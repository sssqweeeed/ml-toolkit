"""The import graph follows the layer order, has no cycles and stays explicit."""

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "src" / "ml_toolkit"
ROOT = "ml_toolkit"

# Lowest layer first; a module may import only from its own or a lower layer.
LAYERS: dict[str, int] = {
    "types": 1,
    "_time": 1,
    "_console": 1,
    "figures": 1,
    "data": 2,
    "metrics": 3,
    "feature_levels": 4,
    "psi": 4,
    "inflation": 4,
    "models": 5,
    "reports": 6,
    "plots": 6,
    "explainability": 6,
    "feature_selection": 7,
    "tracking": 7,
    "config": 8,
    "training": 9,
}

# Private root modules that every layer may use as internal infrastructure.
SHARED_PRIVATE_MODULES = {"_console", "_time"}

# Known violations as (importing module, imported name). Each entry names the
# refactoring phase that removes it; the set must shrink, never grow.
ALLOWED_VIOLATIONS: set[tuple[str, str]] = {
    # Phase 6: the dataset summary printer moves out of the console layer.
    ("ml_toolkit._console", "ml_toolkit.data"),
    # Phase 5: the selection figures move into feature_selection.
    ("ml_toolkit.plots.feature_selection", "ml_toolkit.feature_selection"),
    # Phase 4: feature-level serialization merges into feature_levels.
    ("ml_toolkit.psi.baseline", "ml_toolkit.feature_levels._serialization"),
}


# Package cycles that later phases remove; none remain after the reports merge.
ALLOWED_CYCLES: set[frozenset[str]] = set()

# Function-level imports of toolkit modules as (importing module, imported
# module). Only optional backends may be imported lazily; each entry below
# works around a cycle that the named phase removes.
FUNCTION_LEVEL_IMPORTS: set[tuple[str, str]] = {
    # Phase 2: training-only code (metrics, numba kernels) is split from serving
    # code; serving environments must not import it, see check_model_inference.
    ("ml_toolkit.models.catboost_classification", "ml_toolkit.metrics.classification"),
    ("ml_toolkit.models.catboost_classification", "ml_toolkit.models.losses.pu.cpu"),
    ("ml_toolkit.models.losses.gamma.config", "ml_toolkit.models.losses.gamma.cpu"),
    ("ml_toolkit.models.losses.pu.common", "ml_toolkit.metrics.pu"),
    ("ml_toolkit.models.losses.pu.common", "ml_toolkit.models.losses.pu.cpu"),
}

# Optional backends: CUDA kernels and LightGBM may be imported lazily.
OPTIONAL_MODULES = {"cuda", "lightgbm"}


@dataclass(frozen=True)
class ImportedName:
    importer: str
    target: str
    in_function: bool = False
    is_name: bool = False  # an imported name, not necessarily a module


class ImportCollector(ast.NodeVisitor):
    """Collect every import, including TYPE_CHECKING and function-level ones."""

    def __init__(self, importer: str, package: str) -> None:
        self.importer = importer
        self.package = package
        self.depth = 0
        self.found: list[ImportedName] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    def add(self, target: str, *, is_name: bool = False) -> None:
        self.found.append(ImportedName(self.importer, target, self.depth > 0, is_name))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.add(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            parts = self.package.split(".")
            base = parts[: len(parts) - (node.level - 1)]
            if node.module:
                base += node.module.split(".")
        else:
            base = (node.module or "").split(".")
        module = ".".join(base)
        for alias in node.names:
            self.add(f"{module}.{alias.name}", is_name=True)
            self.add(module)


def module_name(path: Path) -> str:
    parts = path.relative_to(SOURCE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def collect_imports(path: Path) -> list[ImportedName]:
    importer = module_name(path)
    package = importer if path.name == "__init__.py" else importer.rpartition(".")[0]
    collector = ImportCollector(importer, package)
    collector.visit(ast.parse(path.read_text(encoding="utf-8")))
    return [i for i in collector.found if i.target.split(".")[0] == ROOT]


def toolkit_imports() -> list[ImportedName]:
    modules = sorted(SOURCE.rglob("*.py"))
    return [item for path in modules for item in collect_imports(path)]


def top_level(name: str) -> str | None:
    parts = name.split(".")
    return parts[1] if len(parts) > 1 else None


def layer_violations() -> set[tuple[str, str]]:
    violations: set[tuple[str, str]] = set()
    for item in toolkit_imports():
        importer = top_level(item.importer)
        imported = top_level(item.target)
        if importer is None or imported is None or importer == imported:
            continue
        if LAYERS[imported] > LAYERS[importer]:
            violations.add((item.importer, ".".join(item.target.split(".")[:2])))
    return violations


def private_violations() -> set[tuple[str, str]]:
    violations: set[tuple[str, str]] = set()
    for item in toolkit_imports():
        importer = top_level(item.importer)
        parts = item.target.split(".")
        if importer is None or top_level(item.target) == importer:
            continue
        private = [
            index
            for index, part in enumerate(parts[1:], start=1)
            if part.startswith("_") and not part.startswith("__")
        ]
        if not private:
            continue
        if private == [1] and parts[1] in SHARED_PRIVATE_MODULES:
            continue
        violations.add((item.importer, ".".join(parts[: private[0] + 1])))
    return violations


def package_cycles() -> set[frozenset[str]]:
    """Strongly connected groups of top-level packages in the import graph."""
    edges: dict[str, set[str]] = {name: set() for name in LAYERS}
    for item in toolkit_imports():
        source, target = top_level(item.importer), top_level(item.target)
        if source is not None and target is not None and source != target:
            edges[source].add(target)

    def reachable(start: str) -> set[str]:
        seen: set[str] = set()
        pending = [start]
        while pending:
            for nxt in edges[pending.pop()] - seen:
                seen.add(nxt)
                pending.append(nxt)
        return seen

    reach = {name: reachable(name) for name in edges}
    cycles = {
        frozenset({name} | {o for o in reach[name] if name in reach[o]})
        for name in edges
        if name in reach[name]
    }
    return cycles


def function_level_imports() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for item in toolkit_imports():
        if item.importer == ROOT or not item.in_function or item.is_name:
            continue
        parts = item.target.split(".")
        if OPTIONAL_MODULES & set(parts[1:]):
            continue
        found.add((item.importer, item.target))
    return found


def test_every_module_has_a_layer():
    top_levels = {
        top_level(module_name(path))
        for path in SOURCE.rglob("*.py")
        if module_name(path) != ROOT
    }
    assert top_levels <= set(LAYERS)


def test_no_module_imports_from_a_higher_layer():
    found = layer_violations() - ALLOWED_VIOLATIONS
    assert not found, f"imports from a higher layer: {sorted(found)}"


def test_private_modules_and_names_stay_in_their_package():
    found = private_violations() - ALLOWED_VIOLATIONS
    assert not found, f"private names imported across packages: {sorted(found)}"


def test_no_new_package_cycles():
    found = package_cycles() - ALLOWED_CYCLES
    assert not found, f"package import cycles: {sorted(map(sorted, found))}"


def test_function_level_imports_are_limited_to_optional_backends():
    found = function_level_imports() - FUNCTION_LEVEL_IMPORTS
    assert not found, f"function-level imports of toolkit modules: {sorted(found)}"


@pytest.mark.parametrize("entry", sorted(ALLOWED_VIOLATIONS))
def test_allowed_violations_are_still_needed(entry):
    assert entry in layer_violations() | private_violations()


@pytest.mark.parametrize("entry", sorted(FUNCTION_LEVEL_IMPORTS))
def test_allowed_function_level_imports_are_still_needed(entry):
    assert entry in function_level_imports()


@pytest.mark.parametrize("entry", sorted(map(sorted, ALLOWED_CYCLES)))
def test_allowed_cycles_are_still_needed(entry):
    assert frozenset(entry) in package_cycles()
