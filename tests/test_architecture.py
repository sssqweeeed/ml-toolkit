"""The import graph follows the layer order and keeps private modules private."""

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
    "_grouping": 4,
    "models": 5,
    "evaluation": 6,
    "model_evaluation": 6,
    "data_analysis": 6,
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
    # Phase 3: ShapConfig moves down with explainability.
    ("ml_toolkit.explainability", "ml_toolkit.config"),
    ("ml_toolkit.plots.explainability", "ml_toolkit.config"),
    # Phase 5: the selection figures move into feature_selection.
    ("ml_toolkit.plots.feature_selection", "ml_toolkit.feature_selection"),
    # Phase 3: row grouping and report-metric validation join reports.
    ("ml_toolkit.evaluation.grouped", "ml_toolkit._grouping"),
    ("ml_toolkit.evaluation.regression", "ml_toolkit._grouping"),
    ("ml_toolkit.feature_selection", "ml_toolkit.evaluation._metrics"),
    ("ml_toolkit.training.reporting", "ml_toolkit.evaluation._metrics"),
    # Phase 4: feature-level serialization merges into feature_levels.
    ("ml_toolkit.psi.baseline", "ml_toolkit.feature_levels._serialization"),
}


@dataclass(frozen=True)
class ImportedName:
    importer: str
    target: str


def module_name(path: Path) -> str:
    parts = path.relative_to(SOURCE.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def collect_imports(path: Path) -> list[ImportedName]:
    """Every import in the module, including TYPE_CHECKING and function-level ones."""
    importer = module_name(path)
    package = importer if path.name == "__init__.py" else importer.rpartition(".")[0]
    found: list[ImportedName] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.extend(ImportedName(importer, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                base = parts[: len(parts) - (node.level - 1)]
                if node.module:
                    base += node.module.split(".")
            else:
                base = (node.module or "").split(".")
            module = ".".join(base)
            for alias in node.names:
                found.append(ImportedName(importer, f"{module}.{alias.name}"))
                found.append(ImportedName(importer, module))
    return [item for item in found if item.target.split(".")[0] == ROOT]


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


@pytest.mark.parametrize("entry", sorted(ALLOWED_VIOLATIONS))
def test_allowed_violations_are_still_needed(entry):
    assert entry in layer_violations() | private_violations()
