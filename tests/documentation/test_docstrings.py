"""Execute source examples as part of the ordinary documentation test suite."""

import ast
import doctest
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.documentation
PACKAGE = Path(__file__).resolve().parents[2] / "src/ml_toolkit"


def example_modules():
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        docstrings = [
            ast.get_docstring(node) or ""
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if any(">>>" in docstring for docstring in docstrings):
            parts = path.relative_to(PACKAGE.parent).with_suffix("").parts
            yield ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


@pytest.mark.parametrize("module_name", list(example_modules()))
def test_documented_source_examples(module_name):
    module = importlib.import_module(module_name)
    for example in doctest.DocTestFinder().find(module):
        assert not any(item.options.get(doctest.SKIP) for item in example.examples), (
            f"Skipped example needs an executable usage path: {example.name}"
        )
    result = doctest.testmod(module, optionflags=doctest.ELLIPSIS)
    assert result.attempted > 0
    assert result.failed == 0
