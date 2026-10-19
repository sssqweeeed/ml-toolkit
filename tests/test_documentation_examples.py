"""Published examples expose captured output that agrees with executed code."""

import ast
import re
from pathlib import Path

import pytest

from examples.render_examples import RECIPE, render

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('page', ['docs/api/data.md', 'docs/api/psi.md', 'docs/api/inflation.md'])
def test_captured_documentation_output_matches_examples(page):
    render(check=True, pages=(page,))


def test_every_printed_markdown_example_has_captured_output():
    paths = [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md'))]
    for path in paths:
        document = path.read_text()
        for match in re.finditer(r'^```python\n(.*?)\n```', document, re.MULTILINE | re.DOTALL):
            code = match[1]
            if '--8<--' in code:
                continue
            tree = ast.parse(code)
            prints = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id in {'print', 'pprint'}]
            if not prints:
                continue
            recipe = next((item for item in RECIPE.finditer(document)
                           if item.start() <= match.start() < item.end()), None)
            assert recipe is not None, f'Unmarked print example: {path}'
            assert recipe['output'], f'No captured output: {path}: {recipe["name"]}'
