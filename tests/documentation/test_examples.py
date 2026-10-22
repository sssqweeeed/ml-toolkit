"""Published examples expose captured output that agrees with executed code."""

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.documentation

from examples.render_examples import PAGES, RECIPE, recipes, render

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('page', [page for page in PAGES if page not in {
    'docs/configuration.md',
}])
def test_captured_documentation_output_matches_examples(page):
    from examples.gallery_data import bike_cache_available
    from examples.retail_data import retail_cache_available

    if page in {'README.md', 'docs/getting-started.md', 'docs/api/models.md',
                'docs/api/training.md', 'docs/api/tracking.md', 'docs/api/custom-model.md',
                'docs/api/splitting.md', 'docs/api/feature-selection.md',
                'docs/api/tuning.md', 'docs/api/metrics.md'}:
        if not bike_cache_available():
            pytest.skip('Prepare the original Bike Sharing cache before running this recipe')
    if page in {'docs/api/models.md', 'docs/api/inflation.md'} and not retail_cache_available():
        pytest.skip('Prepare the original Online Retail cache before running this recipe')
    if page == 'docs/api/models.md':
        pytest.importorskip('lightgbm')
    render(check=True, pages=(page,))


@pytest.mark.parametrize('page', PAGES)
def test_every_recipe_is_registered_even_without_prints(page):
    assert recipes(ROOT / page)


def test_all_executable_pages_have_one_renderer():
    from examples.render_plots import PAGES as PLOT_PAGES

    assert not set(PAGES).intersection(PLOT_PAGES)
    executable = {
        path.relative_to(ROOT).as_posix()
        for path in [ROOT / 'README.md', *sorted((ROOT / 'docs').rglob('*.md'))]
        if '<!-- example:' in path.read_text()
    }
    assert executable == set(PAGES) | set(PLOT_PAGES)


def test_inflation_calculation_changes_price_and_reverses():
    import numpy as np
    from examples.render_examples import execute

    namespace = {}
    recipe = next(item for item in recipes(ROOT / 'docs/api/inflation.md')
                  if item.name == 'inflation-constant')
    execute(recipe, namespace)
    assert namespace['adjusted'][0] > namespace['restored'][0]
    np.testing.assert_allclose(namespace['restored'], [100.0])


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
