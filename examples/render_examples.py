"""Execute documentation recipes and refresh their captured console output.

Run ``python -m examples.render_examples`` from the repository checkout after
installing the package with its ``all`` extra. Use ``--check`` to verify the
committed output without rewriting Markdown. The command also regenerates the
plot gallery; run ``python -m examples.render_plots`` for figures alone.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECIPE = re.compile(
    r"<!-- example: (?P<name>[\w-]+) -->\n+"
    r"```python\n(?P<code>.*?)\n```"
    r"(?P<output>\n\n<!-- output: (?P=name) -->\n```text\n.*?\n```\n<!-- /output -->)?",
    re.DOTALL,
)
PAGES = (
    "README.md",
    "docs/getting-started.md",
    "docs/api/data.md",
    "docs/api/metrics.md",
    "docs/configuration.md",
    "docs/api/psi.md",
    "docs/api/inflation.md",
    "docs/api/models.md",
    "docs/api/training.md",
    "docs/api/tracking.md",
    "docs/api/configuration.md",
    "docs/custom-model.md",
)


@dataclass(frozen=True)
class Recipe:
    name: str
    code: str


def recipes(path: Path) -> list[Recipe]:
    """Read named executable snippets, excluding source-file includes."""
    document = path.read_text()
    matches = list(RECIPE.finditer(document))
    blocks = list(re.finditer(r"^```python\n.*?\n```", document, re.MULTILINE | re.DOTALL))
    if len(matches) != len(blocks):
        raise ValueError(f"Every top-level Python block in {path.name} needs an example marker")
    names = [match["name"] for match in matches]
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate example markers in {path.name}")
    return [Recipe(match["name"], match["code"]) for match in matches]


def execute(recipe: Recipe, namespace: dict[str, object]) -> str:
    """Capture the actual stdout of one documentation snippet."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(recipe.code, recipe.name, 'exec'), namespace)
    return stream.getvalue().strip()


def prepare_model(namespace: dict[str, object]) -> None:
    # This cross-page prerequisite is linked explicitly by the tracking guide.
    by_name = {recipe.name: recipe for recipe in recipes(ROOT / 'docs/api/models.md')}
    for name in ('models-data', 'models-regression'):
        execute(by_name[name], namespace)


def render_page(path: Path, *, check: bool) -> None:
    """Execute a page with its documented prerequisites and verify its output."""
    relative = path.relative_to(ROOT).as_posix()
    source = path.read_text()
    namespace: dict[str, object] = {'__name__': '__documentation__'}
    if relative == 'docs/api/tracking.md':
        prepare_model(namespace)
    outputs: dict[str, str] = {}
    frequency: dict[str, dict[str, object]] = {}
    for recipe in recipes(path):
        if recipe.name == 'configuration-run':
            namespace.update(frequency['configuration-frequency'])
        elif recipe.name == 'configuration-full-run':
            namespace.update(frequency['configuration-frequency-full'])
        outputs[recipe.name] = execute(recipe, namespace)
        if recipe.name in {'configuration-frequency', 'configuration-frequency-full'}:
            frequency[recipe.name] = {
                name: namespace[name]
                for name in (
                    'data', 'splitter', 'config', 'params', 'metric', 'model_factory',
                    'feature_selector', 'param_space', 'prediction_evaluators',
                )
            }
        if recipe.name == 'configuration-full-run':
            from matplotlib.testing.compare import compare_images

            generated = Path('frequency-full-shap.png')
            asset = ROOT / 'docs/assets/plots/frequency-full-shap.png'
            assert generated.is_file(), 'Full frequency recipe did not create its figure'
            if check:
                assert asset.is_file(), 'Missing full frequency SHAP figure'
                assert compare_images(str(asset), str(generated), tol=1) is None, (
                    'Stale full frequency SHAP figure; regenerate the configuration page'
                )
            else:
                shutil.copyfile(generated, asset)

    def replace_output(match: re.Match[str]) -> str:
        printed = outputs[match['name']]
        body = f"<!-- example: {match['name']} -->\n\n```python\n{match['code']}\n```"
        if printed:
            body += f"\n\n<!-- output: {match['name']} -->\n```text\n{printed}\n```\n<!-- /output -->"
        return body

    updated = RECIPE.sub(replace_output, source)
    if check:
        if updated != source:
            raise AssertionError(f'Stale example output: {relative}; run python -m examples.render_examples')
    else:
        path.write_text(updated.rstrip() + '\n')
    print(f'PASS {relative}: {len(outputs)} snippets, {sum(bool(value) for value in outputs.values())} captured outputs', flush=True)


def render(*, check: bool = False, pages: tuple[str, ...] = PAGES) -> None:
    """Render pages in separate temporary working directories and MLflow stores."""
    previous = Path.cwd()
    try:
        for relative in pages:
            path = ROOT / relative
            with tempfile.TemporaryDirectory(prefix='ml-toolkit-doc-example-') as directory:
                os.chdir(directory)
                # Resolve each relative SQLite URI against this page's own store.
                # MLflow caches engines by URI, so use an absolute URI for isolation.
                import mlflow
                original_set_uri = mlflow.set_tracking_uri
                def set_local_uri(uri: str | Path) -> None:
                    original_set_uri(f'sqlite:///{directory}/mlflow.db' if uri == 'sqlite:///mlflow.db' else uri)
                mlflow.set_tracking_uri = set_local_uri
                try:
                    render_page(path, check=check)
                finally:
                    mlflow.set_tracking_uri = original_set_uri
                    os.chdir(previous)
    finally:
        os.chdir(previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='execute recipes and fail if captured output is stale')
    parser.add_argument(
        '--page', action='append', choices=PAGES,
        help='render only the selected page (repeat for multiple pages)',
    )
    arguments = parser.parse_args()
    render(check=arguments.check, pages=tuple(arguments.page) if arguments.page else PAGES)
    if not arguments.page:
        from examples.render_plots import render as render_plots
        render_plots(check=arguments.check)


if __name__ == '__main__':
    main()
