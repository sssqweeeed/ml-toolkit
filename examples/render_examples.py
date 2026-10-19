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
    return [Recipe(match['name'], match['code']) for match in RECIPE.finditer(path.read_text())]


def execute(recipe: Recipe, namespace: dict[str, object]) -> str:
    """Capture the actual stdout of one documentation snippet."""
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(recipe.code, recipe.name, 'exec'), namespace)
    return stream.getvalue().strip()


def prepare_data(namespace: dict[str, object]) -> None:
    execute(recipes(ROOT / 'docs/api/data.md')[0], namespace)


def prepare_model(namespace: dict[str, object]) -> None:
    prepare_data(namespace)
    execute(recipes(ROOT / 'docs/api/models.md')[0], namespace)


def render_page(path: Path, *, check: bool) -> None:
    """Execute a page with its documented prerequisites and verify its output."""
    relative = path.relative_to(ROOT).as_posix()
    source = path.read_text()
    namespace: dict[str, object] = {'__name__': '__documentation__'}
    if relative in {'docs/api/models.md','docs/api/training.md','docs/api/inflation.md','docs/configuration.md','docs/custom-model.md'}:
        prepare_data(namespace)
    if relative == 'docs/api/tracking.md':
        prepare_model(namespace)
    outputs: dict[str, str] = {}
    frequency: dict[str, object] = {}
    for recipe in recipes(path):
        if recipe.name == 'configuration-run':
            namespace.update(frequency)
        outputs[recipe.name] = execute(recipe, namespace)
        if recipe.name == 'configuration-frequency':
            frequency = {name: namespace[name] for name in ('config','params','metric','model_factory')}

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
    arguments = parser.parse_args()
    render(check=arguments.check)
    from examples.render_plots import render as render_plots
    render_plots(check=arguments.check)


if __name__ == '__main__':
    main()
