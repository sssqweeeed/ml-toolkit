import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / ".github" / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


checks = load("workflow_checks")
release = load("publish_release")
docs = load("publish_docs")


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "tests/conftest.py",
        "tests/support/training.py",
        "src/ml_toolkit/_rows.py",
        "new-module.py",
    ],
)
def test_shared_or_unknown_changes_expand_coverage(path):
    selected, _ = checks.select_tests([path])
    assert selected == ["tests"]


def test_model_changes_cover_training_and_serving_consumers():
    selected, documentation = checks.select_tests(["src/ml_toolkit/models/catboost.py"])
    assert {"tests/models", "tests/training", "tests/tracking"} <= set(selected)
    assert documentation


def test_docs_changes_keep_core_and_enable_recipes():
    selected, documentation = checks.select_tests(["docs/getting-started.md"])
    assert set(selected) == checks.CORE
    assert documentation


@pytest.mark.parametrize(
    "title,body",
    [
        ("Add a feature", "- Add evaluation support."),
        ("feat: add evaluation", "## Research\nMemory dropped from 50 to 20 MB."),
        ("feat: add evaluation", "- [ ] Run tests"),
        ("feat: add evaluation", ""),
    ],
)
def test_pr_contract_rejects_non_change_descriptions(title, body):
    with pytest.raises(ValueError):
        checks.validate_pr(title, body)


def test_pr_contract_accepts_breaking_change_and_migration_bullet():
    checks.validate_pr(
        "refactor(models)!: require explicit state",
        "- Require explicit model state; pass state to the constructor.",
    )


@pytest.mark.parametrize("version", ["0.1.0", "0.0.9"])
def test_release_version_cannot_reuse_or_downgrade_tag(version):
    with pytest.raises(ValueError):
        checks.validate_version(version, ["unrelated", "v0.1.0"])


def test_release_version_order_is_numeric():
    checks.validate_version("0.10.0", ["v0.9.0"])


@pytest.mark.parametrize("field", ["tree", "revision", "base"])
def test_release_rejects_mismatched_validation_evidence(field):
    candidate = {"tree": "validated-tree", "revision": "validated-head", "base": "main"}
    candidate[field] = "different"
    with pytest.raises(ValueError):
        release.verify_candidate(candidate, "validated-tree", "validated-head")


def test_release_accepts_same_tree_with_a_new_merge_commit():
    release.verify_candidate(
        {"tree": "tree", "revision": "head", "base": "main"}, "tree", "head"
    )


def test_legacy_redirects_preserve_versioned_pages(tmp_path):
    page = tmp_path / "0.1.0" / "api" / "training" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("released page")
    (tmp_path / "index.html").write_text("stable landing")
    docs.add_legacy_redirects(tmp_path, "0.1.0")
    assert page.read_text() == "released page"
    assert (tmp_path / "index.html").read_text() == "stable landing"
    assert (
        "../../stable/api/training/index.html"
        in (tmp_path / "api/training/index.html").read_text()
    )
    assert "../../stable/evaluation/" in (tmp_path / "api/plots/index.html").read_text()
