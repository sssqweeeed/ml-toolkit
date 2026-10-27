import importlib.util
import json
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


@pytest.mark.parametrize("branch", ["dev", "main"])
@pytest.mark.parametrize("event_name", ["pull_request", "workflow_dispatch"])
@pytest.mark.parametrize("tags", ["v0.1.0", "v0.2.0"])
def test_full_validation_is_reserved_for_main(tmp_path, monkeypatch, branch, event_name, tags):
    event = tmp_path / "event.json"
    payload = {}
    if event_name == "pull_request":
        payload["pull_request"] = {
            "base": {"ref": branch},
            "head": {"sha": "candidate-head"},
            "title": "chore(release): 0.2.0",
            "body": "- Improve model training.",
        }
    event.write_text(json.dumps(payload))
    output = tmp_path / "output"
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("GITHUB_REF_NAME", branch)
    monkeypatch.setattr(
        checks, "git", lambda *args: tags if args[0] == "tag" else "manual-head"
    )
    monkeypatch.setattr(checks, "package_version", lambda: "0.2.0")
    if branch == "dev":
        with pytest.raises(ValueError, match="only available for main"):
            checks.plan()
        assert not output.exists()
        return
    if event_name == "pull_request" and tags == "v0.2.0":
        with pytest.raises(ValueError, match="newer than every published version"):
            checks.plan()
        assert not output.exists()
        return
    checks.plan()
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert json.loads(values["matrix"]) == {"python": ["3.11", "3.12"]}
    assert values["revision"] == (
        "candidate-head" if event_name == "pull_request" else "manual-head"
    )
    assert values["version"] == "0.2.0"


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


def test_retired_dev_links_redirect_without_changing_released_docs(tmp_path):
    page = tmp_path / "0.2.0" / "api" / "training" / "index.html"
    page.parent.mkdir(parents=True)
    page.write_text("released page")
    (tmp_path / "0.2.0/index.html").write_text("release home")
    (tmp_path / "index.html").write_text("stable landing")
    docs.add_legacy_redirects(tmp_path, "0.2.0")
    assert '../stable/' in (tmp_path / "dev/index.html").read_text()
    assert (
        "../../../stable/api/training/index.html"
        in (tmp_path / "dev/api/training/index.html").read_text()
    )
    assert "../../stable/api/custom-model/" in (
        tmp_path / "dev/custom-model/index.html"
    ).read_text()
    assert page.read_text() == "released page"
    assert (tmp_path / "0.2.0/index.html").read_text() == "release home"
    assert (tmp_path / "index.html").read_text() == "stable landing"
