import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib

CORE = {
    "tests/config",
    "tests/data",
    "tests/metrics",
    "tests/typing",
    "tests/automation",
}
DEPENDENTS = {
    "data.py": {"data", "training", "models", "tracking", "plots"},
    "config": {"config", "training", "models", "tracking", "plots"},
    "metrics": {"metrics", "training", "plots", "tracking"},
    "models": {"models", "training", "tracking"},
    "training": {"training", "models", "tracking"},
    "plots": {"plots", "training", "tracking"},
    "psi": {"psi", "plots", "training"},
    "temporal": {"temporal", "training", "plots"},
    "tracking": {"tracking", "models", "training"},
    "evaluation.py": {"training", "plots", "tracking"},
    "data_analysis.py": {"training", "plots", "tracking"},
    "feature_selection.py": {"training", "models", "tracking"},
}
SUBJECT = re.compile(
    r"(?:feat|fix|perf|refactor|docs|test|build|ci|chore|revert)(?:\([a-z0-9_-]+\))?!?: .+"
)
VERSION = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def select_tests(paths: list[str]) -> tuple[list[str], bool]:
    selected = set(CORE)
    broad = False
    docs = False
    for path in paths:
        if path.startswith(("docs/", "examples/", "tests/documentation/")) or path in {
            "README.md",
            "mkdocs.yml",
            "requirements-docs.txt",
            "requirements-examples.txt",
        }:
            docs = True
        elif path.startswith("src/ml_toolkit/"):
            docs = True
            key = path.removeprefix("src/ml_toolkit/").split("/")[0]
            if key in DEPENDENTS:
                selected.update(f"tests/{name}" for name in DEPENDENTS[key])
            else:
                broad = True
        elif path.startswith("tests/"):
            parts = path.split("/")
            if len(parts) < 3 or parts[1] in {"support", "typecheck"}:
                broad = True
            else:
                selected.add(f"tests/{parts[1]}")
        else:
            broad = True
            docs = True
    return (["tests"] if broad else sorted(selected)), docs


def validate_pr(title: str, body: str) -> None:
    if not SUBJECT.fullmatch(title):
        raise ValueError("PR title must follow Conventional Commits")
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    if not 1 <= len(lines) <= 5 or any(
        not line.startswith("- ") or len(line) < 4 for line in lines
    ):
        raise ValueError("PR description must contain 1-5 concise change bullets")
    if any(line.startswith("- [") for line in lines):
        raise ValueError("Describe delivered changes, not a checklist")


def package_version() -> str:
    value: str = tomllib.loads(Path("pyproject.toml").read_text())["project"]["version"]
    if not VERSION.fullmatch(value):
        raise ValueError("Released versions must use X.Y.Z")
    return value


def validate_version(version: str, tags: list[str]) -> None:
    released = [
        tuple(map(int, tag[1:].split(".")))
        for tag in tags
        if tag.startswith("v") and VERSION.fullmatch(tag[1:])
    ]
    if released and tuple(map(int, version.split("."))) <= max(released):
        raise ValueError("Release version must be newer than every published version")


def plan() -> None:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    pr = event.get("pull_request")
    full = (
        pr["base"]["ref"] == "main" if pr else os.environ["GITHUB_REF_NAME"] == "main"
    )
    if pr:
        validate_pr(pr["title"], pr.get("body") or "")
        base = pr["base"]["sha"]
        revision = pr["head"]["sha"]
    else:
        base = event.get("before", "")
        revision = git("rev-parse", "HEAD")
    if not base or set(base) == {"0"}:
        paths = ["pyproject.toml"]
    else:
        paths = git("diff", "--name-only", base, "HEAD").splitlines()
    tests, docs = select_tests(paths)
    # Protected dev merges have already passed their affected PR checks.
    # Recheck integration with the core suite instead of repeating release tests.
    if not pr and os.environ["GITHUB_EVENT_NAME"] == "push" and not full:
        tests, docs = sorted(CORE), False
    version = package_version()
    if full:
        validate_version(version, git("tag", "--list").splitlines())
        if pr and pr["title"] != f"chore(release): {version}":
            raise ValueError(f"Release PR title must be chore(release): {version}")
    values = {
        "full": str(full).lower(),
        "docs": str(docs or full).lower(),
        "tests": json.dumps(tests),
        "matrix": json.dumps({"python": ["3.11", "3.12"] if full else ["3.12"]}),
        "version": version,
        "revision": revision,
    }
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")


def run_tests() -> None:
    paths: list[str] = json.loads(os.environ["TEST_PATHS"])
    if os.environ["FULL"] == "true":
        paths = ["tests"]
    subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            *paths,
            "-n",
            "2",
            "--dist",
            "loadscope",
            "--ignore=tests/documentation",
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["plan", "test"])
    args = parser.parse_args()
    if args.command == "plan":
        plan()
    else:
        run_tests()
