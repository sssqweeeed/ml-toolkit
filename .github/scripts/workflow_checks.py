import json
import os
from pathlib import Path
import re
import subprocess
import tomllib

SUBJECT = re.compile(
    r"(?:feat|fix|perf|refactor|docs|test|build|ci|chore|revert)(?:\([a-z0-9_-]+\))?!?: .+"
)
VERSION = re.compile(r"(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


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
    branch = pr["base"]["ref"] if pr else os.environ["GITHUB_REF_NAME"]
    if branch != "main":
        raise ValueError("Full CI validation is only available for main candidates")
    if pr:
        validate_pr(pr["title"], pr.get("body") or "")
        revision = pr["head"]["sha"]
    else:
        revision = git("rev-parse", "HEAD")
    version = package_version()
    if pr:
        validate_version(version, git("tag", "--list").splitlines())
    if pr and pr["title"] != f"chore(release): {version}":
        raise ValueError(f"Release PR title must be chore(release): {version}")
    values = {
        "matrix": json.dumps({"python": ["3.11", "3.12"]}),
        "version": version,
        "revision": revision,
    }
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        for key, value in values.items():
            stream.write(f"{key}={value}\n")


if __name__ == "__main__":
    plan()
