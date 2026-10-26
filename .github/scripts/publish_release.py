import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from workflow_checks import git, package_version


def gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True).strip()


def api(path: str) -> Any:
    # GitHub's REST response is a third-party JSON boundary.
    return json.loads(gh("api", f"repos/{os.environ['GITHUB_REPOSITORY']}/{path}"))


def verify_candidate(candidate: dict[str, str], tree: str, revision: str) -> None:
    if candidate != {"tree": tree, "revision": revision, "base": "main"}:
        raise ValueError("Release commit does not match the validated main candidate")


def verify() -> None:
    commit = git("rev-parse", "HEAD")
    if commit != api("git/ref/heads/main")["object"]["sha"]:
        raise ValueError("Only the current main commit can be released")
    prs = api(f"commits/{commit}/pulls")
    prs = [
        pr
        for pr in prs
        if pr["merged_at"]
        and pr["base"]["ref"] == "main"
        and pr["merge_commit_sha"] == commit
    ]
    if len(prs) != 1:
        raise ValueError("Expected exactly one merged release PR for this commit")
    pr = prs[0]
    runs = api(
        f"actions/workflows/tests.yml/runs?event=pull_request&status=success&head_sha={pr['head']['sha']}&per_page=100"
    )["workflow_runs"]
    if not runs:
        raise ValueError("No successful candidate validation run")
    # Use the latest successful run; do not silently reuse older incompatible evidence.
    run = max(runs, key=lambda item: item["id"])
    with tempfile.TemporaryDirectory() as directory:
        gh(
            "run",
            "download",
            str(run["id"]),
            "--name",
            "release-candidate",
            "--dir",
            directory,
        )
        candidate = json.loads((Path(directory) / "candidate.json").read_text())
    verify_candidate(candidate, git("rev-parse", "HEAD^{tree}"), pr["head"]["sha"])
    Path(os.environ["RUNNER_TEMP"], "release-notes.md").write_text(pr["body"] + "\n")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"version={package_version()}\n")


def publish() -> None:
    version = package_version()
    tag = f"v{version}"
    commit = git("rev-parse", "HEAD")
    if tag in git("tag", "--list").splitlines():
        if git("rev-parse", f"{tag}^{{commit}}") != commit:
            raise ValueError("Existing release tag points to a different commit")
    else:
        subprocess.run(
            ["git", "tag", "-a", tag, "-m", f"Release {version}", commit], check=True
        )
        subprocess.run(["git", "push", "origin", f"refs/tags/{tag}"], check=True)
    releases = json.loads(
        gh(
            "api",
            "--paginate",
            f"repos/{os.environ['GITHUB_REPOSITORY']}/releases",
            "--slurp",
        )
    )
    existing = next(
        (
            release
            for page in releases
            for release in page
            if release["tag_name"] == tag
        ),
        None,
    )
    if existing and not existing["draft"]:
        with tempfile.TemporaryDirectory() as directory:
            gh("release", "download", tag, "--dir", directory)
            subprocess.run(
                ["sha256sum", "--check", "SHA256SUMS"], cwd=directory, check=True
            )
        return
    if not existing:
        gh(
            "release",
            "create",
            tag,
            "--verify-tag",
            "--draft",
            "--title",
            tag,
            "--notes-file",
            str(Path(os.environ["RUNNER_TEMP"], "release-notes.md")),
        )
    artifacts = sorted(Path("dist").glob("*"))
    checksums = Path("dist/SHA256SUMS")
    checksums.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
            for path in artifacts
        )
    )
    # Replacing assets is allowed only while the release is still a draft.
    gh(
        "release",
        "upload",
        tag,
        *(str(path) for path in [*artifacts, checksums]),
        "--clobber",
    )
    gh("release", "edit", tag, "--draft=false", "--latest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["verify", "publish"])
    args = parser.parse_args()
    if args.command == "verify":
        verify()
    else:
        publish()
