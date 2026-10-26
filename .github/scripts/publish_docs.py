import html
import json
import os
from pathlib import Path
import subprocess
import tempfile
from urllib.parse import quote

from workflow_checks import git, package_version


def redirect(target: str) -> str:
    safe = html.escape(target, quote=True)
    return f'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0; url={safe}"><link rel="canonical" href="{safe}"><a href="{safe}">Continue to documentation</a>\n'


def add_legacy_redirects(site: Path, version: str) -> None:
    (site / "dev").mkdir(exist_ok=True)
    (site / "dev/index.html").write_text(redirect("../stable/"))
    for page in (site / version).rglob("*.html"):
        relative = page.relative_to(site / version)
        if relative.as_posix() in {"index.html", "404.html"}:
            continue
        destination = site / relative
        # Never replace versioned directories or the root landing page.
        if relative.parts[0] in {"dev", "stable"}:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        prefix = "../" * (len(relative.parts) - 1)
        destination.write_text(redirect(f"{prefix}stable/{quote(relative.as_posix())}"))
        destination = site / "dev" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            redirect(f"../{prefix}stable/{quote(relative.as_posix())}")
        )
    # Paths retired before versioned documentation was introduced.
    for old, new in {
        "custom-model": "api/custom-model",
        "api/plots": "evaluation",
    }.items():
        destination = site / old / "index.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            redirect("../" * len(Path(old).parts) + f"stable/{new}/")
        )
        destination = site / "dev" / old / "index.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            redirect("../" * (len(Path(old).parts) + 1) + f"stable/{new}/")
        )


def main() -> None:
    source = git("rev-parse", "HEAD")
    remote = git("ls-remote", "origin", "refs/heads/main").split()[0]
    if source != remote:
        raise ValueError("Refusing to deploy stale documentation")
    existing = git("ls-remote", "--heads", "origin", "gh-pages")
    if existing:
        subprocess.run(["git", "fetch", "origin", "gh-pages:gh-pages"], check=True)
    version = package_version()
    environment = os.environ.copy()
    environment["DOCS_EDIT_URI"] = f"blob/v{version}/docs/"
    if git("rev-parse", f"v{version}^{{commit}}") != source:
        raise ValueError("Stable documentation must match its release tag")
    versions = (
        json.loads(subprocess.check_output(["mike", "list", "--json"], text=True))
        if existing
        else []
    )
    if any(item["version"] == "dev" for item in versions):
        subprocess.run(["mike", "delete", "dev"], check=True)
    if not any(item["version"] == version for item in versions):
        subprocess.run(
            [
                "mike",
                "deploy",
                "--update-aliases",
                "--alias-type",
                "redirect",
                version,
                "stable",
            ],
            env=environment,
            check=True,
        )
    else:
        subprocess.run(
            [
                "mike",
                "alias",
                "--update-aliases",
                "--alias-type",
                "redirect",
                version,
                "stable",
            ],
            check=True,
        )
    versions = json.loads(
        subprocess.check_output(["mike", "list", "--json"], text=True)
    )
    stable = next(
        (item["version"] for item in versions if "stable" in item.get("aliases", [])),
        None,
    )
    if stable is None:
        raise ValueError("No stable documentation version was published")
    subprocess.run(["mike", "set-default", "stable"], check=True)
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(["git", "worktree", "add", directory, "gh-pages"], check=True)
        try:
            site = Path(directory)
            add_legacy_redirects(site, stable)
            (site / ".nojekyll").touch()
            subprocess.run(["git", "add", "."], cwd=directory, check=True)
            if subprocess.check_output(
                ["git", "diff", "--cached", "--name-only"], cwd=directory
            ):
                subprocess.run(
                    [
                        "git",
                        "commit",
                        "-m",
                        "docs: preserve stable documentation links",
                    ],
                    cwd=directory,
                    check=True,
                )
        finally:
            subprocess.run(["git", "worktree", "remove", directory], check=True)
    if source != git("ls-remote", "origin", "refs/heads/main").split()[0]:
        raise ValueError("Source branch changed while documentation was building")
    subprocess.run(["git", "push", "origin", "gh-pages"], check=True)
    output = Path(os.environ["RUNNER_TEMP"]) / "pages"
    output.mkdir()
    archive = subprocess.Popen(["git", "archive", "gh-pages"], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(output)], stdin=archive.stdout, check=True)
    if archive.wait() != 0:
        raise RuntimeError("Could not export complete documentation tree")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as stream:
        stream.write(f"deployment_sha={git('rev-parse', 'gh-pages')}\n")


if __name__ == "__main__":
    main()
