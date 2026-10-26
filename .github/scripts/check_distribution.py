from pathlib import Path
import subprocess
import sys
import tempfile
import tomllib


def main() -> None:
    version: str = tomllib.loads(Path("pyproject.toml").read_text())["project"][
        "version"
    ]
    artifacts = sorted(Path("dist").resolve().glob("*"))
    if (
        len(artifacts) != 2
        or not any(p.suffix == ".whl" for p in artifacts)
        or not any(p.name.endswith(".tar.gz") for p in artifacts)
    ):
        raise ValueError("Expected exactly one wheel and one sdist")
    for artifact in artifacts:
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / "env"
            subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
            python = str(env / "bin" / "python")
            subprocess.run([python, "-m", "pip", "install", str(artifact)], check=True)
            subprocess.run(
                [
                    python,
                    "-c",
                    "\n".join(
                        [
                            "from importlib.metadata import version",
                            "import numpy as np",
                            "from ml_toolkit.metrics import metrics_regression",
                            f"assert version('ml-toolkit') == {version!r}",
                            "result = metrics_regression(np.array([1., 2.]), np.array([1., 2.]), np.ones(2))",
                            "assert result.root_mean_squared_error == 0.0",
                        ]
                    ),
                ],
                cwd=directory,
                check=True,
            )


if __name__ == "__main__":
    main()
