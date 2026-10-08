"""Static contracts reject misuse without adding runtime introspection."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_types(path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            "--project",
            str(ROOT),
            "--outputjson",
            str(path),
        ],
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    return json.loads(result.stdout)


def test_valid_public_contracts():
    report = check_types(ROOT / "tests/typecheck/valid_contracts.py")
    assert report["summary"]["errorCount"] == 0, report["generalDiagnostics"]


def test_invalid_public_contracts_are_reported_statically():
    path = ROOT / "tests/typecheck/invalid_contracts.py"
    expected = {
        index
        for index, line in enumerate(path.read_text().splitlines())
        if "expected-type-error" in line
    }
    report = check_types(path)
    errors = [
        item for item in report["generalDiagnostics"] if item["severity"] == "error"
    ]
    assert errors
    assert {item["range"]["start"]["line"] for item in errors} == expected, errors
