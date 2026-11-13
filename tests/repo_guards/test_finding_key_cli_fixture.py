"""The interim finding-key CLI agrees with shared's committed parity fixture.

The fixture and shared.finding_key's own tests live in
shared/tests/test_finding_key.py. This check runs the interim reference
implementation under skills/, a tree outside shared's module.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM_CLI = REPO_ROOT / "skills" / "hotspot-survey" / "scripts" / "findings_artefact.py"
FIXTURE = REPO_ROOT / "shared" / "tests" / "finding_key_parity_fixture.json"
CASES = json.loads(FIXTURE.read_text())


def _case_id(case: dict) -> str:
    return f"{case['area']}|{case['anchor'].strip()}|{case['primary_tag']}"


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_interim_cli_agrees_with_the_committed_fixture(case):
    result = subprocess.run(
        [sys.executable, str(INTERIM_CLI), "key", case["area"], case["anchor"], case["primary_tag"]],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == case["key"]
