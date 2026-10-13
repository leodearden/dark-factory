"""Tests for shared.finding_key, the one finding-key implementation.

The parity fixture's keys were sampled by running the interim reference
(`skills/hotspot-survey/scripts/findings_artefact.py key`), never by the
function under test, so a correlated change to both implementations still
fails here (docs/quality-findings-contract.md §2).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from shared.finding_key import KEY_PATTERN, finding_key, is_finding_key, normalise_anchor

REPO_ROOT = Path(__file__).resolve().parents[2]
INTERIM_CLI = REPO_ROOT / "skills" / "hotspot-survey" / "scripts" / "findings_artefact.py"
FIXTURE = Path(__file__).resolve().parent / "finding_key_parity_fixture.json"
CASES = json.loads(FIXTURE.read_text())


def _case_id(case: dict) -> str:
    return f"{case['area']}|{case['anchor'].strip()}|{case['primary_tag']}"


def test_fixture_covers_every_normalisation_branch():
    anchors = [case["anchor"] for case in CASES]
    tags = {case["primary_tag"].lower() for case in CASES}
    assert len(CASES) >= 12
    assert any(a.endswith(":120") for a in anchors)
    assert any(a.endswith(":120-140") for a in anchors)
    assert any(a.endswith(":1,5-7,9") for a in anchors)
    assert any("\\" in a for a in anchors)
    assert any(a != a.strip() for a in anchors)
    assert any(a.startswith("slug:") for a in anchors)
    assert any("::" not in a and not a.startswith("slug:") for a in anchors)
    assert any(re.search(r"::\w+\.\w+$", a) for a in anchors)
    assert any(c["area"] != c["area"].lower() or c["primary_tag"] != c["primary_tag"].lower() for c in CASES)
    assert {"h13", "comments", "tests", "inv-11", "kind:confusion"} <= tags


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_finding_key_matches_the_committed_fixture(case):
    assert finding_key(case["area"], case["anchor"], case["primary_tag"]) == case["key"]


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


@pytest.mark.parametrize(
    "variant, clean",
    [
        ("x.py::f:12", "x.py::f"),
        ("x.py::f:3-9", "x.py::f"),
        ("x.py::f:1,4-6", "x.py::f"),
        (" x.py::f ", "x.py::f"),
        ("dir\\x.py::f", "dir/x.py::f"),
    ],
)
def test_normalisation_variants_share_the_clean_anchors_key(variant, clean):
    assert normalise_anchor(variant) == clean
    assert finding_key("orchestrator", variant, "h13") == finding_key("orchestrator", clean, "h13")


def test_a_renamed_symbol_gets_a_new_key():
    assert finding_key("orchestrator", "x.py::old", "h13") != finding_key("orchestrator", "x.py::new", "h13")


def test_another_area_or_primary_tag_gets_a_new_key():
    base = finding_key("orchestrator", "x.py::f", "h13")
    assert finding_key("shared", "x.py::f", "h13") != base
    assert finding_key("orchestrator", "x.py::f", "h4") != base


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_every_minted_key_matches_key_pattern(case):
    key = finding_key(case["area"], case["anchor"], case["primary_tag"])
    assert re.fullmatch(KEY_PATTERN, key)
    assert is_finding_key(key)


@pytest.mark.parametrize("value", ["fk-XYZ", "fk-0123456789a", "fk-0123456789ab0", None, 123])
def test_is_finding_key_rejects_non_keys(value):
    assert not is_finding_key(value)
