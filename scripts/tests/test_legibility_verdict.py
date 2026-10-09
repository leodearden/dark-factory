"""Tests for scripts/legibility/verdict.py — the census verifier's typed
verdict and its parser (plans/census-incremental-prd.md §4.4 C4, §4.8 row 6).

Every test drives the public surface (``parse_verdict``, ``Verdict``,
``count_normalisations``) against a real tmp_path tree.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import codebook
import pytest
from legibility import verdict

_FULL_RAW = {
    "verified": True,
    "reason": "seen",
    "anchor": "pkg/mod.py::f",
    "tags": ["h13", "inv-11", "tests"],
    "severity": "high",
    "severity_reason": "next change likely wrong",
    "route": "mechanical",
    "remediation": {"path": "pkg/mod.py", "change": " Rename f "},
}


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text("def f():\n    pass\n")
    return tmp_path


def _parse(raw: dict, root: Path, *, title: str = "A confusion") -> verdict.Verdict:
    return verdict.parse_verdict(raw, title=title, project_root=root)


def _valid_fields(**overrides) -> dict:
    fields = {
        "verified": True,
        "reason": "seen",
        "anchor": "pkg/mod.py::f",
        "tags": ("h13", verdict.KIND_TAG),
        "severity": "high",
        "severity_reason": "next change likely wrong",
        "route": verdict.Route.MECHANICAL,
        "remediation": verdict.Remediation("pkg/mod.py", "Rename f"),
    }
    fields.update(overrides)
    return fields


def test_parse_verdict_keeps_a_fully_valid_reply_verbatim(tree: Path) -> None:
    v = _parse(_FULL_RAW, tree)

    assert v.verified is True
    assert v.reason == "seen"
    assert v.anchor == "pkg/mod.py::f"
    assert v.tags == ("h13", "inv-11", "tests", "kind:confusion")
    assert v.severity == "high"
    assert v.severity_reason == "next change likely wrong"
    assert v.route is verdict.Route.MECHANICAL
    assert v.remediation == verdict.Remediation("pkg/mod.py", "Rename f")
    assert v.normalisations == ()


def test_parse_verdict_parses_a_refuted_reply(tree: Path) -> None:
    v = _parse({**_FULL_RAW, "verified": False}, tree)

    assert v.verified is False
    assert v.normalisations == ()


@pytest.mark.parametrize("offered", [None, "true", 1])
def test_parse_verdict_rejects_a_non_bool_verified(tree: Path, offered) -> None:
    with pytest.raises(verdict.VerdictError) as excinfo:
        _parse({**_FULL_RAW, "verified": offered}, tree)

    assert isinstance(excinfo.value, ValueError)
    assert "verified" in str(excinfo.value)
    assert repr(offered) in str(excinfo.value)


def test_parse_verdict_rejects_an_absent_verified(tree: Path) -> None:
    raw = {k: v for k, v in _FULL_RAW.items() if k != "verified"}

    with pytest.raises(verdict.VerdictError, match="verified"):
        _parse(raw, tree)


def _codebook_with_severity(severity: str) -> dict:
    return {
        "version": 2,
        "entries": [
            {
                "id": "entry-a",
                "title": "Some confusion cluster",
                "severity": severity,
                "status": "open",
                "origin_phase": "unknown",
                "manifested_phase": "unknown",
                "sightings": [],
            },
        ],
    }


def test_severities_are_exactly_the_codebook_entry_severities() -> None:
    for severity in verdict.SEVERITIES:
        assert codebook.validate(_codebook_with_severity(severity)) == []
    for severity in ("critical", "urgent"):
        assert severity not in verdict.SEVERITIES
        assert codebook.validate(_codebook_with_severity(severity)) != []
    assert verdict.DEFAULT_SEVERITY == "medium"
    assert verdict.DEFAULT_SEVERITY in verdict.SEVERITIES


@pytest.mark.parametrize(
    ("field", "offered"),
    [
        ("severity", "critical"),
        ("severity_reason", ""),
        ("tags", ("h13",)),
        ("tags", ("h13", "kind:confusion", "kind:confusion")),
        ("tags", ("kind:confusion", "h13")),
        ("tags", ("bogus", "kind:confusion")),
        ("anchor", "a.py:12"),
        ("anchor", "pkg\\mod.py"),
        ("route", "mechanical"),
    ],
)
def test_verdict_construction_enforces_its_invariants(field: str, offered) -> None:
    with pytest.raises(verdict.VerdictError) as excinfo:
        verdict.Verdict(**_valid_fields(**{field: offered}))

    assert field in str(excinfo.value)
    assert repr(offered) in str(excinfo.value)


def test_verdict_construction_accepts_valid_fields() -> None:
    v = verdict.Verdict(**_valid_fields())

    assert v.normalisations == ()


def test_verdict_is_frozen() -> None:
    v = verdict.Verdict(**_valid_fields())

    with pytest.raises(dataclasses.FrozenInstanceError):
        v.severity = "low"  # type: ignore[misc]


def test_count_normalisations_of_nothing_names_every_kind_at_zero() -> None:
    assert verdict.count_normalisations([]) == {
        kind.value: 0 for kind in verdict.NormalisationKind
    }


def test_to_record_is_json_safe(tree: Path) -> None:
    record = _parse(_FULL_RAW, tree).to_record()

    assert json.loads(json.dumps(record)) == record
    assert record["route"] == "mechanical"
    assert record["tags"] == ["h13", "inv-11", "tests", "kind:confusion"]
    assert record["remediation"] == {"path": "pkg/mod.py", "change": "Rename f"}


def test_to_record_of_a_verdict_without_remediation_carries_none() -> None:
    record = verdict.Verdict(**_valid_fields(remediation=None)).to_record()

    assert json.loads(json.dumps(record)) == record
    assert record["remediation"] is None
