"""Tests for scripts/legibility/census_findings.py — the finding identity a
Verdict gives a codebook record, and the contract §1 row that record reports
(plans/census-incremental-prd.md leaf L3)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml
from legibility import census_findings, check_census_report, codebook, finding_area, verdict
from shared.finding_key import finding_key

_RUN = "census-dark_factory-20261010"
_OLD_RUN = "census-dark_factory-20260701"
_ANCHOR = "orchestrator/x.py::f"


def _area_map(tmp_path: Path) -> finding_area.AreaMap:
    briefing = tmp_path / "review" / "briefing.yaml"
    briefing.parent.mkdir(parents=True, exist_ok=True)
    briefing.write_text("subprojects:\n  orchestrator: {}\n  shared: {}\n", encoding="utf-8")
    return finding_area.load_area_map(tmp_path)


def _found(
    *,
    anchor: str = _ANCHOR,
    tags: tuple[str, ...] = ("h13", verdict.KIND_TAG),
    reason: str = "the verifier saw it",
    severity: str = "high",
    severity_reason: str = "it costs every run",
    remediation: verdict.Remediation | None = None,
) -> verdict.Verdict:
    return verdict.Verdict(
        verified=True,
        reason=reason,
        anchor=anchor,
        tags=tags,
        severity=severity,
        severity_reason=severity_reason,
        route=verdict.Route.MECHANICAL if remediation else verdict.Route.STRUCTURAL,
        remediation=remediation,
    )


def _identity(anchor: str = _ANCHOR, tags: tuple[str, ...] = ("h13", verdict.KIND_TAG)):
    return census_findings.FindingIdentity.mint("orchestrator", anchor, tags)


def _entry(entry_id: str, *, sessions: tuple[str, ...] = (), **fields) -> dict:
    entry = {
        "id": entry_id,
        "title": f"title of {entry_id}",
        "severity": "low",
        "status": "open",
        "origin_phase": "implement",
        "manifested_phase": "merge",
        "sightings": [
            {
                "date": "2026-10-10",
                "project": "dark_factory",
                "session": session,
                "origin_phase": "implement",
                "manifested_phase": "merge",
            }
            for session in sessions
        ],
    }
    entry.update(fields)
    return entry


def _keyed_entry(entry_id: str = "entry-a", **fields) -> dict:
    stamped = census_findings.stamped(_entry(entry_id, **fields), _identity())
    return {**stamped, **census_findings.first_sighting_fields(_OLD_RUN)}


def _candidate(cand_id: str = "cand-20261010-1") -> dict:
    return {
        "id": cand_id,
        "title": "a novel shape",
        "first_seen": "2026-10-10",
        "disposition": "rejected",
        "sightings": [],
    }


# identity_of -----------------------------------------------------------------


def test_identity_of_reads_area_anchor_and_tags_from_the_verdict(tmp_path):
    area_map = _area_map(tmp_path)
    found = _found(tags=("h4", "inv-11", verdict.KIND_TAG))
    identity = census_findings.identity_of(found, area_map)
    assert identity.area == area_map.area_of(found.anchor) == "orchestrator"
    assert identity.anchor == found.anchor
    assert identity.tags == ("h4", "inv-11", verdict.KIND_TAG)
    assert identity.key == finding_key("orchestrator", found.anchor, "h4")


def test_a_reworded_verdict_keeps_its_key(tmp_path):
    area_map = _area_map(tmp_path)
    first = _found(reason="one wording", severity="high", severity_reason="a")
    second = _found(reason="another wording", severity="low", severity_reason="b")
    assert census_findings.identity_of(first, area_map).key == census_findings.identity_of(second, area_map).key


# FindingIdentity invariants --------------------------------------------------


@pytest.mark.parametrize(
    "overrides, field, offending",
    [
        ({"key": "fk-XYZ"}, "key", "fk-XYZ"),
        ({"tags": ()}, "tags", "()"),
        ({"tags": ("h99",)}, "tags", "h99"),
        ({"area": "orchestrator/merge-queue"}, "area", "orchestrator/merge-queue"),
        ({"anchor": "orchestrator/x.py:12"}, "anchor", "orchestrator/x.py:12"),
    ],
)
def test_finding_identity_rejects_a_malformed_field(overrides, field, offending):
    fields = {"key": finding_key("orchestrator", _ANCHOR, "h13"), "area": "orchestrator",
              "anchor": _ANCHOR, "tags": ("h13", verdict.KIND_TAG)}
    fields.update(overrides)
    with pytest.raises(ValueError) as raised:
        census_findings.FindingIdentity(**fields)
    assert field in str(raised.value)
    assert offending in str(raised.value)


# from_record / to_fields -----------------------------------------------------


def test_from_record_round_trips_a_keyed_record():
    identity = _identity()
    record = {"id": "entry-a", **identity.to_fields()}
    restored = census_findings.FindingIdentity.from_record(record)
    assert restored == identity
    assert restored is not None and restored.to_fields() == identity.to_fields()


def test_from_record_of_an_unkeyed_record_is_none():
    assert census_findings.FindingIdentity.from_record(_entry("entry-a")) is None


@pytest.mark.parametrize(
    "malformed",
    [
        {"finding_area": None, "finding_anchor": None, "finding_tags": None},
        {"finding_anchor": None},
        {"finding_anchor": "orchestrator/x.py:12"},
        {"finding_tags": ["h99"]},
    ],
)
def test_from_record_refuses_exactly_the_identities_codebook_validate_rejects(malformed):
    """One invariant: an identity validate() rejects never reaches a run as a
    FindingIdentity, and from_record names the record and the field."""
    record = _keyed_entry()
    for field, value in malformed.items():
        if value is None:
            del record[field]
        else:
            record[field] = value
    errors = codebook.validate({"version": 2, "entries": [record], "candidates": []})
    assert errors and all(field in " ".join(errors) for field in malformed)
    with pytest.raises(ValueError) as raised:
        census_findings.FindingIdentity.from_record(record)
    assert "entry-a" in str(raised.value)
    assert all(field in str(raised.value) for field in malformed)


def test_to_fields_is_plain_yaml():
    fields = _identity().to_fields()
    assert set(fields) == {"finding_key", "finding_area", "finding_anchor", "finding_tags"}
    assert isinstance(fields["finding_tags"], list)
    assert yaml.safe_load(yaml.safe_dump(fields)) == fields


# stamped: contract §2's supersedes rule --------------------------------------


def test_stamping_an_unkeyed_record_adds_the_identity_without_mutating_it():
    record = _entry("entry-a")
    before = copy.deepcopy(record)
    identity = _identity()
    result = census_findings.stamped(record, identity)
    assert record == before
    assert result == {**before, **identity.to_fields()}


def test_a_record_keyed_at_the_same_location_keeps_its_standing_key():
    record = census_findings.stamped(_entry("entry-a"), _identity(tags=("h13", verdict.KIND_TAG)))
    restamped = census_findings.stamped(record, _identity(tags=("h4", verdict.KIND_TAG)))
    assert restamped == record


def test_a_renamed_symbol_gets_a_new_key_and_supersedes_the_old_one():
    prior = finding_key("orchestrator", "x.py::older", "h13")
    old = _identity(anchor="x.py::old_name")
    record = {**census_findings.stamped(_entry("entry-a"), old), "supersedes": [prior]}
    new = _identity(anchor="x.py::new_name")
    result = census_findings.stamped(record, new)
    assert result["finding_key"] == new.key != old.key
    assert result["finding_anchor"] == "x.py::new_name"
    assert result["supersedes"] == [prior, old.key]
    assert record["finding_key"] == old.key


# mark_seen -------------------------------------------------------------------


def _sighting_codebook() -> dict:
    return {
        "version": 2,
        "entries": [
            _entry("A", sessions=("s1", "s2"), last_seen_runs=["census-dark_factory-20260901"]),
            _entry("B", sessions=("s1",)),
            _entry("C", sessions=("s2",)),
        ],
        "candidates": [],
    }


def test_mark_seen_appends_the_run_to_every_entry_sighted_this_run():
    cb = _sighting_codebook()
    before = copy.deepcopy(cb)
    result, sighted = census_findings.mark_seen(cb, run_id=_RUN, sessions={"s2"})
    entries = {entry["id"]: entry for entry in result["entries"]}
    assert entries["A"]["last_seen_runs"] == ["census-dark_factory-20260901", _RUN]
    assert entries["B"] == before["entries"][1]
    assert entries["C"]["last_seen_runs"] == [_RUN]
    assert sighted == ("A", "C")
    assert cb == before
    assert codebook.validate(result) == []


def test_mark_seen_is_idempotent():
    once, _ = census_findings.mark_seen(_sighting_codebook(), run_id=_RUN, sessions={"s2"})
    twice, sighted = census_findings.mark_seen(once, run_id=_RUN, sessions={"s2"})
    assert twice == once
    assert sighted == ("A", "C")


# rows ------------------------------------------------------------------------


def _assert_oracle_clean(row: census_findings.FindingRow) -> dict:
    record = row.to_record()
    assert check_census_report.check_findings({"findings": [record]}) == []
    assert json.loads(json.dumps(record)) == record
    return record


def test_confirmed_row_reports_a_promoted_entry():
    entry = _keyed_entry(severity="medium")
    found = _found(remediation=verdict.Remediation("orchestrator/x.py", "split f"))
    record = _assert_oracle_clean(census_findings.confirmed_row(entry, found, cause="the cause"))
    assert record["id"] == "entry-a"
    assert record["key"] == entry["finding_key"]
    assert record["verdict"] == "confirmed"
    assert record["evidence_source"] == ["agent-transcripts", "present-tree"]
    assert record["disposition"] == "open"
    assert "the cause" in record["statement"]
    assert found.reason in record["statement"]
    assert record["proposal"] == "split f"
    assert record["severity"] == "medium"
    assert record["first_seen"] == _OLD_RUN
    assert record["last_seen"] == [_OLD_RUN]
    assert record["supersedes"] == []
    assert record["sub_area"] is None


def test_confirmed_row_without_a_remediation_has_no_proposal():
    record = census_findings.confirmed_row(_keyed_entry(), _found(), cause="c").to_record()
    assert record["proposal"] is None


def test_refuted_row_reports_a_rejected_candidate():
    found = _found(severity="low")
    candidate = census_findings.stamped(_candidate(), _identity())
    record = _assert_oracle_clean(census_findings.refuted_row(candidate, found, "c", run_id=_RUN))
    assert record["id"] == "cand-20261010-1"
    assert record["key"] == candidate["finding_key"]
    assert record["verdict"] == "refuted"
    assert record["disposition"] == "refuted"
    assert record["evidence_source"] == ["agent-transcripts"]
    assert record["proposal"] is None
    assert record["first_seen"] == _RUN
    assert record["last_seen"] == [_RUN]
    assert record["supersedes"] == []
    assert record["severity"] == "low"


def test_observed_row_reports_a_re_observed_entry_as_unverified():
    entry = _keyed_entry()
    record = _assert_oracle_clean(census_findings.observed_row(entry, "c"))
    assert record["id"] == "entry-a"
    assert record["verdict"] == "unverified"
    assert record["disposition"] == "open"
    assert record["evidence_source"] == ["agent-transcripts"]
    assert record["severity"] == "low"
    assert "c" in record["statement"]


def test_a_row_falls_back_to_the_title_when_there_is_no_cause():
    entry = _keyed_entry()
    record = census_findings.observed_row(entry, None).to_record()
    assert entry["title"] in record["statement"]


def test_a_rows_record_keys_are_the_published_row_fields_in_order():
    record = census_findings.observed_row(_keyed_entry(), "c").to_record()
    assert tuple(record) == census_findings.FINDING_ROW_FIELDS


# finding_rows ----------------------------------------------------------------


def test_finding_rows_reports_promotions_rejections_and_keyed_re_sightings():
    promoted = _keyed_entry("entry-promoted")
    re_sighted = _keyed_entry("entry-re-sighted")
    unkeyed = _entry("entry-unkeyed")
    rejected = census_findings.stamped(_candidate(), _identity())
    cb = {"version": 2, "entries": [promoted, re_sighted, unkeyed], "candidates": [rejected]}
    asked: list[str] = []

    def cause_of(entry):
        asked.append(entry["id"])
        return f"cause of {entry['id']}"

    rows = census_findings.finding_rows(
        cb,
        promoted=[census_findings.Adjudication("entry-promoted", _found(), "promoted cause")],
        refuted=[census_findings.Adjudication(rejected["id"], _found(), "refuted cause")],
        sighted=["entry-promoted", "entry-re-sighted", "entry-unkeyed"],
        run_id=_RUN,
        cause_of=cause_of,
    )
    assert [(row.record_id, row.verdict) for row in rows] == [
        ("entry-promoted", census_findings.FindingVerdict.CONFIRMED),
        (rejected["id"], census_findings.FindingVerdict.REFUTED),
        ("entry-re-sighted", census_findings.FindingVerdict.UNVERIFIED),
    ]
    assert asked == ["entry-re-sighted"]
    assert rows[-1].statement == "cause of entry-re-sighted"
