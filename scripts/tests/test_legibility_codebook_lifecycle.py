"""Tests for scripts/legibility/codebook_lifecycle.py: the codebook's
legal-transition table and the census lifecycle transforms built on it."""
from __future__ import annotations

import copy
import re

import pytest
from legibility import census_findings, codebook, codebook_lifecycle, verdict
from shared.finding_key import finding_key

T = codebook_lifecycle.Transition

_LIVE_STATUSES = ("open", "partially", "fixed", "mined-unverified", "accepted")

_EXPECTED_TABLE = {
    T.PROMOTE: ("candidate", "disposition", {"pending"}, "promoted"),
    T.REJECT: ("candidate", "disposition", {"pending"}, "rejected"),
    T.RETIRE: ("entry", "status", set(_LIVE_STATUSES), "retired"),
    T.ACCEPT: ("entry", "status", set(_LIVE_STATUSES), "accepted"),
    T.FIX: ("entry", "status", set(_LIVE_STATUSES), "fixed"),
    T.REOPEN: ("entry", "status", {"accepted", "fixed"}, "open"),
    T.WITHHOLD: ("entry", "filing", {None}, "withheld"),
    T.FILE: ("entry", "filing", {"withheld"}, "filed"),
}

_VOCABULARY = {
    "disposition": tuple(codebook.DISPOSITIONS),
    "status": tuple(codebook.STATUSES),
    "filing": (None, *codebook.ENTRY_FILINGS),
}


def _record(kind: str, field: str, state: str | None) -> dict:
    record = {"id": "cand-20261001-1" if kind == "candidate" else "entry-a", "title": "t"}
    if state is not None:
        record[field] = state
    return record


def _cases(*, legal: bool) -> list:
    return [
        pytest.param(transition, state, id=f"{transition}-from-{state}")
        for transition, (_, field, sources, _) in _EXPECTED_TABLE.items()
        for state in _VOCABULARY[field]
        if (state in sources) == legal
    ]


def test_the_table_states_every_transition_once():
    assert {
        transition: (rule.kind, rule.field, set(rule.sources), rule.target)
        for transition, rule in codebook_lifecycle.TABLE.items()
    } == _EXPECTED_TABLE
    assert set(codebook_lifecycle.TABLE) == set(T)


def test_every_target_is_of_its_fields_vocabulary():
    for rule in codebook_lifecycle.TABLE.values():
        assert rule.target in _VOCABULARY[rule.field]
        assert rule.sources <= set(_VOCABULARY[rule.field])


@pytest.mark.parametrize(("transition", "state"), _cases(legal=True))
def test_a_legal_transition_writes_its_target(transition, state):
    kind, field, _, target = _EXPECTED_TABLE[transition]
    record = _record(kind, field, state)

    codebook_lifecycle.transit(record, transition)

    assert record[field] == target


@pytest.mark.parametrize(("transition", "state"), _cases(legal=False))
def test_an_illegal_source_is_refused_naming_everything_a_reader_needs(transition, state):
    kind, field, sources, _ = _EXPECTED_TABLE[transition]
    record = _record(kind, field, state)
    before = copy.deepcopy(record)

    with pytest.raises(codebook_lifecycle.IllegalTransition) as raised:
        codebook_lifecycle.transit(record, transition)

    message = str(raised.value)
    for fragment in (str(transition), kind, record["id"], field, repr(state), *map(repr, sources)):
        assert fragment in message
    assert record == before


def test_an_illegal_transition_is_a_value_error():
    assert issubclass(codebook_lifecycle.IllegalTransition, ValueError)


# ---------------------------------------------------------------------------
# The census transforms
# ---------------------------------------------------------------------------

def _minimal_v2_codebook() -> dict:
    return {
        "version": 2,
        "entries": [
            {
                "id": "entry-a",
                "title": "Some confusion cluster",
                "severity": "high",
                "status": "open",
                "origin_phase": "implement",
                "manifested_phase": "merge",
                "sightings": [],
            }
        ],
        "candidates": [],
    }


def _codebook_with_candidate(disposition: str = "pending") -> dict:
    cb = _minimal_v2_codebook()
    cb["candidates"] = [
        {
            "id": "cand-20260714-1",
            "title": "A novel confusion shape",
            "first_seen": "2026-07-14",
            "disposition": disposition,
            "sightings": [
                {
                    "date": "2026-07-14",
                    "project": "dark_factory",
                    "session": "sess-1",
                    "origin_phase": "implement",
                    "manifested_phase": "verify",
                },
            ],
        }
    ]
    if disposition == "promoted":
        cb["candidates"][0]["promoted_to"] = "entry-a"
    return cb


def _codebook_with_withheld_entry() -> dict:
    cb = _minimal_v2_codebook()
    cb["entries"].append(
        {
            "id": "entry-withheld",
            "title": "A withheld confusion",
            "severity": "medium",
            "status": "open",
            "origin_phase": "implement",
            "manifested_phase": "verify",
            "sightings": [],
            codebook.ENTRY_FILING_KEY: codebook.ENTRY_FILING_WITHHELD,
        }
    )
    return cb


_PROMOTION_ENTRY_FIELDS = {
    "id": "new-entry-a",
    "title": "A novel confusion shape",
    "severity": "medium",
    "status": "open",
    "origin_phase": "implement",
    "manifested_phase": "verify",
}


def test_promote_candidate_appends_entry_and_stamps_disposition():
    before = _codebook_with_candidate()

    result = codebook_lifecycle.promote_candidate(before, "cand-20260714-1", dict(_PROMOTION_ENTRY_FIELDS))

    new_entry = next(e for e in result["entries"] if e["id"] == "new-entry-a")
    assert new_entry["sightings"] == before["candidates"][0]["sightings"]
    candidate = next(c for c in result["candidates"] if c["id"] == "cand-20260714-1")
    assert candidate["disposition"] == "promoted"
    assert candidate.get("promoted_to") == "new-entry-a", "must name the new entry id"
    assert len(result["candidates"]) == len(before["candidates"]), "candidate is RETAINED"
    assert codebook.validate(result) == []
    codebook.assert_no_deletion(before, result)
    assert before["candidates"][0]["disposition"] == "pending"
    assert len(before["entries"]) == 1


def _promotion_identity(anchor: str, primary_tag: str = "h13") -> census_findings.FindingIdentity:
    return census_findings.FindingIdentity.mint("orchestrator", anchor, (primary_tag, verdict.KIND_TAG))


def _promoted_pair(cb: dict, finding: census_findings.FindingIdentity) -> tuple[dict, dict]:
    result = codebook_lifecycle.promote_candidate(
        cb, "cand-20260714-1", dict(_PROMOTION_ENTRY_FIELDS), finding=finding
    )
    assert codebook.validate(result) == []
    return result["candidates"][0], next(e for e in result["entries"] if e["id"] == "new-entry-a")


def _identity_and_supersedes(record: dict) -> dict:
    return {field: record[field] for field in (*codebook.FINDING_IDENTITY_FIELDS, "supersedes")}


def test_a_promoted_entry_carries_its_unkeyed_candidates_new_identity():
    finding = _promotion_identity("orchestrator/x.py::f")
    candidate, entry = _promoted_pair(_codebook_with_candidate(), finding)
    assert _identity_and_supersedes(entry) == {**finding.to_fields(), "supersedes": []}
    assert candidate["finding_key"] == entry["finding_key"]


def test_a_promoted_entry_keeps_the_key_its_candidate_already_stands_on():
    cb = _codebook_with_candidate()
    standing = _promotion_identity("orchestrator/x.py::f", primary_tag="h13")
    cb["candidates"][0].update(standing.to_fields())
    candidate, entry = _promoted_pair(cb, _promotion_identity("orchestrator/x.py::f", primary_tag="h4"))
    assert entry["finding_key"] == candidate["finding_key"] == standing.key
    assert _identity_and_supersedes(entry) == _identity_and_supersedes({**candidate, "supersedes": []})


def test_a_promoted_entry_carries_the_supersedes_chain_its_moved_candidate_records():
    cb = _codebook_with_candidate()
    prior = finding_key("orchestrator", "orchestrator/x.py::older", "h13")
    standing = _promotion_identity("orchestrator/x.py::old_name")
    cb["candidates"][0].update({**standing.to_fields(), "supersedes": [prior]})
    moved = _promotion_identity("orchestrator/x.py::new_name")
    candidate, entry = _promoted_pair(cb, moved)
    assert entry["finding_key"] == candidate["finding_key"] == moved.key
    assert entry["supersedes"] == candidate["supersedes"] == [prior, standing.key]


@pytest.mark.parametrize(("withheld", "marker"), [(True, codebook.ENTRY_FILING_WITHHELD), (False, None)])
def test_a_promotion_marks_its_entry_withheld_only_when_held(withheld, marker):
    result = codebook_lifecycle.promote_candidate(
        _codebook_with_candidate(), "cand-20260714-1", dict(_PROMOTION_ENTRY_FIELDS), withheld=withheld
    )

    entry = next(e for e in result["entries"] if e["id"] == "new-entry-a")
    assert entry.get(codebook.ENTRY_FILING_KEY) == marker
    assert codebook.validate(result) == []


def test_reject_candidate_stamps_disposition_retained():
    before = _codebook_with_candidate()

    result = codebook_lifecycle.reject_candidate(before, "cand-20260714-1")

    candidate = next(c for c in result["candidates"] if c["id"] == "cand-20260714-1")
    assert candidate["disposition"] == "rejected"
    assert len(result["candidates"]) == len(before["candidates"]), "candidate is RETAINED"
    assert codebook.validate(result) == []
    codebook.assert_no_deletion(before, result)
    assert before["candidates"][0]["disposition"] == "pending"


def test_retire_entry_sets_status_retained():
    before = _minimal_v2_codebook()

    result = codebook_lifecycle.retire_entry(before, "entry-a")

    entry = next(e for e in result["entries"] if e["id"] == "entry-a")
    assert entry["status"] == "retired"
    assert len(result["entries"]) == len(before["entries"]), "entry is RETAINED"
    assert codebook.validate(result) == []
    codebook.assert_no_deletion(before, result)
    assert before["entries"][0]["status"] == "open"


def test_mark_entry_filed_flips_the_marker_retained():
    before = _codebook_with_withheld_entry()

    result = codebook_lifecycle.mark_entry_filed(before, "entry-withheld")

    entry = next(e for e in result["entries"] if e["id"] == "entry-withheld")
    assert entry[codebook.ENTRY_FILING_KEY] == codebook.ENTRY_FILING_FILED
    assert len(result["entries"]) == len(before["entries"]), "entry is RETAINED"
    assert next(e for e in result["entries"] if e["id"] == "entry-a") == before["entries"][0]
    assert codebook.validate(result) == []
    codebook.assert_no_deletion(before, result)
    assert before["entries"][1][codebook.ENTRY_FILING_KEY] == codebook.ENTRY_FILING_WITHHELD, (
        "the input is never mutated"
    )


def test_mark_entry_filed_rejects_an_unknown_entry_id():
    with pytest.raises(ValueError, match="entry-nope"):
        codebook_lifecycle.mark_entry_filed(_codebook_with_withheld_entry(), "entry-nope")


_TRANSFORMS = [
    pytest.param(
        _codebook_with_candidate,
        lambda cb: codebook_lifecycle.promote_candidate(cb, "cand-20260714-1", dict(_PROMOTION_ENTRY_FIELDS)),
        id="promote",
    ),
    pytest.param(
        _codebook_with_candidate,
        lambda cb: codebook_lifecycle.reject_candidate(cb, "cand-20260714-1"),
        id="reject",
    ),
    pytest.param(_minimal_v2_codebook, lambda cb: codebook_lifecycle.retire_entry(cb, "entry-a"), id="retire"),
    pytest.param(
        _codebook_with_withheld_entry,
        lambda cb: codebook_lifecycle.mark_entry_filed(cb, "entry-withheld"),
        id="file",
    ),
]


@pytest.mark.parametrize(("build", "transform"), _TRANSFORMS)
def test_every_transform_returns_a_new_valid_codebook_and_leaves_its_input(build, transform):
    before = build()
    original = copy.deepcopy(before)

    result = transform(before)

    assert result is not before
    assert before == original
    assert codebook.validate(result) == []
    codebook.assert_no_deletion(before, result)


@pytest.mark.parametrize(
    ("build", "transform", "transition", "record_id"),
    [
        pytest.param(
            lambda: _codebook_with_candidate("promoted"),
            lambda cb: codebook_lifecycle.reject_candidate(cb, "cand-20260714-1"),
            T.REJECT,
            "cand-20260714-1",
            id="reject-a-promoted-candidate",
        ),
        pytest.param(
            lambda: _codebook_with_candidate("rejected"),
            lambda cb: codebook_lifecycle.promote_candidate(cb, "cand-20260714-1", dict(_PROMOTION_ENTRY_FIELDS)),
            T.PROMOTE,
            "cand-20260714-1",
            id="promote-a-rejected-candidate",
        ),
        pytest.param(
            lambda: codebook_lifecycle.retire_entry(_minimal_v2_codebook(), "entry-a"),
            lambda cb: codebook_lifecycle.retire_entry(cb, "entry-a"),
            T.RETIRE,
            "entry-a",
            id="retire-a-retired-entry",
        ),
        pytest.param(
            _minimal_v2_codebook,
            lambda cb: codebook_lifecycle.mark_entry_filed(cb, "entry-a"),
            T.FILE,
            "entry-a",
            id="file-an-entry-never-withheld",
        ),
    ],
)
def test_a_transform_refuses_an_illegal_source(build, transform, transition, record_id):
    before = build()
    original = copy.deepcopy(before)

    with pytest.raises(codebook_lifecycle.IllegalTransition, match=rf"{transition}.*{re.escape(record_id)}"):
        transform(before)

    assert before == original
