"""Tests for scripts/legibility/entry_join.py: joining a pending candidate
onto an entry that already stands (plans/census-incremental-prd.md L6, L7)."""
from __future__ import annotations

import copy

import pytest
from legibility import census_findings, codebook, entry_join, verdict

_RUN = "census-dark_factory-20261019"
_STANDING = census_findings.FindingIdentity.mint("orchestrator", "orchestrator/x.py::f", ("h13", verdict.KIND_TAG))
_JOINED_BY = {"method": "similarity", "score": 0.93, "run_id": _RUN}


def _sighting(session: str) -> dict:
    return {
        "date": "2026-07-01",
        "project": "dark_factory",
        "session": session,
        "origin_phase": "implement",
        "manifested_phase": "verify",
    }


def _entry(entry_id: str, title: str, session: str, **fields) -> dict:
    return {
        "id": entry_id,
        "title": title,
        "severity": "medium",
        "status": "open",
        "origin_phase": "implement",
        "manifested_phase": "verify",
        "sightings": [_sighting(session)],
        **fields,
    }


def _candidate(cand_id: str, title: str, disposition: str, sessions: tuple[str, ...], **fields) -> dict:
    return {
        "id": cand_id,
        "title": title,
        "first_seen": "2026-07-01",
        "disposition": disposition,
        "sightings": [_sighting(session) for session in sessions],
        **fields,
    }


def _codebook(*candidate_sessions: str) -> dict:
    """An unkeyed entry, a keyed entry sighted in sess-a with the candidate
    promoted to it, and pending cand-20261019-1 sighted in
    *candidate_sessions* (default sess-b)."""
    keyed = _entry(
        "entry-cand-20261001-1", "A title", "sess-a",
        **_STANDING.to_fields(),
        first_seen_run="census-dark_factory-20261001",
        last_seen_runs=["census-dark_factory-20261001"],
    )
    return {
        "version": 2,
        "entries": [_entry("entry-unkeyed", "Unkeyed", "sess-u"), keyed],
        "candidates": [
            _candidate(
                "cand-20261001-1", "A title", "promoted", ("sess-a",),
                promoted_to="entry-cand-20261001-1", **_STANDING.to_fields(),
            ),
            _candidate("cand-20261019-1", "B title", "pending", candidate_sessions or ("sess-b",)),
        ],
    }


def _by_id(cb: dict, record_id: str) -> dict:
    return next(record for record in (*cb["entries"], *cb["candidates"]) if record["id"] == record_id)


def _join(cb: dict, entry_id: str = "entry-cand-20261001-1", candidate_id: str = "cand-20261019-1") -> dict:
    return entry_join.join_entry(
        cb, candidate_id, entry_id,
        candidate_fields={"attached_by": dict(_JOINED_BY)},
        sighting_fields={"attached_by": dict(_JOINED_BY)},
    )


def test_join_entry_promotes_the_candidate_onto_the_entry_under_its_identity():
    cb = _codebook()
    before = copy.deepcopy(cb)

    joined = _join(cb)

    assert cb == before
    candidate = _by_id(joined, "cand-20261019-1")
    assert candidate["disposition"] == "promoted"
    assert candidate["promoted_to"] == "entry-cand-20261001-1"
    assert candidate["attached_by"] == _JOINED_BY
    assert census_findings.FindingIdentity.from_record(candidate) == _STANDING
    assert len(joined["entries"]) == len(cb["entries"])
    assert codebook.validate(joined) == []


def test_join_entry_onto_an_unkeyed_entry_stamps_no_identity():
    joined = _join(_codebook(), entry_id="entry-unkeyed")

    candidate = _by_id(joined, "cand-20261019-1")
    assert candidate["promoted_to"] == "entry-unkeyed"
    assert census_findings.FindingIdentity.from_record(candidate) is None
    assert codebook.validate(joined) == []


def test_join_entry_appends_only_the_sessions_the_entry_does_not_carry():
    joined = _join(_codebook("sess-a", "sess-b", "sess-b"))

    sightings = _by_id(joined, "entry-cand-20261001-1")["sightings"]
    assert [sighting["session"] for sighting in sightings] == ["sess-a", "sess-b"]
    assert "attached_by" not in sightings[0]
    assert sightings[1]["attached_by"] == _JOINED_BY
    assert "verifier_tags" not in sightings[1]


@pytest.mark.parametrize(
    ("candidate_id", "entry_id", "named"),
    [
        ("cand-20261019-9", "entry-cand-20261001-1", "cand-20261019-9"),
        ("cand-20261019-1", "entry-missing", "entry-missing"),
    ],
    ids=["missing-candidate", "missing-entry"],
)
def test_join_entry_names_a_missing_record(candidate_id, entry_id, named):
    with pytest.raises(ValueError, match=named):
        _join(_codebook(), entry_id=entry_id, candidate_id=candidate_id)


def test_join_entry_refuses_a_candidate_that_is_not_pending():
    with pytest.raises(ValueError, match="cand-20261001-1.*pending"):
        _join(_codebook(), candidate_id="cand-20261001-1")
