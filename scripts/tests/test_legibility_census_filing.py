"""Tests for scripts/legibility/census_filing.py: which codebook entries a
census run re-offers for filing (task 7095)."""
from __future__ import annotations

import pytest
from legibility import census_filing, codebook

_OWED = {codebook.ENTRY_FILING_KEY: codebook.ENTRY_FILING_OWED}
_WITHHELD = {codebook.ENTRY_FILING_KEY: codebook.ENTRY_FILING_WITHHELD}
_FILED = {codebook.ENTRY_FILING_KEY: codebook.ENTRY_FILING_FILED}


def _sighting(session: str) -> dict:
    return {
        "date": "2026-11-10",
        "project": "dark_factory",
        "session": session,
        "origin_phase": "implement",
        "manifested_phase": "verify",
    }


def _entry(entry_id: str, *, sessions: tuple[str, ...] = ("s-1",), **fields) -> dict:
    return {
        "id": entry_id,
        "title": f"Confusion {entry_id}",
        "severity": "medium",
        "status": "open",
        "origin_phase": "implement",
        "manifested_phase": "verify",
        "sightings": [_sighting(session) for session in sessions],
        **fields,
    }


def _codebook(*entries: dict) -> dict:
    return {"version": 2, "entries": list(entries), "candidates": []}


def _fileable_cluster(title: str) -> dict:
    return {"title": title, "summary": title, "sightings": [_sighting("s-new")]}


# ---------------------------------------------------------------------------
# filing_queue: owed entries are re-offered
# ---------------------------------------------------------------------------

def test_the_queue_reoffers_an_owed_singleton_unconditionally():
    cb = _codebook(_entry("entry-owed", **_OWED))

    [pending] = census_filing.filing_queue([], cb)

    assert pending.reoffered_entry_id == "entry-owed"
    assert pending.entry_id == "entry-owed"
    assert pending.verified_present is False
    assert pending.cluster["title"] == "Confusion entry-owed"


@pytest.mark.parametrize(
    ("status", "queued"),
    [("open", True), ("partially", True), ("fixed", False), ("retired", False), ("accepted", False)],
)
def test_the_queue_reoffers_an_owed_entry_only_while_it_is_open_or_partially(status, queued):
    cb = _codebook(_entry("entry-owed", status=status, **_OWED))

    queue = census_filing.filing_queue([], cb)

    assert [pending.reoffered_entry_id for pending in queue] == (["entry-owed"] if queued else [])


def test_an_owed_entry_a_fileable_cluster_lands_on_is_queued_once_as_that_cluster():
    cb = _codebook(_entry("entry-owed", **_OWED))
    cluster = _fileable_cluster("Confusion entry-owed")

    [pending] = census_filing.filing_queue([(cluster, "entry-owed")], cb)

    assert pending.cluster is cluster
    assert pending.verified_present is True
    assert pending.reoffered_entry_id == "entry-owed"
    assert pending.entry_id == "entry-owed"


def test_a_withheld_entry_a_fileable_cluster_lands_on_carries_its_reoffered_entry_id():
    cb = _codebook(_entry("entry-withheld", **_WITHHELD))
    cluster = _fileable_cluster("Confusion entry-withheld")

    [pending] = census_filing.filing_queue([(cluster, "entry-withheld")], cb)

    assert pending.cluster is cluster
    assert pending.reoffered_entry_id == "entry-withheld"
    assert pending.entry_id == "entry-withheld"


@pytest.mark.parametrize("marker", [{}, _FILED], ids=["unmarked", "filed"])
def test_an_unmarked_or_filed_entry_is_not_reoffered(marker):
    cb = _codebook(_entry("entry-a", sessions=("s-1", "s-2"), **marker))
    cluster = _fileable_cluster("Confusion entry-a")

    assert census_filing.filing_queue([], cb) == []
    [pending] = census_filing.filing_queue([(cluster, "entry-a")], cb)
    assert pending.reoffered_entry_id is None
    assert pending.entry_id == "entry-a"
