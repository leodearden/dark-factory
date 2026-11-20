"""Tests for scripts/legibility/census_filing.py: which codebook entries a
census run re-offers for filing (task 7095)."""
from __future__ import annotations

import copy

import pytest
from legibility import census_filing, codebook, disposition, task_store

_OWED = codebook.ENTRY_FILING_OWED
_WITHHELD = codebook.ENTRY_FILING_WITHHELD
_FILED = codebook.ENTRY_FILING_FILED


def _sighting(session: str) -> dict:
    return {
        "date": "2026-11-10",
        "project": "dark_factory",
        "session": session,
        "origin_phase": "implement",
        "manifested_phase": "verify",
    }


def _entry(entry_id: str, *, sessions: tuple[str, ...] = ("s-1",), filing: str | None = None, **fields) -> dict:
    marker = {codebook.ENTRY_FILING_KEY: filing} if filing else {}
    return {
        "id": entry_id,
        "title": f"Confusion {entry_id}",
        "severity": "medium",
        "status": "open",
        "origin_phase": "implement",
        "manifested_phase": "verify",
        "sightings": [_sighting(session) for session in sessions],
        **marker,
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
    cb = _codebook(_entry("entry-owed", filing=_OWED))

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
    cb = _codebook(_entry("entry-owed", status=status, filing=_OWED))

    queue = census_filing.filing_queue([], cb)

    assert [pending.reoffered_entry_id for pending in queue] == (["entry-owed"] if queued else [])


def test_an_owed_entry_a_fileable_cluster_lands_on_is_queued_once_as_that_cluster():
    cb = _codebook(_entry("entry-owed", filing=_OWED))
    cluster = _fileable_cluster("Confusion entry-owed")

    [pending] = census_filing.filing_queue([(cluster, "entry-owed")], cb)

    assert pending.cluster is cluster
    assert pending.verified_present is True
    assert pending.reoffered_entry_id == "entry-owed"
    assert pending.entry_id == "entry-owed"


def test_a_withheld_entry_a_fileable_cluster_lands_on_carries_its_reoffered_entry_id():
    cb = _codebook(_entry("entry-withheld", filing=_WITHHELD))
    cluster = _fileable_cluster("Confusion entry-withheld")

    [pending] = census_filing.filing_queue([(cluster, "entry-withheld")], cb)

    assert pending.cluster is cluster
    assert pending.reoffered_entry_id == "entry-withheld"
    assert pending.entry_id == "entry-withheld"


@pytest.mark.parametrize("filing", [None, _FILED], ids=["unmarked", "filed"])
def test_an_unmarked_or_filed_entry_is_not_reoffered(filing):
    cb = _codebook(_entry("entry-a", sessions=("s-1", "s-2"), filing=filing))
    cluster = _fileable_cluster("Confusion entry-a")

    assert census_filing.filing_queue([], cb) == []
    [pending] = census_filing.filing_queue([(cluster, "entry-a")], cb)
    assert pending.reoffered_entry_id is None
    assert pending.entry_id == "entry-a"


# ---------------------------------------------------------------------------
# owed_after_filing / owed_after_resolving: the owing rule
# ---------------------------------------------------------------------------

_ENTRY_ID = "entry-a"
_RUN = "census-dark_factory-20261120"


def _row(*, ticket: str | None, task_id: int | None, resolution: str | None, step: str = "curator") -> dict:
    return {
        "ticket": ticket,
        "project_root": "/p",
        "run_id": _RUN,
        "finding_key": None,
        "task_id": task_id,
        "resolution": resolution,
        "step": step,
    }


def _codebook_of(*, rows: tuple[dict, ...] = (), filing: str | None = None) -> dict:
    entry = _entry(_ENTRY_ID, filing=filing)
    if rows:
        entry["filed_tickets"] = list(rows)
    return _codebook(entry)


def _filing(entry_id: str | None = _ENTRY_ID) -> disposition.Filing:
    return disposition.Filing(
        entry_id=entry_id, finding_key=None, title="Confusion entry-a", statement="", payload={"project_root": "/p"},
    )


def _unrouted(entry_id: str | None = _ENTRY_ID) -> disposition.Decision:
    return disposition.Decision(_filing(entry_id), disposition.Step.CURATOR)


def _routed(entry_id: str | None = _ENTRY_ID) -> disposition.Decision:
    return disposition.Decision(_filing(entry_id), disposition.Step.CURATOR, ticket="tkt_9")


def _disposal(*decisions: disposition.Decision, unfiled: tuple = (), unreachable: str | None = None):
    return disposition.Disposal(decisions=tuple(decisions), unfiled=tuple(unfiled), unreachable=unreachable)


def _resolutions(status: task_store.TicketStatus, *, task_id: int | None = None, reason: str | None = None):
    outcome = task_store.TicketOutcome(status, task_id, reason)
    return disposition.Resolutions(
        resolved=(disposition.ResolvedTicket(_ENTRY_ID, "tkt_1", "/p", outcome, carried=False),),
    )


def _marker_after(transform, cb: dict) -> str | None:
    original = copy.deepcopy(cb)

    result = transform(cb)

    assert codebook.validate(result) == []
    assert cb == original, "the input is never mutated"
    return codebook.record_by_id(result["entries"], _ENTRY_ID, kind="entry").get(codebook.ENTRY_FILING_KEY)


def _after_filing(disposal: disposition.Disposal, *, dry_run: bool = False):
    return lambda cb: census_filing.owed_after_filing(cb, disposal, dry_run=dry_run)


def test_an_unmarked_entry_whose_decision_routed_nothing_is_owed():
    assert _marker_after(_after_filing(_disposal(_unrouted())), _codebook_of()) == codebook.ENTRY_FILING_OWED


def test_a_filed_entry_whose_decision_routed_nothing_is_owed():
    cb = _codebook_of(filing=_FILED)

    assert _marker_after(_after_filing(_disposal(_unrouted())), cb) == codebook.ENTRY_FILING_OWED


def test_a_withheld_entry_whose_decision_routed_nothing_stays_withheld():
    cb = _codebook_of(filing=_WITHHELD)

    assert _marker_after(_after_filing(_disposal(_unrouted())), cb) == codebook.ENTRY_FILING_WITHHELD


def test_an_entry_a_back_link_names_a_task_for_is_never_owed():
    cb = _codebook_of(rows=(_row(ticket=None, task_id=41, resolution="existing", step="key"),))

    assert _marker_after(_after_filing(_disposal(_unrouted())), cb) is None


def test_the_entry_of_an_awaiting_ticket_decision_is_never_owed():
    cb = _codebook_of(rows=(_row(ticket="tkt_0", task_id=None, resolution=None),))
    awaiting = disposition.Decision(_filing(), disposition.Step.CURATOR, awaiting_ticket="tkt_0")

    assert _marker_after(_after_filing(_disposal(awaiting)), cb) is None


def test_an_entry_another_decision_of_the_run_routed_is_never_owed():
    assert _marker_after(_after_filing(_disposal(_unrouted(), _routed())), _codebook_of()) is None


@pytest.mark.parametrize(("dry_run", "marker"), [(False, codebook.ENTRY_FILING_OWED), (True, None)])
def test_an_unfiled_filing_is_owed_except_under_a_dry_run(dry_run, marker):
    disposal = _disposal(unfiled=(_filing(),), unreachable="find_tasks_by_metadata: down")

    assert _marker_after(_after_filing(disposal, dry_run=dry_run), _codebook_of()) == marker


def test_a_filing_with_no_entry_is_ignored():
    disposal = _disposal(_unrouted(None), unfiled=(_filing(None),))

    assert _marker_after(_after_filing(disposal), _codebook_of()) is None


def _after_resolving(resolutions: disposition.Resolutions):
    return lambda cb: census_filing.owed_after_resolving(cb, resolutions)


def test_an_entry_whose_only_ticket_failed_is_owed():
    cb = _codebook_of(rows=(_row(ticket="tkt_1", task_id=None, resolution="failed"),))
    resolutions = _resolutions(task_store.TicketStatus.FAILED, reason="expired")

    assert _marker_after(_after_resolving(resolutions), cb) == codebook.ENTRY_FILING_OWED


@pytest.mark.parametrize(
    ("status", "task_id"),
    [
        (task_store.TicketStatus.CREATED, 7),
        (task_store.TicketStatus.COMBINED, 7),
        (task_store.TicketStatus.REFUSED, None),
        (task_store.TicketStatus.CANCELLED, None),
    ],
)
def test_a_ticket_the_curator_decided_never_owes_its_entry(status, task_id):
    resolution = status.value
    cb = _codebook_of(rows=(_row(ticket="tkt_1", task_id=task_id, resolution=resolution),))

    assert _marker_after(_after_resolving(_resolutions(status, task_id=task_id, reason="r")), cb) is None


def test_a_failed_ticket_never_owes_an_entry_another_row_names_a_task_for():
    cb = _codebook_of(rows=(
        _row(ticket="tkt_1", task_id=None, resolution="failed"),
        _row(ticket=None, task_id=41, resolution="existing", step="key"),
    ))
    resolutions = _resolutions(task_store.TicketStatus.FAILED, reason="expired")

    assert _marker_after(_after_resolving(resolutions), cb) is None
