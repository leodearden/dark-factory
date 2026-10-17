"""Tests for scripts/legibility/disposition.py: how a verified census finding
becomes work and how its task links back (plans/census-incremental-prd.md L4;
docs/quality-findings-contract.md §6, §7, §8)."""
from __future__ import annotations

from typing import Any

import pytest
from legibility import disposition, task_store
from legibility_task_store_fake import FakeTaskStore

_ROOT = "/abs/observed"
_RUN = "census-dark_factory-20261017"
_KEY_A = "fk-00000000000a"
_KEY_B = "fk-00000000000b"

Step = disposition.Step
Resolution = disposition.Resolution
Standing = disposition.Standing


def _filing(
    key: str | None = _KEY_A,
    title: str = "agents misread the merge queue",
    *,
    entry_id: str | None = "entry-cand-20261017-1",
    statement: str = "the queue state is not legible from the report",
    project_root: str = _ROOT,
) -> disposition.Filing:
    metadata: dict[str, Any] = {"source": disposition.CENSUS_SOURCE, "x_finding_run": _RUN}
    if key is not None:
        metadata["x_finding_key"] = key
    payload = {
        "project_root": project_root,
        "title": disposition.task_title(title),
        "description": statement,
        "task_kind": "normal",
        "priority": "medium",
        "metadata": metadata,
    }
    return disposition.Filing(entry_id=entry_id, finding_key=key, title=title, statement=statement, payload=payload)


class _Submit:
    """A fake curator-path ``submit_task``: answers ``{'ticket': 'tkt_<n>'}``
    unless told otherwise."""

    def __init__(self, reply: Any = None, *, raises: Exception | None = None) -> None:
        self._reply = reply
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **payload: Any) -> Any:
        self.calls.append(payload)
        if self._raises is not None:
            raise self._raises
        if self._reply is not None:
            return self._reply
        return {"ticket": f"tkt_{len(self.calls)}"}


def _hit(task_id: int, status: str) -> task_store.TaskHit:
    return task_store.TaskHit(task_id=task_id, title=f"task {task_id}", status=status)


def _task(task_id: int, status: str, **metadata: Any) -> task_store.TaskRecord:
    return task_store.TaskRecord(task_id=task_id, title=f"task {task_id}", status=status, metadata=metadata)


def _key_hits(*hits: task_store.TaskHit, key: str = _KEY_A) -> dict:
    return {(disposition.FINDING_KEY_METADATA, key): list(hits)}


def _accepting(task_id: int) -> task_store.TaskRecord:
    return _task(task_id, "cancelled", **{disposition.ACCEPTANCE_REASON_METADATA: "kept on purpose"})


def _only(disposal: disposition.Disposal) -> disposition.Decision:
    (decision,) = disposal.decisions
    return decision


def test_a_live_task_carrying_the_key_decides_at_step_one_and_files_nothing() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(501, "pending")))
    submit = _Submit()

    disposal = disposition.dispose([_filing()], store=store, submit=submit)

    decision = _only(disposal)
    assert submit.calls == []
    assert (decision.step, decision.resolution, decision.standing) == (Step.KEY, Resolution.EXISTING, Standing.FILED)
    assert decision.task_id == 501
    assert decision.disposition() == "filed:501"
    assert disposal.unfiled == ()
    assert disposal.unreachable is None
    assert store.arguments_of("find_tasks_by_metadata") == [
        {"project_root": _ROOT, "key": disposition.FINDING_KEY_METADATA, "value": _KEY_A},
    ]


def test_a_cancelled_task_carrying_an_acceptance_reason_is_an_acceptance() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(12, "cancelled")), tasks={12: _accepting(12)})
    submit = _Submit()

    decision = _only(disposition.dispose([_filing()], store=store, submit=submit))

    assert submit.calls == []
    assert (decision.step, decision.standing, decision.task_id) == (Step.KEY, Standing.ACCEPTED, 12)
    assert decision.disposition() == "accepted:12"
    assert "search_tasks" not in store.tools_called()


def test_a_cancelled_task_without_an_acceptance_reason_counts_as_absent() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(12, "cancelled")), tasks={12: _task(12, "cancelled")})
    submit = _Submit()

    decision = _only(disposition.dispose([_filing()], store=store, submit=submit))

    assert "search_tasks" in store.tools_called()
    assert len(submit.calls) == 1
    assert (decision.step, decision.ticket) == (Step.CURATOR, "tkt_1")


def test_a_done_task_carrying_the_key_is_linked_and_not_refiled() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(40, "done")))
    submit = _Submit()

    decision = _only(disposition.dispose([_filing()], store=store, submit=submit))

    assert submit.calls == []
    assert (decision.step, decision.standing, decision.task_id) == (Step.KEY, Standing.DONE, 40)
    link = decision.back_link(_RUN)
    assert link is not None
    assert (link.task_id, link.step, link.resolution) == (40, Step.KEY, Resolution.EXISTING)


@pytest.mark.parametrize(
    ("hits", "expected"),
    [
        ([_hit(10, "done"), _hit(12, "cancelled"), _hit(5, "pending"), _hit(7, "blocked")], (Standing.FILED, 7)),
        ([_hit(10, "done"), _hit(12, "cancelled"), _hit(3, "cancelled")], (Standing.ACCEPTED, 12)),
        ([_hit(3, "done"), _hit(9, "done"), _hit(4, "cancelled")], (Standing.DONE, 9)),
    ],
    ids=["live-wins", "accepted-over-done", "newest-done"],
)
def test_step_one_precedence_is_live_then_accepted_then_done_newest_first(hits, expected) -> None:
    store = FakeTaskStore(
        metadata_hits=_key_hits(*hits),
        tasks={12: _accepting(12), 3: _task(3, "cancelled"), 4: _task(4, "cancelled")},
    )

    decision = _only(disposition.dispose([_filing()], store=store, submit=_Submit()))

    assert (decision.standing, decision.task_id) == expected


def test_an_unreachable_step_one_lookup_files_nothing_at_all() -> None:
    unreachable = task_store.StoreUnreachable("find_tasks_by_metadata: ConnectionError: down")
    store = FakeTaskStore(metadata_hits={(disposition.FINDING_KEY_METADATA, _KEY_B): unreachable})
    submit = _Submit()
    filings = [_filing(_KEY_A, "first"), _filing(_KEY_B, "second")]

    disposal = disposition.dispose(filings, store=store, submit=submit)

    assert submit.calls == []
    assert disposal.decisions == ()
    assert disposal.unfiled == tuple(filings)
    assert disposal.unreachable is not None
    assert "ConnectionError: down" in disposal.unreachable


def test_an_unreachable_acceptance_read_back_is_an_unreachable_step_one() -> None:
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(12, "cancelled")))
    submit = _Submit()

    disposal = disposition.dispose([_filing()], store=store, submit=submit)

    assert submit.calls == []
    assert disposal.unreachable is not None
    assert "get_task" in disposal.unreachable


def test_step_two_finds_a_live_task_by_meaning() -> None:
    filing = _filing()
    store = FakeTaskStore(
        search_hits={filing.title: [task_store.SearchHit(task_id=88, title="close", status="pending", score=0.8)]},
        tasks={88: _task(88, "pending")},
    )
    submit = _Submit()

    decision = _only(disposition.dispose([filing], store=store, submit=submit))

    assert submit.calls == []
    assert (decision.step, decision.resolution, decision.task_id) == (Step.SEMANTIC, Resolution.EXISTING, 88)
    (search,) = store.arguments_of("search_tasks")
    assert search["score_threshold"] == disposition.SEMANTIC_SCORE_THRESHOLD == 0.6
    assert filing.title in search["query"]
    assert store.arguments_of("get_task") == [{"project_root": _ROOT, "task_id": 88}]


def test_step_two_reads_back_the_top_hit_and_files_when_it_is_done() -> None:
    filing = _filing()
    store = FakeTaskStore(
        search_hits={filing.title: [
            task_store.SearchHit(task_id=88, title="close", status="pending", score=0.7),
            task_store.SearchHit(task_id=89, title="closer", status="pending", score=0.9),
        ]},
        tasks={88: _task(88, "pending"), 89: _task(89, "done")},
    )
    submit = _Submit()

    decision = _only(disposition.dispose([filing], store=store, submit=submit))

    assert store.arguments_of("get_task") == [{"project_root": _ROOT, "task_id": 89}]
    assert len(submit.calls) == 1
    assert decision.step is Step.CURATOR


def test_an_unreachable_step_two_goes_to_the_curator_and_says_so() -> None:
    store = FakeTaskStore(unreachable=["search_tasks"])
    submit = _Submit()

    decision = _only(disposition.dispose([_filing()], store=store, submit=submit))

    assert len(submit.calls) == 1
    assert decision.step is Step.CURATOR
    assert decision.note is not None
    assert "search_tasks" in decision.note


def test_step_three_files_through_the_curator_and_holds_the_pending_ticket() -> None:
    submit = _Submit({"ticket": "tkt_9"})
    filing = _filing()

    decision = _only(disposition.dispose([filing], store=FakeTaskStore(), submit=submit))

    assert submit.calls == [dict(filing.payload)]
    assert (decision.step, decision.ticket, decision.resolution, decision.task_id) == (
        Step.CURATOR, "tkt_9", None, None,
    )
    assert "tkt_9" in decision.disposition()
    link = decision.back_link(_RUN)
    assert link is not None
    assert (link.ticket, link.task_id, link.resolution, link.step) == ("tkt_9", None, None, Step.CURATOR)


@pytest.mark.parametrize(
    "submit",
    [_Submit(raises=RuntimeError("curator down")), _Submit({"error": "no", "error_type": "ValidationError"})],
    ids=["raises", "no-ticket"],
)
def test_a_submit_that_yields_no_ticket_is_a_decision_without_a_link(submit) -> None:
    decision = _only(disposition.dispose([_filing()], store=FakeTaskStore(), submit=submit))

    assert decision.step is Step.CURATOR
    assert decision.ticket is None
    assert decision.note
    assert decision.back_link(_RUN) is None


def test_dry_run_stops_before_the_curator() -> None:
    live = _filing(_KEY_A, "live")
    fresh = _filing(_KEY_B, "fresh")
    store = FakeTaskStore(metadata_hits=_key_hits(_hit(501, "pending"), key=_KEY_A))

    disposal = disposition.dispose([live, fresh], store=store, submit=None)

    assert [decision.filing for decision in disposal.decisions] == [live]
    assert disposal.unfiled == (fresh,)
    assert disposal.unreachable is None


def test_an_unkeyed_filing_skips_the_key_lookup() -> None:
    store = FakeTaskStore()

    decision = _only(disposition.dispose([_filing(None)], store=store, submit=_Submit()))

    assert "find_tasks_by_metadata" not in store.tools_called()
    assert decision.step is Step.CURATOR


# The submit_task seam's contract (fused_memory/server/tools.py::submit_task,
# fused_memory/middleware/task_interceptor.py::TaskInterceptor): success is
# {"ticket": "tkt_<id>"}, a TICKET id; there is no "id" key on this path.

def test_ticket_id_from_submit_result_reads_the_curator_ticket_key() -> None:
    assert disposition.ticket_id_from_submit_result({"ticket": "tkt_42"}) == "tkt_42"


def test_ticket_id_from_submit_result_rejects_the_error_shape() -> None:
    assert disposition.ticket_id_from_submit_result({"error": "boom", "error_type": "ValueError"}) is None


def test_ticket_id_from_submit_result_rejects_non_dict_results() -> None:
    assert disposition.ticket_id_from_submit_result(None) is None
    assert disposition.ticket_id_from_submit_result("tkt_1") is None
    assert disposition.ticket_id_from_submit_result(["tkt_1"]) is None


def test_ticket_id_from_submit_result_rejects_empty_dict() -> None:
    assert disposition.ticket_id_from_submit_result({}) is None


def test_ticket_id_from_submit_result_rejects_unusable_ticket_values() -> None:
    assert disposition.ticket_id_from_submit_result({"ticket": ""}) is None
    assert disposition.ticket_id_from_submit_result({"ticket": None}) is None
    assert disposition.ticket_id_from_submit_result({"ticket": {"id": 1}}) is None
    assert disposition.ticket_id_from_submit_result({"ticket": 42}) is None
