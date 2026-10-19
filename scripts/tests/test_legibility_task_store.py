"""Tests for scripts/legibility/task_store.py: the census's typed boundary over
the fused-memory task tools (plans/census-incremental-prd.md L4;
docs/quality-findings-contract.md §8). Every case drives McpTaskStore through
its injected ``post`` callable."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from legibility import task_store

_URL = "http://fused-memory.test/mcp"
_ROOT = "/abs/root"
_KEY = "fk-0123456789ab"


class _RecordingPost:
    """A fake ``post(url, tool_name, arguments, *, timeout)``. Each tool's reply
    comes from a queue (one reply per call, the last one repeating) or from a
    callable that raises."""

    def __init__(self, replies: dict[str, list[Any] | Callable[[], Any]]) -> None:
        self._replies = replies
        self.calls: list[tuple[str, str, dict, float]] = []

    def __call__(self, url: str, tool_name: str, arguments: dict, *, timeout: float) -> Any:
        self.calls.append((url, tool_name, dict(arguments), timeout))
        reply = self._replies[tool_name]
        if callable(reply):
            return reply()
        return reply.pop(0) if len(reply) > 1 else reply[0]

    def arguments_of(self, tool_name: str) -> list[dict]:
        return [arguments for _, tool, arguments, _ in self.calls if tool == tool_name]


def _store(post: _RecordingPost) -> task_store.McpTaskStore:
    return task_store.McpTaskStore(url=_URL, post=post)


def _page(rows: list[dict], *, offset: int, has_more: bool) -> dict:
    return {
        "results": rows,
        "tasks_with_key": 3,
        "pagination": {
            "total": 3,
            "offset": offset,
            "page_size": 100,
            "returned": len(rows),
            "has_more": has_more,
        },
    }


def _row(task_id: int | str, title: str, status: str) -> dict:
    return {"task_id": task_id, "title": title, "status": status, "priority": "medium", "updated_at": "x"}


def test_find_tasks_by_metadata_pages_until_has_more_is_false() -> None:
    post = _RecordingPost({
        "find_tasks_by_metadata": [
            _page([_row(5, "a", "pending"), _row(9, "b", "done")], offset=0, has_more=True),
            _page([_row("12", "c", "cancelled")], offset=2, has_more=False),
        ],
    })

    hits = _store(post).find_tasks_by_metadata(_ROOT, key="x_finding_key", value=_KEY)

    assert hits == (
        task_store.TaskHit(task_id=5, title="a", status="pending"),
        task_store.TaskHit(task_id=9, title="b", status="done"),
        task_store.TaskHit(task_id=12, title="c", status="cancelled"),
    )
    pages = post.arguments_of("find_tasks_by_metadata")
    assert [page["offset"] for page in pages] == [0, 2]
    for page in pages:
        assert page["project_root"] == _ROOT
        assert page["key"] == "x_finding_key"
        assert page["value"] == _KEY
    assert {url for url, *_ in post.calls} == {_URL}


def _error_reply() -> dict:
    return {"error": "store is down", "error_type": "ConnectionError"}


@pytest.mark.parametrize(
    ("tool", "call"),
    [
        ("find_tasks_by_metadata", lambda store: store.find_tasks_by_metadata(_ROOT, key="k", value="v")),
        ("search_tasks", lambda store: store.search_tasks(_ROOT, query="q", score_threshold=0.6)),
        ("get_task", lambda store: store.get_task(_ROOT, 42)),
        ("resolve_ticket", lambda store: store.resolve_ticket(_ROOT, "tkt_x", timeout_secs=5)),
    ],
)
def test_an_error_payload_from_any_tool_raises_store_unreachable_naming_tool_and_error(tool, call) -> None:
    post = _RecordingPost({tool: [_error_reply()]})

    with pytest.raises(task_store.StoreUnreachable) as raised:
        call(_store(post))

    assert tool in str(raised.value)
    assert "store is down" in str(raised.value)


def test_a_raising_post_becomes_store_unreachable_chaining_the_original() -> None:
    original = RuntimeError("connection refused")

    def _raise() -> Any:
        raise original

    post = _RecordingPost({"get_task": _raise})

    with pytest.raises(task_store.StoreUnreachable) as raised:
        _store(post).get_task(_ROOT, 42)

    assert "get_task" in str(raised.value)
    assert "RuntimeError" in str(raised.value)
    assert raised.value.__cause__ is original


@pytest.mark.parametrize(
    "reply",
    [
        _page([], offset=0, has_more=True),
        {"tasks_with_key": 0, "pagination": {"has_more": False, "returned": 0}},
        {"results": "not a list", "pagination": {"has_more": False, "returned": 0}},
        {"results": [{"title": "no id", "status": "pending"}], "pagination": {"has_more": False, "returned": 1}},
        ["not", "a", "dict"],
    ],
    ids=["has-more-without-progress", "missing-results", "results-not-a-list", "row-without-id", "not-a-dict"],
)
def test_a_malformed_find_reply_raises_store_unreachable_never_no_match(reply) -> None:
    post = _RecordingPost({"find_tasks_by_metadata": [reply]})

    with pytest.raises(task_store.StoreUnreachable) as raised:
        _store(post).find_tasks_by_metadata(_ROOT, key="k", value="v")

    assert "find_tasks_by_metadata" in str(raised.value)


def test_a_pager_that_never_ends_is_bounded() -> None:
    post = _RecordingPost({"find_tasks_by_metadata": [_page([_row(1, "a", "pending")], offset=0, has_more=True)]})

    with pytest.raises(task_store.StoreUnreachable):
        _store(post).find_tasks_by_metadata(_ROOT, key="k", value="v")

    assert len(post.calls) < 10_000


def test_search_tasks_forwards_the_threshold_and_parses_hits() -> None:
    post = _RecordingPost({
        "search_tasks": [{
            "results": [
                {"task_id": 7, "title": "seven", "score": 0.91, "status": "pending"},
                {"task_id": "8", "title": "gone", "score": 0.7, "status": None},
            ],
        }],
    })

    hits = _store(post).search_tasks(_ROOT, query="what was filed", score_threshold=0.6)

    assert hits == (
        task_store.SearchHit(task_id=7, title="seven", status="pending", score=0.91),
        task_store.SearchHit(task_id=8, title="gone", status=None, score=0.7),
    )
    (arguments,) = post.arguments_of("search_tasks")
    assert arguments["project_root"] == _ROOT
    assert arguments["query"] == "what was filed"
    assert arguments["score_threshold"] == 0.6


def test_get_task_posts_a_string_id_and_parses_the_record() -> None:
    post = _RecordingPost({
        "get_task": [{"id": 42, "title": "forty-two", "status": "cancelled", "metadata": {"x_acceptance_reason": "r"}}],
    })

    record = _store(post).get_task(_ROOT, 42)

    assert record == task_store.TaskRecord(
        task_id=42, title="forty-two", status="cancelled", metadata={"x_acceptance_reason": "r"},
    )
    (arguments,) = post.arguments_of("get_task")
    assert arguments == {"project_root": _ROOT, "id": "42"}


def test_get_task_without_metadata_reads_an_empty_mapping() -> None:
    post = _RecordingPost({"get_task": [{"id": "42", "title": "t", "status": "pending"}]})

    record = _store(post).get_task(_ROOT, 42)

    assert record.task_id == 42
    assert dict(record.metadata) == {}


def test_resolve_ticket_forwards_its_wait_and_outlasts_it_on_the_transport() -> None:
    post = _RecordingPost({"resolve_ticket": [{"status": "created", "task_id": 7}]})

    _store(post).resolve_ticket(_ROOT, "tkt_x", timeout_secs=50)

    ((_, _, arguments, transport_timeout),) = post.calls
    assert arguments == {"project_root": _ROOT, "ticket": "tkt_x", "timeout_seconds": 50}
    assert transport_timeout > 50


def _resolve(reply: dict) -> task_store.TicketOutcome:
    return _store(_RecordingPost({"resolve_ticket": [reply]})).resolve_ticket(_ROOT, "tkt_x", timeout_secs=5)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ({"status": "created", "task_id": "7"}, task_store.TicketOutcome(task_store.TicketStatus.CREATED, 7, None)),
        ({"status": "combined", "task_id": 9}, task_store.TicketOutcome(task_store.TicketStatus.COMBINED, 9, None)),
        ({"status": "refused", "reason": "r"}, task_store.TicketOutcome(task_store.TicketStatus.REFUSED, None, "r")),
        (
            {"status": "failed", "reason": "expired"},
            task_store.TicketOutcome(task_store.TicketStatus.FAILED, None, "expired"),
        ),
    ],
)
def test_resolve_ticket_parses_each_documented_status(reply, expected) -> None:
    assert _resolve(reply) == expected


def test_a_timed_out_wait_is_undecided_and_any_other_failure_is_not() -> None:
    assert _resolve({"status": "failed", "reason": "timeout"}).undecided is True
    assert _resolve({"status": "failed", "reason": "expired"}).undecided is False
    assert _resolve({"status": "created", "task_id": 1}).undecided is False


@pytest.mark.parametrize(
    "reply",
    [
        {"status": "pending"},
        {"status": "created"},
        {"status": "combined", "task_id": "not-a-number"},
        {"status": "created", "task_id": True},
    ],
    ids=["unknown-status", "created-without-id", "combined-with-garbage-id", "bool-id"],
)
def test_resolve_ticket_rejects_an_unusable_reply(reply) -> None:
    with pytest.raises(task_store.StoreUnreachable) as raised:
        _resolve(reply)

    assert "resolve_ticket" in str(raised.value)
