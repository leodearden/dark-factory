"""``EventProbes.owner_is_shared_host``: is the event's claude a shared ``claude agents`` host?

The answer is read through the event's ``ProcessTree`` and memoized with its
walk, like every other probe.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator

import pytest
from _process_tree_fakes import claude_process_tree
from _shared_host_fixtures import REQUIRES_PROC, running_shared_host

from orchestrator.session_ownership_probes import EventProbes


@pytest.fixture
def shared_host_pid(tmp_path_factory: pytest.TempPathFactory) -> Iterator[int]:
    with running_shared_host(tmp_path_factory.mktemp('agents-host')) as pid:
        yield pid


def _recording_host_reader(answer: bool) -> tuple[Callable[[int], bool], list[int]]:
    """A host reader answering *answer*, plus the pids it was asked about."""
    asked: list[int] = []

    def _reader(pid: int) -> bool:
        asked.append(pid)
        return answer

    return _reader, asked


@REQUIRES_PROC
def test_the_default_reader_recognises_a_live_host_process(shared_host_pid: int) -> None:
    probes = EventProbes({}, claude_process_tree(shared_host_pid))

    assert probes.owner_is_shared_host() is True


def test_an_unresolvable_owning_claude_is_not_a_host_and_asks_no_reader() -> None:
    reader, asked = _recording_host_reader(True)
    tree = dataclasses.replace(claude_process_tree(None), is_shared_session_host=reader)

    assert EventProbes({}, tree).owner_is_shared_host() is False
    assert asked == []


def test_the_host_answer_is_read_once_per_event() -> None:
    reader, asked = _recording_host_reader(True)
    tree = dataclasses.replace(claude_process_tree(4242), is_shared_session_host=reader)
    probes = EventProbes({}, tree)

    assert probes.owner_is_shared_host() is True
    assert probes.owner_is_shared_host() is True
    assert asked == [4242]
