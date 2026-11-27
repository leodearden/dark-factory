"""The call watcher B1 proves default_project_root unused with: what it counts, and what it leaves as it found it."""

import sys
import threading
from types import FrameType
from typing import Any

import pytest
from _call_watch import watching
from shared.project_layout import default_project_root


def target() -> None:
    pass


def other() -> None:
    pass


def test_direct_calls_inside_the_block_are_counted() -> None:
    with watching(target) as calls:
        target()
        target()
        other()

    assert calls.count == 2


def test_a_call_through_another_binding_is_counted() -> None:
    alias = target

    with watching(target) as calls:
        alias()

    assert calls.count == 1


def test_a_call_on_a_thread_started_inside_the_block_is_counted() -> None:
    with watching(target) as calls:
        thread = threading.Thread(target=target)
        thread.start()
        thread.join()

    assert calls.count == 1


def test_calls_before_and_after_the_block_are_not_counted() -> None:
    target()
    with watching(target) as calls:
        pass
    target()

    assert calls.count == 0


def _profiler(frame: FrameType, event: str, arg: Any) -> None:
    pass


def test_the_previous_profilers_are_restored_on_exit() -> None:
    previous_sys, previous_threading = sys.getprofile(), threading.getprofile()
    sys.setprofile(_profiler)
    threading.setprofile(_profiler)
    try:
        with watching(target):
            target()

        assert sys.getprofile() is _profiler
        assert threading.getprofile() is _profiler
    finally:
        sys.setprofile(previous_sys)
        threading.setprofile(previous_threading)


def test_the_previous_profilers_are_restored_when_the_block_raises() -> None:
    previous_sys, previous_threading = sys.getprofile(), threading.getprofile()

    with pytest.raises(RuntimeError), watching(target):
        raise RuntimeError('boom')

    assert sys.getprofile() is previous_sys
    assert threading.getprofile() is previous_threading


def test_a_watch_of_default_project_root_sees_a_call_to_it() -> None:
    with watching(default_project_root) as calls:
        default_project_root()

    assert calls.count == 1
