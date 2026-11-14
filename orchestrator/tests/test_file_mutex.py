"""``orchestrator.file_mutex.exclusive``: a reentrant lock honoured across threads and processes.

Every wait in this module is bounded, so a regression fails instead of hanging
the xdist worker.
"""
from __future__ import annotations

import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from orchestrator.file_mutex import FileMutexTimeout, exclusive, held

_SRC = Path(__file__).resolve().parents[1] / 'src'
_BOUND_SECS = 10.0
_SHORT_SECS = 0.05

_CHILD_ATTEMPT = (
    'import sys\n'
    'sys.path.insert(0, sys.argv[1])\n'
    'from pathlib import Path\n'
    'from orchestrator.file_mutex import exclusive\n'
    'with exclusive(Path(sys.argv[2]), timeout=0.1):\n'
    '    pass\n'
)


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    return tmp_path / 'x.mutex'


@contextmanager
def _held_on_another_thread(lock_path: Path) -> Iterator[None]:
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with exclusive(lock_path, timeout=5):
            held.set()
            release.wait(_BOUND_SECS)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert held.wait(_BOUND_SECS), 'the holder thread never acquired the lock'
    try:
        yield
    finally:
        release.set()
        thread.join(_BOUND_SECS)
    assert not thread.is_alive(), 'the holder thread never released the lock'


def _attempt_on_another_thread(lock_path: Path) -> BaseException | None:
    """Try a short acquisition on a fresh thread; the error it raised, or None."""
    outcome: list[BaseException | None] = []

    def attempt() -> None:
        try:
            with exclusive(lock_path, timeout=_SHORT_SECS):
                pass
        except BaseException as exc:
            outcome.append(exc)
        else:
            outcome.append(None)

    thread = threading.Thread(target=attempt, daemon=True)
    thread.start()
    thread.join(_BOUND_SECS)
    assert outcome, 'the attempt neither acquired the lock nor timed out'
    return outcome[0]


def _is_excluded_elsewhere(lock_path: Path) -> bool:
    return isinstance(_attempt_on_another_thread(lock_path), FileMutexTimeout)


def _held_on_a_fresh_thread(lock_path: Path) -> bool:
    answers: list[bool] = []
    thread = threading.Thread(target=lambda: answers.append(held(lock_path)), daemon=True)
    thread.start()
    thread.join(_BOUND_SECS)
    assert answers, 'the fresh thread never answered'
    return answers[0]


def _child_attempt(lock_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-c', _CHILD_ATTEMPT, str(_SRC), str(lock_path)],
        capture_output=True, text=True, timeout=30,
    )


def test_a_lock_held_by_another_thread_excludes_this_one(lock_path: Path) -> None:
    with (
        _held_on_another_thread(lock_path),
        pytest.raises(FileMutexTimeout),
        exclusive(lock_path, timeout=_SHORT_SECS),
    ):
        pass

    with exclusive(lock_path, timeout=_SHORT_SECS):
        assert _is_excluded_elsewhere(lock_path)


def test_a_nested_acquisition_on_one_thread_enters_at_once_and_keeps_the_lock(
    lock_path: Path,
) -> None:
    with exclusive(lock_path, timeout=5):
        with exclusive(lock_path, timeout=_SHORT_SECS):
            assert _is_excluded_elsewhere(lock_path)
        assert _is_excluded_elsewhere(lock_path), 'the inner exit released the outer hold'

    assert _attempt_on_another_thread(lock_path) is None


def test_a_lock_held_by_this_process_excludes_another_process(lock_path: Path) -> None:
    with exclusive(lock_path, timeout=5):
        while_held = _child_attempt(lock_path)
    after_release = _child_attempt(lock_path)

    assert while_held.returncode != 0
    assert 'FileMutexTimeout' in while_held.stderr, while_held.stderr
    assert after_release.returncode == 0, after_release.stderr


def test_an_exception_in_the_body_propagates_and_releases_the_lock(lock_path: Path) -> None:
    with pytest.raises(ValueError, match='boom'), exclusive(lock_path, timeout=5):
        raise ValueError('boom')

    assert _attempt_on_another_thread(lock_path) is None
    with exclusive(lock_path, timeout=_SHORT_SECS):
        assert _is_excluded_elsewhere(lock_path), 'a stale reentrancy depth survived'


def test_the_timeout_names_the_lock_and_the_wait(lock_path: Path) -> None:
    with (
        _held_on_another_thread(lock_path),
        pytest.raises(FileMutexTimeout) as caught,
        exclusive(lock_path, timeout=_SHORT_SECS),
    ):
        pass

    error = caught.value
    assert not isinstance(error, OSError), 'an `except OSError` would read a lock timeout as an IO failure'
    assert error.lock_path == lock_path
    assert error.timeout == _SHORT_SECS
    assert str(lock_path) in str(error)
    assert str(_SHORT_SECS) in str(error)


def test_a_missing_directory_is_not_created(tmp_path: Path) -> None:
    absent = tmp_path / 'absent'

    with pytest.raises(FileNotFoundError), exclusive(absent / 'x.mutex', timeout=_SHORT_SECS):
        pass

    assert not absent.exists()


def test_held_answers_for_the_calling_thread_at_every_depth(lock_path: Path) -> None:
    assert not held(lock_path)
    with exclusive(lock_path, timeout=5):
        assert held(lock_path)
        with exclusive(lock_path, timeout=5):
            assert held(lock_path)
        assert held(lock_path), 'the inner exit cleared the outer hold'
        assert not _held_on_a_fresh_thread(lock_path), 'another thread inherited this hold'

    assert not held(lock_path)


def test_a_lock_held_by_another_thread_is_not_held_by_this_one(lock_path: Path) -> None:
    with _held_on_another_thread(lock_path):
        assert not held(lock_path)
