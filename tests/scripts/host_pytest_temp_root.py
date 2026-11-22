"""A stand-in for the host's default pytest temp root, seeded with a dead session's dirs, and a recorder witnessing that a child ran against it, for contracts that a nested session leaves that root untouched (tasks 7222, 7228, 7229, 7143)."""
from __future__ import annotations

import getpass
import os
import time
from pathlib import Path

#: Older than pytest's 3-day LOCK_TIMEOUT, so pytest treats the lock as dead.
DEAD_LOCK_AGE_SECS = 5 * 24 * 60 * 60

TEMPROOT_RECORDER_TEST = 'test_records_the_temproot'
_TEMPROOT_RECORD_NAME = 'seen-temproot'


def listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob('*'))


def seed_dead_session_dirs(host: Path) -> None:
    """``pytest-0`` holds a dead session's leftovers and lock; ``pytest-1..3``
    make it a cleanup candidate as soon as a new session adds ``pytest-4``, and
    ``garbage-*`` is swept unconditionally."""
    user_root = host / f'pytest-of-{getpass.getuser()}'
    (user_root / 'pytest-0' / 'leftover').mkdir(parents=True)
    lock = user_root / 'pytest-0' / '.lock'
    lock.write_text('')
    dead = time.time() - DEAD_LOCK_AGE_SECS
    os.utime(lock, (dead, dead))
    for n in (1, 2, 3):
        (user_root / f'pytest-{n}').mkdir()
    (user_root / 'garbage-seeded' / 'leftover').mkdir(parents=True)


def temproot_recorder_source(*, timeout_secs: float | None = None) -> str:
    """A test module whose ``TEMPROOT_RECORDER_TEST`` records the child's
    ``tempfile.gettempdir()`` beside itself, for ``recorded_temproot`` to read:
    the witness that an unchanged stand-in listing is not vacuous.

    The test requests ``tmp_path``, so the session creates its basetemp even
    without xdist. *timeout_secs* marks it for a suite whose ini cap is tighter
    than a cold start."""
    mark = '' if timeout_secs is None else f'@pytest.mark.timeout({timeout_secs!r})\n'
    return (
        'import pathlib\n'
        'import tempfile\n'
        '\n'
        'import pytest\n'
        '\n'
        '\n'
        f'{mark}'
        f'def {TEMPROOT_RECORDER_TEST}(tmp_path):\n'
        f'    record = pathlib.Path(__file__).resolve().with_name({_TEMPROOT_RECORD_NAME!r})\n'
        '    record.write_text(tempfile.gettempdir())\n'
    )


def recorded_temproot(recorder_dir: Path) -> Path | None:
    """The temp root the recorder in *recorder_dir* ran against, or None if it never ran."""
    record = recorder_dir / _TEMPROOT_RECORD_NAME
    return Path(record.read_text(encoding='utf-8')) if record.exists() else None
