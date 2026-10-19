"""The contract of ``_probe_session.py::run_probe_session``: a nested probe
session runs under the basetemp its caller names, and never in the host's
default pytest temp root."""
from __future__ import annotations

import getpass
import os
import time
from pathlib import Path

import pytest
from _probe_session import run_probe_session

_PROBE = Path(__file__).with_name('_probe_session_probe.py')
#: Older than pytest's 3-day LOCK_TIMEOUT, so pytest treats the lock as dead.
_DEAD_LOCK_AGE_SECS = 5 * 24 * 60 * 60


def _listing(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob('*'))


def _seed_a_dead_sessions_dirs(user_root: Path) -> None:
    """``pytest-0`` holds a dead session's leftovers and lock; ``pytest-1..3``
    make it a cleanup candidate as soon as a new session adds ``pytest-4``."""
    (user_root / 'pytest-0' / 'leftover').mkdir(parents=True)
    lock = user_root / 'pytest-0' / '.lock'
    lock.write_text('')
    dead = time.time() - _DEAD_LOCK_AGE_SECS
    os.utime(lock, (dead, dead))
    for n in (1, 2, 3):
        (user_root / f'pytest-{n}').mkdir()


def test_a_probe_session_leaves_the_hosts_default_temp_root_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = tmp_path / 'host-temp'
    _seed_a_dead_sessions_dirs(host / f'pytest-of-{getpass.getuser()}')
    before = _listing(host)
    monkeypatch.setenv('PYTEST_DEBUG_TEMPROOT', str(host))
    basetemp = tmp_path / 'probe-basetemp'

    run = run_probe_session(_PROBE, basetemp=basetemp, kill_secs=60, backstop_secs=240)

    output = run.stdout[-3000:] + run.stderr[-2000:]
    assert run.returncode == 0, output
    assert '1 passed' in run.stdout, output
    assert len(list(basetemp.rglob('written'))) == 1, (
        f'the probe must run under the basetemp it was given ({basetemp}); '
        f'found {_listing(basetemp) if basetemp.exists() else "no such dir"}'
    )
    after = _listing(host)
    assert after == before, (
        "a nested probe session must not create, lock or clean dirs in the "
        "host's default pytest temp root: its exit-time cleanup of other "
        "sessions' dead dirs runs outside pytest-timeout\n"
        f'before: {before}\nafter:  {after}'
    )


@pytest.mark.parametrize(
    'spelling', [('--basetemp={}',), ('--basetemp', '{}')], ids=['joined', 'separate'],
)
def test_a_basetemp_smuggled_through_extra_is_refused_before_any_session_starts(
    tmp_path: Path, spelling: tuple[str, ...],
) -> None:
    smuggled = tmp_path / 'elsewhere'
    extra = tuple(arg.format(smuggled) for arg in spelling)
    basetemp = tmp_path / 'probe-basetemp'

    with pytest.raises(ValueError, match='--basetemp') as refused:
        run_probe_session(
            _PROBE, basetemp=basetemp, kill_secs=60, backstop_secs=240, extra=extra,
        )

    assert extra[0] in str(refused.value)
    assert not basetemp.exists() and not smuggled.exists(), 'a session started anyway'
