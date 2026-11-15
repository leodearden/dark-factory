"""The contract of ``_probe_session.py::run_probe_session``: a nested probe
session runs under the basetemp its caller names, and never in the host's
default pytest temp root."""
from __future__ import annotations

from pathlib import Path

import pytest
from _host_pytest_temp_root import listing, seed_dead_session_dirs
from _probe_session import run_probe_session

_PROBE = Path(__file__).with_name('_probe_session_probe.py')


def test_a_probe_session_leaves_the_hosts_default_temp_root_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = tmp_path / 'host-temp'
    seed_dead_session_dirs(host)
    before = listing(host)
    monkeypatch.setenv('PYTEST_DEBUG_TEMPROOT', str(host))
    basetemp = tmp_path / 'probe-basetemp'

    run = run_probe_session(_PROBE, basetemp=basetemp, kill_secs=60, backstop_secs=240)

    output = run.stdout[-3000:] + run.stderr[-2000:]
    assert run.returncode == 0, output
    assert '1 passed' in run.stdout, output
    recorded = list(basetemp.rglob('inherited-temproot'))
    assert len(recorded) == 1, (
        f'the probe must run under the basetemp it was given ({basetemp}); '
        f'found {listing(basetemp) if basetemp.exists() else "no such dir"}'
    )
    assert recorded[0].read_text() == str(host), (
        'the probe session never inherited PYTEST_DEBUG_TEMPROOT, so it would '
        'use the real host temp root and leave the stand-in untouched with or '
        'without --basetemp: the assertion below would pass vacuously'
    )
    after = listing(host)
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
