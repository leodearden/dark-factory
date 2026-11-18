"""Tests for orchestrator.session_windows: the one home of the wmctrl window-id contract (PRD C6)."""

from __future__ import annotations

import dataclasses
import logging
import subprocess
from collections.abc import Callable

import pytest

from orchestrator import session_windows

WmctrlRun = Callable[[list[str]], subprocess.CompletedProcess[str]]

_UNPROVABLE_IDS = [None, '', '   ', '0xZZ', 'nope', '-1', True]
_UNPROVABLE_ID_NAMES = [
    'none', 'empty', 'whitespace', 'unparseable-hex', 'unparseable-decimal', 'negative', 'bool',
]


def _wmctrl_run(
    stdout_lines: list[str], returncode: int = 0, calls: list[list[str]] | None = None
) -> WmctrlRun:
    """A fake ``wmctrl -l`` runner returning canned output, recording each argv into *calls*."""

    def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        if calls is not None:
            calls.append(argv)
        return subprocess.CompletedProcess(argv, returncode=returncode, stdout='\n'.join(stdout_lines))

    return _run


def _scripted_wmctrl_run(
    *results: tuple[int, str], calls: list[list[str]]
) -> WmctrlRun:
    """A fake runner answering its Nth call with ``results[N]``, then repeating the last one."""

    def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        returncode, stdout = results[min(len(calls), len(results)) - 1]
        return subprocess.CompletedProcess(argv, returncode=returncode, stdout=stdout)

    return _run


def _no_sleep(_seconds: float) -> None:
    return None


# ---------------------------------------------------------------------------
# canonical_window_id / window_id_int
# ---------------------------------------------------------------------------


def test_decimal_and_hex_spellings_of_one_window_share_the_padded_canonical_id() -> None:
    assert (
        session_windows.canonical_window_id('52428807')
        == session_windows.canonical_window_id('0x3200007')
        == session_windows.canonical_window_id(52428807)
        == '0x03200007'
    )


def test_canonical_window_id_lowercases_an_uppercase_hex_prefix_and_digits() -> None:
    assert session_windows.canonical_window_id('0X0000002B') == '0x0000002b'


def test_a_decimal_capture_and_wmctrls_padded_column_compare_equal() -> None:
    assert session_windows.canonical_window_id('26') == '0x0000001a'
    assert session_windows.canonical_window_id('0x0000001a') == '0x0000001a'


def test_an_already_canonical_id_round_trips() -> None:
    assert session_windows.canonical_window_id('0x03200007') == '0x03200007'


@pytest.mark.parametrize('raw', _UNPROVABLE_IDS, ids=_UNPROVABLE_ID_NAMES)
def test_canonical_window_id_is_none_for_an_unprovable_id(raw: str | int | None) -> None:
    assert session_windows.canonical_window_id(raw) is None


def test_window_id_int_agrees_across_decimal_and_hex_spellings() -> None:
    spellings = ['52428807', '0x3200007', '0x03200007', ' 0X3200007 ', 52428807]

    assert {session_windows.window_id_int(raw) for raw in spellings} == {52428807}


@pytest.mark.parametrize('raw', _UNPROVABLE_IDS, ids=_UNPROVABLE_ID_NAMES)
def test_window_id_int_is_none_for_an_unprovable_id(raw: str | int | None) -> None:
    assert session_windows.window_id_int(raw) is None


# ---------------------------------------------------------------------------
# parse_wmctrl_list: the one `wmctrl -l` line parser
# ---------------------------------------------------------------------------


def test_a_four_column_line_parses_to_a_window_with_its_title_intact() -> None:
    windows = session_windows.parse_wmctrl_list('0x3200007  0 host  focus:df#2510 alpha\n')

    assert windows == (
        session_windows.WmWindow(
            window_id='0x03200007', desktop='0', host='host', title='focus:df#2510 alpha'
        ),
    )


def test_a_parsed_window_is_frozen() -> None:
    (window,) = session_windows.parse_wmctrl_list('0x3200007 0 host title')

    with pytest.raises(dataclasses.FrozenInstanceError):
        window.title = 'other'  # type: ignore[misc]


def test_every_window_id_is_canonical_however_wmctrl_spelled_it() -> None:
    windows = session_windows.parse_wmctrl_list('0X0000002B  0 host  some window\n0x1a 1 host x')

    assert [window.window_id for window in windows] == ['0x0000002b', '0x0000001a']


def test_a_titleless_or_id_only_line_is_kept_with_empty_missing_columns() -> None:
    """The reaper fails toward keeping a live session: a window whose _NET_WM_NAME is
    empty prints only three columns, and dropping it would mark that session EXITED."""
    windows = session_windows.parse_wmctrl_list('0x3c 0 host \n0x3d')

    assert windows == (
        session_windows.WmWindow(window_id='0x0000003c', desktop='0', host='host', title=''),
        session_windows.WmWindow(window_id='0x0000003d', desktop='', host='', title=''),
    )


def test_a_blank_line_and_a_line_not_led_by_a_window_id_are_skipped() -> None:
    windows = session_windows.parse_wmctrl_list('\n   \nnot-an-id 0 host title\n0x1a 0 host kept\n')

    assert [window.title for window in windows] == ['kept']


# ---------------------------------------------------------------------------
# run_wmctrl: the never-raising subprocess boundary
# ---------------------------------------------------------------------------


def test_run_wmctrl_maps_a_missing_binary_to_rc_127(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError('wmctrl not found')

    monkeypatch.setattr(subprocess, 'run', _fake_run)

    assert session_windows.run_wmctrl(['wmctrl', '-l']).returncode == 127


def test_run_wmctrl_maps_a_timeout_to_transient_rc_124_not_127(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A momentarily-hung wmctrl is transient, so it must not share the permanent
    missing-binary sentinel the title resolver short-circuits on."""

    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=argv, timeout=session_windows.WMCTRL_TIMEOUT_SECS)

    monkeypatch.setattr(subprocess, 'run', _fake_run)

    assert session_windows.run_wmctrl(['wmctrl', '-l']).returncode == 124


def test_run_wmctrl_bounds_each_probe_by_the_wmctrl_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, object]] = []

    def _fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen.append(kwargs)
        return subprocess.CompletedProcess(argv, returncode=0, stdout='0x1a 0 host title\n')

    monkeypatch.setattr(subprocess, 'run', _fake_run)

    result = session_windows.run_wmctrl(['wmctrl', '-l'])

    assert (result.returncode, result.stdout) == (0, '0x1a 0 host title\n')
    assert [kwargs['timeout'] for kwargs in seen] == [session_windows.WMCTRL_TIMEOUT_SECS]


# ---------------------------------------------------------------------------
# live_window_ids: None means "cannot prove", never "no windows"
# ---------------------------------------------------------------------------


def test_live_window_ids_is_the_set_of_canonical_ids_wmctrl_reports() -> None:
    run = _wmctrl_run(['0x0000001a  0 host  first window title', '0x2b 1 host second-window-title'])

    assert session_windows.live_window_ids(run=run) == frozenset({'0x0000001a', '0x0000002b'})


def test_live_window_ids_is_empty_not_none_when_wmctrl_lists_no_window() -> None:
    result = session_windows.live_window_ids(run=_wmctrl_run([]))

    assert result == frozenset()
    assert result is not None


@pytest.mark.parametrize(
    'returncode', [127, 124, 1], ids=['missing-binary', 'transient-timeout', 'other-nonzero']
)
def test_live_window_ids_is_none_when_wmctrl_fails(returncode: int) -> None:
    run = _wmctrl_run(['0x1a  0 host  irrelevant'], returncode=returncode)

    assert session_windows.live_window_ids(run=run) is None


def test_live_window_ids_is_none_and_warns_when_the_runner_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def _boom(argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise OSError('wmctrl exploded')

    with caplog.at_level(logging.WARNING, logger=session_windows.__name__):
        assert session_windows.live_window_ids(run=_boom) is None

    assert [record.levelno for record in caplog.records] == [logging.WARNING]


# ---------------------------------------------------------------------------
# resolve_window_id_by_title
# ---------------------------------------------------------------------------

_MARKER = 'focus:df#2510 alpha'


def test_resolve_by_title_returns_the_canonical_id_of_the_exactly_matching_window() -> None:
    calls: list[list[str]] = []
    run = _wmctrl_run(['0x1a 0 host unrelated', f'0x3200007 0 host {_MARKER}'], calls=calls)

    result = session_windows.resolve_window_id_by_title(_MARKER, run=run, sleep=_no_sleep, attempts=3)

    assert result == '0x03200007'
    assert calls == [['wmctrl', '-l']]


def test_resolve_by_title_never_matches_a_title_the_marker_only_prefixes() -> None:
    run = _wmctrl_run([f'0x3200007 0 host {_MARKER} extra'])

    assert session_windows.resolve_window_id_by_title(_MARKER, run=run, sleep=_no_sleep, attempts=1) is None


def test_resolve_by_title_retries_with_a_sleep_between_attempts_only() -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []
    run = _scripted_wmctrl_run(
        (1, ''), (1, ''), (0, f'0x3200007 0 host {_MARKER}\n'), calls=calls
    )

    result = session_windows.resolve_window_id_by_title(
        _MARKER, run=run, sleep=sleeps.append, attempts=5
    )

    assert result == '0x03200007'
    assert len(calls) == 3
    assert len(sleeps) == 2
    assert all(seconds > 0 for seconds in sleeps)


def test_resolve_by_title_gives_up_after_its_attempts_without_a_trailing_sleep() -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []
    run = _wmctrl_run([], returncode=1, calls=calls)

    result = session_windows.resolve_window_id_by_title(_MARKER, run=run, sleep=sleeps.append, attempts=4)

    assert result is None
    assert (len(calls), len(sleeps)) == (4, 3)


def test_resolve_by_title_defaults_to_resolve_attempts() -> None:
    calls: list[list[str]] = []

    session_windows.resolve_window_id_by_title(
        _MARKER, run=_wmctrl_run([], returncode=1, calls=calls), sleep=_no_sleep
    )

    assert len(calls) == session_windows.RESOLVE_ATTEMPTS


def test_resolve_by_title_short_circuits_on_the_missing_binary_sentinel() -> None:
    calls: list[list[str]] = []
    sleeps: list[float] = []
    run = _wmctrl_run([], returncode=127, calls=calls)

    result = session_windows.resolve_window_id_by_title(_MARKER, run=run, sleep=sleeps.append, attempts=5)

    assert result is None
    assert (len(calls), sleeps) == (1, [])


def test_resolve_by_title_rides_out_the_transient_timeout_sentinel() -> None:
    calls: list[list[str]] = []
    run = _scripted_wmctrl_run((124, ''), (124, ''), (0, f'0x3200007 0 host {_MARKER}\n'), calls=calls)

    result = session_windows.resolve_window_id_by_title(_MARKER, run=run, sleep=_no_sleep, attempts=5)

    assert result == '0x03200007'
    assert len(calls) == 3


def test_resolve_by_title_is_none_when_the_runner_raises() -> None:
    def _boom(argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise OSError('wmctrl exploded')

    assert session_windows.resolve_window_id_by_title(_MARKER, run=_boom, sleep=_no_sleep, attempts=3) is None


def test_resolve_by_title_is_none_when_the_sleep_raises() -> None:
    def _boom(_seconds: float) -> None:
        raise RuntimeError('sleep exploded')

    run = _wmctrl_run([], returncode=1)

    assert session_windows.resolve_window_id_by_title(_MARKER, run=run, sleep=_boom, attempts=3) is None


def test_resolve_by_title_only_ever_lists_windows() -> None:
    """Read-only probing: it never issues a focus or activate command."""
    calls: list[list[str]] = []

    session_windows.resolve_window_id_by_title(
        _MARKER, run=_wmctrl_run([], returncode=1, calls=calls), sleep=_no_sleep, attempts=3
    )

    assert calls == [['wmctrl', '-l']] * 3


def test_resolve_by_title_never_runs_wmctrl_for_an_empty_title() -> None:
    calls: list[list[str]] = []
    run = _wmctrl_run(['0x3c 0 host '], calls=calls)

    assert session_windows.resolve_window_id_by_title('', run=run, sleep=_no_sleep, attempts=3) is None
    assert calls == []


# ---------------------------------------------------------------------------
# wmctrl_window_probe: the live default every consumer injects
# ---------------------------------------------------------------------------


def test_wmctrl_window_probe_routes_both_probes_through_the_injected_runner() -> None:
    calls: list[list[str]] = []
    run = _wmctrl_run([f'0x3200007 0 host {_MARKER}'], calls=calls)

    probe = session_windows.wmctrl_window_probe(run=run, sleep=_no_sleep)

    assert isinstance(probe, session_windows.WindowProbe)
    assert probe.live_window_ids() == frozenset({'0x03200007'})
    assert probe.resolve_by_title(_MARKER) == '0x03200007'
    assert calls == [['wmctrl', '-l'], ['wmctrl', '-l']]


def test_wmctrl_window_probe_resolves_with_the_injected_sleep() -> None:
    sleeps: list[float] = []
    probe = session_windows.wmctrl_window_probe(run=_wmctrl_run([], returncode=1), sleep=sleeps.append)

    assert probe.resolve_by_title(_MARKER) is None
    assert len(sleeps) == session_windows.RESOLVE_ATTEMPTS - 1


def test_a_window_probe_is_frozen() -> None:
    probe = session_windows.wmctrl_window_probe(run=_wmctrl_run([]), sleep=_no_sleep)

    with pytest.raises(dataclasses.FrozenInstanceError):
        probe.live_window_ids = lambda: None  # type: ignore[misc]
