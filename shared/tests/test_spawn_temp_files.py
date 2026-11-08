"""Tests for shared.spawn_temp_files — the names of the temp files a Claude CLI
spawn is handed by argv path, and the startup sweep that reclaims the ones a
SIGKILLed spawner never unlinked.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from shared.spawn_temp_files import (
    MCP_CONFIG_TEMP_NAME,
    SPAWN_TEMP_NAMES,
    SYSPROMPT_TEMP_NAME,
    SpawnTempName,
    sweep_orphaned_spawn_temp_files,
)

_STALE_SECS = 25 * 3600
_LOGGER = 'shared.spawn_temp_files'


def test_spawn_temp_names_are_exactly_the_two_kinds() -> None:
    assert SPAWN_TEMP_NAMES == (SYSPROMPT_TEMP_NAME, MCP_CONFIG_TEMP_NAME)


@pytest.mark.parametrize('kind', SPAWN_TEMP_NAMES, ids=lambda k: k.prefix)
def test_matches_a_real_mkstemp_name(kind: SpawnTempName, tmp_path: Path) -> None:
    fd, path = tempfile.mkstemp(prefix=kind.prefix, suffix=kind.suffix, dir=tmp_path)
    os.close(fd)
    assert kind.matches(Path(path).name), path


@pytest.mark.parametrize('kind', SPAWN_TEMP_NAMES, ids=lambda k: k.prefix)
def test_matches_its_own_mkstemp_output_in_the_default_tempdir(
    kind: SpawnTempName, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    fd, path = kind.mkstemp()
    os.close(fd)
    assert Path(path).parent == tmp_path
    assert kind.matches(Path(path).name), path


def test_names_are_the_ones_already_on_hosts() -> None:
    """Legacy orphans on hosts carry these names; keeping them keeps those reclaimable."""
    assert SYSPROMPT_TEMP_NAME.prefix == 'sysprompt_'
    assert SYSPROMPT_TEMP_NAME.suffix == '.txt'
    assert MCP_CONFIG_TEMP_NAME.prefix == 'mcp_'
    assert MCP_CONFIG_TEMP_NAME.suffix == '.json'


@pytest.mark.parametrize('name', [
    'mcp_del7442.py',
    'mcp_config.json',
    'sysprompt_abcd1234.txt.bak',
    'startup_probe_sysprompt_abcd1234.txt',
    'sysprompt_ABCD1234.txt',
    'mcp_abcd12345.json',
])
def test_no_kind_matches_a_name_mkstemp_would_not_have_made(name: str) -> None:
    assert not any(kind.matches(name) for kind in SPAWN_TEMP_NAMES)


def test_spawn_temp_name_is_frozen() -> None:
    frozen_field = 'prefix'
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(SYSPROMPT_TEMP_NAME, frozen_field, 'other_')


def _fake_proc(root: Path, cmdlines: dict[int, bytes]) -> Path:
    """A process table of ``<pid>/cmdline`` files; this process is always in it."""
    for pid, cmdline in {**cmdlines, os.getpid(): b'pytest\0'}.items():
        (root / str(pid)).mkdir(parents=True)
        (root / str(pid) / 'cmdline').write_bytes(cmdline)
    return root


def _age(path: Path, age_secs: float) -> None:
    then = time.time() - age_secs
    os.utime(path, (then, then), follow_symlinks=False)


def _plant(base: Path, name: str, age_secs: float) -> Path:
    path = base / name
    path.write_text('x')
    _age(path, age_secs)
    return path


@pytest.fixture
def base(tmp_path: Path) -> Path:
    path = tmp_path / 'tmp'
    path.mkdir()
    return path


@pytest.fixture
def idle_proc(tmp_path: Path) -> Path:
    return _fake_proc(tmp_path / 'proc', {})


def test_reclaims_stale_unreferenced_files_of_both_kinds(base: Path, idle_proc: Path) -> None:
    sysprompt = _plant(base, 'sysprompt_abcd1234.txt', _STALE_SECS)
    mcp = _plant(base, 'mcp_abcd1234.json', _STALE_SECS)

    assert sweep_orphaned_spawn_temp_files(base_dir=base, proc_root=idle_proc) == 2

    assert not sysprompt.exists()
    assert not mcp.exists()


def test_keeps_files_younger_than_the_age_floor(base: Path, idle_proc: Path) -> None:
    fresh = _plant(base, 'sysprompt_fresh123.txt', 60)
    two_hours = _plant(base, 'mcp_twohour1.json', 2 * 3600)

    assert sweep_orphaned_spawn_temp_files(
        base_dir=base, proc_root=idle_proc, min_age_secs=3600,
    ) == 1
    assert fresh.exists()
    assert not two_hours.exists()

    kept = _plant(base, 'mcp_twohour2.json', 2 * 3600)
    assert sweep_orphaned_spawn_temp_files(base_dir=base, proc_root=idle_proc) == 0
    assert kept.exists()


@pytest.mark.parametrize('cmdline', [
    b'claude\0--system-prompt-file\0/tmp/sysprompt_abcd1234.txt\0',
    b'bwrap\0--\0sh\0-c\0claude --mcp-config /tmp/mcp_abcd1234.json\0',
    b'claude\0--mcp-config=/tmp/mcp_abcd1234.json\0',
    b'claude\0--mcp-config\0/var/tmp/x/mcp_abcd1234.json\0',
], ids=['argv-token', 'embedded-in-sh-c', 'flag-equals-value', 'other-dir-same-basename'])
def test_keeps_a_stale_file_a_live_process_names_on_its_cmdline(
    base: Path, tmp_path: Path, cmdline: bytes,
) -> None:
    """The CLI reads these files once at startup from their argv paths and holds
    no fd on them: the 2026-07-27 purge found 8 live processes referencing them
    by argument with zero fds open. So the fake pid dir carries ONLY a cmdline;
    an fd/ or cwd-based liveness check would see nothing here and delete."""
    sysprompt = _plant(base, 'sysprompt_abcd1234.txt', _STALE_SECS)
    mcp = _plant(base, 'mcp_abcd1234.json', _STALE_SECS)
    proc = _fake_proc(tmp_path / 'proc', {os.getpid() + 1: cmdline})

    assert sweep_orphaned_spawn_temp_files(base_dir=base, proc_root=proc) == 1

    referenced, unreferenced = (sysprompt, mcp) if b'sysprompt_' in cmdline else (mcp, sysprompt)
    assert referenced.exists()
    assert not unreferenced.exists()


def test_keeps_stale_entries_that_are_not_a_spawn_temp_file(
    base: Path, tmp_path: Path, idle_proc: Path,
) -> None:
    lookalike = _plant(base, 'mcp_del7442.py', _STALE_SECS)
    directory = base / 'sysprompt_abcd1234.txt'
    directory.mkdir()
    _age(directory, _STALE_SECS)
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    target = _plant(elsewhere, 'target.json', _STALE_SECS)
    link = base / 'mcp_abcd1234.json'
    link.symlink_to(target)
    _age(link, _STALE_SECS)

    assert sweep_orphaned_spawn_temp_files(base_dir=base, proc_root=idle_proc) == 0

    assert lookalike.exists()
    assert directory.is_dir()
    assert link.is_symlink()
    assert target.exists()


def test_keeps_a_file_owned_by_another_user(
    base: Path, idle_proc: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_euid = os.geteuid()
    stale = _plant(base, 'sysprompt_abcd1234.txt', _STALE_SECS)
    monkeypatch.setattr(os, 'geteuid', lambda: real_euid + 1)

    assert sweep_orphaned_spawn_temp_files(base_dir=base, proc_root=idle_proc) == 0
    assert stale.exists()


def _infos(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == _LOGGER and r.levelno == logging.INFO]


def test_logs_one_info_naming_the_count_only_when_it_removed_something(
    base: Path, idle_proc: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=_LOGGER)
    _plant(base, 'sysprompt_abcd1234.txt', _STALE_SECS)
    _plant(base, 'mcp_abcd1234.json', _STALE_SECS)

    assert sweep_orphaned_spawn_temp_files(base_dir=base, proc_root=idle_proc) == 2
    infos = _infos(caplog)
    assert len(infos) == 1, infos
    assert re.search(r'\b2\b', infos[0].getMessage().replace(str(base), '')), infos[0].getMessage()

    caplog.clear()
    assert sweep_orphaned_spawn_temp_files(base_dir=base, proc_root=idle_proc) == 0
    assert _infos(caplog) == []


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage() for r in caplog.records
        if r.name == _LOGGER and r.levelno == logging.WARNING
    ]


def _missing_proc(tmp_path: Path) -> Path:
    return tmp_path / 'no-such-proc'


def _proc_without_this_process(tmp_path: Path) -> Path:
    root = tmp_path / 'proc'
    (root / str(os.getpid() + 1)).mkdir(parents=True)
    (root / str(os.getpid() + 1) / 'cmdline').write_bytes(b'claude\0')
    return root


def _proc_with_an_unreadable_cmdline(tmp_path: Path) -> Path:
    root = _fake_proc(tmp_path / 'proc', {os.getpid() + 1: b'claude\0'})
    (root / str(os.getpid() + 1) / 'cmdline').chmod(0o000)
    return root


_ROOT_READS_ANYTHING = pytest.mark.skipif(
    os.geteuid() == 0, reason='root bypasses file permission bits',
)


@pytest.mark.parametrize('make_proc', [
    _missing_proc,
    _proc_without_this_process,
    pytest.param(_proc_with_an_unreadable_cmdline, marks=_ROOT_READS_ANYTHING),
], ids=['proc-missing', 'own-pid-not-visible', 'cmdline-unreadable'])
def test_deletes_nothing_when_liveness_cannot_be_established(
    base: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    make_proc: Callable[[Path], Path],
) -> None:
    stale = _plant(base, 'sysprompt_abcd1234.txt', _STALE_SECS)
    proc = make_proc(tmp_path)
    try:
        assert sweep_orphaned_spawn_temp_files(base_dir=base, proc_root=proc) == 0
    finally:
        for cmdline in proc.glob('*/cmdline'):
            cmdline.chmod(0o644)

    assert stale.exists()
    messages = _warnings(caplog)
    assert any(
        'liveness' in m.lower() and 'nothing' in m.lower() for m in messages
    ), messages


def test_consults_proc_only_when_there_is_something_to_delete(
    base: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    _plant(base, 'unrelated.txt', _STALE_SECS)

    assert sweep_orphaned_spawn_temp_files(
        base_dir=base, proc_root=_missing_proc(tmp_path),
    ) == 0
    assert _warnings(caplog) == []


def test_an_expired_deadline_stops_the_sweep_and_says_it_is_incomplete(
    base: Path, idle_proc: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    stale = _plant(base, 'sysprompt_abcd1234.txt', _STALE_SECS)

    assert sweep_orphaned_spawn_temp_files(
        base_dir=base, proc_root=idle_proc, deadline_secs=0,
    ) == 0

    assert stale.exists()
    messages = _warnings(caplog)
    assert any(
        'INCOMPLETE' in m and re.search(r'examined \d+', m) and re.search(r'removed \d+', m)
        for m in messages
    ), messages


@_ROOT_READS_ANYTHING
def test_an_unlink_failure_is_logged_and_not_counted(
    base: Path, idle_proc: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    stale = _plant(base, 'sysprompt_abcd1234.txt', _STALE_SECS)
    base.chmod(0o500)
    try:
        assert sweep_orphaned_spawn_temp_files(base_dir=base, proc_root=idle_proc) == 0
    finally:
        base.chmod(0o700)

    assert stale.exists()
    messages = _warnings(caplog)
    assert any(str(stale) in m for m in messages), messages
