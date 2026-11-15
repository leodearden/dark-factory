"""The startup step that gives this orchestrator's project slice its CPUWeight (task 3394).

Driven through the public ``ensure_project_slice_weight``. A recording
``systemctl`` is put first on PATH, so the real ``shutil.which`` and subprocess
spawn run against it; its exit code comes from ``FAKE_SYSTEMCTL_RC``.
"""

from __future__ import annotations

import logging
import os
import shlex
import stat
from pathlib import Path

import pytest

from orchestrator.project_slice_weight import SliceWeightOutcome, ensure_project_slice_weight

_USER_MANAGER = '/user.slice/user-1000.slice/user@1000.service'
_SERVICE_IN_SLICE = f'{_USER_MANAGER}/df.slice/df-my_proj.slice/orchestrator-x.service'
_SERVICE_IN_APP_SLICE = f'{_USER_MANAGER}/app.slice/orchestrator-x.service'
_VERIFY_SCOPE_IN_SLICE = (
    f'{_USER_MANAGER}/df.slice/df-my_proj.slice/df-verify-x-0123abcd-0123456789ab.scope'
)


@pytest.fixture
def fake_systemctl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put a recording ``systemctl`` first on PATH; return its argv-record dir."""
    argv_dir = tmp_path / 'systemctl-argv'
    argv_dir.mkdir()
    fakebin = tmp_path / 'fakectl'
    fakebin.mkdir()
    script = fakebin / 'systemctl'
    script.write_text(
        '#!/bin/sh\n'
        f'rec=$(mktemp {shlex.quote(str(argv_dir))}/argv.XXXXXX)\n'
        'for a in "$@"; do printf \'%s\\0\' "$a" >> "$rec"; done\n'
        'exit "${FAKE_SYSTEMCTL_RC:-0}"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv('PATH', f'{fakebin}{os.pathsep}' + os.environ['PATH'])
    monkeypatch.delenv('FAKE_SYSTEMCTL_RC', raising=False)
    return argv_dir


def _recorded(argv_dir: Path) -> list[list[str]]:
    return [p.read_bytes().decode().split('\0')[:-1] for p in sorted(argv_dir.iterdir())]


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.asyncio
async def test_service_directly_under_its_slice_gets_the_weight(fake_systemctl):
    outcome = await ensure_project_slice_weight('my-proj', _SERVICE_IN_SLICE)

    assert outcome is SliceWeightOutcome.APPLIED
    assert _recorded(fake_systemctl) == [
        ['--user', 'set-property', '--runtime', 'df-my_proj.slice', 'CPUWeight=100'],
    ]


@pytest.mark.asyncio
async def test_service_still_in_app_slice_is_flagged_and_left_alone(fake_systemctl, caplog):
    with caplog.at_level(logging.WARNING):
        outcome = await ensure_project_slice_weight('my-proj', _SERVICE_IN_APP_SLICE)

    assert outcome is SliceWeightOutcome.UNIT_NOT_IN_SLICE
    assert _recorded(fake_systemctl) == []
    (warning,) = _warnings(caplog)
    assert 'df-my_proj.slice' in warning
    assert _SERVICE_IN_APP_SLICE in warning


@pytest.mark.asyncio
async def test_process_inside_a_verify_scope_never_mutates_its_slice(fake_systemctl):
    outcome = await ensure_project_slice_weight('my-proj', _VERIFY_SCOPE_IN_SLICE)

    assert outcome is SliceWeightOutcome.UNIT_NOT_IN_SLICE
    assert _recorded(fake_systemctl) == []


@pytest.mark.asyncio
async def test_unresolved_own_cgroup_is_not_in_slice(fake_systemctl):
    outcome = await ensure_project_slice_weight('my-proj', '')

    assert outcome is SliceWeightOutcome.UNIT_NOT_IN_SLICE
    assert _recorded(fake_systemctl) == []


@pytest.mark.asyncio
async def test_failing_systemctl_is_reported_not_raised(fake_systemctl, monkeypatch, caplog):
    monkeypatch.setenv('FAKE_SYSTEMCTL_RC', '1')
    with caplog.at_level(logging.WARNING):
        outcome = await ensure_project_slice_weight('my-proj', _SERVICE_IN_SLICE)

    assert outcome is SliceWeightOutcome.FAILED
    (warning,) = _warnings(caplog)
    assert 'rc=1' in warning


@pytest.mark.asyncio
async def test_missing_systemctl_is_reported_not_raised(tmp_path, monkeypatch):
    empty = tmp_path / 'empty-path'
    empty.mkdir()
    monkeypatch.setenv('PATH', str(empty))

    outcome = await ensure_project_slice_weight('my-proj', _SERVICE_IN_SLICE)

    assert outcome is SliceWeightOutcome.FAILED
