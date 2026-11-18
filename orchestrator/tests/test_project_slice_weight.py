"""The startup step that gives this orchestrator's project slice its CPUWeight (task 3394).

Driven through the public ``ensure_project_slice_weight``. The recording
``systemctl`` of ``_fake_systemctl`` is put first on PATH, so the real
``shutil.which`` and subprocess spawn run against it.
"""

from __future__ import annotations

import logging

import pytest
from _fake_systemctl import fake_systemctl as fake_systemctl
from _fake_systemctl import recorded_calls

from orchestrator.project_slice_weight import SliceWeightOutcome, ensure_project_slice_weight

_USER_MANAGER = '/user.slice/user-1000.slice/user@1000.service'
_SERVICE_IN_SLICE = f'{_USER_MANAGER}/df.slice/df-my_proj.slice/orchestrator-x.service'
_SERVICE_IN_APP_SLICE = f'{_USER_MANAGER}/app.slice/orchestrator-x.service'
_SERVICE_IN_OTHER_PROJECT_SLICE = (
    f'{_USER_MANAGER}/df.slice/df-other.slice/orchestrator-x.service'
)
_VERIFY_SCOPE_IN_SLICE = (
    f'{_USER_MANAGER}/df.slice/df-my_proj.slice/df-verify-x-0123abcd-0123456789ab.scope'
)


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]


@pytest.mark.asyncio
async def test_service_directly_under_its_slice_gets_the_weight(fake_systemctl):
    outcome = await ensure_project_slice_weight('my-proj', _SERVICE_IN_SLICE)

    assert outcome is SliceWeightOutcome.APPLIED
    assert recorded_calls(fake_systemctl) == [
        ['--user', 'set-property', '--runtime', 'df-my_proj.slice', 'CPUWeight=100'],
    ]


@pytest.mark.asyncio
async def test_service_still_in_app_slice_is_flagged_and_left_alone(fake_systemctl, caplog):
    with caplog.at_level(logging.WARNING):
        outcome = await ensure_project_slice_weight('my-proj', _SERVICE_IN_APP_SLICE)

    assert outcome is SliceWeightOutcome.UNIT_NOT_IN_SLICE
    assert recorded_calls(fake_systemctl) == []
    (warning,) = _warnings(caplog)
    assert 'df-my_proj.slice' in warning
    assert _SERVICE_IN_APP_SLICE in warning


@pytest.mark.asyncio
async def test_service_under_another_projects_slice_is_flagged_and_left_alone(
    fake_systemctl, caplog,
):
    """The unit's Slice= disagrees with the declared fused_memory.project_id."""
    with caplog.at_level(logging.WARNING):
        outcome = await ensure_project_slice_weight('my-proj', _SERVICE_IN_OTHER_PROJECT_SLICE)

    assert outcome is SliceWeightOutcome.UNIT_NOT_IN_SLICE
    assert recorded_calls(fake_systemctl) == []
    (warning,) = _warnings(caplog)
    assert 'df-my_proj.slice' in warning
    assert _SERVICE_IN_OTHER_PROJECT_SLICE in warning


@pytest.mark.asyncio
async def test_process_inside_a_verify_scope_never_mutates_its_slice(fake_systemctl):
    outcome = await ensure_project_slice_weight('my-proj', _VERIFY_SCOPE_IN_SLICE)

    assert outcome is SliceWeightOutcome.UNIT_NOT_IN_SLICE
    assert recorded_calls(fake_systemctl) == []


@pytest.mark.asyncio
async def test_unresolved_own_cgroup_is_not_in_slice(fake_systemctl):
    outcome = await ensure_project_slice_weight('my-proj', '')

    assert outcome is SliceWeightOutcome.UNIT_NOT_IN_SLICE
    assert recorded_calls(fake_systemctl) == []


@pytest.mark.asyncio
async def test_failing_systemctl_is_reported_not_raised(fake_systemctl, monkeypatch, caplog):
    monkeypatch.setenv('FAKE_SYSTEMCTL_RC', '1')
    with caplog.at_level(logging.WARNING):
        outcome = await ensure_project_slice_weight('my-proj', _SERVICE_IN_SLICE)

    assert outcome is SliceWeightOutcome.FAILED
    (warning,) = _warnings(caplog)
    assert 'rc=1' in warning


@pytest.mark.asyncio
async def test_hanging_systemctl_is_killed_and_reported_not_raised(
    fake_systemctl, monkeypatch, caplog,
):
    monkeypatch.setenv('FAKE_SYSTEMCTL_SLEEP_SECS', '30')
    with caplog.at_level(logging.WARNING):
        outcome = await ensure_project_slice_weight(
            'my-proj', _SERVICE_IN_SLICE, set_property_timeout_secs=0.2,
        )

    assert outcome is SliceWeightOutcome.FAILED
    (warning,) = _warnings(caplog)
    assert 'timed out after 0.2s' in warning


@pytest.mark.asyncio
async def test_missing_systemctl_is_reported_not_raised(tmp_path, monkeypatch):
    empty = tmp_path / 'empty-path'
    empty.mkdir()
    monkeypatch.setenv('PATH', str(empty))

    outcome = await ensure_project_slice_weight('my-proj', _SERVICE_IN_SLICE)

    assert outcome is SliceWeightOutcome.FAILED
