"""The env-recovery retry's persisted attempt says which legs it re-ran (task 5624).

The retry re-runs only the test leg, so lint and type are the first pass's
results. The record written to disk marks them ``carried_over``, and the
attempt's own wall time is the recovery leg's, never a max or sum across legs
that ran in an earlier pass.

Driven through the REAL ``run_verification`` with a real failing-then-
recovering ``pytest`` executable put first on PATH through the public
``ModuleConfig.verify_env`` seam; nothing in ``orchestrator.verify`` is patched.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_FAKE_PYTEST = """\
#!/bin/sh
case " $* " in
  *" no:xdist "*) echo "1 passed in 0.01s"; exit 0;;
esac
echo "pytest: error: unrecognized arguments: -n --dist --max-worker-restart=0" >&2
exit 4
"""


def _fake_pytest_bin(tmp_path: Path) -> Path:
    """A `pytest` that fails with the vanished-xdist usage error unless run serial."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    script = bin_dir / 'pytest'
    script.write_text(_FAKE_PYTEST, encoding='utf-8')
    script.chmod(script.stat().st_mode | 0o100)
    return bin_dir


async def _verify(tmp_path, *, concurrent, lint_command='true', type_command='sleep 1'):
    from orchestrator import verify
    from orchestrator.config import ModuleConfig, OrchestratorConfig

    (tmp_path / '.task').mkdir()
    bin_dir = _fake_pytest_bin(tmp_path)
    config = OrchestratorConfig(
        project_root=tmp_path,
        verify_admission_enabled=False,
        concurrent_verify=concurrent,
        verify_cold_preprovision_command='',
    )
    module_config = ModuleConfig(
        prefix='__fallback__',
        test_command='pytest -n 2 tests/',
        lint_command=lint_command,
        type_check_command=type_command,
        verify_env={'PATH': f"{bin_dir}{os.pathsep}{os.environ['PATH']}"},
    )
    result = await verify.run_verification(
        tmp_path, config, module_config, attempt_id=1, task_id='5624', max_retries=0,
    )
    summary_path = tmp_path / '.task' / 'verify' / 'attempt-1.__fallback__.summary.json'
    return result, json.loads(summary_path.read_text(encoding='utf-8'))


def _entry(summary, label):
    return next(c for c in summary['commands'] if c['label'] == label)


@pytest.mark.asyncio
async def test_the_recovery_run_is_the_test_leg_on_record(tmp_path):
    result, summary = await _verify(tmp_path, concurrent=True)

    assert result.passed
    test = _entry(summary, 'test')
    assert '-o addopts=' in test['cmd']
    assert test['carried_over'] is False


@pytest.mark.asyncio
@pytest.mark.parametrize('concurrent', [True, False])
async def test_lint_and_type_are_recorded_as_carried_over(tmp_path, concurrent):
    _, summary = await _verify(tmp_path, concurrent=concurrent)

    assert all('carried_over' in entry for entry in summary['commands'])
    assert _entry(summary, 'lint')['carried_over'] is True
    assert _entry(summary, 'type')['carried_over'] is True


@pytest.mark.asyncio
@pytest.mark.parametrize('concurrent', [True, False])
async def test_the_attempt_wall_time_is_the_recovery_legs_own(tmp_path, concurrent):
    result, summary = await _verify(tmp_path, concurrent=concurrent)

    test, type_ = _entry(summary, 'test'), _entry(summary, 'type')
    assert type_['duration_secs'] > test['duration_secs'], (
        f'precondition: the carried type leg ({type_["duration_secs"]}s) must outlast '
        f'the recovery test leg ({test["duration_secs"]}s) for this test to discriminate'
    )
    assert result.duration_secs == pytest.approx(test['duration_secs'])


@pytest.mark.asyncio
async def test_a_carried_over_failure_still_fails_and_the_top_level_says_it_is_carried(
    tmp_path,
):
    result, summary = await _verify(
        tmp_path, concurrent=True, lint_command='false', type_command='true',
    )

    assert result.passed is False
    assert summary['label'] == 'lint'
    assert summary['carried_over'] is True
    assert '-o addopts=' in _entry(summary, 'test')['cmd']


@pytest.mark.asyncio
async def test_a_green_top_level_names_the_recovery_run(tmp_path):
    _, summary = await _verify(tmp_path, concurrent=True)

    assert summary['label'] == 'test'
    assert summary['carried_over'] is False
