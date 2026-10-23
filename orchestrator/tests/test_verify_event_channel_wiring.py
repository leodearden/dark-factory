"""run_verification allocates, exports and reads one REIFY_VERIFY_EVENT_LOG per leg
invocation, and classifies each leg with its own channel (task 6856).

Drives the real ``orchestrator.verify.run_verification`` over real bash legs that
append real reify producer bytes (``fixtures/reify_verify_event_channel``) to the
channel DF hands them.
"""

from __future__ import annotations

import json
import shlex
import tempfile
from pathlib import Path
from typing import Literal

import pytest

from orchestrator.config import ModuleConfig, OrchestratorConfig
from orchestrator.verify import VerifyResult, run_verification
from orchestrator.verify_categories import FailureCategory
from orchestrator.verify_classify import classify_failure
from orchestrator.verify_cmd import ToolKind
from orchestrator.verify_event_channel import VERIFY_EVENT_LOG_ENV
from orchestrator.verify_runner import result_from_dict, result_to_dict

FIXTURES = Path(__file__).parent / 'fixtures' / 'reify_verify_event_channel'
SLOT_FATAL = shlex.quote(str(FIXTURES / 'slot_timeout-fatal.events.jsonl'))
POOL = shlex.quote(str(FIXTURES / 'run_all-pool.events.jsonl'))
FATAL_SENTINEL = (FIXTURES / 'slot_timeout-fatal.stderr.txt').read_text().rstrip('\n')
ORDINARY_FAILURE = 'FAILED tests/test_x.py::test_y - AssertionError'
ENV_TRANSIENT_LINE = '=== INTERRUPTED (worktree removed) ==='

Role = Literal['merge', 'task', 'background']


@pytest.fixture(autouse=True)
def _no_ambient_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(VERIFY_EVENT_LOG_ENV, raising=False)


def write_leg(worktree: Path, name: str, body: str) -> None:
    (worktree / name).write_text('#!/usr/bin/env bash\n' + body)


def record_channel(side_file: str) -> str:
    """Leg lines writing the channel path, then its first line, to ``side_file``."""
    return (
        f'printf "%s\\n" "${{{VERIFY_EVENT_LOG_ENV}:-}}" > {side_file}\n'
        f'head -n1 "${{{VERIFY_EVENT_LOG_ENV}}}" >> {side_file} 2>/dev/null || true\n'
    )


def side_file(worktree: Path, name: str) -> tuple[str, dict[str, object] | None]:
    lines = (worktree / name).read_text().splitlines()
    first_record = json.loads(lines[1]) if len(lines) > 1 else None
    return lines[0], first_record


async def verify(
    worktree: Path, *, lint: bool = False, role: Role = 'task',
    verify_env: dict[str, str] | None = None,
) -> VerifyResult:
    config = OrchestratorConfig(
        project_root=worktree,
        verify_admission_enabled=False,
        verify_use_cgroup_scope=False,
        verify_cold_preprovision_command='',
        verify_env=verify_env or {},
    )
    module = ModuleConfig(
        prefix='__fallback__',
        test_command='bash ./test_leg.sh',
        lint_command='bash ./lint_leg.sh' if lint else None,
        type_check_command=None,
    )
    return await run_verification(
        worktree, config, module, allow_cold_cache=False, max_retries=0, role=role,
    )


pytestmark = pytest.mark.usefixtures('code_default_config')


@pytest.mark.asyncio
async def test_task_role_fatal_record_without_run_start_is_a_semaphore_timeout(
    tmp_path: Path,
) -> None:
    write_leg(tmp_path, 'test_leg.sh', (
        f'cat {SLOT_FATAL} >> "${VERIFY_EVENT_LOG_ENV}"\n'
        f'echo {shlex.quote(ORDINARY_FAILURE)}\n'
        'exit 1\n'
    ))

    result = await verify(tmp_path)

    assert result.category == FailureCategory.SEMAPHORE_TIMEOUT
    assert result.failing_leg_categories == [FailureCategory.SEMAPHORE_TIMEOUT]


@pytest.mark.asyncio
async def test_the_test_leg_channel_names_the_failed_run_all_members(tmp_path: Path) -> None:
    write_leg(tmp_path, 'test_leg.sh', (
        f'cat {POOL} >> "${VERIFY_EVENT_LOG_ENV}"\n'
        'echo "FAILED other.sh"\n'
        'exit 1\n'
    ))

    result = await verify(tmp_path)

    assert result.run_all_failed_members == ['test_fail.sh']


@pytest.mark.asyncio
async def test_task_role_sentinel_without_a_record_is_not_a_semaphore_timeout(
    tmp_path: Path,
) -> None:
    write_leg(tmp_path, 'test_leg.sh', f'echo {shlex.quote(FATAL_SENTINEL)} >&2\nexit 1\n')

    result = await verify(tmp_path)

    assert result.category != FailureCategory.SEMAPHORE_TIMEOUT
    assert result.run_all_failed_members == []


@pytest.mark.asyncio
async def test_failed_allocation_falls_back_to_the_text_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_leg(tmp_path, 'test_leg.sh', (
        record_channel('channel.txt')
        + f'echo {shlex.quote(FATAL_SENTINEL)} >&2\nexit 1\n'
    ))
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path / 'missing'))

    result = await verify(tmp_path)

    assert result.category == FailureCategory.SEMAPHORE_TIMEOUT
    assert result.run_all_failed_members is None
    assert side_file(tmp_path, 'channel.txt') == ('', None)


@pytest.mark.parametrize('role', ['task', 'background'])
@pytest.mark.asyncio
async def test_each_leg_gets_its_own_channel_removed_after_the_verify(
    tmp_path: Path, role: Role,
) -> None:
    write_leg(tmp_path, 'test_leg.sh', record_channel('test-channel.txt') + 'exit 0\n')
    write_leg(tmp_path, 'lint_leg.sh', record_channel('lint-channel.txt') + 'exit 0\n')

    result = await verify(tmp_path, lint=True, role=role)

    assert result.passed, result.summary
    test_path, test_header = side_file(tmp_path, 'test-channel.txt')
    lint_path, lint_header = side_file(tmp_path, 'lint-channel.txt')
    assert Path(test_path).is_absolute() and Path(lint_path).is_absolute()
    assert test_path != lint_path
    for header in (test_header, lint_header):
        assert header is not None
        assert header['event'] == 'channel_open'
        assert header['v'] == 1
        assert header['role'] == role
        assert isinstance(header['verify_id'], str)
    assert test_header is not None and lint_header is not None
    assert test_header['verify_id'] != lint_header['verify_id']
    assert not Path(test_path).exists()
    assert not Path(lint_path).exists()


@pytest.mark.asyncio
async def test_an_operator_supplied_channel_is_overridden(tmp_path: Path) -> None:
    operator = tmp_path / 'operator.jsonl'
    write_leg(tmp_path, 'test_leg.sh', record_channel('channel.txt') + 'exit 0\n')

    await verify(tmp_path, verify_env={VERIFY_EVENT_LOG_ENV: str(operator)})

    recorded, header = side_file(tmp_path, 'channel.txt')
    assert recorded not in ('', str(operator))
    assert header is not None and header['event'] == 'channel_open'
    assert not operator.exists()


@pytest.mark.asyncio
async def test_the_env_recovery_rerun_classifies_with_its_own_channel(tmp_path: Path) -> None:
    assert classify_failure(ToolKind.OPAQUE, 1, ENV_TRANSIENT_LINE + '\n', False) == (
        FailureCategory.ENV_TRANSIENT
    )
    write_leg(tmp_path, 'test_leg.sh', (
        'n=$(( $(cat counter 2>/dev/null || echo 0) + 1 ))\n'
        'echo "$n" > counter\n'
        'if [ "$n" = 1 ]; then\n'
        f'  echo {shlex.quote(ENV_TRANSIENT_LINE)}\n'
        '  exit 1\n'
        'fi\n'
        f'cat {SLOT_FATAL} >> "${VERIFY_EVENT_LOG_ENV}"\n'
        f'echo {shlex.quote(ORDINARY_FAILURE)}\n'
        'exit 1\n'
    ))

    result = await verify(tmp_path)

    assert (tmp_path / 'counter').read_text().strip() == '2'
    assert result.category == FailureCategory.SEMAPHORE_TIMEOUT


def _bare_result(run_all_failed_members: list[str] | None) -> VerifyResult:
    return VerifyResult(
        passed=False, test_output='', lint_output='', type_output='', summary='',
        run_all_failed_members=run_all_failed_members,
    )


def test_run_all_failed_members_round_trips_the_verify_runner_codec() -> None:
    result = _bare_result(run_all_failed_members=['a.sh'])

    assert result_from_dict(result_to_dict(result)).run_all_failed_members == ['a.sh']


def test_a_payload_without_run_all_failed_members_reads_as_absent() -> None:
    payload = result_to_dict(_bare_result(run_all_failed_members=['a.sh']))
    del payload['run_all_failed_members']

    assert result_from_dict(payload).run_all_failed_members is None
