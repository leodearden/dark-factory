"""Review Phase 1 spawns the configured checks in the main checkout, even one moved after its venv was built.

Drives the public ``ReviewCheckpoint.run_focused`` with a real config and the
real verify spawn path.  The only double is ``uv`` itself: a fake on PATH that
behaves like uv where this bug lives.  It uses ``<cwd>/.venv`` (building it
when absent), trusts packages already installed there, and execs
``.venv/bin/<tool>``, printing uv's own "Failed to spawn" text when that exec
fails.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from _venv_fixtures import make_venv
from shared.cli_invoke import AgentResult

from orchestrator.config import OrchestratorConfig
from orchestrator.review_checkpoint import ReviewCheckpoint, ReviewReport

pytestmark = [pytest.mark.asyncio, pytest.mark.usefixtures('code_default_config')]

_UV_RUN = ['run', '--locked', '--extra', 'dev', '--extra', 'web']
_TEST_COMMAND = (
    "uv run --locked --extra dev --extra web pytest -m 'not slow and not e2e' --ignore=tests/e2e"
)
_TYPE_CHECK_COMMAND = 'uv run --locked --extra dev --extra web mypy src/solar_challenge'

_FAKE_UV_BODY = r'''
import json
import os
import sys
from pathlib import Path

_FLAGS_WITH_VALUE = {'--extra'}


def main(argv):
    assert argv[0] == 'run', argv
    rest = argv[1:]
    i = 0
    while rest[i].startswith('--'):
        i += 2 if rest[i] in _FLAGS_WITH_VALUE else 1
    tool, args = rest[i], rest[i + 1:]
    venv = Path.cwd() / '.venv'
    with open(LOG, 'a') as fh:
        fh.write(json.dumps({
            'argv': argv,
            'cwd': os.getcwd(),
            'virtual_env': os.environ.get('VIRTUAL_ENV'),
            'uv_run_recursion_depth': os.environ.get('UV_RUN_RECURSION_DEPTH'),
            'venv_existed': venv.is_dir(),
        }) + '\n')
    if not venv.is_dir():
        (venv / 'bin').mkdir(parents=True)
        (venv / 'pyvenv.cfg').write_text('uv = fake\n')
        for name in ('pytest', 'mypy'):
            script = venv / 'bin' / name
            script.write_text('#!/bin/sh\nexit 0\n')
            script.chmod(0o755)
    try:
        os.execv(venv / 'bin' / tool, [tool, *args])
    except FileNotFoundError:
        sys.stderr.write(
            f'error: Failed to spawn: `{tool}`\n'
            '  Caused by: No such file or directory (os error 2)\n'
        )
        sys.exit(2)


main(sys.argv[1:])
'''


@dataclass(frozen=True)
class Scene:
    project: Path
    uv_log: Path

    def uv_calls(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.uv_log.read_text().splitlines()]


@pytest.fixture
def scene(monkeypatch: pytest.MonkeyPatch) -> Iterator[Scene]:
    with tempfile.TemporaryDirectory(prefix='df-7077-phase1-spawn-') as scratch:
        root = Path(scratch)
        project = root / 'checkout'
        project.mkdir()
        assert '/tmp/pytest' not in str(project), (
            f'{project} contains "/tmp/pytest", which ReviewCheckpoint._run_review refuses '
            'as a project_root; this fixture must live outside pytest basetemp'
        )
        uv_log = root / 'uv-calls.jsonl'
        fake_uv = root / 'fake_uv.py'
        fake_uv.write_text(f'LOG = {str(uv_log)!r}\n' + _FAKE_UV_BODY)
        fakebin = root / 'fakebin'
        fakebin.mkdir()
        uv = fakebin / 'uv'
        uv.write_text(
            f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(fake_uv))} "$@"\n'
        )
        uv.chmod(0o755)
        orch_venv = root / 'orch-venv'
        monkeypatch.setenv(
            'PATH', os.pathsep.join((str(fakebin), str(orch_venv / 'bin'), os.environ['PATH'])),
        )
        monkeypatch.setenv('VIRTUAL_ENV', str(orch_venv))
        monkeypatch.setenv('UV_RUN_RECURSION_DEPTH', '1')
        yield Scene(project=project, uv_log=uv_log)


@pytest.fixture
def reviewer(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    reviewer = AsyncMock(return_value=AgentResult(
        success=True, output='', subtype='success',
        structured_output={'findings': [], 'summary': 'ok'},
    ))
    monkeypatch.setattr('orchestrator.review_checkpoint.invoke_with_cap_retry', reviewer)
    return reviewer


def _checkpoint(project: Path) -> ReviewCheckpoint:
    config = OrchestratorConfig(
        project_root=project,
        test_command=_TEST_COMMAND,
        lint_command='true',
        type_check_command=_TYPE_CHECK_COMMAND,
        concurrent_verify=False,
        verify_use_cgroup_scope=False,
    )
    mcp = MagicMock()
    mcp.mcp_config_json.return_value = {'mcpServers': {}}
    return ReviewCheckpoint(config, mcp=mcp, usage_gate=None)


def _saved_report(cp: ReviewCheckpoint, report: ReviewReport) -> dict[str, Any]:
    reports_dir = cp.config.project_root / cp.config.review.reports_dir
    return json.loads((reports_dir / f'review-{report.review_id}.json').read_text())


def _phase1_output(report: ReviewReport) -> str:
    return f'{report.phase1.summary}\n{report.phase1.test_output}\n{report.phase1.type_output}'


async def test_relocated_venv_is_rebuilt_and_phase1_measures_the_configured_commands(
    scene: Scene, reviewer: AsyncMock,
) -> None:
    make_venv(scene.project / '.venv', built_at=scene.project.parent / 'my-checkout' / '.venv')
    cp = _checkpoint(scene.project)

    report = await cp.run_focused()

    assert report.phase1.passed is True, _phase1_output(report)
    assert _saved_report(cp, report)['phase1_passed'] is True
    calls = scene.uv_calls()
    by_tool = {call['argv'][len(_UV_RUN)]: call for call in calls}
    assert sorted(call['argv'][len(_UV_RUN)] for call in calls) == ['mypy', 'pytest']
    assert calls[0]['venv_existed'] is False
    for call in calls:
        assert call['cwd'] == str(scene.project)
        assert call['virtual_env'] is None
        assert call['uv_run_recursion_depth'] is None
    pytest_argv = by_tool['pytest']['argv']
    assert pytest_argv[:len(_UV_RUN) + 1] == [*_UV_RUN, 'pytest']
    assert pytest_argv[pytest_argv.index('-m') + 1] == 'not slow and not e2e'
    assert '--ignore=tests/e2e' in pytest_argv
    assert by_tool['mypy']['argv'] == [*_UV_RUN, 'mypy', 'src/solar_challenge']


async def test_healthy_venv_is_measured_in_place(scene: Scene, reviewer: AsyncMock) -> None:
    venv = make_venv(scene.project / '.venv')
    sentinel = venv / 'sentinel'
    sentinel.write_text('keep')

    report = await _checkpoint(scene.project).run_focused()

    assert report.phase1.passed is True, _phase1_output(report)
    calls = scene.uv_calls()
    assert calls and all(call['venv_existed'] for call in calls)
    assert sentinel.read_text() == 'keep'
