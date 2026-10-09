"""Tests for orchestrator.review_pre_test — the review checkpoint's pre-test setup runner.

Every runner test drives real bash subprocesses against real files under
tmp_path; nothing is patched.
"""

from __future__ import annotations

import logging
import shlex
import time

import pytest

from orchestrator.review_pre_test import (
    PreTestCommandResult,
    render_pre_test_section,
    run_pre_test_commands,
)

_LOGGER = 'orchestrator.review_pre_test'


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
class TestRunPreTestCommands:
    async def test_commands_run_in_order_in_cwd(self, tmp_path):
        proj = tmp_path / 'proj'
        proj.mkdir()
        log = shlex.quote(str(tmp_path / 'order.log'))
        commands = [
            f"printf 'one %s\\n' \"$PWD\" >> {log}",
            f"printf 'two\\n' >> {log}",
        ]

        results = await run_pre_test_commands(
            commands, cwd=proj, timeout_secs=30, env=None,
        )

        assert (tmp_path / 'order.log').read_text().splitlines() == [f'one {proj}', 'two']
        assert [r.command for r in results] == commands
        for r in results:
            assert r.returncode == 0
            assert r.timed_out is False
            assert r.failed is False

    async def test_empty_commands_is_a_noop(self, tmp_path):
        assert await run_pre_test_commands([], cwd=tmp_path, timeout_secs=30, env=None) == ()

    async def test_failing_command_is_reported_and_later_commands_still_run(
        self, tmp_path, caplog,
    ):
        marker = tmp_path / 'later-ran'
        failing = 'echo boom-tail-marker; exit 3'
        commands = [failing, f'touch {shlex.quote(str(marker))}']

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            results = await run_pre_test_commands(
                commands, cwd=tmp_path, timeout_secs=30, env=None,
            )

        assert results[0].returncode == 3
        assert results[0].failed is True
        assert 'boom-tail-marker' in results[0].output_tail
        assert marker.exists()
        assert results[1].failed is False
        assert any(
            failing in msg and '3' in msg and 'boom-tail-marker' in msg
            for msg in _warnings(caplog)
        ), _warnings(caplog)

    async def test_timeout_is_bounded_and_reported(self, tmp_path, caplog):
        command = 'sleep 30'
        start = time.monotonic()

        with caplog.at_level(logging.WARNING, logger=_LOGGER):
            results = await run_pre_test_commands(
                [command], cwd=tmp_path, timeout_secs=0.5, env=None,
            )

        assert time.monotonic() - start < 15
        assert results[0].timed_out is True
        assert results[0].failed is True
        assert any(command in msg for msg in _warnings(caplog)), _warnings(caplog)

    async def test_output_tail_is_bounded(self, tmp_path):
        command = "printf 'x%.0s' $(seq 1 20000); echo END-MARKER"

        results = await run_pre_test_commands(
            [command], cwd=tmp_path, timeout_secs=30, env=None,
        )

        tail = results[0].output_tail
        assert len(tail) < 20000
        assert tail.rstrip().endswith('END-MARKER')

    async def test_env_overlay_reaches_command(self, tmp_path):
        out = tmp_path / 'probe.txt'
        command = f'printf %s "$DF_PRE_TEST_PROBE" > {shlex.quote(str(out))}'

        await run_pre_test_commands(
            [command], cwd=tmp_path, timeout_secs=30,
            env={'DF_PRE_TEST_PROBE': 'probe-value'},
        )

        assert out.read_text() == 'probe-value'


def _result(
    command: str,
    *,
    returncode: int = 0,
    timed_out: bool = False,
    output_tail: str = '',
) -> PreTestCommandResult:
    return PreTestCommandResult(
        command=command,
        returncode=returncode,
        timed_out=timed_out,
        duration_secs=1.5,
        output_tail=output_tail,
    )


class TestRenderPreTestSection:
    def test_no_results_render_empty(self):
        assert render_pre_test_section(()) == ''

    def test_all_success_lists_each_command_without_failed(self):
        rendered = render_pre_test_section(
            (_result('npm ci'), _result('npm run build')),
        )

        assert 'npm ci' in rendered
        assert 'npm run build' in rendered
        assert 'FAILED' not in rendered

    def test_failed_result_carries_exit_code_and_tail(self):
        rendered = render_pre_test_section(
            (_result('make bundle', returncode=7, output_tail='tail-xyz'),),
        )

        assert 'make bundle' in rendered
        assert '7' in rendered
        assert 'tail-xyz' in rendered
        assert 'FAILED' in rendered

    def test_timed_out_result_says_timed_out(self):
        rendered = render_pre_test_section(
            (_result('slow build', returncode=-9, timed_out=True),),
        )

        assert 'slow build' in rendered
        assert 'timed out' in rendered.lower()
