"""Warning lines picked out of a reify-style script's tagged stderr."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from _lane_test_helpers import make_warm_lane_repo
from _seed_script_stubs import commit_seed_script, warm_pool_git_config

from orchestrator.git_ops import GitOps
from orchestrator.script_warnings import log_warnings_despite_success, tagged_warning_lines

SEED_WARN = '\x1b[1;33m[warn]\x1b[0m  hooks-armed-guard.sh arm failed (exit 9) — seed continues'
GUARD_WARN = '[warn] hooks-armed-guard: NOT armed in /x: reason'
GUARD_ERROR = '[error] hooks-armed-guard: setup-main-gate-worktree-config.sh failed to pin /x'


class TestTaggedWarningLines:
    def test_coloured_seed_warn_keeps_tag_and_drops_colour(self):
        lines = tagged_warning_lines(SEED_WARN + '\n')

        assert lines == ('[warn]  hooks-armed-guard.sh arm failed (exit 9) — seed continues',)
        assert not any('\x1b' in line for line in lines)

    def test_uncoloured_warn_and_error_lines_kept_in_order(self):
        stderr = f'{GUARD_WARN}\n{GUARD_ERROR}\n'

        assert tagged_warning_lines(stderr) == (GUARD_WARN, GUARD_ERROR)

    def test_info_ok_untagged_and_embedded_tags_excluded(self):
        stderr = '\n'.join(
            [
                '\x1b[1;34m[info]\x1b[0m  seeding lane',
                '[info] plain info',
                '\x1b[1;32m[ok]\x1b[0m  done',
                '[ok] plain ok',
                'an untagged line',
                '[info] previously saw [warn] x',
            ]
        )

        assert tagged_warning_lines(stderr) == ()

    def test_leading_whitespace_tagged_line_included_stripped(self):
        assert tagged_warning_lines('   [warn] indented\n') == ('[warn] indented',)

    def test_empty_and_whitespace_only_stderr(self):
        assert tagged_warning_lines('') == ()
        assert tagged_warning_lines('  \n\t\n   ') == ()

    def test_mixed_multiline_input_returns_only_warn_lines_in_order(self):
        continuation = '\x1b[1;33m[warn]\x1b[0m    run scripts/hooks-armed-guard.sh check /lane'
        stderr = '\n'.join(
            [
                '\x1b[1;34m[info]\x1b[0m  seeding /lane',
                SEED_WARN,
                '\x1b[1;32m[ok]\x1b[0m  target seeded',
                continuation,
            ]
        )

        result = tagged_warning_lines(stderr)

        assert isinstance(result, tuple)
        assert result == (
            '[warn]  hooks-armed-guard.sh arm failed (exit 9) — seed continues',
            '[warn]    run scripts/hooks-armed-guard.sh check /lane',
        )



class TestLogWarningsDespiteSuccess:
    def test_all_warning_lines_land_in_one_record_under_the_headline(
        self, caplog: pytest.LogCaptureFixture,
    ):
        log = logging.getLogger('test_script_warnings.success')
        stderr = f'[info] start\n{SEED_WARN}\n{GUARD_ERROR}\n'

        with caplog.at_level(logging.DEBUG, logger=log.name):
            log_warnings_despite_success(log, 'seed for /lane', stderr)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.WARNING
        message = record.getMessage()
        assert message.startswith('seed for /lane exited 0 but reported 2 warning(s)')
        assert 'hooks-armed-guard.sh arm failed (exit 9)' in message
        assert GUARD_ERROR in message
        assert '[info] start' not in message
        assert '\x1b' not in message

    def test_stderr_without_warning_lines_logs_nothing(
        self, caplog: pytest.LogCaptureFixture,
    ):
        log = logging.getLogger('test_script_warnings.success')

        with caplog.at_level(logging.DEBUG, logger=log.name):
            log_warnings_despite_success(log, 'seed for /lane', '[info] a\n[ok] b\n')

        assert caplog.records == []


_WARNING_SEED_SCRIPT = r"""#!/usr/bin/env bash
mkdir -p "$2/target"
echo seeded > "$2/target/seeded.bin"
printf '\033[1;34m[info]\033[0m  %s\n' 'stash-guard hooks armed' >&2
printf '\033[1;33m[warn]\033[0m  %s\n' 'hooks-armed-guard.sh arm failed (exit 9) — seed continues' >&2
exit 0
"""

_QUIET_SEED_SCRIPT = r"""#!/usr/bin/env bash
mkdir -p "$2/target"
echo seeded > "$2/target/seeded.bin"
printf '\033[1;34m[info]\033[0m  %s\n' 'stash-guard hooks armed' >&2
printf '\033[1;32m[ok]\033[0m  %s\n' 'target seeded' >&2
exit 0
"""


async def _warm_git_ops(tmp_path: Path, seed_body: str) -> GitOps:
    repo = make_warm_lane_repo(tmp_path)
    await commit_seed_script(repo, seed_body)
    return GitOps(warm_pool_git_config(), repo, warm_lane_pool_size=1)


def _seed_warning_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record for record in caplog.records
        if record.levelno == logging.WARNING
        and record.getMessage().startswith('_seed_warm_lane:')
    ]


@pytest.mark.asyncio
class TestSeedWarningsSurfacedOnSuccess:
    async def test_rc0_seed_warn_line_is_logged_once_without_colour(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        git_ops = await _warm_git_ops(tmp_path, _WARNING_SEED_SCRIPT)
        caplog.set_level(logging.DEBUG, logger='orchestrator.git_ops')

        info = await git_ops.create_worktree('A')

        lane = git_ops.worktree_base / '_lane-0'
        assert info.path == lane
        seed_warnings = _seed_warning_records(caplog)
        assert len(seed_warnings) == 1, [r.getMessage() for r in seed_warnings]
        message = seed_warnings[0].getMessage()
        assert 'hooks-armed-guard.sh arm failed (exit 9)' in message
        assert str(lane) in message
        assert not any('\x1b' in r.getMessage() for r in caplog.records)
        assert not any(
            'stash-guard hooks armed' in r.getMessage()
            for r in caplog.records if r.levelno >= logging.WARNING
        )

    async def test_rc0_seed_with_only_info_and_ok_lines_logs_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ):
        git_ops = await _warm_git_ops(tmp_path, _QUIET_SEED_SCRIPT)
        caplog.set_level(logging.DEBUG, logger='orchestrator.git_ops')

        await git_ops.create_worktree('A')

        assert _seed_warning_records(caplog) == []
