"""Unit tests for ``orchestrator/src/orchestrator/staged_gitlinks.py`` (task 3389).

The parser is fed literal ``git diff --cached --raw -z`` output; the unstage
step is driven through its injected runner, which is how its failure branches
are reached.  The real-git behaviour is pinned in test_git_ops_commit_staging.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from orchestrator.staged_gitlinks import newly_staged_gitlinks, unstage_new_gitlinks

_ZERO = '0000000'
_OLD = 'a1b2c3d'
_NEW = 'e4f5a6b'


def _record(old_mode: str, new_mode: str, status: str, path: str) -> str:
    old_sha = _ZERO if old_mode == '000000' else _OLD
    new_sha = _ZERO if new_mode == '000000' else _NEW
    return f':{old_mode} {new_mode} {old_sha} {new_sha} {status}\0{path}\0'


class TestNewlyStagedGitlinks:
    def test_empty_diff_has_none(self) -> None:
        assert newly_staged_gitlinks('') == ()

    def test_added_gitlink_is_returned(self) -> None:
        assert newly_staged_gitlinks(_record('000000', '160000', 'A', 'scratch/t0')) == (
            'scratch/t0',
        )

    def test_file_to_gitlink_typechange_is_returned(self) -> None:
        assert newly_staged_gitlinks(_record('100644', '160000', 'T', 'vendor/x')) == (
            'vendor/x',
        )

    def test_submodule_pointer_bump_is_not_returned(self) -> None:
        assert newly_staged_gitlinks(_record('160000', '160000', 'M', 'graphiti')) == ()

    def test_gitlink_deletion_is_not_returned(self) -> None:
        assert newly_staged_gitlinks(_record('160000', '000000', 'D', 'graphiti')) == ()

    def test_ordinary_file_changes_are_not_returned(self) -> None:
        raw = _record('000000', '100644', 'A', 'new.py') + _record(
            '100644', '100755', 'M', 'run.sh',
        )
        assert newly_staged_gitlinks(raw) == ()

    def test_paths_are_returned_verbatim_in_input_order(self) -> None:
        raw = (
            _record('000000', '160000', 'A', 'dir with space/t0')
            + _record('000000', '100644', 'A', 'work.py')
            + _record('000000', '160000', 'A', 'line\nbreak/t1')
        )
        assert newly_staged_gitlinks(raw) == ('dir with space/t0', 'line\nbreak/t1')

    def test_record_without_a_path_raises(self) -> None:
        raw = _record('000000', '160000', 'A', 'scratch/t0') + f':000000 160000 {_ZERO} {_NEW} A'
        with pytest.raises(ValueError, match='without a path'):
            newly_staged_gitlinks(raw)

    @pytest.mark.parametrize(
        'meta',
        [
            f'000000 160000 {_ZERO} {_NEW} A',
            f':000000 160000 {_ZERO} A',
            f':000000 160000 {_ZERO} {_NEW} A extra',
        ],
    )
    def test_malformed_record_raises(self, meta: str) -> None:
        with pytest.raises(ValueError, match='malformed'):
            newly_staged_gitlinks(f'{meta}\0scratch/t0\0')


class _ScriptedRunner:
    def __init__(self, *replies: tuple[int, str, str]) -> None:
        self._replies = list(replies)
        self.calls: list[list[str]] = []

    async def __call__(self, cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
        self.calls.append(cmd)
        return self._replies.pop(0)


class TestUnstageNewGitlinks:
    @pytest.mark.asyncio
    async def test_nothing_to_unstage_runs_no_reset(self, tmp_path: Path) -> None:
        run = _ScriptedRunner((0, _record('000000', '100644', 'A', 'work.py'), ''))

        assert await unstage_new_gitlinks(tmp_path, run) == ()
        assert len(run.calls) == 1

    @pytest.mark.asyncio
    async def test_failed_staged_listing_raises(self, tmp_path: Path) -> None:
        run = _ScriptedRunner((128, '', 'fatal: index file corrupt\n'))

        with pytest.raises(RuntimeError) as excinfo:
            await unstage_new_gitlinks(tmp_path, run)

        message = str(excinfo.value)
        assert str(tmp_path) in message
        assert 'rc=128' in message
        assert 'index file corrupt' in message
        assert len(run.calls) == 1

    @pytest.mark.asyncio
    async def test_failed_reset_raises(self, tmp_path: Path) -> None:
        run = _ScriptedRunner(
            (0, _record('000000', '160000', 'A', 'scratch/t0'), ''),
            (1, '', 'fatal: Unable to create index.lock\n'),
        )

        with pytest.raises(RuntimeError) as excinfo:
            await unstage_new_gitlinks(tmp_path, run)

        message = str(excinfo.value)
        assert 'scratch/t0' in message
        assert str(tmp_path) in message
        assert 'rc=1' in message
        assert 'index.lock' in message
        assert run.calls[1] == [
            'git', '--literal-pathspecs', 'reset', '-q', 'HEAD', '--', 'scratch/t0',
        ]

    @pytest.mark.asyncio
    async def test_warning_lists_at_most_ten_paths_and_counts_the_rest(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture,
    ) -> None:
        paths = [f's/a{index:02d}' for index in range(12)]
        run = _ScriptedRunner(
            (0, ''.join(_record('000000', '160000', 'A', path) for path in paths), ''),
            (0, '', ''),
        )

        with caplog.at_level(logging.WARNING, logger='orchestrator.staged_gitlinks'):
            assert await unstage_new_gitlinks(tmp_path, run) == tuple(paths)

        assert run.calls[1][-12:] == paths
        [message] = [record.getMessage() for record in caplog.records]
        assert message.startswith('left 12 embedded git repositories')
        assert 's/a09 (+2 more)' in message
        assert 's/a10' not in message
