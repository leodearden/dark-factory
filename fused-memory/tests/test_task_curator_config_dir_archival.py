"""TaskCurator per-process config-dir transcript archival (task 6008)."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from _curator_helpers import agent_result, empty_corpus, escalator_double, make_config
from shared.config_dir import CONFIG_DIR_PREFIX, reset_sweep_once_state, sweep_stale_pid_dirs
from shared.usage_gate import UsageGate

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.middleware.task_curator import CandidateTask, TaskCurator
from fused_memory.reconciliation.cli_stage_runner import recon_archive_root

_CURATOR_MODULE = 'fused_memory.middleware.task_curator'
_INVOKE = f'{_CURATOR_MODULE}.invoke_with_cap_retry'
_SWEEP = f'{_CURATOR_MODULE}.sweep_stale_pid_dirs'
_CURATOR_SWEEP_PREFIX = CONFIG_DIR_PREFIX + 'fm-curator-'
_SWEEP_CEILING_SECONDS = 30.0
_CREATE = {'action': 'create', 'justification': 'x'}

_DEAD_ENC = '-tmp-fm-neutral-classifier-cwd-x'
_DEAD_SESSION = '0b6f3a1e-5c2d-4e8f-9a7b-1c2d3e4f5a6b'
_DEAD_TRANSCRIPTS = {
    Path(_DEAD_ENC, f'{_DEAD_SESSION}.jsonl'): b'{"type":"user","from":"main"}\n',
    Path(_DEAD_ENC, _DEAD_SESSION, 'subagents', 'agent-a.jsonl'): b'{"type":"user","from":"sub"}\n',
}
_MAIN_TRANSCRIPT = Path(_DEAD_ENC, f'{_DEAD_SESSION}.jsonl')


def _find_dead_pid() -> int:
    for candidate in range(999_000, 999_000 + 5000):
        try:
            os.kill(candidate, 0)
        except ProcessLookupError:
            return candidate
        except OSError:
            continue
    raise RuntimeError('could not find a dead PID to test against')


def _backdate(path: Path, secs: float = 3600.0) -> None:
    old = time.time() - secs
    os.utime(path, (old, old))


def _plant_dead_curator_dir(base: Path, pid: int) -> Path:
    """A dead curator's config dir, as a SIGKILL leaves it: credential plus transcripts."""
    config_dir = base / f'{CONFIG_DIR_PREFIX}fm-curator-{pid}'
    config_dir.mkdir(parents=True)
    (config_dir / '.credentials.json').write_text('{"token": "secret"}')
    for rel, body in _DEAD_TRANSCRIPTS.items():
        transcript = config_dir / 'projects' / rel
        transcript.parent.mkdir(parents=True, exist_ok=True)
        transcript.write_bytes(body)
    _backdate(config_dir)
    return config_dir


def _gated_curator(config_dir_base: Path, config: FusedMemoryConfig) -> TaskCurator:
    return TaskCurator(
        config=config,
        taskmaster=None,
        usage_gate=MagicMock(spec=UsageGate),
        config_dir_base=config_dir_base,
        escalator=escalator_double(),
    )


class _CompletingSweep:
    """The real dead-PID sweep, signalling when it has finished.

    The curator fires the sweep on the default executor without awaiting it,
    so this is the only completion signal a test can wait on.
    """

    def __init__(self) -> None:
        self.done = threading.Event()

    def __call__(self, *args: Any, **kwargs: Any) -> int:
        try:
            return sweep_stale_pid_dirs(*args, **kwargs)
        finally:
            self.done.set()


async def _curate_and_sweep(curator: TaskCurator) -> dict[str, Any]:
    """One successful gated curate; returns its LLM call kwargs once the sweep has run."""
    sweep = _CompletingSweep()
    invoke = AsyncMock(return_value=agent_result(_CREATE))
    with patch(_SWEEP, new=sweep), patch(_INVOKE, new=invoke), \
            patch.object(curator, '_build_corpus', side_effect=empty_corpus):
        await curator.curate(CandidateTask(title='T'), project_id='p', project_root='/p')
        assert await asyncio.to_thread(sweep.done.wait, _SWEEP_CEILING_SECONDS)
    return dict(invoke.call_args.kwargs)


def _archive_root(config: FusedMemoryConfig) -> Path:
    root = recon_archive_root(config.reconciliation)
    assert root is not None
    return root


def _make_archive_root_unreachable(config: FusedMemoryConfig) -> None:
    """Point the archive root beneath a regular file, so every archive write fails."""
    (Path(config.reconciliation.explore_codebase_root) / 'blocker').write_text('not a dir')
    object.__setattr__(config.reconciliation, 'transcript_archive_root', 'blocker/archive')


def _reclaim_failures(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The reasons the sweep logged for each dir it failed to reclaim."""
    return [
        str(r.exc_info[1])
        for r in caplog.records
        if r.levelno >= logging.WARNING and 'failed to reclaim' in r.getMessage()
        and r.exc_info is not None
    ]


@pytest.fixture
def _fresh_curator_sweep():
    """The curator sweep runs once per process, so an earlier gated test in this
    worker would otherwise have consumed it."""
    reset_sweep_once_state(_CURATOR_SWEEP_PREFIX)
    yield
    reset_sweep_once_state(_CURATOR_SWEEP_PREFIX)


@pytest.mark.usefixtures('_fresh_curator_sweep')
class TestDeadPidSweepArchivesFirst:
    """A later process's dead-PID sweep archives a SIGKILLed curator's transcripts."""

    @pytest.mark.asyncio
    async def test_a_dead_curators_transcripts_are_archived_before_its_dir_is_removed(
        self, tmp_path,
    ):
        base = tmp_path / 'tmp'
        dead = _find_dead_pid()
        planted = _plant_dead_curator_dir(base, dead)
        config = make_config()
        root = _archive_root(config)

        await _curate_and_sweep(_gated_curator(base, config))

        for rel, body in _DEAD_TRANSCRIPTS.items():
            assert (root / f'fm-curator-{dead}' / rel).read_bytes() == body
        assert not planted.exists()
        assert not list(root.rglob('.credentials.json'))

    @pytest.mark.asyncio
    async def test_an_unarchivable_transcript_is_held_and_archived_by_the_next_processs_sweep(
        self, tmp_path, caplog,
    ):
        base = tmp_path / 'tmp'
        dead = _find_dead_pid()
        planted = _plant_dead_curator_dir(base, dead)
        config = make_config()
        _make_archive_root_unreachable(config)

        with caplog.at_level(logging.WARNING):
            await _curate_and_sweep(_gated_curator(base, config))

        assert (planted / 'projects' / _MAIN_TRANSCRIPT).read_bytes() == _DEAD_TRANSCRIPTS[
            _MAIN_TRANSCRIPT
        ]
        assert not (planted / '.credentials.json').exists()
        assert [
            reason for reason in _reclaim_failures(caplog)
            if f'{len(_DEAD_TRANSCRIPTS)} transcript(s) held' in reason
        ]

        reset_sweep_once_state(_CURATOR_SWEEP_PREFIX)
        _backdate(planted)
        object.__setattr__(config.reconciliation, 'transcript_archive_root', 'reachable/archive')
        root = _archive_root(config)

        await _curate_and_sweep(_gated_curator(base, config))

        assert (root / f'fm-curator-{dead}' / _MAIN_TRANSCRIPT).read_bytes() == (
            _DEAD_TRANSCRIPTS[_MAIN_TRANSCRIPT]
        )
        assert not planted.exists()

    @pytest.mark.skipif(os.geteuid() == 0, reason='root ignores directory write permission')
    @pytest.mark.asyncio
    async def test_a_dir_archived_but_not_removable_is_reported_as_a_removal_failure(
        self, tmp_path, caplog,
    ):
        base = tmp_path / 'tmp'
        dead = _find_dead_pid()
        planted = _plant_dead_curator_dir(base, dead)
        locked = planted / 'sessions' / 'locked'
        locked.mkdir(parents=True)
        (locked / 'pinned').write_text('x')
        locked.chmod(0o500)
        _backdate(planted)
        config = make_config()
        root = _archive_root(config)

        try:
            with caplog.at_level(logging.WARNING):
                await _curate_and_sweep(_gated_curator(base, config))
        finally:
            locked.chmod(0o700)

        for rel, body in _DEAD_TRANSCRIPTS.items():
            assert (root / f'fm-curator-{dead}' / rel).read_bytes() == body
        reasons = _reclaim_failures(caplog)
        assert any('could not be removed' in reason for reason in reasons)
        assert not any('held' in reason for reason in reasons)

    @pytest.mark.asyncio
    async def test_with_archival_switched_off_the_sweep_still_reclaims_without_archiving(
        self, tmp_path,
    ):
        base = tmp_path / 'tmp'
        planted = _plant_dead_curator_dir(base, _find_dead_pid())
        config = make_config()
        would_be_root = _archive_root(config)
        object.__setattr__(config.reconciliation, 'transcript_archive_enabled', False)

        await _curate_and_sweep(_gated_curator(base, config))

        assert not planted.exists()
        assert not would_be_root.exists()


_LIVE_ENC = '-tmp-fm-neutral-classifier-cwd-y'
_LIVE_TRANSCRIPT = b'{"type":"assistant","from":"this process"}\n'


def _plant_live_transcript(invoke_kwargs: dict[str, Any]) -> Path:
    """Write the transcript where the CLI would have for this gated call."""
    transcript = (
        invoke_kwargs['config_dir'].path / 'projects' / _LIVE_ENC
        / f'{invoke_kwargs["session_id"]}.jsonl'
    )
    transcript.parent.mkdir(parents=True, exist_ok=True)
    transcript.write_bytes(_LIVE_TRANSCRIPT)
    return transcript


@pytest.mark.usefixtures('_fresh_curator_sweep')
class TestCloseArchivesThisProcessesDir:
    """Clean shutdown archives this process's own config dir, then removes it."""

    @pytest.mark.asyncio
    async def test_close_archives_this_process_transcripts_and_removes_its_dir(self, tmp_path):
        config = make_config()
        curator = _gated_curator(tmp_path / 'tmp', config)
        kwargs = await _curate_and_sweep(curator)
        _plant_live_transcript(kwargs)

        await curator.close()

        archived = (
            _archive_root(config) / f'fm-curator-{os.getpid()}' / _LIVE_ENC
            / f'{kwargs["session_id"]}.jsonl'
        )
        assert archived.read_bytes() == _LIVE_TRANSCRIPT
        assert not kwargs['config_dir'].path.exists()

    @pytest.mark.asyncio
    async def test_close_with_an_unreachable_archive_root_holds_the_transcript_and_purges_the_credential(
        self, tmp_path,
    ):
        config = make_config()
        curator = _gated_curator(tmp_path / 'tmp', config)
        kwargs = await _curate_and_sweep(curator)
        transcript = _plant_live_transcript(kwargs)
        kwargs['config_dir'].write_credentials('secret')
        _make_archive_root_unreachable(config)

        await curator.close()

        assert transcript.read_bytes() == _LIVE_TRANSCRIPT
        assert not (kwargs['config_dir'].path / '.credentials.json').exists()

    @pytest.mark.asyncio
    async def test_close_with_archival_switched_off_leaves_the_dir_for_the_sweep(self, tmp_path):
        config = make_config()
        curator = _gated_curator(tmp_path / 'tmp', config)
        kwargs = await _curate_and_sweep(curator)
        transcript = _plant_live_transcript(kwargs)
        would_be_root = _archive_root(config)
        object.__setattr__(config.reconciliation, 'transcript_archive_enabled', False)

        await curator.close()

        assert kwargs['config_dir'].path.is_dir()
        assert transcript.read_bytes() == _LIVE_TRANSCRIPT
        assert not would_be_root.exists()
