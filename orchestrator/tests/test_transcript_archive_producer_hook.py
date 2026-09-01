"""Producer-hook coverage: TaskWorkflow._invoke's finally archives transcripts
(task 2742, plans/agent-transcript-archival-prd.md α).

Reuses the _invoke-probe pattern (build TaskWorkflow + patch
invoke_with_cap_retry) from test_invoke_role_config_resolution.py, with the
git_repo/git_ops/task_assignment fixture trio duplicated module-local per the
established convention. These probes drive _invoke directly (skipping run()'s
setup), so workflow._config_dir is set MANUALLY.

Fixtures are kept module-local (no conftest.py) — see
test_config_verify_admission_reload.py's rationale.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from shared.config_dir import TaskConfigDir

from orchestrator.agents.invoke import AgentResult
from orchestrator.agents.roles import SIMPLE_TASK
from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow

# The encoded-project dir the fake transcript is laid down under.
ENC = '-home-leo-projX'


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_init_repo(repo))
    return repo


async def _init_repo(repo: Path) -> None:
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'lib.py').write_text('def greet(name): return name\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


@pytest.fixture
def git_ops(git_repo: Path) -> GitOps:
    return GitOps(
        GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
        git_repo,
    )


@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id='42',
        task={
            'id': '42',
            'title': 'X',
            'description': 'Y',
            'status': 'pending',
            'metadata': {'files': ['lib']},
            'dependencies': [],
        },
        modules=['lib'],
    )


def _config(git_repo: Path, **overrides) -> OrchestratorConfig:
    kwargs: dict[str, Any] = dict(
        project_root=git_repo,
        max_concurrent_tasks=1,
        git=GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
    )
    kwargs.update(overrides)
    return OrchestratorConfig(**kwargs)


async def _make_workflow(config, git_ops, task_assignment):
    """Build a probe TaskWorkflow with _config_dir set manually (run() skipped)."""
    wt_info = await git_ops.create_worktree(task_assignment.task_id)
    cwd = wt_info.path
    workflow = TaskWorkflow(
        assignment=task_assignment,
        config=config,
        git_ops=git_ops,
        scheduler=FakeScheduler(),  # type: ignore[arg-type]
        briefing=FakeBriefing(),  # type: ignore[arg-type]
        mcp=FakeMcp(),  # type: ignore[arg-type]
    )
    workflow.artifacts = None
    # Direct _invoke skips run() setup where _config_dir is created.
    workflow._config_dir = TaskConfigDir(task_assignment.task_id, base_dir=cwd / '.task')
    return workflow, cwd


@pytest.mark.asyncio
class TestProducerHook:

    async def test_end_to_end_transcript_appears(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)  # transcript_archive enabled by default
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)
        config_dir = workflow._config_dir
        fake_bytes = b'{"transcript":"hello"}\n'

        def _side_effect(**kwargs):
            # The session id _invoke generated is forwarded as session_id=.
            assert config_dir is not None
            sid = kwargs['session_id']
            p = config_dir.path / 'projects' / ENC / f'{sid}.jsonl'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(fake_bytes)
            return AgentResult(success=True, output='')

        with patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            side_effect=_side_effect,
        ):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        sid = workflow._last_invoke_session_id
        archived = (
            git_repo / 'data' / 'orchestrator' / 'agent-transcripts'
            / task_assignment.task_id / ENC / f'{sid}.jsonl'
        )
        assert archived.exists()
        assert archived.read_bytes() == fake_bytes

    async def test_flag_guard_disabled_skips_helper(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo, transcript_archive={'enabled': False})
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)

        with patch('orchestrator.workflow.archive_task_transcripts') as mock_helper, patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        mock_helper.assert_not_called()

    async def test_arg_wiring(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)  # enabled by default
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)

        with patch('orchestrator.workflow.archive_task_transcripts') as mock_helper, patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        assert workflow._config_dir is not None
        mock_helper.assert_called_once_with(
            workflow._config_dir.path,
            workflow.task_id,
            workflow._last_invoke_session_id,
            archive_root=config.project_root / 'data/orchestrator/agent-transcripts',
        )

    async def test_helper_error_does_not_mask_in_flight(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """A non-cancellation error escaping the helper is swallowed, not raised.

        archive_task_transcripts guards per-file work with ``except OSError`` but
        NOT its top-level glob/Path/archive_root construction. Awaiting it inside
        _invoke's finally means any such escaped error would REPLACE the in-flight
        exception the finally is unwinding (finally-masks-original). The hook's own
        ``try/except Exception`` is the structural guarantee that it cannot — even
        if the helper's contract regresses (review #1).
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)  # enabled by default
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)

        with patch(
            'orchestrator.workflow.archive_task_transcripts',
            side_effect=RuntimeError('unguarded glob boom'),
        ) as mock_helper, patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ):
            # Must NOT raise RuntimeError out of the finally — _invoke returns
            # its result normally and the hook logs+swallows the archival error.
            result = await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        assert result.success is True
        mock_helper.assert_called_once()

    async def test_cancellation_propagates_not_swallowed(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """CancelledError from the archive await must re-raise, never be swallowed.

        Loop teardown / hard-kill surfaces CancelledError from the ``await``; the
        hook re-raises it (cooperative cancellation must propagate), so a killed
        invocation is deliberately not archived here — that abandoned tail is
        β/task 2729's teardown-backstop's job (review #2). The masking guard's
        ``except Exception`` deliberately does NOT catch CancelledError (a
        BaseException), so this asserts the two clauses are ordered correctly.
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)  # enabled by default
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)

        with patch(
            'orchestrator.workflow.archive_task_transcripts',
            side_effect=asyncio.CancelledError,
        ), patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            return_value=AgentResult(success=True, output=''),
        ), pytest.raises(asyncio.CancelledError):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)


@pytest.mark.asyncio
class TestCleanupConfigDirArchivesFirst:
    """Task 3619: the teardown sites must archive BEFORE they delete.

    The producer hook above archives on the way out of ``_invoke``. It is not
    enough on its own: it is skippable (a cancellation on its ``await``, an
    invocation that never reached it) while the ``rmtree`` in
    ``_cleanup_config_dir`` is not. So the same run could preserve the session
    sidecar and destroy the transcript the sidecar points at — the measured
    bug. These tests move the guard to the deletion site, where it cannot be
    routed around.
    """

    @staticmethod
    def _plant_transcript(workflow, sid: str = 'sess-teardown') -> tuple[Path, bytes]:
        assert workflow._config_dir is not None
        payload = b'{"transcript":"never archived"}\n'
        p = workflow._config_dir.path / 'projects' / ENC / f'{sid}.jsonl'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
        return p, payload

    async def test_cleanup_archives_the_transcript_it_is_about_to_destroy(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """(a) The measured bug, closed at its source."""
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)
        workflow, _cwd = await _make_workflow(config, git_ops, task_assignment)
        assert workflow._config_dir is not None
        config_dir_path = workflow._config_dir.path
        _src, payload = self._plant_transcript(workflow)

        workflow._cleanup_config_dir()

        archived = (
            git_repo / 'data' / 'orchestrator' / 'agent-transcripts'
            / task_assignment.task_id / ENC / 'sess-teardown.jsonl'
        )
        assert archived.read_bytes() == payload
        # ...and the dir is still torn down. Archival is a precondition of the
        # delete, not a replacement for it.
        assert not config_dir_path.exists()

    async def test_the_archive_root_composition_matches_the_other_two_sites(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """(b) One composition, so the already-current skip fires across all three.

        The producer hook, this teardown site and the cleanup_worktree backstop
        must resolve the SAME archive root from the SAME config leaves. If any
        one of them composed a different path, its writes would land in a second
        tree, the already-current check would never match, and every pass would
        re-archive into a corpus no reader looks at.
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)
        workflow, _cwd = await _make_workflow(config, git_ops, task_assignment)
        assert workflow._config_dir is not None
        config_dir_path = workflow._config_dir.path

        with patch('orchestrator.workflow.archive_before_delete') as mock_helper:
            workflow._cleanup_config_dir()

        mock_helper.assert_called_once_with(
            config_dir_path,
            workflow.task_id,
            archive_root=config.project_root / 'data/orchestrator/agent-transcripts',
        )

    async def test_preserve_still_short_circuits_and_touches_nothing(
        self, monkeypatch, git_repo, git_ops, task_assignment, caplog
    ):
        """(c) The forensic breaker path is untouched by this change.

        ``_preserve_config_dir`` exists so an on-call engineer can read a wedged
        task's config dir IN PLACE. Archiving there would be harmless but
        pointless; DELETING there would destroy the evidence the breaker tripped
        to collect. The early return must stay ahead of everything.
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)
        workflow, _cwd = await _make_workflow(config, git_ops, task_assignment)
        assert workflow._config_dir is not None
        config_dir_path = workflow._config_dir.path
        src, payload = self._plant_transcript(workflow)
        workflow._preserve_config_dir = True

        with patch(
            'orchestrator.workflow.archive_before_delete'
        ) as mock_helper, caplog.at_level(
            logging.WARNING, logger='orchestrator.workflow'
        ):
            workflow._cleanup_config_dir()

        mock_helper.assert_not_called()
        assert config_dir_path.exists()
        assert src.read_bytes() == payload
        assert any('preserved for forensic analysis' in r.getMessage() for r in caplog.records)

    async def test_the_kill_switch_gates_archival_never_teardown(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """(d) enabled=False → the plain cleanup() this wraps, nothing else.

        The flag turns the ARCHIVE off. If it also turned the teardown off, an
        operator disabling archival would silently start leaking per-task
        credential dirs — a much worse outcome than the transcripts they meant
        to stop writing.
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo, transcript_archive={'enabled': False})
        workflow, _cwd = await _make_workflow(config, git_ops, task_assignment)
        assert workflow._config_dir is not None
        config_dir_path = workflow._config_dir.path
        self._plant_transcript(workflow)

        with patch('orchestrator.workflow.archive_before_delete') as mock_helper:
            workflow._cleanup_config_dir()

        mock_helper.assert_not_called()
        assert not config_dir_path.exists()
        assert not (git_repo / 'data' / 'orchestrator' / 'agent-transcripts').exists()

    async def test_recycle_also_archives_before_it_destroys(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """(e) The zero-output-hang recycle is a THIRD transcript-destroying cleanup().

        ``_recycle_config_dir`` throws the config dir away between sub-threshold
        zero-output timeouts. turns==0 means the session did no useful work, but
        its transcript is exactly the forensic record of a wedge — the thing an
        operator most wants and the thing this path has been silently deleting.
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)
        workflow.worktree = cwd
        assert workflow._config_dir is not None
        src, payload = self._plant_transcript(workflow, sid='sess-wedged')

        workflow._recycle_config_dir()

        archived = (
            git_repo / 'data' / 'orchestrator' / 'agent-transcripts'
            / task_assignment.task_id / ENC / 'sess-wedged.jsonl'
        )
        assert archived.read_bytes() == payload
        # The recycle rebuilds at the SAME path (TaskConfigDir is named from
        # the task id), so "the dir is gone" is not the observable here — the
        # transcript having left it is.
        assert not src.exists()
        # ...and the recycle still did its job: a fresh, empty dir is in place.
        assert workflow._config_dir is not None
        assert workflow._config_dir.path.exists()
        assert list(workflow._config_dir.path.glob('projects/**/*.jsonl')) == []

    async def test_an_archiver_error_never_blocks_the_teardown(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """Defence in depth: a helper that regresses must not strand the dir.

        archive_before_delete is total by contract, so this is belt-and-braces
        rather than the contract — but the failure it guards against is leaking
        a credential-bearing directory on every task, so the guard is cheap
        relative to the risk.
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)
        workflow, _cwd = await _make_workflow(config, git_ops, task_assignment)
        assert workflow._config_dir is not None
        config_dir_path = workflow._config_dir.path
        self._plant_transcript(workflow)

        with patch(
            'orchestrator.workflow.archive_before_delete',
            side_effect=RuntimeError('unguarded boom'),
        ):
            workflow._cleanup_config_dir()

        assert not config_dir_path.exists()


@pytest.mark.asyncio
class TestProducerHookIsUncancellable:
    """Task 3619: the producer hook must survive the SIGTERM that triggers it.

    ``_invoke``'s finally archived behind ``await asyncio.to_thread(...)``. The
    cancellation that reaches this finally IS the shutdown — and it lands on
    that await, so the archival is skipped and re-raised. The same shutdown
    then sets ``session_preserved = True`` and writes a resume sidecar naming a
    session whose transcript was never made durable. This site still COPIES
    rather than moves: the session may be resumed and must keep reading its own
    live transcript.
    """

    @staticmethod
    def _writes_then(workflow, payload: bytes, then):
        """Side effect that lays down a transcript, then does *then*()."""
        def _side_effect(**kwargs):
            assert workflow._config_dir is not None
            sid = kwargs['session_id']
            p = workflow._config_dir.path / 'projects' / ENC / f'{sid}.jsonl'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(payload)
            return then()
        return _side_effect

    async def test_a_cancelled_invocation_still_archives_its_transcript(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """The SIGTERM shape: cancellation propagates AND the archive exists.

        MEASURED, so the next reader is not misled: this one passes against the
        pre-fix ``await asyncio.to_thread(...)`` too. A ``CancelledError``
        RAISED by the invoke (rather than delivered by ``task.cancel()``) does
        not cancel the surrounding task, so the offloaded archival still gets
        to complete. It is a re-verification of the propagate-and-archive pair,
        not the RED. The RED for this step is the sibling below: only patching
        ``asyncio.to_thread`` distinguishes "archival happened" from "archival
        happened on a cancellable await".
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)
        payload = b'{"transcript":"in flight at SIGTERM"}\n'

        def _boom():
            raise asyncio.CancelledError

        with patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            side_effect=self._writes_then(workflow, payload, _boom),
        ), pytest.raises(asyncio.CancelledError):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        sid = workflow._last_invoke_session_id
        archived = (
            git_repo / 'data' / 'orchestrator' / 'agent-transcripts'
            / task_assignment.task_id / ENC / f'{sid}.jsonl'
        )
        # Cancellation still propagates — teardown is cooperative. What changed
        # is that it can no longer take the archival with it.
        assert archived.read_bytes() == payload

    async def test_archival_does_not_go_through_to_thread(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """Pin the SYNCHRONOUS call, not merely a passing end-to-end result.

        Without this, someone could re-offload the hook to a worker thread and
        every other test here would still pass — while quietly restoring the
        cancellation point that loses the transcript. Making
        ``asyncio.to_thread`` explode proves archival does not route through it.
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)
        payload = b'{"transcript":"synchronous"}\n'

        async def _explode(*_a, **_kw):
            raise AssertionError('archival must not be offloaded to a thread')

        with patch('orchestrator.workflow.asyncio.to_thread', _explode), patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            side_effect=self._writes_then(workflow, payload, lambda: AgentResult(
                success=True, output='')),
        ):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        sid = workflow._last_invoke_session_id
        archived = (
            git_repo / 'data' / 'orchestrator' / 'agent-transcripts'
            / task_assignment.task_id / ENC / f'{sid}.jsonl'
        )
        assert archived.read_bytes() == payload

    async def test_this_site_copies_so_a_resumed_session_keeps_reading(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """The producer COPIES; only the teardown sites move.

        ``_invoke`` can return to a caller that resumes the very same session,
        which reads and appends to its own live transcript. Moving it out from
        under a live session would turn every resume into a no_transcript
        fallback — the opposite of what this task exists to enable.
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)
        payload = b'{"transcript":"still live"}\n'

        with patch(
            'orchestrator.workflow.invoke_with_cap_retry',
            new_callable=AsyncMock,
            side_effect=self._writes_then(workflow, payload, lambda: AgentResult(
                success=True, output='')),
        ):
            await workflow._invoke(SIMPLE_TASK, 'p', cwd)

        assert workflow._config_dir is not None
        sid = workflow._last_invoke_session_id
        src = workflow._config_dir.path / 'projects' / ENC / f'{sid}.jsonl'
        assert src.read_bytes() == payload
