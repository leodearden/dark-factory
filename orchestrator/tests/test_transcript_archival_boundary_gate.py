"""ε B+H boundary gate — orchestrator-consumer half (task 2732).

The end-to-end transcript-archival boundary gate over the ALREADY-INTEGRATED
code paths landed by α (producer hook + archiver primitive), β (teardown
backstop), γ (legibility archive mining) and δ (retention GC). See
``plans/agent-transcript-archival-prd.md`` Appendix B for the matrix::

    E1  a completed session's transcript is archived at completion
    E2  the archive survives worktree teardown
    E3  the teardown backstop is idempotent w.r.t. the producer
    E4  the archive is credential-safe (only projects/**.jsonl is ever copied)
    E5  legibility mining enumerates the archived transcript
    E6  a resumed session re-archives its grown transcript (last-write-wins)
    E7  an archive failure is SOFT (task still succeeds) and LOUD (counted+logged)
    E8  the retention GC prunes by cap, loudly; default caps are a no-op

**This file owns E1, E6, E2, E3 and E7** — every row that runs through the REAL
``TaskWorkflow._invoke`` producer hook and/or the REAL
``GitOps.cleanup_worktree`` teardown backstop. Its two siblings own the rest:

* ``shared/tests/test_transcript_archival_boundary_gate.py`` — E4 (the
  credential-safety row, kept orchestrator-free so ``shared`` stays a leaf).
* ``scripts/tests/test_transcript_archival_boundary_gate.py`` — E5, E8 (the
  legibility-mining and retention-GC rows).

The gate is three files rather than one because ``verify`` is directory-scoped:
each package's ``orchestrator.yaml`` declares its own ``test_command``, so a
single cross-package module would run in exactly one lane and a shared-only or
scripts-only diff would never exercise its rows. Convention copied from the
two-file ``test_liveness_boundary_gate.py`` pair (orchestrator/tests +
shared/tests).

ARCHIVE FORMAT: plain ``.jsonl``, byte-verbatim, NO added suffix. Task 3618
(leaf α of ``plans/transcript-preservation-seam-prd.md``) dropped gzip from the
archive AFTER the PRD was written, so Appendix B's "gz round-trips" wording for
E1/E5 is stale — do not read it as a gap. The residual-``.jsonl.gz`` contract
that survived 3618 (not enumerated, but counted + warned) is pinned by E5 in
the scripts file.

No row mocks the component under test. ``archive_task_transcripts``,
``cleanup_worktree`` and the real ``git worktree remove`` all run for real;
ONLY ``orchestrator.workflow.invoke_with_cap_retry`` is patched (to avoid
spawning a real Claude agent), and its side effect writes a real transcript
from the ``session_id=`` kwarg ``_invoke`` forwards. The per-leaf suites
(``test_transcript_archive_producer_hook.py`` / ``_backstop.py``) patch the
archiver itself to assert argument wiring; this gate deliberately does not, so
it covers the integrated path those cannot.

Fixtures are kept module-local (no conftest.py additions), matching
``test_transcript_archive_backstop.py``'s documented choice.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from _workflow_helpers import FakeBriefing, FakeMcp, FakeScheduler
from shared.config_dir import TaskConfigDir

from orchestrator.config import GitConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.scheduler import TaskAssignment
from orchestrator.workflow import TaskWorkflow

# The encoded-project dir the fake transcript is laid down under.
ENC = '-home-leo-projX'

# The task id every row runs under (matches the branch name `git_ops`
# creates, which is what cleanup_worktree's backstop keys the config-dir path
# and the archive layout on).
TASK_ID = '42'


# ---------------------------------------------------------------------------
# Real-git fixture harness (ported from test_transcript_archive_backstop.py)
# ---------------------------------------------------------------------------

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


def _make_git_ops(git_repo: Path, **kwargs) -> GitOps:
    """Build a GitOps rooted at *git_repo*; ``**kwargs`` pass through to
    ``__init__`` (notably ``transcript_archive=...``, which arms the β
    teardown backstop — omitting it leaves the backstop inert)."""
    return GitOps(
        GitConfig(
            main_branch='main',
            branch_prefix='task/',
            remote='origin',
            worktree_dir='.worktrees',
        ),
        git_repo,
        **kwargs,
    )


@pytest.fixture
def git_ops(git_repo: Path) -> GitOps:
    """A backstop-INERT GitOps (no transcript_archive), the default for rows
    that isolate the producer. E3 builds its own armed instance."""
    return _make_git_ops(git_repo)


def _config_dir(worktree: Path, task_id: str = TASK_ID) -> Path:
    """The on-disk per-task Claude config dir the β backstop reconstructs
    (``<worktree>/.task/claude-config-<branch>``, git_ops.py's derivation)."""
    return worktree / '.task' / f'claude-config-{task_id}'


def _write_transcript(worktree: Path, sid: str, data: bytes, task_id: str = TASK_ID) -> Path:
    """Lay down an un-archived transcript at
    ``<config_dir>/projects/<ENC>/<sid>.jsonl`` and return its path."""
    p = _config_dir(worktree, task_id) / 'projects' / ENC / f'{sid}.jsonl'
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _archive_root(git_repo: Path) -> Path:
    """The durable archive root the producer composes
    (``config.project_root / transcript_archive.root``) — OUTSIDE the worktree."""
    return git_repo / 'data' / 'orchestrator' / 'agent-transcripts'


def _archived(git_repo: Path, sid: str, task_id: str = TASK_ID) -> Path:
    """The durable plain-.jsonl mirror the archiver should produce for *sid*."""
    return _archive_root(git_repo) / task_id / ENC / f'{sid}.jsonl'


# ---------------------------------------------------------------------------
# Probe-TaskWorkflow harness (ported from test_transcript_archive_producer_hook.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def task_assignment() -> TaskAssignment:
    return TaskAssignment(
        task_id=TASK_ID,
        task={
            'id': TASK_ID,
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
    """Build a probe TaskWorkflow over a REAL worktree, with ``_config_dir``
    set manually (driving ``_invoke`` directly skips ``run()``'s setup)."""
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
    workflow._config_dir = TaskConfigDir(task_assignment.task_id, base_dir=cwd / '.task')
    return workflow, cwd
