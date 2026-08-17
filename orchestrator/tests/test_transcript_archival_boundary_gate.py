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


# The per-task OAuth credential every live config dir holds beside projects/.
# _producer_invoke lays it down on EVERY row, so credential-exclusion is probed
# continuously by the whole orchestrator half of the gate rather than only by E1.
_CREDENTIAL_DECOY = b'{"claudeAiOauth":{"accessToken":"sk-ant-oat01-GATE-CANARY"}}'


async def _producer_invoke(
    workflow,
    cwd: Path,
    *,
    payload: bytes | None = None,
    session_id: str | None = None,
) -> tuple[AgentResult, str, Path]:
    """Run ONE real ``_invoke``, patching ONLY ``invoke_with_cap_retry``.

    The stand-in agent writes its transcript at the path implied by the session
    id ``_invoke`` FORWARDS to it — never at a path the test picked — so every
    row proves the producer located the file rather than that the fixture put
    one where the assertion looked.

    *payload* is the bytes the stand-in agent writes. ``None`` means "the
    transcript already on disk IS this session's output": the side effect
    leaves the file untouched, which is the shape a RESUMED session needs (E6
    grows the file itself between invocations, and a rewrite here would undo
    that). *session_id*, when given, asserts ``_invoke`` forwarded exactly that
    id — the check that makes E6's real resume path observable rather than
    assumed.

    Returns ``(result, session_id, source_path)``.
    """
    config_dir = workflow._config_dir
    assert config_dir is not None
    captured: dict[str, Any] = {}

    def _side_effect(**kwargs):
        sid = kwargs['session_id']
        if session_id is not None:
            assert sid == session_id, (
                f'_invoke forwarded a fresh session id {sid!r}, expected the '
                f'resumed {session_id!r}'
            )
        src = config_dir.path / 'projects' / ENC / f'{sid}.jsonl'
        src.parent.mkdir(parents=True, exist_ok=True)
        if payload is not None:
            src.write_bytes(payload)
        (config_dir.path / '.credentials.json').write_bytes(_CREDENTIAL_DECOY)
        captured['sid'] = sid
        captured['src'] = src
        return AgentResult(success=True, output='')

    with patch(
        'orchestrator.workflow.invoke_with_cap_retry',
        new_callable=AsyncMock,
        side_effect=_side_effect,
    ):
        result = await workflow._invoke(SIMPLE_TASK, 'p', cwd)

    return result, captured['sid'], captured['src']


def _archive_files(git_repo: Path) -> set[str]:
    """Every FILE under the archive root, as archive-root-relative posix paths."""
    root = _archive_root(git_repo)
    if not root.is_dir():
        return set()
    return {
        p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
    }


# ---------------------------------------------------------------------------
# E1 — a completed session's transcript is archived at completion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestE1ArchivedAtCompletion:
    """E1 — running a role to completion archives its transcript to the durable
    root OUTSIDE the worktree, byte-verbatim, with no credential alongside.

    Drives the REAL producer hook in ``TaskWorkflow._invoke``'s ``finally`` and
    the REAL ``archive_task_transcripts``. ONLY ``invoke_with_cap_retry`` is
    patched, and its side effect writes the transcript at the path the session
    id ``_invoke`` forwards implies — so the row proves the producer located and
    archived the file, rather than that a fixture put one where it was expected.
    """

    async def test_completed_session_lands_in_the_durable_archive(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo)  # transcript_archive enabled by default
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)
        transcript_bytes = b'{"type":"user","cwd":"/tmp/x"}\n{"type":"assistant"}\n'

        result, sid, src = await _producer_invoke(
            workflow, cwd, payload=transcript_bytes
        )

        assert result.success is True
        assert workflow._last_invoke_session_id == sid

        archive_root = _archive_root(git_repo)
        archived = _archived(git_repo, sid)

        # Archived, at the layout the durable root defines, plain .jsonl.
        assert archived.exists()
        assert archived.read_bytes() == src.read_bytes()
        assert archived.read_bytes() == transcript_bytes

        # Plain .jsonl everywhere: task 3618 dropped gzip, so no `.gz` artefact
        # may appear anywhere under the archive root.
        assert not list(archive_root.rglob('*.gz'))

        # DURABLE means outside the worktree — the whole point of the producer
        # hook. A project_root that WAS the worktree would make this vacuous,
        # so assert the containment directly rather than trusting the fixture.
        assert not archived.is_relative_to(cwd)
        assert not archive_root.is_relative_to(cwd)

        # Integrated half of E4: the credential written beside projects/ did
        # not follow the transcript into the archive.
        assert _archive_files(git_repo) == {f'{TASK_ID}/{ENC}/{sid}.jsonl'}
        assert (workflow._config_dir.path / '.credentials.json').exists()

    async def test_kill_switch_leaves_the_archive_root_uncreated(
        self, monkeypatch, git_repo, git_ops, task_assignment
    ):
        """KILL-SWITCH CONTROL for the row above — the PRODUCER caused that
        archive, not the tmp layout.

        The identical scenario with ``transcript_archive.enabled = False`` must
        leave the archive root NON-EXISTENT. Without this, the row above would
        still pass if the archive were written by anything other than the hook
        under test (a fixture, a stray helper, a leftover tree), and the
        documented green-tier kill switch would be untested at the integrated
        level.
        """
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = _config(git_repo, transcript_archive={'enabled': False})
        assert config.transcript_archive.enabled is False
        workflow, cwd = await _make_workflow(config, git_ops, task_assignment)

        result, sid, src = await _producer_invoke(
            workflow, cwd, payload=b'{"type":"user"}\n'
        )

        assert result.success is True
        # The agent really did produce a transcript this time too...
        assert src.exists()
        # ...and the hook really was the only thing that would have archived it.
        assert not _archive_root(git_repo).exists()
        assert not _archived(git_repo, sid).exists()
