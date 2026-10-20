"""resolve_escalation_shas: a record's cited, since-rebased SHAs resolved to the
commits they landed as on main, read-only (plans/plan-deviation-recording-prd.md D12).

Real-git, like test_warm_worktree_verbs.py: the tool reaches git through
``harness.git_ops``, so the orchestrator import is guarded and the file skips
without it.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from _escalation_seed import seed_escalation

from escalation.queue import EscalationQueue, iter_all_escalation_paths
from escalation.server import create_server

try:
    from orchestrator.config import GitConfig  # type: ignore[reportMissingImports]
    from orchestrator.git_ops import GitOps, _run  # type: ignore[reportMissingImports]
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False
    GitConfig: Any = None
    GitOps: Any = None
    _run: Any = None

pytestmark = pytest.mark.skipif(
    not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed',
)


async def _git(repo: Path, *args: str) -> str:
    rc, out, err = await _run(['git', *args], cwd=repo)
    assert rc == 0, f'git {" ".join(args)} failed: {err}'
    return out.strip()


async def _commit(repo: Path, relpath: str, text: str, subject: str) -> str:
    (repo / relpath).write_text(text)
    await _git(repo, 'add', relpath)
    await _git(repo, 'commit', '-m', subject)
    return await _git(repo, 'rev-parse', 'HEAD')


@dataclass(frozen=True)
class LandedBranch:
    s1: str
    s2: str
    s1_landed: str
    s2_landed: str


async def _land_rebased_branch(repo: Path) -> LandedBranch:
    await _git(repo, 'init', '-b', 'main')
    await _git(repo, 'config', 'user.email', 'test@test.com')
    await _git(repo, 'config', 'user.name', 'Test')
    await _commit(repo, 'README.md', '# Test\n', 'Initial commit')
    await _git(repo, 'checkout', '-b', 'task/7')
    s1 = await _commit(repo, 'feature.py', 'feature = 1\n', 'feat(7): add feature')
    s2 = await _commit(repo, 'second.py', 'second = 2\n', 'feat(7): second')
    await _git(repo, 'checkout', 'main')
    await _commit(repo, 'other.py', 'other = 3\n', 'main moves on')
    await _git(repo, 'checkout', 'task/7')
    await _git(repo, 'rebase', 'main')
    await _git(repo, 'checkout', 'main')
    await _git(repo, 'merge', '--ff-only', 'task/7')
    return LandedBranch(
        s1=s1,
        s2=s2,
        s1_landed=await _git(repo, 'rev-parse', 'HEAD~1'),
        s2_landed=await _git(repo, 'rev-parse', 'HEAD'),
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / 'repo'
    path.mkdir()
    return path


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


@pytest.fixture
def server(queue: EscalationQueue, repo: Path) -> Any:
    harness = types.SimpleNamespace(git_ops=GitOps(GitConfig(), repo))
    return create_server(queue, harness=harness, startup_sweep=False)


async def _resolve(server: Any, escalation_id: str) -> dict[str, Any]:
    tool = await server.get_tool('resolve_escalation_shas')
    return await tool.fn(escalation_id=escalation_id)


def _record_bytes(queue: EscalationQueue, escalation_id: str) -> bytes:
    (path,) = [p for p in iter_all_escalation_paths(queue.queue_dir) if p.stem == escalation_id]
    return path.read_bytes()


@pytest.mark.asyncio
class TestResolveEscalationShas:
    async def test_rebased_citations_resolve_to_their_landed_twins(self, server, queue, repo):
        landed = await _land_rebased_branch(repo)
        esc = seed_escalation(
            queue, level=0, task_id='7',
            detail=f'step-1 commit {landed.s1[:10]} was orphaned by a rebase',
            evidence=[{'observation': f'step-2 at {landed.s2}', 'measured_at': 'x', 'ref': 'y'}],
        )

        result = await _resolve(server, esc.id)

        assert result['escalation_id'] == esc.id
        first, second = result['resolved_shas']
        assert first['cited'] == landed.s1[:10]
        assert first['landed'] == landed.s1_landed
        assert first['method'] == 'patch-id'
        assert first['subject'] == 'feat(7): add feature'
        assert second['cited'] == landed.s2
        assert second['landed'] == landed.s2_landed

    async def test_the_record_is_never_rewritten(self, server, queue, repo):
        landed = await _land_rebased_branch(repo)
        esc = seed_escalation(queue, level=0, task_id='7', detail=f'cites {landed.s1}')
        before = _record_bytes(queue, esc.id)

        await _resolve(server, esc.id)

        assert _record_bytes(queue, esc.id) == before

    async def test_main_shas_and_no_shas_resolve_to_nothing(self, server, queue, repo):
        landed = await _land_rebased_branch(repo)
        on_main = seed_escalation(
            queue, level=0, task_id='7', detail=f'{landed.s1_landed} {landed.s2_landed}',
        )
        no_shas = seed_escalation(queue, level=0, task_id='7', detail='nothing cited here')

        assert (await _resolve(server, on_main.id))['resolved_shas'] == []
        assert (await _resolve(server, no_shas.id))['resolved_shas'] == []

    async def test_unknown_escalation_id_is_an_error_naming_it(self, server):
        result = await _resolve(server, 'esc-7-404')

        assert 'error' in result
        assert 'esc-7-404' in result['error']

    async def test_standalone_server_has_no_git_authority(self, queue):
        standalone = create_server(queue, startup_sweep=False)
        esc = seed_escalation(queue, level=0, task_id='7', detail='cites 1234567')

        assert await _resolve(standalone, esc.id) == {'error': 'no git authority wired'}
