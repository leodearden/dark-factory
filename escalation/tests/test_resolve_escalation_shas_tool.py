"""resolve_escalation_shas: a record's cited SHAs resolved through the harness's
git authority, read-only (plans/plan-deviation-recording-prd.md D12).

The tool's own contract runs against a stub ``git_ops``; one real-git smoke test
drives orchestrator ``GitOps`` end to end, skipped without the orchestrator.
"""

from __future__ import annotations

import subprocess
import types
from pathlib import Path
from typing import Any

import pytest
from _citation_resolver_stub import StubCitationResolver
from _escalation_seed import seed_escalation
from _sha_noise import NOISE
from _tool_calls import call_tool

from escalation.queue import EscalationQueue, iter_all_escalation_paths
from escalation.server import create_server
from escalation.sha_citations import MAX_CITED_SHAS_RESOLVED

try:
    from orchestrator.config import GitConfig
    from orchestrator.git_ops import GitOps
    _ORCHESTRATOR_AVAILABLE = True
except ImportError:
    _ORCHESTRATOR_AVAILABLE = False
    GitConfig: Any = None
    GitOps: Any = None

SHA_A = 'a7ceed6597'
SHA_B = '570dc88ad6'
TWIN_A: dict[str, object] = {'landed': 'a' * 40, 'subject': 'feat: a', 'method': 'patch-id'}


@pytest.fixture
def queue(tmp_path: Path) -> EscalationQueue:
    return EscalationQueue(tmp_path / 'esc')


def _server(queue: EscalationQueue, git_ops: object) -> Any:
    harness = types.SimpleNamespace(git_ops=git_ops)
    return create_server(queue, harness=harness, startup_sweep=False)


async def _resolve(server: Any, escalation_id: str) -> dict[str, Any]:
    return await call_tool(server, 'resolve_escalation_shas', escalation_id=escalation_id)


def _record_bytes(queue: EscalationQueue, escalation_id: str) -> bytes:
    (path,) = [p for p in iter_all_escalation_paths(queue.queue_dir) if p.stem == escalation_id]
    return path.read_bytes()


@pytest.mark.asyncio
class TestResolveEscalationShas:
    async def test_cited_shas_resolve_through_the_harness_git_authority(self, queue):
        git_ops = StubCitationResolver({SHA_A: TWIN_A})
        esc = seed_escalation(
            queue, level=0, task_id='7', detail=f'step-1 commit {SHA_A} was orphaned',
            evidence=[{'observation': f'step-2 at {SHA_B}', 'measured_at': 'x', 'ref': 'y'}],
        )

        result = await _resolve(_server(queue, git_ops), esc.id)

        assert result == {
            'escalation_id': esc.id,
            'resolved_shas': [{'cited': SHA_A, **TWIN_A}],
            'failed_shas': [],
            'skipped_shas': [],
        }
        assert git_ops.lookups == [SHA_A, SHA_B]

    async def test_a_lookup_git_could_not_answer_is_reported_in_failed_shas(self, queue):
        git_ops = StubCitationResolver(faults={SHA_A: RuntimeError('fatal: bad object a7ceed6597')})
        esc = seed_escalation(queue, level=0, task_id='7', detail=f'cites {SHA_A}')

        result = await _resolve(_server(queue, git_ops), esc.id)

        assert result['resolved_shas'] == []
        assert result['failed_shas'] == [
            {'cited': SHA_A, 'error': 'RuntimeError: fatal: bad object a7ceed6597'},
        ]

    async def test_tokens_past_the_cap_are_reported_skipped_and_never_looked_up(self, queue):
        tokens = [f'{n:07x}' for n in range(MAX_CITED_SHAS_RESOLVED + 3)]
        git_ops = StubCitationResolver()
        esc = seed_escalation(queue, level=0, task_id='7', detail=' '.join(tokens))

        result = await _resolve(_server(queue, git_ops), esc.id)

        assert result['skipped_shas'] == tokens[MAX_CITED_SHAS_RESOLVED:]
        assert git_ops.lookups == tokens[:MAX_CITED_SHAS_RESOLVED]

    async def test_the_record_is_never_rewritten(self, queue):
        esc = seed_escalation(queue, level=0, task_id='7', detail=f'cites {SHA_A}')
        before = _record_bytes(queue, esc.id)

        await _resolve(_server(queue, StubCitationResolver({SHA_A: TWIN_A})), esc.id)

        assert _record_bytes(queue, esc.id) == before

    async def test_unknown_escalation_id_is_an_error_naming_it(self, queue):
        git_ops = StubCitationResolver()

        result = await _resolve(_server(queue, git_ops), 'esc-7-404')

        assert 'esc-7-404' in result['error']
        assert git_ops.lookups == []

    async def test_standalone_server_has_no_git_authority(self, queue):
        standalone = create_server(queue, startup_sweep=False)
        esc = seed_escalation(queue, level=0, task_id='7', detail='cites 1234567')

        assert await _resolve(standalone, esc.id) == {'error': 'no git authority wired'}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ['git', *args], cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()


def _commit(repo: Path, relpath: str, text: str, subject: str) -> str:
    (repo / relpath).write_text(text)
    _git(repo, 'add', relpath)
    _git(repo, 'commit', '-m', subject)
    return _git(repo, 'rev-parse', 'HEAD')


def _landed_rebase(repo: Path) -> tuple[str, str]:
    """A new repo at *repo* whose task/7 commit was rebased and landed on main:
    (the cited pre-rebase sha, the sha it landed as)."""
    repo.mkdir()
    _git(repo, 'init', '-b', 'main')
    _git(repo, 'config', 'user.email', 'test@test.com')
    _git(repo, 'config', 'user.name', 'Test')
    _commit(repo, 'README.md', '# Test\n', 'Initial commit')
    _git(repo, 'checkout', '-b', 'task/7')
    cited = _commit(repo, 'feature.py', 'feature = 1\n', 'feat(7): add feature')
    _git(repo, 'checkout', 'main')
    _commit(repo, 'other.py', 'other = 3\n', 'main moves on')
    _git(repo, 'rebase', 'main', 'task/7')
    _git(repo, 'checkout', 'main')
    _git(repo, 'merge', '--ff-only', 'task/7')
    return cited, _git(repo, 'rev-parse', 'HEAD')


@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
@pytest.mark.asyncio
async def test_smoke_a_rebased_citation_resolves_to_its_landed_twin(queue, tmp_path):
    repo = tmp_path / 'repo'
    cited, landed = _landed_rebase(repo)
    esc = seed_escalation(queue, level=0, task_id='7', detail=f'orphaned {cited[:10]}')

    result = await _resolve(_server(queue, GitOps(GitConfig(), repo)), esc.id)

    assert result['resolved_shas'] == [{
        'cited': cited[:10], 'landed': landed,
        'subject': 'feat(7): add feature', 'method': 'patch-id',
    }]


@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
@pytest.mark.asyncio
async def test_smoke_a_git_outage_is_failed_never_a_clean_no_twin(queue, tmp_path):
    not_a_repo = tmp_path / 'not-a-repo'
    not_a_repo.mkdir()
    esc = seed_escalation(queue, level=0, task_id='7', detail=f'cites {SHA_A} and {SHA_B}')

    result = await _resolve(_server(queue, GitOps(GitConfig(), not_a_repo)), esc.id)

    assert result['resolved_shas'] == []
    assert result['skipped_shas'] == []
    assert [f['cited'] for f in result['failed_shas']] == [SHA_A, SHA_B]


@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
@pytest.mark.asyncio
async def test_smoke_an_unknown_main_branch_fails_the_lookup(queue, tmp_path):
    repo = tmp_path / 'repo'
    cited, _ = _landed_rebase(repo)
    esc = seed_escalation(queue, level=0, task_id='7', detail=f'orphaned {cited[:10]}')
    git_ops = GitOps(GitConfig(main_branch='no-such-branch'), repo)

    result = await _resolve(_server(queue, git_ops), esc.id)

    assert result['resolved_shas'] == []
    assert [f['cited'] for f in result['failed_shas']] == [cited[:10]]


@pytest.mark.skipif(not _ORCHESTRATOR_AVAILABLE, reason='orchestrator package not installed')
@pytest.mark.asyncio
async def test_smoke_uuids_and_digit_runs_do_not_crowd_a_real_sha_past_the_cap(queue, tmp_path):
    repo = tmp_path / 'repo'
    cited, landed = _landed_rebase(repo)
    detail = f'{NOISE} orphaned {cited[:10]}'
    esc = seed_escalation(queue, level=0, task_id='7', detail=detail)

    result = await _resolve(_server(queue, GitOps(GitConfig(), repo)), esc.id)

    assert result['resolved_shas'] == [{
        'cited': cited[:10], 'landed': landed,
        'subject': 'feat(7): add feature', 'method': 'patch-id',
    }]
    assert result['skipped_shas'] == []
    assert result['failed_shas'] == []
