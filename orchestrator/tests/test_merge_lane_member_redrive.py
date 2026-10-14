"""The per-member re-drive of parked train members (task 4737).

``orchestrator.merge_lane.member_redrive.MemberRedriver`` decides, for each
member still parked merge-deferred, whether its branch already landed, and
hands it back through the injected ``redrive_member`` callback: done with
found_on_main evidence, or pending for a fresh solo merge.  Real git over a
temp repository; the status and re-drive callbacks are the seam.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, call

import pytest
import pytest_asyncio
from escalation.queue import EscalationQueue
from test_git_ops_unstack import (
    _add_member,
    _commit_files,
    _git,
    _init_repo,
    _make_git_ops,
    _rev_parse,
)

from orchestrator.git_ops import GitOps
from orchestrator.merge_lane.member_redrive import MemberRedriveOutcome, MemberRedriver
from orchestrator.scheduler import StaleEvidenceRejection

_ROLE = 'test-member-redrive'


@dataclass
class _RecordingSink:
    """Stands in for orchestrator/src/orchestrator/provenance_conflict.py::ProvenanceConflictSink."""

    memoized: set[str] = field(default_factory=set)
    rejections: list[tuple[StaleEvidenceRejection, str]] = field(default_factory=list)

    def should_skip(self, task_id: str) -> bool:
        return task_id in self.memoized

    def record_from_rejection(self, exc: StaleEvidenceRejection, *, gate_source: str) -> None:
        self.rejections.append((exc, gate_source))


async def land_attributed(git_ops: GitOps, mid: str) -> str:
    """Land *mid* as the merge lane does; returns the merge commit, its citation."""
    wt = await _add_member(git_ops, mid, 'main', {f'{mid}.py': f'{mid} = 1\n'})
    await _commit_files(wt, {f'{mid}.py': f'{mid} = 2\n'}, f'impl({mid}): add {mid}')
    await _git(
        git_ops.project_root, 'merge', '--no-ff', f'task/{mid}', '-m', f'Merge task/{mid} into main',
    )
    return await _rev_parse(git_ops.project_root, 'main')


async def add_unlanded(git_ops: GitOps, mid: str) -> None:
    await _add_member(git_ops, mid, 'main', {f'{mid}.py': f'{mid} = 1\n'})


@dataclass(frozen=True)
class _Repo:
    root: Path
    git_ops: GitOps
    queue: EscalationQueue
    sink: _RecordingSink

    def redriver(self) -> MemberRedriver:
        return MemberRedriver(
            git_ops=self.git_ops,
            provenance_conflict_sink=self.sink,
            escalation_queue=self.queue,
            agent_role=_ROLE,
        )

    async def land_attributed(self, mid: str) -> str:
        return await land_attributed(self.git_ops, mid)

    async def land_unattributed(self, mid: str) -> None:
        await _add_member(self.git_ops, mid, 'main', {f'{mid}.py': f'{mid} = 1\n'})
        await _git(self.root, 'merge', '--ff-only', f'task/{mid}')

    async def add_unlanded(self, mid: str) -> None:
        await add_unlanded(self.git_ops, mid)

    async def is_ancestor(self, ancestor: str, ref: str) -> bool:
        return await self.git_ops.is_ancestor(ancestor, ref)


@pytest_asyncio.fixture
async def repo(tmp_path: Path) -> _Repo:
    root = tmp_path / 'repo'
    root.mkdir()
    await _init_repo(root)
    await _commit_files(root, {'shared.txt': 'v0\n'}, 'initial')
    return _Repo(
        root=root,
        git_ops=_make_git_ops(root),
        queue=EscalationQueue(tmp_path / 'esc'),
        sink=_RecordingSink(),
    )


def _parked(*mids: str, **others: str) -> AsyncMock:
    statuses = dict.fromkeys(mids, 'merge-deferred') | others
    return AsyncMock(return_value=statuses)


@pytest.mark.asyncio
class TestOutcomes:
    async def test_member_not_parked_is_left_alone(self, repo: _Repo) -> None:
        await repo.add_unlanded('busy')
        redrive_member = AsyncMock()

        outcomes = await repo.redriver().redrive(
            'train-x', ['busy'],
            status_check=_parked(busy='in-progress'),
            redrive_member=redrive_member,
            strip_unlanded_base=True,
        )

        assert outcomes == {'busy': MemberRedriveOutcome.NOT_PARKED}
        redrive_member.assert_not_awaited()

    async def test_unlanded_member_goes_back_to_pending(self, repo: _Repo) -> None:
        await repo.add_unlanded('off')
        redrive_member = AsyncMock()

        outcomes = await repo.redriver().redrive(
            'train-x', ['off'],
            status_check=_parked('off'),
            redrive_member=redrive_member,
            strip_unlanded_base=True,
        )

        assert outcomes == {'off': MemberRedriveOutcome.NOT_ON_MAIN}
        assert redrive_member.await_args_list == [call('off', False, None)]

    async def test_attributed_landing_goes_done_with_its_citation(self, repo: _Repo) -> None:
        citation = await repo.land_attributed('on')
        redrive_member = AsyncMock()

        outcomes = await repo.redriver().redrive(
            'train-x', ['on'],
            status_check=_parked('on'),
            redrive_member=redrive_member,
            strip_unlanded_base=True,
        )

        assert outcomes == {'on': MemberRedriveOutcome.ON_MAIN_ATTRIBUTED}
        assert redrive_member.await_args_list == [call('on', True, citation)]
        assert repo.queue.get_by_task('on', status='pending') == []

    async def test_unattributed_landing_escalates_and_goes_pending(self, repo: _Repo) -> None:
        await repo.land_unattributed('ptr')
        redrive_member = AsyncMock()

        outcomes = await repo.redriver().redrive(
            'train-x', ['ptr'],
            status_check=_parked('ptr'),
            redrive_member=redrive_member,
            strip_unlanded_base=True,
        )

        assert outcomes == {'ptr': MemberRedriveOutcome.ON_MAIN_UNATTRIBUTED}
        assert redrive_member.await_args_list == [call('ptr', False, None)]
        pending = repo.queue.get_by_task('ptr', status='pending')
        assert [(esc.category, esc.agent_role) for esc in pending] == [
            ('provenance_unattributed', _ROLE),
        ]

    async def test_memoized_provenance_conflict_skips_the_write(self, repo: _Repo) -> None:
        await repo.land_attributed('memo')
        repo.sink.memoized.add('memo')
        redrive_member = AsyncMock()

        outcomes = await repo.redriver().redrive(
            'train-x', ['memo'],
            status_check=_parked('memo'),
            redrive_member=redrive_member,
            strip_unlanded_base=True,
        )

        assert outcomes == {'memo': MemberRedriveOutcome.PROVENANCE_CONFLICT_MEMOIZED}
        redrive_member.assert_not_awaited()

    async def test_stale_evidence_goes_to_the_sink(self, repo: _Repo) -> None:
        await repo.land_attributed('stale')
        rejection = StaleEvidenceRejection(
            task_id='stale', evidence_commit='abc', evidence_committed_at='t0',
            reopen_at='t1', agent_id='orchestrator', raw='done_evidence_stale',
        )
        redrive_member = AsyncMock(side_effect=rejection)

        outcomes = await repo.redriver().redrive(
            'train-x', ['stale'],
            status_check=_parked('stale'),
            redrive_member=redrive_member,
            strip_unlanded_base=True,
        )

        assert outcomes == {'stale': MemberRedriveOutcome.STALE_EVIDENCE}
        assert [exc for exc, _ in repo.sink.rejections] == [rejection]

    async def test_one_member_failing_does_not_stop_the_others(self, repo: _Repo) -> None:
        await repo.add_unlanded('boom')
        await repo.add_unlanded('fine')

        async def redrive_member(mid: str, found_on_main: bool, sha: str | None) -> None:
            if mid == 'boom':
                raise RuntimeError('redrive exploded')

        recorder = AsyncMock(side_effect=redrive_member)

        outcomes = await repo.redriver().redrive(
            'train-x', ['boom', 'fine'],
            status_check=_parked('boom', 'fine'),
            redrive_member=recorder,
            strip_unlanded_base=True,
        )

        assert outcomes == {
            'boom': MemberRedriveOutcome.ERRORED,
            'fine': MemberRedriveOutcome.NOT_ON_MAIN,
        }
        assert recorder.await_args_list == [
            call('boom', False, None), call('fine', False, None),
        ]


@pytest.mark.asyncio
class TestUnlandedBase:
    async def _stack_pair(self, repo: _Repo) -> str:
        await _add_member(repo.git_ops, 'A', 'main', {'a.txt': 'a\n'})
        await _add_member(repo.git_ops, 'B', 'main', {'b.txt': 'b\n'})
        result = await repo.git_ops.stack_train_branches(['A', 'B'])
        assert result.survivors == ['A', 'B']
        return await _rev_parse(repo.root, 'task/A')

    async def test_strip_removes_the_unlanded_base(self, repo: _Repo) -> None:
        a_commit = await self._stack_pair(repo)
        assert await repo.is_ancestor(a_commit, 'task/B')

        outcomes = await repo.redriver().redrive(
            'train-x', ['B'],
            status_check=_parked('B'),
            redrive_member=AsyncMock(),
            strip_unlanded_base=True,
        )

        assert outcomes == {'B': MemberRedriveOutcome.NOT_ON_MAIN}
        assert not await repo.is_ancestor(a_commit, 'task/B')
        assert (await _git(repo.root, 'show', 'task/B:b.txt')).strip() == 'b'

    async def test_no_strip_leaves_the_branch_byte_identical(self, repo: _Repo) -> None:
        await self._stack_pair(repo)
        tip_before = await _rev_parse(repo.root, 'task/B')

        outcomes = await repo.redriver().redrive(
            'train-x', ['B'],
            status_check=_parked('B'),
            redrive_member=AsyncMock(),
            strip_unlanded_base=False,
        )

        assert outcomes == {'B': MemberRedriveOutcome.NOT_ON_MAIN}
        assert await _rev_parse(repo.root, 'task/B') == tip_before
