"""The done_provenance write chokepoint, driven through TaskInterceptor.set_task_status.

The per-kind shape contract lives on ``shared.task_metadata.DoneProvenance``;
these tests pin that the chokepoint hands the model's verdict to the caller
unchanged, refuses on identity before it judges shape, and keeps the
self-correcting kind remedies the model cannot know.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import get_args
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from _fm_helpers import _init_git_repo
from pydantic import ValidationError
from shared.task_metadata import DoneProvenance

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer

DECLARED_KINDS = get_args(DoneProvenance.model_fields['kind'].annotation)
CALLER_REFUSAL = 'DeterministicProvenanceCallerNotPermitted'


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_task = AsyncMock(return_value={'id': '1', 'status': 'pending', 'title': 'Test Task'})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.set_status_and_stamp_audit = AsyncMock(return_value={'success': True})
    return tm


@pytest.fixture
def reconciler():
    r = AsyncMock()
    r.reconcile_task = AsyncMock(return_value={'actions': []})
    return r


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'chokepoint_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def interceptor(taskmaster, reconciler, event_buffer):
    return TaskInterceptor(taskmaster, reconciler, event_buffer, config=FusedMemoryConfig())


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / 'repo'
    sha = _init_git_repo(path)
    return str(path), sha


async def _mark_done(interceptor, project_root, provenance, *, agent_id='orchestrator'):
    return await interceptor.set_task_status(
        '1', 'done', project_root, done_provenance=provenance, agent_id=agent_id,
    )


def _assert_refused_without_write(result, taskmaster):
    assert result.get('error') == 'done_provenance_invalid', result
    taskmaster.set_status_and_stamp_audit.assert_not_called()
    taskmaster.set_task_status.assert_not_called()


def _persisted_provenance(taskmaster) -> dict:
    taskmaster.set_status_and_stamp_audit.assert_called_once()
    return taskmaster.set_status_and_stamp_audit.call_args.kwargs['audit_fields']['done_provenance']


PayloadFor = Callable[[str], dict]

_MODEL_REFUSALS: list[tuple[str, PayloadFor]] = [
    ('merged-no-commit', lambda sha: {'kind': 'merged'}),
    ('merged-blank-commit', lambda sha: {'kind': 'merged', 'commit': '   '}),
    ('found-on-main-no-commit', lambda sha: {'kind': 'found_on_main', 'note': 'n'}),
    ('found-on-main-no-note', lambda sha: {'kind': 'found_on_main', 'commit': sha}),
    ('operational-no-escalation', lambda sha: {'kind': 'operational-verified', 'note': 'n'}),
    ('operational-no-note', lambda sha: {'kind': 'operational-verified', 'escalation_id': 'e'}),
    ('gate-no-escalation', lambda sha: {'kind': 'deterministic-gate', 'note': 'n'}),
]


class TestModelVerdictReachesCaller:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'payload_for', [p for _, p in _MODEL_REFUSALS], ids=[i for i, _ in _MODEL_REFUSALS],
    )
    async def test_reason_is_the_models_message(self, interceptor, taskmaster, repo, payload_for):
        project_root, sha = repo
        payload = payload_for(sha)
        with pytest.raises(ValidationError) as exc_info:
            DoneProvenance.parse_for_write(payload)
        expected = exc_info.value.errors()[0]['msg']

        result = await _mark_done(interceptor, project_root, payload)

        _assert_refused_without_write(result, taskmaster)
        assert result['reason'] == expected
        assert 'error_type' not in result


class TestTypedEvidenceIsRefused:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('payload', 'field'),
        [
            ({'kind': 'deterministic-deploy', 'pid': 'not-a-pid', 'unit': 'u.service'}, 'pid'),
            (
                {
                    'kind': 'deterministic-deploy-scheduled',
                    'unit': 'u.service',
                    'fire_delay_secs': 'soon',
                },
                'fire_delay_secs',
            ),
        ],
    )
    async def test_wrongly_typed_evidence_is_refused(
        self, interceptor, taskmaster, repo, payload, field,
    ):
        project_root, _ = repo
        result = await _mark_done(interceptor, project_root, payload)

        _assert_refused_without_write(result, taskmaster)
        assert field in result['reason']


class TestIdentityBeforeShape:
    @pytest.mark.asyncio
    async def test_recon_stage_gate_with_malformed_note_gets_the_caller_refusal(
        self, interceptor, taskmaster, repo,
    ):
        project_root, _ = repo
        result = await _mark_done(
            interceptor,
            project_root,
            {'kind': 'deterministic-gate', 'note': 123, 'escalation_id': 'e'},
            agent_id='recon-stage-2',
        )

        _assert_refused_without_write(result, taskmaster)
        assert result.get('error_type') == CALLER_REFUSAL

    @pytest.mark.asyncio
    async def test_recon_stage_operational_with_malformed_note_gets_the_recon_refusal(
        self, interceptor, taskmaster, repo,
    ):
        project_root, _ = repo
        result = await _mark_done(
            interceptor,
            project_root,
            {'kind': 'operational-verified', 'note': 5},
            agent_id='recon-stage-2',
        )

        _assert_refused_without_write(result, taskmaster)
        assert 'recon' in result['reason'].lower()

    @pytest.mark.asyncio
    async def test_unlisted_caller_gate_without_escalation_gets_the_caller_refusal(
        self, interceptor, taskmaster, repo,
    ):
        project_root, _ = repo
        result = await _mark_done(
            interceptor, project_root, {'kind': 'deterministic-gate'}, agent_id='dashboard',
        )

        _assert_refused_without_write(result, taskmaster)
        assert result.get('error_type') == CALLER_REFUSAL


class TestKindRemediesStaySelfCorrecting:
    @pytest.mark.asyncio
    @pytest.mark.parametrize('kind_entry', [{}, {'kind': '   '}], ids=['absent', 'blank'])
    async def test_missing_kind_names_every_declared_kind(
        self, interceptor, taskmaster, repo, kind_entry,
    ):
        project_root, sha = repo
        result = await _mark_done(interceptor, project_root, {'commit': sha, **kind_entry})

        _assert_refused_without_write(result, taskmaster)
        for kind in DECLARED_KINDS:
            assert f'"{kind}"' in result['reason']

    @pytest.mark.asyncio
    async def test_unknown_kind_is_echoed(self, interceptor, taskmaster, repo):
        project_root, _ = repo
        result = await _mark_done(interceptor, project_root, {'kind': 'cherry_picked'})

        _assert_refused_without_write(result, taskmaster)
        assert 'cherry_picked' in result['reason']

    @pytest.mark.asyncio
    async def test_non_string_kind_is_refused(self, interceptor, taskmaster, repo):
        project_root, _ = repo
        result = await _mark_done(interceptor, project_root, {'kind': 7})

        _assert_refused_without_write(result, taskmaster)


class TestAcceptedWritesRecordTheProjection:
    @pytest.mark.asyncio
    async def test_whitespace_is_trimmed_and_foreign_fields_dropped(
        self, interceptor, taskmaster, repo,
    ):
        project_root, sha = repo
        result = await _mark_done(
            interceptor,
            project_root,
            {'kind': ' merged ', 'commit': f'  {sha}  ', 'escalation_id': 'e', 'pid': 4, 'x_foo': 1},
        )

        assert 'error' not in result, result
        assert _persisted_provenance(taskmaster) == {'kind': 'merged', 'commit': sha}

    @pytest.mark.asyncio
    async def test_caller_stamped_at_of_any_type_is_replaced(self, interceptor, taskmaster, repo):
        project_root, sha = repo
        supplied = 1754179200
        result = await _mark_done(
            interceptor,
            project_root,
            {'kind': 'found_on_main', 'commit': sha, 'note': 'n', 'stamped_at': supplied},
        )

        assert 'error' not in result, result
        stamped_at = _persisted_provenance(taskmaster)['stamped_at']
        assert isinstance(stamped_at, str)
        assert stamped_at != str(supplied)
