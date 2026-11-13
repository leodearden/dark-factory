"""Tests for the reconciliation task-text write guard (task 4650).

Two symptom instances motivate it. Task 3708: a Stage-2 cycle counted a
``details`` amendment as applied while the stored text still held the old
paragraph, because nothing read the field back. Task 4644: two writers
composed full-field ``details`` rewrites against the same read, and the later
one silently clobbered the earlier one, because nothing compared the base the
writer composed against with the text it replaced.
"""

from __future__ import annotations

import copy
import logging
from typing import Any
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from shared.mcp_envelope import write_failure_reason
from shared.task_text_fingerprint import task_text_fingerprint

from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.middleware.task_text_write_guard import (
    TASK_TEXT_FP_KEY,
    TEXT_BASE_REQUIRED,
    TEXT_BASE_STALE,
    TEXT_VERIFIED_KEY,
    TextWriteUnconfirmedFinding,
    check_text_base,
    plan_text_write,
    text_fp_of,
    verify_text_write,
    verify_text_write_landed,
)
from fused_memory.reconciliation.event_buffer import EventBuffer
from fused_memory.reconciliation.prompts.stage2 import (
    STAGE2_SYSTEM_PROMPT,
    build_stage2_system_prompt,
)

EMPTY_TEXT_FP = task_text_fingerprint(None, None)


def _forwarded_kwargs(**overrides: Any) -> dict[str, Any]:
    """The kwargs shape ``server/tools.py::update_task`` forwards to the
    interceptor: every text key present, None when unset."""
    kwargs: dict[str, Any] = {
        'prompt': None,
        'metadata': None,
        'append': None,
        'metadata_mode': None,
        'tag': None,
        'title': None,
        'description': None,
        'details': None,
        'priority': None,
        'status': None,
        'dependencies': None,
    }
    kwargs.update(overrides)
    return kwargs


class TestTextFpOf:
    def test_delegates_to_the_shared_fingerprint(self):
        task = {'description': 'A short summary.', 'details': 'Long body\n\nwith paragraphs.'}

        assert text_fp_of(task) == task_text_fingerprint(task['description'], task['details'])

    def test_absent_or_none_text_is_the_empty_text_fingerprint(self):
        assert text_fp_of({'id': '1'}) == EMPTY_TEXT_FP
        assert text_fp_of({'description': None, 'details': None}) == EMPTY_TEXT_FP

    def test_a_non_mapping_is_the_empty_text_fingerprint(self):
        assert text_fp_of(None) == EMPTY_TEXT_FP
        assert text_fp_of('not a task') == EMPTY_TEXT_FP


class TestPlanTextWrite:
    def test_no_text_kwargs_writes_no_text(self):
        write = plan_text_write(_forwarded_kwargs(metadata='{"a": 1}'))

        assert write.writes_text is False
        assert write.replaces_text is False
        assert write.fields == ()

    def test_details_replace(self):
        write = plan_text_write(_forwarded_kwargs(details='x'))

        assert write.details == 'x'
        assert write.writes_text is True
        assert write.replaces_text is True
        assert write.fields == ('details',)

    def test_details_append_writes_but_does_not_replace(self):
        write = plan_text_write(_forwarded_kwargs(details='x', append=True))

        assert write.writes_text is True
        assert write.replaces_text is False
        assert write.details_appends is True

    def test_prompt_is_the_details_payload_when_details_is_unset(self):
        write = plan_text_write(_forwarded_kwargs(prompt='x'))

        assert write.details == 'x'
        assert write.replaces_text is True
        assert write.fields == ('details',)

    def test_details_wins_over_prompt(self):
        write = plan_text_write(_forwarded_kwargs(details='from-details', prompt='from-prompt'))

        assert write.details == 'from-details'

    def test_description_always_replaces(self):
        assert plan_text_write(_forwarded_kwargs(description='x')).replaces_text is True
        assert plan_text_write(
            _forwarded_kwargs(description='x', append=True),
        ).replaces_text is True

    def test_fields_are_listed_description_first(self):
        write = plan_text_write(_forwarded_kwargs(details='d', description='s'))

        assert write.fields == ('description', 'details')

    def test_absent_keys_read_as_unset(self):
        assert plan_text_write({}).writes_text is False


BEFORE = {'id': '4644', 'description': 'Current summary.', 'details': 'Current body.'}
RECON_AGENT = 'recon-stage-task_knowledge_sync'


def _check(*, expected_text_fp: str | None, fp_required: bool = True, before=BEFORE, **kwargs):
    return check_text_base(
        before=before,
        write=plan_text_write(_forwarded_kwargs(**kwargs)),
        expected_text_fp=expected_text_fp,
        fp_required=fp_required,
        task_id='4644',
        agent_id=RECON_AGENT,
    )


class TestCheckTextBase:
    def test_a_metadata_only_write_needs_no_token(self):
        assert _check(expected_text_fp=None, metadata='{"a": 1}') is None

    def test_a_matching_token_admits_a_replace(self):
        assert _check(expected_text_fp=text_fp_of(BEFORE), details='New body.') is None

    def test_a_stale_token_is_rejected_without_echoing_the_current_fingerprint(self):
        rejection = _check(expected_text_fp='tfp:stale', details='New body.')

        assert rejection is not None
        error = rejection.to_error_dict()
        assert error['error_type'] == 'TaskTextBaseStale'
        assert error['fields'] == ['details']
        assert error['expected_text_fp'] == 'tfp:stale'
        assert 'get_task' in error['hint']
        assert 'task_text_fp' in error['hint']
        assert text_fp_of(BEFORE) not in repr(error)

    def test_a_recon_replace_without_a_token_is_rejected(self):
        rejection = _check(expected_text_fp=None, details='New body.')

        assert rejection is not None
        assert rejection.to_error_dict()['error_type'] == 'ReconTaskTextBaseRequired'

    def test_a_token_is_optional_when_not_required(self):
        assert _check(expected_text_fp=None, fp_required=False, details='New body.') is None

    def test_a_details_append_needs_no_token(self):
        assert _check(expected_text_fp=None, details='More.', append=True) is None

    def test_a_description_write_needs_a_token_even_under_append(self):
        assert _check(expected_text_fp=None, description='New summary.') is not None
        assert _check(expected_text_fp=None, description='New summary.', append=True) is not None

    def test_a_prompt_replace_needs_a_token(self):
        assert _check(expected_text_fp=None, prompt='New body.') is not None

    def test_a_supplied_token_is_honoured_on_an_append(self):
        rejection = _check(expected_text_fp='tfp:stale', details='More.', append=True)

        assert rejection is not None
        assert rejection.to_error_dict()['error_type'] == 'TaskTextBaseStale'

    def test_a_task_with_no_text_accepts_the_empty_text_token(self):
        assert _check(
            expected_text_fp=text_fp_of({}), before={'id': '4644'}, details='First body.',
        ) is None

    def test_every_rejection_reads_as_a_failed_write(self):
        rejections = [
            _check(expected_text_fp='tfp:stale', details='New body.'),
            _check(expected_text_fp=None, details='New body.'),
        ]

        for rejection in rejections:
            assert rejection is not None
            assert write_failure_reason(rejection.to_error_dict()) is not None


OLD_PARAGRAPH = 'Evidence relay: gates 3658/3659/3660 still pending.'
NEW_PARAGRAPH = 'Evidence relay: gates 3658/3659/4856 still pending.'


def _verify(after: object, **kwargs):
    return verify_text_write(after, plan_text_write(_forwarded_kwargs(**kwargs)))


class TestVerifyTextWrite:
    def test_a_write_touching_no_text_is_not_verified(self):
        assert _verify({'details': OLD_PARAGRAPH}, metadata='{"a": 1}') is None

    def test_a_landed_replace_is_confirmed(self):
        verification = _verify({'details': NEW_PARAGRAPH}, details=NEW_PARAGRAPH)

        assert verification is not None
        assert verification.confirmed is True
        assert verification.unconfirmed_fields == ()

    def test_a_landed_append_is_confirmed(self):
        verification = _verify(
            {'details': OLD_PARAGRAPH + '\n\n' + NEW_PARAGRAPH},
            details=NEW_PARAGRAPH,
            append=True,
        )

        assert verification is not None
        assert verification.confirmed is True

    def test_the_old_paragraph_still_stored_is_unconfirmed(self):
        verification = _verify({'details': OLD_PARAGRAPH}, details=NEW_PARAGRAPH)

        assert verification is not None
        assert verification.confirmed is False
        assert verification.unconfirmed_fields == ('details',)

    def test_a_missing_or_none_field_is_unconfirmed(self):
        for after in ({'id': '3708'}, {'details': None}):
            verification = _verify(after, details=NEW_PARAGRAPH)

            assert verification is not None
            assert verification.confirmed is False

    def test_line_ending_and_trailing_whitespace_differences_still_confirm(self):
        verification = _verify(
            {'details': 'First line.\nSecond line.'},
            details='First line.\r\nSecond line.  \n',
        )

        assert verification is not None
        assert verification.confirmed is True

    def test_a_prompt_write_is_checked_against_details(self):
        verification = _verify({'details': NEW_PARAGRAPH}, prompt=NEW_PARAGRAPH)

        assert verification is not None
        assert verification.confirmed is True

    def test_only_the_field_that_did_not_land_is_flagged(self):
        verification = _verify(
            {'description': 'New summary.', 'details': OLD_PARAGRAPH},
            description='New summary.',
            details=NEW_PARAGRAPH,
        )

        assert verification is not None
        assert verification.confirmed is False
        assert verification.unconfirmed_fields == ('details',)

    def test_a_non_mapping_after_is_unconfirmed(self):
        verification = _verify(None, details=NEW_PARAGRAPH)

        assert verification is not None
        assert verification.confirmed is False

    def test_an_unlanded_trimming_replace_is_unconfirmed(self):
        verification = _verify(
            {'details': 'Para A. Para B is a false premise.'}, details='Para A.',
        )

        assert verification is not None
        assert verification.confirmed is False
        assert verification.unconfirmed_fields == ('details',)

    def test_an_unlanded_clear_is_unconfirmed(self):
        verification = _verify({'details': OLD_PARAGRAPH}, details='')

        assert verification is not None
        assert verification.confirmed is False

    def test_a_landed_clear_is_confirmed(self):
        verification = _verify({'details': ''}, details='')

        assert verification is not None
        assert verification.confirmed is True

    def test_a_clear_read_back_against_a_missing_or_none_field_is_unconfirmed(self):
        for after in ({'id': '3708'}, {'details': None}):
            verification = _verify(after, details='')

            assert verification is not None
            assert verification.confirmed is False

    def test_an_unlanded_description_trim_is_unconfirmed(self):
        verification = _verify({'description': 'Fix the reader'}, description='Fix')

        assert verification is not None
        assert verification.confirmed is False
        assert verification.unconfirmed_fields == ('description',)

    def test_an_unlanded_prompt_trim_is_unconfirmed(self):
        verification = _verify({'details': OLD_PARAGRAPH}, prompt=OLD_PARAGRAPH[:20])

        assert verification is not None
        assert verification.confirmed is False

    def test_an_append_whose_payload_is_not_at_the_end_is_unconfirmed(self):
        verification = _verify(
            {'details': NEW_PARAGRAPH + '\n\n' + OLD_PARAGRAPH},
            details=NEW_PARAGRAPH,
            append=True,
        )

        assert verification is not None
        assert verification.confirmed is False

    def test_an_append_onto_an_empty_base_is_confirmed(self):
        verification = _verify({'details': NEW_PARAGRAPH}, details=NEW_PARAGRAPH, append=True)

        assert verification is not None
        assert verification.confirmed is True

    def test_result_keys(self):
        verification = _verify(
            {'description': 'New summary.', 'details': OLD_PARAGRAPH},
            description='New summary.',
            details=NEW_PARAGRAPH,
            append=True,
        )

        assert verification is not None
        assert verification.to_result_keys() == {
            'text_verified': False,
            'text_verification': {
                'description': {'confirmed': True, 'appended': False},
                'details': {'confirmed': False, 'appended': True},
            },
        }


WRITE_RESULT = {'id': '3708', 'details': NEW_PARAGRAPH, 'title': 'γ'}


async def _land(
    result: dict[str, Any],
    *,
    after: object = None,
    get_task: AsyncMock | None = None,
    file_finding: AsyncMock | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return await verify_text_write_landed(
        result,
        task_id='3708',
        project_id='dark_factory',
        project_root='/project',
        write=plan_text_write(_forwarded_kwargs(**kwargs)),
        get_task=get_task if get_task is not None else AsyncMock(return_value=after),
        file_finding=file_finding,
    )


class TestVerifyTextWriteLanded:
    @pytest.mark.asyncio
    async def test_a_write_touching_no_text_is_returned_unchanged_without_a_read(self):
        get_task = AsyncMock()
        result = {'id': '3708', 'metadata': {'a': 1}}

        landed = await _land(result, get_task=get_task, metadata='{"a": 1}')

        assert landed == result
        assert 'text_verified' not in landed
        get_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_confirmed_write_is_annotated_true_and_files_nothing(self):
        file_finding = AsyncMock()
        result = dict(WRITE_RESULT)

        landed = await _land(
            result,
            after={'details': NEW_PARAGRAPH},
            file_finding=file_finding,
            details=NEW_PARAGRAPH,
        )

        assert landed['text_verified'] is True
        assert 'text_verification' in landed
        file_finding.assert_not_awaited()
        assert result == WRITE_RESULT

    @pytest.mark.asyncio
    async def test_an_unconfirmed_write_is_annotated_false_and_files_a_finding(self):
        file_finding = AsyncMock()
        get_task = AsyncMock(return_value={'details': OLD_PARAGRAPH})
        result = dict(WRITE_RESULT)

        landed = await _land(
            result, get_task=get_task, file_finding=file_finding, details=NEW_PARAGRAPH,
        )

        assert landed['text_verified'] is False
        get_task.assert_awaited_once_with('3708', '/project')
        (filed,) = file_finding.await_args_list
        (finding,) = filed.args
        assert isinstance(finding, TextWriteUnconfirmedFinding)
        assert finding.task_id == '3708'
        assert finding.project_id == 'dark_factory'
        assert finding.op == 'update_task'
        assert finding.unconfirmed_fields == ('details',)
        assert finding.flag_type == 'task_text_write_unconfirmed'
        assert result == WRITE_RESULT

    @pytest.mark.asyncio
    async def test_an_unconfirmed_write_with_no_filer_is_still_annotated(self):
        landed = await _land(
            dict(WRITE_RESULT), after={'details': OLD_PARAGRAPH}, details=NEW_PARAGRAPH,
        )

        assert landed['text_verified'] is False

    @pytest.mark.asyncio
    async def test_a_failing_after_read_returns_the_original_result(self, caplog):
        result = dict(WRITE_RESULT)
        get_task = AsyncMock(side_effect=RuntimeError('backend gone'))

        with caplog.at_level(logging.ERROR):
            landed = await _land(result, get_task=get_task, details=NEW_PARAGRAPH)

        assert landed == WRITE_RESULT
        assert 'backend gone' in caplog.text
        assert result == WRITE_RESULT

    @pytest.mark.asyncio
    async def test_a_failing_filer_keeps_the_annotation(self):
        result = copy.deepcopy(WRITE_RESULT)
        file_finding = AsyncMock(side_effect=RuntimeError('report store gone'))

        landed = await _land(
            result,
            after={'details': OLD_PARAGRAPH},
            file_finding=file_finding,
            details=NEW_PARAGRAPH,
        )

        assert landed['text_verified'] is False
        file_finding.assert_awaited_once()
        assert result == WRITE_RESULT


# --------------------------------------------------------------------------- #
# Interceptor boundary fixtures (copied from test_stale_gate_citation_guard.py;
# this repo defines the quartet per test file rather than in a conftest)
# --------------------------------------------------------------------------- #


@pytest.fixture
def taskmaster():
    tm = AsyncMock()
    tm.get_task = AsyncMock(return_value={'id': '1', 'status': 'pending', 'title': 'Test Task'})
    tm.set_task_status = AsyncMock(return_value={'success': True})
    tm.get_tasks = AsyncMock(return_value={'tasks': []})
    tm.add_task = AsyncMock(return_value={'id': '2', 'title': 'New Task'})
    tm.update_task = AsyncMock(return_value={'success': True})
    tm.remove_tasks = AsyncMock(return_value={'success': True})
    tm.add_dependency = AsyncMock(return_value={'success': True})
    tm.remove_dependency = AsyncMock(return_value={'success': True})
    return tm


@pytest.fixture
def reconciler():
    r = AsyncMock()
    r.reconcile_task = AsyncMock(return_value={'actions': [{'type': 'knowledge_captured'}]})
    return r


@pytest_asyncio.fixture
async def event_buffer(tmp_path):
    buf = EventBuffer(db_path=tmp_path / 'interceptor_eb.db', buffer_size_threshold=100)
    await buf.initialize()
    yield buf
    await buf.close()


@pytest.fixture
def interceptor(taskmaster, reconciler, event_buffer):
    return TaskInterceptor(taskmaster, reconciler, event_buffer)


LIVE_ROW = {
    'id': '4644',
    'status': 'pending',
    'title': 'Two-writer clobber',
    'description': 'Current summary.',
    'details': OLD_PARAGRAPH,
    'dependencies': [],
}


@pytest.fixture
def live_row(taskmaster):
    row = dict(LIVE_ROW)
    taskmaster.get_task = AsyncMock(return_value=row)
    return row


def _error_type(result: dict[str, Any]) -> object:
    return result.get('error_type')


class TestInterceptorTextBaseBoundary:
    @pytest.mark.asyncio
    async def test_a_recon_replace_without_a_token_is_rejected_before_the_write(
        self, interceptor, taskmaster, live_row,
    ):
        result = await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id=RECON_AGENT,
        )

        assert _error_type(result) == 'ReconTaskTextBaseRequired'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_recon_replace_with_the_current_token_is_written(
        self, interceptor, taskmaster, live_row,
    ):
        await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id=RECON_AGENT,
            expected_text_fp=text_fp_of(live_row),
        )

        taskmaster.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_recon_replace_with_a_stale_token_is_rejected(
        self, interceptor, taskmaster, live_row,
    ):
        result = await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id=RECON_AGENT,
            expected_text_fp=text_fp_of({'details': 'an earlier read'}),
        )

        assert _error_type(result) == 'TaskTextBaseStale'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_recon_details_append_needs_no_token(
        self, interceptor, taskmaster, live_row,
    ):
        await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, append=True, agent_id=RECON_AGENT,
        )

        taskmaster.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('text_kwarg', ['description', 'prompt'])
    async def test_a_recon_description_or_prompt_write_without_a_token_is_rejected(
        self, interceptor, taskmaster, live_row, text_kwarg,
    ):
        result = await interceptor.update_task(
            '4644', '/project', agent_id=RECON_AGENT, **{text_kwarg: 'Rewritten.'},
        )

        assert _error_type(result) == 'ReconTaskTextBaseRequired'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_recon_metadata_only_write_is_unaffected(
        self, interceptor, taskmaster, live_row,
    ):
        await interceptor.update_task(
            '4644', '/project', metadata='{"x_note": "seen"}', agent_id=RECON_AGENT,
        )

        taskmaster.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize('agent_id', [None, 'claude-interactive'])
    async def test_a_non_recon_replace_without_a_token_is_written_with_no_before_read(
        self, interceptor, taskmaster, live_row, agent_id,
    ):
        reads_before_write: list[int] = []

        async def _record_reads(**_kwargs: Any) -> dict[str, Any]:
            reads_before_write.append(taskmaster.get_task.await_count)
            return {'success': True}

        taskmaster.update_task = AsyncMock(side_effect=_record_reads)

        await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id=agent_id,
        )

        assert reads_before_write == [0]

    @pytest.mark.asyncio
    async def test_a_non_recon_stale_token_is_rejected(self, interceptor, taskmaster, live_row):
        result = await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id='claude-interactive',
            expected_text_fp=text_fp_of({'details': 'an earlier read'}),
        )

        assert _error_type(result) == 'TaskTextBaseStale'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_non_recon_matching_token_is_written(self, interceptor, taskmaster, live_row):
        await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id='claude-interactive',
            expected_text_fp=text_fp_of(live_row),
        )

        taskmaster.update_task.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_the_token_reaches_neither_the_backend_nor_the_journal(
        self, interceptor, taskmaster, live_row,
    ):
        journal = AsyncMock()
        interceptor.set_write_journal(journal)
        token = text_fp_of(live_row)

        await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id=RECON_AGENT,
            expected_text_fp=token,
        )

        assert 'expected_text_fp' not in taskmaster.update_task.await_args.kwargs
        journalled = [call.kwargs['params'] for call in journal.log_write_op.await_args_list]
        assert journalled
        assert all('expected_text_fp' not in params for params in journalled)
        assert all(token not in repr(params) for params in journalled)


def _before_then_after(taskmaster, after: dict[str, Any]) -> dict[str, Any]:
    before = dict(LIVE_ROW)
    taskmaster.get_task = AsyncMock(side_effect=[before, after])
    return before


class TestInterceptorTextVerificationBoundary:
    @pytest.mark.asyncio
    async def test_a_landed_recon_text_write_is_verified(self, interceptor, taskmaster):
        before = _before_then_after(taskmaster, {**LIVE_ROW, 'details': NEW_PARAGRAPH})

        result = await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id=RECON_AGENT,
            expected_text_fp=text_fp_of(before),
        )

        assert result['text_verified'] is True
        assert taskmaster.get_task.await_count == 2

    @pytest.mark.asyncio
    async def test_the_old_paragraph_reading_back_is_unverified(self, interceptor, taskmaster):
        before = _before_then_after(taskmaster, dict(LIVE_ROW))

        result = await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id=RECON_AGENT,
            expected_text_fp=text_fp_of(before),
        )

        assert result['text_verified'] is False

    @pytest.mark.asyncio
    async def test_a_wired_filer_hears_only_the_unconfirmed_write(self, interceptor, taskmaster):
        filer = AsyncMock()
        interceptor.set_text_write_unconfirmed_filer(filer)

        before = _before_then_after(taskmaster, {**LIVE_ROW, 'details': NEW_PARAGRAPH})
        await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id=RECON_AGENT,
            expected_text_fp=text_fp_of(before),
        )
        filer.assert_not_awaited()

        before = _before_then_after(taskmaster, dict(LIVE_ROW))
        await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id=RECON_AGENT,
            expected_text_fp=text_fp_of(before),
        )
        (filed,) = filer.await_args_list
        (finding,) = filed.args
        assert finding.unconfirmed_fields == ('details',)

    @pytest.mark.asyncio
    async def test_a_trim_that_reads_back_the_longer_old_text_is_unverified_and_filed(
        self, interceptor, taskmaster,
    ):
        filer = AsyncMock()
        interceptor.set_text_write_unconfirmed_filer(filer)
        before = _before_then_after(taskmaster, dict(LIVE_ROW))

        result = await interceptor.update_task(
            '4644', '/project', details=OLD_PARAGRAPH[:20], agent_id=RECON_AGENT,
            expected_text_fp=text_fp_of(before),
        )

        assert result['text_verified'] is False
        (filed,) = filer.await_args_list
        (finding,) = filed.args
        assert finding.unconfirmed_fields == ('details',)

    @pytest.mark.asyncio
    async def test_a_recon_metadata_only_write_is_not_read_back(
        self, interceptor, taskmaster, live_row,
    ):
        result = await interceptor.update_task(
            '4644', '/project', metadata='{"x_note": "seen"}', agent_id=RECON_AGENT,
        )

        assert 'text_verified' not in result
        assert taskmaster.get_task.await_count == 1

    @pytest.mark.asyncio
    async def test_a_non_recon_text_write_is_not_read_back(
        self, interceptor, taskmaster, live_row,
    ):
        result = await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id='claude-interactive',
        )

        assert 'text_verified' not in result
        taskmaster.get_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_gate_rejection_is_not_read_back(self, interceptor, taskmaster, live_row):
        result = await interceptor.update_task(
            '4644', '/project', details=NEW_PARAGRAPH, agent_id=RECON_AGENT,
        )

        assert _error_type(result) == 'ReconTaskTextBaseRequired'
        assert taskmaster.get_task.await_count == 1


class TestStage2PromptNamesTheTextWriteContract:
    """Stage 2 writes task text through the MCP tool, so its prompt is the only
    place it learns the contract: a replace-mode write without the token is
    rejected, and only ``text_verified`` confirms an amendment landed."""

    @pytest.mark.parametrize(
        'name',
        ['expected_text_fp', TASK_TEXT_FP_KEY, TEXT_VERIFIED_KEY, TEXT_BASE_STALE, TEXT_BASE_REQUIRED],
    )
    @pytest.mark.parametrize(
        'prompt',
        [STAGE2_SYSTEM_PROMPT, build_stage2_system_prompt('autopilot_video')],
        ids=['default', 'autopilot_video'],
    )
    def test_the_prompt_names_it(self, prompt: str, name: str):
        assert name in prompt
