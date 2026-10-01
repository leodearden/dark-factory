"""Unit tests for fused_memory.middleware.recurring_gate_guard (task 3588).

Step 1 (RED → step-2 GREEN): the SUBMISSION-READER layer — the canonical
``gate_subject`` metadata key, its read-side aliases, and the
``is_gate_submission`` predicate that decides whether a ``submit_task``
payload is a human-gate carrier at all.

The module does not exist until step-2, so every test imports the names it
needs LOCALLY inside its body — the RED-collection idiom documented at
tests/test_operational_routing_guard.py:7-12 (same style in
test_execution_class_guard.py's TestInjectExecutionClass). That keeps this
file COLLECTABLE at RED: each test fails on its own in-body import, so the
count of pending assertions stays visible instead of the whole module
erroring out at collection time.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from shared.task_statuses import ACTIVE, TERMINAL

_DEFAULT_METADATA = object()


def _row(
    task_id: str = '5902',
    status: str = 'pending',
    title: str = 'GATE: stranded task 5879 needs a human ruling',
    metadata: Any = _DEFAULT_METADATA,
    **extra: Any,
) -> dict[str, Any]:
    """A task row in the real `_row_to_task` wire shape (`id` is a STRING).

    Mirrors tests/test_live_task_write_guard.py::_flat — a plain dict
    factory, no model construction.

    ``metadata`` defaults to a distinct ``_DEFAULT_METADATA`` sentinel, NOT
    to ``None``, so a caller can pass a literal ``metadata=None`` row (the
    malformed-corpus case) without silently getting the default gate blob.
    """
    payload: dict[str, Any] = {
        'id': task_id,
        'title': title,
        'status': status,
        'metadata': (
            {'execution_class': 'operational', 'operational_mode': 'gate',
             'gate_subject': '5879'}
            if metadata is _DEFAULT_METADATA
            else metadata
        ),
    }
    payload.update(extra)
    return payload


class TestGateSubjectKeyConstants:
    """The canonical subject key and its ordered read-side alias tuple."""

    def test_canonical_key_is_gate_subject(self):
        from fused_memory.middleware.recurring_gate_guard import GATE_SUBJECT_KEY

        assert GATE_SUBJECT_KEY == 'gate_subject'

    def test_aliases_are_ordered_with_canonical_first(self):
        from fused_memory.middleware.recurring_gate_guard import (
            GATE_SUBJECT_ALIASES,
            GATE_SUBJECT_KEY,
        )

        # Ordered sequence (not a set): resolution order is load-bearing.
        assert isinstance(GATE_SUBJECT_ALIASES, tuple)
        assert GATE_SUBJECT_ALIASES[0] == GATE_SUBJECT_KEY

    def test_aliases_cover_the_already_filed_carrier_spellings(self):
        from fused_memory.middleware.recurring_gate_guard import GATE_SUBJECT_ALIASES

        # 5902/5916/5929 key their subject via stranded_task_id; dark-factory's
        # own gates 3240/3361/3463 key theirs via related_task_id. History is
        # never rewritten, so both must stay readable on the STORED side.
        assert 'stranded_task_id' in GATE_SUBJECT_ALIASES
        assert 'related_task_id' in GATE_SUBJECT_ALIASES

    def test_submission_aliases_exclude_related_task_id(self):
        """The INCOMING order must not honour the see-also spelling.

        Measured in the live corpus: `related_task_id` is a generic
        cross-reference on non-gate work (3042 -> 2885, 3046 -> 3045, both
        code_tdd) as well as a subject on gates. The stored side survives that
        ambiguity because find_open_gate also requires the row to BE a gate;
        the incoming side has no such fallback, so honouring it there would
        hard-reject a novel gate against an unrelated carrier.
        """
        from fused_memory.middleware.recurring_gate_guard import (
            GATE_SUBJECT_SUBMISSION_ALIASES,
        )

        assert 'related_task_id' not in GATE_SUBJECT_SUBMISSION_ALIASES

    def test_submission_aliases_are_ordered_canonical_first_then_stranded(self):
        from fused_memory.middleware.recurring_gate_guard import (
            GATE_SUBJECT_KEY,
            GATE_SUBJECT_SUBMISSION_ALIASES,
        )

        assert isinstance(GATE_SUBJECT_SUBMISSION_ALIASES, tuple)
        assert GATE_SUBJECT_SUBMISSION_ALIASES[0] == GATE_SUBJECT_KEY
        # stranded_task_id is only ever a subject, never a see-also, so it is
        # the one legacy spelling the transition window keeps.
        assert 'stranded_task_id' in GATE_SUBJECT_SUBMISSION_ALIASES

    def test_submission_aliases_are_a_subset_of_the_stored_order(self):
        """Narrower, never wider: every incoming key must also read stored rows.

        A key honoured on the incoming side but not the stored side could
        resolve a subject that no carrier could ever match, silently disabling
        the dedupe for that spelling.
        """
        from fused_memory.middleware.recurring_gate_guard import (
            GATE_SUBJECT_ALIASES,
            GATE_SUBJECT_SUBMISSION_ALIASES,
        )

        assert set(GATE_SUBJECT_SUBMISSION_ALIASES) <= set(GATE_SUBJECT_ALIASES)


class TestExtractGateSubject:
    """extract_gate_subject(metadata) -> str | None."""

    def test_canonical_key_returns_str(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        result = extract_gate_subject({'gate_subject': '5879'})
        assert result == '5879'
        assert isinstance(result, str)

    def test_falls_back_to_stranded_task_id(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'stranded_task_id': '5879'}) == '5879'

    def test_falls_back_to_related_task_id(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        # Default order is the STORED one, which still reads this spelling.
        assert extract_gate_subject({'related_task_id': '5879'}) == '5879'

    def test_explicit_alias_tuple_narrows_resolution(self):
        from fused_memory.middleware.recurring_gate_guard import (
            GATE_SUBJECT_SUBMISSION_ALIASES,
            extract_gate_subject,
        )

        assert (
            extract_gate_subject(
                {'related_task_id': '5879'},
                aliases=GATE_SUBJECT_SUBMISSION_ALIASES,
            )
            is None
        )
        assert (
            extract_gate_subject(
                {'stranded_task_id': '5879'},
                aliases=GATE_SUBJECT_SUBMISSION_ALIASES,
            )
            == '5879'
        )

    def test_alias_order_stranded_beats_related(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert (
            extract_gate_subject(
                {'stranded_task_id': '5879', 'related_task_id': '5858'}
            )
            == '5879'
        )

    def test_canonical_wins_over_alias(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert (
            extract_gate_subject(
                {
                    'gate_subject': '5879',
                    'stranded_task_id': '5858',
                    'related_task_id': '5801',
                }
            )
            == '5879'
        )

    def test_int_scalar_is_coerced_to_str(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        result = extract_gate_subject({'gate_subject': 5879})
        assert result == '5879'
        assert isinstance(result, str)

    def test_surrounding_whitespace_is_stripped(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'gate_subject': '  5879\n'}) == '5879'

    def test_absent_key_returns_none(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'execution_class': 'operational'}) is None

    def test_empty_string_returns_none(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'gate_subject': ''}) is None

    def test_whitespace_only_returns_none(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'gate_subject': '   '}) is None

    def test_none_value_returns_none(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'gate_subject': None}) is None

    def test_non_scalar_values_return_none(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject({'gate_subject': {'id': '5879'}}) is None
        assert extract_gate_subject({'gate_subject': ['5879']}) is None

    def test_bool_is_not_accepted_as_a_scalar(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        # bool is an int subclass — True must not become the subject 'True'.
        assert extract_gate_subject({'gate_subject': True}) is None

    def test_json_string_metadata_blob_is_read(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject(json.dumps({'gate_subject': '5879'})) == '5879'

    def test_unparseable_string_metadata_returns_none_without_raising(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject('not json at all {{{') is None

    def test_none_metadata_returns_none_without_raising(self):
        from fused_memory.middleware.recurring_gate_guard import extract_gate_subject

        assert extract_gate_subject(None) is None


class TestIsGateSubmission:
    """is_gate_submission(metadata) -> bool.

    | execution_class | operational_mode | is_gate_submission |
    |-----------------|------------------|--------------------|
    | operational     | 'gate'           | True               |
    | operational     | absent           | True  (default)    |
    | operational     | 'llm'            | False              |
    | code_tdd        | (any)            | False              |
    | decision        | (any)            | False              |
    | absent          | (any)            | False              |
    """

    def test_operational_with_explicit_gate_is_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert (
            is_gate_submission(
                {'execution_class': 'operational', 'operational_mode': 'gate'}
            )
            is True
        )

    def test_operational_with_absent_mode_is_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        # TaskMetadata.operational_mode is Literal['gate','llm'] = 'gate', and
        # inject_operational_routing coerces operational+absent to a pure gate.
        assert is_gate_submission({'execution_class': 'operational'}) is True

    def test_operational_llm_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert (
            is_gate_submission(
                {'execution_class': 'operational', 'operational_mode': 'llm'}
            )
            is False
        )

    def test_code_tdd_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert (
            is_gate_submission(
                {'execution_class': 'code_tdd', 'operational_mode': 'gate'}
            )
            is False
        )

    def test_decision_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert (
            is_gate_submission(
                {'execution_class': 'decision', 'operational_mode': 'gate'}
            )
            is False
        )

    def test_absent_execution_class_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert is_gate_submission({'operational_mode': 'gate'}) is False

    def test_truthy_non_matching_operational_mode_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        # Value-sensitive, mirroring TaskInterceptor._is_gate_metadata: a
        # truthy non-'gate' value must not satisfy the predicate.
        assert (
            is_gate_submission(
                {'execution_class': 'operational', 'operational_mode': 1}
            )
            is False
        )
        assert (
            is_gate_submission(
                {'execution_class': 'operational', 'operational_mode': 'false'}
            )
            is False
        )

    def test_none_metadata_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert is_gate_submission(None) is False

    def test_unparseable_metadata_is_not_a_gate(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert is_gate_submission('not json at all {{{') is False

    def test_json_string_metadata_blob_is_read(self):
        from fused_memory.middleware.recurring_gate_guard import is_gate_submission

        assert (
            is_gate_submission(json.dumps({'execution_class': 'operational'}))
            is True
        )


class TestFindOpenGate:
    """find_open_gate(tasks, subject) -> mapping | None. PURE — no I/O, no async."""

    def test_matching_pending_carrier_is_returned(self):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        hit = find_open_gate([_row(task_id='5902', status='pending')], '5879')
        assert hit is not None
        assert hit['id'] == '5902'
        assert isinstance(hit['id'], str)
        assert hit['title'] == 'GATE: stranded task 5879 needs a human ruling'

    @pytest.mark.parametrize('status', sorted(str(s) for s in ACTIVE))
    def test_matches_across_every_non_terminal_status(self, status):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        # `deferred` is the planning_mode landing status — it must block too.
        hit = find_open_gate([_row(status=status)], '5879')
        assert hit is not None, f'{status} carrier must block a fresh gate'
        assert hit['id'] == '5902'

    @pytest.mark.parametrize('status', sorted(str(s) for s in TERMINAL))
    def test_terminal_carriers_do_not_block(self, status):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        assert find_open_gate([_row(status=status)], '5879') is None

    def test_different_subject_does_not_match(self):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        assert find_open_gate([_row()], '5858') is None

    def test_non_gate_task_with_same_subject_does_not_match(self):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        code_tdd = _row(
            metadata={'execution_class': 'code_tdd', 'gate_subject': '5879'}
        )
        llm = _row(
            metadata={
                'execution_class': 'operational',
                'operational_mode': 'llm',
                'gate_subject': '5879',
            }
        )
        assert find_open_gate([code_tdd], '5879') is None
        assert find_open_gate([llm], '5879') is None

    def test_empty_corpus_returns_none(self):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        assert find_open_gate([], '5879') is None

    def test_stored_carrier_keyed_by_stranded_task_id_alias_matches(self):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        # The measured 5902/5916/5929 population keys its subject this way.
        row = _row(
            metadata={
                'execution_class': 'operational',
                'operational_mode': 'gate',
                'stranded_task_id': '5879',
            }
        )
        hit = find_open_gate([row], '5879')
        assert hit is not None
        assert hit['id'] == '5902'

    def test_stored_carrier_keyed_by_related_task_id_alias_matches(self):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        row = _row(
            metadata={
                'execution_class': 'operational',
                'related_task_id': '5879',
            }
        )
        hit = find_open_gate([row], '5879')
        assert hit is not None
        assert hit['id'] == '5902'

    def test_stored_int_subject_matches_incoming_string(self):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        row = _row(
            metadata={'execution_class': 'operational', 'gate_subject': 5879}
        )
        hit = find_open_gate([row], '5879')
        assert hit is not None
        assert hit['id'] == '5902'

    def test_first_match_in_corpus_order_wins(self):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        rows = [_row(task_id='5902'), _row(task_id='5916'), _row(task_id='5929')]
        hit = find_open_gate(rows, '5879')
        assert hit is not None
        assert hit['id'] == '5902'

    def test_malformed_rows_never_raise_and_are_skipped(self):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        corpus = [
            _row(task_id='5801', metadata=None),
            _row(task_id='5802', metadata='not json at all {{{'),
            _row(task_id='5803', metadata=['not', 'a', 'dict']),
            _row(task_id='5804', metadata=42),
            'not a mapping at all',
            None,
            42,
        ]
        assert find_open_gate(corpus, '5879') is None

    def test_json_string_metadata_on_a_stored_row_still_matches(self):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        row = _row(
            metadata=json.dumps(
                {'execution_class': 'operational', 'gate_subject': '5879'}
            )
        )
        hit = find_open_gate([row], '5879')
        assert hit is not None
        assert hit['id'] == '5902'

    def test_blank_or_missing_id_row_is_skipped_not_returned(self):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        blank = _row(task_id='')
        missing = _row()
        del missing['id']
        assert find_open_gate([blank], '5879') is None
        assert find_open_gate([missing], '5879') is None
        # ... and a skipped blank row does not mask a real later match.
        hit = find_open_gate([blank, _row(task_id='5916')], '5879')
        assert hit is not None
        assert hit['id'] == '5916'

    def test_none_corpus_returns_none_without_raising(self):
        from fused_memory.middleware.recurring_gate_guard import find_open_gate

        assert find_open_gate(None, '5879') is None


class TestRecurringGateError:
    """recurring_gate_error(existing, subject) -> the rejection payload.

    This is the user-observable contract: "rejected at the boundary with an
    error naming the open carrier's task id".
    """

    @staticmethod
    def _existing() -> dict[str, Any]:
        return {
            'id': '5902',
            'title': 'GATE: stranded task 5879 needs a human ruling',
            'status': 'pending',
        }

    def test_guard_family_shape(self):
        from fused_memory.middleware.recurring_gate_guard import recurring_gate_error

        err = recurring_gate_error(self._existing(), '5879')
        assert isinstance(err['error'], str)
        assert err['error_type'] == 'RecurringGateViolation'
        assert isinstance(err['hint'], str)

    def test_error_message_names_the_carrier_id_and_subject(self):
        from fused_memory.middleware.recurring_gate_guard import recurring_gate_error

        err = recurring_gate_error(self._existing(), '5879')
        assert '5902' in err['error']
        assert '5879' in err['error']

    def test_hint_names_update_task_as_the_amend_path(self):
        from fused_memory.middleware.recurring_gate_guard import recurring_gate_error

        hint = recurring_gate_error(self._existing(), '5879')['hint']
        assert 'update_task' in hint
        assert 'amend' in hint.lower()
        assert '5902' in hint

    def test_hint_warns_that_append_does_not_append_description(self):
        from fused_memory.middleware.recurring_gate_guard import recurring_gate_error

        hint = recurring_gate_error(self._existing(), '5879')['hint']
        assert 'append=True' in hint
        assert 'description' in hint
        assert 'updated_task' in hint

    def test_structured_machine_readable_keys(self):
        from fused_memory.middleware.recurring_gate_guard import recurring_gate_error

        err = recurring_gate_error(self._existing(), '5879')
        # A consumer must never have to regex the prose to learn which
        # carrier to amend.
        assert err['existing_gate_task_id'] == '5902'
        assert isinstance(err['existing_gate_task_id'], str)
        assert err['gate_subject'] == '5879'
        assert isinstance(err['gate_subject'], str)

    def test_total_on_a_carrier_missing_its_title(self):
        from fused_memory.middleware.recurring_gate_guard import recurring_gate_error

        err = recurring_gate_error({'id': '5902'}, '5879')
        assert err['error_type'] == 'RecurringGateViolation'
        assert err['existing_gate_task_id'] == '5902'
        assert '5902' in err['error']


def _gate_meta(subject: str | None = '5879', **extra: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        'execution_class': 'operational',
        'operational_mode': 'gate',
    }
    if subject is not None:
        meta['gate_subject'] = subject
    meta.update(extra)
    return meta


class TestRecurringGateGuardError:
    """The async orchestrator — the only function in the module that does I/O.

    ``fetch_tasks`` is an INJECTED zero-arg async callable (the
    ``live_task_write_guard.GetTaskFn`` precedent), so every test fakes the
    corpus with a one-line AsyncMock and can prove via ``await_count`` that
    the cheap short-circuits do ZERO I/O.
    """

    @staticmethod
    def _corpus(*rows: Any) -> AsyncMock:
        return AsyncMock(return_value={'tasks': list(rows)})

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'agent_id', ['claude-interactive', None, 'orchestrator-merge', 'recon-stage']
    )
    async def test_non_recon_caller_is_exempt_and_does_no_io(self, agent_id):
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        fetch = self._corpus(_row())
        assert (
            await recurring_gate_guard_error(
                _gate_meta(), agent_id, '/p', fetch_tasks=fetch
            )
            is None
        )
        assert fetch.await_count == 0

    @pytest.mark.asyncio
    async def test_non_gate_submission_does_no_io(self):
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        fetch = self._corpus(_row())
        meta = {'execution_class': 'code_tdd', 'gate_subject': '5879'}
        assert (
            await recurring_gate_guard_error(
                meta, 'recon-stage-task_knowledge_sync', '/p', fetch_tasks=fetch
            )
            is None
        )
        assert fetch.await_count == 0

    @pytest.mark.asyncio
    async def test_gate_without_a_subject_does_no_io(self):
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        fetch = self._corpus(_row())
        # No dedupe key => nothing to enforce.
        assert (
            await recurring_gate_guard_error(
                _gate_meta(subject=None),
                'recon-stage-task_knowledge_sync',
                '/p',
                fetch_tasks=fetch,
            )
            is None
        )
        assert fetch.await_count == 0

    @pytest.mark.asyncio
    async def test_no_matching_carrier_passes_with_exactly_one_read(self):
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        fetch = self._corpus(
            _row(metadata={'execution_class': 'operational', 'gate_subject': '5858'})
        )
        assert (
            await recurring_gate_guard_error(
                _gate_meta(), 'recon-stage-task_knowledge_sync', '/p', fetch_tasks=fetch
            )
            is None
        )
        assert fetch.await_count == 1

    @pytest.mark.asyncio
    async def test_matching_open_carrier_is_rejected_naming_its_id(self):
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        fetch = self._corpus(_row(task_id='5902'))
        err = await recurring_gate_guard_error(
            _gate_meta(), 'recon-stage-task_knowledge_sync', '/p', fetch_tasks=fetch
        )
        assert err is not None
        assert err['error_type'] == 'RecurringGateViolation'
        assert err['existing_gate_task_id'] == '5902'
        assert '5902' in err['error']
        assert fetch.await_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize('status', sorted(str(s) for s in TERMINAL))
    async def test_terminal_carrier_does_not_block_a_recurrence(self, status):
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        # The condition genuinely recurred after closure — a fresh gate is
        # exactly what should happen.
        fetch = self._corpus(_row(status=status))
        assert (
            await recurring_gate_guard_error(
                _gate_meta(), 'recon-stage-task_knowledge_sync', '/p', fetch_tasks=fetch
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_incoming_alias_matches_stored_canonical(self):
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        incoming = _gate_meta(subject=None, stranded_task_id='5879')
        fetch = self._corpus(_row(task_id='5902'))
        err = await recurring_gate_guard_error(
            incoming, 'recon-stage-task_knowledge_sync', '/p', fetch_tasks=fetch
        )
        assert err is not None
        assert err['existing_gate_task_id'] == '5902'

    @pytest.mark.asyncio
    async def test_incoming_see_also_related_task_id_is_not_a_subject(self):
        """A see-also pointer must never become the dedupe subject.

        The expensive failure direction: a genuinely novel human decision is
        hard-rejected against a carrier it has nothing to do with, and the
        rejection prose asserts the two share a subject. Asserting
        `await_count == 0` also pins that this resolves to "no subject" and
        short-circuits, rather than reading the corpus and missing.
        """
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        # The incoming gate is about something else entirely; 5879 is only a
        # cross-reference. An open carrier for 5879 exists in the corpus.
        incoming = _gate_meta(subject=None, related_task_id='5879')
        fetch = self._corpus(_row(task_id='5902'))
        assert (
            await recurring_gate_guard_error(
                incoming, 'recon-stage-task_knowledge_sync', '/p', fetch_tasks=fetch
            )
            is None
        )
        assert fetch.await_count == 0

    @pytest.mark.asyncio
    async def test_incoming_canonical_matches_stored_related_task_id(self):
        """dark-factory gates 3240/3361/3463 key their subject this way.

        The stored side keeps the wider alias order precisely so those real
        carriers stay matchable; only the incoming side was narrowed.
        """
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        stored = _row(
            task_id='3463',
            metadata={
                'execution_class': 'operational',
                'operational_mode': 'gate',
                'related_task_id': '5879',
            },
        )
        err = await recurring_gate_guard_error(
            _gate_meta(),
            'recon-stage-task_knowledge_sync',
            '/p',
            fetch_tasks=self._corpus(stored),
        )
        assert err is not None
        assert err['existing_gate_task_id'] == '3463'

    @pytest.mark.asyncio
    async def test_incoming_canonical_matches_stored_alias(self):
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        stored = _row(
            task_id='5916',
            metadata={
                'execution_class': 'operational',
                'operational_mode': 'gate',
                'stranded_task_id': '5879',
            },
        )
        fetch = self._corpus(stored)
        err = await recurring_gate_guard_error(
            _gate_meta(), 'recon-stage-task_knowledge_sync', '/p', fetch_tasks=fetch
        )
        assert err is not None
        assert err['existing_gate_task_id'] == '5916'

    @pytest.mark.asyncio
    async def test_fails_open_when_the_lookup_raises(self):
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        fetch = AsyncMock(side_effect=RuntimeError('backend down'))
        # Must not propagate: failing closed would block EVERY recon human
        # gate on a transient blip, including genuinely novel ones.
        assert (
            await recurring_gate_guard_error(
                _gate_meta(), 'recon-stage-task_knowledge_sync', '/p', fetch_tasks=fetch
            )
            is None
        )
        assert fetch.await_count == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'junk', [None, [], ['not a row'], 42, 'a string', {'no_tasks_key': 1}]
    )
    async def test_junk_lookup_result_returns_none_without_raising(self, junk):
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        fetch = AsyncMock(return_value=junk)
        assert (
            await recurring_gate_guard_error(
                _gate_meta(), 'recon-stage-task_knowledge_sync', '/p', fetch_tasks=fetch
            )
            is None
        )

    @pytest.mark.asyncio
    async def test_bare_list_lookup_result_is_accepted(self):
        from fused_memory.middleware.recurring_gate_guard import (
            recurring_gate_guard_error,
        )

        fetch = AsyncMock(return_value=[_row(task_id='5929')])
        err = await recurring_gate_guard_error(
            _gate_meta(), 'recon-stage-task_knowledge_sync', '/p', fetch_tasks=fetch
        )
        assert err is not None
        assert err['existing_gate_task_id'] == '5929'
