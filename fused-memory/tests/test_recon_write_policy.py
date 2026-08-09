"""Tests for the server-side ReconWritePolicy gate (W5-ζ, task 2224).

Rejects recon-stage task writes at the interceptor boundary so the post-hoc
reconciliation guards (μ's job to delete) become redundant. Three independent
early-return gates in :func:`recon_write_policy.check`:

1. ``op == 'update_task'`` AND ``live_status`` is terminal ->
   ``ReconTerminalWriteRejected``.
2. ``op == 'set_task_status'`` AND a live workflow is detected for the task ->
   ``ReconLiveWorkflowWriteRejected``.
3. ``snapshot_token is not None`` AND it disagrees with ``live_status`` (any
   op) -> ``ReconStaleSnapshotRejected``.

This module unit-tests the pure ``Verdict``/``check()``/``extract_snapshot_token``
building blocks in isolation. Interceptor boundary tests (P1/P2/P3) reuse the
fixtures from ``test_task_write_agent_id.py`` and live further down this file.
"""

from __future__ import annotations

import subprocess
import threading
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from shared.task_metadata import _BLESSED_METADATA_KEYS, parse_metadata

from fused_memory.middleware import recon_write_policy
from fused_memory.middleware.task_interceptor import TaskInterceptor
from fused_memory.reconciliation.event_buffer import EventBuffer

# Recon-stage agent_id used by the interceptor boundary tests (P1/P2/P3) —
# matches test_task_write_agent_id.py's AGENT_ID so the recon-stage
# task-write setup is identical to the epsilon task's.
AGENT_ID = 'recon-stage-task_knowledge_sync'

# ---------------------------------------------------------------------------
# Verdict dataclass
# ---------------------------------------------------------------------------


class TestVerdict:
    def test_ok_verdict_is_not_rejection(self):
        verdict = recon_write_policy.Verdict(outcome='ok')
        assert verdict.is_rejection is False

    def test_ok_verdict_to_error_dict_is_empty(self):
        verdict = recon_write_policy.Verdict(outcome='ok')
        assert verdict.to_error_dict() == {}

    def test_rejection_verdict_is_rejection(self):
        verdict = recon_write_policy.Verdict(
            outcome='rejection',
            op='update_task',
            task_id='1',
            agent_id='recon-stage-x',
            error_type='ReconTerminalWriteRejected',
            reason='task is done',
            live_status='done',
        )
        assert verdict.is_rejection is True

    def test_rejection_verdict_to_error_dict_shape(self):
        verdict = recon_write_policy.Verdict(
            outcome='rejection',
            op='update_task',
            task_id='1',
            agent_id='recon-stage-x',
            error_type='ReconTerminalWriteRejected',
            reason='task is done',
            live_status='done',
        )
        error_dict = verdict.to_error_dict()
        assert isinstance(error_dict['error'], str)
        assert error_dict['error_type'] == 'ReconTerminalWriteRejected'
        assert error_dict['agent_id'] == 'recon-stage-x'
        assert error_dict['task_id'] == '1'
        assert error_dict['op'] == 'update_task'
        assert error_dict['live_status'] == 'done'

    def test_ok_verdict_corrective_path_defaults_empty(self):
        verdict = recon_write_policy.Verdict(outcome='ok')
        assert verdict.corrective_path == ''
        assert verdict.to_error_dict() == {}

    def test_rejection_verdict_corrective_path_surfaces_in_error_dict(self):
        verdict = recon_write_policy.Verdict(
            outcome='rejection',
            op='update_task',
            task_id='1',
            agent_id='recon-stage-x',
            error_type='ReconTerminalWriteRejected',
            reason='task is done',
            live_status='done',
            corrective_path='set_task_status_done_provenance_repair',
        )
        assert (
            verdict.to_error_dict()['corrective_path']
            == 'set_task_status_done_provenance_repair'
        )


# ---------------------------------------------------------------------------
# check() helper
# ---------------------------------------------------------------------------


def _check(op: str, **overrides) -> recon_write_policy.Verdict:
    """Call check() with sensible defaults; override any kwarg via overrides."""
    kwargs = {
        'task_id': '1',
        'project_root': '/p',
        'agent_id': 'recon-stage-x',
        'target_status': None,
        'live_status': 'pending',
        'snapshot_token': None,
    }
    kwargs.update(overrides)
    return recon_write_policy.check(op, **kwargs)


# ---------------------------------------------------------------------------
# check() gate 1 — terminal (update_task only)
# ---------------------------------------------------------------------------


class TestCheckGate1Terminal:
    def test_update_task_on_done_task_rejects(self):
        verdict = _check('update_task', live_status='done')
        assert verdict.is_rejection is True
        assert verdict.error_type == 'ReconTerminalWriteRejected'

    def test_update_task_on_cancelled_task_rejects(self):
        verdict = _check('update_task', live_status='cancelled')
        assert verdict.is_rejection is True
        assert verdict.error_type == 'ReconTerminalWriteRejected'

    def test_update_task_on_in_progress_task_is_ok(self):
        verdict = _check('update_task', live_status='in-progress')
        assert verdict.is_rejection is False

    def test_set_task_status_op_is_not_scoped_by_gate_1(self):
        """Gate 1 is update_task-only: a set_task_status call against a done
        task must never surface ReconTerminalWriteRejected."""
        verdict = _check('set_task_status', target_status='pending', live_status='done')
        assert verdict.error_type != 'ReconTerminalWriteRejected'

    def test_terminal_rejection_on_done_populates_corrective_path(self):
        """Bound to the source-of-truth TERMINAL_CORRECTIVE_PATH constant
        (rather than a repeated literal) so an accidental change to its
        value is caught here. The sibling cancelled-task test below keeps
        one explicit literal pin to lock the on-the-wire value."""
        verdict = _check('update_task', live_status='done')
        assert verdict.corrective_path == recon_write_policy.TERMINAL_CORRECTIVE_PATH
        assert (
            verdict.to_error_dict()['corrective_path']
            == recon_write_policy.TERMINAL_CORRECTIVE_PATH
        )

    def test_terminal_rejection_on_cancelled_populates_corrective_path(self):
        verdict = _check('update_task', live_status='cancelled')
        assert verdict.corrective_path == 'set_task_status_done_provenance_repair'
        assert (
            verdict.to_error_dict()['corrective_path']
            == 'set_task_status_done_provenance_repair'
        )

    def test_other_gate_rejections_leave_corrective_path_empty(self, monkeypatch):
        """Scoping: corrective_path is a Gate-1-only redirect to the
        same-status done_provenance repair seam. Neither a Gate-2
        live-workflow rejection nor a Gate-3 stale-snapshot rejection is
        served by that seam, so both must leave corrective_path == ''."""
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: True,
        )
        gate2_verdict = _check('set_task_status', live_status='in-progress')
        assert gate2_verdict.error_type == 'ReconLiveWorkflowWriteRejected'
        assert gate2_verdict.corrective_path == ''

        gate3_verdict = _check(
            'update_task', live_status='in-progress', snapshot_token='pending',
        )
        assert gate3_verdict.error_type == 'ReconStaleSnapshotRejected'
        assert gate3_verdict.corrective_path == ''

    def test_terminal_hint_routes_to_set_task_status_done_provenance_repair(self):
        """The Gate 1 hint must route metadata/done_provenance corrections
        to the sanctioned same-status set_task_status(..., 'done',
        done_provenance=...) repair seam. Asserted on positive semantic
        substrings only, to avoid a brittle wording pin — a fixed negative
        phrase pin (e.g. asserting some reopen-flavored phrase is absent)
        would be fragile against legitimate future rewording and is not
        needed: the old misleading hint this replaces ("route through
        set_task_status with a reopen_reason") never mentioned
        done_provenance at all, so the 'done_provenance' substring check
        alone already fails against a regression back to it."""
        hint = _check('update_task', live_status='done').hint
        assert 'set_task_status' in hint
        assert 'done_provenance' in hint

    def test_terminal_hint_states_content_fields_have_no_recon_corrective_path(self):
        """The Gate 1 hint must be honest that the done_provenance repair seam
        is the ONLY recon-stage correction on a terminal task — load-bearing
        string content fields (details/description/title) have NO recon-stage
        corrective path and must be corrected via a human-gated workaround
        task. Asserted on positive semantic substrings only (matching
        ``test_terminal_hint_routes_to_set_task_status_done_provenance_repair``'s
        style), to avoid a brittle wording pin: the hint must (a) name a
        content-field example and (b) direct the caller to a human-gated
        workaround task. This fails against the old over-promising hint (which
        claimed the seam could 'correct done_provenance or other metadata' and
        mentioned neither 'details' nor 'workaround')."""
        hint = _check('update_task', live_status='done').hint
        assert 'details' in hint
        assert 'workaround' in hint

    def test_annotation_clear_exempts_done_task_from_gate_1(self):
        verdict = _check('update_task', live_status='done', is_annotation_clear=True)
        assert verdict.is_rejection is False
        assert verdict.error_type != 'ReconTerminalWriteRejected'

    def test_annotation_clear_exempts_cancelled_task_from_gate_1(self):
        verdict = _check('update_task', live_status='cancelled', is_annotation_clear=True)
        assert verdict.is_rejection is False
        assert verdict.error_type != 'ReconTerminalWriteRejected'

    def test_annotation_clear_false_still_rejects(self):
        verdict = _check('update_task', live_status='done', is_annotation_clear=False)
        assert verdict.is_rejection is True
        assert verdict.error_type == 'ReconTerminalWriteRejected'

    def test_annotation_clear_omitted_defaults_false_still_rejects(self):
        """Backward compatibility: existing callers that never pass
        is_annotation_clear must see unchanged Gate 1 behavior."""
        verdict = _check('update_task', live_status='done')
        assert verdict.is_rejection is True
        assert verdict.error_type == 'ReconTerminalWriteRejected'

    def test_annotation_clear_still_subject_to_gate_3_stale_snapshot(self):
        """The exemption bypasses Gate 1 only — Gate 3 (stale snapshot)
        still composes and fires on a clear write carrying a stale token."""
        verdict = _check(
            'update_task',
            live_status='done',
            is_annotation_clear=True,
            snapshot_token='pending',
        )
        assert verdict.is_rejection is True
        assert verdict.error_type == 'ReconStaleSnapshotRejected'

    def test_annotation_clear_on_non_terminal_task_is_unaffected(self):
        verdict = _check(
            'update_task', live_status='in-progress', is_annotation_clear=False,
        )
        assert verdict.is_rejection is False


# ---------------------------------------------------------------------------
# check() gate 2 — live workflow (set_task_status only)
# ---------------------------------------------------------------------------


class TestCheckGate2LiveWorkflow:
    def test_set_task_status_with_live_workflow_rejects(self, monkeypatch):
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: True,
        )
        verdict = _check('set_task_status', live_status='in-progress')
        assert verdict.is_rejection is True
        assert verdict.error_type == 'ReconLiveWorkflowWriteRejected'

    def test_set_task_status_without_live_workflow_is_ok(self, monkeypatch):
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: False,
        )
        verdict = _check('set_task_status', live_status='in-progress')
        assert verdict.is_rejection is False

    def test_update_task_op_is_not_scoped_by_gate_2(self, monkeypatch):
        """Gate 2 is set_task_status-only: update_task must not surface
        ReconLiveWorkflowWriteRejected even when the detector reports live."""
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: True,
        )
        verdict = _check('update_task', live_status='in-progress')
        assert verdict.error_type != 'ReconLiveWorkflowWriteRejected'

    def test_gate_2_forwards_live_status_as_status_kwarg(self, monkeypatch):
        """is_workflow_live_for_task must receive the caller's live_status as
        its `status` kwarg so it can suppress the project-wide
        orchestrator_live signal for done/cancelled/deferred tasks (see
        live_workflow_detector.ORCH_LIVE_INELIGIBLE_STATUSES) — otherwise a
        live orchestrator elsewhere in the project would falsely flag a
        terminal/deferred task's set_task_status write as gate-2-live."""
        captured = {}

        def _spy(*args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs
            return False

        monkeypatch.setattr(recon_write_policy, 'is_workflow_live_for_task', _spy)
        _check('set_task_status', task_id='7', live_status='deferred')

        assert captured['kwargs'].get('status') == 'deferred'

    # -- task_metadata forwarding (task 3751) --------------------------------
    #
    # Gate 2 forwarded ONLY `status`, never `task_kind`, so no task_kind-scoped
    # detector rule was ever reachable here — including task 2067's rule 2. The
    # new `task_metadata` kwarg lets check() derive both `task_kind` and
    # `pure_gate` (via is_pure_gate_metadata) and forward them, which is what
    # makes rule 5 — and hence the fix for task 3845's 3-cycle stall — reachable
    # at the one gate where the bare orchestrator lock actually blocks a write.

    @staticmethod
    def _capture_detector_kwargs(monkeypatch) -> dict:
        """Install a kwarg-capturing is_workflow_live_for_task spy returning False."""
        captured: dict = {}

        def _spy(*args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs
            return False

        monkeypatch.setattr(recon_write_policy, 'is_workflow_live_for_task', _spy)
        return captured

    def test_gate_2_forwards_pure_gate_shape_from_task_metadata(self, monkeypatch):
        """THE FIX — a pending deterministic PURE GATE's metadata yields
        task_kind='deterministic' and pure_gate=True at the detector.

        This is dark_factory task 3845's real metadata (verified first-hand):
        `always_escalates=True` with NO `before_done`. The incidental
        `operational_mode`/`execution_class` labels are deliberately NOT what the
        classification keys on — see is_pure_gate_metadata.
        """
        captured = self._capture_detector_kwargs(monkeypatch)
        _check(
            'set_task_status',
            task_id='3845',
            live_status='pending',
            task_metadata={
                'task_kind': 'deterministic',
                'always_escalates': True,
                'operational_mode': 'gate',
                'execution_class': 'operational',
            },
        )

        assert captured['kwargs'].get('status') == 'pending'
        assert captured['kwargs'].get('task_kind') == 'deterministic'
        assert captured['kwargs'].get('pure_gate') is True

    def test_gate_2_before_done_metadata_is_not_a_pure_gate(self, monkeypatch):
        """NARROWING — a deterministic task WITH `before_done` forwards
        pure_gate=False, so rule 5 stays inert and the orchestrator lock keeps
        protecting it from a recon race while it may be mid-deploy."""
        captured = self._capture_detector_kwargs(monkeypatch)
        _check(
            'set_task_status',
            task_id='7',
            live_status='pending',
            task_metadata={
                'task_kind': 'deterministic',
                'always_escalates': True,
                'before_done': {'kind': 'predicate'},
            },
        )

        assert captured['kwargs'].get('task_kind') == 'deterministic'
        assert captured['kwargs'].get('pure_gate') is False

    def test_gate_2_forwards_normal_task_kind(self, monkeypatch):
        """An ordinary task forwards its task_kind with pure_gate=False."""
        captured = self._capture_detector_kwargs(monkeypatch)
        _check(
            'set_task_status',
            task_id='7',
            live_status='pending',
            task_metadata={'task_kind': 'normal'},
        )

        assert captured['kwargs'].get('task_kind') == 'normal'
        assert captured['kwargs'].get('pure_gate') is False

    def test_gate_2_without_task_metadata_forwards_none_and_false(self, monkeypatch):
        """BACKWARD COMPATIBILITY — omitting task_metadata reproduces today's
        behavior exactly: task_kind=None, pure_gate=False. Every caller that
        does not pass the new kwarg is unaffected."""
        captured = self._capture_detector_kwargs(monkeypatch)
        _check('set_task_status', task_id='7', live_status='pending')

        assert captured['kwargs'].get('task_kind') is None
        assert captured['kwargs'].get('pure_gate') is False

    def test_gate_2_coerces_json_string_task_metadata(self, monkeypatch):
        """A JSON-object-string metadata blob is coerced via
        _coerce_metadata_dict, the module's existing shared idiom."""
        captured = self._capture_detector_kwargs(monkeypatch)
        _check(
            'set_task_status',
            task_id='3845',
            live_status='pending',
            task_metadata='{"task_kind": "deterministic", "always_escalates": true}',
        )

        assert captured['kwargs'].get('task_kind') == 'deterministic'
        assert captured['kwargs'].get('pure_gate') is True

    @pytest.mark.parametrize(
        'task_metadata',
        ['not json', 42, '[]', None, ['a'], ''],
        ids=['invalid-json', 'int', 'json-list', 'none', 'list', 'empty-str'],
    )
    def test_gate_2_malformed_task_metadata_fails_safe_toward_live(
        self, monkeypatch, task_metadata
    ):
        """FAIL-SAFE — anything that is not a dict / JSON-object string degrades
        to task_kind=None, pure_gate=False without raising, so an unparseable
        metadata blob leaves the task live rather than suppressing its signal."""
        captured = self._capture_detector_kwargs(monkeypatch)
        verdict = _check(
            'set_task_status',
            task_id='7',
            live_status='pending',
            task_metadata=task_metadata,
        )

        assert captured['kwargs'].get('task_kind') is None
        assert captured['kwargs'].get('pure_gate') is False
        assert verdict.is_rejection is False


# ---------------------------------------------------------------------------
# check() gate 3 — stale snapshot (op-agnostic) + precedence
# ---------------------------------------------------------------------------


class TestCheckGate3StaleSnapshot:
    def test_stale_snapshot_rejects(self):
        verdict = _check(
            'update_task', live_status='in-progress', snapshot_token='pending',
        )
        assert verdict.is_rejection is True
        assert verdict.error_type == 'ReconStaleSnapshotRejected'

    def test_fresh_snapshot_matching_live_status_is_ok(self):
        verdict = _check(
            'update_task', live_status='in-progress', snapshot_token='in-progress',
        )
        assert verdict.is_rejection is False

    def test_no_snapshot_token_skips_gate(self):
        verdict = _check(
            'update_task', live_status='in-progress', snapshot_token=None,
        )
        assert verdict.is_rejection is False

    def test_terminal_gate_takes_precedence_over_stale_snapshot(self):
        """Gate 1 (terminal) is checked before gate 3 (stale snapshot): a
        done task with a stale snapshot reports the more fundamental
        ReconTerminalWriteRejected, not ReconStaleSnapshotRejected."""
        verdict = _check(
            'update_task', live_status='done', snapshot_token='pending',
        )
        assert verdict.error_type == 'ReconTerminalWriteRejected'


# ---------------------------------------------------------------------------
# Regression pin: the sanctioned same-status repair seam the new
# corrective_path/hint redirect to is NOT itself recon-gated
# ---------------------------------------------------------------------------


class TestCorrectivePathSeamIsReachable:
    def test_set_task_status_done_provenance_repair_on_done_task_is_not_gated(
        self, monkeypatch,
    ):
        """Locks the invariant the Gate 1 redirect depends on: a recon-stage
        set_task_status(task_id, 'done', ...) call against an already-done
        task — the same-status done_provenance repair transition
        (task_interceptor._repair_done_provenance_same_status, task 2401)
        that corrective_path/hint now advertise — is not itself blocked by
        this gate. Gate 1 is update_task-only (doesn't fire here); Gate 2
        forces orchestrator_live False for a done task in the real
        detector, mirrored here by monkeypatching it to False; Gate 3 never
        fires because set_task_status always passes snapshot_token=None.
        Guards against a future Gate-2 tightening silently breaking the
        advertised corrective seam."""
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: False,
        )

        verdict = _check(
            'set_task_status',
            target_status='done',
            live_status='done',
            snapshot_token=None,
        )

        assert verdict.is_rejection is False
        assert verdict.corrective_path == ''


# ---------------------------------------------------------------------------
# SNAPSHOT_TOKEN_KEYS / extract_snapshot_token
# ---------------------------------------------------------------------------


class TestExtractSnapshotToken:
    def test_snapshot_token_keys(self):
        assert recon_write_policy.SNAPSHOT_TOKEN_KEYS == ('snapshot_status', 'observed_status')

    def test_extract_from_dict_snapshot_status(self):
        assert recon_write_policy.extract_snapshot_token(
            {'snapshot_status': 'pending'},
        ) == 'pending'

    def test_extract_from_dict_observed_status_alias(self):
        assert recon_write_policy.extract_snapshot_token(
            {'observed_status': 'done'},
        ) == 'done'

    def test_extract_from_json_string(self):
        assert recon_write_policy.extract_snapshot_token(
            '{"snapshot_status":"pending"}',
        ) == 'pending'

    def test_extract_from_none_is_none(self):
        assert recon_write_policy.extract_snapshot_token(None) is None

    def test_extract_from_non_dict_is_none(self):
        assert recon_write_policy.extract_snapshot_token(42) is None

    def test_extract_from_dict_without_either_key_is_none(self):
        assert recon_write_policy.extract_snapshot_token({'other': 'x'}) is None

    def test_extract_from_dict_with_none_value_is_none(self):
        """A None snapshot value must not be coerced to the string 'None',
        which could never equal a real live_status and would spuriously
        trigger ReconStaleSnapshotRejected."""
        assert recon_write_policy.extract_snapshot_token({'snapshot_status': None}) is None

    def test_extract_from_dict_with_int_value_is_none(self):
        assert recon_write_policy.extract_snapshot_token({'snapshot_status': 42}) is None

    def test_extract_from_dict_with_bool_value_is_none(self):
        assert recon_write_policy.extract_snapshot_token({'snapshot_status': True}) is None

    def test_extract_from_dict_with_empty_string_value_is_none(self):
        assert recon_write_policy.extract_snapshot_token({'snapshot_status': ''}) is None


# ---------------------------------------------------------------------------
# CLEARABLE_ANNOTATION_KEYS / is_terminal_annotation_clear
# ---------------------------------------------------------------------------


class TestIsTerminalAnnotationClear:
    def test_clearable_annotation_keys_contains_possible_scope_mismatch(self):
        assert 'possible_scope_mismatch' in recon_write_policy.CLEARABLE_ANNOTATION_KEYS

    # -- positive: exemption applies ---------------------------------------

    def test_clear_to_none_default_mode_is_true(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None}},
        ) is True

    def test_clear_via_json_string_metadata_is_true(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': '{"possible_scope_mismatch": {"matched_paths": ["a"]}}'},
        ) is True

    def test_explicit_merge_mode_is_true(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {
                'metadata': {'possible_scope_mismatch': None},
                'metadata_mode': 'merge',
            },
        ) is True

    def test_overwrite_to_non_null_value_is_true(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {
                'metadata': {
                    'possible_scope_mismatch': {
                        'matched_paths': ['a'],
                        'suggested_project': 'other_project',
                        'source': 'prose',
                    },
                },
            },
        ) is True

    # -- causation-id tracing co-key tolerance (task 2697) -------------------

    def test_clear_plus_causation_id_is_true(self):
        """Symmetric clear-path fix: the mandatory recon-stage _causation_id
        tracing co-key (reconciliation/stages/base.py's Reconciliation
        Context block) must be transparent to the all-clearable check."""
        assert recon_write_policy.is_terminal_annotation_clear(
            {
                'metadata': {
                    'possible_scope_mismatch': None,
                    '_causation_id': 'run-1',
                },
            },
        ) is True

    def test_causation_id_alone_is_false(self):
        """A bare tracing co-key with no real annotation content is a pure
        trace no-op and must stay rejected on a terminal task."""
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'_causation_id': 'run-1'}},
        ) is False

    # -- negative: task-content fields disqualify ---------------------------

    def test_title_present_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None}, 'title': 'x'},
        ) is False

    def test_description_present_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None}, 'description': 'd'},
        ) is False

    def test_details_present_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None}, 'details': 'd'},
        ) is False

    def test_prompt_present_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None}, 'prompt': 'p'},
        ) is False

    def test_priority_present_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None}, 'priority': 'high'},
        ) is False

    def test_dependencies_present_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None}, 'dependencies': [1, 2]},
        ) is False

    # -- negative: unrecognized kwargs fail closed (robustness amendment) ---

    def test_unrecognized_future_kwarg_is_false(self):
        """A hypothetical `update_task` parameter that doesn't exist yet (and
        so cannot be on any denylist) must still disqualify the exemption —
        the allowlist (_ANNOTATION_CLEAR_ALLOWED_KWARGS) fails CLOSED for any
        kwarg it doesn't recognize, unlike a denylist which would silently
        let an unenumerated field through."""
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None}, 'owner': 'alice'},
        ) is False

    def test_unrecognized_kwarg_alone_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'some_new_field': 'x'},
        ) is False

    # -- positive: allowlisted non-content kwargs don't disqualify -----------

    def test_tag_present_is_still_true(self):
        """`tag` selects the tag-scoped row (addressing); it is never itself
        written, so it is not a content mutation."""
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None}, 'tag': 'master'},
        ) is True

    def test_status_present_is_still_true(self):
        """`status` is deliberately excluded from consideration here (same
        rationale as CLEARABLE_ANNOTATION_KEYS): the write-authority floor
        unconditionally rejects a non-None status for every caller before
        any DB write, regardless of this predicate's verdict."""
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None}, 'status': 'done'},
        ) is True

    # -- negative: non-allowlisted metadata keys -----------------------------

    def test_non_allowlisted_key_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'arbitrary': 1}},
        ) is False

    def test_files_key_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'files': ['a/b.py']}},
        ) is False

    def test_done_provenance_key_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'done_provenance': {'kind': 'merged'}}},
        ) is False

    def test_mixed_allowlisted_and_non_allowlisted_keys_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None, 'arbitrary': 1}},
        ) is False

    # -- negative: non-merge modes -------------------------------------------

    def test_metadata_mode_replace_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {
                'metadata': {'possible_scope_mismatch': None},
                'metadata_mode': 'replace',
            },
        ) is False

    def test_append_true_additive_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None}, 'append': True},
        ) is False

    def test_append_false_replace_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear(
            {'metadata': {'possible_scope_mismatch': None}, 'append': False},
        ) is False

    # -- negative: absent / empty / unparseable metadata ---------------------

    def test_metadata_absent_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear({}) is False

    def test_metadata_none_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear({'metadata': None}) is False

    def test_metadata_empty_dict_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear({'metadata': {}}) is False

    def test_metadata_non_dict_int_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear({'metadata': 42}) is False

    def test_metadata_json_list_string_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear({'metadata': '[1,2]'}) is False

    def test_metadata_unparseable_string_is_false(self):
        assert recon_write_policy.is_terminal_annotation_clear({'metadata': 'not json'}) is False


# ---------------------------------------------------------------------------
# is_terminal_annotation_add
# ---------------------------------------------------------------------------


class TestIsTerminalAnnotationAdd:
    # -- positive: exemption applies ---------------------------------------

    def test_single_x_key_default_mode_is_true(self):
        assert recon_write_policy.is_terminal_annotation_add(
            {'metadata': {'x_foo': 1}},
        ) is True

    def test_incident_two_key_payload_is_true(self):
        """The exact incident payload: ADD of x_refile_superseded_by /
        x_reopen_abandoned_reason to done task 1175."""
        assert recon_write_policy.is_terminal_annotation_add(
            {
                'metadata': {
                    'x_refile_superseded_by': 'task-2431',
                    'x_reopen_abandoned_reason': 'superseded',
                },
            },
        ) is True

    def test_x_key_via_json_string_metadata_is_true(self):
        assert recon_write_policy.is_terminal_annotation_add(
            {'metadata': '{"x_foo": {"a": 1}}'},
        ) is True

    def test_explicit_merge_mode_is_true(self):
        assert recon_write_policy.is_terminal_annotation_add(
            {
                'metadata': {'x_foo': 1},
                'metadata_mode': 'merge',
            },
        ) is True

    def test_tag_present_is_still_true(self):
        """`tag` selects the tag-scoped row (addressing); it is never itself
        written, so it is not a content mutation."""
        assert recon_write_policy.is_terminal_annotation_add(
            {'metadata': {'x_foo': 1}, 'tag': 'master'},
        ) is True

    # -- causation-id tracing co-key tolerance (task 2697) -------------------

    def test_x_keys_plus_causation_id_is_true(self):
        """The exact incident payload plus the mandatory recon-stage
        _causation_id tracing co-key (reconciliation/stages/base.py's
        Reconciliation Context block) — _causation_id must be transparent
        to the all-x_ check."""
        assert recon_write_policy.is_terminal_annotation_add(
            {
                'metadata': {
                    'x_refile_superseded_by': '2673',
                    'x_reopen_abandoned_reason': 'superseded',
                    '_causation_id': '06f2658a-474a-476c-9cf3-232d76e9ffb9',
                },
            },
        ) is True

    def test_causation_id_alone_is_false(self):
        """A bare tracing co-key with no real annotation content is a pure
        trace no-op and must stay rejected on a terminal task."""
        assert recon_write_policy.is_terminal_annotation_add(
            {'metadata': {'_causation_id': 'run-1'}},
        ) is False

    def test_x_keys_plus_causation_id_plus_files_is_false(self):
        """Causation-id tracing tolerance must not open the door to other
        load-bearing keys riding alongside it."""
        assert recon_write_policy.is_terminal_annotation_add(
            {
                'metadata': {
                    'x_foo': 1,
                    '_causation_id': 'run-1',
                    'files': ['a/b.py'],
                },
            },
        ) is False

    # -- negative: task-content fields disqualify ---------------------------

    def test_title_present_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add(
            {'metadata': {'x_foo': 1}, 'title': 'x'},
        ) is False

    # -- negative: unrecognized kwargs fail closed (robustness amendment) ---

    def test_unrecognized_future_kwarg_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add(
            {'metadata': {'x_foo': 1}, 'owner': 'alice'},
        ) is False

    # -- negative: non-x_ metadata keys --------------------------------------

    def test_mixed_with_load_bearing_files_key_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add(
            {'metadata': {'x_foo': 1, 'files': ['a/b.py']}},
        ) is False

    def test_mixed_with_done_provenance_key_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add(
            {'metadata': {'x_foo': 1, 'done_provenance': {'kind': 'merged'}}},
        ) is False

    def test_possible_scope_mismatch_alone_is_false(self):
        """Clearable but non-x_ — the add predicate is x_-only. Clearing
        possible_scope_mismatch is is_terminal_annotation_clear's job."""
        assert recon_write_policy.is_terminal_annotation_add(
            {'metadata': {'possible_scope_mismatch': None}},
        ) is False

    def test_arbitrary_non_x_key_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add(
            {'metadata': {'arbitrary': 1}},
        ) is False

    # -- negative: non-merge modes -------------------------------------------

    def test_metadata_mode_replace_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add(
            {
                'metadata': {'x_foo': 1},
                'metadata_mode': 'replace',
            },
        ) is False

    def test_append_true_additive_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add(
            {'metadata': {'x_foo': 1}, 'append': True},
        ) is False

    def test_append_false_replace_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add(
            {'metadata': {'x_foo': 1}, 'append': False},
        ) is False

    # -- negative: absent / empty / unparseable metadata ---------------------

    def test_metadata_absent_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add({}) is False

    def test_metadata_none_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add({'metadata': None}) is False

    def test_metadata_empty_dict_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add({'metadata': {}}) is False

    def test_metadata_non_dict_int_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add({'metadata': 42}) is False

    def test_metadata_json_list_string_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add({'metadata': '[1,2]'}) is False

    def test_metadata_unparseable_string_is_false(self):
        assert recon_write_policy.is_terminal_annotation_add({'metadata': 'not json'}) is False


# ---------------------------------------------------------------------------
# is_terminal_annotation_exempt — single-parse combined form
# (efficiency amendment, task 2695)
# ---------------------------------------------------------------------------


class TestIsTerminalAnnotationExempt:
    """Parity checks against the ``is_terminal_annotation_clear(...) or
    is_terminal_annotation_add(...)`` two-call form it replaces at the
    Gate 1 call site — not a full re-run of every case in
    TestIsTerminalAnnotationClear/TestIsTerminalAnnotationAdd, since this
    function delegates to the exact same :func:`_pure_terminal_annotation_merge`
    + per-key checks those suites already cover."""

    def test_clear_only_payload_is_true(self):
        assert recon_write_policy.is_terminal_annotation_exempt(
            {'metadata': {'possible_scope_mismatch': None}},
        ) is True

    def test_add_only_payload_is_true(self):
        assert recon_write_policy.is_terminal_annotation_exempt(
            {'metadata': {'x_foo': 1}},
        ) is True

    def test_mixed_clearable_and_x_keys_is_false(self):
        """Parity with the x_-only limitation documented on
        is_terminal_annotation_add: neither predicate's ALL-keys condition
        holds for a mixed payload, so the combined form is also False —
        this is the one case a naive "each key is clearable-or-x_" helper
        would get wrong (see review discussion, task 2695 amendment)."""
        assert recon_write_policy.is_terminal_annotation_exempt(
            {'metadata': {'possible_scope_mismatch': None, 'x_foo': 1}},
        ) is False

    def test_neither_clearable_nor_x_is_false(self):
        assert recon_write_policy.is_terminal_annotation_exempt(
            {'metadata': {'arbitrary': 1}},
        ) is False

    def test_disqualifying_kwarg_is_false(self):
        assert recon_write_policy.is_terminal_annotation_exempt(
            {'metadata': {'x_foo': 1}, 'title': 'x'},
        ) is False

    def test_metadata_absent_is_false(self):
        assert recon_write_policy.is_terminal_annotation_exempt({}) is False

    # -- causation-id tracing co-key tolerance (task 2697) -------------------

    def test_add_plus_causation_id_is_true(self):
        assert recon_write_policy.is_terminal_annotation_exempt(
            {'metadata': {'x_foo': 1, '_causation_id': 'run-1'}},
        ) is True

    def test_clear_plus_causation_id_is_true(self):
        assert recon_write_policy.is_terminal_annotation_exempt(
            {
                'metadata': {
                    'possible_scope_mismatch': None,
                    '_causation_id': 'run-1',
                },
            },
        ) is True

    def test_causation_id_alone_is_false(self):
        """A bare tracing co-key with no real annotation content is a pure
        trace no-op and must stay rejected on a terminal task."""
        assert recon_write_policy.is_terminal_annotation_exempt(
            {'metadata': {'_causation_id': 'run-1'}},
        ) is False

    def test_mixed_clearable_and_x_and_causation_id_is_false(self):
        """The x_-only-OR-clear-only limitation survives stripping only the
        tracing key: a mixed clearable+x_ residual still satisfies neither
        ALL-branch."""
        assert recon_write_policy.is_terminal_annotation_exempt(
            {
                'metadata': {
                    'possible_scope_mismatch': None,
                    'x_foo': 1,
                    '_causation_id': 'run-1',
                },
            },
        ) is False

    def test_add_plus_causation_id_via_json_string_metadata_is_true(self):
        """JSON-string metadata (the _coerce_metadata_dict string-coercion
        path recon-stage writes can also arrive via) carrying an x_ key plus
        _causation_id must still be exempt — _annotation_content_keys strips
        tracing co-keys from the already-parsed dict, after string
        coercion, so the stripping applies identically regardless of
        whether metadata arrived as a dict or a JSON string."""
        assert recon_write_policy.is_terminal_annotation_exempt(
            {'metadata': '{"x_foo": 1, "_causation_id": "run-1"}'},
        ) is True


# ---------------------------------------------------------------------------
# X_ANNOTATION_PREFIX <-> shared.task_metadata.parse_metadata consistency
# (robustness amendment, task 2695)
# ---------------------------------------------------------------------------


class TestXAnnotationPrefixConsistencyWithParseMetadata:
    """Guards the hand-maintained literal-mirror documented on
    :data:`recon_write_policy.X_ANNOTATION_PREFIX`.

    ``X_ANNOTATION_PREFIX`` mirrors the bare ``'x_'`` literal in
    ``shared.task_metadata.parse_metadata`` (the forward-compat namespace
    ``parse_metadata`` admits silently, with no schema warning) BY HAND —
    there is no shared constant tying the two together. Nothing else
    enforces they stay in sync, so a future change to either literal (e.g.
    a different prefix, or a case rule) would otherwise silently diverge:
    the recon exemption would then accept a key ``parse_metadata`` warns
    about, or reject one it silently admits. These tests fail loudly
    instead.
    """

    def test_x_annotation_prefix_key_is_admitted_by_parse_metadata_without_warning(self):
        key = f'{recon_write_policy.X_ANNOTATION_PREFIX}consistency_probe'
        _model, warnings = parse_metadata({key: 'v'}, direction='read')
        assert key not in {w.field for w in warnings}

    def test_non_x_prefixed_unknown_key_still_warns(self):
        """Contrast check: an unknown key NOT under X_ANNOTATION_PREFIX still
        trips parse_metadata's unknown_key warning — proving the positive
        assertion above is a meaningful signal (parse_metadata does warn on
        unrecognised keys in general), not a vacuous pass."""
        key = 'not_forward_compat_probe'
        _model, warnings = parse_metadata({key: 'v'}, direction='read')
        assert any(w.field == key and w.code == 'unknown_key' for w in warnings)


# ---------------------------------------------------------------------------
# _CAUSATION_TRACING_KEYS <-> shared.task_metadata consistency (task 2697)
# ---------------------------------------------------------------------------


class TestCausationTracingKeysConsistencyWithBlessedMetadata:
    """Guards :data:`recon_write_policy._CAUSATION_TRACING_KEYS` against
    silently drifting from ``shared.task_metadata._BLESSED_METADATA_KEYS`` —
    a future rename of the blessed causation key would otherwise silently
    break the recon-stage tolerance this task adds (the predicates would
    resume treating ``_causation_id`` as disqualifying load-bearing content)
    without any test failing to say so.
    """

    def test_causation_tracing_keys_is_subset_of_blessed_metadata_keys(self):
        assert recon_write_policy._CAUSATION_TRACING_KEYS <= _BLESSED_METADATA_KEYS

    def test_causation_id_is_admitted_by_parse_metadata_without_warning(self):
        _model, warnings = parse_metadata({'_causation_id': 'v'}, direction='read')
        assert '_causation_id' not in {w.field for w in warnings}


# ---------------------------------------------------------------------------
# Interceptor boundary fixtures (mirrors test_task_write_agent_id.py)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# P1 — interceptor.update_task terminal-write boundary
# ---------------------------------------------------------------------------


class TestInterceptorUpdateTaskTerminalBoundary:
    @pytest.mark.asyncio
    async def test_recon_stage_update_task_on_done_task_rejects(self, interceptor, taskmaster):
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        result = await interceptor.update_task(
            '1', '/project', title='x', agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconTerminalWriteRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_recon_agent_id_is_not_gated(self, interceptor, taskmaster):
        """Recon-scoping negative: a non-recon-stage agent_id on the same
        done task is never gated — the write proceeds normally."""
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        await interceptor.update_task('1', '/project', title='x', agent_id=None)

        taskmaster.update_task.assert_awaited_once()


# ---------------------------------------------------------------------------
# P1b — interceptor.update_task terminal-annotation-clear exemption boundary
# ---------------------------------------------------------------------------


class TestInterceptorUpdateTaskAnnotationClearBoundary:
    @pytest.mark.asyncio
    async def test_recon_stage_clear_of_possible_scope_mismatch_on_done_task_proceeds(
        self, interceptor, taskmaster,
    ):
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        result = await interceptor.update_task(
            '1', '/project',
            metadata={'possible_scope_mismatch': None},
            agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()
        assert 'error_type' not in result

    @pytest.mark.asyncio
    async def test_recon_stage_clear_with_causation_id_on_done_task_proceeds(
        self, interceptor, taskmaster,
    ):
        """Symmetric clear-path fix (task 2697): the mandatory recon-stage
        _causation_id tracing co-key (reconciliation/stages/base.py's
        Reconciliation Context block) riding alongside a clear of
        possible_scope_mismatch must proceed, not be rejected — the
        interceptor-boundary counterpart to
        test_recon_stage_x_annotation_add_with_causation_id_on_done_task_proceeds
        below, for the clear path rather than the add path."""
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        result = await interceptor.update_task(
            '1', '/project',
            metadata={
                'possible_scope_mismatch': None,
                '_causation_id': '06f2658a-474a-476c-9cf3-232d76e9ffb9',
            },
            agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()
        assert 'error_type' not in result

    @pytest.mark.asyncio
    async def test_clear_with_title_present_still_rejects(self, interceptor, taskmaster):
        """A content field alongside the clearable metadata key disqualifies
        the exemption — this is a content mutation, not a pure clear."""
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        result = await interceptor.update_task(
            '1', '/project',
            metadata={'possible_scope_mismatch': None},
            title='x',
            agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconTerminalWriteRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clear_with_non_allowlisted_key_still_rejects(self, interceptor, taskmaster):
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        result = await interceptor.update_task(
            '1', '/project',
            metadata={'arbitrary_key': 1},
            agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconTerminalWriteRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clear_with_replace_mode_still_rejects(self, interceptor, taskmaster):
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        result = await interceptor.update_task(
            '1', '/project',
            metadata={'possible_scope_mismatch': None},
            metadata_mode='replace',
            agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconTerminalWriteRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_recon_agent_id_clear_proceeds_unchanged(self, interceptor, taskmaster):
        """Recon-scoping negative: unaffected by this change either way —
        this write was never gated regardless of the exemption."""
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        await interceptor.update_task(
            '1', '/project',
            metadata={'possible_scope_mismatch': None},
            agent_id=None,
        )

        taskmaster.update_task.assert_awaited_once()


# ---------------------------------------------------------------------------
# P1c — interceptor.update_task terminal-x_-annotation-add exemption boundary
# ---------------------------------------------------------------------------


class TestInterceptorUpdateTaskXAnnotationAddBoundary:
    @pytest.mark.asyncio
    async def test_recon_stage_x_annotation_add_on_done_task_proceeds(
        self, interceptor, taskmaster,
    ):
        """The exact incident payload: ADD of x_refile_superseded_by /
        x_reopen_abandoned_reason to done task 1175."""
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        result = await interceptor.update_task(
            '1', '/project',
            metadata={
                'x_refile_superseded_by': 'task-2431',
                'x_reopen_abandoned_reason': 'superseded',
            },
            agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()
        assert 'error_type' not in result

    @pytest.mark.asyncio
    async def test_recon_stage_x_annotation_add_with_causation_id_on_done_task_proceeds(
        self, interceptor, taskmaster,
    ):
        """Exact incident reproduction (run 06f2658a): the x_ add payload
        plus the mandatory recon-stage _causation_id tracing co-key that
        every recon-stage write embeds (reconciliation/stages/base.py's
        Reconciliation Context block) must proceed, not be rejected."""
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        result = await interceptor.update_task(
            '1', '/project',
            metadata={
                'x_refile_superseded_by': '2673',
                'x_reopen_abandoned_reason': 'superseded',
                '_causation_id': '06f2658a-474a-476c-9cf3-232d76e9ffb9',
            },
            agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()
        assert 'error_type' not in result

    @pytest.mark.asyncio
    async def test_x_add_with_causation_id_and_load_bearing_key_still_rejects(
        self, interceptor, taskmaster,
    ):
        """Causation-id tracing tolerance must not open the door to other
        load-bearing keys riding alongside it."""
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        result = await interceptor.update_task(
            '1', '/project',
            metadata={
                'x_refile_superseded_by': '2673',
                'x_reopen_abandoned_reason': 'superseded',
                '_causation_id': '06f2658a-474a-476c-9cf3-232d76e9ffb9',
                'files': ['a/b.py'],
            },
            agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconTerminalWriteRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_x_add_with_title_present_still_rejects(self, interceptor, taskmaster):
        """A content field alongside the x_ metadata key disqualifies the
        exemption — this is a content mutation, not a pure add."""
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        result = await interceptor.update_task(
            '1', '/project',
            metadata={'x_foo': 1},
            title='x',
            agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconTerminalWriteRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_x_add_mixed_with_load_bearing_metadata_key_still_rejects(
        self, interceptor, taskmaster,
    ):
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        result = await interceptor.update_task(
            '1', '/project',
            metadata={'x_foo': 1, 'files': ['a/b.py']},
            agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconTerminalWriteRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_recon_agent_id_x_add_proceeds_unchanged(self, interceptor, taskmaster):
        """Recon-scoping negative: unaffected by this change either way —
        this write was never gated regardless of the exemption."""
        taskmaster.get_task = AsyncMock(return_value={'id': '1', 'status': 'done', 'title': 'T'})

        await interceptor.update_task(
            '1', '/project',
            metadata={'x_foo': 1},
            agent_id=None,
        )

        taskmaster.update_task.assert_awaited_once()


# ---------------------------------------------------------------------------
# P2 — interceptor.set_task_status live-workflow boundary
# ---------------------------------------------------------------------------


class TestInterceptorSetTaskStatusLiveWorkflowBoundary:
    @pytest.mark.asyncio
    async def test_recon_stage_set_task_status_with_live_workflow_rejects(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """Default taskmaster.get_task fixture returns status='pending'."""
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: True,
        )

        result = await interceptor.set_task_status(
            '1', 'in-progress', '/project', agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconLiveWorkflowWriteRejected'
        taskmaster.set_task_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_recon_agent_id_is_not_gated(self, interceptor, taskmaster, monkeypatch):
        """Recon-scoping negative: a non-recon-stage agent_id is never gated
        even when the detector reports a live workflow."""
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: True,
        )

        await interceptor.set_task_status('1', 'in-progress', '/project', agent_id=None)

        taskmaster.set_task_status.assert_awaited_once()


# ---------------------------------------------------------------------------
# set_task_status forwards the task's metadata into check() (task 3751)
# ---------------------------------------------------------------------------


# dark_factory task 3845's real, verified-first-hand task shape: a PENDING
# `always_escalates` deterministic gate with NO `before_done` key at all. Its
# whole DeterministicRunner run is "file one born-at-L2 escalation, stamp
# gate_escalated_at, set blocked" — no script, no systemd, no git_ops — so it
# never acquires a worktree or branch and the bare project-wide orchestrator
# lock is never task-specific evidence for it. Before this task, Gate 2
# forwarded only `status`, so the lock alone rejected every recon-stage status
# write for it: 3845 stalled 3+ consecutive reconciliation cycles that way.
_GATE_TASK_3845 = {
    'id': '3845',
    'status': 'pending',
    'title': 'Human gate: consolidate duplicate observations cluster',
    'metadata': {
        'task_kind': 'deterministic',
        'execution_class': 'operational',
        'operational_mode': 'gate',
        'always_escalates': True,
    },
}


class TestInterceptorSetTaskStatusForwardsTaskMetadata:
    @staticmethod
    def _spy_check(monkeypatch) -> dict:
        """Wrap the REAL recon_write_policy.check with a kwarg-capturing spy.

        Same idiom as test_set_task_status_always_passes_snapshot_token_none —
        the real gate still runs, so the captured kwargs are exactly what the
        production path passed.
        """
        captured: dict = {}
        real_check = recon_write_policy.check

        def _spy(op, **kwargs):
            captured['op'] = op
            captured.update(kwargs)
            return real_check(op, **kwargs)

        monkeypatch.setattr(recon_write_policy, 'check', _spy)
        return captured

    @staticmethod
    def _spy_detector(monkeypatch) -> dict:
        """Install a kwarg-capturing is_workflow_live_for_task spy returning False."""
        captured: dict = {}

        def _spy(*args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs
            return False

        monkeypatch.setattr(recon_write_policy, 'is_workflow_live_for_task', _spy)
        return captured

    @pytest.mark.asyncio
    async def test_pure_gate_metadata_reaches_the_detector(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """THE PLUMBING — the task's metadata reaches check(), which derives
        task_kind='deterministic' and pure_gate=True for it.

        The metadata is sourced from the `before` dict the interceptor already
        read under its write lock, so it is guaranteed consistent with the
        `live_status` passed alongside it.
        """
        taskmaster.get_task = AsyncMock(return_value=_GATE_TASK_3845)
        check_kwargs = self._spy_check(monkeypatch)
        detector_kwargs = self._spy_detector(monkeypatch)

        await interceptor.set_task_status(
            '3845', 'cancelled', '/project', agent_id=AGENT_ID,
        )

        assert check_kwargs.get('task_metadata') == _GATE_TASK_3845['metadata']
        assert detector_kwargs['kwargs'].get('status') == 'pending'
        assert detector_kwargs['kwargs'].get('task_kind') == 'deterministic'
        assert detector_kwargs['kwargs'].get('pure_gate') is True

    @pytest.mark.asyncio
    async def test_pure_gate_write_is_not_rejected_under_a_live_orchestrator(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """END-TO-END NON-REJECTION — the exact rejection that stalled 3845.

        The REAL is_workflow_live_for_task / detect_live_workflow run; only the
        project-wide orchestrator lock is forced live. The git subprocesses
        against the non-existent '/project' root fail safe to
        no-worktree / no-commit, so the ONLY signal that could reject this
        write is the bare orchestrator lock — which rule 5 now drops.
        """
        from fused_memory.services import live_workflow_detector

        taskmaster.get_task = AsyncMock(return_value=_GATE_TASK_3845)
        monkeypatch.setattr(
            live_workflow_detector, 'is_orchestrator_live_for', lambda *a, **k: True,
        )

        result = await interceptor.set_task_status(
            '3845', 'cancelled', '/project', agent_id=AGENT_ID,
        )

        assert result.get('error_type') != 'ReconLiveWorkflowWriteRejected'
        taskmaster.set_task_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_before_done_deterministic_task_is_still_rejected(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """NEGATIVE CONTROL — the narrowing holds at the real boundary.

        A pending deterministic task WITH `before_done` may be mid-deploy
        inside DeterministicRunner (Harness._run_deterministic_slot never flips
        it to 'in-progress', and a blocking script run leaves no git evidence),
        so recon must still not race it.
        """
        from fused_memory.services import live_workflow_detector

        taskmaster.get_task = AsyncMock(return_value={
            **_GATE_TASK_3845,
            'metadata': {
                'task_kind': 'deterministic',
                'always_escalates': True,
                'before_done': {'kind': 'predicate', 'script': 'x.sh'},
            },
        })
        monkeypatch.setattr(
            live_workflow_detector, 'is_orchestrator_live_for', lambda *a, **k: True,
        )

        result = await interceptor.set_task_status(
            '3845', 'cancelled', '/project', agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconLiveWorkflowWriteRejected'
        taskmaster.set_task_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_task_without_metadata_key_routes_through_unchanged(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """BACKWARD COMPATIBILITY — a task dict with no `metadata` key at all
        forwards None and degrades to today's exact detector inputs."""
        taskmaster.get_task = AsyncMock(
            return_value={'id': '1', 'status': 'pending', 'title': 'T'},
        )
        check_kwargs = self._spy_check(monkeypatch)
        detector_kwargs = self._spy_detector(monkeypatch)

        await interceptor.set_task_status(
            '1', 'in-progress', '/project', agent_id=AGENT_ID,
        )

        assert check_kwargs.get('task_metadata') is None
        assert detector_kwargs['kwargs'].get('task_kind') is None
        assert detector_kwargs['kwargs'].get('pure_gate') is False
        taskmaster.set_task_status.assert_awaited_once()


# ---------------------------------------------------------------------------
# Gate 2: forwarding task_kind is behavior-preserving (task 3751 amendment)
# ---------------------------------------------------------------------------


def _worktree_porcelain_registering(branch: str) -> str:
    """Minimal `git worktree list --porcelain` output that registers *branch*."""
    return f'worktree /tmp/wt\nHEAD abc1234\nbranch refs/heads/{branch}\n\n'


class TestGate2TaskKindForwardingIsBehaviorPreserving:
    """Pins what forwarding `task_kind` into Gate 2 does — and does NOT — change.

    Task 3751 started passing the task's `task_kind` to the detector from this
    gate, which had previously forwarded only `status`. That makes rule 2
    (blocked + deterministic, task 2067) reachable here for the first time.
    It is NOT, however, a widening of the class of tasks a recon-stage agent may
    write status to. Two facts, each pinned below, are why:

    1. Rule 3 (blocked + normal/absent + no git evidence, task 2409) was ALREADY
       reachable at this gate. Its task_kind clause is
       `task_kind in (None, NORMAL_TASK_KIND)`, and the omitted kwarg defaulted
       to None — so a blocked task with no worktree and no recent commit was
       already exempt from the bare orchestrator lock here, before any metadata
       was plumbed through.
    2. Rule 2's only behavioral delta over rule 3 is that it is unconditional on
       the git signals — it also fires when a worktree/recent commit exists. But
       Gate 2 consumes only `is_workflow_live_for_task`, i.e.
       `is_live = worktree_registered or recent_commit or orchestrator_live`, so
       in exactly that case `is_live` stays True on the per-task evidence and
       the write is still rejected. Rule 2 zeroes `orchestrator_live`, which
       Gate 2 never reads on its own.

    Net: the entire Gate-2 behavior change in task 3751 comes from `pure_gate`
    (rule 5), pinned by TestInterceptorSetTaskStatusForwardsTaskMetadata above.
    These tests exist so that claim in check()'s docstring is backed rather than
    asserted, and so a future reader can see the no-widening argument fail loudly
    if a detector rule changes underneath it.
    """

    @staticmethod
    def _force_orchestrator_live(monkeypatch) -> None:
        """Force the project-wide lock live; leave the REAL detector in place."""
        from fused_memory.services import live_workflow_detector

        monkeypatch.setattr(
            live_workflow_detector, 'is_orchestrator_live_for', lambda *a, **k: True,
        )

    @pytest.mark.asyncio
    async def test_blocked_deterministic_bare_write_is_not_rejected(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """RULE 2 at Gate 2 — a blocked deterministic task with no git evidence.

        The git subprocesses against the non-existent '/project' root fail safe
        to no-worktree / no-commit, so the bare orchestrator lock is the only
        signal in play and it is dropped. (Rule 3 produced this same verdict
        before task_kind was forwarded — see
        test_blocked_bare_write_was_already_allowed_before_task_kind_forwarding.)
        """
        taskmaster.get_task = AsyncMock(return_value={
            'id': '2067', 'status': 'blocked', 'title': 'T',
            'metadata': {'task_kind': 'deterministic'},
        })
        self._force_orchestrator_live(monkeypatch)

        result = await interceptor.set_task_status(
            '2067', 'cancelled', '/project', agent_id=AGENT_ID,
        )

        assert result.get('error_type') != 'ReconLiveWorkflowWriteRejected'
        taskmaster.set_task_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_blocked_normal_bare_write_is_not_rejected(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """RULE 3 at Gate 2 — a blocked NORMAL task with no git evidence.

        Unchanged by the plumbing: `task_kind='normal'` and the previous
        implicit None both satisfy rule 3's `task_kind in (None,
        NORMAL_TASK_KIND)` clause.
        """
        taskmaster.get_task = AsyncMock(return_value={
            'id': '2409', 'status': 'blocked', 'title': 'T',
            'metadata': {'task_kind': 'normal'},
        })
        self._force_orchestrator_live(monkeypatch)

        result = await interceptor.set_task_status(
            '2409', 'cancelled', '/project', agent_id=AGENT_ID,
        )

        assert result.get('error_type') != 'ReconLiveWorkflowWriteRejected'
        taskmaster.set_task_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_blocked_bare_write_was_already_allowed_before_task_kind_forwarding(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """THE NO-WIDENING BASELINE — reproduces the pre-task-3751 Gate-2 inputs.

        A task dict with no `metadata` key forwards `task_metadata=None`, from
        which check() derives `task_kind=None` — byte-for-byte the detector
        inputs Gate 2 used before task 3751 plumbed metadata through. It is
        already NOT rejected, which is what shows the blocked-bare allowance
        this gate now grants a `task_kind='normal'` task is pre-existing rule-3
        behavior and not something the plumbing introduced.
        """
        taskmaster.get_task = AsyncMock(
            return_value={'id': '2335', 'status': 'blocked', 'title': 'T'},
        )
        self._force_orchestrator_live(monkeypatch)

        result = await interceptor.set_task_status(
            '2335', 'cancelled', '/project', agent_id=AGENT_ID,
        )

        assert result.get('error_type') != 'ReconLiveWorkflowWriteRejected'
        taskmaster.set_task_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_blocked_deterministic_with_registered_worktree_is_still_rejected(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """THE NO-WIDENING PIN — the one input combination whose
        `orchestrator_live` verdict rule 2 actually changes at this gate.

        With a LIVE worktree registered for the task's branch, forwarding
        `task_kind='deterministic'` makes rule 2 fire and zero
        `orchestrator_live` (rule 3 would not have fired — it is guarded on the
        git signals). Gate 2's verdict is nevertheless UNCHANGED, because
        `is_live` ORs in `worktree_registered`. If this ever starts passing the
        write through, the no-widening argument in check()'s docstring has
        broken and must be re-derived.
        """
        taskmaster.get_task = AsyncMock(return_value={
            'id': '2067', 'status': 'blocked', 'title': 'T',
            'metadata': {'task_kind': 'deterministic'},
        })
        self._force_orchestrator_live(monkeypatch)

        def _git(args, **kwargs):
            return subprocess.CompletedProcess(
                args=args, returncode=0,
                stdout=_worktree_porcelain_registering('task/2067'), stderr='',
            )

        with patch('subprocess.run', side_effect=_git):
            result = await interceptor.set_task_status(
                '2067', 'cancelled', '/project', agent_id=AGENT_ID,
            )

        assert result.get('error_type') == 'ReconLiveWorkflowWriteRejected'
        taskmaster.set_task_status.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_non_blocked_normal_task_is_still_rejected(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """SCOPE PIN — rules 2 and 3 are blocked-only, so an ordinary
        in-progress task under the bare lock is still protected from a
        recon-stage status write. The plumbing does not make the lock
        universally ignorable."""
        taskmaster.get_task = AsyncMock(return_value={
            'id': '999', 'status': 'in-progress', 'title': 'T',
            'metadata': {'task_kind': 'normal'},
        })
        self._force_orchestrator_live(monkeypatch)

        result = await interceptor.set_task_status(
            '999', 'cancelled', '/project', agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconLiveWorkflowWriteRejected'
        taskmaster.set_task_status.assert_not_awaited()


# ---------------------------------------------------------------------------
# set_task_status recon check: offloaded to a thread, snapshot gate unreachable
# ---------------------------------------------------------------------------


class TestInterceptorSetTaskStatusReconCheckOffload:
    @pytest.mark.asyncio
    async def test_recon_check_runs_off_the_event_loop_thread(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """is_workflow_live_for_task shells out to git synchronously; the
        recon-stage check() call in _apply_status_transition must be
        dispatched via asyncio.to_thread so those blocking subprocess calls
        never run on the event-loop thread while _write_lock is held —
        otherwise they would block the loop and stall every other write to
        the project for as long as git takes."""
        seen_threads = []

        def _spy(*args, **kwargs):
            seen_threads.append(threading.current_thread())
            return False

        monkeypatch.setattr(recon_write_policy, 'is_workflow_live_for_task', _spy)

        await interceptor.set_task_status(
            '1', 'in-progress', '/project', agent_id=AGENT_ID,
        )

        assert len(seen_threads) == 1
        assert seen_threads[0] is not threading.main_thread()
        taskmaster.set_task_status.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_set_task_status_always_passes_snapshot_token_none(
        self, interceptor, taskmaster, monkeypatch,
    ):
        """set_task_status has no metadata payload to source a snapshot
        token from, so recon_write_policy.check() is always invoked with
        snapshot_token=None on this path — gate 3 (stale-snapshot) is
        reachable only via update_task in practice, even though check()'s
        own gate 3 is op-agnostic. Pins the intentional None so a future
        reader does not assume this path enforces snapshot freshness too."""
        captured = {}
        real_check = recon_write_policy.check

        def _spy(op, **kwargs):
            captured['op'] = op
            captured.update(kwargs)
            return real_check(op, **kwargs)

        monkeypatch.setattr(recon_write_policy, 'check', _spy)
        monkeypatch.setattr(
            recon_write_policy, 'is_workflow_live_for_task', lambda *a, **k: False,
        )

        await interceptor.set_task_status(
            '1', 'in-progress', '/project', agent_id=AGENT_ID,
        )

        assert captured.get('op') == 'set_task_status'
        assert captured.get('snapshot_token') is None
        taskmaster.set_task_status.assert_awaited_once()


# ---------------------------------------------------------------------------
# P3 — interceptor.update_task stale-snapshot boundary
# ---------------------------------------------------------------------------


class TestInterceptorUpdateTaskStaleSnapshotBoundary:
    @pytest.mark.asyncio
    async def test_recon_stage_update_task_with_stale_snapshot_rejects(
        self, interceptor, taskmaster,
    ):
        """live_status='in-progress' is non-terminal, so gate 1 does not fire."""
        taskmaster.get_task = AsyncMock(
            return_value={'id': '1', 'status': 'in-progress', 'title': 'T'},
        )

        result = await interceptor.update_task(
            '1', '/project', metadata={'snapshot_status': 'pending'}, agent_id=AGENT_ID,
        )

        assert result.get('error_type') == 'ReconStaleSnapshotRejected'
        taskmaster.update_task.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fresh_snapshot_matching_live_status_proceeds(self, interceptor, taskmaster):
        taskmaster.get_task = AsyncMock(
            return_value={'id': '1', 'status': 'in-progress', 'title': 'T'},
        )

        await interceptor.update_task(
            '1', '/project', metadata={'snapshot_status': 'in-progress'}, agent_id=AGENT_ID,
        )

        taskmaster.update_task.assert_awaited_once()
