"""Tests for ``orchestrator.recovery_emission`` — the shared recovery-decision
emission machinery (task 3535, PRD ``plans/task-escalation-state-graph-prd.md``
D5; spec ``docs/task-escalation-state-spec.md`` S6/E12).

The canonical explanation of WHY this mechanism exists — and why it changes no
disposition — lives in the module docstring under test.  This file pins the
observable contract only.
"""

import json

import pytest

# ---------------------------------------------------------------------------
# S3 — the pure payload primitives
# ---------------------------------------------------------------------------


class TestRecoverySiteVocabulary:
    """RecoverySite / LeaveReason are closed StrEnum vocabularies."""

    def test_recovery_site_members_equal_their_plain_string_spellings(self):
        """Genuine ``str`` members, mirroring ``escalation.pins.PinClass``.

        Equality against a plain string must hold WITHOUT an explicit
        ``.value`` so an emitted payload round-trips through JSON and still
        compares equal to the vocabulary.
        """
        from orchestrator.recovery_emission import RecoverySite

        assert RecoverySite.reconcile_sweep == 'reconcile_sweep'
        assert RecoverySite.scheduler_blocked_redispatch == 'scheduler_blocked_redispatch'
        assert RecoverySite.deterministic_recon_sweep == 'deterministic_recon_sweep'
        assert RecoverySite.deterministic_recon_deploy == 'deterministic_recon_deploy'
        assert RecoverySite.already_landed_gate == 'already_landed_gate'
        # The deterministic-recon PAIR must keep DISTINCT labels: collapsing
        # them is task eta's job, and a shared label would hide the very
        # duplication eta needs to measure.
        assert (
            RecoverySite.deterministic_recon_sweep
            != RecoverySite.deterministic_recon_deploy
        )

    def test_leave_reason_vocabulary_is_closed_and_complete(self):
        from orchestrator.recovery_emission import LeaveReason

        assert LeaveReason.escalation_pinned == 'escalation_pinned'
        assert LeaveReason.escalation_store_unavailable == 'escalation_store_unavailable'
        assert LeaveReason.unmapped_shape == 'unmapped_shape'
        assert LeaveReason.live_claimant == 'live_claimant'
        assert LeaveReason.deploy_phase_in_flight == 'deploy_phase_in_flight'
        assert LeaveReason.provenance_arbitration == 'provenance_arbitration'

    def test_both_vocabularies_json_encode_as_plain_strings(self):
        """The event store JSON-encodes ``data`` — a StrEnum must survive it."""
        from orchestrator.recovery_emission import LeaveReason, RecoverySite

        payload = {'site': RecoverySite.reconcile_sweep, 'reason': LeaveReason.unmapped_shape}
        assert json.loads(json.dumps(payload)) == {
            'site': 'reconcile_sweep',
            'reason': 'unmapped_shape',
        }


class TestRenderShape:
    """``render_shape`` mirrors ``task_ground_truth._shape``'s element order."""

    def test_renders_the_five_tuple_in_shape_order(self):
        from shared.task_statuses import TaskStatus

        from orchestrator.recovery_emission import render_shape
        from orchestrator.task_ground_truth import BranchStateKind

        # _RECOVERY row (f): IN_PROGRESS, no claimant, ON_MAIN, open escalation.
        assert render_shape(
            TaskStatus.IN_PROGRESS, False, BranchStateKind.ON_MAIN, True, None,
        ) == 'in-progress|false|on_main|true|-'

    def test_none_deploy_phase_renders_as_a_dash_not_unknown(self):
        """A missing deploy phase is NORMAL, not unknown — most tasks have none."""
        from orchestrator.recovery_emission import render_shape
        from orchestrator.task_ground_truth import BranchStateKind

        rendered = render_shape('blocked', False, BranchStateKind.GONE_NO_MARKER, False, None)
        assert rendered.endswith('|-')

    def test_deploy_phase_enum_renders_by_value(self):
        from orchestrator.deploy_state import DeployPhase
        from orchestrator.recovery_emission import render_shape
        from orchestrator.task_ground_truth import BranchStateKind

        assert render_shape(
            'in-progress', False, BranchStateKind.GONE_NO_MARKER, False, DeployPhase.RAN,
        ) == 'in-progress|false|gone_no_marker|false|ran'

    def test_unknown_elements_render_as_unknown(self):
        """The dispatch gate resolves no claimant and no branch state by design."""
        from orchestrator.recovery_emission import render_shape

        assert render_shape('pending', None, None, True, None) == (
            'pending|unknown|unknown|true|-'
        )

    def test_accepts_plain_strings_as_well_as_enums(self):
        from orchestrator.recovery_emission import render_shape

        assert render_shape(
            'IN-PROGRESS', False, 'ON_MAIN', True, 'RAN',
        ) == 'in-progress|false|on_main|true|ran'

    def test_output_is_stable_and_lowercase(self):
        from shared.task_statuses import TaskStatus

        from orchestrator.recovery_emission import render_shape
        from orchestrator.task_ground_truth import BranchStateKind

        first = render_shape(TaskStatus.BLOCKED, True, BranchStateKind.EXISTS_OFF_MAIN, False, None)
        second = render_shape('blocked', True, 'exists_off_main', False, None)
        assert first == second == first.lower()
        assert first.count('|') == 4


class _Rec:
    """Minimal structural stand-in for EscalationRef / Escalation."""

    def __init__(self, rec_id, created_at):
        self.id = rec_id
        self.created_at = created_at


class TestEscalationAgesSecs:
    """``escalation_ages_secs`` is TOTAL — an id can never silently vanish."""

    NOW = '2026-08-10T12:00:00+00:00'

    def test_returns_a_mapping_keyed_by_escalation_id(self):
        """A MAPPING, not a parallel list: joining by position mis-attributes."""
        from orchestrator.recovery_emission import escalation_ages_secs

        ages = escalation_ages_secs(
            [_Rec('esc-a', '2026-08-10T11:59:00+00:00'), _Rec('esc-b', '2026-08-10T11:00:00+00:00')],
            now=self.NOW,
        )
        assert ages == {'esc-a': 60.0, 'esc-b': 3600.0}

    @pytest.mark.parametrize('created_at', [None, '', '   ', 'not-a-timestamp', 'garbage'])
    def test_unparseable_created_at_maps_to_none_rather_than_dropping_the_id(self, created_at):
        """A hold that cannot be AGED is still a hold.

        Omitting the key would read as "no such record" to a consumer that
        joins ages_secs against escalation_ids.
        """
        from orchestrator.recovery_emission import escalation_ages_secs

        ages = escalation_ages_secs([_Rec('esc-x', created_at)], now=self.NOW)
        assert ages == {'esc-x': None}

    def test_missing_created_at_attribute_maps_to_none(self):
        """Not every record shape carries created_at — legacy refs do not."""
        from orchestrator.recovery_emission import escalation_ages_secs

        class _NoField:
            id = 'esc-legacy'

        assert escalation_ages_secs([_NoField()], now=self.NOW) == {'esc-legacy': None}

    def test_naive_timestamp_is_assumed_utc(self):
        """Matches the harness.py:4605-4610 legacy-naive-value precedent."""
        from orchestrator.recovery_emission import escalation_ages_secs

        assert escalation_ages_secs(
            [_Rec('esc-naive', '2026-08-10T11:30:00')], now=self.NOW,
        ) == {'esc-naive': 1800.0}

    def test_future_timestamp_clamps_to_zero_never_negative(self):
        """A clock skew must not produce a negative age in the payload."""
        from orchestrator.recovery_emission import escalation_ages_secs

        assert escalation_ages_secs(
            [_Rec('esc-future', '2026-08-10T13:00:00+00:00')], now=self.NOW,
        ) == {'esc-future': 0.0}

    def test_empty_and_none_inputs_return_an_empty_mapping(self):
        from orchestrator.recovery_emission import escalation_ages_secs

        assert escalation_ages_secs([], now=self.NOW) == {}
        assert escalation_ages_secs(None, now=self.NOW) == {}

    def test_never_raises_on_a_wholly_corrupt_record(self):
        """Telemetry must not be able to abort a sweep."""
        from orchestrator.recovery_emission import escalation_ages_secs

        class _Exploding:
            @property
            def id(self):
                raise RuntimeError('boom')

        assert escalation_ages_secs([_Exploding()], now=self.NOW) == {}


class TestBuildRecoveryPayload:
    """The payload's key set is the closed contract EventType documents."""

    KEYS = {
        'task_id', 'site', 'shape', 'reason', 'escalation_ids', 'ages_secs',
        'measured_at', 'store_unavailable', 'streak',
    }

    def _payload(self, **overrides):
        from orchestrator.recovery_emission import (
            LeaveReason,
            RecoverySite,
            build_recovery_payload,
        )

        kwargs = {
            'task_id': '3535',
            'site': RecoverySite.reconcile_sweep,
            'shape': 'in-progress|false|on_main|true|-',
            'reason': LeaveReason.escalation_pinned,
            'escalation_ids': {'queue_handoff': ['esc-b', 'esc-a']},
            'ages_secs': {'esc-a': 60.0, 'esc-b': None},
            'store_unavailable': False,
            'streak': 1,
            'now': '2026-08-10T12:00:00+00:00',
        }
        kwargs.update(overrides)
        return build_recovery_payload(**kwargs)

    def test_returns_exactly_the_documented_key_set(self):
        assert set(self._payload()) == self.KEYS

    def test_is_json_round_trippable(self):
        """EventStore JSON-encodes ``data``; an unserialisable value drops the row."""
        payload = self._payload()
        assert json.loads(json.dumps(payload)) == json.loads(json.dumps(payload))
        decoded = json.loads(json.dumps(payload))
        assert decoded['site'] == 'reconcile_sweep'
        assert decoded['reason'] == 'escalation_pinned'

    def test_escalation_ids_are_sorted_deterministically(self):
        """So two identical vetoes produce two identical signatures."""
        payload = self._payload()
        assert payload['escalation_ids']['queue_handoff'] == ['esc-a', 'esc-b']

    def test_measured_at_is_an_iso8601_utc_string_from_the_injected_clock(self):
        from datetime import datetime

        payload = self._payload(now='2026-08-10T12:00:00+00:00')
        assert payload['measured_at'] == '2026-08-10T12:00:00+00:00'
        # Parses back as tz-aware UTC.
        parsed = datetime.fromisoformat(payload['measured_at'])
        assert parsed.tzinfo is not None

    def test_ages_secs_survives_a_none_age(self):
        payload = self._payload()
        assert payload['ages_secs'] == {'esc-a': 60.0, 'esc-b': None}

    def test_store_unavailable_and_streak_travel_verbatim(self):
        payload = self._payload(store_unavailable=True, streak=4)
        assert payload['store_unavailable'] is True
        assert payload['streak'] == 4


class _RecordingStore:
    def __init__(self):
        self.calls = []

    def emit(self, event_type, **kwargs):
        self.calls.append((event_type, kwargs))


class TestEmitRecoveryEvent:
    """Fire-and-forget: telemetry can never disturb a sweep."""

    def _emit(self, store, **overrides):
        from orchestrator.event_store import EventType
        from orchestrator.recovery_emission import emit_recovery_event

        kwargs = {
            'event_store': store,
            'event_type': EventType.recovery_vetoed,
            'task_id': '3535',
            'payload': {'site': 'reconcile_sweep'},
        }
        kwargs.update(overrides)
        return emit_recovery_event(**kwargs)

    def test_none_event_store_is_a_silent_no_op(self):
        assert self._emit(None) is False

    def test_emits_task_id_as_a_first_class_column(self):
        """Not only inside ``data`` — the rows must stay joinable."""
        from orchestrator.event_store import EventType

        store = _RecordingStore()
        assert self._emit(store) is True
        assert len(store.calls) == 1
        event_type, kwargs = store.calls[0]
        assert event_type is EventType.recovery_vetoed
        assert kwargs['task_id'] == '3535'
        assert kwargs['data'] == {'site': 'reconcile_sweep'}

    def test_a_raising_event_store_is_swallowed_and_logged(self, caplog):
        import logging

        class _Exploding:
            def emit(self, *_args, **_kwargs):
                raise RuntimeError('disk full')

        with caplog.at_level(logging.WARNING, logger='orchestrator.recovery_emission'):
            assert self._emit(_Exploding()) is False
        assert any('recovery' in r.getMessage().lower() for r in caplog.records)

    def test_a_none_task_id_is_accepted_for_phase_scoped_emissions(self):
        """The scheduler's queue-absent emission is process-scoped, not per-task."""
        store = _RecordingStore()
        assert self._emit(store, task_id=None) is True
        assert store.calls[0][1]['task_id'] is None


# ---------------------------------------------------------------------------
# S5 — the streak tracker and the emission-cadence predicate
# ---------------------------------------------------------------------------


def _tracker(clock=None):
    from orchestrator.recovery_emission import RecoveryVetoStreakTracker

    if clock is None:
        return RecoveryVetoStreakTracker()
    return RecoveryVetoStreakTracker(clock=clock)


class TestRecoveryVetoStreakTracker:
    """Per-(site, task) consecutive-identical-veto counting, in memory."""

    def test_first_observation_starts_a_streak_and_reports_changed(self):
        from orchestrator.recovery_emission import RecoverySite

        tracker = _tracker()
        obs = tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        assert obs.streak == 1
        assert obs.changed is True

    def test_identical_signature_accumulates_quietly(self):
        """A quiet repeat is the same hold observed again, not a new fact."""
        from orchestrator.recovery_emission import RecoverySite

        tracker = _tracker()
        first = tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        second = tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        third = tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        assert (first.streak, second.streak, third.streak) == (1, 2, 3)
        assert second.changed is False
        assert third.changed is False

    def test_a_changed_signature_restarts_the_streak(self):
        """So an L1 only ever describes N genuinely IDENTICAL consecutive vetoes."""
        from orchestrator.recovery_emission import RecoverySite

        tracker = _tracker()
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        changed = tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1,esc-2')
        assert changed.streak == 1
        assert changed.changed is True

    def test_sites_keep_independent_streaks_for_one_task(self):
        """A per-TICK site must never inflate a per-SWEEP site's count."""
        from orchestrator.recovery_emission import RecoverySite

        tracker = _tracker()
        for _ in range(5):
            tracker.observe(RecoverySite.already_landed_gate, '3535', 'pinned|esc-1')
        sweep = tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        assert sweep.streak == 1, 'the tick site must not have charged the sweep site'
        assert tracker.streak(RecoverySite.already_landed_gate, '3535') == 5

    def test_tasks_keep_independent_streaks_at_one_site(self):
        from orchestrator.recovery_emission import RecoverySite

        tracker = _tracker()
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        other = tracker.observe(RecoverySite.reconcile_sweep, '3536', 'pinned|esc-9')
        assert other.streak == 1
        assert tracker.streak(RecoverySite.reconcile_sweep, '3535') == 2

    def test_streak_is_zero_for_an_unknown_key(self):
        from orchestrator.recovery_emission import RecoverySite

        assert _tracker().streak(RecoverySite.reconcile_sweep, 'never-seen') == 0

    def test_clear_pops_the_entry_so_the_footprint_stays_bounded(self):
        """Pop-on-reset, not set-to-zero.

        This tracker lives for the whole life of a multi-week orchestrator
        process, and the reconcile sweep visits EVERY in-progress and blocked
        task.  A dict that grew one entry per task ever swept would be a slow
        leak; it must stay proportional to the tasks CURRENTLY held.
        """
        from orchestrator.recovery_emission import RecoverySite

        tracker = _tracker()
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        tracker.observe(RecoverySite.reconcile_sweep, '3536', 'pinned|esc-2')
        assert set(tracker.tracked()) == {
            (RecoverySite.reconcile_sweep, '3535'),
            (RecoverySite.reconcile_sweep, '3536'),
        }
        tracker.clear(RecoverySite.reconcile_sweep, '3535')
        assert set(tracker.tracked()) == {(RecoverySite.reconcile_sweep, '3536')}
        assert tracker.streak(RecoverySite.reconcile_sweep, '3535') == 0

    def test_clear_of_an_unknown_key_is_a_no_op(self):
        """Every swept task calls this; an unheld one must not raise."""
        from orchestrator.recovery_emission import RecoverySite

        tracker = _tracker()
        tracker.clear(RecoverySite.reconcile_sweep, 'never-seen')
        assert tuple(tracker.tracked()) == ()

    def test_clear_returns_the_streak_it_dropped(self):
        """The resolve half needs the pre-reset streak to describe what ended."""
        from orchestrator.recovery_emission import RecoverySite

        tracker = _tracker()
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        assert tracker.clear(RecoverySite.reconcile_sweep, '3535') == 2
        assert tracker.clear(RecoverySite.reconcile_sweep, '3535') == 0


class TestRecoveryVetoStreakTrackerSpan:
    """The second half of the two-dimensional alarm predicate."""

    def _seq_tracker(self, readings):
        seq = iter(readings)
        return _tracker(clock=lambda: next(seq))

    def test_span_measures_from_the_first_observation_of_the_streak(self):
        from orchestrator.recovery_emission import RecoverySite

        tracker = self._seq_tracker([100.0, 400.0, 1900.0])
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')  # t=100
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')  # t=400
        assert tracker.span(RecoverySite.reconcile_sweep, '3535') == 1800.0  # t=1900

    def test_span_is_zero_for_an_unknown_key(self):
        from orchestrator.recovery_emission import RecoverySite

        tracker = self._seq_tracker([100.0])
        assert tracker.span(RecoverySite.reconcile_sweep, 'never-seen') == 0.0

    def test_span_is_never_negative(self):
        """A non-monotonic clock injected by a test must not produce nonsense."""
        from orchestrator.recovery_emission import RecoverySite

        tracker = self._seq_tracker([1000.0, 10.0])
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        assert tracker.span(RecoverySite.reconcile_sweep, '3535') == 0.0

    def test_span_restarts_with_the_streak_on_a_changed_signature(self):
        from orchestrator.recovery_emission import RecoverySite

        tracker = self._seq_tracker([100.0, 2000.0, 2100.0])
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')   # t=100
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-2')   # t=2000, reset
        assert tracker.span(RecoverySite.reconcile_sweep, '3535') == 100.0      # t=2100

    def test_span_is_dropped_with_the_streak_on_clear(self):
        from orchestrator.recovery_emission import RecoverySite

        tracker = self._seq_tracker([100.0, 5000.0])
        tracker.observe(RecoverySite.reconcile_sweep, '3535', 'pinned|esc-1')
        tracker.clear(RecoverySite.reconcile_sweep, '3535')
        assert tracker.span(RecoverySite.reconcile_sweep, '3535') == 0.0

    def test_default_clock_is_monotonic(self):
        """Wall clock would let an NTP step or DST jump corrupt a span."""
        import time

        assert _tracker()._clock is time.monotonic


class TestShouldEmitEvent:
    """The emission-cadence predicate — signature-transition-gated (INV-4)."""

    def _obs(self, *, streak, changed):
        from orchestrator.recovery_emission import Observation, RecoverySite

        return Observation(
            site=RecoverySite.reconcile_sweep,
            task_id='3535',
            signature='pinned|esc-1',
            streak=streak,
            changed=changed,
        )

    def test_emits_on_a_new_or_changed_signature(self):
        from orchestrator.recovery_emission import should_emit_event

        assert should_emit_event(self._obs(streak=1, changed=True), threshold=3) is True

    def test_stays_quiet_on_a_repeat_below_the_threshold(self):
        """Two per-TICK sites call this; one row per tick would storm the store."""
        from orchestrator.recovery_emission import should_emit_event

        assert should_emit_event(self._obs(streak=2, changed=False), threshold=3) is False

    def test_emits_exactly_once_at_the_threshold_crossing(self):
        """The escalation moment must appear in the event store."""
        from orchestrator.recovery_emission import should_emit_event

        assert should_emit_event(self._obs(streak=3, changed=False), threshold=3) is True

    def test_stays_quiet_forever_after_the_crossing(self):
        """A long-lived veto must not re-emit on every later observation."""
        from orchestrator.recovery_emission import should_emit_event

        for streak in (4, 5, 50, 5000):
            assert should_emit_event(
                self._obs(streak=streak, changed=False), threshold=3,
            ) is False, f'streak={streak} must not re-emit'

    def test_a_changed_signature_above_the_threshold_still_emits(self):
        """`changed` is the primary clause — a NEW fact is always worth a row."""
        from orchestrator.recovery_emission import should_emit_event

        assert should_emit_event(self._obs(streak=1, changed=True), threshold=1) is True
