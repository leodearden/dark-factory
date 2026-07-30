"""Task 3068 / Part C — the zero-progress-requeue backstop.

Origin incident (reify esc-5556-1): ~24 tasks requeued ~349 times over 46h
without invoking a single agent, and NOTHING alarmed.  That is by design, and
that is the bug: ``workflow_types._disposition_table()`` sets
``counts_against_requeue_cap=False`` for the warm-lane dispositions, so
``Scheduler.record_requeue(counts_against_cap=False)`` is history-only —
neither ``_requeue_counts`` nor ``_transient_requeue_counts`` moves, so neither
ceiling in ``Harness._apply_retry_cap`` can ever trip.  A hard fault
masquerading as transient backpressure requeues forever, invisibly.

This module's detector is the only backstop for that class, so these tests pin
its shape tightly: a per-task CONSECUTIVE streak of zero-agent-invocation
requeues, reset by any other outcome, alarming once per unresolved incident.
"""

from __future__ import annotations

import pytest

from orchestrator.workflow import WorkflowOutcome

# ---------------------------------------------------------------------------
# Part C.1 — the pure streak tracker
# ---------------------------------------------------------------------------


class TestZeroProgressRequeueTracker:
    """Per-task consecutive zero-progress requeue counting, no I/O."""

    def _tracker(self):
        from orchestrator.zero_progress_requeue import ZeroProgressRequeueTracker

        return ZeroProgressRequeueTracker()

    def test_consecutive_zero_progress_requeues_accumulate(self):
        """Three zero-invocation requeues in a row return 1, 2, 3."""
        tracker = self._tracker()
        streaks = [
            tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
            for _ in range(3)
        ]
        assert streaks == [1, 2, 3]
        assert tracker.streak('3068') == 3

    def test_requeue_with_real_work_resets(self):
        """A requeue that DID invoke an agent is progress — reset the streak.

        This is the false-positive guard: genuine slow progress (a task that
        requeues after doing real work) must never accumulate toward an alarm.
        """
        tracker = self._tracker()
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)

        assert tracker.record(
            '3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=1
        ) == 0
        assert tracker.streak('3068') == 0

        assert tracker.record(
            '3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0
        ) == 1

    def test_done_resets(self):
        """A DONE outcome clears the streak."""
        tracker = self._tracker()
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)

        assert tracker.record(
            '3068', outcome=WorkflowOutcome.DONE, agent_invocations=0
        ) == 0
        assert tracker.streak('3068') == 0

    def test_blocked_resets(self):
        """A BLOCKED outcome clears the streak — it is a terminal, visible state."""
        tracker = self._tracker()
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)

        assert tracker.record(
            '3068', outcome=WorkflowOutcome.BLOCKED, agent_invocations=0
        ) == 0
        assert tracker.streak('3068') == 0

    def test_streaks_are_per_task(self):
        """Two tasks keep independent streaks and do not interfere."""
        tracker = self._tracker()
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        tracker.record('4000', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)

        assert tracker.streak('3068') == 2
        assert tracker.streak('4000') == 1

        # Resetting one must not touch the other.
        tracker.record('4000', outcome=WorkflowOutcome.DONE, agent_invocations=0)
        assert tracker.streak('3068') == 2
        assert tracker.streak('4000') == 0

    def test_streak_of_unknown_task_is_zero(self):
        """Reading a never-seen task is 0, not a KeyError."""
        assert self._tracker().streak('never-seen') == 0

    def test_reset_pops_the_key_rather_than_storing_zero(self):
        """A reset must POP, not store 0.

        The orchestrator runs for weeks; storing a zero per task would leak one
        dict entry per task forever.  Popping keeps the tracker's footprint
        proportional to the number of tasks CURRENTLY looping, which for a
        healthy fleet is zero.
        """
        tracker = self._tracker()
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        tracker.record('4000', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        assert set(tracker.tracked_task_ids()) == {'3068', '4000'}

        tracker.record('3068', outcome=WorkflowOutcome.DONE, agent_invocations=0)
        assert set(tracker.tracked_task_ids()) == {'4000'}
        assert tracker.streak('3068') == 0

        # A DONE for a task that was never tracked must also not create an entry.
        tracker.record('9999', outcome=WorkflowOutcome.DONE, agent_invocations=0)
        assert set(tracker.tracked_task_ids()) == {'4000'}

    @pytest.mark.parametrize(
        'outcome',
        [
            WorkflowOutcome.DONE,
            WorkflowOutcome.PLANNED,
            WorkflowOutcome.BLOCKED,
            WorkflowOutcome.ESCALATED,
            WorkflowOutcome.CANCELLED,
            WorkflowOutcome.MERGE_DEFERRED,
            WorkflowOutcome.SOFT_CANCELLED,
        ],
    )
    def test_every_non_requeue_outcome_resets(self, outcome):
        """Only REQUEUED can ever accumulate — every other outcome resets."""
        tracker = self._tracker()
        tracker.record('3068', outcome=WorkflowOutcome.REQUEUED, agent_invocations=0)
        assert tracker.record('3068', outcome=outcome, agent_invocations=0) == 0
        assert tracker.streak('3068') == 0


# ---------------------------------------------------------------------------
# Part C.2 — the fail-open alert emitter
# ---------------------------------------------------------------------------


def _emit(**kwargs):
    """Call the alert emitter, importing at call time so a missing symbol
    fails only these tests rather than breaking module collection."""
    from orchestrator.zero_progress_requeue import emit_zero_progress_requeue_alert

    return emit_zero_progress_requeue_alert(**kwargs)


def _sentinel(task_id: str) -> str:
    from orchestrator.zero_progress_requeue import ZERO_PROGRESS_SENTINEL_PREFIX

    return f'{ZERO_PROGRESS_SENTINEL_PREFIX}{task_id}'


def _default_kwargs(queue, rec, *, task_id='3068', streak=5, threshold=5):
    return {
        'escalation_queue': queue,
        'event_store': rec,
        'task_id': task_id,
        'streak': streak,
        'threshold': threshold,
        'block_reason': 'warm_lane_pool_hard_down',
        'block_phase': 'plan',
    }


class TestEmitZeroProgressRequeueAlert:
    """One blocking L1 per unresolved incident; never raises."""

    def test_fires_at_threshold_with_monitor_alarm_shape(self, tmp_path):
        """streak >= threshold files exactly one L1 with the alarm shape."""
        from _recording_event_store import _RecordingEventStore
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path)
        rec = _RecordingEventStore()

        assert _emit(**_default_kwargs(queue, rec)) is True

        filed = queue.get_by_task(_sentinel('3068'), status='pending')
        assert len(filed) == 1, f'expected exactly one escalation, got {filed}'
        esc = filed[0]
        assert esc.task_id == _sentinel('3068')
        assert esc.agent_role == 'orchestrator-zero-progress-requeue'
        # `blocking` is the ceiling an agent-authored filer may use, and level 1
        # routes to the auto-watcher rather than a steward that has no real task
        # to work (the sentinel id is synthetic).
        assert esc.severity == 'blocking'
        assert esc.level == 1
        assert esc.category == 'risk_identified'
        # The alert must name the REAL task and the streak, not just the sentinel.
        assert '3068' in esc.summary
        assert '5' in esc.summary
        # ...and carry the WHY that parts A/B make durable, so the operator does
        # not have to go digging in a rotated journald log.
        assert 'warm_lane_pool_hard_down' in esc.detail
        assert 'plan' in esc.detail

    def test_below_threshold_files_nothing(self, tmp_path):
        """streak < threshold is a no-op returning False."""
        from _recording_event_store import _RecordingEventStore
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path)
        rec = _RecordingEventStore()

        assert _emit(**_default_kwargs(queue, rec, streak=4, threshold=5)) is False
        assert queue.get_by_task(_sentinel('3068'), status='pending') == []
        assert rec.events == []

    def test_dedups_while_first_is_still_pending(self, tmp_path):
        """A second call must not stack a duplicate.

        Not clearing the streak on fire would otherwise produce a sawtooth that
        re-files every N requeues — spamming the queue during exactly the 46h
        loop this exists to catch.
        """
        from _recording_event_store import _RecordingEventStore
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path)
        rec = _RecordingEventStore()

        assert _emit(**_default_kwargs(queue, rec, streak=5)) is True
        assert _emit(**_default_kwargs(queue, rec, streak=6)) is False
        assert _emit(**_default_kwargs(queue, rec, streak=7)) is False

        assert len(queue.get_by_task(_sentinel('3068'), status='pending')) == 1

    def test_distinct_tasks_get_distinct_alerts(self, tmp_path):
        """Dedup is per-task — one looping task must not mask another."""
        from _recording_event_store import _RecordingEventStore
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path)
        rec = _RecordingEventStore()

        assert _emit(**_default_kwargs(queue, rec, task_id='3068')) is True
        assert _emit(**_default_kwargs(queue, rec, task_id='4000')) is True

        assert len(queue.get_by_task(_sentinel('3068'), status='pending')) == 1
        assert len(queue.get_by_task(_sentinel('4000'), status='pending')) == 1

    def test_no_queue_is_a_silent_no_op(self, tmp_path):
        """escalation_queue=None returns False without raising."""
        from _recording_event_store import _RecordingEventStore

        rec = _RecordingEventStore()
        assert _emit(**_default_kwargs(None, rec)) is False
        assert rec.events == []

    def test_raising_queue_does_not_propagate(self, tmp_path):
        """A submit() that raises must not escape the emitter.

        This module sits on the dispatch hot path; a bug or a full disk here
        must never disturb requeue-cap accounting.
        """
        from unittest.mock import MagicMock

        from _recording_event_store import _RecordingEventStore

        queue = MagicMock()
        queue.has_open_l1.return_value = False
        queue.make_id.return_value = 'esc-x-1'
        queue.submit.side_effect = RuntimeError('disk full')

        assert _emit(**_default_kwargs(queue, _RecordingEventStore())) is False

    def test_raising_event_store_does_not_unfile_the_escalation(self, tmp_path):
        """An event-emit failure must not un-file the escalation.

        The escalation is the load-bearing output; the event is telemetry.
        They are submitted under INDIVIDUAL try/excepts so a failure of the
        second cannot cost us the first.
        """
        from unittest.mock import MagicMock

        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path)
        broken = MagicMock()
        broken.emit.side_effect = RuntimeError('event store wedged')

        result = _emit(**_default_kwargs(queue, broken))

        assert len(queue.get_by_task(_sentinel('3068'), status='pending')) == 1
        assert result is True, 'the escalation WAS filed; a telemetry failure must not lie'

    def test_emits_zero_progress_requeue_event(self, tmp_path):
        """A paired EventType.zero_progress_requeue carries the payload."""
        from _recording_event_store import _RecordingEventStore
        from escalation.queue import EscalationQueue

        from orchestrator.event_store import EventType

        queue = EscalationQueue(tmp_path)
        rec = _RecordingEventStore()

        _emit(**_default_kwargs(queue, rec))

        events = [
            entry for (etype, entry) in rec.events
            if etype == EventType.zero_progress_requeue
        ]
        assert len(events) == 1, f'expected one zero_progress_requeue event, got {rec.events}'
        entry = events[0]
        assert entry['task_id'] == '3068', 'the event must key on the REAL task id'
        data = entry['data']
        assert data['streak'] == 5
        assert data['threshold'] == 5
        assert data['reason'] == 'warm_lane_pool_hard_down'
        assert data['block_phase'] == 'plan'

    def test_no_event_store_still_files_the_escalation(self, tmp_path):
        """event_store=None is fine — telemetry is optional, the alarm is not."""
        from escalation.queue import EscalationQueue

        queue = EscalationQueue(tmp_path)
        assert _emit(**_default_kwargs(queue, None)) is True
        assert len(queue.get_by_task(_sentinel('3068'), status='pending')) == 1


# ---------------------------------------------------------------------------
# Part C.3 — the config section
# ---------------------------------------------------------------------------


def _load_package_defaults() -> dict:
    """Read the shipped defaults.yaml (test_config_merge_deep.py:63-69 idiom)."""
    import importlib.resources as pkg_resources

    import yaml

    defaults_file = pkg_resources.files('orchestrator') / 'defaults.yaml'
    return yaml.safe_load(defaults_file.read_text())


class TestZeroProgressRequeueConfig:
    """The detector must be operator-visible, tunable, and hot-reloadable."""

    def test_section_is_registered_on_orchestrator_config(self):
        """OrchestratorConfig exposes zero_progress_requeue with the defaults.

        Registration is load-bearing, not cosmetic: model_config sets
        extra='ignore', so an UNREGISTERED yaml block would be silently
        dropped — the operator would edit a stanza that does nothing.  The
        field declaration is also what registers the key with
        census_config_keys.
        """
        from orchestrator.config import OrchestratorConfig

        cfg = OrchestratorConfig()
        assert hasattr(cfg, 'zero_progress_requeue'), (
            'zero_progress_requeue must be a registered field, or its yaml '
            'block is silently dropped by extra="ignore"'
        )
        # Unlike chronic_flake (shipped off, pending an un-landed reify
        # substrate), this detector reads only TaskReport fields that already
        # exist — shipping it off would leave the gap open indefinitely.
        assert cfg.zero_progress_requeue.enabled is True
        assert cfg.zero_progress_requeue.threshold == 5

    def test_threshold_must_be_at_least_one(self):
        """threshold=0 is rejected — it would alarm on every single requeue."""
        import pydantic

        from orchestrator.config import ZeroProgressRequeueConfig

        with pytest.raises(pydantic.ValidationError):
            ZeroProgressRequeueConfig(threshold=0)
        with pytest.raises(pydantic.ValidationError):
            ZeroProgressRequeueConfig(threshold=-1)

    def test_defaults_yaml_block_matches_the_field_defaults(self):
        """The shipped stanza must not drift from the pydantic defaults.

        YamlSettingsSource layers defaults.yaml UNDER the project YAML, so the
        STANZA — not the Field(default=...) — is what actually applies.  A
        drifted stanza would silently win.
        """
        defaults = _load_package_defaults()
        assert 'zero_progress_requeue' in defaults, (
            'the shipped defaults.yaml must declare the zero_progress_requeue: '
            'block so the knob is discoverable without reading pydantic source'
        )
        block = defaults['zero_progress_requeue']
        assert block['enabled'] is True
        assert block['threshold'] == 5

    def test_leaves_are_green_tier_hot_reloadable(self):
        """Both leaves are in RELOADABLE_FIELDS.

        An operator must be able to retune or silence a noisy detector live,
        without a restart — a detector you can only silence by restarting the
        fleet is one that gets silenced by ignoring it.
        """
        from orchestrator.config import RELOADABLE_FIELDS

        assert 'zero_progress_requeue.enabled' in RELOADABLE_FIELDS
        assert 'zero_progress_requeue.threshold' in RELOADABLE_FIELDS


# ---------------------------------------------------------------------------
# Part C.4 — harness wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def wired_harness(tmp_path, mock_orch_config):
    """Harness driven through ``_apply_retry_cap`` directly.

    ``_run_id`` is set so the method's existing ``if self._run_id is None:
    return requeued`` guard (relied on as a test no-op hook elsewhere) does not
    short-circuit, and the mocked scheduler returns a low requeue count so the
    pre-existing cap never trips and cannot be mistaken for our alert.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from _recording_event_store import _RecordingEventStore
    from escalation.queue import EscalationQueue

    from orchestrator.harness import Harness

    with (
        patch('orchestrator.harness.McpLifecycle'),
        patch('orchestrator.harness.OverrideStore'),
        patch('orchestrator.harness.Scheduler'),
        patch('orchestrator.harness.BriefingAssembler'),
    ):
        h = Harness(mock_orch_config)

    h._run_id = 'run-test'
    h._escalation_queue = EscalationQueue(tmp_path)
    h.event_store = _RecordingEventStore()  # type: ignore[assignment]

    h.scheduler = MagicMock()
    h.scheduler.record_requeue = MagicMock(return_value=1)
    h.scheduler.transient_requeue_count = MagicMock(return_value=0)
    h.scheduler.clear_requeue_count = MagicMock()
    h.scheduler.requeue_history = MagicMock(return_value=[])
    h.scheduler.trigger_retry_cap_exhausted = AsyncMock()

    h.config.requeue_cap = 99
    h.config.transient_requeue_cap = 99
    h.config.zero_progress_requeue.enabled = True
    h.config.zero_progress_requeue.threshold = 3
    return h


def _report(
    outcome=None,
    *,
    agent_invocations: int = 0,
    counts_against_requeue_cap: bool = False,
    task_id: str = '3068',
):
    from orchestrator.harness import TaskReport

    if outcome is None:
        outcome = WorkflowOutcome.REQUEUED
    return TaskReport(
        task_id=task_id,
        title='Looping task',
        outcome=outcome,
        agent_invocations=agent_invocations,
        block_reason='warm_lane_pool_hard_down' if outcome is WorkflowOutcome.REQUEUED else '',
        block_phase='plan' if outcome is WorkflowOutcome.REQUEUED else '',
        counts_against_requeue_cap=counts_against_requeue_cap,
    )


def _alerts(harness, task_id: str = '3068'):
    return harness._escalation_queue.get_by_task(_sentinel(task_id), status='pending')


@pytest.mark.asyncio
class TestHarnessZeroProgressWiring:
    """``_apply_retry_cap`` is the chokepoint that feeds the tracker."""

    async def test_fires_exactly_at_threshold(self, wired_harness):
        """Two zero-progress requeues alarm nothing; the third fires once."""
        for _ in range(2):
            await wired_harness._apply_retry_cap('3068', _report(), True)
        assert _alerts(wired_harness) == [], 'fired below threshold'

        await wired_harness._apply_retry_cap('3068', _report(), True)
        assert len(_alerts(wired_harness)) == 1

        # And keeps deduping past the boundary — no sawtooth.
        for _ in range(3):
            await wired_harness._apply_retry_cap('3068', _report(), True)
        assert len(_alerts(wired_harness)) == 1

    async def test_done_between_requeues_resets(self, wired_harness):
        """A DONE resets the streak, so two more requeues do not fire."""
        for _ in range(2):
            await wired_harness._apply_retry_cap('3068', _report(), True)
        await wired_harness._apply_retry_cap(
            '3068', _report(WorkflowOutcome.DONE), False,
        )
        for _ in range(2):
            await wired_harness._apply_retry_cap('3068', _report(), True)

        assert _alerts(wired_harness) == []

    async def test_requeue_with_real_work_resets(self, wired_harness):
        """A requeue that DID invoke an agent resets — genuine slow progress."""
        for _ in range(2):
            await wired_harness._apply_retry_cap('3068', _report(), True)
        await wired_harness._apply_retry_cap(
            '3068', _report(agent_invocations=1), True,
        )
        for _ in range(2):
            await wired_harness._apply_retry_cap('3068', _report(), True)

        assert _alerts(wired_harness) == []

    async def test_disabled_never_fires(self, wired_harness):
        """The kill switch holds however long the streak runs."""
        wired_harness.config.zero_progress_requeue.enabled = False
        for _ in range(20):
            await wired_harness._apply_retry_cap('3068', _report(), True)

        assert _alerts(wired_harness) == []

    async def test_fires_for_non_counting_dispositions(self, wired_harness):
        """counts_against_requeue_cap=False reports still fire — the whole point.

        Gating on the disposition table would couple this backstop to the very
        mechanism it exists to be independent of: a future disposition
        mis-classified as counting would then be invisible to BOTH.
        """
        for _ in range(3):
            await wired_harness._apply_retry_cap(
                '3068', _report(counts_against_requeue_cap=False), True,
            )

        assert len(_alerts(wired_harness)) == 1
        # The pre-existing cap genuinely did not fire for these.
        wired_harness.scheduler.trigger_retry_cap_exhausted.assert_not_called()

    async def test_existing_retry_cap_contract_unchanged(self, wired_harness):
        """The new call must not perturb _apply_retry_cap's existing behaviour."""
        assert await wired_harness._apply_retry_cap('3068', _report(), True) is True
        wired_harness.scheduler.record_requeue.assert_called_once()
        assert wired_harness.scheduler.record_requeue.call_args.kwargs[
            'counts_against_cap'
        ] is False

        wired_harness.scheduler.clear_requeue_count.assert_not_called()

        assert await wired_harness._apply_retry_cap(
            '4000', _report(WorkflowOutcome.DONE, task_id='4000'), False,
        ) is False
        wired_harness.scheduler.clear_requeue_count.assert_called_once_with('4000')

    async def test_run_id_guard_still_short_circuits(self, wired_harness):
        """The _run_id=None no-op hook (relied on by other suites) is preserved."""
        wired_harness._run_id = None
        for _ in range(10):
            await wired_harness._apply_retry_cap('3068', _report(), True)

        assert _alerts(wired_harness) == []
        wired_harness.scheduler.record_requeue.assert_not_called()

    async def test_alert_emitter_failure_does_not_propagate(
        self, wired_harness, monkeypatch,
    ):
        """A raising emitter must not escape or corrupt the return value.

        The backstop must never be able to break the accounting it backstops.
        """
        import orchestrator.harness as harness_mod

        def _boom(**kwargs):
            raise RuntimeError('emitter exploded')

        monkeypatch.setattr(
            harness_mod, 'emit_zero_progress_requeue_alert', _boom,
        )

        for _ in range(4):
            assert await wired_harness._apply_retry_cap('3068', _report(), True) is True

        # Cap accounting still happened for every call.
        assert wired_harness.scheduler.record_requeue.call_count == 4

    async def test_tracker_failure_does_not_propagate(
        self, wired_harness, monkeypatch,
    ):
        """A raising tracker is equally contained."""
        from unittest.mock import MagicMock

        broken = MagicMock()
        broken.record.side_effect = RuntimeError('tracker exploded')
        wired_harness._zero_progress_tracker = broken

        assert await wired_harness._apply_retry_cap('3068', _report(), True) is True
        wired_harness.scheduler.record_requeue.assert_called_once()
