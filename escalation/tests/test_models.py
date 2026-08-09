"""Tests for escalation data model — BORN_AT_L2_SEVERITIES constant, level field, and L2 cluster fields."""

from __future__ import annotations

import json

from escalation.models import (
    BORN_AT_L2_SEVERITIES,
    RESOLUTION_CLASSES,
    Escalation,
    EvidenceEntry,
    TrainState,
)


class TestBornAtL2Severities:
    """BORN_AT_L2_SEVERITIES constant is defined, typed correctly, and contains expected values."""

    def test_constant_is_frozenset(self):
        """BORN_AT_L2_SEVERITIES is a frozenset."""
        assert isinstance(BORN_AT_L2_SEVERITIES, frozenset)

    def test_constant_contains_critical(self):
        """BORN_AT_L2_SEVERITIES contains 'critical'."""
        assert 'critical' in BORN_AT_L2_SEVERITIES

    def test_constant_contains_urgent(self):
        """BORN_AT_L2_SEVERITIES contains 'urgent'."""
        assert 'urgent' in BORN_AT_L2_SEVERITIES

    def test_constant_contains_exactly_critical_and_urgent(self):
        """BORN_AT_L2_SEVERITIES contains exactly {'critical', 'urgent'} — no extras."""
        assert frozenset({'critical', 'urgent'}) == BORN_AT_L2_SEVERITIES


class TestEscalationLevelDefault:
    """Escalation default level is 0."""

    def test_default_level_is_zero(self):
        """Escalation() has level=0 by default."""
        esc = Escalation(
            id='esc-task-1-0001',
            task_id='task-1',
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary='default level test',
        )
        assert esc.level == 0


class TestEscalationLevelRoundTrip:
    """Escalation(level=2) round-trips correctly through to_dict/from_dict and to_json/from_json."""

    def _make_l2_esc(self) -> Escalation:
        return Escalation(
            id='esc-task-1-0001',
            task_id='task-1',
            agent_role='implementer',
            severity='critical',
            category='scope_violation',
            summary='level=2 round-trip test',
            level=2,
        )

    def test_level2_preserved_in_to_dict(self):
        """Escalation(level=2).to_dict() preserves level=2."""
        esc = self._make_l2_esc()
        d = esc.to_dict()
        assert d['level'] == 2

    def test_level2_preserved_in_from_dict(self):
        """Escalation.from_dict(esc.to_dict()) round-trips level=2."""
        esc = self._make_l2_esc()
        restored = Escalation.from_dict(esc.to_dict())
        assert restored.level == 2

    def test_level2_preserved_in_to_json(self):
        """Escalation(level=2).to_json() contains level=2 in the serialised output."""
        esc = self._make_l2_esc()
        import json
        d = json.loads(esc.to_json())
        assert d['level'] == 2

    def test_level2_preserved_in_from_json(self):
        """Escalation.from_json(esc.to_json()) round-trips level=2."""
        esc = self._make_l2_esc()
        restored = Escalation.from_json(esc.to_json())
        assert restored.level == 2


class TestL2Fields:
    """Escalation dataclass has L2 cluster fields: members, root_cause, options."""

    def _make_base_esc(self) -> Escalation:
        return Escalation(
            id='esc-task-1-0001',
            task_id='task-1',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='design_concern',
            summary='L2 cluster test',
        )

    # --- Default values ---

    def test_members_defaults_to_empty_list(self):
        """Escalation() has members=[] by default."""
        esc = self._make_base_esc()
        assert esc.members == []

    def test_root_cause_defaults_to_empty_string(self):
        """Escalation() has root_cause='' by default."""
        esc = self._make_base_esc()
        assert esc.root_cause == ''

    def test_options_defaults_to_empty_list(self):
        """Escalation() has options=[] by default."""
        esc = self._make_base_esc()
        assert esc.options == []

    def test_default_members_is_mutable_and_independent(self):
        """Two Escalation instances do not share the same members list (default_factory)."""
        esc1 = self._make_base_esc()
        esc2 = self._make_base_esc()
        esc1.members.append('esc-x-1')
        assert esc2.members == [], 'Default members lists must be independent (field uses default_factory)'

    def test_default_options_is_mutable_and_independent(self):
        """Two Escalation instances do not share the same options list (default_factory)."""
        esc1 = self._make_base_esc()
        esc2 = self._make_base_esc()
        esc1.options.append('A: some option')
        assert esc2.options == [], 'Default options lists must be independent (field uses default_factory)'

    # --- to_dict round-trips ---

    def test_members_roundtrip_via_to_dict_from_dict(self):
        """Escalation(members=[...]).to_dict()/from_dict() preserves members list."""
        esc = Escalation(
            id='esc-task-1-0001',
            task_id='task-1',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='design_concern',
            summary='test',
            level=2,
            members=['esc-l1-1', 'esc-l1-2'],
            root_cause='bad merge',
            options=['A: rollback', 'B: fix forward'],
        )
        restored = Escalation.from_dict(esc.to_dict())
        assert restored.members == ['esc-l1-1', 'esc-l1-2']
        assert restored.root_cause == 'bad merge'
        assert restored.options == ['A: rollback', 'B: fix forward']

    def test_members_roundtrip_via_to_json_from_json(self):
        """Escalation(members=[...]).to_json()/from_json() preserves members list."""
        esc = Escalation(
            id='esc-task-1-0001',
            task_id='task-1',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='design_concern',
            summary='test',
            level=2,
            members=['esc-l1-3'],
            root_cause='repeated timeout',
            options=['A: disable', 'B: add retry', 'C: alert on-call'],
        )
        restored = Escalation.from_json(esc.to_json())
        assert restored.members == ['esc-l1-3']
        assert restored.root_cause == 'repeated timeout'
        assert restored.options == ['A: disable', 'B: add retry', 'C: alert on-call']

    def test_empty_defaults_serialise_to_json(self):
        """Default-valued L2 fields appear in the serialised JSON (empty list / empty str)."""
        esc = self._make_base_esc()
        d = json.loads(esc.to_json())
        assert 'members' in d, "to_json() must include 'members' key"
        assert 'root_cause' in d, "to_json() must include 'root_cause' key"
        assert 'options' in d, "to_json() must include 'options' key"
        assert d['members'] == []
        assert d['root_cause'] == ''
        assert d['options'] == []

    def test_old_json_without_l2_fields_deserialises_to_defaults(self):
        """from_json() on a pre-L2 JSON blob (missing members/root_cause/options) returns defaults.

        This validates that old on-disk files (written before the L2 fields were added)
        deserialise correctly without migration — new fields default to [] / ''.
        """
        # Build a dict without the L2 fields and serialise it
        old_dict = {
            'id': 'esc-task-1-0001',
            'task_id': 'task-1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'scope_violation',
            'summary': 'old-format escalation',
            'detail': '',
            'suggested_action': '',
            'timestamp': '2026-01-01T00:00:00+00:00',
            'status': 'pending',
            'resolution': None,
            'worktree': None,
            'workflow_state': None,
            'level': 0,
            'resolved_at': None,
            'resolved_by': None,
            'resolution_turns': None,
            'dedupe_count': 0,
            'dedupe_children': [],
            'dedupe_fingerprint': None,
            # NOTE: members, root_cause, options are absent
        }
        old_json = json.dumps(old_dict)
        restored = Escalation.from_json(old_json)
        assert restored.members == [], f"Expected members=[], got {restored.members!r}"
        assert restored.root_cause == '', f"Expected root_cause='', got {restored.root_cause!r}"
        assert restored.options == [], f"Expected options=[], got {restored.options!r}"


class TestEscalationTrainState:
    """Escalation dataclass has a train_state field (PRD § 9.8 park-prefix derail context)."""

    def _make_base_esc(self) -> Escalation:
        return Escalation(
            id='esc-task-103-0001',
            task_id='103',
            agent_role='orchestrator',
            severity='blocking',
            category='task_failure',
            summary='train member blocked',
        )

    # --- (a) default is None ---

    def test_train_state_default_is_none(self):
        """Escalation constructed without train_state has train_state=None."""
        esc = self._make_base_esc()
        assert esc.train_state is None

    # --- (b) round-trip ---

    def test_train_state_round_trip_via_to_dict_from_dict(self):
        """train_state dict is preserved through to_dict() / from_dict()."""
        ts: TrainState = {'id': 'T1', 'order': 2, 'parked_members': ['101', '102'], 'failing_member': '103'}
        esc = Escalation(
            id='esc-task-103-0001',
            task_id='103',
            agent_role='orchestrator',
            severity='blocking',
            category='task_failure',
            summary='train member blocked',
            level=1,
            train_state=ts,
        )
        restored = Escalation.from_dict(esc.to_dict())
        assert restored.train_state == ts

    def test_train_state_round_trip_via_to_json_from_json(self):
        """train_state dict is preserved through to_json() / from_json()."""
        ts: TrainState = {'id': 'T1', 'order': 2, 'parked_members': ['101', '102'], 'failing_member': '103'}
        esc = Escalation(
            id='esc-task-103-0001',
            task_id='103',
            agent_role='orchestrator',
            severity='blocking',
            category='task_failure',
            summary='train member blocked',
            level=1,
            train_state=ts,
        )
        restored = Escalation.from_json(esc.to_json())
        assert restored.train_state == ts

    def test_train_state_appears_in_to_json_output(self):
        """train_state is serialised (not silently dropped) when set."""
        ts: TrainState = {'id': 'T1', 'order': 2, 'parked_members': ['101'], 'failing_member': '103'}
        esc = Escalation(
            id='esc-task-103-0001',
            task_id='103',
            agent_role='orchestrator',
            severity='blocking',
            category='task_failure',
            summary='train member blocked',
            train_state=ts,
        )
        d = json.loads(esc.to_json())
        assert 'train_state' in d
        assert d['train_state'] == ts

    # --- (c) legacy JSON backward compat ---

    def test_from_dict_legacy_json_omits_train_state(self):
        """from_json() on JSON without train_state key returns train_state=None (backward compat)."""
        old_dict = {
            'id': 'esc-task-1-0001',
            'task_id': 'task-1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'scope_violation',
            'summary': 'legacy escalation without train_state',
            'detail': '',
            'suggested_action': '',
            'timestamp': '2026-01-01T00:00:00+00:00',
            'status': 'pending',
            'resolution': None,
            'worktree': None,
            'workflow_state': None,
            'level': 0,
            'resolved_at': None,
            'resolved_by': None,
            'resolution_turns': None,
            'dedupe_count': 0,
            'dedupe_children': [],
            'dedupe_fingerprint': None,
            'members': [],
            'root_cause': '',
            'options': [],
            # NOTE: train_state is intentionally absent
        }
        old_json = json.dumps(old_dict)
        restored = Escalation.from_json(old_json)
        assert restored.train_state is None, (
            f"Expected train_state=None for legacy JSON, got {restored.train_state!r}"
        )


class TestEscalationResolutionAction:
    """Escalation dataclass has a resolution_action field (C1 § action enum for resolve_issue)."""

    def _make_base_esc(self) -> Escalation:
        return Escalation(
            id='esc-task-200-0001',
            task_id='200',
            agent_role='orchestrator',
            severity='blocking',
            category='scope_violation',
            summary='test escalation for resolution_action',
        )

    # --- (a) default is None ---

    def test_resolution_action_default_is_none(self):
        """Escalation constructed without resolution_action has resolution_action=None."""
        esc = self._make_base_esc()
        assert esc.resolution_action is None

    # --- (b) round-trip to_dict / from_dict ---

    def test_resolution_action_round_trip_via_to_dict_from_dict(self):
        """resolution_action='park' is preserved through to_dict() / from_dict()."""
        esc = Escalation(
            id='esc-task-200-0001',
            task_id='200',
            agent_role='orchestrator',
            severity='blocking',
            category='scope_violation',
            summary='test resolution_action round-trip',
            resolution_action='park',
        )
        restored = Escalation.from_dict(esc.to_dict())
        assert restored.resolution_action == 'park'

    # --- (c) round-trip to_json / from_json ---

    def test_resolution_action_round_trip_via_to_json_from_json(self):
        """resolution_action='abandon' is preserved through to_json() / from_json()."""
        esc = Escalation(
            id='esc-task-200-0001',
            task_id='200',
            agent_role='orchestrator',
            severity='blocking',
            category='scope_violation',
            summary='test resolution_action json round-trip',
            resolution_action='abandon',
        )
        restored = Escalation.from_json(esc.to_json())
        assert restored.resolution_action == 'abandon'

    # --- (d) appears in serialised JSON ---

    def test_resolution_action_appears_in_to_json_output(self):
        """resolution_action is serialised (not silently dropped) when set."""
        esc = Escalation(
            id='esc-task-200-0001',
            task_id='200',
            agent_role='orchestrator',
            severity='blocking',
            category='scope_violation',
            summary='test resolution_action in json',
            resolution_action='resume',
        )
        d = json.loads(esc.to_json())
        assert 'resolution_action' in d
        assert d['resolution_action'] == 'resume'

    # --- (e) legacy JSON backward compat ---

    def test_from_dict_legacy_json_omits_resolution_action(self):
        """from_json() on JSON without resolution_action key returns resolution_action=None (backward compat)."""
        old_dict = {
            'id': 'esc-task-1-0001',
            'task_id': 'task-1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'scope_violation',
            'summary': 'legacy escalation without resolution_action',
            'detail': '',
            'suggested_action': '',
            'timestamp': '2026-01-01T00:00:00+00:00',
            'status': 'pending',
            'resolution': None,
            'worktree': None,
            'workflow_state': None,
            'level': 0,
            'resolved_at': None,
            'resolved_by': None,
            'resolution_turns': None,
            'dedupe_count': 0,
            'dedupe_children': [],
            'dedupe_fingerprint': None,
            'members': [],
            'root_cause': '',
            'options': [],
            'train_state': None,
            # NOTE: resolution_action is intentionally absent
        }
        old_json = json.dumps(old_dict)
        restored = Escalation.from_json(old_json)
        assert restored.resolution_action is None, (
            f"Expected resolution_action=None for legacy JSON, got {restored.resolution_action!r}"
        )


class TestEscalationResolutionClass:
    """Escalation dataclass has a resolution_class field (escalation-lifecycle-dashboard-prd.md Seam 1)."""

    def _make_base_esc(self) -> Escalation:
        return Escalation(
            id='esc-task-300-0001',
            task_id='300',
            agent_role='orchestrator',
            severity='blocking',
            category='scope_violation',
            summary='test escalation for resolution_class',
        )

    # --- (a) default is None ---

    def test_resolution_class_default_is_none(self):
        """Escalation constructed without resolution_class has resolution_class=None."""
        esc = self._make_base_esc()
        assert esc.resolution_class is None

    # --- (b) RESOLUTION_CLASSES constant ---

    def test_resolution_classes_is_frozenset(self):
        """RESOLUTION_CLASSES is a frozenset."""
        assert isinstance(RESOLUTION_CLASSES, frozenset)

    def test_resolution_classes_contains_exactly_the_three_legal_values(self):
        """RESOLUTION_CLASSES contains exactly {'benign', 'actionable',
        'moot-terminal-subject'} — no extras. 'moot-terminal-subject' is the
        distinct, non-benign stamp the task-2724 revalidation sweep writes."""
        assert frozenset({'benign', 'actionable', 'moot-terminal-subject'}) == RESOLUTION_CLASSES
        assert 'moot-terminal-subject' in RESOLUTION_CLASSES

    # --- (c) round-trip to_dict/from_dict and to_json/from_json ---

    def test_resolution_class_round_trip_via_to_dict_from_dict(self):
        """resolution_class='benign' is preserved through to_dict() / from_dict()."""
        esc = Escalation(
            id='esc-task-300-0001',
            task_id='300',
            agent_role='orchestrator',
            severity='blocking',
            category='scope_violation',
            summary='test resolution_class round-trip',
            resolution_class='benign',
        )
        restored = Escalation.from_dict(esc.to_dict())
        assert restored.resolution_class == 'benign'

    def test_resolution_class_round_trip_via_to_json_from_json(self):
        """resolution_class='benign' is preserved through to_json() / from_json()."""
        esc = Escalation(
            id='esc-task-300-0001',
            task_id='300',
            agent_role='orchestrator',
            severity='blocking',
            category='scope_violation',
            summary='test resolution_class json round-trip',
            resolution_class='benign',
        )
        restored = Escalation.from_json(esc.to_json())
        assert restored.resolution_class == 'benign'

    def test_resolution_class_appears_in_to_json_output(self):
        """resolution_class is serialised (not silently dropped) when set."""
        esc = Escalation(
            id='esc-task-300-0001',
            task_id='300',
            agent_role='orchestrator',
            severity='blocking',
            category='scope_violation',
            summary='test resolution_class in json',
            resolution_class='actionable',
        )
        d = json.loads(esc.to_json())
        assert 'resolution_class' in d
        assert d['resolution_class'] == 'actionable'

    # --- (d) legacy JSON backward compat ---

    def test_from_dict_legacy_json_omits_resolution_class(self):
        """from_json() on JSON without resolution_class key returns resolution_class=None (backward compat)."""
        old_dict = {
            'id': 'esc-task-1-0001',
            'task_id': 'task-1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'scope_violation',
            'summary': 'legacy escalation without resolution_class',
            'detail': '',
            'suggested_action': '',
            'timestamp': '2026-01-01T00:00:00+00:00',
            'status': 'pending',
            'resolution': None,
            'worktree': None,
            'workflow_state': None,
            'level': 0,
            'resolved_at': None,
            'resolved_by': None,
            'resolution_turns': None,
            'dedupe_count': 0,
            'dedupe_children': [],
            'dedupe_fingerprint': None,
            'members': [],
            'root_cause': '',
            'options': [],
            'train_state': None,
            'resolution_action': None,
            # NOTE: resolution_class is intentionally absent
        }
        old_json = json.dumps(old_dict)
        restored = Escalation.from_json(old_json)
        assert restored.resolution_class is None, (
            f"Expected resolution_class=None for legacy JSON, got {restored.resolution_class!r}"
        )


class TestEscalationTriageFields:
    """Escalation dataclass has triage-ack marker fields: triaged_at, triaged_by, triage_note, updated_at.

    These are an annotation (not a resolution) — a durable "I looked at this" marker
    plus a "changed since I triaged it" signal, so escalation-watcher-auto rotations
    can skip re-deriving the disposition of a still-pending item every rotation.
    """

    def _make_base_esc(self) -> Escalation:
        return Escalation(
            id='esc-task-400-0001',
            task_id='400',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='scope_violation',
            summary='test escalation for triage fields',
        )

    # --- (a) defaults ---

    def test_triaged_at_default_is_none(self):
        """Escalation constructed without triaged_at has triaged_at=None."""
        esc = self._make_base_esc()
        assert esc.triaged_at is None

    def test_triaged_by_default_is_none(self):
        """Escalation constructed without triaged_by has triaged_by=None."""
        esc = self._make_base_esc()
        assert esc.triaged_by is None

    def test_triage_note_default_is_empty_string(self):
        """Escalation constructed without triage_note has triage_note=''."""
        esc = self._make_base_esc()
        assert esc.triage_note == ''

    def test_updated_at_default_is_none(self):
        """Escalation constructed without updated_at has updated_at=None."""
        esc = self._make_base_esc()
        assert esc.updated_at is None

    # --- (b) round-trip via to_dict / from_dict ---

    def test_triage_fields_round_trip_via_to_dict_from_dict(self):
        """triaged_at/triaged_by/triage_note/updated_at are preserved through to_dict()/from_dict()."""
        esc = Escalation(
            id='esc-task-400-0001',
            task_id='400',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='scope_violation',
            summary='test triage fields round-trip',
            triaged_at='2026-07-14T00:00:00+00:00',
            triaged_by='orchestrator-escalation-watcher-auto',
            triage_note='task-604 status==done | probe: get_task 604 -> status=done',
            updated_at='2026-07-14T00:00:00+00:00',
        )
        restored = Escalation.from_dict(esc.to_dict())
        assert restored.triaged_at == '2026-07-14T00:00:00+00:00'
        assert restored.triaged_by == 'orchestrator-escalation-watcher-auto'
        assert restored.triage_note == 'task-604 status==done | probe: get_task 604 -> status=done'
        assert restored.updated_at == '2026-07-14T00:00:00+00:00'

    # --- (c) round-trip via to_json / from_json ---

    def test_triage_fields_round_trip_via_to_json_from_json(self):
        """triaged_at/triaged_by/triage_note/updated_at are preserved through to_json()/from_json()."""
        esc = Escalation(
            id='esc-task-400-0001',
            task_id='400',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='scope_violation',
            summary='test triage fields json round-trip',
            triaged_at='2026-07-14T00:00:00+00:00',
            triaged_by='orchestrator-escalation-watcher-auto',
            triage_note='task-604 status==done | probe: get_task 604 -> status=done',
            updated_at='2026-07-14T00:00:00+00:00',
        )
        restored = Escalation.from_json(esc.to_json())
        assert restored.triaged_at == '2026-07-14T00:00:00+00:00'
        assert restored.triaged_by == 'orchestrator-escalation-watcher-auto'
        assert restored.triage_note == 'task-604 status==done | probe: get_task 604 -> status=done'
        assert restored.updated_at == '2026-07-14T00:00:00+00:00'

    # --- (d) appear in serialised JSON ---

    def test_triage_fields_appear_in_to_json_output(self):
        """triaged_at/triaged_by/triage_note/updated_at are serialised (not silently dropped) when set."""
        esc = Escalation(
            id='esc-task-400-0001',
            task_id='400',
            agent_role='escalation-watcher-auto',
            severity='blocking',
            category='scope_violation',
            summary='test triage fields in json',
            triaged_at='2026-07-14T00:00:00+00:00',
            triaged_by='orchestrator-escalation-watcher-auto',
            triage_note='task-604 status==done | probe: get_task 604 -> status=done',
            updated_at='2026-07-14T00:00:00+00:00',
        )
        d = json.loads(esc.to_json())
        assert 'triaged_at' in d
        assert 'triaged_by' in d
        assert 'triage_note' in d
        assert 'updated_at' in d
        assert d['triaged_at'] == '2026-07-14T00:00:00+00:00'
        assert d['triaged_by'] == 'orchestrator-escalation-watcher-auto'
        assert d['triage_note'] == 'task-604 status==done | probe: get_task 604 -> status=done'
        assert d['updated_at'] == '2026-07-14T00:00:00+00:00'

    # --- (e) legacy JSON backward compat ---

    def test_from_dict_legacy_json_omits_triage_fields(self):
        """from_json() on JSON without the triage fields returns their defaults (backward compat)."""
        old_dict = {
            'id': 'esc-task-1-0001',
            'task_id': 'task-1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'scope_violation',
            'summary': 'legacy escalation without triage fields',
            'detail': '',
            'suggested_action': '',
            'timestamp': '2026-01-01T00:00:00+00:00',
            'status': 'pending',
            'resolution': None,
            'worktree': None,
            'workflow_state': None,
            'level': 0,
            'resolved_at': None,
            'resolved_by': None,
            'resolution_turns': None,
            'dedupe_count': 0,
            'dedupe_children': [],
            'dedupe_fingerprint': None,
            'members': [],
            'root_cause': '',
            'options': [],
            'train_state': None,
            'resolution_action': None,
            'resolution_class': None,
            # NOTE: triaged_at, triaged_by, triage_note, updated_at are intentionally absent
        }
        old_json = json.dumps(old_dict)
        restored = Escalation.from_json(old_json)
        assert restored.triaged_at is None, (
            f"Expected triaged_at=None for legacy JSON, got {restored.triaged_at!r}"
        )
        assert restored.triaged_by is None, (
            f"Expected triaged_by=None for legacy JSON, got {restored.triaged_by!r}"
        )
        assert restored.triage_note == '', (
            f"Expected triage_note='' for legacy JSON, got {restored.triage_note!r}"
        )
        assert restored.updated_at is None, (
            f"Expected updated_at=None for legacy JSON, got {restored.updated_at!r}"
        )


class TestEscalationEvidence:
    """Escalation dataclass has an `evidence` field (task 2558 — structured raw-observation entries).

    Mirrors the zero-migration round-trip pattern of TestEscalationTrainState /
    TestEscalationResolutionAction.  Each entry is a plain dict
    {observation, measured_at, ref} — stored/returned verbatim, no shape
    validation.  The values are plain dict literals (annotated
    `list[EvidenceEntry]` only to satisfy the invariant-`list` type check;
    EvidenceEntry is a plain dict at runtime), so this reads identically pre-
    and post-implementation.
    """

    def _make_base_esc(self) -> Escalation:
        return Escalation(
            id='esc-task-400-0001',
            task_id='400',
            agent_role='orchestrator',
            severity='critical',
            category='infra_issue',
            summary='test escalation for evidence',
        )

    # --- (a) default is [] ---

    def test_evidence_default_is_empty_list(self):
        """Escalation constructed without evidence has evidence == []."""
        esc = self._make_base_esc()
        assert esc.evidence == []

    # --- (b) default_factory independence (no shared mutable default) ---

    def test_evidence_default_not_shared_between_instances(self):
        """Two escalations do not share the same default evidence list object."""
        esc1 = self._make_base_esc()
        esc2 = self._make_base_esc()
        assert esc1.evidence is not esc2.evidence
        esc1.evidence.append({'observation': 'x', 'measured_at': 'y', 'ref': 'z'})
        assert esc2.evidence == []

    # --- (c) round-trip to_dict / from_dict ---

    def test_evidence_round_trip_via_to_dict_from_dict(self):
        """evidence list is preserved through to_dict() / from_dict()."""
        ev: list[EvidenceEntry] = [{'observation': 'main red at abc123', 'measured_at': '2026-07-14T00:00:00+00:00', 'ref': 'HEAD=abc123'}]
        esc = Escalation(
            id='esc-task-400-0001',
            task_id='400',
            agent_role='orchestrator',
            severity='critical',
            category='infra_issue',
            summary='test evidence round-trip',
            evidence=ev,
        )
        restored = Escalation.from_dict(esc.to_dict())
        assert restored.evidence == ev

    # --- (c) round-trip to_json / from_json ---

    def test_evidence_round_trip_via_to_json_from_json(self):
        """evidence list is preserved through to_json() / from_json()."""
        ev: list[EvidenceEntry] = [{'observation': 'main red at abc123', 'measured_at': '2026-07-14T00:00:00+00:00', 'ref': 'HEAD=abc123'}]
        esc = Escalation(
            id='esc-task-400-0001',
            task_id='400',
            agent_role='orchestrator',
            severity='critical',
            category='infra_issue',
            summary='test evidence json round-trip',
            evidence=ev,
        )
        restored = Escalation.from_json(esc.to_json())
        assert restored.evidence == ev

    # --- (d) appears in serialised JSON ---

    def test_evidence_appears_in_to_json_output(self):
        """evidence is serialised (not silently dropped) when set."""
        ev: list[EvidenceEntry] = [{'observation': 'main red at abc123', 'measured_at': '2026-07-14T00:00:00+00:00', 'ref': 'HEAD=abc123'}]
        esc = Escalation(
            id='esc-task-400-0001',
            task_id='400',
            agent_role='orchestrator',
            severity='critical',
            category='infra_issue',
            summary='test evidence in json',
            evidence=ev,
        )
        d = json.loads(esc.to_json())
        assert 'evidence' in d
        assert d['evidence'] == ev

    # --- (e) legacy JSON backward compat ---

    def test_from_dict_legacy_json_omits_evidence(self):
        """from_json() on JSON without evidence key returns evidence == [] (backward compat, zero migration)."""
        old_dict = {
            'id': 'esc-task-1-0001',
            'task_id': 'task-1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'scope_violation',
            'summary': 'legacy escalation without evidence',
            'detail': '',
            'suggested_action': '',
            'timestamp': '2026-01-01T00:00:00+00:00',
            'status': 'pending',
            'resolution': None,
            'worktree': None,
            'workflow_state': None,
            'level': 0,
            'resolved_at': None,
            'resolved_by': None,
            'resolution_turns': None,
            'dedupe_count': 0,
            'dedupe_children': [],
            'dedupe_fingerprint': None,
            'members': [],
            'root_cause': '',
            'options': [],
            'train_state': None,
            'resolution_action': None,
            'resolution_class': None,
            # NOTE: evidence is intentionally absent
        }
        old_json = json.dumps(old_dict)
        restored = Escalation.from_json(old_json)
        assert restored.evidence == [], (
            f"Expected evidence=[] for legacy JSON, got {restored.evidence!r}"
        )


class TestEscalationGrantedFiles:
    """Escalation dataclass has a granted_files field (task 2505 steward scope-grant channel).

    granted_files is the steward's structured scope-expansion grant (file-level,
    project-relative paths), consumed by the orchestrator resume path — distinct
    from the free-text `resolution` rationale string.
    """

    def _make_base_esc(self) -> Escalation:
        return Escalation(
            id='esc-task-500-0001',
            task_id='500',
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary='test escalation for granted_files',
        )

    # --- (a) default is [] (not None) ---

    def test_granted_files_defaults_to_empty_list(self):
        """Escalation constructed without granted_files has granted_files=[] (not None)."""
        esc = self._make_base_esc()
        assert esc.granted_files == []

    def test_default_granted_files_is_mutable_and_independent(self):
        """Two Escalation instances do not share the same granted_files list (default_factory)."""
        esc1 = self._make_base_esc()
        esc2 = self._make_base_esc()
        esc1.granted_files.append('a/b.py')
        assert esc2.granted_files == [], (
            'Default granted_files lists must be independent (field uses default_factory)'
        )

    # --- (b) round-trip via to_dict / from_dict ---

    def test_granted_files_round_trip_via_to_dict_from_dict(self):
        """granted_files=['a/b.py'] is preserved through to_dict() / from_dict()."""
        esc = Escalation(
            id='esc-task-500-0001',
            task_id='500',
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary='test granted_files round-trip',
            granted_files=['a/b.py'],
        )
        restored = Escalation.from_dict(esc.to_dict())
        assert restored.granted_files == ['a/b.py']

    # --- (c) round-trip via to_json / from_json ---

    def test_granted_files_round_trip_via_to_json_from_json(self):
        """granted_files=['a/b.py'] is preserved through to_json() / from_json()."""
        esc = Escalation(
            id='esc-task-500-0001',
            task_id='500',
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary='test granted_files json round-trip',
            granted_files=['a/b.py'],
        )
        restored = Escalation.from_json(esc.to_json())
        assert restored.granted_files == ['a/b.py']

    # --- (d) appears in serialised JSON ---

    def test_granted_files_appears_in_to_json_output(self):
        """granted_files is serialised (not silently dropped) when set."""
        esc = Escalation(
            id='esc-task-500-0001',
            task_id='500',
            agent_role='implementer',
            severity='blocking',
            category='scope_violation',
            summary='test granted_files in json',
            granted_files=['crate/Cargo.toml'],
        )
        d = json.loads(esc.to_json())
        assert 'granted_files' in d
        assert d['granted_files'] == ['crate/Cargo.toml']

    # --- (e) legacy JSON backward compat (zero-migration) ---

    def test_from_dict_legacy_json_omits_granted_files(self):
        """from_json() on JSON without granted_files key returns granted_files=[] (backward compat)."""
        old_dict = {
            'id': 'esc-task-1-0001',
            'task_id': 'task-1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'scope_violation',
            'summary': 'legacy escalation without granted_files',
            'detail': '',
            'suggested_action': '',
            'timestamp': '2026-01-01T00:00:00+00:00',
            'status': 'pending',
            'resolution': None,
            'worktree': None,
            'workflow_state': None,
            'level': 0,
            'resolved_at': None,
            'resolved_by': None,
            'resolution_turns': None,
            'dedupe_count': 0,
            'dedupe_children': [],
            'dedupe_fingerprint': None,
            'members': [],
            'root_cause': '',
            'options': [],
            'train_state': None,
            'resolution_action': None,
            'resolution_class': None,
            'triaged_at': None,
            'triaged_by': None,
            'triage_note': '',
            'updated_at': None,
            # NOTE: granted_files is intentionally absent
        }
        old_json = json.dumps(old_dict)
        restored = Escalation.from_json(old_json)
        assert restored.granted_files == [], (
            f"Expected granted_files=[] for legacy JSON, got {restored.granted_files!r}"
        )


class TestFilingClaimantRunId:
    """Escalation carries the FILING incarnation's claimant identity (task 3533).

    These tests pin the FIELD's storage/round-trip behaviour only.  What the
    identity MEANS, and the fail-safe rule applied to it, are documented once
    on ``escalation.pins.classify_pins`` and exercised in tests/test_pins.py.
    """

    #: Real ``compose_claimant_run_id`` output shape.
    IDENTITY = 'run-abc/3533-f3af2d2a/pid=1234'

    def _make_base_esc(self) -> Escalation:
        return Escalation(
            id='esc-task-3533-0001',
            task_id='3533',
            agent_role='implementer',
            severity='blocking',
            category='infra_issue',
            summary='s',
        )

    # --- (a) default is None ---

    def test_filing_claimant_run_id_defaults_to_none(self):
        """A minimally-constructed Escalation has filing_claimant_run_id=None."""
        esc = self._make_base_esc()
        assert esc.filing_claimant_run_id is None

    # --- (b) construction preserves a real identity string ---

    def test_filing_claimant_run_id_preserved_on_construction(self):
        """filing_claimant_run_id is stored verbatim (no parsing/normalisation)."""
        esc = Escalation(
            id='esc-task-3533-0001',
            task_id='3533',
            agent_role='implementer',
            severity='blocking',
            category='infra_issue',
            summary='s',
            filing_claimant_run_id=self.IDENTITY,
        )
        assert esc.filing_claimant_run_id == self.IDENTITY

    # --- (c) round-trip via to_dict / from_dict ---

    def test_filing_claimant_run_id_appears_in_to_dict(self):
        """to_dict() carries the key (not silently dropped)."""
        esc = Escalation(
            id='esc-task-3533-0001',
            task_id='3533',
            agent_role='implementer',
            severity='blocking',
            category='infra_issue',
            summary='s',
            filing_claimant_run_id=self.IDENTITY,
        )
        d = esc.to_dict()
        assert 'filing_claimant_run_id' in d
        assert d['filing_claimant_run_id'] == self.IDENTITY

    def test_filing_claimant_run_id_round_trip_via_to_dict_from_dict(self):
        """Escalation.from_dict(esc.to_dict()) round-trips the identity exactly."""
        esc = Escalation(
            id='esc-task-3533-0001',
            task_id='3533',
            agent_role='implementer',
            severity='blocking',
            category='infra_issue',
            summary='s',
            filing_claimant_run_id=self.IDENTITY,
        )
        restored = Escalation.from_dict(esc.to_dict())
        assert restored.filing_claimant_run_id == self.IDENTITY

    # --- (d) round-trip via to_json / from_json ---

    def test_filing_claimant_run_id_round_trip_via_to_json_from_json(self):
        """Escalation.from_json(esc.to_json()) round-trips the identity exactly."""
        esc = Escalation(
            id='esc-task-3533-0001',
            task_id='3533',
            agent_role='implementer',
            severity='blocking',
            category='infra_issue',
            summary='s',
            filing_claimant_run_id=self.IDENTITY,
        )
        restored = Escalation.from_json(esc.to_json())
        assert restored.filing_claimant_run_id == self.IDENTITY

    # --- (e) legacy JSON backward compat (zero-migration) ---

    def _legacy_payload(self) -> dict:
        """A pre-3533 on-disk payload — every field EXCEPT filing_claimant_run_id."""
        return {
            'id': 'esc-task-1-0001',
            'task_id': 'task-1',
            'agent_role': 'implementer',
            'severity': 'blocking',
            'category': 'scope_violation',
            'summary': 'legacy escalation without filing_claimant_run_id',
            'detail': '',
            'suggested_action': '',
            'timestamp': '2026-01-01T00:00:00+00:00',
            'status': 'pending',
            'resolution': None,
            'worktree': None,
            'workflow_state': None,
            'level': 0,
            'resolved_at': None,
            'resolved_by': None,
            'resolution_turns': None,
            'dedupe_count': 0,
            'dedupe_children': [],
            'dedupe_fingerprint': None,
            'members': [],
            'root_cause': '',
            'options': [],
            'evidence': [],
            'train_state': None,
            'resolution_action': None,
            'resolution_class': None,
            'triaged_at': None,
            'triaged_by': None,
            'triage_note': '',
            'updated_at': None,
            'granted_files': [],
            # NOTE: filing_claimant_run_id is intentionally absent
        }

    def test_from_dict_legacy_payload_omits_filing_claimant_run_id(self):
        """Legacy JSON without the key deserialises to None — zero migration."""
        restored = Escalation.from_dict(self._legacy_payload())
        assert restored.filing_claimant_run_id is None, (
            f'Expected None for legacy JSON, got {restored.filing_claimant_run_id!r}'
        )

    def test_from_json_legacy_payload_does_not_raise(self):
        """Legacy JSON deserialises through from_json without raising."""
        restored = Escalation.from_json(json.dumps(self._legacy_payload()))
        assert restored.id == 'esc-task-1-0001'
        assert restored.filing_claimant_run_id is None

    # --- (f) the __dataclass_fields__ filter is not weakened ---

    def test_unknown_extra_key_is_still_dropped(self):
        """An unknown payload key is dropped by from_dict's __dataclass_fields__ filter."""
        payload = self._legacy_payload()
        payload['not_a_real_field'] = 'boom'
        restored = Escalation.from_dict(payload)
        assert not hasattr(restored, 'not_a_real_field')


class TestTimestampIsStampedFromTheLiveClock:
    """REGRESSION PIN, not a fix — no timestamp defect exists (task 3236).

    The task asked for an investigation of an escalation whose ``timestamp``
    and ``resolved_at`` looked backdated relative to the incident report.  The
    read: ``Escalation.timestamp`` is stamped by a per-instance
    ``field(default_factory=...)`` reading the live clock, and the queue
    stamps ``resolved_at`` from ``datetime.now(UTC)``.  There is no cached or
    session clock in the stamping code to fix, and the measured values were
    internally consistent as a real sub-second sequence shortly before a UTC
    midnight boundary, which accounts for the reported date discrepancy
    without a clock bug.

    So these assertions guard against a FUTURE regression rather than
    describing a current one: specifically the classic mistake of hoisting the
    stamp to an import-time constant (``field(default=...)`` instead of
    ``default_factory``), which would make every record in a process share one
    timestamp.  They are expected to pass on first run — that is the correct
    outcome for a pin.
    """

    def test_timestamp_is_within_300s_of_now(self):
        """A freshly constructed Escalation is stamped from the live clock."""
        from datetime import UTC, datetime

        esc = Escalation(
            id='esc-1-1', task_id='1', agent_role='implementer',
            severity='blocking', category='infra_issue', summary='s',
        )

        stamped = datetime.fromisoformat(esc.timestamp)
        assert stamped.tzinfo is not None, f'Naive timestamp: {esc.timestamp!r}'
        delta = abs((datetime.now(UTC) - stamped).total_seconds())
        # 300s is derived, not guessed: the value comes from datetime.now(UTC)
        # at construction, so real slack is milliseconds.  300s never flakes on
        # a loaded CI box yet is three orders of magnitude tighter than the
        # >24h discrepancy that prompted the investigation.
        assert delta < 300, f'Timestamp {esc.timestamp!r} is {delta}s from now'

    def test_two_constructions_get_distinct_non_decreasing_timestamps(self):
        """THE assertion that catches a hoisted default.

        If the stamp were ever moved to ``field(default=...)`` — evaluated once
        at import — every Escalation in the process would share one timestamp
        and this would fail.
        """
        import time
        from datetime import datetime

        first = Escalation(
            id='esc-1-1', task_id='1', agent_role='implementer',
            severity='blocking', category='infra_issue', summary='s',
        )
        time.sleep(0.01)
        second = Escalation(
            id='esc-1-2', task_id='1', agent_role='implementer',
            severity='blocking', category='infra_issue', summary='s',
        )

        assert first.timestamp != second.timestamp, (
            'Both Escalations share one timestamp — the stamp looks hoisted to '
            'an import-time constant rather than a per-instance default_factory'
        )
        assert datetime.fromisoformat(second.timestamp) >= datetime.fromisoformat(
            first.timestamp,
        ), 'Timestamps must be non-decreasing across constructions'
