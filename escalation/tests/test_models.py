"""Tests for escalation data model — BORN_AT_L2_SEVERITIES constant, level field, and L2 cluster fields."""

from __future__ import annotations

import json

from escalation.models import BORN_AT_L2_SEVERITIES, RESOLUTION_CLASSES, Escalation, TrainState


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
    validation.  Uses plain dict literals (not the EvidenceEntry TypedDict,
    which is a plain dict at runtime) so this reads identically pre- and
    post-implementation.
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
        ev = [{'observation': 'main red at abc123', 'measured_at': '2026-07-14T00:00:00+00:00', 'ref': 'HEAD=abc123'}]
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
        ev = [{'observation': 'main red at abc123', 'measured_at': '2026-07-14T00:00:00+00:00', 'ref': 'HEAD=abc123'}]
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
        ev = [{'observation': 'main red at abc123', 'measured_at': '2026-07-14T00:00:00+00:00', 'ref': 'HEAD=abc123'}]
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
