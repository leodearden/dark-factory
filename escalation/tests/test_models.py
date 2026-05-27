"""Tests for escalation data model — BORN_AT_L2_SEVERITIES constant, level field, and L2 cluster fields."""

from __future__ import annotations

import json

from escalation.models import BORN_AT_L2_SEVERITIES, Escalation


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
