"""Tests for escalation data model — BORN_AT_L2_SEVERITIES constant and level field."""

from __future__ import annotations

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
