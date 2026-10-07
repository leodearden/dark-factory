"""The recon finding severity vocabulary has one source, and every surface reads it.

``FindingSeverity`` (models/reconciliation.py) is the closed vocabulary of the
recon_report ``add_finding`` channel.  It is deliberately distinct from
``VerdictSeverity`` (judge verdicts, which carry ``ok``) and from the
escalation-queue severities (info/blocking/critical/urgent).
"""

from __future__ import annotations

import json
from enum import StrEnum


class TestFindingSeverityVocabulary:
    def test_enum_members_are_ascending_minor_moderate_serious(self):
        from fused_memory.models.reconciliation import FindingSeverity

        assert issubclass(FindingSeverity, StrEnum)
        assert [m.name for m in FindingSeverity] == ['minor', 'moderate', 'serious']
        for member in FindingSeverity:
            assert member.value == member.name

    def test_values_tuple_and_membership_set_derive_from_the_enum(self):
        from fused_memory.models.reconciliation import (
            FINDING_SEVERITY_SET,
            FINDING_SEVERITY_VALUES,
        )

        assert isinstance(FINDING_SEVERITY_VALUES, tuple)
        assert FINDING_SEVERITY_VALUES == ('minor', 'moderate', 'serious')
        assert isinstance(FINDING_SEVERITY_SET, frozenset)
        assert frozenset(FINDING_SEVERITY_VALUES) == FINDING_SEVERITY_SET

    def test_finding_vocabulary_is_not_the_judge_verdict_vocabulary(self):
        from fused_memory.models.reconciliation import (
            FINDING_SEVERITY_SET,
            FINDING_SEVERITY_VALUES,
            VerdictSeverity,
        )

        assert set(FINDING_SEVERITY_VALUES) != {s.value for s in VerdictSeverity}
        assert 'ok' not in FINDING_SEVERITY_SET

    def test_stage3_finding_schema_enum_is_derived_from_the_vocabulary(self):
        from fused_memory.models.reconciliation import FINDING_SEVERITY_VALUES
        from fused_memory.reconciliation.cli_stage_runner import FINDING_ITEM_SCHEMA

        assert FINDING_ITEM_SCHEMA['properties']['severity']['enum'] == list(
            FINDING_SEVERITY_VALUES
        )
        assert 'severity' in FINDING_ITEM_SCHEMA['required']
        json.dumps(FINDING_ITEM_SCHEMA)
