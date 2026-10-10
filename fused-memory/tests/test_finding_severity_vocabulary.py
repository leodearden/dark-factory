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


def _rendered_severity_placeholder() -> str:
    from fused_memory.models.reconciliation import FINDING_SEVERITY_VALUES

    return '<' + '|'.join(repr(v) for v in FINDING_SEVERITY_VALUES) + '>'


class TestPromptVocabularyIsRendered:
    """The prompt-side statements of the vocabulary are rendered from
    FINDING_SEVERITY_VALUES, never hand-transcribed."""

    def test_gloss_table_covers_exactly_the_vocabulary_in_order(self):
        from fused_memory.models.reconciliation import FINDING_SEVERITY_VALUES
        from fused_memory.reconciliation.prompts import FINDING_SEVERITY_GLOSSES

        assert tuple(FINDING_SEVERITY_GLOSSES) == FINDING_SEVERITY_VALUES
        for severity, gloss in FINDING_SEVERITY_GLOSSES.items():
            assert isinstance(gloss, str) and gloss.strip(), severity

    def test_rendered_vocabulary_lists_every_member_and_gloss_in_order(self):
        from fused_memory.models.reconciliation import FINDING_SEVERITY_VALUES
        from fused_memory.reconciliation.prompts import (
            FINDING_SEVERITY_GLOSSES,
            render_finding_severity_vocabulary,
        )

        rendered = render_finding_severity_vocabulary()
        assert rendered.strip()
        cursor = 0
        for severity in FINDING_SEVERITY_VALUES:
            for token in (severity, FINDING_SEVERITY_GLOSSES[severity]):
                found = rendered.find(token, cursor)
                assert found >= 0, f'{token!r} missing or out of order in {rendered!r}'
                cursor = found + len(token)

    def test_add_finding_call_shape_enumerates_the_vocabulary(self):
        from fused_memory.reconciliation.prompts import (
            get_recon_report_tool_guidance,
            render_recon_report_tool_guidance,
        )

        expected = f'severity={_rendered_severity_placeholder()}'
        for name, guidance in (
            ('render_recon_report_tool_guidance', render_recon_report_tool_guidance()),
            ('get_recon_report_tool_guidance', get_recon_report_tool_guidance()),
        ):
            assert expected in guidance, f'{name} lacks {expected!r}'
            assert 'severity=<severity>' not in guidance, name

    def test_every_stage_prompt_carries_the_enumerated_call_shape(self):
        from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

        expected = f'severity={_rendered_severity_placeholder()}'
        for name, prompt in (
            ('STAGE1_SYSTEM_PROMPT', STAGE1_SYSTEM_PROMPT),
            ('STAGE2_SYSTEM_PROMPT', STAGE2_SYSTEM_PROMPT),
            ('STAGE3_SYSTEM_PROMPT', STAGE3_SYSTEM_PROMPT),
        ):
            assert expected in prompt, f'{name} lacks {expected!r}'

    def test_both_stage3_vocabulary_listings_are_rendered(self):
        from fused_memory.reconciliation.prompts import render_finding_severity_vocabulary
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

        assert STAGE3_SYSTEM_PROMPT.count(render_finding_severity_vocabulary()) >= 2
