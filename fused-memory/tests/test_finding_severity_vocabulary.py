"""The recon finding severity vocabulary has one source, which the schema and prompts read.

``FindingSeverity`` (models/reconciliation.py) is the closed vocabulary of the
recon_report ``add_finding`` channel.  It is deliberately distinct from
``VerdictSeverity`` (judge verdicts, which carry ``ok``) and from the
escalation-queue severities (info/blocking/critical/urgent).
"""

from __future__ import annotations

import functools
import json
import re
from enum import StrEnum

import pytest


class TestFindingSeverityVocabulary:
    def test_enum_members_are_ascending_minor_moderate_serious(self):
        from fused_memory.models.reconciliation import FindingSeverity

        assert issubclass(FindingSeverity, StrEnum)
        assert [m.name for m in FindingSeverity] == ['minor', 'moderate', 'serious']
        for member in FindingSeverity:
            assert member.value == member.name

    def test_every_member_carries_a_gloss(self):
        from fused_memory.models.reconciliation import FindingSeverity

        for member in FindingSeverity:
            assert isinstance(member.gloss, str) and member.gloss.strip(), member

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


_ADD_FINDING_SEVERITY_ARG = re.compile(r'add_finding\([^)]*?\bseverity=([^,)]*)')


def _offers_whole_vocabulary(text: str) -> bool:
    """Some add_finding call shape in ``text`` offers every severity member."""
    from fused_memory.models.reconciliation import FINDING_SEVERITY_VALUES

    return any(
        all(severity in m.group(1) for severity in FINDING_SEVERITY_VALUES)
        for m in _ADD_FINDING_SEVERITY_ARG.finditer(text)
    )


class TestPromptVocabularyIsRendered:
    """The prompt-side statements of the vocabulary are rendered from
    FINDING_SEVERITY_VALUES, never hand-transcribed."""

    def test_rendered_vocabulary_lists_every_member_and_gloss_in_order(self):
        from fused_memory.models.reconciliation import FindingSeverity
        from fused_memory.reconciliation.prompts import render_finding_severity_vocabulary

        rendered = render_finding_severity_vocabulary()
        assert rendered.strip()
        cursor = 0
        for severity in FindingSeverity:
            for token in (severity.value, severity.gloss):
                found = rendered.find(token, cursor)
                assert found >= 0, f'{token!r} missing or out of order in {rendered!r}'
                cursor = found + len(token)

    def test_add_finding_call_shape_enumerates_the_vocabulary(self):
        from fused_memory.reconciliation.prompts import (
            get_recon_report_tool_guidance,
            render_recon_report_tool_guidance,
        )

        for name, guidance in (
            ('render_recon_report_tool_guidance', render_recon_report_tool_guidance()),
            ('get_recon_report_tool_guidance', get_recon_report_tool_guidance()),
        ):
            assert _offers_whole_vocabulary(guidance), name
            assert 'severity=<severity>' not in guidance, name

    def test_every_stage_prompt_carries_the_enumerated_call_shape(self):
        from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
        from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

        for name, prompt in (
            ('STAGE1_SYSTEM_PROMPT', STAGE1_SYSTEM_PROMPT),
            ('STAGE2_SYSTEM_PROMPT', STAGE2_SYSTEM_PROMPT),
            ('STAGE3_SYSTEM_PROMPT', STAGE3_SYSTEM_PROMPT),
        ):
            assert _offers_whole_vocabulary(prompt), name

    def test_both_stage3_vocabulary_listings_are_rendered(self):
        from fused_memory.reconciliation.prompts import render_finding_severity_vocabulary
        from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

        assert STAGE3_SYSTEM_PROMPT.count(render_finding_severity_vocabulary()) >= 2


_STAGE_PROMPT_NAMES = ('STAGE1_SYSTEM_PROMPT', 'STAGE2_SYSTEM_PROMPT', 'STAGE3_SYSTEM_PROMPT')
_ASSEMBLED_PROMPT_NAMES = (*_STAGE_PROMPT_NAMES, 'recon_report_tool_guidance')

_LITERAL_SEVERITY = re.compile(r"severity\s*[=:]\s*[`'\"]?([A-Za-z_]+)")
_HYPHENATED_SEVERITY = re.compile(r'\b([A-Za-z]+)-severity\b')
_ADJECTIVAL_SEVERITY = re.compile(r'\b(info|low|high)\s+finding\b')


@functools.cache
def _assembled_prompt(name: str) -> str:
    """The fully assembled text, so interpolated renderers are scanned too."""
    from fused_memory.reconciliation.prompts import get_recon_report_tool_guidance
    from fused_memory.reconciliation.prompts.stage1 import STAGE1_SYSTEM_PROMPT
    from fused_memory.reconciliation.prompts.stage2 import STAGE2_SYSTEM_PROMPT
    from fused_memory.reconciliation.prompts.stage3 import STAGE3_SYSTEM_PROMPT

    return {
        'STAGE1_SYSTEM_PROMPT': STAGE1_SYSTEM_PROMPT,
        'STAGE2_SYSTEM_PROMPT': STAGE2_SYSTEM_PROMPT,
        'STAGE3_SYSTEM_PROMPT': STAGE3_SYSTEM_PROMPT,
        'recon_report_tool_guidance': get_recon_report_tool_guidance(),
    }[name]


def _off_vocabulary_report(prompt_name: str, prompt: str, matches: list[re.Match[str]]) -> str:
    from fused_memory.models.reconciliation import FINDING_SEVERITY_VALUES

    lines = [
        f'{prompt_name} teaches a finding severity outside {list(FINDING_SEVERITY_VALUES)}:'
    ]
    for m in matches:
        excerpt = prompt[max(0, m.start() - 30): m.end() + 30].replace('\n', ' ')
        lines.append(f'  {m.group(1)!r} in ...{excerpt}...')
    lines.append(
        'Finding severities are FindingSeverity (models/reconciliation.py). '
        '"Present but not automatically fixable" is actionable=false, not a '
        'severity. Escalation-queue severities (info/blocking/critical/urgent) '
        'and judge verdicts (ok) are different vocabularies: if this prompt '
        'legitimately names one, reword it so it does not read as a finding '
        'severity rather than allow-listing the token here.'
    )
    return '\n'.join(lines)


class TestAssembledPromptsNameOnlyEnumSeverities:
    """Backstop, not proof: the severity phrasings that have drifted before
    (``severity=X``, ``X-severity``, ``info/low/high finding``) name only
    FindingSeverity members in the assembled prompts.

    Other phrasings pass unseen, and policies/*.py text or the recon_self_model
    signature are scanned only where interpolated into these prompts.  The
    primary defence is prompt prose interpolating FindingSeverity members and
    render_finding_severity_vocabulary() rather than spelling a severity out.
    """

    @pytest.mark.parametrize('prompt_name', _ASSEMBLED_PROMPT_NAMES)
    def test_literal_severities_are_enum_members(self, prompt_name):
        from fused_memory.models.reconciliation import FINDING_SEVERITY_SET

        prompt = _assembled_prompt(prompt_name)
        bad = [
            m for m in _LITERAL_SEVERITY.finditer(prompt)
            if m.group(1) not in FINDING_SEVERITY_SET
        ]
        assert not bad, _off_vocabulary_report(prompt_name, prompt, bad)

    @pytest.mark.parametrize('prompt_name', _ASSEMBLED_PROMPT_NAMES)
    def test_hyphenated_severities_are_enum_members(self, prompt_name):
        from fused_memory.models.reconciliation import FINDING_SEVERITY_SET

        prompt = _assembled_prompt(prompt_name)
        bad = [
            m for m in _HYPHENATED_SEVERITY.finditer(prompt)
            if m.group(1) not in FINDING_SEVERITY_SET
        ]
        assert not bad, _off_vocabulary_report(prompt_name, prompt, bad)

    @pytest.mark.parametrize('prompt_name', _ASSEMBLED_PROMPT_NAMES)
    def test_no_off_vocabulary_word_is_used_as_a_finding_adjective(self, prompt_name):
        prompt = _assembled_prompt(prompt_name)
        bad = list(_ADJECTIVAL_SEVERITY.finditer(prompt))
        assert not bad, _off_vocabulary_report(prompt_name, prompt, bad)

    @pytest.mark.parametrize('prompt_name', _STAGE_PROMPT_NAMES)
    def test_literal_extractor_is_not_vacuous(self, prompt_name):
        assert _LITERAL_SEVERITY.search(_assembled_prompt(prompt_name)), (
            f'the severity-literal extractor matched nothing in {prompt_name}; '
            'a regex that silently matches nothing would pass the scan above'
        )
