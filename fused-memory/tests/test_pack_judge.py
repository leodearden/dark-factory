"""The pack judge: plans/followup-packing-sweep-prd.md §4.2 and decision 9.

The spec, the output schema, the verdict models and parser, and the one member renderer are
driven through their public signatures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

import pytest
from pydantic import ValidationError
from shared.followup_items import observed_item
from shared.prompt_artifact import PromptArtifactStore

from fused_memory.middleware.curator_types import clip_for_prompt
from fused_memory.middleware.followup_packing import CandidatePack, PackMember
from fused_memory.middleware.pack_judge import (
    CURATOR_PACK_OUTPUT_SCHEMA,
    CURATOR_PACK_SPEC,
    MAX_JUSTIFICATION_CHARS,
    PACK_LEDGER_ROLE,
    PACK_PROMPT_HARNESS_VERSION,
    PackGroup,
    PackVerdict,
    VerdictDegrade,
    parse_pack_verdict,
    render_pack_member,
)
from fused_memory.middleware.task_curator import CURATOR_BATCH_SPEC, CURATOR_SINGLE_SPEC

PACK = CandidatePack('parent', '1', (1, 2, 3, 4, 5), ('a/x.py',), survivor=1, pack_key='0' * 16)

FORBIDDEN_FIELDS = frozenset({'survivor', 'priority', 'files', 'items', 'files_to_modify'})


def _member(*, band: Literal['narrow'] | None) -> PackMember:
    return PackMember(
        task_id=4242,
        title='Relocate the inline test mod of foo.rs',
        kind='review_suggestion',
        parent=7,
        lock_keys=('a/x.py', 'b/y.py'),
        item_count=1,
        item_keys=('k4242',),
        band=band,
        pending_since=datetime(2026, 10, 1, tzinfo=UTC),
        text_fp='0' * 16,
        status='pending',
        dependencies=frozenset(),
        held=True,
        is_holder=False,
        has_live_dependents=False,
        claimed=False,
        has_branch=False,
        has_plan=False,
        dedupe_skipped=False,
        rider_of=None,
        rider_declines=0,
        rider_last_decline_reason=None,
    )


def _group(*member_ids: Any, **overrides: Any) -> dict[str, Any]:
    return {
        'member_ids': list(member_ids),
        'title': 'One coherent task',
        'description': 'what changes',
        'details': 'how',
        'justification': 'one concern',
        **overrides,
    }


def _raw(verdict: str, *groups: dict[str, Any], justification: str = 'rule 1 fired') -> dict[str, Any]:
    return {'verdict': verdict, 'groups': list(groups), 'justification': justification}


def test_the_pack_spec_shares_the_curator_heuristics_under_its_own_contract():
    assert CURATOR_PACK_SPEC.prompt_id == 'curator_pack'
    assert PACK_PROMPT_HARNESS_VERSION == 'curator-pack-v1'
    assert PACK_LEDGER_ROLE == 'curator_pack'
    assert CURATOR_PACK_SPEC.baseline_heuristics is CURATOR_SINGLE_SPEC.baseline_heuristics
    assert CURATOR_PACK_SPEC.contract not in {CURATOR_SINGLE_SPEC.contract, CURATOR_BATCH_SPEC.contract}


def test_an_unpinned_pack_prompt_resolves_to_the_in_code_baseline(tmp_path):
    resolved = PromptArtifactStore(tmp_path).resolve(
        CURATOR_PACK_SPEC, executor_model='sonnet', harness_version=PACK_PROMPT_HARNESS_VERSION,
    )
    assert resolved.source == 'in_code'
    assert resolved.text == CURATOR_PACK_SPEC.in_code_constant


def _property_names(schema: Any) -> set[str]:
    if isinstance(schema, dict):
        own = set(schema.get('properties', {}))
        return own.union(*(_property_names(value) for value in schema.values()))
    if isinstance(schema, list):
        return set().union(*(_property_names(value) for value in schema))
    return set()


def test_the_output_schema_asks_for_prose_and_member_ids_only():
    schema = CURATOR_PACK_OUTPUT_SCHEMA
    group = schema['properties']['groups']['items']
    assert {'verdict', 'groups', 'justification'} <= set(schema['required'])
    assert schema['properties']['verdict']['enum'] == ['pack', 'split', 'reject']
    assert set(group['properties']) == {'member_ids', 'title', 'description', 'details', 'justification'}
    assert group['properties']['member_ids'] == {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 2}
    assert _property_names(schema).isdisjoint(FORBIDDEN_FIELDS)


def _model_group(*member_ids: int) -> PackGroup:
    return PackGroup.model_validate(_group(*member_ids))


def test_verdict_models_are_frozen():
    group: Any = _model_group(1, 2)
    verdict: Any = PackVerdict(verdict='pack', groups=(group,), justification='j')
    with pytest.raises(ValidationError):
        group.title = 'renamed'
    with pytest.raises(ValidationError):
        verdict.verdict = 'reject'


def test_verdict_models_refuse_extra_keys():
    with pytest.raises(ValidationError):
        PackGroup.model_validate(_group(1, 2, survivor=1))
    with pytest.raises(ValidationError):
        PackVerdict.model_validate({**_raw('reject'), 'priority': 'low'})


@pytest.mark.parametrize('member_ids', [(1,), (), (3, 3)])
def test_a_group_needs_two_distinct_members(member_ids):
    with pytest.raises(ValidationError):
        _model_group(*member_ids)


def test_a_group_sorts_its_member_ids():
    assert _model_group(3, 1, 2).member_ids == (1, 2, 3)


@pytest.mark.parametrize(
    ('verdict', 'groups', 'degraded'),
    [
        ('reject', ((1, 2),), None),
        ('pack', (), None),
        ('pack', ((1, 2), (3, 4)), None),
        ('split', (), None),
        ('split', ((1, 2), (2, 3)), None),
        ('pack', ((1, 2),), VerdictDegrade.SCHEMA_INVALID),
    ],
)
def test_a_verdict_refuses_an_incoherent_shape(verdict, groups, degraded):
    with pytest.raises(ValidationError):
        PackVerdict(
            verdict=verdict,
            groups=tuple(_model_group(*ids) for ids in groups),
            justification='j',
            degraded=degraded,
        )


def test_a_whole_pack_verdict_parses_with_sorted_member_ids():
    parsed = parse_pack_verdict(_raw('pack', _group(5, 3, 1, 2, 4)), PACK)
    assert parsed.verdict == 'pack'
    assert [g.member_ids for g in parsed.groups] == [(1, 2, 3, 4, 5)]
    assert parsed.degraded is None


def test_a_split_verdict_leaves_members_outside_every_group():
    parsed = parse_pack_verdict(_raw('split', _group(1, 2), _group(3, 4)), PACK)
    assert parsed.verdict == 'split'
    assert [g.member_ids for g in parsed.groups] == [(1, 2), (3, 4)]
    assert parsed.degraded is None


def test_a_reject_verdict_parses_without_groups():
    parsed = parse_pack_verdict(_raw('reject', justification='rule 2: lock widens'), PACK)
    assert (parsed.verdict, parsed.groups, parsed.justification) == ('reject', (), 'rule 2: lock widens')
    assert parsed.degraded is None


@pytest.mark.parametrize(
    ('raw', 'degraded'),
    [
        (_raw('pack', _group(1, 2, 3, 4, 99)), VerdictDegrade.UNKNOWN_MEMBER),
        (_raw('split', _group(1, 3), _group(3, 4)), VerdictDegrade.OVERLAPPING_GROUPS),
        (_raw('split', _group(3, 3, 4)), VerdictDegrade.OVERLAPPING_GROUPS),
        (_raw('split', _group(1, 2), _group(3)), VerdictDegrade.GROUP_TOO_SMALL),
        (_raw('pack', _group(1, 2, 3, 4)), VerdictDegrade.PACK_NOT_WHOLE),
        (_raw('split', _group(1, 2, 3, 4, 5)), VerdictDegrade.SPLIT_NOT_PROPER),
        (_raw('split'), VerdictDegrade.SPLIT_NOT_PROPER),
        (_raw('reject', _group(1, 2)), VerdictDegrade.REJECT_WITH_GROUPS),
    ],
)
def test_a_verdict_breaking_the_pack_degrades_to_reject(raw, degraded):
    parsed = parse_pack_verdict(raw, PACK)
    assert (parsed.verdict, parsed.groups, parsed.degraded) == ('reject', (), degraded)
    assert parsed.justification == raw['justification']


JUNK = [
    {},
    _raw('maybe'),
    {'verdict': 'pack', 'groups': 'all of them', 'justification': 'j'},
    _raw('pack', _group('1', '2', '3', '4', '5')),
    _raw('pack', _group(1, 2, 3, 4, 5, title='   ')),
    _raw('pack', {'member_ids': [1, 2, 3, 4, 5]}),
    {'verdict': 'reject', 'groups': [], 'justification': None},
    [_raw('reject')],
    None,
    'pack',
]


@pytest.mark.parametrize('raw', JUNK)
def test_a_malformed_verdict_degrades_to_a_schema_invalid_reject(raw):
    parsed = parse_pack_verdict(raw, PACK)
    assert isinstance(parsed, PackVerdict)
    assert (parsed.verdict, parsed.groups, parsed.degraded) == ('reject', (), VerdictDegrade.SCHEMA_INVALID)


def test_a_long_justification_is_clipped_not_degraded():
    parsed = parse_pack_verdict(_raw('reject', justification='j' * 800), PACK)
    assert parsed.degraded is None
    assert parsed.justification == 'j' * MAX_JUSTIFICATION_CHARS == 'j' * 500


def test_a_long_group_justification_is_clipped_not_degraded():
    parsed = parse_pack_verdict(_raw('pack', _group(1, 2, 3, 4, 5, justification='g' * 800)), PACK)
    assert parsed.degraded is None
    assert parsed.groups[0].justification == 'g' * 500


def _rendered(*, band: Literal['narrow'] | None) -> tuple[PackMember, str, str, str]:
    member = _member(band=band)
    description, details = 'D' * 1999 + 'd', 'E' * 2999 + 'e'
    rendered = render_pack_member(
        member,
        priority='low',
        description=description,
        details=details,
        items=[observed_item(summary='fix the thing', files=['a/x.py'], origin_task='1')],
        desc_cap=100,
        details_cap=100,
    )
    return member, description, details, rendered


def test_a_rendered_member_shows_what_the_judge_needs():
    member, description, details, rendered = _rendered(band='narrow')
    for shown in ('4242', 'review_suggestion', member.title, 'low', '7', 'narrow', 'a/x.py', 'b/y.py',
                  'fix the thing', clip_for_prompt(description, 100), clip_for_prompt(details, 100)):
        assert shown in rendered
    assert description not in rendered
    assert details not in rendered


def test_a_rendered_member_names_an_unstamped_band():
    _, _, _, rendered = _rendered(band=None)
    assert 'band=unstamped' in rendered
