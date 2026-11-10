"""``metadata.trigger_chain``: the chain record's shape, registration, blessing and capabilities.

PRD plans/completion-driven-triggers-prd.md C2. The model owns the shape (the
role and skill vocabularies, the bare skill spelling, the run_id prefix); the
carrier rules live in the fused-memory submit guard and are not tested here.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.task_metadata import (
    TASK_METADATA_CAPABILITIES,
    TRIGGER_CHAIN_KEY,
    BeforeDone,
    TriggerChain,
    is_known_metadata_key,
    parse_metadata,
    registered_metadata_submodel,
)

SKILLS = ('review', 'hotspot-survey', 'review-all')


def _chain(**fields: object) -> dict:
    return {
        'role': 'completion_gate',
        'skill': 'review',
        'run_id': 'review-dark_factory-20261004',
        **fields,
    }


class TestTriggerChainModel:
    def test_completion_gate_defaults_superseded_by_to_none(self) -> None:
        chain = TriggerChain.model_validate(_chain())
        assert chain.role == 'completion_gate'
        assert chain.superseded_by is None

    def test_superseded_human_gate_is_accepted(self) -> None:
        chain = TriggerChain.model_validate(
            _chain(role='human_gate', superseded_by='review-dark_factory-20261102')
        )
        assert chain.superseded_by == 'review-dark_factory-20261102'

    @pytest.mark.parametrize('skill', SKILLS)
    def test_every_attended_skill_is_accepted(self, skill: str) -> None:
        chain = TriggerChain.model_validate(_chain(skill=skill, run_id=f'{skill}-x'))
        assert chain.skill == skill

    @pytest.mark.parametrize(
        ('slashed', 'bare'), [('/review', 'review'), ('/hotspot-survey', 'hotspot-survey')]
    )
    def test_leading_slash_names_the_bare_skill(self, slashed: str, bare: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            TriggerChain.model_validate(_chain(skill=slashed, run_id=f'{bare}-x'))
        assert (
            f'trigger_chain.skill={slashed!r} must be the bare skill name {bare!r} '
            '(no leading slash)'
        ) in str(excinfo.value)

    @pytest.mark.parametrize(
        'fields',
        [{'skill': 'census', 'run_id': 'census-x'}, {'role': 'gate'}],
        ids=['unknown-skill', 'unknown-role'],
    )
    def test_unknown_vocabulary_is_refused(self, fields: dict) -> None:
        with pytest.raises(ValidationError):
            TriggerChain.model_validate(_chain(**fields))

    @pytest.mark.parametrize('run_id', ['hotspot-survey-x', 'review'])
    def test_run_id_must_carry_the_skill_prefix(self, run_id: str) -> None:
        with pytest.raises(ValidationError) as excinfo:
            TriggerChain.model_validate(_chain(skill='review', run_id=run_id))
        message = str(excinfo.value)
        assert 'run_id' in message
        assert repr(run_id) in message
        assert "'review-'" in message

    def test_a_longer_skills_run_is_not_a_run_of_its_prefix(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            TriggerChain.model_validate(_chain(skill='review', run_id='review-all-x'))
        assert (
            "trigger_chain.run_id='review-all-x' is a run of 'review-all', not of 'review'"
        ) in str(excinfo.value)

    def test_the_longer_skill_accepts_its_own_run(self) -> None:
        chain = TriggerChain.model_validate(_chain(skill='review-all', run_id='review-all-x'))
        assert chain.skill == 'review-all'

    @pytest.mark.parametrize(
        ('superseded_by', 'expected'),
        [
            ('', "trigger_chain.superseded_by='' must start with 'review-'"),
            ('hotspot-survey-x', "trigger_chain.superseded_by='hotspot-survey-x' must start with 'review-'"),
            ('review-all-x', "trigger_chain.superseded_by='review-all-x' is a run of 'review-all'"),
        ],
        ids=['empty', 'other-skill', 'longer-skill'],
    )
    def test_superseded_by_must_name_a_run_of_the_same_skill(
        self, superseded_by: str, expected: str
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            TriggerChain.model_validate(_chain(superseded_by=superseded_by))
        assert expected in str(excinfo.value)

    @pytest.mark.parametrize('missing', ['run_id', 'role'])
    def test_required_fields(self, missing: str) -> None:
        raw = _chain()
        del raw[missing]
        with pytest.raises(ValidationError):
            TriggerChain.model_validate(raw)


class TestTriggerChainRegistration:
    def test_key_spelling(self) -> None:
        assert TRIGGER_CHAIN_KEY == 'trigger_chain'

    def test_the_key_is_registered_to_the_model(self) -> None:
        assert registered_metadata_submodel(TRIGGER_CHAIN_KEY) is TriggerChain

    def test_an_unregistered_key_has_no_model(self) -> None:
        assert registered_metadata_submodel('x_never_registered') is None

    def test_write_boundary_refuses_a_slashed_skill(self) -> None:
        with pytest.raises(ValidationError, match='no leading slash'):
            parse_metadata(
                {'trigger_chain': _chain(skill='/review')}, direction='write', enforce=True
            )

    def test_write_boundary_types_a_valid_chain(self) -> None:
        model, warnings = parse_metadata(
            {'trigger_chain': _chain()}, direction='write', enforce=True
        )
        assert warnings == []
        assert model.model_extra is not None
        assert isinstance(model.model_extra['trigger_chain'], TriggerChain)

    def test_list_shaped_chain_is_refused_by_the_cardinality_gate(self) -> None:
        with pytest.raises(TypeError, match='cardinality'):
            parse_metadata({'trigger_chain': [_chain()]}, direction='write', enforce=True)


class TestBlessing:
    @pytest.mark.parametrize(
        ('key', 'value'),
        [
            ('trigger_chain', _chain()),
            ('predicate_recheck', {'streak': 1}),
            ('predicate_verdict', {'last_checked_at': '2026-11-02T00:00:00+00:00'}),
        ],
    )
    def test_key_is_known_and_silent_on_read(self, key: str, value: dict) -> None:
        assert is_known_metadata_key(key)
        _model, warnings = parse_metadata({key: value}, direction='read')
        assert [w for w in warnings if w.field == key and w.code == 'unknown_key'] == []


class TestCapabilities:
    def test_the_capability_set(self) -> None:
        expected = {'before_done.recheck_secs', 'before_done.script_root', 'trigger_chain'}
        assert set(TASK_METADATA_CAPABILITIES) == expected
        assert isinstance(TASK_METADATA_CAPABILITIES, frozenset)

    def test_every_capability_names_something_this_schema_declares(self) -> None:
        for capability in TASK_METADATA_CAPABILITIES:
            head, _, field = capability.partition('.')
            if field:
                assert head == 'before_done', capability
                assert field in BeforeDone.model_fields, capability
            else:
                assert registered_metadata_submodel(capability) is not None, capability
