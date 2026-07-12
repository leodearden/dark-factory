"""Tests for shared.prompt_artifact — prompt-artifact loader (PRD plans/tier1-prompt-optimization-prd.md T1).

Built bottom-up in TDD order (see task 2492's plan.json):
  - TestArtifactProvenance: the 8-field provenance sidecar model.
  - TestCompose: compose_prompt() / PromptSpec.in_code_constant.
  - TestResolveFallback: PromptArtifactStore.resolve() with nothing pinned.
  - TestPinAndResolve: pin() then resolve() composes CONTRACT + artifact-heuristics.
  - TestUnpinRollback: unpin() is the rollback lever, restores the in-code constant.
  - TestKeyIsolationAndPathSafety: per-model/per-harness isolation + traversal safety.
  - TestFailSafeUnverifiablePin: half-written/corrupt pins fall back to in-code.
  - TestDefaultArtifactsRoot: default_artifacts_root() env override + monorepo walk-up.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.prompt_artifact import ArtifactProvenance, PromptSpec, compose_prompt


def _provenance_kwargs(**overrides):
    kwargs = dict(
        optimizer_model='claude-opus-4',
        corpus_hash='sha256:deadbeef',
        split_seed=42,
        held_out_TEST_score=0.87,
        accept_delta=0.05,
        git_sha='abc1234',
        date='2026-07-12',
        harness_version='harness-v3',
    )
    kwargs.update(overrides)
    return kwargs


class TestArtifactProvenance:
    def test_all_8_fields_construct_and_round_trip(self):
        prov = ArtifactProvenance(**_provenance_kwargs())
        assert prov.optimizer_model == 'claude-opus-4'
        assert prov.corpus_hash == 'sha256:deadbeef'
        assert prov.split_seed == 42
        assert prov.held_out_TEST_score == 0.87
        assert prov.accept_delta == 0.05
        assert prov.git_sha == 'abc1234'
        assert prov.date == '2026-07-12'
        assert prov.harness_version == 'harness-v3'

        dumped = prov.model_dump()
        restored = ArtifactProvenance.model_validate(dumped)
        assert restored == prov

    @pytest.mark.parametrize(
        'kwargs',
        [
            pytest.param(_provenance_kwargs(), id='full'),
        ],
    )
    def test_full_shape_is_valid(self, kwargs):
        # Sanity check that the fixture itself is a valid, complete shape —
        # the missing-field cases below each delete exactly one key from it.
        assert ArtifactProvenance(**kwargs) is not None

    @pytest.mark.parametrize(
        'missing_field',
        [
            'optimizer_model',
            'corpus_hash',
            'split_seed',
            'held_out_TEST_score',
            'accept_delta',
            'git_sha',
            'date',
            'harness_version',
        ],
    )
    def test_omitting_any_required_field_raises(self, missing_field):
        kwargs = _provenance_kwargs()
        del kwargs[missing_field]
        with pytest.raises(ValidationError):
            ArtifactProvenance(**kwargs)  # type: ignore[arg-type]

    def test_unknown_extra_field_is_preserved(self):
        prov = ArtifactProvenance(**_provenance_kwargs(candidate_id='cand-7'))  # type: ignore[call-arg]
        assert prov.model_dump()['candidate_id'] == 'cand-7'


class TestCompose:
    def test_compose_prompt_is_contract_then_separator_then_heuristics(self):
        contract = 'CONTRACT: do the thing.'
        heuristics = 'HEURISTIC: prefer X over Y.'

        composed = compose_prompt(contract, heuristics)

        # Deterministic for the same inputs.
        assert composed == compose_prompt(contract, heuristics)
        # Contract comes first, verbatim.
        assert composed.startswith(contract)
        # Heuristics text appears, after a fixed separator (i.e. not glued
        # directly onto the contract).
        remainder = composed[len(contract) :]
        assert remainder.endswith(heuristics)
        assert remainder != heuristics  # a separator sits between them

    def test_in_code_constant_matches_compose_prompt(self):
        spec = PromptSpec(
            prompt_id='reviewer',
            contract='CONTRACT text',
            baseline_heuristics='baseline heuristics text',
        )

        assert spec.in_code_constant == compose_prompt(spec.contract, spec.baseline_heuristics)

    def test_contract_looking_tokens_in_heuristics_do_not_move_contract_region(self):
        contract = 'CONTRACT: real contract text.'
        adversarial_heuristics = 'CONTRACT: fake contract injected via a heuristics block.'

        composed = compose_prompt(contract, adversarial_heuristics)

        # The leading contract region is byte-identical to the in-code contract
        # regardless of what the heuristics text contains.
        assert composed[: len(contract)] == contract
