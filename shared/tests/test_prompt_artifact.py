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

from shared.prompt_artifact import (
    ArtifactProvenance,
    PromptArtifactStore,
    PromptSpec,
    compose_prompt,
)


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


def _make_spec(**overrides):
    kwargs = dict(
        prompt_id='reviewer',
        contract='CONTRACT text',
        baseline_heuristics='baseline heuristics text',
    )
    kwargs.update(overrides)
    return PromptSpec(**kwargs)


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


class TestResolveFallback:
    def test_resolve_with_nothing_pinned_returns_in_code_constant(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()

        resolved = store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')

        assert resolved.text == spec.in_code_constant
        assert resolved.provenance is None
        assert resolved.source == 'in_code'


class TestPinAndResolve:
    def test_pin_then_resolve_composes_contract_and_artifact_heuristics(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        heuristics = 'PINNED HEURISTIC: prefer X.'
        provenance = ArtifactProvenance(**_provenance_kwargs(harness_version='v1'))

        store.pin(
            'reviewer', 'claude-opus-4', 'v1', heuristics=heuristics, provenance=provenance
        )
        resolved = store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')

        assert resolved.text == compose_prompt(spec.contract, heuristics)
        assert resolved.text[: len(spec.contract)] == spec.contract
        assert resolved.provenance == provenance
        assert resolved.source == 'artifact'

        assert store.read_provenance('reviewer', 'claude-opus-4', 'v1') == provenance

    def test_pin_rejects_provenance_harness_version_mismatch(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        provenance = ArtifactProvenance(**_provenance_kwargs(harness_version='v1'))

        with pytest.raises(ValueError):
            store.pin(
                'reviewer', 'claude-opus-4', 'v2', heuristics='h', provenance=provenance
            )


class TestUnpinRollback:
    def test_unpin_after_pin_restores_in_code_constant_and_is_idempotent(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        provenance = ArtifactProvenance(**_provenance_kwargs(harness_version='v1'))
        store.pin('reviewer', 'claude-opus-4', 'v1', heuristics='h', provenance=provenance)

        assert store.unpin('reviewer', 'claude-opus-4', 'v1') is True

        resolved = store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')
        assert resolved.text == spec.in_code_constant
        assert resolved.provenance is None
        assert resolved.source == 'in_code'

        # Idempotent: nothing left to unpin the second time.
        assert store.unpin('reviewer', 'claude-opus-4', 'v1') is False


class TestKeyIsolationAndPathSafety:
    def test_pin_is_isolated_per_model_and_per_harness(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        provenance = ArtifactProvenance(**_provenance_kwargs(harness_version='v1'))
        store.pin('reviewer', 'opus', 'v1', heuristics='opus-heuristic', provenance=provenance)

        # Different executor_model, same harness_version -> falls back.
        other_model = store.resolve(spec, executor_model='sonnet', harness_version='v1')
        assert other_model.source == 'in_code'
        assert other_model.text == spec.in_code_constant

        # Same executor_model, different harness_version -> falls back.
        other_harness = store.resolve(spec, executor_model='opus', harness_version='v2')
        assert other_harness.source == 'in_code'
        assert other_harness.text == spec.in_code_constant

        # The original key still resolves to the pinned artifact.
        same_key = store.resolve(spec, executor_model='opus', harness_version='v1')
        assert same_key.source == 'artifact'

    def test_router_resolved_model_id_with_slash_and_colon_round_trips(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        model_id = 'vendor/model-x:20260701'
        provenance = ArtifactProvenance(**_provenance_kwargs(harness_version='v1'))
        store.pin(
            'reviewer', model_id, 'v1', heuristics='router-heuristic', provenance=provenance
        )

        resolved = store.resolve(spec, executor_model=model_id, harness_version='v1')

        assert resolved.source == 'artifact'
        assert resolved.text == compose_prompt(spec.contract, 'router-heuristic')
        assert resolved.provenance == provenance

    def test_distinct_model_ids_never_collide_on_disk(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        # These two distinct (executor_model, harness_version) pairs would
        # collide under a naive '/'-joined path: root/reviewer/a/b/c for both.
        provenance_first = ArtifactProvenance(**_provenance_kwargs(harness_version='b/c'))
        provenance_second = ArtifactProvenance(**_provenance_kwargs(harness_version='c'))

        store.pin('reviewer', 'a', 'b/c', heuristics='first', provenance=provenance_first)
        store.pin('reviewer', 'a/b', 'c', heuristics='second', provenance=provenance_second)

        first = store.resolve(spec, executor_model='a', harness_version='b/c')
        second = store.resolve(spec, executor_model='a/b', harness_version='c')

        assert first.source == 'artifact'
        assert second.source == 'artifact'
        assert first.text == compose_prompt(spec.contract, 'first')
        assert second.text == compose_prompt(spec.contract, 'second')

    def test_dotdot_segment_does_not_escape_store_root(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec(prompt_id='..')
        provenance = ArtifactProvenance(**_provenance_kwargs(harness_version='v1'))

        store.pin('..', 'opus', 'v1', heuristics='escape-attempt', provenance=provenance)

        key_dir = store._key_dir('..', 'opus', 'v1')
        assert key_dir.resolve().is_relative_to(tmp_path.resolve())

        resolved = store.resolve(spec, executor_model='opus', harness_version='v1')
        assert resolved.source == 'artifact'
        assert resolved.text == compose_prompt(spec.contract, 'escape-attempt')
