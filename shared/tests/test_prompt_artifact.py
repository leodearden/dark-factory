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

from shared.prompt_artifact import ArtifactProvenance


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
