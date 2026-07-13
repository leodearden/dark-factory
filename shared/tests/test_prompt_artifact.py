"""Tests for shared.prompt_artifact — prompt-artifact loader (PRD plans/tier1-prompt-optimization-prd.md T1).

Built bottom-up in TDD order (see task 2492's plan.json):
  - TestArtifactProvenance: the 8-field provenance sidecar model.
  - TestCompose: compose_prompt() / PromptSpec.in_code_constant.
  - TestResolveFallback: PromptArtifactStore.resolve() with nothing pinned.
  - TestPinAndResolve: pin() then resolve() composes CONTRACT + artifact-heuristics.
  - TestUnpinRollback: unpin() is the rollback lever, restores the in-code constant.
  - TestKeyIsolationAndPathSafety: per-model/per-harness isolation + traversal safety.
  - TestFailSafeUnverifiablePin: half-written/corrupt pins fall back to in-code.
  - TestAtomicWriteText: _atomic_write_text's failure/cleanup path.
  - TestDefaultArtifactsRoot: default_artifacts_root() env override + monorepo walk-up.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from shared import prompt_artifact
from shared.prompt_artifact import (
    ArtifactProvenance,
    PromptArtifactStore,
    PromptSpec,
    compose_prompt,
    default_artifacts_root,
)


def _provenance_kwargs(**overrides: Any) -> dict[str, Any]:
    # dict[str, Any] is intentional: ArtifactProvenance's fields have
    # heterogeneous types and spreading a concrete dict[str, <union>] defeats
    # pyright's per-parameter type checking at the ** call site.
    kwargs: dict[str, Any] = dict(
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

    def test_extra_provenance_field_survives_pin_disk_round_trip(self, tmp_path):
        """test_unknown_extra_field_is_preserved (TestArtifactProvenance) only
        checks in-memory model_dump() preservation. This locks the same
        guarantee through a full pin() -> on-disk provenance.json -> fresh
        PromptArtifactStore -> resolve()/read_provenance() round trip, so a
        future change to how the sidecar is serialized/reloaded (e.g.
        switching model_dump_json args, or adding a field filter) cannot
        silently drop forward-compat fields without failing a test.
        """
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        provenance = ArtifactProvenance(
            **_provenance_kwargs(harness_version='v1', candidate_id='cand-7')
        )  # type: ignore[call-arg]
        store.pin('reviewer', 'claude-opus-4', 'v1', heuristics='h', provenance=provenance)

        # A fresh store instance -- nothing carried over in memory -- so both
        # reads below come purely from what pin() persisted to disk.
        reloaded_store = PromptArtifactStore(tmp_path)

        reloaded_provenance = reloaded_store.read_provenance('reviewer', 'claude-opus-4', 'v1')
        assert reloaded_provenance is not None
        assert reloaded_provenance.model_dump()['candidate_id'] == 'cand-7'

        resolved = reloaded_store.resolve(
            spec, executor_model='claude-opus-4', harness_version='v1'
        )
        assert resolved.provenance is not None
        assert resolved.provenance.model_dump()['candidate_id'] == 'cand-7'

    def test_pin_rejects_provenance_harness_version_mismatch(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        provenance = ArtifactProvenance(**_provenance_kwargs(harness_version='v1'))

        with pytest.raises(ValueError):
            store.pin(
                'reviewer', 'claude-opus-4', 'v2', heuristics='h', provenance=provenance
            )

    def test_read_provenance_returns_none_when_nothing_pinned(self, tmp_path):
        store = PromptArtifactStore(tmp_path)

        assert store.read_provenance('reviewer', 'claude-opus-4', 'v1') is None

    def test_repin_overwrites_previous_heuristics_and_provenance(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        old_provenance = ArtifactProvenance(
            **_provenance_kwargs(harness_version='v1', git_sha='old-sha')
        )
        store.pin(
            'reviewer', 'claude-opus-4', 'v1', heuristics='old heuristics',
            provenance=old_provenance,
        )

        new_provenance = ArtifactProvenance(
            **_provenance_kwargs(harness_version='v1', git_sha='new-sha')
        )
        store.pin(
            'reviewer', 'claude-opus-4', 'v1', heuristics='new heuristics',
            provenance=new_provenance,
        )

        resolved = store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')

        # The second pin's heuristics + provenance win outright — no trace of
        # the first pin survives.
        assert resolved.text == compose_prompt(spec.contract, 'new heuristics')
        assert resolved.provenance == new_provenance
        assert resolved.source == 'artifact'
        assert store.read_provenance('reviewer', 'claude-opus-4', 'v1') == new_provenance

    def test_first_pin_never_exposes_heuristics_before_provenance_lands(
        self, tmp_path, monkeypatch
    ):
        """Regression locking in "provenance written last" for a *first-time*
        pin, not just a re-pin: if a reader observes the key dir right after
        heuristics.txt lands but before provenance.json is written, it must
        see the fail-safe in-code fallback — never an orphan heuristics file
        treated as pinned.
        """
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()

        observed = []
        real_atomic_write_text = prompt_artifact._atomic_write_text

        def spying_write(path, text):
            real_atomic_write_text(path, text)
            if path.name == 'heuristics.txt':
                # Snapshot resolve() right after heuristics.txt lands but
                # before provenance.json is written. There is no prior pin
                # at this key, so an interrupted first pin must resolve to
                # in_code, not merely "the previous artifact".
                observed.append(
                    store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')
                )

        monkeypatch.setattr(prompt_artifact, '_atomic_write_text', spying_write)

        provenance = ArtifactProvenance(**_provenance_kwargs(harness_version='v1'))
        store.pin(
            'reviewer', 'claude-opus-4', 'v1', heuristics='first heuristics',
            provenance=provenance,
        )

        assert len(observed) == 1
        mid_write = observed[0]
        assert mid_write.source == 'in_code'
        assert mid_write.text == spec.in_code_constant
        assert mid_write.provenance is None

    def test_repin_never_exposes_new_heuristics_with_stale_provenance(self, tmp_path, monkeypatch):
        """Regression for the re-pin race: a reader interleaved between the
        two writes of a re-pin must never see NEW heuristics stitched to the
        OLD provenance sidecar — it must see the fail-safe in-code fallback
        instead (pin() now drops the stale provenance.json before writing the
        new heuristics.txt).
        """
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        old_provenance = ArtifactProvenance(
            **_provenance_kwargs(harness_version='v1', git_sha='old-sha')
        )
        store.pin(
            'reviewer', 'claude-opus-4', 'v1', heuristics='old heuristics',
            provenance=old_provenance,
        )

        observed = []
        real_atomic_write_text = prompt_artifact._atomic_write_text

        def spying_write(path, text):
            real_atomic_write_text(path, text)
            if path.name == 'heuristics.txt':
                # Snapshot resolve() right after the new heuristics land but
                # before the new provenance.json is written.
                observed.append(
                    store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')
                )

        monkeypatch.setattr(prompt_artifact, '_atomic_write_text', spying_write)

        new_provenance = ArtifactProvenance(
            **_provenance_kwargs(harness_version='v1', git_sha='new-sha')
        )
        store.pin(
            'reviewer', 'claude-opus-4', 'v1', heuristics='new heuristics',
            provenance=new_provenance,
        )

        assert len(observed) == 1
        mid_write = observed[0]
        assert mid_write.source == 'in_code'
        assert mid_write.text == spec.in_code_constant
        assert mid_write.provenance is None

    def test_failed_repin_provenance_write_reverts_to_baseline(self, tmp_path, monkeypatch):
        """Documents pin()'s "failed re-pin can silently revert to baseline"
        hazard as an intentional, locked contract: if the *second* atomic
        write of a re-pin (provenance.json) raises after the first
        (heuristics.txt) already landed, the key is left holding the NEW
        heuristics with NO provenance -- the same on-disk shape resolve()
        treats as not-pinned. The previously-good pin is therefore silently
        replaced by the in-code fallback rather than preserved. Pairs with
        the TOCTOU/race regression tests above; this one exercises the
        write-failure path instead of a reader interleaving.
        """
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        old_provenance = ArtifactProvenance(
            **_provenance_kwargs(harness_version='v1', git_sha='old-sha')
        )
        store.pin(
            'reviewer', 'claude-opus-4', 'v1', heuristics='old heuristics',
            provenance=old_provenance,
        )
        # Sanity: the pin from setup resolves as a genuine artifact before
        # the failure injection below.
        before = store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')
        assert before.source == 'artifact'

        real_atomic_write_text = prompt_artifact._atomic_write_text

        def failing_on_provenance_write(path, text):
            if path.name == 'provenance.json':
                raise OSError('simulated disk-full on provenance write')
            real_atomic_write_text(path, text)

        monkeypatch.setattr(prompt_artifact, '_atomic_write_text', failing_on_provenance_write)

        new_provenance = ArtifactProvenance(
            **_provenance_kwargs(harness_version='v1', git_sha='new-sha')
        )
        with pytest.raises(OSError):
            store.pin(
                'reviewer', 'claude-opus-4', 'v1', heuristics='new heuristics',
                provenance=new_provenance,
            )

        monkeypatch.setattr(prompt_artifact, '_atomic_write_text', real_atomic_write_text)
        resolved = store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')

        assert resolved.source == 'in_code'
        assert resolved.text == spec.in_code_constant
        assert resolved.provenance is None


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

    def test_unpin_prunes_now_empty_parent_directories(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        provenance = ArtifactProvenance(**_provenance_kwargs(harness_version='v1'))
        store.pin('reviewer', 'claude-opus-4', 'v1', heuristics='h', provenance=provenance)
        key_dir = store._key_dir('reviewer', 'claude-opus-4', 'v1')
        model_dir = key_dir.parent
        prompt_dir = key_dir.parent.parent
        assert model_dir.exists()
        assert prompt_dir.exists()

        store.unpin('reviewer', 'claude-opus-4', 'v1')

        # The per-model and per-prompt_id levels are pruned once empty...
        assert not model_dir.exists()
        assert not prompt_dir.exists()
        # ...but root itself is left alone.
        assert tmp_path.exists()

    def test_unpin_does_not_prune_a_directory_still_holding_a_sibling_key(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        provenance = ArtifactProvenance(**_provenance_kwargs(harness_version='v1'))
        store.pin('reviewer', 'opus', 'v1', heuristics='opus-h', provenance=provenance)
        store.pin('reviewer', 'sonnet', 'v1', heuristics='sonnet-h', provenance=provenance)

        store.unpin('reviewer', 'opus', 'v1')

        # 'reviewer' survives because 'sonnet' is still pinned under it.
        prompt_dir = store._key_dir('reviewer', 'opus', 'v1').parent.parent
        assert prompt_dir.exists()
        sonnet_key_dir = store._key_dir('reviewer', 'sonnet', 'v1')
        assert sonnet_key_dir.exists()


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


class TestFailSafeUnverifiablePin:
    """Half-written or corrupt on-disk state must never surface as a pinned artifact.

    Each case constructs the half-pinned state directly on tmp_path (bypassing
    ``pin()``, which always writes a coherent pair) and asserts ``resolve()``
    falls back to the in-code constant rather than raising or returning an
    unverified artifact.
    """

    def test_heuristics_without_provenance_falls_back_to_in_code(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        key_dir = store._key_dir('reviewer', 'claude-opus-4', 'v1')
        key_dir.mkdir(parents=True)
        (key_dir / 'heuristics.txt').write_text('orphan heuristics', encoding='utf-8')
        # No provenance.json written.

        resolved = store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')

        assert resolved.text == spec.in_code_constant
        assert resolved.provenance is None
        assert resolved.source == 'in_code'

    def test_provenance_without_heuristics_falls_back_to_in_code(self, tmp_path):
        """The symmetric orphan to the heuristics-without-provenance case: a
        schema-valid provenance.json with NO heuristics.txt. resolve() short
        -circuits on heuristics_path.exists() before ever loading provenance,
        so this is a distinct code path that must independently fail safe.
        """
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        key_dir = store._key_dir('reviewer', 'claude-opus-4', 'v1')
        key_dir.mkdir(parents=True)
        (key_dir / 'provenance.json').write_text(
            json.dumps(_provenance_kwargs(harness_version='v1')), encoding='utf-8'
        )
        # No heuristics.txt written.

        resolved = store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')

        assert resolved.text == spec.in_code_constant
        assert resolved.provenance is None
        assert resolved.source == 'in_code'

    def test_provenance_with_mismatched_harness_version_falls_back_to_in_code(self, tmp_path):
        """A schema-valid provenance.json recorded for a different
        harness_version than this key (e.g. a sidecar manually relocated or
        hand-crafted outside pin(), which always enforces this equality)
        must not be trusted — both resolve() and read_provenance() fail safe.
        """
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        key_dir = store._key_dir('reviewer', 'claude-opus-4', 'v1')
        key_dir.mkdir(parents=True)
        (key_dir / 'heuristics.txt').write_text('some heuristics', encoding='utf-8')
        (key_dir / 'provenance.json').write_text(
            json.dumps(_provenance_kwargs(harness_version='v2')), encoding='utf-8'
        )

        resolved = store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')

        assert resolved.text == spec.in_code_constant
        assert resolved.provenance is None
        assert resolved.source == 'in_code'
        assert store.read_provenance('reviewer', 'claude-opus-4', 'v1') is None

    def test_corrupt_provenance_json_falls_back_to_in_code(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        key_dir = store._key_dir('reviewer', 'claude-opus-4', 'v1')
        key_dir.mkdir(parents=True)
        (key_dir / 'heuristics.txt').write_text('some heuristics', encoding='utf-8')
        (key_dir / 'provenance.json').write_text('{not valid json', encoding='utf-8')

        resolved = store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')

        assert resolved.text == spec.in_code_constant
        assert resolved.provenance is None
        assert resolved.source == 'in_code'

    def test_provenance_missing_required_field_falls_back_to_in_code(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        key_dir = store._key_dir('reviewer', 'claude-opus-4', 'v1')
        key_dir.mkdir(parents=True)
        (key_dir / 'heuristics.txt').write_text('some heuristics', encoding='utf-8')
        incomplete = _provenance_kwargs(harness_version='v1')
        del incomplete['git_sha']  # a schema-valid ArtifactProvenance requires this
        (key_dir / 'provenance.json').write_text(json.dumps(incomplete), encoding='utf-8')

        resolved = store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')

        assert resolved.text == spec.in_code_constant
        assert resolved.provenance is None
        assert resolved.source == 'in_code'

    def test_read_provenance_returns_none_for_corrupt_provenance(self, tmp_path):
        store = PromptArtifactStore(tmp_path)
        key_dir = store._key_dir('reviewer', 'claude-opus-4', 'v1')
        key_dir.mkdir(parents=True)
        (key_dir / 'provenance.json').write_text('{not valid json', encoding='utf-8')

        assert store.read_provenance('reviewer', 'claude-opus-4', 'v1') is None

    def test_resolve_falls_back_when_heuristics_vanishes_after_provenance_check(
        self, tmp_path, monkeypatch
    ):
        """Regression for a resolve() TOCTOU: heuristics.txt can vanish (e.g. a
        concurrent unpin(), or an external deletion) between the provenance
        validation and the heuristics read. resolve() must fail safe to the
        in-code constant rather than let FileNotFoundError propagate — it is
        documented to never raise.
        """
        store = PromptArtifactStore(tmp_path)
        spec = _make_spec()
        provenance = ArtifactProvenance(**_provenance_kwargs(harness_version='v1'))
        store.pin('reviewer', 'claude-opus-4', 'v1', heuristics='h', provenance=provenance)
        heuristics_path = store._key_dir('reviewer', 'claude-opus-4', 'v1') / 'heuristics.txt'

        real_load_valid_provenance = prompt_artifact._load_valid_provenance

        def racing_load_valid_provenance(path, *, expected_harness_version=None):
            result = real_load_valid_provenance(
                path, expected_harness_version=expected_harness_version
            )
            # Simulate a concurrent unpin()/external deletion landing right
            # after the provenance check succeeds but before resolve() reads
            # the heuristics file.
            heuristics_path.unlink()
            return result

        monkeypatch.setattr(
            prompt_artifact, '_load_valid_provenance', racing_load_valid_provenance
        )

        resolved = store.resolve(spec, executor_model='claude-opus-4', harness_version='v1')

        assert resolved.source == 'in_code'
        assert resolved.text == spec.in_code_constant
        assert resolved.provenance is None


class TestAtomicWriteText:
    """Direct coverage of the private write-then-replace helper's failure path."""

    def test_cleans_up_temp_file_and_leaves_original_untouched_on_failure(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / 'out.txt'
        target.write_text('original', encoding='utf-8')

        def boom(*_args, **_kwargs):
            raise OSError('simulated write failure')

        monkeypatch.setattr('shared.prompt_artifact.os.replace', boom)

        with pytest.raises(OSError):
            prompt_artifact._atomic_write_text(target, 'new content')

        # The failed replace never landed: original contents survive...
        assert target.read_text(encoding='utf-8') == 'original'
        # ...and no leftover temp file was left behind next to it.
        leftover = [p.name for p in tmp_path.iterdir() if p.name != 'out.txt']
        assert leftover == []


def _build_fake_monorepo_module_file(root: Path) -> Path:
    """Create a fake ``orchestrator/``, ``fused-memory/``, ``shared/`` tree under
    *root* and return the path a ``shared/src/shared/prompt_artifact.py`` would
    live at inside it.

    Lets tests drive :func:`default_artifacts_root`'s walk-up deterministically
    (via monkeypatching the module's ``__file__``) instead of depending on the
    real checkout's physical layout, which breaks when the package is
    installed/tested outside the monorepo (e.g. a wheel install).
    """
    (root / 'orchestrator').mkdir(parents=True)
    (root / 'fused-memory').mkdir(parents=True)
    package_dir = root / 'shared' / 'src' / 'shared'
    package_dir.mkdir(parents=True)
    module_file = package_dir / 'prompt_artifact.py'
    module_file.write_text('', encoding='utf-8')
    return module_file


class TestDefaultArtifactsRoot:
    def test_env_override_wins(self, monkeypatch, tmp_path):
        custom = tmp_path / 'custom-artifacts'
        monkeypatch.setenv('DARK_FACTORY_PROMPT_ARTIFACTS', str(custom))

        assert default_artifacts_root() == Path(custom)

    def test_unset_env_anchors_at_monorepo_root(self, monkeypatch, tmp_path):
        monkeypatch.delenv('DARK_FACTORY_PROMPT_ARTIFACTS', raising=False)
        fake_monorepo = tmp_path / 'fake-monorepo'
        module_file = _build_fake_monorepo_module_file(fake_monorepo)
        monkeypatch.setattr(prompt_artifact, '__file__', str(module_file))

        root = default_artifacts_root()

        assert root == fake_monorepo / 'data' / 'prompt_artifacts'

    def test_empty_string_env_override_falls_through_to_walkup(self, monkeypatch, tmp_path):
        # An empty string is falsy -> treated the same as "unset", not as an
        # explicit override of "use this (empty) path".
        monkeypatch.setenv('DARK_FACTORY_PROMPT_ARTIFACTS', '')
        fake_monorepo = tmp_path / 'fake-monorepo'
        module_file = _build_fake_monorepo_module_file(fake_monorepo)
        monkeypatch.setattr(prompt_artifact, '__file__', str(module_file))

        root = default_artifacts_root()

        assert root == fake_monorepo / 'data' / 'prompt_artifacts'

    def test_no_monorepo_marker_falls_back_to_cwd(self, monkeypatch, tmp_path):
        monkeypatch.delenv('DARK_FACTORY_PROMPT_ARTIFACTS', raising=False)
        # A module location with no orchestrator/fused-memory/shared marker
        # anywhere above it -> the walk-up exhausts every parent unsatisfied.
        isolated_module_file = tmp_path / 'isolated' / 'nested' / 'prompt_artifact.py'
        isolated_module_file.parent.mkdir(parents=True)
        isolated_module_file.write_text('', encoding='utf-8')
        monkeypatch.setattr(prompt_artifact, '__file__', str(isolated_module_file))
        fake_cwd = tmp_path / 'cwd-anchor'
        fake_cwd.mkdir()
        monkeypatch.chdir(fake_cwd)

        root = default_artifacts_root()

        assert root == fake_cwd / 'data' / 'prompt_artifacts'
