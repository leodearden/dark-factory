"""Tests for scripts/retro_stamp_topics.py — PRD leaf θ (task 3201).

The bounded retro topic/canonical stamping sweep.  Loaded via importlib so
the script (``scripts/`` is not a package and is not on PYTHONPATH) can be
tested without sys.path pollution — the same idiom as
``test_sweep_orphan_flag_markers.py`` / ``test_cleanup_count_snapshots.py``.

Every derivation function in the script is pure, so almost everything here
is a plain unit test; the single I/O boundary (an injected
``memory_service``) is exercised with ``AsyncMock``.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from fused_memory import topic_slug as topic_slug_module

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'retro_stamp_topics.py'


def _load_module() -> types.ModuleType:
    """Load retro_stamp_topics.py from its file path.

    The module is registered in sys.modules under its name so that
    reflection-based decorators work correctly.
    """
    mod_name = 'retro_stamp_topics'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()


# ===========================================================================
# derive_topic_slug — the fold from a raw value to ε's slug shape
# ===========================================================================

class TestDeriveTopicSlug:
    """``derive_topic_slug(value) -> str | None``.

    Folds a raw topic value into the shape
    :mod:`fused_memory.topic_slug` defines, or returns ``None`` when no
    honest fold exists.  Never guesses: a value that cannot conform is
    reported, not repaired into something plausible.
    """

    def test_conforming_slug_round_trips_unchanged(self):
        """A value already in ε's shape is returned byte-identical.

        This is what makes the sweep idempotent at the derivation layer:
        run two folds the target topic to itself, so ``compute_patch``
        sees no change to write.
        """
        assert _mod.derive_topic_slug('docs-prd-landing') == 'docs-prd-landing'
        assert _mod.derive_topic_slug('a') == 'a'
        assert _mod.derive_topic_slug('x1-2y') == 'x1-2y'

    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            # The single live dark_factory `canonical: true` record's topic.
            (
                'eval_worktree_plan_tools_missing',
                'eval-worktree-plan-tools-missing',
            ),
            # One of the five live reify `canonical: true` topics.
            (
                'merge_request_bare_task_id_branch_arg',
                'merge-request-bare-task-id-branch-arg',
            ),
        ],
    )
    def test_measured_live_snake_case_topics_fold_to_hyphens(
        self, raw: str, expected: str
    ):
        """The two measured snake_case live topics are exactly θ's job.

        ``eval_worktree_plan_tools_missing`` is the snake_case twin of the
        seeded cluster id ``eval-worktree-plan-tools-missing``; ε's
        enforcement note names this normalization as the precondition for
        flipping ``memory_metadata.enforce``.  These are not invented
        examples — they are the values the live corpus carries.
        """
        assert _mod.derive_topic_slug(raw) == expected

    @pytest.mark.parametrize(
        ('raw', 'expected'),
        [
            ('Docs-PRD-Landing', 'docs-prd-landing'),
            ('  docs-prd-landing  ', 'docs-prd-landing'),
            ('\tDocs PRD Landing\n', 'docs-prd-landing'),
            ('docs/prd/landing', 'docs-prd-landing'),
            ('docs -- prd  landing', 'docs-prd-landing'),
            ('--docs-prd-landing--', 'docs-prd-landing'),
            ('Docs_PRD__Landing!!!', 'docs-prd-landing'),
        ],
    )
    def test_case_whitespace_and_punctuation_runs_collapse(
        self, raw: str, expected: str
    ):
        """Case folds, runs collapse, edges strip — never a doubled hyphen."""
        got = _mod.derive_topic_slug(raw)
        assert got == expected
        assert '--' not in got
        assert not got.startswith('-')
        assert not got.endswith('-')

    @pytest.mark.parametrize(
        ('raw', 'why'),
        [
            ('', 'empty'),
            ('   ', 'whitespace only'),
            ('\n\t', 'whitespace only, non-space'),
            ('!!!', 'all punctuation — nothing survives the fold'),
            ('---', 'all separators'),
            ('a' * 120, 'exceeds TOPIC_SLUG_MAX_LEN'),
            (None, 'not a string'),
            (12345, 'not a string'),
            (['docs-prd-landing'], 'not a string'),
        ],
    )
    def test_unfoldable_values_return_none_rather_than_a_guess(
        self, raw: object, why: str
    ):
        """No honest fold exists -> ``None``, so the caller can report it.

        Truncating the over-long value or inventing a slug for ``'!!!'``
        would silently file a record under a topic no human chose.  The
        loud-over-silent read is to refuse and surface it.
        """
        assert _mod.derive_topic_slug(raw) is None, why

    def test_over_length_boundary_is_the_shared_cap_not_a_local_number(self):
        """Exactly at the cap passes; one over fails — via ε's constant."""
        cap = topic_slug_module.TOPIC_SLUG_MAX_LEN
        assert _mod.derive_topic_slug('a' * cap) == 'a' * cap
        assert _mod.derive_topic_slug('a' * (cap + 1)) is None

    @pytest.mark.parametrize(
        'raw',
        [
            'docs-prd-landing',
            'eval_worktree_plan_tools_missing',
            'Docs PRD Landing',
            'docs/prd/landing',
            'a',
            'x1-2y',
            '--docs--prd--landing--',
        ],
    )
    def test_every_non_none_return_satisfies_the_shared_predicate(
        self, raw: str
    ):
        """The output contract: whatever comes back is a valid slug.

        Checked against :func:`fused_memory.topic_slug.is_valid_topic_slug`
        directly (not the script's re-export) so the property holds against
        the normative home even if the re-export were ever broken.
        """
        got = _mod.derive_topic_slug(raw)
        assert got is not None
        assert topic_slug_module.is_valid_topic_slug(got)


class TestTopicSlugNamespaceIsShared:
    """INV-5: the script gets ε's rule by import, never by copy.

    ``tests/test_topic_slug_namespace.py`` pins the same identity for the
    metadata registry and the config schema.  A second copy of the regex or
    the cap anywhere in the tree is a bug; these ``is`` assertions make a
    copy fail mechanically rather than by review.
    """

    def test_predicate_is_the_same_object(self):
        assert _mod.is_valid_topic_slug is topic_slug_module.is_valid_topic_slug

    def test_cap_is_the_same_object(self):
        assert _mod.TOPIC_SLUG_MAX_LEN is topic_slug_module.TOPIC_SLUG_MAX_LEN

    def test_script_does_not_define_its_own_slug_pattern(self):
        """No local ``re.Pattern`` re-expressing the slug shape.

        The fold itself needs a character-class pattern, so this does not
        forbid ``re`` outright — it forbids a *second anchored slug
        validator*, which is the copy that would silently diverge from ε.
        """
        source = SCRIPT_PATH.read_text()
        assert '[a-z0-9]+(?:-[a-z0-9]+)*' not in source


# ===========================================================================
# compute_patch — the idempotence heart
# ===========================================================================

class TestComputePatchTopic:
    """The ``topic`` half of ``compute_patch``.

    ``compute_patch(existing_metadata, *, target_topic, make_canonical)``
    returns a :class:`PatchDecision` carrying the metadata patch to write
    (possibly empty) and the dispositions that explain it.  Pure: it takes a
    plain dict and no service, so a wrong answer here can never be masked by
    a mock.
    """

    def test_absent_topic_is_stamped(self):
        decision = _mod.compute_patch(
            {'category': 'observations_and_summaries'},
            target_topic='docs-prd-landing',
            make_canonical=False,
        )
        assert decision.patch == {'topic': 'docs-prd-landing'}
        assert 'topic_stamped' in decision.dispositions

    def test_matching_topic_writes_nothing(self):
        """Idempotence: run two must not re-issue the same write."""
        decision = _mod.compute_patch(
            {'topic': 'docs-prd-landing'},
            target_topic='docs-prd-landing',
            make_canonical=False,
        )
        assert 'topic' not in decision.patch
        assert 'topic_already_present' in decision.dispositions

    def test_snake_case_twin_is_normalized_in_place(self):
        """The measured live case — and ε's stated precondition.

        ``eval_worktree_plan_tools_missing`` and the target
        ``eval-worktree-plan-tools-missing`` are the SAME topic wearing two
        shapes.  Rewriting the record to the conforming shape is exactly the
        normalization ε's enforcement note delegates to θ; leaving it is what
        blocks ``memory_metadata.enforce``.
        """
        decision = _mod.compute_patch(
            {'topic': 'eval_worktree_plan_tools_missing'},
            target_topic='eval-worktree-plan-tools-missing',
            make_canonical=False,
        )
        assert decision.patch == {'topic': 'eval-worktree-plan-tools-missing'}
        assert 'topic_normalized' in decision.dispositions
        assert 'conflicting_existing_topic' not in decision.dispositions

    def test_different_existing_topic_is_never_clobbered(self):
        """A human-set topic outranks the sweep's guess, always.

        Overwriting here would silently re-file someone's record under
        another topic — a retro sweep must not be able to destroy a fact it
        did not create.  Refusing costs one report line; guessing costs a
        fact.
        """
        decision = _mod.compute_patch(
            {'topic': 'some-other-topic'},
            target_topic='docs-prd-landing',
            make_canonical=False,
        )
        assert 'topic' not in decision.patch
        assert 'conflicting_existing_topic' in decision.dispositions
        assert 'topic_normalized' not in decision.dispositions
        assert 'topic_stamped' not in decision.dispositions

    def test_unfoldable_existing_topic_is_a_conflict_not_an_overwrite(self):
        """An existing value ``derive_topic_slug`` cannot fold is still a fact.

        ``None`` from the fold means "no honest comparison available" — which
        is a reason to refuse, not a licence to overwrite.
        """
        decision = _mod.compute_patch(
            {'topic': '!!!'},
            target_topic='docs-prd-landing',
            make_canonical=False,
        )
        assert 'topic' not in decision.patch
        assert 'conflicting_existing_topic' in decision.dispositions

    def test_nothing_to_do_yields_an_empty_patch(self):
        """The property ``run`` keys its no-op on.

        An empty patch means the caller issues NO ``update_memory`` at all —
        not an update that happens to change nothing.  That is what makes a
        second run cost zero writes rather than zero net effect.
        """
        decision = _mod.compute_patch(
            {'topic': 'docs-prd-landing', 'canonical': True},
            target_topic='docs-prd-landing',
            make_canonical=True,
        )
        assert not decision.patch

    def test_is_pure_and_does_not_mutate_its_input(self):
        existing = {'topic': 'docs-prd-landing', 'category': 'preferences_and_norms'}
        before = dict(existing)
        _mod.compute_patch(
            existing, target_topic='docs-prd-landing', make_canonical=False
        )
        assert existing == before

    def test_dispositions_is_a_tuple(self):
        """Frozen, hashable, and safe to fold straight into a report."""
        decision = _mod.compute_patch(
            {}, target_topic='docs-prd-landing', make_canonical=False
        )
        assert isinstance(decision.dispositions, tuple)
        assert isinstance(decision.patch, dict)


#: The real ``supersedes`` value carried by two of the six live
#: ``canonical: true`` records (``9a4e568b`` capability-manifest-sidecar-and-g7
#: and ``9b01e961`` docs-prd-landing).  Transcribed, not invented — the whole
#: point of the case is that prose like this EXISTS in the corpus today.
LIVE_PROSE_SUPERSEDES = (
    '0d542614 (2026-07-20 07:03) and b5cc4f3c (2026-07-20 14:07), '
    'both deleted 2026-07-25 as part of the same consolidation'
)


class TestComputePatchCanonicalAndSupersedes:
    """The ``canonical`` and ``supersedes`` halves of ``compute_patch``."""

    # -- canonical: stamping only, never demotion ---------------------------

    def test_canonical_is_stamped_when_requested_and_absent(self):
        decision = _mod.compute_patch(
            {'topic': 'docs-prd-landing'},
            target_topic='docs-prd-landing',
            make_canonical=True,
        )
        assert decision.patch == {'canonical': True}
        assert 'canonical_stamped' in decision.dispositions

    def test_canonical_already_true_writes_nothing(self):
        decision = _mod.compute_patch(
            {'topic': 'docs-prd-landing', 'canonical': True},
            target_topic='docs-prd-landing',
            make_canonical=True,
        )
        assert 'canonical' not in decision.patch
        assert 'canonical_already_present' in decision.dispositions

    @pytest.mark.parametrize(
        'existing',
        [
            {'topic': 'docs-prd-landing'},
            {'topic': 'docs-prd-landing', 'canonical': True},
            {'topic': 'docs-prd-landing', 'canonical': False},
        ],
        ids=['absent', 'true', 'false'],
    )
    def test_make_canonical_false_never_emits_the_key(self, existing: dict):
        """Stamping only — θ never demotes, in any spelling.

        Not writing ``canonical: False`` matters as much as not deleting an
        existing ``True``: an explicit ``False`` is a *claim* ("this is not
        the canonical"), and θ has no basis for it.  Members of a cluster
        whose canonical is disputed or plural must come out of the sweep
        carrying a ``topic`` and no opinion at all about canonicity.
        """
        decision = _mod.compute_patch(
            existing, target_topic='docs-prd-landing', make_canonical=False
        )
        assert 'canonical' not in decision.patch

    # -- supersedes: PRD D2, normalize only where it is honest --------------

    def test_scalar_uuid_supersedes_is_folded_to_a_list(self):
        """D2's scalar->list fold, riding the sweep where it touches anyway."""
        uuid_value = '0d542614-1b2c-4d3e-8f90-abcdef123456'
        decision = _mod.compute_patch(
            {'topic': 'docs-prd-landing', 'supersedes': uuid_value},
            target_topic='docs-prd-landing',
            make_canonical=False,
        )
        assert decision.patch == {'supersedes': [uuid_value]}
        assert 'supersedes_normalized' in decision.dispositions

    def test_conforming_list_supersedes_is_left_alone(self):
        """Already a list of full UUIDs -> no write, so run two stays empty."""
        uuids = [
            '0d542614-1b2c-4d3e-8f90-abcdef123456',
            'b5cc4f3c-1b2c-4d3e-8f90-abcdef123456',
        ]
        decision = _mod.compute_patch(
            {'topic': 'docs-prd-landing', 'supersedes': list(uuids)},
            target_topic='docs-prd-landing',
            make_canonical=False,
        )
        assert 'supersedes' not in decision.patch

    def test_prose_supersedes_is_left_byte_identical_and_reported(self):
        """The measured case that makes blind normalization wrong.

        ``normalize_supersedes`` faithfully wraps ANY scalar, so folding this
        sentence would write ``['<prose>']`` — a one-member list whose member
        fails ``_is_full_uuid``, turning a record that merely has a legacy
        shape into one that fails ``validate_memory_metadata`` outright.  D2
        offers the fold as a convenience; it does not license manufacturing a
        validation failure.  Leave it, name it, move on.
        """
        decision = _mod.compute_patch(
            {'topic': 'docs-prd-landing', 'supersedes': LIVE_PROSE_SUPERSEDES},
            target_topic='docs-prd-landing',
            make_canonical=False,
        )
        assert 'supersedes' not in decision.patch
        assert 'supersedes_not_normalizable' in decision.dispositions

    def test_list_with_a_non_uuid_member_is_also_left_alone(self):
        """Same reasoning, list shape: never rewrite a value we cannot vouch for."""
        decision = _mod.compute_patch(
            {'topic': 'docs-prd-landing', 'supersedes': ['0d542614', 'not-a-uuid']},
            target_topic='docs-prd-landing',
            make_canonical=False,
        )
        assert 'supersedes' not in decision.patch
        assert 'supersedes_not_normalizable' in decision.dispositions

    def test_absent_supersedes_produces_no_key_and_no_disposition(self):
        """θ never INVENTS a supersedes edge — it only reshapes an existing one."""
        decision = _mod.compute_patch(
            {'topic': 'docs-prd-landing'},
            target_topic='docs-prd-landing',
            make_canonical=False,
        )
        assert 'supersedes' not in decision.patch
        assert not [d for d in decision.dispositions if d.startswith('supersedes')]

    # -- the three halves compose ------------------------------------------

    def test_all_three_halves_can_land_in_one_patch(self):
        uuid_value = '0d542614-1b2c-4d3e-8f90-abcdef123456'
        decision = _mod.compute_patch(
            {'topic': 'docs_prd_landing', 'supersedes': uuid_value},
            target_topic='docs-prd-landing',
            make_canonical=True,
        )
        assert decision.patch == {
            'topic': 'docs-prd-landing',
            'canonical': True,
            'supersedes': [uuid_value],
        }
        assert set(decision.dispositions) == {
            'topic_normalized',
            'canonical_stamped',
            'supersedes_normalized',
        }

    def test_shape_helpers_come_from_the_registry_not_a_local_copy(self):
        """INV-5 again: the UUID predicate and the fold have one home.

        Same private-helper reuse ``strip_leaked_control_keys.py`` already
        establishes for ``_drop_reserved_control_keys``.
        """
        from fused_memory import memory_metadata

        assert _mod.normalize_supersedes is memory_metadata.normalize_supersedes
        assert _mod._is_full_uuid is memory_metadata._is_full_uuid


# ===========================================================================
# plan_canonical_clusters — source (1), the live `canonical: true` scroll
# ===========================================================================
#
# Every fixture below is TRANSCRIBED from a live record measured on
# 2026-08-02 (`get_memories_by_metadata({'canonical': True})`, 1 in
# dark_factory + 5 in reify).  They are not invented shapes: the point of
# this source is that the member-bearing keys are heterogeneous in the wild
# (`consolidates`, `retires`, scalar `replaces`, list `supersedes`, prose
# `supersedes`), and a planner tested only against a tidy invented shape
# would miss exactly that.

#: dark_factory 0929cff6 — topic in snake_case, members under `consolidates`.
LIVE_DF_CANONICAL = {
    'id': '0929cff6-8efc-4a49-a672-e83f0a41bbfb',
    'created_at': '2026-07-23T05:40:48.789851+00:00',
    'metadata': {
        'topic': 'eval_worktree_plan_tools_missing',
        'canonical': True,
        'consolidates': [
            '0460ebc2-b23f-4366-91ea-4c5d42a7b82b',
            'ec96e8de-d7f5-4f50-9ce6-53611ed8df7e',
            '8b38cf14-dfc0-472e-b5eb-a03850a08cda',
            '25a51e44-cb94-42d5-a236-8ff896f68652',
            '3d5665c6-6afc-4815-ad7b-513dce2762ad',
            'd3cded4d-659b-48b4-962a-63a50db0ede0',
            '7c3f6259-020a-4898-8a33-6d610d5a3eb3',
        ],
        'category': 'procedural_knowledge',
    },
}

#: reify 28cbf1c4 — members split across `retires` (list) and `replaces`
#: (SCALAR).  The scalar arm is the one a list-only harvester would drop.
LIVE_REIFY_RETIRES_AND_REPLACES = {
    'id': '28cbf1c4-555e-430f-82a7-92578381c8df',
    'created_at': '2026-07-28T20:08:13.475183+00:00',
    'metadata': {
        'topic': 'cross-project-path-scope-guard',
        'canonical': True,
        'replaces': 'f2f0cb49-7403-4f10-8f26-2a33ec3b9247',
        'retires': [
            'ddc9b76e-d8fb-41d4-b4eb-0d6076f7a18a',
            '4d2f5b96-d57d-4896-a15c-c5bc660434b5',
            'bab98983-dc79-4332-9bc1-b3c7aec426f7',
            'fd179b32-a422-44ab-bf1a-1043e7f96aa6',
            '2c1f21a2-a812-46ea-af6e-4f0b98bad61e',
        ],
        'category': 'procedural_knowledge',
    },
}

#: reify bbc063a7 — members under a list `supersedes`.  Also carries
#: `replaces_verbatim`, which is deliberately NOT a member key: it names the
#: record this one reissues, and it already appears in `supersedes`.
LIVE_REIFY_LIST_SUPERSEDES = {
    'id': 'bbc063a7-ea8f-4262-b9dd-eef3002a99a8',
    'created_at': '2026-07-28T20:05:44.622926+00:00',
    'metadata': {
        'topic': 'harness-layout-gate-decision-rule',
        'canonical': True,
        'replaces_verbatim': '168c3a6b-55dd-4f53-88e4-3a829ea210fc',
        'supersedes': [
            '168c3a6b-55dd-4f53-88e4-3a829ea210fc',
            '1a265e20-6e4d-4c7c-a1dd-cdf880c865e4',
            'ad415479-f6b6-441b-b95f-6bf1d4168804',
            '80936e18-6a91-4b74-9bb8-cae79e6490b8',
            '8110c809-04cf-4de8-9aad-1c2c00e25053',
            '18cbefc3-5845-443b-95cb-1d14b265befd',
        ],
        'category': 'procedural_knowledge',
    },
}

#: reify dbc478b8 — snake_case topic AND an overlap between `consolidates`
#: and `supersedes` (``4c2786de`` appears in both), i.e. the live dedup case.
LIVE_REIFY_OVERLAPPING_MEMBERS = {
    'id': 'dbc478b8-6357-417f-b114-303ebfe650e5',
    'created_at': '2026-07-28T00:05:37.968722+00:00',
    'metadata': {
        'topic': 'merge_request_bare_task_id_branch_arg',
        'canonical': True,
        'supersedes': [
            '4c2786de-2511-462f-a643-576ddc63fdf5',
            'f71c92ff-08d5-4264-9a29-f3b1bfcaca61',
        ],
        'consolidates': [
            '4c2786de-2511-462f-a643-576ddc63fdf5',
            '99ddc1d3-ed7c-44f5-8eae-5c461818da3b',
            'a5216a28-824f-44ee-828c-fd294dd119fd',
            'f522b1c8-74e0-4578-9371-bc5749911682',
            '156b0e6d-2088-444d-a38c-7bea7f5168f3',
            '40737468-cb02-462a-9a12-5f8b2cd6fb43',
        ],
        'category': 'procedural_knowledge',
    },
}

#: reify 9a4e568b — the prose `supersedes` again, here as a MEMBER source.
LIVE_REIFY_PROSE_SUPERSEDES = {
    'id': '9a4e568b-1b7e-408c-afeb-efde9bde33aa',
    'created_at': '2026-07-25T16:15:54.557669+00:00',
    'metadata': {
        'topic': 'capability-manifest-sidecar-and-g7',
        'canonical': True,
        'supersedes': LIVE_PROSE_SUPERSEDES,
        'category': 'procedural_knowledge',
    },
}


def _only(plans: list) -> object:
    assert len(plans) == 1, f'expected exactly one plan, got {plans!r}'
    return plans[0]


def _reasons(skips: list[dict]) -> list[str]:
    return [s['reason'] for s in skips]


class TestPlanCanonicalClusters:
    """``plan_canonical_clusters(records, *, project_id)``.

    Turns scroll-shaped ``canonical: true`` records into cluster plans.
    Pure: records in, ``(plans, skips)`` out — no service argument, so it
    cannot be tested into passing against a mock.
    """

    def test_member_keys_are_a_reviewable_ordered_constant(self):
        """The harvested keys are a named constant, not a buried literal.

        Order is part of the contract: it makes member ordering deterministic
        across runs, so two dry-runs diff cleanly.
        """
        assert _mod.CLUSTER_MEMBER_KEYS == (
            'consolidates',
            'retires',
            'replaces',
            'supersedes',
        )

    def test_snake_case_topic_folds_and_consolidates_supplies_members(self):
        """Case (a): the single live dark_factory canonical."""
        plans, skips = _mod.plan_canonical_clusters(
            [LIVE_DF_CANONICAL], project_id='dark_factory'
        )
        plan = _only(plans)
        assert plan.topic == 'eval-worktree-plan-tools-missing'
        assert plan.project_id == 'dark_factory'
        assert plan.canonical_memory_id == '0929cff6-8efc-4a49-a672-e83f0a41bbfb'
        assert plan.member_memory_ids == tuple(
            LIVE_DF_CANONICAL['metadata']['consolidates']
        )
        assert plan.canonical_plural is False
        assert skips == []

    def test_scalar_replaces_is_harvested_alongside_list_retires(self):
        """Case (b): a scalar member key is a member key.

        ``replaces`` is a bare string on this record.  A harvester that only
        understood lists would silently drop that edge — the same
        scalar-vs-list split D2 documents for ``supersedes``.
        """
        plans, _ = _mod.plan_canonical_clusters(
            [LIVE_REIFY_RETIRES_AND_REPLACES], project_id='reify'
        )
        plan = _only(plans)
        meta = LIVE_REIFY_RETIRES_AND_REPLACES['metadata']
        assert plan.member_memory_ids == (*meta['retires'], meta['replaces'])

    def test_list_supersedes_supplies_members(self):
        """Case (c) — and ``replaces_verbatim`` is not a member key."""
        plans, _ = _mod.plan_canonical_clusters(
            [LIVE_REIFY_LIST_SUPERSEDES], project_id='reify'
        )
        plan = _only(plans)
        assert plan.member_memory_ids == tuple(
            LIVE_REIFY_LIST_SUPERSEDES['metadata']['supersedes']
        )

    def test_members_are_deduplicated_across_keys(self):
        """Case (d): ``4c2786de`` is in BOTH consolidates and supersedes."""
        plans, _ = _mod.plan_canonical_clusters(
            [LIVE_REIFY_OVERLAPPING_MEMBERS], project_id='reify'
        )
        plan = _only(plans)
        assert len(plan.member_memory_ids) == len(set(plan.member_memory_ids))
        assert plan.member_memory_ids == (
            '4c2786de-2511-462f-a643-576ddc63fdf5',
            '99ddc1d3-ed7c-44f5-8eae-5c461818da3b',
            'a5216a28-824f-44ee-828c-fd294dd119fd',
            'f522b1c8-74e0-4578-9371-bc5749911682',
            '156b0e6d-2088-444d-a38c-7bea7f5168f3',
            '40737468-cb02-462a-9a12-5f8b2cd6fb43',
            'f71c92ff-08d5-4264-9a29-f3b1bfcaca61',
        )

    def test_a_canonical_is_never_its_own_member(self):
        """Case (d), second half.

        A self-reference would make ``stamp_one`` visit the same record twice
        with contradictory ``make_canonical`` values — the kind of ambiguity
        that is far cheaper to exclude here than to adjudicate later.
        """
        record = {
            'id': '0929cff6-8efc-4a49-a672-e83f0a41bbfb',
            'metadata': {
                'topic': 'docs-prd-landing',
                'canonical': True,
                'consolidates': [
                    '0929cff6-8efc-4a49-a672-e83f0a41bbfb',
                    '0460ebc2-b23f-4366-91ea-4c5d42a7b82b',
                ],
            },
        }
        plans, _ = _mod.plan_canonical_clusters([record], project_id='dark_factory')
        plan = _only(plans)
        assert plan.canonical_memory_id not in plan.member_memory_ids
        assert plan.member_memory_ids == ('0460ebc2-b23f-4366-91ea-4c5d42a7b82b',)

    def test_non_uuid_member_is_dropped_and_reported_never_swallowed(self):
        """Case (e): the prose ``supersedes``, seen as a member source.

        The cluster is still planned — its topic is good — but the
        unresolvable member is named in the skip list.  Dropping it silently
        would let the report claim a clean sweep over a cluster whose
        membership was never actually determined.
        """
        plans, skips = _mod.plan_canonical_clusters(
            [LIVE_REIFY_PROSE_SUPERSEDES], project_id='reify'
        )
        plan = _only(plans)
        assert plan.topic == 'capability-manifest-sidecar-and-g7'
        assert plan.member_memory_ids == ()
        assert _reasons(skips) == ['member_not_a_uuid']
        assert skips[0]['memory_id'] == '9a4e568b-1b7e-408c-afeb-efde9bde33aa'
        assert skips[0]['key'] == 'supersedes'
        assert skips[0]['value'] == LIVE_PROSE_SUPERSEDES

    @pytest.mark.parametrize(
        ('topic', 'why'),
        [(None, 'no topic at all'), ('!!!', 'unfoldable'), ('a' * 120, 'over the cap')],
    )
    def test_underivable_topic_yields_no_plan_and_a_named_skip(
        self, topic: object, why: str
    ):
        """Case (f): no topic, no plan — and a line in the report saying so.

        Guessing a slug here would file a whole cluster under a topic nobody
        chose.  Emitting an empty plan would be worse: it would look like a
        cluster the sweep handled.
        """
        metadata: dict = {'canonical': True, 'consolidates': [
            '0460ebc2-b23f-4366-91ea-4c5d42a7b82b'
        ]}
        if topic is not None:
            metadata['topic'] = topic
        plans, skips = _mod.plan_canonical_clusters(
            [{'id': '0929cff6-8efc-4a49-a672-e83f0a41bbfb', 'metadata': metadata}],
            project_id='dark_factory',
        )
        assert plans == [], why
        assert _reasons(skips) == ['topic_underivable']
        assert skips[0]['memory_id'] == '0929cff6-8efc-4a49-a672-e83f0a41bbfb'

    def test_all_six_measured_live_records_plan_without_error(self):
        """The whole measured source (1), end to end.

        Five reify records plus one dark_factory record is the entire live
        ``canonical: true`` population at authoring time.  Planning all of
        them at once is the closest a unit test gets to the real input.
        """
        reify_records = [
            LIVE_REIFY_RETIRES_AND_REPLACES,
            LIVE_REIFY_LIST_SUPERSEDES,
            LIVE_REIFY_OVERLAPPING_MEMBERS,
            LIVE_REIFY_PROSE_SUPERSEDES,
        ]
        plans, skips = _mod.plan_canonical_clusters(reify_records, project_id='reify')
        assert len(plans) == 4
        assert all(topic_slug_module.is_valid_topic_slug(p.topic) for p in plans)
        assert all(p.source == 'canonical_scroll' for p in plans)
        # The one prose member is the only thing this source cannot resolve.
        assert _reasons(skips) == ['member_not_a_uuid']

    def test_is_pure_and_does_not_mutate_its_input(self):
        import copy

        records = [copy.deepcopy(LIVE_DF_CANONICAL)]
        before = copy.deepcopy(records)
        _mod.plan_canonical_clusters(records, project_id='dark_factory')
        assert records == before


# ===========================================================================
# plan_calibration_clusters — sources (2)-reify and (3), joined
# ===========================================================================

def _row(
    cluster_id: str,
    memory_id: str,
    label: str,
    *,
    gate_id: str = 'esc-5560',
) -> dict:
    """Build a 3130 calibration-fixture row (the keys the join reads)."""
    return {
        'cluster_id': cluster_id,
        'memory_id': memory_id,
        'label': label,
        'content': f'content of {memory_id}',
        'category': 'procedural_knowledge',
        'provenance': {'gate_id': gate_id, 'source': 'add_memory_input'},
    }


def _entry(
    topic: str,
    cluster_id: str,
    *,
    project_id: str = 'reify',
) -> object:
    """Build a minimal registry entry keyed to *cluster_id*."""
    return _mod.RegistryEntry(
        topic=topic,
        project_id=project_id,
        derived_from='curator_gate',
        canonical=_mod.Canonical(content_hash='0' * 16, last_known_id=cluster_id),
        phrasings=(_mod.Phrasing(text=f'what about {topic}'),),
        provenance={'cluster_id': cluster_id},
    )


def _registry(*entries: object) -> object:
    return _mod.TopicRegistry(schema_version=1, entries=tuple(entries))


CLUSTER_A = '68f19644-c3f3-49d5-af61-c983de358358'


class TestPlanCalibrationClusters:
    """``plan_calibration_clusters(rows, registry)`` — ONE join, two sources.

    3130's labeled fixture supplies memory ids and curator labels; the
    committed E1 topic registry supplies the hand-authored topic slug, keyed
    by ``provenance.cluster_id``.  That join resolves both source (3) and the
    reify half of source (2) at once: the registry's 20 ``curator_gate``
    entries ARE 3112's reify gate census, already materialized and
    topic-completed by a human.
    """

    def test_canonical_row_and_duplicate_rows_form_the_cluster(self):
        rows = [
            _row(CLUSTER_A, 'aaaaaaaa-0000-4000-8000-000000000001', 'canonical'),
            _row(CLUSTER_A, 'aaaaaaaa-0000-4000-8000-000000000002', 'duplicate'),
            _row(CLUSTER_A, 'aaaaaaaa-0000-4000-8000-000000000003', 'duplicate'),
            _row(CLUSTER_A, 'aaaaaaaa-0000-4000-8000-000000000004', 'duplicate'),
        ]
        plans, skips = _mod.plan_calibration_clusters(
            rows, _registry(_entry('architect-plan-files-write-set', CLUSTER_A))
        )
        plan = _only(plans)
        assert plan.topic == 'architect-plan-files-write-set'
        assert plan.canonical_memory_id == 'aaaaaaaa-0000-4000-8000-000000000001'
        assert plan.member_memory_ids == (
            'aaaaaaaa-0000-4000-8000-000000000002',
            'aaaaaaaa-0000-4000-8000-000000000003',
            'aaaaaaaa-0000-4000-8000-000000000004',
        )
        assert plan.source == 'calibration_registry_join'
        assert skips == []

    @pytest.mark.parametrize('label', ['distinct', 'pseudo_contradiction'])
    def test_distinct_and_pseudo_contradiction_rows_are_not_members(
        self, label: str
    ):
        """The curator's own adjudication, reused rather than re-invented.

        ``_derive_curator_gate_candidates`` counts ONLY ``duplicate`` rows as
        members: ``distinct`` and ``pseudo_contradiction`` are separate claims
        that merely READ as contradictory.  Stamping them into the cluster
        would tell every downstream reader a legitimately different answer is
        contamination — and θ inventing a second, looser notion of "cluster"
        than the deriver's is exactly how two definitions drift apart.
        """
        rows = [
            _row(CLUSTER_A, 'aaaaaaaa-0000-4000-8000-000000000001', 'canonical'),
            _row(CLUSTER_A, 'aaaaaaaa-0000-4000-8000-000000000002', 'duplicate'),
            _row(CLUSTER_A, 'aaaaaaaa-0000-4000-8000-000000000009', label),
        ]
        plans, _ = _mod.plan_calibration_clusters(
            rows, _registry(_entry('architect-plan-files-write-set', CLUSTER_A))
        )
        plan = _only(plans)
        assert plan.member_memory_ids == ('aaaaaaaa-0000-4000-8000-000000000002',)

    def test_cluster_with_no_registry_entry_is_skipped_not_slugified(self):
        """The join is the whole point — never fall back to a slugified UUID.

        ``_derive_curator_gate_candidates`` falls back to
        ``_slugify(cluster_id)`` when a row carries no topic, and these rows
        carry none.  That fallback would mint a slugified UUID as a topic and
        stamp it across a real cluster.  The registry's hand-authored slugs
        are the part a machine cannot regenerate; without one there is
        nothing honest to stamp.
        """
        rows = [
            _row('deadbeef-0000-4000-8000-000000000000',
                 'aaaaaaaa-0000-4000-8000-000000000001', 'canonical'),
            _row('deadbeef-0000-4000-8000-000000000000',
                 'aaaaaaaa-0000-4000-8000-000000000002', 'duplicate'),
        ]
        plans, skips = _mod.plan_calibration_clusters(rows, _registry())
        assert plans == []
        assert _reasons(skips) == ['no_registry_topic']
        assert skips[0]['cluster_id'] == 'deadbeef-0000-4000-8000-000000000000'

    def test_cluster_without_a_canonical_row_still_gets_a_topic(self):
        """Topic-only plan plus a note — a missing canonical is not a blocker.

        The members are still a real cluster and still benefit from a topic;
        what the sweep must not do is *pick* a canonical for them.
        """
        rows = [
            _row(CLUSTER_A, 'aaaaaaaa-0000-4000-8000-000000000002', 'duplicate'),
            _row(CLUSTER_A, 'aaaaaaaa-0000-4000-8000-000000000003', 'duplicate'),
        ]
        plans, skips = _mod.plan_calibration_clusters(
            rows, _registry(_entry('architect-plan-files-write-set', CLUSTER_A))
        )
        plan = _only(plans)
        assert plan.canonical_memory_id is None
        assert len(plan.member_memory_ids) == 2
        assert _reasons(skips) == ['cluster_without_canonical']

    def test_project_id_comes_from_the_registry_entry(self):
        """Sources span two corpora; project cannot be a run-level constant."""
        rows = [_row(CLUSTER_A, 'aaaaaaaa-0000-4000-8000-000000000002', 'duplicate')]
        plans, _ = _mod.plan_calibration_clusters(
            rows,
            _registry(_entry('architect-plan-files-write-set', CLUSTER_A,
                             project_id='reify')),
        )
        assert _only(plans).project_id == 'reify'

    def test_non_uuid_memory_id_is_reported_not_stamped(self):
        rows = [
            _row(CLUSTER_A, 'aaaaaaaa-0000-4000-8000-000000000001', 'canonical'),
            _row(CLUSTER_A, 'not-a-uuid', 'duplicate'),
        ]
        plans, skips = _mod.plan_calibration_clusters(
            rows, _registry(_entry('architect-plan-files-write-set', CLUSTER_A))
        )
        assert _only(plans).member_memory_ids == ()
        assert 'member_not_a_uuid' in _reasons(skips)

    def test_is_pure_and_does_not_mutate_its_input(self):
        import copy

        rows = [_row(CLUSTER_A, 'aaaaaaaa-0000-4000-8000-000000000002', 'duplicate')]
        before = copy.deepcopy(rows)
        _mod.plan_calibration_clusters(
            rows, _registry(_entry('architect-plan-files-write-set', CLUSTER_A))
        )
        assert rows == before


class TestCalibrationJoinAgainstTheRealCommittedFixtures:
    """The join, run against the two fixtures actually on disk.

    A unit test with hand-built rows proves the join LOGIC; it cannot notice
    that the two fixtures stopped agreeing on ``cluster_id``.  If a future
    edit to either file breaks the join, the sweep would quietly plan nothing
    and report a clean run — so the breakage is caught here instead.
    """

    def test_fixture_paths_point_at_the_committed_files(self):
        assert Path(_mod.CALIBRATION_FIXTURE_PATH).is_file()
        assert Path(_mod.TOPIC_REGISTRY_PATH).is_file()

    def test_real_join_is_non_empty_and_every_topic_is_a_valid_slug(self):
        rows = _mod.load_calibration_rows(_mod.CALIBRATION_FIXTURE_PATH)
        registry = _mod.load_topic_registry(_mod.TOPIC_REGISTRY_PATH)
        assert len(rows) > 0
        plans, _skips = _mod.plan_calibration_clusters(rows, registry)
        assert plans, 'the two committed fixtures no longer join on cluster_id'
        for plan in plans:
            assert topic_slug_module.is_valid_topic_slug(plan.topic), plan.topic
            assert plan.source == 'calibration_registry_join'

    def test_registry_is_loaded_through_the_probes_strict_loader(self):
        """Not a hand-rolled ``json.load``.

        That loader is required-strict and additive-tolerant, and
        ``RegistryEntry.extra`` exists precisely so this consumer can load a
        richer entry against it without a fixture rewrite — its docstring
        names task 3201.
        """
        probe = _mod._load_probe_module()
        assert _mod.load_topic_registry is probe.load_topic_registry
        assert _mod.RegistryEntry is probe.RegistryEntry


# ===========================================================================
# DF_CURATOR_GATE_CLUSTERS — source (2), dark_factory half
# ===========================================================================

#: The seven dark_factory curator gates PRD D11 enumerates.
DF_GATE_IDS = {'2969', '2973', '3011', '3016', '3036', '3063', '3092'}

#: The three whose member ids exist ONLY in title prose — measured against
#: ``.taskmaster/tasks/tasks.db`` on 2026-08-02, not assumed.
DF_GATES_WITHOUT_ENUMERATION = {'2969', '2973', '3016'}

#: Gate 3036's ``metadata.memory_ids``, transcribed verbatim.
GATE_3036_MEMORY_IDS = (
    '19705df4-3f01-41cb-8eab-41624a160a00',
    '8ef8cf8d-4bd7-45ff-b99a-4f2e0d1d4855',
    '9cf1e664-8a27-4e1e-9d3d-c6c210476142',
    'ae4a759a-2405-4f9d-875d-9ac2433c87c7',
    'b3d4d44e-1906-45dc-b156-f894264f3b25',
    'adcfa20b-7d2a-4eb1-9d83-8f6a5c871f25',
    '8fcdafbc-6776-4875-bc29-3ce3724dc794',
    '4a74ed20-6076-41a2-b967-7e339883e800',
)


def _manifest_by_gate() -> dict:
    return {e.gate_task_id: e for e in _mod.DF_CURATOR_GATE_CLUSTERS}


class TestDfCuratorGateManifest:
    """The committed manifest IS the artifact, so its shape is contract.

    Its ids were transcribed from live gate-task metadata rather than read at
    run time: the seven gates' enumeration keys are genuinely heterogeneous
    (``memory_ids``, ``cited_memories``, ``stale_cluster_memory_ids`` +
    ``canonical_fact_memory_ids``, ``legacy_cluster_candidates_resolved``),
    a live read would need a task-backend dependency this script otherwise
    does not have, and it would still hit the same three-gate hole.  Pinning
    the table here makes the hole VISIBLE instead of turning "covered 4 of 7
    gates" into silence.
    """

    def test_covers_exactly_the_seven_prd_gates(self):
        """A dropped gate fails a test instead of vanishing from the sweep."""
        assert {e.gate_task_id for e in _mod.DF_CURATOR_GATE_CLUSTERS} == DF_GATE_IDS

    def test_every_topic_is_a_valid_slug(self):
        for entry in _mod.DF_CURATOR_GATE_CLUSTERS:
            assert topic_slug_module.is_valid_topic_slug(entry.topic), entry

    def test_every_entry_targets_dark_factory(self):
        for entry in _mod.DF_CURATOR_GATE_CLUSTERS:
            assert entry.project_id == 'dark_factory'

    @pytest.mark.parametrize('gate_id', sorted(DF_GATES_WITHOUT_ENUMERATION))
    def test_the_three_unenumerated_gates_declare_the_hole(self, gate_id: str):
        """Measured: these three carry NO id list in metadata at all.

        Their member ids live only in title prose (2969's are even 8-hex
        prefixes, not uuids), which no parser should be asked to mine.
        ``None`` plus a stated reason is how the gap stays legible; an empty
        tuple would read as "a gate with no members", which is a different
        and false claim.
        """
        entry = _manifest_by_gate()[gate_id]
        assert entry.member_memory_ids is None
        assert entry.no_enumeration_reason
        assert isinstance(entry.no_enumeration_reason, str)

    def test_gate_3036_is_plural_canonical_with_its_eight_measured_ids(self):
        """The task's own rule: "canonical ENTRIES", plural, so canonical none.

        Topic on every member and ``canonical`` on none.  This composes with
        ε's <=1-canonical-per-topic invariant the safe way round: refusing to
        pick can never violate uniqueness, whereas picking can.
        """
        entry = _manifest_by_gate()['3036']
        assert entry.canonical_plural is True
        assert entry.canonical_memory_id is None
        assert entry.member_memory_ids == GATE_3036_MEMORY_IDS
        assert entry.source_key == ('memory_ids',)

    def test_every_enumerated_entry_names_the_key_its_ids_came_from(self):
        """So an auditor can re-derive the transcription without trusting it."""
        for entry in _mod.DF_CURATOR_GATE_CLUSTERS:
            if entry.member_memory_ids is None:
                assert entry.source_key == ()
                continue
            assert entry.source_key
            assert all(isinstance(k, str) and k for k in entry.source_key)

    def test_every_transcribed_id_is_a_full_uuid(self):
        """A truncated 8-hex prefix (gate 3063's ``mem0_entry``) is not an id.

        Transcribing one would produce a lookup that can never resolve, and
        a ``memory_not_found`` line indistinguishable from a genuinely
        consolidated-away record.
        """
        for entry in _mod.DF_CURATOR_GATE_CLUSTERS:
            for memory_id in entry.member_memory_ids or ():
                assert _mod._is_full_uuid(memory_id), (entry.gate_task_id, memory_id)
            if entry.canonical_memory_id is not None:
                assert _mod._is_full_uuid(entry.canonical_memory_id), entry

    def test_no_memory_id_is_claimed_by_two_gates(self):
        seen: dict[str, str] = {}
        for entry in _mod.DF_CURATOR_GATE_CLUSTERS:
            ids = list(entry.member_memory_ids or ())
            if entry.canonical_memory_id:
                ids.append(entry.canonical_memory_id)
            for memory_id in ids:
                assert memory_id not in seen, (memory_id, seen.get(memory_id))
                seen[memory_id] = entry.gate_task_id


class TestPlanGateClusters:
    """``plan_gate_clusters(manifest)`` — manifest rows to cluster plans."""

    def test_emits_one_plan_per_enumerated_gate(self):
        plans, _ = _mod.plan_gate_clusters(_mod.DF_CURATOR_GATE_CLUSTERS)
        assert len(plans) == len(DF_GATE_IDS) - len(DF_GATES_WITHOUT_ENUMERATION)
        assert all(p.source == 'df_curator_gate_manifest' for p in plans)
        assert all(p.project_id == 'dark_factory' for p in plans)

    def test_the_three_holes_are_a_named_bucket_not_silence(self):
        """no-silent-caps: a gate the sweep cannot cover says so, by name."""
        _, skips = _mod.plan_gate_clusters(_mod.DF_CURATOR_GATE_CLUSTERS)
        no_enum = [s for s in skips if s['reason'] == 'skipped_no_enumeration']
        assert {s['gate_task_id'] for s in no_enum} == DF_GATES_WITHOUT_ENUMERATION
        assert all(s['detail'] for s in no_enum)

    def test_canonical_plural_is_propagated_to_the_plan(self):
        plans, _ = _mod.plan_gate_clusters(_mod.DF_CURATOR_GATE_CLUSTERS)
        by_gate = {p.provenance['gate_task_id']: p for p in plans}
        assert by_gate['3036'].canonical_plural is True
        assert by_gate['3036'].canonical_memory_id is None
        assert by_gate['3036'].member_memory_ids == GATE_3036_MEMORY_IDS

    def test_plan_provenance_carries_the_gate_and_its_source_keys(self):
        plans, _ = _mod.plan_gate_clusters(_mod.DF_CURATOR_GATE_CLUSTERS)
        for plan in plans:
            assert plan.provenance['gate_task_id'] in DF_GATE_IDS
            assert plan.provenance['source_key']

    def test_every_planned_topic_is_a_valid_slug(self):
        plans, _ = _mod.plan_gate_clusters(_mod.DF_CURATOR_GATE_CLUSTERS)
        assert all(topic_slug_module.is_valid_topic_slug(p.topic) for p in plans)


# ===========================================================================
# merge_plans — three planners -> one per-record work list
# ===========================================================================

M1 = 'aaaaaaaa-0000-4000-8000-000000000001'
M2 = 'aaaaaaaa-0000-4000-8000-000000000002'
M3 = 'aaaaaaaa-0000-4000-8000-000000000003'
C1 = 'cccccccc-0000-4000-8000-00000000000a'
C2 = 'cccccccc-0000-4000-8000-00000000000b'


def _plan(
    topic: str,
    *,
    canonical: str | None = None,
    members: tuple[str, ...] = (),
    source: str = 'canonical_scroll',
    project_id: str = 'dark_factory',
    canonical_plural: bool = False,
    provenance: dict | None = None,
) -> object:
    return _mod.ClusterPlan(
        topic=topic,
        project_id=project_id,
        canonical_memory_id=canonical,
        member_memory_ids=members,
        source=source,
        canonical_plural=canonical_plural,
        provenance=provenance or {},
    )


class TestMergePlans:
    """``merge_plans(*plan_lists) -> (targets, skips)``.

    Flattens the three sources into the per-record work list ``run``
    executes.  Everything ambiguous is reported rather than resolved: an
    ambiguous cluster is a curation question, and letting iteration order
    answer it would make the outcome depend on which planner ran first.
    """

    def test_same_topic_from_two_sources_merges_into_one(self):
        """Overlap between sources is EXPECTED, not an error.

        The live canonical scroll and the E1 registry both cover
        ``eval-worktree-plan-tools-missing``; two sources agreeing is the
        normal case and must not produce duplicate work or a false conflict.
        """
        targets, skips = _mod.merge_plans(
            [_plan('eval-worktree-plan-tools-missing', canonical=C1, members=(M1,))],
            [_plan('eval-worktree-plan-tools-missing', canonical=C1, members=(M2,),
                   source='calibration_registry_join')],
        )
        assert skips == []
        assert {t.memory_id for t in targets} == {C1, M1, M2}
        assert [t.topic for t in targets] == ['eval-worktree-plan-tools-missing'] * 3
        assert [t.make_canonical for t in targets].count(True) == 1

    def test_one_id_claimed_by_two_topics_is_reported_not_guessed(self):
        """No stamp for the disputed id, and both topics named in the skip.

        Iteration order deciding which topic wins would make the corpus
        depend on planner ordering — and the wrong answer would be invisible.
        """
        targets, skips = _mod.merge_plans(
            [_plan('topic-one', members=(M1, M2))],
            [_plan('topic-two', members=(M2, M3))],
        )
        assert {t.memory_id for t in targets} == {M1, M3}
        conflicts = [s for s in skips if s['reason'] == 'conflicting_topic_claim']
        assert len(conflicts) == 1
        assert conflicts[0]['memory_id'] == M2
        assert sorted(conflicts[0]['topics']) == ['topic-one', 'topic-two']

    def test_disagreeing_canonicals_stamp_the_topic_but_no_canonical(self):
        """Same conservative rule as the plural case.

        Two sources naming different canonicals for one topic is a genuine
        disagreement.  Picking one would risk stamping the wrong record —
        and, since the loser keeps no marker, would erase the evidence that
        anything was ever in doubt.
        """
        targets, skips = _mod.merge_plans(
            [_plan('topic-one', canonical=C1, members=(M1,))],
            [_plan('topic-one', canonical=C2, members=(M2,),
                   source='df_curator_gate_manifest')],
        )
        assert all(t.make_canonical is False for t in targets)
        assert {t.memory_id for t in targets} == {C1, C2, M1, M2}
        disputes = [s for s in skips if s['reason'] == 'canonical_disagreement']
        assert len(disputes) == 1
        assert sorted(disputes[0]['canonical_memory_ids']) == sorted([C1, C2])
        assert disputes[0]['topic'] == 'topic-one'

    def test_at_most_one_canonical_per_project_topic(self):
        """ε's <=1-canonical-per-topic invariant, pinned at PLAN level.

        Checked before any write, because ``update_memory`` never reaches the
        service-side uniqueness probe — so a plan that violated this would
        reach Qdrant unchallenged.
        """
        targets, _ = _mod.merge_plans(
            [_plan('topic-one', canonical=C1, members=(M1,)),
             _plan('topic-two', canonical=C2, members=(M2,))],
            [_plan('topic-one', canonical=C1, members=(M3,))],
            [_plan('topic-one', canonical=C1, project_id='reify')],
        )
        seen: dict[tuple[str, str], int] = {}
        for target in targets:
            if target.make_canonical:
                key = (target.project_id, target.topic)
                seen[key] = seen.get(key, 0) + 1
        assert seen == {
            ('dark_factory', 'topic-one'): 1,
            ('dark_factory', 'topic-two'): 1,
            ('reify', 'topic-one'): 1,
        }

    def test_canonical_plural_contributes_topic_only_targets(self):
        targets, skips = _mod.merge_plans(
            [_plan('pyright-worktree-import-resolution', members=(M1, M2),
                   canonical_plural=True, source='df_curator_gate_manifest',
                   provenance={'gate_task_id': '3036', 'note': 'several canonicals'})],
        )
        assert all(t.make_canonical is False for t in targets)
        plural = [s for s in skips if s['reason'] == 'canonical_plural']
        assert len(plural) == 1
        assert plural[0]['topic'] == 'pyright-worktree-import-resolution'
        assert plural[0]['note'] == 'several canonicals'

    def test_output_order_is_deterministic(self):
        """Two runs produce byte-comparable reports.

        A report whose ordering wobbles cannot be diffed, which is most of
        what a dry-run is for.
        """
        lists = (
            [_plan('topic-two', canonical=C2, members=(M3, M1))],
            [_plan('topic-one', canonical=C1, members=(M2,), project_id='reify')],
        )
        targets, _ = _mod.merge_plans(*lists)
        assert [(t.project_id, t.topic, t.memory_id) for t in targets] == sorted(
            (t.project_id, t.topic, t.memory_id) for t in targets
        )
        again, _ = _mod.merge_plans(*lists)
        assert targets == again

    def test_a_canonical_is_never_also_a_member_target(self):
        targets, _ = _mod.merge_plans(
            [_plan('topic-one', canonical=C1, members=(C1, M1))],
        )
        assert len(targets) == len({t.memory_id for t in targets})
        canonical_targets = [t for t in targets if t.memory_id == C1]
        assert len(canonical_targets) == 1
        assert canonical_targets[0].make_canonical is True

    def test_empty_input_is_an_empty_work_list_not_an_error(self):
        assert _mod.merge_plans() == ([], [])
        assert _mod.merge_plans([], []) == ([], [])


# ===========================================================================
# stamp_one — the single I/O boundary
# ===========================================================================


def _target(
    *,
    memory_id: str = M1,
    topic: str = 'topic-one',
    make_canonical: bool = False,
    project_id: str = 'dark_factory',
    source: str = 'canonical_scroll',
) -> object:
    return _mod.StampTarget(
        project_id=project_id,
        memory_id=memory_id,
        topic=topic,
        make_canonical=make_canonical,
        source=source,
    )


def _record(memory_id: str = M1, **metadata) -> dict:
    """A ``get_memory_by_id`` response — ``{'id', 'content', 'metadata'}``."""
    return {'id': memory_id, 'content': 'some remembered fact', 'metadata': dict(metadata)}


#: Distinguishes "caller did not specify a record" from "the record is
#: genuinely absent" — ``None`` is a meaningful ``get_memory_by_id`` answer
#: here (a consolidated-away member), so it cannot double as the default.
_UNSET = object()


def _service(
    *,
    record: dict | None = _UNSET,  # type: ignore[assignment]
    count: int = 0,
    incumbent: str | None = None,
) -> AsyncMock:
    """An injected ``MemoryService`` double.

    Children are configured through ``.return_value`` rather than reassigned
    to fresh ``AsyncMock``s, because a reassigned child stops propagating into
    the parent's ``mock_calls`` — and the ORDER of the probe relative to the
    write is one of the things this suite has to pin.
    """
    service = AsyncMock()
    service.get_memory_by_id.return_value = _record() if record is _UNSET else record
    service.count_memories_by_metadata.return_value = count
    service.get_memories_by_metadata.return_value = (
        [{'id': incumbent}] if incumbent else []
    )
    service.update_memory.side_effect = lambda **kwargs: {
        'status': 'updated',
        'store': 'mem0',
        'id': kwargs['memory_id'],
        'content_amended': False,
        'metadata_patched': True,
    }
    return service


def _call_names(service: AsyncMock) -> list[str]:
    """The service methods that were called, in call order."""
    return [name for name, _args, _kwargs in service.mock_calls if name]


class TestStampOne:
    """``stamp_one(memory_service, target, *, apply) -> dict``.

    The one function in the script that talks to a store.  Everything it
    refuses to do is as load-bearing as what it does: it never re-embeds
    content, never writes an empty patch, and never stamps ``canonical``
    it could not first prove unique.
    """

    @pytest.mark.asyncio
    async def test_happy_path_writes_a_metadata_only_merge_patch(self):
        """One ``update_memory``, metadata-only, merge mode, attributed.

        The kwargs asserted here ARE the in-place-update contract
        (``plans/mem0-in-place-update-decision.md`` §3): passing ``content``
        would re-embed the record and rewrite ``updated_at`` for what is a
        purely cosmetic tag, and passing ``metadata_delete_keys`` alongside
        the patch would flip the service off its ``set_payload`` fast path
        onto a read-modify-overwrite of the WHOLE payload.  Neither is
        wanted, so both are asserted absent rather than merely not passed.
        """
        service = _service(record=_record())
        result = await _mod.stamp_one(service, _target(), apply=True)

        assert service.update_memory.await_count == 1
        kwargs = service.update_memory.await_args.kwargs
        assert kwargs['memory_id'] == M1
        assert kwargs['project_id'] == 'dark_factory'
        assert kwargs['metadata_patch'] == {'topic': 'topic-one'}
        assert kwargs['metadata_mode'] == 'merge'
        assert kwargs['_source'] == 'retro_stamp_topics'
        assert 'content' not in kwargs, (
            f'a content argument would re-embed the record: {kwargs}'
        )
        assert 'metadata_delete_keys' not in kwargs, (
            f'delete keys would leave the set_payload fast path: {kwargs}'
        )
        assert result['outcome'] == 'stamped'
        assert result['memory_id'] == M1
        # Point-id stability, read straight off the response envelope — the
        # service docstring echoes `id` precisely so a caller need not refetch.
        assert result['response']['id'] == M1

    @pytest.mark.asyncio
    async def test_already_stamped_record_issues_no_write(self):
        """Run two costs ZERO writes, not zero net effect.

        An ``update_memory`` that happens to be a no-op still journals a
        write op and still trips the amendment storm counters, so "the patch
        is empty" has to mean "no call" — which is also what makes the
        report's stamped count an honest measure of what the corpus gained.
        """
        service = _service(record=_record(topic='topic-one'))
        result = await _mod.stamp_one(service, _target(), apply=True)

        service.update_memory.assert_not_awaited()
        assert result['outcome'] == 'already_stamped'
        assert 'topic_already_present' in result['dispositions']

    @pytest.mark.asyncio
    async def test_missing_record_is_reported_not_written(self):
        """A consolidated-away member is a fact, not a failure to swallow.

        The manifest and the live cluster keys both point at ids that were
        already merged away (measured: gate 3036's ``19705df4`` no longer
        resolves), so this is an expected outcome — but it must reach the
        report, because a silently-dropped id reads as a stamped one.
        """
        service = _service(record=None)
        result = await _mod.stamp_one(service, _target(), apply=True)

        service.update_memory.assert_not_awaited()
        assert result['outcome'] == 'memory_not_found'

    @pytest.mark.asyncio
    async def test_update_memory_error_envelope_is_not_counted_as_a_stamp(self):
        """``{'error_type': ...}`` is a REJECTION, and returns 200-shaped.

        ``update_memory`` reports a not-found by returning a structured
        envelope rather than raising, so a caller that only catches
        exceptions would count a refused write as a successful stamp.
        """
        service = _service(record=_record())
        service.update_memory.side_effect = None
        service.update_memory.return_value = {
            'error': 'Memory does not exist in mem0',
            'error_type': 'MemoryNotFound',
            'store': 'mem0',
            'id': M1,
        }
        result = await _mod.stamp_one(service, _target(), apply=True)

        assert service.update_memory.await_count == 1
        assert result['outcome'] == 'update_failed'
        assert result['error_type'] == 'MemoryNotFound'

    @pytest.mark.asyncio
    async def test_canonical_uniqueness_is_probed_before_the_write(self):
        """The probe runs FIRST, with ε's exact filter.

        ``update_memory`` never reaches
        ``_apply_memory_metadata_validation``, so this script is the only
        layer standing between a plan and a second ``canonical: true`` for
        one ``(project, topic)``.  A probe issued after the write would
        observe the violation it was meant to prevent.
        """
        service = _service(record=_record())
        await _mod.stamp_one(service, _target(make_canonical=True), apply=True)

        service.count_memories_by_metadata.assert_awaited_once_with(
            'dark_factory', {'topic': 'topic-one', 'canonical': True},
        )
        names = _call_names(service)
        assert names.index('count_memories_by_metadata') < names.index('update_memory')
        assert service.update_memory.await_args.kwargs['metadata_patch'] == {
            'topic': 'topic-one', 'canonical': True,
        }

    @pytest.mark.asyncio
    async def test_an_incumbent_canonical_blocks_the_write(self):
        """A different incumbent means the plan was wrong — refuse it whole.

        Not "stamp the topic and skip the canonical": if two records claim
        one topic's canonical, which cluster this record belongs to is
        exactly what is in doubt.  The incumbent's id is named so the report
        line is actionable without a follow-up query.
        """
        service = _service(record=_record(), count=1, incumbent=C1)
        result = await _mod.stamp_one(service, _target(make_canonical=True), apply=True)

        service.update_memory.assert_not_awaited()
        assert result['outcome'] == 'canonical_uniqueness_blocked'
        assert result['incumbent_id'] == C1

    @pytest.mark.asyncio
    async def test_when_the_incumbent_is_the_target_itself_the_stamp_proceeds(self):
        """Finding ourselves is not a conflict.

        The probe cannot exclude the record being written, so a re-run over a
        partially-stamped corpus sees its own row come back — treating that
        as a violation would make the sweep unable to finish what it started.
        """
        service = _service(record=_record(), count=1, incumbent=M1)
        result = await _mod.stamp_one(service, _target(make_canonical=True), apply=True)

        assert result['outcome'] == 'stamped'
        assert service.update_memory.await_count == 1

    @pytest.mark.asyncio
    async def test_zero_count_proceeds_without_a_second_round_trip(self):
        """The happy path pays exactly one count and never scrolls.

        Same guard order ``_check_canonical_uniqueness`` documents: the
        incumbent-naming scroll is confined to the path that is already
        refusing the write.
        """
        service = _service(record=_record(), count=0)
        result = await _mod.stamp_one(service, _target(make_canonical=True), apply=True)

        assert result['outcome'] == 'stamped'
        service.get_memories_by_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_probe_timeout_fails_closed_and_stamps_nothing(self):
        """Fail CLOSED — the opposite of the service seam's warn-mode default.

        ``_check_canonical_uniqueness`` fails open under warn mode because
        failing a valid write when only the complaint machinery broke is
        worse than the duplicate.  Here the calculus inverts: this script IS
        the enforcement layer, it is writing canonicals in bulk unattended,
        and a re-run is free.  Reported under its own outcome rather than
        folded into ``canonical_uniqueness_blocked``, because "the store was
        unreachable" and "a duplicate exists" are different facts.
        """
        service = _service(record=_record())
        service.count_memories_by_metadata.side_effect = TimeoutError('qdrant timeout')
        result = await _mod.stamp_one(service, _target(make_canonical=True), apply=True)

        service.update_memory.assert_not_awaited()
        assert result['outcome'] == 'canonical_probe_failed'
        assert 'TimeoutError' in result['error']

    @pytest.mark.asyncio
    async def test_a_non_canonical_target_never_probes(self):
        """Guard 1: almost every target is a plain member and pays no probe."""
        service = _service(record=_record())
        await _mod.stamp_one(service, _target(make_canonical=False), apply=True)

        service.count_memories_by_metadata.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dry_run_reads_and_decides_but_never_writes(self):
        """``apply=False`` is a real rehearsal, not a different code path.

        The read and the decision still happen, so the dry-run report states
        what an apply would actually do — including the probe verdict, which
        is where a plan-level surprise would surface.
        """
        service = _service(record=_record())
        result = await _mod.stamp_one(service, _target(make_canonical=True), apply=False)

        service.update_memory.assert_not_awaited()
        service.get_memory_by_id.assert_awaited_once()
        service.count_memories_by_metadata.assert_awaited_once()
        assert result['outcome'] == 'would_stamp'
        assert result['patch'] == {'topic': 'topic-one', 'canonical': True}

    @pytest.mark.asyncio
    async def test_a_conflicting_human_topic_refuses_the_whole_write(self):
        """A retro sweep must not be able to destroy a topic a human set.

        And it must not stamp ``canonical`` either: ``canonical`` is scoped
        BY topic, so asserting it on a record filed under a different topic
        would put the wrong record at the head of the wrong cluster.
        """
        service = _service(record=_record(topic='a-different-topic'))
        result = await _mod.stamp_one(service, _target(make_canonical=True), apply=True)

        service.update_memory.assert_not_awaited()
        assert result['outcome'] == 'conflicting_existing_topic'
        assert result['existing_topic'] == 'a-different-topic'
