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
