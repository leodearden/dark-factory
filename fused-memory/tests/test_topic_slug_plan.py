"""Tests for ``fused_memory/maintenance/topic_slug_plan.py``, the pure planner
behind ``scripts/normalize_topic_slugs.py`` (task 4878).

Pure: plain scroll-shaped dicts in, plan rows and refusals out, and no service,
so these tests load neither the script nor a store.
"""
from __future__ import annotations

import dataclasses

import pytest
from _topic_slug_migration_doubles import by_reason, rec

from fused_memory.maintenance.topic_slug_plan import (
    in_selection,
    is_non_conforming_topic,
    plan_renames,
    refuse_invalid_selection,
)

# ===========================================================================
# plan_renames — the pure planner
# ===========================================================================

class TestPlanRenames:
    """``plan_renames(records, *, project_id) -> (renames, skips)``.

    Pure: it takes plain scroll-shaped dicts and no service, so a wrong answer
    here can never be masked by a mock.  It decides what the migration WOULD
    do; nothing in it touches a store.
    """

    def test_non_conforming_topic_that_folds_yields_one_rename(self):
        renames, skips = plan_renames(
            [rec('m1', 'mem0_tombstone_coverage')], project_id='dark_factory',
        )
        assert skips == []
        assert len(renames) == 1
        (rename,) = renames
        assert rename.project_id == 'dark_factory'
        assert rename.memory_id == 'm1'
        assert rename.old_topic == 'mem0_tombstone_coverage'
        assert rename.new_topic == 'mem0-tombstone-coverage'

    def test_rename_is_frozen(self):
        """A plan row must not be mutated between planning and writing.

        The whole point of planning the corpus before touching it is that the
        artifact and the writes agree; a mutable row lets them diverge.
        """
        (rename,), _ = plan_renames(
            [rec('m1', 'a_b')], project_id='dark_factory',
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            rename.new_topic = 'something-else'  # type: ignore[misc]

    def test_conforming_topic_is_neither_a_rename_nor_a_skip(self):
        """Already-good is not work, and a skip line would MISREPORT it.

        Most of the corpus conforms.  Filing each conforming record as a skip
        would bury the ~141 records that actually need attention under tens of
        thousands of lines saying "nothing to do" — and would make the skip
        buckets, which exist to surface coverage holes, unreadable.
        """
        renames, skips = plan_renames(
            [rec('m1', 'already-good'), rec('m2', 'x1-2y')],
            project_id='dark_factory',
        )
        assert renames == []
        assert skips == []

    def test_record_without_a_topic_key_is_ignored_entirely(self):
        """No ``topic`` is retro_stamp_topics' job, not this script's.

        This sweep NORMALIZES the value of a topic that exists; stamping one
        onto a record that carries none is the bounded sibling's job, and
        claiming it here would double-report the same corpus gap in two
        artifacts with two different definitions of it.
        """
        renames, skips = plan_renames(
            [rec('m1'), rec('m2', canonical=True)], project_id='dark_factory',
        )
        assert renames == []
        assert skips == []

    def test_topic_present_but_none_is_ignored_entirely(self):
        """A null ``topic`` is an absent topic, not a malformed one.

        ``derive_topic_slug(None)`` is ``None``, so without this case a null
        would land in the ``topic_unfoldable`` bucket and read as a value a
        human needs to adjudicate — when in fact there is nothing there.
        """
        renames, skips = plan_renames(
            [{'id': 'm1', 'created_at': None, 'metadata': {'topic': None}}],
            project_id='dark_factory',
        )
        assert renames == []
        assert skips == []

    def test_unfoldable_topic_is_a_skip_naming_the_raw_value(self):
        """No honest fold exists — report it, never guess one.

        The skip has to carry the RAW value: an operator resolving this by
        hand needs to see what is actually on the record, and the whole
        reason it is here is that the script refused to name a slug for it.
        """
        renames, skips = plan_renames(
            [rec('m1', '!!!')], project_id='dark_factory',
        )
        assert renames == []
        entries = by_reason(skips, 'topic_unfoldable')
        assert len(entries) == 1
        entry = entries[0]
        assert entry['project_id'] == 'dark_factory'
        assert entry['memory_id'] == 'm1'
        assert entry['topic'] == '!!!'
        assert 'new_topic' not in entry, 'a refusal must not carry a guessed slug'

    def test_over_long_topic_is_unfoldable_not_truncated(self):
        """The cap refuses rather than repairs — pinned at the planner too.

        ``derive_topic_slug`` already returns ``None`` here, but the planner
        is where that ``None`` becomes an operator-visible line rather than a
        silently dropped record.
        """
        over = 'a' * 101
        renames, skips = plan_renames(
            [rec('m1', over)], project_id='dark_factory',
        )
        assert renames == []
        assert by_reason(skips, 'topic_unfoldable')[0]['topic'] == over

    def test_non_str_topic_is_unfoldable(self):
        """Untrusted values come straight off live records."""
        renames, skips = plan_renames(
            [{'id': 'm1', 'created_at': None, 'metadata': {'topic': 42}}],
            project_id='dark_factory',
        )
        assert renames == []
        assert by_reason(skips, 'topic_unfoldable')[0]['topic'] == 42

    def test_an_unfoldable_entry_names_the_value_type(self):
        """In markdown, ``topic=42`` and ``topic='42'`` differ only by quote marks."""
        _, skips = plan_renames(
            [rec('m1', '!!!'), rec('m2', 42), rec('m3', ['a'])],
            project_id='dark_factory',
        )
        assert [
            (s['memory_id'], s['topic'], s['topic_type'])
            for s in by_reason(skips, 'topic_unfoldable')
        ] == [('m1', '!!!', 'str'), ('m2', 42, 'int'), ('m3', ['a'], 'list')]

    def test_many_records_sharing_one_legacy_slug_each_get_a_rename(self):
        """One legacy VALUE can sit on many records; each needs its own write.

        The distinct-value count is what the baseline delta grades (item 4);
        the per-record count is what the write loop does.  Conflating them is
        how a migration reports "1 topic normalized" and leaves 7 records
        behind.
        """
        renames, _ = plan_renames(
            [rec(f'm{i}', 'shared_legacy') for i in range(8)],
            project_id='dark_factory',
        )
        assert len(renames) == 8
        assert {r.new_topic for r in renames} == {'shared-legacy'}

    def test_output_is_sorted_for_a_clean_diff(self):
        """Sorted by ``(project_id, new_topic, memory_id)``.

        Two dry runs over an unchanged corpus must produce byte-comparable
        artifacts, and scroll order is not stable.  Without this an operator
        diffing consecutive rehearsals sees churn that means nothing.
        """
        renames, _ = plan_renames(
            [
                rec('m9', 'zeta_topic'),
                rec('m2', 'alpha_topic'),
                rec('m1', 'zeta_topic'),
                rec('m5', 'alpha_topic'),
            ],
            project_id='dark_factory',
        )
        assert [(r.new_topic, r.memory_id) for r in renames] == [
            ('alpha-topic', 'm2'),
            ('alpha-topic', 'm5'),
            ('zeta-topic', 'm1'),
            ('zeta-topic', 'm9'),
        ]

    def test_skips_are_sorted_too(self):
        """Same argument as the renames: a rehearsal diff must be readable."""
        _, skips = plan_renames(
            [rec('m9', '!!!'), rec('m1', '???')], project_id='dark_factory',
        )
        assert [s['memory_id'] for s in by_reason(skips, 'topic_unfoldable')] == [
            'm1', 'm9',
        ]

    def test_record_without_an_id_is_skipped_not_planned(self):
        """A row with no id cannot be written to; refuse it loudly.

        ``update_memory`` is addressed by memory id.  A record whose id is
        missing or empty would otherwise reach the write boundary and fail
        there, one row at a time, inside a report that already claimed it as
        planned work.
        """
        renames, skips = plan_renames(
            [{'id': '', 'created_at': None, 'metadata': {'topic': 'a_b'}}],
            project_id='dark_factory',
        )
        assert renames == []
        assert len(by_reason(skips, 'record_without_id')) == 1

    def test_empty_input_is_empty_output(self):
        assert plan_renames([], project_id='dark_factory') == ([], [])


# ===========================================================================
# Collision classification — the most dangerous case in the migration
# ===========================================================================

class TestSlugCollision:
    """A fold that lands on an OCCUPIED slug is refused, never merged.

    Renaming ``foo_bar`` to ``foo-bar`` when records already carry
    ``foo-bar`` is not a normalization — it MERGES two topic namespaces,
    changing cluster membership for records this migration never examined.

    Worse, ``topic`` is what SCOPES canonical uniqueness
    (``memory_service._check_canonical_uniqueness`` probes
    ``{'topic': T, 'canonical': True}``), so merging two clusters that each
    hold a canonical manufactures exactly the second canonical that invariant
    exists to prevent.  Under the shipped ``memory_metadata.enforce = false``
    the service seam WARN-fails open, so it would land silently — and this
    script would be the thing that created it.

    Refusing makes the verdict a property of the CORPUS rather than of scroll
    order.  That is the same posture ``retro_stamp_topics.stamp_one`` takes for
    ``stray_canonical_on_member``, for the same fail-closed-because-unattended
    -and-bulk reason.
    """

    def test_fold_onto_an_occupied_slug_is_refused(self):
        renames, skips = plan_renames(
            [
                rec('m1', 'foo_bar'),
                rec('m2', 'foo_bar'),
                rec('incumbent', 'foo-bar'),
            ],
            project_id='dark_factory',
        )
        assert renames == [], 'the merge must not be planned'
        entries = by_reason(skips, 'slug_collision')
        assert len(entries) == 1, 'one entry per collision, not one per record'
        entry = entries[0]
        assert entry['project_id'] == 'dark_factory'
        assert entry['old_topic'] == 'foo_bar'
        assert entry['new_topic'] == 'foo-bar'
        assert entry['colliding_count'] == 2
        assert entry['memory_ids'] == ['m1', 'm2']

    def test_the_incumbent_records_are_left_alone(self):
        """The incumbent already conforms — it is not the migration's business.

        It appears in the skip only as the REASON for the refusal.  Touching
        it would be the very namespace merge being refused, in the other
        direction.
        """
        renames, _ = plan_renames(
            [rec('m1', 'foo_bar'), rec('incumbent', 'foo-bar')],
            project_id='dark_factory',
        )
        assert 'incumbent' not in [r.memory_id for r in renames]

    def test_collision_is_scoped_per_project(self):
        """``foo-bar`` in reify does not block ``foo_bar`` in dark_factory.

        Topics are per-corpus; a cross-project "collision" is two unrelated
        clusters that happen to share a name, and refusing on it would strand
        real work for no reason.  ``plan_renames`` sees one project at a time,
        so this is pinned by construction — the test guards the construction.
        """
        renames, skips = plan_renames(
            [rec('m1', 'foo_bar')], project_id='dark_factory',
        )
        assert [r.new_topic for r in renames] == ['foo-bar']
        assert skips == []

    def test_two_legacy_slugs_folding_to_one_target_is_a_collision(self):
        """``foo_bar`` and ``foo.bar`` both fold to ``foo-bar`` — refuse BOTH.

        Never resolved by scroll order.  Letting the first one through would
        make the outcome depend on which page the backend returned first, and
        would leave the second refused for a reason ("occupied") this script
        itself manufactured a moment earlier.
        """
        renames, skips = plan_renames(
            [rec('m1', 'foo_bar'), rec('m2', 'foo.bar')],
            project_id='dark_factory',
        )
        assert renames == []
        entries = by_reason(skips, 'slug_collision')
        assert len(entries) == 2, 'both source slugs are refused'
        assert sorted(e['old_topic'] for e in entries) == ['foo.bar', 'foo_bar']
        assert {e['new_topic'] for e in entries} == {'foo-bar'}

    def test_two_legacy_slugs_refusal_is_order_independent(self):
        """Reversing the input must not change the verdict for either slug."""
        forward, _ = plan_renames(
            [rec('m1', 'foo_bar'), rec('m2', 'foo.bar')],
            project_id='dark_factory',
        )
        reverse, _ = plan_renames(
            [rec('m2', 'foo.bar'), rec('m1', 'foo_bar')],
            project_id='dark_factory',
        )
        assert forward == reverse == []

    def test_an_unrelated_clean_rename_still_proceeds(self):
        """A refusal is scoped to its slug, not to the whole run.

        ~141 records need this migration; one unresolvable collision must not
        strand the rest.
        """
        renames, skips = plan_renames(
            [
                rec('m1', 'foo_bar'),
                rec('incumbent', 'foo-bar'),
                rec('m3', 'clean_topic'),
            ],
            project_id='dark_factory',
        )
        assert [(r.memory_id, r.new_topic) for r in renames] == [
            ('m3', 'clean-topic'),
        ]
        assert len(by_reason(skips, 'slug_collision')) == 1


class TestCanonicalCollision:
    """The severe case: a merge that would manufacture a second canonical.

    Distinct from ``slug_collision`` because the consequence is different in
    kind.  A plain namespace merge changes cluster membership; a merge across
    two clusters that EACH hold a ``canonical: true`` record violates the
    uniqueness invariant ``_check_canonical_uniqueness`` exists to hold — and
    does so through a seam that warn-fails OPEN under the shipped
    ``enforce = false``, so nothing downstream would object.

    Reported separately so an operator triaging the artifact can see at a
    glance which refusals are bookkeeping and which are corruption avoided.
    """

    def test_two_canonicals_across_the_merge_is_the_severe_outcome(self):
        renames, skips = plan_renames(
            [
                rec('legacy-canon', 'foo_bar', canonical=True),
                rec('legacy-member', 'foo_bar'),
                rec('incumbent-canon', 'foo-bar', canonical=True),
            ],
            project_id='dark_factory',
        )
        assert renames == []
        assert by_reason(skips, 'slug_collision') == [], (
            'the severe outcome must not ALSO be filed as the mild one'
        )
        entries = by_reason(skips, 'canonical_collision')
        assert len(entries) == 1
        entry = entries[0]
        assert entry['project_id'] == 'dark_factory'
        assert entry['old_topic'] == 'foo_bar'
        assert entry['new_topic'] == 'foo-bar'
        assert entry['old_canonical_ids'] == ['legacy-canon']
        assert entry['new_canonical_ids'] == ['incumbent-canon']

    def test_only_the_source_side_canonical_is_the_mild_outcome(self):
        """One canonical across the merge does not violate uniqueness.

        The merged cluster would still hold exactly one, so the refusal is
        the ordinary namespace-merge one — reported, but not escalated to the
        severe bucket, which would cry wolf and devalue the real cases.
        """
        _, skips = plan_renames(
            [
                rec('legacy-canon', 'foo_bar', canonical=True),
                rec('incumbent', 'foo-bar'),
            ],
            project_id='dark_factory',
        )
        assert len(by_reason(skips, 'slug_collision')) == 1
        assert by_reason(skips, 'canonical_collision') == []

    def test_only_the_target_side_canonical_is_the_mild_outcome(self):
        _, skips = plan_renames(
            [
                rec('legacy', 'foo_bar'),
                rec('incumbent-canon', 'foo-bar', canonical=True),
            ],
            project_id='dark_factory',
        )
        assert len(by_reason(skips, 'slug_collision')) == 1
        assert by_reason(skips, 'canonical_collision') == []

    def test_two_legacy_slugs_each_canonical_is_also_severe(self):
        """The two-legacy-slugs collision escalates the same way.

        ``foo_bar`` and ``foo.bar`` each holding a canonical would merge into
        one cluster with two — the same violation, arrived at without any
        incumbent conforming record involved at all.
        """
        _, skips = plan_renames(
            [
                rec('c1', 'foo_bar', canonical=True),
                rec('c2', 'foo.bar', canonical=True),
            ],
            project_id='dark_factory',
        )
        assert len(by_reason(skips, 'canonical_collision')) == 2
        assert by_reason(skips, 'slug_collision') == []

    def test_canonical_must_be_true_not_merely_truthy(self):
        """``canonical: 'false'`` is a string and must not read as canonical.

        Metadata comes off a live store and is not schema-enforced here; a
        truthiness test would turn any stray non-empty value into a phantom
        canonical and escalate a mild refusal to the severe bucket.
        """
        _, skips = plan_renames(
            [
                rec('legacy', 'foo_bar', canonical='false'),
                rec('incumbent', 'foo-bar', canonical='false'),
            ],
            project_id='dark_factory',
        )
        assert by_reason(skips, 'canonical_collision') == []
        assert len(by_reason(skips, 'slug_collision')) == 1


# ===========================================================================
# The --topic selection: which legacy values a plan returns verdicts for
# ===========================================================================

#: Selections that name no foldable legacy value, each with the text the
#: refusal must name.  The conforming target slug is the likely typo.
INVALID_SELECTIONS = [
    pytest.param(frozenset(), 'empty', id='empty'),
    pytest.param(frozenset({'legacy-a'}), 'legacy-a', id='conforming-target'),
    pytest.param(frozenset({'a' * 101}), 'a' * 101, id='no-fold'),
    pytest.param(frozenset({42}), '42', id='non-str'),
]


class TestTopicSelection:
    """A selection names LEGACY source values and narrows the verdicts only.

    The collision index stays corpus-wide (see ``TestSlugCollision``); what a
    selection removes is every verdict about an unselected value.
    """

    def test_an_unselected_value_gets_no_verdict_of_any_kind(self):
        records = [
            rec('m1', '!!!'),
            rec('', 'no_id'),
            rec('m2', 'dup_topic'), rec('m3', 'dup-topic'),
            rec('m4', 'plain_legacy'),
        ]

        unnarrowed = plan_renames(records, project_id='dark_factory')
        narrowed = plan_renames(records, project_id='dark_factory',
                                topics=frozenset({'other_legacy'}))

        assert sorted(entry['reason'] for entry in unnarrowed[1]) == [
            'record_without_id', 'slug_collision', 'topic_unfoldable',
        ]
        assert [rename.old_topic for rename in unnarrowed[0]] == ['plain_legacy']
        assert narrowed == ([], [])

    @pytest.mark.parametrize(('topics', 'named'), INVALID_SELECTIONS)
    def test_an_invalid_selection_is_refused(self, topics, named):
        with pytest.raises(ValueError) as refused:
            refuse_invalid_selection(topics)

        assert named in str(refused.value)

    @pytest.mark.parametrize(('topics', 'named'), INVALID_SELECTIONS)
    def test_the_planner_refuses_an_invalid_selection_itself(self, topics, named):
        """Planning nothing for a typo would read as a clean run to any caller."""
        with pytest.raises(ValueError) as refused:
            plan_renames([rec('m1', 'legacy_a')], project_id='dark_factory',
                         topics=topics)

        assert named in str(refused.value)

    def test_no_selection_is_not_refused(self):
        assert refuse_invalid_selection(None) is None

    def test_a_drained_legacy_value_is_not_refused(self):
        """A re-run after a successful ``--apply`` finds the value gone."""
        drained = frozenset({'drained_legacy'})

        assert refuse_invalid_selection(drained) is None
        assert plan_renames([rec('m1', 'drained-legacy')], project_id='dark_factory',
                            topics=drained) == ([], [])

    def test_no_selection_covers_every_value(self):
        assert in_selection('legacy_a', None)
        assert in_selection(42, None)

    def test_a_selection_covers_only_its_own_values(self):
        selection = frozenset({'legacy_a'})

        assert in_selection('legacy_a', selection)
        assert not in_selection('legacy_b', selection)

    def test_a_non_string_topic_is_never_in_a_selection(self):
        assert not in_selection(42, frozenset({'42'}))


class TestIsNonConformingTopic:
    """A PRESENT topic the slug rule rejects; an absent one is not this sweep's."""

    @pytest.mark.parametrize(('topic', 'non_conforming'), [
        pytest.param(None, False, id='absent'),
        pytest.param('good-topic', False, id='conforming'),
        pytest.param('legacy_a', True, id='foldable'),
        pytest.param('!!!', True, id='unfoldable'),
        pytest.param(42, True, id='non-str'),
    ])
    def test_verdict(self, topic, non_conforming):
        assert is_non_conforming_topic(topic) is non_conforming
