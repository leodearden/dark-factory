"""Tests for :mod:`fused_memory.memory_metadata` — the normative Mem0
metadata vocabulary registry (task 3195, leaf β of
``docs/prds/memory-metadata-vocabulary.md``).

The registry is the single normative home for the Mem0 metadata
vocabulary (PRD V1 / INV-5): consumers import it, they never restate it.
These tests pin that contract mechanically.
"""

import re

import pytest

from fused_memory.config.schema import _default_topic_guard_clusters
from fused_memory.memory_metadata import (
    TOPIC_SLUG_MAX_LEN,
    TOPIC_SLUG_RE,
    normalize_supersedes,
)


class TestTopicSlug:
    """`topic` slug shape — PRD D4 (one topic namespace, one shared regex)."""

    def test_accepts_every_seeded_topic_cluster_id(self):
        """PRD §10 hard requirement: the regex MUST accept all 5 seeded
        ``ProceduralTopicCluster.topic_id`` values.

        The seeded ids are imported rather than hand-copied so this stays
        enforced if the seeded set is ever edited (D4 makes cluster ids and
        ``metadata.topic`` the same namespace, so a regex that rejected a
        seeded id would split the namespace it exists to unify).
        """
        clusters = _default_topic_guard_clusters()
        assert len(clusters) == 5, 'seeded cluster set changed — re-verify the regex'
        for cluster in clusters:
            assert TOPIC_SLUG_RE.match(cluster.topic_id), (
                f'seeded cluster id {cluster.topic_id!r} must match the slug regex'
            )
            assert len(cluster.topic_id) <= TOPIC_SLUG_MAX_LEN

    @pytest.mark.parametrize(
        'value',
        [
            'a',
            'a-b',
            'x1-2y',
            'eval-worktree-venv-shadowing',
            'a' * TOPIC_SLUG_MAX_LEN,  # exactly at the cap
        ],
    )
    def test_accepts_conforming_slugs(self, value):
        assert TOPIC_SLUG_RE.match(value)
        assert len(value) <= TOPIC_SLUG_MAX_LEN

    @pytest.mark.parametrize(
        ('value', 'why'),
        [
            ('escalation_server_ops', 'snake_case — the shape 98 of 352 live topics have'),
            ('Foo-Bar', 'uppercase'),
            ('-lead', 'leading separator'),
            ('trail-', 'trailing separator'),
            ('a--b', 'doubled separator'),
            ('', 'empty'),
        ],
    )
    def test_rejects_nonconforming_slugs(self, value, why):
        assert not TOPIC_SLUG_RE.match(value), f'{value!r} must be rejected ({why})'

    def test_rejects_over_length_slug(self):
        """The cap is enforced by length, not by the regex itself."""
        over = 'a' * (TOPIC_SLUG_MAX_LEN + 1)
        assert len(over) > TOPIC_SLUG_MAX_LEN

    def test_max_len_is_100(self):
        # Basis (measured against plans/memory-metadata-census-report.json
        # @ b5af3e4b03): longest conforming live `topic` is 69 chars, longest
        # seeded ProceduralTopicCluster.topic_id is 52 chars. 100 therefore
        # bounds the key while rejecting nothing observed.
        assert TOPIC_SLUG_MAX_LEN == 100

    def test_regex_is_anchored(self):
        """An unanchored regex would accept embedded junk — pin both ends."""
        assert not TOPIC_SLUG_RE.match('bad topic-slug')
        assert not TOPIC_SLUG_RE.match('good-slug\nevil')
        assert isinstance(TOPIC_SLUG_RE, re.Pattern)


class TestNormalizeSupersedes:
    """PRD D2 — `supersedes` is a list; the helper accepts scalar/list/None.

    Readers: ``reconciliation/targeted.py:1464`` and leaf 3112's closure
    predicate. The scalar writer is ``reconciliation/harness.py:1167``.
    """

    def test_none_becomes_empty_list(self):
        assert normalize_supersedes(None) == []

    def test_scalar_str_becomes_single_element_list(self):
        assert normalize_supersedes('abc-uuid') == ['abc-uuid']

    def test_list_is_copied_not_aliased(self):
        """The caller's list must not be aliased — a later in-place mutation
        of the returned list would otherwise reach back into caller state."""
        original = ['a', 'b']
        result = normalize_supersedes(original)
        assert result == ['a', 'b']
        assert result is not original

    def test_tuple_becomes_list(self):
        assert normalize_supersedes(('a', 'b')) == ['a', 'b']

    def test_empty_list_stays_empty(self):
        assert normalize_supersedes([]) == []

    def test_non_str_member_is_preserved_not_dropped(self):
        """Silently dropping or coercing a malformed member would be a
        silent-fail-soft: the SHAPE VALIDATOR rejects it by name, so the
        normalizer must hand it through intact for that to be possible."""
        assert normalize_supersedes([42]) == [42]
        assert normalize_supersedes(['ok-uuid', 42]) == ['ok-uuid', 42]

    @pytest.mark.parametrize(
        'value', [None, 'scalar', ['a', 'b'], ('a',), [], [42], ['a', 42]]
    )
    def test_idempotent(self, value):
        once = normalize_supersedes(value)
        assert normalize_supersedes(once) == once
