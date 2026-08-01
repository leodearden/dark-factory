"""Tests for :mod:`fused_memory.topic_slug` — the ONE topic namespace
(task 3198, leaf ε of ``docs/prds/memory-metadata-vocabulary.md``).

PRD D4 makes ``metadata.topic`` and ``ProceduralTopicCluster.topic_id`` a
single namespace with a single regex.  INV-5 says a shared rule gets ONE
home that both consumers import — never two copies kept in sync by prose.
These tests pin that mechanically: the leaf exists, it is import-light,
and both consumers validate through the *same objects*.
"""

import re
import subprocess
import sys

import pytest

from fused_memory.config.schema import _default_topic_guard_clusters
from fused_memory.topic_slug import (
    TOPIC_SLUG_MAX_LEN,
    TOPIC_SLUG_RE,
    is_valid_topic_slug,
)

#: Accept/reject table shared by the leaf-module tests and the
#: one-namespace-one-verdict closure test (D4).  Kept as a single table so
#: the two consumers are provably checked against the SAME cases rather
#: than against two tables that could drift apart.
_SLUG_CASES: list[tuple[str, bool, str]] = [
    ('a', True, 'single char'),
    ('a-good-slug', True, 'canonical shape'),
    ('x1-2y', True, 'digits are legal inside segments'),
    ('a' * TOPIC_SLUG_MAX_LEN, True, 'exactly at the cap'),
    ('bad_slug', False, 'snake_case — the shape 98 of 352 live topics have'),
    ('Bad-Slug', False, 'uppercase'),
    ('bad topic-slug', False, 'embedded space'),
    ('good-slug\nevil', False, r'trailing newline — the \Z-not-$ case'),
    ('-lead', False, 'leading separator'),
    ('trail-', False, 'trailing separator'),
    ('double--hyphen', False, 'doubled separator'),
    ('', False, 'empty'),
    ('a' * (TOPIC_SLUG_MAX_LEN + 1), False, 'over the length cap'),
]


class TestTopicSlugLeafModule:
    """The leaf module's surface: two constants and one predicate."""

    def test_exposes_the_regex_and_the_cap(self):
        assert isinstance(TOPIC_SLUG_RE, re.Pattern)
        assert TOPIC_SLUG_MAX_LEN == 100

    def test_accepts_every_seeded_topic_cluster_id(self):
        """PRD §10's one hard requirement, re-pinned on the config side.

        The seeded ids are IMPORTED, never hand-copied — D4 makes cluster
        ids and ``metadata.topic`` one namespace, so a predicate that
        rejected a seeded id would split the namespace it exists to unify
        (and, once step-4 lands, would make the shipped default config
        unloadable).
        """
        clusters = _default_topic_guard_clusters()
        assert len(clusters) == 5, 'seeded cluster set changed — re-verify the slug rule'
        for cluster in clusters:
            assert is_valid_topic_slug(cluster.topic_id), (
                f'seeded cluster id {cluster.topic_id!r} must be a valid topic slug'
            )

    @pytest.mark.parametrize(('value', 'expected', 'why'), _SLUG_CASES)
    def test_predicate_verdicts(self, value, expected, why):
        assert is_valid_topic_slug(value) is expected, f'{value!r}: {why}'

    def test_predicate_returns_bool_not_a_match_object(self):
        """``is True``/``is False`` above only means anything if it is a bool.

        A naive ``return TOPIC_SLUG_RE.match(value)`` would be truthy and
        would satisfy an ``assert is_valid_topic_slug(x)`` — but it is a
        ``re.Match``, not a verdict, and pydantic/JSON consumers would
        choke on it.
        """
        assert is_valid_topic_slug('a-good-slug') is True
        assert is_valid_topic_slug('bad_slug') is False

    def test_predicate_rejects_non_strings_without_raising(self):
        """It takes ``object`` — a non-str value is a verdict, not a crash.

        The config-side validator and the metadata validator both hand it
        untrusted values; ``TOPIC_SLUG_RE.match(None)`` would raise
        ``TypeError`` and turn a rejection into a stack trace.
        """
        for value in (None, 1, 1.5, True, b'a-good-slug', ['a-good-slug']):
            assert is_valid_topic_slug(value) is False

    def test_cap_is_applied_by_length_not_by_the_regex(self):
        """The over-length case must prove the LENGTH check, not the regex.

        Mirrors the tautology note at ``test_memory_metadata.py:101``: the
        regex ACCEPTS the over-length value, so if the predicate only
        consulted the regex the ``'a' * (MAX + 1)`` case above would pass
        for the wrong reason and deleting the length clause would go
        unnoticed.
        """
        over = 'a' * (TOPIC_SLUG_MAX_LEN + 1)
        assert TOPIC_SLUG_RE.match(over), 'the regex alone must NOT reject it'
        assert is_valid_topic_slug(over) is False

    def test_module_is_import_light(self):
        """The whole point of the extraction: stdlib-only, no cycle, no mem0.

        Probed in a FRESH interpreter, because this test process has
        already imported ``fused_memory.memory_metadata`` (and therefore
        ``mem0``) via the module-level imports above — an in-process
        ``sys.modules`` check would pass vacuously.

        If this fails, someone re-inlined a ``fused_memory`` import into
        the leaf and the config side is back on the measured
        ``ImportError: cannot import name 'FusedMemoryConfig'`` cycle.
        """
        probe = (
            'import sys; import fused_memory.topic_slug as t; '
            "assert 'mem0' not in sys.modules, sorted(k for k in sys.modules if 'mem0' in k); "
            "assert 'fused_memory.config.schema' not in sys.modules; "
            "assert 'fused_memory.memory_metadata' not in sys.modules; "
            'assert t.TOPIC_SLUG_MAX_LEN == 100'
        )
        result = subprocess.run(
            [sys.executable, '-c', probe], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
