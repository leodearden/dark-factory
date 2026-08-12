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
        # 4 since b536ae972c retired both eval-worktree clusters
        # (`eval-worktree-plan-tools-missing`, `eval-worktree-venv-shadowing`)
        # from the seed, their gate tasks 2841/2844 being done. The count is a
        # canary, not the contract — the slug check below is. Re-verified: all
        # four remaining ids are valid slugs.
        assert len(clusters) == 4, 'seeded cluster set changed — re-verify the slug rule'
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


class TestOneNamespaceOneConstant:
    """INV-5: ONE home for the rule, both consumers importing the SAME objects.

    Every assertion here is ``is``, never ``==``. Equality would be
    satisfied by a re-typed copy of the pattern in either consumer — which
    is precisely the drift INV-5 forbids and which prose alone cannot
    prevent. Identity fails the moment someone inlines a second copy.
    """

    def test_memory_metadata_re_exports_the_same_objects(self):
        import fused_memory.memory_metadata as mm
        import fused_memory.topic_slug as ts

        assert mm.TOPIC_SLUG_RE is ts.TOPIC_SLUG_RE
        assert mm.TOPIC_SLUG_MAX_LEN is ts.TOPIC_SLUG_MAX_LEN
        assert mm.is_valid_topic_slug is ts.is_valid_topic_slug

    def test_registry_still_advertises_the_names(self):
        """The re-export is not a silent deprecation.

        ``memory_metadata`` remains the advertised vocabulary home, so the
        names must stay in its ``__all__`` — every existing importer keeps
        working through it.
        """
        import fused_memory.memory_metadata as mm

        for name in ('TOPIC_SLUG_RE', 'TOPIC_SLUG_MAX_LEN', 'is_valid_topic_slug'):
            assert name in mm.__all__

    def test_config_schema_validates_through_the_same_predicate(self):
        import fused_memory.config.schema as schema
        import fused_memory.topic_slug as ts

        assert schema.is_valid_topic_slug is ts.is_valid_topic_slug

    def test_every_seeded_cluster_id_satisfies_the_shared_predicate(self):
        for cluster in _default_topic_guard_clusters():
            assert is_valid_topic_slug(cluster.topic_id)

    @pytest.mark.parametrize(('value', 'expected', 'why'), _SLUG_CASES)
    def test_both_consumers_reach_the_same_verdict(self, value, expected, why):
        """D4's closure property: one namespace, one verdict.

        The memory side (``validate_memory_metadata``'s ``topic`` check) and
        the config side (``ProceduralTopicCluster.topic_id``) are driven
        over the SAME table. If they ever disagree on a single value, the
        namespace has split in two and 3135's auto-seed invariant
        (``cluster.topic_id == canonical.metadata.topic``) becomes
        unsatisfiable for that value.
        """
        import pydantic

        from fused_memory.config.schema import ProceduralTopicCluster
        from fused_memory.memory_metadata import validate_memory_metadata

        memory_codes = {
            v.code
            for v in validate_memory_metadata({'topic': value}, enforce_kind_registry=False)
        }
        memory_accepts = 'invalid_topic_slug' not in memory_codes

        try:
            ProceduralTopicCluster(topic_id=value, phrases=['a', 'b'])
            config_accepts = True
        except pydantic.ValidationError:
            config_accepts = False

        assert memory_accepts is expected, f'memory side disagreed on {value!r}: {why}'
        assert config_accepts is expected, f'config side disagreed on {value!r}: {why}'


class TestConfigSchemaImportStaysMem0Free:
    """Regression guard for the measured import cycle (task 3198).

    Reproduced before the fix: a ``config/schema.py`` carrying
    ``from fused_memory.memory_metadata import TOPIC_SLUG_RE`` raises
    ``ImportError: cannot import name 'FusedMemoryConfig' from
    'fused_memory.config.schema'`` — because ``memory_metadata`` imports
    ``backends.mem0_client``, which imports ``config.schema`` at module
    scope.

    Without this guard a future refactor could re-inline that import and
    break config loading for every mem0-less consumer, including
    ``orchestrator/tests/test_reopen_sticks_e2e.py:58``, which imports
    ``FusedMemoryConfig`` with no mem0 SDK present.
    """

    def test_importing_config_schema_pulls_neither_mem0_nor_the_registry(self):
        probe = (
            'import sys; from fused_memory.config.schema import FusedMemoryConfig; '
            "assert 'mem0' not in sys.modules, sorted(k for k in sys.modules if 'mem0' in k); "
            "assert 'fused_memory.memory_metadata' not in sys.modules; "
            'from fused_memory.config.schema import ProceduralTopicCluster as C; '
            "C(topic_id='a-good-slug', phrases=['a', 'b'])"
        )
        result = subprocess.run(
            [sys.executable, '-c', probe], capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, result.stderr
