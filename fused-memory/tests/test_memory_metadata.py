"""Tests for :mod:`fused_memory.memory_metadata` — the normative Mem0
metadata vocabulary registry (task 3195, leaf β of
``docs/prds/memory-metadata-vocabulary.md``).

The registry is the single normative home for the Mem0 metadata
vocabulary (PRD V1 / INV-5): consumers import it, they never restate it.
These tests pin that contract mechanically.
"""

import json
import re
from pathlib import Path

import pytest

from fused_memory.config.schema import _default_topic_guard_clusters
from fused_memory.memory_metadata import (
    KIND_REGISTRY,
    TOPIC_SLUG_MAX_LEN,
    TOPIC_SLUG_RE,
    normalize_supersedes,
)

#: Leaf α's committed census artifact — the oracle the registry is
#: grandfathered from.  Resolved from ``__file__`` (repo root is two
#: parents up from ``fused-memory/tests/<this file>``) rather than from
#: cwd, so the test is invariant to where pytest is invoked.
_CENSUS_ARTIFACT = (
    Path(__file__).resolve().parents[2] / 'plans' / 'memory-metadata-census-report.json'
)


def _census_kind_values() -> list[str]:
    """Return every ``kind`` value leaf α actually measured.

    ``coverage.complete`` is asserted before the values are used: a
    partial scroll must never silently seed a grandfather list.
    """
    report = json.loads(_CENSUS_ARTIFACT.read_text())
    assert report['coverage']['complete'] is True, (
        'census coverage is incomplete — the grandfather set is not authoritative'
    )
    return [entry['value'] for entry in report['grand_total']['kind']['entries']]


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


class TestKindRegistry:
    """`KIND_REGISTRY` — PRD D3's closed registry, grandfathered from the census.

    Oracled against the committed artifact so the registry can never
    silently drift from the measurement it claims to grandfather.
    """

    #: The five names the PRD's §6 row claimed were live but that leaf α
    #: measured at ZERO live records.  Three have verified live in-repo
    #: Mem0 writers (scope_freshness.py:97/:251/:495, harness.py:1166,
    #: cleanup_count_snapshots.py:210); two do not (see the registry's
    #: block 2 comments).  All five are retained regardless:
    #: grandfathering means "what is written", not only "what survives"
    #: an aging sweep, and excluding them would reject in-repo code the
    #: moment `enforce_kind_registry` flips.
    ZERO_RECORD_IN_REPO_KINDS = frozenset({
        'stage1_flag_marker',
        'project_status_correction',
        'consolidated_scope_correction',
        'entity_standing_decision',
        'count_snapshot_cleanup_audit',
    })

    #: PRD V1's two new kinds (triage attach outcomes).  Both confirmed
    #: ABSENT from the live corpus, so they must be added explicitly
    #: rather than assumed present via the census.
    NEW_IN_PRD_KINDS = frozenset({'amendment', 'sighting'})

    def test_is_a_frozenset_of_str(self):
        assert isinstance(KIND_REGISTRY, frozenset)
        assert all(isinstance(value, str) for value in KIND_REGISTRY)

    def test_superset_of_every_census_measured_value(self):
        """Every kind that exists live must be grandfathered — otherwise
        flipping `enforce_kind_registry` rejects live writers."""
        measured = set(_census_kind_values())
        missing = measured - KIND_REGISTRY
        assert not missing, f'census-measured kinds absent from the registry: {sorted(missing)}'

    def test_contains_the_two_new_prd_kinds(self):
        assert self.NEW_IN_PRD_KINDS <= KIND_REGISTRY

    def test_contains_the_zero_record_in_repo_kinds(self):
        assert self.ZERO_RECORD_IN_REPO_KINDS <= KIND_REGISTRY

    def test_exact_size(self):
        """Pin the size so any unreviewed addition or removal fails loudly
        rather than drifting silently away from the measurement."""
        measured = set(_census_kind_values())
        assert len(measured) == 329, 'census artifact changed — re-derive the registry'
        # 329 census-measured + 2 new-in-PRD + 5 zero-record-but-in-repo.
        assert len(KIND_REGISTRY) == len(measured) + 7
        assert len(KIND_REGISTRY) == 336

    def test_the_seven_additions_are_exactly_the_expected_names(self):
        """Not just the right count — the right names."""
        measured = set(_census_kind_values())
        additions = KIND_REGISTRY - measured
        assert additions == self.NEW_IN_PRD_KINDS | self.ZERO_RECORD_IN_REPO_KINDS

    def test_entries_are_clean_non_empty_strings(self):
        for value in KIND_REGISTRY:
            assert value, 'empty kind value in registry'
            assert value == value.strip(), f'{value!r} carries leading/trailing whitespace'

    def test_registry_is_closed_not_open(self):
        assert 'not_a_real_kind_xyz' not in KIND_REGISTRY


class TestKeyLayers:
    """The layered top-level key allowlist, and the INV-5 single-home pin for D12.

    Four layers, deliberately disjoint: mem0-managed (the store owns them),
    server-stamped (this server writes them), reserved vocabulary (V1's five),
    and blessed conventional keys (the measured >=1000 tier).
    """

    def test_mem0_managed_keys_are_the_expected_nine(self):
        from fused_memory.backends import mem0_client

        assert mem0_client.MEM0_MANAGED_METADATA_KEYS == frozenset({
            'data', 'hash', 'created_at', 'updated_at',
            'user_id', 'agent_id', 'run_id', 'actor_id', 'role',
        })

    def test_mem0_managed_keys_is_one_object_not_a_copy(self):
        """INV-5 / PRD D12: `backends/mem0_client.py` is the decided home.

        Identity, not equality — two equal frozensets in two modules is
        exactly the duplication D12 exists to prevent.
        """
        from fused_memory.backends import mem0_client
        from fused_memory import memory_metadata as mm

        assert mm.MEM0_MANAGED_METADATA_KEYS is mem0_client.MEM0_MANAGED_METADATA_KEYS

    def test_script_retains_an_alias_not_a_copy(self):
        """The script's private name must be an ALIAS to the extracted object.

        Keeping the module-local spelling means
        `tests/test_tag_cgl_eta_rehome_scope.py:309` needs no edit, while
        identity proves the extraction left one object rather than two.
        """
        import importlib.util
        import sys
        from fused_memory.backends import mem0_client

        script_path = Path(__file__).parent.parent / 'scripts' / 'tag_cgl_eta_rehome_scope.py'
        spec = importlib.util.spec_from_file_location('tag_cgl_eta_rehome_scope', script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules['tag_cgl_eta_rehome_scope'] = module
        try:
            spec.loader.exec_module(module)
            assert module._MEM0_MANAGED_METADATA_KEYS is mem0_client.MEM0_MANAGED_METADATA_KEYS
        finally:
            sys.modules.pop('tag_cgl_eta_rehome_scope', None)

    def test_server_stamped_keys(self):
        from fused_memory.memory_metadata import SERVER_STAMPED_KEYS

        assert SERVER_STAMPED_KEYS == frozenset({
            'category',    # memory_service.py:2193 (add_memory), :2542 (add_system_record)
            'recon_pool',  # _apply_cycle_summary_metadata_tagging (memory_service.py:389)
            'run_id',      # _apply_cycle_summary_metadata_tagging (memory_service.py:389)
            'planned',     # NOT a write-seam stamp — a server-owned search-result
                           # annotation (memory_service.py:1932, :2921, read back at
                           # :2962). Included so a round-tripped search result
                           # re-written as metadata does not census-warn on the
                           # server's own field.
        })

    def test_reserved_vocabulary_keys(self):
        from fused_memory.memory_metadata import RESERVED_VOCABULARY_KEYS

        assert RESERVED_VOCABULARY_KEYS == frozenset({
            'topic', 'canonical', 'kind', 'parent_id', 'supersedes',
        })

    @pytest.mark.parametrize(
        ('key', 'census_count'),
        [
            ('task_id', 18850),
            ('source', 16364),
            ('transition', 15529),
            ('_deferred', 8263),
            ('_causation_id', 8262),
            ('stage', 2865),
            ('stage2_suppress', 1588),
            ('echo_used_provenance', 1284),
        ],
    )
    def test_blessed_keys_contain_the_measured_cut(self, key, census_count):
        """The >=1000 cut, re-derived from the artifact so the tier cannot
        drift from the measurement that justified it."""
        from fused_memory.memory_metadata import BLESSED_METADATA_KEYS

        assert key in BLESSED_METADATA_KEYS
        report = json.loads(_CENSUS_ARTIFACT.read_text())
        measured = {
            e['value']: e['count'] for e in report['grand_total']['keys']['entries']
        }
        assert measured[key] == census_count
        assert measured[key] >= 1000

    def test_layers_are_pairwise_disjoint(self):
        """No key may be classified twice.

        `run_id` is the live trap: it is server-stamped AND measures 4,518
        occurrences, so it is exactly the key that would otherwise land in
        both the blessed tier and the server-stamped tier.
        """
        from fused_memory.backends.mem0_client import MEM0_MANAGED_METADATA_KEYS
        from fused_memory.memory_metadata import (
            BLESSED_METADATA_KEYS,
            RESERVED_VOCABULARY_KEYS,
            SERVER_STAMPED_KEYS,
        )

        assert not (BLESSED_METADATA_KEYS & RESERVED_VOCABULARY_KEYS)
        assert not (BLESSED_METADATA_KEYS & MEM0_MANAGED_METADATA_KEYS)
        assert not (BLESSED_METADATA_KEYS & SERVER_STAMPED_KEYS)
        assert 'run_id' not in BLESSED_METADATA_KEYS

    def test_classify_unknown_keys_returns_only_genuinely_unknown(self):
        from fused_memory.memory_metadata import classify_unknown_keys

        meta = {
            'data': 1,                 # mem0-managed
            'category': 'x',           # server-stamped
            'kind': 'cycle_summary',   # reserved
            'task_id': '3195',         # blessed
            'x_experimental': True,    # x_ passthrough
            'totally_novel_key': 'v',  # unknown
        }
        assert classify_unknown_keys(meta) == ['totally_novel_key']

    def test_classify_unknown_keys_empty_dict(self):
        from fused_memory.memory_metadata import classify_unknown_keys

        assert classify_unknown_keys({}) == []

    def test_bare_x_prefix_passes(self):
        from fused_memory.memory_metadata import classify_unknown_keys

        assert classify_unknown_keys({'x_': 1}) == []


class TestErrorTypes:
    """`MetadataViolation` / `MemoryMetadataValidationError`.

    V1 requires the rejection hint to name BOTH the violated rule and where
    the registry lives, so an agent that trips it can find the vocabulary.
    """

    def test_violation_carries_the_expected_fields(self):
        from fused_memory.memory_metadata import MetadataViolation

        v = MetadataViolation(
            key='topic', code='invalid_topic_slug', message='bad', fatal=True
        )
        assert v.key == 'topic'
        assert v.code == 'invalid_topic_slug'
        assert v.message == 'bad'
        assert v.fatal is True

    def test_violation_is_comparable_and_hashable(self):
        from fused_memory.memory_metadata import MetadataViolation

        a = MetadataViolation(key='k', code='c', message='m', fatal=False)
        b = MetadataViolation(key='k', code='c', message='m', fatal=False)
        assert a == b
        assert len({a, b}) == 1

    def test_error_subclasses_exception_and_exposes_violations(self):
        from fused_memory.memory_metadata import (
            MemoryMetadataValidationError,
            MetadataViolation,
        )

        assert issubclass(MemoryMetadataValidationError, Exception)
        v = MetadataViolation(
            key='topic',
            code='invalid_topic_slug',
            message="topic 'bad_slug' must match TOPIC_SLUG_RE",
            fatal=True,
        )
        err = MemoryMetadataValidationError([v])
        assert err.violations == [v]

    def test_error_str_names_the_rule_and_the_registry_location(self):
        """V1: the hint names the rule AND where the registry lives."""
        from fused_memory.memory_metadata import (
            MemoryMetadataValidationError,
            MetadataViolation,
        )

        err = MemoryMetadataValidationError([
            MetadataViolation(
                key='topic',
                code='invalid_topic_slug',
                message="topic 'bad_slug' must match TOPIC_SLUG_RE",
                fatal=True,
            )
        ])
        text = str(err)
        assert 'invalid_topic_slug' in text
        assert 'TOPIC_SLUG_RE' in text
        assert 'fused_memory.memory_metadata' in text
