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
    MemoryMetadataValidationError,
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
        """PRD §10 hard requirement: the regex MUST accept all 6 seeded
        ``ProceduralTopicCluster.topic_id`` values.

        The seeded ids are imported rather than hand-copied so this stays
        enforced if the seeded set is ever edited (D4 makes cluster ids and
        ``metadata.topic`` the same namespace, so a regex that rejected a
        seeded id would split the namespace it exists to unify).
        """
        clusters = _default_topic_guard_clusters()
        assert len(clusters) == 6, 'seeded cluster set changed — re-verify the regex'
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
        """The cap is enforced by length, not by the regex itself.

        Both halves of that sentence are asserted. The regex ACCEPTS the
        over-length value (so the cap is not redundant with it), and the
        validator still rejects it (so the cap is actually applied). An
        earlier version of this test asserted only
        ``len(over) > TOPIC_SLUG_MAX_LEN`` — a tautology about a string the
        test had just constructed, which would have kept passing if the
        ``len(topic) > TOPIC_SLUG_MAX_LEN`` clause were deleted outright.
        """
        from fused_memory.memory_metadata import validate_memory_metadata

        over = 'a' * (TOPIC_SLUG_MAX_LEN + 1)
        assert TOPIC_SLUG_RE.match(over), 'the regex alone must NOT reject it'
        violations = validate_memory_metadata({'topic': over}, enforce_kind_registry=False)
        assert {v.code for v in violations} == {'invalid_topic_slug'}

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

    The writer/reader map is stated once, in ``normalize_supersedes``'s own
    docstring — in short: the harness writer emitted a scalar until task 3196
    migrated it to the canonical list shape, and scalar tolerance remains on
    read for the 81 un-migrated corpus records.
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
    #: Mem0 writers (scope_freshness.py:97/:251/:495,
    #: ReconciliationHarness._reconcile_status_correction,
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

        assert frozenset({
            'data', 'hash', 'created_at', 'updated_at',
            'user_id', 'agent_id', 'run_id', 'actor_id', 'role',
        }) == mem0_client.MEM0_MANAGED_METADATA_KEYS

    def test_mem0_managed_keys_is_one_object_not_a_copy(self):
        """INV-5 / PRD D12: `backends/mem0_client.py` is the decided home.

        Identity, not equality — two equal frozensets in two modules is
        exactly the duplication D12 exists to prevent.
        """
        from fused_memory import memory_metadata as mm
        from fused_memory.backends import mem0_client

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

        assert frozenset({
            'category',    # memory_service.py:2193 (add_memory), :2542 (add_system_record)
            'recon_pool',  # _apply_cycle_summary_metadata_tagging (memory_service.py:389)
            # NOTE: `run_id` is stamped by that same helper, but mem0 also owns
            # it, so it is classified under MEM0_MANAGED_METADATA_KEYS only —
            # see test_layers_are_pairwise_disjoint.
            'planned',     # NOT a write-seam stamp — a server-owned search-result
                           # annotation (memory_service.py:1932, :2921, read back at
                           # :2962). Included so a round-tripped search result
                           # re-written as metadata does not census-warn on the
                           # server's own field.
        }) == SERVER_STAMPED_KEYS

    def test_reserved_vocabulary_keys(self):
        from fused_memory.memory_metadata import RESERVED_VOCABULARY_KEYS

        assert frozenset({
            'topic', 'canonical', 'kind', 'parent_id', 'supersedes',
        }) == RESERVED_VOCABULARY_KEYS

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
        """No key may be classified twice -- all SIX pairs, not just the
        blessed-vs-rest three.

        `run_id` is the live trap: it is server-stamped (memory_service.py:389)
        AND mem0-managed AND measures 4,518 occurrences, so it is the one key
        that would otherwise land in three tiers at once.  It is classified
        under the mem0-managed layer only.
        """
        from fused_memory.backends.mem0_client import MEM0_MANAGED_METADATA_KEYS
        from fused_memory.memory_metadata import (
            BLESSED_METADATA_KEYS,
            RESERVED_VOCABULARY_KEYS,
            SERVER_STAMPED_KEYS,
        )

        layers = {
            'blessed': BLESSED_METADATA_KEYS,
            'reserved': RESERVED_VOCABULARY_KEYS,
            'mem0_managed': MEM0_MANAGED_METADATA_KEYS,
            'server_stamped': SERVER_STAMPED_KEYS,
        }
        names = sorted(layers)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = layers[a] & layers[b]
                assert not overlap, f'{a} & {b} overlap on {sorted(overlap)}'

        # The trap, pinned explicitly: run_id lives in exactly one layer.
        owners = [n for n, keys in layers.items() if 'run_id' in keys]
        assert owners == ['mem0_managed']

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


class TestCanonicalUniquenessViolationType:
    """The second contract-fixed rejection type (V1, task 3198 leaf ε).

    Pure type contract only -- no service. The type is raised from the
    async seam (it needs live store state), but its SHAPE is pinned here,
    next to its sibling, because that is what callers program against.
    """

    #: The live record leaf α measured; used as a realistic incumbent id.
    _INCUMBENT = '0929cff6-8efc-4a49-a672-e83f0a41bbfb'

    def test_is_importable_and_exported(self):
        import fused_memory.memory_metadata as mm

        assert hasattr(mm, 'CanonicalUniquenessViolation')
        assert 'CanonicalUniquenessViolation' in mm.__all__

    def _make(self):
        from fused_memory.memory_metadata import CanonicalUniquenessViolation

        return CanonicalUniquenessViolation(
            project_id='dark_factory',
            topic='some-topic',
            incumbent_id=self._INCUMBENT,
        )

    def test_carries_the_structured_fields(self):
        """INV-2: a rejection carries structured facts, not just prose."""
        exc = self._make()
        assert exc.project_id == 'dark_factory'
        assert exc.topic == 'some-topic'
        assert exc.incumbent_id == self._INCUMBENT

    def test_str_names_the_incumbent_verbatim(self):
        """V1's contract-fixed obligation: NAME the existing canonical's id.

        Counting is not enough -- an operator who trips this must be able
        to go look at the record that already holds the topic. This is
        also why the seam counts first and then scrolls: an int cannot
        carry an id.
        """
        text = str(self._make())
        assert self._INCUMBENT in text
        assert 'some-topic' in text

    def test_str_names_the_registry_location(self):
        """Asserted via the constant, not a literal, so it tracks a rename."""
        from fused_memory.memory_metadata import MemoryMetadataValidationError

        assert MemoryMetadataValidationError.REGISTRY_LOCATION in str(self._make())

    def test_is_a_sibling_of_the_shape_error_not_a_subclass(self):
        """Deliberate: the two rejection contracts must be distinguishable.

        Subclassing would let the five pre-existing
        `pytest.raises(MemoryMetadataValidationError)` sites in
        test_memory_service.py silently swallow a uniqueness violation --
        a test passing for the wrong reason. Keeping them siblings makes
        "which contract rejected this write" observable at the `except`.
        """
        from fused_memory.memory_metadata import (
            CanonicalUniquenessViolation,
            MemoryMetadataValidationError,
        )

        assert issubclass(CanonicalUniquenessViolation, Exception)
        assert not issubclass(CanonicalUniquenessViolation, MemoryMetadataValidationError)
        assert not issubclass(MemoryMetadataValidationError, CanonicalUniquenessViolation)

    def test_shape_error_handler_does_not_catch_it(self):
        """The sibling relationship, asserted behaviourally rather than by type."""
        from fused_memory.memory_metadata import (
            CanonicalUniquenessViolation,
            MemoryMetadataValidationError,
        )

        with pytest.raises(CanonicalUniquenessViolation):
            try:
                raise self._make()
            except MemoryMetadataValidationError:  # pragma: no cover - must not catch
                pytest.fail('the shape-error handler swallowed a uniqueness violation')


_UUID = '3f2504e0-4f89-11d3-9a0c-0305e82c3301'
_UUID2 = '7c9e6679-7425-40de-944b-e07fc1f90ae7'
_UUID3 = 'b1e0f2c4-5a6d-4e8f-9b0a-1c2d3e4f5a6b'


class TestParentHasChildrenError:
    """`ParentHasChildrenError` — the delete-time refusal (leaf δ, task 3197).

    The contract-fixed half of PRD V3's lifecycle rule ("no operation may
    silently orphan a child"). It lives in the registry module, next to
    `MemoryMetadataValidationError`, so both contract-fixed errors read
    identically to a caller and an out-of-process consumer can import the
    error without pulling in `MemoryService`.
    """

    def test_is_exported_and_subclasses_exception(self):
        import fused_memory.memory_metadata as mm
        from fused_memory.memory_metadata import ParentHasChildrenError

        assert issubclass(ParentHasChildrenError, Exception)
        assert 'ParentHasChildrenError' in mm.__all__

    def test_exposes_parent_id_and_child_ids(self):
        from fused_memory.memory_metadata import ParentHasChildrenError

        err = ParentHasChildrenError(parent_id=_UUID, child_ids=[_UUID2, _UUID3])
        assert err.parent_id == _UUID
        assert err.child_ids == [_UUID2, _UUID3]
        assert err.truncated is False

    def test_child_ids_is_a_defensive_copy(self):
        """Mirrors `MemoryMetadataValidationError.__init__`'s `list(violations)`.

        The caller builds the id list from a live scroll; if the error
        aliased it, a later mutation of that list would retroactively
        rewrite what the refusal claimed.
        """
        from fused_memory.memory_metadata import ParentHasChildrenError

        children = [_UUID2]
        err = ParentHasChildrenError(parent_id=_UUID, child_ids=children)
        children.append(_UUID3)
        assert err.child_ids == [_UUID2]

    def test_str_names_parent_every_child_the_count_cascade_and_registry(self):
        """The refusal must be actionable from the message alone.

        This is the user-observable signal the task exists to produce: an
        agent that hits the refusal at the MCP wire gets `str(err)` and
        nothing else, so it has to carry the ids, the way out (`cascade`)
        and where the vocabulary lives.
        """
        from fused_memory.memory_metadata import (
            MemoryMetadataValidationError,
            ParentHasChildrenError,
        )

        err = ParentHasChildrenError(parent_id=_UUID, child_ids=[_UUID2, _UUID3])
        text = str(err)
        assert _UUID in text
        assert _UUID2 in text
        assert _UUID3 in text
        assert '2' in text
        assert 'cascade' in text
        assert MemoryMetadataValidationError.REGISTRY_LOCATION in text

    def test_untruncated_message_does_not_imply_a_partial_listing(self):
        from fused_memory.memory_metadata import ParentHasChildrenError

        err = ParentHasChildrenError(parent_id=_UUID, child_ids=[_UUID2])
        assert 'at least' not in str(err)

    def test_truncated_message_says_the_listing_is_partial(self):
        """A bounded scroll must never read as an exhaustive listing.

        `truncated=True` is set when the id scroll returned fewer rows than
        the live child count (limit or race); saying "2 children" there
        would understate the damage a cascade is about to do.
        """
        from fused_memory.memory_metadata import ParentHasChildrenError

        err = ParentHasChildrenError(
            parent_id=_UUID, child_ids=[_UUID2, _UUID3], truncated=True
        )
        assert err.truncated is True
        text = str(err)
        assert 'at least' in text
        assert _UUID2 in text
        assert _UUID3 in text


class TestParentLivenessViolation:
    """The two write-time liveness codes and their public factory (leaf δ).

    The LOOKUP happens at the service seam — `validate_memory_metadata` is
    pure by construction and cannot reach a store (leaf β pinned that
    boundary structurally). But the violation CODES and the message WORDING
    stay here, behind this factory, so the registry remains the single
    normative home for the rule text and the seam cannot grow a second copy
    of it (INV-5).
    """

    def test_codes_are_distinct_constants_and_exported(self):
        import fused_memory.memory_metadata as mm
        from fused_memory.memory_metadata import (
            PARENT_ID_DEAD_CODE,
            PARENT_ID_UNAVAILABLE_CODE,
        )

        assert PARENT_ID_DEAD_CODE == 'dead_parent_id'
        assert PARENT_ID_UNAVAILABLE_CODE == 'parent_id_liveness_unavailable'
        assert PARENT_ID_DEAD_CODE != PARENT_ID_UNAVAILABLE_CODE
        for name in (
            'PARENT_ID_DEAD_CODE',
            'PARENT_ID_UNAVAILABLE_CODE',
            'parent_liveness_violation',
        ):
            assert name in mm.__all__

    def test_dead_parent_violation_shape(self):
        from fused_memory.memory_metadata import (
            PARENT_ID_DEAD_CODE,
            MetadataViolation,
            parent_liveness_violation,
        )

        v = parent_liveness_violation(_UUID, code=PARENT_ID_DEAD_CODE)
        assert isinstance(v, MetadataViolation)
        assert v.key == 'parent_id'
        assert v.code == 'dead_parent_id'
        assert v.fatal is True

    def test_dead_parent_message_names_the_id_the_rule_and_the_registry(self):
        from fused_memory.memory_metadata import (
            PARENT_ID_DEAD_CODE,
            MemoryMetadataValidationError,
            parent_liveness_violation,
        )

        text = parent_liveness_violation(_UUID, code=PARENT_ID_DEAD_CODE).message
        assert _UUID in text
        assert 'live' in text
        assert 'same-project' in text
        assert MemoryMetadataValidationError.REGISTRY_LOCATION in text

    def test_unavailable_violation_shape(self):
        from fused_memory.memory_metadata import (
            PARENT_ID_UNAVAILABLE_CODE,
            MetadataViolation,
            parent_liveness_violation,
        )

        v = parent_liveness_violation(_UUID, code=PARENT_ID_UNAVAILABLE_CODE)
        assert isinstance(v, MetadataViolation)
        assert v.key == 'parent_id'
        assert v.code == 'parent_id_liveness_unavailable'
        # Fatal is INV-3 read literally: an actor that cannot corroborate
        # must not act. Under `enforce` this rejects; in warn mode it
        # censuses and the write proceeds, via the same uniform arms.
        assert v.fatal is True

    def test_unavailable_message_says_unchecked_not_dead(self):
        """An operator reading the census line must never be told a parent is
        dead when the backend merely timed out.

        `Mem0Backend.get_point_by_id` deliberately PROPAGATES a read timeout
        rather than collapsing it into None, precisely to preserve this
        distinction; collapsing both codes into `dead_parent_id` would throw
        it away at the seam.
        """
        from fused_memory.memory_metadata import (
            PARENT_ID_UNAVAILABLE_CODE,
            MemoryMetadataValidationError,
            parent_liveness_violation,
        )

        text = parent_liveness_violation(
            _UUID, code=PARENT_ID_UNAVAILABLE_CODE
        ).message
        assert _UUID in text
        assert 'could not be checked' in text.lower()
        assert 'dead' not in text.lower()
        assert MemoryMetadataValidationError.REGISTRY_LOCATION in text

    def test_unrecognised_code_raises(self):
        """No silent minting of an unknown census code.

        Census codes are what operators grep; a typo'd code that quietly
        became a new census axis would be indistinguishable from a real one.
        """
        from fused_memory.memory_metadata import parent_liveness_violation

        with pytest.raises(ValueError):
            parent_liveness_violation(_UUID, code='parent_id_something_else')


def _codes(violations):
    return {v.code for v in violations}


def _by_key(violations, key):
    return [v for v in violations if v.key == key]


class TestValidateMemoryMetadata:
    """Pure shape checks (leaf β owns SHAPES ONLY).

    Signature: ``validate_memory_metadata(meta, *, enforce_kind_registry)``
    -> ``list[MetadataViolation]``.  It mutates *meta* in place ONLY to
    normalize `supersedes`, mirroring `parse_metadata`'s (value, warnings)
    contract: the caller owns the warn-vs-reject decision.
    """

    def _validate(self, meta, *, enforce_kind_registry=False):
        from fused_memory.memory_metadata import validate_memory_metadata

        return validate_memory_metadata(meta, enforce_kind_registry=enforce_kind_registry)

    # -- valid path ------------------------------------------------------

    def test_fully_valid_metadata_yields_no_violations(self):
        meta = {
            'topic': 'a-good-slug',
            'canonical': True,
            'kind': 'cycle_summary',
            'parent_id': _UUID,
            'supersedes': [_UUID2],
            'task_id': '3195',        # blessed
            'x_experimental': 'ok',   # passthrough
        }
        assert self._validate(meta) == []
        assert meta['supersedes'] == [_UUID2]

    def test_absent_reserved_keys_are_never_a_violation(self):
        """None of V1's five keys is required."""
        assert self._validate({}) == []

    # -- topic -----------------------------------------------------------

    @pytest.mark.parametrize('bad', ['bad_slug', 'Bad-Slug', '-x', 'a--b', ''])
    def test_bad_topic_slug(self, bad):
        violations = _by_key(self._validate({'topic': bad}), 'topic')
        assert [v.code for v in violations] == ['invalid_topic_slug']
        assert 'TOPIC_SLUG_RE' in violations[0].message

    def test_over_length_topic_slug(self):
        from fused_memory.memory_metadata import TOPIC_SLUG_MAX_LEN

        meta = {'topic': 'a' * (TOPIC_SLUG_MAX_LEN + 1)}
        assert _codes(self._validate(meta)) == {'invalid_topic_slug'}

    def test_non_str_topic(self):
        assert _codes(self._validate({'topic': 123})) == {'invalid_topic_type'}

    # -- canonical -------------------------------------------------------

    def test_non_bool_canonical_rejected(self):
        assert _codes(self._validate({'canonical': 'yes'})) == {'invalid_canonical_type'}

    def test_canonical_one_is_rejected_not_treated_as_true(self):
        """The check must be isinstance(x, bool), NOT truthiness.

        `1 == True` in Python, so a truthiness check would silently accept
        an int here — and the census counts exactly 1 live non-bool value,
        which is the record this rule exists to surface.
        """
        assert _codes(self._validate({'canonical': 1})) == {'invalid_canonical_type'}
        assert _codes(self._validate({'canonical': 0})) == {'invalid_canonical_type'}

    @pytest.mark.parametrize('good', [True, False])
    def test_bool_canonical_accepted(self, good):
        """A bool ``canonical`` passes the TYPE check (§2b).

        ``True`` is paired with a topic here because task 3198 (leaf ε)
        additionally makes a bare ``{'canonical': True}`` a shape
        violation -- see ``TestCanonicalRequiresTopic`` below. Keeping the
        topic in this case preserves what this test is actually for (the
        bool type check) instead of letting it double as an assertion that
        a topic-less canonical is legal, which it no longer is.
        """
        meta = {'canonical': good}
        if good:
            meta['topic'] = 'a-good-slug'
        assert self._validate(meta) == []

    # -- canonical requires a topic (task 3198, leaf ε) -------------------

    def test_canonical_true_without_topic_is_fatal(self):
        """``canonical: true`` is meaningless outside a topic.

        ``canonical`` asserts "this is THE entry for its topic", so
        without a topic there is no set for it to be canonical OF, and the
        per-(project, topic) uniqueness rule has no key to check. Caught
        in the PURE validator because it needs no store access at all.
        """
        violations = _by_key(self._validate({'canonical': True}), 'canonical')
        assert [v.code for v in violations] == ['canonical_without_topic']
        assert violations[0].fatal is True
        assert 'topic' in violations[0].message
        # Asserted via the constant, not a hard-coded literal, so the message
        # tracks a rename of the registry's advertised location.
        assert MemoryMetadataValidationError.REGISTRY_LOCATION in violations[0].message

    def test_canonical_false_without_topic_is_legal(self):
        """Only an ASSERTED canonical is meaningless without a topic.

        Writers that stamp the key uniformly (``canonical: False`` on
        every record) must keep working -- rejecting an explicit False
        would break them for no benefit.
        """
        assert self._validate({'canonical': False}) == []

    def test_canonical_true_with_a_good_topic_is_legal(self):
        assert self._validate({'canonical': True, 'topic': 'a-good-slug'}) == []

    def test_malformed_topic_is_not_double_counted_as_missing(self):
        """A present-but-invalid topic is ONE defect, reported once.

        The ``topic`` key IS present, so §2a's slug violation already
        describes the problem. Also emitting ``canonical_without_topic``
        would report one defect twice and imply the writer forgot a key
        they did supply.
        """
        codes = _codes(self._validate({'canonical': True, 'topic': 'bad_slug'}))
        assert 'invalid_topic_slug' in codes
        assert 'canonical_without_topic' not in codes

    def test_topic_without_canonical_is_legal(self):
        """A topic needs no canonical marker; the implication is one-way."""
        assert self._validate({'topic': 'a-good-slug'}) == []

    def test_validator_does_not_short_circuit_on_canonical(self):
        """Every applicable violation comes back in ONE pass (contract).

        An int ``canonical`` with no topic is two independent defects. A
        validator that returned at the first would hide the second, and
        the caller would fix one, re-submit, and be rejected again.
        """
        codes = _codes(self._validate({'canonical': 1}))
        assert codes == {'invalid_canonical_type'}, (
            'a non-bool canonical is a TYPE defect; canonical_without_topic is '
            'reserved for a genuine `is True`, per §2b\'s isinstance-not-truthiness rule'
        )
        codes = _codes(
            self._validate(
                {'canonical': True, 'kind': 'not_in_registry_xyz'},
                enforce_kind_registry=True,
            )
        )
        assert codes == {'canonical_without_topic', 'unknown_kind'}, (
            'unrelated violations must still be reported alongside, in one pass'
        )

    # -- kind ------------------------------------------------------------

    def test_known_kind_accepted(self):
        assert self._validate({'kind': 'cycle_summary'}) == []

    def test_unknown_kind_is_fatal_when_registry_enforced(self):
        violations = self._validate(
            {'kind': 'not_in_registry_xyz'}, enforce_kind_registry=True
        )
        assert len(violations) == 1
        assert violations[0].code == 'unknown_kind'
        assert violations[0].fatal is True
        assert 'KIND_REGISTRY' in violations[0].message

    def test_unknown_kind_still_censuses_when_registry_not_enforced(self):
        """Same violation CODE, but non-fatal: it censuses, it does not reject.

        This is the census-refuted-premise carve-out (PRD D3 amendment):
        242 of 329 live kinds are singletons, so day-one rejection would
        break the live fleet.
        """
        violations = self._validate(
            {'kind': 'not_in_registry_xyz'}, enforce_kind_registry=False
        )
        assert len(violations) == 1
        assert violations[0].code == 'unknown_kind'
        assert violations[0].fatal is False

    def test_non_str_kind(self):
        assert _codes(self._validate({'kind': 42})) == {'invalid_kind_type'}

    # -- parent_id (SHAPE only; liveness is leaf delta's) -----------------

    @pytest.mark.parametrize('bad', ['deadbeef', 'deadbe', 'not-a-uuid', ''])
    def test_bad_parent_id_shape(self, bad):
        assert _codes(self._validate({'parent_id': bad})) == {'invalid_parent_id_shape'}

    def test_non_str_parent_id(self):
        assert _codes(self._validate({'parent_id': 123})) == {'invalid_parent_id_shape'}

    def test_well_formed_parent_id_accepted(self):
        assert self._validate({'parent_id': _UUID}) == []

    def test_validator_is_a_pure_sync_dict_function(self):
        """`parent_id` LIVENESS (leaf delta) is structurally out of reach.

        The validator is SYNCHRONOUS, so it cannot await the store lookup
        that `parent_id` liveness requires. That makes the beta/delta scope
        boundary structural rather than a comment a later leaf could
        overlook and duplicate.

        Deliberately the only introspection here. The parameter NAMES are
        not pinned: they are cosmetic, every other test in this file and
        the seam helper in `memory_service.py` call through the real
        signature, so a signature change already fails on its own merits.
        """
        import inspect

        from fused_memory.memory_metadata import validate_memory_metadata

        assert not inspect.iscoroutinefunction(validate_memory_metadata)

    # -- supersedes ------------------------------------------------------

    def test_scalar_supersedes_is_normalized_not_rejected(self):
        """The beta-before-gamma hazard fix.

        The recon harness wrote a scalar (81 live records) until task 3196
        (gamma) migrated it to the canonical list shape. Beta landed BEFORE
        gamma, so rejecting the scalar form here would have broken the recon
        harness's own writes for the whole window between.

        Post-3196 this guards the LEGACY CORPUS rather than a live in-repo
        writer: gamma shipped no corpus rewrite, so the 81 pre-migration
        records still carry scalars (PRD D2 defers retro normalization to leaf
        theta's stamping sweep), and out-of-repo writers are unbound by it.
        """
        meta = {'supersedes': _UUID}
        assert self._validate(meta) == []
        assert meta['supersedes'] == [_UUID]

    def test_list_supersedes_accepted(self):
        meta = {'supersedes': [_UUID, _UUID2]}
        assert self._validate(meta) == []
        assert meta['supersedes'] == [_UUID, _UUID2]

    def test_short_hex_supersedes_member_rejected(self):
        """3 live records carry this shape."""
        violations = self._validate({'supersedes': [_UUID, 'deadbe']})
        assert _codes(violations) == {'invalid_supersedes_member'}
        assert 'deadbe' in violations[0].message

    def test_non_str_supersedes_member_rejected(self):
        """8 live records carry an 'other' member shape."""
        violations = self._validate({'supersedes': [_UUID, 42]})
        assert _codes(violations) == {'invalid_supersedes_member'}
        assert '42' in violations[0].message

    def test_empty_and_absent_supersedes_are_fine(self):
        meta = {'supersedes': []}
        assert self._validate(meta) == []
        assert self._validate({}) == []

    # -- unknown keys ----------------------------------------------------

    def test_unknown_key_censuses_non_fatally(self):
        violations = self._validate({'weird_key': 1})
        assert len(violations) == 1
        assert violations[0].code == 'unknown_key'
        assert violations[0].key == 'weird_key'
        assert violations[0].fatal is False

    def test_x_prefixed_key_is_silent(self):
        assert self._validate({'x_weird': 1}) == []

    @pytest.mark.parametrize(
        'key',
        ['data', 'user_id', 'category', 'recon_pool', 'planned', 'task_id', 'source'],
    )
    def test_known_layer_keys_are_silent(self, key):
        assert self._validate({key: 'v'}) == []

    # -- no short-circuit ------------------------------------------------

    def test_all_violations_are_returned_not_just_the_first(self):
        """A validator that stopped at the first failure would make an
        operator fix one violation per write attempt."""
        meta = {
            'topic': 'bad_slug',
            'canonical': 'yes',
            'kind': 'not_in_registry_xyz',
            'parent_id': 'deadbeef',
            'supersedes': [42],
            'weird_key': 1,
        }
        violations = self._validate(meta, enforce_kind_registry=True)
        assert _codes(violations) == {
            'invalid_topic_slug',
            'invalid_canonical_type',
            'unknown_kind',
            'invalid_parent_id_shape',
            'invalid_supersedes_member',
            'unknown_key',
        }

    def test_every_message_names_the_registry_location(self):
        """V1: the hint must name where the vocabulary lives."""
        meta = {'topic': 'bad_slug', 'canonical': 'yes', 'parent_id': 'x'}
        for violation in self._validate(meta):
            assert 'fused_memory.memory_metadata' in violation.message

    # -- fatal flag ------------------------------------------------------

    @pytest.mark.parametrize(
        ('meta', 'code'),
        [
            ({'topic': 'bad_slug'}, 'invalid_topic_slug'),
            ({'topic': 123}, 'invalid_topic_type'),
            ({'canonical': 'yes'}, 'invalid_canonical_type'),
            ({'kind': 42}, 'invalid_kind_type'),
            ({'parent_id': 'deadbeef'}, 'invalid_parent_id_shape'),
            ({'supersedes': [42]}, 'invalid_supersedes_member'),
        ],
    )
    def test_shape_violations_are_fatal(self, meta, code):
        """Malformed RESERVED keys are fatal; `enforce` is what decides
        whether a fatal violation actually rejects.

        This is the pin step-15's seam behaviour rests on: with
        `enforce=True` a bad `topic` slug must RAISE, which is only true if
        the shape violation carries fatal=True. Asserting it here — on the
        pure function — means the seam test cannot be the only thing
        standing between a silent downgrade and a lost contract.
        """
        violations = [v for v in self._validate(meta) if v.code == code]
        assert violations, f'expected a {code} violation'
        assert all(v.fatal is True for v in violations)

    @pytest.mark.parametrize(
        ('meta', 'kwargs', 'code'),
        [
            ({'weird_key': 1}, {}, 'unknown_key'),
            ({'kind': 'not_in_registry_xyz'}, {}, 'unknown_kind'),
        ],
    )
    def test_census_only_violations_are_not_fatal(self, meta, kwargs, code):
        """The two census axes never reject under the shipped defaults.

        `unknown_key` is census-only by design (1,627 distinct live keys);
        `unknown_kind` is non-fatal unless `enforce_kind_registry` is on,
        because leaf alpha measured D3's closed-registry premise false.
        """
        violations = [v for v in self._validate(meta, **kwargs) if v.code == code]
        assert violations, f'expected a {code} violation'
        assert all(v.fatal is False for v in violations)
