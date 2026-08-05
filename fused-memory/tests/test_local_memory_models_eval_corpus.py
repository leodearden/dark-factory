"""Tests for scripts/local_memory_models_eval/build_corpus.py — the LME replay corpus.

PRD ``plans/local-memory-models-eval-prd.md`` task δ: a committed, re-derivable,
stratified sample of real ``dark_factory`` episodes that is **never conditioned
on the incumbent pipeline's outcome**. Consumers: ε (replay engine input),
ζ (control replays), θ (full arm replays).

Loaded via importlib so the script (not on PYTHONPATH) can be tested without
sys.path pollution — the same ``_load_module()`` helper as
test_memory_eval_transcript_corpus.py / test_memory_eval_retrieval_probe.py.

Every test here runs against in-memory record lists or a hand-written store
double, EXCEPT the single ``@pytest.mark.integration`` smoke, which issues one
``GRAPH.RO_QUERY`` against the live graph. Nothing here ever writes: the store
double's tripwires enforce it offline, and ``GRAPH.RO_QUERY`` enforces it
server-side on the live path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parent.parent / 'scripts' / 'local_memory_models_eval' / 'build_corpus.py'
)


def _load_module() -> types.ModuleType:
    """Load build_corpus.py from its file path.

    The module is registered in sys.modules under its name BEFORE
    ``exec_module`` so that ``@dataclass`` and other reflection-based
    decorators work correctly (they call ``sys.modules.get(cls.__module__)``),
    and build_corpus.py defines frozen dataclasses. See the note at
    test_memory_eval_retrieval_probe.py's copy of this helper.
    """
    mod_name = 'lme_build_corpus'
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # required for @dataclass __module__ lookup
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()


# ===========================================================================
# Parsing core — month bucket, payload kind, content hash
# ===========================================================================


class TestMonthBucket:
    """``month_bucket`` derives the time stratum from ``created_at``."""

    def test_iso_offset_timestamp(self):
        """The shape FalkorDB actually returns for dark_factory episodes."""
        assert _mod.month_bucket('2026-04-06T00:19:33.967261+00:00') == '2026-04'

    def test_naive_and_z_suffixed_normalize_identically(self):
        """A 'Z' suffix, a '+00:00' offset and a naive stamp are the same instant.

        The store has been written by several code paths over its life; a
        bucket that depended on which spelling was persisted would silently
        split one month into two strata.
        """
        buckets = {
            _mod.month_bucket('2026-05-16T12:00:00Z'),
            _mod.month_bucket('2026-05-16T12:00:00+00:00'),
            _mod.month_bucket('2026-05-16T12:00:00'),
        }
        assert buckets == {'2026-05'}

    def test_sub_second_and_space_separator(self):
        assert _mod.month_bucket('2026-08-05 21:17:57.123456+00:00') == '2026-08'

    @pytest.mark.parametrize(
        'bad',
        ['', 'not-a-timestamp', '2026-13-01T00:00:00Z', None, 17],
    )
    def test_malformed_raises_rather_than_bucketing_to_unknown(self, bad):
        """Loud over silent degradation: never a catch-all 'unknown' bucket.

        An 'unknown' bucket would quietly become a 17th stratum and the
        stratification report would look complete while describing a corpus
        whose time axis had partially collapsed.
        """
        with pytest.raises(_mod.CorpusBuildError) as exc:
            _mod.month_bucket(bad)
        assert 'created_at' in str(exc.value)


class TestPayloadKind:
    """``payload_kind`` derives the payload stratum from ``source_description``."""

    def test_add_memory_shape(self):
        """The shape all 2770 dark_factory episodes carry today."""
        assert _mod.payload_kind('add_memory:decisions_and_rationale') == (
            'decisions_and_rationale'
        )

    @pytest.mark.parametrize(
        'kind',
        [
            'decisions_and_rationale',
            'temporal_facts',
            'entities_and_relations',
            'procedural_knowledge',
        ],
    )
    def test_every_observed_kind(self, kind):
        assert _mod.payload_kind(f'add_memory:{kind}') == kind

    def test_replay_from_mem0_shape(self):
        """The second in-tree episode writer (memory_service replay path)."""
        assert _mod.payload_kind('replay_from_mem0:temporal_facts') == 'temporal_facts'

    def test_add_episode_caller_supplied_description(self):
        """Real add_episode ingestion carries an arbitrary caller string.

        It is not one of the six categories, so it buckets under a single
        explicit ``add_episode`` kind rather than fanning the payload axis out
        into one stratum per caller.
        """
        assert _mod.payload_kind('interactive session recap') == 'add_episode'

    def test_temporal_context_prefix_is_stripped(self):
        """``graphiti_client`` prepends ``[temporal:<ctx>] `` when set.

        Verified absent from dark_factory today — all rows are bare
        ``add_memory:*`` — but the parser must not mis-bucket it if it appears.
        """
        assert _mod.payload_kind('[temporal:2026-05-16] add_memory:temporal_facts') == (
            'temporal_facts'
        )

    @pytest.mark.parametrize('bad', ['', None, 17, '   '])
    def test_unusable_value_raises_naming_the_offender(self, bad):
        with pytest.raises(_mod.CorpusBuildError) as exc:
            _mod.payload_kind(bad)
        assert 'source_description' in str(exc.value)
        assert repr(bad) in str(exc.value)


class TestContentHash:
    """``content_hash`` is the drift detector for replay inputs."""

    def test_is_full_length_sha256_of_utf8_bytes(self):
        text = 'a decision was made because of X'
        assert _mod.content_hash(text) == hashlib.sha256(text.encode('utf-8')).hexdigest()
        assert len(_mod.content_hash(text)) == 64

    def test_is_byte_exact_not_whitespace_normalized(self):
        """Deliberately NOT ``content_key()``'s whitespace-collapsing digest.

        For epsilon/zeta/theta the bytes fed to the extraction pipeline are the
        experiment's independent variable, so a whitespace change IS a change
        and must fail verification.
        """
        assert _mod.content_hash('a b') != _mod.content_hash('a  b')
        assert _mod.content_hash('a b') != _mod.content_hash('a\nb')

    def test_non_ascii_is_hashed_as_utf8(self):
        assert _mod.content_hash('δ') == hashlib.sha256('δ'.encode()).hexdigest()


class TestEpisodeRecord:
    """The record type — and the outcome field it deliberately does not have."""

    def _record(self, **over):
        fields = {
            'uuid': 'e622a9bf-f1c8-431b-ad36-92762d69436d',
            'name': 'ep',
            'group_id': 'dark_factory',
            'source_description': 'add_memory:temporal_facts',
            'created_at': '2026-05-16T12:00:00+00:00',
            'content': 'some episode text',
        }
        fields.update(over)
        return _mod.EpisodeRecord(**fields)

    def test_is_frozen(self):
        record = self._record()
        with pytest.raises(dataclasses.FrozenInstanceError):
            record.uuid = 'other'  # type: ignore[misc]

    def test_exposes_no_outcome_field(self):
        """The no-outcome-filter hazard, enforced at the type level.

        ``entity_edges`` is the per-episode record of what the INCUMBENT
        extraction pipeline produced. If the record cannot carry it, no
        sampling rule downstream can condition on it — the guarantee holds by
        construction rather than by promise.
        """
        names = {f.name for f in dataclasses.fields(_mod.EpisodeRecord)}
        assert names == {
            'uuid',
            'name',
            'group_id',
            'source_description',
            'created_at',
            'content',
        }
        assert not hasattr(self._record(), 'entity_edges')
        forbidden = ('entity_edges', 'edges', 'extracted', 'outcome', 'score', 'success')
        assert not [n for n in names if any(f in n for f in forbidden)]

    def test_stratum_key_is_month_by_payload_kind(self):
        assert _mod.stratum_key(self._record()) == ('2026-05', 'temporal_facts')

    def test_stratum_key_propagates_parse_failure(self):
        with pytest.raises(_mod.CorpusBuildError):
            _mod.stratum_key(self._record(created_at='garbage'))


# ===========================================================================
# Allocation — min-1 floor + largest-remainder proportional
# ===========================================================================

# The REAL cross-tab, measured read-only from the live dark_factory graph on
# 2026-08-05 via `docker exec docker-falkordb-1 redis-cli GRAPH.RO_QUERY`.
# 16 non-empty cells of 20; 2770 episodes total. Frozen here as a test
# constant deliberately: the allocator's hard cases are this census's shape
# (a singleton cell, a 571-row cell, 4 empty cells), and a synthetic census
# would not exercise them.
CENSUS: dict[tuple[str, str], int] = {
    ('2026-04', 'decisions_and_rationale'): 464,
    ('2026-04', 'entities_and_relations'): 56,
    ('2026-04', 'procedural_knowledge'): 1,
    ('2026-04', 'temporal_facts'): 237,
    ('2026-05', 'decisions_and_rationale'): 327,
    ('2026-05', 'entities_and_relations'): 38,
    ('2026-05', 'temporal_facts'): 391,
    ('2026-06', 'decisions_and_rationale'): 171,
    ('2026-06', 'entities_and_relations'): 10,
    ('2026-06', 'temporal_facts'): 85,
    ('2026-07', 'decisions_and_rationale'): 571,
    ('2026-07', 'entities_and_relations'): 77,
    ('2026-07', 'temporal_facts'): 163,
    ('2026-08', 'decisions_and_rationale'): 153,
    ('2026-08', 'entities_and_relations'): 9,
    ('2026-08', 'temporal_facts'): 17,
}

CENSUS_POPULATION = 2770

# The singleton cell. Its proportional share at N=200 is 1/2770 * 200 = 0.07,
# so pure proportional allocation rounds it to zero and deletes an entire
# payload kind from every downstream arm comparison.
SINGLETON_CELL = ('2026-04', 'procedural_knowledge')


class TestAllocate:
    """``allocate`` splits N across the non-empty cells, exactly and deterministically."""

    def test_census_constant_matches_its_stated_population(self):
        """Guards the test's own premise — a typo'd census would test nothing."""
        assert sum(CENSUS.values()) == CENSUS_POPULATION
        assert len(CENSUS) == 16

    @pytest.mark.parametrize('n', [150, 200, 300])
    def test_sums_to_exactly_n(self, n):
        """Across the whole PRD band (150-300), not just the default."""
        assert sum(_mod.allocate(CENSUS, n).values()) == n

    @pytest.mark.parametrize('n', [150, 200, 300])
    def test_every_non_empty_cell_gets_at_least_one(self, n):
        alloc = _mod.allocate(CENSUS, n)
        assert set(alloc) == set(CENSUS)
        assert all(v >= 1 for v in alloc.values()), {
            cell: v for cell, v in alloc.items() if v < 1
        }

    @pytest.mark.parametrize('n', [150, 200, 300])
    def test_singleton_cell_survives_the_floor(self, n):
        """The whole reason a min-1 floor exists rather than pure proportional.

        procedural_knowledge has exactly ONE episode in the entire store. Its
        proportional share is 0.07 seats at N=200; without the floor the payload
        kind vanishes from the corpus and no downstream arm comparison can ever
        see it — an invisible coverage loss that would make the eval lie.
        """
        assert _mod.allocate(CENSUS, n)[SINGLETON_CELL] == 1

    @pytest.mark.parametrize('n', [16, 20, 150, 200, 300, 1000, 2770])
    def test_never_allocates_more_than_a_cell_holds(self, n):
        """You cannot draw 5 episodes from a cell of 1."""
        alloc = _mod.allocate(CENSUS, n)
        over = {c: (alloc[c], CENSUS[c]) for c in alloc if alloc[c] > CENSUS[c]}
        assert not over, over
        assert sum(alloc.values()) == n

    def test_allocation_is_broadly_proportional(self):
        """The floor must not swamp the proportionality it is protecting.

        The largest cell (2026-07 d_and_r, 20.6% of the store) should still
        receive roughly its share, not be flattened toward the singleton.
        """
        alloc = _mod.allocate(CENSUS, 200)
        biggest = ('2026-07', 'decisions_and_rationale')
        share = CENSUS[biggest] / CENSUS_POPULATION
        assert abs(alloc[biggest] - share * 200) <= 3

    def test_is_deterministic_no_rng(self):
        """Same input, same output — repeatedly, and independent of dict order."""
        first = _mod.allocate(CENSUS, 200)
        for _ in range(5):
            assert _mod.allocate(CENSUS, 200) == first
        reordered = dict(sorted(CENSUS.items(), reverse=True))
        assert _mod.allocate(reordered, 200) == first

    def test_n_greater_than_population_raises(self):
        """Never silently return fewer than asked for."""
        with pytest.raises(_mod.CorpusBuildError) as exc:
            _mod.allocate(CENSUS, CENSUS_POPULATION + 1)
        assert str(CENSUS_POPULATION) in str(exc.value)

    def test_n_equal_to_population_takes_everything(self):
        alloc = _mod.allocate(CENSUS, CENSUS_POPULATION)
        assert alloc == CENSUS

    def test_n_below_cell_count_raises(self):
        """The floor is not satisfiable, so the request is rejected, not shaved.

        Silently dropping cells to fit would be exactly the invisible coverage
        loss the floor exists to prevent.
        """
        with pytest.raises(_mod.CorpusBuildError) as exc:
            _mod.allocate(CENSUS, 15)
        assert '16' in str(exc.value)

    @pytest.mark.parametrize('n', [0, -1])
    def test_non_positive_n_raises(self, n):
        with pytest.raises(_mod.CorpusBuildError):
            _mod.allocate(CENSUS, n)

    def test_empty_cells_are_rejected_not_silently_seated(self):
        """A zero-count cell cannot receive the min-1 floor.

        It must be excluded before allocation, not floored into existence — a
        seat in an empty cell is a seat that can never be filled, and the total
        would then not sum to N.
        """
        with pytest.raises(_mod.CorpusBuildError):
            _mod.allocate({**CENSUS, ('2026-09', 'temporal_facts'): 0}, 200)
