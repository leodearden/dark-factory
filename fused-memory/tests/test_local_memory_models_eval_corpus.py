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

import asyncio
import dataclasses
import hashlib
import importlib.util
import json
import random
import re
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from _fm_helpers import falkor_skipif

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


# ===========================================================================
# Selection — prefix of a seeded permutation, per cell
# ===========================================================================

# The real eligibility anchor, measured read-only on 2026-08-05: the ONE
# episode of 2770 with size(entity_edges) == 0, i.e. the one the incumbent
# extraction pipeline produced nothing from. If corpus membership were ever
# conditioned on the incumbent's outcome, this is the episode that would
# silently disappear. It must stay eligible.
ANCHOR_UUID = 'e622a9bf-f1c8-431b-ad36-92762d69436d'


def _record(uuid: str, month: str = '2026-05', kind: str = 'temporal_facts', content=None):
    """One EpisodeRecord with a stratum key of exactly (month, kind)."""
    return _mod.EpisodeRecord(
        uuid=uuid,
        name=f'ep-{uuid[:8]}',
        group_id='dark_factory',
        source_description=f'add_memory:{kind}',
        created_at=f'{month}-16T12:00:00+00:00',
        content=content if content is not None else f'content of {uuid}',
    )


def _population(per_cell: dict[tuple[str, str], int]):
    """A synthetic population with the requested per-cell counts.

    UUIDs are zero-padded and globally unique so canonical uuid sorting is
    well-defined and cells cannot alias each other.
    """
    out = []
    counter = 0
    for (month, kind), count in sorted(per_cell.items()):
        for _ in range(count):
            out.append(_record(f'{counter:08d}-0000-0000-0000-000000000000', month, kind))
            counter += 1
    return out


SYNTHETIC_CELLS: dict[tuple[str, str], int] = {
    ('2026-04', 'decisions_and_rationale'): 40,
    ('2026-04', 'procedural_knowledge'): 1,
    ('2026-05', 'decisions_and_rationale'): 30,
    ('2026-05', 'temporal_facts'): 25,
    ('2026-06', 'entities_and_relations'): 12,
    ('2026-07', 'temporal_facts'): 60,
}


def _uuids(records) -> list[str]:
    return [r.uuid for r in records]


class TestSelect:
    """``select`` draws N records: deterministic, order-blind, cell-prefix."""

    def test_returns_exactly_n(self):
        pop = _population(SYNTHETIC_CELLS)
        assert len(_mod.select(pop, 50, seed='s').selected) == 50

    def test_db_row_order_does_not_matter(self):
        """The guard against FalkorDB returning rows in an arbitrary order.

        Row order is not guaranteed stable between runs. Without a canonical
        re-sort before seeding, a re-run could silently produce a DIFFERENT
        corpus while still claiming the same seed — the worst kind of failure,
        because the artifact would look reproducible and not be.
        """
        pop = _population(SYNTHETIC_CELLS)
        baseline = _uuids(_mod.select(pop, 50, seed='s').selected)
        for shuffle_seed in range(5):
            shuffled = list(pop)
            random.Random(shuffle_seed).shuffle(shuffled)
            assert _uuids(_mod.select(shuffled, 50, seed='s').selected) == baseline

    def test_same_seed_is_byte_identical_across_calls(self):
        pop = _population(SYNTHETIC_CELLS)
        first = _uuids(_mod.select(pop, 50, seed='s').selected)
        for _ in range(5):
            assert _uuids(_mod.select(pop, 50, seed='s').selected) == first

    def test_different_seed_changes_the_selection_not_just_its_order(self):
        pop = _population(SYNTHETIC_CELLS)
        a = set(_uuids(_mod.select(pop, 50, seed='seed-a').selected))
        b = set(_uuids(_mod.select(pop, 50, seed='seed-b').selected))
        assert a != b

    def test_uses_no_module_global_random_state(self):
        """RNG is derived from the seed, never from the ambient random module.

        A module-global draw would make the result depend on whatever else ran
        first in the process — reproducible only by accident.
        """
        pop = _population(SYNTHETIC_CELLS)
        random.seed(1)
        first = _uuids(_mod.select(pop, 50, seed='s').selected)
        random.seed(999)
        [random.random() for _ in range(50)]
        assert _uuids(_mod.select(pop, 50, seed='s').selected) == first

    def test_per_cell_counts_match_allocate_exactly(self):
        pop = _population(SYNTHETIC_CELLS)
        result = _mod.select(pop, 50, seed='s')
        expected = _mod.allocate(SYNTHETIC_CELLS, 50)
        got: dict[tuple[str, str], int] = {}
        for record in result.selected:
            got[_mod.stratum_key(record)] = got.get(_mod.stratum_key(record), 0) + 1
        assert got == expected

    def test_selection_is_a_prefix_of_each_cells_permutation(self):
        """The property that makes zeta's N re-tune (PRD Open Q4) cheap.

        Asserted per cell, deliberately NOT as global nestedness: largest
        remainder allocation can move a single seat BETWEEN cells as N grows,
        so global nestedness is not guaranteed and asserting it would be
        asserting something false.
        """
        pop = _population(SYNTHETIC_CELLS)
        small = _mod.select(pop, 50, seed='s')
        large = _mod.select(pop, 120, seed='s')
        for cell in SYNTHETIC_CELLS:
            perm = _mod.cell_permutation(pop, cell, seed='s')
            for result in (small, large):
                taken = [r.uuid for r in result.selected if _mod.stratum_key(r) == cell]
                assert taken == perm[: len(taken)], (cell, taken)

    def test_a_cell_whose_allocation_grew_keeps_its_earlier_picks(self):
        """The consequence epsilon cares about: replays already done stay valid."""
        pop = _population(SYNTHETIC_CELLS)
        small = _mod.select(pop, 50, seed='s')
        large = _mod.select(pop, 120, seed='s')
        for cell in SYNTHETIC_CELLS:
            before = [r.uuid for r in small.selected if _mod.stratum_key(r) == cell]
            after = [r.uuid for r in large.selected if _mod.stratum_key(r) == cell]
            if len(after) >= len(before):
                assert after[: len(before)] == before, cell

    def test_output_order_is_fully_determined(self):
        """Ordered by (cell key, position in permutation) — no incidental order."""
        pop = _population(SYNTHETIC_CELLS)
        selected = _mod.select(pop, 50, seed='s').selected
        cells_in_order = [_mod.stratum_key(r) for r in selected]
        assert cells_in_order == sorted(cells_in_order)

    def test_every_record_is_accounted_for_exactly_once(self):
        """Exhaustive disposition — no episode disappears unexplained.

        The SampleResult convention from scripts/legibility/sampling.py: every
        record handed in leaves by exactly ONE door. Without it, a corpus that
        quietly lost records would report a clean total.
        """
        pop = _population(SYNTHETIC_CELLS)
        result = _mod.select(pop, 50, seed='s')
        selected = _uuids(result.selected)
        not_selected = sorted(result.not_selected)
        assert len(set(selected)) == len(selected)
        assert set(selected).isdisjoint(not_selected)
        assert set(selected) | set(not_selected) == {r.uuid for r in pop}
        assert len(selected) + len(not_selected) == len(pop)

    def test_every_non_selected_record_carries_a_reason(self):
        pop = _population(SYNTHETIC_CELLS)
        result = _mod.select(pop, 50, seed='s')
        assert result.not_selected
        for uuid, reason in result.not_selected.items():
            assert isinstance(reason, str) and reason.strip(), uuid
        assert set(result.not_selected.values()) <= set(_mod.DISPOSITIONS)

    def test_duplicate_uuids_are_rejected(self):
        """Two records claiming one uuid would break disposition accounting."""
        pop = _population(SYNTHETIC_CELLS)
        with pytest.raises(_mod.CorpusBuildError) as exc:
            _mod.select([*pop, pop[0]], 50, seed='s')
        assert pop[0].uuid in str(exc.value)

    def test_empty_population_raises(self):
        with pytest.raises(_mod.CorpusBuildError):
            _mod.select([], 10, seed='s')


class TestSelectionIsNotConditionedOnOutcome:
    """The binding hazard, at the sampler: an outcome-failed episode stays eligible."""

    def _population_with_anchor(self):
        pop = _population(SYNTHETIC_CELLS)
        # The anchor lands in its own singleton cell so its allocation is 1
        # under the min-1 floor and it is therefore ALWAYS selected — the
        # strongest form of "never excluded".
        return [*pop, _record(ANCHOR_UUID, '2026-08', 'temporal_facts')]

    def test_anchor_is_in_the_population(self):
        pop = self._population_with_anchor()
        assert ANCHOR_UUID in _uuids(pop)

    @pytest.mark.parametrize('seed', ['s', 'other', 'zeta-2026', ''])
    def test_anchor_is_selected_under_every_seed(self, seed):
        """The one episode the incumbent extracted NOTHING from.

        It is the concrete test of the no-outcome-filter hazard: if membership
        were conditioned on incumbent success this episode would be first out.
        """
        result = _mod.select(self._population_with_anchor(), 50, seed=seed)
        assert ANCHOR_UUID in _uuids(result.selected)

    def test_anchor_is_never_dispositioned_as_ineligible(self):
        result = _mod.select(self._population_with_anchor(), 50, seed='s')
        assert ANCHOR_UUID not in result.not_selected


# ===========================================================================
# The reader seam — read-only tripwires and the outcome-blind projection
# ===========================================================================


class _ReadOnlyViolation(AssertionError):
    """Raised by the double when the reader touches a write-capable path."""


class _FalkorDouble:
    """A FalkorDB graph handle stand-in that cannot be written through.

    A real class, never MagicMock (enforced repo-wide by
    fused-memory/scripts/check_bare_magicmock_config.py) — a MagicMock would
    cheerfully absorb ``graph.query('CREATE …')`` and report success, which is
    precisely the failure this double exists to make impossible.

    It records every Cypher string it is handed so the projection can be
    asserted on rather than guessed at, and raises from ``query()`` — the
    write-capable command path — and from any ``execute_command`` verb other
    than ``GRAPH.RO_QUERY``.
    """

    def __init__(self, rows: list[list[object]] | None = None):
        self.rows = rows if rows is not None else []
        self.ro_queries: list[str] = []
        self.violations: list[str] = []

    async def ro_query(self, cypher, *args, **kwargs):
        self.ro_queries.append(cypher)
        return types.SimpleNamespace(result_set=[list(row) for row in self.rows])

    # -- tripwires: every write-capable path -------------------------------
    async def query(self, cypher, *args, **kwargs):
        self.violations.append(f'query:{cypher}')
        raise _ReadOnlyViolation(f'reader used the write-capable query() path: {cypher!r}')

    async def execute_command(self, verb, *args, **kwargs):
        if verb != 'GRAPH.RO_QUERY':
            self.violations.append(f'execute_command:{verb}')
            raise _ReadOnlyViolation(f'reader issued a non-read-only command: {verb!r}')
        self.ro_queries.append(str(args[-1]) if args else '')
        return [[], [list(row) for row in self.rows], []]

    async def delete(self, *args, **kwargs):
        self.violations.append('delete')
        raise _ReadOnlyViolation('reader called delete()')

    async def create_node(self, *args, **kwargs):
        self.violations.append('create_node')
        raise _ReadOnlyViolation('reader called create_node()')

    async def merge(self, *args, **kwargs):
        self.violations.append('merge')
        raise _ReadOnlyViolation('reader called merge()')

    async def create_index(self, *args, **kwargs):
        self.violations.append('create_index')
        raise _ReadOnlyViolation('reader called create_index()')

    async def build_indices_and_constraints(self, *args, **kwargs):
        self.violations.append('build_indices_and_constraints')
        raise _ReadOnlyViolation(
            'reader triggered index construction — this is the evidence-destroying '
            'hazard owned by docs/prds/falkordb-index-provisioning.md'
        )


def _row(uuid: str, month: str = '2026-05', kind: str = 'temporal_facts') -> list[object]:
    """One result_set row, in PROJECTED_FIELDS order."""
    return [
        uuid,
        f'ep-{uuid[:8]}',
        'dark_factory',
        f'add_memory:{kind}',
        f'{month}-16T12:00:00+00:00',
        f'content of {uuid}',
    ]


def _fetch(double) -> list:
    """Drive the reader against *double* and return the population."""
    return asyncio.run(_mod.EpisodeReader(graph=double).fetch_population())


class TestReaderProjectionCarriesNoOutcomeSignal:
    """Hazard leg 1: the query cannot see the incumbent's output at all."""

    def test_cypher_never_mentions_entity_edges(self):
        double = _FalkorDouble([_row('u1')])
        _fetch(double)
        assert double.ro_queries
        for cypher in double.ro_queries:
            assert 'entity_edges' not in cypher, cypher

    @pytest.mark.parametrize(
        'signal',
        ['entity_edges', 'edges', 'extracted', 'valid_at', 'score', 'confidence'],
    )
    def test_cypher_mentions_no_incumbent_success_signal(self, signal):
        double = _FalkorDouble([_row('u1')])
        _fetch(double)
        assert signal not in ' '.join(double.ro_queries)

    def test_return_list_is_exactly_the_allowed_field_set(self):
        """The projection is closed, not merely 'does not currently include'."""
        double = _FalkorDouble([_row('u1')])
        _fetch(double)
        cypher = double.ro_queries[0]
        returned = re.findall(r'e\.([A-Za-z_]+)', cypher)
        assert tuple(returned) == _mod.PROJECTED_FIELDS

    def test_query_has_no_where_clause(self):
        """No filter of any kind — the population is the whole population.

        A WHERE clause is where outcome conditioning would live, so its
        absence is asserted directly rather than inferred from the field list.
        """
        double = _FalkorDouble([_row('u1')])
        _fetch(double)
        for cypher in double.ro_queries:
            assert 'WHERE' not in cypher.upper(), cypher

    def test_query_has_no_limit(self):
        """The FULL population is needed or the stratum denominators are wrong.

        A LIMIT would silently bias the census toward whatever order the store
        returned, and every proportional share computed from it would be wrong.
        """
        double = _FalkorDouble([_row('u1')])
        _fetch(double)
        for cypher in double.ro_queries:
            assert 'LIMIT' not in cypher.upper(), cypher


class TestOutcomeFailedEpisodesStayEligible:
    """Hazard leg 2: the anchor survives a full fetch."""

    def test_anchor_is_fetched(self):
        """The one episode of 2770 the incumbent extracted ZERO edges from."""
        double = _FalkorDouble([_row('u1'), _row(ANCHOR_UUID), _row('u2')])
        population = _fetch(double)
        assert ANCHOR_UUID in [r.uuid for r in population]

    def test_anchor_is_selectable_after_a_real_fetch(self):
        rows = [_row(f'u{i:04d}') for i in range(40)]
        rows.append(_row(ANCHOR_UUID, '2026-08', 'procedural_knowledge'))
        population = _fetch(_FalkorDouble(rows))
        result = _mod.select(population, 20, seed='s')
        assert ANCHOR_UUID in [r.uuid for r in result.selected]

    def test_fetched_records_carry_no_outcome_attribute(self):
        population = _fetch(_FalkorDouble([_row(ANCHOR_UUID)]))
        assert not hasattr(population[0], 'entity_edges')


class TestReaderIsReadOnly:
    """Hazard leg 3: the read-only guarantee, and the tripwires that prove it."""

    def test_full_fetch_trips_no_write_path(self):
        double = _FalkorDouble([_row(f'u{i}') for i in range(20)])
        population = _fetch(double)
        assert len(population) == 20
        assert double.violations == []
        assert double.ro_queries

    def test_the_double_would_have_caught_a_write(self):
        """The guarantee is only worth what the tripwires are worth.

        A tripwire nobody proved fires is not a guarantee. Precedent:
        test_memory_eval_retrieval_probe.py::test_the_double_would_have_caught_a_write.
        """
        double = _FalkorDouble()
        for name in (
            'query',
            'delete',
            'create_node',
            'merge',
            'create_index',
            'build_indices_and_constraints',
        ):
            with pytest.raises(_ReadOnlyViolation):
                asyncio.run(getattr(double, name)('CREATE (n:Junk)'))
        assert len(double.violations) == 6

    def test_the_double_would_have_caught_a_non_ro_command(self):
        double = _FalkorDouble()
        with pytest.raises(_ReadOnlyViolation):
            asyncio.run(double.execute_command('GRAPH.QUERY', 'g', 'CREATE (n:Junk)'))
        assert double.violations == ['execute_command:GRAPH.QUERY']

    def test_reader_rejects_a_non_ro_command_client_side(self):
        """A client-side guard on top of the server-side one.

        GRAPH.RO_QUERY is refused server-side, but the guard makes the
        violation a typed CorpusBuildError at the seam rather than a redis
        error surfacing from three layers down.
        """
        with pytest.raises(_mod.CorpusBuildError) as exc:
            _mod.EpisodeReader.assert_read_only_command('GRAPH.QUERY')
        assert 'GRAPH.RO_QUERY' in str(exc.value)
        _mod.EpisodeReader.assert_read_only_command('GRAPH.RO_QUERY')


class TestNoGraphitiDriverIsConstructed:
    """The evidence-destroying hazard, tested rather than promised."""

    def test_fetch_succeeds_with_falkor_driver_init_booby_trapped(self, monkeypatch):
        """FalkorDriver.__init__ fire-and-forgets build_indices_and_constraints().

        Constructing one would create indices on dark_factory and destroy the
        no-index evidence owned by docs/prds/falkordb-index-provisioning.md. So
        the test makes construction fatal and asserts the fetch still succeeds.
        """
        from graphiti_core.driver import falkordb_driver  # noqa: PLC0415

        def _boom(*args, **kwargs):
            raise AssertionError(
                'build_corpus constructed a graphiti FalkorDriver — this fires '
                'build_indices_and_constraints() and destroys the no-index evidence'
            )

        monkeypatch.setattr(falkordb_driver.FalkorDriver, '__init__', _boom)
        population = _fetch(_FalkorDouble([_row('u1'), _row('u2')]))
        assert len(population) == 2

    def test_builder_source_never_names_the_graphiti_driver_as_an_import(self):
        """A source-level guard: the import cannot creep back in unnoticed."""
        source = SCRIPT_PATH.read_text(encoding='utf-8')
        imports = [
            line
            for line in source.splitlines()
            if line.strip().startswith(('import ', 'from '))
        ]
        assert not [ln for ln in imports if 'falkordb_driver' in ln or 'FalkorDriver' in ln]
        assert not [ln for ln in imports if 'graphiti_core' in ln]


# ===========================================================================
# Manifest — structure, reconciliation, byte-stability
# ===========================================================================

# The literal the delivered_check greps for on the COMMITTED tree
# (plans/local-memory-models-eval-prd.capability-manifest.yaml). Assembled from
# fragments so this test module never itself satisfies the check it is
# asserting on — the manifest-pair docs use the same trick with 'PRD[-]MARKER'.
MARKER = 'PRD' + '-MARKER:local-memory-models-eval corpus' + '-manifest'


def _built(n: int = 20, seed: str = 's', population=None):
    """Build a manifest over a synthetic population."""
    population = population if population is not None else _population(SYNTHETIC_CELLS)
    result = _mod.select(population, n, seed=seed)
    return _mod.build_manifest(result, population, n=n, seed=seed, graph='dark_factory')


class TestManifestEpisodes:
    """The ids + content hashes the task requires."""

    def test_one_entry_per_selected_episode(self):
        manifest = _built(n=20)
        assert len(manifest['episodes']) == 20

    def test_entry_shape(self):
        entry = _built()['episodes'][0]
        assert set(entry) == {'uuid', 'content_hash', 'month', 'payload_kind'}

    def test_content_hash_is_the_full_digest_of_the_episode_content(self):
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        by_uuid = {r.uuid: r for r in population}
        for entry in manifest['episodes']:
            assert entry['content_hash'] == _mod.content_hash(by_uuid[entry['uuid']].content)
            assert len(entry['content_hash']) == 64

    def test_episode_content_itself_is_not_copied_into_the_manifest(self):
        """The manifest is an index, not a copy of the store.

        The replay engines read content from the store by uuid; duplicating it
        here would create a second copy that can silently disagree with the
        first, and hash drift would then be undetectable.
        """
        entry = _built()['episodes'][0]
        assert 'content' not in entry

    def test_episodes_are_in_the_selection_order(self):
        manifest = _built(n=20)
        result = _mod.select(_population(SYNTHETIC_CELLS), 20, seed='s')
        assert [e['uuid'] for e in manifest['episodes']] == _uuids(result.selected)


class TestManifestCriteria:
    """Everything needed to re-derive the sample, recorded in the artifact."""

    def test_records_every_re_derivation_input(self):
        criteria = _built(n=20, seed='s')['criteria']
        assert criteria['seed'] == 's'
        assert criteria['n'] == 20
        assert criteria['graph'] == 'dark_factory'
        assert criteria['stratification_dimensions'] == [
            'month(created_at)',
            'payload_kind(source_description)',
        ]
        assert criteria['allocation_rule'] == _mod.ALLOCATION_RULE
        assert criteria['selection_rule'] == _mod.SELECTION_RULE
        assert criteria['builder_version'] == _mod.BUILDER_VERSION
        assert criteria['population_size'] == len(_population(SYNTHETIC_CELLS))
        # Existence only. PRD Open Q4 defers the final N to zeta, so the field
        # must survive in the artifact's surface — but its prose is prose, and
        # pinning a substring of it would test wording, not the contract.
        assert 'n_rationale' in criteria

    def test_records_the_observed_window(self):
        criteria = _built()['criteria']
        assert criteria['window']['min_created_at'] <= criteria['window']['max_created_at']
        assert criteria['window']['min_created_at'].startswith('2026-04')


class TestStratificationReport:
    """Per-cell accounting that reconciles to the whole population."""

    def test_covers_every_non_empty_cell(self):
        report = _built(n=20)['stratification_report']
        assert set(report['cells']) == {f'{m}|{k}' for m, k in SYNTHETIC_CELLS}

    def test_each_cell_reports_population_allocated_selected(self):
        report = _built(n=20)['stratification_report']
        for key, cell in report['cells'].items():
            assert set(cell) == {'population', 'allocated', 'selected'}
            assert cell['allocated'] == cell['selected'], key
            assert cell['selected'] <= cell['population'], key

    def test_selected_totals_to_n(self):
        report = _built(n=20)['stratification_report']
        assert sum(c['selected'] for c in report['cells'].values()) == 20
        assert report['totals']['selected'] == 20

    def test_population_totals_reconcile_no_cell_silently_dropped(self):
        """The check that makes the report trustworthy rather than decorative.

        If the per-cell populations do not sum to population_size, a cell was
        dropped between the fetch and the report — and every proportional share
        in the artifact is then computed against a denominator that does not
        describe the store.
        """
        manifest = _built(n=20)
        report = manifest['stratification_report']
        summed = sum(c['population'] for c in report['cells'].values())
        assert summed == report['totals']['population']
        assert summed == manifest['criteria']['population_size']

    def test_reports_the_not_selected_count(self):
        """Exhaustive disposition survives into the artifact."""
        manifest = _built(n=20)
        totals = manifest['stratification_report']['totals']
        assert totals['selected'] + totals['not_selected'] == totals['population']


class TestNoOutcomeFilterStatement:
    """The guarantee, recorded so a reviewer can CHECK it rather than trust it."""

    def test_records_the_projected_field_list_as_a_machine_checkable_fact(self):
        block = _built()['no_outcome_filter']
        assert block['projected_fields'] == list(_mod.PROJECTED_FIELDS)

    def test_projected_fields_contain_no_outcome_signal(self):
        block = _built()['no_outcome_filter']
        for field in block['projected_fields']:
            assert 'entity_edges' not in field
            assert 'edges' not in field

    def test_records_that_entity_edges_was_neither_queried_nor_used(self):
        block = _built()['no_outcome_filter']
        assert block['entity_edges_queried'] is False
        assert block['entity_edges_used_as_filter'] is False
        assert block['entity_edges_used_as_stratum'] is False

    def test_records_the_query_that_was_actually_issued(self):
        """The strongest checkable fact: the reviewer reads the real Cypher."""
        block = _built()['no_outcome_filter']
        # Existence only — the prose statement stays in the artifact's surface,
        # but its wording is not the guarantee. The asserts below are.
        assert isinstance(block['statement'], str) and block['statement']
        assert block['population_query'] == _mod.POPULATION_CYPHER
        assert 'entity_edges' not in block['population_query']
        assert 'WHERE' not in block['population_query'].upper()

    def test_records_the_stratification_dimensions_used(self):
        block = _built()['no_outcome_filter']
        assert block['stratification_dimensions'] == [
            'month(created_at)',
            'payload_kind(source_description)',
        ]


class TestManifestSerialization:
    """Canonical, re-runnable, and carrying the marker."""

    def test_carries_the_prd_marker_as_a_top_level_field(self):
        assert _built()['prd_marker'] == MARKER

    def test_marker_survives_serialization_intact(self):
        """The delivered_check greps the SERIALIZED text, not the dict."""
        assert MARKER in _mod.serialize_manifest(_built())

    def test_serialization_is_canonical(self):
        text = _mod.serialize_manifest(_built())
        assert text.endswith('\n')
        assert json.loads(text) == _built()
        keys = list(json.loads(text))
        assert keys == sorted(keys)

    def test_two_builds_over_an_unchanged_population_are_byte_identical(self):
        """The determinism convention: a re-run must diff cleanly against itself.

        Without it a real change would be unreadable, buried in incidental
        reordering.
        """
        first = _mod.serialize_manifest(_built())
        for _ in range(3):
            assert _mod.serialize_manifest(_built()) == first

    def test_shuffled_population_yields_a_byte_identical_manifest(self):
        """Store row order must not reach the artifact."""
        population = _population(SYNTHETIC_CELLS)
        baseline = _mod.serialize_manifest(_built(population=population))
        shuffled = list(population)
        random.Random(3).shuffle(shuffled)
        assert _mod.serialize_manifest(_built(population=shuffled)) == baseline

    def test_non_ascii_is_kept_verbatim_not_escaped(self):
        population = [
            *_population(SYNTHETIC_CELLS),
            _record('ffffffff-0000-0000-0000-000000000000', '2026-08', 'temporal_facts',
                    content='δ corpus — naïve'),
        ]
        text = _mod.serialize_manifest(_built(n=20, population=population))
        assert '\\u' not in text

    def test_write_manifest_round_trips(self, tmp_path):
        path = tmp_path / 'corpus_manifest.json'
        manifest = _built()
        _mod.write_manifest(manifest, path)
        assert json.loads(path.read_text(encoding='utf-8')) == manifest
        assert MARKER in path.read_text(encoding='utf-8')

    def test_default_manifest_path_is_under_scripts_not_gitignored_data(self):
        """fused-memory/.gitignore ignores data/, so a manifest written to the
        usual eval-artifact directory would be uncommittable and would fail the
        delivered_check, which greps the COMMITTED tree.
        """
        path = Path(_mod.DEFAULT_MANIFEST_PATH)
        assert path.parent.name == 'local_memory_models_eval'
        assert 'scripts' in path.parts
        assert 'data' not in path.parts


# ===========================================================================
# Re-derivability — the task's user-observable signal
# ===========================================================================


class TestVerifyManifestReDerives:
    """A reviewer re-derives the sample from the manifest's OWN criteria."""

    def test_clean_verify_reports_ok_with_counts(self):
        population = _population(SYNTHETIC_CELLS)
        report = _mod.verify_manifest(_built(population=population), population)
        assert report.status == 'ok'
        assert report.ok is True
        assert report.checked == 20
        assert not report.id_mismatch
        assert not report.hash_drift
        assert not report.missing

    def test_re_derivation_reproduces_the_exact_uuid_list_in_order(self):
        """The property a reviewer actually checks."""
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        report = _mod.verify_manifest(manifest, population)
        assert report.rederived == [e['uuid'] for e in manifest['episodes']]

    def test_verification_does_not_read_the_episode_list_as_its_own_answer(self):
        """The check must not be vacuous.

        If verify re-read the manifest's own `episodes` list as the source of
        truth for what to select, it would pass on ANY manifest — including a
        tampered one. So: replace the episode list wholesale with records that
        the recorded criteria would never select, and demand a failure.
        """
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        wrong = [
            {
                'uuid': r.uuid,
                'content_hash': _mod.content_hash(r.content),
                'month': _mod.stratum_key(r)[0],
                'payload_kind': _mod.stratum_key(r)[1],
            }
            for r in population[:20]
        ]
        assert [e['uuid'] for e in wrong] != [e['uuid'] for e in manifest['episodes']]
        report = _mod.verify_manifest({**manifest, 'episodes': wrong}, population)
        assert report.status == 'id_mismatch'


class TestVerifyIsPinnedToTheRecordedWindow:
    """Re-derivation samples the frame the manifest names, not today's store.

    The regression this class exists for: `dark_factory` is written
    continuously, so between building the corpus and checking it the population
    has already grown. Sampling the grown population shifts the cell
    denominators, largest-remainder moves a seat, and `--verify` reports
    `id_mismatch` on a perfectly good artifact — measured live during this
    task, population 2775 at build and 2776 minutes later. Without the window
    bound a green verify is a statement about a store that no longer exists.
    """

    def _later(self, uuid: str, created_at: str, kind: str = 'temporal_facts'):
        """One record at an explicit instant, bypassing `_record`'s fixed time."""
        return _mod.EpisodeRecord(
            uuid=uuid,
            name=f'ep-{uuid[:8]}',
            group_id='dark_factory',
            source_description=f'add_memory:{kind}',
            created_at=created_at,
            content=f'content of {uuid}',
        )

    def test_the_recorded_window_bounds_the_population_under_test(self):
        """Guard the fixture: the appended episodes really are outside it."""
        manifest = _built(population=_population(SYNTHETIC_CELLS))
        assert manifest['criteria']['window']['max_created_at'] == '2026-07-16T12:00:00+00:00'

    def test_an_episode_written_after_the_window_does_not_break_verification(self):
        """The bug, directly: the factory keeps writing and verify stays green."""
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        assert _mod.verify_manifest(manifest, population).status == 'ok'

        grown = [
            *population,
            # A brand-new month cell, as an episode written "now" would be.
            self._later('aaaaaaaa-0000-0000-0000-000000000000', '2026-08-01T09:00:00+00:00'),
            # And a later episode inside an EXISTING cell, which shifts that
            # cell's denominator and permutation rather than adding a stratum.
            self._later('bbbbbbbb-0000-0000-0000-000000000000', '2026-07-16T12:00:01+00:00'),
        ]
        report = _mod.verify_manifest(manifest, grown)
        assert report.status == 'ok', report.detail
        assert report.rederived == [e['uuid'] for e in manifest['episodes']]

    def test_the_appended_episodes_would_have_broken_an_unwindowed_check(self):
        """Prove the regression test bites.

        If growth did not perturb the sample, the test above would pass with
        the window filter deleted and would be guarding nothing. Re-deriving
        from the grown population WITHOUT the bound must differ.
        """
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        grown = [
            *population,
            self._later('aaaaaaaa-0000-0000-0000-000000000000', '2026-08-01T09:00:00+00:00'),
            self._later('bbbbbbbb-0000-0000-0000-000000000000', '2026-07-16T12:00:01+00:00'),
        ]
        unwindowed = _uuids(_mod.select(grown, manifest['criteria']['n'], seed='s').selected)
        assert unwindowed != [e['uuid'] for e in manifest['episodes']]

    def test_an_episode_inside_the_window_still_fails_verification(self):
        """The negative companion — the window must not disable the check.

        An episode dated INSIDE the recorded frame but absent from the build
        population means the frame itself is not what the manifest says it was.
        If that passed too, windowing would be indistinguishable from deleting
        the re-derivation.
        """
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        infill = [
            *population,
            self._later('cccccccc-0000-0000-0000-000000000000', '2026-05-16T12:00:00+00:00'),
        ]
        report = _mod.verify_manifest(manifest, infill)
        assert report.status == 'id_mismatch'

    def test_timestamps_are_compared_normalized_not_as_strings(self):
        """The store spells the same instant four ways.

        `'2026-07-16 12:00:00Z'` sorts after `'2026-07-16T12:00:00+00:00'` as a
        string while naming the same instant, so a string comparison would drop
        a boundary episode out of the frame and fail re-derivation.
        """
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        restyled = [
            _mod.EpisodeRecord(
                **{
                    **dataclasses.asdict(r),
                    'created_at': r.created_at.replace('T', ' ').replace('+00:00', 'Z'),
                }
            )
            for r in population
        ]
        report = _mod.verify_manifest(manifest, restyled)
        assert report.status == 'ok', report.detail

    def test_a_naive_timestamp_is_read_as_utc_not_rejected(self):
        """Aware-vs-naive comparison is a TypeError, not a verdict."""
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        naive = [
            _mod.EpisodeRecord(
                **{**dataclasses.asdict(r), 'created_at': r.created_at.replace('+00:00', '')}
            )
            for r in population
        ]
        report = _mod.verify_manifest(manifest, naive)
        assert report.status == 'ok', report.detail

    @pytest.mark.parametrize(
        'window',
        [None, {}, {'min_created_at': '2026-04-06T00:19:33+00:00'}, {'max_created_at': 7},
         {'max_created_at': 'not-a-timestamp'}],
    )
    def test_a_manifest_that_cannot_state_its_frame_is_bad_manifest(self, window):
        """The dependency is explicit, not assumed.

        Verification now reads `criteria.window.max_created_at`, so a manifest
        that omits or mangles it must fail as structurally unusable rather than
        be silently checked against an unbounded population.
        """
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        criteria = {k: v for k, v in manifest['criteria'].items() if k != 'window'}
        if window is not None:
            criteria['window'] = window
        report = _mod.verify_manifest({**manifest, 'criteria': criteria}, population)
        assert report.status == 'bad_manifest'
        assert 'window' in report.detail or 'max_created_at' in report.detail

    def test_the_window_does_not_hide_a_deleted_episode(self):
        """`missing` stays on the UNFILTERED population.

        "The store still holds what the corpus names" is a different question
        from "what was the sampling frame". If the window were applied to the
        presence check too, a deletion could hide behind a bound the manifest
        itself supplies.
        """
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        gone = manifest['episodes'][0]['uuid']
        grown_and_reduced = [
            *(r for r in population if r.uuid != gone),
            self._later('aaaaaaaa-0000-0000-0000-000000000000', '2026-08-01T09:00:00+00:00'),
        ]
        report = _mod.verify_manifest(manifest, grown_and_reduced)
        assert report.status == 'missing_episodes'
        assert gone in report.missing

    def test_the_window_does_not_hide_content_drift_outside_it(self):
        """Hash drift is checked against the store as it is, not as it was."""
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        target = manifest['episodes'][0]['uuid']
        drifted = [
            _mod.EpisodeRecord(**{**dataclasses.asdict(r), 'content': 'EDITED'})
            if r.uuid == target
            else r
            for r in population
        ] + [self._later('aaaaaaaa-0000-0000-0000-000000000000', '2026-08-01T09:00:00+00:00')]
        report = _mod.verify_manifest(manifest, drifted)
        assert report.status == 'hash_drift'
        assert target in report.hash_drift


class TestVerifyCatchesTampering:
    """A tampered criterion must fail loudly, with a structured diff."""

    @pytest.mark.parametrize('field,value', [('seed', 'tampered'), ('n', 25)])
    def test_mutated_criterion_fails(self, field, value):
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        tampered = {
            **manifest,
            'criteria': {**manifest['criteria'], field: value},
        }
        report = _mod.verify_manifest(tampered, population)
        assert report.ok is False
        assert report.status == 'id_mismatch'

    def test_failure_names_expected_vs_actual_not_a_bare_boolean(self):
        """A bare False tells a reviewer nothing about what to fix."""
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        tampered = {**manifest, 'criteria': {**manifest['criteria'], 'seed': 'x'}}
        report = _mod.verify_manifest(tampered, population)
        assert report.id_mismatch['only_in_manifest']
        assert report.id_mismatch['only_in_rederivation']
        assert 'seed' in report.detail or 'id' in report.detail.lower()

    def test_a_single_swapped_id_is_caught(self):
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        episodes = [dict(e) for e in manifest['episodes']]
        outside = next(
            r.uuid for r in population if r.uuid not in {e['uuid'] for e in episodes}
        )
        episodes[0] = {**episodes[0], 'uuid': outside}
        report = _mod.verify_manifest({**manifest, 'episodes': episodes}, population)
        assert report.status == 'id_mismatch'


class TestVerifyCatchesContentDrift:
    """Hash drift is a DIFFERENT failure from an id mismatch."""

    def _drifted(self):
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        target = manifest['episodes'][0]['uuid']
        drifted = [
            _record(r.uuid, *_mod.stratum_key(r), content='EDITED')
            if r.uuid == target
            else r
            for r in population
        ]
        return manifest, drifted, target

    def test_drift_is_reported_with_the_affected_uuids(self):
        manifest, drifted, target = self._drifted()
        report = _mod.verify_manifest(manifest, drifted)
        assert report.ok is False
        assert report.status == 'hash_drift'
        assert target in report.hash_drift

    def test_drift_entry_carries_expected_and_actual_hashes(self):
        manifest, drifted, target = self._drifted()
        report = _mod.verify_manifest(manifest, drifted)
        entry = report.hash_drift[target]
        assert entry['expected'] != entry['actual']
        assert entry['actual'] == _mod.content_hash('EDITED')

    def test_drift_is_distinct_from_an_id_mismatch(self):
        """Different failures, different remedies.

        An id mismatch means the manifest no longer describes what its criteria
        produce — re-derive or investigate tampering. Hash drift means the
        SELECTION is still right but the replay inputs changed underneath it —
        the corpus needs re-hashing and epsilon/zeta/theta results computed
        against the old bytes are suspect. Collapsing both into one non-zero
        status would send a reviewer down the wrong path.
        """
        manifest, drifted, _ = self._drifted()
        report = _mod.verify_manifest(manifest, drifted)
        assert report.status == 'hash_drift'
        assert not report.id_mismatch
        assert _mod.EXIT_CODES['hash_drift'] != _mod.EXIT_CODES['id_mismatch']


class TestVerifyCatchesMissingEpisodes:
    """An id in the manifest but absent from the store is reported, not skipped."""

    def _without_a_selected_episode(self):
        population = _population(SYNTHETIC_CELLS)
        manifest = _built(population=population)
        gone = manifest['episodes'][0]['uuid']
        return manifest, [r for r in population if r.uuid != gone], gone

    def test_vanished_episode_is_reported_as_missing(self):
        manifest, reduced, gone = self._without_a_selected_episode()
        report = _mod.verify_manifest(manifest, reduced)
        assert report.ok is False
        assert gone in report.missing

    def test_missing_is_never_silently_skipped(self):
        """The absorb-and-continue failure this seam exists to prevent."""
        manifest, reduced, _ = self._without_a_selected_episode()
        report = _mod.verify_manifest(manifest, reduced)
        assert report.status != 'ok'
        assert _mod.EXIT_CODES[report.status] != 0


class TestVerifyStatusSemantics:
    """Each failure class is distinguishable by exit code (the INV-2/INV-4 seam)."""

    def test_exit_codes_cover_every_status(self):
        assert set(_mod.EXIT_CODES) == {
            'ok',
            'id_mismatch',
            'hash_drift',
            'missing_episodes',
            'bad_manifest',
        }

    def test_ok_is_zero_and_every_failure_is_distinctly_non_zero(self):
        codes = _mod.EXIT_CODES
        assert codes['ok'] == 0
        failures = [v for k, v in codes.items() if k != 'ok']
        assert all(v != 0 for v in failures)
        assert len(set(failures)) == len(failures), 'failure codes must be distinct'

    @pytest.mark.parametrize(
        'broken',
        [
            {},
            {'episodes': []},
            {'criteria': {'seed': 's'}},
            {'criteria': {'seed': 's', 'n': 20}, 'episodes': 'not-a-list'},
        ],
    )
    def test_a_structurally_broken_manifest_is_bad_manifest_not_a_traceback(
        self, broken
    ):
        report = _mod.verify_manifest(broken, _population(SYNTHETIC_CELLS))
        assert report.status == 'bad_manifest'
        assert report.ok is False
        assert report.detail

    def test_report_exposes_its_own_exit_code(self):
        population = _population(SYNTHETIC_CELLS)
        assert _mod.verify_manifest(_built(population=population), population).exit_code == 0
        assert _mod.verify_manifest({}, population).exit_code == (
            _mod.EXIT_CODES['bad_manifest']
        )


# ===========================================================================
# CLI — flat flags, build and verify modes, driven in-process
# ===========================================================================

# Captured BEFORE any monkeypatching, so the stub factory can construct the
# real reader without recursing into itself.
_REAL_READER = _mod.EpisodeReader


class _StubReaderFactory:
    """Stands in for the module-level ``EpisodeReader`` so main() opens no socket.

    Hands back a REAL reader bound to the offline ``_FalkorDouble`` rather than
    a fake reader: the CLI tests then exercise the actual fetch path — and its
    read-only tripwires — instead of a parallel re-implementation of it that
    could pass while the real one writes.
    """

    def __init__(self, double: _FalkorDouble):
        self.double = double
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return _REAL_READER(graph=self.double, **kwargs)


def _cli_rows(population=None) -> list[list[object]]:
    """Result-set rows for a population, in PROJECTED_FIELDS order."""
    population = population if population is not None else _population(SYNTHETIC_CELLS)
    return [
        [r.uuid, r.name, r.group_id, r.source_description, r.created_at, r.content]
        for r in population
    ]


def _run_cli(monkeypatch, *argv, rows=None) -> tuple[int, _StubReaderFactory]:
    """Drive ``main()`` in-process against the offline double.

    Precedent: ``_run_cli`` in test_memory_eval_transcript_corpus.py. Returns
    the exit code plus the factory, so a test can assert on what the CLI asked
    the reader for as well as what it produced.
    """
    factory = _StubReaderFactory(_FalkorDouble(rows if rows is not None else _cli_rows()))
    monkeypatch.setattr(_mod, 'EpisodeReader', factory)
    return _mod.main(list(argv)), factory


class TestCliFlagSurface:
    """Flat mode flags, no subcommands — the memory_eval_transcript_corpus shape."""

    def test_flag_set_is_pinned_by_equality(self):
        """By EQUALITY, so a flag cannot appear without a test saying so.

        Same treatment as memory_eval_retrieval_probe's no-mutating-flag test:
        this script is read-only against the store, and the CLI is the surface
        where a write band would be added.
        """
        parser = _mod._build_parser()
        flags = {opt for action in parser._actions for opt in action.option_strings}
        assert flags == {
            '-h', '--help',
            '--verify', '--n', '--seed', '--graph', '--out', '--dry-run',
        }

    def test_there_are_no_subcommands_and_no_positionals(self):
        """Mode is a flag, not a verb — `--verify` selects it, build is default."""
        parser = _mod._build_parser()
        assert [a.dest for a in parser._actions if not a.option_strings] == []

    def test_defaults(self):
        args = _mod._build_parser().parse_args([])
        assert args.verify is False
        assert args.dry_run is False
        assert args.n == _mod.DEFAULT_N == 200
        assert args.graph == _mod.DEFAULT_GRAPH == 'dark_factory'
        assert args.out == _mod.DEFAULT_MANIFEST_PATH
        assert args.seed == _mod.DEFAULT_SEED

    def test_n_is_parsed_as_an_int_not_a_string(self):
        """`allocate` rejects a non-int n, so argparse must do the conversion."""
        assert _mod._build_parser().parse_args(['--n', '150']).n == 150

    def test_every_flag_has_help_text(self):
        """Each help names the failure mode its flag guards."""
        parser = _mod._build_parser()
        assert all(
            a.help for a in parser._actions if a.dest not in ('help',)
        ), [a.dest for a in parser._actions if not a.help]


class TestCliBuildMode:
    """Build is the default mode and writes the committed artifact."""

    def test_writes_the_manifest_and_exits_zero(self, monkeypatch, tmp_path):
        out = tmp_path / 'corpus_manifest.json'
        code, _ = _run_cli(monkeypatch, '--n', '20', '--seed', 's', '--out', str(out))
        assert code == 0
        manifest = json.loads(out.read_text(encoding='utf-8'))
        assert len(manifest['episodes']) == 20
        assert manifest['criteria']['seed'] == 's'

    def test_written_manifest_carries_the_marker(self, monkeypatch, tmp_path):
        out = tmp_path / 'corpus_manifest.json'
        _run_cli(monkeypatch, '--n', '20', '--seed', 's', '--out', str(out))
        assert MARKER in out.read_text(encoding='utf-8')

    def test_a_second_build_over_an_unchanged_population_is_byte_identical(
        self, monkeypatch, tmp_path
    ):
        """Re-running must diff cleanly, or a real change is unreadable in noise."""
        out = tmp_path / 'corpus_manifest.json'
        args = ('--n', '20', '--seed', 's', '--out', str(out))
        _run_cli(monkeypatch, *args)
        first = out.read_bytes()
        _run_cli(monkeypatch, *args)
        assert out.read_bytes() == first

    def test_row_order_from_the_store_does_not_change_the_artifact(
        self, monkeypatch, tmp_path
    ):
        """FalkorDB does not guarantee a stable row order between runs."""
        out = tmp_path / 'a.json'
        rows = _cli_rows()
        _run_cli(monkeypatch, '--n', '20', '--seed', 's', '--out', str(out), rows=rows)
        shuffled = list(rows)
        random.Random(7).shuffle(shuffled)
        other = tmp_path / 'b.json'
        _run_cli(
            monkeypatch, '--n', '20', '--seed', 's', '--out', str(other), rows=shuffled
        )
        assert other.read_bytes() == out.read_bytes()

    def test_n_flag_is_honoured(self, monkeypatch, tmp_path):
        out = tmp_path / 'corpus_manifest.json'
        _run_cli(monkeypatch, '--n', '30', '--seed', 's', '--out', str(out))
        manifest = json.loads(out.read_text(encoding='utf-8'))
        assert len(manifest['episodes']) == 30 == manifest['criteria']['n']

    def test_seed_flag_changes_the_corpus(self, monkeypatch, tmp_path):
        first, second = tmp_path / 'a.json', tmp_path / 'b.json'
        _run_cli(monkeypatch, '--n', '20', '--seed', 'one', '--out', str(first))
        _run_cli(monkeypatch, '--n', '20', '--seed', 'two', '--out', str(second))
        ids = [
            [e['uuid'] for e in json.loads(p.read_text(encoding='utf-8'))['episodes']]
            for p in (first, second)
        ]
        assert set(ids[0]) != set(ids[1])

    def test_graph_flag_reaches_the_reader(self, monkeypatch, tmp_path):
        out = tmp_path / 'corpus_manifest.json'
        _, factory = _run_cli(
            monkeypatch, '--n', '20', '--seed', 's', '--graph', 'other_graph',
            '--out', str(out),
        )
        assert factory.calls == [{'graph_name': 'other_graph'}]

    def test_build_prints_the_stratification_report(self, monkeypatch, tmp_path, capsys):
        out = tmp_path / 'corpus_manifest.json'
        _run_cli(monkeypatch, '--n', '20', '--seed', 's', '--out', str(out))
        printed = capsys.readouterr().out
        assert '2026-04|procedural_knowledge' in printed
        assert str(out) in printed

    def test_build_trips_no_write_path(self, monkeypatch, tmp_path):
        out = tmp_path / 'corpus_manifest.json'
        _, factory = _run_cli(monkeypatch, '--n', '20', '--seed', 's', '--out', str(out))
        assert factory.double.violations == []
        assert factory.double.ro_queries

    def test_an_unsatisfiable_n_exits_non_zero_and_writes_nothing(
        self, monkeypatch, tmp_path, capsys
    ):
        """A too-large n is a loud usage error, never a quietly shorter corpus."""
        out = tmp_path / 'corpus_manifest.json'
        code, _ = _run_cli(monkeypatch, '--n', '99999', '--out', str(out))
        assert code != 0
        assert not out.exists()
        assert capsys.readouterr().err.strip()


class TestCliDryRun:
    """`--dry-run` reports and writes nothing."""

    def test_writes_no_file(self, monkeypatch, tmp_path):
        out = tmp_path / 'corpus_manifest.json'
        code, _ = _run_cli(
            monkeypatch, '--n', '20', '--seed', 's', '--out', str(out), '--dry-run'
        )
        assert code == 0
        assert not out.exists()

    def test_prints_the_stratification_report(self, monkeypatch, tmp_path, capsys):
        out = tmp_path / 'corpus_manifest.json'
        _run_cli(monkeypatch, '--n', '20', '--seed', 's', '--out', str(out), '--dry-run')
        printed = capsys.readouterr().out
        for cell in SYNTHETIC_CELLS:
            assert f'{cell[0]}|{cell[1]}' in printed
        assert 'dry-run' in printed.lower()

    def test_does_not_overwrite_an_existing_manifest(self, monkeypatch, tmp_path):
        out = tmp_path / 'corpus_manifest.json'
        out.write_text('{"sentinel": true}\n', encoding='utf-8')
        _run_cli(
            monkeypatch, '--n', '20', '--seed', 's', '--out', str(out), '--dry-run'
        )
        assert json.loads(out.read_text(encoding='utf-8')) == {'sentinel': True}


class TestCliVerifyMode:
    """`--verify` re-derives the committed artifact and exits on its status code."""

    def _built_manifest(self, monkeypatch, tmp_path) -> Path:
        out = tmp_path / 'corpus_manifest.json'
        _run_cli(monkeypatch, '--n', '20', '--seed', 's', '--out', str(out))
        return out

    def test_a_good_manifest_exits_zero(self, monkeypatch, tmp_path):
        out = self._built_manifest(monkeypatch, tmp_path)
        code, _ = _run_cli(monkeypatch, '--verify', '--out', str(out))
        assert code == 0

    def test_a_good_manifest_says_so(self, monkeypatch, tmp_path, capsys):
        out = self._built_manifest(monkeypatch, tmp_path)
        capsys.readouterr()
        _run_cli(monkeypatch, '--verify', '--out', str(out))
        assert 'ok' in capsys.readouterr().out.lower()

    def test_a_tampered_seed_exits_id_mismatch(self, monkeypatch, tmp_path):
        out = self._built_manifest(monkeypatch, tmp_path)
        manifest = json.loads(out.read_text(encoding='utf-8'))
        manifest['criteria']['seed'] = 'tampered'
        out.write_text(json.dumps(manifest), encoding='utf-8')
        code, _ = _run_cli(monkeypatch, '--verify', '--out', str(out))
        assert code == _mod.EXIT_CODES['id_mismatch'] != 0

    def test_content_drift_exits_hash_drift(self, monkeypatch, tmp_path):
        out = self._built_manifest(monkeypatch, tmp_path)
        population = _population(SYNTHETIC_CELLS)
        manifest = json.loads(out.read_text(encoding='utf-8'))
        drifted_uuid = manifest['episodes'][0]['uuid']
        rows = _cli_rows(
            [
                dataclasses.replace(r, content='rewritten') if r.uuid == drifted_uuid else r
                for r in population
            ]
        )
        code, _ = _run_cli(monkeypatch, '--verify', '--out', str(out), rows=rows)
        assert code == _mod.EXIT_CODES['hash_drift']

    def test_a_vanished_episode_exits_missing_episodes(self, monkeypatch, tmp_path):
        out = self._built_manifest(monkeypatch, tmp_path)
        manifest = json.loads(out.read_text(encoding='utf-8'))
        gone = manifest['episodes'][0]['uuid']
        rows = [row for row in _cli_rows() if row[0] != gone]
        code, _ = _run_cli(monkeypatch, '--verify', '--out', str(out), rows=rows)
        assert code == _mod.EXIT_CODES['missing_episodes']

    def test_an_absent_manifest_is_bad_manifest_not_a_traceback(
        self, monkeypatch, tmp_path, capsys
    ):
        code, _ = _run_cli(
            monkeypatch, '--verify', '--out', str(tmp_path / 'nope.json')
        )
        assert code == _mod.EXIT_CODES['bad_manifest']
        assert capsys.readouterr().err.strip()

    def test_unparseable_json_is_bad_manifest(self, monkeypatch, tmp_path):
        out = tmp_path / 'corpus_manifest.json'
        out.write_text('{not json', encoding='utf-8')
        code, _ = _run_cli(monkeypatch, '--verify', '--out', str(out))
        assert code == _mod.EXIT_CODES['bad_manifest']

    def test_verify_writes_nothing(self, monkeypatch, tmp_path):
        """Verification is a read of the artifact, not a re-write of it."""
        out = self._built_manifest(monkeypatch, tmp_path)
        before = out.read_bytes()
        _run_cli(monkeypatch, '--verify', '--out', str(out))
        assert out.read_bytes() == before


class TestBuilderScriptSatisfiesTheDeliveredCheck:
    """The delivered_check greps fused-memory/scripts/ on the COMMITTED tree."""

    def test_module_source_carries_the_literal_marker(self):
        """Exactly how the delivered_check evaluates: a substring of the file.

        Deliberately NOT asserting the marker sits on a line of its own, nor
        where in the file it sits. ``git grep -cE`` matches a substring anywhere
        on a line, so a marker mid-sentence satisfies the check identically —
        pinning its layout would go red on an editorial reflow while the
        contract still holds.
        """
        assert MARKER in SCRIPT_PATH.read_text(encoding='utf-8')

    def test_main_returns_an_int_exit_code(self, monkeypatch, tmp_path):
        code, _ = _run_cli(
            monkeypatch, '--n', '20', '--seed', 's',
            '--out', str(tmp_path / 'corpus_manifest.json'),
        )
        assert isinstance(code, int) and not isinstance(code, bool)


# ===========================================================================
# Live smoke — ONE read-only query against the real dark_factory graph
# ===========================================================================
#
# Marked @pytest.mark.integration PER-TEST, never as a module-level pytestmark:
# fused-memory/pyproject.toml sets `addopts = -m 'not integration'`, so a
# module-wide marker would deselect every offline test above and this file
# would silently stop running in the default lane.

# Drift-tolerant floors, never equalities. The store grows continuously — the
# PRD measured ~2,635 episodes at authoring and this task measured 2,770 two
# months later — so `assert len(population) == 2770` would be a guaranteed
# future red that says nothing about correctness.
LIVE_MIN_POPULATION = 2000
LIVE_MIN_MONTH_BUCKETS = 5
LIVE_WINDOW_STARTS_BY = '2026-04-06'

# The payload axis IS asserted by equality, unlike the counts. A new kind
# appearing is not drift to tolerate: it adds a row to the cross-tab, changes
# every stratum denominator, and means delta's census needs re-measuring
# before the next corpus build. A red here is information, not noise.
LIVE_PAYLOAD_KINDS = {
    'decisions_and_rationale',
    'temporal_facts',
    'entities_and_relations',
    'procedural_knowledge',
}


class _RecordingLiveGraph:
    """Wraps a REAL FalkorDB graph handle: forwards ``ro_query``, trips on the rest.

    Lets the live smoke MEASURE the read-only claim on the real path rather
    than infer it from the source. ``__getattr__`` raises for every other
    attribute, so any write-capable call — ``query``, ``delete``,
    ``create_node_range_index`` — fails the test instead of reaching the store.
    """

    def __init__(self, graph: Any):
        self._graph = graph
        self.commands: list[str] = []

    async def ro_query(self, cypher, *args, **kwargs):
        self.commands.append('ro_query')
        return await self._graph.ro_query(cypher, *args, **kwargs)

    def __getattr__(self, name: str):
        raise _ReadOnlyViolation(
            f'the live smoke reached a non-read-only path on the real graph: {name!r}'
        )


@pytest.mark.integration
@falkor_skipif()
@pytest.mark.timeout(120)
def test_live_dark_factory_population_smoke(monkeypatch):
    """One GRAPH.RO_QUERY against the live graph; nothing is written.

    Asserts the premises the whole builder rests on are still true of the real
    store — the two stratification dimensions actually discriminate, and the
    population is big enough for a 150-300 corpus to mean anything — plus the
    two hazards, measured rather than promised: only the read-only command path
    was used, and no graphiti FalkorDriver was constructed (its ``__init__``
    fire-and-forgets ``build_indices_and_constraints()``, which would create
    indices on dark_factory and destroy the no-index evidence owned by
    docs/prds/falkordb-index-provisioning.md).
    """
    from graphiti_core.driver import falkordb_driver  # noqa: PLC0415

    def _boom(*args, **kwargs):
        raise AssertionError(
            'the live smoke constructed a graphiti FalkorDriver — this fires '
            'build_indices_and_constraints() against dark_factory'
        )

    monkeypatch.setattr(falkordb_driver.FalkorDriver, '__init__', _boom)

    recorders: list[_RecordingLiveGraph] = []

    async def _run() -> list:
        reader = _mod.EpisodeReader(graph_name=_mod.DEFAULT_GRAPH)
        # Resolve through the reader's OWN lazy client construction, so the
        # host/port env wiring is exercised live rather than bypassed by an
        # injected handle; then wrap it to record the command path.
        live: Any = reader._resolve_graph()
        recorder = _RecordingLiveGraph(live)
        recorders.append(recorder)
        reader._graph = recorder
        try:
            return await reader.fetch_population()
        finally:
            await live.client.aclose()

    population = asyncio.run(_run())

    assert len(population) > LIVE_MIN_POPULATION, (
        f'only {len(population)} episodes — too few for a 150-300 corpus to be a '
        f'sample rather than a census'
    )

    kinds = {_mod.payload_kind(r.source_description) for r in population}
    assert kinds == LIVE_PAYLOAD_KINDS, (
        f'the payload axis changed: {kinds ^ LIVE_PAYLOAD_KINDS} — re-measure the '
        f'census before building the next corpus'
    )

    months = {_mod.month_bucket(r.created_at) for r in population}
    assert len(months) >= LIVE_MIN_MONTH_BUCKETS, months

    window_start = min(r.created_at for r in population)
    assert window_start[:10] <= LIVE_WINDOW_STARTS_BY, window_start

    # The hazards, measured: exactly one query, on the read-only path only.
    assert [r.commands for r in recorders] == [['ro_query']]
