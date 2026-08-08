"""Tests for memory_eval_staleness_sweep.py — the E4 staleness sweep.

The script is loaded via importlib so it can be tested without sys.path
pollution — mirrors the pattern in test_memory_eval_retrieval_probe.py and
test_audit_duplicate_memories.py. The loader is invoked lazily (``_mod()``).

**Lane discipline.** Every test in this file except the single seeded
live-store test is free of network, Qdrant, OPENAI_API_KEY and any live
store: the sweep's three metric families are pure functions over
already-fetched records, precisely so the merge lane (which runs under
``addopts = -m 'not integration'``) covers all of them. The one integration
test carries ``@pytest.mark.integration`` PER-TEST rather than as a module
``pytestmark``, so marking it never deselects the pure tests here. Note also
``asyncio_mode = "strict"``: every async test needs an explicit
``@pytest.mark.asyncio``.

**No thresholds.** Per the plan's G6 decision, no test in this file asserts a
rate, tolerance, bound or pass/fail limit. Assertions are boolean flips on
named item_keys and exact counts on seeded fixtures.
"""
from __future__ import annotations

import functools
import importlib.util
import types
from pathlib import Path
from typing import Any

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'memory_eval_staleness_sweep.py'


def _load_module() -> types.ModuleType:
    """Load memory_eval_staleness_sweep.py from its file path.

    The module is registered in sys.modules under its name so that
    @dataclass and other reflection-based decorators work correctly
    (they call sys.modules.get(cls.__module__)).
    """
    import sys  # noqa: PLC0415

    mod_name = 'memory_eval_staleness_sweep'
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


@functools.cache
def _mod() -> types.ModuleType:
    return _load_module()


def _source() -> str:
    """The script's own text, for the INV-5 single-parser assertions."""
    return SCRIPT_PATH.read_text(encoding='utf-8')


class TestPinnedVocabulary:
    """The metric ids and eval_id are a contract with leaf α, not free choice."""

    def test_the_eval_id_is_this_leafs_own(self):
        m = _mod()
        assert m.EVAL_ID == 'e4-staleness-sweep'
        # Sharing beta's eval_id would make write_metric_series clobber beta's
        # artifact on every scheduled run (they share a stamp by design).
        assert m.EVAL_ID != 'e1-retrieval-health'

    def test_the_reserved_metric_ids_are_spelled_exactly(self):
        m = _mod()
        assert m.METRIC_SUPERSEDED_STILL_SURFACING == 'superseded-still-surfacing'
        assert m.METRIC_DANGLING_POINTERS == 'dangling-pointers'
        assert m.METRIC_SUCCESSOR_POINTER_PRESENT == 'successor-pointer-present'
        assert m.METRIC_TASK_TERMINAL_STALENESS == 'task-terminal-staleness'

    def test_all_three_pointer_keys_are_swept(self):
        m = _mod()
        assert m.POINTER_KEYS == ('supersedes', 'parent_id', 'corrects')


# ---------------------------------------------------------------------------
# Record builders (in-memory; no store needed)
# ---------------------------------------------------------------------------

UUID_A = '0b746438-6ce8-435c-885c-b3ac82666764'
UUID_B = '9f2c1d5e-1111-4a2b-8c3d-4e5f60718293'
UUID_C = 'c3d4e5f6-2222-4b3c-9d4e-5f6071829304'


def _record(record_id: str = 'rec-1', content: str = 'a memory', **metadata) -> dict:
    """The ``{'id', 'content', 'metadata'}`` shape the fetch band normalises to."""
    return {'id': record_id, 'content': content, 'metadata': dict(metadata)}


class TestPointerTargets:
    """Every (source, key, target) edge a record's metadata declares."""

    def test_a_uuid_string_yields_one_target_not_thirty_six(self):
        """The 3112 char-iteration regression pin.

        A bare ``for target in value`` over a 36-character UUID *string*
        iterates it into 36 single characters, none of which resolve —
        manufacturing a systematic false dangling-pointer report. This is the
        exact bug ``normalize_supersedes`` exists to prevent, and the reason
        this leaf may not carry a second parser.
        """
        m = _mod()
        refs = m.pointer_targets(_record(supersedes=UUID_A))
        assert len(refs) == 1
        assert refs[0].target == UUID_A
        assert refs[0].key == 'supersedes'
        assert refs[0].source_id == 'rec-1'

    def test_a_list_valued_pointer_yields_one_ref_per_member(self):
        m = _mod()
        refs = m.pointer_targets(_record(supersedes=[UUID_A, UUID_B]))
        assert [r.target for r in refs] == [UUID_A, UUID_B]

    def test_absent_and_none_yield_nothing(self):
        m = _mod()
        assert m.pointer_targets(_record()) == []
        assert m.pointer_targets(_record(supersedes=None, parent_id=None, corrects=None)) == []

    @pytest.mark.parametrize('key', ['parent_id', 'corrects'])
    def test_the_other_pointer_keys_get_the_same_tolerance(self, key):
        """Same None/scalar/list ambiguity, same normalizer (INV-5)."""
        m = _mod()
        # Typed locals, not inline **{...}: a bare dict literal splatted into
        # **metadata reads to pyright as a possible bind of record_id/content.
        absent: dict[str, Any] = {key: None}
        one: dict[str, Any] = {key: UUID_A}
        many: dict[str, Any] = {key: [UUID_A, UUID_B]}
        assert m.pointer_targets(_record(**absent)) == []
        scalar = m.pointer_targets(_record(**one))
        assert [r.target for r in scalar] == [UUID_A]
        assert [r.key for r in scalar] == [key]
        listed = m.pointer_targets(_record(**many))
        assert [r.target for r in listed] == [UUID_A, UUID_B]

    def test_a_malformed_member_is_retained_not_dropped(self):
        """``normalize_supersedes`` never drops a member; neither may this.

        A dropped member is a silently discarded supersession edge — the
        census would report a clean sweep over a corpus it never looked at.
        """
        m = _mod()
        refs = m.pointer_targets(_record(supersedes=[UUID_A, 'deadbeef', 42, None]))
        assert [r.target for r in refs] == [UUID_A, 'deadbeef', 42, None]
        malformed = m.malformed_pointer_refs(refs)
        assert [r.target for r in malformed] == ['deadbeef', 42, None]

    def test_ordering_is_deterministic_across_pointer_keys(self):
        m = _mod()
        refs = m.pointer_targets(
            _record(corrects=UUID_C, parent_id=UUID_B, supersedes=UUID_A),
        )
        assert [r.key for r in refs] == list(m.POINTER_KEYS)
        # And the same record built with the keys inserted in another order
        # produces the same sequence: metadata dict order must not leak into
        # a per-run artifact leaf alpha trends.
        reordered = m.pointer_targets(
            _record(supersedes=UUID_A, corrects=UUID_C, parent_id=UUID_B),
        )
        assert refs == reordered

    def test_the_source_content_rides_along_for_the_tripwire_key(self):
        m = _mod()
        refs = m.pointer_targets(_record(content='the successor says X', supersedes=UUID_A))
        assert refs[0].source_content == 'the successor says X'

    def test_the_script_imports_the_one_sanctioned_parser(self):
        """INV-5 / D7, and this task's delivered_checks grep."""
        source = _source()
        assert 'normalize_supersedes' in source
        assert 'from fused_memory.memory_metadata import normalize_supersedes' in source

    def test_the_script_defines_no_second_pointer_parser(self):
        """No local re-implementation, and exactly one import site.

        Asserted on code shapes rather than on prose: the module docstring
        NAMES the 3112 failure mode on purpose, so a banned-substring sweep
        over the whole file would constrain wording rather than behaviour.
        """
        source = _source()
        assert 'def normalize_supersedes' not in source
        assert source.count('import normalize_supersedes') == 1


class TestDanglingCensus:
    """Resolved vs unresolved, per key, with the unresolved targets NAMED."""

    def test_resolved_and_unresolved_totals(self):
        m = _mod()
        refs = m.pointer_targets(_record(supersedes=[UUID_A, UUID_B]))
        census = m.dangling_census(refs, {UUID_A: True, UUID_B: False})
        assert census.examined == 2
        assert census.resolved == 1
        assert census.unresolved == 1

    def test_the_breakdown_is_per_pointer_key(self):
        m = _mod()
        refs = m.pointer_targets(_record(supersedes=UUID_A, corrects=UUID_B))
        census = m.dangling_census(refs, {UUID_A: True, UUID_B: False})
        assert census.by_key['supersedes'] == {'examined': 1, 'resolved': 1, 'unresolved': 0}
        assert census.by_key['corrects'] == {'examined': 1, 'resolved': 0, 'unresolved': 1}
        # A key with no live population is absent, not a fabricated zero row.
        assert 'parent_id' not in census.by_key

    def test_the_unresolved_targets_are_reported_not_just_counted(self):
        """A bare count says something dangles, not which pointer to go look at."""
        m = _mod()
        refs = m.pointer_targets(_record(supersedes=[UUID_A, UUID_B]))
        census = m.dangling_census(refs, {UUID_A: True, UUID_B: False})
        assert [r.target for r in census.unresolved_refs] == [UUID_B]
        assert census.unresolved_refs[0].source_id == 'rec-1'
        assert census.unresolved_refs[0].key == 'supersedes'

    def test_a_shared_target_is_examined_once_per_citing_source(self):
        """`examined` counts POINTERS, and each source keeps its own attribution."""
        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-1', supersedes=UUID_A)),
            *m.pointer_targets(_record('rec-2', supersedes=UUID_A)),
        ]
        census = m.dangling_census(refs, {UUID_A: False})
        assert census.examined == 2
        assert census.unresolved == 2
        assert sorted(r.source_id for r in census.unresolved_refs) == ['rec-1', 'rec-2']
        assert m.unique_pointer_targets(refs) == [UUID_A]

    def test_a_target_missing_from_the_resolution_map_is_unresolved(self):
        """The map is the caller's ground-truth read; an absent key is a miss.

        Never a silent 'assume fine': a target the resolver never reached is
        exactly the case a self-confirming census would paper over.
        """
        m = _mod()
        refs = m.pointer_targets(_record(supersedes=UUID_A))
        census = m.dangling_census(refs, {})
        assert census.unresolved == 1

    def test_an_empty_ref_set_is_zero_exposure(self):
        """Which build_series then declines to emit as a metric at all."""
        m = _mod()
        census = m.dangling_census([], {})
        assert census.examined == 0
        assert census.resolved == 0
        assert census.unresolved == 0
        assert census.by_key == {}
        assert census.unresolved_refs == []


class TestSuccessorPointerItems:
    """The tripwire covers the supersedes edge ONLY, keyed by content."""

    def test_one_item_per_supersedes_edge_and_passed_tracks_resolution(self):
        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-1', 'successor one', supersedes=UUID_A)),
            *m.pointer_targets(_record('rec-2', 'successor two', supersedes=UUID_B)),
        ]
        items = m.successor_pointer_items(refs, {UUID_A: True, UUID_B: False})
        assert len(items) == 2
        assert {item.passed for item in items} == {True, False}

    def test_the_item_key_is_content_derived_not_a_raw_source_uuid(self):
        """D5 UUID rot: alpha grandfathers by item_key.

        The same content under a rotated source id must keep its key, or a
        re-consolidation would read to the evaluator as a brand-new failure.
        """
        m = _mod()
        first = m.successor_pointer_items(
            m.pointer_targets(_record(UUID_C, 'the same words', supersedes=UUID_A)),
            {UUID_A: False},
        )
        rotated = m.successor_pointer_items(
            m.pointer_targets(_record(UUID_B, 'the same words', supersedes=UUID_A)),
            {UUID_A: False},
        )
        assert first[0].item_key == rotated[0].item_key
        assert UUID_C not in first[0].item_key
        assert first[0].item_key.startswith(m.TRIPWIRE_ITEM_PREFIX)

    def test_the_target_discriminates_two_edges_from_one_source(self):
        m = _mod()
        items = m.successor_pointer_items(
            m.pointer_targets(_record('rec-1', 'one successor', supersedes=[UUID_A, UUID_B])),
            {UUID_A: True, UUID_B: False},
        )
        assert len({item.item_key for item in items}) == 2

    def test_item_keys_are_unique_and_sorted(self):
        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-2', 'zeta content', supersedes=UUID_B)),
            *m.pointer_targets(_record('rec-1', 'alpha content', supersedes=UUID_A)),
        ]
        items = m.successor_pointer_items(refs, {UUID_A: True, UUID_B: True})
        keys = [item.item_key for item in items]
        assert keys == sorted(keys)
        assert len(set(keys)) == len(keys)

    def test_parent_id_and_corrects_produce_no_tripwire_items(self):
        """They are counted by dangling-pointers only.

        Those targets have no stable per-item identity to grandfather on, so a
        tripwire over them could not express alpha's ratchet.
        """
        m = _mod()
        refs = m.pointer_targets(_record(parent_id=UUID_A, corrects=UUID_B))
        assert m.successor_pointer_items(refs, {UUID_A: False, UUID_B: False}) == []

    def test_no_supersedes_edges_yields_no_items(self):
        m = _mod()
        assert m.successor_pointer_items([], {}) == []


class TestSupersededSurfacing:
    """Both-present-only exposure over corpus-discovered (successor, superseded) pairs."""

    def test_a_pair_with_only_one_member_present_contributes_nothing(self):
        """Not to the count AND not to the exposure.

        An absent successor is a findability question; charging it here as
        well would double-weight one defect against two metrics.
        """
        m = _mod()
        obs = m.superseded_surfacing([(UUID_A, UUID_B)], [UUID_A, UUID_C])
        assert obs.pairs_comparable == 0
        assert obs.still_surfacing == 0
        assert obs.inversions == ()
        assert m.superseded_surfacing([(UUID_A, UUID_B)], [UUID_B]).pairs_comparable == 0

    def test_a_superseded_entry_ranked_above_its_successor_is_counted(self):
        m = _mod()
        obs = m.superseded_surfacing([(UUID_A, UUID_B)], [UUID_B, UUID_A])
        assert obs.pairs_comparable == 1
        assert obs.still_surfacing == 1
        assert len(obs.inversions) == 1

    def test_a_superseded_entry_ranked_below_is_comparable_but_not_counted(self):
        m = _mod()
        obs = m.superseded_surfacing([(UUID_A, UUID_B)], [UUID_A, UUID_B])
        assert obs.pairs_comparable == 1
        assert obs.still_surfacing == 0
        assert obs.inversions == ()

    def test_a_superseded_entry_appearing_at_all_is_in_the_detail_records(self):
        m = _mod()
        obs = m.superseded_surfacing([(UUID_A, UUID_B)], [UUID_A, UUID_B])
        assert len(obs.records) == 1
        record = obs.records[0]
        assert record.successor_id == UUID_A
        assert record.superseded_id == UUID_B
        assert record.successor_rank == 1
        assert record.superseded_rank == 2

    def test_rank_is_list_position_so_two_runs_agree(self):
        """Equal scores resolve by returned order, never by an unstable re-sort.

        A tie that flapped between runs would read to leaf alpha as a real
        regression.
        """
        m = _mod()
        ranked = [UUID_B, UUID_A, UUID_C]
        assert m.rank_index(ranked) == {UUID_B: 1, UUID_A: 2, UUID_C: 3}
        first = m.superseded_surfacing([(UUID_A, UUID_B)], ranked)
        second = m.superseded_surfacing([(UUID_A, UUID_B)], list(ranked))
        assert first == second

    def test_a_repeated_id_keeps_its_best_rank(self):
        m = _mod()
        assert m.rank_index([UUID_A, UUID_B, UUID_A]) == {UUID_A: 1, UUID_B: 2}

    def test_several_pairs_against_one_ranked_list(self):
        m = _mod()
        obs = m.superseded_surfacing(
            [(UUID_A, UUID_B), (UUID_C, UUID_A)], [UUID_B, UUID_A, UUID_C],
        )
        assert obs.pairs_comparable == 2
        assert obs.still_surfacing == 2

    def test_an_empty_pair_set_is_zero_exposure(self):
        m = _mod()
        obs = m.superseded_surfacing([], [UUID_A])
        assert obs.pairs_comparable == 0
        assert obs.records == ()


class TestTaskTerminalStaleness:
    """Entries asserting LIVE task state for a task that has gone terminal."""

    def test_a_live_status_assertion_about_a_terminal_task_is_reported(self):
        m = _mod()
        records = [_record('rec-1', 'Task 4802 status=in-progress, claimant_run_id=abc')]
        obs = m.terminal_staleness(records, {'4802': 'done'})
        assert obs.entries_referencing_terminal == 1
        assert obs.stale == 1
        assert obs.records[0].record_id == 'rec-1'
        assert obs.records[0].task_id == '4802'
        assert obs.records[0].status == 'done'

    def test_the_same_assertion_about_a_non_terminal_task_is_not_reported(self):
        m = _mod()
        records = [_record('rec-1', 'Task 4802 status=in-progress, claimant_run_id=abc')]
        obs = m.terminal_staleness(records, {'1234': 'done'})
        assert obs.entries_referencing_terminal == 0
        assert obs.stale == 0
        assert obs.records == ()

    def test_a_timestamped_point_in_time_framing_is_exempt(self):
        """Even for a terminal task: it was true when it was checked."""
        m = _mod()
        records = [
            _record('rec-1', 'Task 4802 liveness check performed at 2026-08-01: status=in-progress'),
        ]
        obs = m.terminal_staleness(records, {'4802': 'done'})
        assert obs.entries_referencing_terminal == 1  # still exposed
        assert obs.stale == 0                          # but not counted

    def test_task_ids_come_from_content_and_from_metadata(self):
        m = _mod()
        assert m.referenced_task_ids(_record('r', 'see task 4802 and df 91 and #7')) == {
            '4802', '91', '7',
        }
        assert m.referenced_task_ids(_record('r', 'no refs', task_id=4802)) == {'4802'}
        assert m.referenced_task_ids(_record('r', 'task 12', task_id='4802')) == {'12', '4802'}
        # A bare number is not a task reference.
        assert m.referenced_task_ids(_record('r', 'there were 4802 records')) == set()

    def test_exposure_is_entries_referencing_a_terminal_task_not_the_corpus(self):
        m = _mod()
        records = [
            _record('rec-1', 'Task 4802 status=in-progress'),
            _record('rec-2', 'Task 4802 was a good idea'),
            _record('rec-3', 'nothing to do with tasks at all'),
            _record('rec-4', 'Task 9999 status=in-progress'),
        ]
        obs = m.terminal_staleness(records, {'4802': 'cancelled'})
        assert obs.entries_referencing_terminal == 2
        assert obs.stale == 1

    def test_one_entry_naming_two_terminal_tasks_is_exposed_once(self):
        m = _mod()
        records = [_record('rec-1', 'Task 4802 and task 4803 status=in-progress')]
        obs = m.terminal_staleness(records, {'4802': 'done', '4803': 'done'})
        assert obs.entries_referencing_terminal == 1
        assert obs.stale == 1
        assert sorted(r.task_id for r in obs.records) == ['4802', '4803']

    def test_a_bare_set_of_terminal_ids_works_too(self):
        m = _mod()
        obs = m.terminal_staleness(
            [_record('rec-1', 'Task 4802 status=in-progress')], {'4802'},
        )
        assert obs.stale == 1
        assert obs.records[0].status == 'terminal'

    def test_no_terminal_ids_is_zero_exposure(self):
        m = _mod()
        obs = m.terminal_staleness([_record('rec-1', 'Task 4802 status=in-progress')], set())
        assert obs.entries_referencing_terminal == 0
        assert obs.records == ()

    def test_the_live_state_judgement_is_delegated_to_task_filter(self):
        """INV-5, and the helper carries the mandatory cheap-prefilter ordering.

        ``POINT_IN_TIME_CHECK_RE``'s two lookaheads under ``re.DOTALL`` are
        quadratic in content length; the helper prefilters with the
        lookahead-free ``LIVE_TASK_STATUS_RE``, which is what keeps a
        corpus-scale scan tractable. Re-deriving either regex here would drop
        that ordering silently.
        """
        source = _source()
        assert 'frames_live_task_status_as_current_fact' in source
        assert 'LIVE_TASK_STATUS_RE = ' not in source
        assert 'POINT_IN_TIME_CHECK_RE = ' not in source

    def test_the_helper_is_actually_called(self, monkeypatch):
        m = _mod()
        from fused_memory.reconciliation import task_filter  # noqa: PLC0415

        calls: list[str] = []

        def _spy(text: str) -> bool:
            calls.append(text)
            return True

        monkeypatch.setattr(task_filter, 'frames_live_task_status_as_current_fact', _spy)
        obs = m.terminal_staleness([_record('rec-1', 'Task 4802 whatever')], {'4802': 'done'})
        assert calls == ['Task 4802 whatever']
        assert obs.stale == 1


STAMP = '20260808T120000Z'


def _full_inputs():
    """One of every family, all with non-zero exposure."""
    m = _mod()
    refs = [
        *m.pointer_targets(_record('rec-1', 'successor one', supersedes=UUID_B)),
        *m.pointer_targets(_record('rec-2', 'a correction', corrects=UUID_C)),
    ]
    resolution = {UUID_B: True, UUID_C: False}
    return {
        'census': m.dangling_census(refs, resolution),
        'tripwire_items': m.successor_pointer_items(refs, resolution),
        'surfacing': m.superseded_surfacing([(UUID_A, UUID_B)], [UUID_B, UUID_A]),
        'staleness': m.terminal_staleness(
            [_record('rec-3', 'Task 4802 status=in-progress')], {'4802': 'done'},
        ),
        'corpus_counts': {'procedural_knowledge': 12},
        'project_id': 'dark_factory',
        'stamp': STAMP,
    }


def _ids(series) -> set[str]:
    return {metric.metric_id for metric in series.metrics}


def _metric(series, metric_id):
    for metric in series.metrics:
        if metric.metric_id == metric_id:
            return metric
    raise AssertionError(f'series carries no {metric_id!r}: {sorted(_ids(series))}')


class TestBuildSeries:
    """The artifact this leaf owns — and the metrics it must NOT emit."""

    def test_the_series_identifies_this_leaf(self):
        from shared.memory_eval_metrics import SCHEMA_VERSION  # noqa: PLC0415

        series = _mod().build_series(**_full_inputs())
        assert series.eval_id == 'e4-staleness-sweep'
        assert series.schema_version == SCHEMA_VERSION
        assert series.run_stamp == STAMP
        assert series.corpus.project_id == 'dark_factory'

    def test_it_emits_exactly_the_four_metrics_this_leaf_owns(self):
        m = _mod()
        series = m.build_series(**_full_inputs())
        assert _ids(series) == set(m.pinned_metric_ids())
        assert _ids(series) == {
            'superseded-still-surfacing',
            'dangling-pointers',
            'successor-pointer-present',
            'task-terminal-staleness',
        }

    def test_it_never_emits_beta_metrics(self):
        series = _mod().build_series(**_full_inputs())
        assert 'superseded-above-successor' not in _ids(series)
        assert 'canonical-in-top-5' not in _ids(series)
        assert 'claim-recall' not in _ids(series)
        assert 'contamination-share' not in _ids(series)

    def test_kinds_and_directions_per_family(self):
        m = _mod()
        series = m.build_series(**_full_inputs())
        for metric_id in (
            'superseded-still-surfacing', 'dangling-pointers', 'task-terminal-staleness',
        ):
            metric = _metric(series, metric_id)
            assert metric.kind == 'count'
            assert metric.direction == 'higher_is_worse'
            assert metric.denominator is None
            assert metric.items is None
        tripwire = _metric(series, 'successor-pointer-present')
        assert tripwire.kind == 'tripwire'
        assert tripwire.direction is None
        assert tripwire.denominator is None

    def test_the_tripwire_value_is_its_failing_item_count(self):
        m = _mod()
        inputs = _full_inputs()
        tripwire = _metric(m.build_series(**inputs), 'successor-pointer-present')
        assert tripwire.n == len(tripwire.items)
        assert tripwire.value == sum(1 for i in tripwire.items if not i.passed)

    def test_the_count_values_and_exposures(self):
        m = _mod()
        inputs = _full_inputs()
        series = m.build_series(**inputs)
        dangling = _metric(series, 'dangling-pointers')
        assert dangling.value == inputs['census'].unresolved
        assert dangling.n == inputs['census'].examined
        surfacing = _metric(series, 'superseded-still-surfacing')
        assert surfacing.value == inputs['surfacing'].still_surfacing
        assert surfacing.n == inputs['surfacing'].pairs_comparable
        staleness = _metric(series, 'task-terminal-staleness')
        assert staleness.value == inputs['staleness'].stale
        assert staleness.n == inputs['staleness'].entries_referencing_terminal

    def test_a_zero_exposure_family_is_absent_not_zero(self):
        """A fabricated 0/0 datapoint entering alpha's baseline window is worse
        than a gap in it — and parent_id makes this live, not hypothetical."""
        m = _mod()
        inputs = _full_inputs()
        inputs['census'] = m.dangling_census([], {})
        inputs['tripwire_items'] = []
        inputs['surfacing'] = m.EMPTY_SURFACING
        inputs['staleness'] = m.terminal_staleness([], set())
        series = m.build_series(**inputs)
        assert series.metrics == []
        assert m.metric_families_not_measured(series) == list(m.pinned_metric_ids())

    def test_each_family_can_be_absent_independently(self):
        m = _mod()
        inputs = _full_inputs()
        inputs['surfacing'] = m.EMPTY_SURFACING
        series = m.build_series(**inputs)
        assert 'superseded-still-surfacing' not in _ids(series)
        assert 'dangling-pointers' in _ids(series)
        assert m.metric_families_not_measured(series) == ['superseded-still-surfacing']

    def test_details_path_is_a_filename_never_an_absolute_path(self):
        """The artifact directory gets copied and served; an absolute path
        from this machine would be a dangling pointer there."""
        series = _mod().build_series(**_full_inputs())
        for metric in series.metrics:
            assert metric.details_path == f'report-{STAMP}.txt'
            assert '/' not in metric.details_path

    def test_corpus_counts_carry_the_scan_disclosure(self):
        m = _mod()
        inputs = _full_inputs()
        inputs['corpus_counts'] = {
            'procedural_knowledge_scanned': 12,
            'procedural_knowledge_truncated': 1,
        }
        counts = m.build_series(**inputs).corpus.counts
        assert counts['procedural_knowledge_scanned'] == 12
        assert counts['procedural_knowledge_truncated'] == 1
        # And the runner's own disclosures ride along in the same mapping.
        assert counts['pointer_refs_malformed'] == 0
        assert counts['pointers_supersedes_examined'] == 1

    def test_a_caller_key_colliding_with_a_computed_disclosure_raises(self):
        """Silently overwriting either one would hide a narrowing."""
        m = _mod()
        inputs = _full_inputs()
        inputs['corpus_counts'] = {'pointer_refs_malformed': 999}
        with pytest.raises(ValueError, match='collides'):
            m.build_series(**inputs)

    def test_the_series_round_trips_and_passes_the_real_validator(self):
        import json  # noqa: PLC0415

        from shared.memory_eval_metrics import (  # noqa: PLC0415
            parse_metric_series,
            serialize_metric_series,
            validate_metric_series,
        )

        series = _mod().build_series(**_full_inputs())
        validate_metric_series(series)
        assert parse_metric_series(json.loads(serialize_metric_series(series))) == series


# ---------------------------------------------------------------------------
# The async I/O band
#
# Thin by design: fetch, normalise, hand off to the pure functions above. What
# is asserted here is the CALL SHAPE (which primitive, how many times, with
# what filter) and the disclosure of anything the fetch narrowed — not any
# metric arithmetic, which is already covered above without a store.
# ---------------------------------------------------------------------------

def _raw_point(point_id: str, payload: dict) -> dict:
    """One record as ``scroll_by_metadata`` returns it."""
    return {'id': point_id, 'created_at': '2026-08-01T00:00:00+00:00', 'metadata': payload}


def _mock_memory(*, scroll_return=None, by_id=None):
    """A MagicMock MemoryService whose two read primitives are AsyncMocks."""
    from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

    memory = MagicMock()
    memory.mem0 = MagicMock()
    memory.mem0.scroll_by_metadata = AsyncMock(return_value=scroll_return or [])
    memory.get_memory_by_id = AsyncMock(side_effect=lambda _p, mid: (by_id or {}).get(mid))
    return memory


@pytest.mark.asyncio
class TestFetchPointerRecords:
    """One capped scroll per Mem0 category, with the cap disclosed."""

    async def test_one_scroll_per_category_with_the_category_filter(self):
        m = _mod()
        memory = _mock_memory()

        _records, stats = await m.fetch_pointer_records(
            memory, 'dark_factory', categories=('procedural_knowledge', 'preferences_and_norms'),
            scan_limit=1234,
        )

        # There is no single "scan everything" call to make: the primitive
        # builds an AND-equality filter and REJECTS an empty filter dict, so
        # the per-category loop is the enumeration, not a missed optimisation.
        assert memory.mem0.scroll_by_metadata.await_count == 2
        filters = [call.args[1] for call in memory.mem0.scroll_by_metadata.await_args_list]
        assert filters == [
            {'category': 'procedural_knowledge'},
            {'category': 'preferences_and_norms'},
        ]
        for call in memory.mem0.scroll_by_metadata.await_args_list:
            assert call.args[0].project_id == 'dark_factory'
            assert call.kwargs.get('limit') == 1234
        assert set(stats) == {'procedural_knowledge', 'preferences_and_norms'}

    async def test_records_are_normalised_to_the_pure_bands_shape(self):
        m = _mod()
        raw = [_raw_point('m1', {
            'data': 'the successor text',
            'category': 'procedural_knowledge',
            'supersedes': UUID_B,
        })]
        memory = _mock_memory(scroll_return=raw)

        records, _stats = await m.fetch_pointer_records(
            memory, 'dark_factory', categories=('procedural_knowledge',), scan_limit=10,
        )

        assert len(records) == 1
        # The exact {'id','content','metadata'} shape _record() builds, so the
        # pure functions above are the SAME code path a live run drives.
        assert records[0]['id'] == 'm1'
        assert records[0]['content'] == 'the successor text'
        assert records[0]['metadata']['supersedes'] == UUID_B
        assert m.pointer_targets(records[0]) == [
            m.PointerRef(
                source_id='m1', key='supersedes', target=UUID_B,
                source_content='the successor text',
            ),
        ]

    async def test_a_firing_cap_is_disclosed_per_category(self):
        m = _mod()
        raw = [_raw_point(f'm{i}', {'data': f'text {i}'}) for i in range(3)]
        memory = _mock_memory(scroll_return=raw)

        _records, stats = await m.fetch_pointer_records(
            memory, 'dark_factory', categories=('procedural_knowledge',), scan_limit=3,
        )

        # scanned == scan_limit means the scroll may have stopped short. A cap
        # firing on one category must never be readable as a clean sweep of
        # the whole corpus, so it is counted per category and reported.
        assert stats['procedural_knowledge'] == {'scanned': 3, 'truncated': 1}

    async def test_an_unfilled_scan_is_disclosed_as_untruncated(self):
        m = _mod()
        memory = _mock_memory(scroll_return=[_raw_point('m1', {'data': 'x'})])

        _records, stats = await m.fetch_pointer_records(
            memory, 'dark_factory', categories=('procedural_knowledge',), scan_limit=50,
        )

        assert stats['procedural_knowledge'] == {'scanned': 1, 'truncated': 0}

    async def test_a_repeated_category_is_scrolled_once(self):
        m = _mod()
        memory = _mock_memory(scroll_return=[_raw_point('m1', {'data': 'x'})])

        records, stats = await m.fetch_pointer_records(
            memory, 'dark_factory',
            categories=('procedural_knowledge', 'procedural_knowledge'),
            scan_limit=10,
        )

        # Re-scrolling appends a second copy of every record under the SAME
        # id, which would double every pointer edge in the census.
        assert memory.mem0.scroll_by_metadata.await_count == 1
        assert [r['id'] for r in records] == ['m1']
        assert stats['procedural_knowledge']['scanned'] == 1

    async def test_a_scan_timeout_propagates(self):
        m = _mod()
        memory = _mock_memory()
        memory.mem0.scroll_by_metadata.side_effect = TimeoutError('qdrant read timed out')

        with pytest.raises(TimeoutError):
            await m.fetch_pointer_records(
                memory, 'dark_factory', categories=('procedural_knowledge',), scan_limit=10,
            )


@pytest.mark.asyncio
class TestResolvePointerTargets:
    """Every target is corroborated against the live store (INV-3)."""

    async def test_one_live_read_per_unique_target(self):
        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-1', 'one', supersedes=UUID_B)),
            *m.pointer_targets(_record('rec-2', 'two', corrects=UUID_B)),
            *m.pointer_targets(_record('rec-3', 'three', supersedes=UUID_C)),
        ]
        memory = _mock_memory(by_id={UUID_B: {'id': UUID_B, 'content': 'x', 'metadata': {}}})

        resolution = await m.resolve_pointer_targets(memory, 'dark_factory', refs)

        # A target cited by several sources costs ONE read, but the read is
        # never skipped: resolving against the already-scrolled snapshot would
        # make the census self-confirming, since the scroll is capped and
        # category-scoped and a target outside it exists but is not in hand.
        assert memory.get_memory_by_id.await_count == 2
        assert [call.args[1] for call in memory.get_memory_by_id.await_args_list] == [
            UUID_B, UUID_C,
        ]
        assert all(call.args[0] == 'dark_factory'
                   for call in memory.get_memory_by_id.await_args_list)
        assert resolution == {UUID_B: True, UUID_C: False}

    async def test_a_none_return_maps_to_unresolved(self):
        m = _mod()
        refs = m.pointer_targets(_record('rec-1', 'one', supersedes=UUID_A))
        memory = _mock_memory(by_id={})

        assert await m.resolve_pointer_targets(memory, 'dark_factory', refs) == {UUID_A: False}

    async def test_a_resolve_timeout_is_never_reported_as_a_dangling_pointer(self):
        m = _mod()
        refs = m.pointer_targets(_record('rec-1', 'one', supersedes=UUID_A))
        memory = _mock_memory()
        memory.get_memory_by_id.side_effect = TimeoutError('qdrant read timed out')

        # get_memory_by_id propagates TimeoutError precisely so "absent" and
        # "backend blipped" stay distinguishable. Folding it into the
        # unresolved count would fabricate a defect and fire an alpha alarm on
        # an infrastructure blip.
        with pytest.raises(TimeoutError):
            await m.resolve_pointer_targets(memory, 'dark_factory', refs)

    async def test_a_malformed_target_costs_no_read_and_is_not_resolved(self):
        m = _mod()
        refs = m.pointer_targets(_record('rec-1', 'one', supersedes=['not-a-uuid', 7]))
        memory = _mock_memory()

        resolution = await m.resolve_pointer_targets(memory, 'dark_factory', refs)

        # A non-id-shaped value cannot be handed to a point-id read at all --
        # Qdrant's retrieve() rejects it, and because retrieve takes a LIST of
        # ids one bad pointer would fail the read and take down the very sweep
        # that exists to report it. It is not thereby forgiven: dangling_census
        # still counts it unresolved off its absence from this map.
        assert memory.get_memory_by_id.await_count == 0
        assert resolution == {}
        census = m.dangling_census(refs, resolution)
        assert census.unresolved == 2

    async def test_the_read_plan_and_the_disclosure_share_one_predicate(self):
        m = _mod()
        refs = m.pointer_targets(_record('rec-1', 'one', supersedes=['not-a-uuid', UUID_A]))

        # Two spellings of "is this target readable?" drift, and the drift is
        # what sends an unreadable id to Qdrant. Whatever one names malformed,
        # the other must decline to read -- asserted as a partition, not as
        # two independently restated shapes.
        readable = set(m.unique_pointer_targets(refs))
        malformed = {ref.target for ref in m.malformed_pointer_refs(refs)}
        assert readable == {UUID_A}
        assert malformed == {'not-a-uuid'}
        assert readable & malformed == set()
        assert readable | malformed == {ref.target for ref in refs}

    async def test_an_empty_ref_set_issues_no_reads(self):
        m = _mod()
        memory = _mock_memory()
        assert await m.resolve_pointer_targets(memory, 'dark_factory', []) == {}
        assert memory.get_memory_by_id.await_count == 0


class _Hit:
    """One search result, in the duck-typed shape MemoryResult presents."""

    def __init__(self, result_id: str):
        self.id = result_id


@pytest.mark.asyncio
class TestFetchSurfacingRanks:
    """One search per supersedes edge, derived from the successor's own text."""

    async def test_the_query_is_the_successors_content(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        m = _mod()
        refs = m.pointer_targets(_record('rec-1', 'the successor text', supersedes=UUID_B))
        memory = MagicMock()
        memory.search = AsyncMock(return_value=[_Hit('rec-1'), _Hit(UUID_B)])

        await m.fetch_surfacing_ranks(memory, 'dark_factory', refs, limit=7)

        # The family asks "when the corpus is asked about what the successor
        # says, does the entry it replaced come back above it?" -- only a query
        # derived from the successor's own text poses that question.
        memory.search.assert_awaited_once()
        assert memory.search.await_args.args[0] == 'the successor text'
        assert memory.search.await_args.kwargs == {'project_id': 'dark_factory', 'limit': 7}

    async def test_only_supersedes_edges_are_searched(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-1', 'a correction', corrects=UUID_B)),
            *m.pointer_targets(_record('rec-2', 'a child', parent_id=UUID_C)),
        ]
        memory = MagicMock()
        memory.search = AsyncMock(return_value=[])

        obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)

        # corrects/parent_id are counted by dangling-pointers only: neither
        # asserts that one entry REPLACED another, so neither has a surfacing
        # order to be wrong about.
        assert memory.search.await_count == 0
        assert obs.pairs_comparable == 0

    async def test_an_empty_successor_content_is_not_searched(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        m = _mod()
        refs = m.pointer_targets(_record('rec-1', '   ', supersedes=UUID_B))
        memory = MagicMock()
        memory.search = AsyncMock(return_value=[])

        obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)

        # An empty query's ranking is arbitrary; scoring against it would
        # manufacture inversions out of noise.
        assert memory.search.await_count == 0
        assert obs.pairs_comparable == 0

    async def test_each_edge_is_scored_against_its_own_query_and_folded(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-1', 'successor one', supersedes=UUID_B)),
            *m.pointer_targets(_record('rec-2', 'successor two', supersedes=UUID_C)),
        ]
        memory = MagicMock()
        memory.search = AsyncMock(side_effect=[
            [_Hit(UUID_B), _Hit('rec-1')],   # superseded ABOVE its successor
            [_Hit('rec-2'), _Hit(UUID_C)],   # superseded below: comparable, not counted
        ])

        obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)

        assert memory.search.await_count == 2
        assert obs.pairs_comparable == 2
        assert obs.still_surfacing == 1
        assert [r.successor_id for r in obs.inversions] == ['rec-1']

    async def test_a_half_present_pair_is_neither_counted_nor_exposed(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        m = _mod()
        refs = m.pointer_targets(_record('rec-1', 'successor one', supersedes=UUID_B))
        memory = MagicMock()
        memory.search = AsyncMock(return_value=[_Hit('rec-1')])

        obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)

        # Both-present-only exposure survives the round trip through the store:
        # an absent superseded entry carries no possibility of an inversion.
        assert obs.pairs_comparable == 0
        assert obs.still_surfacing == 0


class _BackendDouble:
    """A SqliteTaskBackend stand-in recording its lifecycle calls."""

    def __init__(self, taskmaster_config, *, statuses=None, fail=None):
        self.config = taskmaster_config
        self._statuses = statuses or {}
        self._fail = fail
        self.calls: list[str] = []

    async def start(self) -> None:
        self.calls.append('start')

    async def get_statuses(self, project_root: str) -> dict[str, str]:
        self.calls.append(f'get_statuses:{project_root}')
        if self._fail is not None:
            raise self._fail
        return self._statuses

    async def close(self) -> None:
        self.calls.append('close')


def _taskmaster_config(project_root: str = '/repo'):
    return types.SimpleNamespace(
        taskmaster=types.SimpleNamespace(project_root=project_root),
    )


def _install_backend_double(monkeypatch, double_holder, **kwargs):
    """Patch the module attribute the sweep's function-local import resolves."""
    from fused_memory.backends import sqlite_task_backend  # noqa: PLC0415

    def _factory(taskmaster_config):
        double = _BackendDouble(taskmaster_config, **kwargs)
        double_holder.append(double)
        return double

    monkeypatch.setattr(sqlite_task_backend, 'SqliteTaskBackend', _factory)


@pytest.mark.asyncio
class TestFetchTerminalTaskIds:
    """The task join: terminal means {done, cancelled}, and a skip says so."""

    async def test_it_starts_reads_and_closes_the_backend(self, monkeypatch):
        m = _mod()
        doubles: list[_BackendDouble] = []
        _install_backend_double(monkeypatch, doubles, statuses={'1': 'done'})

        await m.fetch_terminal_task_ids(_taskmaster_config('/repo'))

        assert doubles[0].calls == ['start', 'get_statuses:/repo', 'close']

    async def test_only_the_shared_terminal_statuses_are_selected(self, monkeypatch):
        from shared.task_statuses import TERMINAL  # noqa: PLC0415

        m = _mod()
        doubles: list[_BackendDouble] = []
        _install_backend_double(monkeypatch, doubles, statuses={
            '4802': 'done',
            '4803': 'cancelled',
            '4804': 'in-progress',
            '4805': 'deferred',
            '4806': 'pending',
        })

        join = await m.fetch_terminal_task_ids(_taskmaster_config())

        # deferred is NOT terminal: a deferred task can still be worked, so a
        # live-state assertion about one is not stale.
        # The STATUS rides along, not just the id -- it is what lets the
        # report say which terminal state the entry is contradicting.
        assert join.statuses == {'4802': 'done', '4803': 'cancelled'}
        assert 'deferred' not in TERMINAL
        assert join.skipped_reason is None

    async def test_ids_are_normalised_to_strings(self, monkeypatch):
        m = _mod()
        doubles: list[_BackendDouble] = []
        _install_backend_double(monkeypatch, doubles, statuses={4802: 'done'})

        join = await m.fetch_terminal_task_ids(_taskmaster_config())

        # referenced_task_ids() yields strings off TASK_REF_RE, so an int key
        # here would silently never match and the family would read as clean.
        assert join.statuses == {'4802': 'done'}

    async def test_the_backend_is_closed_even_when_the_read_raises(self, monkeypatch):
        m = _mod()
        doubles: list[_BackendDouble] = []
        _install_backend_double(monkeypatch, doubles, fail=RuntimeError('db locked'))

        join = await m.fetch_terminal_task_ids(_taskmaster_config())

        assert 'close' in doubles[0].calls
        assert join.statuses == {}
        assert join.skipped_reason is not None
        assert 'db locked' in join.skipped_reason

    async def test_an_unconfigured_taskmaster_is_a_named_skip_not_an_empty_set(
        self, monkeypatch,
    ):
        m = _mod()
        doubles: list[_BackendDouble] = []
        _install_backend_double(monkeypatch, doubles)

        join = await m.fetch_terminal_task_ids(types.SimpleNamespace(taskmaster=None))

        # No backend is even constructed, and -- the load-bearing part -- the
        # skip is DISTINGUISHABLE from a genuine "no terminal tasks". Both
        # omit the metric (zero exposure), but only one of them means the
        # family was measured and found clean.
        assert doubles == []
        assert join.statuses == {}
        assert join.skipped_reason is not None
        assert join.available is False

    async def test_a_genuine_empty_result_is_available_not_skipped(self, monkeypatch):
        m = _mod()
        doubles: list[_BackendDouble] = []
        _install_backend_double(monkeypatch, doubles, statuses={'1': 'pending'})

        join = await m.fetch_terminal_task_ids(_taskmaster_config())

        assert join.statuses == {}
        assert join.available is True
        assert join.skipped_reason is None

    async def test_a_skipped_join_omits_the_family_rather_than_emitting_a_zero(
        self, monkeypatch,
    ):
        m = _mod()
        _install_backend_double(monkeypatch, [])

        join = await m.fetch_terminal_task_ids(types.SimpleNamespace(taskmaster=None))
        inputs = _full_inputs()
        inputs['staleness'] = m.terminal_staleness(
            [_record('rec-3', 'Task 4802 status=in-progress')], join.statuses,
        )
        series = m.build_series(**inputs)

        assert 'task-terminal-staleness' not in _ids(series)
        assert 'task-terminal-staleness' in m.metric_families_not_measured(series)


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
