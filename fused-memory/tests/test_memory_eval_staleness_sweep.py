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

import asyncio
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

    def test_every_pointer_key_is_parsed_by_the_one_sanctioned_helper(self, monkeypatch):
        """INV-5 / D7, asserted as a CALL rather than as a substring.

        The spy delegates to the real helper, so ``pointer_targets`` still
        returns correct refs and this test cannot pass against a stub that
        merely swallowed the values. No rename, re-spelling or annotated
        copy-paste of a local parser can defeat it — which is precisely what a
        sweep over the script's own text could not promise: the module
        docstring names the helper on purpose, so a substring check for it was
        satisfied by prose and would have passed with every call site deleted.

        The patch lands BECAUSE the script's import is function-local (inside
        ``pointer_targets``, ``# noqa: PLC0415``) and therefore re-resolves the
        module attribute on every call. If that import is ever hoisted to
        module level it would bind at import time and silently no-op this spy;
        patch the SCRIPT module's own attribute instead.
        """
        m = _mod()
        from fused_memory import memory_metadata  # noqa: PLC0415

        real = memory_metadata.normalize_supersedes
        seen: list[Any] = []

        def _spy(value: Any) -> list[Any]:
            seen.append(value)
            return real(value)

        monkeypatch.setattr(memory_metadata, 'normalize_supersedes', _spy)
        refs = m.pointer_targets(
            _record(supersedes=UUID_A, parent_id=UUID_B, corrects=[UUID_C]),
        )

        # All three keys, each value handed over untouched — not just the one
        # D7 names, because parent_id and corrects carry the same ambiguity.
        assert seen == [UUID_A, UUID_B, [UUID_C]]
        assert [(r.key, r.target) for r in refs] == [
            ('supersedes', UUID_A), ('parent_id', UUID_B), ('corrects', UUID_C),
        ]


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

    def test_two_content_less_sources_do_not_collapse_into_one_item(self):
        """An empty content hashes to ONE key, so a collision is not "one fact".

        ``fetch_pointer_records`` normalises content to ``''`` when no payload
        key yields a string, and ``content_key('')`` is a constant. Two
        DIFFERENT sources with unreadable content citing the same target would
        therefore share an item_key, and the AND fold would hand a resolving
        edge the broken edge's ``passed=False`` — a healthy edge reported
        failing, in the metric whose whole purpose is per-edge grandfathering
        in leaf alpha. Worse, alpha persists item_keys, so a key that means
        two different edges poisons the ratchet permanently.
        """
        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-1', '', supersedes=UUID_A)),
            *m.pointer_targets(_record('rec-2', '', supersedes=UUID_A)),
        ]
        items = m.successor_pointer_items(refs, {UUID_A: True})
        # Unkeyable, so absent (this leaf's zero-exposure posture) — never
        # merged into a single item whose verdict belongs to neither source.
        assert items == []

    def test_a_content_less_edge_never_drags_a_keyable_one_down(self):
        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-1', '', supersedes=UUID_A)),
            *m.pointer_targets(_record('rec-2', 'real words', supersedes=UUID_A)),
        ]
        items = m.successor_pointer_items(refs, {UUID_A: True})
        assert [item.passed for item in items] == [True]

    def test_content_less_supersedes_edges_are_disclosed_not_silently_dropped(self):
        """Skipped is fine; SILENTLY skipped is the thing this leaf forbids."""
        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-1', '', supersedes=UUID_A)),
            *m.pointer_targets(_record('rec-2', '   ', supersedes=UUID_B)),
            *m.pointer_targets(_record('rec-3', 'real words', supersedes=UUID_A)),
            # Other keys are not tripwired at all, so they are not "skipped".
            *m.pointer_targets(_record('rec-4', '', corrects=UUID_C)),
        ]
        unkeyable = m.unkeyable_successor_refs(refs)
        assert [(ref.source_id, ref.key) for ref in unkeyable] == [
            ('rec-1', 'supersedes'), ('rec-2', 'supersedes'),
        ]
        # The edges the census still counts are unchanged — only the tripwire
        # declines them, so nothing that was measured becomes unmeasured.
        assert m.dangling_census(refs, {UUID_A: True}).examined == 4

    @pytest.mark.parametrize('member', [
        {'id': UUID_A},          # a dict member — unhashable
        [UUID_A],                # a nested list member — unhashable
        {UUID_A: 'why'},
    ])
    def test_an_unhashable_pointer_member_does_not_abort_the_sweep(self, member):
        """A dict- or list-valued member must be REPORTED, never fatal.

        ``normalize_supersedes`` wraps a non-list/non-str value verbatim ("any
        other scalar is wrapped rather than rejected") and copies a list member
        verbatim, so an unhashable member reaches :class:`PointerRef` intact —
        and ``PointerRef.target`` is typed ``Any`` precisely to let it. Any code
        that HASHES a ref therefore raises ``TypeError: unhashable type`` and
        takes down the whole sweep — every family, for the entire corpus —
        because of one malformed pointer. That is the exact collapse
        ``is_readable_target`` and ``_tripwire_item_key``'s ``repr()`` were
        written to prevent: this runner exists to report that kind of damage, so
        the damage must never be what kills it.

        Reachable, not hypothetical: ``validate_memory_metadata`` only became
        fatal on non-UUID members after the fact and records live non-string
        members already in the corpus, and ``parent_id``/``corrects`` get no
        member validation at all.
        """
        m = _mod()
        refs = m.pointer_targets(_record('rec-1', 'real words', supersedes=[member]))
        assert [r.target for r in refs] == [member]

        # Neither the tripwire nor its disclosure may HASH the ref. The source
        # has real content, so the edge is keyable (``_tripwire_item_key`` goes
        # through ``repr``) and gets an item — failing, because a target that
        # cannot name a memory has no predecessor to find.
        items = m.successor_pointer_items(refs, {})
        assert [item.passed for item in items] == [False]
        assert m.unkeyable_successor_refs(refs) == []

        # And the member is reported as malformed rather than swallowed: it can
        # never name a memory, so no read is issued and the edge is unresolved.
        assert [r.target for r in m.malformed_pointer_refs(refs)] == [member]
        assert m.unique_pointer_targets(refs) == []
        assert m.dangling_census(refs, {}).unresolved == 1

    def test_a_keyable_edge_survives_an_unhashable_one_beside_it(self):
        """One malformed member must not cost the tripwire its healthy items."""
        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-1', 'real words', supersedes=[{'id': UUID_A}])),
            *m.pointer_targets(_record('rec-2', 'other words', supersedes=UUID_B)),
            # Content-less AND unhashable: the skip predicate must reach its
            # verdict without hashing either.
            *m.pointer_targets(_record('rec-3', '', supersedes=[[UUID_C]])),
        ]
        items = m.successor_pointer_items(refs, {UUID_B: True})
        # Keyed by content hash, so the expectation is stated per SOURCE rather
        # than by list position: rec-2's live edge passes, rec-1's unresolvable
        # one fails, and rec-3 is skipped for having no content to key on.
        verdicts = {item.item_key: item.passed for item in items}
        assert verdicts == {
            m._tripwire_item_key(refs[0]): False,
            m._tripwire_item_key(refs[1]): True,
        }
        assert [r.source_id for r in m.unkeyable_successor_refs(refs)] == ['rec-3']


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

    def test_the_helper_is_actually_called(self, monkeypatch):
        """INV-5, and the helper carries the mandatory cheap-prefilter ordering.

        ``POINT_IN_TIME_CHECK_RE``'s two lookaheads under ``re.DOTALL`` are
        quadratic in content length; the helper prefilters with the
        lookahead-free ``LIVE_TASK_STATUS_RE``, which is what keeps a
        corpus-scale scan tractable. Re-deriving either regex here would drop
        that ordering silently — so the delegation is asserted by patching the
        real predicate and observing the call, not by grepping the script for
        a regex name it would no longer be spelling.
        """
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


def _full_refs():
    """The refs behind :func:`_full_inputs` — one citation per target."""
    m = _mod()
    return [
        *m.pointer_targets(_record('rec-1', 'successor one', supersedes=UUID_B)),
        *m.pointer_targets(_record('rec-2', 'a correction', corrects=UUID_C)),
    ]


def _multi_cited_refs():
    """Refs whose targets are cited more than once — the read-plan case.

    Deliberately a SIBLING of :func:`_full_refs` rather than a widening of it:
    the single-citation fixture is load-bearing for the rest of
    ``TestBuildSeries``, and changing its multiplicity in place would move
    every one of their expected numbers at once.

    THREE refs cite ONE resolved target (``rec-1``/``rec-2`` supersede it,
    ``rec-3`` corrects it), one ref names a target no read can be issued for
    at all, and one names a readable target that does not resolve.
    """
    m = _mod()
    return [
        *m.pointer_targets(_record('rec-1', 'successor one', supersedes=UUID_B)),
        *m.pointer_targets(_record('rec-2', 'successor two', supersedes=UUID_B)),
        *m.pointer_targets(_record('rec-3', 'a correction', corrects=UUID_B)),
        *m.pointer_targets(_record('rec-4', 'a broken pointer', supersedes='not-a-uuid')),
        *m.pointer_targets(_record('rec-5', 'cites a ghost', corrects=UUID_C)),
    ]


def _multi_cited_inputs():
    """:func:`_full_inputs`'s shape over :func:`_multi_cited_refs`."""
    m = _mod()
    refs = _multi_cited_refs()
    resolution = {UUID_B: True}
    return {
        'census': m.dangling_census(refs, resolution),
        'tripwire_items': m.successor_pointer_items(refs, resolution),
        'surfacing': m.superseded_surfacing([(UUID_A, UUID_B)], [UUID_B, UUID_A]),
        'staleness': m.terminal_staleness(
            [_record('rec-6', 'Task 4802 status=in-progress')], {'4802': 'done'},
        ),
        'corpus_counts': {},
        'project_id': 'dark_factory',
        'stamp': STAMP,
        'refs': refs,
    }


def _full_inputs():
    """One of every family, all with non-zero exposure."""
    m = _mod()
    refs = _full_refs()
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
        'refs': refs,
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
        inputs['refs'] = []  # a scan that found nothing also planned no reads
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

    def test_every_disclosure_key_is_asserted_not_merely_spot_checked(self):
        """The emitted key SET, and every value in it.

        The test above spot-checks two of these keys, which is how a disclosure
        that reported the wrong number shipped: nothing asserted it. An
        emitted-SET assertion makes the next disclosure added without coverage
        fail here rather than slip through the same gap.
        """
        m = _mod()
        inputs = _full_inputs()
        inputs['corpus_counts'] = {'procedural_knowledge_scanned': 12}
        counts = m.build_series(**inputs).corpus.counts

        assert set(counts) == {
            'procedural_knowledge_scanned',
            'pointer_refs_malformed',
            'pointer_targets_unique_reads',
            'successor_edges_unkeyable',
            'surfacing_edges_unsearchable',
            'surfacing_queries_degraded',
            'surfacing_search_depth',
            'task_terminal_entry_task_pairs',
            'pointers_supersedes_examined',
            'pointers_supersedes_resolved',
            'pointers_supersedes_unresolved',
            'pointers_corrects_examined',
            'pointers_corrects_resolved',
            'pointers_corrects_unresolved',
        }

        census = inputs['census']
        assert counts['pointer_refs_malformed'] == len(
            m.malformed_pointer_refs(census.unresolved_refs),
        )
        assert counts['pointer_targets_unique_reads'] == len(
            m.unique_pointer_targets(_full_refs()),
        )
        assert counts['successor_edges_unkeyable'] == len(
            m.unkeyable_successor_refs(_full_refs()),
        )
        assert counts['surfacing_edges_unsearchable'] == len(
            m.unsearchable_supersedes_refs(_full_refs()),
        )
        assert counts['surfacing_queries_degraded'] == len(inputs['surfacing'].degraded)
        assert counts['surfacing_search_depth'] == m.SURFACING_SEARCH_DEPTH
        assert counts['task_terminal_entry_task_pairs'] == len(inputs['staleness'].records)
        for key, row in census.by_key.items():
            for field_name, value in row.items():
                assert counts[f'pointers_{key}_{field_name}'] == value

    def test_the_surfacing_search_depth_rides_in_the_artifact(self):
        """The retrieval depth SETS family 1's denominator, so it is a narrowing.

        A pair whose superseded member ranks below the depth never appears in
        the ranked list, so it is dropped from ``pairs_comparable`` exactly the
        way ``--scan-limit`` drops records from the census — and that cap is
        disclosed. Undisclosed, a later change from 10 to 20 would move the
        trend leaf α reads with nothing in the series to explain it, which is
        the one reading of a count-shift this runner must never permit.
        """
        m = _mod()
        inputs = _full_inputs()
        inputs['surfacing_depth'] = 25
        assert m.build_series(**inputs).corpus.counts['surfacing_search_depth'] == 25

    def test_a_degraded_surfacing_query_reaches_the_machine_readable_artifact(self):
        """Prose-only disclosure is a silent cap for every JSON consumer.

        Which is all of them — leaf α's evaluator reads the artifact, never the
        report. Without this row a mem0 outage and a corpus that genuinely stopped
        surfacing superseded entries produce the SAME artifact: a small (or
        absent) ``superseded-still-surfacing`` with nothing to say why.
        """
        m = _mod()
        inputs = _full_inputs()
        degraded = m.SurfacingObservation(
            pairs_comparable=0, still_surfacing=0, records=(), inversions=(),
            degraded=(
                m.DegradedSurfacingQuery(
                    source_id='rec-1', target=UUID_B, failed_stores=('mem0',),
                ),
            ),
        )
        inputs['surfacing'] = degraded
        series = m.build_series(**inputs)

        assert series.corpus.counts['surfacing_queries_degraded'] == 1
        # Zero exposure, so the family declines to emit rather than fabricate a
        # clean datapoint — and the disclosure is what says WHY it is missing.
        assert m.METRIC_SUPERSEDED_STILL_SURFACING not in _ids(series)
        assert m.METRIC_SUPERSEDED_STILL_SURFACING in m.metric_families_not_measured(series)

    def test_the_family_1_exposure_is_published_once_not_under_a_second_name(self):
        """One number, one authoritative home: the metric's own ``n``.

        A ``surfacing_pairs_observed`` disclosure would be identically
        ``pairs_comparable``, and two names for one number can only ever
        disagree — at which point a consumer has no way to tell which is
        authoritative. What the counts DO carry is the pair of narrowings the
        metric's ``n`` cannot express: edges never searched, and searches that
        did not answer.
        """
        m = _mod()
        inputs = _full_inputs()
        series = m.build_series(**inputs)
        counts = series.corpus.counts

        assert 'surfacing_pairs_observed' not in counts
        assert _metric(series, m.METRIC_SUPERSEDED_STILL_SURFACING).n == (
            inputs['surfacing'].pairs_comparable
        )
        assert {'surfacing_edges_unsearchable', 'surfacing_queries_degraded'} <= set(counts)

    def test_the_edges_family_1_never_searched_are_disclosed(self):
        """A shrunken ``pairs_comparable`` with nothing to explain it is a silent cap.

        Two supersedes edges are unsearchable — one with no successor text to
        derive a query from, one whose target is not a memory id — so neither
        can ever join ``pairs_comparable``. Undisclosed, leaf α's count-shift
        trend would read their absence as a corpus that stopped superseding.
        """
        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-1', 'successor one', supersedes=UUID_B)),
            *m.pointer_targets(_record('rec-blank', '', supersedes=UUID_C)),
            *m.pointer_targets(_record('rec-bad', 'has text', supersedes={'oops': 1})),
            # A `corrects` edge is not family 1's population at all, so it is
            # not "unsearchable" — it was never in scope to be searched.
            *m.pointer_targets(_record('rec-2', '', corrects=UUID_C)),
        ]
        inputs = _full_inputs()
        inputs['refs'] = refs
        inputs['census'] = m.dangling_census(refs, {UUID_B: True})
        inputs['tripwire_items'] = m.successor_pointer_items(refs, {UUID_B: True})

        counts = m.build_series(**inputs).corpus.counts

        assert counts['surfacing_edges_unsearchable'] == 2
        assert [ref.source_id for ref in m.unsearchable_supersedes_refs(refs)] == [
            'rec-blank', 'rec-bad',
        ]
        # It OVERLAPS the tripwire's own narrowing rather than partitioning it:
        # every blank-content edge is unsearchable too. Disclosed separately
        # because they narrow different families, so a consumer reads each
        # against its own metric and never sums the two.
        assert counts['successor_edges_unkeyable'] == 1

    def test_the_disclosed_read_count_is_reads_not_citations(self):
        """``pointer_targets_unique_reads`` IS the plan ``resolve_pointer_targets`` runs.

        Three records citing one resolved target cost ONE live
        ``get_memory_by_id``, not three — the de-duplication lives in
        ``unique_pointer_targets``, and that helper is the read plan. Computed
        from the helper here rather than restated as a literal, so the
        assertion tracks the read plan instead of a hand-copied number.

        A read-cost field that instead grew with citation density would drift
        upward as the corpus cross-references itself, and leaf α — which reads
        this artifact, not this source — would see a trend that no read ever
        made.
        """
        m = _mod()
        counts = m.build_series(**_multi_cited_inputs()).corpus.counts
        assert counts['pointer_targets_unique_reads'] == len(
            m.unique_pointer_targets(_multi_cited_refs()),
        )

    def test_an_unreadable_target_is_disclosed_but_never_counted_as_a_read(self):
        """Excluded from the read count, still named by its own key.

        ``resolve_pointer_targets`` issues no read for a target that is not
        memory-id-shaped, so counting it as a read is a fabricated cost. It is
        not thereby forgiven: ``pointer_refs_malformed`` is where it stays
        visible, which is what makes the exclusion a correction rather than a
        lost disclosure.
        """
        m = _mod()
        inputs = _multi_cited_inputs()
        counts = m.build_series(**inputs).corpus.counts

        assert 'not-a-uuid' not in m.unique_pointer_targets(_multi_cited_refs())
        assert counts['pointer_targets_unique_reads'] == len({UUID_B, UUID_C})
        assert counts['pointer_refs_malformed'] == len(
            m.malformed_pointer_refs(inputs['census'].unresolved_refs),
        )
        assert counts['pointer_refs_malformed'] == 1

    def test_reads_never_exceed_the_edges_that_asked_for_them(self):
        """One read per distinct readable target, so reads ≤ pointers examined.

        True by construction once the field counts distinct readable targets,
        and violated by any expression that sums a de-duplicated half with a
        per-edge one. Asserted against the per-key rows in the same artifact,
        so the two disclosures cannot disagree about what a unit is.
        """
        m = _mod()
        inputs = _multi_cited_inputs()
        counts = m.build_series(**inputs).corpus.counts
        examined = sum(
            value for key, value in counts.items() if key.endswith('_examined')
        )
        assert examined == inputs['census'].examined
        assert counts['pointer_targets_unique_reads'] <= examined

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


class _ConcurrencyProbe:
    """An async call double that records how many calls were IN FLIGHT at once.

    The ``asyncio.sleep(0)`` is what makes overlap observable at all: a
    coroutine that never suspends runs to completion before its sibling is
    scheduled, so a sequential band and a concurrent one would produce the
    same peak of 1 and the assertion would be vacuous.

    *result* is returned to the caller, or called first if it is a callable
    (so a per-call answer can be seeded without a second double).
    """

    def __init__(self, result=None):
        self._result = result
        self.in_flight = 0
        self.peak = 0
        self.calls: list[types.SimpleNamespace] = []

    async def __call__(self, *args, **kwargs):
        self.calls.append(types.SimpleNamespace(args=args, kwargs=kwargs))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0)
            return self._result(*args, **kwargs) if callable(self._result) else self._result
        finally:
            self.in_flight -= 1


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

    async def test_the_reads_overlap_rather_than_serialising_one_per_target(self):
        """Concurrent, bounded by the module's own knob — and still ordered.

        Not a threshold (G6 does not reach it): every read still happens and
        no measured population depends on the fan-out. It is asserted because
        a scheduled run (leaf ε) issues one point read per unique pointer
        target, and a band that quietly went back to awaiting them one at a
        time would still pass every other test in this class.
        """
        from unittest.mock import MagicMock  # noqa: PLC0415

        m = _mod()
        targets = [f'{i:08d}-1111-4a2b-8c3d-4e5f60718293' for i in range(20)]
        refs = [
            ref for index, target in enumerate(targets)
            for ref in m.pointer_targets(_record(f'rec-{index}', 'text', supersedes=target))
        ]
        probe = _ConcurrencyProbe(result=None)
        memory = MagicMock()
        memory.get_memory_by_id = probe

        resolution = await m.resolve_pointer_targets(memory, 'dark_factory', refs)

        assert 1 < probe.peak <= m.STORE_READ_CONCURRENCY
        # Every target still read exactly once, and the map still keyed in
        # first-seen order: overlap must not cost determinism.
        assert [call.args[1] for call in probe.calls] == targets
        assert list(resolution) == targets


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
        assert memory.search.await_args.kwargs == {
            'project_id': 'dark_factory',
            'limit': 7,
            'stores': ['mem0'],
            'categories': list(m.SWEEP_CATEGORIES),
        }

    async def test_the_search_is_pinned_to_the_swept_population(self):
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        m = _mod()
        # "before" matches ReadRouter's temporal heuristic, which routes to
        # GRAPHITI ALONE -- so an unpinned search over this perfectly ordinary
        # successor text would return no Mem0 point ids at all and drop the
        # pair out of pairs_comparable. The explicit stores override
        # short-circuits route() before any classification (heuristic OR the
        # per-edge LLM fallback), so the population searched is the population
        # swept, and two runs over an unchanged corpus agree.
        refs = m.pointer_targets(
            _record('rec-1', 'the state before the migration', supersedes=UUID_B),
        )
        memory = MagicMock()
        memory.search = AsyncMock(return_value=[_Hit(UUID_B), _Hit('rec-1')])

        obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)

        assert memory.search.await_args.kwargs['stores'] == ['mem0']
        assert memory.search.await_args.kwargs['categories'] == list(m.SWEEP_CATEGORIES)
        assert obs.pairs_comparable == 1
        assert obs.still_surfacing == 1

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
        # ...and it is NOT a degraded query: the store answered, the pair simply
        # did not come back. Those two must stay distinguishable.
        assert obs.degraded == ()

    async def test_a_degraded_search_is_disclosed_not_scored_as_a_missing_pair(self):
        """``search`` swallows a store failure and returns an EMPTY list.

        ``MemoryService.search`` never raises on a store error or a search
        timeout: it catches / cancels and returns a ``SearchResults`` that is
        empty but carries ``degraded=True`` and ``failed_stores``. Scoring that
        empty list would drop the pair for not being both-present, silently
        SHRINKING ``pairs_comparable`` — to zero in the limit, which omits the
        metric entirely — with nothing recorded anywhere. Leaf α would then trend
        a mem0 outage as a real corpus change. This module refuses that reading
        everywhere else (``fetch_pointer_records`` and ``resolve_pointer_targets``
        both propagate ``TimeoutError`` so "backend blipped" is never read as
        "corpus is clean"), and β already excludes degraded observations from
        every denominator while disclosing them.
        """
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        from fused_memory.services.memory_service import SearchResults  # noqa: PLC0415

        m = _mod()
        refs = m.pointer_targets(_record('rec-1', 'the successor text', supersedes=UUID_B))
        memory = MagicMock()
        memory.search = AsyncMock(return_value=SearchResults(
            [], degraded=True, failed_stores=['mem0'],
            failure_diagnostics=[{'store': 'mem0', 'error': 'TimeoutError'}],
        ))

        obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)

        # Excluded from the denominator, not scored as a clean absence...
        assert obs.pairs_comparable == 0
        assert obs.still_surfacing == 0
        # ...and named, with the store that failed: "the run was degraded" tells
        # an operator to distrust the numbers, "mem0 raised TimeoutError" tells
        # them what to restart.
        assert len(obs.degraded) == 1
        assert obs.degraded[0].source_id == 'rec-1'
        assert obs.degraded[0].target == UUID_B
        assert obs.degraded[0].failed_stores == ('mem0',)
        assert obs.degraded[0].diagnostics == ({'store': 'mem0', 'error': 'TimeoutError'},)

    async def test_the_degrade_metadata_is_read_before_any_list_operation(self):
        """It rides on a list SUBCLASS and does not survive ``results or []``.

        ``SearchResults`` documents that ``degraded``/``failed_stores`` are lost
        by slicing, ``sorted()``, concatenation and comprehensions — and an EMPTY
        ``SearchResults`` is falsy, so the ``results or []`` guard itself hands
        back a bare ``list`` with the metadata stripped. A degraded EMPTY result
        is exactly the case that matters, so reading the attributes after that
        guard would disclose nothing precisely when there is something to
        disclose.
        """
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        from fused_memory.services.memory_service import SearchResults  # noqa: PLC0415

        m = _mod()
        empty = SearchResults([], degraded=True, failed_stores=['mem0'])
        # The premise, asserted rather than assumed.
        assert not empty
        assert not getattr(empty or [], 'degraded', False)

        refs = m.pointer_targets(_record('rec-1', 'the successor text', supersedes=UUID_B))
        memory = MagicMock()
        memory.search = AsyncMock(return_value=empty)

        assert len((await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)).degraded) == 1

    async def test_a_degraded_search_that_named_no_store_still_discloses(self):
        """``degraded=True`` alone is enough; so is a store name alone."""
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        from fused_memory.services.memory_service import SearchResults  # noqa: PLC0415

        m = _mod()
        refs = m.pointer_targets(_record('rec-1', 'the successor text', supersedes=UUID_B))
        for results in (
            SearchResults([], degraded=True),
            SearchResults([_Hit(UUID_B), _Hit('rec-1')], failed_stores=['graphiti']),
        ):
            memory = MagicMock()
            memory.search = AsyncMock(return_value=results)
            obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)
            # The second case would otherwise score a full inversion off a
            # partial answer — a ranking verdict read from an incomplete list.
            assert obs.pairs_comparable == 0
            assert len(obs.degraded) == 1

    async def test_one_degraded_edge_never_suppresses_a_healthy_one(self):
        """The exclusion is PER EDGE, so a blip costs only its own query."""
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        from fused_memory.services.memory_service import SearchResults  # noqa: PLC0415

        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-1', 'successor one', supersedes=UUID_B)),
            *m.pointer_targets(_record('rec-2', 'successor two', supersedes=UUID_C)),
        ]
        memory = MagicMock()
        memory.search = AsyncMock(side_effect=[
            SearchResults([], degraded=True, failed_stores=['mem0']),
            SearchResults([_Hit(UUID_C), _Hit('rec-2')]),
        ])

        obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)

        assert obs.pairs_comparable == 1
        assert obs.still_surfacing == 1
        assert [r.successor_id for r in obs.inversions] == ['rec-2']
        assert [q.source_id for q in obs.degraded] == ['rec-1']

    async def test_a_search_that_raises_is_disclosed_rather_than_aborting_the_run(self):
        """An escaping exception here would discard three measured families.

        ``MemoryService.search`` swallows the failures it knows about, but a
        backend error on a path that does not swallow reaches this band — and
        by then the pointer scan and the target reads have already completed.
        Propagating would throw away dangling-pointers, successor-pointer-present
        AND task-terminal-staleness and write no artifact at all, over a family
        that was only ever going to be narrower. ``fetch_terminal_task_ids``
        degrades a failing task backend to a NAMED skip for exactly this reason.
        """
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        m = _mod()
        refs = m.pointer_targets(_record('rec-1', 'the successor text', supersedes=UUID_B))
        memory = MagicMock()
        memory.search = AsyncMock(side_effect=TimeoutError('mem0 search timed out'))

        obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)

        assert obs.pairs_comparable == 0
        # Named, not swallowed: the edge lands in the same disclosure a
        # sentinel-degraded search does, so the report and corpus.counts say
        # this family was narrowed and by what.
        assert len(obs.degraded) == 1
        assert obs.degraded[0].source_id == 'rec-1'
        assert obs.degraded[0].target == UUID_B
        # 'mem0' because the search is PINNED to that store; the exception
        # type rides in the diagnostics, where the store's own failures put it.
        assert obs.degraded[0].failed_stores == ('mem0',)
        assert obs.degraded[0].diagnostics[0]['error'] == 'TimeoutError'

    async def test_one_raising_edge_never_costs_a_healthy_one_its_score(self):
        """Per edge, exactly as the sentinel path is."""
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-1', 'successor one', supersedes=UUID_B)),
            *m.pointer_targets(_record('rec-2', 'successor two', supersedes=UUID_C)),
        ]
        memory = MagicMock()
        memory.search = AsyncMock(side_effect=[
            RuntimeError('the backend fell over'),
            [_Hit(UUID_C), _Hit('rec-2')],
        ])

        obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)

        assert obs.pairs_comparable == 1
        assert obs.still_surfacing == 1
        assert [q.source_id for q in obs.degraded] == ['rec-1']

    async def test_an_unsearchable_edge_costs_no_search_and_is_reported(self):
        """The skip and its disclosure go through ONE predicate.

        A non-string target can never match a returned memory id and a blank
        successor content has no query to derive, so neither edge is searched.
        Both narrow ``pairs_comparable``, which is why the same predicate that
        skips them is the one that names them.
        """
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        m = _mod()
        refs = [
            *m.pointer_targets(_record('rec-blank', '  ', supersedes=UUID_B)),
            *m.pointer_targets(_record('rec-bad', 'has text', supersedes=[{'id': UUID_C}])),
        ]
        memory = MagicMock()
        memory.search = AsyncMock(return_value=[])

        obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)

        assert memory.search.await_count == 0
        assert obs.pairs_comparable == 0
        assert [ref.source_id for ref in m.unsearchable_supersedes_refs(refs)] == [
            'rec-blank', 'rec-bad',
        ]

    async def test_the_searches_overlap_rather_than_serialising_one_per_edge(self):
        """Concurrent, bounded by the module's own knob — and still ordered.

        Each search is embedding-backed and the live ``supersedes`` population
        is ~150 records, so a sequential band is ~150 serialised round trips on
        a runner leaf ε schedules beside leaf β. Not a threshold: every edge is
        still searched and the fold is still handed its inputs in ref order.
        """
        from unittest.mock import MagicMock  # noqa: PLC0415

        m = _mod()
        refs = [
            ref for index in range(20)
            for ref in m.pointer_targets(
                _record(f'rec-{index}', f'successor {index}', supersedes=UUID_B),
            )
        ]
        probe = _ConcurrencyProbe(result=[])
        memory = MagicMock()
        memory.search = probe

        obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)

        assert 1 < probe.peak <= m.STORE_READ_CONCURRENCY
        assert [call.args[0] for call in probe.calls] == [
            f'successor {index}' for index in range(20)
        ]
        assert obs.pairs_comparable == 0

    async def test_a_plain_list_search_double_is_never_read_as_degraded(self):
        """The old duck-typed doubles (and any plain list) stay non-degraded."""
        from unittest.mock import AsyncMock, MagicMock  # noqa: PLC0415

        m = _mod()
        refs = m.pointer_targets(_record('rec-1', 'the successor text', supersedes=UUID_B))
        memory = MagicMock()
        memory.search = AsyncMock(return_value=[_Hit(UUID_B), _Hit('rec-1')])

        obs = await m.fetch_surfacing_ranks(memory, 'dark_factory', refs)

        assert obs.degraded == ()
        assert obs.pairs_comparable == 1


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


# ---------------------------------------------------------------------------
# The read-only run band, driven end to end through the real main()
#
# "Never writes" is asserted as BEHAVIOUR, not as a comment: the double below
# raises on every mutating method, so a run that completes is a run that never
# wrote. The meta-test proves the tripwire actually fires, because a double
# that silently tolerated a write would make every test above vacuous.
# ---------------------------------------------------------------------------

class _ReadOnlyViolation(AssertionError):
    """Raised by the double when the sweep touches a mutation path."""


class _ServiceDouble:
    """A MemoryService stand-in that cannot be written to.

    Implements only the three read primitives this sweep is allowed to use
    and records every call, so the band can be asserted on rather than
    guessed at. Every write path is a tripwire.
    """

    def __init__(self, *, scrolls=None, by_id=None, searches=None):
        self._scrolls = dict(scrolls or {})
        self._by_id = dict(by_id or {})
        self._searches = dict(searches or {})
        self.scroll_calls: list[tuple[str, dict, int]] = []
        self.id_reads: list[str] = []
        self.search_calls: list[str] = []
        # Recorded separately from search_calls so the depth a run actually
        # searched at can be asserted against the depth it DISCLOSED.
        self.search_limits: list[int] = []
        self.initialized = False
        self.closed = False
        self.mem0 = self

    # -- the read paths the sweep is allowed to use ------------------------
    async def scroll_by_metadata(self, scope, filters, limit=1000, **kwargs):
        self.scroll_calls.append((scope.project_id, dict(filters), limit))
        return list(self._scrolls.get(filters.get('category'), []))

    async def get_memory_by_id(self, project_id, memory_id):
        self.id_reads.append(memory_id)
        return self._by_id.get(memory_id)

    async def search(self, query, project_id='main', limit=10, **kwargs):
        self.search_calls.append(query)
        self.search_limits.append(limit)
        seeded = self._searches.get(query, [])
        # A seeded SearchResults is handed back VERBATIM: its degraded /
        # failed_stores ride on a list SUBCLASS and would not survive the
        # list(...) copy, which is the very metadata the degrade tests seed.
        from fused_memory.services.memory_service import SearchResults  # noqa: PLC0415
        if isinstance(seeded, SearchResults):
            return seeded
        return list(seeded)

    # -- lifecycle ---------------------------------------------------------
    async def initialize(self):
        self.initialized = True

    async def close(self):
        self.closed = True

    # -- every write path is a tripwire ------------------------------------
    async def add_memory(self, *a, **kw):
        raise _ReadOnlyViolation('the sweep called add_memory')

    async def add_episode(self, *a, **kw):
        raise _ReadOnlyViolation('the sweep called add_episode')

    async def add_system_record(self, *a, **kw):
        raise _ReadOnlyViolation('the sweep called add_system_record')

    async def delete_memory(self, *a, **kw):
        raise _ReadOnlyViolation('the sweep called delete_memory')

    async def delete_episode(self, *a, **kw):
        raise _ReadOnlyViolation('the sweep called delete_episode')

    async def update_edge(self, *a, **kw):
        raise _ReadOnlyViolation('the sweep called update_edge')

    async def merge_entities(self, *a, **kw):
        raise _ReadOnlyViolation('the sweep called merge_entities')

    async def delete_entity(self, *a, **kw):
        raise _ReadOnlyViolation('the sweep called delete_entity')


def _seeded_double(*, search_results=None) -> _ServiceDouble:
    """A corpus with one live supersedes edge, one dangling edge, one stale entry.

    *search_results* overrides what the one surfacing query returns, so a test
    can hand back a degraded ``SearchResults`` without reaching into the
    double's internals.
    """
    return _ServiceDouble(
        scrolls={'procedural_knowledge': [
            _raw_point(UUID_A, {
                'data': 'the successor text', 'supersedes': UUID_B,
            }),
            _raw_point('rec-corrects', {
                'data': 'a correction', 'corrects': UUID_C,
            }),
            _raw_point('rec-stale', {
                'data': 'Task 4802 status=in-progress claimant_run_id=abc',
            }),
        ]},
        # UUID_B resolves; UUID_C was never written, so that edge dangles.
        by_id={UUID_B: {'id': UUID_B, 'content': 'the predecessor', 'metadata': {}}},
        searches={'the successor text': (
            [_Hit(UUID_B), _Hit(UUID_A)] if search_results is None else search_results
        )},
    )


def _install_run_band(monkeypatch, double, *, taskmaster_statuses=None, project_root='/repo'):
    """Point the lazily-imported MemoryService and config at test stand-ins.

    No test-only seam in the script: ``_run`` imports both inside the function
    (the D8 pattern), so patching the module attributes is enough to drive the
    real argparse/_run/emit path end to end.
    """
    import fused_memory.services.memory_service as ms  # noqa: PLC0415
    from fused_memory.backends import sqlite_task_backend  # noqa: PLC0415
    from fused_memory.config import schema  # noqa: PLC0415

    taskmaster = (
        None if taskmaster_statuses is None
        else types.SimpleNamespace(project_root=project_root)
    )
    monkeypatch.setattr(ms, 'MemoryService', lambda config: double)
    monkeypatch.setattr(
        schema, 'FusedMemoryConfig',
        lambda *a, **kw: types.SimpleNamespace(taskmaster=taskmaster),
    )
    monkeypatch.setattr(
        sqlite_task_backend, 'SqliteTaskBackend',
        lambda cfg: _BackendDouble(cfg, statuses=taskmaster_statuses or {}),
    )


def _run_main(monkeypatch, tmp_path, double, *extra_argv, taskmaster_statuses=None):
    """Drive the real ``main()`` with the stamp pinned, returning its exit code."""
    from shared.memory_eval_metrics import RUN_STAMP_ENV_VAR  # noqa: PLC0415

    _install_run_band(monkeypatch, double, taskmaster_statuses=taskmaster_statuses)
    monkeypatch.setenv(RUN_STAMP_ENV_VAR, STAMP)
    return _mod().main([
        '--project-id', 'dark_factory',
        '--metrics-root', str(tmp_path),
        *extra_argv,
    ])


class TestReadOnlyGuarantee:
    """A run that completes is a run that never wrote."""

    def test_a_full_run_never_touches_a_write_path(self, monkeypatch, tmp_path, capsys):
        double = _seeded_double()

        code = _run_main(
            monkeypatch, tmp_path, double, taskmaster_statuses={'4802': 'done'},
        )

        assert code == 0
        assert double.initialized and double.closed
        # And it did real work rather than completing by doing nothing.
        assert double.scroll_calls
        assert double.id_reads
        assert capsys.readouterr().out

    def test_the_double_would_have_caught_a_write(self):
        """The tripwire fires — otherwise every assertion above is vacuous."""
        import asyncio  # noqa: PLC0415

        double = _ServiceDouble()
        for method in (
            'add_memory', 'add_episode', 'add_system_record', 'delete_memory',
            'delete_episode', 'update_edge', 'merge_entities', 'delete_entity',
        ):
            with pytest.raises(_ReadOnlyViolation):
                asyncio.run(getattr(double, method)())

    def test_the_store_is_closed_even_when_the_sweep_raises(self, monkeypatch, tmp_path):
        double = _seeded_double()
        monkeypatch.setattr(
            double, 'scroll_by_metadata',
            _raising(TimeoutError('qdrant read timed out')),
        )

        with pytest.raises(TimeoutError):
            _run_main(monkeypatch, tmp_path, double)

        assert double.closed


def _raising(exc):
    async def _fail(*a, **kw):
        raise exc
    return _fail


class TestArgparseBand:
    """The CLI surface — and, load-bearing, what is absent from it."""

    def test_the_flag_set_is_exactly_this(self):
        parser = _mod().build_parser()
        flags = {opt for action in parser._actions for opt in action.option_strings}
        assert flags == {
            '-h', '--help',
            '--project-id',
            '--scan-limit',
            '--config',
            '--metrics-root',
            '--eval-id',
            '--no-metrics',
        }

    def test_there_is_no_mutation_flag(self):
        parser = _mod().build_parser()
        flags = {opt for action in parser._actions for opt in action.option_strings}
        # Asserted by equality above too, but named here because THIS is the
        # guarantee: a sweep that reports has no --apply to grow into.
        assert flags.isdisjoint({'--apply', '--fix', '--prune', '--delete', '--repair'})

    def test_the_defaults_are_this_leafs_own(self):
        m = _mod()
        args = m.build_parser().parse_args(['--project-id', 'dark_factory'])
        assert args.eval_id == 'e4-staleness-sweep'
        assert args.metrics_root == m._DEFAULT_METRICS_ROOT
        assert args.no_metrics is False
        assert args.scan_limit > 0

    def test_project_id_is_required(self):
        with pytest.raises(SystemExit):
            _mod().build_parser().parse_args([])


class TestArtifactEmission:
    """What a run leaves on disk, under the stamp it was pinned to."""

    def test_both_artifacts_land_under_the_eval_id_and_round_trip(
        self, monkeypatch, tmp_path,
    ):
        from shared.memory_eval_metrics import load_metric_series  # noqa: PLC0415

        _run_main(
            monkeypatch, tmp_path, _seeded_double(), taskmaster_statuses={'4802': 'done'},
        )

        metrics_path = tmp_path / 'e4-staleness-sweep' / f'metrics-{STAMP}.json'
        report_path = tmp_path / 'e4-staleness-sweep' / f'report-{STAMP}.txt'
        assert metrics_path.exists()
        assert report_path.exists()

        series = load_metric_series(metrics_path)
        assert series.eval_id == 'e4-staleness-sweep'
        assert series.run_stamp == STAMP
        # Every family measurable from this seeded corpus is measured.
        assert _ids(series) == {
            'superseded-still-surfacing',
            'dangling-pointers',
            'successor-pointer-present',
            'task-terminal-staleness',
        }

    def test_eval_id_overrides_the_directory(self, monkeypatch, tmp_path):
        _run_main(monkeypatch, tmp_path, _seeded_double(), '--eval-id', 'e4-scratch')

        assert (tmp_path / 'e4-scratch' / f'metrics-{STAMP}.json').exists()
        assert not (tmp_path / 'e4-staleness-sweep').exists()

    def test_no_metrics_prints_the_report_and_writes_nothing(
        self, monkeypatch, tmp_path, capsys,
    ):
        code = _run_main(monkeypatch, tmp_path, _seeded_double(), '--no-metrics')

        assert code == 0
        assert capsys.readouterr().out
        assert list(tmp_path.iterdir()) == []

    def test_the_report_on_disk_is_this_runners_text_not_the_shared_table(
        self, monkeypatch, tmp_path,
    ):
        """Asserting the file EXISTS would pass with none of this leaf's disclosures.

        ``write_metric_series`` already writes a report at this exact path —
        the shared minimal metric table — and ``write_report_text`` exists only
        to replace it with the fuller sectioned text. Dropped, mis-pathed, or
        ordered BEFORE the shared write, the artifact silently degrades to that
        table: no not_measured, no scope, no scan_truncation, no
        malformed_pointers, no surfacing_degraded. Every one of those is a
        disclosure this module argues must never be inferable-only, and an
        existence check cannot tell the two files apart.
        """
        outcome = _sweep(
            monkeypatch, tmp_path, _seeded_double(),
            taskmaster_statuses={'4802': 'done'}, write_metrics=True,
        )

        assert outcome.report_path is not None
        on_disk = outcome.report_path.read_text(encoding='utf-8')
        # The whole in-memory report, byte for byte: the file IS what the run
        # says it produced, not a subset of it.
        assert on_disk == outcome.report
        # Section by section rather than by prose, so a copy edit does not fail
        # this but a lost block does.
        for section in outcome.sections:
            assert section.text in on_disk
        assert {'scope', 'dangling_pointers', 'successor_pointer_tripwire'} <= {
            section.key for section in outcome.sections
        }
        # And the replacement landed AFTER the shared writer, not before it.
        assert outcome.metrics_path is not None
        assert outcome.metrics_path.exists()


class TestWriteReportText:
    """The atomic replace that widens the shared writer's report."""

    def test_it_replaces_an_existing_report_leaving_no_temp_sibling(self, tmp_path):
        m = _mod()
        path = tmp_path / f'report-{STAMP}.txt'
        path.write_text('the shared writer\'s minimal metric table\n', encoding='utf-8')

        m.write_report_text(path, 'the fuller sectioned text\n')

        assert path.read_text(encoding='utf-8') == 'the fuller sectioned text\n'
        # mkstemp creates a real sibling under the shared artifact root the
        # dashboard reads as plain files; one left behind is a permanent
        # half-written report beside a valid one.
        assert [p.name for p in tmp_path.iterdir()] == [path.name]

    def test_an_interrupted_write_leaves_neither_a_temp_file_nor_a_truncation(
        self, tmp_path, monkeypatch,
    ):
        """``except BaseException``, not ``except Exception``, and deliberately.

        A KeyboardInterrupt or a cancellation between mkstemp and the replace
        is exactly when the temp file would be orphaned, and those do not
        derive from ``Exception``. The previous report must also survive
        intact: the whole reason this is not a ``write_text`` is that a reader
        must never see a truncated report where a valid one was.
        """
        m = _mod()
        path = tmp_path / f'report-{STAMP}.txt'
        path.write_text('the previous report\n', encoding='utf-8')

        class _Interrupted(BaseException):
            pass

        def _explode(*args, **kwargs):
            raise _Interrupted('interrupted between write and replace')

        monkeypatch.setattr(m.os, 'replace', _explode)

        with pytest.raises(_Interrupted):
            m.write_report_text(path, 'never lands\n')

        assert path.read_text(encoding='utf-8') == 'the previous report\n'
        assert [p.name for p in tmp_path.iterdir()] == [path.name]


def _sweep(
    monkeypatch, tmp_path, double, *,
    taskmaster_statuses=None,
    scan_limit=100,
    project_ids=('dark_factory',),
    write_metrics=False,
    **run_kwargs,
):
    """Run the sweep band directly and return its outcome.

    ``run_sweep`` returns the sections under their machine keys (β's
    ProbeOutcome precedent), so what a run DISCLOSED is answerable without a
    test-only global in the script and without pattern-matching English.

    *project_ids*, *write_metrics* and ``**run_kwargs`` are passed through
    verbatim so a test can drive the real band at a non-default depth, project
    scope, or with the artifacts actually written, rather than re-implementing
    the call.
    """
    import asyncio  # noqa: PLC0415

    from fused_memory.backends import sqlite_task_backend  # noqa: PLC0415

    taskmaster = (
        None if taskmaster_statuses is None
        else types.SimpleNamespace(project_root='/repo')
    )
    monkeypatch.setattr(
        sqlite_task_backend, 'SqliteTaskBackend',
        lambda cfg: _BackendDouble(cfg, statuses=taskmaster_statuses or {}),
    )
    return asyncio.run(_mod().run_sweep(
        double,
        project_ids=project_ids,
        scan_limit=scan_limit,
        out_root=tmp_path,
        stamp=STAMP,
        config=types.SimpleNamespace(taskmaster=taskmaster),
        write_metrics=write_metrics,
        **run_kwargs,
    ))


class TestRunSweepProjectScope:
    """One project per run, enforced rather than merely documented."""

    def test_more_than_one_project_id_is_refused_rather_than_under_reported(
        self, monkeypatch, tmp_path,
    ):
        """The failure this refuses is silent and in the wrong direction.

        ``get_memory_by_id`` is project-scoped but the resolution map is keyed
        by bare target id, so sweeping two projects lets a target present in
        project A mark project B's ref RESOLVED — a dangling pointer reported
        as healthy, by a runner whose whole job is to find dangling pointers.
        A docstring warning is the weakest possible guard for that; the run
        must not start.
        """
        with pytest.raises(ValueError) as excinfo:
            _sweep(
                monkeypatch, tmp_path, _seeded_double(),
                project_ids=('dark_factory', 'other_project'),
            )
        message = str(excinfo.value)
        # Names the ids it was given, so the operator does not have to guess
        # which call site passed what.
        assert 'dark_factory' in message
        assert 'other_project' in message

    def test_no_project_id_at_all_is_refused_too(self, monkeypatch, tmp_path):
        """Zero ids would emit a clean-looking artifact for a corpus never read.

        The same guard, in the other direction: nothing would be scrolled, so
        every family would report zero exposure and the artifact would land
        under a stamp leaf α trends — an unread corpus presented as a measured
        one.
        """
        with pytest.raises(ValueError) as excinfo:
            _sweep(monkeypatch, tmp_path, _seeded_double(), project_ids=())
        assert 'exactly ONE project' in str(excinfo.value)

    def test_one_id_repeated_is_not_several_projects(self, monkeypatch, tmp_path):
        """The guard counts DISTINCT ids — the band already de-duplicates."""
        double = _seeded_double()
        outcome = _sweep(
            monkeypatch, tmp_path, double,
            project_ids=('dark_factory', 'dark_factory'),
        )
        assert outcome.series.corpus.project_id == 'dark_factory'
        # De-duplicated before the scroll, not after: a repeat must not double
        # the scanned counts either.
        assert [call[0] for call in double.scroll_calls] == ['dark_factory'] * len(
            _mod().SWEEP_CATEGORIES,
        )

    def test_the_disclosed_depth_is_the_depth_the_run_searched_at(
        self, monkeypatch, tmp_path,
    ):
        """A disclosure that cannot drift from the value it describes.

        The depth is read once in ``run_sweep`` and used twice — for the search
        and for the disclosure — so the artifact cannot claim a depth the run
        did not use.
        """
        double = _seeded_double()
        outcome = _sweep(monkeypatch, tmp_path, double, surfacing_depth=37)

        assert double.search_limits == [37]
        assert outcome.series.corpus.counts['surfacing_search_depth'] == 37
        # And the human-facing report names it beside the denominator it sets.
        sections = {section.key: section for section in outcome.sections}
        assert '37' in sections['superseded_surfacing'].text


class TestReport:
    """Every family is NAMED, so an absence can never read as health."""

    def _sections(self, monkeypatch, tmp_path, double, **kwargs):
        outcome = _sweep(monkeypatch, tmp_path, double, **kwargs)
        return {section.key: section for section in outcome.sections}

    def test_every_family_has_a_named_section(self, monkeypatch, tmp_path):
        sections = self._sections(
            monkeypatch, tmp_path, _seeded_double(), taskmaster_statuses={'4802': 'done'},
        )
        # Keyed on structure, not on prose: a copy edit must not fail this,
        # but a section that stops being emitted must.
        assert {
            'superseded_surfacing', 'dangling_pointers', 'task_terminal_staleness',
        } <= set(sections)

    def test_the_swept_scope_is_named_rather_than_left_to_be_inferred(
        self, monkeypatch, tmp_path,
    ):
        """The largest narrowing in the run, and the one nothing else names.

        Only the Mem0-primary categories are reachable by a payload scroll, so
        half the category vocabulary is out of scope on every run. Without a
        section saying so, the only trace is the ``scanned_<category>`` rows —
        which a reader can decode only if they already know six categories
        exist and which three a Qdrant scroll cannot reach.
        """
        m = _mod()
        sections = self._sections(monkeypatch, tmp_path, _seeded_double())

        assert 'scope' in sections
        text = sections['scope'].text
        # The categories the run ACTUALLY scrolled, named one by one.
        for category in m.SWEEP_CATEGORIES:
            assert category in text
        # And that the rest of the vocabulary was never in reach — the claim a
        # low count must not be read against.
        assert 'scroll_by_metadata' in text

    def test_the_scope_section_names_what_the_run_swept_not_the_default(
        self, monkeypatch, tmp_path,
    ):
        """A scope line copied from the module default could describe another run.

        ``fetch_pointer_records`` takes its categories from the caller, so the
        one honest source is the scan's own per-category stats.
        """
        double = _ServiceDouble(scrolls={'procedural_knowledge': []})
        outcome = _sweep(monkeypatch, tmp_path, double)
        scope = {s.key: s for s in outcome.sections}['scope']

        swept = list(outcome.scan_stats)
        assert swept  # the premise: something was scanned
        for category in swept:
            assert category in scope.text

    def test_family_1s_unsearchable_edges_are_named_beside_its_denominator(
        self, monkeypatch, tmp_path,
    ):
        """A shrunken comparable-pair count must never be unexplained.

        Two supersedes edges here are never searched — one with no successor
        text, one whose target is not a memory id — so neither can join
        ``pairs_comparable``. The section that publishes that denominator is
        where the narrowing has to be stated.
        """
        double = _ServiceDouble(scrolls={'procedural_knowledge': [
            _raw_point('rec-blank', {'supersedes': UUID_A}),
            _raw_point('rec-bad', {'data': 'has text', 'supersedes': [{'id': UUID_C}]}),
        ]})
        outcome = _sweep(monkeypatch, tmp_path, double)
        sections = {s.key: s for s in outcome.sections}

        assert outcome.series.corpus.counts['surfacing_edges_unsearchable'] == 2
        assert '2' in sections['superseded_surfacing'].text
        # Counted here, NAMED under the disclosures that already own them:
        # a non-id target is a malformed pointer, a blank source is unkeyable.
        assert 'rec-bad' in sections['malformed_pointers'].text
        assert 'rec-blank' in sections['unkeyable_successor_edges'].text

    def test_an_unmeasured_family_is_named_rather_than_omitted_silently(
        self, monkeypatch, tmp_path,
    ):
        # No task backend -> the staleness family has no exposure and emits no
        # metric. The run must SAY so; a metric that quietly stops existing
        # reads to a human as one that had nothing to report.
        sections = self._sections(monkeypatch, tmp_path, _seeded_double())
        assert 'not_measured' in sections
        assert 'task-terminal-staleness' in sections['not_measured'].text

    def test_a_skipped_task_backend_is_its_own_disclosure(self, monkeypatch, tmp_path):
        sections = self._sections(monkeypatch, tmp_path, _seeded_double())
        assert 'task_backend_skipped' in sections

    def test_a_reached_task_backend_produces_no_skip_disclosure(
        self, monkeypatch, tmp_path,
    ):
        sections = self._sections(
            monkeypatch, tmp_path, _seeded_double(), taskmaster_statuses={'4802': 'done'},
        )
        assert 'task_backend_skipped' not in sections

    def test_a_truncated_scan_is_its_own_disclosure(self, monkeypatch, tmp_path):
        sections = self._sections(monkeypatch, tmp_path, _seeded_double(), scan_limit=3)
        # Three seeded records against a limit of three: the scroll may have
        # stopped short, and a capped sample presented as a census is the one
        # thing this report must never do.
        assert 'scan_truncation' in sections

    def test_an_uncapped_scan_produces_no_truncation_disclosure(
        self, monkeypatch, tmp_path,
    ):
        sections = self._sections(monkeypatch, tmp_path, _seeded_double())
        assert 'scan_truncation' not in sections

    def test_malformed_pointer_members_get_their_own_disclosure(
        self, monkeypatch, tmp_path,
    ):
        double = _ServiceDouble(scrolls={'procedural_knowledge': [
            _raw_point('rec-bad', {'data': 'x', 'supersedes': ['not-a-uuid']}),
        ]})
        sections = self._sections(monkeypatch, tmp_path, double)
        # "The target is gone" and "the pointer was never writable" have
        # different fixes, so a dangling count must not merge them.
        assert 'malformed_pointers' in sections
        assert 'not-a-uuid' in sections['malformed_pointers'].text

    def test_content_less_successor_edges_get_their_own_disclosure(
        self, monkeypatch, tmp_path,
    ):
        double = _ServiceDouble(scrolls={'procedural_knowledge': [
            _raw_point('rec-blank', {'supersedes': [UUID_A]}),
        ]})
        sections = self._sections(monkeypatch, tmp_path, double)
        # The edge is real and the census counts it; only the TRIPWIRE
        # declines it, because an empty content cannot produce a key that
        # distinguishes this source from any other content-less one.
        assert 'unkeyable_successor_edges' in sections
        assert 'rec-blank' in sections['unkeyable_successor_edges'].text

    def test_a_keyable_run_produces_no_unkeyable_disclosure(
        self, monkeypatch, tmp_path,
    ):
        sections = self._sections(monkeypatch, tmp_path, _seeded_double())
        assert 'unkeyable_successor_edges' not in sections

    def test_a_degraded_surfacing_search_gets_its_own_disclosure(
        self, monkeypatch, tmp_path,
    ):
        """The whole run, against a search that reports a failed store.

        ``search`` returns an EMPTY ``SearchResults`` carrying
        ``degraded=True``/``failed_stores=['mem0']`` rather than raising, so
        without this the run would report ``pairs_comparable: 0`` — a mem0
        outage rendered as a corpus with nothing to compare, in the report an
        operator reads to decide whether to trust the numbers.
        """
        from fused_memory.services.memory_service import SearchResults  # noqa: PLC0415

        m = _mod()
        outcome = _sweep(monkeypatch, tmp_path, _seeded_double(
            search_results=SearchResults([], degraded=True, failed_stores=['mem0']),
        ))
        sections = {section.key: section for section in outcome.sections}

        assert 'surfacing_degraded' in sections
        text = sections['surfacing_degraded'].text
        # Named with the failing store, so the operator knows what to restart.
        assert 'mem0' in text
        assert UUID_A in text
        # The metric declines rather than reporting a fabricated clean zero, and
        # the not-measured section says so.
        assert m.METRIC_SUPERSEDED_STILL_SURFACING not in _ids(outcome.series)
        assert m.METRIC_SUPERSEDED_STILL_SURFACING in sections['not_measured'].text
        assert outcome.series.corpus.counts['surfacing_queries_degraded'] == 1

    def test_a_healthy_run_produces_no_degraded_disclosure(
        self, monkeypatch, tmp_path,
    ):
        outcome = _sweep(monkeypatch, tmp_path, _seeded_double())
        sections = {section.key: section for section in outcome.sections}
        assert 'surfacing_degraded' not in sections
        # And the count is disclosed as zero rather than omitted, so a consumer
        # can tell "no degradation" from "this runner does not report it".
        assert outcome.series.corpus.counts['surfacing_queries_degraded'] == 0

    def test_an_unhashable_pointer_member_does_not_take_the_whole_run_down(
        self, monkeypatch, tmp_path,
    ):
        """End to end: one dict-valued member must not cost every family.

        The sweep exists to REPORT malformed pointers, so a malformed pointer
        that aborts it before any artifact is produced is the worst available
        outcome. ``normalize_supersedes`` wraps a non-string member verbatim, so
        an unhashable target reaches the pure band intact and any hashing of a
        ``PointerRef`` raises ``TypeError`` for the entire corpus.
        """
        double = _ServiceDouble(scrolls={'procedural_knowledge': [
            _raw_point('rec-dict', {'data': 'x', 'supersedes': [{'id': UUID_A}]}),
            _raw_point('rec-list', {'data': 'y', 'supersedes': [[UUID_B]]}),
            # A healthy edge beside them, to prove the run still measures.
            _raw_point('rec-ok', {'data': 'the successor text', 'supersedes': UUID_C}),
        ]}, by_id={UUID_C: {'id': UUID_C, 'content': 'pred', 'metadata': {}}})

        outcome = _sweep(monkeypatch, tmp_path, double)

        sections = {section.key: section for section in outcome.sections}
        # Reported as malformed rather than raising...
        assert 'malformed_pointers' in sections
        assert 'rec-dict' in sections['malformed_pointers'].text
        assert 'rec-list' in sections['malformed_pointers'].text
        # ...the census still counts all three edges...
        assert outcome.census.examined == 3
        assert outcome.census.unresolved == 2
        # ...and the run still MEASURES: three keyable edges, of which only the
        # healthy one passes. A TypeError would have produced no artifact at all.
        assert sorted(item.passed for item in outcome.tripwire_items) == [False, False, True]
        assert outcome.series.corpus.counts['pointer_refs_malformed'] == 2

    def test_the_staleness_detail_does_not_assert_per_task_liveness(
        self, monkeypatch, tmp_path,
    ):
        """The analysis is per ENTRY; the report must not claim per TASK.

        ``frames_live_task_status_as_current_fact`` judges the WHOLE content
        and never establishes which referenced task the live-status framing
        was about. For "task 4802 (which motivated this) landed; task 5000
        status=in-progress", counting the entry is right — it does frame live
        state and does reference a terminal task — but rendering "claims task
        4802 is live" states something the analysis did not establish, in an
        operator-facing report whose entire premise is honest disclosure.
        """
        double = _ServiceDouble(scrolls={'procedural_knowledge': [
            _raw_point('rec-mixed', {
                'data': 'task 4802 (which motivated this) landed; '
                        'task 5000 status=in-progress claimant_run_id=abc',
            }),
        ]})
        sections = self._sections(
            monkeypatch, tmp_path, double, taskmaster_statuses={'4802': 'done'},
        )
        text = sections['task_terminal_staleness'].text
        # Still fully attributed — the operator gets entry, task and status.
        assert 'rec-mixed' in text
        assert '4802' in text
        assert 'done' in text
        # And attributed in the SANCTIONED shape. Asserted positively, not as
        # the absence of one hand-picked over-claiming sentence: a negative
        # prose assertion only rules out the exact wording it names, so
        # rewording the renderer to "asserts task 4802 is live" would commit
        # the very over-claim while still passing. Pinning the honest line
        # instead fails on ANY reword, which makes changing it a decision.
        assert (
            'rec-mixed frames live task state and references task 4802 (done)'
        ) in text

    def test_the_unresolved_targets_are_named_not_just_counted(
        self, monkeypatch, tmp_path,
    ):
        sections = self._sections(monkeypatch, tmp_path, _seeded_double())
        # A bare count tells an operator that something dangles but not which
        # pointer to go and look at.
        assert UUID_C in sections['dangling_pointers'].text


# ---------------------------------------------------------------------------
# The seeded live-store test — the task's user-observable signal
#
# On an EPHEMERAL per-worker collection, never the live corpus. Marked
# PER-TEST (never a module pytestmark): fused-memory runs under
# `-m 'not integration'`, so a module-level mark would deselect every pure
# test above and the guards they carry would guard nothing.
# ---------------------------------------------------------------------------

import contextlib  # noqa: E402
import os  # noqa: E402

from _fm_helpers import QDRANT_URL, qdrant_skipif  # noqa: E402

EPHEMERAL_COLLECTION_PREFIX = '_test_mem0_qdrant_integration'
"""The ONLY prefix scripts/cleanup_test_collections.py reaps.

mock_config's default `fused` prefix would leak a collection forever. Asserted
against that script's own constant below rather than restated here.
"""

SEED_PREDECESSOR = (
    'The staleness sweep resolves each pointer target with one live '
    'get_memory_by_id read per unique target id.'
)
SEED_SUCCESSOR = (
    'The staleness sweep resolves each pointer target against the live store, '
    'deduplicating by target id so one target costs one read.'
)
SEED_DANGLING_SOURCE = (
    'Corrects an earlier note about how the staleness sweep counts pointer '
    'targets it cannot resolve.'
)
NEVER_WRITTEN_UUID = 'deadbeef-0000-4000-8000-000000000001'


@pytest.fixture
def sweep_project_id(worker_id):
    """Per-xdist-worker so concurrent workers cannot share a collection."""
    return f'sweep_e4_{worker_id}'


@pytest.fixture
def sweep_config(mock_config, sweep_project_id):
    """mock_config pointed at an ephemeral collection with a REAL embedder.

    Clearing the fake api_key makes mem0's OpenAIEmbedding fall back to the
    real OPENAI_API_KEY. A stub constant vector would make the surfacing
    family meaningless — its whole question is about real retrieval order.
    """
    config = mock_config.model_copy(deep=True)
    config.mem0.collection_prefix = EPHEMERAL_COLLECTION_PREFIX
    config.embedder.providers.openai.api_key = None
    return config


@pytest.fixture
def clean_sweep_collection(sweep_config, sweep_project_id):
    """Delete the seeded collection before AND after, so a swallowed teardown self-heals."""
    from qdrant_client import QdrantClient  # noqa: PLC0415

    from fused_memory.models.scope import Scope  # noqa: PLC0415

    collection = Scope(project_id=sweep_project_id).mem0_collection_name(
        sweep_config.mem0.collection_prefix,
    )
    client = QdrantClient(url=QDRANT_URL, timeout=10)
    with contextlib.suppress(Exception):
        client.delete_collection(collection)
    yield collection
    with contextlib.suppress(Exception):
        client.delete_collection(collection)
    client.close()


class TestSeededStalenessSweep:
    """A real supersedes pair and a real dangling pointer, both reported."""

    def test_the_ephemeral_collection_is_one_the_reaper_can_reclaim(
        self, monkeypatch, sweep_config, sweep_project_id,
    ):
        """A leaked collection under the default prefix would live forever.

        Deliberately NOT via ``clean_sweep_collection``: that fixture opens a
        real QdrantClient, and this assertion is about a NAME. Taking it would
        drag the one pure test in this class onto the network.
        """
        import importlib.util as _ilu  # noqa: PLC0415
        import sys as _sys  # noqa: PLC0415

        from fused_memory.models.scope import Scope  # noqa: PLC0415

        collection = Scope(project_id=sweep_project_id).mem0_collection_name(
            sweep_config.mem0.collection_prefix,
        )

        path = SCRIPT_PATH.parent / 'cleanup_test_collections.py'
        spec = _ilu.spec_from_file_location('cleanup_test_collections', path)
        assert spec is not None and spec.loader is not None
        cleanup = _ilu.module_from_spec(spec)
        monkeypatch.setitem(_sys.modules, 'cleanup_test_collections', cleanup)
        spec.loader.exec_module(cleanup)

        # Asserted against the reaper's OWN constant, not a restated string:
        # a prefix rename over there must not silently strand collections here.
        assert collection.startswith(cleanup.PREFIX)

    @pytest.mark.integration
    @pytest.mark.timeout(300)
    @pytest.mark.asyncio
    @qdrant_skipif()
    @pytest.mark.skipif(
        not os.environ.get('OPENAI_API_KEY'),
        reason='the seeded sweep needs a real embedder',
    )
    async def test_a_superseded_pair_and_a_dangling_pointer_are_both_reported(
        self, sweep_config, sweep_project_id, clean_sweep_collection, tmp_path,
    ):
        from shared.memory_eval_metrics import load_metric_series  # noqa: PLC0415

        from fused_memory.models.scope import Scope  # noqa: PLC0415
        from fused_memory.services.memory_service import MemoryService  # noqa: PLC0415

        m = _mod()
        memory = MemoryService(sweep_config)
        await memory.initialize()
        try:
            # mem0's SQLite history writer is process-shared and xdist-
            # contended; it is not the question under test, and its failure
            # would mask the one that is.
            instance = await memory.mem0._get_instance(Scope(project_id=sweep_project_id))
            instance.db.add_history = lambda *a, **kw: None

            predecessor = await memory.add_memory(
                SEED_PREDECESSOR, category='procedural_knowledge',
                project_id=sweep_project_id, agent_id='e4-sweep-seed',
            )
            predecessor_id = predecessor.memory_ids[0]
            successor = await memory.add_memory(
                SEED_SUCCESSOR, category='procedural_knowledge',
                project_id=sweep_project_id, agent_id='e4-sweep-seed',
                metadata={'supersedes': [predecessor_id]},
            )
            successor_id = successor.memory_ids[0]
            await memory.add_memory(
                SEED_DANGLING_SOURCE, category='procedural_knowledge',
                project_id=sweep_project_id, agent_id='e4-sweep-seed',
                metadata={'corrects': [NEVER_WRITTEN_UUID]},
            )

            outcome = await m.run_sweep(
                memory,
                project_ids=(sweep_project_id,),
                scan_limit=100,
                out_root=tmp_path,
                stamp=STAMP,
            )
        finally:
            await memory.close()

        series = load_metric_series(outcome.metrics_path)

        # (1) The dangling edge is counted AND its target is named -- a bare
        # count would not tell an operator which pointer to go and look at.
        assert _metric(series, 'dangling-pointers').value >= 1
        unresolved = {ref.target for ref in outcome.census.unresolved_refs}
        assert NEVER_WRITTEN_UUID in unresolved
        # ...and the live predecessor is NOT in it: the read really happened.
        assert predecessor_id not in unresolved

        # (2) The seeded supersedes edge produced a tripwire item that passes,
        # keyed by content rather than by the successor's UUID (D5).
        tripwire = _metric(series, 'successor-pointer-present')
        seeded_key = m._tripwire_item_key(m.PointerRef(
            source_id=successor_id, key='supersedes', target=predecessor_id,
            source_content=SEED_SUCCESSOR,
        ))
        # items is Optional on the shared model (only a tripwire carries one),
        # so assert it is populated before indexing rather than narrowing with
        # a cast: an empty items list is itself a schema violation here.
        assert tripwire.items
        matching = [item for item in tripwire.items if item.item_key == seeded_key]
        assert len(matching) == 1
        assert matching[0].passed is True
        assert successor_id not in seeded_key

        # (3) The pair was searched against the real embedder and scored.
        assert {r.successor_id for r in outcome.surfacing.records} == {successor_id}
        assert {r.superseded_id for r in outcome.surfacing.records} == {predecessor_id}

        # Both artifacts landed, under this leaf's own eval_id.
        assert outcome.metrics_path == tmp_path / 'e4-staleness-sweep' / f'metrics-{STAMP}.json'
        assert outcome.report_path is not None and outcome.report_path.exists()


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(pytest.main([__file__]))
