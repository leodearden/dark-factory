"""Unit tests of the pure faces the commit-time ratchet gate calls.

``scripts/check_staged_ratchet_raise.py`` asks what a commit's STAGED baseline
and ledger change against HEAD -- or, finishing a conflicted merge, against both
parents. ``compare_baseline_files`` and ``compare_merged_baseline_files`` give
what the staged image raises; ``own_entry_names`` gives which entry files the
commit adds, and ``retire_legacy_ledger`` holds the one commit that splits the
retired array; ``unrecorded_raises`` gives which raises those additions leave
uncovered. The gate's git plumbing is
pinned end to end in ``test_merge_lane_ratchet_commit_gate.py``. Every image and
ledger here is built from the shared seed in ``_merge_lane_ratchet_fixtures.py``,
and no live measurement is taken.
"""
from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path

import pytest
from _merge_lane_ratchet_fixtures import blank_file_entry, metrics, synthetic_report


def _baseline_image(tmp_path: Path, name: str, report: dict) -> Path:
    """*report* written as committed baseline bytes, for a path-taking face."""
    target = tmp_path / name
    target.write_text(metrics.render_baseline(report), encoding='utf-8')
    return target


def _violation_fields(violation: metrics.Violation) -> tuple[str, str, int, int]:
    """The four fields a ledger record is projected from, message excluded.

    Asserting on the tuple rather than on ``message`` is what keeps these
    tests about the COMPARISON: the wording lives in ``Violation.rose`` and is
    pinned by ``test_merge_lane_ratchet.py::TestViolationShape``, so
    re-deriving it here would be a second copy that drifts.
    """
    return (
        violation.measure,
        violation.key,
        violation.baseline,
        violation.current,
    )


class TestCompareBaselineFiles:
    """The comparison the WRITER cannot make: two committed IMAGES, not a tree.

    ``write_baseline``'s gate compares a fresh report against whatever happens to
    be at the destination, so deleting that destination -- or rendering elsewhere
    and copying over -- leaves it nothing to compare and every measure resets.
    These pin the face that reads two baseline FILES, which is what lets the
    committed-diff auditor ask "did this diff raise anything" without the tree.

    Synthetic images throughout, written with ``render_baseline`` into tmp_path:
    microseconds, and never a live measurement.
    """

    @staticmethod
    def _pair(tmp_path: Path, mutate) -> tuple[Path, Path]:
        """Two images: the seed, and the seed with one measure perturbed."""
        moved = copy.deepcopy(synthetic_report())
        mutate(moved)
        return (
            _baseline_image(tmp_path, 'previous.json', synthetic_report()),
            _baseline_image(tmp_path, 'current.json', moved),
        )

    def test_a_rise_is_reported_per_path_and_in_the_derived_total(
        self, tmp_path: Path
    ) -> None:
        previous, current = self._pair(
            tmp_path, lambda report: report['files']['a.py'].__setitem__('cognitive', 125)
        )

        raises = metrics.compare_baseline_files(previous, current)

        # The per-path violation is field-for-field what the writer's own gate
        # would have produced, so a ledger record derived from either describes
        # the same raise. The total comes along because it is DERIVED -- moving
        # the mass to a new path is the shape that check exists for.
        assert [_violation_fields(v) for v in raises] == [
            _violation_fields(metrics.Violation.rose('cognitive', 'a.py', 120, 125)),
            ('total:cognitive', metrics.CLUSTER_TOTAL_KEY, 150, 155),
        ]

    def test_a_fall_is_clean(self, tmp_path: Path) -> None:
        previous, current = self._pair(
            tmp_path, lambda report: report['files']['a.py'].__setitem__('cognitive', 115)
        )

        # Lowering is the point of a ratchet, so it must never need authorizing.
        assert metrics.compare_baseline_files(previous, current) == []

    def test_byte_identical_images_are_clean(self, tmp_path: Path) -> None:
        previous, current = self._pair(tmp_path, lambda report: None)

        assert previous.read_bytes() == current.read_bytes()
        assert metrics.compare_baseline_files(previous, current) == []

    def test_the_argument_order_cannot_read_clean_when_crossed(
        self, tmp_path: Path
    ) -> None:
        # THE ONE MISTAKE THAT WOULD DISARM THE AUDITOR SILENTLY.
        # ``_measure_raises`` takes (current, previous) while this face takes
        # (previous, current), so a crossed call is a live hazard -- and a
        # crossed call reads CLEAN, which is the failure a gate must never have.
        previous, current = self._pair(
            tmp_path, lambda report: report['files']['a.py'].__setitem__('cognitive', 125)
        )

        assert metrics.compare_baseline_files(previous, current)
        assert metrics.compare_baseline_files(current, previous) == []

    @pytest.mark.parametrize(
        'mutate',
        [
            pytest.param(
                lambda report: report['files']['a.py'].__setitem__('lines', 1005),
                id='lines',
            ),
            pytest.param(
                lambda report: report['files']['a.py'].__setitem__('prose_lines', 460),
                id='prose_lines',
            ),
            pytest.param(
                lambda report: report['files'].__setitem__(
                    'new.py', blank_file_entry(lines=1500, prose_lines=700)
                ),
                id='new-path-at-the-ceiling',
            ),
        ],
    )
    def test_a_size_only_move_between_images_raises_nothing(
        self, tmp_path: Path, mutate
    ) -> None:
        # PRD plans/review-all-s25-scripts-instruments-prd.md D10: size is
        # measured and stored, never ratcheted, per path or in the total.
        previous, current = self._pair(tmp_path, mutate)

        assert metrics.compare_baseline_files(previous, current) == []

    def test_a_new_oversized_path_breaches_the_ceiling(self, tmp_path: Path) -> None:
        # Proves ``_check_ceilings`` is reached, not just the four rise arms: a
        # path ABSENT from the previous image is held to the ceiling, so writing
        # an oversized new file into the baseline cannot grandfather it.
        oversized = metrics.FILE_LINE_CEILING + 1
        previous, current = self._pair(
            tmp_path,
            lambda report: report['files'].__setitem__(
                'new_big.py', blank_file_entry(lines=oversized)
            ),
        )

        raises = metrics.compare_baseline_files(previous, current)

        assert (
            'new_file_over_ceiling',
            'new_big.py',
            metrics.FILE_LINE_CEILING,
            oversized,
        ) in [_violation_fields(v) for v in raises]

    def test_a_missing_image_is_a_named_hard_failure(self, tmp_path: Path) -> None:
        # Inherited from load_baseline, never re-implemented: an unreadable
        # image must never read as an empty-baseline pass (INV-11).
        previous = _baseline_image(tmp_path, 'previous.json', synthetic_report())
        absent = tmp_path / 'gone.json'

        with pytest.raises(metrics.RatchetError) as excinfo:
            metrics.compare_baseline_files(previous, absent)

        assert str(absent) in str(excinfo.value)

    def test_a_malformed_image_is_a_named_hard_failure(self, tmp_path: Path) -> None:
        previous = _baseline_image(tmp_path, 'previous.json', synthetic_report())
        broken = tmp_path / 'broken.json'
        broken.write_text('{"files": {', encoding='utf-8')

        with pytest.raises(metrics.RatchetError) as excinfo:
            metrics.compare_baseline_files(previous, broken)

        message = str(excinfo.value)
        assert str(broken) in message
        assert 'not valid JSON' in message


class TestCompareMergedBaselineFiles:
    """The 3-WAY BOUND a merge commit's staged baseline is compared against.

    A conflicted merge finished with `git commit` stages a baseline descended
    from two parents (esc-3620-11). Read against HEAD alone, every measure
    MERGE_HEAD moved is a raise; read against either parent, keeping one side's
    stale value re-absorbs the other side's lowering. The bound is what git does
    to the file text, applied per measure against the merge base: a measure one
    side moved stands at that side's value, up or down, and where both moved,
    the higher side bounds it.

    The seed's numbers: a.py cognitive 120, b.py cognitive 30, total:cognitive 150.
    """

    @staticmethod
    def _moved(*moves: tuple[str, str, int]) -> dict:
        """The seed with each ``(path, measure, value)`` set in its ``files``."""
        report = synthetic_report()
        for path, measure, value in moves:
            report['files'][path][measure] = value
        return report

    @staticmethod
    def _bounded(
        tmp_path: Path,
        current: dict,
        *,
        base: dict | None,
        ours: dict | None,
        theirs: dict | None,
    ) -> list[tuple[str, str, int, int]]:
        """The raises *current* makes over the bound, as field tuples."""

        def image(name: str, report: dict | None) -> Path | None:
            return None if report is None else _baseline_image(tmp_path, name, report)

        raises = metrics.compare_merged_baseline_files(
            base=image('base.json', base),
            ours=image('ours.json', ours),
            theirs=image('theirs.json', theirs),
            current=_baseline_image(tmp_path, 'current.json', current),
        )
        return [_violation_fields(violation) for violation in raises]

    def test_a_measure_only_theirs_moved_stands_at_theirs_value(
        self, tmp_path: Path
    ) -> None:
        # THE INCIDENT'S SHAPE: ours never touched the baseline, so the bound IS
        # theirs' image -- and the reported baseline is the bound, not HEAD's.
        seed, theirs = synthetic_report(), self._moved(('a.py', 'cognitive', 125))

        assert self._bounded(tmp_path, theirs, base=seed, ours=seed, theirs=theirs) == []
        assert self._bounded(
            tmp_path,
            self._moved(('a.py', 'cognitive', 126)),
            base=seed,
            ours=seed,
            theirs=theirs,
        ) == [
            ('cognitive', 'a.py', 125, 126),
            ('total:cognitive', metrics.CLUSTER_TOTAL_KEY, 155, 156),
        ]

    @pytest.mark.parametrize('lowered_by', ['ours', 'theirs'])
    def test_a_lowering_one_side_made_is_not_reabsorbed_from_the_other(
        self, tmp_path: Path, lowered_by: str
    ) -> None:
        # THE CASE EVERY EITHER-PARENT RULE ADMITS. Keeping the unmoved side's
        # 120 is clean against that side, and in a conflicted merge on main it
        # would land an unrecorded widening.
        seed, lowered = synthetic_report(), self._moved(('a.py', 'cognitive', 115))

        assert self._bounded(
            tmp_path,
            seed,
            base=seed,
            ours=lowered if lowered_by == 'ours' else seed,
            theirs=lowered if lowered_by == 'theirs' else seed,
        ) == [
            ('cognitive', 'a.py', 115, 120),
            ('total:cognitive', metrics.CLUSTER_TOTAL_KEY, 145, 150),
        ]

    def test_own_moves_stand_per_measure_not_per_path(self, tmp_path: Path) -> None:
        # Both sides moved a.py, but different MEASURES of it: taking theirs'
        # whole entry keeps theirs' cognitive lowering and undoes ours'
        # function_local_imports one.
        ours = self._moved(('a.py', 'function_local_imports', 2))
        theirs = self._moved(('a.py', 'cognitive', 110))

        assert self._bounded(
            tmp_path, theirs, base=synthetic_report(), ours=ours, theirs=theirs
        ) == [
            ('function_local_imports', 'a.py', 2, 3),
            ('total:function_local_imports', metrics.CLUSTER_TOTAL_KEY, 3, 4),
        ]

    def test_where_both_sides_moved_a_measure_the_higher_bounds_it(
        self, tmp_path: Path
    ) -> None:
        seed = synthetic_report()
        ours = self._moved(('a.py', 'cognitive', 110))
        theirs = self._moved(('a.py', 'cognitive', 125))

        assert self._bounded(tmp_path, theirs, base=seed, ours=ours, theirs=theirs) == []
        assert self._bounded(
            tmp_path,
            self._moved(('a.py', 'cognitive', 126)),
            base=seed,
            ours=ours,
            theirs=theirs,
        ) == [
            ('cognitive', 'a.py', 125, 126),
            ('total:cognitive', metrics.CLUSTER_TOTAL_KEY, 155, 156),
        ]

    def test_moves_on_different_paths_add_up_in_the_derived_total(
        self, tmp_path: Path
    ) -> None:
        # ONE side moved each path, so each move stands, and the total is
        # DERIVED from the bound: 165. A per-measure max of the parents' own
        # totals would read 160 and refuse the merge.
        ours = self._moved(('a.py', 'cognitive', 125))
        theirs = self._moved(('b.py', 'cognitive', 40))
        merged = self._moved(('a.py', 'cognitive', 125), ('b.py', 'cognitive', 40))

        assert self._bounded(
            tmp_path, merged, base=synthetic_report(), ours=ours, theirs=theirs
        ) == []

    def test_a_path_one_side_deleted_stays_deleted(self, tmp_path: Path) -> None:
        seed, ours = synthetic_report(), synthetic_report()
        del ours['files']['b.py']

        assert ('total:cognitive', metrics.CLUSTER_TOTAL_KEY, 120, 150) in self._bounded(
            tmp_path, seed, base=seed, ours=ours, theirs=seed
        )

    def test_a_deletion_does_not_undo_the_other_sides_move(
        self, tmp_path: Path
    ) -> None:
        # git's modify/delete conflict: the modification stands.
        ours = synthetic_report()
        del ours['files']['b.py']
        theirs = self._moved(('b.py', 'cognitive', 20))

        assert self._bounded(
            tmp_path, theirs, base=synthetic_report(), ours=ours, theirs=theirs
        ) == []

    def test_an_absent_base_bounds_each_measure_by_the_higher_parent(
        self, tmp_path: Path
    ) -> None:
        # No common ancestor, or no baseline at it: git merges such histories
        # against the empty tree, so each side moved every measure it holds.
        ours = synthetic_report()
        theirs = self._moved(('a.py', 'cognitive', 125), ('b.py', 'cognitive', 20))

        assert self._bounded(
            tmp_path,
            self._moved(('a.py', 'cognitive', 125)),
            base=None,
            ours=ours,
            theirs=theirs,
        ) == []
        assert ('cognitive', 'b.py', 30, 31) in self._bounded(
            tmp_path,
            self._moved(('a.py', 'cognitive', 125), ('b.py', 'cognitive', 31)),
            base=None,
            ours=ours,
            theirs=theirs,
        )

    def test_an_absent_parent_image_moved_nothing(self, tmp_path: Path) -> None:
        # A parent without a baseline must not read as having lowered every
        # measure to nothing, which would refuse a resolver who keeps theirs.
        seed, theirs = synthetic_report(), self._moved(('a.py', 'cognitive', 125))

        assert self._bounded(tmp_path, theirs, base=seed, ours=None, theirs=theirs) == []
        assert ('cognitive', 'a.py', 125, 126) in self._bounded(
            tmp_path,
            self._moved(('a.py', 'cognitive', 126)),
            base=seed,
            ours=None,
            theirs=theirs,
        )

    def test_a_size_only_excess_over_the_bound_raises_nothing(
        self, tmp_path: Path
    ) -> None:
        seed, theirs = synthetic_report(), self._moved(('a.py', 'lines', 1005))

        assert self._bounded(
            tmp_path,
            self._moved(('a.py', 'lines', 1100)),
            base=seed,
            ours=seed,
            theirs=theirs,
        ) == []

    def test_a_name_set_both_sides_moved_is_bounded_by_the_side_naming_more(
        self, tmp_path: Path
    ) -> None:
        # DISTINCT names, the unit the per-path comparison counts in.
        seed, ours, theirs = synthetic_report(), synthetic_report(), synthetic_report()
        ours['tests']['t1.py']['patch_targets'] = ['foo', 'bar', 'qux']
        theirs['tests']['t1.py']['patch_targets'] = ['foo']
        widened = copy.deepcopy(ours)
        widened['tests']['t1.py']['patch_targets'].append('zap')

        assert self._bounded(tmp_path, ours, base=seed, ours=ours, theirs=theirs) == []
        assert ('patch_targets', 't1.py', 3, 4) in self._bounded(
            tmp_path, widened, base=seed, ours=ours, theirs=theirs
        )


def _ledger_record(task_id: str, reason: str = 'net-additive work') -> dict:
    """One ledger entry, built by the real writer rather than hand-written."""
    return metrics.authorization_record(
        metrics.RaiseAuthorization(task_id=task_id, reason=reason),
        [metrics.Violation.rose('lines', 'a.py', 1000, 1005)],
    )


def _listing(*records: dict) -> dict[str, str]:
    """Entry file name -> content identity, as the gate lists a tree or the index."""
    return {metrics.entry_filename(record): metrics.render_entry(record) for record in records}


class TestOwnEntryNames:
    """The ledger delta, and what makes the append-only promise checkable.

    The ledger is a directory of one file per authorization, so a commit's own
    entries are the files no parent holds, and append-only is a comparison of
    listings: every file a parent holds must be staged unchanged.
    """

    def test_an_unchanged_ledger_adds_nothing(self) -> None:
        listing = _listing(_ledger_record('5485'))

        assert metrics.own_entry_names([listing], dict(listing)) == []

    def test_the_added_files_are_returned_sorted(self) -> None:
        history = _ledger_record('5485')
        first, second = _ledger_record('5723'), _ledger_record('5722')

        added = metrics.own_entry_names(
            [_listing(history)], _listing(history, first, second)
        )

        assert added == sorted(
            [metrics.entry_filename(first), metrics.entry_filename(second)]
        )

    def test_with_no_recorded_entries_every_staged_file_is_own(self) -> None:
        record = _ledger_record('5722')

        assert metrics.own_entry_names([{}], _listing(record)) == [
            metrics.entry_filename(record)
        ]

    def test_dropping_a_recorded_entry_is_refused_by_name(self) -> None:
        kept, dropped = _ledger_record('5485'), _ledger_record('5675')

        with pytest.raises(metrics.AppendOnlyViolation) as excinfo:
            metrics.own_entry_names([_listing(kept, dropped)], _listing(kept))

        assert 'append-only' in str(excinfo.value)
        assert metrics.entry_filename(dropped) in str(excinfo.value)

    def test_rewriting_a_recorded_entry_under_its_name_is_refused(self) -> None:
        # Same NAME, different content: a count of files would pass this.
        history = _ledger_record('5485')
        staged = {metrics.entry_filename(history): 'a different blob'}

        with pytest.raises(metrics.AppendOnlyViolation):
            metrics.own_entry_names([_listing(history)], staged)

    def test_a_merge_takes_either_parents_files_as_history(self) -> None:
        history = _ledger_record('5485')
        ours_own, theirs_own = _ledger_record('5722'), _ledger_record('3620')
        merge_own = _ledger_record('5792')

        added = metrics.own_entry_names(
            [_listing(history, ours_own), _listing(history, theirs_own)],
            _listing(history, ours_own, theirs_own, merge_own),
        )

        # A file either parent recorded is HISTORY: counting it as the merge's
        # own would be the standing permission LEDGER_README forbids.
        assert added == [metrics.entry_filename(merge_own)]

    def test_a_merge_dropping_the_other_parents_entry_is_refused(self) -> None:
        theirs_own = _ledger_record('3620')

        with pytest.raises(metrics.AppendOnlyViolation):
            metrics.own_entry_names([{}, _listing(theirs_own)], {})


class TestRetireLegacyLedger:
    """The one-commit, set-equal rule for splitting the retired array into files."""

    @staticmethod
    def _entries() -> tuple[dict, dict, dict]:
        return _ledger_record('5485'), _ledger_record('5675'), _ledger_record('5722')

    def test_with_nothing_retired_own_entries_pass_through(self) -> None:
        first, second, _third = self._entries()

        assert metrics.retire_legacy_ledger([], _listing(first, second), [first, second]) == [
            first,
            second,
        ]

    def test_an_exact_split_passes_and_covers_nothing(self) -> None:
        entries = list(self._entries())

        # Split entries are HISTORY: the commit that moves them authorizes nothing.
        assert metrics.retire_legacy_ledger(entries, _listing(*entries), entries) == []

    def test_dropping_an_entry_in_the_split_is_refused(self) -> None:
        first, second, third = self._entries()

        with pytest.raises(metrics.AppendOnlyViolation) as excinfo:
            metrics.retire_legacy_ledger(
                [first, second, third], _listing(first, second), [first, second]
            )

        assert metrics.entry_filename(third) in str(excinfo.value)

    def test_rewriting_an_entry_in_the_split_is_refused(self) -> None:
        first, second, third = self._entries()
        rewritten = {**third, 'reason': 'actually it was a refactor'}
        staged = [first, second, rewritten]

        with pytest.raises(metrics.AppendOnlyViolation):
            metrics.retire_legacy_ledger([first, second, third], _listing(*staged), staged)

    def test_an_authorization_of_its_own_in_the_split_is_refused(self) -> None:
        first, second, fresh = self._entries()
        staged = [first, second, fresh]

        with pytest.raises(metrics.AppendOnlyViolation) as excinfo:
            metrics.retire_legacy_ledger([first, second], _listing(*staged), staged)

        assert metrics.entry_filename(fresh) in str(excinfo.value)

    def test_a_merge_may_find_the_retired_entries_already_split_by_its_other_parent(
        self,
    ) -> None:
        # A branch still carrying the array merges a main that split it: main's
        # files are staged as history, only the branch's own entry is new here.
        shared_history, branch_own, _other = self._entries()

        assert metrics.retire_legacy_ledger(
            [shared_history, branch_own], _listing(shared_history, branch_own), [branch_own]
        ) == []


class TestLegacyLedgerEntries:
    @pytest.mark.parametrize(
        'text',
        ['{"raises": ', '[]', '{"raises": {}}', '{}', '{"raises": [7]}'],
        ids=['truncated', 'array', 'raises-not-list', 'raises-missing', 'entry-not-object'],
    )
    def test_a_malformed_retired_array_is_a_named_hard_failure(self, text: str) -> None:
        # Its entries are history being moved: one that cannot be read cannot be
        # checked into a file, and must never read as "nothing to move".
        with pytest.raises(metrics.RatchetError) as excinfo:
            metrics.legacy_ledger_entries(text)

        assert metrics.LEGACY_LEDGER_RELPATH in str(excinfo.value)

    def test_the_entries_are_returned_in_order(self) -> None:
        first, second = _ledger_record('5485'), _ledger_record('5722')
        text = json.dumps({'_README': 'x', 'schema_version': 1, 'raises': [first, second]})

        assert metrics.legacy_ledger_entries(text) == [first, second]


class TestUnrecordedRaises:
    """Which measured raises THIS commit's own ledger entries do not cover.

    The inverse of ``authorization_record``: the writer projects Violations into
    a record, this asks whether records cover a Violation. Records here are
    built by calling ``authorization_record`` for real rather than hand-written,
    so what is pinned is the round trip against the actual producer.

    THE RULE: per ``(measure, key)``, the rise may not exceed the SUM of the
    covering rows' ``current - baseline``; a ceiling breach is covered by name.
    """

    @staticmethod
    def _record(raises: list[metrics.Violation], task_id: str = '5722') -> dict:
        return metrics.authorization_record(
            metrics.RaiseAuthorization(task_id=task_id, reason=f'reason {task_id}'),
            raises,
        )

    @staticmethod
    def _lines_raise() -> metrics.Violation:
        return metrics.Violation.rose('lines', 'a.py', 1000, 1005)

    @staticmethod
    def _cognitive_raise() -> metrics.Violation:
        return metrics.Violation.rose('cognitive', 'a.py', 120, 127)

    def test_a_record_derived_from_the_raises_covers_them_all(self) -> None:
        raises = [self._lines_raise(), self._cognitive_raise()]

        assert metrics.unrecorded_raises(raises, [self._record(raises)]) == []

    def test_no_records_leaves_every_raise_unrecorded_in_order(self) -> None:
        raises = [self._lines_raise(), self._cognitive_raise()]

        # The common refusal: a commit that staged a raising baseline and never
        # recorded an entry. Order is the input's, so the refusal message is
        # diffable run to run.
        assert metrics.unrecorded_raises(raises, []) == raises

    def test_a_record_authorizing_less_than_the_rise_covers_nothing(self) -> None:
        # A stale or hand-written entry names the right measure and key but a
        # smaller rise than the one this commit makes.
        stale = self._record([metrics.Violation.rose('lines', 'a.py', 1000, 1002)])
        raises = [self._lines_raise()]

        assert metrics.unrecorded_raises(raises, [stale]) == raises

    def test_a_record_for_an_unrelated_measure_covers_nothing(self) -> None:
        raises = [self._lines_raise()]

        unrelated = self._record([self._cognitive_raise()])

        assert metrics.unrecorded_raises(raises, [unrelated]) == raises

    def test_records_of_different_measures_union(self) -> None:
        lines, cognitive = self._lines_raise(), self._cognitive_raise()
        records = [self._record([lines]), self._record([cognitive], task_id='5723')]

        assert metrics.unrecorded_raises([lines, cognitive], records) == []

    def test_records_of_one_key_sum(self) -> None:
        # An agent who runs --write-baseline --authorize-raise TWICE in one
        # commit records b->m and m->f, while the commit's delta reads b->f;
        # a branch landing several authorizing commits sums the same way.
        records = [
            self._record([metrics.Violation.rose('lines', 'a.py', 1000, 1002)]),
            self._record([metrics.Violation.rose('lines', 'a.py', 1002, 1005)], '5723'),
        ]

        assert metrics.unrecorded_raises([self._lines_raise()], records) == []

    def test_three_partial_records_cover_exactly_their_sum(self) -> None:
        # No single record reaches the rise; only their SUM does. One count
        # short of it, the same records refuse.
        records = [
            self._record([metrics.Violation.rose('lines', 'a.py', 1000, 1001)]),
            self._record([metrics.Violation.rose('lines', 'a.py', 1001, 1002)], '5723'),
            self._record([metrics.Violation.rose('lines', 'a.py', 1002, 1005)], '5724'),
        ]

        assert metrics.unrecorded_raises([self._lines_raise()], records) == []
        assert metrics.unrecorded_raises(
            [metrics.Violation.rose('lines', 'a.py', 1000, 1006)], records
        ) != []

    def test_derived_total_rows_sum_across_records_too(self) -> None:
        total = metrics.CLUSTER_TOTAL_KEY
        records = [
            self._record([metrics.Violation.rose('total:lines', total, 1200, 1202)]),
            self._record([metrics.Violation.rose('total:lines', total, 1300, 1303)], '5723'),
        ]

        landed = metrics.Violation.rose('total:lines', total, 1400, 1405)
        assert metrics.unrecorded_raises([landed], records) == []

    def test_one_record_of_part_of_the_rise_does_not_cover_it(self) -> None:
        # The other half of the sum: the midpoint record alone authorized 3, and
        # the commit raised 5. (Under the retired absolute rule its matching
        # `current` covered the whole rise.)
        midpoint = self._record([metrics.Violation.rose('lines', 'a.py', 1002, 1005)])

        assert metrics.unrecorded_raises([self._lines_raise()], [midpoint]) == [
            self._lines_raise()
        ]

    def test_a_record_measured_from_another_base_covers_the_same_rise(self) -> None:
        # A derived total authorized against base X covers the same raise
        # landing on X + a sibling's lines -- the case the absolute rule refused.
        landed = metrics.Violation.rose('total:lines', metrics.CLUSTER_TOTAL_KEY, 1300, 1305)
        authored = self._record(
            [metrics.Violation.rose('total:lines', metrics.CLUSTER_TOTAL_KEY, 1200, 1205)]
        )

        assert metrics.unrecorded_raises([landed], [authored]) == []

    def test_a_ceiling_breach_is_covered_by_naming_it(self) -> None:
        breach = metrics.Violation(
            measure=metrics.NEW_FUNCTION_OVER_CEILING,
            key='a.py::new',
            baseline=metrics.NEW_FUNCTION_COGNITIVE_CEILING,
            current=40,
            message='breach',
        )
        named = self._record(
            [dataclasses.replace(breach, current=19)]
        )

        assert metrics.unrecorded_raises([breach], [named]) == []
        assert metrics.unrecorded_raises([breach], [self._record([self._lines_raise()])]) == [
            breach
        ]

    @pytest.mark.parametrize(
        'measures', [None, 'lines', 7], ids=['missing', 'string', 'int']
    )
    def test_a_record_without_a_measures_list_covers_nothing_and_does_not_crash(
        self, measures: object
    ) -> None:
        # Hand-mutated on purpose -- authorization_record cannot produce these.
        # A ledger a human edited badly must REFUSE the commit, not raise a
        # TypeError out of the gate: a crash reads as a broken instrument.
        record = self._record([self._lines_raise()])
        if measures is None:
            del record['measures']
        else:
            record['measures'] = measures
        raises = [self._lines_raise()]

        assert metrics.unrecorded_raises(raises, [record]) == raises

    @pytest.mark.parametrize('current', ['1005', None, True], ids=['string', 'null', 'bool'])
    def test_a_row_whose_numbers_are_not_counts_authorizes_nothing(
        self, current: object
    ) -> None:
        record = self._record([self._lines_raise()])
        record['measures'][0]['current'] = current

        assert metrics.unrecorded_raises([self._lines_raise()], [record]) == [
            self._lines_raise()
        ]
