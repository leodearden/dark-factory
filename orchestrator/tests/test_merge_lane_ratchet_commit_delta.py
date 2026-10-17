"""Unit tests of the five pure faces the commit-time ratchet gate calls.

``scripts/check_staged_ratchet_raise.py`` asks what a commit's STAGED baseline
and ledger change against HEAD -- or, finishing a conflicted merge, against both
parents. ``compare_baseline_files`` and ``compare_merged_baseline_files`` give
what the staged image raises; ``ledger_appended_entries`` and
``ledger_merged_entries`` give what the staged ledger adds; ``unrecorded_raises``
gives which raises those additions leave uncovered. The gate's git plumbing is
pinned end to end in ``test_merge_lane_ratchet_commit_gate.py``. Every image and
ledger here is built from the shared seed in ``_merge_lane_ratchet_fixtures.py``,
and no live measurement is taken.
"""
from __future__ import annotations

import copy
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
            tmp_path, lambda report: report['files']['a.py'].__setitem__('lines', 1005)
        )

        raises = metrics.compare_baseline_files(previous, current)

        # The per-path violation is field-for-field what the writer's own gate
        # would have produced, so a ledger record derived from either describes
        # the same raise. The total comes along because it is DERIVED -- moving
        # the mass to a new path is the shape that check exists for.
        assert [_violation_fields(v) for v in raises] == [
            _violation_fields(metrics.Violation.rose('lines', 'a.py', 1000, 1005)),
            ('total:lines', metrics.CLUSTER_TOTAL_KEY, 1200, 1205),
        ]

    def test_a_fall_is_clean(self, tmp_path: Path) -> None:
        previous, current = self._pair(
            tmp_path, lambda report: report['files']['a.py'].__setitem__('lines', 995)
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
            tmp_path, lambda report: report['files']['a.py'].__setitem__('lines', 1005)
        )

        assert metrics.compare_baseline_files(previous, current)
        assert metrics.compare_baseline_files(current, previous) == []

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

    The seed's numbers: a.py lines 1000, b.py lines 200, total:lines 1200.
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
        seed, theirs = synthetic_report(), self._moved(('a.py', 'lines', 1005))

        assert self._bounded(tmp_path, theirs, base=seed, ours=seed, theirs=theirs) == []
        assert self._bounded(
            tmp_path,
            self._moved(('a.py', 'lines', 1006)),
            base=seed,
            ours=seed,
            theirs=theirs,
        ) == [
            ('lines', 'a.py', 1005, 1006),
            ('total:lines', metrics.CLUSTER_TOTAL_KEY, 1205, 1206),
        ]

    @pytest.mark.parametrize('lowered_by', ['ours', 'theirs'])
    def test_a_lowering_one_side_made_is_not_reabsorbed_from_the_other(
        self, tmp_path: Path, lowered_by: str
    ) -> None:
        # THE CASE EVERY EITHER-PARENT RULE ADMITS. Keeping the unmoved side's
        # 1000 is clean against that side, and in a conflicted merge on main it
        # would land an unrecorded widening.
        seed, lowered = synthetic_report(), self._moved(('a.py', 'lines', 995))

        assert self._bounded(
            tmp_path,
            seed,
            base=seed,
            ours=lowered if lowered_by == 'ours' else seed,
            theirs=lowered if lowered_by == 'theirs' else seed,
        ) == [
            ('lines', 'a.py', 995, 1000),
            ('total:lines', metrics.CLUSTER_TOTAL_KEY, 1195, 1200),
        ]

    def test_own_moves_stand_per_measure_not_per_path(self, tmp_path: Path) -> None:
        # Both sides moved a.py, but different MEASURES of it: taking theirs'
        # whole entry keeps theirs' cognitive lowering and undoes ours' lines one.
        ours = self._moved(('a.py', 'lines', 995))
        theirs = self._moved(('a.py', 'cognitive', 110))

        assert self._bounded(
            tmp_path, theirs, base=synthetic_report(), ours=ours, theirs=theirs
        ) == [
            ('lines', 'a.py', 995, 1000),
            ('total:lines', metrics.CLUSTER_TOTAL_KEY, 1195, 1200),
        ]

    def test_where_both_sides_moved_a_measure_the_higher_bounds_it(
        self, tmp_path: Path
    ) -> None:
        seed = synthetic_report()
        ours = self._moved(('a.py', 'lines', 990))
        theirs = self._moved(('a.py', 'lines', 1005))

        assert self._bounded(tmp_path, theirs, base=seed, ours=ours, theirs=theirs) == []
        assert self._bounded(
            tmp_path,
            self._moved(('a.py', 'lines', 1006)),
            base=seed,
            ours=ours,
            theirs=theirs,
        ) == [
            ('lines', 'a.py', 1005, 1006),
            ('total:lines', metrics.CLUSTER_TOTAL_KEY, 1205, 1206),
        ]

    def test_moves_on_different_paths_add_up_in_the_derived_total(
        self, tmp_path: Path
    ) -> None:
        # ONE side moved each path, so each move stands, and the total is
        # DERIVED from the bound: 1215. A per-measure max of the parents' own
        # totals would read 1210 and refuse the merge.
        ours = self._moved(('a.py', 'lines', 1005))
        theirs = self._moved(('b.py', 'lines', 210))
        merged = self._moved(('a.py', 'lines', 1005), ('b.py', 'lines', 210))

        assert self._bounded(
            tmp_path, merged, base=synthetic_report(), ours=ours, theirs=theirs
        ) == []

    def test_a_path_one_side_deleted_stays_deleted(self, tmp_path: Path) -> None:
        seed, ours = synthetic_report(), synthetic_report()
        del ours['files']['b.py']

        assert ('total:lines', metrics.CLUSTER_TOTAL_KEY, 1000, 1200) in self._bounded(
            tmp_path, seed, base=seed, ours=ours, theirs=seed
        )

    def test_a_deletion_does_not_undo_the_other_sides_move(
        self, tmp_path: Path
    ) -> None:
        # git's modify/delete conflict: the modification stands.
        ours = synthetic_report()
        del ours['files']['b.py']
        theirs = self._moved(('b.py', 'lines', 190))

        assert self._bounded(
            tmp_path, theirs, base=synthetic_report(), ours=ours, theirs=theirs
        ) == []

    def test_an_absent_base_bounds_each_measure_by_the_higher_parent(
        self, tmp_path: Path
    ) -> None:
        # No common ancestor, or no baseline at it: git merges such histories
        # against the empty tree, so each side moved every measure it holds.
        ours = synthetic_report()
        theirs = self._moved(('a.py', 'lines', 1005), ('b.py', 'lines', 190))

        assert self._bounded(
            tmp_path,
            self._moved(('a.py', 'lines', 1005)),
            base=None,
            ours=ours,
            theirs=theirs,
        ) == []
        assert ('lines', 'b.py', 200, 201) in self._bounded(
            tmp_path,
            self._moved(('a.py', 'lines', 1005), ('b.py', 'lines', 201)),
            base=None,
            ours=ours,
            theirs=theirs,
        )

    def test_an_absent_parent_image_moved_nothing(self, tmp_path: Path) -> None:
        # A parent without a baseline must not read as having lowered every
        # measure to nothing, which would refuse a resolver who keeps theirs.
        seed, theirs = synthetic_report(), self._moved(('a.py', 'lines', 1005))

        assert self._bounded(tmp_path, theirs, base=seed, ours=None, theirs=theirs) == []
        assert ('lines', 'a.py', 1005, 1006) in self._bounded(
            tmp_path,
            self._moved(('a.py', 'lines', 1006)),
            base=seed,
            ours=None,
            theirs=theirs,
        )

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


def _ledger(*records: dict) -> dict:
    return {**metrics.empty_ledger(), 'raises': list(records)}


class TestLedgerAppendedEntries:
    """The ledger delta, and what makes append_authorization's promise checkable.

    ``append_authorization``'s docstring promises that "a reviewer reading the
    file reads every raise this baseline has ever absorbed". That is a claim
    about every FUTURE writer of the file, not just about that function, and
    nothing enforced it: a commit could quietly drop or rewrite a historical
    entry and the promise would stay standing and false. These pin the reader
    that checks it.
    """

    def test_an_unchanged_ledger_appended_nothing(self) -> None:
        ledger = _ledger(_ledger_record('5485'))

        assert metrics.ledger_appended_entries(ledger, copy.deepcopy(ledger)) == []

    def test_the_appended_suffix_is_returned_in_order(self) -> None:
        history = _ledger_record('5485')
        first, second = _ledger_record('5722'), _ledger_record('5723')

        appended = metrics.ledger_appended_entries(
            _ledger(history), _ledger(history, first, second)
        )

        assert appended == [first, second]

    def test_the_day_one_shape_is_the_whole_list(self) -> None:
        # Previous is the fail-CLOSED empty ledger -- the state a commit that
        # authorizes the very first raise starts from.
        record = _ledger_record('5722')

        assert metrics.ledger_appended_entries(
            metrics.empty_ledger(), _ledger(record)
        ) == [record]

    def test_dropping_a_historical_entry_is_refused_by_name(self) -> None:
        history = [_ledger_record('5485'), _ledger_record('5675')]

        with pytest.raises(metrics.RatchetError) as excinfo:
            metrics.ledger_appended_entries(_ledger(*history), _ledger(history[0]))

        message = str(excinfo.value)
        assert 'append-only' in message
        # HOW MANY went missing, not merely that something did: the number is
        # what tells a reviewer whether this was one bad rebase or a wipe.
        assert '1' in message

    def test_rewriting_an_entry_in_place_is_refused(self) -> None:
        # LENGTH EQUALITY IS NOT PREFIX EQUALITY. A rewrite keeps the count
        # identical, so any check that compared lengths would pass it -- and a
        # rewritten `reason` is exactly how a raise stops reading as what it was.
        history = _ledger_record('5485')
        forged = _ledger_record('5485', reason='actually it was a refactor')

        with pytest.raises(metrics.RatchetError):
            metrics.ledger_appended_entries(_ledger(history), _ledger(forged))

    def test_reordering_history_is_refused(self) -> None:
        first, second = _ledger_record('5485'), _ledger_record('5675')

        with pytest.raises(metrics.RatchetError):
            metrics.ledger_appended_entries(
                _ledger(first, second), _ledger(second, first)
            )


class TestLedgerMergedEntries:
    """The ledger delta of a MERGE commit, read against BOTH parents' histories.

    A conflicted merge finished with `git commit` runs pre-commit with
    MERGE_HEAD set (esc-3620-11), so its staged ledger descends from two
    recorded histories, not one. Each parent's entries must survive whole AND
    in that parent's own order; the two histories may interleave, the way git's
    merge of the text interleaves them, but nothing foreign may sit among them.
    Only what follows both histories and belongs to NEITHER parent is the
    merge's OWN -- the one thing that may cover a raise the merge makes.
    """

    @staticmethod
    def _entries() -> tuple[dict, dict, dict]:
        """History both parents share, then one entry only ours / only theirs added."""
        return _ledger_record('5485'), _ledger_record('5722'), _ledger_record('3620')

    def test_a_parent_without_a_ledger_constrains_nothing(self) -> None:
        # THE INCIDENT'S SHAPE: the task branch predates the ledger, so HEAD
        # reads as the fail-closed empty one and main's entries are history.
        _history, _ours_own, theirs_own = self._entries()

        assert metrics.ledger_merged_entries(
            metrics.empty_ledger(), _ledger(theirs_own), _ledger(theirs_own)
        ) == []

    def test_one_sides_additions_are_history_not_the_merges_own(self) -> None:
        history, _ours_own, theirs_own = self._entries()

        assert metrics.ledger_merged_entries(
            _ledger(history),
            _ledger(history, theirs_own),
            _ledger(history, theirs_own),
        ) == []

    @pytest.mark.parametrize(
        'theirs_first', [True, False], ids=['theirs-then-ours', 'ours-then-theirs']
    )
    def test_both_sides_additions_are_accepted_in_either_block_order(
        self, theirs_first: bool
    ) -> None:
        # A resolver facing a both-sides-appended conflict keeps both entries in
        # whichever order the conflict shows them; either interleaving keeps
        # each parent's entries unchanged and in that parent's own order.
        history, ours_own, theirs_own = self._entries()
        blocks = [theirs_own, ours_own] if theirs_first else [ours_own, theirs_own]

        assert metrics.ledger_merged_entries(
            _ledger(history, ours_own),
            _ledger(history, theirs_own),
            _ledger(history, *blocks),
        ) == []

    def test_the_merges_own_additions_are_returned_in_order(self) -> None:
        history, ours_own, theirs_own = self._entries()
        first, second = _ledger_record('5792'), _ledger_record('5793')

        appended = metrics.ledger_merged_entries(
            _ledger(history, ours_own),
            _ledger(history, theirs_own),
            _ledger(history, theirs_own, ours_own, first, second),
        )

        assert appended == [first, second]

    @pytest.mark.parametrize('dropped', ['ours', 'theirs'])
    def test_dropping_an_entry_either_side_recorded_is_refused(
        self, dropped: str
    ) -> None:
        # BOTH histories, not either: a merge resolved on main that kept only
        # the task's entries would silently drop main's recorded ones.
        history, ours_own, theirs_own = self._entries()
        kept = theirs_own if dropped == 'ours' else ours_own

        with pytest.raises(metrics.AppendOnlyViolation) as excinfo:
            metrics.ledger_merged_entries(
                _ledger(history, ours_own),
                _ledger(history, theirs_own),
                _ledger(history, kept),
            )

        assert 'append-only' in str(excinfo.value)

    def test_rewriting_an_entry_one_side_recorded_is_refused(self) -> None:
        # Same count, changed `reason`: length equality is not prefix equality
        # here either.
        history, ours_own, theirs_own = self._entries()
        forged = _ledger_record('3620', reason='actually it was a refactor')

        with pytest.raises(metrics.AppendOnlyViolation):
            metrics.ledger_merged_entries(
                _ledger(history, ours_own),
                _ledger(history, theirs_own),
                _ledger(history, forged, ours_own),
            )

    @staticmethod
    def _merged_before() -> tuple[dict, dict, dict, dict]:
        """Shared history, ours' own, theirs' own, and theirs' next addition.

        The shape of a branch that merged main once, kept ours-then-theirs, and
        now merges main again after main appended one more entry.
        """
        return (
            _ledger_record('5485'),
            _ledger_record('5792'),
            _ledger_record('3620'),
            _ledger_record('3790'),
        )

    def test_a_branch_that_kept_ours_then_theirs_can_merge_again(self) -> None:
        # The reviewer's case, measured: git merges [h,o,t] and [h,t,t2] over
        # base [h,t] cleanly to [h,o,t,t2]. No block order reproduces that.
        history, ours_own, theirs_own, theirs_next = self._merged_before()

        assert metrics.ledger_merged_entries(
            _ledger(history, ours_own, theirs_own),
            _ledger(history, theirs_own, theirs_next),
            _ledger(history, ours_own, theirs_own, theirs_next),
        ) == []

    def test_a_branch_merging_after_another_landed_interleaves(self) -> None:
        # Main now carries another branch's ours-then-theirs order, and this
        # branch appended past the entry the two share.
        history, other_own, shared, _theirs_next = self._merged_before()
        ours_next = _ledger_record('5801')

        assert metrics.ledger_merged_entries(
            _ledger(history, shared, ours_next),
            _ledger(history, other_own, shared),
            _ledger(history, other_own, shared, ours_next),
        ) == []

    def test_the_merges_own_entries_follow_an_interleaved_history(self) -> None:
        history, ours_own, theirs_own, theirs_next = self._merged_before()
        own = _ledger_record('5900')

        assert metrics.ledger_merged_entries(
            _ledger(history, ours_own, theirs_own),
            _ledger(history, theirs_own, theirs_next),
            _ledger(history, ours_own, theirs_own, theirs_next, own),
        ) == [own]

    def test_reordering_one_parents_own_entries_is_refused(self) -> None:
        # Ours recorded its own entry BEFORE the shared one; moving it after
        # theirs' history rewrites the order ours recorded.
        history, ours_own, theirs_own, theirs_next = self._merged_before()

        with pytest.raises(metrics.AppendOnlyViolation) as excinfo:
            metrics.ledger_merged_entries(
                _ledger(history, ours_own, theirs_own),
                _ledger(history, theirs_own, theirs_next),
                _ledger(history, theirs_own, theirs_next, ours_own),
            )

        assert 'append-only' in str(excinfo.value)

    def test_a_foreign_entry_inside_the_histories_is_refused(self) -> None:
        history, ours_own, theirs_own, theirs_next = self._merged_before()
        foreign = _ledger_record('5900')

        with pytest.raises(metrics.AppendOnlyViolation):
            metrics.ledger_merged_entries(
                _ledger(history, ours_own, theirs_own),
                _ledger(history, theirs_own, theirs_next),
                _ledger(history, foreign, ours_own, theirs_own, theirs_next),
            )

    @pytest.mark.parametrize('with_own', [False, True], ids=['copy-only', 'copy-then-own'])
    def test_a_copied_parent_entry_is_not_the_merges_own(self, with_own: bool) -> None:
        # A duplicate of a recorded entry after both histories is still
        # HISTORY: counting it as coverage is the standing permission
        # LEDGER_README forbids.
        history, ours_own, theirs_own, _theirs_next = self._merged_before()
        own = [_ledger_record('5900')] if with_own else []

        assert metrics.ledger_merged_entries(
            _ledger(history, ours_own),
            _ledger(history, theirs_own),
            _ledger(history, ours_own, theirs_own, theirs_own, *own),
        ) == own


class TestUnrecordedRaises:
    """Which measured raises THIS commit's ledger entries do not name.

    The inverse of ``authorization_record``: the writer projects Violations into
    a record, this asks whether a record covers a Violation. Records here are
    built by calling ``authorization_record`` for real rather than hand-written,
    so what is pinned is the round trip against the actual producer -- a
    hand-written dict would pin this module's idea of the vocabulary instead of
    the writer's, which is the drift the shared field names exist to prevent.
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
        # touched the ledger. Order is the input's, so the refusal message is
        # diffable run to run.
        assert metrics.unrecorded_raises(raises, []) == raises

    def test_a_stale_record_naming_a_different_landing_value_covers_nothing(
        self,
    ) -> None:
        # THE CASE THIS FUNCTION EXISTS FOR. A hand-written or stale entry names
        # the right measure and the right key, so any (measure, key) check would
        # wave it through -- while the baseline it accompanies landed somewhere
        # else entirely. `current` is the discriminator.
        stale = self._record([metrics.Violation.rose('lines', 'a.py', 1000, 1002)])
        raises = [self._lines_raise()]

        assert metrics.unrecorded_raises(raises, [stale]) == raises

    def test_a_record_for_an_unrelated_measure_covers_nothing(self) -> None:
        raises = [self._lines_raise()]

        unrelated = self._record([self._cognitive_raise()])

        assert metrics.unrecorded_raises(raises, [unrelated]) == raises

    def test_two_records_union_rather_than_the_last_one_winning(self) -> None:
        lines, cognitive = self._lines_raise(), self._cognitive_raise()

        # Two authorized writes in one commit append two records; coverage is
        # the union of both, not whichever landed last.
        records = [self._record([lines]), self._record([cognitive], task_id='5723')]

        assert metrics.unrecorded_raises([lines, cognitive], records) == []

    def test_a_differing_baseline_still_covers_when_current_matches(self) -> None:
        # DELIBERATE: the triple is (measure, key, current), not the full
        # four-tuple. An agent who runs --write-baseline --authorize-raise TWICE
        # in one commit records b->m and m->f, while the HEAD-to-staged delta
        # reads b->f. Requiring `baseline` to match would refuse that correctly
        # authorized commit.
        midpoint = self._record([metrics.Violation.rose('lines', 'a.py', 1002, 1005)])

        assert metrics.unrecorded_raises([self._lines_raise()], [midpoint]) == []

    @pytest.mark.parametrize(
        'measures', [None, 'lines', 7], ids=['missing', 'string', 'int']
    )
    def test_a_record_without_a_measures_list_covers_nothing_and_does_not_crash(
        self, measures: object
    ) -> None:
        # Hand-mutated on purpose -- authorization_record cannot produce these.
        # A ledger a human edited badly must REFUSE the commit, not raise a
        # TypeError out of the gate: a crash reads as a broken instrument and
        # sends the reader hunting the wrong thing.
        record = self._record([self._lines_raise()])
        if measures is None:
            del record['measures']
        else:
            record['measures'] = measures
        raises = [self._lines_raise()]

        assert metrics.unrecorded_raises(raises, [record]) == raises
