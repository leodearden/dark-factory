"""Tests for scripts/scan_plan_decision_pairing.py — the READ-ONLY,
detection-only prevalence scanner for semantically cross-paired
``design_decisions`` entries in task plans.

Task 3967. The damage class is an entry whose ``decision`` and ``rationale``
are each perfectly well-formed prose but whose *association* is wrong. Nothing
is malformed and no sentinel is present, so ``shared.toolcall_markup.detect``
is structurally blind to it; the predicate keying on the correction entry an
author appends after noticing lives in :mod:`shared.decision_pairing`, and this
script is a re-runnable CLI over it. Every accept/reject verdict is DELEGATED
to that module — no marker literal is re-spelled here or in the script (INV-5),
so these tests never assert on a marker string they invented.

Mirrors ``test_scan_task_toolcall_leaks.py``: the pure functions
(``scan_plan_file``, ``scan_tree``, ``format_report``, ``format_json``) get
direct pytest coverage here; ``main()`` gets subprocess coverage in
``TestCli`` (step 13). The bare ``from scan_plan_decision_pairing import ...``
spelling resolves via ``scripts/tests/conftest.py``'s sys.path insert.

## The scanner NEVER reads the live corpus in a test

Every tree below is SYNTHETIC and written to ``tmp_path``. Nothing here globs
``.worktrees/.task-meta``, and nothing here pins a live prevalence count. That
corpus grew from 1196 to 1299 plans in about eight days and new victims are
still landing, so a pinned count would be flaky by construction — and worse, it
would invert the signal, making a predicate improvement that legitimately
detects MORE entries read as a regression. Live figures belong in the dated
findings doc and in the scanner's own output, never in an assertion.

## The read-only contract is ASSERTED, not merely documented

``_snapshot`` captures every input file's bytes and mtime before a scan;
``_assert_unchanged`` requires both to be identical afterwards. That is the
same structural-incapability posture ``scan_task_toolcall_leaks.py`` takes with
its ``mode=ro`` SQLite URIs, expressed for a scanner whose inputs are files. It
matters more here than there: the scanner's real inputs are live plan documents
that a RUNNING task may be reading, so a stray write would be mutation of
another agent's runtime state, not merely a dirty test.

## Sentinel-literal hazard — DO NOT "helpfully" un-escape these

Every envelope literal below is spelled with the ``\\x3c`` escape for the
opening angle bracket, exactly as ``shared/tests/test_toolcall_markup.py`` and
``shared/tests/test_decision_pairing.py`` require. Writing one verbatim here
would force any agent editing this file to emit that literal inside its own
tool-call envelope, reproducing a defect adjacent to the one this scanner
reports on. It is byte-identical at runtime and never appears verbatim in the
file text. Leave it escaped.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from scan_plan_decision_pairing import (
    PairingRecord,
    SkippedFile,
    TreeScan,
    format_json,
    format_report,
    scan_plan_file,
    scan_tree,
)

# ---------------------------------------------------------------------------
# Specimens, paraphrased from live victim plans (task ids named per specimen).
# ---------------------------------------------------------------------------

# Modelled on 3098[1] / 3216[1] / 3473[1] — the commonest live shape: a later
# entry opening on a correction header and saying, in words, that a preceding
# entry's two texts were mis-paired.
MISPAIRED_DECISION = (
    'CORRECTION of the preceding entry, whose rationale was mis-paired with '
    'its decision text at authoring time. That rationale argues for the '
    'discovery-flag choice, not for the detector-class choice it is attached '
    'to; read it against the former. The correct decision here is to key the '
    'walker on the plan document rather than on the lane directory.'
)
MISPAIRED_RATIONALE = (
    'Both texts were composed in a single call and the arguments were '
    'transposed, so neither field is damaged — only their association is. '
    'Measured over the live plans, this shape leaves no other trace.'
)

# Modelled on 3042[2] — the second observed header form, with the pairing
# language carried by the rationale rather than the decision.
SECOND_MISPAIRED_DECISION = (
    'READ THIS INSTEAD OF DECISIONS 1 AND 2, whose texts were transposed by a '
    'composition error. The scanner reports and never repairs.'
)
SECOND_MISPAIRED_RATIONALE = (
    'Decisions 1 and 2 were mis-paired: 1\'s rationale belongs to 2 and vice '
    'versa. Nothing can reconstruct the original association from the '
    'document, so this entry states it in prose instead.'
)

# An ordinary, correctly-paired entry. Must never be reported.
CLEAN_DECISION = (
    'Resolve the scan root at the CLI boundary rather than inside the walker, '
    'so a caller passing an explicit path is never second-guessed.'
)
CLEAN_RATIONALE = (
    'The walker is handed live plan documents that a running task may be '
    'reading, so keeping it free of discovery logic keeps it read-only by '
    'construction rather than by convention.'
)

# Modelled on 3382[5] — a GENUINE design reversal: a start-anchored header with
# no pairing language anywhere. A real supersession is not a mis-pairing, and
# the conjunction in shared.decision_pairing is what keeps it out of a report.
SUPERSEDES_ONLY_DECISION = (
    'SUPERSEDES decision #3: the scanner takes a --root flag after all, '
    'because an operator sweeping an orphaned lane tree has no other way in.'
)
SUPERSEDES_ONLY_RATIONALE = (
    'Decision #3 assumed a single well-known root. The orphaned-lane tree is a '
    'second one, and hardcoding either would make the other unreachable.'
)

# The `\x3c` escape is mandatory — see the module docstring. This specimen is
# BOTH mis-paired AND carries envelope residue, which is the 3382 shape: the
# scanner must report both damage classes on one record rather than making an
# operator run two sweeps to see them.
ENVELOPE_TAINTED_RATIONALE = (
    MISPAIRED_RATIONALE + '\x3c/invoke>\n'
)


def _entry(decision: str, rationale: str) -> dict:
    return {'decision': decision, 'rationale': rationale}


MISPAIRED_ENTRY = _entry(MISPAIRED_DECISION, MISPAIRED_RATIONALE)
SECOND_MISPAIRED_ENTRY = _entry(SECOND_MISPAIRED_DECISION, SECOND_MISPAIRED_RATIONALE)
CLEAN_ENTRY = _entry(CLEAN_DECISION, CLEAN_RATIONALE)
SUPERSEDES_ONLY_ENTRY = _entry(SUPERSEDES_ONLY_DECISION, SUPERSEDES_ONLY_RATIONALE)
ENVELOPE_TAINTED_ENTRY = _entry(MISPAIRED_DECISION, ENVELOPE_TAINTED_RATIONALE)


# ---------------------------------------------------------------------------
# Synthetic tree construction, and the read-only snapshot helpers.
# ---------------------------------------------------------------------------


def write_plan(root: Path, lane: str, decisions: list, **plan_fields) -> Path:
    """Write ``<root>/<lane>/plan.json`` carrying *decisions*, return it.

    The first parameter is named *lane* rather than ``task_id`` deliberately:
    the lane DIRECTORY and the document's own ``task_id`` field are distinct
    things that disagree on four live plans, and
    ``test_task_id_comes_from_the_lane_directory_not_the_document`` pins which
    of them the scanner keys on by passing ``task_id=`` through
    ``plan_fields``. Sharing one name here would make that test unwritable.

    Extra ``plan_fields`` are merged into the document, so a test can override
    or omit ``task_id``/``design_decisions`` to build an adversarial shape.
    """
    lane_dir = root / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    plan: dict = {
        'task_id': lane,
        'title': f'synthetic plan {lane}',
        'design_decisions': decisions,
    }
    plan.update(plan_fields)
    path = lane_dir / 'plan.json'
    path.write_text(json.dumps(plan, indent=2))
    return path


def write_raw_plan(root: Path, lane: str, raw: str) -> Path:
    """Write ``<root>/<lane>/plan.json`` with *raw* text verbatim."""
    lane_dir = root / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    path = lane_dir / 'plan.json'
    path.write_text(raw)
    return path


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    """Every file under *root*, mapped to its (bytes, mtime_ns)."""
    return {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob('*'))
        if p.is_file()
    }


def _assert_unchanged(root: Path, before: dict[str, tuple[bytes, int]]) -> None:
    """Require every input file byte-identical and mtime-identical."""
    after = _snapshot(root)
    assert after.keys() == before.keys(), (
        'the scan created or removed files: '
        f'{sorted(set(after) ^ set(before))}'
    )
    for path, (raw, mtime) in before.items():
        assert after[path][0] == raw, f'the scan rewrote {path}'
        assert after[path][1] == mtime, f'the scan touched the mtime of {path}'


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A synthetic plan tree: two victims, one clean plan, one near-miss.

    ``10`` carries two mis-paired entries (the real 3209/3567/4096 shape) with
    a correctly-paired entry between them, so index reporting is exercised
    against a document where the hits are not adjacent. ``2`` carries one
    mis-paired entry whose rationale ALSO carries envelope residue. ``30`` is
    the near-miss: a genuine supersession beside a clean entry, and must be
    reported by nothing.
    """
    root = tmp_path / 'task-meta'
    write_plan(root, '10', [MISPAIRED_ENTRY, CLEAN_ENTRY, SECOND_MISPAIRED_ENTRY])
    write_plan(root, '2', [CLEAN_ENTRY, ENVELOPE_TAINTED_ENTRY])
    write_plan(root, '30', [SUPERSEDES_ONLY_ENTRY, CLEAN_ENTRY])
    return root


# ---------------------------------------------------------------------------
# scan_plan_file
# ---------------------------------------------------------------------------


class TestScanPlanFile:
    """One plan document in, one record per mis-paired entry out."""

    def test_reports_one_record_per_mispaired_entry_with_both_markers(self, tmp_path):
        path = write_plan(
            tmp_path, '10', [MISPAIRED_ENTRY, CLEAN_ENTRY, SECOND_MISPAIRED_ENTRY]
        )

        records = scan_plan_file(path)

        assert [r.index for r in records] == [0, 2]
        assert all(isinstance(r, PairingRecord) for r in records)
        assert all(r.task_id == '10' for r in records)
        assert all(r.path == str(path) for r in records)
        # The matched literals travel ON the record, so a triager is never sent
        # to log-scrape which marker fired (INV-2).
        assert all(r.header and r.marker for r in records)
        assert all(r.field in ('decision', 'rationale') for r in records)

    def test_matched_markers_are_the_shared_modules_declared_literals(self, tmp_path):
        """The record quotes shared.decision_pairing's tuples, never free text.

        Imported from the owning module rather than re-spelled here, so this
        test cannot drift from the predicate it describes (INV-5).
        """
        from shared.decision_pairing import HEADER_MARKERS, PAIRING_MARKERS

        path = write_plan(tmp_path, '10', [MISPAIRED_ENTRY, SECOND_MISPAIRED_ENTRY])

        for record in scan_plan_file(path):
            assert record.header in HEADER_MARKERS
            assert record.marker in PAIRING_MARKERS

    def test_pairing_language_in_rationale_alone_is_attributed_to_rationale(self, tmp_path):
        """3042[2]'s shape: header on the decision, pairing word in the rationale."""
        path = write_plan(tmp_path, '10', [SECOND_MISPAIRED_ENTRY])

        (record,) = scan_plan_file(path)

        assert record.field == 'rationale'

    def test_clean_plan_yields_no_records(self, tmp_path):
        path = write_plan(tmp_path, '10', [CLEAN_ENTRY, CLEAN_ENTRY])

        assert scan_plan_file(path) == []

    def test_genuine_supersession_is_not_reported(self, tmp_path):
        """A header with NO pairing language is a design reversal, not damage."""
        path = write_plan(tmp_path, '30', [SUPERSEDES_ONLY_ENTRY])

        assert scan_plan_file(path) == []

    def test_envelope_column_is_false_when_no_literal_is_present(self, tmp_path):
        """The two damage classes are disjoint at the detector, and shown so."""
        path = write_plan(tmp_path, '10', [MISPAIRED_ENTRY])

        (record,) = scan_plan_file(path)

        assert record.envelope_leak is False

    def test_envelope_column_is_true_when_the_entry_also_carries_residue(self, tmp_path):
        """The 3382 shape: one entry damaged BOTH ways, reported on one record.

        Carrying the envelope verdict as a column is what lets a single report
        distinguish this damage class from the one
        ``shared.toolcall_markup`` owns — and flag a plan suffering both —
        without an operator running two sweeps and joining them by hand.
        """
        path = write_plan(tmp_path, '2', [ENVELOPE_TAINTED_ENTRY])

        (record,) = scan_plan_file(path)

        assert record.envelope_leak is True

    def test_task_id_comes_from_the_lane_directory_not_the_document(self, tmp_path):
        """Measured 2026-08-16 over 1299 live plans: the document disagrees on 4.

        Tasks 2421, 2460, 2579 and 2921 carry a ``task_id`` of ``df_task_<n>``
        while their lane directory is ``<n>``. The directory is the spelling
        the rest of the system uses and the one an operator navigates by, and —
        decisively — it is the only one available for a plan whose bytes cannot
        be parsed at all, so the skip report and the hit report key alike.
        """
        path = write_plan(tmp_path, '2421', [MISPAIRED_ENTRY], task_id='df_task_2421')

        (record,) = scan_plan_file(path)

        assert record.task_id == '2421'

    @pytest.mark.parametrize(
        'decisions',
        [
            pytest.param('not-a-list', id='non-list-design-decisions'),
            pytest.param([None, 7, 'text'], id='non-dict-items'),
            pytest.param([{'decision': 42, 'rationale': None}], id='non-str-fields'),
            pytest.param([{}], id='entry-missing-both-fields'),
            pytest.param([], id='empty-list'),
        ],
    )
    def test_adversarial_entry_shapes_yield_no_hit_and_no_exception(self, tmp_path, decisions):
        """Totality mirrors ``plan_tools._walk_repairable``'s guard set."""
        path = write_plan(tmp_path, '10', decisions)

        assert scan_plan_file(path) == []

    def test_plan_without_a_design_decisions_key_yields_no_hit(self, tmp_path):
        path = write_raw_plan(tmp_path, '10', json.dumps({'task_id': '10'}))

        assert scan_plan_file(path) == []

    def test_plan_whose_top_level_is_not_an_object_yields_no_hit(self, tmp_path):
        """``json.loads`` can legitimately return a list or a scalar."""
        path = write_raw_plan(tmp_path, '10', '[1, 2, 3]')

        assert scan_plan_file(path) == []

    def test_scanning_does_not_touch_the_file(self, tmp_path):
        write_plan(tmp_path, '10', [MISPAIRED_ENTRY])
        before = _snapshot(tmp_path)

        scan_plan_file(tmp_path / '10' / 'plan.json')

        _assert_unchanged(tmp_path, before)


# ---------------------------------------------------------------------------
# scan_tree
# ---------------------------------------------------------------------------


class TestScanTree:
    """Walks ``*/plan.json``, tolerating every unreadable input by REPORTING it."""

    def test_returns_records_from_every_victim_plan_in_the_tree(self, tree):
        scan = scan_tree(tree)

        assert isinstance(scan, TreeScan)
        assert {(r.task_id, r.index) for r in scan.records} == {
            ('10', 0), ('10', 2), ('2', 1)
        }
        assert scan.skipped == []

    def test_counts_what_it_scanned_as_well_as_what_it_matched(self, tree):
        """A zero-hit run must be distinguishable from a read-nothing run.

        Without ``scanned``, an empty ``records`` list means either "the corpus
        is clean" or "the root was wrong and nothing was opened" — the two
        outcomes an operator most needs to tell apart, and the reason exit 3
        exists in the precedent scanner.
        """
        scan = scan_tree(tree)

        assert scan.scanned == 3

    def test_a_clean_tree_reports_zero_hits_but_a_nonzero_scanned_count(self, tmp_path):
        root = tmp_path / 'task-meta'
        write_plan(root, '10', [CLEAN_ENTRY])
        write_plan(root, '30', [SUPERSEDES_ONLY_ENTRY])

        scan = scan_tree(root)

        assert scan.records == []
        assert scan.skipped == []
        assert scan.scanned == 2

    def test_a_missing_root_reports_nothing_scanned_rather_than_raising(self, tmp_path):
        scan = scan_tree(tmp_path / 'no-such-tree')

        assert scan.records == []
        assert scan.scanned == 0

    def test_unparseable_plan_is_reported_and_the_rest_of_the_tree_is_scanned(self, tmp_path):
        """One bad file must never abort the sweep — warn-and-continue."""
        root = tmp_path / 'task-meta'
        write_raw_plan(root, '4', '{"design_decisions": [truncated')
        write_plan(root, '10', [MISPAIRED_ENTRY])

        scan = scan_tree(root)

        assert [r.task_id for r in scan.records] == ['10']
        assert [(s.task_id, s.reason) for s in scan.skipped] == [('4', 'unparseable')]
        assert scan.skipped[0].detail, 'a skip must say what went wrong'
        assert scan.scanned == 1, 'a file that could not be parsed was not scanned'

    def test_unreadable_plan_is_reported_and_the_rest_of_the_tree_is_scanned(self, tmp_path):
        root = tmp_path / 'task-meta'
        bad = write_plan(root, '4', [MISPAIRED_ENTRY])
        bad.chmod(0o000)
        write_plan(root, '10', [MISPAIRED_ENTRY])
        try:
            scan = scan_tree(root)
        finally:
            bad.chmod(0o644)

        assert [r.task_id for r in scan.records] == ['10']
        assert [(s.task_id, s.reason) for s in scan.skipped] == [('4', 'unreadable')]
        assert scan.scanned == 1

    def test_dangling_plan_symlink_is_reported_as_missing(self, tmp_path):
        """The live shape: every lane's plan.json is a symlink into .task-meta.

        A lane whose target has been reclaimed leaves the name in place with
        nothing behind it. Reporting it is the point — a silently dropped lane
        would understate a count this scanner already only ever states as a
        lower bound.
        """
        root = tmp_path / 'task-meta'
        write_plan(root, '10', [MISPAIRED_ENTRY])
        lane = root / '4'
        lane.mkdir(parents=True)
        (lane / 'plan.json').symlink_to(tmp_path / 'gone' / 'plan.json')

        scan = scan_tree(root)

        assert [r.task_id for r in scan.records] == ['10']
        assert [(s.task_id, s.reason) for s in scan.skipped] == [('4', 'missing')]
        assert scan.scanned == 1

    def test_a_lane_without_a_plan_is_neither_scanned_nor_skipped(self, tmp_path):
        """An absent name is not an error — most lanes never got a plan."""
        root = tmp_path / 'task-meta'
        write_plan(root, '10', [MISPAIRED_ENTRY])
        (root / '4').mkdir()

        scan = scan_tree(root)

        assert scan.skipped == []
        assert scan.scanned == 1

    def test_records_are_sorted_numerically_by_task_then_by_index(self, tmp_path):
        """Numeric-aware, so lane 10 sorts after lane 2 rather than before it.

        An operator diffs one run's report against the next; a lexicographic
        order would manufacture churn that reads as new damage, and would
        scatter a numeric corpus in a way no reader can scan by eye.
        """
        root = tmp_path / 'task-meta'
        write_plan(root, '10', [MISPAIRED_ENTRY])
        write_plan(root, '2', [CLEAN_ENTRY, MISPAIRED_ENTRY, SECOND_MISPAIRED_ENTRY])
        write_plan(root, '_iact-spawn', [MISPAIRED_ENTRY])

        scan = scan_tree(root)

        assert [(r.task_id, r.index) for r in scan.records] == [
            ('2', 1), ('2', 2), ('10', 0), ('_iact-spawn', 0)
        ]

    def test_skips_are_sorted_by_the_same_key(self, tmp_path):
        root = tmp_path / 'task-meta'
        write_raw_plan(root, '10', 'nope')
        write_raw_plan(root, '2', 'nope')

        scan = scan_tree(root)

        assert [s.task_id for s in scan.skipped] == ['2', '10']
        assert all(isinstance(s, SkippedFile) for s in scan.skipped)

    def test_repeated_scans_of_the_same_tree_are_identical(self, tree):
        assert scan_tree(tree) == scan_tree(tree)

    def test_the_whole_tree_is_untouched_by_a_scan(self, tree):
        before = _snapshot(tree)

        scan_tree(tree)

        _assert_unchanged(tree, before)

    def test_the_tree_is_untouched_even_when_files_are_unparseable(self, tmp_path):
        """The error path is where a scanner is most tempted to 'fix' an input."""
        root = tmp_path / 'task-meta'
        write_raw_plan(root, '4', '{"design_decisions": [truncated')
        write_plan(root, '10', [MISPAIRED_ENTRY])
        before = _snapshot(root)

        scan_tree(root)

        _assert_unchanged(root, before)


# ---------------------------------------------------------------------------
# format_report / format_json
# ---------------------------------------------------------------------------


class TestFormatReport:
    def test_names_every_victim_task_index_and_both_matched_markers(self, tree):
        scan = scan_tree(tree)

        report = format_report(scan)

        for record in scan.records:
            assert record.task_id in report
            assert record.header in report
            assert record.marker in report

    def test_summary_counts_entries_plans_and_files_scanned(self, tree):
        report = format_report(scan_tree(tree))

        # 3 mis-paired entries across 2 plans, out of 3 plan files read.
        assert '3' in report and '2' in report
        assert 'scanned' in report.lower()

    def test_a_clean_tree_yields_an_explicit_message_naming_what_was_read(self, tmp_path):
        """Never a blank report, and never one that hides a zero-file run."""
        root = tmp_path / 'task-meta'
        write_plan(root, '10', [CLEAN_ENTRY])

        report = format_report(scan_tree(root))

        assert report.strip()
        assert 'scanned' in report.lower()

    def test_skipped_files_appear_in_the_report_with_their_reason(self, tmp_path):
        root = tmp_path / 'task-meta'
        write_raw_plan(root, '4', 'nope')

        report = format_report(scan_tree(root))

        assert '4' in report
        assert 'unparseable' in report

    def test_report_is_deterministic(self, tree):
        scan = scan_tree(tree)

        assert format_report(scan) == format_report(scan)


class TestFormatJson:
    def test_is_parseable_and_carries_every_record_field(self, tree):
        scan = scan_tree(tree)

        payload = json.loads(format_json(scan))

        assert payload['scanned'] == scan.scanned
        assert len(payload['records']) == len(scan.records)
        assert payload['records'][0] == {
            'task_id': scan.records[0].task_id,
            'path': scan.records[0].path,
            'index': scan.records[0].index,
            'header': scan.records[0].header,
            'marker': scan.records[0].marker,
            'field': scan.records[0].field,
            'envelope_leak': scan.records[0].envelope_leak,
        }

    def test_carries_the_skipped_files_too(self, tmp_path):
        """A consumer reading only ``records`` must still be able to see gaps."""
        root = tmp_path / 'task-meta'
        write_raw_plan(root, '4', 'nope')

        payload = json.loads(format_json(scan_tree(root)))

        assert [s['reason'] for s in payload['skipped']] == ['unparseable']

    def test_an_empty_scan_is_still_a_well_formed_object(self, tmp_path):
        payload = json.loads(format_json(scan_tree(tmp_path / 'no-such-tree')))

        assert payload['records'] == []
        assert payload['skipped'] == []
        assert payload['scanned'] == 0

    def test_json_is_deterministic(self, tree):
        scan = scan_tree(tree)

        assert format_json(scan) == format_json(scan)
