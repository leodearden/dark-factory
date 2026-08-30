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
import os
import subprocess
import sys
from pathlib import Path

import pytest
from scan_plan_decision_pairing import (
    _WORKTREES_DIR,
    DEFAULT_ROOT,
    SKIP_UNDECODABLE,
    PairingRecord,
    SkippedFile,
    TreeScan,
    _default_root,
    format_json,
    format_report,
    scan_plan_file,
    scan_tree,
)

# Resolvable only AFTER the import above: the scanner module carries the
# shared/src bootstrap (its lines 94-101), and this file deliberately reaches
# through it rather than re-spelling that path. ``detect`` is imported for ONE
# purpose — the non-circular control asserting the self-name specimen is
# invisible to the BLANKET predicate — never as the thing under test.
from shared.toolcall_markup import detect  # noqa: E402

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

# The SELF-NAME dialect (task 4696): a field mis-closed with its OWN
# name-echoing tag and no other signature. Measured 2026-08-25 over
# .worktrees/.task-meta/*/plan.json, this is 296 of the corrupted rationale
# strings and it matches NO fixed envelope literal, so the param-blind
# predicate reports it envelope-CLEAN while it is not.
SELF_NAME_TAINTED_RATIONALE = MISPAIRED_RATIONALE + '\x3c/rationale>'
SELF_NAME_TAINTED_ENTRY = _entry(MISPAIRED_DECISION, SELF_NAME_TAINTED_RATIONALE)


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
    # utf-8, matching what scan_plan_file reads with. json.dumps defaults to
    # ensure_ascii=True so today's bytes are ASCII either way, but a helper
    # that WRITES in the ambient locale and a scanner that READS utf-8 is an
    # asymmetry that only shows up on somebody else's machine.
    path.write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return path


def write_raw_plan(root: Path, lane: str, raw: str) -> Path:
    """Write ``<root>/<lane>/plan.json`` with *raw* text verbatim, as utf-8.

    Encoding pinned for the same reason :func:`write_plan`'s is: the scanner
    reads utf-8, so a specimen written in the ambient locale would be a
    different specimen on a differently-configured machine. Bytes that are
    deliberately NOT text go through :func:`write_undecodable_plan` instead.
    """
    lane_dir = root / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    path = lane_dir / 'plan.json'
    path.write_text(raw, encoding='utf-8')
    return path


#: A plan.json that is not TEXT at all: a valid UTF-8 prefix ending in a lone
#: multibyte LEAD byte, so the sequence is truncated mid-character.
#:
#: Built by dropping the closing brace and appending ``\xc3`` rather than by
#: slicing at a magic byte offset. The offset spelling works, but it is pinned
#: to the exact length of this document — reword the title and the cut lands
#: somewhere harmless and the specimen silently stops being undecodable, which
#: would make every test below pass vacuously. Truncating at the END cannot
#: drift that way, and it reproduces the identical failure:
#: ``UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position N:
#: unexpected end of data``.
#:
#: That exception is the point. It is a ``ValueError`` subclass but is NEITHER
#: an ``OSError`` NOR a ``json.JSONDecodeError``, so it escapes both of
#: ``scan_tree``'s handlers and aborts the ENTIRE sweep — one bad file hiding
#: the whole corpus, which is exactly what that function's docstring promises
#: cannot happen.
_UNDECODABLE_PLAN_BYTES = (
    json.dumps(
        {'task_id': '4', 'title': 'café résumé synthetic plan',
         'design_decisions': []},
        ensure_ascii=False,
    ).encode('utf-8')[:-1] + b'\xc3'
)


def write_undecodable_plan(root: Path, lane: str) -> Path:
    """Write ``<root>/<lane>/plan.json`` as bytes that are not valid UTF-8.

    Asserts the specimen really is undecodable before handing it back, so a
    future edit that accidentally makes these bytes decode fails HERE, loudly,
    instead of leaving the callers below green against an input that no longer
    exercises the path they exist to cover.
    """
    lane_dir = root / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    path = lane_dir / 'plan.json'
    path.write_bytes(_UNDECODABLE_PLAN_BYTES)
    with pytest.raises(UnicodeDecodeError):
        path.read_text(encoding='utf-8')
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

    def test_a_self_name_closer_is_reported_as_envelope_damage(self, tmp_path):
        """Task 4696: the verdict must see the invoked tool's OWN parameters.

        The blanket predicate enumerates a fixed literal set that spells none
        of plan-tools' parameter names, so a rationale mis-closed with its own
        tag was reported envelope-CLEAN. That is a WRONG column rather than a
        missing one, and it defeats this scanner's whole purpose: the reason
        the envelope verdict rides along as a column is so ONE report
        distinguishes the two damage classes instead of an operator running two
        sweeps and joining them by hand.
        """
        path = write_plan(tmp_path, '11', [SELF_NAME_TAINTED_ENTRY])

        (record,) = scan_plan_file(path)

        assert record.envelope_leak is True

    def test_the_self_name_specimen_carries_no_other_signature(self, tmp_path):
        """The non-circular control: it must be INVISIBLE to the blanket set.

        If a future widening of the fixed literals caught this specimen anyway,
        the row above would pass for the wrong reason and stop proving the
        verdict is parameter-aware.
        """
        assert detect(SELF_NAME_TAINTED_RATIONALE) is None
        assert '\x3c/invoke>' not in SELF_NAME_TAINTED_RATIONALE

    def test_a_clean_entry_is_still_reported_envelope_clean(self, tmp_path):
        """The widening must not turn ordinary prose into a false positive."""
        path = write_plan(tmp_path, '12', [MISPAIRED_ENTRY])

        (record,) = scan_plan_file(path)

        assert record.envelope_leak is False

    def test_scanning_a_self_name_specimen_does_not_touch_the_file(self, tmp_path):
        """The read-only guarantee is unchanged by the widened verdict.

        This scanner NEVER writes — the wider predicate reads more, it does not
        make the scanner a repairer.
        """
        write_plan(tmp_path, '11', [SELF_NAME_TAINTED_ENTRY])
        before = _snapshot(tmp_path)

        scan_plan_file(tmp_path / '11' / 'plan.json')

        _assert_unchanged(tmp_path, before)

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

    def test_an_unlistable_root_is_reported_rather_than_raising(self, tmp_path):
        """DISCOVERY can fail too, not just the per-file read.

        The listing is the one step whose failure loses the whole sweep, so it
        is guarded the same way every read is. The report's shape is the part
        worth pinning: there is no lane to key the skip by, so it carries the
        ROOT's own directory name in the ``task_id`` slot and the root
        directory in ``path`` — a convention a refactor could silently change,
        leaving an operator grepping for a lane id that was never emitted.

        ``scanned == 0`` beside an empty ``records`` is the whole point: that
        pair is what tells a reader the sweep read nothing, as opposed to
        reading a clean corpus.
        """
        root = tmp_path / 'task-meta'
        write_plan(root, '10', [MISPAIRED_ENTRY])
        root.chmod(0o000)
        try:
            scan = scan_tree(root)
        finally:
            root.chmod(0o755)

        assert scan.records == []
        assert scan.scanned == 0
        assert [(s.task_id, s.reason, s.path) for s in scan.skipped] == [
            ('task-meta', 'unreadable', str(root))
        ]
        assert scan.skipped[0].detail, 'a skip must say what went wrong'

    def test_an_unreadable_lane_directory_does_not_hide_its_siblings(self, tmp_path):
        """A lane reclaimed or chmod'd mid-sweep makes the PROBES raise.

        ``lane.is_dir()`` still answers — it stats the lane itself — but
        ``candidate.exists()`` needs search permission on that directory and
        raises ``PermissionError`` (measured on this interpreter; ``exists()``
        swallows ENOENT/ENOTDIR/ELOOP, not EACCES). Without the guard around
        the probes that escapes the loop and discards every lane already
        scanned, which is the same total-report failure the per-file arms
        exist to prevent — so the load-bearing assertion here is that the good
        sibling SURVIVES, not merely that the bad lane is named.
        """
        root = tmp_path / 'task-meta'
        write_plan(root, '10', [MISPAIRED_ENTRY])
        bad_lane = root / '4'
        bad_lane.mkdir()
        (bad_lane / 'plan.json').write_text('{}', encoding='utf-8')
        bad_lane.chmod(0o000)
        try:
            scan = scan_tree(root)
        finally:
            bad_lane.chmod(0o755)

        assert [(r.task_id, r.index) for r in scan.records] == [('10', 0)], (
            'a lane that cannot be probed must not take its siblings with it'
        )
        assert [(s.task_id, s.reason) for s in scan.skipped] == [('4', 'unreadable')]
        assert scan.skipped[0].path == str(bad_lane / 'plan.json')
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
        """All four numbers, asserted on the SUMMARY LINE itself.

        A bare ``'3' in report`` cannot pin this: the body already carries
        tmp_path components, entry indices and lane names, so both digits are
        present however the summary counts — a summary reporting ``0 ... across
        0 plans (scanned 0 ...)`` would satisfy it. The counts are the whole
        product of a prevalence sweep, so they are pinned verbatim, and pinned
        as the LAST line so they cannot be buried mid-report where an operator
        reading the tail would not see them.
        """
        report = format_report(scan_tree(tree))

        # 3 mis-paired entries across 2 plans, out of 3 plan files read.
        summary = report.splitlines()[-1]
        assert summary.startswith(
            '3 mis-paired entries across 2 plans (scanned 3 plan files, skipped 0).'
        ), f'the summary line reported: {summary!r}'
        assert 'STRICT LOWER BOUND' in summary, (
            'the count must never be readable as a prevalence estimate'
        )

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


# ---------------------------------------------------------------------------
# Where the sweep points when nobody says — a constant, pinned in-process.
# ---------------------------------------------------------------------------


class TestDefaultRoot:
    """Both checkout shapes must resolve to the ONE real lane tree.

    Measured 2026-08-16: a task worktree lives at ``<repo>/.worktrees/<lane>/``,
    so the naive ``<checkout>/.worktrees/.task-meta`` resolves inside a worktree
    to ``<repo>/.worktrees/<lane>/.worktrees/.task-meta`` — a directory that
    does not exist. The sweep then reports zero hits over zero files, which
    reads as a clean corpus. ``TreeScan.scanned`` makes that visible, but a
    default that walks into it by construction is a defect, not a diagnostic.
    """

    def test_a_main_checkout_resolves_to_its_own_worktrees_lane_tree(self):
        assert _default_root(Path('/repo')) == Path('/repo/.worktrees/.task-meta')

    def test_a_task_worktree_resolves_to_the_shared_lane_tree_beside_it(self):
        assert (
            _default_root(Path('/repo/.worktrees/3967'))
            == Path('/repo/.worktrees/.task-meta')
        )

    def test_the_default_resolves_to_a_tree_that_EXISTS_in_both_layouts(
        self, tmp_path
    ):
        """EXISTENCE, which is the property the two cases above cannot check.

        They compare paths, and a wrong path compares equal to itself — the
        earliest spelling of this test asserted ``_default_root(<repo root>) ==
        DEFAULT_ROOT``, which is the same function applied to the same
        directory and so could only fail if ``_default_root`` were
        non-deterministic. The defect the helper was written for is a default
        that resolves to a directory which is NOT THERE (from a task worktree
        the naive spelling yields ``<repo>/.worktrees/<lane>/.worktrees/
        .task-meta``), and only a filesystem probe can catch that.

        The probe runs against a SYNTHETIC tree, not this checkout. The
        previous spelling probed the live tree and so conflated two unrelated
        facts: it skipped on a missing ``.worktrees`` but FAILED on a checkout
        that has lanes while ``.task-meta`` has not been materialised yet,
        reporting a defect in ``_default_root`` when the operator's plan state
        was merely empty. Here the naive spelling still fails by construction —
        from ``lane`` it yields ``<tmp>/repo/.worktrees/3967/.worktrees/
        .task-meta``, which this tree does not contain — while no state of the
        real checkout can colour the result, so the case never skips either.
        """
        repo = tmp_path / 'repo'
        (repo / _WORKTREES_DIR / '.task-meta').mkdir(parents=True)
        lane = repo / _WORKTREES_DIR / '3967'
        lane.mkdir()

        assert _default_root(repo).is_dir()
        assert _default_root(lane).is_dir()

    def test_the_shipped_default_nests_the_worktrees_dir_exactly_ONCE(self):
        """The shipped constant itself, pinned structurally and unconditionally.

        The synthetic probe above covers the function; this covers
        ``DEFAULT_ROOT``, which is that function applied to ``_CHECKOUT_ROOT``,
        and so is the only case that can catch a wrong checkout root. Counting
        the nestings is what discriminates: the naive spelling names
        ``.worktrees`` TWICE (``.worktrees/<lane>/.worktrees/.task-meta``)
        where the correct one names it once. Note that
        ``DEFAULT_ROOT.parent.name == '.worktrees'`` holds of BOTH spellings
        and so distinguishes nothing. Reads no filesystem state, so it cannot
        go red on a checkout whose lane tree is simply not populated.
        """
        assert DEFAULT_ROOT.name == '.task-meta'
        assert DEFAULT_ROOT.parts.count(_WORKTREES_DIR) == 1, (
            f'the shipped default nests {_WORKTREES_DIR!r} more than once: '
            f'{DEFAULT_ROOT} — that is the naive spelling resolved from inside '
            f'a task worktree, naming a directory that does not exist, and a '
            f'sweep with no --root would read nothing and report a clean corpus'
        )


# ---------------------------------------------------------------------------
# CLI (main), driven via subprocess.run — mirrors test_scan_task_toolcall_leaks.py
# ---------------------------------------------------------------------------

SCRIPT = Path(__file__).parent.parent / 'scan_plan_decision_pairing.py'


def _run_cli(*args, timeout=60, env=None, python_args=()):
    """Run the scanner CLI in a subprocess and return the CompletedProcess.

    *python_args* go BEFORE the script path, so a caller can pass interpreter
    flags such as ``-W error::EncodingWarning``; *env* is merged over a copy of
    the ambient environment rather than replacing it, so PATH and the harness's
    own redirects survive. Both are extensions of the one subprocess mechanism
    rather than a second helper, so every CLI case here runs the same way.
    """
    full_env = None
    if env is not None:
        full_env = dict(os.environ)
        full_env.update(env)
    return subprocess.run(
        [sys.executable, *python_args, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=full_env,
    )


class TestCli:
    """The invocation an operator actually uses, driven end to end.

    Every case passes an explicit ``--root`` at ``tmp_path``. None of these
    ever sweeps the live corpus: a bare run would, and its result would be a
    live prevalence count that no assertion here may depend on.
    """

    def test_help_succeeds(self):
        result = _run_cli('--help')

        assert result.returncode == 0, result.stderr
        assert '--root' in result.stdout
        assert '--fail-on-hit' in result.stdout
        assert '--require-scanned' in result.stdout

    def test_json_output_matches_the_in_process_scan_of_the_same_tree(self, tree):
        """The CLI is a thin shell: it must not re-derive or re-order anything."""
        result = _run_cli('--root', str(tree), '--json')

        assert result.returncode == 0, f'stderr={result.stderr!r}'
        assert json.loads(result.stdout) == json.loads(format_json(scan_tree(tree)))

    def test_plain_report_names_the_victims_and_what_was_scanned(self, tree):
        result = _run_cli('--root', str(tree))

        assert result.returncode == 0, f'stderr={result.stderr!r}'
        assert 'task 2' in result.stdout
        assert 'task 10' in result.stdout
        assert 'scanned 3' in result.stdout

    def test_a_dirty_tree_exits_zero_without_fail_on_hit(self, tree):
        """Reporting is the default posture; failing is opt-in."""
        result = _run_cli('--root', str(tree))

        assert result.returncode == 0, f'stderr={result.stderr!r}'

    def test_fail_on_hit_exits_nonzero_when_the_tree_has_a_mispairing(self, tree):
        result = _run_cli('--root', str(tree), '--fail-on-hit')

        assert result.returncode != 0
        assert 'task 10' in result.stdout

    def test_fail_on_hit_exits_zero_on_a_clean_tree(self, tmp_path):
        root = tmp_path / 'task-meta'
        write_plan(root, '10', [CLEAN_ENTRY])
        write_plan(root, '30', [SUPERSEDES_ONLY_ENTRY])

        result = _run_cli('--root', str(root), '--fail-on-hit')

        assert result.returncode == 0, f'stdout={result.stdout!r} stderr={result.stderr!r}'

    def test_fail_on_hit_over_a_missing_root_exits_zero_but_says_it_read_nothing(
        self, tmp_path
    ):
        """The one case where exit 0 is NOT evidence of a clean corpus.

        ``--fail-on-hit`` keys on hits, and a root that does not exist produces
        none — so ITS exit code alone cannot distinguish "clean" from "swept
        nothing". That is why the summary always states the scanned count: an
        operator reading the output can see it, even though an operator
        branching only on ``$?`` cannot. Pinned so the caveat is visible rather
        than discovered in an incident.

        The flag's semantics are deliberately left as they are — it means what
        its name says — and the unattended gap is closed by ``--require-scanned``
        instead (see the four cases below), so a caller who wants coverage
        enforced asks for it rather than having it silently redefined.
        """
        result = _run_cli('--root', str(tmp_path / 'no-such-tree'), '--fail-on-hit')

        assert result.returncode == 0
        assert 'scanned 0' in result.stdout

    def test_require_scanned_turns_a_read_nothing_sweep_into_a_FAILURE(
        self, tmp_path
    ):
        """The gap that made ``--fail-on-hit`` unsafe unattended, closed.

        A mistyped ``--root``, a root not created yet, or a root whose listing
        raised ``PermissionError`` all yield zero hits, so a hit-keyed gate
        reports success for a sweep that read nothing at all. A gate is
        consumed by ``$?``, not by a human reading the scanned count, so the
        coverage floor has to be expressible in the exit code.

        Exit 3, matching the precedent scanner's reserved "nothing was scanned,
        never treat this as a clean run" code rather than inventing a fourth
        meaning for 1.
        """
        result = _run_cli(
            '--root', str(tmp_path / 'no-such-tree'),
            '--fail-on-hit', '--require-scanned', '1',
        )

        assert result.returncode == 3, (
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )
        assert 'scanned 0' in result.stdout, (
            'the report is still printed before the code is decided'
        )

    def test_require_scanned_is_satisfied_by_a_sweep_that_read_enough(self, tmp_path):
        """The floor is a floor, not an equality: reading MORE is fine."""
        root = tmp_path / 'task-meta'
        write_plan(root, '10', [CLEAN_ENTRY])
        write_plan(root, '30', [SUPERSEDES_ONLY_ENTRY])

        result = _run_cli(
            '--root', str(root), '--fail-on-hit', '--require-scanned', '1',
        )

        assert result.returncode == 0, (
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )

    def test_a_coverage_failure_outranks_a_hit_in_the_exit_code(self, tree):
        """Both are non-zero, so the question is only which one is REPORTED.

        A sweep that fell short of its coverage floor found an unknown fraction
        of what is there, so "at least one hit" understates the situation: the
        honest report is that the run cannot be read as a measurement at all.
        Exit 3 wins, and the hits are still printed.
        """
        result = _run_cli(
            '--root', str(tree), '--fail-on-hit', '--require-scanned', '99',
        )

        assert result.returncode == 3, (
            f'stdout={result.stdout!r} stderr={result.stderr!r}'
        )
        assert 'task 10' in result.stdout, 'the hits found are still reported'

    def test_a_negative_require_scanned_is_rejected_not_silently_disabled(self):
        """A mistyped floor must not quietly switch the gate off.

        ``scanned < -1`` is never true, so a negative would disable the very
        check it was passed to enable — the silent degradation this flag exists
        to remove, reintroduced by a typo. argparse's own usage error (exit 2)
        is the loud answer.
        """
        result = _run_cli('--root', '/nonexistent', '--require-scanned', '-1')

        assert result.returncode == 2
        assert '--require-scanned' in result.stderr

    def test_the_tree_is_untouched_by_the_real_cli_path(self, tree):
        """The read-only contract must survive the invocation operators use.

        Asserted against the SUBPROCESS, not just the in-process functions: a
        CLI is where an ``--apply`` convenience or a "normalize while we're
        here" would land, and the scanner's real inputs are live plan documents
        a running task may be reading.
        """
        before = _snapshot(tree)

        result = _run_cli('--root', str(tree), '--fail-on-hit')

        assert result.returncode != 0
        _assert_unchanged(tree, before)

    def test_a_tree_with_unreadable_files_is_reported_not_fatal(self, tmp_path):
        root = tmp_path / 'task-meta'
        write_raw_plan(root, '4', '{"design_decisions": [truncated')
        write_plan(root, '10', [MISPAIRED_ENTRY])
        before = _snapshot(root)

        result = _run_cli('--root', str(root), '--json')

        assert result.returncode == 0, f'stderr={result.stderr!r}'
        payload = json.loads(result.stdout)
        assert [s['reason'] for s in payload['skipped']] == ['unparseable']
        assert payload['scanned'] == 1
        _assert_unchanged(root, before)


# ---------------------------------------------------------------------------
# Undecodable plan bytes — the one per-file failure that is currently FATAL
# ---------------------------------------------------------------------------


class TestUndecodablePlan:
    """A plan.json whose bytes are not text must be REPORTED, never fatal.

    ``scan_tree``'s docstring already promises the contract: "Every per-file
    failure is REPORTED as a SkippedFile and the sweep continues, so one
    truncated plan cannot hide the rest of the corpus." Two handlers implement
    it — ``json.JSONDecodeError`` and ``OSError`` — and an undecodable file
    falls between them. ``UnicodeDecodeError`` is a ``ValueError`` subclass and
    is neither of those types, so it propagates out of the loop and takes the
    whole sweep with it, DISCARDING every lane already scanned.

    The existing coverage misses this by construction: the unparseable case
    uses ASCII-only malformed JSON (a genuine ``JSONDecodeError``) and the
    unreadable case uses ``chmod 000`` (a genuine ``PermissionError``). Neither
    ever produces bytes that fail to decode, so the gap between the two
    handlers was never probed.

    The load-bearing half of the tolerance test is not that the bad lane is
    reported — it is that the CLEAN lane is still scanned. A scanner that
    reports a lower bound and then silently loses the rest of the corpus to one
    bad file reports a number that is not merely low but arbitrary.
    """

    def test_an_undecodable_plan_is_reported_and_the_rest_of_the_tree_is_scanned(
        self, tmp_path
    ):
        """One file that is not text must not hide every file that is.

        Asserted against ``scan_tree``'s SORTED output rather than against an
        assumed ``iterdir`` order. That order is arbitrary, and today's failure
        is order-dependent — a sweep that happens to reach the bad lane last
        still loses everything it had already collected, so an assertion keyed
        to one traversal order would be a coin flip.
        """
        root = tmp_path / 'task-meta'
        write_plan(root, '10', [MISPAIRED_ENTRY])
        write_undecodable_plan(root, '4')

        scan = scan_tree(root)

        assert [(r.task_id, r.index) for r in scan.records] == [('10', 0)], (
            'the clean lane must survive an undecodable sibling'
        )
        assert [(s.task_id, s.reason) for s in scan.skipped] == [
            ('4', SKIP_UNDECODABLE)
        ]
        assert scan.skipped[0].detail, 'a skip must say what went wrong'
        assert scan.scanned == 1, 'a file that could not be decoded was not scanned'

    def test_scan_plan_file_still_raises_on_an_undecodable_plan(self, tmp_path):
        """The raise/report split the module already declares is PRESERVED.

        ``scan_plan_file``'s docstring propagates ``OSError`` and
        ``json.JSONDecodeError`` deliberately: ``scan_tree`` is what turns a
        per-file failure into a reported skip, and a caller pointing the
        function at one known file gets the error rather than an empty list
        that reads as "this plan is clean". The new arm must join that split on
        the same side — tolerating it HERE would make a single-file caller
        unable to tell a clean plan from an unreadable one.
        """
        bad = write_undecodable_plan(tmp_path / 'task-meta', '4')

        with pytest.raises(UnicodeDecodeError):
            scan_plan_file(bad)

    def test_the_tree_is_untouched_even_when_a_plan_is_undecodable(self, tmp_path):
        """The read-only contract holds on the new error path too.

        The error paths are where a scanner is most tempted to "normalize" an
        input, and these inputs are live plan documents a running task may be
        reading.
        """
        root = tmp_path / 'task-meta'
        write_plan(root, '10', [MISPAIRED_ENTRY])
        write_undecodable_plan(root, '4')
        before = _snapshot(root)

        scan_tree(root)

        _assert_unchanged(root, before)

    def test_the_cli_never_decodes_a_plan_in_the_ambient_locale_encoding(self, tree):
        """PEP 597's own mechanism for detecting a locale-dependent read.

        ``PYTHONWARNDEFAULTENCODING=1`` makes every ``read_text()`` that omits
        ``encoding=`` emit an ``EncodingWarning``, and ``-W
        error::EncodingWarning`` promotes it to an exception, so the pin is a
        real failure rather than a string search over noise. A scanner that
        inherits ``locale.getpreferredencoding()`` decodes the same corpus
        differently on two machines, which for a tool whose entire output is a
        prevalence count is a silent wrong answer, not a crash.

        Deliberately NOT tested by running under ``LC_ALL=C``: that route was
        checked and does not reproduce, because PEP 540 UTF-8 mode coercion
        leaves the C locale decoding UTF-8 anyway. It would be a test that
        passes for a reason unrelated to the property it claims to check.
        """
        result = _run_cli(
            '--root', str(tree), '--json',
            python_args=('-W', 'error::EncodingWarning'),
            env={'PYTHONWARNDEFAULTENCODING': '1'},
        )

        assert result.returncode == 0, (
            f'the CLI read a file in the ambient locale encoding:\n{result.stderr}'
        )
        assert 'EncodingWarning' not in result.stderr, result.stderr
