"""Validate the COMMITTED toolcall-XML sweep artifacts (task 3567).

This module is the anti-fabrication gate for an OPERATIONAL task. Task 3567's
deliverable is not code — it is a measurement: the first real incidence rate of
leaked tool-call XML in the live Mem0/Qdrant corpus. For that kind of task the
dominant failure mode is not a bug, it is a plausible-looking hand-typed
report, and nothing else downstream would catch one.

So this module does not trust the capture. For every record in every committed
report it re-runs the PRODUCTION detector (``classify_record``) over that
record's OWN stored ``content`` and asserts the detector reproduces the
classification the artifact claims. That is exactly the call shape the sweep
itself uses (``classify_record({'data': text})``, script line ~329), so the
test and the sweep share one source of truth rather than re-encoding the leak
rules here. A report typed from imagination cannot satisfy it: the author would
have to hand-craft content that the real detector independently classifies the
claimed way, for every record.

It also pins the run's own disclosure fields — ``exhaustive is True``,
``truncated is False``, ``limit is None``, ``scanned > 0``, no ``aborted`` — so
a partial, capped or aborted sweep can never be committed AS an incidence
measurement. The script discloses its own incompleteness; this makes that
disclosure load-bearing.

## Why this reads files instead of sweeping

A test that re-ran the sweep against Qdrant would be non-deterministic (the
corpus takes concurrent writes) and would fail whenever the stores are down,
converting an infrastructure state into a spurious task failure. Reading the
committed artifact asserts exactly what this task is accountable for — that the
evidence it committed is well-formed, complete and internally consistent — and
stays green independent of infrastructure. This module touches NO live store:
pure file reads plus the pure detector. It mirrors
``test_sweep_toolcall_xml_leak.py``, which drives the whole sweep through an
``AsyncMock`` MemoryService and likewise never opens a socket.

## Why the date is globbed

The glob is ``toolcall-xml-leak-sweep-*`` rather than the one 2026-08-05
directory, so a LATER capture committed by a future sweep is held to the same
contract automatically instead of being validated once and then drifting.

## Sentinel-literal hazard (inherited from task 3083)

Do not paste a raw tool-call sentinel into this file. Spell ``<`` as ``\\x3c``
in any leak literal, as ``test_sweep_toolcall_xml_leak.py`` does — writing one
verbatim makes THIS file's own authoring tool call terminate early,
reproducing the very bug under test. This module needs no such literal today
(every specimen it checks is read from disk), and it should stay that way.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'sweep_toolcall_xml_leak.py'

# Mirrors fused-memory/tests/conftest.py:54 rather than inventing a second
# root-finding scheme.
REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_GLOB = 'toolcall-xml-leak-sweep-*'


def _load_module() -> types.ModuleType:
    """Load sweep_toolcall_xml_leak.py from its file path.

    Reuse of test_sweep_toolcall_xml_leak.py's loader: scripts/ is not a
    package and the script is not on PYTHONPATH, so importlib is how the
    production detector is reached without sys.path pollution.

    One deliberate difference from that copy: an already-loaded module for the
    SAME file is reused rather than re-executed. Both test modules register the
    key ``sweep_toolcall_xml_leak`` in ``sys.modules``, so an unconditional
    load makes the second import execute the 965-line script a second time and
    silently REPLACE the first module object -- two live module objects whose
    identity depends on collection order. Harmless while only pure functions
    are used, but a latent hazard the moment anything holds module state.

    The remaining half of the reviewer's ask -- hoisting this loader into
    ``fused-memory/tests/conftest.py`` so the loading contract has ONE
    definition -- is not applied here: conftest.py is outside this task's
    locked module set.
    """
    mod_name = 'sweep_toolcall_xml_leak'
    cached = sys.modules.get(mod_name)
    cached_file = getattr(cached, '__file__', None)
    if cached is not None and cached_file is not None and (
        Path(cached_file).resolve() == SCRIPT_PATH.resolve()
    ):
        return cached
    spec = importlib.util.spec_from_file_location(mod_name, SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {SCRIPT_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(mod_name, None)
        raise
    return module


_mod = _load_module()

classify_record = _mod.classify_record
CLASSIFICATIONS = _mod.CLASSIFICATIONS
CLEAN = _mod.CLEAN
# Sourced from the script, never re-typed. A literal 'manual_review' here would
# make test_manual_review_records_were_never_mutated skip every record and pass
# VACUOUSLY the day the name changes -- green, not red, which is the failure
# direction this module exists to eliminate.
MANUAL_REVIEW = _mod.MANUAL_REVIEW
REPAIRABLE_TAIL = _mod.REPAIRABLE_TAIL
REPAIRABLE_DUPLICATE = _mod.REPAIRABLE_DUPLICATE

# The exact key set build_report() emits (script line ~508). Asserted as an
# EQUALITY, not a subset: an aborted run adds 'aborted', and a report carrying
# it covered only part of the corpus and is not an incidence measurement.
REPORT_KEYS = frozenset(
    {
        'project_id',
        'collection',
        'dry_run',
        'exhaustive',
        'scanned',
        'truncated',
        'limit',
        'counts',
        'repaired',
        'records',
    }
)


def _report_paths() -> list[Path]:
    """Every committed sweep report, oldest directory first."""
    return sorted((REPO_ROOT / 'docs').glob(f'{ARTIFACT_GLOB}/*-report.json'))


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _provenance_path(report_path: Path) -> Path:
    """The sidecar that belongs to *report_path*.

    ``<name>-report.json`` pairs with ``<name>-provenance.json``, so the two
    halves of one capture cannot drift apart or be silently reattributed.
    """
    return report_path.with_name(
        report_path.name.removesuffix('-report.json') + '-provenance.json'
    )


_SHA_RE = re.compile(r'\A[0-9a-f]{40}\Z')

# The four per-record outcomes the runbook (docs/mcp-toolcall-xml-leak.md §5
# remediation table) says a HUMAN must adjudicate. Lifted verbatim as a
# vocabulary rather than re-derived, so this module's notion of "a mutation
# somebody has to deal with" is the runbook's, not a fresh invention.
#
# There is no shared constant to import: the script spells the same four
# inline in resolve_exit_code(). That makes this tuple a HAND COPY, and a hand
# copy of a growth-prone list is a silent-drift hazard -- add a fifth danger
# outcome to the sweep and _unrepaired_danger_records() would return [] for a
# record carrying it, every test below would hit its "if not outstanding"
# early exit, and the whole class would go GREEN while a real unrepaired live
# mutation landed untracked. TestDangerFlagsMatchTheProductionScript pins this
# tuple against resolve_exit_code's own source so that drift is RED, not
# silent. (Exporting the constant FROM the script, as the reviewer suggested,
# is the cleaner fix but edits scripts/sweep_toolcall_xml_leak.py, which is
# outside this task's locked module set.)
DANGER_FLAGS = (
    'record_error',
    'content_lost_in_flight',
    'skipped_not_mem0_routed',
    'skipped_metadata_would_be_rejected',
)

# The two shapes repair_content() knows how to fix. Only these carry a
# repaired_content, so only for these can the recovery payload be checked for
# leak-freeness. Imported, not re-typed -- see MANUAL_REVIEW above.
REPAIRABLE_CLASSIFICATIONS = (REPAIRABLE_TAIL, REPAIRABLE_DUPLICATE)

# NAMING: deliberately NOT '*-report.json'. _report_paths() globs that suffix,
# so a tracking file named that way would be swept into _REPORT_PATHS and fail
# every shape and provenance test in this module.
RECOVERY_TRACKING_NAME = 'recovery-tracking.json'

_DIGITS_RE = re.compile(r'\A[0-9]+\Z')

_SHA256_RE = re.compile(r'\A[0-9a-f]{64}\Z')

# How the recovery payload is pinned: by CONTENT, not by commit. Each entry
# maps a tracking-file digest field to the payload-record field it must be the
# sha256 of. A commit sha cannot do this job on a branch the merge lane
# rebases — it is an unreachable orphan by the time the work reaches main —
# whereas a digest over the bytes survives any rebase, and is checkable
# hermetically from the working tree with no git invocation at all.
_PAYLOAD_DIGEST_FIELDS = {
    'payload_content_sha256': 'content',
    'payload_repaired_content_sha256': 'repaired_content',
}


def _ids(paths: list[Path]) -> list[str]:
    return [f'{p.parent.name}/{p.name}' for p in paths]


_REPORT_PATHS = _report_paths()

# One entry per dated capture directory. The converse tracking check is a
# property of a DIRECTORY (its tracking file vs all its reports), not of a
# single report, so it is parametrized over these instead.
_ARTIFACT_DIRS = sorted({p.parent for p in _REPORT_PATHS})


def _recovery_tracking_path(artifact_dir: Path) -> Path:
    """The tracking file that belongs to a capture directory.

    Derived from the directory rather than hard-coding the 2026-08-05 date, so
    a future capture is held to exactly the same contract.
    """
    return artifact_dir / RECOVERY_TRACKING_NAME


def _unrepaired_danger_records(report: dict) -> list[dict]:
    """Records where the sweep ATTEMPTED a mutation and did not repair.

    Dry runs are excluded by construction: they mutate nothing, so a flag on
    one is not an outstanding live-corpus change.
    """
    if report.get('dry_run') is not False:
        return []
    return [
        record
        for record in report['records']
        if any(record.get(flag) for flag in DANGER_FLAGS)
        and record.get('repaired') is not True
    ]


# ---------------------------------------------------------------------------
# Vocabulary drift — the hand copy vs the production script
# ---------------------------------------------------------------------------


def _production_danger_flags() -> tuple[set[str], str]:
    """The danger-flag set the SCRIPT actually acts on, plus where it came from.

    Prefers a real exported constant if the script ever grows one (the clean
    fix, which needs an edit outside this task's locked scope). Falls back to
    reading the flag names straight out of ``resolve_exit_code``'s source: that
    function IS the production consumer of these flags, so its own text is the
    closest available substitute for a shared constant.
    """
    for name in ('HUMAN_ADJUDICATION_FLAGS', 'DANGER_FLAGS'):
        exported = getattr(_mod, name, None)
        if exported is not None:
            return set(exported), f'{name} exported by the script'
    source = inspect.getsource(_mod.resolve_exit_code)
    # Only record.get(...) — report.get('counts'/'truncated'/'records') are the
    # run-level conditions, not per-record human-adjudication outcomes.
    names = re.findall(r"record\.get\(\s*'([A-Za-z_][A-Za-z0-9_]*)'", source)
    return set(names), 'resolve_exit_code() source'


class TestDangerFlagsMatchTheProductionScript:
    """DANGER_FLAGS is a hand copy. This is what keeps it honest.

    The script spells the same four names inline in ``resolve_exit_code`` and
    exports no shared constant, so nothing structurally prevents the two lists
    from diverging. Divergence here fails in the WORST direction: a fifth
    danger outcome added upstream would make ``_unrepaired_danger_records()``
    return ``[]`` for a record carrying it, every test in
    ``TestUnrepairedMutationsAreTracked`` would take its ``if not outstanding``
    early exit, and the suite would report all-green while a real unrepaired
    live-corpus mutation landed with no tracking entry required — this module's
    own failure class, reintroduced one level up.
    """

    def test_hand_copy_equals_what_the_script_acts_on(self) -> None:
        production, origin = _production_danger_flags()
        assert set(DANGER_FLAGS) == production, (
            f'DANGER_FLAGS {sorted(DANGER_FLAGS)} != the flags the script acts '
            f'on {sorted(production)} (read from {origin}).'
            '\n  missing here: '
            f'{sorted(production - set(DANGER_FLAGS))}'
            '\n  stale here:   '
            f'{sorted(set(DANGER_FLAGS) - production)}'
            '\nRe-bind DANGER_FLAGS. Do NOT delete this assertion: while the '
            'two lists disagree, an unrepaired mutation carrying a flag this '
            'module does not know about is tracked by nobody and the whole '
            'TestUnrepairedMutationsAreTracked class silently no-ops.'
        )

    def test_no_duplicates_in_the_hand_copy(self) -> None:
        assert len(set(DANGER_FLAGS)) == len(DANGER_FLAGS), (
            f'DANGER_FLAGS {DANGER_FLAGS} repeats a name'
        )

    @pytest.mark.parametrize('flag', DANGER_FLAGS)
    def test_each_flag_forces_a_non_zero_exit(self, flag: str) -> None:
        """Behavioural confirmation, independent of how the source is parsed.

        Each name must be one the production exit-code predicate genuinely
        treats as needing a human — not merely a string that appears in the
        file.
        """
        clean_apply = {
            'dry_run': False,
            'truncated': False,
            'counts': dict.fromkeys(CLASSIFICATIONS, 0),
            'records': [{'id': 'synthetic', 'classification': CLEAN}],
        }
        assert _mod.resolve_exit_code(clean_apply) == 0, (
            'baseline synthetic apply report should exit 0; the fixture, not '
            'the flag, is wrong'
        )
        flagged = json.loads(json.dumps(clean_apply))
        flagged['records'][0][flag] = True
        assert _mod.resolve_exit_code(flagged) == 1, (
            f'{flag!r} does not force a non-zero exit in resolve_exit_code, so '
            'it is not one of the outcomes the script escalates to a human. '
            'Either it is misspelled here or it no longer belongs in '
            'DANGER_FLAGS.'
        )


# ---------------------------------------------------------------------------
# Existence — the assertion an empty parametrize set cannot make
# ---------------------------------------------------------------------------


class TestArtifactsExist:
    """A parametrized suite over zero artifacts is silently green. This is not."""

    def test_at_least_one_report_committed(self) -> None:
        assert _REPORT_PATHS, (
            f'No sweep report found under docs/{ARTIFACT_GLOB}/*-report.json. '
            'Task 3567 delivers a captured live-corpus sweep; without a '
            'committed report there is no measurement.'
        )

    def test_at_least_one_report_is_a_dry_run(self) -> None:
        """The pre-mutation capture is the one that must always survive.

        Per the runbook, a repair is delete-then-re-add: if the delete lands
        and the re-add does not persist, the original text survives ONLY in the
        printed report. The dry run is that safety net, so its presence is a
        contract, not a nicety.
        """
        dry_runs = [p for p in _REPORT_PATHS if _load(p).get('dry_run') is True]
        assert dry_runs, (
            'No committed report has dry_run=true. The pre-mutation dry-run '
            'capture is the only surviving copy of any content lost in flight.'
        )


# ---------------------------------------------------------------------------
# Report shape and self-consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('path', _REPORT_PATHS, ids=_ids(_REPORT_PATHS))
class TestReportShape:
    def test_parses_and_has_exact_key_set(self, path: Path) -> None:
        report = _load(path)
        assert isinstance(report, dict)
        # Checked before the key-set equality so an aborted run fails with the
        # reason rather than with a confusing key-set diff.
        assert not report.get('aborted'), (
            f'{path.name} carries aborted=true: the run covered only part of '
            'the corpus and must not be committed as the incidence rate. '
            'Keep the partial output (it may hold the only copy of some '
            "record's text) and escalate."
        )
        assert set(report) == REPORT_KEYS, (
            f'{path.name} key set differs from build_report()'
            f'\n  missing: {sorted(REPORT_KEYS - set(report))}'
            f'\n  extra:   {sorted(set(report) - REPORT_KEYS)}'
        )

    def test_run_was_authoritative_and_complete(self, path: Path) -> None:
        """A truncated, capped or zero-scan run is not an incidence rate.

        ``--exhaustive`` skips the server-side MatchText prefilter and
        paginates the whole collection, so the result depends on nothing but
        the shared Python detector. Runbook §7: MatchText on the un-indexed
        ``data`` field would silently flip to tokenized matching if a payload
        index were ever added, so only the exhaustive walk is authoritative.
        """
        report = _load(path)
        assert report['exhaustive'] is True, 'run must be --exhaustive'
        assert report['truncated'] is False, 'run must not be truncated'
        assert report['limit'] is None, 'run must not be --limit capped'
        assert isinstance(report['scanned'], int)
        assert report['scanned'] > 0, (
            'scanned==0 means nothing was walked — an unreachable or '
            'wrong-scoped store reads as a clean corpus, the exact '
            'silent-wrong-answer this line of work exists to kill.'
        )

    def test_counts_are_the_classification_vocabulary(self, path: Path) -> None:
        report = _load(path)
        counts = report['counts']
        assert set(counts) == set(CLASSIFICATIONS), (
            f'counts keys {sorted(counts)} != CLASSIFICATIONS '
            f'{sorted(CLASSIFICATIONS)}'
        )
        assert all(isinstance(v, int) for v in counts.values())

    def test_counts_reconcile_with_records(self, path: Path) -> None:
        report = _load(path)
        counts = report['counts']
        records = report['records']
        assert sum(counts.values()) == len(records), (
            f'counts sum {sum(counts.values())} != len(records) '
            f'{len(records)} — the summary and the evidence disagree.'
        )
        tallied: dict[str, int] = {name: 0 for name in CLASSIFICATIONS}
        for record in records:
            tallied[record['classification']] += 1
        assert tallied == counts, (
            f'per-classification tally {tallied} != reported counts {counts}'
        )

    def test_scanned_is_at_least_the_records_returned(self, path: Path) -> None:
        """Records are a subset of the points walked.

        This is the ONLY relation asserted between scanned and anything else.
        Deliberately no ratio against the live Qdrant point count: the corpus
        takes concurrent writes, consolidation actively DELETES entries, and
        scan_payload_text scopes by group_id while a collection count does not
        — so no such bound is sound. Coverage interpretation belongs in the
        prose, where a human reads it.
        """
        report = _load(path)
        assert report['scanned'] >= len(report['records'])

    def test_repaired_count_reconciles(self, path: Path) -> None:
        report = _load(path)
        actual = sum(1 for r in report['records'] if r.get('repaired'))
        assert report['repaired'] == actual
        if report['dry_run'] is True:
            assert report['repaired'] == 0, (
                'a dry run mutates nothing, so it cannot have repaired a record'
            )


# ---------------------------------------------------------------------------
# The load-bearing check: re-derive every verdict with the real detector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('path', _REPORT_PATHS, ids=_ids(_REPORT_PATHS))
class TestDetectorReproducesEveryClassification:
    def test_every_classification_is_in_the_vocabulary(self, path: Path) -> None:
        report = _load(path)
        for record in report['records']:
            assert record['classification'] in CLASSIFICATIONS, (
                f'record {record.get("id")!r} carries unknown '
                f'classification {record["classification"]!r}'
            )

    def test_production_detector_reproduces_each_verdict(self, path: Path) -> None:
        """Re-classify each record's own stored content and compare.

        ``classify_record({'data': text})`` is the exact call the sweep makes,
        so this re-derives every verdict from the evidence the artifact itself
        carries. A fabricated report fails here.
        """
        report = _load(path)
        mismatches = []
        for record in report['records']:
            claimed = record['classification']
            derived = classify_record({'data': record['content']})
            if derived != claimed:
                mismatches.append(
                    f'  {record.get("id")!r}: claims {claimed!r}, '
                    f'detector says {derived!r}'
                )
        assert not mismatches, (
            f'{path.name}: the production detector disagrees with the '
            f'recorded classification for {len(mismatches)} record(s). The '
            'report does not describe its own evidence:\n' + '\n'.join(mismatches)
        )

    def test_every_repair_actually_removes_the_leak(self, path: Path) -> None:
        """A repaired_content that still classifies dirty is not a repair."""
        report = _load(path)
        checked = 0
        for record in report['records']:
            repaired = record.get('repaired_content')
            if repaired is None:
                continue
            checked += 1
            derived = classify_record({'data': repaired})
            assert derived == CLEAN, (
                f'record {record.get("id")!r} has repaired_content the '
                f'detector still classifies {derived!r}, not {CLEAN!r}'
            )
            assert repaired, 'a repair must never leave the memory empty'
        if checked:
            print(f'{path.name}: verified {checked} repair(s) land clean')

    def test_manual_review_records_were_never_mutated(self, path: Path) -> None:
        """The sweep refuses to touch manual_review by construction.

        Asserted on the artifact too, so a future run that regressed that
        refusal cannot land its evidence quietly.
        """
        report = _load(path)
        for record in report['records']:
            if record['classification'] != MANUAL_REVIEW:
                continue
            assert not record.get('repaired'), (
                f'manual_review record {record.get("id")!r} is marked '
                'repaired — the sweep must never mutate one'
            )
            assert record.get('repaired_content') is None, (
                f'manual_review record {record.get("id")!r} carries a '
                'repaired_content; repair_content() returns None to REFUSE'
            )


# ---------------------------------------------------------------------------
# Provenance sidecar — ties a committed report back to the run that made it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('path', _REPORT_PATHS, ids=_ids(_REPORT_PATHS))
class TestProvenanceSidecar:
    """Every report must be attributable to a real, identified run.

    The sidecar exists for a specific reason beyond bookkeeping. ``new_progress()``
    (script line ~532) seeds ``collection`` to ``''`` and the field can land
    BLANK in the emitted report — which it did for this capture, and which is
    exactly the defect task 3243 (still pending) will fix upstream. A report
    that does not say which collection it swept cannot be trusted as a
    measurement of that collection.

    Recording the collection out-of-band here closes that gap LOCALLY, for this
    capture, without editing the sweep script — which would duplicate and
    probably conflict with 3243's work, and would put production-code changes
    inside a task whose whole point is an operational run.
    """

    def test_sidecar_exists(self, path: Path) -> None:
        sidecar = _provenance_path(path)
        assert sidecar.is_file(), (
            f'{path.name} has no provenance sidecar at {sidecar.name}. A report '
            'that cannot be tied back to the run that produced it -- the sha, '
            'the argv, the collection, the bracketing point counts -- is not '
            'auditable evidence.'
        )

    def test_records_the_collection_the_report_may_omit(self, path: Path) -> None:
        prov = _load(_provenance_path(path))
        collection = prov['collection']
        assert isinstance(collection, str)
        assert collection.strip(), (
            'provenance.collection is blank. This field is the whole reason the '
            "sidecar exists: the sweep's own report can emit collection='' "
            '(task 3243), so if the sidecar is blank too, nothing identifies '
            'what was swept.'
        )

    def test_point_counts_bracket_the_run(self, path: Path) -> None:
        """Measured exact Qdrant counts either side of the sweep.

        Deliberately NO assertion relating these to report['scanned']. The
        corpus takes concurrent writes so the count drifts during the run;
        consolidation actively DELETES entries (this task's own premise) so it
        is not even monotonic and points_after can fall BELOW scanned; and
        scan_payload_text scopes by group_id while a collection-level count does
        not, so the two are different populations. Any ratio would be a guess
        tuned to whatever the first run happened to print.
        """
        prov = _load(_provenance_path(path))
        for field in ('collection_points_before', 'collection_points_after'):
            value = prov[field]
            assert isinstance(value, int) and not isinstance(value, bool), (
                f'{field} must be an int, got {type(value).__name__}'
            )
            assert value > 0, (
                f'{field}={value}: a zero or absent count must never be '
                'recorded as a clean corpus -- an unreachable or wrong-scoped '
                'store returns an empty match list that reads exactly like one.'
            )

    def test_git_sha_is_a_full_sha(self, path: Path) -> None:
        prov = _load(_provenance_path(path))
        sha = prov['git_sha']
        assert isinstance(sha, str)
        assert _SHA_RE.match(sha), (
            f'git_sha {sha!r} is not a 40-char lowercase hex sha; an '
            'abbreviated or dirty-marked sha does not pin the code that ran.'
        )

    def test_swept_at_is_parseable_iso8601(self, path: Path) -> None:
        prov = _load(_provenance_path(path))
        swept_at = prov['swept_at']
        assert isinstance(swept_at, str)
        parsed = datetime.fromisoformat(swept_at.replace('Z', '+00:00'))
        assert parsed.tzinfo is not None, (
            f'swept_at {swept_at!r} carries no timezone; a naive timestamp on a '
            'shared live corpus is not a fact anyone can line up with anything.'
        )

    def test_command_is_the_argv_actually_run(self, path: Path) -> None:
        """The argv is what makes the report's own claims checkable.

        A report can SAY exhaustive=true; the recorded command is the
        independent record of what was actually invoked.
        """
        prov = _load(_provenance_path(path))
        command = prov['command']
        assert isinstance(command, list)
        assert command, 'command must not be empty'
        assert all(isinstance(part, str) for part in command)
        assert '--exhaustive' in command, (
            f'command {command!r} lacks --exhaustive; only the exhaustive walk '
            'is the authoritative incidence rate (runbook 7).'
        )
        assert not any(part.startswith('--limit') for part in command), (
            f'command {command!r} carries --limit; a capped run covers an '
            'unknown fraction of the corpus.'
        )
        report = _load(path)
        assert ('--apply' in command) is (report['dry_run'] is False), (
            f'command {command!r} and report dry_run={report["dry_run"]} '
            'disagree about whether this run mutated the store.'
        )

    def test_exit_code_is_recorded(self, path: Path) -> None:
        prov = _load(_provenance_path(path))
        exit_code = prov['exit_code']
        assert isinstance(exit_code, int) and not isinstance(exit_code, bool)
        assert exit_code != 2, (
            'exit 2 means the run aborted partway and covered only part of the '
            'corpus; it is not an incidence measurement.'
        )


# ---------------------------------------------------------------------------
# Unrepaired live-corpus mutations: named owner + still-recoverable payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('path', _REPORT_PATHS, ids=_ids(_REPORT_PATHS))
class TestUnrepairedMutationsAreTracked:
    """An unrepaired mutation of the LIVE corpus must not land on prose alone.

    This class exists because of what actually happened on this task's first
    ``--apply``: a record was deleted from live Qdrant and never re-added
    (``record_error``, ``repaired: false``). The only things standing between
    that and permanent data loss were a paragraph in ``investigation.md`` and
    escalation ``esc-3567-2``, which was AUTO-DISMISSED on timeout rather than
    adjudicated. Prose is not a tracker, and a dismissed escalation is not an
    owner.

    So the tracker becomes a machine-checked artifact. Two things are asserted
    that prose can never assert:

    1. every unrepaired mutation has a NAMED, durable owner (a Taskmaster id);
    2. the recovery PAYLOAD is still present in a committed report and is still
       judged leak-free by the production detector.

    (2) is the part that keeps mattering after today. A future "tidy up these
    big JSON artifacts" commit that drops the report holding the only surviving
    copy of the deleted text turns this suite RED instead of quietly making the
    loss permanent — the repo's loud-over-silent norm, applied to evidence.

    Hermetic like the rest of the module: pure file reads plus the pure
    detector, so it is green in VERIFY whether or not Qdrant is up.
    """

    def test_unrepaired_mutation_has_a_tracking_entry(self, path: Path) -> None:
        report = _load(path)
        outstanding = _unrepaired_danger_records(report)
        if not outstanding:
            return
        tracking_path = _recovery_tracking_path(path.parent)
        assert tracking_path.is_file(), (
            f'{path.name} records {len(outstanding)} unrepaired live-corpus '
            f'mutation(s) but {path.parent.name}/{RECOVERY_TRACKING_NAME} does '
            'not exist. A mutation that changed the shared store and did not '
            'complete needs a tracked owner, not a paragraph.'
        )
        tracking = _load(tracking_path)
        assert isinstance(tracking, dict), (
            f'{RECOVERY_TRACKING_NAME} must be a JSON object keyed by memory id'
        )
        missing = [r['id'] for r in outstanding if r['id'] not in tracking]
        assert not missing, (
            f'{path.name}: unrepaired mutation(s) {missing} have no entry in '
            f'{RECOVERY_TRACKING_NAME}. This is precisely the failure this '
            'check exists to catch — a real change to the live corpus with '
            'nobody accountable for finishing it.'
        )

    def test_tracking_entry_names_a_durable_task_owner(self, path: Path) -> None:
        """A Taskmaster id, asserted by SHAPE so it survives any id minted."""
        report = _load(path)
        outstanding = _unrepaired_danger_records(report)
        if not outstanding:
            return
        tracking = _load(_recovery_tracking_path(path.parent))
        for record in outstanding:
            entry = tracking[record['id']]
            owner = entry.get('tracked_as_task')
            assert isinstance(owner, str) and _DIGITS_RE.match(owner), (
                f'{record["id"]}: tracked_as_task={owner!r} is not a bare '
                'all-digits task id. An escalation can auto-dismiss on '
                'timeout; a task is the durable owner.'
            )

    def test_tracking_entry_points_at_a_committed_payload(self, path: Path) -> None:
        report = _load(path)
        outstanding = _unrepaired_danger_records(report)
        if not outstanding:
            return
        tracking = _load(_recovery_tracking_path(path.parent))
        repo_root = REPO_ROOT.resolve()
        for record in outstanding:
            entry = tracking[record['id']]
            rel = entry.get('payload_source_report')
            assert isinstance(rel, str) and rel, (
                f'{record["id"]}: payload_source_report is missing; nothing '
                'names where the recoverable text lives.'
            )
            assert not Path(rel).is_absolute(), (
                f'{record["id"]}: payload_source_report {rel!r} is absolute; '
                'it must be repo-relative so it survives a different checkout.'
            )
            resolved = (REPO_ROOT / rel).resolve()
            assert resolved.is_relative_to(repo_root), (
                f'{record["id"]}: payload_source_report {rel!r} escapes the '
                'repo; the payload must be a committed artifact.'
            )
            assert resolved.is_file(), (
                f'{record["id"]}: payload_source_report {rel!r} does not '
                'exist. The recovery payload is GONE — this is the deletion '
                'this check is here to make loud.'
            )
            # Membership in the validated set, NOT merely a '-report.json'
            # suffix: only the paths in _REPORT_PATHS are the ones every other
            # class in this module actually checks (shape, provenance,
            # detector reproduction). A file elsewhere in the repo could match
            # the suffix while never having been validated at all, so the
            # suffix test would assert something weaker than its own comment
            # claimed.
            assert resolved in {p.resolve() for p in _REPORT_PATHS}, (
                f'{record["id"]}: payload_source_report {rel!r} is not one of '
                'the validated captures under '
                f'docs/{ARTIFACT_GLOB}/. The payload must live in a report '
                'this module checks end to end, otherwise its integrity is '
                'asserted nowhere.'
            )
            required = ['payload_content_sha256']
            if entry.get('classification') in REPAIRABLE_CLASSIFICATIONS:
                # Only these carry a repaired_content to hash.
                required.append('payload_repaired_content_sha256')
            for field in required:
                digest = entry.get(field)
                assert isinstance(digest, str) and _SHA256_RE.match(digest), (
                    f'{record["id"]}: {field}={digest!r} is not a 64-char '
                    'lowercase hex sha256. The payload is pinned by CONTENT, '
                    'not by commit — see test_payload_digests_reproduce.'
                )
            assert 'payload_source_commit' not in entry, (
                f'{record["id"]}: payload_source_commit is back. A commit sha '
                'naming a same-branch commit is broken BY CONSTRUCTION here: '
                'the merge lane rebases the branch, so the sha is an '
                'unreachable orphan by the time the work lands on main and '
                '`git show <sha>:<path>` fails once gc prunes it — taking the '
                'documented recovery path with it. Pin the payload with the '
                'repo-relative path plus the sha256 digests instead.'
            )

    def test_recovered_flag_and_new_id_agree(self, path: Path) -> None:
        """``recovered`` must never overstate what happened.

        While it is False the entry carries no ``new_id`` (nothing was
        re-added, so there is no id to carry). If a future session flips it to
        True it MUST carry the id of the memory it re-added — otherwise
        "recovered" is an unfalsifiable claim.
        """
        report = _load(path)
        outstanding = _unrepaired_danger_records(report)
        if not outstanding:
            return
        tracking = _load(_recovery_tracking_path(path.parent))
        for record in outstanding:
            entry = tracking[record['id']]
            recovered = entry.get('recovered')
            assert isinstance(recovered, bool), (
                f'{record["id"]}: recovered={recovered!r} must be a bool'
            )
            if recovered:
                new_id = entry.get('new_id')
                assert isinstance(new_id, str) and new_id.strip(), (
                    f'{record["id"]}: recovered=true but new_id={new_id!r}. '
                    'A re-added memory necessarily gets a NEW id; without it '
                    'the recovery claim cannot be checked against the store.'
                )
            else:
                assert 'new_id' not in entry, (
                    f'{record["id"]}: recovered=false but a new_id is present. '
                    'Nothing has been re-added; the two disagree.'
                )

    def test_recovery_payload_is_still_present_and_leak_free(
        self, path: Path
    ) -> None:
        """THE load-bearing assertion: the documented recovery is executable.

        Tracking an unrepaired mutation is worthless if the text a human would
        re-add has since been deleted, emptied, or is itself still leaking. So
        the payload is opened, located by the SAME memory id, and — where the
        record was repairable — its ``repaired_content`` is re-run through the
        production detector. That re-derives from the real detector that the
        text on the recovery path is genuinely clean, rather than trusting a
        prose promise that it is.
        """
        report = _load(path)
        outstanding = _unrepaired_danger_records(report)
        if not outstanding:
            return
        tracking = _load(_recovery_tracking_path(path.parent))
        for record in outstanding:
            memory_id = record['id']
            entry = tracking[memory_id]
            payload_path = (REPO_ROOT / entry['payload_source_report']).resolve()
            payload_report = _load(payload_path)
            matches = [
                r for r in payload_report['records'] if r.get('id') == memory_id
            ]
            assert matches, (
                f'{memory_id}: payload_source_report '
                f'{entry["payload_source_report"]} carries no record with that '
                'id. The tracking entry points at a report that cannot recover '
                'it.'
            )
            payload = matches[0]
            assert payload.get('content'), (
                f'{memory_id}: the payload record has empty content. The only '
                'surviving copy of the deleted text is gone.'
            )
            if payload['classification'] in REPAIRABLE_CLASSIFICATIONS:
                repaired = payload.get('repaired_content')
                assert repaired, (
                    f'{memory_id}: classified {payload["classification"]!r} '
                    'but the payload carries no repaired_content, so there is '
                    'nothing clean to re-add.'
                )
                derived = classify_record({'data': repaired})
                assert derived == CLEAN, (
                    f'{memory_id}: the repaired_content a human would re-add '
                    f'is still classified {derived!r} by the production '
                    f'detector, not {CLEAN!r}. Recovering it would re-introduce '
                    'the leak.'
                )

    def test_payload_digests_reproduce(self, path: Path) -> None:
        """The durability pin must be REPRODUCIBLE, not merely well-shaped.

        This is the assertion the previous commit-sha pin could never make. A
        sha recorded for a same-branch commit is stale by construction — the
        merge lane rebases, so it names an orphan that ``git gc`` eventually
        prunes — and a test can only ever check its SHAPE, which is exactly as
        true of a fabricated sha as of a real one. A content digest instead
        re-derives from the committed bytes: the payload is located by the same
        memory id and re-hashed here, so an edited, truncated, swapped or
        hand-typed payload fails. Hermetic — file reads and hashlib, no git.
        """
        report = _load(path)
        outstanding = _unrepaired_danger_records(report)
        if not outstanding:
            return
        tracking = _load(_recovery_tracking_path(path.parent))
        for record in outstanding:
            memory_id = record['id']
            entry = tracking[memory_id]
            payload_report = _load(
                (REPO_ROOT / entry['payload_source_report']).resolve()
            )
            matches = [
                r for r in payload_report['records'] if r.get('id') == memory_id
            ]
            assert matches, (
                f'{memory_id}: payload_source_report carries no record with '
                'that id; there is nothing to hash.'
            )
            payload = matches[0]
            for field, payload_field in _PAYLOAD_DIGEST_FIELDS.items():
                claimed = entry.get(field)
                if claimed is None:
                    continue
                text = payload.get(payload_field)
                assert isinstance(text, str), (
                    f'{memory_id}: {field} is recorded but the payload record '
                    f'has no {payload_field!r} string to hash it against.'
                )
                actual = hashlib.sha256(text.encode('utf-8')).hexdigest()
                assert actual == claimed, (
                    f'{memory_id}: {field} claims {claimed}, but the '
                    f'{payload_field!r} committed at '
                    f'{entry["payload_source_report"]} hashes to {actual}. '
                    'The recovery payload is not the text this entry pins — '
                    'either it was altered after the digest was taken, or the '
                    'digest was never derived from it.'
                )
            # The char counts are human-facing commentary on the same bytes;
            # if they disagree with the payload the entry is internally
            # inconsistent and a reader cannot tell which half to believe.
            for count_field, payload_field in (
                ('payload_content_chars', 'content'),
                ('payload_repaired_content_chars', 'repaired_content'),
            ):
                claimed_len = entry.get(count_field)
                if claimed_len is None:
                    continue
                assert claimed_len == len(payload.get(payload_field) or ''), (
                    f'{memory_id}: {count_field}={claimed_len} disagrees with '
                    f'the committed {payload_field!r} '
                    f'({len(payload.get(payload_field) or "")} chars).'
                )
            # Same principle for the two remaining self-describing fields.
            # They restate facts the reports already carry, so a hand-edited
            # or copy-pasted entry that misstates them would otherwise pass
            # every check here — the exact fabrication class this module
            # exists to close. Both are pure file reads.
            claimed_classification = entry.get('classification')
            if claimed_classification is not None:
                assert claimed_classification == payload['classification'], (
                    f'{memory_id}: entry claims classification '
                    f'{claimed_classification!r}, but the payload committed '
                    f'at {entry["payload_source_report"]} is classified '
                    f'{payload["classification"]!r}. The entry decides from '
                    'this field whether a repaired_content is required, so a '
                    'wrong value silently weakens the recovery check.'
                )
            claimed_flag = entry.get('flag')
            if claimed_flag is not None:
                assert claimed_flag in DANGER_FLAGS, (
                    f'{memory_id}: flag={claimed_flag!r} is not one of the '
                    f'runbook danger flags {DANGER_FLAGS}.'
                )
                assert record.get(claimed_flag), (
                    f'{memory_id}: entry names flag {claimed_flag!r}, but that '
                    f'flag is not truthy on the record in {path.name}. The '
                    'tracker describes a mutation the report does not.'
                )


@pytest.mark.parametrize(
    'artifact_dir', _ARTIFACT_DIRS, ids=[d.name for d in _ARTIFACT_DIRS]
)
class TestRecoveryTrackingHasNoPhantomEntries:
    """The converse: a tracking entry must correspond to a real mutation.

    Without this, a fabricated or stale entry could stand in for a real one —
    the tracker would look populated while the actual unrepaired mutation went
    unowned, which is the same silent failure in a new costume.
    """

    def test_every_tracked_id_is_a_real_unrepaired_mutation(
        self, artifact_dir: Path
    ) -> None:
        tracking_path = _recovery_tracking_path(artifact_dir)
        if not tracking_path.is_file():
            return
        tracking = _load(tracking_path)
        real: set[str] = set()
        for report_path in sorted(artifact_dir.glob('*-report.json')):
            for record in _unrepaired_danger_records(_load(report_path)):
                real.add(record['id'])
        phantom = sorted(set(tracking) - real)
        assert not phantom, (
            f'{artifact_dir.name}/{RECOVERY_TRACKING_NAME} tracks {phantom}, '
            'which no committed apply report in that directory records as an '
            'unrepaired mutation. A tracking file that does not describe real '
            'store state reads as evidence while protecting nothing.'
        )
