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

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / 'scripts' / 'sweep_toolcall_xml_leak.py'

# Mirrors fused-memory/tests/conftest.py:54 rather than inventing a second
# root-finding scheme.
REPO_ROOT = Path(__file__).resolve().parents[2]

ARTIFACT_GLOB = 'toolcall-xml-leak-sweep-*'


def _load_module() -> types.ModuleType:
    """Load sweep_toolcall_xml_leak.py from its file path.

    Verbatim reuse of test_sweep_toolcall_xml_leak.py's loader: scripts/ is not
    a package and the script is not on PYTHONPATH, so importlib is how the
    production detector is reached without sys.path pollution.
    """
    mod_name = 'sweep_toolcall_xml_leak'
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


def _ids(paths: list[Path]) -> list[str]:
    return [f'{p.parent.name}/{p.name}' for p in paths]


_REPORT_PATHS = _report_paths()


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
                f'record {record.get("memory_id")!r} carries unknown '
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
                    f'  {record.get("memory_id")!r}: claims {claimed!r}, '
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
                f'record {record.get("memory_id")!r} has repaired_content the '
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
            if record['classification'] != 'manual_review':
                continue
            assert not record.get('repaired'), (
                f'manual_review record {record.get("memory_id")!r} is marked '
                'repaired — the sweep must never mutate one'
            )
            assert record.get('repaired_content') is None, (
                f'manual_review record {record.get("memory_id")!r} carries a '
                'repaired_content; repair_content() returns None to REFUSE'
            )
