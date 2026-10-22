"""``seam_scan.py`` listing G4 seam candidates through its CLI, over a seeded home and a store copy.

The home has S1 and S2 authored and S1's applied author return in ``returns/``;
the store is a copy of ``store.json`` that each case may edit in place. The
return's prd_path and sidecar_path exist nowhere on disk, so every candidate
can only come from the store. Assertions name task ids, touched paths and check
names, not prose.
"""
from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from review_all_program_seeded_run import FIXTURE_HOME, RUN_ID, SeededRun, seeded_run

SCRIPT = 'seam_scan.py'
LISTED = 0
UNSCANNED = 2
S1_PRD_PATH = 'plans/review-all-fixture-s1-prd.md'
G4_COLUMNS = ['Other PRD', 'Direction', 'Seam mechanism', 'Owner', 'Status']
NONE_LINE = 'no non-terminal task outside the program declares any touched path'


@pytest.fixture
def scan(tmp_path: Path) -> Iterator[SeededRun]:
    with seeded_run(tmp_path, store='store.json', returns=('S1',)) as run:
        yield run


def _scan_s1(run: SeededRun) -> subprocess.CompletedProcess[str]:
    return run.run(SCRIPT, 'S1', '--run', RUN_ID)


def _candidate_rows(result: subprocess.CompletedProcess[str]) -> dict[int, list[str]]:
    rows: dict[int, list[str]] = {}
    for line in result.stdout.splitlines():
        named = re.match(r'\| task (\d+):', line)
        if named is not None:
            rows.setdefault(int(named[1]), []).append(line)
    return rows


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip('|').split('|')]


def test_lists_each_outside_task_declaring_or_anchoring_a_touched_path(scan: SeededRun) -> None:
    result = _scan_s1(scan)

    assert result.returncode == LISTED, result.stdout + result.stderr
    header = [line for line in result.stdout.splitlines() if line.startswith('seam_scan: S1')]
    assert len(header) == 1, result.stdout
    assert S1_PRD_PATH in header[0] and re.search(r'\b5 paths\b', header[0]), header
    assert G4_COLUMNS in [_cells(line) for line in result.stdout.splitlines()], result.stdout

    rows = _candidate_rows(result)
    assert sorted(rows) == [201, 202, 203, 204], result.stdout
    assert all(len(lines) == 1 for lines in rows.values()), rows
    expected = {
        201: ('src/fixture/seam.py', 'metadata.files', 'plans/other-prd.md'),
        202: ('src/fixture/seam.py', 'seam-contract-defined'),
        203: ('src/fixture/old_name.py', 'old-thing-defined'),
        204: ('src/fixture/removed.py', 'removed-module-present', 'in-progress'),
    }
    for task_id, fragments in expected.items():
        for fragment in fragments:
            assert fragment in rows[task_id][0], (fragment, rows[task_id][0])
    assert NONE_LINE not in result.stdout


def test_a_program_task_by_prd_path_alone_is_not_a_seam(scan: SeededRun) -> None:
    scan.store.edit({104: {'metadata': {'files': ['src/fixture/beta.py']}}})

    result = _scan_s1(scan)

    assert result.returncode == LISTED, result.stdout + result.stderr
    assert 104 not in _candidate_rows(result)


def test_candidates_come_from_the_store(scan: SeededRun) -> None:
    scan.store.edit({201: {'metadata': {'files': []}}})

    result = _scan_s1(scan)

    assert result.returncode == LISTED, result.stdout + result.stderr
    assert sorted(_candidate_rows(result)) == [202, 203, 204]


def test_no_candidate_prints_the_none_line(scan: SeededRun) -> None:
    scan.store.edit({task_id: {'status': 'done'} for task_id in (201, 202, 203, 204)})

    result = _scan_s1(scan)

    assert result.returncode == LISTED, result.stdout + result.stderr
    assert _candidate_rows(result) == {}
    assert result.stdout.splitlines()[-1] == NONE_LINE


def test_a_stream_with_no_applied_author_return_is_unscanned(scan: SeededRun) -> None:
    result = scan.run(SCRIPT, 'S2', '--run', RUN_ID)

    assert result.returncode == UNSCANNED, result.stdout + result.stderr
    assert str(scan.home.returns_file('S2', 'author')) in result.stderr
    assert 'author return' in result.stderr
    assert _candidate_rows(result) == {}


def test_a_stream_outside_the_program_is_unscanned(scan: SeededRun) -> None:
    result = scan.run(SCRIPT, 'S9', '--run', RUN_ID)

    assert result.returncode == UNSCANNED, result.stdout + result.stderr
    for stream in ('S9', 'S1', 'S2', 'S3'):
        assert stream in result.stderr, result.stderr


def test_a_store_without_include_rows_is_unscanned(tmp_path: Path) -> None:
    with seeded_run(tmp_path, returns=('S1',), store_flags=('--no-include-rows',)) as run:
        result = _scan_s1(run)

    assert result.returncode == UNSCANNED, result.stdout + result.stderr
    assert 'include_rows' in result.stderr
    assert NONE_LINE not in result.stdout


def test_a_return_touching_nothing_prints_the_none_line(scan: SeededRun) -> None:
    untouched = json.loads(FIXTURE_HOME.returns_file('S1', 'author').read_text())
    untouched['touches'] = []
    scan.home.returns_file('S1', 'author').write_text(json.dumps(untouched))

    result = _scan_s1(scan)

    assert result.returncode == LISTED, result.stdout + result.stderr
    header = [line for line in result.stdout.splitlines() if line.startswith('seam_scan: S1')]
    assert len(header) == 1 and re.search(r'\b0 paths\b', header[0]), result.stdout
    assert result.stdout.splitlines()[-1] == NONE_LINE


def test_no_run_argument_is_unscanned(scan: SeededRun) -> None:
    assert scan.run(SCRIPT, 'S1').returncode == UNSCANNED
