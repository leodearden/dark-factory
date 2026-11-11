"""Synthetic verify-summary corpora for the tests/scripts/test_verify_budget_census*.py suites.

NO CENSUS TEST ASSERTS A NUMBER READ FROM THE LIVE SUMMARY CORPUS.
This norm is inherited verbatim from the sibling suites in this directory —
tests/scripts/test_census_tagger_debris.py's module docstring records a
candidate count that moved 40 -> 43 -> 45 across a single task's planning
sessions — and scripts/tests/test_merge_lane_throughput.py runs entirely on
synthetic roots for the same reason. The corpus here is mutated continuously by
a running fleet: worktrees are created and pruned, and each attempt OVERWRITES
the previous summary, so it is both non-deterministic and survivorship-biased.
A test pinning "the corpus yields n=14, max 3753" is a guessed threshold that
goes red the moment any task verifies.

D17 does name numbers the census "must reproduce". Those are discharged as a
recorded reproduction run in the iteration log plus a provenance-stamped
constant beside `MEASURED_MODULE_SUITE_WORST_SECS` — which is what that
constant's own provenance block already demands ("RE-MEASURE BY REPEATING THAT
CENSUS, so the next figure is a repeat rather than a re-derivation") — never as
an assertion in a census test. Every census assertion runs against a synthetic
`tmp_path` corpus whose contents the test controls exactly and whose expected
numbers are computed by hand.

Like ``module_budget_family.py``, this is a non-test helper, importable by bare
name because ``tests/scripts/conftest.py`` inserts this directory on sys.path.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import verify_budget_census

# Both archive stamp spellings seen in the wild. The microsecond-bearing form
# dominates; the second-resolution form is what older records carry.
STAMP_US = '20260914T123016_283575Z'
STAMP_SEC = '20260816T080255Z'


def write_worktree_record(root: Path, lane: str, name: str) -> Path:
    path = root / '.worktrees' / lane / '.task' / 'verify' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'commands': []}), encoding='utf-8')
    return path


def write_archive_record(root: Path, task_id: str, name: str) -> Path:
    path = root / 'data' / 'verify-logs' / task_id / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'commands': []}), encoding='utf-8')
    return path


# The orchestrator module's declared test_command, as of this writing. Used
# ONLY to shape the synthetic fixtures below — never as the value under test.
# The census reads the real one out of `<root>/<prefix>/orchestrator.yaml`,
# which is what
# `tests/scripts/test_verify_budget_census_selection.py::TestTheExpectedCommandHasOneHome`
# pins.
FULL_SUITE = (
    'uv run --directory orchestrator pytest tests/ --tb=short -q --timeout=300'
)


def summary_payload(*commands: dict[str, Any], **top: Any) -> dict[str, Any]:
    """A summary.json payload. Top-level fields default to a LOUD lint leg.

    Defaulting the top level to something LOUD and SHORT is deliberate: it is
    the shape that catches a census reading the wrong place. See
    `tests/scripts/test_verify_budget_census_selection.py::TestTheSelectorReadsTheCommandsArray`.
    """
    payload: dict[str, Any] = {
        'category': 'lint_error',
        'cause_hint': '',
        'rc': -9,
        'timed_out': False,
        'cmd': 'uv run --directory orchestrator ruff check src/',
        'started_at': '2026-09-13T04:00:00+00:00',
        'duration_secs': 4.5,
        'commands': list(commands),
    }
    payload.update(top)
    return payload


def command_entry(
    label: str = 'test', cmd: str | None = FULL_SUITE, **overrides: Any,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        'label': label,
        'cmd': cmd,
        'rc': 0,
        'timed_out': False,
        'started_at': '2026-09-13T04:00:00+00:00',
        'duration_secs': 3300.0,
        'segments': None,
        'load': None,
    }
    entry.update(overrides)
    return entry


def write_module_yaml(root: Path, prefix: str = 'orchestrator') -> Path:
    """The `<root>/<prefix>/orchestrator.yaml` the selector reads its command from."""
    path = root / prefix / 'orchestrator.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'test_command: "{FULL_SUITE}"\n', encoding='utf-8')
    return path


def write_corpus(
    root: Path,
    *commands: dict[str, Any],
    prefix: str = 'orchestrator',
    lane: str = '3353',
    **top: Any,
) -> Path:
    """A one-record synthetic root, plus the module yaml the selector reads."""
    write_module_yaml(root, prefix)

    path = write_worktree_record(root, lane, f'attempt-1.{prefix}.summary.json')
    path.write_text(json.dumps(summary_payload(*commands, **top)), encoding='utf-8')
    return path


# A frozen clock, injected into every main() call below, so the trailing-window
# boundary is deterministic. The sibling suites do the same; a window resolved
# against the real clock makes a test's own fixtures age out of it.
_NOW = '2026-09-14T12:00:00+00:00'


def run_main(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    """Drive `main()` and return `(rc, stdout, stderr)` — the sibling's idiom."""
    code = verify_budget_census.main(list(argv), now=datetime.fromisoformat(_NOW))
    captured = capsys.readouterr()
    return code, captured.out, captured.err
