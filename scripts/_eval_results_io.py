"""Shared persisted-``EvalResult`` loader for the ``scripts/`` eval drivers.

Extracted from ``scripts/run_fable_trial_v2_campaign.py`` (task 3632, review
finding 6) after ``scripts/run_judge_ofat_pilot.py`` grew a near-verbatim
sibling copy of the same load-and-filter body. Task 3632's amendment pass
hardened only its own copy — an ``is_dir()`` check that exits naming the path,
and a malformed-JSON handler that exits naming the offending file — while
``run_judge_ofat_pilot.py``'s copy kept neither, even though it reads the same
shared packaged results dir (``orchestrator/src/orchestrator/evals/results/``,
written by many campaigns over time) where a single truncated or partial
write is the realistic failure. Extracting this module means both drivers get
the hardening from one implementation instead of two independently-maintained
copies.

Both drivers import ``load_results_from_dir`` directly rather than each
keeping a private wrapper — there is exactly one implementation to keep in
sync with ``orchestrator.evals.runner.load_results``, which this mirrors over
an arbitrary dir rather than the packaged ``RESULTS_DIR``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# scripts/ is not on sys.path when this module is imported as a plain script
# sibling rather than through scripts/tests/conftest.py's insertion — add the
# orchestrator src so `from orchestrator.evals.runner import EvalResult` below
# resolves the same way it does for both callers (idempotent — a no-op when
# already present).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ORCHESTRATOR_SRC = str(_REPO_ROOT / 'orchestrator' / 'src')
if _ORCHESTRATOR_SRC not in sys.path:
    sys.path.insert(0, _ORCHESTRATOR_SRC)


def load_results_from_dir(results_dir: Path) -> list[Any]:
    """Load every persisted ``EvalResult`` JSON in *results_dir*.

    Filters each loaded dict to the known dataclass fields, mirroring
    ``runner.load_results``, so a cell persisted before a field existed still
    loads instead of raising on an unexpected keyword.

    TWO LOUD FAILURES, both because this path is aimed by default at the
    SHARED packaged results dir holding cells written by many campaigns over
    time:

    * A MISSING DIR exits naming it. ``Path.glob`` returns EMPTY for a
      nonexistent dir rather than raising, so without this check a plain typo
      falls through to a zero-survivors branch and gets diagnosed as
      contamination when the real problem is the path.
    * A MALFORMED FILE exits NAMING THE FILE. One truncated or partial write
      among hundreds of cells would otherwise take down the whole re-analysis
      with a bare ``JSONDecodeError`` traceback naming neither the offending
      file nor the dir — unactionable. Failing loudly rather than skipping is
      deliberate: a silently skipped cell shifts every count in a downstream
      report.
    """
    from orchestrator.evals.runner import EvalResult

    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise SystemExit(
            f'Error: results dir not found (or is not a directory): {results_dir}\n'
            '  This is a path problem, not a contamination problem: Path.glob returns '
            'empty for a nonexistent dir rather than raising, so without this check the '
            'typo would be reported downstream as "no in-campaign cells found".'
        )
    known = {f.name for f in EvalResult.__dataclass_fields__.values()}
    results = []
    for path in sorted(results_dir.glob('*.json')):
        try:
            data = json.loads(path.read_text())
            results.append(EvalResult(**{k: v for k, v in data.items() if k in known}))
        except (ValueError, OSError, TypeError, AttributeError) as e:
            raise SystemExit(
                f'Error: could not load persisted eval cell {path}: '
                f'{type(e).__name__}: {e}\n'
                '  Expected a JSON object shaped like EvalResult.to_dict(). A truncated '
                'or partial write in the shared packaged results dir is the usual cause. '
                'The file is named rather than skipped: dropping a cell silently would '
                'shift every count in the report.'
            ) from e
    return results
