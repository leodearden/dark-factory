"""The ``scripts/`` eval drivers' operator-facing face of ``ResultStore.load_all``.

Both drivers (``run_fable_trial_v2_campaign.py``, ``run_judge_ofat_pilot.py``)
import :func:`load_results_from_dir`, which turns the store's structured load
errors into a ``SystemExit`` naming the missing dir or the bad cell.
"""
from __future__ import annotations

import sys
from pathlib import Path

# scripts/ is not on sys.path when this module is imported as a plain script
# sibling rather than through scripts/tests/conftest.py's insertion — add the
# orchestrator src so the import below resolves the same way it does for both
# callers (idempotent — a no-op when already present).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_ORCHESTRATOR_SRC = str(_REPO_ROOT / 'orchestrator' / 'src')
if _ORCHESTRATOR_SRC not in sys.path:
    sys.path.insert(0, _ORCHESTRATOR_SRC)

from orchestrator.evals.results import (  # noqa: E402
    EvalResult,
    MissingResultsDirError,
    ResultLoadError,
    ResultStore,
)


def load_results_from_dir(results_dir: Path) -> list[EvalResult]:
    """Every persisted cell in *results_dir*, or an exit naming what could not be read."""
    try:
        return ResultStore(Path(results_dir)).load_all()
    except MissingResultsDirError as e:
        raise SystemExit(
            f'Error: results dir not found (or is not a directory): {e.path}\n'
            '  This is a path problem, not a contamination problem: Path.glob returns '
            'empty for a nonexistent dir rather than raising, so without this check the '
            'typo would be reported downstream as "no in-campaign cells found".'
        ) from e
    except ResultLoadError as e:
        raise SystemExit(
            f'Error: could not load persisted eval cell {e.path}: {e.detail}\n'
            '  Expected a JSON object shaped like EvalResult.to_dict(). A truncated '
            'or partial write in the shared packaged results dir is the usual cause. '
            'The file is named rather than skipped: dropping a cell silently would '
            'shift every count in the report.'
        ) from e
