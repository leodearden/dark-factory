import importlib.util
import subprocess
import sys

import pytest

_REPORT_LEAVES = ('stats', 'table_render', 'prices', 'cell_admission', 'elo_report')
_REPORT_SURFACES = ('plan_quality_report', 'composite_report')
_EVERY_REPORT_SURFACE = frozenset(
    {'orchestrator.evals.report'}
    | {f'orchestrator.evals.{surface}' for surface in _REPORT_SURFACES}
)


def _evals_modules_loaded_by(module: str) -> set[str]:
    code = (
        'import sys\n'
        f'import orchestrator.evals.{module}\n'
        "print('\\n'.join(sorted(m for m in sys.modules"
        " if m.startswith('orchestrator.evals.'))))\n"
    )
    completed = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return set(completed.stdout.split())


class TestReportLeavesImportNothingFromReport:
    """A report leaf must be importable without loading any report surface (PRD D9, heuristic 9 layering)."""

    @pytest.mark.parametrize('leaf', _REPORT_LEAVES, ids=_REPORT_LEAVES)
    def test_leaf_import_leaves_every_report_surface_unloaded(self, leaf: str) -> None:
        assert not _evals_modules_loaded_by(leaf) & _EVERY_REPORT_SURFACE


class TestReportSurfacesAreSiblings:
    """A report surface must be importable without loading report or its sibling (PRD D9)."""

    @pytest.mark.parametrize('surface', _REPORT_SURFACES, ids=_REPORT_SURFACES)
    def test_surface_import_leaves_report_and_sibling_unloaded(self, surface: str) -> None:
        others = _EVERY_REPORT_SURFACE - {f'orchestrator.evals.{surface}'}
        assert not _evals_modules_loaded_by(surface) & others


def test_report_module_is_gone() -> None:
    """PRD D9: report.py is deleted with no re-export shim (the `report-module-gone` delivered check)."""
    assert importlib.util.find_spec('orchestrator.evals.report') is None
