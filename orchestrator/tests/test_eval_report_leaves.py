import subprocess
import sys

import pytest

_REPORT_LEAVES = ('stats', 'table_render', 'prices', 'cell_admission', 'elo_report')


class TestReportLeavesImportNothingFromReport:
    """A report leaf must be importable without loading the report surface (PRD D9, heuristic 9 layering)."""

    @pytest.mark.parametrize('leaf', _REPORT_LEAVES, ids=_REPORT_LEAVES)
    def test_leaf_import_leaves_report_unloaded(self, leaf: str) -> None:
        code = (
            'import sys\n'
            f'import orchestrator.evals.{leaf}\n'
            "print('orchestrator.evals.report' in sys.modules)\n"
        )
        completed = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == 'False'
