"""The merge gate's de-flake and regression texts as they read before task 6864 (ε1).

Both are ``string.Template`` files in ``fixtures/flake_ledger_golden/``.  Every variable
value and the instruction constants are placeholders, filled here from a row, an item
and the LIVE constants, so rewording a constant does not churn the golden while any
other change to the merge gate's text turns its pinning test red.  See that directory's
PROVENANCE.md.
"""

from __future__ import annotations

from pathlib import Path
from string import Template

from orchestrator.flake_ledger import (
    NEVER_WIDEN_A_TIMEOUT,
    RESPONSIBILITY_FIX_ROOT_DEFECT,
    RESPONSIBILITY_REMOVE_FROM_LEDGER,
    DebtRow,
    DueRow,
)

_GOLDEN_DIR = Path(__file__).parent / 'fixtures' / 'flake_ledger_golden'


def _template(name: str) -> Template:
    return Template((_GOLDEN_DIR / name).read_text())


def merge_gate_deflake_description(row: DebtRow) -> str:
    """The merge gate's de-flake task description for *row*."""
    return _template('merge_gate_deflake_description.txt').substitute(
        test_id=row.test_id,
        project_id=row.project_id,
        opened_at=row.opened_at,
        open_count=row.open_count,
        responsibility_fix_root_defect=RESPONSIBILITY_FIX_ROOT_DEFECT,
        responsibility_remove_from_ledger=RESPONSIBILITY_REMOVE_FROM_LEDGER,
        never_widen_a_timeout=NEVER_WIDEN_A_TIMEOUT,
    )


def merge_gate_regression_detail(row: DebtRow, item: DueRow) -> str:
    """The merge gate's ``regressed_after_resolution`` detail for *row*, observed
    through *item*."""
    return _template('merge_gate_regression_detail.txt').substitute(
        test_id=row.test_id,
        project_id=row.project_id,
        opened_at=row.opened_at,
        open_count=row.open_count,
        prior_resolving_commit=row.prior_resolving_commit or '(none recorded)',
        prior_resolved_at=row.prior_resolved_at,
        head=item.head or '(none)',
        observing_task_id=item.observing_task_id or '(none)',
        observed_at=item.observed_at,
        never_widen_a_timeout=NEVER_WIDEN_A_TIMEOUT,
    )
