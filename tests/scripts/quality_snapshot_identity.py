"""The comparison behind the measurement-identity invariant: which lines of a snapshot file a re-run may change.

plans/quality-dashboard-prd.md decision 17: two measurements at one identity
are byte-identical below ``run_id`` and ``cost``. The instrument's identity
test and the cadence wrapper's ``--force`` test both compare files through this
one filter, so a new per-run field changes both together.

IMPORT ME, DO NOT COPY ME. Importable from ``tests/scripts/test_*.py`` because
``tests/scripts/conftest.py`` puts this directory on ``sys.path``.
"""
from __future__ import annotations

from pathlib import Path

#: The rendered lines a re-run changes: the top-level run_id and the cost block's wall clock.
_PER_RUN_LINE_PREFIXES = ('  "run_id": ', '    "wall_clock_s": ')


def below_run_id_and_cost(path: Path) -> list[str]:
    """*path*'s lines without the two a re-run changes."""
    return [
        line
        for line in path.read_text(encoding='utf-8').splitlines()
        if not line.startswith(_PER_RUN_LINE_PREFIXES)
    ]
