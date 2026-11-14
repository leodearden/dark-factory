"""Hold an INV-12 declarations module to being read by the register as it evaluates.

``exception_register.py --report`` finds governed lists statically, so a
declaration it cannot read -- one built inside a function or through a helper,
or with a non-literal list_id, rule or disposition -- still raises at import
but drops out of the report. Each declarations module calls
:func:`assert_register_reads_as_evaluated` with its own file and globals.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import exception_register
from shared.governed_exceptions import GovernedList

REPO_ROOT = Path(__file__).resolve().parents[1]


def assert_register_reads_as_evaluated(module_file: str, namespace: Mapping[str, object]) -> None:
    """Raise AssertionError unless the register reads *module_file* as *namespace* holds it."""
    path = Path(module_file).resolve()
    relpath = path.relative_to(REPO_ROOT).as_posix()
    discovery = exception_register.declarations_in(
        exception_register.Corpus(files={relpath: path.read_bytes()}, enumerated=1)
    )
    if discovery.faults:
        raise AssertionError(f'exception_register cannot read {relpath}: {discovery.faults}')
    read = {
        d.list_id: (d.rule, d.default, d.default_covers, dict(d.overrides))
        for d in discovery.declarations
    }
    evaluated = {
        g.list_id: (g.rule, g.default, g.default_covers, dict(g.overrides))
        for g in namespace.values()
        if isinstance(g, GovernedList)
    }
    if read != evaluated:
        raise AssertionError(
            f'exception_register reads the declarations of {relpath} differently from how '
            f'they evaluate, so `exception_register.py --report` misstates them.\n'
            f'  read statically: {read}\n'
            f'  evaluated:       {evaluated}'
        )
