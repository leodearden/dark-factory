"""The one place a broken environment becomes the inline scanner's exit 2.

Two things live here and nowhere else: :class:`InstrumentFailure`, the single
exception every stratum raises when it could not take a measurement, and
:func:`import_shared`, the scanner's only route to the ``shared`` workspace
member.  The ``shared/src`` bootstrap below the imports is what makes that route
read THIS checkout's ``shared``; :data:`REPO_ROOT`, the checkout it resolves,
is public because every other path the scanner derives from its own checkout
starts there too.  Nothing here imports another stratum.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

# The shared/src bootstrap. Every path this scanner derives from its own
# checkout is resolved from __file__, NEVER from the working directory: a run
# inside a task worktree must read THAT checkout's disposition grammar, ratchet
# kernel, tracked corpus and configuration, wherever it was launched from. The
# idiom and the precedence argument are scripts/scan_plan_decision_pairing.py's:
# the editable install of `shared` is an ordinary .pth entry, so sys.path ORDER
# decides the winner, which is why this inserts at sys.path[0]. It sits BELOW
# every import in this file rather than above them, which is the whole reason no
# import in the scanner needs a suppression for E402: every `shared` name is
# fetched lazily through :func:`import_shared`, so there is no module-level
# import left to sit after this statement.
REPO_ROOT = Path(__file__).resolve().parents[1]
_SHARED_SRC = REPO_ROOT / 'shared' / 'src'
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))


class InstrumentFailure(Exception):
    """The scanner could not take a measurement it was asked for — exit 2.

    THE NAME STATES THE CONSUMER-RELEVANT FACT RATHER THAN A CAUSE, the same
    choice ``shared.ratchet.BaselineUnusable`` argues for: every cause of an
    exit 2 means one identical thing to a caller —
    *this run measured nothing you can trust*.

    Categorically apart from a VIOLATION, which is exit 1.  That separation is
    the whole point: a broken instrument reported as a finding sends an agent
    to fix code that was never the problem, and a finding reported as a broken
    instrument is an INV-12 breach nobody is told about.  Every message names
    the file, path or key at fault, because that is the one thing the operator
    cannot derive from the rest of it.
    """



def import_shared(name: str) -> ModuleType:
    """Import ``shared.<name>`` lazily, as exit 2 rather than exit 1.

    THE ONLY PLACE THE SCANNER IMPORTS ``shared``, and the reason is the PRD
    Contract: an ImportError must be 2, never 1.  A top-level ``from shared… import
    …`` in any of its modules cannot satisfy that — it raises while the entry
    script is still executing its imports, so ``main`` is never defined, the
    ``__main__`` block never runs, and Python's own
    uncaught-exception exit is 1, the exact code the ladder reserves for a FINDING;
    :class:`InstrumentFailure` says why the two must never be confused.

    Returning the module rather than the names is what keeps the conversion in one
    place, following ``scripts/merge_lane_metrics.py::_import_complexipy``.  The
    cost is measured and bounded: attributes come back as ``Any``, so a caller that
    needs ``isinstance`` NARROWING binds the class through a ``type[X]``
    annotation first (verified: ``isinstance(x, module.Debt)`` does not narrow,
    ``debt_type: type[Debt] = module.Debt`` then ``isinstance(x, debt_type)``
    does).  A caller that only needs the runtime class — an ``except`` clause, a
    branch predicate — uses the attribute directly.
    """
    try:
        return importlib.import_module(f'shared.{name}')
    except ImportError as exc:
        raise InstrumentFailure(
            f'`shared.{name}` could not be imported, so this scanner has no disposition '
            f'grammar and no ratchet kernel to work with -- {exc}. It is a workspace '
            'member of this repository: run the scanner as `uv run --project shared '
            'python scripts/inline_suppressions.py`, which is how the merge gate and '
            'every declared check invoke it.'
        ) from exc
