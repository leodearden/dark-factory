"""The exit statuses of a deterministic task's predicate script (docs/task-authoring.md section 6).

The gate re-arms only on EXIT_NOT_YET, 75 (``os.EX_TEMPFAIL``), and reads any
other non-zero status as a failing verdict that it escalates. A predicate
therefore answers 0 when its condition holds, waits only by exiting 75, and
exits EXIT_ERROR, 1, on every error. That includes a usage error, which stock
argparse would exit as 2: :class:`PredicateParser` exits it as 1. Stdlib only.
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import NoReturn

__all__ = ['EXIT_ERROR', 'EXIT_NOT_YET', 'PredicateParser']

EXIT_NOT_YET = os.EX_TEMPFAIL
EXIT_ERROR = 1


class PredicateParser(argparse.ArgumentParser):
    """An ArgumentParser whose usage error exits EXIT_ERROR, never argparse's 2 or the wait, 75."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f'{self.prog}: error: {message}\n')
