"""Measure a first-party module's transitive import closure in a fresh interpreter.

The pytest process has already imported graphiti_core and most of fused_memory,
so asking its own ``sys.modules`` what a module pulls in answers nothing. The
probe therefore imports the module in a new interpreter and reports which of the
named candidates that import left in ``sys.modules``.

It runs ``-c`` with ``cwd`` set to this tree's ``src/`` rather than ``-I``:
``-I`` drops the cwd from ``sys.path``, so ``fused_memory`` would resolve
through whichever venv's editable ``.pth`` is active, and in a worktree using
main's interpreter that measures MAIN's source. With ``-c`` and ``cwd=SRC``,
``sys.path[0]`` is the tree under test, and the returned ``file`` lets a caller
prove which tree was measured. ``-E`` and ``-s`` still drop the ``PYTHON*``
environment and the user site.
"""

import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / 'src'

_REPORT_SENTINEL = 'IMPORT-CLOSURE-REPORT: '

_PROBE = """\
import importlib, json, sys
module = importlib.import_module({module!r})
print({sentinel!r} + json.dumps({{
    'file': module.__file__,
    'loaded': [name for name in {candidates!r} if name in sys.modules],
}}))
"""


@dataclass(frozen=True)
class ImportProbe:
    file: str
    loaded: tuple[str, ...]


def modules_loaded_by_importing(module: str, candidates: Sequence[str]) -> ImportProbe:
    result = subprocess.run(
        [
            sys.executable, '-E', '-s', '-c',
            _PROBE.format(
                module=module, candidates=list(candidates), sentinel=_REPORT_SENTINEL
            ),
        ],
        cwd=SRC,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            f'importing {module} in a fresh interpreter failed '
            f'(exit {result.returncode}):\n{result.stderr}'
        )
    report = _parse_report(module, result)
    return ImportProbe(file=report['file'], loaded=tuple(report['loaded']))


def _parse_report(module: str, result: subprocess.CompletedProcess[str]) -> dict:
    """The probe's own report line, found by its sentinel so import-time stdout cannot corrupt it."""
    reports = [
        line.removeprefix(_REPORT_SENTINEL)
        for line in result.stdout.splitlines()
        if line.startswith(_REPORT_SENTINEL)
    ]
    try:
        return json.loads(reports[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise AssertionError(
            f'probing {module} printed no readable report ({error!r}):\n'
            f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
        ) from error
