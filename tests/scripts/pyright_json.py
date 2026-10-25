"""One ``pyright --outputjson`` run, parsed: the subprocess discipline guards share.

A guard that asserts on pyright's diagnostics needs the same three things from
the run:

* ``sys.executable -m pyright``, so the interpreter's own pyright runs, and a
  missing or broken pyright FAILS the guard (an exit code other than 0 or 1)
  instead of skipping it. A skip would be exactly the vacuous green such a
  guard exists to prevent.
* ``--outputjson``, parsed, so a failure message can name the rule, the file
  and the line rather than an exit code. It also suppresses pyright-python's
  newer-version notice, which keeps stdout pure JSON.
* undecodable stdout surfaced as an ``AssertionError`` carrying stdout and
  stderr, never as a bare ``JSONDecodeError``.

IMPORT ME, DO NOT COPY ME. This was extracted from
``tests/scripts/test_root_py_type_gate.py::_pyright_report`` (task 5086), which
still carries that copy until it imports this instead; a fix to one must reach
the other until then.

Importable from ``tests/scripts/test_*.py`` only because
``tests/scripts/conftest.py`` puts this directory on ``sys.path``.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = pathlib.Path(__file__).parents[2]


def pyright_json_report(
    paths: Sequence[pathlib.Path], *, project_dir: pathlib.Path | None = None
) -> dict[str, Any]:
    """The parsed ``pyright --outputjson [-p <project_dir>] <paths>`` payload.

    Runs from the repo root, so without *project_dir* pyright resolves the root
    ``[tool.pyright]`` table; with it, the table in
    ``<project_dir>/pyproject.toml``.
    """
    project_args = [] if project_dir is None else ["-p", str(project_dir)]
    proc = subprocess.run(
        [
            sys.executable, "-m", "pyright", "--outputjson",
            *project_args, *[str(p) for p in paths],
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert proc.returncode in (0, 1), (
        f"`pyright --outputjson` exited {proc.returncode} — expected 0 (clean) "
        f"or 1 (diagnostics). A missing pyright module or a bad invocation must "
        f"fail the guard rather than skip it; stderr: {proc.stderr.strip()!r}"
    )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise AssertionError(
            f"could not parse `pyright --outputjson` output: {exc}; "
            f"stdout: {proc.stdout[:500]!r}; stderr: {proc.stderr.strip()!r}"
        ) from exc
