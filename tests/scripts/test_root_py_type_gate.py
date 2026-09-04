"""Gate contract: every repo-root ``*.py`` must be pyright-clean.

Task 3960. This is the TYPE half of the gate whose LINT half is
``tests/scripts/test_root_lint_covers_nonmember_py.py`` (task 3485). That task
closed the identical hole for LINT over the identical file set; the two halves
are independent, and neither implies the other — a file can be lint-clean under
a wide rule set and carry a type error nothing ever runs a type checker over.

THE HOLE THIS CLOSES, traced against the production helpers rather than assumed.
A repo-root ``.py`` can never belong to a module config:
``config._discover_module_configs`` explicitly SKIPS a root-level config
(``prefix == '.'``), and ``config.OrchestratorConfig.for_module`` splits the
path on ``/`` — for ``df_pytest_isolation.py`` the single candidate is the bare
filename, which is never a registry key. So neither ``scripts/`` nor
``tests/scripts/`` can own these files no matter how their commands are worded.

A diff touching ONLY repo-root ``.py`` is nevertheless gated today:
``verify._build_fallback_config`` scopes the type leg to the touched files. The
REAL hole is a diff touching a repo-root ``.py`` ALONGSIDE subproject files —
``module_configs`` is then non-empty, the fallback builder is never consulted,
no prefix owns the root path, the root file is dropped with only a WARNING, and
it is type-checked by NOTHING.

MEASURED RED at base main ``23ce883356``: ``python -m pyright --outputjson
conftest.py df_pytest_isolation.py`` returned exactly ONE error diagnostic —
``df_pytest_isolation.py`` line 788 col 46, rule ``reportArgumentType``,
'Argument of type "float | None" cannot be assigned to parameter "timeout" of
type "float" in function "__init__"'. It was landed by task 3798's commit
``61b62df192`` and survived task 3799 on top of it, on checks reporting green —
exactly the diff shape described above. Measured cost 5.9s cold / ~1.4s warm
over 2 files, against this suite's 600s budget and its 233.50s measured worst
run, so no budget literal moves and ``tests/scripts/orchestrator.yaml`` needs no
edit.

WHY A PYTEST GUARD AND NOT A CONFIG EDIT. Both config levers are dead. Widening
``scripts/orchestrator.yaml`` or ``tests/scripts/orchestrator.yaml`` gates
nothing for these files (see the prefix argument above) and is separately
blocked by ``test_scripts_module_config.py::test_scripts_diff_is_type_gated``,
which requires the directory-wide element and rejects narrowing to a file list.
Appending to the repo-root ``type_check_command`` is dead code on any
``.py``-bearing diff: ``verify._scope_to_keyword`` truncates the head at the
FIRST ``pyright``, and ``verify_cmd.split_chain_tail`` refuses tail preservation
outright because that chain contains ``cd`` — measured with the production
helpers, the base chain and the chain-plus-append scope to a byte-identical
command.

PLACEMENT IS LOAD-BEARING, NOT STYLISTIC. This file lives in ``tests/scripts/``
because that directory carries its own module config, so the guard actually runs
under FULL_SUITE and merge-role ``merge_verify_breadth: full``. Under that
breadth every registered module's declared command runs VERBATIM, so this guard
fires on EVERY merge regardless of the diff's shape — which is precisely the
coverage the config levers cannot give. A guard against a vacuous gate that
itself never ran on merge full-verify would be vacuous in the same way
(``test_scripts_module_config.py``'s own rationale).

MUST NOT SKIP. The pyright probe resolves the interpreter's own ``-m pyright``,
so a missing pyright FAILS rather than silently skipping. A
``pytest.importorskip`` or try/skip here would reintroduce precisely the
vacuous-green failure mode this task closes — the same discipline task 3485's
ruff probe insists on.

Production code is cited BY SYMBOL, deliberately never by file:line — task
3445's explicit correction of the convention task 3350 established: every line
pin copied forward had already rotted at HEAD, and a stale pin is worse than no
pin because it reads as authoritative.
"""
from __future__ import annotations

import fnmatch
import json
import pathlib
import subprocess
import sys
import tomllib
from typing import Any

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The repo-root pyproject.toml, which carries the [tool.pyright] table a bare
# `pyright` invocation resolves from this directory. Loading it by this exact
# path means the guard fails loudly if the table disappears, rather than quietly
# guarding nothing.
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

# Glob metacharacters. An `include`/`exclude` entry carrying one is a PATTERN,
# so the stale-target check must resolve it rather than stat it.
_GLOB_CHARS = ("*", "?", "[")


def _guarded_py_files() -> list[pathlib.Path]:
    """Every repo-root-level ``*.py``.

    A SCOPED pathlib glob, deliberately not a repo-wide walk and not
    ``git ls-files`` — the reasoning is copied from ``_guarded_py_files`` in
    ``test_root_lint_covers_nonmember_py.py``. A recursive walk from
    ``REPO_ROOT`` would descend into ``.worktrees/``, which holds full sibling
    checkouts of this repo, so the guard would enumerate other tasks' trees and
    fail nondeterministically. ``git ls-files`` avoids that but buys a
    subprocess-plus-git dependency for no gain here.

    This tree is exactly the durability surface no module config can own, and
    deliberately excludes ``scripts/**`` and ``tests/scripts/**``, which carry
    their own module configs and their own declared type_check_commands. A NEW
    repo-root ``.py`` is therefore picked up automatically and fails this guard
    until it is clean and declared.
    """
    return sorted(REPO_ROOT.glob("*.py"))


def _pyright_report(paths: list[pathlib.Path]) -> dict[str, Any]:
    """The parsed ``pyright --outputjson`` payload for *paths*.

    Asserting on the parsed diagnostic SET rather than on the exit code is what
    makes a failure message name the rule, the file and the line. It also keeps
    the probe honest if pyright ever exits non-zero for a reason other than a
    diagnostic — that surfaces below as a decode failure, not as a silent pass.

    ``--outputjson`` also suppresses pyright-python's newer-version notice, so
    stdout stays pure JSON.
    """
    proc = subprocess.run(
        [
            sys.executable, "-m", "pyright", "--outputjson",
            *[str(p) for p in paths],
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    # No try/except-and-skip: a missing or broken pyright must FAIL this guard.
    assert proc.returncode in (0, 1), (
        f"`pyright --outputjson` exited {proc.returncode} (task 3960) — expected "
        f"0 (clean) or 1 (diagnostics). A missing pyright module or a bad "
        f"invocation must fail this guard rather than skip it; stderr: "
        f"{proc.stderr.strip()!r}"
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise AssertionError(
            f"could not parse `pyright --outputjson` output (task 3960): {exc}; "
            f"stdout: {proc.stdout[:500]!r}; stderr: {proc.stderr.strip()!r}"
        ) from exc
    return payload


def _errors(payload: dict[str, Any]) -> list[tuple[str, int, int, str, str]]:
    """``(rel path, line, col, rule, first message line)`` per error diagnostic.

    Warnings and informations are deliberately NOT collected: the declared
    module type gates assert on pyright's exit status, which is driven by the
    error count alone, so gating on anything wider here would red-wall merges
    over findings no other gate in this repo treats as blocking.

    Rows are 1-based in the message, matching how pyright prints them to a
    human, while the payload itself is 0-based.
    """
    rows: list[tuple[str, int, int, str, str]] = []
    for item in payload.get("generalDiagnostics", []):
        if item.get("severity") != "error":
            continue
        start = item.get("range", {}).get("start", {})
        rows.append((
            pathlib.Path(item["file"]).resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
            int(start.get("line", -1)) + 1,
            int(start.get("character", -1)) + 1,
            item.get("rule", "<no rule>"),
            str(item.get("message", "")).splitlines()[0],
        ))
    return sorted(rows)


def test_repo_root_py_is_pyright_clean() -> None:
    """Every repo-root ``*.py`` must type-check clean.

    This is the invariant the task exists to establish, and it is RED on
    arrival — see the MEASURED RED paragraph in the module docstring.

    Findings here are FIXED, never carved back out with an ``exclude`` or a
    ``# pyright: ignore`` (task 3350's fix-don't-weaken precedent; the
    no-carve-out half is enforced by the config test below). An ignore comment
    would leave this gate green while the defect stands, which is the exact
    vacuous-gate class this guard closes.
    """
    files = _guarded_py_files()

    # NON-VACUITY, side one: a bad glob (a parents[] off-by-one, a moved file)
    # would otherwise let this invariant pass by checking nothing at all.
    assert files, (
        f"no repo-root-level *.py found under {REPO_ROOT} (task 3960) — this "
        "cleanliness invariant would pass vacuously; the enumeration glob is "
        "almost certainly wrong"
    )

    payload = _pyright_report(files)

    # NON-VACUITY, side two: pyright silently analysing fewer files than were
    # named (a bad path, a config exclude swallowing one) would report a clean
    # run over a file this guard never actually checked.
    analyzed = payload.get("summary", {}).get("filesAnalyzed")
    assert analyzed == len(files), (
        f"pyright analysed {analyzed} file(s) but {len(files)} were named "
        f"(task 3960) — a clean result over fewer files than the enumeration "
        f"is a vacuous pass; named: "
        f"{[p.relative_to(REPO_ROOT).as_posix() for p in files]}"
    )

    errors = _errors(payload)
    assert not errors, (
        f"repo-root *.py files are not pyright-clean (task 3960): {errors}. "
        f"Nothing else type-checks them: _discover_module_configs skips a "
        f"root-level config (prefix == '.') and for_module splits the path on "
        f"'/', so a repo-root .py can never belong to a module config, and the "
        f"repo-root type_check_command is truncated at its first `pyright` on "
        f"any .py-bearing diff. FIX the diagnostic — do not add a "
        f"`# pyright: ignore`, and do not carve the file out of "
        f"[tool.pyright] include/exclude. Probe: {sys.executable} -m pyright "
        f"--outputjson {' '.join(p.relative_to(REPO_ROOT).as_posix() for p in files)}"
    )


def _pyright_table() -> dict[str, Any]:
    """The repo-root ``[tool.pyright]`` table, asserted present."""
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    table = data.get("tool", {}).get("pyright")
    assert isinstance(table, dict), (
        f"{PYPROJECT_PATH} declares no [tool.pyright] table (task 3960) — a "
        f"bare `pyright` run from the repo root would then fall back to "
        f"scanning the whole tree, and this durability pin would have nothing "
        f"to assert against"
    )
    return table


def _matches(rel: str, entry: str) -> bool:
    """True if config *entry* names *rel*, an ancestor directory of it, or a glob for it.

    Ancestor-directory coverage is real coverage in both directions: an
    ``include`` of a directory traverses it, and an ``exclude`` of one carves
    out everything beneath. ``fnmatch`` is used for entries carrying glob
    metacharacters, which pyright's include/exclude syntax permits.
    """
    normalized = entry.rstrip("/") or "."
    if normalized in (".", rel):
        return True
    if rel.startswith(normalized + "/"):
        return True
    return fnmatch.fnmatch(rel, normalized)


def test_root_pyright_include_targets_every_repo_root_py() -> None:
    """``[tool.pyright] include`` must name every repo-root ``*.py``.

    THIS PASSES ON ARRIVAL — it is a regression pin, not a driver for any edit
    in this task. It mirrors assertions (b) and (c) of task 3485's
    ``test_root_lint_command_targets_every_root_level_and_skills_py``, and the
    ``[tool.pyright]``-loaded-with-``tomllib`` shape of
    ``orchestrator/tests/test_pyright_gate_for_workflow_e2e.py``.

    What it buys: the cleanliness probe above names its files EXPLICITLY on the
    command line, which makes it immune to the config — so a newly-added
    repo-root ``.py`` that never reaches ``include`` would still be caught
    there, but would be invisible to every OTHER pyright entry point (a bare
    ``pyright`` run from the repo root, an editor's language server, the
    repo-root ``type_check_command``'s unscoped form). This pin keeps the two
    in step, and keeps a later ``exclude`` from silently un-gating a file while
    every check still reports green.
    """
    table = _pyright_table()
    include = table.get("include")
    files = _guarded_py_files()

    # (a) NON-VACUITY, both sides.
    assert files, (
        f"no repo-root-level *.py found under {REPO_ROOT} (task 3960) — this "
        "coverage invariant would pass vacuously; the enumeration glob is "
        "almost certainly wrong"
    )
    assert isinstance(include, list) and include, (
        f"[tool.pyright] in {PYPROJECT_PATH} declares no non-empty `include` "
        f"(task 3960) — this coverage invariant would pass vacuously; got "
        f"{include!r}"
    )

    # The coverage invariant itself.
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        assert any(_matches(rel, entry) for entry in include), (
            f"[tool.pyright] include does not cover {rel!r} (task 3960) — a "
            f"bare `pyright` from the repo root, and every editor language "
            f"server reading this table, would not see it. Add {rel!r} (or an "
            f"ancestor directory) to include; current include: {include}"
        )

    # (b) NO CARVE-OUTS. An exclude/ignore silently un-gates whatever it names
    # while the check still reports green — the exact vacuous-gate class this
    # guard closes (task 3445's precedent). Findings are FIXED instead.
    for key in ("exclude", "ignore"):
        entries = table.get(key) or []
        for path in files:
            rel = path.relative_to(REPO_ROOT).as_posix()
            carving = [e for e in entries if _matches(rel, e)]
            assert not carving, (
                f"[tool.pyright] {key} carves out {rel!r} via {carving} (task "
                f"3960) — that silently un-gates the file while pyright still "
                f"reports green. Fix the diagnostic instead of excluding it."
            )

    # (c) STALE-TARGET guard. A bogus include entry is invisible to the
    # coverage loop above, yet makes a bare `pyright` run misreport its scope.
    for entry in include:
        if any(ch in entry for ch in _GLOB_CHARS):
            assert list(REPO_ROOT.glob(entry)), (
                f"[tool.pyright] include names the pattern {entry!r}, which "
                f"matches nothing under {REPO_ROOT} (task 3960) — a stale "
                f"target reads as coverage while gating nothing; include: "
                f"{include}"
            )
        else:
            assert (REPO_ROOT / entry).exists(), (
                f"[tool.pyright] include names {entry!r}, which does not exist "
                f"under {REPO_ROOT} (task 3960) — a stale target reads as "
                f"coverage while gating nothing; include: {include}"
            )
