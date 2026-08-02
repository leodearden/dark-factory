"""Gate contract: repo-root ``*.py`` and ``skills/**/*.py`` must actually be linted.

Task 3485. ``conftest.py``, ``df_pytest_isolation.py`` and ``skills/**/*.py``
were linted by NOTHING, on checks reporting green. The repo-root
``dark-factory-orchestrator.yaml`` ``lint_command``'s ruff leg targeted only
the 7 ``[tool.uv.workspace].members``; ``scripts/orchestrator.yaml`` targets
``scripts/``; ``tests/scripts/orchestrator.yaml`` targets ``tests/scripts/``.
Nothing targeted these three files — the same vacuous-gate class tasks 3350 and
3445 closed for ``tests/scripts/`` and ``scripts/``.

This guard is the FILES half of the gate — "which files are targeted?". Task
3457's ``tests/scripts/test_nonmember_ruff_config.py`` is the complementary
RULES half — "which rules apply to them?". Neither implies the other: a file
can be targeted under a weak rule set, or carry the right rule set and be
targeted by no command at all.

MEASURED RED at base main ``1f83dbed15``: ``ruff check --select E,F,UP,B,SIM,I
--ignore E501 --line-length 100 conftest.py df_pytest_isolation.py skills/``
returned exactly 2 findings, both SIM105 (``try``/``except ImportError``/
``pass``) in ``conftest.py`` at rows 40 and 45; ``df_pytest_isolation.py`` 0,
``skills/`` 0. And the root command's ruff leg named none of the three paths.

RULE-SET / LANDING-ORDER NOTE. The cleanliness probe below passes the members'
rule set as EXPLICIT ``--select``/``--ignore``/``--line-length`` flags rather
than resolving it from config. Task 3457 (which declares ``[tool.ruff]`` at the
repo root) had not landed at this base, so these paths currently resolve ruff's
BUILT-IN defaults (E4/E7/E9 + F). Explicit flags make this guard independent of
3457's landing order, and are exactly how the original observation was taken.
Because the built-in defaults are a strict SUBSET of ``{E,F,UP,B,SIM,I}``,
proving cleanliness under the wide set also proves it under today's — one
assertion covers both landing orders, so 3457 landing cannot retroactively
red-wall main.

MUST NOT SKIP. The ruff probe resolves the interpreter's own ``-m ruff``, so a
missing ruff FAILS rather than silently skipping. A ``pytest.importorskip`` or
try/skip here would reintroduce precisely the vacuous-green failure mode this
task closes.

Production code is cited BY SYMBOL, deliberately never by file:line — task
3445's explicit correction of the convention task 3350 established: every line
pin copied forward had already rotted at HEAD, and a stale pin is worse than no
pin because it reads as authoritative.

PLACEMENT IS LOAD-BEARING, NOT STYLISTIC. This file lives in ``tests/scripts/``
because that directory carries its own module config, so the guard actually
runs under FULL_SUITE and merge-role ``merge_verify_breadth: full``. A guard
against a vacuous gate that itself never ran on merge full-verify would be
vacuous in the same way (``test_scripts_module_config.py``'s own rationale).
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The members' rule set, spelled out rather than resolved from config — see the
# RULE-SET / LANDING-ORDER NOTE in the module docstring.
MEMBER_RULE_SET_FLAGS = [
    "--select", "E,F,UP,B,SIM,I",
    "--ignore", "E501",
    "--line-length", "100",
]


def _guarded_py_files() -> list[pathlib.Path]:
    """Repo-root-level ``*.py`` plus every ``skills/**/*.py``.

    Two SCOPED pathlib globs, deliberately not a repo-wide walk and not
    ``git ls-files``. A recursive walk from ``REPO_ROOT`` would descend into
    ``.worktrees/``, which holds full sibling checkouts of this repo — the
    guard would enumerate other tasks' trees and fail nondeterministically.
    ``git ls-files`` avoids that but buys a subprocess-plus-git dependency for
    no gain here.

    These two trees are exactly the durability surface this task owns, and
    deliberately exclude ``scripts/**`` and ``tests/scripts/**``, which carry
    their own module configs and their own declared lint commands. A NEW
    root-level or ``skills/`` ``.py`` is therefore picked up automatically and
    fails this guard until it is clean and targeted.
    """
    return sorted(REPO_ROOT.glob("*.py")) + sorted((REPO_ROOT / "skills").rglob("*.py"))


def _ruff_findings(paths: list[pathlib.Path]) -> list[tuple[str, str]]:
    """``(rule code, repo-relative path)`` for every finding over *paths*.

    Asserting on the parsed finding SET rather than on the exit code is what
    makes a failure message name the rule and the file. It also keeps the probe
    honest if ruff ever exits non-zero for a reason other than a lint finding —
    that surfaces below as a decode failure, not as a silent pass.
    """
    proc = subprocess.run(
        [
            sys.executable, "-m", "ruff", "check", "--no-cache",
            "--output-format", "json",
            *MEMBER_RULE_SET_FLAGS,
            *[str(p) for p in paths],
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    # No try/except-and-skip: a missing or broken ruff must FAIL this guard.
    assert proc.returncode in (0, 1), (
        f"`ruff check` exited {proc.returncode} (task 3485) — expected 0 (clean) "
        f"or 1 (findings). A missing ruff module or a bad invocation must fail "
        f"this guard rather than skip it; stderr: {proc.stderr.strip()!r}"
    )
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise AssertionError(
            f"could not parse `ruff check --output-format json` output (task "
            f"3485): {exc}; stdout: {proc.stdout[:500]!r}; stderr: "
            f"{proc.stderr.strip()!r}"
        ) from exc
    return [
        (
            item["code"],
            pathlib.Path(item["filename"]).resolve().relative_to(REPO_ROOT).as_posix(),
        )
        for item in payload
    ]


def test_root_level_and_skills_py_are_clean_under_the_members_rule_set() -> None:
    """Every guarded ``.py`` must be ruff-clean under the members' rule set.

    This is the precondition that makes widening the root ``lint_command``
    safe to land. ``verify.run_full_verification`` asyncio-gathers over ALL
    ``module_configs.values()``, not just those a diff touches, and the repo
    root sets ``merge_verify_breadth: "full"`` — so a declared target carrying
    a finding blocks every merge, review checkpoint and main-tip sweep
    repo-wide, on branches with no defect. Findings are FIXED, never carved
    back out with an exclude (task 3350's precedent; enforced by the
    exclude-ban in the coverage test).
    """
    files = _guarded_py_files()

    # NON-VACUITY: a bad glob (renamed skills/, a parents[] off-by-one) would
    # otherwise let this invariant pass by checking nothing at all.
    assert files, (
        f"no repo-root-level *.py or skills/**/*.py found under {REPO_ROOT} "
        "(task 3485) — this cleanliness invariant would pass vacuously; the "
        "enumeration globs are almost certainly wrong"
    )

    findings = _ruff_findings(files)
    assert not findings, (
        f"repo-root-level / skills/ .py files are not clean under the members' "
        f"ruff rule set (task 3485): {sorted(findings)}. These paths are "
        f"declared lint targets of the repo-root lint_command, so a finding "
        f"here red-walls every merge and main-tip sweep repo-wide. FIX the "
        f"finding — do not weaken the command with an --exclude or a narrowed "
        f"target. Probe: ruff check {' '.join(MEMBER_RULE_SET_FLAGS)} over "
        f"{[p.relative_to(REPO_ROOT).as_posix() for p in files]}"
    )
