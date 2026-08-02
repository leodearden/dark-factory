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
import shlex
import subprocess
import sys

import yaml

from orchestrator import verify_cmd

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The canonical root config filename — what the dashboard's escalation-URL
# discovery (_discover_escalation_urls) keys on. Loading it by this exact path
# means the guard fails loudly if the config is renamed or lint_command
# disappears, rather than quietly guarding nothing.
DF_CONFIG_PATH = REPO_ROOT / "dark-factory-orchestrator.yaml"

# The members' rule set, spelled out rather than resolved from config — see the
# RULE-SET / LANDING-ORDER NOTE in the module docstring.
MEMBER_RULE_SET_FLAGS = [
    "--select", "E,F,UP,B,SIM,I",
    "--ignore", "E501",
    "--line-length", "100",
]

# The chain-shape tail. verify._reproject_str relies on
# split_chain_tail(cmd, 'uv run') carrying this through verbatim, so adding
# target names to the ruff head is safe while restructuring the chain is not —
# a warning the root config states about itself and assertion (d) below pins.
MAGICMOCK_CHECKER = "check_bare_magicmock_config.py"


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


def _root_lint_command() -> str:
    return yaml.safe_load(DF_CONFIG_PATH.read_text(encoding="utf-8"))["lint_command"]


def _ruff_segment(cmd: str) -> str:
    """The ``&&``-chained segment of *cmd* that actually invokes ``ruff check``.

    Uses the production splitter ``verify_cmd.split_top_level_and`` (quote-aware)
    rather than a naive ``str.split('&&')`` — matching ``_ruff_segment`` in
    ``test_scripts_module_config.py``. Extracting the ruff segment FIRST is what
    keeps the target assertions honest: tokenising the whole chain would read
    ``&&``, ``python3`` and the magicmock checker's own directory arguments as
    ruff lint targets.
    """
    segments = verify_cmd.split_top_level_and(cmd)
    ruff_segments = [s for s in segments if "ruff check" in s]
    assert len(ruff_segments) == 1, (
        f"expected exactly one `ruff check` segment in the root lint_command "
        f"(task 3485), got {ruff_segments!r}"
    )
    return ruff_segments[0]


def _ruff_targets(cmd: str) -> list[str]:
    """The positional path arguments the ``ruff check`` segment of *cmd* lints.

    Returns whole path TOKENS. Callers must compare against them by exact
    element and never substring-match the raw command — the documented contract
    on ``_lint_leg_targets`` in ``test_fallback_verify_config.py``.

    MEASURED at base ``1f83dbed15``, so the rule is not hypothetical:
    ``'shared' in cmd`` is ALREADY TRUE via the ``check_bare_magicmock_config.py``
    TAIL leg's ``shared/tests`` argument, so a substring test would pass
    vacuously for a path the ruff leg never checks. (The plan for this task
    offered ``'skills' in cmd`` as that example; it was verified FALSE at this
    base — the token ``skills`` appears nowhere in the command — so the
    verified ``shared`` case is cited instead. The rule itself is unchanged:
    the ruff leg and the tail leg have DISJOINT target lists, and only
    per-leg token extraction can tell them apart.)
    """
    tokens = shlex.split(_ruff_segment(cmd))
    assert "check" in tokens, f"no ruff `check` subcommand in {cmd!r} (task 3485)"
    tail = tokens[tokens.index("check") + 1:]
    return [t for t in tail if not t.startswith("-")]


def _ruff_exclude_flags(cmd: str) -> list[str]:
    """Any ``--exclude`` / ``--extend-exclude`` / ``--force-exclude`` flags.

    Both spellings are caught: ``--exclude foo`` and ``--exclude=foo``.
    """
    prefixes = ("--exclude", "--extend-exclude", "--force-exclude")
    return [t for t in shlex.split(_ruff_segment(cmd)) if t.startswith(prefixes)]


def _is_covered(rel_path: str, targets: list[str]) -> bool:
    """True if *rel_path* or one of its ancestor directories is an exact target.

    Ancestor-directory coverage is real coverage: ``ruff check skills``
    traverses the directory. Membership is tested element-wise against the
    extracted target tokens, never by substring — see ``_ruff_targets``.
    """
    candidate = pathlib.PurePosixPath(rel_path)
    names = {rel_path, rel_path.rstrip("/")}
    for parent in candidate.parents:
        if parent == pathlib.PurePosixPath("."):
            continue
        names.add(parent.as_posix())
        names.add(parent.as_posix() + "/")
    names.add(rel_path + "/")
    return any(t in names for t in targets)


def test_root_lint_command_targets_every_root_level_and_skills_py() -> None:
    """The root ``lint_command``'s ruff leg must target the guarded ``.py`` files.

    The root ``lint_command`` is the ONLY available lever for the two repo-root
    files: ``_discover_module_configs`` explicitly SKIPS a root-level config
    (``prefix == '.'``), so ``conftest.py`` / ``df_pytest_isolation.py`` can
    never belong to a module config.

    WHY THE WIDENING IS LOAD-BEARING, NOT COSMETIC. This command runs on the
    module-config-less path (``verify._build_fallback_config``). There,
    ``verify_plan._scope_to_keyword`` NARROWS the ruff head to the touched
    ``.py`` files — so a ``conftest.py``-only diff is already ruff'd. The gap is
    the VERBATIM (unscoped) form, which runs when a touched file classifies
    STRUCTURAL or under FULL_SUITE. ``conftest.py`` is itself a STRUCTURAL
    trigger, so a diff touching it runs the chain verbatim and, before this
    task, never checked ``conftest.py``. A docs-only / zero-``.py`` diff hits
    the same verbatim path.

    WHAT THIS GUARD DOES NOT PROVE. It parses the CONFIG COMMAND STRING, so it
    is not evidence that runtime routing actually reaches that command — the
    same disclaimer ``TestFleetLintCoversEveryWorkspaceMember`` makes about its
    own ``KNOWN_FLEET_MEMBERS`` floor. And it deliberately does not cover
    ``scripts/**`` or ``tests/scripts/**``, which carry their own module
    configs and their own declared lint commands (tasks 3445 and 3350).

    MEASURED RED at base main ``1f83dbed15``: the ruff leg's targets were
    exactly the 7 workspace members, so ``conftest.py``,
    ``df_pytest_isolation.py`` and
    ``skills/factory-init/scripts/find_escalation_port.py`` were all uncovered.
    """
    cmd = _root_lint_command()
    targets = _ruff_targets(cmd)
    files = _guarded_py_files()

    # (a) NON-VACUITY, both sides — neither an empty file inventory nor an
    # empty target list may let this invariant pass by checking nothing.
    assert files, (
        f"no repo-root-level *.py or skills/**/*.py found under {REPO_ROOT} "
        "(task 3485) — this coverage invariant would pass vacuously; the "
        "enumeration globs are almost certainly wrong"
    )
    assert targets, (
        "the root lint_command's ruff-check leg had no positional targets at "
        f"all (task 3485) — this coverage invariant would pass vacuously; "
        f"command: {cmd!r}"
    )

    # The coverage invariant itself.
    for path in files:
        rel = path.relative_to(REPO_ROOT).as_posix()
        assert _is_covered(rel, targets), (
            f"the root lint_command's ruff-check leg does not target {rel!r} "
            f"(task 3485) — nothing else lints it either: scripts/"
            f"orchestrator.yaml targets scripts/, tests/scripts/"
            f"orchestrator.yaml targets tests/scripts/, and "
            f"_discover_module_configs skips a root-level config "
            f"(prefix == '.'), so a repo-root .py can never belong to a module "
            f"config. Add {rel!r} (or an ancestor directory) to the ruff head; "
            f"targets: {targets}"
        )

    # (b) NO EXCLUDE FLAGS. The measured findings are FIXED, not carved back
    # out; an exclude silently un-gates whatever it names while the check still
    # reports green (task 3445's precedent).
    excludes = _ruff_exclude_flags(cmd)
    assert not excludes, (
        f"the root lint_command's ruff-check leg carries exclude flag(s) "
        f"{excludes} (task 3485) — an exclude silently un-gates whatever it "
        "names while LINT still reports green, which is the exact vacuous-gate "
        "class this guard closes. Fix the finding instead of excluding it."
    )

    # (c) TYPO / STALE-TARGET guard. A bogus extra target is invisible to the
    # missing-target loop above, yet makes `ruff check` exit non-zero on every
    # fallback and merge-queue verify.
    for target in targets:
        assert (REPO_ROOT / target).exists(), (
            f"the root lint_command's ruff-check leg names {target!r}, which "
            f"does not exist under {REPO_ROOT} (task 3485) — this would make "
            "`ruff check` exit non-zero on every fallback/merge-queue verify; "
            f"targets: {targets}"
        )

    # (d) CHAIN SHAPE preserved. verify._reproject_str relies on
    # split_chain_tail(cmd, 'uv run') carrying the tail through verbatim, so
    # the command must still be a leading `uv run ruff check ...` head plus a
    # `&& python3 .../check_bare_magicmock_config.py ...` tail.
    segments = verify_cmd.split_top_level_and(cmd)
    assert segments[0].strip().startswith("uv run ruff check"), (
        f"the root lint_command no longer LEADS with `uv run ruff check` "
        f"(task 3485) — verify._reproject_str relies on "
        f"split_chain_tail(cmd, 'uv run'); adding target names to the head is "
        f"safe, restructuring the chain is not. First segment: "
        f"{segments[0]!r}"
    )
    assert any(MAGICMOCK_CHECKER in s for s in segments[1:]), (
        f"the root lint_command no longer chains a `&& python3 "
        f"{MAGICMOCK_CHECKER} ...` tail after the ruff head (task 3485) — the "
        f"bare-MagicMock config gate must survive edits to the ruff head; "
        f"segments: {segments!r}"
    )
