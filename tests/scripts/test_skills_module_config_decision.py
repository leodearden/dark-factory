"""Decision gate: ``skills/`` is deliberately NOT a registered module config.

Task 3554, deciding the question task 3485 deferred. The DECISION is option
(a): do **not** add ``skills/orchestrator.yaml``. The repo-root
``lint_command``'s ruff head remains the declaration lever for ``skills``, and
``skills/**/*.py`` ruff cleanliness is carried into
``verify.run_full_verification``'s gather by a module config's
``test_command`` — not by any ``lint_command``. The full record, with the
measurement and the two corrections it forces on 3485's block, lives in the
``DECIDED — skills/orchestrator.yaml`` block in
``dark-factory-orchestrator.yaml``.

This file is the EXECUTABLE half of that record. A documented-but-ungated
invariant is the same defect class these config tasks exist to close: 3485's
block was pure prose and was already carrying two falsified claims when this
task re-measured it. Every assertion here is on RUNTIME state — discovered
module-config prefixes, tracked-file sweeps, parsed command targets, file
existence — never on comment or docstring prose.

WHY THESE RATCHETS NEED A MEASURED RED. Because the decision is "leave as-is",
every assertion is green at HEAD BY CONSTRUCTION, so a green run proves
nothing about whether the guard bites. Each test below therefore records the
failure text observed against a named scratch mutation, following
``test_root_lint_covers_nonmember_py.py``'s "MEASURED RED at base main
``1f83dbed15``" precedent. The scratch artifacts are reverted before commit.

MUST NOT SKIP. No ``pytest.importorskip`` and no try/except-and-skip anywhere:
a missing ``git`` or an unimportable ``orchestrator.config`` must FAIL this
guard, not silently pass it. A guard against a vacuous gate that can itself go
vacuous is worthless.

Production code is cited BY SYMBOL, deliberately never by file:line — task
3445's explicit correction of the convention task 3350 established, after
every line pin copied forward had already rotted at HEAD.

PLACEMENT IS LOAD-BEARING, NOT STYLISTIC. This file lives in ``tests/scripts/``
because that directory carries its own module config, so the guard actually
runs under FULL_SUITE and merge-role ``merge_verify_breadth: full``. A guard
against a vacuous gate that itself never ran on merge full-verify would be
vacuous in the same way (``test_scripts_module_config.py``'s own rationale).
"""
from __future__ import annotations

import pathlib
import subprocess

from orchestrator.config import _discover_module_configs

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The repo-root config carrying the DECIDED block this file gates. Named by
# its canonical, REQUIRED filename (what the dashboard's escalation-URL
# discovery keys on), so the pointer in every failure message below stays
# findable.
DF_CONFIG_NAME = "dark-factory-orchestrator.yaml"


def _git_ls_files(*pathspecs: str) -> list[str]:
    """Tracked paths matching *pathspecs*, via real ``git ls-files``.

    MEASURED, not assumed (in a throwaway repo at this base): the default
    pathspec matcher treats ``**`` as crossing ``/``, so
    ``skills/**/orchestrator.yaml`` really does match a nested
    ``skills/foo/orchestrator.yaml`` and ``skills/**/test_*.py`` really does
    match ``skills/foo/bar/test_probe.py``. Intent-to-add (``git add -N``)
    entries ARE listed, which is what lets the scratch mutations below be
    observed without committing them.

    No try/except-and-skip: a missing or broken ``git`` must fail the caller.
    """
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "--", *pathspecs],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        f"`git ls-files -- {' '.join(pathspecs)}` exited {proc.returncode} "
        f"(task 3554). A broken git must fail this guard rather than let it "
        f"pass vacuously; stderr: {proc.stderr.strip()!r}"
    )
    return [line for line in proc.stdout.splitlines() if line]


def test_no_skills_prefixed_module_config_is_registered() -> None:
    """No ``skills``-prefixed module config may exist. This is THE decision.

    Two independent observations, asserted together so one failure report
    shows both: the real ``config._discover_module_configs`` walk over this
    worktree, and a tracked-file sweep. The discovery walk is the primary — it
    is production behaviour, and it registers a directory iff that directory
    DIRECTLY contains an ``orchestrator.yaml`` carrying at least one
    overridable field. The ls-files sweep is belt-and-braces: it keeps biting
    if discovery's excluded-directory pruning ever changes such that a
    committed ``skills/orchestrator.yaml`` stops being walked.

    ``skills`` (prefix depth 1) is well inside the effective ``lock_depth`` of
    4, and ``_discover_module_configs`` only skips ``prefix == '.'``, so such a
    config WOULD be discovered. Registering one is feasible; it is declined.

    TWO REASONS, both measured (task 3554, full record in the DECIDED block):

    1. It opens a NEW vacuous gate. ``verify_plan._derive_module_runs`` emits
       an explicit ``ScopeKind.SKIPPED`` ``PlannedRun`` with ``cmd=None`` for a
       falsy ``test_command``, ``verify._executed_module_configs_from_plan``
       renders that slot back to ``None``, and ``verify._run_or_skip_timed``
       turns a ``None`` command into a ``CheckRun.skipped`` that is VACUOUSLY
       PASSING at rc=0 — the exact failure class tasks 3350, 3445 and 3485
       exist to prevent.
    2. It buys nothing that is not already held. The ``skills/**/*.py`` ruff
       gate is ALREADY inside ``verify.run_full_verification``'s gather over
       ``module_configs.values()``, carried there by a registered module
       config's ``test_command`` rather than by any ``lint_command`` — the
       lever ``test_skills_py_ruff_probe_is_collected_by_a_registered_module_config_test_command``
       pins.

    MEASURED RED at base main ``7c6039327d``, against an uncommitted scratch
    ``skills/orchestrator.yaml`` carrying ``lint_command: "true"``, ``git add
    -N``'d so both halves could be observed at once::

        AssertionError: a `skills`-prefixed module config exists (task 3554),
        but skills/ is DELIBERATELY unregistered. Discovered prefixes:
        ['skills']. Tracked configs: ['skills/orchestrator.yaml'].

    Both halves populated, confirming neither is decorative.
    """
    discovered = _discover_module_configs(REPO_ROOT)

    # NON-VACUITY: an empty discovery result would let the prefix half of this
    # invariant pass while proving nothing — the walk itself would be broken.
    assert discovered, (
        f"config._discover_module_configs found NO module configs at all under "
        f"{REPO_ROOT} (task 3554). This decision ratchet would pass vacuously; "
        f"the discovery walk is broken, which is a far larger problem than "
        f"anything this guard owns."
    )

    offending_prefixes = sorted(
        prefix
        for prefix in discovered
        if prefix == "skills" or prefix.startswith("skills/")
    )
    tracked_configs = _git_ls_files(
        "skills/**/orchestrator.yaml", "skills/orchestrator.yaml"
    )

    assert not offending_prefixes and not tracked_configs, (
        f"a `skills`-prefixed module config exists (task 3554), but skills/ is "
        f"DELIBERATELY unregistered. Discovered prefixes: "
        f"{offending_prefixes}. Tracked configs: {tracked_configs}. "
        f"TWO REASONS, and neither is stale: (1) a module config with a falsy "
        f"test_command yields a ScopeKind.SKIPPED PlannedRun that "
        f"verify._run_or_skip_timed renders as a VACUOUSLY PASSING "
        f"CheckRun.skipped at rc=0 — closing one vacuous gate by opening "
        f"another; (2) the skills/**/*.py ruff gate is ALREADY inside "
        f"verify.run_full_verification's gather over module_configs.values(), "
        f"carried by a registered module config's test_command, so "
        f"registering buys a redundant second ruff pass and nothing else. "
        f"Do NOT delete this assertion to land a config — re-open the "
        f"`DECIDED — skills/orchestrator.yaml` block in {DF_CONFIG_NAME} and "
        f"re-take the decision on the record there."
    )


def test_skills_has_no_tests_of_its_own() -> None:
    """``skills/`` must carry no test suite of its own. This is the REVISIT TRIGGER.

    The decision above rests on there being no HONEST ``test_command`` to
    declare. That is true today only because ``skills/`` has no tests: the
    tests that actually read or execute paths under ``skills/`` live in three
    OTHER uv projects (see
    ``test_every_skills_consuming_test_stays_under_a_gated_directory``), so an
    honest ``test_command`` would be a three-project union whose duplicate-run
    cost dwarfs the one ``tests/scripts/orchestrator.yaml`` already documents
    about itself.

    The moment ``skills/`` grows a suite of its own, that premise dies: a
    narrow, honest, non-duplicating ``test_command`` becomes available for the
    first time, and the decision must be RE-TAKEN rather than inherited. This
    test failing is not a defect to route around — it is the trigger firing.

    MEASURED RED at base main ``7c6039327d``, against an uncommitted scratch
    ``skills/_scratch/test_probe.py``, ``git add -N``'d::

        AssertionError: skills/ now has tests of its own (task 3554):
        ['skills/_scratch/test_probe.py']. This is the REVISIT TRIGGER for the
        `DECIDED — skills/orchestrator.yaml` block in
        dark-factory-orchestrator.yaml, not a test to edit away.
    """
    own_tests = _git_ls_files(
        "skills/**/tests/**", "skills/**/test_*.py", "skills/**/*_test.py"
    )

    assert not own_tests, (
        f"skills/ now has tests of its own (task 3554): {own_tests}. This is "
        f"the REVISIT TRIGGER for the `DECIDED — skills/orchestrator.yaml` "
        f"block in {DF_CONFIG_NAME}, not a test to edit away. That decision "
        f"declined a module config partly because no HONEST test_command "
        f"existed to declare — every test exercising skills/ lived in another "
        f"project's suite, so pointing at them meant a three-project union "
        f"duplicated on every review checkpoint, main-tip sweep and merge "
        f"full-verify. A suite living under skills/ falsifies that premise: a "
        f"narrow, non-duplicating test_command is now available, and the "
        f"registration question must be re-decided on the record in "
        f"{DF_CONFIG_NAME} — including whether declaring one still opens the "
        f"vacuous lint/type gate the sibling configs warn about."
    )
