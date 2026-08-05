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
import posixpath
import shlex
import subprocess

from orchestrator import verify_cmd
from orchestrator.config import _discover_module_configs

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The repo-root config carrying the DECIDED block this file gates. Named by
# its canonical, REQUIRED filename (what the dashboard's escalation-URL
# discovery keys on), so the pointer in every failure message below stays
# findable.
DF_CONFIG_NAME = "dark-factory-orchestrator.yaml"

_PYTEST = "pytest"

# The probe that actually runs ruff over skills/**/*.py. Named as a path
# because what matters is which module config COLLECTS it, not what it
# asserts. It rglobs (REPO_ROOT/'skills').rglob('*.py') and subprocesses ruff
# with --output-format json — so it is the only thing running ruff over
# skills/ inside run_full_verification's gather.
SKILLS_PY_RUFF_PROBE = "tests/scripts/test_root_lint_covers_nonmember_py.py"

# Every tracked test that READS or EXECUTES a real path under skills/ at
# runtime. Enumerated first-hand at base main 7c6039327d, and deliberately
# NOT a count: a hard-coded count of a directory's contents rots on the next
# file added (task 3460 recorded that lesson twice against a "40" that was
# already stale at HEAD). Membership and coverage are asserted; the total
# never is.
#
# The three uv projects this spans are the whole reason the DECIDED block
# rejects option (b): an honest skills/ test_command would be their UNION,
# including orchestrator/tests/ — the fleet's largest suite — duplicated on
# every review checkpoint, main-tip sweep and merge full-verify.
#
# Synthetic mentions are EXCLUDED by construction, so this stays an inventory
# of real consumers: cockpit/tests/ builds '/repo/skills/spawn/spawn-claude.sh'
# as a fake argv string, and orchestrator/tests/test_verify*.py write a
# tmp_path 'skills/foo.md' fixture. Neither touches the repo's skills/ tree.
SKILLS_CONSUMING_TESTS = (
    # Exercises orchestrator.agents.skill_prompt.load_skill_system_prompt,
    # which walks up from its own __file__ to the IN-REPO skills/<name>/
    # SKILL.md (not ~/.claude/skills); reads skills/unblock-auto/ and
    # skills/escalation-watcher-auto/.
    "orchestrator/tests/test_skill_prompt.py",
    # Uses that same real loader's output as its oracle.
    "orchestrator/tests/test_harness_watcher_supervisor.py",
    # EXECUTES the real skills/spawn/hooks/*.sh.
    "orchestrator/tests/test_session_hooks.py",
    # EXECUTES the real skills/spawn/spawn-claude.sh.
    "tests/scripts/test_spawn_claude.py",
    # Reads skills/factory-init/references/supervised-unit.md.
    "tests/scripts/test_systemd_restart_backoff.py",
    # Rglobs skills/**/*.py and RUNS ruff over them — the deciding fact.
    SKILLS_PY_RUFF_PROBE,
    # Reads skills/spawn/spawn-claude.sh to cross-check a bash/Python mirror.
    "scripts/tests/test_legibility_inventory.py",
)


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


def _pytest_segment(cmd: str) -> str:
    """The ``&&``-chained segment of *cmd* that actually invokes ``pytest``.

    Uses the production splitter ``verify_cmd.split_top_level_and``
    (quote-aware) rather than a naive ``str.split('&&')`` — DUPLICATED, with
    this pointer comment, from ``_ruff_segment`` in
    ``test_root_lint_covers_nonmember_py.py`` and ``_segment`` in
    ``test_scripts_module_config.py``. That is the established convention here
    rather than an oversight: ``tests/scripts/`` modules are not an importable
    package, so these guards duplicate the helper and name their siblings.

    Splitting matters for real commands in this repo, not hypothetically: the
    repo-root fleet ``test_command`` is a seven-segment ``&&`` chain of
    ``cd <dir> && uv run pytest ...`` clauses, so tokenising the whole string
    would read ``cd``, ``&&`` and directory names as pytest targets.
    """
    segments = verify_cmd.split_top_level_and(cmd)
    matching = [s for s in segments if _PYTEST in s]
    assert matching, (
        f"no `{_PYTEST}` segment in {cmd!r} (task 3554), so its collected "
        f"targets cannot be located"
    )
    return matching[0]


def _pytest_collected_dirs(cmd: str) -> list[str]:
    """Repo-relative paths *cmd*'s pytest segment collects.

    Two things this must get right, both MEASURED against the real configs at
    base main ``7c6039327d`` rather than assumed:

    1. ``--directory``. Most module ``test_command``s are
       ``uv run --project X --directory X pytest tests/ ...`` — the target
       ``tests/`` is relative to ``X``, not to the worktree root. Ignoring the
       ``--directory`` prefix would resolve ``orchestrator``'s target to a
       repo-root ``tests/`` that no consumer test lives under, silently
       breaking the coverage assertions. Both ``--directory X`` and
       ``--directory=X`` spellings are handled.
    2. Positional targets only, never flag values. Anchoring at the ``pytest``
       token drops the ``--project shared`` argument (it precedes ``pytest``),
       and the ``-``-prefix filter drops ``--tb=short``, ``-q`` and
       ``--timeout=300``. Exact TOKEN extraction, never a substring test:
       ``pytest tests/scripts/ --ignore=scripts/tests/`` and
       ``pytest scripts/tests/test_x.py`` both satisfy a naive
       ``'scripts/tests/' in cmd`` while collecting something else — the first
       collects the exact opposite of what the substring appears to prove
       (``_pytest_targets``' documented contract in
       ``test_scripts_module_config.py``).
    """
    segment = _pytest_segment(cmd)
    tokens = shlex.split(segment)

    base = ""
    for index, token in enumerate(tokens):
        if token == "--directory" and index + 1 < len(tokens):
            base = tokens[index + 1]
        elif token.startswith("--directory="):
            base = token.split("=", 1)[1]

    assert _PYTEST in tokens, (
        f"no bare `{_PYTEST}` token in the pytest segment of {cmd!r} "
        f"(task 3554), so the positional targets cannot be located"
    )
    tail = tokens[tokens.index(_PYTEST) + 1:]
    targets = [t for t in tail if not t.startswith("-")]
    return [posixpath.normpath(posixpath.join(base, t)) for t in targets]


def _all_collected_dirs() -> dict[str, list[str]]:
    """``{module prefix: repo-relative pytest targets}`` for every registered config.

    Built from the real ``config._discover_module_configs`` walk, so the
    ratchets below track production discovery rather than a hand-mirrored copy
    of it. A config with a falsy ``test_command`` contributes nothing — which
    is itself the vacuous-gate hazard the DECIDED block cites, so it is skipped
    rather than crashing the helper.
    """
    return {
        prefix: _pytest_collected_dirs(mc.test_command)
        for prefix, mc in _discover_module_configs(REPO_ROOT).items()
        if mc.test_command
    }


def _is_collected(rel_path: str, targets: list[str]) -> bool:
    """True if *rel_path* or one of its ancestor directories is an exact target.

    Ancestor-directory coverage is real coverage: ``pytest tests/scripts/``
    collects everything under that directory. Membership is tested
    element-wise against normalised target tokens, never by substring — see
    ``_pytest_collected_dirs``. Trailing slashes are already gone via
    ``posixpath.normpath``, so ``tests/scripts/`` and ``tests/scripts`` compare
    equal: the production deriver is not committed to either spelling
    (``verify_plan._fallback_pytest_targets`` maps a touched conftest to its
    PARENT DIRECTORY, which yields the slashless form), and a literal
    comparison would fail with a message accusing the author of removing a gate
    at the moment nothing changed.
    """
    candidate = pathlib.PurePosixPath(rel_path)
    names = {rel_path}
    for parent in candidate.parents:
        if parent == pathlib.PurePosixPath("."):
            continue
        names.add(parent.as_posix())
    return any(t in names for t in targets)


def test_skills_py_ruff_probe_is_collected_by_a_registered_module_config_test_command() -> None:
    """Some registered module config's ``test_command`` must collect the skills/ ruff probe.

    THIS IS THE DECIDING FACT of task 3554, as an executable assertion rather
    than prose. ``test_root_lint_covers_nonmember_py.py`` subprocesses ruff over
    ``REPO_ROOT.glob('*.py') + (REPO_ROOT/'skills').rglob('*.py')`` and asserts
    an empty finding set. It is the ONLY thing running ruff over
    ``skills/**/*.py`` inside ``verify.run_full_verification``'s asyncio.gather
    over ``module_configs.values()`` — i.e. on every review checkpoint,
    ``run_main_tip_sweep`` and merge full-verify.

    The repo-root ``lint_command``'s ruff head DECLARES ``skills`` as a target
    but cannot deliver that: its verbatim form is reached only through
    ``verify.run_verification`` with ``module_config is None``. So a module
    config's ``test_command`` is what actually carries the skills/ lint gate,
    and narrowing the owning command away from the probe's directory would
    SILENTLY remove it — nothing would report skipped, nothing would exit
    non-zero.

    DELIBERATELY NOT ASSERTED: which config carries it, or how many do. Today
    both ``tests/scripts`` and ``scripts`` collect ``tests/scripts/``, and task
    3383's pending dedupe-by-command fix will legitimately change that
    arrangement. Pinning the identity or the count would go red on a correct
    refactor and get suppressed; pinning "SOMETHING still carries it" keeps
    biting on the failure that actually matters.

    MEASURED RED at base main ``7c6039327d``, against uncommitted scratch edits
    narrowing BOTH collectors' ``test_command``s from the ``tests/scripts/``
    directory to a single unrelated file::

        AssertionError: NO registered module config's test_command collects
        'tests/scripts/test_root_lint_covers_nonmember_py.py' (task 3554).
        Collected targets by prefix: {...'scripts':
        ['tests/scripts/test_spawn_claude.py', 'scripts/tests'],
        'tests/scripts': ['tests/scripts/test_spawn_claude.py']...}

    Narrowing only ONE of the two was also run, and correctly stayed GREEN —
    confirming the assertion pins the LEVER and not a particular collector.
    """
    collected = _all_collected_dirs()

    # NON-VACUITY: an empty discovery result, or configs that declare no
    # pytest targets at all, would let this pass while gating nothing.
    assert collected, (
        f"config._discover_module_configs produced no module config with a "
        f"test_command under {REPO_ROOT} (task 3554) — this coverage "
        f"invariant would pass vacuously; discovery is broken."
    )

    carriers = sorted(
        prefix
        for prefix, targets in collected.items()
        if _is_collected(SKILLS_PY_RUFF_PROBE, targets)
    )

    assert carriers, (
        f"NO registered module config's test_command collects "
        f"{SKILLS_PY_RUFF_PROBE!r} (task 3554). Collected targets by prefix: "
        f"{collected}. That file is the ONLY thing running ruff over "
        f"skills/**/*.py inside verify.run_full_verification's gather over "
        f"module_configs.values(), so narrowing the owning command SILENTLY "
        f"removes the skills/ lint gate — nothing reports skipped and nothing "
        f"exits non-zero. The repo-root lint_command's ruff head declares "
        f"`skills` but cannot deliver this: its verbatim form is reached only "
        f"via verify.run_verification with module_config is None. Task 3554 "
        f"declined to register skills/orchestrator.yaml BECAUSE this gate "
        f"already holds, so removing it invalidates that decision rather than "
        f"merely losing coverage — re-open the `DECIDED — "
        f"skills/orchestrator.yaml` block in {DF_CONFIG_NAME}."
    )


def test_every_skills_consuming_test_stays_under_a_gated_directory() -> None:
    """Each skills/-consuming test must still exist and still be collected by a gated dir.

    CORRECTION 1 of the DECIDED block rests on this inventory: the coverage
    skills/ actually has is held by its CONSUMERS' suites, spread across three
    uv projects, which is why an honest skills/ ``test_command`` would be their
    union and why option (b) was rejected on cost. Two ways that premise can
    rot silently, both asserted here:

    * A consumer is MOVED or DELETED. The inventory then names a file that no
      longer exists, and the claim "these suites cover skills/" quietly
      describes a smaller set than the reader believes.
    * A consumer's directory stops being collected by any registered module
      config's ``test_command``. The file still exists, still reads skills/,
      and never runs under ``run_full_verification``.

    NO COUNT is asserted anywhere — house convention (task 3460): a hard-coded
    count of a directory's contents rots on the next file added. Non-emptiness
    is asserted, because an emptied tuple would make the whole test vacuous.

    Note what this does NOT claim. These suites run on THEIR OWN diffs and on
    full verify; they do NOT run on a diff that changes the skills/ artifact
    they read, because ``verify.run_scoped_verification`` short-circuits on
    ``_has_source_files`` for a .md/.sh-only diff. That residual gap is
    recorded in the DECIDED block with its own follow-up ticket; this guard
    pins only what the decision actually relies on.

    MEASURED RED at base main ``7c6039327d``, both halves, each against its own
    uncommitted scratch mutation. Renaming one inventory entry to a path that
    does not exist (``orchestrator/tests/test_skill_prompt_RENAMED.py``)::

        AssertionError: skills/-consuming tests have MOVED or been DELETED
        (task 3554): ['orchestrator/tests/test_skill_prompt_RENAMED.py'].

    And narrowing both ``tests/scripts/``-collecting ``test_command``s to a
    single unrelated file::

        AssertionError: skills/-consuming tests are no longer collected by ANY
        registered module config's test_command (task 3554):
        ['tests/scripts/test_root_lint_covers_nonmember_py.py',
        'tests/scripts/test_systemd_restart_backoff.py'].

    That second run also confirms ``_pytest_collected_dirs`` resolves
    ``--directory`` correctly: the surviving prefixes reported
    ``'orchestrator': ['orchestrator/tests']``, not a repo-root ``['tests']``,
    which is what keeps the three orchestrator/tests/ entries above genuinely
    checked rather than accidentally uncovered.
    """
    collected = _all_collected_dirs()

    # NON-VACUITY, both sides.
    assert SKILLS_CONSUMING_TESTS, (
        "the skills/-consuming test inventory is EMPTY (task 3554) — this "
        "guard would pass vacuously, and CORRECTION 1 of the DECIDED block in "
        f"{DF_CONFIG_NAME} would be asserting nothing."
    )
    assert collected, (
        f"config._discover_module_configs produced no module config with a "
        f"test_command under {REPO_ROOT} (task 3554) — this coverage "
        f"invariant would pass vacuously; discovery is broken."
    )

    missing = [
        rel for rel in SKILLS_CONSUMING_TESTS if not (REPO_ROOT / rel).is_file()
    ]
    assert not missing, (
        f"skills/-consuming tests have MOVED or been DELETED (task 3554): "
        f"{missing}. These files are the evidence that skills/ coverage is "
        f"held by its CONSUMERS' suites — the premise CORRECTION 1 of the "
        f"`DECIDED — skills/orchestrator.yaml` block in {DF_CONFIG_NAME} "
        f"rests on. Do NOT just edit the tuple to match: if a consumer was "
        f"deleted rather than moved, skills/ lost real coverage and the "
        f"registration decision must be RE-TAKEN on the record in that block. "
        f"If it merely moved, update the tuple AND the enumeration in the "
        f"block, which must not drift apart."
    )

    ungated = sorted(
        rel
        for rel in SKILLS_CONSUMING_TESTS
        if not any(_is_collected(rel, targets) for targets in collected.values())
    )
    assert not ungated, (
        f"skills/-consuming tests are no longer collected by ANY registered "
        f"module config's test_command (task 3554): {ungated}. Collected "
        f"targets by prefix: {collected}. Such a test still reads or executes "
        f"real skills/ paths but never runs inside "
        f"verify.run_full_verification's gather, so the coverage CORRECTION 1 "
        f"of the `DECIDED — skills/orchestrator.yaml` block in "
        f"{DF_CONFIG_NAME} credits to the consumers' suites is no longer "
        f"there. Re-take the decision on the record in that block rather than "
        f"editing this tuple."
    )
