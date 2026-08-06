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

RE-CONFIRMED at base main ``5078f6df15``, 28 commits ahead of the
``7c6039327d`` every MEASURED RED below cites. Three of the surviving ratchets
were re-run against freshly recreated scratch mutations at the handed-off base
— the direct-child ``skills/test_probe.py`` trigger, the ``_guarded_py_files``
narrowing (still ``1 failed, 5 passed``, the path-level sibling still green),
and the ``-k some_expr`` phantom-target parse (still ``OLD
['tests/scripts/', 'some_expr']`` vs ``NEW ['tests/scripts']``) — and every
failure text reproduced VERBATIM. The original shas are kept rather than
overwritten: rewriting them would claim a first observation that was not taken
at that base.

A FOURTH RATCHET WAS REMOVED ON REVIEW, and the removal is recorded rather
than silently absorbed (task 3460's correct-rather-than-delete precedent).
``test_no_unlisted_skills_mentioning_test_escapes_triage`` swept every tracked
test module for the literal ``skills/`` and required each hit to appear in one
of two hand-maintained sets here. Its staleness half had already been removed
for making other packages' comment PROSE a merge-blocking gate on this module
(measured: a cosmetic reword of a comment in
``scripts/tests/test_legibility_sampling.py`` produced ``1 failed, 5 passed``
here, from a diff touching nothing this file gates). The surviving half had
the identical coupling merely inverted — ADDING the substring ``skills/`` to a
docstring anywhere in ``escalation/tests/``, ``fused-memory/tests/``,
``orchestrator/tests/`` or ``scripts/tests/`` red-walled this guard — while
being a self-confessed LOWER BOUND that misses the case that matters:
``orchestrator/tests/test_harness_watcher_supervisor.py`` is a real consumer
containing the token ZERO times. High false-positive rate on cosmetic edits in
unrelated packages, known false negatives on real consumers, so the whole
detector went. DO NOT REINTRODUCE IT, and specifically do not reintroduce it
as a word-boundary or regex variant: that preserves the identical dependency
on other packages' prose behind a cleverer matcher. The ADD-direction drift is
knowingly ungated — the enumeration in the DECIDED block may under-describe the
real consumer set, which only makes CORRECTION 1's cost argument against
option (b) weaker (more consumers = more expensive union), never unsound.

MUST NOT SKIP. No ``pytest.importorskip`` and no try/except-and-skip anywhere:
a missing ``git``, an unimportable ``orchestrator.config`` or an unimportable
``test_root_lint_covers_nonmember_py`` must FAIL this guard, not silently pass
it. A guard against a vacuous gate that can itself go vacuous is worthless.

DUPLICATED COMMAND-PARSING HELPERS, KNOWINGLY. ``_pytest_segment`` /
``_pytest_collected_dirs`` / ``_is_collected`` are a THIRD hand-maintained
copy of a trio that also exists as ``_ruff_segment`` / ``_ruff_targets`` /
``_is_covered`` in ``test_root_lint_covers_nonmember_py.py`` and ``_segment``
/ ``_pytest_targets`` in ``test_scripts_module_config.py``. This file's first
draft justified that with "``tests/scripts/`` modules are not an importable
package" — which is FALSE, and is corrected here rather than deleted (the
house convention this file's own DECIDED block follows). ``conftest.py`` in
this directory puts it on ``sys.path`` precisely so siblings import;
``systemd_unit_invariants.py`` is that pattern in production, lifted by task
3408 with the explicit rationale that duplicating a shared invariant "is how
the two copies drift until one silently stops catching the defect". The
copies HAVE already drifted: ``_is_covered`` normalises trailing slashes with
``rstrip('/')``, ``_is_collected`` with ``posixpath.normpath``.

The extraction into a shared ``verify_command_invariants.py`` is therefore
correct and is NOT done here only because it requires editing the two sibling
guards, which are outside this task's locks. Filed as
[tkt_0RS47G1QXJ5XDPH4T0HKKA1A9S] — a curator TICKET id, not a task id: the
curator runs asynchronously and decides create/combine/drop, so the resulting
task number is not knowable at this commit (the DECIDED block's own convention
for its two residual-gap follow-ups). That ticket carries the four behaviours
the unified helper must not regress, since ``_pytest_collected_dirs`` is
strictly richer than either sibling: ``--directory`` base resolution,
``_PYTEST_VALUE_FLAGS`` consumption, the target-EXISTS assertion, and the
``None``-not-assert contract for a non-pytest command. What
IS done here: the false rationale is corrected, the drift is named, and the
one cross-module dependency this file genuinely needs — the skills/ ruff
probe's runtime target list — is taken by IMPORT rather than by re-derivation,
demonstrating in-place that the sibling-import mechanism works.

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

# The skills/ ruff probe itself, imported so the DECIDING FACT can be asserted
# against its RUNTIME target list rather than only against its path. Resolves
# because tests/scripts/conftest.py puts this directory on sys.path (pytest's
# --import-mode=importlib deliberately does not) — the same mechanism
# test_dashboard_service_template.py and test_systemd_restart_backoff.py use
# for systemd_unit_invariants. A bare `import`, never an importorskip: if the
# probe stops importing, this guard must fail loudly.
import test_root_lint_covers_nonmember_py as skills_ruff_probe

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

    THE ASYMMETRY EVERY CALLER MUST HANDLE. ``skills/**/<x>`` does NOT match a
    DIRECT child ``skills/<x>``: git's non-pathname wildmatch requires at
    least one more ``/`` after ``skills/``. Measured in that same throwaway
    repo, with all six paths ``git add -N``'d::

        $ git ls-files -- 'skills/**/tests/**' 'skills/**/test_*.py' \
              'skills/**/*_test.py'
        skills/foo/bar_test.py
        skills/foo/test_d.py
        skills/foo/tests/test_c.py
        skills/tests/test_a.py

    ``skills/test_b.py`` and ``skills/tests/conftest.py`` are BOTH absent.
    Adding the direct-child spellings (``skills/tests/**``, ``skills/test_*.py``,
    ``skills/*_test.py``) returns all six. Every caller here therefore passes
    both the nested and the direct-child spelling of each pattern.

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

    SIX PATHSPECS, NOT THREE. Each pattern is passed in both its nested
    (``skills/**/...``) and its DIRECT-CHILD (``skills/...``) spelling, because
    git's non-pathname wildmatch will not match a direct child through ``**``
    — measured, with the raw output, in ``_git_ls_files``. Without the
    direct-child half, a suite landing at ``skills/tests/`` with only a
    ``conftest.py`` plus non-``test_``-prefixed modules, or a bare
    ``skills/test_probe.py``, would leave this trigger silently unfired. This
    mirrors what the module-config sweep above already does by passing both
    ``skills/**/orchestrator.yaml`` and ``skills/orchestrator.yaml``.

    MEASURED RED at base main ``7c6039327d``, against an uncommitted scratch
    ``skills/_scratch/test_probe.py``, ``git add -N``'d::

        AssertionError: skills/ now has tests of its own (task 3554):
        ['skills/_scratch/test_probe.py']. This is the REVISIT TRIGGER for the
        `DECIDED — skills/orchestrator.yaml` block in
        dark-factory-orchestrator.yaml, not a test to edit away.

    RE-MEASURED RED for the direct-child half, against an uncommitted scratch
    ``skills/test_probe.py`` (which the original three pathspecs did NOT
    catch), ``git add -N``'d::

        AssertionError: skills/ now has tests of its own (task 3554):
        ['skills/test_probe.py']. This is the REVISIT TRIGGER for the
        `DECIDED — skills/orchestrator.yaml` block in
        dark-factory-orchestrator.yaml, not a test to edit away.
    """
    own_tests = _git_ls_files(
        "skills/**/tests/**",
        "skills/**/test_*.py",
        "skills/**/*_test.py",
        "skills/tests/**",
        "skills/test_*.py",
        "skills/*_test.py",
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


def _pytest_segment(cmd: str) -> str | None:
    """The ``&&``-chained segment of *cmd* that invokes ``pytest``, or ``None``.

    Uses the production splitter ``verify_cmd.split_top_level_and``
    (quote-aware) rather than a naive ``str.split('&&')`` — see the module
    docstring's DUPLICATED COMMAND-PARSING HELPERS note for why this is a
    third copy and why the extraction is a follow-up rather than a defect
    left unnamed.

    Splitting matters for real commands in this repo, not hypothetically: the
    repo-root fleet ``test_command`` is a seven-segment ``&&`` chain of
    ``cd <dir> && uv run pytest ...`` clauses, so tokenising the whole string
    would read ``cd``, ``&&`` and directory names as pytest targets.

    RETURNS ``None`` RATHER THAN ASSERTING. A module whose ``test_command``
    runs something other than pytest contributes no pytest targets — that is
    the correct semantic, not an error. Asserting here would make the first
    module to declare ``cargo test`` fail BOTH ratchets in this file with a
    message naming an unrelated module and saying nothing about ``skills/``,
    which invites suppressing the guard instead of fixing anything. That
    module is not hypothetical policy: ``verify._has_source_files`` already
    keys on ``.py`` AND ``.rs``, and ``tests/scripts/orchestrator.yaml``
    documents a ``config.test_command != 'pytest'`` branch.

    Matched on the BARE ``pytest`` TOKEN after ``shlex.split``, not on a
    substring: a segment mentioning ``pytest-timeout`` or a
    ``--rootdir=/x/pytest`` value is not a pytest invocation.

    MEASURED at base main ``7c6039327d``, against an uncommitted scratch
    ``sampler/orchestrator.yaml`` whose ``test_command`` was replaced with
    ``cargo test --workspace``. The PREVIOUS asserting form produced::

        AssertionError: no `pytest` segment in 'cargo test --workspace'
        (task 3554), so its collected targets cannot be located

    — a message naming ``sampler`` and saying nothing about ``skills/``, from
    BOTH ratchets in this file. With this form the same scratch leaves the
    module GREEN (``6 passed``): ``sampler`` contributes no pytest targets,
    which is the truth, and the ``skills/`` invariants are unaffected because
    no ``skills/`` consumer lives under ``sampler/``.
    """
    for segment in verify_cmd.split_top_level_and(cmd):
        try:
            tokens = shlex.split(segment)
        except ValueError:  # unbalanced quotes in a segment — not parseable
            continue
        if _PYTEST in tokens:
            return segment
    return None


# pytest flags whose VALUE is a SEPARATE token. Only the space-separated form
# needs listing: the ``=`` form (``--timeout=300``) is already dropped whole by
# the ``-``-prefix filter. Without this, ``-k some_expr`` and ``--timeout 300``
# donate ``some_expr`` and ``300`` to the target list as phantom paths — and a
# phantom target can only ever satisfy ``_is_collected`` spuriously, never
# break it, so the failure direction is the SILENT one (a false PASS).
_PYTEST_VALUE_FLAGS = frozenset({
    "-c", "-k", "-m", "-n", "-o", "-p", "-r", "-W",
    "--basetemp", "--color", "--deselect", "--dist", "--durations",
    "--ignore", "--ignore-glob", "--import-mode", "--junitxml",
    "--log-level", "--maxfail", "--rootdir", "--tb", "--timeout",
})


def _pytest_collected_dirs(cmd: str) -> list[str]:
    """Repo-relative paths *cmd*'s pytest segment collects; ``[]`` if not pytest.

    Three things this must get right, all MEASURED against the real configs at
    base main ``7c6039327d`` rather than assumed:

    1. ``--directory``. Most module ``test_command``s are
       ``uv run --project X --directory X pytest tests/ ...`` — the target
       ``tests/`` is relative to ``X``, not to the worktree root. Ignoring the
       ``--directory`` prefix would resolve ``orchestrator``'s target to a
       repo-root ``tests/`` that no consumer test lives under, silently
       breaking the coverage assertions. Both ``--directory X`` and
       ``--directory=X`` spellings are handled.
    2. Positional targets, with VALUE-TAKING FLAGS CONSUMED. Anchoring at the
       ``pytest`` token drops the ``--project shared`` argument (it precedes
       ``pytest``), and the ``-``-prefix filter drops ``--tb=short``, ``-q``
       and ``--timeout=300``. What the ``-``-prefix filter does NOT drop is a
       space-separated flag VALUE, so ``_PYTEST_VALUE_FLAGS`` is consulted and
       the following token skipped. Note ``--ignore``/``--ignore-glob`` are in
       that set for a second reason beyond parsing: an ignored path is
       precisely NOT collected, so admitting its value as a target would
       invert the meaning.

       WHAT THIS GUARANTEES, stated no more strongly than it holds: every
       token returned was a positional argument to ``pytest`` and survived
       assertion (3). It is NOT a proof that no unlisted value-taking flag
       exists — an unknown one would still donate its value, which is exactly
       why (3) is an assertion and not a filter.
    3. EVERY EXTRACTED TARGET EXISTS. Asserted, not filtered. A phantom target
       ('300', 'some_expr') can only ever make ``_is_collected`` pass
       spuriously, so silently discarding unresolvable tokens would preserve
       the false-pass hazard the parsing above closes. Failing instead turns
       both a mis-parse and a genuinely stale target in a module config into a
       loud, named failure. ``::`` node-id suffixes are stripped before the
       check, since ``tests/test_x.py::TestC::test_y`` is a legal target whose
       full spelling is not a path.

    Exact TOKEN extraction throughout, never a substring test:
    ``pytest tests/scripts/ --ignore=scripts/tests/`` and
    ``pytest scripts/tests/test_x.py`` both satisfy a naive
    ``'scripts/tests/' in cmd`` while collecting something else — the first
    collects the exact opposite of what the substring appears to prove
    (``_pytest_targets``' documented contract in
    ``test_scripts_module_config.py``).

    BOTH HAZARDS MEASURED at base main ``7c6039327d``, against uncommitted
    scratch edits to ``tests/scripts/orchestrator.yaml``. Inserting
    ``-k some_expr`` into its pytest segment, parsed side by side::

        OLD targets: ['tests/scripts/', 'some_expr']
        NEW targets: ['tests/scripts']

    ``'some_expr'`` is a phantom that the old ``-``-prefix-only filter admitted
    silently — and a phantom can only ever make ``_is_collected`` pass, so
    nothing would have gone red. Separately, adding a genuinely stale
    positional target ``tests/scripts/does_not_exist/`` produced::

        AssertionError: the pytest segment of 'uv run --project shared pytest
        tests/scripts/ tests/scripts/does_not_exist/ --tb=short -q
        --timeout=300' names target 'tests/scripts/does_not_exist', which does
        not exist under /home/leo/src/dark-factory/.worktrees/3554 (task 3554).

    confirming assertion (3) bites rather than filtering the token away.
    """
    segment = _pytest_segment(cmd)
    if segment is None:
        return []
    tokens = shlex.split(segment)

    base = ""
    for index, token in enumerate(tokens):
        if token == "--directory" and index + 1 < len(tokens):
            base = tokens[index + 1]
        elif token.startswith("--directory="):
            base = token.split("=", 1)[1]

    targets: list[str] = []
    skip_next = False
    for token in tokens[tokens.index(_PYTEST) + 1:]:
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-"):
            skip_next = token in _PYTEST_VALUE_FLAGS
            continue
        targets.append(token)

    resolved = [posixpath.normpath(posixpath.join(base, t)) for t in targets]
    for target in resolved:
        assert (REPO_ROOT / target.split("::", 1)[0]).exists(), (
            f"the pytest segment of {cmd!r} names target {target!r}, which "
            f"does not exist under {REPO_ROOT} (task 3554). Either a module "
            f"config carries a stale target — which makes pytest exit "
            f"non-zero on every verify that runs it — or this parser admitted "
            f"a value-taking flag's VALUE as a positional target, in which "
            f"case add that flag to _PYTEST_VALUE_FLAGS. A phantom target can "
            f"only ever satisfy _is_collected spuriously, so this is asserted "
            f"rather than filtered out."
        )
    return resolved


def _collected_by_prefix() -> tuple[dict[str, list[str]], list[str]]:
    """``({prefix: pytest targets}, [prefixes skipped as non-pytest])``.

    Built from the real ``config._discover_module_configs`` walk, so the
    ratchets below track production discovery rather than a hand-mirrored copy
    of it.

    TWO KINDS OF NON-CONTRIBUTOR, both skipped rather than crashing the
    helper, and the second REPORTED so a failure stays diagnosable:

    * A falsy ``test_command`` — itself the vacuous-gate hazard the DECIDED
      block cites as reason (1) for declining a ``skills/`` config.
    * A ``test_command`` with no bare ``pytest`` token (see
      ``_pytest_segment``). Its prefix is returned in the second element so
      the assertions below can say "and these N configs declare no pytest
      command" instead of reporting a bare empty set.
    """
    collected: dict[str, list[str]] = {}
    non_pytest: list[str] = []
    for prefix, module_config in _discover_module_configs(REPO_ROOT).items():
        command = module_config.test_command
        if not command:
            non_pytest.append(prefix)
            continue
        if _pytest_segment(command) is None:
            non_pytest.append(prefix)
            continue
        collected[prefix] = _pytest_collected_dirs(command)
    return collected, sorted(non_pytest)


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

    THIS IS ONLY HALF THE DECIDING FACT. It pins that some config collects the
    probe's PATH. That the probe still points ruff AT ``skills/`` is the other
    half, and is pinned separately by
    ``test_skills_py_ruff_probe_still_globs_skills_at_runtime`` — narrowing
    ``_guarded_py_files`` would otherwise remove the gate with every assertion
    here staying green.
    """
    collected, non_pytest = _collected_by_prefix()

    # NON-VACUITY: an empty result — no configs discovered, or none declaring
    # a pytest command — would let this pass while gating nothing.
    assert collected, (
        f"no registered module config under {REPO_ROOT} declares a pytest "
        f"test_command (task 3554) — this coverage invariant would pass "
        f"vacuously. Either discovery is broken, or every config now runs a "
        f"non-pytest runner; skipped prefixes: {non_pytest}."
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


def test_skills_py_ruff_probe_still_globs_skills_at_runtime() -> None:
    """The collected probe must still point ruff AT ``skills/``, not just exist.

    THE OTHER HALF OF THE DECIDING FACT. Its sibling above pins that some
    registered module config collects
    ``test_root_lint_covers_nonmember_py.py``. That is worth nothing if the
    file stops probing ``skills/``: narrowing ``_guarded_py_files`` to
    ``REPO_ROOT.glob('*.py')`` — an entirely reasonable-looking edit while
    splitting the repo-root and ``skills/`` concerns apart — deletes the
    ``skills/`` lint gate outright while every path-level assertion in this
    module stays green. That is precisely the silent-removal failure mode this
    file exists to prevent, so it is asserted on the probe's RUNTIME OUTPUT.

    Asserted by IMPORT, not by re-deriving the glob: ``_guarded_py_files`` is
    called and its result inspected, so this tracks whatever the probe
    actually does rather than a mirrored copy that could drift from it. The
    private name is used deliberately — the coupling is the point, and a
    rename that breaks this import fails loudly at collection, which is the
    correct outcome.

    DELIBERATELY NOT ASSERTED: how many ``skills/`` files, or which ones. A
    count rots on the next file added (task 3460), and the gate's meaning is
    "``skills/`` is in the probe's scope", not "``skills/`` has N .py files".

    MEASURED RED at base main ``7c6039327d``, against an uncommitted scratch
    edit narrowing ``_guarded_py_files`` to
    ``return sorted(REPO_ROOT.glob("*.py"))``::

        AssertionError: test_root_lint_covers_nonmember_py._guarded_py_files()
        no longer returns ANY path under
        /home/leo/src/dark-factory/.worktrees/3554/skills (task 3554). It
        returned 2 file(s), all outside skills/.

    THE OTHER FIVE TESTS IN THIS MODULE ALL STAYED GREEN on that run
    (``1 failed, 5 passed``) — including the path-level sibling above. That is
    the whole justification for this assertion existing separately: the gate
    was gone and only this test noticed.
    """
    skills_dir = REPO_ROOT / "skills"
    probed = skills_ruff_probe._guarded_py_files()

    # NON-VACUITY: an empty probe result would make the skills/ half below
    # unfalsifiable AND means the probe itself is gating nothing. Its own
    # non-vacuity assertion covers this too; duplicating it here keeps THIS
    # failure message pointed at the decision rather than at the sibling.
    assert probed, (
        "test_root_lint_covers_nonmember_py._guarded_py_files() returned "
        "NOTHING (task 3554) — the probe that carries the skills/ lint gate "
        "into verify.run_full_verification is enumerating no files at all, "
        "so the gate is vacuous regardless of which module config collects "
        "it."
    )

    under_skills = [p for p in probed if p.is_relative_to(skills_dir)]
    assert under_skills, (
        f"test_root_lint_covers_nonmember_py._guarded_py_files() no longer "
        f"returns ANY path under {skills_dir} (task 3554). It returned "
        f"{len(probed)} file(s), all outside skills/. That function is the "
        f"ONLY thing pointing ruff at skills/**/*.py inside "
        f"verify.run_full_verification's gather over module_configs.values(), "
        f"so narrowing it SILENTLY removes the skills/ lint gate — nothing "
        f"reports skipped and nothing exits non-zero, and the sibling "
        f"assertion that a module config still COLLECTS that file stays green "
        f"while gating nothing. Task 3554 declined to register "
        f"skills/orchestrator.yaml BECAUSE this gate already holds, so "
        f"narrowing it invalidates that decision rather than merely losing "
        f"coverage — re-open the `DECIDED — skills/orchestrator.yaml` block "
        f"in {DF_CONFIG_NAME} and re-take it on the record."
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

    THE OPPOSITE DRIFT — a NEW consumer that never reaches this tuple — is not
    detectable here and is DELIBERATELY UNGATED anywhere. The ratchet that once
    pinned it was removed on review for coupling this module to other packages'
    docstring and comment prose; the full record, and why a regex or
    word-boundary variant is not an acceptable replacement, is in this module's
    docstring. The asymmetry is tolerable in a way the drift above is not: an
    unrecorded extra consumer makes CORRECTION 1's cost argument against option
    (b) understate the union's size, i.e. errs toward the conclusion already
    taken, whereas a consumer that MOVES, DIES or leaves a gated directory
    removes coverage the decision actually credits.
    """
    collected, non_pytest = _collected_by_prefix()

    # NON-VACUITY, both sides. Spelled `len(...) > 0` and NOT as a truthiness
    # test: the inventory is a tuple LITERAL in this module, so pyright folds
    # `assert SKILLS_CONSUMING_TESTS` to a constant and emits
    # reportAssertAlwaysTrue — MEASURED as the single warning of an otherwise
    # clean `npx pyright tests/scripts/` run. The runtime guard is real
    # regardless (emptying the tuple must fail here rather than silently make
    # the whole test pass), so the spelling is changed rather than the check
    # dropped or the diagnostic suppressed. Do not "simplify" it back.
    assert len(SKILLS_CONSUMING_TESTS) > 0, (
        "the skills/-consuming test inventory is EMPTY (task 3554) — this "
        "guard would pass vacuously, and CORRECTION 1 of the DECIDED block in "
        f"{DF_CONFIG_NAME} would be asserting nothing."
    )
    assert collected, (
        f"no registered module config under {REPO_ROOT} declares a pytest "
        f"test_command (task 3554) — this coverage invariant would pass "
        f"vacuously. Either discovery is broken, or every config now runs a "
        f"non-pytest runner; skipped prefixes: {non_pytest}."
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
