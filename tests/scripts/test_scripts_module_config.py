"""Routing contract: the ``scripts/`` module config must actually GATE lint.

Task 3445. ``scripts/orchestrator.yaml`` declared only ``test_command``, so
every diff confined to ``scripts/`` cleared the LINT check without ruff ever
running — 71 tracked ``.py`` files (operator tooling, the ``scripts/legibility/``
monitors, 40 test modules) gated by nothing.

Omitting ``lint_command`` does not leave a fallback in place, it DELETES the
gate: ``verify_plan._derive_module_runs`` emits an explicit
``ScopeKind.SKIPPED`` PlannedRun with ``cmd=None`` for a falsy
``lint_command``; ``verify._executed_module_configs_from_plan`` renders that
SKIPPED slot back to ``None``; ``verify._run_or_skip_timed`` turns a None
command into a ``CheckRun.skipped`` that is VACUOUSLY PASSING at rc=0. This is
the same gap task 3350 closed for ``tests/scripts/``, against whose comment
block ``scripts/`` was measured.

Cited by SYMBOL, deliberately never by file:line. This guard's first draft
pinned line numbers — adapted from the sibling guard's own pins, which had
already rotted — and every single one was wrong at HEAD, so the failure
messages sent an operator to unrelated code (the cited ``verify.py`` line for
``_executed_module_configs_from_plan`` had drifted ~125 lines and landed on an
unrelated assignment). Symbols are greppable and survive edits above them;
line numbers in prose this long cannot be kept true, and a stale pin is worse
than no pin because it reads as authoritative.

Asserted STRUCTURALLY — through the production ``derive_verify_plan`` ->
``_executed_module_configs_from_plan`` bridge — rather than by shelling out to
ruff or reading the yaml. An exit code cannot carry the claim: a None command
already exits 0, so "ruff exited 0" is exactly what the bug produces. Reading
the yaml with ``yaml.safe_load`` would be weaker still — it would pass even if
a routing regression made the config unreachable.

PLACEMENT IS LOAD-BEARING, NOT STYLISTIC. This file lives in ``tests/scripts/``
rather than ``scripts/tests/`` because under FULL_SUITE — a conftest/test-data
trigger, or merge-role ``merge_verify_breadth: full`` — the ``scripts`` module
config runs its ``test_command`` VERBATIM, and that command targets
``tests/scripts/``, not ``scripts/tests/``. The repo-root fleet chain likewise
ends in ``pytest tests/scripts/``. A guard against a vacuous gate that itself
never runs on merge full-verify would be vacuous in the same way.

Importing ``orchestrator.config`` from this suite is established precedent —
see ``test_tests_scripts_module_config.py`` and ``test_fallback_verify_config.py``
in this same directory; the root conftest.py puts every subproject's ``src/``
on sys.path.
"""
from __future__ import annotations

import pathlib
import shlex

from orchestrator.config import OrchestratorConfig, _discover_module_configs
from orchestrator.module_charter import derive_modules

from orchestrator import verify, verify_cmd, verify_plan

REPO_ROOT = pathlib.Path(__file__).parents[2]

MODULE_PREFIX = 'scripts'

# The near-homograph sibling. `scripts/orchestrator.yaml`'s test_command is
# already byte-identical to this module's (a fact tests/scripts/orchestrator.yaml
# documents about itself), so a copy-pasted lint_command left pointing here is
# the realistic wrong fix — see assertion (5), which is the ONLY place that
# copy-paste is detectable (assertion (4) cannot see it; the reason is recorded
# there).
SIBLING_PREFIX = 'tests/scripts'

# A real tracked file under scripts/, used as the representative touched-file
# for the derive_modules -> for_module routing assertions below.
SAMPLE_TOUCHED_FILE = 'scripts/tests/test_census_trigger.py'

# The mechanism, restated once so each failure message can point at it. By
# SYMBOL, not file:line — see the module docstring: the line pins this string
# originally carried were all stale at HEAD and sent readers to unrelated code.
_VACUOUS_PASS = (
    'verify_plan._derive_module_runs emits a SKIPPED PlannedRun with cmd=None '
    'for a falsy lint_command, verify._executed_module_configs_from_plan '
    'renders that back to None, and verify._run_or_skip_timed turns a None '
    'command into a CheckRun.skipped that is VACUOUSLY PASSING at rc=0'
)


def _discovered() -> dict:
    return _discover_module_configs(REPO_ROOT)


def _executed_for_touched(files: list[str]):
    """Run the PRODUCTION plan->execution bridge and return the single executed config.

    ``derive_verify_plan`` decides scope; ``_executed_module_configs_from_plan``
    renders those PlannedRuns into the exact ModuleConfig ``run_verification``
    executes. Asserting on THAT is what makes "ruff ran over scripts/" a
    structural claim rather than an exit-code claim.

    The ``lambda _f: None`` worktree_reader keeps this hermetic: no file reads,
    and nothing classifies STRUCTURAL, so the lint/type legs stay FILE_SCOPED.
    """
    mc = _discovered()[MODULE_PREFIX]
    cfg = OrchestratorConfig(project_root=REPO_ROOT)
    plan = verify_plan.derive_verify_plan(files, [mc], cfg, lambda _f: None)
    executed = verify._executed_module_configs_from_plan([mc], plan)
    assert len(executed) == 1, (
        f'expected exactly one executed module config for {files!r}, got '
        f'{[e.prefix for e in executed]!r}'
    )
    return executed[0]


def _ruff_segment(cmd: str) -> str:
    """The ``&&``-chained segment of *cmd* that actually invokes ``ruff check``.

    Reuses the production splitter (``verify_cmd.split_top_level_and``, which
    is quote-aware) rather than a naive ``str.split('&&')``.

    Chaining is an ESTABLISHED pattern here, not a hypothetical:
    ``verify_plan._scope_prefix_to_keyword``'s own docstring records that
    "every subproject's lint_command chains a ``python3 .../check_*.py <dir>``
    gate after ``ruff check``". Extracting the ruff segment first is what lets
    the target assertions below stay true if this module later adopts that
    shape — tokenising the whole chain would otherwise read ``&&``, ``python3``
    and the checker's own arguments as ruff lint targets.
    """
    segments = verify_cmd.split_top_level_and(cmd)
    ruff_segments = [s for s in segments if 'ruff check' in s]
    assert len(ruff_segments) == 1, (
        f'expected exactly one `ruff check` segment in {cmd!r}, got '
        f'{ruff_segments!r}'
    )
    return ruff_segments[0]


def _ruff_targets(cmd: str) -> list[str]:
    """The positional path arguments the ``ruff check`` segment of *cmd* lints.

    Substring checks alone cannot carry assertion (5): ``'scripts/'`` is a
    substring of ``'tests/scripts/'``, so a copy-pasted sibling command would
    satisfy a naive ``'scripts/' in cmd``. Splitting out the actual targets and
    testing LIST MEMBERSHIP (exact-element, so ``'tests/scripts/'`` does not
    match) is what makes the anti-copy-paste assertion real.
    """
    tokens = shlex.split(_ruff_segment(cmd))
    assert 'check' in tokens, f'no ruff `check` subcommand in {cmd!r}'
    tail = tokens[tokens.index('check') + 1:]
    return [t for t in tail if not t.startswith('-')]


def _ruff_exclude_flags(cmd: str) -> list[str]:
    """Any ``--exclude`` / ``--extend-exclude`` / ``--force-exclude`` flags.

    Both spellings are caught: ``--exclude foo`` and ``--exclude=foo``.
    """
    prefixes = ('--exclude', '--extend-exclude', '--force-exclude')
    return [t for t in shlex.split(_ruff_segment(cmd)) if t.startswith(prefixes)]


def test_scripts_diff_is_lint_gated() -> None:
    """A diff confined to scripts/ must actually run ruff over scripts/.

    Five assertions, one contract. (1) and (2) are routing PRECONDITIONS: they
    are asserted so that a future routing regression cannot quietly make the
    lint assertion vacuous — a config that is discovered but unreachable, or
    reachable but resolving elsewhere, would let (3) pass while nothing is
    gated in production.

    NOTE on (2) — written in the lock_depth-AGNOSTIC form, never pinning a
    literal such as ``derive_modules(...) == ['scripts']``. The pydantic Field
    default for ``lock_depth`` is 2, but the EFFECTIVE value is 4: the
    package-bundled ``orchestrator/src/orchestrator/defaults.yaml`` ships
    ``lock_depth: 4`` and is layered over the Field default on every load. At
    depth 4, ``derive_modules([SAMPLE_TOUCHED_FILE], 4)`` returns the full
    path — 3 path components is below the depth-4 truncation threshold — so
    ``normalize_lock`` leaves it whole. What matters is that each derived key
    RESOLVES back to this config. Task 3350's sibling guard hit this exact trap
    and documented it; pinning the literal would re-encode a falsified constant.
    """
    discovered = _discovered()

    # (1) ROUTING PRECONDITION — discovery registers it, under the repo-relative
    # POSIX prefix that for_module resolves by.
    assert MODULE_PREFIX in discovered, (
        f'{MODULE_PREFIX}/orchestrator.yaml is not discovered by the production '
        f'config._discover_module_configs walk (task 3445), so nothing below can '
        f'gate a {MODULE_PREFIX}/ diff. Discovered: {sorted(discovered)}'
    )

    mc = discovered[MODULE_PREFIX]
    assert mc.prefix == MODULE_PREFIX, (
        f'module config discovered for {MODULE_PREFIX} carries prefix '
        f'{mc.prefix!r}; for_module resolves by repo-relative POSIX prefix, so a '
        'mismatch makes it unroutable'
    )

    # (2) ROUTING PRECONDITION — reachable by the path that actually dispatches
    # verify. A prefix deeper than lock_depth is honoured by
    # run_full_verification (which iterates module_configs.values() directly)
    # but unreachable via scheduler/workflow, which pass normalize_lock-
    # truncated keys; config._discover_module_configs only WARNS, it does not
    # fail, so nothing but this assertion would surface the mismatch.
    cfg = OrchestratorConfig(project_root=REPO_ROOT)
    prefix_depth = len(MODULE_PREFIX.split('/'))
    assert prefix_depth <= cfg.lock_depth, (
        f'module config prefix {MODULE_PREFIX!r} has depth {prefix_depth} but '
        f'lock_depth={cfg.lock_depth}; scheduler._limit_for and '
        'workflow._resolve_module_configs truncate module paths to lock_depth '
        'components via shared.locking.normalize_lock, so this config would be '
        'unreachable through the path that dispatches verify — '
        'config._discover_module_configs logs a warning for exactly this case '
        'and carries on'
    )

    cfg._module_configs = discovered
    derived = derive_modules([SAMPLE_TOUCHED_FILE], cfg.lock_depth)
    assert derived, (
        f'derive_modules([{SAMPLE_TOUCHED_FILE!r}], {cfg.lock_depth}) derived no '
        'module lock keys at all, so the workflow would fall through to its '
        'task-<id> synthetic lock and never resolve a module config'
    )
    for key in derived:
        resolved = cfg.for_module(key)
        assert resolved is not None and resolved.prefix == MODULE_PREFIX, (
            f'derived module lock key {key!r} resolves to '
            f'{resolved.prefix if resolved else None!r}, not {MODULE_PREFIX!r} '
            f'(task 3445) — workflow._resolve_module_configs would then produce '
            'an EMPTY module list and this config would gate nothing'
        )

    executed = _executed_for_touched([SAMPLE_TOUCHED_FILE])

    # (3) THE GATE ITSELF. A None command here is not "lint deferred to some
    # other config" — it is lint DELETED, and it reports green.
    assert executed.lint_command is not None and 'ruff' in executed.lint_command, (
        f'executed lint_command is {executed.lint_command!r} for a '
        f'{MODULE_PREFIX}/-only diff (task 3445). Declaring only test_command on '
        f'this module config downgrades LINT to a vacuously-passing '
        f'CheckRun.skipped at rc=0: {_VACUOUS_PASS}. Every .py file under '
        f'{MODULE_PREFIX}/ — operator tooling and the legibility monitors '
        'included — is then linted by nothing, on a check that reports green. '
        'The repo-root lint_command does not cover it either: that command '
        'targets only shared/escalation/fused-memory/orchestrator/dashboard'
    )

    # (4) SCOPING, on the FILE_SCOPED render: the executed command must lint
    # the file that actually changed.
    #
    # This assertion deliberately does NOT claim to catch the tests/scripts/
    # copy-paste, and an earlier draft that did was WRONG. Measured against the
    # production bridge: _scope_prefix_to_keyword REPLACES the declared targets
    # with the touched-file list, so a config carrying the copy-pasted
    # `ruff check tests/scripts/` renders to
    # 'uv run --project shared ruff check scripts/tests/test_census_trigger.py'
    # — byte-identical to what the correct config renders. A
    # `SIBLING_PREFIX not in executed.lint_command` check here is therefore
    # unfalsifiable: it passes for the right config and the wrong one alike.
    # The copy-paste is only detectable on the DECLARED value — see (5), which
    # is where that claim now lives, and where it is genuinely falsifiable.
    assert SAMPLE_TOUCHED_FILE in executed.lint_command, (
        f'executed lint_command {executed.lint_command!r} does not target the '
        f'touched file {SAMPLE_TOUCHED_FILE!r} (task 3445) — '
        'verify_plan._scope_prefix_to_keyword rewrites the declared directory '
        'target to the touched file list, so a command that does not mention it '
        'is linting some other tree than the one that changed'
    )

    # (5) The FULL_SUITE / merge-role form, and the ONLY falsifiable
    # anti-copy-paste coverage in this file (see (4)). Under
    # merge_verify_breadth=full and on STRUCTURAL diffs the DECLARED command
    # runs verbatim and unscoped, so the raw value has to be right on its own
    # terms, not merely after scoping.
    assert mc.lint_command is not None and 'ruff check' in mc.lint_command, (
        f'{MODULE_PREFIX}/orchestrator.yaml declares lint_command='
        f'{mc.lint_command!r} (task 3445). Under merge-role '
        f'merge_verify_breadth=full this value runs VERBATIM, so an absent or '
        f'non-ruff command leaves the merge path ungated too: {_VACUOUS_PASS}'
    )

    # Membership, not list equality: equality would also reject a LEGITIMATE
    # strengthening — `ruff check scripts/ && python3 scripts/check_x.py
    # scripts`, the chained-sibling-gate shape _scope_prefix_to_keyword's
    # docstring says every subproject's lint_command already uses — and would
    # do so with a message accusing the author of narrowing the gate. What must
    # hold is narrower and exact: the directory itself is a ruff target
    # (exact-element, so the copy-pasted 'tests/scripts/' does NOT satisfy it).
    targets = _ruff_targets(mc.lint_command)
    assert f'{MODULE_PREFIX}/' in targets, (
        f'{MODULE_PREFIX}/orchestrator.yaml declares lint_command='
        f'{mc.lint_command!r}, whose ruff targets are {targets!r} — '
        f'{MODULE_PREFIX + "/"!r} is not among them (task 3445). The gate must '
        f'be the DIRECTORY-WIDE form: narrowing it to a file list, or leaving a '
        f'copy-pasted {SIBLING_PREFIX}/ target in place, leaves {MODULE_PREFIX}/ '
        f'ungated under merge full-verify behind a check that reports green — '
        'the exact defect this guard exists to prevent. Chaining an additional '
        '&&-joined gate after ruff is fine and does NOT trip this assertion'
    )

    # The other half of "directory-wide": nothing carved back out of it.
    excludes = _ruff_exclude_flags(mc.lint_command)
    assert not excludes, (
        f'{MODULE_PREFIX}/orchestrator.yaml declares lint_command='
        f'{mc.lint_command!r}, which carves files back out with {excludes!r} '
        f'(task 3445). The three findings measured at declaration time were '
        f'FIXED rather than excluded, per the task-3350 precedent; an exclude here '
        f'silently un-gates whatever it names while the check still reports '
        'green'
    )

    sibling = discovered.get(SIBLING_PREFIX)
    assert sibling is not None, (
        f'{SIBLING_PREFIX}/orchestrator.yaml is no longer discovered, so the '
        'anti-copy-paste comparison below cannot be made (task 3445)'
    )
    assert mc.lint_command != sibling.lint_command, (
        f'{MODULE_PREFIX} and {SIBLING_PREFIX} declare a BYTE-IDENTICAL '
        f'lint_command {mc.lint_command!r} (task 3445). These two directories '
        f'are distinct trees; a shared command means one of them is linting the '
        f'other and its own files are gated by nothing. Note the two '
        f'test_commands ARE byte-identical by design — that is a different, '
        'already-recorded issue and is not license to duplicate this one'
    )
