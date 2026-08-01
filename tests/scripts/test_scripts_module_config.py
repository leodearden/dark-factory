"""Routing contract: the ``scripts/`` module config must actually GATE lint.

Task 3445. ``scripts/orchestrator.yaml`` declared only ``test_command``, so
every diff confined to ``scripts/`` cleared the LINT check without ruff ever
running — 71 tracked ``.py`` files (operator tooling, the ``scripts/legibility/``
monitors, 40 test modules) gated by nothing.

Omitting ``lint_command`` does not leave a fallback in place, it DELETES the
gate: ``verify_plan.py:394-403`` emits an explicit ``ScopeKind.SKIPPED``
PlannedRun with ``cmd=None`` for a falsy ``lint_command``,
``_executed_module_configs_from_plan`` (``verify.py:4897``) renders SKIPPED
back to ``None``, and ``_run_or_skip_timed`` (``verify.py:4202-4203``) turns a
None command into a ``CheckRun.skipped`` that is VACUOUSLY PASSING at rc=0.
This is the same gap task 3350 closed for ``tests/scripts/``, against whose
comment block ``scripts/`` was measured.

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

from orchestrator import verify, verify_plan
from orchestrator.config import OrchestratorConfig, _discover_module_configs
from orchestrator.module_charter import derive_modules

REPO_ROOT = pathlib.Path(__file__).parents[2]

MODULE_PREFIX = 'scripts'

# The near-homograph sibling. `scripts/orchestrator.yaml`'s test_command is
# already byte-identical to this module's (a fact tests/scripts/orchestrator.yaml
# documents about itself), so a copy-pasted lint_command left pointing here is
# the realistic wrong fix — see the anti-copy-paste assertions below.
SIBLING_PREFIX = 'tests/scripts'

# A real tracked file under scripts/, used as the representative touched-file
# for the derive_modules -> for_module routing assertions below.
SAMPLE_TOUCHED_FILE = 'scripts/tests/test_census_trigger.py'

# The mechanism, restated once so each failure message can point at it.
_VACUOUS_PASS = (
    'verify_plan.py:394-403 emits a SKIPPED PlannedRun with cmd=None for a '
    'falsy lint_command, _executed_module_configs_from_plan renders that back '
    'to None (verify.py:4897), and _run_or_skip_timed turns a None command '
    'into a CheckRun.skipped that is VACUOUSLY PASSING at rc=0 '
    '(verify.py:4202-4203)'
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


def _ruff_targets(cmd: str) -> list[str]:
    """The positional path arguments a ``ruff check`` command lints.

    Substring checks alone cannot carry assertion (5): ``'scripts/'`` is a
    substring of ``'tests/scripts/'``, so a copy-pasted sibling command would
    satisfy a naive ``'scripts/' in cmd``. Splitting out the actual targets is
    what makes the anti-copy-paste assertion real.
    """
    tokens = shlex.split(cmd)
    assert 'check' in tokens, f'no ruff `check` subcommand in {cmd!r}'
    tail = tokens[tokens.index('check') + 1:]
    return [t for t in tail if not t.startswith('-')]


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
    # truncated keys; config.py:4712-4725 warns, it does not fail.
    cfg = OrchestratorConfig(project_root=REPO_ROOT)
    prefix_depth = len(MODULE_PREFIX.split('/'))
    assert prefix_depth <= cfg.lock_depth, (
        f'module config prefix {MODULE_PREFIX!r} has depth {prefix_depth} but '
        f'lock_depth={cfg.lock_depth}; the scheduler (_limit_for) and workflow '
        '(_resolve_module_configs) truncate module paths to lock_depth '
        'components via normalize_lock, so this config would be unreachable '
        'through the path that dispatches verify (config.py:4712-4725)'
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

    # (4) ANTI-COPY-PASTE, on the FILE_SCOPED render. `scripts/` and
    # `tests/scripts/` are near-homographs and this module's test_command is
    # already byte-identical to the sibling's, so a lint_command left pointing
    # at tests/scripts/ is the single most likely wrong fix — it would present
    # as fully green while leaving every file under scripts/ unlinted,
    # reproducing the original defect behind a config that now looks correct.
    assert SAMPLE_TOUCHED_FILE in executed.lint_command, (
        f'executed lint_command {executed.lint_command!r} does not target the '
        f'touched file {SAMPLE_TOUCHED_FILE!r} (task 3445) — '
        '_scope_prefix_to_keyword rewrites the declared directory target to the '
        'touched file list, so a command that does not mention it is linting '
        'some other tree than the one that changed'
    )
    assert SIBLING_PREFIX not in executed.lint_command, (
        f'executed lint_command {executed.lint_command!r} targets '
        f'{SIBLING_PREFIX}/ for a {MODULE_PREFIX}/-only diff (task 3445) — this '
        f'is the copy-paste failure: {SIBLING_PREFIX}/ has its own module config '
        f'and its own lint_command, so this one would report green while every '
        f'file under {MODULE_PREFIX}/ stays unlinted'
    )

    # (5) The FULL_SUITE / merge-role form. Under merge_verify_breadth=full and
    # on STRUCTURAL diffs the DECLARED command runs verbatim and unscoped, so
    # the raw value has to be right on its own terms, not merely after scoping.
    assert mc.lint_command is not None and 'ruff check' in mc.lint_command, (
        f'{MODULE_PREFIX}/orchestrator.yaml declares lint_command='
        f'{mc.lint_command!r} (task 3445). Under merge-role '
        f'merge_verify_breadth=full this value runs VERBATIM, so an absent or '
        f'non-ruff command leaves the merge path ungated too: {_VACUOUS_PASS}'
    )
    assert _ruff_targets(mc.lint_command) == [f'{MODULE_PREFIX}/'], (
        f'{MODULE_PREFIX}/orchestrator.yaml declares lint_command='
        f'{mc.lint_command!r}, whose ruff targets are '
        f'{_ruff_targets(mc.lint_command)!r} rather than [{MODULE_PREFIX + "/"!r}] '
        f'(task 3445). The gate must be the directory-wide form — narrowing it '
        f'to a file list or carving out an exclude would leave part of '
        f'{MODULE_PREFIX}/ ungated under merge full-verify, which is the defect '
        'this guard exists to prevent'
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
