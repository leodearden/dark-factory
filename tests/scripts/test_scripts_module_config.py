"""Routing contract: the ``scripts/`` module config must actually GATE lint and type.

Task 3445. ``scripts/orchestrator.yaml`` declared only ``test_command``, so
every diff confined to ``scripts/`` cleared the LINT check without ruff ever
running — 71 tracked ``.py`` files (operator tooling, the ``scripts/legibility/``
monitors, this directory's own test suite) gated by nothing. Task 3456 closed
the identical TYPE gap, which 3445 measured and recorded in that yaml as
knowingly-open rather than leaving it a silent absence; the burn-down had to
land first, because declaring a red command here is a fleet-wide outage, not a
transient failure.

Omitting ``lint_command`` or ``type_check_command`` does not leave a fallback
in place, it DELETES the gate: ``verify_plan._derive_module_runs`` emits an
explicit ``ScopeKind.SKIPPED`` PlannedRun with ``cmd=None`` for a falsy
command; ``verify._executed_module_configs_from_plan`` renders that SKIPPED
slot back to ``None``; ``verify._run_or_skip_timed`` turns a None command into
a ``CheckRun.skipped`` that is VACUOUSLY PASSING at rc=0. This is the same gap
task 3350 closed for ``tests/scripts/``, against whose comment block
``scripts/`` was measured.

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
rather than ``scripts/tests/``, and the CONCLUSION outlived the reason that
originally justified it. That reason was that under FULL_SUITE the ``scripts``
module config ran its ``test_command`` verbatim against ``tests/scripts/``
only, and the repo-root fleet chain likewise ended in ``pytest tests/scripts/``
— so a guard placed in ``scripts/tests/`` would never have run on merge
full-verify, vacuous in the same way as the gate it guards. BOTH of those facts
are now FALSE: task 3384 (commit 1eaaf26ab9) made the root chain's trailing
segment ``pytest tests/scripts/ scripts/tests/``, and task 3460 did the same
for this module config — which is what
``test_scripts_full_suite_pytest_covers_scripts_tests`` below exists to pin.

Keep the file here anyway. Both directories now run under FULL_SUITE, so either
home would execute — but ``tests/scripts/`` is ADDITIONALLY covered by its own
registered module config (``tests/scripts/orchestrator.yaml``), so it remains
the strictly safer home for a guard whose entire purpose is to be unable to go
unrun. A guard that depends on the very command it asserts about is one edit
away from silencing itself.

Importing ``orchestrator.config`` from this suite is established precedent —
see ``test_tests_scripts_module_config.py`` and ``test_fallback_verify_config.py``
in this same directory; the root conftest.py puts every subproject's ``src/``
on sys.path.
"""
from __future__ import annotations

import pathlib
import shlex
import tomllib
from typing import Any

import pytest
from module_budget_family import min_budget
from orchestrator.config import OrchestratorConfig, _discover_module_configs
from orchestrator.module_charter import derive_modules

from orchestrator import verify, verify_cmd, verify_plan

REPO_ROOT = pathlib.Path(__file__).parents[2]

# This worktree's own top-level orchestrator config — the repo-root fleet chain
# `_root_config` reads. `dark-factory-orchestrator.yaml` is the canonical,
# REQUIRED filename for a project's top-level config (it is what the
# dashboard's escalation-URL discovery keys on); the legacy spellings are a
# discovery fallback for unmigrated projects, not a choice this repo has.
# Anchored to REPO_ROOT rather than taken from the ambient ORCH_CONFIG_PATH for
# the reason that helper's docstring records at length.
ROOT_CONFIG_PATH = REPO_ROOT / 'dark-factory-orchestrator.yaml'

MODULE_PREFIX = 'scripts'

# The near-homograph sibling. The two module configs' test_commands were
# byte-identical until task 3460 (a fact tests/scripts/orchestrator.yaml
# documented about itself, and which was itself the coverage defect 3460
# closed), so a copy-pasted lint_command left pointing here is the realistic
# wrong fix — see assertion (5), which is the ONLY place that copy-paste is
# detectable (assertion (4) cannot see it; the reason is recorded there). The
# copy-paste risk is unchanged by 3460: the two commands still differ by a
# single added target, and the directory names remain near-homographs.
SIBLING_PREFIX = 'tests/scripts'

# A real tracked file under scripts/, used as the representative touched-file
# for the derive_modules -> for_module routing assertions below.
SAMPLE_TOUCHED_FILE = 'scripts/tests/test_census_trigger.py'

# The two DIRECTORY targets the pytest gate is about, spelled with the trailing
# slash the commands actually carry. Kept as named constants because the two
# strings are near-homographs and a transposition between them is exactly the
# defect `test_scripts_full_suite_pytest_covers_scripts_tests` guards against.
OWN_TESTS_DIR = 'scripts/tests/'
SIBLING_TESTS_DIR = 'tests/scripts/'

# A real tracked PRODUCTION module under scripts/, whose only tests live in
# scripts/tests/test_census_trigger.py. Load-bearing that this is production
# and not a test file: verify_plan._derive_module_runs arm 3 (the task-3294
# source-only floor) runs the owning module's test_command VERBATIM for a diff
# like this one, whereas a touched scripts/tests/test_*.py takes the FILE_SCOPED
# arm instead and narrows correctly even with the gap open. The FULL_SUITE
# claim is therefore only falsifiable through a production file.
SAMPLE_TOUCHED_PRODUCTION_FILE = 'scripts/legibility/census_trigger.py'

# scripts/tests/ own conftest — the CONFTEST trigger for this very directory.
SAMPLE_TOUCHED_CONFTEST = 'scripts/tests/conftest.py'

# The mechanism, restated once so each failure message can point at it. By
# SYMBOL, not file:line — see the module docstring: the line pins this string
# originally carried were all stale at HEAD and sent readers to unrelated code.
_VACUOUS_PASS = (
    'verify_plan._derive_module_runs emits a SKIPPED PlannedRun with cmd=None '
    'for a falsy lint_command or type_check_command, '
    'verify._executed_module_configs_from_plan renders that back to None, and '
    'verify._run_or_skip_timed turns a None command into a CheckRun.skipped '
    'that is VACUOUSLY PASSING at rc=0'
)


# The two extraPaths entries the declared type gate depends on, and the flat
# modules that stop resolving without them. Measured, not assumed: at the
# commit before task 3456 added these entries, the then-declared
# `npx pyright scripts/` reported exactly 9 reportMissingImports naming these
# five modules. That npx spelling is kept because it names the command that was
# ACTUALLY RUN for this measurement; task 4358 later switched the declaration to
# `uv run --project shared pyright scripts/`, which resolves the same root
# [tool.pyright] table, so the measurement carries over unchanged.
_REQUIRED_EXTRA_PATHS = ('scripts', 'scripts/legibility')
_UNRESOLVED_WITHOUT = ('census', 'codebook', 'coder', 'digest', 'inventory')


def _load_root_pyright_config() -> dict[str, Any]:
    """Return the ``[tool.pyright]`` section of the ROOT pyproject.toml, or {}.

    Same shape as ``dashboard/tests/test_pyright_config.py::_load_pyright_config``,
    pointed at REPO_ROOT instead of a package root — the root table is the one
    that governs, because the declared type gate runs from the repo root.
    """
    toml_path = REPO_ROOT / 'pyproject.toml'
    assert toml_path.is_file(), f'pyproject.toml not found at {toml_path}'
    with open(toml_path, 'rb') as fh:
        config = tomllib.load(fh)
    return config.get('tool', {}).get('pyright', {})


# The two ROOT [tool.pyright] keys that carve paths back out of a run without
# the command line changing at all: `exclude` drops the files from analysis
# entirely, `ignore` keeps analysing them but suppresses every diagnostic they
# produce. Either one naming scripts/ un-gates it while the declared
# `uv run --project shared pyright scripts/` stays byte-identical and still
# exits 0 — the same reports-green failure mode as a None command, reached a
# different way.
_CARVE_OUT_KEYS = ('exclude', 'ignore')


def _root_carve_outs_naming(segment: str) -> list[str]:
    """Root ``[tool.pyright]`` exclude/ignore entries that can reach INTO ``<segment>/``.

    Component-wise and ROOT-ANCHORED, not substring, for the same reason
    ``_targets`` tests exact-element membership: ``'scripts'`` is a substring of
    ``'tests/scripts'``, and these entries are resolved relative to the repo
    root. An entry is reported only when it can match a path under
    ``<segment>/`` — either it is rooted there (``scripts``, ``./scripts``,
    ``scripts/tests/**``) or it opens with the recursive wildcard and names
    *segment* later (``**/scripts/**``).

    Deliberately NOT reported: ``tests/scripts`` (rooted in the SIBLING tree —
    excluding it un-gates that module, not this one, and reporting it here
    would mis-diagnose which gate is affected) and ``**/node_modules`` and
    friends (pyright's own defaults, which name nothing under scripts/).
    """
    config = _load_root_pyright_config()
    found: list[str] = []
    for key in _CARVE_OUT_KEYS:
        entries = config.get(key, [])
        if isinstance(entries, str):
            entries = [entries]
        for entry in entries:
            raw = str(entry).replace('\\', '/').split('/')
            parts = [p for p in raw if p not in ('', '.')]
            if not parts:
                continue
            if parts[0] == segment or (parts[0] == '**' and segment in parts[1:]):
                found.append(f'{key}={entry!r}')
    return found


def _discovered() -> dict:
    return _discover_module_configs(REPO_ROOT)


def _executed_for_touched(files: list[str], cfg: OrchestratorConfig):
    """Run the PRODUCTION plan->execution bridge and return the single executed config.

    ``derive_verify_plan`` decides scope; ``_executed_module_configs_from_plan``
    renders those PlannedRuns into the exact ModuleConfig ``run_verification``
    executes. Asserting on THAT is what makes "ruff ran over scripts/" a
    structural claim rather than an exit-code claim.

    *cfg* IS A REQUIRED PARAMETER, not a convenience (task 3703, applying the
    shape commit 6c72a7da5a landed in ``test_module_verify_budgets.py``). It
    must be a config built by ``_root_config``, whose docstring spells out why
    the ``ORCH_CONFIG_PATH`` anchor is load-bearing: an unset anchor collapses
    every value to the pydantic defaults, SILENTLY.

    This helper used to construct its own
    ``OrchestratorConfig(project_root=REPO_ROOT)``, and in THIS file's call
    graph that made it ORDERING-DEPENDENT in a way nothing could report.
    ``test_scripts_module_carries_its_own_measured_verify_budget`` builds a
    ``_root_config(monkeypatch)`` for assertion (c) and only then reaches
    assertion (e)'s call here, so the helper read the right yaml purely as a
    SIDE EFFECT of that earlier line — while the other three callers
    (``test_scripts_diff_is_lint_gated``, ``test_scripts_diff_is_type_gated``,
    ``test_scripts_full_suite_pytest_covers_scripts_tests``) anchored nothing
    at all and were reading the ambient environment outright. Reordering (c)
    after (e) would have broken the one working case with no failure signal.
    Taking the config as an argument makes the dependency structural instead of
    ordering-dependent.

    The ``lambda _f: None`` worktree_reader keeps this hermetic: no file reads,
    and nothing classifies STRUCTURAL, so the lint/type legs stay FILE_SCOPED.
    """
    mc = _discovered()[MODULE_PREFIX]
    plan = verify_plan.derive_verify_plan(files, [mc], cfg, lambda _f: None)
    executed = verify._executed_module_configs_from_plan([mc], plan)
    assert len(executed) == 1, (
        f'expected exactly one executed module config for {files!r}, got '
        f'{[e.prefix for e in executed]!r}'
    )
    return executed[0]


def _root_config(monkeypatch: pytest.MonkeyPatch) -> OrchestratorConfig:
    """Load the repo-root config through the PRODUCTION loader, anchored at ROOT_CONFIG_PATH.

    Shared by every guard in this file that needs to compare against the
    repo-root fleet chain (task 3458's amendment pass extracted this from
    three near-identical copies — see git history for the pre-extraction
    shape).

    ANCHORING ``ORCH_CONFIG_PATH`` IS LOAD-BEARING, not hygiene, and an earlier
    draft omitted it on the false premise that ``project_root=REPO_ROOT``
    selects which yaml is read. It does not: ``project_root`` is only a model
    FIELD, and ``OrchestratorConfig.settings_customise_sources`` builds its
    ``YamlSettingsSource`` from ``os.environ['ORCH_CONFIG_PATH']`` alone,
    falling back to a CWD-relative ``config.yaml``. Both ambient states are
    wrong here, in opposite directions:

      * UNSET — which is the state INSIDE VERIFY, because
        ``verify._target_subprocess_env`` deliberately scrubs the whole
        ``ORCH_`` prefix (task 2957) — finds no file, so every value collapses
        to the pydantic defaults, where e.g. ``test_command`` is the bare
        literal ``'pytest'``. A caller would then fail with a message about
        the fleet chain having dropped a suite, when the chain is in fact
        correct and was simply never read.
      * SET, as an operator's shell has it, points at whichever checkout that
        orchestrator serves — typically the MAIN one, not this worktree. A
        caller would then assert about a different checkout's yaml and report
        GREEN on a worktree that had actually regressed: the exact
        reports-green-while-checking-something-else failure this file exists
        to prevent, one env var over.

    Setting the env var IS the production load path (``config.load_config``
    stamps ``os.environ['ORCH_CONFIG_PATH']`` before constructing), so this
    stays a read through the real loader — pinned to THIS worktree's committed
    yaml rather than left to the ambient environment. Same remedy, same
    reason, as ``tests/scripts/test_orchestrator_watchdog.py``'s
    ``test_orch_restart_min_interval_secs_matches_config_default``.

    Fails LOUDLY on a missing file rather than silently: ``YamlSettingsSource``
    skips a non-existent ``config_path`` instead of raising, so a bad path
    would silently yield the pydantic DEFAULTS — a config this repo does not
    declare — rather than an error.
    """
    assert ROOT_CONFIG_PATH.is_file(), (
        f'{ROOT_CONFIG_PATH} does not exist, so anchoring ORCH_CONFIG_PATH at '
        'it would silently load the pydantic DEFAULTS instead (YamlSettingsSource '
        'skips a non-existent path rather than raising), and every value read '
        'from the returned config would be about a config this repo does not '
        'declare. dark-factory-orchestrator.yaml is the canonical, required '
        "filename for a project's top-level orchestrator config"
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(ROOT_CONFIG_PATH))
    return OrchestratorConfig(project_root=REPO_ROOT)


# The two checker spellings these helpers understand, keyed by the phrase that
# identifies the invoking segment. The ANCHOR — the last whitespace-separated
# token of the keyword — is the token after which positional targets begin, and
# it is what makes one implementation serve both CLIs: ruff's is
# `ruff check <targets>`, so the anchor is its `check` SUBCOMMAND; pyright's is
# `pyright <targets>` with no subcommand at all, so the anchor is the program
# name itself. Nothing else about the two invocations differs for these
# purposes.
_RUFF = 'ruff check'
_PYRIGHT = 'pyright'

# pytest's invocation is `pytest <targets>` with no subcommand, so — like
# pyright's and unlike ruff's — the anchor is the program name itself. Note the
# anchor placement is what excludes the pre-anchor positional `shared` of
# `uv run --project shared pytest ...` from the target list.
#
# Deliberately absent from _NARROWING_FLAGS below: that table is consulted only
# by _narrowing_flag_args, which is never called for pytest. pytest narrows a
# directory target through -k/-m/--deselect/--ignore rather than through the
# exclude spellings ruff and pyright use, and listing a set nothing checks
# would read as coverage while checking nothing — the same objection the
# _NARROWING_FLAGS comment already records against copying ruff's flags to
# pyright. The pytest gate's real failure mode is a MISSING target, which
# assertions (2)-(5) of test_scripts_full_suite_pytest_covers_scripts_tests
# test directly.
_PYTEST = 'pytest'

# Flag PREFIXES that narrow what a directory-wide target actually gets checked,
# per checker. Prefix-matched, so each entry covers both the `--flag value` and
# the `--flag=value` spelling.
#
# ruff's three exclude spellings are real. PYRIGHT'S SET IS NOT THE SAME, and an
# earlier draft of this table simply copied ruff's across. Measured against
# `pyright --help` (v1.1.408): pyright has NO `--exclude` and NO `--ignore` —
# those two are pyproject `[tool.pyright]` KEYS, not CLI flags, and that vector
# is checked where it actually lives, by `_root_carve_outs_naming` in assertion
# (d) below. `--ignoreexternal` does exist but applies only to `--verifytypes`,
# so it cannot narrow a normal run. Listing flags a CLI does not have is not
# free defence: it reads as coverage while leaving the real vectors unchecked.
#
# The two spellings that genuinely narrow a `pyright <dir>` run:
#   --skip*       `--skipunannotated` drops every unannotated function from
#                 analysis. Prefix-matched so a future `--skip<x>` is caught.
#   -p/--project  points pyright at a DIFFERENT config file, which can relax
#                 typeCheckingMode, add excludes, or drop extraPaths wholesale.
#                 Invisible to assertion (c): `_targets` discards every
#                 `-`-prefixed token, so `pyright -p /tmp/lax.json scripts/`
#                 still lists 'scripts/' among its targets and satisfies (c)
#                 while checking almost nothing. Both spellings are listed
#                 because neither is a prefix of the other, and neither is a
#                 prefix of pyright's `--python*` flags (those begin `--p`).
#
# `-p`/`--project` MUST STAY IN THIS SET, and the reason is worth stating
# because task 4358 hit the collision that makes deleting it tempting. `uv run
# --project <member> pyright <dir>` — the invocation shape every module config
# in this repo now uses — contains the SAME CHARACTERS as pyright's own
# config-redirect flag, naming a completely different thing: uv's selects the
# ENVIRONMENT the checker binary is resolved from and narrows NOTHING, while
# pyright's selects a CONFIG FILE that can relax typeCheckingMode wholesale.
#
# The two are told apart by POSITION, not by dropping the flag from this table:
# uv's flags precede the checker anchor, pyright's follow it. That is why
# `_narrowing_flag_args` scans `_post_anchor_tokens` rather than the whole
# segment. Dropping `--project` here would silence the same false positive and
# simultaneously un-guard the sharpest carve-out vector this file documents —
# see test_narrowing_flag_detection_is_scoped_to_the_checkers_own_arguments,
# whose assertion (b) exists specifically to fail on that shortcut.
#
# The property is not new: `_targets` has always sliced after the anchor, and
# the `_PYTEST` comment above already records it ("the anchor placement is what
# excludes the pre-anchor positional `shared` of `uv run --project shared
# pytest ...`"). `_narrowing_flag_args` was simply the one helper it had never
# been applied to — latent because ruff's three exclude spellings below happen
# not to collide with any `uv run` flag, so the already-uv-fronted
# lint_command never tripped it, and the type_check_command was `npx`-fronted
# until 4358.
_NARROWING_FLAGS = {
    _RUFF: ('--exclude', '--extend-exclude', '--force-exclude'),
    _PYRIGHT: ('--skip', '-p', '--project'),
}


def _segment(cmd: str, keyword: str) -> str:
    """The ``&&``-chained segment of *cmd* that actually invokes *keyword*.

    Reuses the production splitter (``verify_cmd.split_top_level_and``, which
    is quote-aware) rather than a naive ``str.split('&&')``.

    Chaining is an ESTABLISHED pattern here, not a hypothetical:
    ``verify_plan._scope_prefix_to_keyword``'s own docstring records that
    "every subproject's lint_command chains a ``python3 .../check_*.py <dir>``
    gate after ``ruff check``". Extracting the checker's own segment first is
    what lets the target assertions below stay true if this module later adopts
    that shape — tokenising the whole chain would otherwise read ``&&``,
    ``python3`` and the checker's own arguments as lint/type targets.
    """
    segments = verify_cmd.split_top_level_and(cmd)
    matching = [s for s in segments if keyword in s]
    assert len(matching) == 1, (
        f'expected exactly one `{keyword}` segment in {cmd!r}, got {matching!r}'
    )
    return matching[0]


def _anchor_split(cmd: str, keyword: str) -> tuple[list[str], list[str]]:
    """*keyword*'s segment of *cmd*, split at the checker anchor into (pre, post).

    The ANCHOR is the last whitespace-separated token of *keyword* (see the
    ``_RUFF``/``_PYRIGHT`` comment above) and belongs to neither half. Sole
    implementation of the anchor location and the anchor-presence assertion, so
    every caller that cares about position shares one notion of where the
    wrapper stops and the checker starts.
    """
    anchor = keyword.split()[-1]
    tokens = shlex.split(_segment(cmd, keyword))
    assert anchor in tokens, (
        f'no `{anchor}` token in the `{keyword}` segment of {cmd!r}, so the '
        "checker's own arguments cannot be located"
    )
    at = tokens.index(anchor)
    return tokens[:at], tokens[at + 1:]


def _pre_anchor_tokens(cmd: str, keyword: str) -> list[str]:
    """The WRAPPER's tokens in *cmd* — everything before the checker anchor.

    The mirror of ``_post_anchor_tokens``, for the assertions that are about
    how the checker is RESOLVED rather than what it checks: in
    ``uv run --project shared pyright scripts/`` these four tokens are uv's, and
    they decide which member environment supplies the ``pyright`` binary.

    Shares ``_anchor_split`` with its mirror deliberately. A hand-rolled
    ``tokens.index(_PYRIGHT)`` at the call site is a third copy of the same
    logic — the drift this file already paid for once (task 4358:
    ``_narrowing_flag_args`` had missed the slice ``_targets`` always had) — and
    it raises a bare ``ValueError`` instead of the diagnostic above when the
    anchor is absent.
    """
    return _anchor_split(cmd, keyword)[0]


def _post_anchor_tokens(cmd: str, keyword: str) -> list[str]:
    """*keyword*'s OWN arguments in *cmd* — the tokens after the checker anchor.

    Everything before the anchor belongs to the WRAPPER, not the checker: in
    ``uv run --project shared pyright scripts/`` the pre-anchor tokens are uv's,
    and reading them as the checker's is a category error.

    Both callers need exactly this slice, for the same reason from opposite
    directions — ``_targets`` must not count uv's positional ``shared`` as a
    checked path, and ``_narrowing_flag_args`` must not count uv's
    ``--project`` as pyright's config redirect. Sharing one implementation is
    what keeps the two from drifting apart again (task 4358: ``_targets`` had
    the slice from the start, ``_narrowing_flag_args`` never got it).
    """
    return _anchor_split(cmd, keyword)[1]


def _targets(cmd: str, keyword: str) -> list[str]:
    """The positional path arguments *keyword*'s segment of *cmd* checks.

    Substring checks alone cannot carry the anti-copy-paste assertions:
    ``'scripts/'`` is a substring of ``'tests/scripts/'``, so a copy-pasted
    sibling command would satisfy a naive ``'scripts/' in cmd`` for BOTH the
    lint and the type command. Splitting out the actual targets and testing
    LIST MEMBERSHIP (exact-element, so ``'tests/scripts/'`` does not match) is
    what makes those assertions real.
    """
    return [t for t in _post_anchor_tokens(cmd, keyword) if not t.startswith('-')]


def _narrowing_flag_args(cmd: str, keyword: str) -> list[str]:
    """Any flag in *keyword*'s segment that carves files back out of the target.

    Both spellings are caught: ``--exclude foo`` and ``--exclude=foo``.

    Scoped to the checker's OWN arguments via ``_post_anchor_tokens``, not the
    whole segment: uv's pre-anchor ``--project <member>`` names an ENVIRONMENT,
    while pyright's post-anchor ``--project <file>`` names a CONFIG FILE, and
    only position distinguishes them. See the ``_NARROWING_FLAGS`` comment for
    why the flag stays in the table rather than being deleted to dodge that
    collision, and
    ``test_narrowing_flag_detection_is_scoped_to_the_checkers_own_arguments``
    for the four cases that pin it.
    """
    prefixes = _NARROWING_FLAGS[keyword]
    return [t for t in _post_anchor_tokens(cmd, keyword) if t.startswith(prefixes)]


def _uv_project_member(cmd: str, keyword: str) -> str | None:
    """The member named by uv's PRE-anchor ``--project`` in *keyword*'s segment.

    Both spellings, matching ``_narrowing_flag_args``' documented both-spellings
    behaviour: ``--project shared`` and ``--project=shared`` are the same uv
    invocation, and a reader that recognises only one of them rejects a correct
    command with a message claiming it selects no environment at all.

    ``None`` means NO USABLE PRE-ANCHOR SELECTOR, which covers three cases the
    caller reports identically because uv fails them identically:

      * no ``--project`` before the anchor at all;
      * a POST-anchor ``--project``, which is pyright's CONFIG-FILE redirect and
        a different flag entirely (see ``_NARROWING_FLAGS``) — that one is
        ``_narrowing_flag_args``' business, not this helper's;
      * a dangling ``--project`` with the anchor as its next token, e.g.
        ``uv run --project pyright scripts/``, where uv would consume
        ``pyright`` as the project name and never invoke the checker.
    """
    pre = _pre_anchor_tokens(cmd, keyword)
    for i, token in enumerate(pre):
        if token == '--project':
            return pre[i + 1] if i + 1 < len(pre) else None
        if token.startswith('--project='):
            return token.split('=', 1)[1]
    return None


# Thin ruff-spelling wrappers, kept so test_scripts_diff_is_lint_gated below is
# untouched by the task-3456 generalization above.
def _ruff_segment(cmd: str) -> str:
    return _segment(cmd, _RUFF)


def _ruff_targets(cmd: str) -> list[str]:
    return _targets(cmd, _RUFF)


def _ruff_exclude_flags(cmd: str) -> list[str]:
    return _narrowing_flag_args(cmd, _RUFF)


def _pytest_targets(cmd: str) -> list[str]:
    """The directories/files *cmd*'s pytest segment actually collects.

    Same thin-wrapper shape as ``_ruff_targets``, but NOT for the same reason,
    and an earlier draft of this docstring copied ``_targets``' justification
    across without re-deriving it. ``_targets`` is defending against a
    SUBSTRING relation between its two targets — ``'scripts/'`` really is a
    substring of ``'tests/scripts/'``, which is the LINT/TYPE case, where the
    target is the bare ``scripts/``. The two PYTEST targets here are
    ``'scripts/tests/'`` and ``'tests/scripts/'``, and NEITHER is a substring
    of the other, so that particular confusion cannot arise on this pair.

    Exact-element membership is still load-bearing, for a different and real
    reason: a substring check cannot tell a DIRECTORY target apart from a file
    or a flag that merely MENTIONS the same path. Both
    ``pytest scripts/tests/test_census_trigger.py`` and
    ``pytest tests/scripts/ --ignore=scripts/tests/`` satisfy
    ``'scripts/tests/' in cmd`` while collecting something other than that
    directory — the second one collects the exact opposite of what the
    substring appears to prove. Splitting out the positional targets is what
    makes "the directory is collected" a claim about what pytest will actually
    do rather than about which characters occur in the string.
    """
    return _targets(cmd, _PYTEST)


def _dir_key(target: str) -> str:
    """*target* with any trailing slash removed, for directory comparison.

    The exact-element property ``_pytest_targets`` provides is kept; only the
    trailing-slash COUPLING is relaxed. ``'scripts/tests/'`` and
    ``'scripts/tests'`` name the same directory to pytest, and the production
    deriver is not committed to either spelling: ``verify_plan``'s
    ``_fallback_pytest_targets`` already maps a touched conftest to its PARENT
    DIRECTORY, which yields the slashless form. If the module path ever adopts
    that same (strictly better-scoped) shape, a literal ``'scripts/tests/'``
    comparison would fail with a message accusing the author of leaving the
    directory ungated at the moment coverage actually improved — the mirror of
    the false-positive the lint test designs around when it chooses membership
    over list equality.
    """
    return target.rstrip('/')


def _dir_keys(targets: list[str]) -> list[str]:
    """``_dir_key`` over a target list, order and multiplicity preserved."""
    return [_dir_key(t) for t in targets]


def test_narrowing_flag_detection_is_scoped_to_the_checkers_own_arguments() -> None:
    """``_narrowing_flag_args`` must read only the tokens AFTER the checker anchor.

    Task 4358. ``uv run --project <member> pyright <dir>`` and
    ``pyright --project <file> <dir>`` contain the SAME CHARACTERS naming two
    unrelated things, and only POSITION tells them apart:

      * ``uv``'s ``--project`` (PRE-anchor) selects the ENVIRONMENT the checker
        is resolved from — which member venv supplies the ``pyright`` binary. It
        narrows NOTHING about what gets checked.
      * pyright's own ``-p``/``--project`` (POST-anchor) points pyright at a
        DIFFERENT CONFIG FILE, which can relax ``typeCheckingMode``, add
        excludes, or drop ``extraPaths`` wholesale. That is the sharpest
        carve-out vector this file guards, and assertion (c) of
        ``test_scripts_diff_is_type_gated`` provably cannot see it — ``_targets``
        discards every ``-``-prefixed token, so ``pyright -p /tmp/lax.json
        scripts/`` still lists ``'scripts/'`` among its targets.

    THE OBVIOUS "FIX" FOR CASE (a) IS THE WRONG ONE, which is why case (b)
    exists. Deleting ``-p``/``--project`` from ``_NARROWING_FLAGS[_PYRIGHT]``
    also makes (a) pass, and simultaneously un-guards the real config-redirect
    vector — trading a false positive for a silent hole. Discriminate by
    position, not by dropping the flag.

    The property is not new to this file: ``_targets`` has always sliced
    ``tokens[tokens.index(anchor) + 1:]``, and the ``_PYTEST`` comment above
    records that "the anchor placement is what excludes the pre-anchor
    positional ``shared`` of ``uv run --project shared pytest ...``".
    ``_narrowing_flag_args`` is the one helper that never received it, latent
    only because ruff's three exclude spellings happen not to collide with any
    ``uv run`` flag — so the already-uv-fronted ``lint_command`` never tripped
    it.
    """
    # (a) uv's environment selector is NOT a pyright narrowing flag. RED before
    # task 4358: whole-segment tokenisation returns ['--project'] here.
    assert _narrowing_flag_args('uv run --project shared pyright scripts/', _PYRIGHT) == [], (
        "uv's PRE-anchor `--project shared` selects the ENVIRONMENT pyright is "
        'resolved from, not a pyright config file, so it narrows nothing about '
        'what gets checked. Reading it as a narrowing flag makes '
        'test_scripts_diff_is_type_gated assertion (d) fail with a message '
        'accusing the author of un-gating the type check, on a command that '
        'gates it fully — a false positive on two homographic flags in '
        'different segments of one command line. Scope the scan to the tokens '
        'AFTER the checker anchor, as _targets already does; do NOT fix this by '
        'removing --project from _NARROWING_FLAGS (see case (b))'
    )

    # (b) A REAL post-anchor redirect is still caught, so (a) does not buy its
    # pass by disabling the check.
    assert _narrowing_flag_args(
        'uv run --project shared pyright -p /tmp/lax.json scripts/', _PYRIGHT
    ) == ['-p'], (
        "pyright's OWN post-anchor -p/--project points it at a different config "
        'file, which can relax typeCheckingMode, add excludes or drop '
        'extraPaths wholesale while the declared command still names scripts/ '
        'and still exits 0. This assertion is what stops case (a) being '
        '"fixed" by deleting -p/--project from _NARROWING_FLAGS[_PYRIGHT]: that '
        'edit would silence the false positive AND leave the sharpest carve-out '
        'vector in this file unguarded'
    )

    # (c) The other genuine pyright narrowing spelling, also post-anchor.
    assert _narrowing_flag_args(
        'uv run --project shared pyright --skipunannotated scripts/', _PYRIGHT
    ) == ['--skipunannotated'], (
        '--skipunannotated drops every unannotated function from analysis, so a '
        'directory-wide target checks far less than it appears to. It is '
        'prefix-matched via --skip so a future --skip<x> is caught too; '
        'anchor-scoping must not cost that'
    )

    # (d) ruff shares this helper through _ruff_exclude_flags and must not be
    # regressed by the fix — its spellings are all post-anchor too.
    assert _ruff_exclude_flags(
        'uv run --project shared ruff check scripts/ --exclude foo'
    ) == ['--exclude'], (
        "ruff's three exclude spellings never collided with a `uv run` flag, "
        'which is why this helper survived un-scoped for so long. '
        'Anchor-scoping is a strict improvement for ruff rather than a '
        'trade-off, and this assertion pins that it stays one'
    )
    assert _ruff_exclude_flags('uv run --project shared ruff check scripts/') == [], (
        "the repo's actual scripts/ lint_command carries no exclude flag, and "
        "uv's pre-anchor --project must not be mistaken for one here either"
    )


def test_uv_project_member_reads_both_spellings_and_only_pre_anchor() -> None:
    """``_uv_project_member`` must accept ``--project=X`` as well as ``--project X``.

    Task 4358 amendment. ``uv run --project=shared pyright scripts/`` is an
    exactly equivalent uv invocation to the space-separated spelling, so a
    reader that recognises only one of them fails a command that does precisely
    what the contract asks — a FALSE POSITIVE whose message would accuse the
    author of not selecting an environment. The sibling helper
    ``_narrowing_flag_args`` already prefix-matches for this same reason ("Both
    spellings are caught: ``--exclude foo`` and ``--exclude=foo``"); this pins
    that ``test_type_gates_resolve_pyright_without_npx`` follows the convention
    its own file established rather than re-deriving a stricter one.

    Cases (c)-(e) carry the POSITION half, which is the same homograph
    ``test_narrowing_flag_detection_is_scoped_to_the_checkers_own_arguments``
    guards from the other side: this helper must see ONLY uv's pre-anchor
    environment selector, ``_narrowing_flag_args`` ONLY pyright's post-anchor
    config redirect. Case (e) is the mixed command that carries both at once —
    neither helper alone can judge it, and each must read exactly its own half.
    """
    # (a) The spelling the repo actually declares.
    assert _uv_project_member('uv run --project shared pyright scripts/', _PYRIGHT) == 'shared'

    # (b) The equals spelling. RED before this amendment: the caller scanned for
    # an exact `--project` token, so this equivalent command was reported as
    # carrying no environment selector at all.
    assert _uv_project_member('uv run --project=shared pyright scripts/', _PYRIGHT) == 'shared', (
        '`--project=shared` selects the same environment as `--project shared`; '
        'reading only the space-separated spelling turns an equivalent, correct '
        'command into a false red'
    )

    # (c) No selector at all is None, not a crash — the caller turns that into
    # its own diagnostic rather than an IndexError.
    assert _uv_project_member('uv run pyright scripts/', _PYRIGHT) is None

    # (d) A POST-anchor --project is pyright's config-file redirect, NOT a uv
    # environment selector, so this helper must not report one. The command is
    # caught instead by _narrowing_flag_args.
    redirect = 'uv run pyright --project /tmp/lax.json scripts/'
    assert _uv_project_member(redirect, _PYRIGHT) is None, (
        "pyright's own post-anchor --project names a CONFIG FILE, not a uv "
        'workspace member; reporting it as the environment selector would then '
        'check `/tmp/lax.json` for membership in [tool.uv.workspace].members '
        'and fail with a message about the wrong flag entirely'
    )
    assert _narrowing_flag_args(redirect, _PYRIGHT) == ['--project']

    # (e) BOTH at once, each read by exactly one helper. This is the command
    # that passes every assertion of test_type_gates_resolve_pyright_without_npx
    # except the post-anchor-redirect one, which is why that assertion has to be
    # a real check rather than a restatement of the pre-anchor one.
    mixed = 'uv run --project=shared pyright --project /tmp/lax.json scripts/'
    assert _uv_project_member(mixed, _PYRIGHT) == 'shared'
    assert _narrowing_flag_args(mixed, _PYRIGHT) == ['--project']

    # (f) The pre/post split itself, shared with _post_anchor_tokens.
    assert _pre_anchor_tokens('uv run --project shared pyright scripts/', _PYRIGHT) == [
        'uv',
        'run',
        '--project',
        'shared',
    ]


def test_executed_for_touched_is_hermetic_against_the_ambient_orch_config_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_executed_for_touched`` must not read the ambient ``ORCH_CONFIG_PATH``.

    Task 3703, reviewer-flagged — the mirror of the repair commit 6c72a7da5a
    landed in ``test_module_verify_budgets.py``, and of the same repair in
    ``test_tests_scripts_module_config.py``. This helper used to construct its
    own ``OrchestratorConfig(project_root=REPO_ROOT)``, and ``project_root``
    selects NOTHING — see ``_root_config``'s docstring, which spells out at
    length that ``settings_customise_sources`` builds its ``YamlSettingsSource``
    from ``os.environ['ORCH_CONFIG_PATH']`` alone.

    THE ORDERING HAZARD THIS FILE CARRIES IS CONCRETE, and sharper than the
    sibling's. ``test_scripts_module_carries_its_own_measured_verify_budget``
    calls ``_root_config(monkeypatch)`` for assertion (c) and then
    ``_executed_for_touched(...)`` for assertion (e), so that helper read the
    right yaml ONLY as a SIDE EFFECT of an assertion made earlier in the same
    test body. Two edits nobody would think twice about silently broke it:
    reordering (c) after (e), or calling the helper from any of the three tests
    that anchor nothing at all —
    ``test_scripts_diff_is_lint_gated``, ``test_scripts_diff_is_type_gated``,
    ``test_scripts_full_suite_pytest_covers_scripts_tests``. Either hands it a
    defaults-collapsed config (inside verify, where
    ``verify._target_subprocess_env`` scrubs the whole ``ORCH_`` prefix) or a
    FOREIGN CHECKOUT's config (in an operator shell, where the variable points
    at whichever checkout that orchestrator serves) — with no failure signal in
    either direction. Taking the config as an argument makes the dependency
    structural instead of ordering-dependent.

    WHY THE HOSTILE YAML IS ONE THE LOADER REJECTS, rather than one that merely
    holds different values. This is the only formulation that can fail BEFORE
    the fix. MEASURED at base d6a5e32535: ``derive_verify_plan`` consults its
    ``config`` argument in exactly two places — ``_merge_breadth_is_full``
    (role='merge') and ``_derive_fallback_runs`` (``module_configs`` EMPTY) —
    and this call is role='task' with one non-empty ModuleConfig, so it reaches
    neither. A differently-VALUED config therefore yields a byte-identical
    plan, and a test built on one would read as enforcement while enforcing
    nothing. A type-invalid value instead makes the PRE-fix helper's own
    ``OrchestratorConfig`` construction raise pydantic ``ValidationError`` — the
    single observable signal that the ambient dependency exists.

    The post-fix GREEN is reachable rather than assumed, measured at the same
    base: ``_discover_module_configs``, ``derive_verify_plan`` and
    ``verify._executed_module_configs_from_plan`` are unaffected by the
    poisoned env.

    ORDER IS LOAD-BEARING: the anchored config is built FIRST, while the
    environment is still sane, and the poison applied SECOND. Anchoring
    afterwards would overwrite the poison and leave this test vacuous.
    """
    # (1) The anchored config, built while the environment is still sane.
    cfg = _root_config(monkeypatch)

    # (2) NOW poison the ambient environment, with a config the PRODUCTION
    # loader REJECTS — see the docstring for why rejected and not merely
    # different.
    hostile = tmp_path / 'hostile.yaml'
    hostile.write_text(
        'verify_command_timeout_secs: "not-a-number"\n', encoding='utf-8'
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(hostile))

    # (3) The helper must not consult that variable. A helper that builds its
    # own config raises pydantic ValidationError here instead of returning.
    executed = _executed_for_touched([SAMPLE_TOUCHED_FILE], cfg)

    assert executed.prefix == MODULE_PREFIX, (
        f'under a poisoned ORCH_CONFIG_PATH the production bridge executed '
        f'{executed.prefix!r}, not {MODULE_PREFIX!r} (task 3703) — module '
        'routing is reading the ambient environment'
    )

    # The three legs, each asserted non-None and TARGETING the file that
    # changed, for the reason the lint/type gate tests below exist: a None
    # command is not a deferral, it is DELETED gating that reports green (see
    # _VACUOUS_PASS, quoted into each message below).
    #
    # SCOPED ON SAMPLE_TOUCHED_FILE, NOT ON MODULE_PREFIX (task 3703 amendment
    # pass, reviewer-flagged). MODULE_PREFIX here is the bare string 'scripts',
    # which is a SUBSTRING of the near-homograph sibling tree SIBLING_TESTS_DIR
    # — so `MODULE_PREFIX in cmd` is satisfied by a command targeting ONLY
    # tests/scripts/ and cannot tell the two trees apart, in the one file where
    # that confusion is the documented realistic defect (see SIBLING_PREFIX).
    # These three assertions therefore added less than they appeared to. The
    # touched file is rooted at OWN_TESTS_DIR and discriminates outright; it is
    # also the exact token the sibling gate tests below pin, because all three
    # legs render FILE_SCOPED for this diff (measured through the production
    # bridge: `pytest|ruff check|pyright scripts/tests/test_census_trigger.py`).
    #
    # What this does NOT buy is copy-paste detection, and it does not claim to:
    # under a FILE_SCOPED render _scope_prefix_to_keyword REPLACES the declared
    # targets with the touched-file list, so a config carrying the copy-pasted
    # tests/scripts/ targets renders byte-identically — the reason assertion (4)
    # of test_scripts_diff_is_lint_gated records at length, and why that claim
    # lives on the DECLARED value in its assertion (5). The load-bearing
    # discriminators for THIS test are executed.prefix above and the module
    # budget below, neither of which the sibling config could satisfy.
    assert executed.test_command is not None and 'pytest' in executed.test_command, (
        f'executed test_command is {executed.test_command!r} under a poisoned '
        f'ORCH_CONFIG_PATH (task 3703); {MODULE_PREFIX} must run its own suite '
        f'regardless of what the ambient environment points at: {_VACUOUS_PASS}'
    )
    assert SAMPLE_TOUCHED_FILE in executed.test_command, (
        f'executed test_command {executed.test_command!r} does not target the '
        f'touched file {SAMPLE_TOUCHED_FILE!r} under a poisoned '
        f'ORCH_CONFIG_PATH (task 3703). Asserted on the file rather than on a '
        f'bare {MODULE_PREFIX!r} substring, which a command targeting only '
        f'{SIBLING_TESTS_DIR!r} would also satisfy'
    )
    assert executed.lint_command is not None and 'ruff' in executed.lint_command, (
        f'executed lint_command is {executed.lint_command!r} under a poisoned '
        f'ORCH_CONFIG_PATH (task 3703): {_VACUOUS_PASS}'
    )
    assert SAMPLE_TOUCHED_FILE in executed.lint_command, (
        f'executed lint_command {executed.lint_command!r} does not target the '
        f'touched file {SAMPLE_TOUCHED_FILE!r} under a poisoned '
        f'ORCH_CONFIG_PATH (task 3703). Asserted on the file rather than on a '
        f'bare {MODULE_PREFIX!r} substring, which a command targeting only '
        f'{SIBLING_TESTS_DIR!r} would also satisfy'
    )
    assert (
        executed.type_check_command is not None
        and 'pyright' in executed.type_check_command
    ), (
        f'executed type_check_command is {executed.type_check_command!r} under '
        f'a poisoned ORCH_CONFIG_PATH (task 3703): {_VACUOUS_PASS}'
    )
    assert SAMPLE_TOUCHED_FILE in executed.type_check_command, (
        f'executed type_check_command {executed.type_check_command!r} does not '
        f'target the touched file {SAMPLE_TOUCHED_FILE!r} under a poisoned '
        f'ORCH_CONFIG_PATH (task 3703). Asserted on the file rather than on a '
        f'bare {MODULE_PREFIX!r} substring, which a command targeting only '
        f'{SIBLING_TESTS_DIR!r} would also satisfy'
    )

    # The module budget survives too: the figure must come from THIS module's
    # yaml, which the poisoned env cannot reach, not from whatever the ambient
    # config declares.
    declared = _discovered()[MODULE_PREFIX].verify_command_timeout_secs
    assert executed.verify_command_timeout_secs == declared, (
        f'executed verify_command_timeout_secs='
        f'{executed.verify_command_timeout_secs} under a poisoned '
        f'ORCH_CONFIG_PATH, not the {declared} this module declares (task '
        '3703) — the budget is being resolved from the ambient environment'
    )


def test_scripts_diff_is_lint_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A diff confined to scripts/ must actually run ruff over scripts/.

    Five assertions, one contract. (1) and (2) are routing PRECONDITIONS: they
    are asserted so that a future routing regression cannot quietly make the
    lint assertion vacuous — a config that is discovered but unreachable, or
    reachable but resolving elsewhere, would let (3) pass while nothing is
    gated in production.

    NOTE on (2) — written in the lock_depth-AGNOSTIC form, never pinning a
    literal such as ``derive_modules(...) == ['scripts']``. Three layers stack
    here and they do not agree: the pydantic Field default, the
    package-bundled ``orchestrator/src/orchestrator/defaults.yaml`` layered
    over it on every load, and THIS project's ``dark-factory-orchestrator
    .yaml`` overriding both — the last of which is what the autouse config
    binding resolves. The conclusion does not turn on which value that is:
    what matters is that each derived key RESOLVES back to this config. Task
    3350's sibling guard hit this exact trap and documented it; pinning the
    literal would re-encode a falsified constant.

        CORRECTED IN PLACE (task 3866): this note used to assert "the
        EFFECTIVE value is 4" and reason "at depth 4", which stopped being
        true when this project moved lock_depth 4 -> 12. Recorded rather than
        silently rewritten, because the note exists precisely to document a
        falsified constant — and for the same reason the repair states the
        invariant instead of naming 12, which would only queue up the next
        correction. Read the number from ``cfg.lock_depth`` if you ever need
        it; do not assert it in prose.

        CORRECTED AGAIN (task 3866, review repair): that first repair did
        assert one in prose. Replacing the constant, it claimed
        ``SAMPLE_TOUCHED_FILE`` is "below the truncation threshold at every
        layer, so ``derive_modules(...)`` returns the full path whole
        regardless" — false at the smallest of the three layers this note
        enumerates. MEASURED:
        ``normalize_lock('scripts/tests/test_census_trigger.py', 2)`` ->
        ``'scripts/tests'``, truncated, not whole; depths 3 / 4 / 12 return
        it whole. A 3-component path survives only at ``lock_depth >= 3``,
        and the pydantic Field default is 2. Deleted rather than re-scoped to
        ``>= 3``: a resolved lock key is itself a depth assertion in prose,
        so the paragraph above already forbade it, and the ``>= 3`` form
        would still be hostage to ``SAMPLE_TOUCHED_FILE`` staying 3
        components. The load-bearing sentence — each derived key must RESOLVE
        back to this config, which is what the assertions check — is
        depth-invariant and stands unchanged.
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

    executed = _executed_for_touched([SAMPLE_TOUCHED_FILE], _root_config(monkeypatch))

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
        f'other and its own files are gated by nothing. The two test_commands '
        f'are held distinct by their own assertions — see '
        'test_scripts_full_suite_pytest_covers_scripts_tests, whose TARGET '
        'assertions (5) and (6) carry that claim: this config must collect '
        'BOTH test trees while the sibling collects only its own. (Its '
        'byte-inequality check is a redundant restatement of those two, kept '
        'for the diagnostic, so read the target assertions for the actual '
        'contract.) Nothing about this repo licenses duplicating a command '
        'across these two configs'
    )


def test_root_pyright_extrapaths_resolves_scripts_imports() -> None:
    """The ROOT ``[tool.pyright] extraPaths`` must carry scripts/ and scripts/legibility.

    A PRECONDITION for the type gate, not a general pyright-config preference,
    which is why it lives beside the gate it protects rather than in its own
    file. The declared ``type_check_command`` is
    ``uv run --project shared pyright scripts/`` (task 4358; it was
    ``npx pyright scripts/`` when this test was written): it carries no
    ``--directory``, so it runs from the repo root either way and resolves
    against the ROOT ``[tool.pyright]`` table — NOT against any per-package
    pyproject.toml, and NOT against ``shared/``'s, since ``--project shared``
    selects only the environment pyright is resolved from.

    ``scripts/tests/conftest.py`` inserts BOTH ``scripts/`` and
    ``scripts/legibility/`` onto sys.path at runtime, so the test modules
    import flat names (``import census``, ``import digest``) that only resolve
    for pyright if the same two directories are on extraPaths. Measured at the
    commit before these entries were added, with the command declared AT THAT
    TIME (``npx pyright scripts/`` — named as run, not retro-fitted to the
    current uv spelling): exactly 9 reportMissingImports naming
    census/codebook/coder/digest/inventory. The gate would then be RED for
    reasons unrelated to any diff —
    an unresolved import is not a finding about the change under review, and a
    permanently-red gate gets suppressed or ignored, which is how a gate dies.

    Both entries are needed and neither implies the other: ``scripts`` alone
    makes ``scripts/legibility/`` importable only as a namespace package
    (``import legibility``), not its contents as bare top-level names — a
    distinction ``scripts/tests/conftest.py`` records about itself, and the
    reason it performs two separate sys.path insertions.

    TWO module gates depend on these entries, not one. The ``tests/scripts``
    module config next door declares its own
    ``uv run --project shared pyright tests/scripts/`` (task 3842), which also
    runs from the repo root against this same root table, and five
    of its modules import flat ``scripts/`` names. Those imports used to carry
    ``# pyright: ignore[reportMissingImports]``; task 3456 dropped the pragmas
    precisely BECAUSE the extraPaths entries made them resolve, so the masking
    is gone and the dependency is now live. RE-MEASURED at the branch tip by
    deleting both entries and re-running each command — under the npx-fronted
    declarations both configs carried at that time, which changes which pyright
    BINARY ran but not which config table it read: ``pyright scripts/`` ->
    23 errors, 9 of them reportMissingImports; ``pyright tests/scripts/`` -> 8
    errors, ALL reportMissingImports, naming migrate_metadata_modules_to_files,
    drain_check, audit_wiped_metadata_files, repair_wiped_metadata_files,
    reviewer_redundancy_diagnostic and trial_module_tagger_haiku. Removing an
    entry is therefore a two-gate outage; this test failing alone would
    under-report it.

    MEMBERSHIP, never list equality or a length pin: adding a future entry is
    a legitimate change, not a regression, and an equality assertion would
    reject it with a message accusing the author of removing these two.
    """
    pyright_config = _load_root_pyright_config()
    extra_paths = pyright_config.get('extraPaths', [])

    for required in _REQUIRED_EXTRA_PATHS:
        assert required in extra_paths, (
            f'{required!r} missing from ROOT [tool.pyright] extraPaths = '
            f'{extra_paths!r} in {REPO_ROOT / "pyproject.toml"} (task 3456). '
            f'The declared type gate for the {MODULE_PREFIX} module runs '
            f'`uv run --project shared pyright {MODULE_PREFIX}/` FROM THE REPO '
            f'ROOT (no --directory), so it resolves '
            f'against this root table. Without both '
            f'{list(_REQUIRED_EXTRA_PATHS)!r} entries, pyright cannot resolve '
            f'the flat modules {list(_UNRESOLVED_WITHOUT)!r} that '
            f'{MODULE_PREFIX}/tests/conftest.py puts on sys.path at runtime — '
            f'measured as exactly 9 reportMissingImports before task 3456 added '
            f'them. The gate then reports RED for reasons unrelated to any '
            f'diff, which is how a gate gets suppressed and dies. Note '
            f'{_REQUIRED_EXTRA_PATHS[0]!r} alone is NOT sufficient: it makes '
            f'{_REQUIRED_EXTRA_PATHS[1]!r} importable only as a namespace '
            f'package, not its contents as bare top-level names. '
            f'BLAST RADIUS IS TWO GATES, NOT ONE: the {SIBLING_PREFIX} module '
            f'config declares its own '
            f'`uv run --project shared pyright {SIBLING_PREFIX}/`, which '
            f'also runs from the repo root against this same table, and task '
            f'3456 dropped the `# pyright: ignore[reportMissingImports]` '
            f'pragmas that were masking its five modules\' flat '
            f'{MODULE_PREFIX}/ imports. Re-measured at the branch tip with '
            f'both entries deleted: `pyright {MODULE_PREFIX}/` -> 23 errors '
            f'(9 reportMissingImports), `pyright {SIBLING_PREFIX}/` -> 8 '
            f'errors, ALL reportMissingImports. Restoring these entries is the '
            f'fix for BOTH; re-adding pragmas to {SIBLING_PREFIX} is not'
        )


def test_scripts_diff_is_type_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A diff confined to scripts/ must actually run pyright over scripts/.

    The TYPE half of the same contract ``test_scripts_diff_is_lint_gated``
    above pins for LINT, and it has the same failure mode: task 3445 declared
    only ``test_command`` and ``lint_command`` here, so every diff confined to
    ``scripts/`` cleared the TYPE check without pyright ever running. That was
    stated in ``scripts/orchestrator.yaml`` as a known-open gap rather than
    left silent, and task 3456 closed it — burning the tree down to zero
    FIRST, because declaring a red command is a fleet-wide outage rather than a
    transient failure (``verify.run_full_verification`` asyncio-gathers over
    ALL ``module_configs.values()``, and the repo root sets
    ``merge_verify_breadth: full``).

    The routing PRECONDITIONS — discovery, prefix, lock_depth reachability,
    derive_modules resolution — are asserted once by the lint test and are not
    restated here; they are properties of the module config, not of either
    command, and duplicating them would double the maintenance surface for no
    added coverage. What is NOT shared is asserted below.

    (a)/(b) are the two forms the command takes in production and both must
    hold: (a) the FILE_SCOPED render a normal scripts/-only diff executes, and
    (b) the DECLARED value, which runs VERBATIM and unscoped under merge-role
    ``merge_verify_breadth=full`` and on STRUCTURAL diffs. Note the type leg
    reaches FULL_SUITE more readily than the lint leg does:
    ``verify_plan._derive_module_runs`` sets ``need_structural`` from
    ``bool(mc.type_check_command)``, so declaring this command is what makes
    scripts/ diffs read file CONTENT at all, and a diff touching a module that
    defines a TypedDict or Protocol widens pyright to the unscoped form.

    (c) is the anti-copy-paste half. It cannot live on the executed value for
    the same reason the lint test's assertion (4) records:
    ``_scope_prefix_to_keyword`` REPLACES the declared targets with the touched
    file list, so a config carrying the sibling's ``pyright tests/scripts/``
    renders byte-identically to the correct one for a scripts/ diff. Only the
    DECLARED value can carry it, and only as exact-element membership —
    ``'scripts/'`` is a SUBSTRING of ``'tests/scripts/'``, so a substring check
    is satisfied by the wrong command.

    (d) is the fix-don't-exclude half. The 395 findings this gate's declaration
    waited on were FIXED — by annotation and narrowing, with zero ``# type:
    ignore`` and zero ``# pyright: ignore`` added — not suppressed and not
    carved out of the target, following the precedent tasks 3350 and 3445 set
    for the lint gate. It is asserted in TWO parts because there are two
    carve-out vectors and the command string only exposes one: the CLI flags
    (``--skip*``, and ``-p``/``--project``, which redirects pyright at another
    config entirely and is invisible to (c)), and the ROOT ``[tool.pyright]``
    table's ``exclude``/``ignore`` keys, which narrow the run with the declared
    command left byte-identical. pyright's flag set is NOT ruff's — it has no
    ``--exclude`` and no ``--ignore`` at all — so this half cannot be written
    by copying the lint test's; see ``_NARROWING_FLAGS``.

    Cited by SYMBOL, never by file:line, for the reason the module docstring
    records.
    """
    discovered = _discovered()
    mc = discovered[MODULE_PREFIX]

    executed = _executed_for_touched([SAMPLE_TOUCHED_FILE], _root_config(monkeypatch))

    # (a) THE GATE ITSELF, on the FILE_SCOPED render. A None command here is
    # not "type-checking deferred to some other config" — it is TYPE DELETED,
    # and it reports green.
    assert (
        executed.type_check_command is not None
        and 'pyright' in executed.type_check_command
    ), (
        f'executed type_check_command is {executed.type_check_command!r} for a '
        f'{MODULE_PREFIX}/-only diff (task 3456). Declaring only test_command '
        f'and lint_command on this module config downgrades TYPE to a '
        f'vacuously-passing CheckRun.skipped at rc=0: {_VACUOUS_PASS}. Every '
        f'.py file under {MODULE_PREFIX}/ is then type-checked by nothing, on a '
        'check that reports green. The repo-root type_check_command does not '
        'cover it either'
    )

    # (b) THE DECLARED VALUE, which runs VERBATIM under merge-role
    # merge_verify_breadth=full and on STRUCTURAL diffs.
    assert mc.type_check_command is not None and 'pyright' in mc.type_check_command, (
        f'{MODULE_PREFIX}/orchestrator.yaml declares type_check_command='
        f'{mc.type_check_command!r} (task 3456). Under merge-role '
        f'merge_verify_breadth=full this value runs VERBATIM and unscoped, so '
        f'an absent or non-pyright command leaves the merge path ungated too: '
        f'{_VACUOUS_PASS}'
    )

    # (c) ANTI-COPY-PASTE, part 1 — exact-element target membership. Membership
    # rather than list equality so that a legitimate strengthening (an
    # &&-chained additional gate, an extra target) is not rejected with a
    # message accusing the author of narrowing the gate.
    targets = _targets(mc.type_check_command, _PYRIGHT)
    assert f'{MODULE_PREFIX}/' in targets, (
        f'{MODULE_PREFIX}/orchestrator.yaml declares type_check_command='
        f'{mc.type_check_command!r}, whose pyright targets are {targets!r} — '
        f'{MODULE_PREFIX + "/"!r} is not among them (task 3456). The gate must '
        f'be the DIRECTORY-WIDE form: narrowing it to a file list, or leaving a '
        f'copy-pasted {SIBLING_PREFIX}/ target in place, leaves {MODULE_PREFIX}/ '
        f'ungated under merge full-verify behind a check that reports green — '
        f'{_VACUOUS_PASS}. Exact-element membership is deliberate: '
        f'{MODULE_PREFIX + "/"!r} is a SUBSTRING of {SIBLING_PREFIX + "/"!r}, so '
        'a substring check would pass for the sibling command'
    )

    # (d) NOTHING CARVED OUT of the directory-wide target — part 1, the COMMAND
    # LINE. See _NARROWING_FLAGS for why pyright's set is not ruff's, and why
    # -p/--project belongs here rather than being caught by (c).
    narrowing = _narrowing_flag_args(mc.type_check_command, _PYRIGHT)
    assert not narrowing, (
        f'{MODULE_PREFIX}/orchestrator.yaml declares type_check_command='
        f'{mc.type_check_command!r}, which narrows what is actually checked '
        f'with {narrowing!r} (task 3456). The findings measured at declaration '
        f'time were FIXED by annotation and narrowing — zero `# type: ignore`, '
        f'zero `# pyright: ignore`, zero excludes — per the task 3350/3445 '
        f'fix-don\'t-exclude precedent; any of these silently un-gates what it '
        f'names while the check still reports green. `-p`/`--project` is the '
        f'sharpest of them: it redirects pyright at another config file, and '
        f'assertion (c) above cannot see it because _targets discards every '
        '`-`-prefixed token'
    )

    # (d) part 2 — the OTHER carve-out vector, which NO inspection of the
    # command can reach. `uv run --project shared pyright scripts/` runs from
    # the repo root, so it
    # resolves against the ROOT [tool.pyright] table — the same table
    # test_root_pyright_extrapaths_resolves_scripts_imports pins, read here
    # through the same _load_root_pyright_config helper. An `exclude` or
    # `ignore` entry added there narrows the run while this module config, and
    # every assertion above it, stays byte-for-byte unchanged.
    carved = _root_carve_outs_naming(MODULE_PREFIX)
    assert not carved, (
        f'ROOT [tool.pyright] in {REPO_ROOT / "pyproject.toml"} carves '
        f'{MODULE_PREFIX} back out with {carved!r} (task 3456). '
        f'`{mc.type_check_command}` runs from the repo root and resolves '
        f'against that table, so an `exclude` there drops those files from '
        f'analysis and an `ignore` suppresses their diagnostics — either way '
        f'the gate keeps exiting 0 over a tree it is no longer checking, which '
        f'is the same reports-green failure as {_VACUOUS_PASS}, reached '
        f'without touching the command. The findings were FIXED, not excluded; '
        'if a future finding genuinely cannot be fixed, suppress it at the '
        'single site with a justified inline pragma, where a reader of that '
        'code can see it — not repo-wide from a config table nothing in '
        f'{MODULE_PREFIX}/ mentions'
    )

    # (c) ANTI-COPY-PASTE, part 2 — the sibling comparison. These two
    # directories are distinct trees.
    sibling = discovered.get(SIBLING_PREFIX)
    assert sibling is not None, (
        f'{SIBLING_PREFIX}/orchestrator.yaml is no longer discovered, so the '
        'anti-copy-paste comparison below cannot be made (task 3456)'
    )
    assert mc.type_check_command != sibling.type_check_command, (
        f'{MODULE_PREFIX} and {SIBLING_PREFIX} declare a BYTE-IDENTICAL '
        f'type_check_command {mc.type_check_command!r} (task 3456). A shared '
        f'command means one of them is type-checking the other and its own '
        f'files are gated by nothing. The two test_commands are held distinct '
        f'by their own assertions — see '
        'test_scripts_full_suite_pytest_covers_scripts_tests, whose TARGET '
        'assertions (5) and (6) carry that claim: this config must collect '
        'BOTH test trees while the sibling collects only its own. (Its '
        'byte-inequality check is a redundant restatement of those two, kept '
        'for the diagnostic, so read the target assertions for the actual '
        'contract.) Nothing about this repo licenses duplicating a command '
        'across these two configs'
    )


def test_type_gates_resolve_pyright_without_npx() -> None:
    """Both non-member type gates must resolve pyright through uv, never npx.

    Task 4358, mirroring task 3842's switch on the sibling. NOT a style rule —
    the failure it prevents was MEASURED, twice, on the GREEN path:

    esc-3473-2 recorded a bare ``npx pyright`` re-resolving through the shared,
    mutable, concurrently-written npm cache under ``$HOME`` on EVERY invocation,
    and once turning a clean 0-error type leg RED on a transient npm-cache write
    failure (npm could not write ``~/.npm/_logs``) with no real type errors in
    the tree. Because ``verify.run_full_verification`` asyncio-gathers over ALL
    ``module_configs.values()`` and the repo root sets
    ``merge_verify_breadth: "full"``, that is a fleet-wide false-red blocking
    every merge, review checkpoint and main-tip sweep — on branches with no
    defect. Task 3842 fixed ``tests/scripts/``; this task fixed ``scripts/``,
    which was the last holdout among the nine discovered module configs.

    (b) IS WHY THIS IS NOT JUST A BAN. Satisfying (a) by dropping to a bare
    ``pyright scripts/`` would resolve whatever ``pyright`` happens to sit on
    PATH — or none at all, since pyright is not on PATH at the worktree root,
    only inside a member venv (``verify.py``'s ``_FALLBACK_UV_PROJECT =
    'shared'`` encodes the same pairing). That trades a flaky red for a
    command-not-found red. The contract is the uv-resolved form specifically.

    Scoped to the two NON-WORKSPACE-MEMBER configs deliberately. They are the
    only two that were ever npx-fronted and the only realistic regression
    surface; the seven workspace members have always been
    ``uv run --project MEMBER ... pyright src/ tests/`` because they need their
    own member environment anyway. Asserting over all nine from inside a
    scripts-module-specific guard would put a repo-wide invariant in a file
    whose ownership and lock scope are module-local — the confusion
    ``tests/scripts/test_module_verify_budgets.py``'s own PLACEMENT docstring
    warns about. That file is the established home for the promoted form, and
    it was GENERALISED FROM these per-module guards after both existed rather
    than bolted on ahead of them.
    """
    discovered = _discovered()

    # Guarded read, in the .get-based shape _load_root_pyright_config already
    # uses here. Raw chained indexing raises a bare KeyError naming only the
    # missing key if the root workspace table is ever restructured, which reads
    # as a broken test rather than the real finding: the membership check below
    # has nothing left to check against.
    with open(REPO_ROOT / 'pyproject.toml', 'rb') as fh:
        root_pyproject = tomllib.load(fh)
    members = root_pyproject.get('tool', {}).get('uv', {}).get('workspace', {}).get('members')
    assert members, (
        f'the root {REPO_ROOT / "pyproject.toml"} declares no non-empty '
        '[tool.uv.workspace].members, so `uv run --project <member>` cannot be '
        'checked against the workspace it names (task 4358). Either the table '
        'moved and this read needs re-pointing, or the workspace itself is '
        'gone and every uv-resolved gate in the repo is broken — both are real '
        'findings, and neither should surface as a bare KeyError'
    )

    for prefix in (MODULE_PREFIX, SIBLING_PREFIX):
        mc = discovered.get(prefix)
        assert mc is not None, (
            f'{prefix}/orchestrator.yaml is no longer discovered by '
            '_discover_module_configs, so its type gate cannot be checked '
            '(task 4358)'
        )
        cmd = mc.type_check_command
        assert cmd, (
            f'{prefix}/orchestrator.yaml declares no type_check_command, so '
            f'{_VACUOUS_PASS} — the invocation-style contract below is moot '
            'until the gate exists at all (tasks 3456 / 3350)'
        )

        # Position matters to every assertion below, so read the two halves
        # through the shared helpers rather than re-locating the anchor here —
        # a hand-rolled `tokens.index(_PYRIGHT)` is the duplication
        # _anchor_split exists to end.
        pre = _pre_anchor_tokens(cmd, _PYRIGHT)
        post = _post_anchor_tokens(cmd, _PYRIGHT)

        # (a) EXACT-TOKEN, not substring, and over the WHOLE segment (both
        # halves; the anchor itself is `pyright`, never `npx`). RED for
        # `scripts` before this task, which declared `npx pyright scripts/`;
        # `tests/scripts` has satisfied it since 3842.
        assert 'npx' not in pre + post, (
            f'{prefix}/orchestrator.yaml declares type_check_command={cmd!r}, '
            'which fronts pyright with npx (task 4358). esc-3473-2 measured '
            'bare `npx pyright` re-resolving through the shared, mutable, '
            'concurrently-written npm cache under $HOME on every invocation, '
            'and once turning a clean 0-error type leg RED on a transient '
            'npm-cache write failure (npm could not write ~/.npm/_logs) with '
            'no real type errors. verify.run_full_verification gathers over '
            'ALL module_configs and the repo root sets merge_verify_breadth: '
            'full, so that is a fleet-wide false-red on the green path, not a '
            'local flake. Resolve pyright through uv instead: '
            f'`uv run --project <member> pyright {prefix}/`'
        )

        # (b) POSITIVELY the uv-resolved form, so (a) cannot be satisfied by
        # dropping to a bare `pyright <dir>` that resolves off PATH or nowhere.
        assert pre[:2] == ['uv', 'run'], (
            f'{prefix}/orchestrator.yaml declares type_check_command={cmd!r}, '
            f'whose pyright segment does not begin `uv run` (task 4358). '
            'Merely removing npx is not the contract: pyright is not on PATH '
            'at the worktree root, only inside a member venv, so a bare '
            f'`pyright {prefix}/` trades a flaky red for a command-not-found '
            "red. verify.py's _FALLBACK_UV_PROJECT = 'shared' encodes the same "
            'environment pairing'
        )
        # Both spellings count — `--project shared` and `--project=shared` are
        # the same uv invocation — so this is not a spelling complaint. See
        # _uv_project_member for the three no-usable-selector cases it folds
        # into None.
        member = _uv_project_member(cmd, _PYRIGHT)
        assert member is not None, (
            f'{prefix}/orchestrator.yaml declares type_check_command={cmd!r}, '
            'which carries no PRE-anchor `--project <member>` selecting the '
            'environment pyright is resolved from (task 4358). A post-anchor '
            '--project is a different thing entirely — it redirects pyright at '
            'another CONFIG file — and is rejected separately below; see the '
            '_NARROWING_FLAGS comment on why the two are distinguished by '
            'position'
        )
        assert member in members, (
            f'{prefix}/orchestrator.yaml declares type_check_command={cmd!r}, '
            f'whose `uv run --project {member}` names {member!r}, which is not '
            'among the declared [tool.uv.workspace].members '
            f'{sorted(members)!r} in the root pyproject.toml (task 4358). uv '
            'resolves a non-member project name to nothing usable, so the gate '
            'would fail to start rather than run — the same reports-nothing '
            'outcome a missing command produces, reached a different way'
        )
        # The POSITION half, and a real check rather than a restatement of the
        # one above: a command may carry BOTH a pre-anchor uv selector and a
        # post-anchor pyright redirect, and every assertion so far passes it.
        # test_scripts_diff_is_type_gated assertion (d) catches that shape via
        # _narrowing_flag_args, but only for MODULE_PREFIX — this loop covers
        # the SIBLING config too, which nothing else checks for it.
        redirects = [
            t for t in post if t in ('--project', '-p') or t.startswith(('--project=', '-p='))
        ]
        assert not redirects, (
            f'{prefix}/orchestrator.yaml declares type_check_command={cmd!r}, '
            f'which carries {redirects!r} AFTER the `pyright` anchor — that is '
            "PYRIGHT's config-file redirect, not uv's environment selector "
            '(task 4358). It points pyright at another config file, which can '
            'relax typeCheckingMode, add excludes or drop extraPaths wholesale '
            'while the declared command still names the directory and still '
            'exits 0'
        )


def test_scripts_full_suite_pytest_covers_scripts_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A FULL_SUITE pytest run for the ``scripts`` module must collect scripts/tests/.

    The TEST third of the contract ``test_scripts_diff_is_lint_gated`` and
    ``test_scripts_diff_is_type_gated`` pin for LINT and TYPE, closing the gap
    task 3445 recorded in ``scripts/orchestrator.yaml`` and task 3460 fixed.
    ``scripts/tests/`` holds this module's own test suite, and the declared
    ``test_command`` targeted only ``tests/scripts/`` — the SIBLING tree — so
    under FULL_SUITE that whole suite never ran. (Deliberately stated without a
    module COUNT: the claim is "a whole tree was ungated", which is true at any
    size, and a hard-coded count of a directory's contents rots on the next
    test file added — this file already carried one that was stale at HEAD.)

    Unlike the lint/type gaps this is NOT a vacuous-pass: the command was
    present and green, it simply collected the wrong tree. That makes it
    strictly harder to notice, because no check reports skipped and no command
    is None — which is why the claim has to be made on the TARGETS rather than
    on presence.

    THREE production paths reach the FULL_SUITE form, and all three were
    affected; each is asserted separately because each is derived by a
    different symbol and a regression could restore any one of them alone:

      (2) ``verify_plan._derive_module_runs`` arm 3 — the task-3294 source-only
          floor — runs the owning module's ``test_command`` VERBATIM for ANY
          ``scripts/`` production diff. Measured before the fix: touching
          ``scripts/legibility/census_trigger.py`` rendered
          ``pytest tests/scripts/`` at scope ``full_suite``, reason "pytest:
          source-only diff — owning-module full suite (task role)". Its own
          tests in ``scripts/tests/test_census_trigger.py`` never ran.
      (3) The same function's CONFTEST trigger (arm 1). Touching
          ``scripts/tests/conftest.py`` — the conftest of the very directory —
          also rendered ``pytest tests/scripts/``.
      (4) ``verify_plan._derive_full_suite_runs``, the merge-role
          ``merge_verify_breadth='full'`` deriver the repo root actually
          enables, which never consults the diff at all.

    Only the arm-4b FILE_SCOPED path (a touched ``scripts/tests/test_*.py``)
    narrowed correctly, which is why (2) uses a PRODUCTION file: a test file
    would take that arm and the claim would be unfalsifiable.

    (5) is the reason this is a UNION and not a SWAP, and it is a real
    assertion rather than a note: ``tests/scripts/`` genuinely tests
    ``scripts/`` PRODUCTION code (``test_orchestrator_watchdog.py`` <->
    ``scripts/orchestrator-watchdog.py``, ``test_spawn_claude.py``,
    ``test_check_dashboard_unit_parity.py``,
    ``test_restart_all_orchestrators.py``, ...). Because arm 3 runs this
    command verbatim for every ``scripts/`` production diff, narrowing to
    ``scripts/tests/`` alone would STOP running those tests for exactly the
    diffs they cover — trading one coverage gap for another. Coverage must be
    monotone in the diff, the same principle task 3294 encoded when it moved
    the floor above the collectable-test branch.

    Asserted STRUCTURALLY through the production ``derive_verify_plan`` ->
    ``_executed_module_configs_from_plan`` bridge, never by ``yaml.safe_load``
    and never by an exit code, for the reasons the module docstring records.
    Cited by SYMBOL, never by file:line, for the same reason.
    """
    discovered = _discovered()

    # (1) PRECONDITION. Asserted so a discovery regression cannot make every
    # assertion below vacuous — an absent config would raise a KeyError with no
    # explanation, and a None test_command would make _pytest_targets fail on a
    # TypeError rather than on the claim.
    assert MODULE_PREFIX in discovered, (
        f'{MODULE_PREFIX}/orchestrator.yaml is not discovered by the production '
        f'config._discover_module_configs walk, so nothing below can gate a '
        f'{MODULE_PREFIX}/ diff. Discovered: {sorted(discovered)}'
    )
    mc = discovered[MODULE_PREFIX]
    assert mc.test_command, (
        f'{MODULE_PREFIX}/orchestrator.yaml declares test_command='
        f'{mc.test_command!r}. A falsy test_command is not "tests deferred to '
        f'another config" — verify_plan._derive_module_runs emits a SKIPPED '
        f'PlannedRun with cmd=None for it, and {_VACUOUS_PASS}'
    )

    # (2) TASK-ROLE ARM 3 — the task-3294 source-only floor, which runs the
    # declared command VERBATIM for any scripts/ production diff.
    cfg = _root_config(monkeypatch)
    executed_production = _executed_for_touched(
        [SAMPLE_TOUCHED_PRODUCTION_FILE], cfg
    )
    assert executed_production.test_command is not None, (
        f'executed test_command is None for a diff touching '
        f'{SAMPLE_TOUCHED_PRODUCTION_FILE!r} (task 3460), so pytest is not run '
        f'at all: {_VACUOUS_PASS}'
    )
    production_targets = _pytest_targets(executed_production.test_command)
    assert _dir_key(OWN_TESTS_DIR) in _dir_keys(production_targets), (
        f'a diff touching the PRODUCTION module {SAMPLE_TOUCHED_PRODUCTION_FILE!r} '
        f'executes pytest over {production_targets!r}, which does not include '
        f'{OWN_TESTS_DIR!r} (task 3460). verify_plan._derive_module_runs arm 3 — '
        f'the task-3294 source-only floor — runs the owning module config\'s '
        f'test_command VERBATIM for a source-only diff, so this module\'s own '
        f'test modules under {OWN_TESTS_DIR} never run for the very diffs they '
        f'cover; scripts/tests/test_census_trigger.py is the direct counterpart '
        f'of this file. This is NOT a vacuous pass — the command is present and '
        f'exits 0 — it simply collects the {SIBLING_TESTS_DIR} tree instead, '
        f'which is why the claim is made on the TARGETS and not on presence. '
        f'The check is exact-element on the positional targets rather than a '
        f'substring of the command, because `pytest {OWN_TESTS_DIR}test_x.py` '
        f'and `--ignore={OWN_TESTS_DIR}` both CONTAIN {OWN_TESTS_DIR!r} while '
        f'collecting something else — see _pytest_targets. A trailing slash is '
        'not required: _dir_key normalizes it away'
    )

    # (3) CONFTEST TRIGGER (arm 1) — the sharpest case, because the touched
    # file IS scripts/tests/'s own conftest.
    executed_conftest = _executed_for_touched([SAMPLE_TOUCHED_CONFTEST], cfg)
    assert executed_conftest.test_command is not None, (
        f'executed test_command is None for a diff touching '
        f'{SAMPLE_TOUCHED_CONFTEST!r} (task 3460): {_VACUOUS_PASS}'
    )
    conftest_targets = _pytest_targets(executed_conftest.test_command)
    assert _dir_key(OWN_TESTS_DIR) in _dir_keys(conftest_targets), (
        f'touching {SAMPLE_TOUCHED_CONFTEST!r} executes pytest over '
        f'{conftest_targets!r}, which does not include {OWN_TESTS_DIR!r} '
        f'(task 3460). verify_plan._derive_module_runs\' CONFTEST trigger '
        f'widens to FULL_SUITE precisely BECAUSE a conftest change can affect '
        f'every test in its directory — and then runs a command that collects a '
        'different directory entirely, so the widening buys nothing for the '
        'suite it was widened for. Note the SLASHLESS spelling '
        f'{_dir_key(OWN_TESTS_DIR)!r} satisfies this assertion: '
        'verify_plan._fallback_pytest_targets already maps a touched conftest '
        'to its parent DIRECTORY in that form, and _dir_key normalizes the '
        'trailing slash away so adopting that better-scoped shape here would '
        'not read as a regression'
    )

    # (4) MERGE FULL BREADTH — the leg the repo root actually enables with
    # merge_verify_breadth: "full". Derived by a DIFFERENT symbol from (2)/(3)
    # and never consults the diff, so it is asserted separately.
    full_suite_runs = verify_plan._derive_full_suite_runs(mc, role='merge')
    pytest_runs = [r for r in full_suite_runs if r.reason.startswith('pytest:')]
    assert len(pytest_runs) == 1, (
        f'verify_plan._derive_full_suite_runs({MODULE_PREFIX!r}, role="merge") '
        f'emitted {len(pytest_runs)} pytest PlannedRuns, expected exactly 1: '
        f'{[r.reason for r in full_suite_runs]!r}'
    )
    pytest_run = pytest_runs[0]
    assert (
        pytest_run.scope_kind is verify_plan.ScopeKind.FULL_SUITE
        and pytest_run.cmd is not None
    ), (
        f'verify_plan._derive_full_suite_runs({MODULE_PREFIX!r}, role="merge") '
        f'planned the pytest slot as scope_kind={pytest_run.scope_kind!r} '
        f'cmd={pytest_run.cmd!r} (task 3460). Under the repo root\'s '
        f'merge_verify_breadth="full" this slot must run the declared command '
        f'FULL_SUITE and unconditionally; a SKIPPED slot renders back to None '
        f'and {_VACUOUS_PASS}'
    )

    # The command that FULL_SUITE slot runs is mc.test_command VERBATIM — which
    # is also what (2) and (3) render — so asserting on the DECLARED value here
    # covers all three paths from one place.
    declared_targets = _pytest_targets(mc.test_command)
    assert _dir_key(OWN_TESTS_DIR) in _dir_keys(declared_targets), (
        f'{MODULE_PREFIX}/orchestrator.yaml declares test_command='
        f'{mc.test_command!r}, whose pytest targets are {declared_targets!r} — '
        f'{OWN_TESTS_DIR!r} is not among them (task 3460). '
        f'verify_plan._derive_full_suite_runs runs this value VERBATIM and '
        f'unscoped under merge-role merge_verify_breadth="full", so every test '
        f'module under {OWN_TESTS_DIR} is ungated on the merge path too'
    )

    # (5) NON-REGRESSION — why this is a UNION and not a SWAP. See the
    # docstring: tests/scripts/ tests scripts/ PRODUCTION code, and arm 3 runs
    # this command verbatim for every scripts/ production diff.
    assert _dir_key(SIBLING_TESTS_DIR) in _dir_keys(declared_targets), (
        f'{MODULE_PREFIX}/orchestrator.yaml declares test_command='
        f'{mc.test_command!r}, whose pytest targets are {declared_targets!r} — '
        f'{SIBLING_TESTS_DIR!r} is no longer among them (task 3460). Adding '
        f'{OWN_TESTS_DIR!r} must be ADDITIVE: {SIBLING_TESTS_DIR} genuinely '
        f'tests {MODULE_PREFIX}/ PRODUCTION code '
        f'(test_orchestrator_watchdog.py <-> scripts/orchestrator-watchdog.py, '
        f'test_spawn_claude.py, test_check_dashboard_unit_parity.py, '
        f'test_restart_all_orchestrators.py, ...), and '
        f'verify_plan._derive_module_runs arm 3 runs this command verbatim for '
        f'every {MODULE_PREFIX}/ production diff — so dropping it stops running '
        'those tests for exactly the diffs they cover, trading one coverage gap '
        'for another. Coverage must be MONOTONE in the diff'
    )

    # (6) ANTI-COPY-PASTE, completing the family test_scripts_diff_is_lint_gated
    # and test_scripts_diff_is_type_gated already guard for the other two
    # commands. Both halves matter: the gap must not be "closed" by widening the
    # SIBLING config instead of this one, which would leave a scripts/-only diff
    # — routed by derive_modules -> for_module's longest-prefix walk to prefix
    # `scripts`, never to `tests/scripts` — still running the wrong tree.
    sibling = discovered.get(SIBLING_PREFIX)
    assert sibling is not None, (
        f'{SIBLING_PREFIX}/orchestrator.yaml is no longer discovered, so the '
        'anti-copy-paste comparison below cannot be made (task 3460)'
    )
    # REDUNDANT BY CONSTRUCTION, kept only as a better first diagnostic — do not
    # mistake it for independent coverage. It is strictly IMPLIED by the two
    # assertions that bracket it: the declared targets must contain both
    # directories (above) while the sibling's must be exactly the one (below),
    # and two commands with different positional-target lists cannot be
    # byte-identical. What it buys is the failure MESSAGE: when the mis-fix is a
    # wholesale copy-paste of one config's command into the other, this fires
    # first and names that directly, instead of leaving a reader to infer it
    # from two target lists. The claim itself is carried by the target
    # assertions, which is where the lint/type tests' cross-references point.
    assert mc.test_command != sibling.test_command, (
        f'{MODULE_PREFIX} and {SIBLING_PREFIX} declare a BYTE-IDENTICAL '
        f'test_command {mc.test_command!r} (task 3460). They were identical '
        f'before this task, and that IS the defect: {MODULE_PREFIX} was running '
        f'the sibling\'s suite and none of its own. The two directories are '
        f'distinct trees and {MODULE_PREFIX} must additionally collect '
        f'{OWN_TESTS_DIR}'
    )
    assert sibling.test_command, (
        f'{SIBLING_PREFIX}/orchestrator.yaml declares test_command='
        f'{sibling.test_command!r} (task 3460), so it now gates nothing and the '
        f'comparison above is satisfied for the wrong reason'
    )
    sibling_targets = _pytest_targets(sibling.test_command)
    assert _dir_keys(sibling_targets) == [_dir_key(SIBLING_TESTS_DIR)], (
        f'{SIBLING_PREFIX}/orchestrator.yaml declares test_command='
        f'{sibling.test_command!r}, whose pytest targets are '
        f'{sibling_targets!r}, expected exactly {[SIBLING_TESTS_DIR]!r} '
        f'(trailing slash optional — compared through _dir_key) '
        f'(task 3460). The {MODULE_PREFIX} gap must be closed on the '
        f'{MODULE_PREFIX} module config — widening the SIBLING to also collect '
        f'{OWN_TESTS_DIR} does NOT close it, because derive_modules -> '
        f'OrchestratorConfig.for_module\'s longest-prefix walk routes every '
        f'{MODULE_PREFIX}/** path to prefix {MODULE_PREFIX!r} and never to '
        f'{SIBLING_PREFIX!r}, so a {MODULE_PREFIX}/-only diff would never reach '
        'the widened command. Equality rather than membership here precisely '
        'because a legitimate strengthening of the sibling does not exist: this '
        'assertion exists to reject exactly that mis-fix'
    )


def _root_scripts_suites_pytest_targets(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Positional targets of the fleet chain's pytest segment for the scripts suites.

    Read through the PRODUCTION loader (``_root_config`` — see its docstring
    for why anchoring ``ORCH_CONFIG_PATH`` is load-bearing here, not hygiene)
    rather than by ``yaml.safe_load``: the ``test_command`` an
    ``OrchestratorConfig`` carries is the same value
    ``verify._build_fallback_config`` receives as its ``config``.

    The segment is selected by CONTENT (the one pytest segment whose targets
    name either scripts test tree) and not by POSITION ("the trailing
    segment"). A future subproject appended after it would silently move a
    positional pick onto the wrong segment, and the guard would then be
    checking something else entirely while still reporting green — the same
    reports-green failure mode this whole file exists to prevent.
    """
    root_cmd = _root_config(monkeypatch).test_command
    assert root_cmd, (
        f'the repo-root orchestrator config declares test_command={root_cmd!r}, '
        'so the fleet chain gates nothing and the comparison below would be '
        'satisfied vacuously'
    )
    wanted = {_dir_key(OWN_TESTS_DIR), _dir_key(SIBLING_TESTS_DIR)}
    segments = [s for s in verify_cmd.split_top_level_and(root_cmd) if _PYTEST in s]
    matching = [s for s in segments if wanted & set(_dir_keys(_targets(s, _PYTEST)))]
    assert len(matching) == 1, (
        f'expected exactly one pytest segment naming {sorted(wanted)!r} in the '
        f'repo-root fleet chain, got {matching!r} out of {segments!r}. Zero '
        f'means the chain no longer runs either scripts test tree at all; more '
        'than one means the two trees were split across segments, which this '
        'guard cannot compare as a single unit'
    )
    return _targets(matching[0], _PYTEST)


def test_root_fleet_chain_and_scripts_module_agree_on_the_scripts_suites(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fleet chain and the ``scripts`` module config must name the SAME suites.

    Both yamls ASSERT this coupling in prose and, until this guard, nothing
    enforced it. ``dark-factory-orchestrator.yaml``: "The two are kept spelled
    identically deliberately, so this path and that one cannot drift into
    different notions of 'the scripts suites'; if you widen one, widen the
    other." ``scripts/orchestrator.yaml``: "The target ORDER and spelling copy
    the repo-root fleet chain's trailing segment verbatim, so the fallback path
    and this module config cannot drift."

    ``test_scripts_full_suite_pytest_covers_scripts_tests`` above reads only
    ``discovered['scripts'].test_command``, so it cannot see the root chain;
    dropping ``scripts/tests/`` from the chain would leave a documented
    invariant silently violated — a documented-but-ungated claim, which is the
    exact defect class task 3460 exists to close, reintroduced one file over.

    NOT covered by ``tests/scripts/test_fallback_verify_config.py::
    test_fallback_verify_runs_tests_scripts`` next door: that one asserts only
    that SOME pytest segment mentions ``tests/scripts``, which a chain that
    dropped ``scripts/tests/`` entirely still satisfies.

    SET equality on the normalized targets. Equality because the claim the two
    comment blocks make is BIDIRECTIONAL — widening either side alone is
    precisely what must fail — and a set because pytest collects directories
    order-insensitively, so pinning ORDER would reject a harmless reordering
    with a message about coverage. The yamls' "same order" wording is a
    readability convention, not a correctness property, and is deliberately
    not encoded as one here.

    The chain is DEFENCE-IN-DEPTH only — ``run_scoped_verification`` reaches
    ``_build_fallback_config`` solely past its ``if module_configs:`` check, and
    a ``scripts/**`` diff always routes to the ``scripts`` module config
    instead — so this guard is about keeping the two spellings honest, not
    about the chain gating scripts/ diffs. It does not license "fixing" a
    scripts/ coverage question in the root yaml.
    """
    root_targets = _root_scripts_suites_pytest_targets(monkeypatch)

    discovered = _discovered()
    assert MODULE_PREFIX in discovered, (
        f'{MODULE_PREFIX}/orchestrator.yaml is not discovered by the production '
        f'config._discover_module_configs walk. Discovered: {sorted(discovered)}'
    )
    mc = discovered[MODULE_PREFIX]
    assert mc.test_command, (
        f'{MODULE_PREFIX}/orchestrator.yaml declares test_command='
        f'{mc.test_command!r}, so there is nothing to compare the fleet chain '
        'against and the equality below would be satisfied for the wrong reason'
    )
    module_targets = _pytest_targets(mc.test_command)

    assert set(_dir_keys(root_targets)) == set(_dir_keys(module_targets)), (
        f'the repo-root fleet chain\'s scripts-suites pytest segment targets '
        f'{root_targets!r} while {MODULE_PREFIX}/orchestrator.yaml\'s '
        f'test_command targets {module_targets!r} (task 3460). BOTH yamls state '
        f'in prose that these are kept identical and that widening one means '
        f'widening the other; this guard is what makes that a property rather '
        f'than an aspiration. Fix the drift or delete the claim from both '
        f'comment blocks — do not leave a documented invariant ungated, which '
        'is the defect class task 3460 closed'
    )

    # Belt and braces: identical-but-empty would satisfy the equality above.
    # This is the claim the equality is FOR, stated directly, so a chain and a
    # module config that drifted together still fail here with a message about
    # coverage rather than about agreement.
    for required in (OWN_TESTS_DIR, SIBLING_TESTS_DIR):
        assert _dir_key(required) in _dir_keys(root_targets), (
            f'{required!r} is not among the fleet chain\'s scripts-suites '
            f'pytest targets {root_targets!r} (task 3460). Both trees must run: '
            f'{SIBLING_TESTS_DIR} tests {MODULE_PREFIX}/ PRODUCTION code '
            f'(test_orchestrator_watchdog.py <-> scripts/orchestrator-watchdog.py, '
            f'test_spawn_claude.py, ...) and {OWN_TESTS_DIR} is that directory\'s '
            'own suite, so dropping either trades one coverage gap for another'
        )


# The scripts module's own measured wall-clock. Task 3458 ran two fresh,
# independent, sequential runs of the verbatim union test_command at this
# branch's base commit 37f761f5a4 (360.47s and 293.32s wall) and combined
# them with the four runs of the byte-identical command already recorded in
# scripts/orchestrator.yaml's MEASURED GREEN / COST DELTA blocks (444.17s,
# 565.37s wall; 310.33s pytest-only; 914.61s pytest / 930.59s wall). The
# floor below is set against the WORST wall-clock across ALL SIX of those
# (930.59s, the "amendment-pass verification" run) — never the mean: that
# same block records a ~3x spread (310.33s -> 914.61s) for a BYTE-IDENTICAL
# command on a BYTE-IDENTICAL tree, which is this oversubscribed host's LOAD
# at measurement time, not suite variance. Neither of task 3458's own two
# fresh runs came anywhere near this worst figure, which is itself evidence
# for sizing against the max rather than any single run: the worst case is
# real but not the common case, so a mean or a fresh-only measurement would
# both have under-sized the floor.
MEASURED_SUITE_WORST_SECS = 930.59
# DERIVED from MEASURED_SUITE_WORST_SECS, not hand-set, so the two cannot
# silently diverge — that exact pair has already rotted once, in the sibling
# guard tests/scripts/test_tests_scripts_module_config.py, which left a
# HAND-SET floor standing against a stale worst-run figure while its own
# tests/scripts/orchestrator.yaml had since recorded a worse one, and nothing
# in the repo could fail on the drift (task 3458 amendment pass,
# reviewer-flagged; the sibling was out of THAT task's locked scope, so it was
# filed as a follow-up rather than fixed there).
#
# THAT FOLLOW-UP IS TASK 3703, AND IT LANDED — so the paragraph above is
# HISTORY, not current state: the sibling now pins its MEASURED_SUITE_WORST_SECS
# to a recorded run of its own VERBATIM test_command and DERIVES its floor from
# it by this same expression, so that yaml's worst figure is gated. Corrected in
# place rather than deleted, because the rot is why this constant is derived at
# all; corrected rather than left standing, because an authoritative-reading
# comment that a later change has falsified is precisely the defect both tasks
# exist to remove — and this one had already been falsified by its own
# follow-up (task 3703 amendment pass, reviewer-flagged).
#
# NO FIGURE OF THE SIBLING'S IS REPEATED HERE, deliberately, and that is
# unchanged by task 4320 even though the DERIVATION is now shared. Cross-module
# provenance has ONE home — the MEASURED_BY_SIBLING_GUARD table in
# tests/scripts/test_module_verify_budgets.py — so that a re-measurement over
# there does not have to be chased into this comment, which is the lockstep
# duplication that table's own comment exists to refuse. Importing
# module_budget_family.min_budget imports an EXPRESSION, not a MEASUREMENT: this
# module's worst-run figure above stays this module's own, and the sibling's
# stays the sibling's.
#
# ~2x the worst observed run, rounded DOWN to the nearest 100s:
# 2 * 930.59 -> 1861.18 -> 1800.
#
# CORRECTED IN PLACE (task 4320): this line used to INLINE that expression as
# `(int(2 * MEASURED_SUITE_WORST_SECS) // 100) * 100`, which was the third
# spelling of one derivation across this family — the other two being a
# `def _min_budget` copy in each sibling guard. Nothing could observe drift
# between them, so the family's shared shape was a convention rather than a
# property. It is now a property: test_module_verify_budgets.py::
# test_the_budget_family_derives_every_floor_from_one_canonical_expression
# evaluates this expression in a namespace holding only `min_budget` and
# `MEASURED_SUITE_WORST_SECS` and NO __builtins__, so re-inlining it here would
# raise NameError on `int` and fail there — even though it would still compute
# 1800 today.
MIN_MODULE_BUDGET_SECS = min_budget(MEASURED_SUITE_WORST_SECS)


def test_scripts_module_carries_its_own_measured_verify_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The module must carry its own warm verify budget, narrower than the repo-root ceiling.

    Mirrors ``tests/scripts/test_tests_scripts_module_config.py::
    test_tests_scripts_module_carries_its_own_tight_verify_budget``
    assertion-for-assertion (task 3350's shape, applied here by task 3458).
    Two sides are bounded:

    - Below (b): at least 1800s, ~2x the worst of six measured runs of the
      verbatim union test_command (930.59s — see MEASURED_SUITE_WORST_SECS
      above). An achievable floor derived from measurement, not a guess.
    - Above (c): strictly below the repo-root ceiling (3600s): a real
      narrowing, not a relabelling.

    SCOPE, stated honestly rather than copied from the sibling: this is a
    floor-REGRESSION guard, not a suite-growth detector. Nothing here
    re-measures anything, and if the suite doubles tomorrow the frozen
    constant still passes. The narrowing this buys is also modest, unlike
    the sibling's "surfaces a hang in minutes": 3600s -> 2400s means a hang
    surfaces in ~40 minutes instead of ~60, because an observed HONEST GREEN
    run of this suite has measured 930.59s, and nothing tighter than ~2.6x
    that is declarable without manufacturing infra_timeout on the green path
    under load — the same failure this task exists to prevent, one level
    down. The duplicate-run cost that inflates the measured figure
    (tests/scripts/ collected twice per full verify) is task 3383's
    dedupe-by-command guard in verify.run_full_verification, not something a
    yaml budget can fix.

    (d) exercises the REAL precedence mechanism
    (``verify._resolve_verify_timeout``) rather than restating it, so the
    assertion cannot drift from the code that implements it.
    """
    mc = _discovered()[MODULE_PREFIX]

    # (a) Declared at all — otherwise the global ceiling silently applies.
    assert mc.verify_command_timeout_secs is not None, (
        f'{MODULE_PREFIX}/orchestrator.yaml declares no '
        'verify_command_timeout_secs (task 3458), so this module silently '
        'inherits the repo-root whole-fleet ceiling — the budget sized for '
        f'seven subprojects, applied to a suite that has measured up to '
        f'{MEASURED_SUITE_WORST_SECS}s'
    )

    # (b) Measurement-derived floor.
    assert mc.verify_command_timeout_secs >= MIN_MODULE_BUDGET_SECS, (
        f'{MODULE_PREFIX} verify_command_timeout_secs='
        f'{mc.verify_command_timeout_secs} is below the {MIN_MODULE_BUDGET_SECS}s '
        f'floor (task 3458). The suite has measured up to '
        f'{MEASURED_SUITE_WORST_SECS}s across six independent runs (see '
        'scripts/orchestrator.yaml\'s MEASURED GREEN / COST DELTA blocks); a '
        'budget under the floor would manufacture infra_timeout on the honest '
        'green path — the exact defect this task exists to remove, '
        'reintroduced one level down'
    )

    # (c) Strictly tighter than the repo-root ceiling: a real narrowing. Read
    # through the PRODUCTION loader (`_root_config` — see its docstring for
    # why ORCH_CONFIG_PATH anchoring is load-bearing here, not hygiene).
    cfg = _root_config(monkeypatch)
    root_warm = cfg.verify_command_timeout_secs
    assert mc.verify_command_timeout_secs < root_warm, (
        f'{MODULE_PREFIX} verify_command_timeout_secs='
        f'{mc.verify_command_timeout_secs} is not strictly below the repo-root '
        f'verify_command_timeout_secs={root_warm} (task 3458). A per-module '
        'budget at or above the global one is a relabelling, not a narrowing: '
        'it surfaces a hang no sooner than the whole-fleet ceiling would'
    )

    # (d) verify.py's documented precedence actually honours it, warm.
    resolved = verify._resolve_verify_timeout(cfg, mc, is_cold=False)
    assert resolved == mc.verify_command_timeout_secs, (
        f'_resolve_verify_timeout returned {resolved} for a warm verify, not '
        f'the module budget {mc.verify_command_timeout_secs} (task 3458) — the '
        'per-module override is not reaching the code path that enforces it'
    )
    # Redundant by construction with the equality just asserted plus (c)'s
    # `mc.verify_command_timeout_secs < root_warm` above — cannot fail
    # independently of those two. Kept anyway for a message that names the
    # specific failure mode (module override not reaching the resolver)
    # rather than making a reader re-derive it from (c).
    assert resolved != root_warm, (
        f'_resolve_verify_timeout returned the repo-root global {root_warm} '
        f'rather than the module budget for {MODULE_PREFIX}'
    )

    # (e) Survives the PRODUCTION plan -> execution bridge
    # (verify._executed_module_configs_from_plan), not just the resolver in
    # isolation. That bridge uses dataclasses.replace, so every ModuleConfig
    # field carries over today — but verify._apply_cargo_scope (applied to
    # cargo-scoped modules on the same bridge) instead reconstructs
    # ModuleConfig by hand-listing every field, precisely the shape where a
    # per-module budget could be silently dropped by a future field addition
    # or refactor. This module is not cargo-scoped, but the bridge itself is
    # real production code this budget must survive, not just the resolver
    # exercised in isolation above.
    # Passed the SAME cfg assertion (c) already anchored above, rather than
    # letting the helper re-derive one from the env that call left behind —
    # that side effect was the ordering dependency task 3703 removed.
    executed = _executed_for_touched([SAMPLE_TOUCHED_FILE], cfg)
    assert executed.verify_command_timeout_secs == mc.verify_command_timeout_secs, (
        'the production plan->execution bridge '
        '(verify._executed_module_configs_from_plan) rendered '
        f'verify_command_timeout_secs={executed.verify_command_timeout_secs} for a '
        f'{MODULE_PREFIX}-scoped run, not the declared module budget '
        f'{mc.verify_command_timeout_secs} (task 3458) — the per-module override '
        'is dropped somewhere between discovery and execution'
    )


def test_scripts_module_cold_verify_falls_through_to_the_root_cold_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The new WARM budget must NOT narrow the COLD path — a deliberate asymmetry.

    Task 3458 declares ``verify_command_timeout_secs`` but leaves
    ``verify_cold_command_timeout_secs`` UNSET on this module (no build step
    to budget for). Per ``verify._resolve_verify_timeout``'s documented
    cascade, an unset module cold knob falls through to
    ``config.verify_cold_command_timeout_secs`` (the repo-root 5400s) — NOT
    to the new warm value — and ``is_merge_verify=True`` wins first with
    ``config.merge_verify_cold_command_timeout_secs`` (7200s, shipped by the
    package-bundled ``orchestrator/src/orchestrator/defaults.yaml`` and not
    overridden by the root yaml).

    This is exactly the misreading the task description flags as likely:
    that an unset cold knob "inherits" the new warm budget. It does not.
    scripts/orchestrator.yaml's new budget block documents this in prose;
    this guard makes it un-silent rather than leaving it a
    documented-but-ungated invariant — the same defect class tasks 3445/3460
    closed elsewhere in this file.
    """
    mc = _discovered()[MODULE_PREFIX]

    # (a) The deliberate asymmetry: warm is set, cold is not.
    assert mc.verify_command_timeout_secs is not None, (
        f'{MODULE_PREFIX}/orchestrator.yaml declares no '
        'verify_command_timeout_secs (task 3458) — this test cannot check the '
        'cold fall-through behaves correctly relative to a warm budget that '
        'does not exist yet'
    )
    assert mc.verify_cold_command_timeout_secs is None, (
        f'{MODULE_PREFIX}/orchestrator.yaml now declares '
        f'verify_cold_command_timeout_secs={mc.verify_cold_command_timeout_secs} '
        '(task 3458 deliberately left this UNSET: no build step on this module '
        'to budget for). If this was set intentionally, this guard and its '
        'assertions below must be updated together, not just this line'
    )

    # Read the root cold ceilings through the PRODUCTION loader (`_root_config`
    # — same anchoring precedent as the warm-budget guard above).
    cfg = _root_config(monkeypatch)

    # (b) Warm (non-merge) cold verify falls through to the ROOT cold ceiling,
    # NOT the new warm value.
    root_cold = cfg.verify_cold_command_timeout_secs
    resolved_cold = verify._resolve_verify_timeout(
        cfg, mc, is_cold=True, is_merge_verify=False
    )
    assert resolved_cold == root_cold, (
        f'_resolve_verify_timeout(is_cold=True) returned {resolved_cold}, not '
        f'the repo-root verify_cold_command_timeout_secs={root_cold} (task '
        f'3458). {MODULE_PREFIX}/orchestrator.yaml leaves its own cold knob '
        'UNSET deliberately, so a cold verify must fall through to the root '
        'ceiling'
    )
    # NOT redundant with (b) above: nothing before this line bounds root_cold
    # against the module's own warm budget, so this independently catches the
    # root cold ceiling ever being configured down to (or below)
    # verify_command_timeout_secs — a real, if currently slack (5400 vs 2400),
    # cross-file misconfiguration nothing else in this file checks for. (It
    # would follow from the warm-budget test's (c) — module warm < root warm
    # — only if root_cold >= root_warm is also assumed, which is true on this
    # config but is not itself asserted anywhere.)
    assert resolved_cold != mc.verify_command_timeout_secs, (
        f'_resolve_verify_timeout(is_cold=True) returned the module\'s WARM '
        f'budget ({mc.verify_command_timeout_secs}) rather than falling '
        'through to the root cold ceiling (task 3458) — this is precisely the '
        'misreading this guard exists to make un-silent: an unset cold knob '
        'does not "inherit" the new warm budget'
    )

    # (c) The merge-cold knob wins first when is_merge_verify=True.
    merge_cold = cfg.merge_verify_cold_command_timeout_secs
    assert merge_cold is not None, (
        'config.merge_verify_cold_command_timeout_secs is None — the '
        'package-bundled orchestrator/src/orchestrator/defaults.yaml is '
        'expected to ship this (task 3458), so there is nothing for the '
        'merge-cold assertion below to check against'
    )
    resolved_merge_cold = verify._resolve_verify_timeout(
        cfg, mc, is_cold=True, is_merge_verify=True
    )
    assert resolved_merge_cold == merge_cold, (
        f'_resolve_verify_timeout(is_cold=True, is_merge_verify=True) returned '
        f'{resolved_merge_cold}, not '
        f'config.merge_verify_cold_command_timeout_secs={merge_cold} (task '
        '3458) — the merge-cold knob should win before the module/root cold '
        'cascade'
    )

    # (d) Coherence: the module's warm budget must not exceed the resolved
    # (non-merge) cold budget, since a cold run does strictly more work
    # (fresh worktree setup) than a warm one.
    assert mc.verify_command_timeout_secs <= resolved_cold, (
        f'{MODULE_PREFIX} warm verify_command_timeout_secs='
        f'{mc.verify_command_timeout_secs} exceeds the resolved cold budget '
        f'{resolved_cold} (task 3458) — a cold run does strictly more work '
        'than a warm one, so the warm budget must not be looser than the cold '
        'one'
    )
