"""The merge gate's parallelism configuration, pinned where it is DECLARED.

Task 5408. Four of the nine discovered module configs ran their pytest leg
single-process at the merge role while ``orchestrator`` and ``fused-memory``
had carried ``-n auto --dist loadgroup`` in their own addopts for some time.
This guard pins the four that were brought onto that same footing, and pins
the exclusions that were left serial DELIBERATELY, so a later reader cannot
tell a ruled-out module from a forgotten one.

The exclusions are TWO tables, because there are two kinds. Workspace members
are recorded in ``RULED_SERIAL_MEMBERS`` and read out of their own
``pyproject.toml``; module-config prefixes that own no pyproject.toml are
recorded in ``RULED_SERIAL_MODULE_LEGS`` and read out of their
``test_command``. An entry in either is not automatically a RULING: each states
what it actually is, and the one open divergence carries the task it is filed
as.

PLACEMENT IS LOAD-BEARING, and follows the family convention recorded in
``test_module_verify_budgets.py`` and ``test_module_type_check_invocation.py``:
``tests/scripts/`` carries its OWN registered module config, so a guard living
here cannot be silenced by editing the very command it asserts about. A NEW
file rather than an append to a sibling, matching how both of those guards were
created.

WHY ``auto`` AND NOT A LITERAL ``-n 8``. ``verify._resolve_verify_env`` injects
the repo-root ``verify_env`` for EVERY verify role, and
``dark-factory-orchestrator.yaml`` pins ``PYTEST_XDIST_AUTO_NUM_WORKERS``
there. No module config declares its own ``verify_env``, so nothing shadows
that pin: one knob governs every parallel module, and the A/B cut that moves it
moves them together. A numeral baked into a module's addopts would silently
opt that module OUT of the knob, which is why the numeral is rejected with its
own message below rather than merely failing the ``auto`` comparison.

Citations are by ``path::symbol``, never file:line — the house style
``test_line_pin_policy.py`` enforces in this directory.
"""
from __future__ import annotations

import pathlib
import re
import shlex
import subprocess
import tomllib
from typing import TYPE_CHECKING

import pytest

# The shared verify/lint/type command parser. "IMPORT ME, DO NOT COPY ME" is its
# own docstring's instruction, and it resolves by the conftest.py sys.path
# insertion this directory relies on under --import-mode=importlib.
import verify_command_invariants as vci

if TYPE_CHECKING:
    from collections.abc import Callable

    from orchestrator.config import ModuleConfig

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The workspace members whose OWN pyproject.toml addopts gained the parallel
# flags. `scripts` is absent on purpose and is covered separately: it has no
# pyproject.toml of its own, so its flags live on its test_command instead.
PARALLEL_ADDOPTS_MEMBERS = ('dashboard', 'escalation', 'cockpit')

# The spelling every parallel module shares, copied from
# `fused-memory/pyproject.toml::[tool.pytest.ini_options]` so all of them read
# as one family rather than as independently-invented variants.
WORKERS_FLAG = '-n'
WORKERS_VALUE = 'auto'
DIST_FLAG = '--dist'
DIST_VALUE = 'loadgroup'

# cockpit's marker deselection must survive the addition, and must stay a
# single flat `-m 'not X'`: that is the only shape
# `test_pytest_workspace_collection.py::_FLAT_DESELECT_RE` claims to
# understand, and a richer expression there is REJECTED at parse time rather
# than approximated.
COCKPIT_MARKER_EXPRESSION = 'not smoke'


def _pyproject(member: str) -> dict:
    """*member*'s parsed ``pyproject.toml``.

    Read with ``tomllib`` from the file on disk rather than through the
    orchestrator config loader, because neither addopts nor a dependency group
    is orchestrator configuration: addopts is what governs a bare
    ``cd <member> && uv run pytest tests/`` as much as the verify leg, and the
    dependency group is what uv resolves the plugin from.
    """
    pyproject = REPO_ROOT / member / 'pyproject.toml'
    assert pyproject.is_file(), (
        f'{member}/pyproject.toml does not exist, so every assertion below '
        'would be about a file this repo does not have'
    )
    return tomllib.loads(pyproject.read_text(encoding='utf-8'))


def _declared_addopts(member: str) -> str:
    """*member*'s ``[tool.pytest.ini_options].addopts``, ``''`` when it declares none.

    NON-asserting, unlike :func:`_addopts`. The structural walk below reads
    every discovered module, and a module that declares no addopts at all is a
    legitimate answer there — it simply contributes no ``-n``.

    *member* is a repo-relative directory, so ``'.'`` names the ROOT
    pyproject.toml — which is what the non-leakage assertion reads, and which is
    a genuinely different file from any member's rather than a special case of
    one.
    """
    ini_options = _pyproject(member).get('tool', {}).get('pytest', {}).get('ini_options', {})
    return ini_options.get('addopts', '')


def _addopts(member: str) -> str:
    """*member*'s addopts, asserted PRESENT.

    The asserting entry point, for the three members this file pins by name. A
    missing addopts key there is the defect, not an absence to tolerate.
    """
    addopts = _declared_addopts(member)
    assert addopts, (
        f'{member}/pyproject.toml declares no '
        '[tool.pytest.ini_options].addopts, so its pytest leg runs '
        'single-process at the merge role (task 5408). Add '
        f'{WORKERS_FLAG} {WORKERS_VALUE} {DIST_FLAG} {DIST_VALUE}, copying '
        'fused-memory/pyproject.toml verbatim'
    )
    return addopts


def _flag_value(tokens: list[str], flag: str) -> str | None:
    """The token IMMEDIATELY following *flag* in *tokens*, or ``None``.

    Adjacency is the claim, not mere presence: ``-n`` takes its worker count as
    the next argv element, so a *flag* found as the last token names no value at
    all and a *flag* whose value sits further along is a different invocation.
    Returns ``None`` for both, which the callers report as "declares no value".

    The ``--flag=value`` spelling is deliberately NOT accepted. It is a single
    ``shlex`` token, so it would need its own branch — and admitting it here
    would let the four configs drift away from the one spelling they are
    supposed to share, which is the property this file exists to hold.
    """
    for i, token in enumerate(tokens):
        if token == flag:
            return tokens[i + 1] if i + 1 < len(tokens) else None
    return None


@pytest.mark.parametrize('member', PARALLEL_ADDOPTS_MEMBERS)
def test_parallel_member_addopts_declare_xdist_with_loadgroup(member: str) -> None:
    """Each parallelised member declares ``-n auto --dist loadgroup`` in its addopts.

    ``--dist loadgroup`` accompanies ``-n`` rather than being left at xdist's
    default ``load``: it is what lets a test declare ``@pytest.mark.
    xdist_group`` and be guaranteed a single worker, which is how the two
    already-parallel modules keep their order-sensitive and resource-sharing
    tests honest. Pinning the pair together stops a later edit adding workers
    without the grouping discipline that makes them safe.
    """
    tokens = shlex.split(_addopts(member))

    workers = _flag_value(tokens, WORKERS_FLAG)
    assert workers is not None, (
        f"{member}/pyproject.toml's addopts is {_addopts(member)!r}, which "
        f'declares no {WORKERS_FLAG} value, so its pytest leg runs '
        'single-process at the merge role (task 5408)'
    )
    # Rejected with its OWN message, because failing the `auto` comparison
    # below would say nothing about WHY a numeral is wrong here.
    assert not workers.isdigit(), (
        f'{member}/pyproject.toml declares {WORKERS_FLAG} {workers!r}, a '
        f'literal worker count. It must be {WORKERS_VALUE!r}: pytest-xdist '
        'resolves `auto` through PYTEST_XDIST_AUTO_NUM_WORKERS, which '
        'dark-factory-orchestrator.yaml pins in its `verify_env` block and '
        'verify._resolve_verify_env injects for every verify role. No module '
        'config declares its own verify_env, so that one knob governs every '
        'parallel module — a numeral here opts this module out of it, and the '
        'next A/B cut of the worker count would move every module except this '
        'one'
    )
    assert workers == WORKERS_VALUE, (
        f'{member}/pyproject.toml declares {WORKERS_FLAG} {workers!r}, not '
        f'{WORKERS_VALUE!r}. Copy fused-memory/pyproject.toml\'s spelling '
        'verbatim so every parallel module in the workspace reads identically'
    )

    dist = _flag_value(tokens, DIST_FLAG)
    assert dist == DIST_VALUE, (
        f"{member}/pyproject.toml's addopts declares {DIST_FLAG} {dist!r}, not "
        f'{DIST_VALUE!r} (task 5408). `{DIST_FLAG} {DIST_VALUE}` is what makes '
        '@pytest.mark.xdist_group a guarantee rather than a hint, so it travels '
        f'with {WORKERS_FLAG} rather than being left at xdist\'s default '
        '`load`; it is also what the two already-parallel modules '
        '(orchestrator, fused-memory) declare'
    )


@pytest.mark.parametrize('member', PARALLEL_ADDOPTS_MEMBERS)
def test_parallel_member_addopts_do_not_copy_max_worker_restart(member: str) -> None:
    """No parallelised member copies ``--max-worker-restart`` across.

    ``orchestrator/pyproject.toml`` carries ``--max-worker-restart=0``, and
    copying it here would look like completing the family. It is deliberately
    NOT copied: that flag turns a worker killed by pytest-timeout's thread
    handler into a false-failing per-test "node down" on whatever test happened
    to be running — a SHIFTING VICTIM rather than the starved test — and
    whether orchestrator should keep it is its own open question (tasks 5114 /
    5115), not something this task pre-answers for four more modules.
    """
    offenders = [
        token
        for token in shlex.split(_addopts(member))
        if token.startswith('--max-worker-restart')
    ]
    assert not offenders, (
        f"{member}/pyproject.toml's addopts declares {offenders!r} (task 5408). "
        'That flag was deliberately not carried over from '
        'orchestrator/pyproject.toml: under it a worker that pytest-timeout '
        "os._exit()s becomes a per-test 'node down' failure attributed to a "
        'shifting victim rather than to the starved test, and tasks 5114/5115 '
        "are still open on whether orchestrator should keep it. Do not widen "
        'that exposure to four more modules as a side effect of adding workers'
    )


def test_cockpit_addopts_keeps_its_smoke_deselection() -> None:
    """cockpit's ``-m 'not smoke'`` survives the parallel flags, as ONE flat term.

    cockpit's smoke tests drive real X11/tmux against the live host's DISPLAY,
    so they are opt-in only — the guarantee
    ``cockpit/tests/test_smoke_marker_config.py`` proves behaviourally. Adding
    workers must not disturb it, and two hazards make that worth pinning rather
    than assuming:

      * ``-m`` is a SINGLE argparse option, so a second ``-m`` token silently
        REPLACES the first and drops the deselection outright;
      * the expression must stay the flat ``not X`` form, the only shape
        ``test_pytest_workspace_collection.py::_FLAT_DESELECT_RE`` claims to
        understand — a richer one is rejected there at parse time rather than
        approximated, which would red the root-mirror guard on a config that is
        semantically fine.

    Asserted as a literal equality rather than by re-deriving that regex here:
    equality is strictly stronger, and restating the pattern would give this
    directory a second copy of it to drift.
    """
    tokens = shlex.split(_addopts('cockpit'))

    marker_flags = [token for token in tokens if token == '-m']
    assert len(marker_flags) == 1, (
        f"cockpit/pyproject.toml's addopts carries {len(marker_flags)} `-m` "
        f'flags: {tokens!r}. `-m` is a single argparse option, so a second one '
        'silently replaces the first — one combined expression, never two flags'
    )
    assert _flag_value(tokens, '-m') == COCKPIT_MARKER_EXPRESSION, (
        f"cockpit/pyproject.toml's addopts is {_addopts('cockpit')!r}, whose "
        f'-m expression is not the expected {COCKPIT_MARKER_EXPRESSION!r} '
        '(task 5408). Adding the parallel flags must PRESERVE the smoke '
        'deselection: cockpit smoke tests drive real X11/tmux against the live '
        'host DISPLAY and are opt-in only. Keep it a single flat `not X` term — '
        'test_pytest_workspace_collection.py::_FLAT_DESELECT_RE rejects any '
        'richer shape rather than approximating it'
    )


# ---------------------------------------------------------------------------
# The internal-sense invariant: `-n` requires a DECLARED plugin (task 5408)
# ---------------------------------------------------------------------------

# The distribution that supplies `-n`, PEP 503-normalised for comparison.
XDIST_DISTRIBUTION = 'pytest-xdist'

# The dependency group uv installs by default, and so the only group in which a
# declaration actually reaches the interpreter that runs the verify leg. Every
# member of this workspace declares its test-time plugins here.
DEFAULT_DEPENDENCY_GROUP = 'dev'

# uv's member selectors, read from a command's PRE-anchor tokens only. The
# pre/post split is exactly the category distinction `vci.anchor_split` records:
# before the anchor `--project` selects the ENVIRONMENT the binary resolves
# from, after it the identically spelled flag would redirect the CHECKER's own
# config. Only POSITION tells the two apart.
MEMBER_SELECTORS = ('--directory', '--project')

_REQUIREMENT_NAME_RE = re.compile(r'^\s*([A-Za-z0-9._-]+)')


def _canonical(name: str) -> str:
    """*name* PEP 503-normalised, so ``pytest_xdist`` and ``PyTest-XDist`` compare equal."""
    return re.sub(r'[-_.]+', '-', name).lower()


def _dependency_group_requirements(member: str, group: str) -> set[str]:
    """Canonical distribution names in *member*'s *group*, following ``include-group``.

    ``include-group`` is resolved TRANSITIVELY rather than skipped, because
    skipping it would make this a guard that passes vacuously the moment a
    member factors its plugins into a base group — reporting a missing
    declaration as present is the one failure mode a dependency check must not
    have. Cycles are impossible to follow twice: a group already visited is not
    re-entered.
    """
    groups = _pyproject(member).get('dependency-groups', {})
    names: set[str] = set()
    pending = [group]
    seen: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        for entry in groups.get(current, []):
            if isinstance(entry, str):
                matched = _REQUIREMENT_NAME_RE.match(entry)
                if matched:
                    names.add(_canonical(matched.group(1)))
            elif isinstance(entry, dict) and 'include-group' in entry:
                pending.append(entry['include-group'])
    return names


def _selected_member(pre_anchor_tokens: list[str]) -> str | None:
    """The workspace member uv's PRE-anchor selector names, or ``None`` if there is none.

    Both spellings, because ``--project shared`` and ``--project=shared`` are the
    same uv invocation and a reader that recognises only one of them rejects a
    correct command. Same both-spellings contract as
    ``test_scripts_module_config.py::_uv_project_member``, which reads the same
    slice for the same reason.

    ``None`` covers a command with no selector at all — where uv resolves
    against the workspace ROOT project, which declares no dependency groups, so
    there is no member whose declaration could be asserted about.
    """
    for i, token in enumerate(pre_anchor_tokens):
        for selector in MEMBER_SELECTORS:
            if token == selector:
                return pre_anchor_tokens[i + 1] if i + 1 < len(pre_anchor_tokens) else None
            if token.startswith(selector + '='):
                return token.split('=', 1)[1]
    return None


def _pytest_legs_carrying_workers(
    module_configs: dict[str, ModuleConfig],
) -> list[tuple[str, str, str]]:
    """``(module prefix, selected member, where -n came from)`` for every parallel leg.

    DERIVED STRUCTURALLY from the production config walk, never from a
    hardcoded list: a tenth module config that declares ``-n`` must be caught by
    this guard on the day it lands, not on the day someone remembers to extend a
    table here.

    ``-n`` can reach pytest from either of two places, and both count because
    pytest cannot tell them apart:

      * the command's own POST-anchor argv (how the ``scripts`` leg gets it, its
        targets being repo-root paths under no member's pyproject.toml);
      * the SELECTED member's own addopts (how the ``--directory <m>`` legs get
        it).

    Read no more strongly than it holds: the addopts consulted is the selected
    member's, which for a ``--directory <m>`` command is also the rootdir
    inifile pytest reads, but for a root-cwd ``--project <m>`` command is not.
    In that second shape this can only ever require a declaration pytest would
    not in fact have needed — over-requiring, never under-requiring — which is
    the safe direction for a guard whose failure mode is a missing plugin.
    """
    legs: list[tuple[str, str, str]] = []
    for prefix, module_config in sorted(module_configs.items()):
        command = module_config.test_command
        if not command:
            continue
        segment = vci.optional_token_segment(command, vci.PYTEST)
        if segment is None:
            # A module whose test_command runs something other than pytest
            # contributes no pytest leg — the documented contract of
            # `optional_token_segment`, and the correct semantic rather than an
            # error (`verify._has_source_files` already keys on .rs as well).
            continue
        pre, post = vci.anchor_split(segment, vci.PYTEST, label=f'{prefix} test_command')
        member = _selected_member(pre)
        if member is None:
            continue
        sources = []
        if WORKERS_FLAG in post:
            sources.append(f'{prefix}/orchestrator.yaml::test_command argv')
        if WORKERS_FLAG in shlex.split(_declared_addopts(member)):
            sources.append(f'{member}/pyproject.toml addopts')
        if sources:
            legs.append((prefix, member, ' and '.join(sources)))
    return legs


def test_every_pytest_leg_running_workers_declares_the_plugin(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """A module config that runs ``-n`` must select a member that DECLARES pytest-xdist.

    HEURISTIC 13 — a file has to make internal sense in isolation. A
    ``pyproject.toml`` whose addopts says ``-n auto`` while its dependency
    groups never mention the plugin that implements ``-n`` does not: read on its
    own it describes a configuration that cannot run.

    AND IT IS NOT MERELY UNTIDY — MEASURED on the task-5408 tree. From a venv
    lacking the plugin, ``uv run --directory dashboard python -c "import
    xdist"`` raises ModuleNotFoundError; after ``uv sync --all-packages`` it
    imports, and a later ``uv run --directory dashboard`` does not prune it. So
    an undeclared ``-n`` works only by ACCIDENT of
    ``verify_cold_preprovision_command``'s ``uv sync --all-packages`` dragging
    ANOTHER member's dev dependency into the one shared root ``.venv`` — the
    identical accident ``dark-factory-orchestrator.yaml`` already records for
    psutil/sampler. Lose that sync order and pytest exits 4 with
    ``unrecognized arguments: -n``: a latent red for every warm worktree, agent
    shell and contributor following CONTRIBUTING.md.

    THE MEMBER ASSERTED ABOUT is the one uv's PRE-anchor selector names, because
    that is the environment the plugin has to be installed into — not whichever
    module config happens to declare the command. For the ``scripts`` leg those
    differ: it selects ``shared``.
    """
    legs = _pytest_legs_carrying_workers(discover_module_configs())
    assert legs, (
        f'no discovered module config runs pytest with {WORKERS_FLAG} at all, so '
        'this guard has nothing to check and would pass vacuously. At least '
        'orchestrator and fused-memory have carried it in their addopts since '
        'before task 5408 — a zero here means the walk stopped seeing the real '
        'commands, not that the repo went serial'
    )

    undeclared = [
        (prefix, member, source)
        for prefix, member, source in legs
        if XDIST_DISTRIBUTION
        not in _dependency_group_requirements(member, DEFAULT_DEPENDENCY_GROUP)
    ]
    assert not undeclared, (
        'these pytest legs run with '
        f'{WORKERS_FLAG} while the member their uv selector names declares no '
        f'{XDIST_DISTRIBUTION} in its [dependency-groups] '
        f'{DEFAULT_DEPENDENCY_GROUP} group (task 5408):\n'
        + '\n'.join(
            f'  - module {prefix!r} selects member {member!r}; {WORKERS_FLAG} '
            f'comes from {source}'
            for prefix, member, source in undeclared
        )
        + f'\nAdd "{XDIST_DISTRIBUTION}>=3.5.0" to that member\'s '
        f'{DEFAULT_DEPENDENCY_GROUP} group and re-run a plain `uv lock`. '
        'Without the declaration the flag works only by ACCIDENT of '
        'verify_cold_preprovision_command\'s `uv sync --all-packages` pulling '
        'another member\'s dev dependency into the single root .venv — measured '
        'on the 5408 tree, from a venv lacking it `uv run --directory dashboard '
        'python -c "import xdist"` raises ModuleNotFoundError, and pytest then '
        f'exits 4 with `unrecognized arguments: {WORKERS_FLAG}`. '
        f'{DEFAULT_DEPENDENCY_GROUP} specifically: it is the group uv installs '
        'by default, so a declaration parked anywhere else never reaches the '
        'interpreter that runs the leg'
    )


# ---------------------------------------------------------------------------
# The `scripts` leg, the non-leakage claim, and the ruled exclusions (task 5408)
# ---------------------------------------------------------------------------

# The module config that defines the `scripts` verify leg. It has no
# pyproject.toml of its own, so its flags live on its test_command instead —
# see `test_root_addopts_never_leaks_the_parallel_flags` for why that is the
# SPOT rather than the root inifile.
SCRIPTS_MODULE_PREFIX = 'scripts'

# The members left SERIAL on purpose, each with the reason it was ruled out.
# A table rather than prose, so a later "helpful" widening fails against a
# message that states the reason instead of merely the expectation.
#
# EVERY REASON HERE IS A WALL-CLOCK MEASUREMENT, and that is a rule rather than
# a coincidence. An xdist-SAFETY worry cannot be a reason in this table, because
# `-n` does not in fact arrive only from addopts:
# `verify._with_pytest_numprocesses_str` injects
# `-n <config.verify_admission_pytest_n>` into every pytest leg at the roles
# `shared.verify_admission.is_gated_role` names (task, background), and
# dark-factory-orchestrator.yaml pins that knob to "8". So a member declaring no
# addopts ALREADY runs 8-way at those roles — and, declaring no `--dist`, under
# xdist's default `load`, where an @pytest.mark.xdist_group mark is ignored
# OUTRIGHT. Whatever cross-worker exposure such a suite has is therefore already
# live, and `-n auto --dist loadgroup` would REDUCE it rather than create it. A
# reason recorded below must be something the addopts change would actually
# change.
RULED_SERIAL_MEMBERS = {
    'shared': (
        'measured FASTER serial on the task-5408 tree — 44.9s serial against '
        '80.9s at -n 8 — so workers cost this suite wall-clock instead of '
        'buying it. That measurement is the WHOLE reason. Task 5408 still '
        'declared pytest-xdist for shared, because the `scripts` leg selects '
        "this member's environment, WITHOUT touching shared's own addopts. "
        '(tests/test_proc_group.py, tests/test_concurrency.py and '
        'tests/test_cli_invoke.py do want @pytest.mark.xdist_group pins, but '
        'that is deliberately NOT recorded here as a reason: per the note '
        'above those suites already run 8-way under the default `load` at task '
        'role, where the mark would be ignored anyway, so it is an exposure '
        'this exclusion does not remove)'
    ),
    'sampler': (
        "pyright, not pytest, is this module's LONG POLE — the figures are "
        'recorded once at PYRIGHT_LONG_POLE_MODULES below (~11s pyright '
        "against ~4s pytest), and sampler/orchestrator.yaml's own budget block "
        'carries the contended per-run spread that ~4s is the fastest of. '
        'Workers on a test leg that short buy nothing measurable, so task 5408 '
        "spent this module's budget on `pyright --threads 8` instead. NOT an "
        'oversight: sampler is the one module 5408 touched for its TYPE leg '
        'and deliberately left alone on its test leg'
    ),
}


# The module-config LEGS left serial on purpose, keyed by module PREFIX rather
# than by member. A second table and not a row in the one above, because these
# are a different kind of thing: a prefix here owns no pyproject.toml, so its
# flags would live on a test_command and its absence from the parallel set is
# invisible in every pyproject in the tree — exactly the case
# RULED_SERIAL_MEMBERS cannot record.
RULED_SERIAL_MODULE_LEGS = {
    'tests/scripts': (
        'NOT ruled out on a measurement, and this entry exists to say so '
        'rather than let silence read as a ruling. Task 5408 was scoped to '
        'four named modules and this is not one of them, so '
        "tests/scripts/orchestrator.yaml was never in that task's lock set. It "
        'is an acknowledged DIVERGENCE rather than a settled exclusion: this '
        "leg's test_command runs `tests/scripts/`, a strict SUBSET of the "
        "`scripts` leg's `tests/scripts/ scripts/tests/` targets, so as of "
        '5408 the same directory runs parallel on one leg and serial on the '
        'other at the same gate. Filed as residue 1 of task 5470 — when that '
        'lands, add the flags and DELETE this entry in the same commit'
    ),
}

# The two modules that were already parallel before task 5408 and are NOT this
# task's to re-spell. Task 3589 is the one that proposes replacing `auto` with a
# literal `-n 8` here, and reconciling that against
# dark-factory-orchestrator.yaml's verify_env pin is its whole subject.
TASK_3589_MEMBERS = ('orchestrator', 'fused-memory')


def _scripts_pytest_argv(module_configs: dict[str, ModuleConfig]) -> list[str]:
    """pytest's OWN argv in the ``scripts`` module's test_command.

    Post-anchor only, via ``vci.anchor_split``: the pre-anchor
    ``uv run --project shared`` tokens are uv's, and reading the whole segment
    would confuse an environment selector with one of pytest's own flags.
    """
    assert SCRIPTS_MODULE_PREFIX in module_configs, (
        f'{SCRIPTS_MODULE_PREFIX}/orchestrator.yaml is not discovered by the '
        'production config._discover_module_configs walk, so there is no '
        f'scripts verify leg to assert about. Discovered: '
        f'{sorted(module_configs)}'
    )
    command = module_configs[SCRIPTS_MODULE_PREFIX].test_command
    assert command, (
        f'{SCRIPTS_MODULE_PREFIX}/orchestrator.yaml declares no test_command, '
        'so the assertions below would be satisfied for the wrong reason'
    )
    segment = vci.required_segment(
        command, vci.PYTEST, label=f'{SCRIPTS_MODULE_PREFIX}/orchestrator.yaml test_command'
    )
    return vci.anchor_split(
        segment, vci.PYTEST, label=f'{SCRIPTS_MODULE_PREFIX}/orchestrator.yaml test_command'
    )[1]


def test_scripts_leg_carries_the_parallel_flags_on_its_test_command(
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """The ``scripts`` verify leg runs parallel, declared on its test_command.

    ``scripts/`` has no pyproject.toml, so it is the one parallelised leg whose
    flags cannot live in addopts. Its command runs from the worktree root over
    ``tests/scripts/ scripts/tests/``, so pytest's rootdir — and therefore its
    inifile — resolves to the REPO ROOT, not to anything scripts-specific. The
    single place that defines this leg is its ``test_command``, so that is where
    the flags belong (heuristic 11); the companion assertion in
    ``test_root_addopts_never_leaks_the_parallel_flags`` is what makes the other
    half of that claim checkable.

    With this in place the structural invariant in
    ``test_every_pytest_leg_running_workers_declares_the_plugin`` then requires
    ``shared`` — the member this command's ``--project`` selects — to declare
    pytest-xdist, even though ``shared``'s own suite stays serial.
    """
    argv = _scripts_pytest_argv(discover_module_configs())

    assert _flag_value(argv, WORKERS_FLAG) == WORKERS_VALUE, (
        f"{SCRIPTS_MODULE_PREFIX}/orchestrator.yaml's test_command passes pytest "
        f'{argv!r}, which does not carry `{WORKERS_FLAG} {WORKERS_VALUE}` '
        '(task 5408). This leg is the slowest in the gate — MEASURED on the 5408 '
        'tree at `-n 8 --dist loadgroup`: 5333 passed / 2 skipped in 117.71s, '
        'against 394-490s serial. The flags go HERE and not in the root '
        "pyproject.toml's addopts, because that inifile also governs every bare "
        'root-bound pytest run'
    )
    assert _flag_value(argv, DIST_FLAG) == DIST_VALUE, (
        f"{SCRIPTS_MODULE_PREFIX}/orchestrator.yaml's test_command passes pytest "
        f'{argv!r}, which does not carry `{DIST_FLAG} {DIST_VALUE}` (task 5408) — '
        f'the grouping discipline that must travel with {WORKERS_FLAG} so '
        '@pytest.mark.xdist_group is a guarantee rather than a hint'
    )


def test_root_addopts_never_leaks_the_parallel_flags() -> None:
    """The ROOT pyproject.toml's addopts must carry neither ``-n`` nor ``--dist``.

    THE SPOT CLAIM, MADE CHECKABLE. The root inifile is NOT the ``scripts``
    module's private config even though that leg resolves its rootdir there: the
    root pyproject's own comment records that it equally governs a bare
    root-bound ``pytest``, a ``-c pyproject.toml`` run, and any argument set
    spanning two subprojects. Confirmed empirically on the 5408 tree — a
    collect-only of the live scripts command reported
    ``5335/5345 tests collected (10 deselected)``, and only the root addopts'
    ``-m 'not smoke and not integration and not warm_lane_bash'`` can deselect
    anything there.

    So putting the flags here would parallelise all of those as a side effect of
    a change scoped to ONE verify leg — including runs in members that declare no
    pytest-xdist, where pytest would exit 4. This assertion holds from the
    start; it exists so a later widening fails loudly rather than passing as a
    convenience.
    """
    tokens = shlex.split(_declared_addopts('.'))
    offenders = [token for token in tokens if token in (WORKERS_FLAG, DIST_FLAG)]
    assert not offenders, (
        f"the ROOT pyproject.toml's addopts is {_declared_addopts('.')!r}, which "
        f'carries {offenders!r} (task 5408). Those flags belong on '
        f'{SCRIPTS_MODULE_PREFIX}/orchestrator.yaml::test_command, the single '
        'place that defines the scripts verify leg. This inifile is not that '
        "leg's private config: it is also what a bare root-bound `pytest`, a "
        '`-c pyproject.toml` run and any argument set spanning two subprojects '
        'read, so parallelising here reaches every one of those — including '
        'members that declare no pytest-xdist, where pytest exits 4 with '
        f'`unrecognized arguments: {WORKERS_FLAG}`'
    )


@pytest.mark.parametrize('member', sorted(RULED_SERIAL_MEMBERS))
def test_ruled_serial_members_stay_serial(member: str) -> None:
    """A member ruled out of the parallel set stays out, with the reason attached.

    An exclusion that is merely ABSENT is indistinguishable from an oversight,
    and the next reader completing the set is doing the obvious thing. Pinning it
    here is what makes the omission legible as a decision — and the failure
    message carries the REASON, so whoever trips it argues with the measurement
    rather than with the assertion.
    """
    tokens = shlex.split(_declared_addopts(member))
    offenders = [token for token in tokens if token in (WORKERS_FLAG, DIST_FLAG)]
    assert not offenders, (
        f"{member}/pyproject.toml's addopts now carries {offenders!r}, but this "
        f'member was RULED serial by task 5408, not overlooked: '
        f'{RULED_SERIAL_MEMBERS[member]}. Re-measure before widening, and record '
        'the measurement here rather than deleting this pin'
    )


@pytest.mark.parametrize('prefix', sorted(RULED_SERIAL_MODULE_LEGS))
def test_ruled_serial_module_legs_stay_serial(
    prefix: str,
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """A module LEG left serial stays serial, with its reason attached.

    The sibling of :func:`test_ruled_serial_members_stay_serial` for the
    configs that own no pyproject.toml. Same contract, different place to read:
    the flags would live on ``test_command``, so that is what is inspected —
    post-anchor, so uv's own wrapper tokens can never be mistaken for pytest's.

    Read the failure message before deleting this: an entry here is not always
    a RULING. The ``tests/scripts`` one records an acknowledged divergence with
    a follow-up task attached, so the correct response to tripping it is to
    finish that work and remove the entry, not to argue with the assertion.
    """
    module_configs = discover_module_configs()
    assert prefix in module_configs, (
        f'{prefix}/orchestrator.yaml is not discovered by the production '
        'config._discover_module_configs walk, so this exclusion is recorded '
        f'about a leg that no longer exists. Discovered: {sorted(module_configs)}'
    )
    command = module_configs[prefix].test_command
    assert command, (
        f'{prefix}/orchestrator.yaml declares no test_command, so the '
        'assertion below would be satisfied for the wrong reason'
    )
    label = f'{prefix}/orchestrator.yaml test_command'
    segment = vci.optional_token_segment(command, vci.PYTEST)
    assert segment is not None, (
        f'{label} no longer runs pytest at all, so the parallel flags this '
        'entry is about have nowhere to go. Re-read '
        f'RULED_SERIAL_MODULE_LEGS[{prefix!r}] and retire it if it is stale'
    )
    _, post = vci.anchor_split(segment, vci.PYTEST, label=label)

    offenders = [token for token in post if token in (WORKERS_FLAG, DIST_FLAG)]
    assert not offenders, (
        f'{label} now passes pytest {offenders!r}, but this leg was recorded '
        f'as staying serial by task 5408: {RULED_SERIAL_MODULE_LEGS[prefix]}. '
        'If that work has now been done, DELETE this entry in the same commit '
        'rather than loosening the assertion — the table is the record of '
        'which legs are deliberately out of the parallel set, and an entry '
        'that no longer holds is worse than no entry'
    )


@pytest.mark.parametrize('member', TASK_3589_MEMBERS)
def test_the_already_parallel_members_still_resolve_through_the_shared_knob(
    member: str,
) -> None:
    """orchestrator and fused-memory keep ``-n auto``; re-spelling it is task 3589's.

    Both were parallel before task 5408 and neither is this task's to re-spell.
    Task 3589 is the one that proposes a literal ``-n 8`` in these two addopts,
    and reconciling that against ``dark-factory-orchestrator.yaml``'s
    ``verify_env`` pin — deciding which layer is authoritative — is 3589's whole
    subject, stated in that key's own comment block.

    Pinned so the reconciliation is a deliberate edit here rather than a silent
    divergence: today all five parallel modules resolve their worker count
    through the same one knob, and the next A/B cut moves them together.
    """
    tokens = shlex.split(_declared_addopts(member))
    workers = _flag_value(tokens, WORKERS_FLAG)
    assert workers == WORKERS_VALUE, (
        f'{member}/pyproject.toml declares {WORKERS_FLAG} {workers!r}, not '
        f'{WORKERS_VALUE!r}. This member was already parallel before task 5408 '
        'and 5408 deliberately left it alone — replacing `auto` with a literal '
        'here is TASK 3589\'s, whose subject is reconciling that literal against '
        "dark-factory-orchestrator.yaml's verify_env pin and stating which layer "
        'is authoritative. If 3589 is what you are landing, update this pin in '
        'that commit and say so; do not let the five parallel modules diverge '
        'silently onto two different worker-count sources'
    )


# ---------------------------------------------------------------------------
# `pyright --threads 8` where pyright is the long pole (task 5408)
# ---------------------------------------------------------------------------

THREADS_FLAG = '--threads'
THREADS_VALUE = '8'

# The two modules whose type leg is SLOWER than their test leg, so threading
# pyright is where their wall-clock actually is: shared 65s pyright vs 45s
# pytest, sampler 11s vs 4s.
PYRIGHT_LONG_POLE_MODULES = ('shared', 'sampler')

# Whose configured runner spelling the EXECUTION probes below use. ONE module,
# not both: `--threads` acceptance is a property of the pyright BINARY, and both
# modules resolve the same pinned one, so a second probe would pay another uv +
# node startup for no new information.
THREADS_PROBE_MODULE = 'shared'

# The flag `--threads` is mutually exclusive with, and the exact refusal pyright
# emits. This pair is the negative control: it is what proves the positive probe
# is really exercising flag acceptance.
PYRIGHT_STATS_FLAG = '--stats'
PYRIGHT_THREADS_STATS_REFUSAL = "'threads' option cannot be used with 'stats' option"
PYRIGHT_USAGE_ERROR_RC = 4

# Co-locates the two subprocess probes on ONE xdist worker WHEREVER
# `--dist loadgroup` is in force, so the pair cannot land on two workers and
# run two concurrent pyright processes on an already-loaded host.
#
# THAT QUALIFIER IS LOAD-BEARING — the guarantee is not universal, because TWO
# registered module configs collect this file. The `scripts` leg passes
# `--dist loadgroup`, so the grouping holds there. The `tests/scripts` leg
# passes no `--dist` at all (RULED_SERIAL_MODULE_LEGS above), so at the roles
# `shared.verify_admission.is_gated_role` names, where verify injects
# `-n <config.verify_admission_pytest_n>`, it runs under xdist's default
# `load` — and there an xdist_group mark is ignored OUTRIGHT, so the two probes
# can land on different workers. Neither probe mutates shared state and each
# measured ~1.4s on the 5408 tree, so that case costs a small optimisation
# rather than correctness; it is stated rather than claimed away. Task 5470
# residue 1 is where `--dist loadgroup` would reach that leg.
PYRIGHT_PROBE_GROUP = 'pyright_threads_probe'

# Bounds the probe subprocess itself, INSIDE each probe's @pytest.mark.timeout,
# so a wedged pyright surfaces as a TimeoutExpired carrying its captured output
# rather than as pytest's axe carrying nothing.
#
# Those markers are 120s, which TIGHTENS rather than raises: the scripts verify
# leg passes --timeout=300 on the command line, and a probe that checks ONE
# trivially clean file has no business taking two minutes even behind a uv
# resolve and a node startup on a loaded 32-core host. Failing at 2 minutes with
# pyright's own captured output beats consuming the full 300s to say nothing.
PROBE_SUBPROCESS_TIMEOUT_SECS = 100

_PROBE_SRC = 'def f(x: int) -> int:\n    return x + 1\n'


def _pyright_argv(
    module_configs: dict[str, ModuleConfig], prefix: str
) -> tuple[list[str], list[str]]:
    """``(pre, post)`` tokens of *prefix*'s ``type_check_command``, split at the anchor.

    PRE is the ``uv run --directory <m>`` wrapper's; POST is pyright's own argv.
    Reading the whole segment instead would be the category error
    ``vci.anchor_split`` exists to prevent — ``--project`` before the anchor
    selects an ENVIRONMENT, after it redirects pyright's CONFIG FILE.
    """
    assert prefix in module_configs, (
        f'{prefix}/orchestrator.yaml is not discovered by the production '
        f'config._discover_module_configs walk. Discovered: {sorted(module_configs)}'
    )
    command = module_configs[prefix].type_check_command
    assert command, (
        f'{prefix}/orchestrator.yaml declares no type_check_command, so the '
        'assertions below would be satisfied for the wrong reason'
    )
    label = f'{prefix}/orchestrator.yaml type_check_command'
    segment = vci.required_segment(command, vci.PYRIGHT, label=label)
    return vci.anchor_split(segment, vci.PYRIGHT, label=label)


def _run_probe(
    pre_tokens: list[str], probe: pathlib.Path, *extra: str
) -> subprocess.CompletedProcess[str]:
    """Run the configured pyright spelling over *probe*, with *extra* flags appended.

    The wrapper tokens come from the module's OWN configured command rather than
    being spelled here, so this probe cannot drift into testing an invocation the
    gate does not use. Run from the repo root, which is where verify runs these
    commands from and what the wrapper's own ``--directory`` is relative to.
    """
    return subprocess.run(
        [*pre_tokens, vci.PYRIGHT, THREADS_FLAG, THREADS_VALUE, *extra, str(probe)],
        capture_output=True,
        text=True,
        timeout=PROBE_SUBPROCESS_TIMEOUT_SECS,
        cwd=str(REPO_ROOT),
        check=False,
    )


def _write_probe(tmp_path: pathlib.Path) -> pathlib.Path:
    """A trivially type-clean file under *tmp_path*, never a repo path.

    A repo path would make these probes go red on any unrelated type error
    anywhere in the checked tree — reporting "pyright rejects --threads" for a
    defect that has nothing to do with the flag.
    """
    probe = tmp_path / 'probe.py'
    probe.write_text(_PROBE_SRC, encoding='utf-8')
    return probe


@pytest.mark.parametrize('prefix', PYRIGHT_LONG_POLE_MODULES)
def test_pyright_long_pole_modules_pass_threads_eight(
    prefix: str,
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """``shared`` and ``sampler`` pass ``--threads 8`` to pyright.

    These are the two modules where pyright, not pytest, is the long pole —
    shared 65s vs 45s, sampler 11s vs 4s — so parallelising the type checker is
    where their wall-clock is. MEASURED on orchestrator: 2.0x, 240s -> 118s, 0
    errors either arm.

    THIS EDIT IS ALSO WHAT REACHES THE MERGE GATE'S LEG-2 UNSCOPED TYPE CHECK.
    ``merge_queue._run_unscoped_typechecks`` re-runs each module's OWN
    ``type_check_command`` verbatim (``dataclasses.replace(mc,
    test_command=None, lint_command=None)`` at role='merge'), so there is no
    separate leg-2 command to assert about — this one is it.

    Asserted on the POST-anchor argv, so uv's pre-anchor wrapper can never be
    mistaken for one of pyright's own flags.
    """
    _, post = _pyright_argv(discover_module_configs(), prefix)

    assert _flag_value(post, THREADS_FLAG) == THREADS_VALUE, (
        f"{prefix}/orchestrator.yaml's type_check_command passes pyright "
        f'{post!r}, which does not carry `{THREADS_FLAG} {THREADS_VALUE}` '
        '(task 5408). pyright is this module\'s LONG POLE, so this is where its '
        'merge-gate wall-clock is; the same command is what '
        'merge_queue._run_unscoped_typechecks re-runs for leg 2'
    )
    assert PYRIGHT_STATS_FLAG not in post, (
        f"{prefix}/orchestrator.yaml's type_check_command passes pyright both "
        f'{THREADS_FLAG} and {PYRIGHT_STATS_FLAG}, which pyright REFUSES: '
        f'{PYRIGHT_THREADS_STATS_REFUSAL} (exit {PYRIGHT_USAGE_ERROR_RC}). The '
        'two are mutually exclusive — pick one'
    )


@pytest.mark.xdist_group(PYRIGHT_PROBE_GROUP)
@pytest.mark.timeout(120)
def test_the_configured_pyright_actually_accepts_threads_eight(
    tmp_path: pathlib.Path,
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """RUN pyright with the flag. A string assertion cannot see a usage error.

    The task asked for this deliberately, and INV-10 is why: ``--threads`` is
    mutually exclusive with ``--stats``, so a config that carried both would
    satisfy every string assertion above while pyright exited on a usage error
    (``PYRIGHT_USAGE_ERROR_RC``) having checked nothing. Only running it tells
    the difference between an accepted flag and a refused one.

    Pinned at the 5408 measurement: on the repo's pyright 1.1.408,
    ``uv run --directory sampler pyright --threads 8 src/ tests/`` returned rc=0
    with "0 errors, 0 warnings, 0 informations" — the verdict is unchanged by the
    flag and the flag parses correctly before the positional targets.

    Its companion negative control is
    ``test_pyright_refuses_threads_together_with_stats``; the two share an
    ``xdist_group`` so they do not run as two concurrent pyright processes on
    any leg that passes ``--dist loadgroup`` — which is not every leg that
    collects this file. See ``PYRIGHT_PROBE_GROUP``.
    """
    pre, _ = _pyright_argv(discover_module_configs(), THREADS_PROBE_MODULE)
    result = _run_probe(pre, _write_probe(tmp_path))

    assert result.returncode == 0, (
        f'the configured {THREADS_PROBE_MODULE} pyright spelling exited '
        f'{result.returncode} with `{THREADS_FLAG} {THREADS_VALUE}` over a '
        'trivially type-clean probe file, so the flag the merge gate now passes '
        'is not accepted by the pinned pyright. Check the pyright version pin '
        f'(root package.json and uv.lock) before changing the flag.\n'
        f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    )


@pytest.mark.xdist_group(PYRIGHT_PROBE_GROUP)
@pytest.mark.timeout(120)
def test_pyright_refuses_threads_together_with_stats(
    tmp_path: pathlib.Path,
    discover_module_configs: Callable[[], dict[str, ModuleConfig]],
) -> None:
    """THE NEGATIVE CONTROL, and the only reason the positive probe is not vacuous.

    If this stops returning ``PYRIGHT_USAGE_ERROR_RC`` with pyright's refusal,
    then a usage error no longer distinguishes itself from success on this
    binary, and its sibling's ``returncode == 0`` stops proving that
    ``--threads`` was accepted rather than merely tolerated.

    It is also the executable form of the warning in both configs' comments: no
    one may add ``--stats`` (or anything else mutually exclusive with
    ``--threads``) to those type_check_commands.

    MEASURED on the 5408 tree: ``pyright --threads 8 --stats <file>`` -> rc=4,
    "'threads' option cannot be used with 'stats' option".
    """
    pre, _ = _pyright_argv(discover_module_configs(), THREADS_PROBE_MODULE)
    result = _run_probe(pre, _write_probe(tmp_path), PYRIGHT_STATS_FLAG)
    combined = result.stdout + result.stderr

    assert result.returncode == PYRIGHT_USAGE_ERROR_RC, (
        f'pyright exited {result.returncode}, not {PYRIGHT_USAGE_ERROR_RC}, for '
        f'`{THREADS_FLAG} {THREADS_VALUE} {PYRIGHT_STATS_FLAG}` — a combination '
        'it is supposed to refuse. Until this holds, the sibling probe\'s rc==0 '
        'does not distinguish an ACCEPTED flag from a tolerated one, and both '
        'configs\' "never add --stats" comments are unenforced.\n'
        f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    )
    assert PYRIGHT_THREADS_STATS_REFUSAL in combined, (
        f'pyright exited {PYRIGHT_USAGE_ERROR_RC} as expected but did not say '
        f'{PYRIGHT_THREADS_STATS_REFUSAL!r}, so the exit code may be reporting a '
        'DIFFERENT usage error and this control is no longer anchored to the '
        f'{THREADS_FLAG}/{PYRIGHT_STATS_FLAG} conflict.\n'
        f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'
    )
