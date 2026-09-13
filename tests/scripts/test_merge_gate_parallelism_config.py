"""The merge gate's parallelism configuration, pinned where it is DECLARED.

Task 5408. Four of the nine discovered module configs ran their pytest leg
single-process at the merge role while ``orchestrator`` and ``fused-memory``
had carried ``-n auto --dist loadgroup`` in their own addopts for some time.
This guard pins the four that were brought onto that same footing, and pins
the exclusions that were left serial DELIBERATELY, so a later reader cannot
tell a ruled-out module from a forgotten one.

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
