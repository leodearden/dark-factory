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
import shlex
import tomllib

import pytest

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


def _addopts(member: str) -> str:
    """*member*'s declared ``[tool.pytest.ini_options].addopts``.

    Read with ``tomllib`` from the file on disk rather than through the
    orchestrator config loader, because addopts is pytest's OWN configuration:
    it is what governs a bare ``cd <member> && uv run pytest tests/`` as well as
    the verify leg, and no module config restates it.
    """
    pyproject = REPO_ROOT / member / 'pyproject.toml'
    assert pyproject.is_file(), (
        f'{member}/pyproject.toml does not exist, so every assertion below '
        'would be about a file this repo does not have'
    )
    data = tomllib.loads(pyproject.read_text(encoding='utf-8'))
    ini_options = data.get('tool', {}).get('pytest', {}).get('ini_options', {})
    addopts = ini_options.get('addopts', '')
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
