"""Boundary gate H (task 2309): dispatch<->CLI verify-seam two-way boundary tests.

PRD: plans/laptop-warm-verify-flock-orphan-prd.md SS8 contract + SS9 boundary-test
table.  This is the LEAF integration gate for the internal dispatch (RemoteRunner)
<-> laptop `verify-merge` CLI seam: every test here DRIVES THE REAL SEAM (spawns a
real `orchestrator verify-merge` subprocess, acts on it, observes real
process/tree/escalation-queue/worktree state) rather than asserting on synthetic
inputs.

The "build" driven by each spawned verify-merge is a controllable /bin/bash
sleeper (crafted MergeVerifySpec.test_command) against a throwaway git repo --
never a real reify build.  verify.py's `_run_cmd` runs every module command via
``create_subprocess_shell(cmd, executable='/bin/bash', start_new_session=True)``,
so the sleeper faithfully reproduces the exact cargo/rustc session-escape the
watchdog/cancel tree-kill machinery must handle, with zero reify toolchain
dependency.

Fakes/helpers are reused from the sibling test_cli.py harness (established
cross-test-module import pattern -- see test_concurrent_verify_boundary.py) and
from orchestrator.verify_cancel (the SAME /proc walkers the production
watchdog/cancel kill paths use) -- single source of truth, zero divergent
ad-hoc process-tree parsing.

This module holds the SS9 rows and the AST guards over every test that
drives the harness.  The rest of the gate lives beside it:

* _verify_boundary_harness.py -- the real-subprocess harness, and the
  rationale for its discovery/assertion split.
* _verify_boundary_ceilings.py -- the timing ceilings and per-test budgets.
* test_verify_boundary_harness.py / test_verify_boundary_ceilings.py --
  their self-tests.
* test_verify_merge_env_overrides.py -- the CLI env-override seam tests.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import fcntl
import math
import os
import signal
import socket
import subprocess
import textwrap
import uuid
import warnings
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import _verify_boundary_ceilings
import pytest
from _gate_fixtures import gated_remote_runner
from _leader_stderr import assert_watchdog_fire_trigger, drain_exited_stderr
from _verify_boundary_ceilings import (
    ROW5_HOLDER_TEARDOWN_CEILING_SECS,
    ROW5_PER_TEST_TIMEOUT_SECS,
    ROW5_WAITER_COMPLETION_CEILING_SECS,
    ROW_DISCOVERY_CEILING_MAX_SECS,
    ROW_MARKER_CEILING_SECS,
    ROW_PER_TEST_TIMEOUT_SECS,
    ROW_TREE_KILL_CEILING_SECS,
    ROW_WATCHDOG_ENV,
    row_discovery_ceiling_secs,
)
from _verify_boundary_harness import (
    HOLDER_TAG_ENV,
    HeartbeatWriter,
    PreKillTree,
    apply_dispatcher_env,
    assert_pre_kill_tree_exited,
    fast_spec,
    git_push_succeeds,
    kill_holder_tree,
    loopback_ssh_run,
    sleeper_spec,
    snapshot_pre_kill_tree,
    spawn_ssh_heartbeat_dispatcher,
    spawn_verify_merge,
    stderr_tail,
    sweep_tagged,
    tagged_pids,
    verify_merge_argv,
    wait_for_marker,
    wait_for_marker_stable,
    wait_for_pgid_file,
    wait_pids_exited,
    wait_subtree_live,
    worktree_base_for,
    write_verify_config,
)
from click.testing import CliRunner
from escalation.queue import EscalationQueue
from test_cli import _setup_verify_repo  # noqa: F401 -- reused cross-module
from test_merge_queue_multihost_wiring import (
    _drive_runner_unavailable as _mq_drive_runner_unavailable,
)
from test_merge_queue_multihost_wiring import (  # noqa: F401 -- reused cross-module
    _make_config as _mq_make_config,
)
from test_merge_queue_multihost_wiring import (
    _make_git_ops_mock as _mq_make_git_ops_mock,
)
from test_merge_queue_multihost_wiring import (
    _make_merge_request as _mq_make_merge_request,
)

import orchestrator.cli as cli_module
from orchestrator.cli import main
from orchestrator.config import OrchestratorConfig
from orchestrator.merge_lane.worker import (
    _run_post_merge_verify,
    _verify_worktree_contention_sentinel,
)
from orchestrator.verify_cancel import (  # noqa: F401 -- reused by row tests
    WatchdogTrigger,
    lane_lock_path,
    pgid_file,
    read_lock_holder_pgid,
)
from orchestrator.verify_runner import (
    FLOCK_CONTENTION_CATEGORY,
    RunnerUnavailable,
    _default_ssh_heartbeat_run,
    is_flock_contention_failure,
    result_from_json,
)

# ---------------------------------------------------------------------------
# Task 4014 (review fix) -- widen the coupling guard from "the derived
# constant is big enough" to "every CALL SITE performing a scaled discovery
# wait is actually covered by the mark it runs under".
#
# The defect this pair exists to catch is not a wrong number, it is a guard
# whose SCOPE did not include a call site.  test_verify_boundary_ceilings.py::
# test_row_per_test_timeout_still_covers_the_max_discovery_ceiling pins
# ROW_PER_TEST_TIMEOUT_SECS against ROW_DISCOVERY_CEILING_MAX_SECS but never
# looks at WHO performs a discovery wait -- so Row 5, which relies on the bare
# (now load-scaled) defaults while carrying task 2921's unrelated LITERAL
# @pytest.mark.timeout(120), was invisible to it.  Two 120s-capable waits
# inside a 120s mark means a wedged Row 5 discovery gets truncated by
# pytest-timeout's thread-mode os._exit() of the whole xdist worker instead of
# raising the harness's self-diagnosing AssertionError (leader rc taxonomy +
# bounded stderr tail) -- the exact silent-truncation class the derivation
# chain exists to prevent.
#
# The sweep is ENUMERATION-FREE over call sites: it derives its call-site
# list from the AST of every test module that drives the harness
# (:data:`_HARNESS_DRIVING_TEST_MODULES`), so a future Row 6 or helper
# self-test registers itself with the guard simply by calling the helper.  A
# hand-maintained row list would have the identical failure mode the day one
# lands.  The MODULE list is explicit, not a directory walk, which would turn
# this guard into a whole-tree scanner; a new module that drives the harness
# must be added to it.  Self-parsing AST guards are an established idiom in
# this suite (test_marker_registration_drift.py,
# test_git_repo_isolation_guard.py, test_lock_release_single_writer_guard.py,
# test_event_loop_antipattern_guard.py) and cost nothing at runtime -- pure
# parse, no subprocess, no timing.
#
# Both sweeps live here, beside the rows, because their sibling
# test_row5_per_test_timeout_covers_its_own_bounded_work must read Row 5's
# live pytestmark.  The matchers key on BARE-NAME (ast.Name) calls to the
# harness helpers, so every swept module imports them by bare name and never
# calls them as module attributes.
# ---------------------------------------------------------------------------

#: Every test module that drives the harness -- the modules both sweeps parse.
#: A missing path raises FileNotFoundError, which is loud.
_HARNESS_DRIVING_TEST_MODULES: tuple[Path, ...] = (
    Path(__file__),
    Path(__file__).with_name('test_verify_boundary_harness.py'),
    Path(__file__).with_name('test_verify_merge_env_overrides.py'),
)


def _swept_test_functions() -> Iterator[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """``(module filename, node)`` for every ``test_*`` function the sweeps cover."""
    for path in _HARNESS_DRIVING_TEST_MODULES:
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith(
                'test_'
            ):
                yield path.name, node

#: The helpers whose ``timeout`` default now resolves to
#: :func:`row_discovery_ceiling_secs`, i.e. can spend up to
#: ROW_DISCOVERY_CEILING_MAX_SECS each.
_SCALED_DISCOVERY_HELPERS: frozenset[str] = frozenset(
    {'wait_for_pgid_file', 'wait_subtree_live'}
)

#: The seams that TOGETHER mean "this call never touches real /proc" -- see
#: :func:`_scaled_discovery_call_count`.  BOTH are required: either one alone
#: leaves the other half of the poll tick reading real /proc.
_DISCOVERY_SEAM_KWARGS: frozenset[str] = frozenset({'_probe_children', '_ppid_map'})

#: Last-resort stand-in for the pytest-timeout budget an unmarked test
#: inherits, used ONLY when pytest-timeout did not configure this run at all
#: (``-p no:timeout``, or a version that stops exposing its resolved value).
#: The live value is read by :func:`_inherited_timeout_budget_secs` --
#: hardcoding orchestrator/pyproject.toml's ``timeout = 60`` would make this
#: guard reason from exactly the kind of unreachable literal its closing
#: assertion bans.
_INHERITED_TIMEOUT_FALLBACK_SECS: float = 60.0


def _inherited_timeout_budget_secs(config) -> float:
    """The pytest-timeout budget an UNMARKED swept test inherits.

    Read from ``config._env_timeout`` -- the plugin's OWN resolved value, i.e.
    literally what ``pytest_timeout._get_item_settings`` hands an item carrying
    no ``timeout`` marker -- rather than duplicated as a literal, so a change to
    orchestrator/pyproject.toml cannot leave the sweep below reasoning from, and
    misreporting, a stale number.

    ``getini('timeout')`` alone will NOT do, MEASURED here on pytest-timeout
    2.4.0: it returns the STRING ``'60'`` (so float() coercion is mandatory),
    returns ``''`` on a root-bound run because the repo-root pyproject declares
    no ``timeout`` key at all, and -- the one that matters -- stays STALE at
    ``'60'`` under ``--timeout=300``, which this repo's per-module verify
    commands do pass.  ``get_env_settings()`` resolves ``--timeout`` ->
    ``PYTEST_TIMEOUT`` -> truthy ini and stores the winner as ``_env_timeout``;
    reading the winner cannot drift from the plugin's chain the way a local
    re-implementation of that chain would.

    ``None`` means no timeout was configured anywhere (measured: a root-bound
    run), so nothing can truncate an unmarked test and its budget really is
    unbounded.  Reporting the literal there would make this guard fail tests
    over a truncation that cannot happen.
    """
    if not hasattr(config, '_env_timeout'):
        warnings.warn(
            'pytest-timeout did not configure this run (no config._env_timeout), '
            'so the discovery-wait sweep is falling back to the duplicated '
            f'literal {_INHERITED_TIMEOUT_FALLBACK_SECS}s for unmarked tests',
            stacklevel=2,
        )
        return _INHERITED_TIMEOUT_FALLBACK_SECS
    resolved = config._env_timeout
    if resolved is None:
        return math.inf  # no timeout anywhere: an unmarked test cannot be truncated
    return float(resolved)


#: Anti-vacuity floor for the sweep below: an AST matcher that silently stops
#: matching is a guard reporting PASS while guarding nothing.
_KNOWN_DISCOVERY_ROWS: frozenset[str] = frozenset({
    'test_cancel_verify_tree_kills_under_live_watchdog',                  # Row 4
    'test_orchestrator_killed_mid_build_tree_killed_via_eof',             # Row 1
    'test_ssh_dropped_mid_build_tree_killed_via_eof_dispatcher_alive',    # Row 2
    'test_heartbeat_starved_hard_partition_tree_killed_via_timeout',      # Row 3
    'test_flock_contention_full_two_way_seam_blocks_and_escalates',       # Row 5
})


def _scaled_discovery_call_count(func_node) -> int:
    """How many LOAD-SCALED discovery waits *func_node* performs.

    A call to one of :data:`_SCALED_DISCOVERY_HELPERS` counts when its
    ``timeout=`` kwarg is ABSENT (the bare default, which now resolves to
    ``row_discovery_ceiling_secs()``) or is exactly
    ``row_discovery_ceiling_secs()``.  An explicit unscaled value (e.g.
    ``timeout=30.0``) does not count -- it cannot exceed what it names.

    A call passing the FULL private seam set (``_probe_children`` AND
    ``_ppid_map``) is EXEMPT, and that rule is load-bearing: a fully
    seam-injected call never touches real /proc and returns in microseconds at
    ``interval=0``.  Without it the three deterministic wait_subtree_live
    tests in test_verify_boundary_harness.py would each be told to carry a
    240s mark -- the guard would push the harness self-tests toward SLOWER
    tests rather than safer ones.

    Requiring BOTH is equally load-bearing, in the other direction.  A PARTIAL
    injection is not exempt, because it still reads real /proc every tick:
    ``_ppid_map=`` alone still runs the real :func:`read_direct_children`
    probe, and ``_probe_children=`` alone still runs the real
    ``read_ppid_map()`` full walk on every non-negative tick.  Either can also
    still run the poll loop to the full ``row_discovery_ceiling_secs()``
    deadline if the injected half never yields a descendant -- exactly the
    truncatable wait this sweep exists to catch.  No call site does this today;
    exempting on ANY seam would have left the hole open for the first one that
    does.
    """
    count = 0
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        if not (
            isinstance(node.func, ast.Name)
            and node.func.id in _SCALED_DISCOVERY_HELPERS
        ):
            continue
        # Superset, not intersection: the FULL seam set must be injected.
        if {kw.arg for kw in node.keywords} >= _DISCOVERY_SEAM_KWARGS:
            continue
        timeout_arg = next(
            (kw.value for kw in node.keywords if kw.arg == 'timeout'), None
        )
        if timeout_arg is None:
            count += 1  # bare default -> row_discovery_ceiling_secs()
        elif (
            isinstance(timeout_arg, ast.Call)
            and isinstance(timeout_arg.func, ast.Name)
            and timeout_arg.func.id == 'row_discovery_ceiling_secs'
        ):
            count += 1
    return count


def _timeout_mark_argument(func_node):
    """The single argument node of *func_node*'s ``@pytest.mark.timeout(...)``.

    Returns None when the function carries no such decorator (it then
    inherits the run's budget -- see :func:`_inherited_timeout_budget_secs`).
    """
    for decorator in func_node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == 'timeout'
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == 'mark'
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == 'pytest'
        ):
            return decorator.args[0] if decorator.args else None
    return None


def _timeout_mark_budget(arg, inherited: float) -> tuple[float, str | None]:
    """The budget a swept test's timeout-mark argument *arg* grants, and the
    offence that bans it (None for an allowed mark).

    *arg* is :func:`_timeout_mark_argument`'s result.  Allowed: no mark (the
    run's *inherited* budget), or the bare name of a budget defined in
    _verify_boundary_ceilings.py -- the one home of the derived budgets, so a
    mark resolves the same way whichever swept module carries it.  A banned
    LITERAL still grants its own value, so an under-budget literal is
    reported as such too; any other banned shape grants *inherited*.
    """
    if arg is None:
        return inherited, None
    if isinstance(arg, ast.Name):
        if hasattr(_verify_boundary_ceilings, arg.id):
            return float(getattr(_verify_boundary_ceilings, arg.id)), None
        return inherited, (
            f'names {arg.id}, which is not a budget defined in _verify_boundary_ceilings.py'
        )
    if isinstance(arg, ast.Constant):
        granted = float(arg.value) if isinstance(arg.value, int | float) else inherited
        return granted, (
            'pins a LITERAL, which no re-derivation of the discovery ceiling can reach into'
        )
    return inherited, 'is not the bare name of a budget defined in _verify_boundary_ceilings.py'


def test_row5_per_test_timeout_covers_its_own_bounded_work():
    """Row 5's pytest timeout must cover ITS bounded worst case, and be DERIVED.

    Row 5 does not share Rows 1-4's shape -- no watchdog window, no kill
    confirmation -- so it cannot share their constant.  Its bounded terms are: two discovery waits (both on the
    load-scaled ceiling), the marker wait, the waiter subprocess's completion
    ceiling, and the ``finally`` block's TWO holder teardowns (waiter then
    holder).

    The marker term is counted TWICE on purpose:
    :func:`wait_for_marker_stable` spends its ``timeout`` on the
    :func:`wait_for_marker` existence gate and then a FRESH ``timeout`` on the
    settle loop, so its worst case is 2x, not 1x -- a term the old literal 120
    hid entirely.

    The second half reads the mark ACTUALLY APPLIED to the live function
    object rather than trusting the constant in isolation, so the derivation
    and the decorator cannot diverge.
    """
    required = (
        2 * ROW_DISCOVERY_CEILING_MAX_SECS
        + 2 * ROW_MARKER_CEILING_SECS
        + ROW5_WAITER_COMPLETION_CEILING_SECS
        + 2 * ROW5_HOLDER_TEARDOWN_CEILING_SECS
    )
    assert required <= ROW5_PER_TEST_TIMEOUT_SECS, (
        f'ROW5_PER_TEST_TIMEOUT_SECS ({ROW5_PER_TEST_TIMEOUT_SECS}) does not cover '
        f"Row 5's bounded worst case ({required}) -- both discovery waits at the "
        f'MAX ceiling, the marker existence gate AND its settle loop, the waiter '
        f"subprocess's completion ceiling, and both holder teardowns"
    )

    applied = [
        mark
        for mark in getattr(
            test_flock_contention_full_two_way_seam_blocks_and_escalates,
            'pytestmark',
            [],
        )
        if mark.name == 'timeout'
    ]
    assert len(applied) == 1, (
        f'expected exactly one @pytest.mark.timeout on Row 5, got {applied!r}'
    )
    assert applied[0].args[0] == ROW5_PER_TEST_TIMEOUT_SECS, (
        f'Row 5 runs under a {applied[0].args[0]}s mark but its derivation says '
        f'{ROW5_PER_TEST_TIMEOUT_SECS}s -- the decorator and the derived constant '
        f'have diverged, so the derivation is guarding nothing'
    )


def test_every_scaled_discovery_wait_is_covered_by_its_test_timeout_mark(request):
    """No test may perform more discovery waiting than its own mark allows.

    Enumeration-free sweep of every :data:`_HARNESS_DRIVING_TEST_MODULES`
    AST (see the banner above).  For every ``test_*`` function it counts the
    LOAD-SCALED discovery waits the source actually contains, resolves the
    EFFECTIVE pytest-timeout budget that test runs under -- from the LIVE
    config for unmarked tests, never from a duplicated ini literal -- and
    asserts the budget covers those waits at the widest ceiling they can
    reach.

    It additionally BANS, on any test performing such a wait, every mark that
    is not the bare name of a budget defined in _verify_boundary_ceilings.py
    (:func:`_timeout_mark_budget`) -- most pointedly a LITERAL.  That kills
    the root cause directly rather than the symptom: task
    2921's ``@pytest.mark.timeout(120)`` was a literal that no re-derivation
    could reach into, which is precisely why task 4014's load-scaled
    discovery ceiling could not propagate to Row 5.
    """
    swept: dict[tuple[str, str], tuple[int, ast.AST]] = {}
    for module, node in _swept_test_functions():
        n_scaled = _scaled_discovery_call_count(node)
        if n_scaled:
            swept[(module, node.name)] = (n_scaled, node)

    matched = {name for _, name in swept}
    missing = sorted(_KNOWN_DISCOVERY_ROWS - matched)
    assert not missing, (
        f'the AST sweep matched no scaled discovery wait in {missing} -- a matcher '
        f'that silently stops matching is a guard reporting PASS while guarding '
        f'nothing (matched: {sorted(matched)})'
    )

    inherited = _inherited_timeout_budget_secs(request.config)
    under_budget: list[str] = []
    banned_marks: list[str] = []
    for (module, name), (n_scaled, func_node) in sorted(swept.items()):
        arg = _timeout_mark_argument(func_node)
        effective, offence = _timeout_mark_budget(arg, inherited)
        if offence is not None:
            banned_marks.append(
                f'{module}::{name}: @pytest.mark.timeout({ast.unparse(arg)}) {offence}'
            )
        required = n_scaled * ROW_DISCOVERY_CEILING_MAX_SECS
        if effective < required:
            under_budget.append(
                f'{module}::{name}: {n_scaled} scaled discovery wait(s) needing {required}s '
                f'but running under a {effective}s budget'
            )

    assert not under_budget, (
        'these tests can spend more time in discovery waits than their pytest '
        'timeout allows, so a wedged discovery is truncated by pytest-timeout\'s '
        'thread-mode os._exit() of the xdist worker instead of raising the '
        'self-diagnosing AssertionError:\n  ' + '\n  '.join(under_budget)
    )
    assert not banned_marks, (
        'these tests perform a load-scaled discovery wait under a pytest timeout '
        'mark that is not a budget derived in _verify_boundary_ceilings.py -- '
        'name one defined there instead:\n  '
        + '\n  '.join(banned_marks)
    )


def test_timeout_mark_budget_names_the_offence_of_each_banned_mark():
    """_timeout_mark_budget grants each mark shape its budget, and names what
    is wrong with a banned one.

    A Name that is not a budget defined in _verify_boundary_ceilings.py --
    e.g. _orch_helpers' VERIFY_CLI_PER_TEST_TIMEOUT, which
    test_verify_merge_env_overrides.py marks its tests with -- is a different
    offence from a LITERAL and must not be reported as one, or the next author
    hunts for a literal that is not there.
    """

    def mark_argument(source: str) -> ast.expr:
        return ast.parse(source, mode='eval').body

    inherited = 60.0
    assert _timeout_mark_budget(None, inherited) == (inherited, None)
    assert _timeout_mark_budget(mark_argument('ROW_PER_TEST_TIMEOUT_SECS'), inherited) == (
        float(ROW_PER_TEST_TIMEOUT_SECS),
        None,
    )

    budget, offence = _timeout_mark_budget(
        mark_argument('VERIFY_CLI_PER_TEST_TIMEOUT'), inherited
    )
    assert budget == inherited
    assert offence is not None
    assert 'VERIFY_CLI_PER_TEST_TIMEOUT' in offence
    assert '_verify_boundary_ceilings.py' in offence
    assert 'LITERAL' not in offence

    budget, offence = _timeout_mark_budget(
        mark_argument('2 * ROW_PER_TEST_TIMEOUT_SECS'), inherited
    )
    assert budget == inherited
    assert offence is not None
    assert '_verify_boundary_ceilings.py' in offence
    assert 'LITERAL' not in offence

    budget, offence = _timeout_mark_budget(mark_argument('120'), inherited)
    assert budget == 120.0
    assert offence is not None
    assert 'LITERAL' in offence


#: Anti-vacuity floor for the sweep below: an AST matcher that silently stops
#: matching is a guard reporting PASS while guarding nothing.  "At minimum"
#: -- the sweep may match more holder teardowns than this names; the
#: assertion below only requires this set to be a SUBSET of what was matched.
#: The two env-override tests live in test_verify_merge_env_overrides.py, so
#: their presence here is what proves that sibling module is actually swept.
_KNOWN_HOLDER_TEARDOWN_ROWS: frozenset[str] = frozenset({
    'test_flock_contention_full_two_way_seam_blocks_and_escalates',       # Row 5
    'test_live_verify_merge_holds_lane_lock_real_subprocess',             # lane lock
    'test_flock_wait_env_override_speeds_up_contention_result',           # flock env seam
    'test_watchdog_timeout_env_override_fires_fast_without_heartbeat',    # watchdog env seam
})


def _spawn_verify_merge_bound_names(func_node) -> set[str]:
    """Local names *func_node* binds via a bare ``X = spawn_verify_merge(...)`` call."""
    names: set[str] = set()
    for node in ast.walk(func_node):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == 'spawn_verify_merge'
        ):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
    return names


def _bare_kill_offenders(func_node, holder_names: set[str]) -> list[str]:
    """``"func:lineno"`` for every swept holder's raw-kill spelling called directly
    in a ``try`` handler/finalbody that does NOT also call ``kill_holder_tree``.

    Three raw-kill spellings are matched, each a way a future teardown
    could reintroduce the exact orphan task 4092 fixed: ``<holder>.kill()``,
    ``<holder>.terminate()``, and an ``os.kill``/``os.killpg`` call whose
    argument subtree mentions a swept holder name -- most pointedly the
    ``os.killpg(os.getpgid(holder.pid), SIGKILL)`` one-liner
    ``kill_holder_tree``'s own docstring calls out as the dangerous thing a
    future author would reach for instead.

    Each ``finally``/``except`` body is checked independently: a raw kill
    in one body is not excused by a ``kill_holder_tree`` call living in a
    DIFFERENT body of the same ``try``.
    """
    offenders: list[str] = []
    for try_node in ast.walk(func_node):
        if not isinstance(try_node, ast.Try):
            continue
        bodies = [try_node.finalbody] + [handler.body for handler in try_node.handlers]
        for body in bodies:
            if not body:
                continue
            bare_kill_lines: list[int] = []
            has_tree_kill = False
            for stmt in body:
                for sub in ast.walk(stmt):
                    if not isinstance(sub, ast.Call):
                        continue
                    if (
                        isinstance(sub.func, ast.Attribute)
                        and sub.func.attr in ('kill', 'terminate')
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id in holder_names
                    ) or (
                        isinstance(sub.func, ast.Attribute)
                        and isinstance(sub.func.value, ast.Name)
                        and sub.func.value.id == 'os'
                        and sub.func.attr in ('kill', 'killpg')
                        and any(
                            isinstance(name_node, ast.Name) and name_node.id in holder_names
                            for arg in sub.args
                            for name_node in ast.walk(arg)
                        )
                    ):
                        bare_kill_lines.append(sub.lineno)
                    elif isinstance(sub.func, ast.Name) and sub.func.id == 'kill_holder_tree':
                        has_tree_kill = True
            if bare_kill_lines and not has_tree_kill:
                offenders.extend(
                    f'{func_node.name}:{lineno}' for lineno in bare_kill_lines
                )
    return offenders


def _untorn_down_holders(func_node, holder_names: set[str]) -> list[str]:
    """``"func:holder"`` for every swept holder never passed to a
    ``kill_holder_tree(...)`` call in a ``try`` handler/finalbody anywhere
    in *func_node*.

    Scoped to the WHOLE FUNCTION, across ALL of its ``try`` statements --
    deliberately unlike :func:`_bare_kill_offenders`, which only excuses a
    raw kill found in the SAME body of the SAME ``try`` -- because the
    failure this catches is the ABSENCE of any teardown, which no per-``try``
    rule can see: a holder torn down in a completely different ``try`` from
    where it was bound is still torn down.

    Only a call inside a ``finally``/``except`` body counts as teardown: a
    ``kill_holder_tree(holder)`` reachable only along the happy path -- as
    the last statement of a ``try`` BODY, or a bare statement outside any
    ``try`` -- would not run if an earlier statement raises, which is
    exactly the failure this check exists to catch, so it must not be
    credited.  Only a DIRECT ``ast.Name`` argument of the call (positional
    or keyword value) counts as teardown; the argument subtree is
    deliberately not walked, so an indirection such as
    ``kill_holder_tree(procs[0])`` is reported as a finding rather than
    silently excused.  Keyword values are included because
    ``kill_holder_tree``'s first parameter is named ``proc``, making
    ``kill_holder_tree(proc=holder)`` a legal spelling.
    """
    torn_down: set[str] = set()
    for try_node in ast.walk(func_node):
        if not isinstance(try_node, ast.Try):
            continue
        bodies = [try_node.finalbody] + [handler.body for handler in try_node.handlers]
        for body in bodies:
            for stmt in body:
                for node in ast.walk(stmt):
                    if not (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == 'kill_holder_tree'
                    ):
                        continue
                    for arg in node.args:
                        if isinstance(arg, ast.Name):
                            torn_down.add(arg.id)
                    for keyword in node.keywords:
                        if isinstance(keyword.value, ast.Name):
                            torn_down.add(keyword.value.id)
    return sorted(f'{func_node.name}:{name}' for name in holder_names - torn_down)


def test_every_real_subprocess_holder_teardown_uses_the_tree_killer():
    """Every spawn_verify_merge-bound holder's teardown goes through kill_holder_tree.

    Enumeration-free sweep of every :data:`_HARNESS_DRIVING_TEST_MODULES`
    AST (see the banner above
    :func:`test_every_scaled_discovery_wait_is_covered_by_its_test_timeout_mark`).
    For every ``test_*`` function, collects the local names bound from a
    bare ``X = spawn_verify_merge(...)`` call, then makes TWO independent
    claims about every such holder:

    (i) no ``try`` handler/finalbody in that function calls
    ``<holder>.kill()``, ``<holder>.terminate()``, or ``os.kill``/
    ``os.killpg`` (with a swept holder name anywhere in the argument
    subtree) directly unless the SAME body also calls ``kill_holder_tree``
    -- any of those raw-kill spellings SIGKILLs/terminates only the leader,
    orphaning any session-escaped descendant (e.g. one of verify.py's
    ``start_new_session`` build commands) for the rest of its natural life
    (task 4092); and

    (ii) every swept holder is passed to a ``kill_holder_tree(...)`` call
    inside a ``try``'s ``finally``/``except`` body SOMEWHERE in that function
    (:func:`_untorn_down_holders`) -- (i) only judges the quality of a
    teardown that exists, so a holder with no exception-safe teardown
    anywhere satisfies (i) vacuously; (ii) closes that silent-pass hole
    (task 4946), including the narrower case of a ``kill_holder_tree`` call
    that is textually present but reachable only along the happy path (e.g.
    the last statement of a ``try`` body, never its ``finally``/``except``).

    Carries the sibling sweep's anti-vacuity discipline: a matcher that
    silently stops matching is a guard reporting PASS while guarding
    nothing, so :data:`_KNOWN_HOLDER_TEARDOWN_ROWS` anchors the functions
    the sweep must always find a spawn_verify_merge-bound holder in,
    independent of whether their teardown currently passes or fails.
    """
    swept: dict[tuple[str, str], set[str]] = {}
    offenders: list[str] = []
    untorn: list[str] = []
    for module, node in _swept_test_functions():
        holder_names = _spawn_verify_merge_bound_names(node)
        if not holder_names:
            continue
        swept[(module, node.name)] = holder_names
        offenders.extend(
            f'{module}::{finding}' for finding in _bare_kill_offenders(node, holder_names)
        )
        untorn.extend(
            f'{module}::{finding}' for finding in _untorn_down_holders(node, holder_names)
        )

    matched = {name for _, name in swept}
    missing = sorted(_KNOWN_HOLDER_TEARDOWN_ROWS - matched)
    assert not missing, (
        f'the AST sweep matched no spawn_verify_merge-bound holder in {missing} '
        f'-- a matcher that silently stops matching is a guard reporting PASS '
        f'while guarding nothing (matched: {sorted(matched)})'
    )

    assert not offenders, (
        'these holder teardown(s) call .kill()/.terminate()/os.kill()/'
        'os.killpg() directly instead of routing through kill_holder_tree, '
        'so a session-escaped descendant (e.g. a verify.py '
        "start_new_session build command) survives the leader's own death "
        'as an orphan:\n  ' + '\n  '.join(offenders)
    )

    assert not untorn, (
        'these spawn_verify_merge-bound holders are never passed to '
        "kill_holder_tree from inside their test's try finally/except "
        'bodies, so a raised communicate()/wait() timeout -- or any '
        'failing assertion before the teardown -- leaks the leader AND '
        'every session-escaped descendant verify.py started under it; a '
        'holder with no exception-safe teardown must be a finding, not a '
        'silent pass (the raw-kill sweep above only sees teardowns that '
        'exist):\n  ' + '\n  '.join(untorn)
    )


def test_untorn_down_holders_flags_only_holders_never_passed_to_kill_holder_tree():
    """_untorn_down_holders flags a swept holder iff it never reaches
    kill_holder_tree from inside a try's finally/except body.

    Deterministic AST-only cases (``ast.parse`` over a ``textwrap.dedent(...)``
    source, zero subprocesses, zero timing).  ``holder_names`` is supplied
    directly to each case, so this test pins ONLY the new helper, not
    :func:`_spawn_verify_merge_bound_names`.

    Case (d) is load-bearing: a matcher that merely asks "does this function
    call kill_holder_tree anywhere?" would pass cases (a)-(c) AND vacuously
    green Row 5 (which already calls ``kill_holder_tree(holder, ...)``) while
    catching nothing -- (d) is the only case that kills that implementation,
    because ``other``'s teardown must not excuse ``holder``'s absence.

    Case (f) is equally load-bearing, for the finally/except requirement:
    a matcher that credits a ``kill_holder_tree`` call reachable anywhere in
    the function -- including the happy path, e.g. the last statement of a
    ``try`` BODY -- would pass every other case here while still leaving a
    holder unprotected against an earlier statement in that same body
    raising before the call is reached.

    Cases (g) and (h) pin two properties this helper's own docstring already
    claims but that no case previously exercised: (g) that an indirect
    argument (not a direct ``ast.Name``) is reported as a finding rather
    than silently excused, and (h) that the whole-function scope credits a
    teardown living in a completely different ``try`` from where the holder
    was bound.
    """

    def parse_func(source: str) -> ast.FunctionDef:
        tree = ast.parse(textwrap.dedent(source))
        (func,) = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        return func

    # (a) holder bound, no kill_holder_tree call at all.
    func_a = parse_func("""
        def f():
            holder = spawn_verify_merge()
    """)
    assert _untorn_down_holders(func_a, {'holder'}) == ['f:holder'], (
        'a holder with NO kill_holder_tree call anywhere in its function '
        'must be flagged'
    )

    # (b) holder passed positionally in a finally:.
    func_b = parse_func("""
        def f():
            holder = spawn_verify_merge()
            try:
                do_work()
            finally:
                kill_holder_tree(holder, timeout=5)
    """)
    assert _untorn_down_holders(func_b, {'holder'}) == [], (
        'a holder passed positionally to kill_holder_tree in a finally: '
        'must not be flagged'
    )

    # (c) holder passed inside a nested `if holder is not None:` guard
    # within the finally: -- pins that the helper walks nested bodies; this
    # is the exact shape step-5 uses for Row 5's `waiter`.
    func_c = parse_func("""
        def f():
            holder = None
            try:
                holder = spawn_verify_merge()
                do_work()
            finally:
                if holder is not None:
                    kill_holder_tree(holder, timeout=5)
    """)
    assert _untorn_down_holders(func_c, {'holder'}) == [], (
        'a holder passed to kill_holder_tree inside a nested if-guard '
        'within the finally: must not be flagged -- the helper must walk '
        'nested bodies'
    )

    # (d) ANTI-VACUITY (load-bearing, do not drop): two holders, only
    # `other` passed to kill_holder_tree -- `holder` must still be flagged.
    func_d = parse_func("""
        def f():
            holder = spawn_verify_merge()
            other = spawn_verify_merge()
            try:
                do_work()
            finally:
                kill_holder_tree(other, timeout=5)
    """)
    assert _untorn_down_holders(func_d, {'holder', 'other'}) == ['f:holder'], (
        "kill_holder_tree(other) must not excuse holder's absence -- a "
        'matcher that merely checks "is kill_holder_tree called anywhere?" '
        'would wrongly return [] here'
    )

    # (e) keyword spelling: kill_holder_tree's first parameter is named `proc`.
    func_e = parse_func("""
        def f():
            holder = spawn_verify_merge()
            try:
                do_work()
            finally:
                kill_holder_tree(proc=holder, timeout=5)
    """)
    assert _untorn_down_holders(func_e, {'holder'}) == [], (
        'a holder passed as the proc= keyword must not be flagged -- '
        "kill_holder_tree's first parameter is named proc"
    )

    # (f) LOAD-BEARING for the finally/except requirement: kill_holder_tree
    # is called with `holder`, but only as the last statement of the try
    # BODY itself -- never inside a finally/except.  An earlier statement in
    # that same try body raising would skip this call entirely, which is
    # exactly the failure this whole check exists to catch, so it must
    # still be flagged even though a kill_holder_tree(holder) call is
    # textually present in the function.
    func_f = parse_func("""
        def f():
            holder = spawn_verify_merge()
            try:
                do_work()
                kill_holder_tree(holder, timeout=5)
            except RuntimeError:
                pass
    """)
    assert _untorn_down_holders(func_f, {'holder'}) == ['f:holder'], (
        'a kill_holder_tree call reachable only along the happy path (the '
        'last statement of a try BODY, not its finally/except) must still '
        'be flagged -- an earlier statement in the same try body raising '
        'would skip it entirely'
    )

    # (g) indirection defeats the matcher: kill_holder_tree(procs[0]) does
    # not count as tearing down `holder`, even though `procs` holds it at
    # runtime -- only a DIRECT ast.Name argument counts, so an indirection
    # is reported as a finding rather than silently excused.
    func_g = parse_func("""
        def f():
            holder = spawn_verify_merge()
            procs = [holder]
            try:
                do_work()
            finally:
                kill_holder_tree(procs[0], timeout=5)
    """)
    assert _untorn_down_holders(func_g, {'holder'}) == ['f:holder'], (
        'kill_holder_tree(procs[0]) is an indirection, not a direct '
        'ast.Name argument -- it must not excuse holder, even though procs '
        'holds it at runtime'
    )

    # (h) whole-function scope: holder is bound before one try and torn down
    # in a COMPLETELY DIFFERENT, later try's finally -- unlike
    # _bare_kill_offenders (which is per-try), this check must not flag it.
    func_h = parse_func("""
        def f():
            holder = spawn_verify_merge()
            try:
                do_work()
            except RuntimeError:
                pass
            try:
                do_more_work()
            finally:
                kill_holder_tree(holder, timeout=5)
    """)
    assert _untorn_down_holders(func_h, {'holder'}) == [], (
        'a holder torn down in a completely different try from where it '
        'was bound is still torn down -- this check is scoped to the '
        'whole function, not per-try'
    )


@pytest.mark.timeout(ROW_PER_TEST_TIMEOUT_SECS)
def test_spawned_verify_merge_build_carries_the_holder_tag(tmp_path):
    """The real CLI's session-escaped build inherits its holder's tag.

    Anti-vacuity anchor for kill_holder_tree's tag sweep: it only reaches an
    orphaned build because verify.py::_target_subprocess_env is a denylist
    (venv/uv/ORCH_*) that passes the tag through.  An allowlist there would
    silently disable the sweep; this fails loudly instead.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=False)
    build_pgf = tmp_path / 'build.pgid'

    holder = spawn_verify_merge(
        sha=head_sha,
        spec=sleeper_spec(300.0, build_pgid_file=build_pgf),
        cfg_file=cfg_file,
    )
    try:
        build_pid = wait_for_pgid_file(build_pgf, timeout=row_discovery_ceiling_secs())
        assert os.getsid(build_pid) == build_pid, (
            f'harness bug: build pid={build_pid} must lead its own session '
            f'(verify.py::_run_cmd start_new_session), or this test is not '
            f'exercising the escape the sweep exists for'
        )
        assert build_pid in tagged_pids(holder.tag), (
            f'the build pid={build_pid} spawned by holder pid={holder.pid} '
            f'does not carry {HOLDER_TAG_ENV}={holder.tag} -- the target-env '
            f'scrub dropped it, so kill_holder_tree cannot sweep an orphaned build'
        )
    finally:
        kill_holder_tree(holder, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)
        for pipe in (holder.stdout, holder.stderr):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()


# ---------------------------------------------------------------------------
# Task 2309 step-3/4 -- SS9 Row 6: normal warm path (also the harness smoke
# test).  Real dispatcher (live heartbeat), run twice on the same
# worktree_base.  Exercises SS8.1's happy-path invariant (heartbeat must not
# alter VerifyResult) + PRD outcome #1 (retained target/ across merges).
# ---------------------------------------------------------------------------


def test_normal_warm_path_reuses_fixed_worktree_twice_no_escalation(tmp_path, monkeypatch):
    """SS9 Row 6: two consecutive warm verify-merge runs, real live-heartbeat dispatcher.

    A knob-on config + a --request-id'd verify-merge is driven by the REAL
    dispatcher (verify_runner._default_ssh_heartbeat_run, called directly --
    in-process asyncio, not a wrapper subprocess -- so its concurrent
    heartbeat-writer task sends real heartbeats down the child's real stdin
    pipe every 0.2s; see PRD SS8.1) run TWICE against the SAME worktree_base.

    Asserts:

    * both runs use the FIXED <worktree_base>/_merge-verify path -- no
      ephemeral _merge-<uuid> dir is ever created (PRD SS8 eta);
    * target/warm.marker (written by the first run's build) survives the
      second run's reset_persistent_merge_worktree -- reap_build_artifact_dirs
      retention (PRD SS10 invariant 1: source bit-identical to fresh checkout,
      build-cache dirs retained for warmth);
    * both VerifyResults are passed=True, category=='passed' (rc==0's
      ``_classify_failure`` value -- verify.py:643-644 -- not one of the
      failure sentinels like FLOCK_CONTENTION_CATEGORY), contention is None --
      structurally unchanged by the live heartbeat;
    * is_flock_contention_failure(result) is False for both -- the REAL beta
      discriminant predicate that gates _run_post_merge_verify's escalation
      branch (task 2307 beta) never fires on this happy path, so nothing
      downstream would file a born-at-L2 escalation.  (A CLI-only harness
      has no merge_queue in the loop to observe an EscalationQueue directly
      -- Row 5 (step-13/14) drives that consumer explicitly; this is the
      producer-side proof that the happy path never emits the discriminant
      the consumer keys on.)
    * the watchdog never fires -- both runs exit 0 via normal completion,
      not the watchdog's non-zero self-kill exit.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    worktree_base = worktree_base_for(repo)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=True)

    apply_dispatcher_env(monkeypatch)

    def run_once(request_id: str):
        argv = verify_merge_argv(
            sha=head_sha, spec=fast_spec(), cfg_file=cfg_file, request_id=request_id,
        )
        rc, out, err = asyncio.run(
            _default_ssh_heartbeat_run(argv, heartbeat_interval=0.2)
        )
        assert rc == 0, (
            f'verify-merge (request_id={request_id!r}) exited {rc} (watchdog '
            f'fired?) stderr={err[:2000]!r}'
        )
        return result_from_json(out)

    result_1 = run_once('row6-run-1')
    result_2 = run_once('row6-run-2')

    persistent_wt = worktree_base / '_merge-verify'
    assert persistent_wt.is_dir(), f'fixed warm worktree missing: {persistent_wt}'

    leaked_ephemeral = sorted(
        p.name for p in worktree_base.iterdir()
        # An ephemeral leak is a _merge-<uuid> *directory*; the shared
        # <lane_dir>.lock flock sibling (<worktree_base>/_merge-verify.lock,
        # task 2685) is a regular file and is expected -- require is_dir().
        if p.is_dir() and p.name.startswith('_merge-') and p.name != '_merge-verify'
    )
    assert leaked_ephemeral == [], (
        f'ephemeral _merge-<uuid> dir(s) leaked (warm path should never '
        f'create one): {leaked_ephemeral}'
    )

    target_dir = persistent_wt / 'target'
    marker = target_dir / 'warm.marker'
    assert marker.exists(), (
        f'target/warm.marker missing after both runs -- reap_build_artifact_dirs '
        f'retention did not survive reset_persistent_merge_worktree: {marker}'
    )
    assert any(target_dir.iterdir()), f'target/ retained but empty: {target_dir}'

    for n, result in ((1, result_1), (2, result_2)):
        assert result.passed is True, f'run {n}: expected passed=True, got {result!r}'
        assert result.category == 'passed', (
            f'run {n}: expected the happy-path category (rc==0 -> "passed"), got '
            f'{result.category!r}'
        )
        assert result.contention is None, (
            f'run {n}: expected no contention payload, got {result.contention!r}'
        )
        assert is_flock_contention_failure(result) is False, (
            f'run {n}: real beta discriminant fired on a happy-path result -- '
            f'_run_post_merge_verify would wrongly escalate: {result!r}'
        )


# ---------------------------------------------------------------------------
# Task 2309 step-5/6 -- SS9 Row 4: cancel-verify under a LIVE watchdog (B2
# coexistence).  Extends test_cli.py:1656's real e2e cancel pattern with a
# live HeartbeatWriter so the stdin watchdog stays armed throughout but never
# self-fires, proving an explicit cancel-verify tree-kills cleanly anyway.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(ROW_PER_TEST_TIMEOUT_SECS)  # task 4025 amendment: shares the derived per-row budget -- same real-subprocess exposure as Rows 1/2/3, previously left on the module's bare 60s ini timeout
def test_cancel_verify_tree_kills_under_live_watchdog(tmp_path, monkeypatch):
    """SS9 Row 4: cancel-verify leaves no orphan while the stdin watchdog is live.

    Spawns a real ``verify-merge --request-id`` with a long-running sleeper
    spec (the start_new_session escape) and a LIVE :class:`HeartbeatWriter`
    driving its stdin, so :func:`verify_cancel.run_stdin_watchdog` stays armed
    for the whole test but never times out or self-fires.  Waits for the
    build to publish its session pgid (:func:`sleeper_spec`'s
    *build_pgid_file*), then cancels via ``cancel-verify --request-id``
    through :class:`CliRunner` (the same in-process pattern as test_cli.py's
    ``test_verify_merge_cancel_end_to_end``).

    Asserts:

    * cancel-verify exits 0 and removes the pgid file;
    * the verify-merge subprocess exits promptly;
    * every pid snapshotted from the leader's tree while it was alive -- the
      leader and the start_new_session build -- has exited (see
      :func:`assert_pre_kill_tree_exited`): the live watchdog and an explicit
      cancel coexist harmlessly and no orphan survives.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    config_obj = OrchestratorConfig(project_root=repo)
    worktree_base = worktree_base_for(repo)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=False)

    REQUEST_ID = 'row4-cancel-under-watchdog'
    pgf = pgid_file(worktree_base, REQUEST_ID)
    build_pgf = tmp_path / 'row4-build.pgid'

    child = spawn_verify_merge(
        sha=head_sha,
        spec=sleeper_spec(300.0, build_pgid_file=build_pgf),
        cfg_file=cfg_file,
        request_id=REQUEST_ID,
    )
    heartbeat = HeartbeatWriter(child, interval=0.2).start()
    try:
        leader = wait_for_pgid_file(pgf, proc=child, timeout=row_discovery_ceiling_secs())
        build = wait_for_pgid_file(build_pgf, proc=child, timeout=row_discovery_ceiling_secs())
        tree = snapshot_pre_kill_tree(leader, build)

        monkeypatch.setattr(cli_module, 'load_config', lambda _path: config_obj)
        result = CliRunner().invoke(main, [
            'cancel-verify', '--request-id', REQUEST_ID, '--config', str(cfg_file),
        ])
        assert result.exit_code == 0, (
            f'cancel-verify expected exit 0, got {result.exit_code}; '
            f'output={result.output!r}'
        )
        assert not pgf.exists(), 'pgid file must be removed by cancel-verify on success'

        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pytest.fail(
                'verify-merge subprocess did not exit within 20s after cancel-verify'
            )

        assert_pre_kill_tree_exited(
            tree, timeout=10.0, after='cancel-verify (live-watchdog coexistence left an orphan)',
        )
    finally:
        heartbeat.stop_heartbeats()
        kill_holder_tree(child, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)


# ---------------------------------------------------------------------------
# Task 2309 step-7/8 -- SS9 Row 1: orchestrator (dispatcher) killed mid-build
# (EOF path via whole-dispatcher death).  A SEPARATE killable process runs the
# REAL verify_runner._default_ssh_heartbeat_run; SIGKILLing it closes its end
# of the child's stdin pipe, delivering a clean EOF -- distinct from Row 3's
# heartbeat-timeout (select-timeout) branch, which keeps stdin open but silent.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(ROW_PER_TEST_TIMEOUT_SECS)  # task 4025: pinned 10s+5s fast window (production is 90s+5s) + real subprocesses under full-suite load
def test_orchestrator_killed_mid_build_tree_killed_via_eof(tmp_path):
    """SS9 Row 1: dispatcher process killed -> child sees stdin EOF -> tree-killed.

    Models "the orchestrator holding the ssh child died": spawns a SEPARATE
    dispatcher process running the REAL ``_default_ssh_heartbeat_run`` against
    a local ``verify-merge --request-id`` argv (small heartbeat_interval so
    real heartbeats flow while the dispatcher is alive).  Waits for the BUILD
    itself to publish its session pgid (:func:`sleeper_spec`'s
    *build_pgid_file* -- not merely any descendant, which is often a
    transient ``git worktree add``), snapshots the leader's tree while the
    leader is provably alive, then SIGKILLs the dispatcher process itself --
    when the OS reclaims its file descriptors, ITS end of the child's stdin
    pipe closes, giving the grandchild a clean EOF on fd 0.

    ROW_WATCHDOG_ENV is threaded through the dispatcher's own environment
    (:func:`spawn_ssh_heartbeat_dispatcher`'s *extra_env*, ambiently inherited
    by the grandchild verify-merge) -- the pinned 10s+5s window, deliberately
    NOT production's own (90s+5s since task 4195); see ROW_WATCHDOG_ENV's
    comment in _verify_boundary_ceilings.py for the pin rationale.  Per
    ``run_stdin_watchdog``, EOF fires on the very next
    ``select`` readiness check regardless of *heartbeat_timeout* -- only
    ``grace_secs`` (the SIGTERM->SIGKILL pause in ``fire_watchdog_kill``)
    materially bounds this row's timing (~5s, measured ~7.3s end-to-end).

    Asserts within a bounded T that every snapshotted pid -- the leader and
    the start_new_session build escape -- has EXITED.  A zombie awaiting its
    subreaper counts as exited (see :func:`wait_pids_exited`); a pgid walk
    from the leader could not judge this, since it goes blind to the escaped
    build the moment the leader dies.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=False)
    worktree_base = worktree_base_for(repo)

    REQUEST_ID = 'row1-orchestrator-killed'
    pgf = pgid_file(worktree_base, REQUEST_ID)
    build_pgf = tmp_path / 'row1-build.pgid'

    argv = verify_merge_argv(
        sha=head_sha,
        spec=sleeper_spec(300.0, build_pgid_file=build_pgf),
        cfg_file=cfg_file,
        request_id=REQUEST_ID,
    )
    dispatcher = spawn_ssh_heartbeat_dispatcher(
        argv=argv,
        heartbeat_interval=0.2,
        extra_env=ROW_WATCHDOG_ENV,
    )
    tree: PreKillTree | None = None
    try:
        leader = wait_for_pgid_file(
            pgf,
            proc=dispatcher,
            proc_label='dispatcher',
            timeout=row_discovery_ceiling_secs(),
        )
        build_pgid = wait_for_pgid_file(
            build_pgf,
            proc=dispatcher,
            proc_label='dispatcher',
            timeout=row_discovery_ceiling_secs(),
        )
        tree = snapshot_pre_kill_tree(leader, build_pgid)

        dispatcher.kill()
        dispatcher.wait(timeout=10)

        assert_pre_kill_tree_exited(
            tree,
            timeout=ROW_TREE_KILL_CEILING_SECS,
            after=(
                'the dispatcher was killed (the EOF-triggered watchdog tree-kill '
                'did not finish)'
            ),
        )
    finally:
        if dispatcher.poll() is None:
            dispatcher.kill()
            dispatcher.wait(timeout=5)
        if tree is not None:
            for pid in wait_pids_exited(tree.pids, timeout=0):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.kill(pid, signal.SIGKILL)


# ---------------------------------------------------------------------------
# Task 2309 step-9/10 -- SS9 Row 2: ssh connection dropped mid-build (EOF
# path, dispatcher ALIVE).  Distinct from Row 1 (whole dispatcher process
# death): here the in-process dispatcher (this test) owns the child's
# stdin=PIPE directly and closes ONLY the write end via
# HeartbeatWriter.close_stdin(), while the dispatcher/test process itself
# keeps running throughout -- modeling "the ssh transport dropped", not "the
# orchestrator died".
# ---------------------------------------------------------------------------


@pytest.mark.timeout(ROW_PER_TEST_TIMEOUT_SECS)  # task 4025: pinned 10s+5s fast window (production is 90s+5s) + real subprocesses under full-suite load
def test_ssh_dropped_mid_build_tree_killed_via_eof_dispatcher_alive(tmp_path):
    """SS9 Row 2: ssh channel closes but the dispatcher stays alive -> EOF tree-kill.

    Spawns a real ``verify-merge --request-id`` with a live
    :class:`HeartbeatWriter` driving its stdin (real heartbeats flow while
    connected), waits for the build to publish its session pgid
    (:func:`sleeper_spec`'s *build_pgid_file*), then calls
    ``heartbeat.close_stdin()`` -- stopping heartbeats AND closing the
    child's stdin write-end -- while this test process (the dispatcher
    stand-in) keeps running.  The child sees the same fd-0 EOF signal as Row
    1 and the same ``run_stdin_watchdog`` EOF branch fires; the difference
    from Row 1 is that no dispatcher PROCESS dies here, only the transport.

    Uses ROW_WATCHDOG_ENV directly on the spawned verify-merge (already an
    in-process dispatcher, so no extra remove is needed, unlike Row 1's
    separate-process case) -- the pinned 10s+5s window, deliberately NOT
    production's own (90s+5s since task 4195).  Row 2 is an EOF row like Row
    1, not a timeout row: ``close_stdin()`` closes the write end, so only
    ``grace_secs`` bounds it (~5s, measured ~6.7s end-to-end), NOT the ~15s a
    reader might assume from the pinned heartbeat timeout.

    Asserts within a bounded T that the verify-merge process itself
    self-exits non-zero (the watchdog's controlled ``os._exit(1)``, not an
    external kill signal) with the ``eof`` trigger, and only that trigger, on
    its stderr, and that every pid snapshotted from the leader's tree while
    it was alive -- the leader and the start_new_session build -- has exited
    (see :func:`assert_pre_kill_tree_exited`).
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=False)
    worktree_base = worktree_base_for(repo)

    REQUEST_ID = 'row2-ssh-dropped'
    pgf = pgid_file(worktree_base, REQUEST_ID)
    build_pgf = tmp_path / 'row2-build.pgid'

    child = spawn_verify_merge(
        sha=head_sha,
        spec=sleeper_spec(300.0, build_pgid_file=build_pgf),
        cfg_file=cfg_file,
        request_id=REQUEST_ID,
        extra_env=ROW_WATCHDOG_ENV,
    )
    heartbeat = HeartbeatWriter(child, interval=0.2).start()
    try:
        leader = wait_for_pgid_file(pgf, proc=child, timeout=row_discovery_ceiling_secs())
        build = wait_for_pgid_file(build_pgf, proc=child, timeout=row_discovery_ceiling_secs())
        tree = snapshot_pre_kill_tree(leader, build)

        heartbeat.close_stdin()

        # The reap feeds the returncode assertion; the exit verdict is zombie-aware.
        try:
            child.wait(timeout=ROW_TREE_KILL_CEILING_SECS)
        except subprocess.TimeoutExpired:
            pytest.fail(
                f'verify-merge did not exit within '
                f'{ROW_TREE_KILL_CEILING_SECS}s after stdin EOF'
            )
        assert child.returncode != 0, (
            f'expected non-zero exit (watchdog self-kill), got {child.returncode}'
        )
        assert_watchdog_fire_trigger(
            drain_exited_stderr(child).decode(errors='replace'),
            WatchdogTrigger.EOF,
            source='Row 2 leader stderr',
        )

        assert_pre_kill_tree_exited(
            tree,
            timeout=ROW_TREE_KILL_CEILING_SECS,
            after=(
                'the ssh channel EOF with the dispatcher alive (the watchdog '
                'tree-kill did not finish)'
            ),
        )
    finally:
        heartbeat.stop_heartbeats()
        kill_holder_tree(child, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)


# ---------------------------------------------------------------------------
# Task 2309 step-11/12 -- SS9 Row 3: heartbeat starved / simulated hard
# partition (heartbeat-TIMEOUT path).  Distinct from Rows 1/2 (both EOF-
# triggered): stdin stays OPEN throughout, only heartbeats stop, exercising
# run_stdin_watchdog's select-timeout branch (empty ready set) rather than
# its EOF branch (readable, read() == b'').
# ---------------------------------------------------------------------------


@pytest.mark.timeout(ROW_PER_TEST_TIMEOUT_SECS)  # task 4025: pinned 10s+5s fast window (production is 90s+5s) + real subprocesses under full-suite load
def test_heartbeat_starved_hard_partition_tree_killed_via_timeout(tmp_path):
    """SS9 Row 3: heartbeat starved, stdin stays OPEN -> select-timeout tree-kill.

    Spawns a real ``verify-merge --request-id`` with a live
    :class:`HeartbeatWriter`, waits for the build to publish its session pgid
    (:func:`sleeper_spec`'s *build_pgid_file*), then calls
    ``heartbeat.stop_heartbeats()`` -- which stops writing WITHOUT
    closing stdin (contrast :func:`HeartbeatWriter.close_stdin`, Row 2) --
    modeling a hard network partition where the channel is never cleanly
    closed.  ``run_stdin_watchdog``'s ``select()`` call times out (empty
    ready set) rather than seeing EOF, taking the branch that calls
    ``on_fire`` directly without ever attempting a read.

    Uses ROW_WATCHDOG_ENV -- the pinned fast window (10.0s heartbeat timeout
    + 5.0s kill grace; production is 90.0s+5.0s since task 4195, injected down
    to this) -- directly on the spawned verify-merge, so the assertion window
    here is ~15s rather than production's ~95s; see ROW_WATCHDOG_ENV's comment
    in _verify_boundary_ceilings.py for the pin rationale and the honest
    cliff-moved caveat.

    Asserts the leader self-exits non-zero with the ``heartbeat_starvation``
    trigger, and only that trigger, on its stderr, and that every pid
    snapshotted from the leader's tree while it was alive -- the leader and
    the start_new_session build -- has exited (see
    :func:`assert_pre_kill_tree_exited`).
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=False)
    worktree_base = worktree_base_for(repo)

    REQUEST_ID = 'row3-heartbeat-starved'
    pgf = pgid_file(worktree_base, REQUEST_ID)
    build_pgf = tmp_path / 'row3-build.pgid'

    child = spawn_verify_merge(
        sha=head_sha,
        spec=sleeper_spec(300.0, build_pgid_file=build_pgf),
        cfg_file=cfg_file,
        request_id=REQUEST_ID,
        extra_env=ROW_WATCHDOG_ENV,
    )
    heartbeat = HeartbeatWriter(child, interval=0.2).start()
    try:
        leader = wait_for_pgid_file(pgf, proc=child, timeout=row_discovery_ceiling_secs())
        build = wait_for_pgid_file(build_pgf, proc=child, timeout=row_discovery_ceiling_secs())
        tree = snapshot_pre_kill_tree(leader, build)

        heartbeat.stop_heartbeats()
        assert child.stdin is not None and not child.stdin.closed, (
            'harness bug: stdin must stay OPEN for Row 3 (hard partition) -- '
            'only heartbeats stop; this is what distinguishes it from Row 2'
        )

        try:
            child.wait(timeout=ROW_TREE_KILL_CEILING_SECS)
        except subprocess.TimeoutExpired:
            pytest.fail(
                f'verify-merge did not self-exit within '
                f'{ROW_TREE_KILL_CEILING_SECS}s of heartbeat starvation '
                f'(select-timeout branch did not fire)'
            )
        assert child.returncode != 0, (
            f'expected non-zero exit (watchdog self-kill), got {child.returncode}'
        )
        assert_watchdog_fire_trigger(
            drain_exited_stderr(child).decode(errors='replace'),
            WatchdogTrigger.HEARTBEAT_STARVATION,
            source='Row 3 leader stderr',
        )

        assert_pre_kill_tree_exited(
            tree,
            timeout=ROW_TREE_KILL_CEILING_SECS,
            after=(
                'heartbeat starvation (the select-timeout watchdog tree-kill did '
                'not finish)'
            ),
        )
    finally:
        heartbeat.stop_heartbeats()
        kill_holder_tree(child, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)


# ---------------------------------------------------------------------------
# Task 5556 -- trigger attribution, end to end.  A real RemoteRunner dispatches
# a real verify-merge through the real heartbeat dispatcher, starves it, and
# follows the watchdog's trigger token through every hop that relays it: the
# dispatcher's stderr capture and the RunnerUnavailable reason, live; then the
# merge worker's elided `remote runner unavailable` WARNING, by replaying that
# captured reason through the real worker rather than a live dispatch.
# ---------------------------------------------------------------------------

#: Heads the RunnerUnavailable reason this row's dispatch produces.
LOOPBACK_SSH_HOST: str = 'loopback-verify-host'

#: Far beyond any wait in this module.  run_stdin_heartbeat's first beat lands
#: at t=interval, so no beat ever arrives and the leader's watchdog takes its
#: select-timeout branch.
STARVED_HEARTBEAT_INTERVAL_SECS: float = 3600.0


@pytest.mark.timeout(ROW_PER_TEST_TIMEOUT_SECS)
def test_starvation_trigger_survives_real_remote_dispatch_and_replayed_worker_warning(
    tmp_path, monkeypatch, caplog
):
    """A starved remote self-kill stays attributable all the way to the worker's log.

    Drives a real RemoteRunner whose only doubles are ``git push``, the
    gate's publish and ensure calls, and an ssh that is a transparent pipe,
    so the leader's stdin is the real
    heartbeat pipe and its stderr comes back exactly as ssh relays it.  The
    reason must carry the watchdog's controlled rc 1 at its head and the
    starvation token at its tail.  That captured reason is then replayed,
    through a runner double, into the real merge worker, whose elided
    WARNING must still name the token.  The EOF trigger is pinned at Row 2
    instead, because the real dispatcher closes the heartbeat pipe only
    after its child has exited.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = write_verify_config(tmp_path / 'config.yaml', repo, persistent_merge_worktree=False)
    tag = uuid.uuid4().hex
    # CLI startup, then one watchdog window plus load headroom: a wedge
    # detector that costs nothing on the success path.
    dispatch_ceiling = row_discovery_ceiling_secs() + ROW_TREE_KILL_CEILING_SECS

    with monkeypatch.context() as env:
        apply_dispatcher_env(env)
        for key, value in ROW_WATCHDOG_ENV.items():
            env.setenv(key, value)
        env.setenv(HOLDER_TAG_ENV, tag)
        runner = gated_remote_runner(
            'loopback',
            LOOPBACK_SSH_HOST,
            git_remote=str(repo),
            cwd=repo,
            config_path=str(cfg_file),
            run=git_push_succeeds,
            ssh_run=loopback_ssh_run(heartbeat_interval=STARVED_HEARTBEAT_INTERVAL_SECS),
            id_factory=lambda: 'trigger-attribution-e2e',
        )
        try:
            with pytest.raises(RunnerUnavailable) as excinfo:
                asyncio.run(asyncio.wait_for(
                    runner.run_merge_verify(head_sha, sleeper_spec(300.0)), dispatch_ceiling,
                ))
        except TimeoutError:
            pytest.fail(
                f'the remote dispatch did not end within {dispatch_ceiling}s: the starved '
                f'leader\'s watchdog never fired, so its self-kill could not be attributed'
            )
        finally:
            sweep_tagged(tag)

    reason = str(excinfo.value)
    assert reason.startswith(f'ssh {LOOPBACK_SSH_HOST} exited 1:'), (
        f'expected the watchdog\'s controlled self-exit (rc 1), not a transport death; '
        f'reason head: {reason[:300]!r}'
    )
    assert_watchdog_fire_trigger(
        reason, WatchdogTrigger.HEARTBEAT_STARVATION, source='RunnerUnavailable reason'
    )

    consumer_dir = tmp_path / 'consumer'
    consumer_dir.mkdir()
    _, records = asyncio.run(_mq_drive_runner_unavailable(consumer_dir, caplog, reason))
    assert len(records) == 1, [record.message for record in records]
    assert 'exited 1' in records[0].message, records[0].message
    assert_watchdog_fire_trigger(
        records[0].message,
        WatchdogTrigger.HEARTBEAT_STARVATION,
        source='remote runner unavailable WARNING',
    )


# ---------------------------------------------------------------------------
# Task 2309 step-13/14 -- SS9 Row 5: flock contention, FULL two-way seam
# (producer + consumer).  The load-bearing SS8.2 seam assertion: a real
# verify-merge #1 holds .merge_verify.lock mid-build; a real verify-merge #2
# on the SAME worktree_base reports the FLOCK_CONTENTION_CATEGORY
# discriminant on stdout WITHOUT ever touching the tree; #2's parsed stdout
# is then fed through the REAL merge_queue consumer (_run_post_merge_verify +
# a real EscalationQueue), asserting a born-at-L2 escalation is filed and the
# MergeOutcome is 'blocked'.
# ---------------------------------------------------------------------------


# task 2921: two real verify-merge subprocesses (holder + waiter) under
# full-suite load -- the module's most storm-exposed row.  Now DERIVED rather
# than the literal 120 task 2921 wrote: task 4014 made this row's two
# discovery waits load-scaled (up to ROW_DISCOVERY_CEILING_MAX_SECS each), and
# a literal cannot track that -- which is exactly what the review caught.
@pytest.mark.timeout(ROW5_PER_TEST_TIMEOUT_SECS)
def test_flock_contention_full_two_way_seam_blocks_and_escalates(tmp_path):
    """SS9 Row 5: producer discriminant -> real consumer -> born-at-L2 + blocked.

    Producer side: a real ``verify-merge #1`` (sleeper build, --request-id)
    materialises the persistent ``_merge-verify`` worktree and holds the real
    ``.merge_verify.lock`` flock for the full duration of its build (cli.py
    acquires the flock BEFORE awaiting the build and releases it only in the
    outer ``finally``).  A real ``verify-merge #2`` on the SAME worktree_base
    with the step-2 env seam (small flock wait) then reports contention.

    Asserts on #2: exits 0 with the FLOCK_CONTENTION_CATEGORY discriminant on
    stdout, contention={host, holder_pgid, waiter_pgid} with holder_pgid
    matching #1's own pgid; no ``_merge-<uuid>`` ephemeral dir is ever
    created; #1's ``_merge-verify`` tree is never touched (marker file mtime
    unchanged -- #2's code path returns before
    ``acquire_host_verify_worktree`` is ever called on contention).

    Consumer side: parses #2's stdout via ``result_from_json`` and feeds it
    through the REAL ``_run_post_merge_verify`` (a stub runner returning that
    exact parsed result -- the established pattern from
    ``TestRunPostMergeVerifyFlockContention`` in
    ``test_merge_queue_multihost_wiring.py``) with a REAL
    ``EscalationQueue(tmp_path)``.  Asserts a born-at-L2 escalation is queued
    (level=2, severity='critical',
    agent_role='orchestrator-verify-host-monitor',
    category='verify_worktree_contention', detail names both pgids) and the
    returned ``MergeOutcome`` is 'blocked'.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=True)
    worktree_base = worktree_base_for(repo)

    HOLDER_REQUEST_ID = 'row5-holder'
    pgf_holder = pgid_file(worktree_base, HOLDER_REQUEST_ID)

    holder = spawn_verify_merge(
        sha=head_sha, spec=sleeper_spec(300.0), cfg_file=cfg_file, request_id=HOLDER_REQUEST_ID,
    )
    # task 3318: --request-id arms the holder's stdin watchdog (see
    # spawn_verify_merge docstring), but unlike Row 3 this holder was never
    # given a live HeartbeatWriter -- it self-killed at
    # ~(WATCHDOG_HEARTBEAT_TIMEOUT_SECS + WATCHDOG_KILL_GRACE_SECS), measured
    # ~15-17s back when those summed to 15.0 (task 4195 has since derived the
    # timeout from the ssh transport, so an unpaired holder would now last
    # ~95s -- this holder gets no ROW_WATCHDOG_ENV)
    # instead of surviving the full 300s sleeper_spec, well inside the
    # waiter's own budget under full-suite xdist load (Python import alone
    # ~9s). Pair it with a live heartbeat, mirroring Row 3, so it survives
    # for the whole span the test needs it.
    heartbeat_holder = HeartbeatWriter(holder, interval=0.2).start()
    # task 4946: bound before the try so the finally below can tear it down
    # even if wait_for_pgid_file/wait_subtree_live/wait_for_marker_stable
    # raises before the waiter is ever spawned.
    waiter = None
    try:
        holder_pgid_val = wait_for_pgid_file(pgf_holder, proc=holder)
        wait_subtree_live(holder_pgid_val, proc=holder)

        persistent_wt = worktree_base / '_merge-verify'
        marker = persistent_wt / 'target' / 'warm.marker'
        # Capture the baseline only AFTER the holder's touch (create +
        # utimensat) has settled -- under full-xdist load those two syscalls
        # can straddle a preemption, so the instant-of-appearance mtime can
        # still drift ~tens of ms (task 2819), which would spuriously trip
        # the retention equality assertion below.
        marker_mtime_before = wait_for_marker_stable(marker)

        waiter = spawn_verify_merge(
            sha=head_sha,
            spec=fast_spec(),
            cfg_file=cfg_file,
            request_id='row5-waiter',
            extra_env={'ORCH_MERGE_VERIFY_FLOCK_WAIT_SECS': '0.5'},
        )
        # task 2921 (load-robustness): widened from 15s. The waiter is a full
        # real verify-merge subprocess (Python import ~9s under a full-suite
        # storm + config load + git worktree setup + the 0.5s flock wait), so a
        # 15s ceiling is import-bound, not flock-bound, under parallel load and
        # flaked (observed: this subprocess timed out at 15s under load). There
        # is NO timing assertion on the waiter here -- the test asserts on the
        # returned FLOCK_CONTENTION_CATEGORY discriminant, not on duration -- so
        # widening the completion ceiling has zero discrimination cost.
        stdout, stderr = waiter.communicate(timeout=ROW5_WAITER_COMPLETION_CEILING_SECS)

        assert waiter.returncode == 0, (
            f'expected exit 0 (contention result on stdout), got '
            # task 3318: tail-sliced (not head-truncated) -- the actual
            # failure cause (e.g. a watchdog self-kill traceback) is at the
            # END of stderr, and a 2000-char head slice was hiding it.
            f'{waiter.returncode}; stderr={stderr_tail(stderr, limit=4000)!r}'
        )
        result = result_from_json(stdout.decode())
        assert is_flock_contention_failure(result), (
            f'expected the flock-contention discriminant, got {result!r}'
        )
        assert result.category == FLOCK_CONTENTION_CATEGORY
        assert result.contention is not None
        assert result.contention['host'] == socket.gethostname()
        assert result.contention['holder_pgid'] == holder_pgid_val
        waiter_pgid_val = result.contention['waiter_pgid']
        assert waiter_pgid_val != holder_pgid_val

        leaked_ephemeral = sorted(
            p.name for p in worktree_base.iterdir()
            # An ephemeral leak is a _merge-<uuid> *directory*; the shared
            # <lane_dir>.lock flock sibling (<worktree_base>/_merge-verify.lock,
            # task 2685) is a regular file and is expected -- require is_dir().
            if p.is_dir() and p.name.startswith('_merge-') and p.name != '_merge-verify'
        )
        assert leaked_ephemeral == [], (
            f'waiter must never create an ephemeral _merge-<uuid> dir on '
            f'contention: {leaked_ephemeral}'
        )
        assert marker.stat().st_mtime_ns == marker_mtime_before, (
            "waiter mutated the holder's _merge-verify tree on contention "
            "(marker mtime changed)"
        )
    finally:
        heartbeat_holder.stop_heartbeats()
        # task 4946: a TimeoutExpired from waiter.communicate() above, or
        # any of the six assertions before this finally failing, previously
        # left the waiter leader -- and its session-escaped descendants --
        # alive.  Tear the waiter down BEFORE the holder: it is the process
        # contending for the lock the holder owns.  Cheap on the success
        # path: kill_holder_tree's pinned-leader gate skips the walk, leaving
        # only its tag sweep.
        if waiter is not None:
            kill_holder_tree(waiter, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)
        kill_holder_tree(holder, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)

    # --- Consumer side: feed #2's real discriminant through the real beta
    # consumer (_run_post_merge_verify) with a real EscalationQueue. ---

    stub_runner = MagicMock()
    stub_runner.is_local = False
    stub_runner.run_merge_verify = AsyncMock(return_value=result)

    eq = EscalationQueue(tmp_path / 'escalations')

    async def _drive_consumer():
        config = _mq_make_config()
        req = _mq_make_merge_request(config, task_files=['src/foo.py'], worktree=tmp_path)
        git_ops = _mq_make_git_ops_mock()
        return await _run_post_merge_verify(
            git_ops, req, tmp_path,
            timeouts={}, enospc_retries={},
            max_timeouts=2, max_enospc=1,
            merge_sha=head_sha,
            runner=stub_runner,
            escalation_queue=eq,
        )

    outcome = asyncio.run(_drive_consumer())

    assert outcome is not None
    assert outcome.status == 'blocked'
    assert outcome.failure_category == FLOCK_CONTENTION_CATEGORY

    sentinel = _verify_worktree_contention_sentinel(socket.gethostname())
    matches = eq.get_by_task(sentinel, status='pending', level=2)
    assert len(matches) == 1, f'expected exactly one born-at-L2 escalation, got {matches!r}'
    esc = matches[0]
    assert esc.level == 2
    assert esc.severity == 'critical'
    assert esc.agent_role == 'orchestrator-verify-host-monitor'
    assert esc.category == 'verify_worktree_contention'
    assert str(holder_pgid_val) in esc.detail
    assert str(waiter_pgid_val) in esc.detail


# ---------------------------------------------------------------------------
# Task 2830 step-4 -- the SS9 Signal at the real subprocess seam: a live
# knob-ON verify-merge HOLDS the shared <lane_dir>.lock mid-build, so a
# non-blocking flock probe on it from this process is DENIED.  Stronger
# evidence than the mock-based step-1 pin (test_cli.py
# test_verify_merge_holds_lane_lock_during_warm_run): drives the real CLI
# span end-to-end and pins the lock PATH the split lane-lock lifetime
# (esc-2830-1) re-acquires for the build's duration.
# ---------------------------------------------------------------------------


def test_live_verify_merge_holds_lane_lock_real_subprocess(tmp_path):
    """SS9 Signal: a live knob-ON verify-merge holds the SHARED lane lock mid-build (task 2830).

    The task's Signal, at the REAL subprocess seam: spawn a real knob-ON
    ``orchestrator verify-merge`` with a sleeper build, poll until it is genuinely
    mid-build (the sleeper's marker inside the persistent ``_merge-verify`` worktree
    appears), then from THIS test process take a non-blocking exclusive flock on
    ``lane_lock_path(persistent_merge_worktree_path)`` = ``<worktree_base>/
    _merge-verify.lock`` -- the SAME lock reify's seed/thin/gc and
    ``GitOps.merge_verify_lease`` take (task 2685) -- and assert it is DENIED
    (BlockingIOError). This confirms the live CLI span is the holder of the LANE lock
    during the build, so a laptop lane actor (reseed/thin/gc) is mutually excluded from
    a live laptop verify -- closing the divergence DF 2685 left for the remote host and
    that the two remote-twin incidents (reify 5034/5187) turned on.

    With the split lane-lock lifetime (esc-2830-1) the lane lock is re-acquired for the
    build's duration inside ``_run()`` (AFTER ``acquire_host_verify_worktree``'s own
    reset has released it), and the shared holder-pgid is written before the build
    runs. The sleeper's ``touch`` executes inside that build-scoped hold, so the
    marker's existence implies the lock is already held and the holder-pgid recorded.

    Stronger than the mock-based step-1 pin
    (``test_verify_merge_holds_lane_lock_during_warm_run``): it drives the real CLI span
    end-to-end (no monkeypatched ``acquire_host_verify_worktree`` / ``run_merge_verify_``
    ``on_worktree``), so it pins the lock path the production code actually computes and
    holds.
    """
    repo, head_sha = _setup_verify_repo(tmp_path)
    worktree_base = worktree_base_for(repo)
    worktree_base.mkdir(parents=True, exist_ok=True)

    cfg_file = tmp_path / 'config.yaml'
    write_verify_config(cfg_file, repo, persistent_merge_worktree=True)

    # No --request-id -> the stdin watchdog is never armed, so the sleeper holder
    # stays alive until we kill it in the finally (mirrors the env-override probe
    # test's spawn shape, test_flock_wait_env_override_speeds_up_contention_result).
    holder = spawn_verify_merge(sha=head_sha, spec=sleeper_spec(300.0), cfg_file=cfg_file)
    try:
        persistent_wt = worktree_base / '_merge-verify'
        marker = persistent_wt / 'target' / 'warm.marker'
        # Poll until the sleeper build is genuinely mid-run. The build-scoped lane-lock
        # hold is acquired (and the holder-pgid written) BEFORE the build runs, so the
        # marker's appearance implies the lane lock is currently held -- no race.
        wait_for_marker(marker)

        # THE SIGNAL: a non-blocking exclusive flock on the shared lane lock from this
        # (independent) process must be DENIED while the live CLI span holds it.
        lane_lock = lane_lock_path(persistent_wt)
        probe_fd = os.open(lane_lock, os.O_RDWR | os.O_CREAT)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe_fd)

        # Corroboration: the build-scoped hold records the shared holder-pgid, so a
        # live positive int (not a stale/absent file) confirms the live CLI owns the
        # lane lock right now.
        holder_pgid = read_lock_holder_pgid(worktree_base)
        assert isinstance(holder_pgid, int) and holder_pgid > 0, (
            f'the live CLI must record a positive shared holder-pgid while holding the '
            f'lane lock mid-build, got {holder_pgid!r}'
        )
    finally:
        kill_holder_tree(holder, timeout=ROW5_HOLDER_TEARDOWN_CEILING_SECS)
