"""Guard tests: this suite — the fleet heartbeat's PRODUCER — cannot reach the
LIVE fleet directory (task 3951).

Sibling of ``tests/scripts/test_fleet_dir_isolation.py``, which carries the
``tests/`` root's copy of the same proofs; ``scripts/tests/`` carries a third in
``test_restart_all_orchestrators.py``.  The three roots are wired SEPARATELY, so
a green run in one says nothing about the others — hence one module per root,
all findable by one grep on the name.

WHY THIS ROOT NEEDED ITS OWN WIRING, and why the two existing copies did not
cover it.  Task 3799 wired ``_df_fleet_dir_redirect`` /
``_df_no_synthetic_heartbeats_in_live_fleet`` into the repo-root ``conftest.py``
and ``scripts/tests/conftest.py`` — the two rootdirs that SPAWN the drain
script, i.e. the two that READ the fleet dir.  ``orchestrator/tests/`` is the
rootdir of the WRITER: ``Harness._write_merge_heartbeat`` calls
``fleet_heartbeat.resolve_fleet_dir()``, which reads bare ``os.environ`` and
falls back to the machine-global, CROSS-PROJECT
``/home/leo/src/dark-factory/data/fleet``, then hands the result to
``write_heartbeat``.  About twenty modules here drive ``harness.run()`` for
real, so every one of them wrote a heartbeat straight into that live directory.
MEASURED: ``data/fleet/unknown-unit.json`` observed at mtime 2026-08-10
00:50:32 (esc-3799-6, which reproduced the advance inside a ``pytest tests/``
run of this suite) and again 2026-08-17 02:46:23 during task 3951.  Worse than
inert pollution whenever ``ORCH_UNIT`` is set: it is AMBIENT-SET in an
orchestrator-dispatched session (measured: ``orchestrator-dark-factory.service``),
so the forgetful test overwrites the REAL unit's heartbeat — the file
``scripts/drain_check.py`` and ``scripts/orchestrator-watchdog.py`` read BY NAME
— rather than the inert ``unknown-unit.json``.

And the repo-root conftest cannot reach here: the verify lane runs ``cd
orchestrator && uv run pytest tests/``, which makes rootdir the SUBPROJECT, so
this suite must wire the defence itself.  Same framing, and the same reason, as
``test_git_repo_isolation_guard.py::TestSessionCeilingIsLiveInThisSuite``.

NOTHING here writes a heartbeat.  The obvious end-to-end shape — drive a write
with the ambient env and assert where the file landed — is RED precisely BY
writing into the live cross-project directory, so it would BE the defect under
guard on every red run.  ``resolve_fleet_dir()`` is the same production function
the harness calls, reading the same bare ``os.environ``, so it proves
containment at the identical seam with zero filesystem effect.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import df_pytest_isolation
from df_pytest_isolation import fleet_dir_redirect_violation_reason

from orchestrator.fleet_heartbeat import DEFAULT_FLEET_DIR, resolve_fleet_dir

# NOT `from df_pytest_isolation import _df_...`. Importing a FIXTURE into a test
# module binds it as a module-scoped fixture that SHADOWS the conftest's — which
# would make the liveness tests below resolve their own import and pass even
# with the conftest wiring removed, i.e. exactly the dead defence they exist to
# detect. Reach them through the module instead.
_REDIRECT_NAME = '_df_fleet_dir_redirect'
_GUARD_NAME = '_df_no_synthetic_heartbeats_in_live_fleet'

_FLEET_DIR_ENV = 'ORCH_FLEET_DIR'


def test_the_production_resolver_lands_in_tmp_space_for_this_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``resolve_fleet_dir()`` — called as PRODUCTION calls it — is contained.

    This root's distinctive proof, and the reason a copy of the other two roots'
    env-var check alone would not be enough here.  Those roots are READERS: they
    check ``ORCH_FLEET_DIR`` as the bash script's ``${VAR:-…}`` default and
    ``drain_check`` see it.  This root is the WRITER's, so the assertion is made
    against the writer's own resolution seam —
    ``orchestrator.fleet_heartbeat.resolve_fleet_dir``, called with NO argument
    so it reads the ambient session ``os.environ`` exactly as
    ``Harness._write_merge_heartbeat`` does at its single production call site.

    A pure resolver READ, deliberately: it creates nothing, so the RED phase of
    this test cannot itself leak into the live directory.
    """
    resolved = resolve_fleet_dir()
    basetemp = tmp_path_factory.getbasetemp().resolve()

    assert resolved.resolve() != DEFAULT_FLEET_DIR.resolve(), (
        f'resolve_fleet_dir() returned the DEFAULT fleet dir {DEFAULT_FLEET_DIR}. '
        'That is a MACHINE-GLOBAL, CROSS-PROJECT rendezvous directory holding '
        "seven projects' live orchestrator heartbeats, and ~20 modules in this "
        'suite drive harness.run() for real — each one writing into it. '
        'orchestrator/tests/conftest.py must import _df_fleet_dir_redirect from '
        'df_pytest_isolation (the import IS the wiring, since pytest only '
        "collects fixtures bound into a conftest's namespace)."
    )
    assert resolved.resolve().is_relative_to(basetemp), (
        f'resolve_fleet_dir() returned {resolved}, which is outside this run\'s '
        f'basetemp {basetemp}. The production writer must resolve into pytest '
        'tmp space, or this suite is writing heartbeats somewhere that outlives '
        'it. Fix: df_pytest_isolation._df_fleet_dir_redirect.'
    )


def test_fleet_dir_is_redirected_away_from_the_live_checkout(
    _df_fleet_dir_redirect: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """``ORCH_FLEET_DIR`` points somewhere hermetic for the WHOLE session.

    Takes the redirect from the fixture BY NAME rather than reading
    ``os.environ`` bare: deleting the conftest binding then fails SETUP with a
    message naming ``_df_fleet_dir_redirect``, instead of this test quietly
    passing off some other suite's leftover env var or failing with a bare
    KeyError that names nothing.

    Only the WIRING is duplicated across the three roots' copies — the
    comparison and its messages live once, in
    ``df_pytest_isolation.fleet_dir_redirect_violation_reason``, because two
    copies of that ~20-line assertion body had already drifted in message text
    before they were a day old.
    """
    value = os.environ.get(_FLEET_DIR_ENV)
    reason = fleet_dir_redirect_violation_reason(
        value, tmp_path_factory.getbasetemp(),
    )
    assert reason is None, reason

    # The fixture's yielded value IS the redirect, not a parallel path. This is
    # the part that is genuinely per-root: it proves THIS rootdir's conftest
    # bound the fixture that set the variable checked above.
    assert Path(_df_fleet_dir_redirect).resolve() == Path(value or '').resolve()
