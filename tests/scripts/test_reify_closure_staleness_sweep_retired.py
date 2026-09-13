"""Repo-structure guard: the nightly reify closure-staleness sweep and its
re-dispatch consumer are retired, and the surviving owners of the
stranded-blocked population they duplicated are preserved (task 5247).

dark-factory owned the INVOCATION half of reify's nightly closure-staleness
sweep: a systemd user timer at 04:30 that ran a wrapper script, which in turn
ran a re-dispatch-request consumer against the task store.  Leo ruled the whole
arrangement retired on 2026-09-09: its ``gate_closure`` predicate was a second,
opposite-policy owner of the stranded-blocked population, and over the 15
retained journal runs it cancelled 7 reify tasks as collateral.

The timer was disabled by hand before this task ran, so the live signal was
already quiet.  The INSTALLER is what makes that disable revocable — any run of
``scripts/install-reify-closure-staleness-sweep-timer.sh`` re-enables the timer
from the tracked unit files — which is why it heads the absence list below and
is deleted first.

This is a runtime-state (filesystem / tracked-tree) assertion — NOT a
``__doc__``/introspection meta-test — so it has a proper RED (every path
present on the base tree) -> GREEN (removed) home, and it guards against the
wiring being re-introduced.

Each absence assertion also checks ``is_symlink()`` so a *dangling*-symlink
leftover (``exists()`` would report ``False``, silently passing) is caught.

HARD CONSTRAINT: only the invocation wiring is retired.  The stranded-blocked
population is still owned — by the orchestrator scheduler's blocked-redispatch
phase, by the harness deterministic-recon sweep, and by fused-memory's Stage 2
task-knowledge reconciliation.  The preservation tests below are green on
arrival by design: they exist so an over-broad deletion fails loudly instead of
silently removing a surviving owner.
"""
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_RETIRED_MESSAGE = (
    "task 5247: {path} is part of the retired nightly reify closure-staleness "
    "sweep wiring ({why}); it must stay deleted, but is still present (or is a "
    "dangling-symlink leftover)"
)


def _assert_retired(relpath: str, why: str) -> None:
    p = REPO_ROOT / relpath
    assert not p.exists() and not p.is_symlink(), _RETIRED_MESSAGE.format(
        path=p, why=why
    )


def test_installer_retired():
    """The installer — the re-arm vector — is gone."""
    _assert_retired(
        "scripts/install-reify-closure-staleness-sweep-timer.sh",
        "it re-enables the timer on any run, which is what makes a by-hand "
        "disable revocable; this is the reason the guard exists",
    )


def test_timer_unit_retired():
    """The systemd timer unit template is gone."""
    _assert_retired(
        "scripts/reify-closure-staleness-sweep.timer",
        "the 04:30 systemd user timer unit the installer copied into place",
    )


def test_service_unit_retired():
    """The systemd service unit template is gone."""
    _assert_retired(
        "scripts/reify-closure-staleness-sweep.service",
        "the thin oneshot service unit the timer triggered",
    )


def test_wrapper_script_retired():
    """The sweep wrapper script is gone."""
    _assert_retired(
        "scripts/reify-closure-staleness-sweep.sh",
        "the wrapper the service ran, which invoked reify's sweep and then the "
        "re-dispatch consumer",
    )


def test_consumer_retired():
    """The re-dispatch-request consumer is gone."""
    _assert_retired(
        "scripts/consume_redispatch_requests.py",
        "the gate_closure-predicate consumer — the second, opposite-policy "
        "owner of the stranded-blocked population, and the only invoker of "
        "reify's re-dispatch requests",
    )


def test_installer_suite_retired():
    """The installer's test suite is gone."""
    _assert_retired(
        "scripts/tests/test_install_reify_closure_staleness_sweep_timer.py",
        "the suite covering the deleted installer",
    )


def test_wrapper_suite_retired():
    """The wrapper's test suite is gone."""
    _assert_retired(
        "scripts/tests/test_reify_closure_staleness_sweep_wrapper.py",
        "the suite covering the deleted wrapper script",
    )


def test_consumer_suite_retired():
    """The consumer's test suite is gone."""
    _assert_retired(
        "scripts/tests/test_consume_redispatch_requests.py",
        "the suite covering the deleted consumer",
    )


_HARD_CONSTRAINT_MESSAGE = (
    "task 5247 HARD CONSTRAINT VIOLATED: {what} is a SURVIVING owner of the "
    "stranded-blocked population and must not be touched by the retirement."
)


def test_hard_constraint_scheduler_redispatch_phase_preserved():
    """HARD CONSTRAINT: the scheduler's blocked-redispatch phase survives."""
    p = REPO_ROOT / "orchestrator" / "src" / "orchestrator" / "scheduler.py"
    assert (
        p.is_file() and "_phase_redispatch_stranded_blocked" in p.read_text()
    ), _HARD_CONSTRAINT_MESSAGE.format(
        what=f"{p}::_phase_redispatch_stranded_blocked"
    )


def test_hard_constraint_harness_deterministic_recon_sweep_preserved():
    """HARD CONSTRAINT: the harness deterministic-recon sweep survives."""
    p = REPO_ROOT / "orchestrator" / "src" / "orchestrator" / "harness.py"
    text = p.read_text() if p.is_file() else ""
    missing = [
        symbol
        for symbol in (
            "_run_deterministic_recon_sweep",
            "_recover_stranded_deterministic_task",
        )
        if symbol not in text
    ]
    assert p.is_file() and not missing, _HARD_CONSTRAINT_MESSAGE.format(
        what=f"{p}::{{{', '.join(missing) or 'the deterministic-recon sweep'}}}"
    )


def test_hard_constraint_stage2_reconciliation_preserved():
    """HARD CONSTRAINT: fused-memory's Stage 2 reconciliation survives."""
    p = (
        REPO_ROOT
        / "fused-memory"
        / "src"
        / "fused_memory"
        / "reconciliation"
        / "stages"
        / "task_knowledge_sync.py"
    )
    assert p.is_file(), _HARD_CONSTRAINT_MESSAGE.format(
        what=f"{p} (Stage 2: Task-Knowledge Sync)"
    )
