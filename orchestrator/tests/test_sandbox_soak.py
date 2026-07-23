"""Tests for orchestrator.sandbox_soak — the OS-sandbox rollout soak predicate
(PRD γ1/γ5) fronted by scripts/check_sandbox_soak.sh.

Structured over four layers, mirroring scripts/tests/test_recon_busy_check.py:
  * pure evaluate_soak / _sandbox_attributable_blocks taxonomy (no I/O),
  * read-only DB readers against constructed fixture SQLite stores,
  * a git-fixture check for the containment-probe-report-on-main condition,
  * subprocess-driven CLI tests asserting the full 0/1/2 exit-code contract.

Everything is derived from STRUCTURED queries over the event store + task
records — never transcript-grep (INV-2).
"""
from __future__ import annotations

from orchestrator import sandbox_soak

PROBE_REPORT_PATH = "docs/sandbox-containment-probe-report.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _applied_and_status(n_sandboxed: int, n_done: int):
    """Build (sandbox_applied_task_ids, task_status) with *n_sandboxed* distinct
    sandboxed tasks of which the first *n_done* are `done` (the rest
    `in_progress`)."""
    applied = {f"t{i}" for i in range(n_sandboxed)}
    status = {
        f"t{i}": ("done" if i < n_done else "in_progress")
        for i in range(n_sandboxed)
    }
    return applied, status


# ---------------------------------------------------------------------------
# Pure evaluate_soak verdict taxonomy — conditions (a) done-count & (b) report
# ---------------------------------------------------------------------------

def test_all_green_is_pass():
    applied, status = _applied_and_status(12, 12)
    v = sandbox_soak.evaluate_soak(applied, status, set(), [], True, min_done=10)
    assert isinstance(v, sandbox_soak.SoakVerdict)
    assert isinstance(v.metrics, dict)
    assert v.ok is True
    assert "PASS" in v.reason
    assert v.metrics["done_count"] == 12


def test_done_count_shortfall_fails_and_names_ratio():
    # Only 3 of the 12 sandboxed tasks reached done → below the >=10 bound.
    applied, status = _applied_and_status(12, 3)
    v = sandbox_soak.evaluate_soak(applied, status, set(), [], True, min_done=10)
    assert v.ok is False
    assert "3/10" in v.reason
    assert v.metrics["done_count"] == 3


def test_report_absent_fails_and_names_probe_report():
    applied, status = _applied_and_status(12, 12)
    v = sandbox_soak.evaluate_soak(applied, status, set(), [], False, min_done=10)
    assert v.ok is False
    assert "probe report" in v.reason.lower()
    assert PROBE_REPORT_PATH in v.reason
    assert v.metrics["report_present"] is False


def test_boundary_nine_fails_ten_passes():
    # The >=10 bound is a PRD-D6 spec constant — boundary-test 9 vs 10.
    applied9, status9 = _applied_and_status(10, 9)
    v9 = sandbox_soak.evaluate_soak(applied9, status9, set(), [], True, min_done=10)
    assert v9.ok is False
    assert "9/10" in v9.reason

    applied10, status10 = _applied_and_status(10, 10)
    v10 = sandbox_soak.evaluate_soak(applied10, status10, set(), [], True, min_done=10)
    assert v10.ok is True


def test_distinct_only_counts_sandboxed_done():
    # A `done` task that is NOT in the sandbox_applied set must not count.
    applied = {"t0", "t1"}
    status = {"t0": "done", "t1": "done", "t99": "done"}
    v = sandbox_soak.evaluate_soak(applied, status, set(), [], True, min_done=10)
    assert v.metrics["done_count"] == 2
