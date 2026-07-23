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


# ---------------------------------------------------------------------------
# Condition (c): sandbox-attribution heuristic (PRD Open Q2). A >=10-done +
# report-present base isolates condition (c) as the sole verdict driver.
# ---------------------------------------------------------------------------

def _green_base(extra_status=None):
    """10 distinct done sandboxed tasks — a base where only condition (c) can
    flip the verdict. *extra_status* injects additional (non-sandboxed) tasks."""
    applied = {f"s{i}" for i in range(10)}
    status = {f"s{i}": "done" for i in range(10)}
    if extra_status:
        status.update(extra_status)
    return applied, status


def test_arm1_blocked_with_sandbox_unavailable_is_attributed():
    applied, status = _green_base({"b1": "blocked"})
    v = sandbox_soak.evaluate_soak(applied, status, {"b1"}, [], True, min_done=10)
    assert v.ok is False
    assert "b1" in v.reason
    assert v.metrics["attributable_block_count"] == 1


def test_arm2_blocked_with_eacces_escalation_is_attributed():
    applied, status = _green_base({"b2": "blocked"})
    escalations = [
        {"task_id": "b2", "summary": "write denied: EACCES on /etc/foo",
         "category": "sandbox_denial"},
    ]
    v = sandbox_soak.evaluate_soak(applied, status, set(), escalations, True, min_done=10)
    assert v.ok is False
    assert "b2" in v.reason


def test_arm2_blocked_with_erofs_escalation_is_attributed():
    applied, status = _green_base({"b3": "blocked"})
    escalations = [
        {"task_id": "b3", "summary": "EROFS: read-only file system", "category": "x"},
    ]
    v = sandbox_soak.evaluate_soak(applied, status, set(), escalations, True, min_done=10)
    assert v.ok is False
    assert "b3" in v.reason


def test_blocked_with_neither_signal_is_not_attributed():
    applied, status = _green_base({"b4": "blocked"})
    v = sandbox_soak.evaluate_soak(applied, status, set(), [], True, min_done=10)
    assert v.ok is True
    assert v.metrics["attributable_block_count"] == 0


def test_recovered_sandbox_unavailable_task_is_not_attributed():
    # In sandbox_unavailable_task_ids but current status is `done` — recovered.
    applied, status = _green_base({"r1": "done"})
    v = sandbox_soak.evaluate_soak(applied, status, {"r1"}, [], True, min_done=10)
    assert v.ok is True
    assert v.metrics["attributable_block_count"] == 0


def test_sandbox_word_without_errno_token_is_not_attributed():
    # Arm 2 matches only the EACCES/EROFS tokens, never the word "sandbox".
    applied, status = _green_base({"b5": "blocked"})
    escalations = [
        {"task_id": "b5", "summary": "sandbox worktree containment note",
         "category": "sandbox"},
    ]
    v = sandbox_soak.evaluate_soak(applied, status, set(), escalations, True, min_done=10)
    assert v.ok is True
    assert v.metrics["attributable_block_count"] == 0


def test_multiple_attributable_blocks_named_and_counted():
    applied, status = _green_base({"b1": "blocked", "b2": "blocked"})
    escalations = [{"task_id": "b2", "summary": "EACCES denied", "category": "x"}]
    v = sandbox_soak.evaluate_soak(applied, status, {"b1"}, escalations, True, min_done=10)
    assert v.ok is False
    assert "b1" in v.reason and "b2" in v.reason
    assert v.metrics["attributable_block_count"] == 2


def test_attributable_blocks_helper_arms_and_exclusions():
    task_status = {
        "b1": "blocked",   # arm 1 (sandbox_unavailable)
        "b2": "blocked",   # arm 2 (EACCES escalation)
        "b3": "blocked",   # neither signal -> excluded
        "r1": "done",      # in unavailable but recovered -> excluded
        "b5": "blocked",   # 'sandbox' word only -> excluded
    }
    unavailable = {"b1", "r1"}
    escalations = [
        {"task_id": "b2", "summary": "write denied EACCES", "category": "x"},
        {"task_id": "b5", "summary": "sandbox containment", "category": "x"},
    ]
    blocks = sandbox_soak._sandbox_attributable_blocks(
        task_status, unavailable, escalations
    )
    assert blocks == ["b1", "b2"]
