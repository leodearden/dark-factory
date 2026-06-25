"""Baseline allowlist for the silent-fallthrough-on-error lint gate.

Each entry in ALLOWLIST_ENTRIES documents a pre-existing violation that the
gate tolerates.  The gate FAILS on any violation NOT in this list (ratchet
semantics): adding new violations is not allowed; only removing them is.

Ratchet contract
----------------
- The step-7 integrity test asserts every entry here corresponds to a *real*
  current violation in the source tree.  When a violation is fixed, the entry
  becomes stale and the integrity test fails, forcing you to remove it.
- Every entry must carry a non-empty ``reason`` so the baseline is documented
  (non-silent), per the "loud escalation" directive.
- Entry keys are ``(relpath, lineno, qualname)``; ``relpath`` is relative to
  the repository root and the file must exist; ``qualname`` must be a valid
  dotted Python identifier chain or the sentinel ``"<module>"``.

Follow-up recommendation
------------------------
Task σ (this gate) seeds the baseline with the ~15 residuals present on the
migrated tree.  A follow-up task should:
1. Widen signature (b) to the *full* spec (add False/0/'' exclusion removal;
   add narrow typed-handler coverage) and burn the baseline down to zero.
2. After each violation is fixed, delete the corresponding ALLOWLIST_ENTRIES
   row so the integrity test confirms the shrinkage.

Entry schema
------------
Each entry is a 4-tuple:
  (relpath: str, lineno: int, qualname: str, reason: str)

  relpath  : path relative to repo root (forward slashes)
  lineno   : line number of the ExceptHandler or Assign node in the CURRENT
             tree — update if the file is edited and lines shift
  qualname : enclosing function dotted name computed by the scanner
  reason   : one-line non-empty human explanation; must not be blank
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Baseline entries
# ---------------------------------------------------------------------------
# Format: (relpath, lineno, qualname, reason)

ALLOWLIST_ENTRIES: list[tuple[str, int, str, str]] = [
    # --- Signature (a): discarded get_statuses error slot ---
    # These three sites in Harness.run use get_statuses() for advisory
    # reporting/counting only; the next or surrounding call binds err properly.
    # Fixing them is PRD-out-of-scope (48-site migration is complete).
    (
        "orchestrator/src/orchestrator/harness.py",
        985,
        "Harness.run",
        "pre-existing best-effort get_statuses: PRD-tag pre-scan only; "
        "next get_statuses at :990 binds err; out of 48-site scope",
    ),
    (
        "orchestrator/src/orchestrator/harness.py",
        1034,
        "Harness.run",
        "pre-existing best-effort get_statuses: advisory report.total_tasks "
        "counter only; non-fatal on error; out of 48-site scope",
    ),
    (
        "orchestrator/src/orchestrator/harness.py",
        1146,
        "Harness.run",
        "pre-existing best-effort get_statuses: advisory cycle-reset "
        "total_tasks counter; non-fatal on error; out of 48-site scope",
    ),

    # --- Signature (b): silent broad-except returning empty literal ---
    # Each is a pre-existing benign handler outside the 48-site remediation
    # scope.  Fixing them requires editing files in multiple packages — a
    # cross-package scope violation for this task (scope: shared/tests/ only).
    (
        "dashboard/src/dashboard/data/metrics.py",
        738,
        "get_memory_24h_ago",
        "debug-logged fail-safe: 24h-ago DB query failure returns empty metrics "
        "dict (dashboard display only, not on critical path)",
    ),
    (
        "fused-memory/src/fused_memory/reconciliation/flag_dedup.py",
        1470,
        "filter_terminal_metadata_flags._safe_get_task",
        "debug-logged fail-safe with exc context: task-lookup error preserves "
        "reconciliation flag in-place (explicit fail-safe comment in code)",
    ),
    (
        "orchestrator/src/orchestrator/agents/briefing.py",
        923,
        "BriefingAssembler._mcp_search",
        "debug-logged fail-safe: MCP search error returns None for graceful "
        "briefing degradation (non-critical context enrichment)",
    ),
    (
        "orchestrator/src/orchestrator/b3_gate.py",
        434,
        "_read_latest_proposal",
        "pre-existing optional DB accessor: any exception returns None; "
        "callers handle None as 'no proposal'; narrow fix deferred to follow-up",
    ),
    (
        "orchestrator/src/orchestrator/cargo_scope.py",
        54,
        "discover_workspace_crates",
        "debug-logged fail-safe: cargo workspace TOML parse failure returns "
        "empty crate set (non-fatal, workspace scoping is best-effort)",
    ),
    (
        "orchestrator/src/orchestrator/dry_run_unblock.py",
        141,
        "_capture_worktree_shas._rev_parse",
        "pre-existing git subprocess inner helper: subprocess error returns "
        "None; callers treat None as missing SHA (non-fatal)",
    ),
    (
        "orchestrator/src/orchestrator/harness.py",
        5884,
        "Harness._schedule_coro_threadsafe._log_if_raised",
        "deliberate: future.exception() may itself raise CancelledError; "
        "bare return avoids infinite escalation loop inside done-callback",
    ),
    (
        "orchestrator/src/orchestrator/merge_queue.py",
        780,
        "_classify_main_health_red",
        "pre-existing optional probe helper: exception during health check "
        "returns None (no proposal); callers handle None gracefully",
    ),
    (
        "orchestrator/src/orchestrator/verify.py",
        2960,
        "run_main_tip_sweep",
        "debug-logged fail-safe: get_main_sha failure returns None to skip "
        "sweep entirely (background probe, non-critical)",
    ),
    (
        "orchestrator/src/orchestrator/verify.py",
        3014,
        "run_main_tip_sweep",
        "debug-logged fail-safe: unexpected error during main-tip sweep "
        "returns None; sweeps are background checks, not on critical path",
    ),
    (
        "scripts/orchestrator-watchdog.py",
        190,
        "_unit_start_elapsed_secs",
        "pre-existing watchdog helper: systemctl timestamp parse error returns "
        "None; watchdog handles None as 'no elapsed time' (marked noqa:BLE001 in source)",
    ),
    (
        "shared/src/shared/cli_invoke.py",
        277,
        "_resolve_transcript_path",
        "debug-logged fail-safe: glob error returns None; caller treats None "
        "as missing transcript (non-fatal optional lookup)",
    ),
]

# ---------------------------------------------------------------------------
# Derived lookup set (used by the gate test for O(1) matching)
# ---------------------------------------------------------------------------

#: Set of ``(relpath, lineno)`` pairs present in the baseline.
#: Used by test_silent_fallthrough_gate.py for gate-time violation filtering.
ALLOWLIST_VIOLATIONS: frozenset[tuple[str, int]] = frozenset(
    (relpath, lineno)
    for relpath, lineno, _qualname, _reason in ALLOWLIST_ENTRIES
)
