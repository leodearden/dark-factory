"""Baseline allowlist for the silent-fallthrough-on-error lint gate.

Each entry in ALLOWLIST_ENTRIES documents a pre-existing violation that the
gate tolerates.  The gate FAILS on any violation NOT in this list (ratchet
semantics): adding new violations is not allowed; only removing them is.

Ratchet contract
----------------
- The integrity test asserts every entry here corresponds to a *real* current
  violation in the source tree.  When a violation is fixed, its entry becomes
  stale and the integrity test fails, forcing you to remove it.
- Every entry must carry a non-empty ``reason`` so the baseline is documented
  (non-silent), per the "loud escalation" directive.
- Entry keys are ``(relpath, qualname, content_hash)``; ``relpath`` is relative
  to the repository root and the file must exist; ``qualname`` must be a valid
  dotted Python identifier chain or the sentinel ``"<module>"``;
  ``content_hash`` is ``sha256(ast.unparse(node))[:12]`` — invariant under
  pure line-number drift and reindentation, changes on any edit to the handler.

Drift-resistance guarantee
--------------------------
Keys omit ``lineno`` entirely.  Pure line shifts (from edits above the site in
the same file) do not invalidate any entry.  Only a genuine change to the
exception handler or assignment (which alters the AST and therefore the hash)
causes a mismatch — forcing the entry to be re-blessed with the new hash.

Multiset ratchet
----------------
The gate uses ``collections.Counter`` subtraction, not set membership.
Duplicate keys (two byte-identical sites in the same function — e.g. the two
``statuses, _ = await self.scheduler.get_statuses()`` calls in Harness.run)
are legitimate: each copy in the tree must be matched by a copy here.
A full-tuple duplicate ``(relpath, qualname, content_hash, reason)`` is still
a copy-paste bug and is caught by the integrity test.

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
  (relpath: str, qualname: str, content_hash: str, reason: str)

  relpath      : path relative to repo root (forward slashes)
  qualname     : enclosing function dotted name computed by the scanner
  content_hash : sha256(ast.unparse(node).encode())[:12] — 12 lowercase hex
                 chars; recompute with the scanner when the handler changes
  reason       : one-line non-empty human explanation; must not be blank
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Baseline entries
# ---------------------------------------------------------------------------
# Format: (relpath, qualname, content_hash, reason)

ALLOWLIST_ENTRIES: list[tuple[str, str, str, str]] = [
    # --- Signature (a): discarded get_statuses error slot ---
    # These three sites in Harness.run use get_statuses() for advisory
    # reporting/counting only; the next or surrounding call binds err properly.
    # Fixing them is PRD-out-of-scope (48-site migration is complete).
    (
        "orchestrator/src/orchestrator/harness.py",
        "Harness.run",
        "fd62af5dda0b",
        "pre-existing best-effort get_statuses: PRD-tag pre-scan only; "
        "next get_statuses at :990 binds err; out of 48-site scope",
    ),
    (
        "orchestrator/src/orchestrator/harness.py",
        "Harness.run",
        "03a74c7dc80d",
        "pre-existing best-effort get_statuses: advisory report.total_tasks "
        "counter only; non-fatal on error; out of 48-site scope",
    ),
    (
        "orchestrator/src/orchestrator/harness.py",
        "Harness.run",
        "03a74c7dc80d",
        "pre-existing best-effort get_statuses: advisory cycle-reset "
        "total_tasks counter; non-fatal on error; out of 48-site scope",
    ),

    # --- Signature (b): silent broad-except returning empty literal ---
    # Each is a pre-existing benign handler outside the 48-site remediation
    # scope.  Fixing them requires editing files in multiple packages — a
    # cross-package scope violation for this task (scope: shared/tests/ only).
    (
        "dashboard/src/dashboard/data/metrics.py",
        "get_memory_24h_ago",
        "0dd11661f9ee",
        "debug-logged fail-safe: 24h-ago DB query failure returns empty metrics "
        "dict (dashboard display only, not on critical path)",
    ),
    (
        "fused-memory/src/fused_memory/reconciliation/flag_dedup.py",
        "filter_terminal_metadata_flags._safe_get_task",
        "673e1da28bdc",
        "debug-logged fail-safe with exc context: task-lookup error preserves "
        "reconciliation flag in-place (explicit fail-safe comment in code)",
    ),
    (
        "orchestrator/src/orchestrator/agents/briefing.py",
        "BriefingAssembler._mcp_search",
        "9c9af4cd3b98",
        "debug-logged fail-safe: MCP search error returns None for graceful "
        "briefing degradation (non-critical context enrichment)",
    ),
    (
        "orchestrator/src/orchestrator/b3_gate.py",
        "_read_latest_proposal",
        "92b98f5b67f9",
        "pre-existing optional DB accessor: any exception returns None; "
        "callers handle None as 'no proposal'; narrow fix deferred to follow-up",
    ),
    (
        "orchestrator/src/orchestrator/cargo_scope.py",
        "discover_workspace_crates",
        "4c97c2dcc30c",
        "debug-logged fail-safe: cargo workspace TOML parse failure returns "
        "empty crate set (non-fatal, workspace scoping is best-effort)",
    ),
    (
        "orchestrator/src/orchestrator/dry_run_unblock.py",
        "_capture_worktree_shas._rev_parse",
        "92b98f5b67f9",
        "pre-existing git subprocess inner helper: subprocess error returns "
        "None; callers treat None as missing SHA (non-fatal)",
    ),
    (
        "orchestrator/src/orchestrator/harness.py",
        "Harness._schedule_coro_threadsafe._log_if_raised",
        "87f35df5cdcf",
        "deliberate: future.exception() may itself raise CancelledError; "
        "bare return avoids infinite escalation loop inside done-callback",
    ),
    (
        "orchestrator/src/orchestrator/merge_queue.py",
        "_classify_main_health_red",
        "92b98f5b67f9",
        "pre-existing optional probe helper: exception during health check "
        "returns None (no proposal); callers handle None gracefully",
    ),
    (
        "orchestrator/src/orchestrator/verify.py",
        "run_main_tip_sweep",
        "933d5ce757a9",
        "debug-logged fail-safe: get_main_sha failure returns None to skip "
        "sweep entirely (background probe, non-critical)",
    ),
    (
        "orchestrator/src/orchestrator/verify.py",
        "run_main_tip_sweep",
        "e2a807e01521",
        "debug-logged fail-safe: unexpected error during main-tip sweep "
        "returns None; sweeps are background checks, not on critical path",
    ),
    (
        "scripts/orchestrator-watchdog.py",
        "_unit_start_elapsed_secs",
        "92b98f5b67f9",
        "pre-existing watchdog helper: systemctl timestamp parse error returns "
        "None; watchdog handles None as 'no elapsed time' (marked noqa:BLE001 in source)",
    ),
    (
        "shared/src/shared/cli_invoke.py",
        "_resolve_transcript_path",
        "c13b0ae5dbc0",
        "debug-logged fail-safe: glob error returns None; caller treats None "
        "as missing transcript (non-fatal optional lookup)",
    ),
]

# ---------------------------------------------------------------------------
# Derived lookup list (used by the gate test for multiset matching)
# ---------------------------------------------------------------------------

#: List of ``(relpath, qualname, content_hash)`` triples present in the baseline.
#: Used by test_silent_fallthrough_gate.py via reconcile_against_allowlist().
#: Duplicates are intentional — two same-key entries (e.g. the two identical
#: get_statuses() call sites in Harness.run) each need a separate copy here
#: so the Counter-based ratchet counts them correctly.
ALLOWLIST_KEYS: list[tuple[str, str, str]] = [
    (relpath, qualname, content_hash)
    for relpath, qualname, content_hash, _reason in ALLOWLIST_ENTRIES
]
