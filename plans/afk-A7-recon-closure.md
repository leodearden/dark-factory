# AFK A7: Investigate dormant reconciliation findings (paused — needs scoping in another session)

## Status

**Paused.** The original plan assumed "escalations pile up without closure"; investigation showed the observable problem is different. Reframing required before re-queueing.

## Direct observations

1. `data/reconciliation/reconciliation.db`, table `runs`, column `stage_reports` (JSON TEXT): aggregating across recent runs yields **5,315 findings marked `actionable=true`**, broken down approximately as:
   - 2,299 `memory_stale`
   - 1,007 `task_memory_mismatch`
   - 407 `systemic_pattern`
   - 406 `cross_store_inconsistency`
   - remainder: `missing_knowledge`, `memory_contradiction`, `duplicate`, `other`

2. Filed-escalation totals on disk across both projects:
   - `dark_factory` (`/home/leo/src/dark-factory/data/escalations/`): 692 files, 10 `status=pending`
   - `reify` (`/home/leo/src/reify/data/escalations/`): 557 files, 2 `status=pending`
   - Combined `status=pending` AND `actionable=true` AND `category=recon_integrity_issue`: **0**

3. The original plan cited "~2,347 actionable pending escalations". That count does not appear in the escalation queue. It approximately matches the `memory_stale` finding subset in `stage_reports`.

4. `fused-memory/src/fused_memory/reconciliation/harness.py` contains:
   - `_escalate` at lines 615-638 (files a finding as an escalation; `detail` carries finding JSON)
   - `_run_remediation_pass()` at line 1149 (dispatches an LLM with actionable findings as context)
   - No code path auto-closes escalations after a remediation pass completes

## Open questions for the next session

- Why do 5,315 actionable findings sit in `stage_reports` without corresponding escalations on disk? (Conditional filing? Recent code change? Intentional triage filter?)
- Is the gap between findings and escalations the actual bug, or is the dormant pile by design?
- What does `_run_remediation_pass()` do with findings today, and does it produce or close anything that would shift these counts?

## Out of scope for now

The original A7 design (closure wiring + backfill) operated on the wrong problem. Forward-linkage / queue-driven closure choices documented in earlier discussion are preserved here for reference if a future session decides closure is part of the fix:

- **Forward linkage:** add optional `escalation_id` field to `FINDING_ITEM_SCHEMA`; capture submit-response at file-sites (`harness.py:1122-1127`, `:1265-1270`, `:1303-1307`); closure iterates findings.
- **Queue-driven:** revalidate all `status=pending AND actionable=true` each cycle; close those that re-pass. Cost ~200ms per finding (serial via CLI agent).
