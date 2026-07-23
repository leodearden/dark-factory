# RCA: reify `stage1_flag_marker` backlog non-drain (esc-2866-1) — research findings

Date: 2026-07-22 · Session: research-df-2866-3921791 · Status: research complete, escalation left PENDING for Leo's decision
Method: 4 parallel research agents (lifecycle code study, live deploy forensics, 10-task fix lineage, reify recon config) + direct live Mem0/Qdrant queries.

## TL;DR verdict

**The fixes on main are live and working as designed. The 07-21 "still not draining" finding is a timing artifact, not a defect.** Task 2853's in-cycle per-project Mem0 sweep has a deliberate **14-day age floor**; at finding time (07-21) **zero** of reify's 52 monitored records were old enough to be eligible. The first record crosses the floor **tonight, 2026-07-22 ~21:26 UTC**, and the pool self-drains cohort-by-cohort to zero by **~2026-07-28** with no operator action. The write path is retired + gated, so the pool is capped (no new records since 07-14 in *any* project).

Two genuine residual defects remain (see §4): **6 reify records are permanently invisible to every drain mechanism**, and the nightly sweep timer is hard-scoped to dark_factory.

## 1. Answers to the escalation's three asks

### Ask 1 — Are 2853/2596 actually deployed and live? YES, verified.
- Single shared fused-memory process (systemd `fused-memory.service`, editable install of `/home/leo/src/dark-factory` main) reconciles ALL projects including reify (reify spawns no server; `reify/dark-factory-orchestrator.yaml:514` → `127.0.0.1:8002`, registered via `DASHBOARD_KNOWN_PROJECT_ROOTS`).
- **2853** (`fdb263234f`, merged 07-20 13:47 UTC): live since the **07-20 14:21 UTC** restart (~34 min after merge) and across every restart since. Today's reify remediation run (07-22 06:22–06:39 UTC) recorded the fix's stat key `stale_mem0_flag_markers_gc_swept: 0` — the sweep **executes** and correctly deletes nothing (nothing eligible yet).
- **2596** (`68d72d7ea2`): live since ~07-15/16. Its `add_memory` tool-boundary gate demonstrably holds — zero new markers after 07-14 in reify AND dark_factory.
- The clean 07-14T09:25 accumulation cutoff = the ~07-14 restart finally picking up task 2406's writer retirement (merged 07-08/09) + 2596's gate (07-14 22:41 +0100) closing the LLM mimic path.

### Ask 2 — Do the fixes drain the pre-existing backlog, or only prevent new? BOTH — but lazily.
- 2853's `_sweep_stale_mem0_flag_markers` (task_knowledge_sync.py:1055-1120, wired :2279) enumerates the whole `source='stage1_flag_marker'` pool per project per cycle and deletes everything `created_at` > 14 days. The pre-existing backlog IS in scope — it just drains as it ages: reify cohorts 1+1+1 on 07-22/23/24, then 16, 10, 11, 12 → **zero by ~07-28**. dark_factory's 16 drain on the same schedule (in-cycle + nightly timer, both 14d).
- One-shot backfill exists if wanted: `sweep_orphan_flag_markers.py --apply --project-id reify --max-age-days 0` (script supports it; never run for reify — the nightly timer structurally sweeps dark_factory only).
- **Except the invisible 6** — see §4.1. No committed mechanism can ever delete those.

### Ask 3 — Is an 11th point-fix the right strategy? NO — and the "10 failed fixes" framing is wrong.
The lineage is not 10 independent failures of the same fix (see §3): 8 were prompt-level fixes that failed the way prompt-level fixes do; the deterministic fixes all **held**; and the current backlog exists because task **2228 W5-κ deleted the two working in-cycle sweeps** (1944, 2103) on 07-10 under the incorrect belief that the recon-ledger `gc()` replaced them (it never touches Mem0). 2853 is the restoration. Nothing is left to point-fix; what's warranted is a **watch** that the predicted drain actually completes, plus optional small hardening (§5).

## 2. Live evidence snapshot (2026-07-22)

| project | `source=` count | `kind=` count | created range | touched since creation |
|---|---|---|---|---|
| reify | 52 | 58 | 07-08T21:26 → 07-14T09:25 (source-tagged); kind-only back to 07-01 | never (`updated_at==created_at` on all) |
| dark_factory | 16 | 17 | 07-08 → 07-14 | never |

- dark_factory's own pool is equally stale — disproving the brief's leading hypothesis (a reify-only scoping gap). No project's pool drained, because no collector existed anywhere from 07-10 (2228 regression) to 07-20 (2853), and nothing has been age-eligible since.
- The escalation's "52" is the `source=` count; the true `kind=` pool is 58. The 6-record delta is defect §4.1.

## 3. Fix-lineage narrative (10 tasks: 1146, 1369, 1659, 1944, 2095, 2103, 2108, 2312, 2596, 2853)

1. **Prompt-level fixes repeatedly failed** (8 of them; 2596's own description: "8 prior prompt-level fixes exhausted"; backlog net-grew 31→43 mid-fix). Every fix that held is deterministic: retire the write path (2406, 07-08/09), reject off-script writes at the tool boundary (2596 gate, 07-14), code-run sweeps (1944/2103/2853 in-cycle; 1659/2108/2596 script).
2. **Self-inflicted regression**: 1944 (14d age sweep) + 2103 (terminal-task sweep) worked, project-agnostic, in-cycle. Task 2228 W5-κ deleted both on 07-10 (merge `9e6f9765f0`) assuming ledger `gc()` superseded them; `gc()` collects SQLite ledger rows only. Pool collector-less in every project 07-10→07-20. **2853 is the third implementation of the same sweep.**
3. **Scripted drains only ever ran against dark_factory**: `sweep_orphan_flag_markers.py` defaults `--project-id dark_factory` (:732) and the nightly timer wrapper (`scripts/fused-memory-flag-marker-sweep.sh:46`) passes no override. Non-df projects were only ever hand-drained ad hoc.
4. **Marker purpose**: cross-cycle dedup state per (task_id, flag_type) — NOT the Stage1→Stage2 relay (that's the separate `flag_for_stage2` channel; FIX C deletes those). Since 2406, dedup state lives solely in the recon_ledger (SQLite, `expires_at=now+14d`); the Mem0 records are dead weight.

## 4. Genuine residual defects

### 4.1 Six reify records are permanently invisible (P1 of the leftovers)
6 records have `kind='stage1_flag_marker'` but **no `source` key** (legacy shape; dates 07-01→07-13; ids `5df46ff5`, `f1afcff8`, `c740aacd`, `922e637d`, `18df7117`, `9ab0facf`). Every mechanism filters on `source=`: in-cycle sweep (:918), script enumeration (:474/487), the count short-circuit probe, even `--delete-ids` (intersected with the source-filtered list :533). Proof: the 07-01 record is already >14d past cutoff and untouched. Once the 52 drain, the `source=` count probe returns 0 and short-circuits the sweep every cycle, cementing them. dark_factory has 1 analogous record. Only direct `delete_memory` calls (or new code) can remove them. Recurrence risk ≈ 0 (write paths retired + gated), so hand-deletion without a code fix is defensible.

### 4.2 Nightly timer is dark_factory-only (now mostly moot)
`fused-memory-flag-marker-sweep.timer` (03:30, active, exit 0) structurally never sweeps any other project. With 2853's per-project in-cycle sweep live, the timer is largely redundant — but its census output ("orphan_count: 0") is misleading fleet-wide health signal.

### 4.3 Observability traps
- `recon_markers_gc_swept` counts SQLite **ledger** rows only; the Mem0-pool stat is `stale_mem0_flag_markers_gc_swept`. Monitoring the former to conclude "GC works" is the exact mistake the 2228 regression made.
- Recon findings count the pool via `source=` — blind to the §4.1 class by construction.

## 5. Options for Leo

### O1 — Pure wait (zero action)
Let the 52 self-drain by ~07-28; leave the 6 invisible records. Cost: recon Stage-1 will likely keep re-raising the non-drain finding during the window (noise); the 6 sit forever (unmonitored junk, but junk). Validates 2853 end-to-end in production.

### O2 — Minimal targeted cleanup (recommended)
1. **Hand-delete the 7 invisible records** (6 reify + 1 df) via direct `delete_memory` — nothing else will ever remove them. ~5 min.
2. **Let the 52+16 self-drain naturally** — deliberately, as a production validation of 2853 given this lineage's history of "done ≠ working".
3. **File ONE deterministic watch task**: `task_kind='deterministic'`, `milestone={mode:'delayed', after_secs:~7d}` (or `dated` 2026-07-29), `before_done={kind:'predicate', script: sweep script wrapper, args: ['--check','--max-backlog','0','--project-id','reify']}` — exactly the pattern 2596's `--check`/`backlog_verdict` was built for and CLAUDE.md's milestone exemplar. Exit 0 → silently done; non-zero → born-at-L2 `milestone_check_failed`. Zero LLM cost, closes the loop without an 11th point-fix. (Optionally a df twin.)
4. Then resolve esc-2866-1 with this RCA as resolution.

### O3 — Immediate manual drain
`sweep_orphan_flag_markers.py --apply --project-id reify --max-age-days 0` now (+ df, + hand-delete the 7). Instant zero, silences recon noise immediately — but forfeits the production validation of 2853's drain path, which is worth having after two prior "done" sweeps evaporated.

### Optional hardening (could be one `complexity=simple` task or folded into a small /prd; none urgent)
- a. `kind=`-filtered enumeration fallback in script + in-cycle sweep (closes §4.1 class structurally; arguably YAGNI since writes are retired+gated).
- b. Retire the nightly timer, or give the wrapper a per-project loop (redundancy/misleading-census cleanup).
- c. Doc/stat clarification for §4.3 (e.g. comment at `recon_markers_gc_swept` emission; recon prompts already partially updated by 2596).

### NOT recommended
- An 11th investigate/point-fix task — nothing left to fix in the drain path.
- A full flag-relay pipeline audit — the pipeline was already structurally retired (2406 + 2596 gate); remaining items are cleanup-grade.

## 6a. O2 APPLIED (2026-07-22, Leo-ratified)

1. **7 invisible records hand-deleted** via `delete_memory` (6 reify: `5df46ff5`, `f1afcff8`, `c740aacd`, `922e637d`, `18df7117`, `9ab0facf`; 1 df: `bfc30fd3`). Counts converged: reify `source==kind==52`, df `16==16`.
2. **52+16 left to self-drain** as production validation of 2853.
3. **Watch task 2902 filed** (`task_kind=deterministic`, milestone dated `2026-07-29T12:00Z`, `before_done.kind=predicate` → `scripts/fused-memory-flag-marker-check.sh --project-id reify --max-backlog 0`). New check wrapper committed `0f1cf3547e`, validated live (exit 1 @ ceiling 0 / exit 0 @ ceiling 52). NOTE: the normal-path submit was drop-combined into 2866 by the curator ("the human resolving 2866 will check this") — refiled via `planning_mode` + `commit_planning` since the watch must outlive 2866.
4. **esc-2866-1 resolved** (`action=resume`, class `actionable`); task 2866 → done (`done_provenance.kind=deterministic-gate`).
5. **Hardening (Leo-ratified follow-up): task 2917** (`complexity=simple`, low) bundles option b (per-projectize the nightly sweep wrapper) + option c's cheap tier (warning comment at `recon_markers_gc_swept` emission + one clarifying line in prompts/stage2.py). Option a (`kind=`-filter fallback) deliberately parked pending 2902's verdict.

## 6. Verification hooks (for whoever acts)
- Tonight after ~21:30 UTC: reify `count_memories_by_metadata(source=stage1_flag_marker)` should read ≤51; by 07-28 → 0.
- Per-cycle: `stale_mem0_flag_markers_gc_swept` > 0 in reify task_knowledge_sync stats once cohorts age in.
- The 6 invisible ids enumerable via `get_memories_by_metadata(project_id='reify', filters={'kind':'stage1_flag_marker'})` minus the source-tagged set.
