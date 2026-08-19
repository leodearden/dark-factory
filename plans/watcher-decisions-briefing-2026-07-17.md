# Discussion briefing — harness/process decisions surfaced by the overnight L2 watcher (2026-07-17)

You are a discussion session for **Leo**. Walk him through each item below in detail, discuss the
options, help him decide the disposition, and **act on his decisions** (file follow-up tasks via the
fused-memory MCP with `project_root=/home/leo/src/dark-factory`; use `/prd` if an item is
design-heavy). Do NOT act until Leo decides each one.

Full context lives in: the cockpit decision registry
(`python3 orchestrator/src/orchestrator/session_registry.py` — inspect the decisions dir under
`~/.claude/fleet/` or `data/`), `data/escalations/afk-digest.md`, and the named escalation records
(`mcp__escalation__get_escalation`). Read what you need before diving in.

## 1. Revalidation-sweep drops operator-action escalations on done tasks
Cockpit decision id: `revalidation-sweep-drops-operator-action-on-done-task` (severity: blocking).

**Problem:** `harness-escalation-revalidation-sweep` auto-closes escalations as "moot (subject task
done)" on a **status-only** heuristic. That silently DROPS a still-required post-merge operator
action whenever *acceptance ≠ task-done*.
- Real instance: it closed **esc-2693-3** (a required flag-marker-sweep-timer deploy) at 22:33Z,
  ~4 min before the deploy actually ran. Only because the watcher was already mid-execution did it
  get done. Deploy-completion had to be recorded on task 2693's metadata (`x_flag_marker_sweep_deploy`)
  because the escalation record wrongly read "moot".
- It's recurring: task 2596's `esc-2596-1` was likewise auto-dismissed ("stale from prior run") and
  its follow-up "never filed" — which is why task 2693 had to be created later after the backlog grew.

**Recommendation to discuss:** exclude `risk_identified`/operational-action escalations from the
done→moot sweep, OR gate the auto-close on acceptance-criteria (e.g. a predicate/`before_done` check)
rather than task status alone. Decide whether to file a fix task (or `/prd`).

## 2. Orphan-reaper false-positives on rebase-superseded step SHAs
Cockpit decision id: `orphan-reaper-false-positive-rebase-superseded-step-sha` (severity: info).

**Problem:** the harness-orphan-reaper files false-positive "orphan step commit" notices for tasks
whose recorded step SHA was superseded by a requeue/rebase-merge — the SHA isn't an ancestor of main,
but the step's *content* landed under a new SHA, so no work is lost.
- ≥4 verified-benign instances in one window: tasks **2679, 2692, 2678, 2586** (all done/merged).
  Verified the mechanism first-hand on esc-2679-9 (step-7 `fb62c8e439` → rebased `4b1510ba28` on main).
- Each needs manual triage+close = recurring noise.

**Options (from esc-2586-10):** (A) harden the reaper to check the parent task's terminal/merged
status (done_provenance) before filing; (B) leave as-is (cheap info-severity no-ops, auto-watcher
closes them); (C) rate-limit/dedupe this class like the `curator_zero_output_hang` burst-suppression.
Decide.

## 3. Optional: re-scope task 2627 cadence-stamp atomicity
Cockpit decision id: `rescope-2627-cadence-stamp-atomicity` (severity: info).

Task 2627 was **cancelled** (premise verified false — there is NO deterministic dark_factory code
path that writes the 5 cadence-stamp fields; the stamps are written by the Stage 2 reconciliation
**LLM agent** via `update_task`, governed by an autopilot_video Mem0 procedural_knowledge memory,
id `349646af`, `prompts/stage2.py:404`). Both known partial-write incidents on autopilot_video task
452 are already corrected.

**Optional, low-value re-scope:** harden `prompts/stage2.py` + the autopilot_video Mem0 cadence memory
to mandate all cadence-stamp fields be written in ONE atomic `update_task` (prevent a 3rd partial-write
drift). It targets an autopilot_video LLM convention, not dark_factory code. **Default = leave
cancelled.** Discuss whether the drift risk warrants the work.

## Bonus finding (related to #1) — StaleServiceRestartCoordinator didn't fire
While resolving the fused-memory restart gate (esc-2686, task 2686 — now done; restart performed
2026-07-17 08:13 BST, activated task 2622's fix `731e1cf315`), the task body flagged that the
**StaleServiceRestartCoordinator (task 1592, merged 2026-06-04)** should have auto-restarted
fused-memory.service after task 2622's merge to `fused_memory/reconciliation/targeted.py` but did
NOT (the pre-merge PID was still live ~1h26m post-merge). Worth checking whether its `require_idle`
gate or file-scope-matching logic regressed — a likely follow-up task, and it's the same "operational
restart didn't happen automatically" theme as #1.

---
_Suggested flow: read context → walk Leo through #1, #2, #3, then the bonus → act on each decision he
makes (file tasks / PRDs), and close/annotate the corresponding cockpit decision records as resolved._
