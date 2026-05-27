# Escalation L2 Tiering — design & plan

**Status:** planned 2026-05-27. Tasks filed in `dark_factory` (see "Task batch" below).
**Origin:** `/deb` investigation into the reify `/escalation-watcher` session colliding with the
orchestrator's autonomous escalation handler (auto-watcher PID 219222 + per-task stewards), where the
human's `resolve_issue` calls returned idempotent no-ops stamped `resolved_by=steward`.

## Problem (what we're fixing)

Two defects, one structural and one data-integrity:

1. **Uncoordinated handlers on one level.** The orchestrator's `escalation-watcher-auto` agent and a
   human-run `/escalation-watcher` session both drain the *same* L1 queue and both call `resolve_issue`.
   `resolve()` is idempotent (`queue.py:291` — `status != 'pending'` → no-op returns the existing esc),
   so the loser silently loses. Per-task L0 stewards add a third competing resolver. There is no lock,
   lease, or level separation.

2. **Steward resurrects archived escalations into the queue root.** `steward._patch_resolution_metadata`
   (`steward.py:740-755`) calls `_rewrite` → `_atomic_write`, which always writes to `queue_dir/{id}.json`
   (the root) *after* `resolve()` already moved the file to the archive (`queue.py:306`). This re-creates
   the file in the root stamped `resolved_by=steward`, while the archive copy keeps `resolved_by=null`,
   same `resolved_at`. Result: **561 of 564 files in the reify queue root are actually resolved/dismissed**
   (307 resolved + 254 dismissed), only 3 genuinely pending — and the audit trail (root vs archive)
   disagrees. This is also the proximate source of the misleading `resolved_by=steward` the operator saw,
   because the MCP server's `get()` finds the resurrected root copy first.

## Target design: a 3-tier escalation ladder

Disjoint consumers per level — the race is removed *by construction* (no shared resource), not by locking.

| Level | Producer → Consumer | Consumer |
|------|---------------------|----------|
| **L0** | agent → per-task steward | in-process steward (handles or re-escalates) |
| **L1** | steward / workflow → **escalation-watcher-auto** | autonomous agent (handles or re-escalates) |
| **L2** | escalation-watcher-auto → **human** | interactive `/escalation-watcher` session |

**Consumer-per-level is a documented contract** (`models.py`, both SKILL.md, roles.py). The bug we
diagnosed grew from "two consumers silently ended up on L1"; making the mapping explicit is what stops
it drifting back.

### Auto-watcher's new job: triage + root-cause analysis, not just admin dispatch

The auto-watcher (L1 consumer) becomes the funnel to the human. For every L1 escalation it either:
- **Handles autonomously** (existing admin classes: scope_violation, dependency_discovered, cleanup_needed), or
- **Promotes to L2** via a new `promote_to_l2` tool when human judgement is genuinely needed — including
  single judgement-class items (1-member L2) and **causal clusters**.

**Causal cluster detection (the high-value signal).** The capability that makes interactive sessions
valuable today moves *down* a tier. The auto-watcher forms a root-cause *hypothesis* that explains several
escalations and promotes the cluster as one L2 with: hypothesized root cause, evidence, member escalation/
task ids, and concrete options (A/B/C + "something else") — mirroring *"a cluster appears to be forming
around X — what would you like to do?"*. Clusters are **causal, not superficial** — members will not
reliably share a category/signature; the common factor is often causal.

**Shallow-by-default RCA.** RCA stays shallow until escalations carry reasons to suspect a common cause
(e.g. repeated failures on the same module/merge, a burst of infra symptoms, sibling tasks of one PRD all
stalling). Depth scales with that signal so a quiet queue stays cheap (auto-watcher is opus/high-effort,
$40/rotation under a $50/day ceiling).

**Embedded architecture map (priors).** The skill ships a high-level system map so the agent has good
priors about likely root causes:

- **fused-memory** (MCP :8002) — Graphiti (KG) + Mem0 (vectors) + Taskmaster behind one interface; task
  store, reconciliation, curator. Symptom of trouble: MCP calls failing / reconciliation/curator escalations.
- **orchestrator** — harness (lifecycle, supervisors), scheduler (v2: parks/module-locks/preemption),
  per-task steward (L0), workflow (TDD phases), agents (architect/implementer/reviewer).
- **escalation** — file-backed queue + MCP server + inotify watcher; the ladder above.
- **merge queue** — serialized merges via escalation MCP `merge_request`.
- **per-project targets** — reify (Rust CAD kernel), know-live, etc., each with its own queue dir.

Root-cause classes and where they surface:
- **infra** — fused-memory/Neo4j/Qdrant/jobserver down, disk full → bursts of `infra_issue` + MCP errors
  across unrelated tasks. (Auto-watcher is read-only; it *hypothesizes* infra from symptoms, it cannot
  probe host health. The case where infra kills the auto-watcher itself is covered by the supervisor
  failsafe below.)
- **implementation** — a bad merge to main breaks dependents (many tasks fail verify/build on the same
  module); a task marked done that didn't fulfil its contract (`bypass_done`; dependents fail to use it).
- **design** — PRD mis-decomposition / wrong architecture → sibling tasks hit scope_violation /
  design_concern around one area.

### Decisions locked in this design

- **Failsafe = supervisor-files-L2 only.** When the auto-watcher supervisor pauses/dies (crashloop guard,
  cost ceiling, disabled), the *supervisor* files an L2 "auto-watcher down, N L1 pending". (No read-only
  L1 peek in the interactive session — keeps it strictly L2.) This is the backstop for "nobody watches L1".
- **One evolving L2 per root cause.** Before filing, the auto-watcher checks pending L2s for the same root
  cause and *updates* the existing one (adds newly-implicated members) instead of re-pinging. Member L1s
  **stay pending at L1**, referenced by the L2; resolving the L2 cascades to them — so dissolving a cluster
  loses nothing.
- **Born-at-L2 = severity-gated.** Escalations whose severity is critical/urgent are born straight at L2 at
  the creation chokepoint (uniform rule, not a hand-enumerated category list).
- **Defect-2 fix bundled** with a one-time cleanup sweep of the stale resolved-in-root files.
- **Graceful degradation:** the new auto-watcher skill must feature-detect `promote_to_l2`; if absent
  (server not yet redeployed) it falls back to the old leave-pending + digest behavior. This makes the
  skill safe to land before the orchestrators are restarted onto the new server.

## Task batch (single-package scopes — respects the architect $12 / one-package rule)

Package **escalation**:
- **E1** — Add L2 to the level model + document the consumer-per-level contract + severity-gated
  born-at-L2 at the creation chokepoint. Files: `escalation/src/escalation/models.py`,
  `escalation/src/escalation/server.py`, tests.
- **E3** — Make resolution-metadata patching archive-aware: add a queue method that updates a resolved
  escalation *in place wherever it lives* (root or archive), never resurrecting it to root. Fixes Defect-2
  at the queue layer. Files: `escalation/src/escalation/queue.py`, tests.
- **E2** — `promote_to_l2` MCP tool + queue mechanics (root-cause dedup, member-linking, resolve-cascade).
  Files: `escalation/src/escalation/server.py`, `escalation/src/escalation/queue.py`, tests. Deps: E1, E3.
- **E4** — Tested one-time **sweep tool** to relocate resolved/dismissed files from queue root → archive and
  reconcile root/archive divergence. Tool only — the live run against real queue dirs is a manual deploy
  step (do not have the orchestrator touch live escalation data). Files: new script under
  `escalation/` or `scripts/`, tests. Deps: E3.

Package **orchestrator**:
- **O1** — Add `promote_to_l2` to `_WATCHER_ALLOWED_TOOLS` + supervisor failsafe (file L2 on auto-watcher
  pause/death). Files: `orchestrator/src/orchestrator/harness.py`, tests. Deps: E2.
- **O2** — Steward Defect-2 fix: use the archive-aware patch (E3) instead of `_rewrite`-to-root; no
  resurrection. Files: `orchestrator/src/orchestrator/steward.py`, tests. Deps: E3.
- **O3** — Fix "L1 = human" assumptions: `roles.py:783,823` ("re-escalate to level=1 (steward→human)")
  and audit L1-means-human comments in workflow/scheduler. Files: `orchestrator/src/orchestrator/agents/roles.py`,
  `orchestrator/src/orchestrator/workflow.py`, tests. Deps: E1.

Package **skills**:
- **S1** — Rewrite `escalation-watcher-auto`: shallow-by-default RCA that deepens on common-cause signal;
  embedded architecture map; `promote_to_l2` for judgement-class items and causal clusters
  (hypothesis/evidence/options); one-evolving-L2-per-root-cause dedup; feature-detect `promote_to_l2` with
  digest fallback. File: `skills/escalation-watcher-auto/SKILL.md`. Deps: E2, O1.
- **S2** — Switch interactive `escalation-watcher` to L2-only (skill filter + watcher subprocess
  `--level 2`); drop its own cluster analysis (now upstream). File: `skills/escalation-watcher/SKILL.md`.
  Deps: E1.

### DAG
```
E1 ─┬─ E2 ── O1 ── S1
    ├─ O3        ╱
    └─ S2      E2
E3 ─┬─ E2
    ├─ O2
    └─ E4
```

## Deployment & rollout (manual, after the batch lands)

Code (E*/O*) takes effect on process **restart**; skills (S*) are read fresh per rotation/session.
1. Land all E*/O* code tasks.
2. Restart **both** orchestrators (`systemctl --user restart orchestrator-dark-factory.service
   orchestrator-reify.service`) so they load the new escalation server + harness; verify `promote_to_l2`
   is registered.
3. S1/S2 are safe to land anytime (S1 feature-detects the tool). Once the servers have the tool, the L2
   path is live; the interactive session switches to L2.
4. Run the E4 sweep tool against the live queue dirs (reify + dark-factory) in a quiet window. Never delete
   `-wal`/`-shm`; move, don't delete; verify counts before/after.

Note: this is self-hosting — the df orchestrator modifies the factory's own escalation/orchestrator code.
Each orchestrator runs its old code until restarted, and they operate on separate queue dirs, so there is
no live cross-contamination during the work.
