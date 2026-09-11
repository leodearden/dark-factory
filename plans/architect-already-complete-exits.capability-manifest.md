# Capability manifest — `plans/architect-already-complete-exits.md`

Mechanizes G3 + G6 per task. Each capability a task's signal asserts is bound to
evidence on main (`grep:<file>:<line> wired`) or to an upstream producer in the
batch (`producer:<label> upstream`). Every binding is **PASS** — no FAIL, no
waiver. Line numbers verified against main 2026-07-24 (they drift slightly from
the PRD's own citations, which were written ~82 min before decompose; every
symbol + semantic is present).

DAG (real dependency edges): `α→γ`, `α→ζ`, `β→ζ`, `β→ε`, `γ→ε`, `δ→ε`.
`δ` *relates* `β` (informational; the two halves of C are independently landable
so there is no `β→δ` edge). Leaf-by-DAG: `ε`, `ζ`. Every other task carries its
own end-to-end signal (`β`, `γ`, `δ`) or is a pure intermediate substrate
extension (`α`).

---

## α — `mark_step_committed` plan-tool (intermediate; unlocks γ, ζ)

Signal: calling it flips a step `status→done` + tags `[COMMITTED <sha>]`;
`get_pending_steps()` excludes it; `confirm_plan` accepts an all-done plan.

| Capability | Evidence | Verdict |
|---|---|---|
| plan-tools MCP registers a new `@mcp.tool()` exit (α extends this surface) | `grep:orchestrator/src/orchestrator/mcp/plan_tools.py:712` — `report_task_already_done` sibling tool + `create_..._server` registration at `:500` | PASS (wired) |
| step objects default `status:'pending'`; only a status flip is needed | `grep:orchestrator/src/orchestrator/mcp/plan_tools.py:184,212` — `'status': 'pending'` | PASS (wired) |
| `get_pending_steps()` excludes `done` steps ⇒ all-done ⇒ EXECUTE no-op | `grep:orchestrator/src/orchestrator/artifacts.py:706` — `def get_pending_steps` | PASS (wired) |
| `confirm_plan` exists (α ensures it accepts an all-`done` plan) | `grep:orchestrator/src/orchestrator/mcp/plan_tools.py:660` — `def confirm_plan`; `_finalized_at` set at `artifacts.py:919` | PASS (wired) |
| authoring guard `git cat-file -e <sha>` corroborates the sha on-branch (INV-3) | git plumbing; standard subprocess pattern used across `git_ops.py` | PASS (wired) |

## β — `report_ready_to_merge` exit + `_handle_ready_to_merge_report` handler (integration-gate; unlocks ζ, ε)

Signal (boundary tests C-1..C-4): on a clean-FF + verify-passed + review-PASS
task, the exit enqueues a `MergeRequest` (`source='architect-desync'`) that lands
the branch on main with **no L1 filed**; idempotent on re-invocation.

| Capability | Evidence | Verdict |
|---|---|---|
| `.task/ready_to_merge.json` artifact follows the established exit-report convention | `grep:orchestrator/src/orchestrator/workflow.py:5212` — `.task/already_done.json` handler (`:5286` unactionable, `:5327` false_premise) | PASS (wired) |
| handler sibling to `_handle_already_done_report` | `grep:orchestrator/src/orchestrator/workflow.py:5211` — `async def _handle_already_done_report` | PASS (wired) |
| clean-FF corroboration via `git merge-base --is-ancestor` both directions (INV-3) | git plumbing; `_handle_already_done_report` already uses `--is-ancestor` (`workflow.py:5211`+) | PASS (wired) |
| latest `workflow_verify` passed + `tip_sha`==branch tip | `grep:orchestrator/src/orchestrator/stranded_verified_green.py:49` — `def last_verified_green_tip` | PASS (wired) |
| build `MergeRequest` + `enqueue_merge_request(..., source=...)` + done-callback + durable marker — **reuse/extract the reaper block, do not copy (INV-5)** | `grep:orchestrator/src/orchestrator/harness.py:4400` — `req = MergeRequest(`; `:4418` `enqueue_merge_request(..., source='stranded-reaper')`; `:4431` marker stamp; `merge_queue.py:3691` `def enqueue_merge_request` | PASS (wired) |
| merge success → `mark_done(kind='found_on_main')` | `grep:orchestrator/src/orchestrator/harness.py:975,4160` — `kind='found_on_main'` | PASS (wired) |
| merge worker scoped re-verify is the sole main-advancing gate | `merge_queue.py:81` — `_reverify_rebased_tree` import; queue re-verify machinery | PASS (wired) |
| "no L1 filed" — behavioral end-to-end (branch taken instead of `report_unactionable`'s L1) | boundary test C-1 (event log shows no escalation filed) — β's own deliverable | PASS (manual) |

## γ — architect behavior wiring (has own leaf signal; unlocks ε)

Signal (boundary tests A1-1..A1-3): a re-invoked architect on an
already-committed-green branch emits an all-`mark_step_committed` plan ⇒ **0
EXECUTE iterations** in the event log ⇒ branch lands via VERIFY→MERGE.

| Capability | Evidence | Verdict |
|---|---|---|
| `mark_step_committed` exists to author against | `producer:α upstream` (γ depends on α) | PASS (DAG-direction) |
| EXECUTE empty-pending-steps loop is a total no-op | `grep:orchestrator/src/orchestrator/workflow.py:5931` — `while self.artifacts.get_pending_steps():` | PASS (wired) |
| EXECUTE iteration count is observable | `grep:orchestrator/src/orchestrator/workflow.py:922` — `execute_iterations: int = 0` metric | PASS (wired) |
| VERIFY remains the semantic gate for a falsely pre-satisfied step (A1-3) | VERIFY stage exists downstream of EXECUTE (`workflow.py` PLAN→EXECUTE→VERIFY) | PASS (wired) |

## δ — veto relax (has own leaf signal; unlocks ε)

Signal (boundary tests D-1, D-2): a verified-green blocked task whose only open
escalation is merge-remediable auto-submits (`source='stranded-reaper'`); a
human-concern escalation still vetoes.

| Capability | Evidence | Verdict |
|---|---|---|
| `_maybe_submit_stranded_verified_green` submit path | `grep:orchestrator/src/orchestrator/harness.py:4324` — `async def _maybe_submit_stranded_verified_green` (call-site `:5087`) | PASS (wired) |
| `detect_verified_green` three-part shape check retained (INV-3) | `grep:orchestrator/src/orchestrator/stranded_verified_green.py:115` — `async def detect_verified_green` | PASS (wired) |
| `RE_FILE_ESCALATION` classification + `open_escalations` veto sites | `grep:orchestrator/src/orchestrator/harness.py:4792,4812,5087` — the three veto sites; **relax via ONE shared predicate helper, not three copies (INV-5)** | PASS (wired) |
| escalations carry a classifiable `category` (to define `MERGE_REMEDIABLE_ESC_CATEGORIES`) | escalation records carry category vocabulary (harness classifies at `:4167`+); exact membership is Open-Q #2 (impl-time) | PASS (wired) |
| existing `stranded_merge_request` marker suppresses re-submit storm (INV-4) | `grep:orchestrator/src/orchestrator/harness.py:4431` — marker stamp | PASS (wired) |

## ε — SOP memory update (leaf; operational, no-code)

Signal: `get_memory_by_id` shows the updated canonical text referencing the
shipped exits with no "in flight" pointer.

| Capability | Evidence | Verdict |
|---|---|---|
| the SOP canonicals exist to rewrite (topic consolidated in esc-3011-1) | task 3011 (`done`) consolidated the 12→3 subcase canonicals; Mem0 reachable | PASS (wired) |
| the exits ε references are shipped (`mark_step_committed`, `report_ready_to_merge`, relaxed veto) | `producer:β,γ,δ upstream` (ε depends on all three) | PASS (DAG-direction) |
| memory read/write path (`get_memory_by_id`, `add_memory`/update) | fused-memory MCP live (`get_status` connected) | PASS (wired) |

## ζ — architect prompt/skill exit-enumeration update (leaf)

Signal: the architect prompt/skill source enumerates both new surfaces.

| Capability | Evidence | Verdict |
|---|---|---|
| the architect exit enumeration lives in an editable source file | `grep:orchestrator/src/orchestrator/agents/roles.py:205-210,334-339,1446-1450` — `report_task_already_done`/`report_unactionable_task` enumerated; `agents/briefing.py:273-274,425` | PASS (wired) |
| the two enumerated surfaces exist to reference | `producer:α,β upstream` (ζ depends on both) | PASS (DAG-direction) |
