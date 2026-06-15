# Capability manifest — merge-status-false-unknown-prd

Mechanizes G3 (substrate exists + wired) and G6 (premise validity) per leaf.
All bindings PASS → batch clears. No leaf asserts a numeric/exactness premise
(G6 branches 1–2 N/A; branch-3 dependency-direction checked below).

## α — server self-resolution + skill alignment

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `GitOps.is_ancestor(ref, main)` exists & wired | `git_ops.py:1240` (def); already invoked from the escalation server at `server.py:769` (coalesce-gate scan) | PASS (wired) |
| `GitOps.find_merge_marker(branch)` — deleted-branch landing check | `git_ops.py` `find_merge_marker` (companion to `is_ancestor`; searches main for `Merge {branch} into main`) | PASS |
| `resolve_branch_sha` / `get_main_sha` | `git_ops.py:~1230` / `git_ops.py get_main_sha` | PASS |
| `merge_status` Tier-4 branch (insertion point) | `server.py:1298` (`return {state:'unknown', ...}`) | PASS |
| `git_ops` handle reachable from escalation server | `server.py:769` `git_ops_for_scan.is_ancestor(...)` | PASS |
| `done_provenance` `kind="found_on_main"` semantics to mirror | `set_task_status` schema (`is-ancestor` against main) | PASS |
| task→branch convention `task/<id>` | used throughout merge queue / land.sh | PASS |
| skill files to edit exist | `skills/merge-queue/SKILL.md`, `skills/unblock/SKILL.md` | PASS |
| DAG-direction (G6 branch-3): every capability α's signal needs is in α or upstream | all primitives above pre-exist on main; no downstream dep | PASS |

## β — widen retention-ring keys + alias coalesced ids

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `TerminalOutcomeRetention` ring to re-key | `merge_queue.py:2358` (deque + `dict` index keyed by `request_id`) | PASS |
| `merge_finalized` record carries `branch` + `task_id` + `request_id` (no new data needed at record time) | `merge_queue.py:2851` (`data={request_id, branch, ...}`, `task_id=req.task_id`) | PASS |
| coalesced/superseded `request_id` concept exists | `merge_request` coalesce path; `merge_cancel` docstring (`server.py:1324`) | PASS |
| `merge_status` Tier-2 lookup to widen | `server.py:1186` (`if ring is not None and request_id is not None`) | PASS |
| DAG-direction | depends_on α (server.py `merge_status` region serialization); no downstream dep | PASS |

## γ — run-spanning "did it land" (filed DEFERRED, gated)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `event_store.latest_merge_finalized` run_id scoping to relax for landed-check | `event_store.py:301` (run_id-scoped by design) | PASS |
| `merge_finalized` rows carry `branch` + `merge_sha` | `merge_queue.py:2851` (`data={branch, merge_sha, ...}`) | PASS |
| DAG-direction | depends_on α, β | PASS |

**Result:** 0 FAIL bindings — batch clears the manifest gate.
