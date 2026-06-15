# Capability manifest — recon-watchdog-kill-get-statuses-prd

Mechanizes G3 (substrate exists + wired) and G6 (premise validity) per leaf.
All bindings PASS → batch clears. Numeric premises: α asserts no test-time number
(the "~6/day" / "drains below 500" are motivating operational outcomes, not the
pass/fail oracle — the oracle is contract-equality + zero-decode); β's threshold is
a configurable default it *defines*, not an external number it must match, so G6
branches 1–2 reduce to the dependency-direction check (branch-3), checked below.

## α — `get_statuses` O(K) status-only read (no metadata decode)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `tasks.status` is a top-level column (selectable without decode) | `sqlite_task_backend.py:62` (`status TEXT NOT NULL`) | PASS |
| index supports status/id lookup | `:69` `CREATE INDEX ix_tasks_status ON tasks(tag, status)`; PK `(tag, id)` `:66` | PASS |
| targeted-query pattern to mirror exists | `get_task` `:522-547` (`SELECT * FROM tasks WHERE tag=? AND id=?`, off-thread via `_get_connection`) | PASS |
| current full-tree path to replace | `get_statuses` `task_interceptor.py:3194-3230` → `tm.get_tasks` → `_get_tasks_internal` `:497-510` → `_row_to_task`/`json.loads` `:251` | PASS |
| contract to preserve is observable & total | `get_statuses` docstring `:3197-3213`: `{id_str:status_str}`, missing→`'unknown'`, `ids=[]`→`{}`, `ids=None`→all, unknown omitted | PASS |
| caller passes `ids` as str; column `id` is INTEGER (str/int boundary handled) | `server/tools.py:2199` `task_ids=[tid…]` (str from `partition(':')`); `_parse_task_id` int-cast pattern in `get_task` `:527` | PASS |
| DAG-direction (G6 branch-3): every capability α needs is on main today | all primitives above pre-exist; no downstream dep | PASS |

## β — `dead_owner_shielded` suppression-storm aggregate alarm

| Capability asserted | Evidence | Verdict |
|---|---|---|
| suppression site to instrument | `harness.py:708-725` (`disposition == 'dead_owner_shielded'` → INFO log + suppress) | PASS |
| `_escalate` exists, routes to 8103 recon queue, dedup-folds on ingest | `harness.py:838` (`def _escalate(category, run_id, summary, detail, *, finding=…)`; A7b routing contract in docstring) | PASS |
| stable-identity fold available (one pending item, not per-event) | `_escalate` `finding=` path computes `compute_content_fingerprint` keyed on finding identity | PASS |
| config home for threshold/window knobs | `self.config = config.reconciliation` `harness.py:272`; recon knobs under `reconciliation:` `config/config.yaml:87` (`stale_lock_seconds`, `max_staleness_seconds`, …) | PASS |
| works without config (in-code defaults) | `cutoff = self.config.stale_run_recovery_seconds` shows direct attr access; β defaults `*_storm_threshold=6`, `*_storm_window_seconds=3600` in code | PASS |
| recon-watcher consumes the 8103 queue (consumer exists) | `recon-watch/mcp.json` escalation→`127.0.0.1:8103`; journal `Reconciliation escalation server starting on 127.0.0.1:8103` | PASS |
| threshold is a defined default, not a matched external number (G6 branch 1–2 N/A) | β owns the knob; test feeds N events and asserts the boundary it defines | PASS |
| DAG-direction (G6 branch-3) | independent of α; all substrate on main | PASS |

## γ1 — fused-memory opt-in status filter on get_tasks (follow-up 2026-06-15)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| get_tasks layers to thread a filter through | server tool `server/tools.py:2020`; interceptor `task_interceptor.py:3187`; backend `sqlite_task_backend.py:514` → `_get_tasks_internal:497` (`SELECT * … WHERE tag=?`) | PASS |
| `status` column + index for `status IN` predicate | `sqlite_task_backend.py:62` (`status TEXT NOT NULL`), `:69` `ix_tasks_status(tag,status)` | PASS |
| default-unchanged contract is observable | `get_tasks` returns `{tasks:[…]}` full tree today; param-omitted path must match byte-for-byte | PASS |
| DAG-direction | no deps; all substrate on main | PASS |

## γ2 — scheduler adopts active-only on hot paths (depends γ1)

| Capability asserted | Evidence | Verdict |
|---|---|---|
| scheduler get_tasks consumers to audit | `scheduler.py:1129` (_get_train_members), `:2392` (dispatch); `workflow.py:918, 7208` | PASS |
| lean dep-satisfaction path exists (so filtering can't blind done-deps) | `scheduler.get_statuses` `scheduler.py:1429` ("~95% less data"), already used at `workflow.py:795` | PASS |
| producer: the `statuses` param γ2 calls | shipped by γ1 (dep wired γ2→γ1) | PASS (producer:γ1 upstream) |
| DAG-direction (G6 branch-3) | γ2 depends_on γ1; γ1 ships the param γ2 needs | PASS |

**Result:** 0 FAIL bindings — batch clears the manifest gate.
