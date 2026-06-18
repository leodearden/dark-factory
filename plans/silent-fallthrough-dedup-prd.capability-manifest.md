# Capability manifest — silent-fallthrough-dedup-prd

Mechanizes G3 (assumed-substrate) + G6 (premise validity) per leaf. Each binding cites evidence on `main`.
**Result: all bindings PASS — no FAIL binding blocks the batch.** Evidence verified during the sweep + the
substrate checks in this session (workspace deps, `(value,error)` and `{'offline'}` contracts, pytest gate).

Substrate facts (shared across leaves):
- `dark-factory-shared` is a workspace dep of orchestrator/fused-memory/dashboard/sampler — `grep` of each
  `pyproject.toml` (`dark-factory-shared = { workspace = true }`). Import style `from shared.<mod> import …`.
- `escalation` does NOT depend on `shared` → its fix (ξ) is in-package only.
- pytest gate exists (`conftest.py` at repo root + per-package `tests/`); `ast` stdlib available → σ host.
- All Python modules have/can-add a `logging` logger; `b3_gate` currently has **none** → ε adds one (self-delivered).

## Foundation (intermediate — unlock migrations; their signals are behavioral unit tests, not synthetic-pass)

- **α** `shared.mcp_envelope` → [ new module: producer:task-α (self) · `logging`: exists · unlocks δ,ζ,θ,ν ]
- **β** `shared.safe_io` → [ new module: producer:task-β (self) · unlocks ε,κ,ν ]
- **γ** `shared.agent_result` → [ new module: producer:task-γ (self) · mirror-pattern `workflow._run_reviewer`
  ERROR-verdict-on-unparseable: grep workflow.py:4470-4486 wired on main · unlocks ζ,ι ]
- **φ** `shared.timestamps.parse_timestamp_or_warn` → [ new module: producer:task-φ (self) · mirror-pattern
  `datetime.min` fallback: grep escalation/queue.py:636-644, watcher.py:97-101 wired on main · unlocks ξ,ν ]

## Migrations (each carries a behavioral RED signal; all required capabilities are UPSTREAM)

- **δ** [orchestrator] →
  - `parse_tool_result` → producer:task-α **upstream** (dep wired) ✓
  - scheduler `(value, error)` resolver contract → grep scheduler.py:1502/1508/1534/1542 (`return {}, None`,
    `-> tuple[dict, Exception|None]`) wired on main ✓
  - `_external_resolver_failed` flag + grace counter pattern to mirror → grep scheduler.py:2593-2609 +
    `_apply_external_dep_policy` wired on main ✓
  - `get_external_statuses` fold-in → producer:task-**1799 upstream** (dep wired; G3 resolution-b prerequisite) ✓
- **ε** [orchestrator] → `load_json_or_warn` → producer:task-β upstream ✓ · b3_gate logger → self-delivered ✓ ·
  corrupt-handling twins `merge_queue_store._save_raw/load` WARNING → grep merge_queue_store.py:138/163/316 ✓
- **ζ** [orchestrator] → `parse_tool_result`/resolver-guard → α upstream ✓ · `extract_agent_verdict` → γ upstream ✓ ·
  loud twin `harness._reap_orphan_worktrees` → grep harness.py:1569-1576 ✓
- **θ** [fused-memory] → `parse_tool_result` → α upstream ✓ · recon exception-branch WARNING twins →
  grep targeted.py:557-562, reconciliation/harness.py:715 ✓
- **ι** [fused-memory] → `extract_agent_verdict` → γ upstream ✓ · agent `{'warning':...}` dict shape →
  grep agent_loop.py:119/159 wired on main ✓
- **κ** [fused-memory] → `load_json_or_warn` → β upstream ✓ · loud twin recon harness journal-read WARNING →
  grep reconciliation/harness.py:1677-1686 ✓ · `memory_service.search` degraded-channel consumer
  `server/tools.py:780-799` same-package ✓
- **λ** [fused-memory] (indep) → `_row_to_task` deduped-WARNING handler `_warned_malformed_task_ids` →
  grep sqlite_task_backend.py:252-268 wired on main ✓ (extract within same package)
- **ν** [dashboard] → `parse_tool_result` → α upstream ✓ · `load_json_or_warn` → β upstream ✓ ·
  `parse_timestamp_or_warn` (#39) → φ upstream ✓ · `{'offline':True,'error':...}` marker contract →
  grep dashboard/data/tasks.py fetch_tasks + active_tasks.py:175 `_shape_one_project` wired on main ✓ ·
  uvicorn default INFO (DEBUG invisible) confirmed ✓
- **ξ** [escalation] → `parse_timestamp_or_warn` → φ upstream ✓ · escalation→shared dep: escalation is a uv
  workspace member (root pyproject `tool.uv.workspace.members`) + `shared` has no first-party app deps →
  `{ workspace = true }` resolves, no cycle ✓ · datetime.min fallback twins to unify →
  grep dedupe.py:291, queue.py:636-644, watcher.py:97-101 ✓
- **ο** [shared] (indep) → timeout-branch WARNING twin → grep pytest_jobserver.py:87-92 ✓
- **π** [sampler] (indep) → `__main__` outer visible-degrade handler ('writing PSI metrics only') →
  grep sampler `__main__.py` wired on main ✓
- **ρ** [scripts] (indep) → caller skip-on-None + file-exists check → grep reviewer_redundancy_diagnostic.py:95/98 ✓

## Enforcement leaf

- **σ** [shared/tests] (dep all migrations upstream) →
  - pytest + `ast` host → exists ✓
  - **Rejection-mechanism (G6 branch 4):** σ BUILDS the checker; the RED test authors a known-bad sample
    (`x, _ = await resolver()` and `except: return {}`) and **observes the scan FAIL** (diagnostic fires), then
    passes once removed → `rejection-check` binding satisfied by σ's own deliverable + observed RED ✓
