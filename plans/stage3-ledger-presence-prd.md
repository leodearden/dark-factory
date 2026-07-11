# Stage-3 Ledger-Presence Check — PRD

Close the reconciliation Stage-3 gap that task 2229 (W5-λ) deliberately deferred:
its per-cycle-summary presence check reads only the best-effort **Mem0 mirror**,
never the **authoritative `ReconLedgerStore` row**, because no MCP tool exposes a
ledger read to the read-only Stage-3 agent. Add a small read-only MCP tool and
wire Stage 3 to trust it.

*Substrate confirmed on `main` = `6525ad7152` (brief baseline `4c40b7fa23` is an
ancestor; deltas re-checked). This is a **LOW-risk, fail-safe** follow-on —
scoped deliberately small; see §2 proportionality.*

## 1. Goal (user-observable)

In a cycle where the authoritative `cycle_summary` ledger row is **present** but
the Mem0 mirror is absent/inconclusive, Stage 3 does **not** report the summary
missing; and when the ledger row is **genuinely absent** (e.g. the ledger upsert
failed while the mirror landed), Stage 3 **does** report it missing — driven by a
real ledger read, not by the non-authoritative mirror. The only thing this buys
is a **more reliable Stage-3 audit** of summary presence; correctness of the
authoritative record itself is unaffected (it already lives safely in the
ledger regardless of what Stage 3 reads).

## 2. Background — why the gap exists, and why it's small

- Task **2229** (merged `b0dab33eca`) made `summary_pool.write_cycle_summary()`
  write the authoritative `cycle_summary` as a `ReconLedgerStore.upsert(...)` row
  keyed by `(project_id, 'cycle_summary', task_id='', flag_type=<stage>, run_id)`
  with `ON CONFLICT` idempotency, **plus** a best-effort, dedup-exempt Mem0
  mirror via `add_system_record`.
- 2229's reviewer flagged (architecture-coherence) that Stage 3 still checks
  presence only via the mirror and **deferred** the fix — closing it needs a new
  server-side MCP tool (`fused_memory/server`), outside 2229's module locks. The
  deferral is documented, not silently assumed, in
  `reconciliation/prompts/stage3.py:9-22` ("Known gap … Tracked as a follow-up").
- As of 2026-07-11 **no filed task tracks it** (only related-but-distinct ledger
  tasks 2219/2229/2227). Task 2421 was CANCELLED as a *different, obsolete*
  concern; this ledger-presence gap is the genuinely-live residual it surfaced.

**Proportionality (read before decomposing / at META).** This already **fails
safe**: the mirror write is dedup-exempt, and a full Mem0 outage makes
`count_memories_by_metadata` *inconclusive*, so Stage 3 does **not** report the
summary missing (it never false-positives into wasteful reconstruction). So the
delta is purely a *better audit*, not a correctness fix. The intended scope is a
**single additive read-only tool + one prompt edit + one boundary test** — two
leaf tasks. Do **not** grow this into a general ledger-browse API, a new finding
taxonomy, or a B+H fleet. If decompose surfaces disproportionate cost, concluding
the minimal version — or recommending the gap stay a documented Known Gap — is a
legitimate outcome.

### 2.1 Substrate corrections vs the design brief (read before decomposing)

The design brief (distilled from the task-2421 `/unblock` session) was accurate
on the core substrate but three points were re-verified and corrected here:

1. **The named seam-test file does not exist.** The brief's
   `fused-memory/tests/test_recon_reliability_integration.py` (attributed to task
   2232 / W5-ο) was **never** in git history. The real, verified homes are:
   - write→read seam test → **`fused-memory/tests/test_summary_pool.py`** (already
     imports `write_cycle_summary` **and** `ReconLedgerStore` — the natural home),
     or a new `tests/test_stage3_ledger_presence.py`.
   - tool-level unit test → **`fused-memory/tests/server/test_count_by_metadata_tool.py`**
     is the copy-precedent → new `tests/server/test_get_cycle_summary_presence_tool.py`.
2. **Stage-3 tool access is allow-by-default via a DISALLOW list, not an
   allowlist.** `cli_stage_runner.py:65` sets
   `STAGE3_DISALLOWED = DISALLOW_TASK_WRITES + DISALLOW_MEMORY_WRITES + DISALLOW_BUILTIN`.
   A new **read-only** tool is auto-allowed **as long as it is kept OUT of every
   `DISALLOW_*` list** — there is **no allowlist edit**. The only prompt edit is
   the human-facing "## Available Tools" list in `stage3.py` (documentation for
   the agent). Mirror `count_memories_by_metadata`'s docstring note ("intentionally
   read-only … NOT in any DISALLOW_* list").
3. **Optional consistency site:** `reconciliation/recon_self_model.py:208`
   (`MCP_CALL_SIGNATURES`) carries a one-line signature per Stage tool. Adding the
   new tool's signature there is a small consistency touch (not load-bearing for
   Stage 3, which reads the `stage3.py` Available-Tools list) — fold it into τ1.

## 3. Sketch of approach

A new read-only fused-memory MCP tool `get_cycle_summary_presence(project_id,
run_id, stage)` delegates to a thin `MemoryService` method that calls the
**existing** `ReconLedgerStore.get_by_identity(project_id, 'cycle_summary',
task_id='', flag_type=stage, run_id=run_id)` and reports presence. Stage 3's
"Cycle-Summary Verification" section consults it as the **authoritative** path,
falling back to the existing two Mem0 paths **only when the ledger read is
inconclusive** (ledger not wired/disabled, or a backend error). The change can
only *add* a true-positive (catch a genuinely-absent ledger row); it never removes
the existing inconclusive→don't-report fail-safe.

## 4. G3 — substrate verification (each confirmed on `main` 6525ad7152)

| Capability | Evidence | Status |
|---|---|---|
| `ReconLedgerStore.get_by_identity(project_id, record_kind, task_id='', flag_type='', run_id='')` | `reconciliation/recon_ledger.py:203` — five-part identity, default-`''` args fit a cycle_summary lookup exactly | **PASS** |
| `memory_service.recon_ledger: ReconLedgerStore \| None` populated + None-guard precedent | decl `services/memory_service.py:516`, set `:535`; `getattr(...,'recon_ledger',None)` precedent in `write_cycle_summary` | **PASS** |
| Cycle_summary identity == `(project_id, 'cycle_summary', '', <stage>, <run_id>)` | `write_cycle_summary` record construction, `summary_pool.py` (`record_kind='cycle_summary'`, `task_id=''`, `flag_type=stage`, `run_id`) | **PASS** (verified live) |
| Read-only MCP tool registration shape | `server/tools.py:1128` `count_memories_by_metadata` (`@mcp.tool()` + `@mcp_tool_errors()`, `_canonicalize_project_id_arg`, delegates to `services/memory_service.py:2684`) — exact copy-precedent | **PASS** |
| Adding a new read-only tool to a stage's toolset is a blessed pattern | task 78 (`get_edges_by_episode` for Stage 1) — done | **PASS** |
| Stage 3 auto-allows a read tool | `cli_stage_runner.py:65` DISALLOW-model — keep the new tool out of every `DISALLOW_*` list (§2.1 pt 2) | **PASS** |

## 5. Resolved design decisions (do not relitigate)

- **D1 — Narrow, purpose-built tool, not a general ledger read.**
  `get_cycle_summary_presence(project_id, run_id, stage)`, not
  `get_ledger_record(record_kind, …)`. Single named consumer (Stage 3); a general
  browse API with only one consumer would trip G1 in reverse (see Out of scope).
- **D2 — `stage` is an explicit, required arg mapped to `flag_type`.** The ledger
  keys the summary by `flag_type=stage`; Stage 1 and Stage 2 both write a
  `cycle_summary` under the **same** run_id (`memory_consolidator` vs
  `task_knowledge_sync`). Passing `stage` prevents the exact Stage-1/Stage-2
  collision the mirror Path-2 filter already guards against (task 1653). Stage 3
  hardcodes `stage='task_knowledge_sync'` (the Stage-2 summary it verifies).
- **D3 — Ledger is authoritative for presence; Mem0 is fallback-on-inconclusive
  only** (full decision rule in §8.2). The Mem0 two-path logic is **retained**,
  not deleted, so today's fail-safe is preserved verbatim whenever the ledger read
  can't answer.
- **D4 — A genuinely-absent ledger row is reported as `missing_knowledge`,
  actionable, suggested_action = reconstruct** — same category/remediation Stage 3
  uses today (re-running `write_cycle_summary` fixes both ledger and mirror). No
  new finding category. (Richer `cross_store_inconsistency` labeling for
  ledger-absent-but-mirror-present is deferred — see Open questions.)
- **D5 — Presence is a boolean, not a payload dump.** The tool returns
  `{present, ledger_available, project_id, run_id, stage}` — no record body — to
  stay a pure presence check and not become a browse API (D1).

## 6. Pre-conditions

- 2229 (`b0dab33eca`, ledger-backed `write_cycle_summary`) — **done, on main.**
- 2219 / W5-α (`ReconLedgerStore`) — **done, on main.**
- No migration, no schema change, no new config. Fully additive.

## 7. Cross-PRD relationship (G4)

This PRD **solely owns** the new MCP tool. It **reuses** `ReconLedgerStore`
(task 2219, done) and `write_cycle_summary` (task 2229, done) read-only. Lineage
parent is the recon-reliability program (`plans/recon-reliability-prd.md`,
stream W5) which explicitly deferred this — treat this as a small W5 follow-on,
**not** a new cross-PRD seam. No reciprocal "the other owns it" pattern.

## 8. Contract section (B + H)

### 8.1 New tool: `get_cycle_summary_presence`

```
get_cycle_summary_presence(project_id: str, run_id: str, stage: str)
  -> { present: bool,
       ledger_available: bool,
       project_id: str, run_id: str, stage: str }
```

- **Read-only.** Delegates to `MemoryService.get_cycle_summary_presence(...)`,
  which calls `recon_ledger.get_by_identity(project_id, 'cycle_summary',
  task_id='', flag_type=stage, run_id=run_id)` and returns
  `present = (record is not None)`.
- **Ledger-unavailable guard:** if `getattr(memory_service, 'recon_ledger', None)
  is None` (unwired, or `recon_ledger_enabled=False`) → return
  `{present: False, ledger_available: False, …}`. This is **inconclusive**, not a
  definitive "absent" (mirrors `write_cycle_summary` returning `False` when
  unwired).
- **Backend error:** a raised exception flows through `@mcp_tool_errors()` to an
  error dict, which Stage 3 treats as inconclusive (same as the count path).
- **Registration:** `@mcp.tool()` + `@mcp_tool_errors()` in `server/tools.py`,
  `_canonicalize_project_id_arg` + `validate_project_id` at the top, copying the
  `count_memories_by_metadata` shape. Carry the "intentionally read-only, NOT in
  any DISALLOW_* list → auto-allowed in Stage 3" docstring note. **Do not** add it
  to any `DISALLOW_*` list.

### 8.2 Stage 3 consult policy (the decision rule the prompt encodes)

Rewrite "## Cycle-Summary Verification" so that, before reporting a Stage-2
per-cycle summary missing for `run_id`:

1. **Authoritative path (new, primary):** call
   `get_cycle_summary_presence(project_id, run_id, stage='task_knowledge_sync')`.
   - `ledger_available=true, present=true` → **present** → do not report missing. Done.
   - `ledger_available=true, present=false` → the authoritative row is **genuinely
     absent** → **report missing** (`missing_knowledge`, actionable,
     suggested_action = reconstruct). *This is the new value.*
   - `ledger_available=false` **or** tool error → **inconclusive** → fall to (2).
2. **Fallback (existing two Mem0 paths), used ONLY on inconclusive:** the current
   Path 1 (semantic) + Path 2 (`count_memories_by_metadata`) rule, **unchanged** —
   declare missing only if BOTH return nothing; treat a count error as
   inconclusive → do **not** report. Preserves today's fail-safe exactly.

Then **delete** the "Known gap" comment (`stage3.py:9-22`), and add the tool to
"## Available Tools".

### 8.3 Invariants

- **Fail-safe monotonicity:** the change can only add a true-positive (a real
  ledger absence now reported); it never removes the inconclusive→don't-report
  protection, because the Mem0 fallback is retained for every non-definitive read.
- **Stage disambiguation:** presence is always queried with an explicit `stage`;
  the Stage-2 verification hardcodes `'task_knowledge_sync'`.
- **No write path touched:** `write_cycle_summary` is unchanged (read/visibility
  fix only).

## 9. Boundary-test sketch (B + H) — the integration-gate signal

One focused two-way boundary test over the write→read seam (proportionate to the
low risk; not a full fleet):

- **Present direction:** `write_cycle_summary(memory_service, report, run_id=R,
  stage='task_knowledge_sync', …)` upserts the ledger row → the new service
  method / tool returns `present=true, ledger_available=true` for identity
  `(project_id, 'cycle_summary', '', 'task_knowledge_sync', R)`.
- **Absent direction:** for an un-written `run_id`, the tool returns
  `present=false, ledger_available=true` (definitive absent, not inconclusive).
- **Inconclusive direction:** with `recon_ledger=None`, the tool returns
  `ledger_available=false` (drives Stage 3 to the Mem0 fallback).
- **Anti-inversion:** the test asserts **both** present→true and absent→false (the
  seam is not wired backwards).

Home: `fused-memory/tests/test_summary_pool.py` (§2.1 pt 1). The tool's own
present/absent/inconclusive unit test lives in
`tests/server/test_get_cycle_summary_presence_tool.py`.

**Honesty note (G2):** the Stage-3 agent's *in-loop reasoning* is not
unit-testable. The seam test proves the mechanical path the prompt tells the agent
to trust; a cheap prompt-content assertion (tool named + rule present +
"Known gap" text gone) proves the wiring landed. Together these are the
non-fake-done signal — not a synthetic-input pass.

## 10. Decomposition plan (task DAG)

Two leaf tasks. Disjoint file-lock sets; τ2 depends on τ1 (needs the tool to
exist). Both take the full architect path (omit `complexity='simple'` — τ1
introduces a new tool surface; τ2 carries the integration boundary test, a
hard-blocker token). `task_kind='normal'`.

### τ1 — `get_cycle_summary_presence` read-only MCP tool *(producer; tool-level signal)*
- **Files:** `server/tools.py` (register, copy `count_memories_by_metadata`
  shape), `services/memory_service.py` (new method delegating to
  `recon_ledger.get_by_identity`, None-guard per §8.1),
  `reconciliation/recon_self_model.py` (optional `MCP_CALL_SIGNATURES` entry),
  `tests/server/test_get_cycle_summary_presence_tool.py` (new; copy
  `test_count_by_metadata_tool.py`).
- **Keep the tool OUT of every `DISALLOW_*` list** (§2.1 pt 2).
- **Signal (G2-a, tool-level observable):** the tool returns a correct
  present / absent / inconclusive answer for a known ledger identity, proven by a
  unit test driving a **real** `ReconLedgerStore.upsert` (or `write_cycle_summary`)
  → read — not a mocked store.
- **Deps:** none.

### τ2 — Wire Stage 3 to consult the ledger authoritatively *(consumer; integration signal)*
- **Files:** `reconciliation/prompts/stage3.py` (add to "## Available Tools";
  rewrite "## Cycle-Summary Verification" to §8.2's rule; **delete** the
  `:9-22` "Known gap" comment), `tests/test_summary_pool.py` (add the §9
  write→read boundary test + inconclusive case).
- **Signal (G2-b, integration/seam observable):** the two-way boundary test proves
  `write_cycle_summary → ledger upsert → get_cycle_summary_presence` returns the
  correct present/absent for the same identity (both directions); **plus** a
  prompt-content assertion that `stage3.py` now names the tool and the
  ledger-authoritative rule and no longer contains the "Known gap" text.
- **Deps:** τ1.

### DAG
```
τ1  (tool)
 └─► τ2  (Stage-3 wiring + boundary test)
```

## 11. Out of scope

- Any change to the write path (`write_cycle_summary`) — it is correct; this is a
  read/visibility fix only.
- A general ledger-browse / `get_ledger_record` API — broadening beyond what
  Stage 3 needs, with no second consumer named, trips G1 in reverse (D1).
- A new finding taxonomy for ledger/mirror divergence (see Open questions).
- Back-filling ledger rows for legacy (pre-2229) summaries — Stage 3 only verifies
  the current cycle's run, which is always ledger-era.

## 12. Open questions (tactical — decide at impl time, non-blocking)

- **Richer divergence signal:** when the ledger is absent but a Mem0 mirror is
  present, τ2 could emit `cross_store_inconsistency` instead of plain
  `missing_knowledge`. Default: reuse `missing_knowledge` (reconstruct fixes both).
  Upgrade only if a run surfaces a case where the distinction changes remediation.
- **Return payload width:** whether to include the row's `state` / `created_at`
  alongside `present` for richer Stage-3 reporting. Default: boolean-only (D5);
  add fields only if a reporting need is named (guards against browse-API creep).
- **Test home:** `test_summary_pool.py` (co-located with the writer) vs a new
  `test_stage3_ledger_presence.py`. Either satisfies §9; implementer's call.

## 13. Capability manifest (draft — committed beside the PRD at decompose)

See `plans/stage3-ledger-presence.capability-manifest.md` (written at decompose,
task IDs filled in after filing). All bindings resolve **PASS**: the change is a
read of existing substrate (2219/2229) plus one new read-only tool produced
upstream of its single consumer.

## 14. Meta

*Would decompose-and-queue without further oversight produce a complete, coherent,
cohesive, good design?* **Yes.** G1 consumer named (Stage 3, same PRD — no orphan
producer). G2 both leaves carry honest user-observable signals (tool-level +
write→read seam + prompt-content), not synthetic-input passes. G3 all substrate
verified on current main; the one brief inaccuracy (a nonexistent test file) is
corrected to real homes (§2.1). G4 single owner, no reciprocal seam. G5 one
proportionate boundary test (B+H right-sized for low risk). G6 premise verified
live (Stage 3 has no ledger read path today; `stage3.py:9-22` + the Available-Tools
list confirm it) and the presence claim is boolean — achievable, backed by a real
DB read, disambiguated by `stage`. Scope is held deliberately minimal per §2.
