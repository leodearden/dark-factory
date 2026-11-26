# Capability manifest — memory-referent-fidelity

PRD: `plans/memory-referent-fidelity-prd.md`
Built at decompose, 2026-08-05, against main tip `57ef5b4ead`.
Machine-readable twin: `plans/memory-referent-fidelity-prd.capability-manifest.yaml`.

Mechanizes G3 + G6 per the PRD's 11-label decomposition. Every binding below was
re-derived at decompose time against main and against the live `dark_factory`
graph via **read-only** probes (`GRAPH.RO_QUERY`); no graph state was mutated and
no graphiti driver was constructed (upstream `FalkorDriver.__init__` fire-and-forgets
`build_indices_and_constraints` at `falkordb_driver.py:161-169`, which would have
destroyed the protected missing-index evidence).

## Substrate re-verification (G3)

Every capability the PRD lists as "verified present on main" was re-checked at the
**exact** line the PRD cites. All 10 hold:

| Capability | Cited | Observed | Verdict |
|---|---|---|---|
| `reassign_edge` | `graphiti_client.py:989` | `:989` | PASS |
| `_resolve_or_create_entity` | `:2251` | `:2251` | PASS |
| `get_nodes_by_exact_name` | `:2043` | `:2043` | PASS |
| `find_duplicate_entity_nodes` | `:2087` | `:2087` | PASS |
| `refresh_entity_summary` | `:2439` | `:2439` | PASS |
| `_identity_lock_for` | `:438` | `:438` | PASS |
| `build_indices_and_constraints` (`pass` stub) | `:291` | `:291`, body is `pass` | PASS |
| `_ensure_indices` (calls the stub) | `:474` | `:474` | PASS |
| `canonicalize_task_node_name` | `utils/task_naming.py:26` | `:26` | PASS |
| `_reconcile_episode_identity` | `memory_service.py:2075` | `:2075` | PASS |

Upstream `graphiti_core` refs (`.venv/lib/python3.13/site-packages/graphiti_core`):
`dedup_helpers.py:220` (`len(existing_matches) > 1` abstain) PASS ·
`node_operations.py:424` (module-global `_resolve_with_llm` call) PASS ·
`bulk_utils.py:128` (`add_nodes_and_edges_bulk`) PASS ·
`falkordb_driver.py:161-169` (the index-build HAZARD) PASS.

**α pre-condition re-verified.** The PRD pinned the four `task/3335` commits against
main tip `52e27ff13f`; main has since moved to `57ef5b4ead`. Re-applied in a throwaway
detached worktree **outside** `.worktrees/`: all four cherry-pick with **zero conflicts**,
`829 insertions(+)` across the same 4 files, all four parsing — identical to the PRD's
recorded figure. `task/3335` is 592 behind / 19 ahead (PRD said 580 — drift only), tip
`8c616a0921`, contained in no branch but its own. No dependency on task 3335 is wired.

**Protected evidence re-confirmed, read-only.** `CALL db.indexes()` on `dark_factory`
returns only RANGE indices on `uuid` for `Entity` and `RELATES_TO` — **no fulltext index
exists**, exactly as the PRD's root cause states. 801 canonical `Task N` nodes (PRD: 800;
+1 drift on a live graph). **38** duplicate normalized name keys over **78** nodes —
matches the PRD exactly.

## G6 premise re-validation

Two findings, both resolved at filing; everything else passes.

### Finding 1 — ι asserts a capability outside its dependency closure (branch 3)

ι's signal is "an operator can read **declared** / metadata / derived / none counts".
The `declared` bucket is produced by the `entities` parameter, which **δ** owns. The
PRD gives ι the prereq set `{ζ}`, and ζ←ε←γ — so δ is **not** in ι's transitive closure.
Filed as-is, ι's `declared` counter would be structurally pinned at zero and the leaf
could close green while its headline signal was vacuous.

**Resolution (G6 (a), dependency-correctness):** wire `ι depends_on δ` **in addition to**
ζ. Recorded here rather than as a PRD edit — it tightens the DAG without changing the
architecture.

### Finding 2 — θ's signal covers 4 of the 5 live conflations (branch 3, extent)

θ is titled "remediate the **5** live conflations" but its signal names four task
numbers. Read-only probes confirm all four currently carry foreign facts:

| Node | Live foreign fact | Mode |
|---|---|---|
| `Task 2434` | "Task **2435** depends on Task 2169 for its original user-observable signal." | (i) |
| `Task 3220` | "The stale note about .gitignore was carried forward into task **3222** from task 3072." | (i) |
| `Task 1030` | "Task **1031** transitioned to done in the remediation cycle on 2026-04-29." | (ii) |
| `Task 3074` | "Task **3075**'s design implication is that any subprocess timeout on lane_protect_glob must be sized against the loaded figure…" | (iii) |

The **fifth** is on `Task 2520`: *"Umbrella task **2519** was filed and then cancelled to
avoid orphaning its vector."* — a unary fact about 2519 attached to the 2520 node. This is
precisely the boundary sketch's *"Unary fact, no correct target"* row, which ζ's
postconditions require be **recorded and left alone, never guessed at**. So the count is
correct and the signal is correct; they simply describe different halves of a 4-repair +
1-record split that θ's text never states.

**Resolution (G6 (b)):** θ's filed description states the 4+1 split explicitly and its
signal is extended to require the 5th be **recorded as flagged-unrepairable**, so
"5 conflations" reconciles with a 4-node observable and no verifier reads the gap as
incomplete work.

### Premises checked and upheld

- **η's summary leg — the highest-risk premise in the batch.** η asserts "the `Task N±1`
  summary no longer contains it". Task 2057 ("`refresh_entity_summary` can't clear
  baked-in stale entity summary text") makes that non-obvious. Read at
  `graphiti_client.py:2439`: `refresh_entity_summary` regenerates the summary **from
  currently-valid edges** (dedupe + join). Once η repoints the edge away, the fact leaves
  the source node's valid-edge set and drops out by construction. 2057's residual case is
  explicitly the *other* one — a stale sentence still carried by a valid edge — for which
  `set_entity_summary` (`:2500`, verified on main) is the documented escape hatch.
  **Premise holds.**
- **δ's rejection mechanism (branch 4).** `add_memory`'s gate stack already returns
  structured `{'error': …, 'error_type': 'ValidationError'}` payloads and demonstrably
  fires today (`validate_project_id`, `_known_project_gate`, `_backlog_gate`,
  `_markup_gate` at `tools.py:2001-2020`). δ's `_entities_gate` joins an existing,
  observed-firing chain rather than asserting a rejection capability that does not exist.
- **β's claim that `task #1153` is unreachable today.** `_TASK_NODE_NAME_PATTERN` is
  `^\s*tasks?\s+(\d+)\s*$` (`task_naming.py:26`) — anchored, no `#` alternation. It is
  structurally unable to match. **Premise holds; β is a fix, not a refactor.**
- **ι's storage claim.** `write_ops` carries free-text `operation` + JSON `params`, with
  `idx_wo_project_time(project_id, created_at)` (`write_journal.py:18-36`). Per-project
  counts over a window need **no migration**, as Open Question 2 assumed.
- **κ's five doc targets are all tracked in-repo.** `CLAUDE.md` ·
  `.claude/commands/memory.md` (this is where `/memory` actually resolves from — *not*
  `~/.claude`, and not `skills/memory/`, which does not exist) · `skills/reflect/SKILL.md` ·
  `reconciliation/prompts/{stage1,stage2,stage3,judge}.py` ·
  `orchestrator/src/orchestrator/review_checkpoint.py`. Pinned into κ's filed description
  so no implementer edits an out-of-worktree path.
- **λ's milestone shape.** `metadata.milestone` `mode='delayed'` + `after_secs` is
  validated by the shared `Milestone` model, with `at` **forbidden** in delayed mode
  (`docs/task-authoring.md:509-529`). λ is filed to that shape.

## G7 walk — `docs/legibility/design-invariants.md`, all 11 tasks

Walked every task (not only leaves) against INV-1..INV-7. The PRD's own walk covered five
slugs; three further hits surfaced, all resolved by tightening the filed task. **No waivers.**

| Task | Hit | Resolution |
|---|---|---|
| **ε** | **INV-4 `storm-escape-required`** — the "absent key ⇒ no referents" back-compat fallback is a silent degradation path. If ε's plumbing regresses, every row arrives referent-less and the entire feature no-ops in silence. The PRD's G7 walk counted η's repair streak and δ's gate rejections but not this. | ε must emit the `source='none'` counter on the absent path, feeding ι. A sustained 100% `none` rate is then observable rather than silent. Resolved by design, not waived. |
| **θ** | **INV-3 `corroborate-before-acting`** — θ would act on a snapshot measured 2026-08-05, but dispatches only after η lands; the graph will have moved. | θ must re-run the read-only census immediately before remediating, and the before/after in the `esc-3375-1` resolution must be θ's **own** measurement, not the PRD's. |
| **κ** | **INV-5 `no-lockstep-duplication`** — κ writes `entities=` examples into five sites that must agree byte-for-byte with the live `add_memory` signature. That is the exact shape INV-5 names. | κ ships a drift/pinning test that fails when a documented example diverges from the live signature — the INV-5 house pattern (render-from-schema, task 2559 precedent) — rather than five hand-transcribed copies. |

Upheld without change: **β** is itself the INV-5 remedy for the four-site regex · **ζ**
satisfies INV-2 by replacing the `logger.debug`-only `ReconcileStats` shape with structured
repair records · **η** satisfies INV-3 (`reassign_edge` re-reads endpoints from topology and
no-ops if already moved) and INV-4 (repair-streak escalation) · **δ** is the INV-1 house
pattern (ValidationError+hint at the submit boundary) · **λ** is the INV-7 bound on decision
4's transitional hold — named owner, dated deadline · **ι** satisfies INV-2 (structured
counters, not log-scraped).

**α, INV-1, satisfied by construction.** `ensure_entity_node`'s lock precondition ("callers
MUST hold `_identity_lock_for(group_id)`") is stated in a docstring — prose. It is not
waived because α is a verbatim cherry-pick and its sole caller, η, executes inside
`_reconcile_episode_identity`'s existing identity-lock critical section, which the PRD makes
load-bearing and ζ/η pin with tests. The contract is enforced at the single call site.

## Per-leaf capability bindings

Structural G2 leaves (nothing in-batch depends on them) are **θ** and **λ**; both carry a
user-observable signal. The PRD additionally gives δ, η, ι, κ genuine user-observable
signals though they are structurally intermediate — stronger than G2 requires. Every
intermediate names its downstream consumer. Bindings below cover all 11 labels.

### α — cherry-pick `cross_project_refs` + `ensure_entity_node`
- `cross-project-refs-module-on-main` → **producer:α**, zero-conflict cherry-pick re-verified against `57ef5b4ead`; exports `find_cross_project_task_refs`, `CrossProjectRef`, `CrossProjectRefScan` — **PASS**
- `ensure-entity-node-mint-primitive` → **producer:α**, `graphiti_client.py:2298` post-pick; delegates to `_resolve_or_create_entity` (`:2251`, on main) and mints only on its `None` branch — **PASS**
- `no-dependency-on-unlanded-3335` → DAG-direction: α takes the two dependency-free primitives only; `_split_cross_project_task_nodes` stays with 3335; no dep edge filed — **PASS**

### β — one canonical-label vocabulary
- `canonical-labels-single-site` → **producer:β** — **PASS**
- `task-hash-1153-parses` → G6 branch 3; today's anchored `^\s*tasks?\s+(\d+)\s*$` structurally cannot match; β delivers it — **PASS**
- `no-second-copy-of-the-pattern` → INV-5, `expect: absent` on the old private pattern surviving in `task_naming.py` — **PASS**

### γ — scanner + `resolve_referents`
- `resolve-referents-precedence` → **producer:γ**, declared > metadata > derived — **PASS**
- `scan-content-precision` → **producer:γ**; source locations / URL authorities / clock times must not match — **PASS**
- `referent-digits-verbatim` → **producer:β** upstream (`Referent.number` is `str`) — **PASS**

### δ — `entities` param + `_entities_gate` *(user-observable signal)*
- `entities-param-on-add-memory` → **producer:δ**; the tri-state `None`/`[]`/`[…]` contract — **PASS**
- `structured-conflict-rejection` → G6 branch 4; joins the observed-firing `ValidationError` gate chain at `tools.py:2001-2020` — **PASS**
- `metadata-task-id-bridge` → **producer:δ**; `metadata.task_id` is currently discarded by the Graphiti enqueue — **PASS**
- `resolve-referents-available` → **producer:γ** upstream — **PASS**

### ε — thread the referent set through the queue payload
- `referents-ride-existing-payload` → **producer:ε**; additional key on `add_memory_graphiti` / `add_episode` (`memory_service.py:2633,3130`), no new operation, no `payload_version` — **PASS**
- `old-row-executes-unchanged` → back-compat; absent key ⇒ today's behaviour — **PASS**
- `absent-path-counted` → **INV-4 resolution**; `source='none'` counter emitted on the fallback — **PASS**

### ζ — verification sub-pass
- `verify-episode-referents-runs-last` → **producer:ζ**; ordering after `_normalize_task_node_names` (`memory_service.py:1955`) is load-bearing — **PASS**
- `set-membership-and-per-edge-pairing` → both checks, per resolved decision 7 — **PASS**
- `structured-repair-records` → INV-2 — **PASS**
- `identity-lock-critical-section` → `_identity_lock_for` (`graphiti_client.py:438`, on main) — **PASS**

### η — repair path + streak escalation *(user-observable signal)*
- `ensure-entity-node-mint` → **producer:α** upstream — **PASS**
- `reassign-edge-lossless` → `graphiti_client.py:989` on main; stamps `reassigned_from_node_uuid` — **PASS**
- `summary-drops-repointed-fact` → G6 branch 3, **the batch's highest-risk premise**; `refresh_entity_summary` (`:2439`) regenerates from currently-valid edges, so the repointed fact leaves by construction; `set_entity_summary` (`:2500`) is the 2057 escape hatch — **PASS**
- `repair-streak-escalation` → INV-4; `merge_liveness.py:446` consecutive-streak gate is the named shape — **PASS**
- `unrepairable-recorded-not-guessed` → the `Task 2520` / 2519 case — **PASS**

### θ — remediate the live conflations; close `esc-3375-1` *(structural leaf, deterministic)*
- `four-conflations-repairable` → G6 branch 3; all four foreign facts observed live at decompose — **PASS**
- `fifth-conflation-recorded-not-repaired` → **Finding 2 resolution**; the 2519→`Task 2520` unary case — **PASS**
- `repair-path-available` → **producer:η** upstream — **PASS**
- `pre-act-recensus` → **INV-3 resolution**; θ re-measures before acting — **PASS**
- `gate-untouched-until-execution` → `esc-3375-1` / task 3375 (`blocked`, `operational_mode=gate`, `always_escalates=true`) is **not** mutated at decompose; θ closes it at execution with human approval — **PASS**

### ι — declaration-rate telemetry *(user-observable signal)*
- `source-field-on-resolution` → **producer:γ** upstream — **PASS**
- `declared-bucket-reachable` → **Finding 1 resolution**; requires **producer:δ**, now wired upstream — **PASS**
- `write-ops-params-no-migration` → `write_journal.py:18-36`, JSON `params` + `idx_wo_project_time` — **PASS**
- `per-project-window-query` → **producer:ι** — **PASS**

### κ — drive adoption *(user-observable signal)*
- `entities-param-exists-to-document` → **producer:δ** upstream — **PASS**
- `all-five-targets-in-repo` → G3; every target tracked in-repo (`/memory` → `.claude/commands/memory.md`) — **PASS**
- `example-drift-pinned` → **INV-5 resolution**; drift test over the documented examples — **PASS**

### λ — dated undeclared-referent census *(structural leaf, delayed milestone)*
- `telemetry-to-census` → **producer:ι** upstream — **PASS**
- `adoption-in-flight` → **producer:κ** upstream — **PASS**
- `delayed-milestone-shape` → `mode='delayed'` + `after_secs`, `at` forbidden (`docs/task-authoring.md:509-529`) — **PASS**
- `seven-days-production-data` → the `after_secs` bound is what makes ≥7 days true at dispatch — **PASS**

## Verdict

**No FAIL bindings.** Two G6 findings and three G7 hits, all resolved by tightening the
filed tasks (one added dependency edge, four description/signal tightenings) — no PRD
architecture change, no waivers. Batch cleared to queue.
