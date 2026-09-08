# PRD: Deterministic auto-consolidation of Mem0 near-duplicate clusters

**Status:** active — authored 2026-09-08 (design-first: implementation plan + adversarial
review, four reviewers, Leo ruled every finding in-window the same day).
**Project:** dark_factory. **Approach:** B+H (contracts + boundary-test sketch; G5 hit:
four packages touched, twelve mechanisms, the MCP write boundary and the task interceptor
are load-bearing seams).
**Code anchors** verified against main `98da920301` (2026-09-08). Main moves fast —
cite-by-symbol; re-locate at implementation time.
**Design record:** `~/.claude/fleet/sessions/gates-df-2206924/auto-consolidation-implementation-plan.md`
(11-item plan + "Adversarial review resolutions (2026-09-08)", authoritative where it
conflicts with the plan body). This PRD carries every load-bearing decision itself; the plan
is cited for the evidence trail, not re-derived.

## 1. Goal (G1 consumer + user-observable surface)

Retire the memory-consolidation human gate for the routine case. When reconciliation Stage 1
recognises a cluster of near-duplicate Mem0 records on one topic, the ratified retain-and-tag
consolidation (`docs/prds/memory-metadata-vocabulary.md` §3 Option C, gate 3200: stamp
`metadata.topic` on every peer, mint ONE short index canonical, delete nothing) executes
**automatically when a deterministic predicate holds**, and a human gate is filed **only on a
hazard reason code**. Leo's standing rule (ratified 2026-09-08): anything decidable by code is
code; the LLM keeps only cluster recognition, slug naming, and the one-line claim.

User-observable surface, after one full Stage 1 cycle on a project with the feature enabled:

- A proposal that **passes** ends with every member carrying `metadata.topic`, exactly one
  record carrying `canonical: true` plus an `x_auto_consolidated` pointer, one
  `consolidation_proposal` ledger row in state `addressed` (the authoritative home), one
  `auto_consolidation` system record, an `auto_consolidations` line in the cycle summary, and
  **no** `milestone_gate` escalation and **no** recovery pin for it. `search` then returns the
  canonical first with `topic_anchored: true` whenever any member lands in the window.
- A proposal that **fails on a hazard** ends with a gate task whose description names the
  machine reason code and the exact `update_task(...)` call that stamps
  `human_curator_adjudicated_at` before the sitting resolves it.
- A proposal that **fails benignly** ends with an `info` record on the recon escalation queue
  (port 8103, folded per topic) plus a ledger row in state `refused` carrying the code.
- A `set_task_status(done)` carrying a `deterministic-*` provenance from a recon stage or a
  non-orchestrator MCP client is **refused with a named error**, and the write journal records
  the resolved caller and provenance kind.
- With `consolidation_auto.enabled: false` (the supervised cycle), the same cycle produces
  ledger rows in state `dry_run` and a stats counter, and files **no task** for would-pass
  proposals; `scripts/report_auto_consolidations.py` lists them for the operator.

Consumers (G1): reconciliation Stage 1's post-step
(`fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py::MemoryConsolidator.run`);
the curate-fused-memories sitting (`skills/curate-fused-memories/SKILL.md`, which keeps
`consolidate_memories` under the `curator-` prefix and learns the new gate shape); the
recon-escalation-watcher on 8103 (running since 2026-09-08, lease `recon-watcher-df`); the
orchestrator's `DeterministicRunner` (fewer gates; hazard gates carry `human_curator_gate`);
Leo as operator reading the report script. The legibility census is **not** a consumer: it
reads agent transcripts only and the executor never enters one.

## 2. Background

Measured 2026-09-07/08 (parent session; ledgers on tasks 5117/5123/5130/5169/5180/5182/5183
and mem0 record `686a6bdc`):

- **47 of 47** prior consolidation gates were ruled retain-and-tag, **0 declined**. Leo ruled
  the shape (gate 3200, esc-3200-3) and two batch dispositions; every per-gate ruling was a
  Claude `curator-` session. The human gate never changed an outcome.
- The gate description's "irreversible, content-losing judgment call" is a hardcoded template
  (`reconciliation/consolidation_gate.py::_default_description`), and it contradicts the
  ratified end state rendered into the **same** prompt by `render_end_state_brief` (retain in
  place, never delete). The 2026-08-02 "stop filing gates" policy on task 3524 lives only in
  the sitting skill and a plans ledger; nothing the stages read reflects it.
- Stage 1 already executed the full retain arm for `verify-absence-claims-positive-control`
  42 minutes **after** filing gate 5156: the automatic path runs de facto, unpredicated.
- Task 5156 was then closed by reconciliation Stage 2 with a hand-passed
  `deterministic-gate` provenance; `middleware/task_interceptor.py::_validate_done_provenance`
  has no caller check for `deterministic-*` kinds, and the interceptor's own journaling
  records every `set_task_status` as `agent_id='task-interceptor'`, so the caller is
  unrecoverable after the fact.
- Each gate costs exactly 3 escalation records (13/13 measured), a recovery pin on every
  sweep, the 48 h stall detector, and a human sitting. Every ruled member list undercounted
  the live cluster 2–3×; the genuinely risky cases (5130's stale designate, 5180's false
  "harness built" banner) were caught by an LLM reading the tree, never by the gate.
- Task 3136's binding 2026-08-11 amendment: cosine-of-full-content detectors are inert for gate
  filing (paraphrase recall 8.3%, AUC inverted). Detection stays LLM.
- Retrieval (reviewer 3, 2026-09-08, `text-embedding-3-small`, 5 topics × 3 queries, calibrated
  to live `store_score` within 0.04): a template-built canonical retrieves within 0.015 cosine
  of the sitting's LLM-written canonicals; the topic pin is **member-triggered** in 14/15
  queries; every pinned body is paid per query (one live window carried 6,390 chars of pinned
  canonicals and evicted three real hits). The sitting's canonicals (1,059–1,682 chars) are
  above the §3 shape.

This is a **control-and-audit change**, not a new capability: the writes already happen; the
PRD puts a predicate, a ledger, caps, a reversal, and a caller check around them.

## 3. Sketch of approach

The seam exists: `MemoryConsolidator.run` awaits the LLM turn, then runs pure post-steps on
full cycles only (remediation early-returns; `targeted.py` never instantiates the
consolidator). The executor is one more post-step there. Mechanisms and their consumers:

| # | Mechanism | Consumer |
|---|---|---|
| M1 | `consolidation_auto:` config block (green tier) + `reconciliation.deterministic_provenance_allowed_agent_prefixes` | M5 executor; M7 authz; `reload_config` |
| M2 | Pure predicate `reconciliation/consolidation_auto.py::evaluate_auto_predicate` → `AutoVerdict` | M5; M8 migration script |
| M3 | One argument validator (`server/consolidation.py::validate_consolidate_args`, extended) + pure `build_auto_canonical` | `consolidate_memories`, M4, M2 |
| M4 | `propose_consolidation` MCP tool → `ReconLedgerRecord(record_kind='consolidation_proposal')`; `ReconLedgerStore.list_by_kind` | Stage 1 LLM (producer); M5 (consumer) |
| M5 | Executor post-step `reconciliation/consolidation_auto_executor.py::run_auto_consolidation` + shared flood control + reversal helper | `MemoryConsolidator.run`; report script; sitting |
| M6 | Service-level `services/consolidation_ops.py::execute_retain_consolidation` (retain arm extracted, tag-only mode) | `consolidate_memories` tool; M5 |
| M7 | Provenance caller check in `_validate_done_provenance` + journaled caller; runner pure-gate `escalation_id` | orchestrator `set_task_status`; the interceptor's refusal path |
| M8 | Hazard gate shape (`reason_codes`, `human_curator_gate`, `observed_members`, stamping instruction) + `scripts/report_auto_consolidations.py` + `scripts/migrate_open_consolidation_gates.py` | DeterministicRunner; the sitting; Leo |

Flow per full cycle: LLM turn emits `propose_consolidation(...)` (validated at the emit
boundary, persisted to the ledger) → post-step lists unaddressed proposals → ranks them under
the shared flood-control policy → for the top `max_auto_per_cycle + max_gate_filings_per_cycle`
reads members and the canonical count deterministically → verdict → PASS / PASS_TAG_ONLY
execute via M6, hazard FAIL files a gate, benign FAIL escalates `info`, NOOP writes nothing →
closure check via task 4808's helper → ledger row state updated, stats appended.

## 4. Contracts (H)

### C1 — Proposal contract (LLM emits, code validates)

```
propose_consolidation(project_id, run_id, topic, member_ids, category, claim, rationale,
                      seeded_from=None) -> {'ok': True, 'ledger_key': ...}
                                        | {'error', 'error_type': 'ValidationError', 'hint'}
```

- Gated on `agent_id.startswith('recon-stage-')` (precedent `tools.py::add_system_record`).
  Denied to Stage 2 and Stage 3 via an **additive** sublist in `cli_stage_runner.py`
  (`DISALLOW_RECON_REPORT_LEDGER_WRITES` pattern) — one proposer.
- **No `metadata` parameter** (the markup override forwards `allow_mcp_markup` to any tool
  that has one; the markup guard wraps every tool automatically).
- Validation collects every offender: slug via the shared regex; `member_ids` UUIDs, count in
  `[member_min, member_max]`; `claim` one line, ≤ `claim_max_chars`, no leading `[`, not a
  label (`^(INDEX|CANONICAL)\b` rejected; alphanumeric tokens ⊆ slug tokens ∪ stopwords
  rejected; ≥ 8 words); `category` in the six-category vocabulary.
- Persisted as `ReconLedgerRecord(project_id, record_kind='consolidation_proposal',
  flag_type=<topic>, run_id, payload_json={member_ids, category, claim, rationale,
  seeded_from}, state='proposed', expires_at=now+ttl)`. Last-write-wins per (project, topic,
  run). `rationale` and `seeded_from` are reviewer-facing and never reach canonical text.

### C2 — Predicate contract

```
evaluate_auto_predicate(proposal, *, members: dict[id, MemoryRecord|None],
                        canonical_count: int | None, open_gate_id: str | None,
                        existing_canonical_slugs: Sequence[str], config) -> AutoVerdict
AutoVerdict = {outcome: PASS|PASS_TAG_ONLY|NOOP|FAIL, reasons: [{code, ids, detail}],
               retain_ids: [...], stripped_ids: [...], predicate_version: '1'}
```

Import-leaf (like `consolidation_gate.py`), pinned by a leaf-import test. Never calls
`search`; the caller reads `get_memory_by_id` per member,
`count_memories_by_metadata({'topic': T, 'canonical': True})`, the open-gate scan
(`curator_gate_resolution_sweep.py::extract_open_gate_task_ids`), and the canonical slug list.

Evaluation order (binding — see D4):

1. `already_gated` → NOOP carrying the open gate id.
2. Hazard codes (any ⇒ FAIL): `member_unreadable` (fail closed), `member_not_found`,
   `member_already_canonical` (a canonical of a **different** topic),
   `member_different_topic` (`topic ∉ {None, T}`), `member_carries_correction_metadata`
   (`corrects` / `superseded_by` / `stale_pre_fix_state`), `member_carries_correction_banner`
   (regex, refusal-only, can never grant a pass), `canonical_carries_correction` (banner or
   correction keys **on the incumbent**), `mixed_category`, `canonical_category_mismatch`
   (incumbent category ≠ members'), `multiple_canonicals`, `canonical_count_unavailable`,
   `slug_near_collision` (token-Jaccard against existing canonical slugs above a config
   threshold).
3. An incumbent canonical of topic T inside `member_ids` is **stripped** into
   `stripped_ids` with disclosure, not a FAIL.
4. `already_consolidated` (all retained members stamped T ∧ canonical_count == 1) → NOOP.
5. Any unstamped member ∧ canonical_count == 1 → PASS_TAG_ONLY.
6. canonical_count == 0 → PASS.

Benign codes (`member_count_out_of_range`, `invalid_slug`, `claim_not_index_shaped`) are
refused at the emit boundary by C1; the predicate re-derives none of them (one validator).

### C3 — Retain / tag-only op

```
execute_retain_consolidation(memory_service, *, project_id, topic, canonical_content: str|None,
                             retain_ids, category, agent_id, run_id, extra_canonical_meta)
    -> ConsolidationResult (the same `survivors` envelope `build_consolidation_result` returns)
```

The peer-tag loop, canonical mint, and step-6 closure scroll move **unchanged** out of the
`server/tools.py::consolidate_memories` closure. `canonical_content=None` ⇒ tag-only: skip the
mint, return the incumbent id, touch nothing on the incumbent. The helper calls
`resolve_mem0_update_authorization` itself with the caller's identity — the executor runs as
`recon-stage-memory_consolidator`, which the default `mem0_update` prefixes already allow. The
tool keeps authz-of-delete, citations, and the delete arm, and calls the helper.

Canonical text is `build_auto_canonical(claim, topic, n, run_id)`, verbatim:

```
f'{claim}\n\nIndex canonical for topic `{topic}` over {N} short peers; the live metadata.topic scroll is the member list (auto-consolidated, run {run_id}).'
```

Bound ≈ 470 chars at claim=200, slug=100, N=20 — asserted by a unit test of the builder, not
by a runtime cap (see D5). Canonical metadata: `topic`, `canonical: true`, `category`,
`x_auto_consolidated: {run_id, predicate_version, members_count, canonical_id, at}` — counts
and pointers only; the full member list lives on the ledger row and the system record.
Each newly tagged member gets `x_auto_consolidated_run: <run_id>`.

### C4 — Executor post-step + ledger

```
run_auto_consolidation(memory_service, taskmaster, escalation_queue, *, project_id, run_id,
                       config, report) -> None   # appends report.stats[...]
```

- Runs only on full cycles, only when `project_id ∈ config.enabled_projects`; otherwise
  returns after appending `auto_consolidations_skipped_project`.
- Lists `ReconLedgerStore.list_by_kind(project_id, 'consolidation_proposal', 'proposed')`;
  ranks under `consolidation_flood_control.rank_and_cap(...)` (N × category weight ×
  canonical-absent bonus; execute ≤ `max_auto_per_cycle`, file ≤
  `max_gate_filings_per_cycle`; if proposals > cap × `backlog_multiplier`, execute nothing
  and emit ONE aggregate **escalation** to 8103). Predicate reads happen only for the ranked
  top slice (INV-8).
- Per verdict: PASS → C3 with a minted canonical; PASS_TAG_ONLY → C3 tag-only; NOOP → nothing;
  hazard FAIL → gate task (C6) unless `already_gated`; benign/aggregate → in-stage
  `Escalation(level=1, severity='info', category='recon_auto_consolidation_refused',
  summary=f'{topic}: {code}')` via `submit_or_dedupe` (precedent
  `stage1_stall_detector.py::maybe_escalate_stalled_gate_backlog`; the harness's `_escalate`
  hardcodes severity and is not used).
- After a PASS / PASS_TAG_ONLY: `evaluate_closure(..., unstamped_live_ids=
  resolve_unstamped_live_ids(...))` using task 4808's helper; a non-closed verdict is a FAIL
  that files a gate. Then `add_system_record(metadata={'record_kind': 'auto_consolidation',
  'topic', 'canonical_id', 'run_id', 'member_ids'})`, ledger row → `addressed`.
- `enabled=false`: PASS / PASS_TAG_ONLY / NOOP → ledger `state='dry_run'` +
  `report.stats['auto_consolidations_dry_run']`, no writes to memories, no task; hazard FAIL
  files a gate exactly as when enabled; benign FAIL → info. This is the supervised cycle.
- Per-(project, topic) consecutive refusals ≥ `refusal_streak_threshold` promote the next
  refusal record to `severity='blocking'` (INV-4 escape).
- Disclosure: `report.stats` carries `auto_consolidations`, `auto_consolidations_tag_only`,
  `auto_consolidations_dry_run`, `auto_consolidation_refusals{code: n}`,
  `auto_consolidation_gates_filed`, `applied_cap`, `unexecuted`, `unfiled` (INV-2/INV-11).
- `revert_auto_consolidation(memory_service, ledger_row)`: for a minted canonical
  `delete_memory(canonical_id)`; for every member the row tagged, `update_memory(
  metadata_delete_keys=['topic', 'x_auto_consolidated_run'])`; never touches an incumbent.
  Called by the report script's `--revert <ledger_key>` and by the sitting.

### C5 — Provenance caller contract

- `_validate_done_provenance(...)` refuses `kind ∈ {deterministic-gate, deterministic-milestone,
  deterministic-deploy, deterministic-deploy-scheduled}` when `is_recon_stage=True`
  (extends the existing `operational-verified` branch) with error type
  `DeterministicProvenanceCallerNotPermitted`.
- Otherwise the resolved `agent_id` must match a prefix in
  `reconciliation.deterministic_provenance_allowed_agent_prefixes` (default
  `['orchestrator']`; live-read, deny-on-missing, `mem0_update_authz.py` shape). The
  orchestrator sends no `agent_id` and `tools.py::_resolve_identity` falls back to
  `clientInfo.name`, which `orchestrator/src/orchestrator/mcp_lifecycle.py::McpSession.initialize`
  advertises as `'orchestrator'`. Self-reported: a deterrent for cooperating callers, not a
  boundary — stated in `docs/task-authoring.md`.
- The write journal row for a done write (accepted or refused) carries the resolved
  `agent_id` and `done_provenance.kind`.
- `deterministic-gate` requires `escalation_id`; `deterministic_runner.py`'s pure-gate branch
  passes `_gate_esc_id` like the curator branch already does. Recorded verbatim, no lookup.

### C6 — Hazard gate shape

`build_consolidation_gate_task(topic, member_ids, category, *, reason_codes,
auto_predicate_version, ...)`: `task_kind='deterministic'`, `always_escalates=True`, no
`before_done`, `metadata.human_curator_gate: True`,
`x_recon_consolidation_gate.provenance.observed_members = proposal.member_ids`
(`authoritative: False`), description "Auto-consolidation refused: `<code>` (ids …).
Before resolving, stamp `update_task(<id>, metadata={'human_curator_adjudicated_at':
'<ISO>'}, metadata_mode='merge')`." Filed through the raw backend's `add_task` (Stage 2
precedent `stages/task_knowledge_sync.py`); the skipped guards are named in D8.

## 5. Boundary-test sketch (the integration gate's signal — task η)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Fresh cluster passes | 5 unstamped members, no canonical, valid claim, `enabled=true`, project enabled | 5 members carry `topic`, one canonical with `x_auto_consolidated`, ledger `addressed`, system record, stats `auto_consolidations=1`, escalation queue has no `milestone_gate`, no task filed |
| B2 | Re-emission with regrowth | Topic consolidated (canonical C, m1–m5 stamped); LLM proposes m1–m5 + n1–n3 + C | PASS_TAG_ONLY; C stripped and disclosed; n1–n3 stamped with `x_auto_consolidated_run`; C's content and metadata unchanged |
| B3 | Fully consolidated re-emission | All members stamped, count==1 | NOOP; zero writes; ledger `addressed` with reason `already_consolidated` |
| B4 | Hazard: correction on incumbent | Incumbent carries `superseded_by` | FAIL `canonical_carries_correction`; gate task with `human_curator_gate`, reason code in description, `observed_members` populated; no memory writes |
| B5 | Hazard already gated | Open gate on (project, topic) | NOOP `already_gated` naming the gate id; no second task |
| B6 | Benign refusal at emit | `claim` = "INDEX — foo" | `propose_consolidation` returns `ValidationError` + hint; no ledger row |
| B7 | Benign refusal in-stage | Proposal > `member_max` members slipped past (older row) | `info` escalation on the queue, summary `<topic>: member_count_out_of_range`, ledger `refused`; after `refusal_streak_threshold` repeats the next record is `blocking` |
| B8 | Supervised cycle | `enabled=false`, B1 inputs | Ledger `dry_run`, stats `auto_consolidations_dry_run=1`, zero memory writes, zero tasks; hazard input still files a gate |
| B9 | Backlog trip | proposals > cap × multiplier | Nothing executed, one aggregate escalation, stats disclose `unexecuted` |
| B10 | Reversal round-trip | After B1, `revert_auto_consolidation` | Canonical deleted, members' `topic` removed, census shows no orphan; after B2 reversal only n1–n3 un-stamped |
| B11 | Provenance refusal | `set_task_status(done, done_provenance={kind: 'deterministic-gate'})` as `recon-stage-task_knowledge_sync` | Refused `DeterministicProvenanceCallerNotPermitted`; journal row carries agent id + kind; the same call with clientInfo `orchestrator` and `escalation_id` is accepted |
| B12 | Stage surface | Stage 2 / Stage 3 tool allowlists | `propose_consolidation` absent; `consolidate_memories` absent from Stage 1 and 2; every fused-memory tool classified |
| B13 | Project scoping | `enabled_projects=['dark_factory']`, proposal on `reify` | Skipped with stats `auto_consolidations_skipped_project`; no reads of reify members |

## 6. Resolved design decisions

- **D1 Detection stays LLM; everything after emission is code.** Binding per task 3136. The
  optional seed from `scripts/audit_duplicate_memories.py` is report-only and recorded as
  `seeded_from` on the proposal (seeding is not wiring; the field yields the recall measurement
  task 4916 is waiting for).
- **D2 The LLM does not write the canonical.** Measured (§2). It writes `claim` only, inside the
  same Stage 1 turn — zero extra LLM calls. `rationale` never enters canonical text.
- **D3 Proposal channel is a tool, not `add_finding` / `flagged_items`.** Refusal at the emit
  boundary so the LLM fixes shape in-turn; durable across a crashed stage; `flagged_items`
  carries no validated schema (the plan's "warns on task-less items" was overstated).
- **D4 Predicate order is C2's, with `already_gated` first and hazards before NOOP.** A new
  member with a correction banner must FAIL even beside a live canonical; an incumbent in the
  member list is stripped, not failed; PASS_TAG_ONLY is the majority verdict over time (gate
  3200 decision 3: recurrence is reduced, still real).
- **D5 No runtime canonical length cap.** `canonical_max_chars` would be unreachable dead code
  (max ≈ 470 < 600); the builder's bound is a unit-test assertion. The claim cap (200) is a
  green-tier leaf `claim_max_chars`.
- **D6 One validator, one flood-control home, one fact-home.** `server/consolidation.py`
  validates for the tool, the proposal tool, and the predicate. Flood control lives in
  `reconciliation/consolidation_flood_control.py` with ONE config block shared with task 4916's
  detector report (3136's binding amendment: config-resident). The ledger row is the
  authoritative record of "topic T auto-consolidated by run R"; every other surface carries
  `{canonical_id, run_id}` pointers.
- **D7 Reversal deletes a minted canonical and un-stamps only that run's members.** A
  `canonical: False` record still carrying `topic` is an orphan the anchor selector ignores
  and no census code names.
- **D8 Hazard gates are filed through the raw backend, and the skipped guards are named.** No
  TaskCurator LLM call (cost win), no `_inject_deterministic_pure_gate`, no premise/markup
  guard, no candidate-key dedup — hence `already_gated` in the predicate and a post-hoc test
  that a raw-filed gate passes `deterministic_task_guard`.
- **D9 Hazard gates keep `human_curator_gate: True`**, and the sitting skill gains Phase 3a
  (stamp `human_curator_adjudicated_at` before `resolve_issue(resume)`). Nothing else stamps
  it today; the runner's rung (ii) otherwise re-blocks up to `_REBLOCK_GUARD_THRESHOLD` times.
- **D10 Benign refusals go to 8103 as `info`, built in-stage.** Leo: the queue was unread
  recently by accident, not intent; a recon watcher now runs and will be kept running. Both
  new categories join `escalation/server.py::CATEGORIES` and the watcher skill's playbook.
- **D11 Provenance check is both factors plus a journaled caller.** The recon-stage branch is
  config-free and catches the 5156 writer class; the allowlist catches an omitted `agent_id`;
  the journal makes the residual (a deliberate spoof) visible.
- **D12 `consolidate_memories` leaves the Stage 1/2 surface additively**, retiring task 3134's
  advertisement (its test, the `STALE_KNOWLEDGE_ANNOTATION_NORM` reference, five prompt sites)
  with a dated pointer on `docs/prds/memory-write-path-convergence.md` §9 ι / C2 that Stage 1's
  caller is now the executor. Sittings keep the tool via the `curator-` prefix.
- **D13 Scope is all projects; rollout is per project.** `enabled_projects` (default `[]`,
  `summary_rebuild.projects` / `backlog_hard_limit_overrides` precedent) stages the
  supervised cycle on `dark_factory` first, then every reconciled project. The 31 open gates
  (7 df pending, 24 reify) are migrated by script (task θ) after each project's clean cycle;
  until then they stay sitting-owned.
- **D14 Tag-only refreshes nothing on the incumbent** (3112 doctrine; `content_amend=False` is
  load-bearing). A stale-but-unbannered incumbent (5130, 5180) is undetectable by code and
  stays with the sitting.
- **D15 Cross-topic overlap:** a member already stamped with a different topic is decidable and
  a FAIL naming both slugs; overlap among unstamped members is the LLM's slug choice, backstopped
  by `slug_near_collision`.

Code-quality heuristics (`docs/code-quality.md`, cited by name, not restated) that the
decisions above rest on: **SPOT** and **clear invariants, uniformly enforced** (D6: one
validator, one flood-control home, one fact-home; every enforcement point calls the one
definition); **structured data instead of meaningful strings** (`AutoVerdict` outcomes and
reason codes are typed values; the ledger row's free-string `state` is parsed into an enum at
the executor boundary; the `recon-stage-` prefix gate is used only where the
`add_system_record` precedent already routes on it and is not extended); **carefully factored
orthogonal dimensions of variability** (predicate, execution, filing, and disclosure are four
mechanisms, each with one axis of change); **deep modules with coherent narrow interfaces**
and **files make internal sense in isolation** (`consolidation_auto.py` is an import-leaf;
`consolidation_ops.py` owns execution and reaches into nothing); **no file too large** (β
moves the retain arm out of `server/tools.py` rather than adding to it, and γ's tool body is a
thin registration over `server/consolidation.py`).

## 7. Pre-conditions / substrate (G3 — verified 2026-09-08 against main `98da920301`)

Verified present and wired:

- The post-LLM deterministic seam: `MemoryConsolidator.run` runs `sweep_resolved_curator_gates`,
  `filter_stale_count_snapshot_corrections`, `dedup_flags`, `maybe_escalate_stalled_gate_backlog`
  after `super().run()`; remediation passes early-return; `run` holds `self.memory`,
  `self.taskmaster` (raw `SqliteTaskBackend`), `self._escalation_queue`, `run_id`;
  `summary_pool.py::write_cycle_summary` serialises `report.stats` whole.
- The retain arm inside `tools.py::consolidate_memories` (peer-tag loop → `update_memory(
  content=None, metadata_patch, metadata_mode='merge')`; mint → `memory_service.add_memory`;
  closure scroll), closing over `memory_service, project_id, topic, agent_id, session_id,
  causation_id, source`. `server/consolidation.py` is pure validation/envelope.
- `ReconLedgerRecord` fields and PK `(project_id, record_kind, task_id, flag_type, run_id)`;
  `upsert` last-write-wins; `mark_addressed`; index `ix_recon_ledger_project_kind_state`.
  **Absent:** a list-by-kind reader; TTL for NULL `expires_at` (task γ adds both).
- `_resolve_identity` clientInfo fallback; `Scheduler.set_task_status` sends no `agent_id`;
  `McpSession.initialize` advertises `'orchestrator'`; only `deterministic_runner.py::
  _build_done_provenance` produces `deterministic-*` kinds in production.
- `_validate_done_provenance(task_id, raw, project_root, *, require, is_recon_stage=False)`
  with the `is_recon_stage` refusal branch for `operational-verified`.
- `escalation/pins.py::_classify_record`: only `severity='info'` is NON_PINNING.
- `shared/task_metadata.py::HUMAN_CURATOR_GATE_KEY` / `HUMAN_CURATOR_ADJUDICATED_AT_KEY`; the
  runner's rung (ii) `_curator_adjudication_confirmed`. **Absent:** any stamper (task ζ adds
  the skill step).
- `config/schema.py::Mem0UpdateConfig` bare-mounted (an `X | None` submodel would be
  restart-only); `config/reload.py::RELOADABLE_FIELDS`; `mem0_update_authz.py` live-read,
  deny-on-missing.
- `MemoryService.update_memory(metadata_delete_keys=...)`; `count_memories_by_metadata -> int`;
  `services/topic_anchor.py::select_canonical_payload` (pure metadata, member-triggered pin).
- `server/markup_guard.py::install_markup_guard` wraps every tool except `scan_memory_content`.
- `curator_gate_resolution_sweep.py::extract_open_gate_task_ids` (open-gate scan).
- `stage1_stall_detector.py::maybe_escalate_stalled_gate_backlog` (in-stage `Escalation` +
  `submit_or_dedupe` precedent).

External prerequisites (hard, in the DAG):

- **Task 4808** (closure helper `resolve_unstamped_live_ids`, `provenance.observed_members`):
  branch tip `8d3af634ea`, **not on main**. Task δ depends on it.
- **Task 3136 / 4916** (detector report, `duplicate_audit` config block): **not on main**.
  This PRD creates the shared flood-control helper and block; 4916 consumes it (seam table).
- `memory_metadata.enforce` is `False` (task 3626, red-tier, untouched): canonical uniqueness
  is censused, not enforced, so the predicate probes the count itself.

No novel substrate beyond these.

## 8. Out of scope

- Any read transform on the search path (gate 3200: not ratified; the promoting pin landed
  separately via task 3111).
- Deleting memories, or any change to the delete arm of `consolidate_memories`.
- Wiring cosine detector signals to filing (task 3136 binding; task 4916 owns the report).
- The `memory_metadata.enforce` flip (task 3626).
- Recon transcript archival (task 5202) and the markup leak (task 5055).
- Detecting a stale-but-unbannered canonical by code (D14).
- The orchestrator's `milestone_gate` category model and `L2_AUTO_CLOSE_DENY_CATEGORIES`
  (`docs/prds/recurring-deterministic-tasks.md`), which this PRD uses unchanged.

## 9. Cross-PRD / seam ownership (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `docs/prds/memory-metadata-vocabulary.md` §3 Option C / gate 3200 | consumes (write shape) | retain-and-tag semantics; `topic`/`canonical` keys | other (ratified) | wired |
| `docs/prds/memory-write-path-convergence.md` §9 ι, C2 / task 3134 | produces (retires Stage-1 advertisement) | `STAGE1_DISALLOWED` sublist; advertisement test; norm text | this PRD (task γ + ζ pointer) | queued |
| `docs/prds/memory-write-path-convergence.md` leaf κ / tasks 3136, 4916 | produces (shared flood-control block) | `consolidation_flood_control.py` + one config block | this PRD (task δ); 4916 consumes | queued; 4916 to be amended at decompose |
| Task 4808 | consumes | `resolve_unstamped_live_ids`, `provenance.observed_members` | other (in merge lane) | blocked until landed |
| Task 3626 | consumes | `_check_canonical_uniqueness` (census only) | other | untouched |
| `docs/prds/recurring-deterministic-tasks.md` / task 3341 | consumes | pure gate, `milestone_gate`, rung (ii), `human_curator_gate` | other | wired |
| `plans/toolcall-markup-containment-prd.md` | consumes | `install_markup_guard` (automatic) | other | wired |
| `plans/config-hot-reload-prd.md` | consumes | fused-memory green tier (`RELOADABLE_FIELDS`) | other | wired |
| Task 3524 (DECIDE-FIRST) | consumes | open-gate cross-check before filing | this PRD (`already_gated`) | queued |
| `skills/curate-fused-memories/SKILL.md` | produces | Phase 3a stamp; builder convergence; stale anchored-read line | this PRD (task ζ) | queued |
| `skills/recon-escalation-watcher/SKILL.md` | produces | two category playbook rows | this PRD (task ζ) | queued |

No reciprocal-ownership ambiguity: every seam above has one owner.

## 10. Decomposition plan (signals are the G2 gate; Greek labels this-PRD-local)

Deps: γ ← {α, β}; δ ← {γ, task 4808}; ε ← α; ζ ← δ; η ← δ; θ ← δ. α ‖ β ‖ (ε after α).
Same-file serialisation is carried by those edges (`server/tools.py`: β → γ → ζ;
`consolidation_gate.py`: γ → δ; `config/schema.py`: α → ε).

- **α — Config, predicate, validator, builder** (fused-memory; ~600 LOC; `config/schema.py`,
  `config/reload.py`, `config/config.yaml`, new `reconciliation/consolidation_auto.py`,
  `server/consolidation.py`, tests). `ConsolidationAutoConfig` mounted bare as
  `consolidation_auto` with leaves `enabled=False`, `enabled_projects=[]`,
  `predicate_version='1'`, `member_min=2`, `member_max=20`, `claim_max_chars=200`,
  `max_auto_per_cycle=3`, `max_gate_filings_per_cycle=3`, `backlog_multiplier=5`,
  `refusal_streak_threshold=5`, `slug_collision_jaccard=0.6`, `proposal_ttl_hours=168`,
  `category_weights`; plus `reconciliation.deterministic_provenance_allowed_agent_prefixes=
  ['orchestrator']`; all in `RELOADABLE_FIELDS`. `evaluate_auto_predicate` per C2,
  `build_auto_canonical` per C3, `validate_consolidate_args` extended with the claim/label
  rules. *Signal (intermediate → γ, δ, ε):* `reload_config` on `consolidation_auto.enabled`
  reports `applied`; one test per reason code executes the predicate against fixture members;
  the builder-bound test asserts < 500 chars at the extremes. G7: INV-10 (tests execute the
  predicate and builder, no prose pins).
- **β — Extract the retain arm** (fused-memory; ~400 LOC; new `services/consolidation_ops.py`,
  `server/tools.py`, tests). Per C3. *Signal (intermediate → γ, δ):* the existing
  `tests/test_consolidate_memories_tool.py` fixture cluster returns a byte-identical
  `survivors` envelope before and after; tag-only returns the incumbent id and leaves the
  incumbent's content and metadata unchanged (two-way test). G7: INV-5 (one loop for tool and
  executor).
- **γ — Proposal tool, ledger reader, stage surface, prompt retirement** (fused-memory; ~700
  LOC, ~11 files; `server/tools.py`, `reconciliation/recon_ledger.py`, `cli_stage_runner.py`,
  `prompts/stage1.py`, `prompts/__init__.py`, `recon_self_model.py`,
  `consolidation_gate.py::render_consolidation_gate_section`, tests incl. a new
  every-fused-memory-tool-is-classified test and the replacement of
  `test_the_advertised_op_is_one_the_stage_actually_holds` with a referential-integrity test
  for `propose_consolidation`). Per C1, D3, D12. *Signal (intermediate → δ):* a
  `recon-stage-memory_consolidator` call with a label claim returns `ValidationError` with a
  hint and writes nothing; a valid call lands a row that `list_by_kind` returns; the Stage 2
  and Stage 3 allowlists exclude the tool and Stage 1/2 exclude `consolidate_memories` (B6,
  B12). G7: INV-1 (tool schema is the contract), INV-10 (referential-integrity pins, not
  substrings).
- **δ — Executor post-step, flood control, gate shape, reversal** (fused-memory + escalation;
  ~1,200 LOC; new `reconciliation/consolidation_auto_executor.py`, new
  `reconciliation/consolidation_flood_control.py`, `stages/memory_consolidator.py`,
  `consolidation_gate.py::build_consolidation_gate_task` + `_default_description`,
  `escalation/src/escalation/server.py::CATEGORIES`, tests). Per C4, C6, D4–D11. Depends on
  task 4808 for the closure helper. *Signal (leaf-bearing intermediate → ζ, η, θ):* B1, B3,
  B4, B5, B7, B8, B9, B10 against a fake `MemoryService` and the real predicate, ops, and
  ledger; the escalation queue fixture shows severity `info` for a refusal and no
  `milestone_gate` for a PASS. G7: INV-4 (refusal streak), INV-7 (proposal TTL + state),
  INV-8 (reads only over the ranked top slice), INV-9 (ledger row is the home), INV-11 (every
  verdict disclosed in stats and the ledger).
- **ε — Close the hand-passed-provenance hole** (fused-memory + orchestrator; ~400 LOC;
  `middleware/task_interceptor.py`, new `middleware/done_provenance_authz.py`,
  `services/write_journal.py`, `orchestrator/src/orchestrator/deterministic_runner.py`,
  `docs/task-authoring.md`, tests). Per C5. *Signal (leaf):* B11 — the recon-stage call is
  refused with `DeterministicProvenanceCallerNotPermitted`, the write-journal row names the
  caller and kind, the orchestrator-clientInfo call with `escalation_id` is accepted, an empty
  prefix list denies all, and `reload_config` flips the list live. G7: INV-1 (named error
  type), INV-2 (journal facts at the refusal).
- **ζ — Review surfaces, skills, operator docs, cross-PRD pointers** (scripts + skills + docs +
  one docstring; ~400 LOC; new `scripts/report_auto_consolidations.py` + test,
  `skills/curate-fused-memories/SKILL.md` (Phase 3a stamp, builder convergence, retire the
  stale "nothing should assume an anchored read" line), `skills/recon-escalation-watcher/
  SKILL.md` (two rows), `OPERATIONS.md` (fused-memory green-tier paragraph), the `search`
  tool docstring / `FUSED_MEMORY_INSTRUCTIONS` (the accretion instruction, once),
  `docs/prds/memory-write-path-convergence.md` §9 ι / C2 and
  `docs/prds/memory-metadata-vocabulary.md` §3 dated pointers to this PRD). *Signal (leaf):*
  the script exits 0 listing executions, dry-runs and refusals for a run id from the ledger
  and system records, exits 2 when the run has none, and `--revert <ledger_key>` performs
  B10; the skill's Phase 3 names the stamp call. G7: INV-9 (pointers, not copies).
- **η — End-to-end integration gate** (fused-memory tests; ~500 LOC; one new test module).
  A fixture harness drives the real `propose_consolidation` tool into the real ledger, then
  `MemoryConsolidator.run`'s post-step with the real predicate and ops against a fake
  `MemoryService`, and asserts every row of §5 (B1–B13). *Signal (leaf):* the boundary-test
  table passes as one suite; it is RED against the reverted executor. G7: INV-10.
- **θ — Migrate the open gates** (scripts; ~300 LOC; new
  `scripts/migrate_open_consolidation_gates.py` + test). For each open consolidation gate of
  a project: re-read the live members (never the gate's snapshot list — every ruled list
  undercounted), run the predicate, execute PASS / PASS_TAG_ONLY under the cycle caps, stamp
  `observed_members`, and print the gates now closable via `resolve_issue(resume)` (the runner
  closes them; the script never sets a task status). *Signal (leaf):* `--dry-run` on
  dark_factory prints the 7 pending gates with verdicts; `--execute` leaves each PASS gate with
  `evaluate_closure` closed and lists it. G7: INV-3 (re-read live state before acting).

Decompose-time actions for the decomposing session, not tasks: amend task 4916's details with
the shared flood-control block name via `update_task`; dep-gate δ on 4808 as an out-of-batch
dependency; file ζ and θ with `complexity` reflecting their docs/scripts weight.

## 11. Rollout

1. Land α, β, γ, ε, δ (δ after 4808). Config default: `enabled=false`, `enabled_projects=[]`.
2. `reload_config`: `enabled_projects=['dark_factory']`, `enabled=false`. One full Stage 1
   cycle (dark_factory runs ~7.8/day). Leo reads the dry-run rows with the report script; a
   false PASS is a predicate bug and blocks step 3.
3. `reload_config`: `enabled=true`. Watch one cycle: executions, zero `milestone_gate` records
   for them, canonical pinned first in `search`.
4. Run θ on dark_factory. Then repeat 2–4 per project (reify next), ending with every
   reconciled project in `enabled_projects`.
5. Kill switch: `enabled=false` (green tier) stops writes within one cycle; `--revert` undoes a
   specific execution.

## 12. Open questions (tactical, implementation-time)

1. **Category weights for ranking.** Default `{procedural_knowledge: 1.0,
   preferences_and_norms: 1.0, observations_and_summaries: 0.7}`? Decide in δ; the block is
   green-tier so it tunes live.
2. **`slug_collision_jaccard` default.** 0.6 is a guess bounded by the 4774/4957 collisions;
   δ's test fixtures should include those two slugs. Calibrate during the supervised cycle.
3. **Proposal TTL.** 168 h; a proposal older than one week that no cycle addressed is stale by
   construction. Decide in γ.
4. **Stopword list for the label check.** Reuse the slug tokenizer's list if one exists in
   `fused_memory.topic_slug`; otherwise a 30-word inline list. Decide in α.
5. **Where the accretion instruction lives** — the `search` docstring or
   `FUSED_MEMORY_INSTRUCTIONS`: whichever the pin caveat already lives in. Decide in ζ.
