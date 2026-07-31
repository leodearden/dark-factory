# Capability manifest — memory-eval-program

Machine-readable twin: `memory-eval-program.capability-manifest.yaml` (same stem; path
strictly derived from the PRD path, so `commit_planning` can find it from
`metadata.prd_path` alone). Batch authored 2026-07-29, task ids stamped by
`commit_planning`; all eight leaves filed `planning_mode`. PRD committed at
`b799b37194`; substrate re-verified against main on 2026-07-29 at the decompose walk
(PRD §6 was itself produced by a three-agent pass earlier the same day).

38 capabilities across 8 leaves, **all PASS**, 25 of them carrying a mechanical
(`grep`) `delivered_check`; 13 recorded `kind: manual` and therefore excluded from the
dispatch gate.

| Leaf | Task | Load-bearing capabilities | Verdict |
|---|---|---|---|
| α M1/M2 schema + evaluator | (stamped) | `shared/` single-home metric schema + limits evaluator; wrong-direction `orchestrator.evals` import absent (D2/INV-5); budget-derived α with no a-priori threshold (G6) | PASS |
| β E1 retrieval probe | (stamped) | read-only probe runner (D8 `_run` pattern); committed topic registry keyed off the vocabulary namespace, no dep on gate 3200 (D5); held-out-phrasing Goodhart guard; seeded-ephemeral fixtures, never the live store | PASS |
| γ E4 staleness sweep | (stamped) | `normalize_supersedes()` imported from **3196 upstream** (D7/INV-5 — never a second parser); three metric families in one M1 artifact; live `get_memory_by_id` dangling resolution (INV-3) | PASS |
| δ E6 detector + series | (stamped) | `with_vectors` widened on `scroll_by_metadata`; `query_points` ANN feeding the **existing** union-find; 3130's labeled fixture already on main; measured-not-asserted similarity (G6); detector CLI path stable for 3136; dropped-candidate disclosure (INV-2/4) | PASS |
| ε scheduled runner + M3 | (stamped) | flag-marker systemd quadruple; direct `EscalationQueue` filing (D4); dedup reusing `has_open_l1` (INV-5); `eval_regression` in the `CATEGORIES` doc list + the watcher's SKILL.md triage row (INV-1); storm escape + runner-failure self-escalation (INV-4/INV-2); first-run grandfather snapshot (D1/INV-3) | PASS |
| ζ E7 telemetry | (stamped) | `caller_agent_id`/`caller_task_id` declared as first-class search params distinct from the `agent_id` **filter** (INV-1); briefing threading; `_MEMORY_INSTRUCTIONS` pinned to the real param names (INV-5); hint-execution journaling; `prune_write_ops` retention; journal-drop visibility counter (INV-4) | PASS |
| η write-after-miss | (stamped) | ζ's ids+scores+caller rows upstream; **replays** the production pure guard `find_near_duplicate_memory` rather than re-implementing similarity (INV-5); metric wired into ε's evaluation; structured incident evidence (INV-2) | PASS |
| θ retro transcript corpus | (stamped) | one-shot script over the live agent-transcript archive; reuses `load_transcript` / `_iter_json_lines` — no third parser (D9/INV-5); coverage report distinguishes wholesale parse failure from an empty archive (INV-2/4) | PASS |

## Bindings that needed work

**No FAIL bindings; no dependency edge had to be added and no signal had to be
re-homed or relaxed.** Three bindings were investigated as candidate FAILs and cleared:

- **δ ← 3130's labeled dataset.** PRD §8 assigns the labeled duplicate-pair fixture to
  task 3130 (write-path PRD α), which is still `in-progress`, and §5's deps line wires
  no edge to it — the `producer-absent` shape. Cleared on evidence: 3130's fixture
  **is already on main** (`fused-memory/tests/fixtures/write_triage_calibration.jsonl`,
  104 curator-ground-truth records with per-record provenance, plus
  `fixtures/README.md`), so the capability binds `grep`-on-main and δ needs no edge to
  3130. Bound mechanically so a later removal of the fixture re-opens the gate.
- **γ ← `normalize_supersedes()`.** Verified the symbol exists today **only in PRD
  prose**, nowhere in code — so the capability rests entirely on producer task 3196,
  which is `pending` and wired **upstream** of γ (DAG-direction verified). PASS as
  `producer:task-3196 upstream`.
- **η's near-duplicate predicate.** Bound to `find_near_duplicate_memory`
  (`fused-memory/src/fused_memory/server/near_duplicate_guard.py:59`), which is
  documented pure/synchronous and is wired into the production `add_memory` path at
  `tools.py:1257` — a *wired*, not merely declared, capability. This is what keeps η
  from growing a second similarity rule (INV-5).

## Amendment 2026-07-29 (dashboard-PRD commission)

The dashboard PRD's authoring session (`docs/prds/memory-eval-dashboard.md`) surfaced
two seam gaps in ε and resolved them by amendment while 3211 was still
pending/unclaimed (M2/M3 amendment bullets in the PRD; 3211's description + two new
sidecar capability rows updated in the same commit):

- **`verdicts-artifact-persisted`** — the evaluator's per-metric verdicts (alarm /
  no_alarm / insufficient_data / grandfathered) were computed but never persisted
  machine-readably, which would have forced the dashboard to re-run statistics
  dashboard-side (a G6 failure) or scrape the human report (INV-2). ε's runner now
  persists `verdicts-<STAMP>.json` + `verdicts-current.json` and commits an exemplar
  under `shared/tests/fixtures/`.
- **`escalation-carries-fingerprint`** — the M3 fingerprint had no pinned structured
  carrier on the filed escalation. Now `dedupe_fingerprint` (existing model field)
  carries the exact verdict fingerprint string; parity is string equality, never
  format parsing.

Both rows carry mechanical `delivered_check`s and were mirrored into 3211's
`metadata.delivered_checks` via `update_task(metadata_mode='additive')`.

## Re-walk spot-check: §6 row confirmed clean

One §6 row was re-checked at the re-walk and found accurate — no drift:

- **`with_vectors` on `scroll_by_metadata`.** §6 reads "`with_vectors` not yet passed —
  δ adds it". Confirmed on main: `scroll_by_metadata` (`backends/mem0_client.py:341-424`)
  passes no `with_vectors` argument to `client.scroll` at all, so δ genuinely
  introduces the parameter rather than widening an existing literal. The module's only
  `with_vectors=False` literal belongs to a different method — `get_point_by_id`'s
  `client.retrieve` at `backends/mem0_client.py:454` — which δ does not touch. An
  earlier pass had it backwards, claiming §6 was imprecise; that claim has been
  corrected on δ's binding in this manifest (task 3359). Task 3210's task text may
  still carry the old wording, which is out of scope here.

All other spot-checked rows verified clean on main: unvalidated `category: str`
(`escalation/models.py:93`) ⇒ `eval_regression` filable; direct `EscalationQueue`
filing (`backfill_recon_escalations.py:68-71`); `has_open_l1` dedup precedent
(`stage1_stall_detector.py:385,483`); recon queue at 8103 with
`escalation_queue_dir` defaulted under `data/reconciliation/escalations`; the
flag-marker `{sh,service,timer}` + installer + `-check.sh` quintet; the `STAMP`
artifact idiom (`cgl_eta_auto_apply_impl.py:45-46`); `query_points`
(`task_curator.py:1890`); the count-only `result_summary` and `query[:200]`
truncation (`tools.py:1441-1447`); the unlogged hint path
(`context_assembler.py`); prune precedent (`write_journal.py:494` →
`server/main.py:560`); transcript readers (`digest.py:32`, `inventory.py:81`) over a
populated `data/orchestrator/agent-transcripts/`; `fused-memory/pyproject.toml`
carrying no orchestrator dependency and `shared/` carrying no stats module.

## Gate outcomes

- **G1** — every mechanism has a named consumer: M1/M2 → ε's evaluator + the
  commissioned dashboard PRD (artifact-only read surface); M3 → the live
  recon-escalation-watcher (`recon-watch/run.sh` → 8103, sole closer); M4 → η plus
  operators. θ, whose downstream shadow-replay/E3/E8 runners are explicitly future
  work (§7), is justified by its own operator-observable artifact + coverage report and
  by η's use of the corpus as a validation set — not by an unfiled PRD.
- **G2** — α–ζ are **intermediates** (each has in-batch consumers) and each names them;
  η and θ are the **leaves**. Every task additionally carries a user-observable signal,
  which exceeds what G2 asks of an intermediate.
- **G6** — re-checked leaf-by-leaf and the PRD's G6-clean authoring is **preserved**:
  no signal asserts an a-priori numeric limit. The only numerals that appear are
  parameterisations and config inputs, never achievement bounds — `k=5,10` (which
  top-k the canonical-presence metric is computed at), the false-alarm budget (α's
  *input*, from which per-test significance is *derived*), and the storm cap `K`
  (config, no default asserted). δ's fixture clause explicitly requires cosine **and**
  lexical ratio to be *measured at authoring*.
- **G7** — all eight tasks walked against INV-1..5. **No waivers**, matching the PRD's
  §5 anticipation. Most invariants are actively satisfied by the PRD's own decisions
  (D2/D6/D7/D9 are INV-5 resolutions; M3's storm escape and runner-failure
  self-escalation are INV-4/INV-2; M3's pre-filing pending-queue re-read and D1's
  grandfather re-read are INV-3). Three hits were resolved **by redesign, folded into
  the filed task text** rather than waived:
  1. **ζ / INV-4** — `_log_read`'s pre-existing catch-and-warn swallow
     (`tools.py:711`) becomes load-bearing once η computes a metric from these rows;
     ζ must count and surface dropped journal writes.
  2. **ζ / INV-5** — the `_MEMORY_INSTRUCTIONS` sentence naming the new caller params
     is hand-transcribed prose that must agree with the tool signature (the survey's
     "prompt text drifted twice in one file" shape); ζ carries a one-line pinning
     assertion, composing with 3202's registry↔prompt drift test rather than
     duplicating it.
  3. **θ / INV-2+INV-4** — "0 searches extracted" is ambiguous between an empty
     archive and total parse failure; θ must make the two distinguishable.
- **G4** — seam ownership in §8 is one-directional throughout; no reciprocal-ownership
  ambiguity. One bookkeeping gap: §8 claims the δ↔**3136** detector/scheduling split
  carries an "ordering note in both", but 3136's record carries no note about this
  PRD (its 2026-07-29 amendment is from the vocabulary PRD). Not blocking (G4 is
  prompt-level, and ownership itself is unambiguous); the note is carried in δ's task
  text instead, and amending 3136 was deliberately left out of scope for this batch.
- **G5** — B+H confirmed. ε is the integration-gate task and its signal names M3's
  boundary-test sketch directly; α's names M1/M2's.
