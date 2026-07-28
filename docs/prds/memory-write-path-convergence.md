# PRD: Memory write-path convergence — server-side write triage, transactional consolidation, decay & boundary hygiene

**Project:** dark-factory (fused-memory + orchestrator). **Status:** active, 2026-07-28. **Approach:** B+H (contracts + two-way boundary tests).
**Origin:** reify curator session 2026-07-27 (19 `milestone_gate` L2 escalations; 90 Mem0 deletions, 21 canonicals) + the forensic RCA at `/home/leo/.claude/fleet/sessions/review-reify-1693879/result.md`. This PRD implements that RCA's 10 recommendations, with recs 1–2 amended per the 2026-07-28 design review (server-side dedup, cosine-as-candidate-generator + LLM judge, non-destructive attach).
**Siblings:** DF **3083** (XML-leak root-cause + Qdrant payload text-match tool + corpus sweep — pending, owns that territory; leaf ο here is containment only). DF **3055** (in-place Mem0 update decision — in-progress; this PRD is deliberately add-only-compatible and takes no dependency on its outcome).

## 1. Goal (G1 consumer + user-observable surface)

The Mem0 corpus **converges instead of accreting**: an agent rediscovering a known gotcha lands its novel fragment on the canonical instead of minting duplicate #9, consolidation passes cannot strand or ratchet, and stale/corrupt entries are flagged mechanically instead of by heroic curator sessions. Observable surfaces:

- **`add_memory` writers** (every `claude-task-*` agent, interactive sessions, recon stages) get a structured routing ack — `{routed: stored | restated | amended | contested, canonical_id?}` — instead of a silent store. Writers' briefings say "write freely; the server deduplicates" and nothing about pre-searching.
- **Readers** (`search`, `get_memory_by_id`) see one grouped document per topic: canonical + amendment digests + sighting count — instead of N competing near-duplicates ranked by embedding luck. Search results carry full provenance (`agent_id`, `task_id`, `created_at`).
- **Stage-1 consolidation** runs through one transactional MCP op that proves closure (returns post-delete survivors); a failed pass can no longer net +1 entry. `delete_memory` on a truncated id fails loudly instead of `{status: deleted}`-no-op'ing.
- **Operators** get a scheduled deterministic duplicate-cluster report; consolidation gates are filed from that report, with the LLM demoted from detector to adjudicator. Resolved consolidations auto-seed the topic-cluster guard — no manual hop.
- **Boundary**: a write whose content contains raw MCP envelope markup (`</content>`, `<parameter name=`, `</invoke>`) is rejected with a structured error naming the pattern; an episode claiming "fix applied" against a non-matching task/git state is tagged unverified.
- **Decay**: flipping a task terminal flags memories that cite that task as a blocker/limitation for re-verification; gotcha entries can carry machine-checkable `reexamine_when` conditions.

## 2. Background — evidence (why this PRD exists)

Full RCA: `review-reify-1693879/result.md` (this section is a pointer, not a restatement — INV-5). Load-bearing facts, all verified against source/git on 2026-07-27/28:

- 89 defective entries deleted by the curator were written by a long tail of ~40 one-shot task agents (1 each), **17 by `recon-stage-memory_consolidator` itself**, 8 interactive. 35 of 89 written in the 2 days before curation — accretion outruns gate-clearing.
- The write guards could not have worked: cosine guard live 2026-07-14, topic guard 2026-07-20 (half the entries predate them); default threshold 0.92 vs a measured genuine-rediscovery pair at **0.824** (paraphrases with novel fragments, not verbatim dups); the seeded topic-cluster list is 5 dark-factory-only topics fed by a manual human hop that no reify topic ever got; guards compare same-category only; `allow_near_duplicate` appeared in zero defective entries (no bypass); kill-switch always default-True.
- Stage-1 ratchet: guard-exempt canonical write + duplicate deletes with no transaction/ordering/verification, and `delete_memory` **silently no-ops on 8-char hex prefixes** (`prompts/stage1.py:110-113` admits prompt warnings "have not prevented" it). Failed pass ⇒ net +1. esc-5541's canonical claimed closure while 8 survivors persisted through five wrong enumerations.
- The one deterministic dedup tool (`fused-memory/scripts/audit_duplicate_memories.py`, 2026-07-14) has no timer and no caller. Cluster detection is LLM prose (`stage1.py:55`).
- 5 of 19 gates were staleness (code moved under the memory; every falsifying event was machine-visible in task store/git). 9 confirmed XML-leak instances, all found by eye (sweep tooling owned by 3083). 1 fabricated "fix applied" episode. Stage-1's injection heuristic flagged its own output (never checked `agent_id`); `search` results strip `agent_id`.

## 3. Sketch of approach

Five thrusts, decomposed as chains (§9):

- **A — write triage** (recs 2, 1, 3-as-auto-seed): server-side routing at the `add_memory` chokepoint. Cosine is candidate generator only; band split with calibrated thresholds; haiku-class judge with closed outputs for the middle band; non-destructive attach (child records); server-side grouped reads; then invert the writer instruction.
- **B — consolidation integrity** (recs 4, 5, 7): UUID-strict deletes; one transactional consolidate op with server-side closure proof and topic-cluster auto-seeding; Stage-1 rewired onto it; the deterministic audit script scheduled and feeding gate filing.
- **C — provenance** (rec 8): full metadata in search results; injection heuristic checks `agent_id` first.
- **D — decay** (rec 9): `reexamine_when` metadata schema; task-terminal invalidation hook.
- **E — boundary hygiene** (rec 10): MCP-markup write tripwire; completion-claim verification gate on episodes.

## 4. Contracts (H)

### C1 — Write-triage contract (thrust A)

> **Every `add_memory` write is triaged at the server; triage may re-route a write but may never lose content, block a write, or edit a canonical.**

- Pipeline: candidate retrieval (top-k across **all three Mem0 categories**, topic-tag-aware) → band decision → outcome ∈ `{stored, restated(canonical_id), amended(canonical_id), contested(canonical_id)}`, returned in the ack (machine-readable schema — INV-1).
- Bands: similarity ≥ `T_high` ⇒ deterministic `restated`; `T_low ≤ s < T_high` ⇒ LLM judge (closed 4-way output); `s < T_high` with a topic-cluster hit still goes to the judge; else `stored`. **`T_high`/`T_low` are calibration outputs of leaf α — no number in this PRD or any leaf signal asserts them a priori** (G6).
- Judge: haiku-class, input = new entry + top 3–5 candidates + closed schema (≈2.5k tokens), no repo/task context. The judge **detects** contradictions (`contested` = attach + flag for the existing gate machinery); it never adjudicates them.
- Attach outcomes are **non-destructive**: an amendment/sighting is a child record with its own UUID and `metadata.parent_id`; canonical text is never auto-edited; a wrong attach is re-parentable. Text-level synthesis remains a curated activity.
- **Fail-open + storm escape (INV-4)**: any triage/judge failure ⇒ `stored` (a write is never blocked or errored by triage), AND every fail-open event increments a counter; a streak/rate threshold raises a structured escalation — a judge outage must not silently degrade every write to today's behavior.
- Rollout: behind `write_triage_enabled` (default off). The existing cosine/topic **reject** guards retire when triage enables (redirect supersedes reject); `allow_near_duplicate: True` is reinterpreted as force-`stored`.
- Two-way boundary tests: server side (each band routes correctly, fail-open fires, storm counter escalates, canonical never mutated) and writer side (ack schema stable; a rejected-in-old-world duplicate now yields `restated` + no new standalone entry).

### C2 — Consolidation contract (thrust B)

> **`consolidate_memories(canonical_content, supersedes=[full UUIDs], topic)` is the only sanctioned multi-entry merge path, and it proves closure.**

- Validates every supersede id as a full UUID (η's hard-error backs this); writes the canonical (topic-tagged); deletes supersedes; **re-runs the discovery query (topic + similarity) post-delete and returns survivors** — the caller sees `survivors: []` or a concrete leftover list, never a claimed closure (INV-2, INV-3).
- Partial failure is reported structurally (which deletes failed); the op is add-only-compatible (no in-place update assumed — 3055 seam).
- **Auto-seeds the topic guard**: derives a `ProceduralTopicCluster` (phrases from the merged texts, hint = canonical UUID) and persists it to a runtime cluster store merged with the config-seeded list — killing the manual hop, at the chokepoint that holds exactly the right inputs.
- Once Stage-1 uses this op, the blanket `recon-stage-*` guard exemption retires.
- Ordering note (resolved decision D6): Stage-1's rewire is **not** hard-ordered behind 3083's corpus sweep; 3083's own description records "run sweep before further large consolidation passes" and the operator sequences that. Leaf ο's tripwire stops new specimens regardless.

### C3 — Boundary contract (thrust E)

> **Content crossing the MCP write surface (`add_memory`, `add_episode`, `submit_task`, `update_task`) that contains raw MCP envelope markup is rejected with a structured error naming the matched pattern.**

- Containment only; the cure (root cause + retroactive sweep) is 3083's. Rejections increment a storm counter (INV-4): a rejection burst = the leak is active ⇒ escalate, don't just bounce writes.

### C4 — Decay contract (thrust D)

> **A memory's validity conditions are machine-checkable where declared, and task-terminal events are joined against the corpus.**

- `metadata.reexamine_when` schema: `{task_terminal: <id>}` and/or `{file_changed: <path>}`; validated at write; documented in the writer instruction site (single normative copy — INV-5).
- On a task reaching terminal status, reconciliation re-corroborates against the live task store (INV-3) then flags — never deletes — memories whose `reexamine_when` names that task **or** whose content cites it as a blocker/limitation; flags feed the existing gate machinery.

## 5. Resolved design decisions

- **D1** Cosine is a candidate generator, never a judge. Bands calibrated from the labeled 89-entry curator dataset (retrieval recall@k + judge accuracy vs curator dispositions); no uncalibrated numeric floor appears in any leaf signal (G6).
- **D2** Redirect supersedes reject: the 0.92 reject guard retires when triage enables; `allow_near_duplicate` becomes the force-store escape hatch.
- **D3** Judge is sync-in-`add_memory`, fail-open, closed outputs, detect-not-adjudicate.
- **D4** Amendments/sightings are add-only child records (3055-independent); reads are **server-side grouped** (canonical + digests + sighting count) — the reader is the consumer, the slice ends there.
- **D5** Topic-cluster auto-seeding lives inside `consolidate_memories` (not escalation plumbing): the op holds canonical + superseded texts, exactly the derivation input. Runtime store merged with config seeds; `reload_config` unaffected.
- **D6** No hard dep edge from the Stage-1 rewire onto 3083's sweep (operator decision 2026-07-28); soft ordering note in C2.
- **D7** roles.py inversion (ε) lands only after triage is live end-to-end (deps γ): capture stays eager; dedup is the server's job. The instruction gets *simpler*, not longer.
- **D8** `audit_duplicate_memories.py` scheduled via systemd user timer (precedent: `fused-memory-flag-marker-sweep.timer`); its deterministic report becomes the consolidation-gate filing input; LLM demoted to adjudicator.
- **D9** Rediscovery count = count of sighting children, computed at read; consumers: grouped read (D4) and κ's report ranking (which topics aren't reaching agents through briefings).
- **D10** Staged rollout: `write_triage_enabled` default off; flipped on when γ's boundary tests pass; ε lands after.

## 6. Pre-conditions / substrate (G3 — verified 2026-07-27/28 against live source)

| Assumed capability | Evidence |
|---|---|
| Writer instruction site exists | `orchestrator/src/orchestrator/agents/roles.py:218-250` `_MEMORY_INSTRUCTIONS` (all task roles) |
| Cosine guard + threshold resolution + kill-switch | `fused-memory/src/fused_memory/server/near_duplicate_guard.py` (threshold default `:30`, category/source filter `:79-81`) |
| Topic-cluster model + seeded list + manual-hop admission | `fused-memory/src/fused_memory/config/schema.py:377-570`; `reload_config` MCP tool live |
| Deterministic dedup script exists (unscheduled) | `fused-memory/scripts/audit_duplicate_memories.py` (added dfdbcc32e6 2026-07-14; no unit/timer/caller) |
| systemd user-timer precedent | `~/.config/systemd/user/fused-memory-flag-marker-sweep.{service,timer}` |
| LLM plumbing reachable from server write path | `add_memory` auto-classification (category=None) + `add_episode` extraction pipeline |
| Metadata-filtered lookup for parent/child | `get_memories_by_metadata` / `count_memories_by_metadata` (exact-match, live) |
| The defect η fixes is real | `reconciliation/prompts/stage1.py:110-113` (silent prefix no-op, "recurrent failure") |
| Task-terminal reconciliation trigger | `set_task_status` triggers reconciliation (MCP server instructions; interceptor events) |
| Labeled calibration raw source | curator transcript `/home/leo/.claude/projects/-home-leo-src-reify/bceaf4a6-d79e-44f3-8422-b152906f70cb.jsonl` (89 entry payloads + dispositions); 19 escalation resolutions (escalation store) |
| Stage-1 consolidation call sites | `fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py`; prompt `prompts/stage1.py:55,65-72` |

## 7. Out of scope

- XML-leak root cause, Qdrant payload text-match read tool, retroactive corpus sweep — **DF 3083**.
- In-place Mem0 update / metadata-patch tool decision — **DF 3055**. (If 3055 lands a tool, a follow-up may upgrade θ's closure-stamping; nothing here waits on it.)
- reify-repo guidance edits (CLAUDE.md, memory docs) — post-ε follow-up in the reify repo, not this batch.
- Graphiti-side dedup; retroactive re-categorization sweeps; changes to the `add_episode` extraction pipeline beyond π's completion-claim gate.

## 8. Cross-PRD / seam ownership (G4)

| Seam | Owner | This PRD's edge |
|---|---|---|
| XML-leak cure (root-cause, text-match, sweep) | DF 3083 (pending) | ο = containment tripwire only; C3 storm counter names 3083 in its escalation hint |
| In-place update primitive | DF 3055 (in-progress) | none — add-only design throughout |
| Consolidation-gate filing machinery (`milestone_gate`) | existing orchestrator/recon (deterministic_runner.py:1433, recon_self_model.py:552) | κ changes only the *input* (deterministic report), not the filing mechanism |
| Writer briefings (roles) | orchestrator repo (same batch, leaf ε) | ε edits `roles.py` only; no reify-side text |
| Task-metadata citation repointing on consolidation deletes | DF 3108 (in-progress, from reify esc-5710-1) | θ's op does **not** claim citation repointing; 3108's sweep remains valid whether deletes go through θ or not |
| Prompt-level UUID-resolution discipline (failed predecessor of η) | DF 1144 (done 2026-05-09; documentedly insufficient per stage1.py:113) | η is the API-level enforcement 1144 could not provide |
| G7 invariants doc | `docs/legibility/design-invariants.md` (single normative copy) | contracts cite slugs, no restatement |

## 9. Decomposition plan (one bullet per leaf; signals are the G2 gate)

Deps: β←{α, δ}; γ←β; ε←γ; θ←η; ι←θ; ζ←θ; ξ←ν. All others independent.

- **α — calibration dataset + eval:** commit the labeled fixture (89 entries: content, category, ground-truth cluster, curator disposition) extracted from the curator transcript + `fused-memory/scripts/calibrate_write_triage.py`; running it emits a report (similarity distributions for true-dup vs unrelated pairs, retrieval recall@k) and writes chosen `T_high`/`T_low` into config with the report path recorded. *Signal:* fixture + script committed; `calibrate_write_triage.py` run produces the report artifact and the config values are traceable to it.
- **β — triage skeleton (deterministic bands):** candidate retrieval (cross-category, topic-tag-aware) + band routing in `add_memory` behind `write_triage_enabled`; judge slot stubbed fail-open (`stored`); structured ack schema; storm counter for fail-open events. *Signal:* with the flag on, a near-verbatim dup write returns `{routed: restated, canonical_id}` and creates no standalone entry; with LLM path failing, write returns `stored` and the counter increments (tests).
- **γ — judge integration:** closed-4-way judge call for the middle band; judge-accuracy eval vs α's labels committed as a report; `contested` routes to attach+flag; boundary tests both sides (C1). *Signal:* middle-band fixture write routes per judge verdict; eval report exists with per-class accuracy; canonical-never-mutated test passes.
- **δ — amendment representation + grouped reads:** child-record schema (`parent_id`, kind ∈ amendment|sighting); `search`/`get_memory_by_id` return canonical with amendment digests + sighting count inlined. *Signal:* reading a canonical with 2 amendments + 3 sightings returns one grouped document showing both digests and count 3 (test).
- **ε — writer-instruction inversion:** `roles.py` memory section says "write freely; the server deduplicates — don't pre-check" and documents the ack meanings; no pre-search instruction remains. *Signal:* rendered architect/implementer briefing contains the new sentence and not the old write-eager-only text (briefing render test).
- **ζ — topic-cluster auto-seed:** `consolidate_memories` derives + persists a runtime `ProceduralTopicCluster` (phrases from merged texts, hint = canonical UUID); runtime store merges with config seeds at guard/triage read. *Signal:* consolidating a fixture cluster yields a persisted cluster that a probe write then matches (test).
- **η — UUID-strict delete:** `delete_memory` hard-errors on non-full-UUID ids; regression test extends `test_delete_memory_truncated_uuid.py`; stage1 prompt line updated to state the hard error (INV-5: points at the enforcement, doesn't restate). *Signal:* `delete_memory('8charhex')` returns a structured error, not `{status: deleted}` (test).
- **θ — transactional consolidate op:** `consolidate_memories` MCP tool per C2 (validate → write → delete → re-query → return survivors; structured partial-failure). *Signal:* consolidating a fixture cluster returns `survivors: []`; injecting a failing delete returns that id in the failure list and the survivor list (two-way tests).
- **ι — Stage-1 rewire + exemption retirement:** Stage-1 consolidation instructions/stage call θ's op; blanket `recon-stage-*` exemption removed from the guard path. *Signal:* Stage-1 integration fixture consolidates via the op; grep shows no `recon-stage` exemption in `near_duplicate_guard.py`; a recon-agent direct near-dup `add_memory` now triages like anyone else (test).
- **κ — scheduled deterministic dedup report:** systemd user timer + service running `audit_duplicate_memories.py` on cadence; report artifact per run; gate filing consumes the report (Stage-2/gate path reads clusters from it rather than LLM enumeration). *Signal:* timer unit files committed + `systemctl --user list-timers` shows it; a run produces the report; a filed consolidation gate's detail cites report clusters (integration fixture).
- **λ — full search provenance:** `search` result payloads include `agent_id`, `task_id`, `created_at`. *Signal:* search over a fixture entry returns those fields populated (test).
- **μ — injection-heuristic provenance check:** Stage-1's possible-injection flag requires an `agent_id` mismatch before firing. *Signal:* fixture entry authored by `recon-stage-memory_consolidator` is not flagged by the heuristic; a genuinely foreign-id fixture still is (test).
- **ν — `reexamine_when` schema:** metadata schema validated at write (`task_terminal`, `file_changed`); documented in the single writer-instruction site. *Signal:* `add_memory` with a valid `reexamine_when` accepts; malformed shape rejects with hint (tests).
- **ξ — task-terminal invalidation hook:** on terminal transition, reconciliation re-corroborates live status then flags memories matching `reexamine_when.task_terminal` or citing the task as blocker/limitation; flags feed existing gate machinery; never deletes. *Signal:* flipping a fixture task to done produces a flag on the citing memory (integration test).
- **ο — MCP-markup tripwire:** C3 rejection on `add_memory`/`add_episode`/`submit_task`/`update_task` content fields; storm counter + escalation hint naming 3083. *Signal:* `add_memory` content containing `</invoke>` rejects with the pattern named; `submit_task` description with `<parameter name=` rejects loudly instead of silently mis-parsing (tests).
- **π — completion-claim gate on episodes:** extraction pipeline detects applied-work claims ("fix applied", "de-flake landed") and verifies against task/git state, tagging unverified claims (extends the terminal-state pre-check discipline). *Signal:* fixture episode claiming a fix on a not-done task is ingested tagged `unverified_claim` and flagged (test).

## 10. Open questions (tactical, implementation-time)

- Judge prompt wording and few-shot picks (bounded by γ's eval — accuracy report is the arbiter).
- Phrase-derivation algorithm for ζ (deterministic n-gram vs judge-suggested-then-validated); runtime cluster store location (file vs DB) — must survive restart, merge cleanly with config seeds.
- κ report format and cadence; whether the report also carries D9 sighting-count rankings from day one.
- Exact ack field names / error codes (schema'd in β; INV-1 requires they live where callers see them).
- ξ's citation heuristic breadth for legacy entries lacking `reexamine_when` (task-id regex vs semantic) — start narrow (explicit `#id` / "task NNNN" citation), widen from flag precision data.
