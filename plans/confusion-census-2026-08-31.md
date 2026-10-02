# confusion census 2026-08-31

Project: dark_factory

## Saturation

- batches: 2
- stop reason: saturated
  - batch 0: dup_rate=1.00 (total=20, succeeded=19, failed=1, saturated=True)
  - batch 1: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | implement |
| --- | --- |
| unknown | 1 |

## Synthesis

All grounding is done. The finding is verified against the harness contract itself, discriminated against five neighboring codebook mechanisms, and both carried-forward method notes from 08-26 resolved (08-21 R1 lives as pending task 3606's Cluster B; 08-26 R1 landed as task 4751). One anomaly noted for the runner: a completed zero-cluster report dated today already exists on disk. Here is the synthesis document for the runner:

---

# Confusion census — 2026-08-31

**Date:** 2026-08-31
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests → per-finding verification against current main (Sonnet) → this synthesis (Fable). The one finding restated below survived the verification stage; this synthesis adds context-reading against the current tree, the live harness tool surface, and the codebook only — no diagnosis appears here that was not itself verified.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml` (dispositions in §5 are inputs to the merger, which promotes/rejects in place).
**Run notes:** eighth completed periodic census, at the PRD's 5-day hard floor after the previous one. Previous: 2026-08-26 (1 verified finding), 08-21 (2), 08-16 (3), 08-10 (1), 08-05 (0), 07-31 (15 findings / 4 clusters + 1 one-off). One anomaly of record: at synthesis time a *completed* `plans/confusion-census-2026-08-31.md` already exists on disk recording **zero** verified clusters (`sonnet verify=0`, `fable synthesis=1`), and `census-state.json` has already advanced to it — this synthesis, invoked with one verified cluster, belongs to a distinct run bearing the same date. The runner owns reconciling the dated report file; this document states the fact and does not guess at the cause. Saturation statistics and filed-task ids are appended by the census runner outside this synthesis.

## Corpus

- **1 verified finding, 1 session** (7dae04c6), one sighting. Area: `tool-contract`.
- Composition: a **harness-rooted** finding — the defect is in agent tool-call construction against the Claude Code harness's Edit contract, upstream of the repo, the same class as the 08-21 census's finding 1.1 (Agent-tool call missing required `description`). No in-repo code path produces or can intercept it.
- Session 7dae04c6 has no prior codebook presence, and neither `replace_all` nor the Edit multi-match error text appears anywhere in the codebook — the mechanism is novel to the registry.
- Phase-stamp coverage: **1 of 2 stamps is `unknown`** (origin unknown, manifested `implement`), ending the two-cycle full-coverage streak (08-21: 0/4; 08-26: 0/2). The unknown is reported, not guessed, per the PRD's standing instruction: the evidence pins where the misparse manifested (turn 70) but contains nothing upstream to pin an origin.
- A one-finding corpus supports verification and placement, not trend claims. Continuity notes below are labeled as observations over a minimal base.

## Executive summary (observations)

1. **Edit's multi-match error hint was invoked as a nonexistent standalone `replace_all` tool.** Session 7dae04c6, turn 69: an Edit call failed on an ambiguous `old_string` with `Found 3 matches of the string to replace, but replace_all is false...` — a message whose remedy clause names the fix as "set replace_all to true". Turn 70: the agent called a bare tool literally named `replace_all` with an empty argument object, which the harness rejected with `Error: No such tool available: replace_all`. The agent then abandoned the parameter remedy entirely and fell back to disambiguating the string manually. The verified cost is one wasted turn; the remedy hint named a real parameter of the very tool that errored, and the agent bound the bare identifier to the wrong syntactic category — a callable tool rather than a parameter on a re-issued Edit call.

## Origin × manifestation matrix

Rows = `origin_phase`, columns = `manifested_phase`. Counts are verified sightings (1 total). `merge` and `verify` kept explicitly to show their zeros; the `unknown` row is a stamp reported as such, not a synthesis guess.

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| unknown | · | · | 1 | · | · | · | · | · | · | **1** |
| **total** | **0** | **0** | **1** | **0** | **0** | **0** | **0** | **0** | **0** | **1** |

Readings (observational): no merge- or verify-manifested sighting for a **seventh consecutive cycle**; the six corpora since 08-05 total 9 findings, so the absence still carries limited evidence on the PRD's motivating architect/implement→merge hypothesis. This is the first `implement`-manifested verified finding since the 08-16 cycle's corpus.

## 1. Verified clusters

### 1.1 Edit's multi-match remedy hint invoked as a standalone `replace_all` tool (1 sighting, session 7dae04c6)

The trace is fully contained in two adjacent turns. Turn 69: `Edit(...)` → `<tool_use_error>Found 3 matches of the string to replace, but replace_all is false...`. Turn 70: `replace_all({})` → `<tool_use_error>Error: No such tool available: replace_all`. The agent then recovered by disambiguating the `old_string` manually — itself a documented remedy for the multi-match condition, so the terminal state is a successful edit; the established cost is the one wasted turn and the abandonment of the parameter route. Whether `replace_all: true` would have been the semantically correct fix for that particular edit (replacing all three occurrences) is not established by the evidence and is not claimed.

**Verified mechanism (checked against the live harness at synthesis time):** the Edit tool's schema carries `replace_all` as an optional boolean parameter, default false, documented as "Replace all occurrences of old_string"; the schema's own description names the non-unique-match failure and the `replace_all: true` alternative. No tool named `replace_all` exists anywhere on the harness surface — neither among the always-loaded tools nor on the deferred-tool list — verified against this synthesis session's own harness, which is the same Claude Agent SDK surface fleet agents run. The error's remedy clause names the parameter bare ("set replace_all to true") without syntactically anchoring it to the Edit call that produced the error, and the sighting shows the bare identifier bound to the wrong category. The empty argument object is consistent with that reading: invoked as a "tool", `replace_all` carries no way to name which edit to redo — nothing from the failed call was carried over.

**Blast surface (structural facts, no frequency claim from one sighting):** Edit is universal to implement-phase agents, and the triggering condition — a non-unique `old_string` — is common enough to have already produced a catalogued cascade under a *different* mechanism (`entry-cand-20260721-6`, where an oversized test file forces blind Edit calls that "collide on multiple near-identical matches"). The failure is client-side and self-contained: one rejected call, one turn, no durable state touched — the same tool-call-boundary containment the 08-21 census observed for both its findings.

**Relation to the codebook (observation, not a merge):** this is **not** a sighting of any of the four adjacent items. Not `entry-cand-20260721-1` (deferred tool called before ToolSearch discovery): Edit is not deferred, its schema was demonstrably in context — the turn-69 call reached the tool and failed on content — and the turn-70 rejection is the flat "No such tool available" for a tool absent from the harness entirely, not an `InputValidationError` against a withheld schema. Not `cand-20260720-6` (architect invents `report_task_already_done`; disposition `rejected`): the surface signature matches — a minted tool name drawing "No such tool available" — but the invention route differs: there the agent coined a name from its own intent to fill a genuinely missing affordance; here the name was lifted verbatim from harness error text, and the needed affordance exists one parameter away. Not `cand-20260812-15` / 08-21 finding 1.1 (missing-required-parameter class): no field was omitted from a real tool's call — the tool name itself was wrong. The nearest relative is `cand-20260812-18` (mcp_markup_middleware hints instruct `metadata={'allow_mcp_markup': True}` on tools that have no `metadata` parameter): both are members of a remedy-text family — printed remedial instructions in tool results treated as executable — with inverse polarity. There the hint is wrong and following it faithfully fails; here the hint is correct and the failure is in parsing it. Together they bracket the failure mode: remedy text is followed literally, so both its accuracy and its syntactic anchoring are load-bearing.

## 2. One-off sightings

None beyond the cluster above (itself single-sighting this cycle).

## 3. Cross-cutting observations

1. **"No such tool available" now discriminates at least five catalogued mechanisms:** deferred-tool-before-discovery (`entry-cand-20260721-1`); the one-shot-subagent prose-contract shims (`grep_search`/`read_file`/`verification_complete` named in verifier contracts the runtime doesn't provide); capability-envelope rediscovery (Write/Edit "exists but is not enabled in this context" in watcher rotations); missing-affordance invention (`cand-20260720-6`); and now remedy-hint misparse. This extends the 08-21 census's method note — that the "required parameter missing" string spans two mechanisms and must not be routed by string match — to this string as well: one error text, five mechanisms, and only surrounding context discriminates them.
2. **Remedy text in tool results is a confusion vector in both directions.** `cand-20260812-18` documents a wrong hint followed faithfully; this finding documents a right hint parsed wrongly. In both, the agent's first move after an error is to execute the printed remedy literally — an observation about where harness error-message wording carries real weight.
3. **The harness-rooted class recurs.** Like 08-21's finding 1.1, the defect sits in tool-call construction upstream of the repo; the codebook remains the only in-repo surface that can hold it.
4. **Phase-stamp coverage regressed** to 1-of-2 unknown after two consecutive full-coverage cycles — on a minimal base, and the unknown is the honest stamp for an origin the evidence cannot pin.

## 4. Remediation candidates

None filed this cycle, following the 08-21 precedent for its finding 1.1: the defect is in harness-side tool-call construction, upstream of the repo — no in-repo code path produces or intercepts it, and unlike 08-21's finding 1.2 there is no in-repo skill or prompt surface that shapes the erroring behavior either (the exchange is entirely between the agent and the harness's own Edit contract). The codebook action (§5) is the accrual anchor; a single sighting does not yet warrant an upstream harness report.

## 5. Codebook dispositions (input to the merger; promote/reject in place, never delete)

| Cluster / finding | Suggested disposition |
|---|---|
| 1.1 Edit multi-match remedy hint invoked as standalone `replace_all` tool | Mint a new pending candidate (suggested `cand-20260831-1`; area `tool-contract`, first_seen 2026-08-31, session 7dae04c6, origin `unknown` / manifested `implement`). Record three discriminators in its evidence notes so future coders don't route by the shared error string: not deferred-discovery (tool absent from the harness entirely, no withheld schema, no `InputValidationError`); not missing-affordance invention (the affordance exists as a parameter; the name came from error text, not from intent); not missing-required-parameter (the tool name was wrong, no field was omitted). Cross-reference `cand-20260812-18` as the inverse-polarity relative in the remedy-text family. Explicitly **not** a sighting of `entry-cand-20260721-6` — the Edit multi-match error is the shared trigger, but that entry's mechanism is a file-oversize cascade and its sessions responded with grep/blind-edit churn, not a phantom tool call |

## 6. Method notes for the next census

- **Discharged from 08-26's carry-forwards, with positive answers:** (a) the 08-21 census's R1 (rejected-call clause in the watcher skills) was **folded, not dropped** — task **3606** ("Census R2+STOP: watcher skill+banner must branch on WATCHER_REARM_OUTCOME and honor in-tool-result STOP directives — never re-issue an identical failed or rejected bounded wait", status `pending`, updated 2026-08-21) carries that exact scope as its Cluster B; consistent with `pending`, a grep of both watcher skills on current main finds no rejected-call clause in the re-arm loops yet. Treat 08-21 cluster 1.2 as filed-but-not-landed. (b) The 08-26 census's R1 **landed as task 4751**: the PRD's §7.2.1 now documents the retry-loop designed-outcome annotation with the 08-26 census named as finding of record. The digest-level confirmation — healthy watcher rotations ceasing to mint rearm-churn candidates — is still the nightly trickle's to demonstrate.
- **Still open from 08-26:** the stray-comma Read candidates `cand-20260806-12` and `cand-20260812-19` remain separate and both `pending` — not yet consolidated. Also observed in passing: the merger catalogued the 08-26 finding under `cand-20260824-28` (now promoted) rather than folding `cand-20260819-1`, which still pends separately with its founding sighting; the two facets of the retry-loop/designed-outcome join remain unconsolidated.
- **New watches:** whether other harness error texts that name a parameter bare (without anchoring it to the erroring call) draw the same tool-category misparse; and whether the five-way "No such tool available" discrimination survives trickle coding — the string is a magnet for string-match routing, which observation §3.1 says is now wrong in at least four out of five cases.

---

*Synthesis note to the runner: written from the 1 verified finding supplied. Mechanism claims verified against the live harness tool surface at synthesis time (Edit schema's `replace_all` boolean parameter and multi-match contract; absence of any standalone `replace_all` tool from both the loaded and deferred tool lists). Codebook relations verified by reading `docs/legibility/confusion-codebook.yaml` on current main (`entry-cand-20260721-1`, `cand-20260720-6`, `cand-20260812-15`, `cand-20260812-18`, `entry-cand-20260721-6`, `cand-20260824-28`, `cand-20260819-1`, `cand-20260806-12`, `cand-20260812-19`; session 7dae04c6 confirmed absent). Method-note claims verified via `plans/confusion-reduction-prd.md` §7.2.1 (task 4751), fused-memory `search_tasks` (task 3606, pending), and grep of both watcher skills. No tasks filed and no codebook edits made from this synthesis — filing and merger application are the runner's steps. Flag for the runner: a completed zero-cluster report already occupies `plans/confusion-census-2026-08-31.md` with `census-state.json` advanced to it; reconcile before writing this run's report to the same dated path.*


## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=40, sonnet verify=1, fable synthesis=1, haiku headroom-probe=2
