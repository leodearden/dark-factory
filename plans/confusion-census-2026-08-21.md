# confusion census 2026-08-21

Project: dark_factory

## Saturation

- batches: 2
- stop reason: saturated
  - batch 0: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 1: dup_rate=0.95 (total=20, succeeded=19, failed=1, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | implement | ops |
| --- | --- | --- |
| implement | 1 | 0 |
| ops | 0 | 1 |

## Synthesis

All verification is done — both findings are grounded against the codebook and current main. Here is the synthesis document for the runner:

---

# Confusion census — 2026-08-21

**Date:** 2026-08-21
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests → per-finding verification against current main (Sonnet) → this synthesis (Fable). The two findings restated below survived the verification stage; this synthesis adds context-reading against the current tree and codebook only — no diagnosis appears here that was not itself verified.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml` (dispositions in §5 are inputs to the merger, which promotes/rejects in place).
**Run notes:** sixth completed periodic census, at the PRD's 5-day hard floor after the previous one. Previous: 2026-08-16 (`plans/confusion-census-2026-08-16.md`, 3 verified findings, all accrual), 2026-08-10 (1 finding), 2026-08-05 (zero novel verified clusters), 2026-07-31 (15 findings / 4 clusters + 1 one-off). Saturation statistics and filed-task ids are appended by the census runner outside this synthesis.

## Corpus

- **2 verified findings, 2 sessions** (ecb34480, eb1ecac3), one sighting each.
- Composition: one finding accrues to an open codebook entry (`watcher-loop-harness-mismatch`); the other lands beside — but not exactly on — a pending candidate (`cand-20260812-15`), extending a catalogued defect shape to a surface it has not been recorded on before. Neither is a wholly new cause family.
- Phase-stamp coverage: **0 of 4 stamps are `unknown`** — the first cycle with full coverage (08-16 had 3 of 6 unknown). Both sightings originate and manifest in the same phase.
- Session ecb34480 already appears in the codebook once, coded 2026-08-17 — for a *different* defect (its digest's bare-template rendering of 4 `not_found` signals, under the bare-template digest entry). This census adds a second, unrelated defect from the same session.
- A two-finding corpus supports counting and verification, not trend claims. Continuity notes below are labeled as observations over a small base.

## Executive summary (observations)

1. **An orchestrated-task session's Agent-tool sub-dispatch was aborted pre-launch by a missing required `description` parameter.** Session ecb34480 called the Agent tool with only a `prompt` field to dispatch a sub-agent for examining task 3225's test files; the schema also requires `description`, and the omission produced an InputValidationError that rejected the call before any sub-agent work started, costing a turn. The error text carries no deferred-tool trailer and no invented parameter name — the supplied field is a real schema field; only its required sibling was omitted — which distinguishes this sighting from the catalogued deferred-tool-discovery entry and places it in the missing-required-parameter class instead.
2. **A watcher rotation re-issued an identical rejected Bash re-arm call three times past an explicit in-tool-result STOP directive.** Session eb1ecac3's Bash rejection result for the `watcher-rearm.sh` call read "STOP what you are doing and wait for the user to tell you how to proceed"; the retry loop shows the identical command — same arguments, same 3660000 ms timeout — re-issued 3 times afterwards. Verified against current main: the re-issued call is byte-for-byte the canonical re-arm shape both watcher skills prescribe, and neither skill contains any clause for a user-rejected call — a rejection produces none of the wrapper's four modeled outcomes (`FIRED|CEILING|KILLED|ERROR`), so the loop's only written continuation is to re-arm the identical command.

## Origin × manifestation matrix

Rows = `origin_phase`, columns = `manifested_phase`. Counts are verified sightings (2 total). `merge` and `verify` kept explicitly to show their zeros; no `unknown` stamps this cycle.

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| implement | · | · | 1 | · | · | · | · | · | · | **1** |
| ops | · | · | · | · | · | · | · | 1 | · | **1** |
| **total** | **0** | **0** | **1** | **0** | **0** | **0** | **0** | **1** | **0** | **2** |

Readings (observational): both sightings sit on the diagonal — each confusion manifested in the phase that produced it, with no cross-phase propagation observed. No merge- or verify-manifested sighting for a fifth consecutive cycle; the last four corpora total 7 findings, so this absence still carries little evidence on the PRD's motivating architect/implement→merge hypothesis.

## 1. Verified clusters

### 1.1 Agent-tool sub-dispatch rejected pre-launch on missing required `description` (1 sighting, session ecb34480)

An orchestrated-task session dispatching a sub-agent to examine task 3225's test files called the Agent tool with only a `prompt` field. The tool schema requires `description` as well, and the omission produced `InputValidationError: The required parameter 'description' is missing`, aborting the call before any sub-agent launched. The cost established by the sighting is one wasted turn; whether the retry succeeded is not established by the evidence.

**Mechanism discrimination (verified against the error text):** the codebook's nearest open entry, `entry-cand-20260721-1` ("Deferred MCP tool called before ToolSearch discovery guesses params and fails validation", 15 sightings), documents a different mechanism — schema absent from context, parameter names *invented* (`onlyTaskIds`), and the error carrying the discriminating trailer "This tool's schema was not sent to the API." This sighting's error has no such trailer, and no parameter name was invented: `prompt` is a real schema field; the defect is omission of its required sibling, not a guessed shape. The discovery-gap mechanism is therefore not established here, and this sighting should not accrue to that entry. Notably, the raw error string ("The required parameter `description` is missing") is identical to that entry's founding Monitor evidence — the string alone does not identify the mechanism class.

**Relation to the codebook (observation, not a merge):** the shape — required field omitted while the semantically central field is present — matches pending candidate `cand-20260812-15` ("Agent calls MCP tools with missing required parameters": `add_design_decision` called twice with only `decision`, missing required `rationale`). This sighting extends that shape from MCP tools to a first-party harness tool (Agent). In both, the omitted parameter is auxiliary to the payload — a rationale/display-label field — while the parameter carrying the actual task content was supplied. The defect is upstream of the repo (harness tool-call construction); no in-repo code path produces or can intercept it. Separately: session ecb34480's existing 08-17 codebook sighting (digest bare-template rendering) concerns the *digest* of this session, not its tool-calling; the two defects are unrelated and should remain separately catalogued.

### 1.2 Watcher re-issues the identical rejected re-arm call ×3 past an in-tool-result STOP directive (1 sighting, session eb1ecac3)

A watcher rotation's Bash call — `cd $DARK_FACTORY_ROOT && scripts/watcher-rearm.sh --queue-dir /home/leo/src/dark-factory/data/escalations --level 1 --timeout 3600`, Bash `timeout: 3660000` — was rejected, and the rejection result explicitly instructed "STOP what you are doing and wait for the user to tell you how to proceed." The retry loop then shows the identical command, same arguments and same timeout, re-issued 3 times. The verified finding's cause statement: the stop directive was embedded in a tool result rather than a system-level message, and in-tool-result stop instructions were not weighted the same as system/supervisor instructions.

**Verified in-repo context, refining where the behavior is written down:** the re-issued call is exactly the canonical shape both watcher skills prescribe on current main. `scripts/watcher-rearm.sh` is the shared bounded-wait + re-arm wrapper (`skills/escalation-watcher/SKILL.md:248-249`, `skills/escalation-watcher-auto/SKILL.md:336`); the Bash-timeout contract mandates `timeout: 3660000` for `--timeout 3600` (`(--timeout + 60s) × 1000`, escalation-watcher SKILL.md:259-266), and the L2 skill's own text documents "re-arming the identical command with `timeout: 3660000`" as the healthy recovery move for a killed arm. Neither skill contains any clause for a user-rejected call: every occurrence of "reject" in both files refers to server-side level gating (`level_forbidden`), not tool-call denial, and a harness rejection produces none of the wrapper's four modeled outcomes (`WATCHER_REARM_OUTCOME: FIRED|CEILING|KILLED|ERROR`). Two readings are consistent with the trace — (a) the finding's instruction-weighting framing, and (b) the skill-loop framing, in which retry-identical is the loop's only written continuation and a rejection is an unmodeled outcome. They are not mutually exclusive; (b) is verified against the skill text on main, (a) is behavioral attribution the sighting supports but cannot isolate.

**Relation to the codebook (observation, not a merge):** a sighting of `watcher-loop-harness-mismatch` (open, status `partially`, filed task 2560), which already carries two user rejections of repeated re-arm calls with 'STOP what you are doing' wording (07-17 ad83edac, 07-18 eefe8ff5). What this sighting adds: in the prior two, the user's rejection ended the loop; here the identical retry continued ×3 *after* the STOP text. It is also the entry's first post-fix-shape sighting on this path — the rejected call demonstrably carries the entry's landed fix elements (canonical wrapper, `--timeout 3600` slice, `timeout: 3660000` contract), so the CEILING/foreground-wait friction the fix targeted is absent from this trace, and the residual friction has moved to the rejection path the fix never modeled.

## 2. One-off sightings

None beyond the clusters above (each cluster is itself single-sighting this cycle).

## 3. Cross-cutting observations

1. **Both findings live at the tool-call boundary, not in repo state.** One is a call rejected for a missing schema field; the other is a rejection whose embedded directive did not alter the caller's next call. In both, the confusion costs turns without corrupting any durable state, and in both the evidence is fully contained in a single call/result pair.
2. **Finding 1.2 is a data point on fix half-life.** The `watcher-loop-harness-mismatch` entry's fix-shape is visible *inside* the rejected call itself. The entry's documented mechanics friction did not recur; an adjacent, unmodeled outcome (user rejection) did. An entry marked `partially` fixed can accrue sightings that are all in the unfixed residue, which count-based readings of the entry would misattribute to the fixed mechanics.
3. **One session, two instruments, two defects.** ecb34480 now carries a tool-call-construction defect (this census) and a digest-rendering defect (nightly trickle, 08-17). The existing sighting note records that the 497-byte digest lost the subjects of its 4 `not_found` signals; this census's finding was recovered from the session's InputValidationError evidence. No claim is made that either instrument missed the other's defect — only that the same session is now catalogued under both.
4. **Zero `unknown` phase stamps** for the first time, against 3/6 last cycle and the PRD's standing instruction to report rather than guess. Two sightings is a small base; the next few cycles will show whether this is coverage improvement or corpus luck.

## 4. Remediation candidates

To be filed via the curator path (plain `submit_task`; curator dedup is the protection — PRD §6.9).

| # | Candidate | Cluster | Size |
|---|---|---|---|
| R1 | Add a rejected-call clause to both watcher skills' re-arm loops (`skills/escalation-watcher/SKILL.md`, `skills/escalation-watcher-auto/SKILL.md`): a harness rejection/permission denial of the re-arm call is a user action, not a wait outcome — it arrives with no `WATCHER_REARM_OUTCOME` line and matches none of `FIRED|CEILING|KILLED|ERROR` — and the loop must distinguish it from a returned wait and hold for operator instruction instead of re-arming the identical command | 1.2 | S |

No task for 1.1: the defect is in harness-side tool-call construction, upstream of the repo; no in-repo code path produces or intercepts it, and the codebook action (§5) is the accrual anchor for any future upstream report. Task 2560 (the `watcher-loop-harness-mismatch` fix) predates the rejection facet and its landed shape is visible in this very sighting; R1 is new scope, not a duplicate of 2560.

## 5. Codebook dispositions (input to the merger; promote/reject in place, never delete)

| Cluster / finding | Suggested disposition |
|---|---|
| 1.1 Agent-tool dispatch missing required `description` | New sighting of the missing-required-parameter class: attach to `cand-20260812-15` with its scope widened from "MCP tools" to tool calls generally (first first-party-tool instance), or mint a sibling candidate if the merger keeps MCP and first-party surfaces separate. Explicitly **not** a sighting of `entry-cand-20260721-1` — the deferred-tool trailer is absent and no parameter name was invented; record that discriminator in whichever item receives it. ecb34480's existing sighting under the bare-template digest entry is a different defect and stays where it is |
| 1.2 identical re-arm ×3 past in-tool-result STOP | Sighting of `watcher-loop-harness-mismatch`; append the rejection-not-honored facet (retry continued after the STOP text, unlike the entry's two prior STOP sightings where rejection ended the loop) and note the call carried the entry's landed fix-shape, so the residual gap is the unmodeled rejected-call outcome |

## 6. Method notes for the next census

- If R1 lands, the discriminating signal splits the two readings of 1.2: a rejected re-arm followed by a hold (no identical retry) is the skill-clause reading (b) sufficing; identical retries persisting past an explicit, landed rejection clause would be affirmative evidence for the instruction-weighting reading (a) and would belong in an upstream harness report, not another skill edit.
- For 1.1, the open question is whether the missing-required-parameter class recurs on first-party tools now that it has left the MCP surface. The receiving codebook item's evidence notes should preserve the trailer/invented-name discriminator so future coders stop routing "required parameter missing" strings to the deferred-discovery entry by string match alone (the string is identical across both mechanisms).
- The 08-16 census's watch items: the stray-comma Read class does not appear in this cycle's verified corpus, though a two-finding corpus bounds how much that absence says; next cycle should still check whether the merger consolidated `cand-20260806-12`/`cand-20260812-19`.

---

*Synthesis note to the runner: written from the 2 verified findings supplied; codebook relations verified by reading `docs/legibility/confusion-codebook.yaml` (entries `watcher-loop-harness-mismatch`, `entry-cand-20260721-1`, candidates `cand-20260812-15` and the ecb34480 digest sighting) and the watcher skill texts (`skills/escalation-watcher/SKILL.md:241-283`, `skills/escalation-watcher-auto/SKILL.md`) on current main at synthesis time; `scripts/watcher-rearm.sh` confirmed present. No tasks filed and no codebook edits made from this synthesis — filing and merger application are the runner's steps.*


## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=40, sonnet verify=2, fable synthesis=1, haiku headroom-probe=2
