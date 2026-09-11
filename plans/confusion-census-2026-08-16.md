# confusion census 2026-08-16

Project: dark_factory

## Saturation

- batches: 3
- stop reason: saturated
  - batch 0: dup_rate=0.89 (total=20, succeeded=19, failed=1, saturated=False)
  - batch 1: dup_rate=0.90 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 2: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | implement | ops | unknown |
| --- | --- | --- | --- |
| ops | 0 | 1 | 0 |
| unknown | 1 | 0 | 1 |

## Synthesis

# Confusion census — 2026-08-16

**Date:** 2026-08-16
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests → per-finding verification against current main (Sonnet) → this synthesis (Fable). The three findings restated below survived the verification stage; this synthesis adds context-reading against the current tree and codebook only — no diagnosis appears here that was not itself verified.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml` (dispositions in §5 are inputs to the merger, which promotes/rejects in place).
**Run notes:** fifth completed periodic census. Previous: 2026-08-10 (`plans/confusion-census-2026-08-10.md`, 1 verified finding), 2026-08-05 (zero novel verified clusters), 2026-07-31 (15 findings / 4 clusters + 1 one-off). Saturation statistics and filed-task ids are appended by the census runner outside this synthesis.

## Corpus

- **3 verified findings, 3 sessions** (5ea0604f, 4396db7a, b976febe), one sighting each.
- Composition: **none of the three is a new cause.** Each finding lands on an already-catalogued codebook item — two pending candidates and one open entry — so this cycle's verified corpus is entirely accrual, not discovery.
- The legibility pipeline's own instruments are back in the corpus after two clean cycles: finding 1.3 is the digest instrument itself, and finding 1.2 manifested inside a census-support sub-agent (though its defect is harness-wide, with ~20 prior catalogued sightings in unrelated session types).
- Phase-stamp coverage: 3 of 6 stamps are `unknown` (finding 1.1's origin; both of finding 1.2's). Reported explicitly in the matrix per PRD §6.6, not guessed.
- A three-finding corpus supports counting and verification, not trend claims. Continuity notes below are labeled as observations over a small base.

## Executive summary (observations)

1. **The stray-comma Read tool-call defect recurred a third time in ten days, and the retry shows it is not self-correcting.** Session 5ea0604f's Read call for a large source file carried an empty parameter slot between `offset` and `limit` (`{"offset": 1240, , "limit": 1}`), rejected as unparseable by InputValidationError. Three turns later the agent reissued the same malformed structure — offset and stray comma unchanged, only `limit` edited (1 → 260) — i.e. it treated the failure as a bad limit value, not a JSON syntax defect. The shape is identical to the two catalogued sightings (`cand-20260806-12`: `"offset": 255, , "limit": 255`; `cand-20260812-19`: `"offset": 3500, , "limit": 340`): in all three, the empty slot sits between `offset` and `limit` in a Read call.
2. **The harness's usage-limit resume injection claimed a continuity that did not exist.** A sub-agent ("Find prior legibility census fixes", session 4396db7a) was terminated by the weekly usage limit having produced only a statement of intent ("I'll start with the git log searches and repo structure exploration in parallel.") as its captured output. The harness then injected the standard "Your previous run was interrupted by a usage limit. Continue where you left off and complete your task." — but there was no persisted partial progress for the resume to continue from. This extends the `subagent-runner-protocol-defects` entry's usage-limit-resume evidence (previously: 5-hour-window limits on watcher rotations and one-shot CLI runners) to the weekly cap and to harness Agent-tool sub-agents.
3. **That same resume boilerplate is harvested by the digest's gold "User Corrections" section.** Session b976febe's digest captured the injected resume message twice (turns 199 and 273) and the turn-3 `# Context` memory block as User Corrections, while all five `signal_counts` read 0. Verified against current main: `scripts/legibility/digest.py` *does* filter harness-injected turns (`is_harness_injected_turn`, digest.py:580), but its coverage is exactly two shapes — the orchestrator briefing (which requires `# context`, `## agent identity`, and `# task` to co-occur as line-anchored headings) and the trickle-coder prompt literal in `HARNESS_PROMPT_MARKERS` (digest.py:570). The resume boilerplate matches neither, and a lone `# Context` block near session start passes the deliberate all-three co-occurrence guard.

## Origin × manifestation matrix

Rows = `origin_phase`, columns = `manifested_phase`. Counts are verified sightings (3 total). `merge` and `verify` kept explicitly to show their zeros; `unknown` reported, not imputed.

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| ops | · | · | · | · | · | · | · | 1 | · | **1** |
| unknown | · | · | 1 | · | · | · | · | · | 1 | **2** |
| **total** | **0** | **0** | **1** | **0** | **0** | **0** | **0** | **1** | **1** | **3** |

Readings (observational): no merge- or verify-manifested sighting for a fourth consecutive cycle; the last three corpora total 5 findings, so this absence still carries little evidence on the PRD's motivating architect/implement→merge hypothesis. Finding 1.1's origin stays `unknown` deliberately: whether the stray comma is model-emitted or introduced in harness-side serialization is not established by any of the three catalogued sightings, and the distinction determines where a fix would live.

## 1. Verified clusters

### 1.1 Stray-comma Read tool-call JSON, reissued unchanged on retry with only the limit edited (1 sighting, session 5ea0604f)

The agent's Read call for a large source file (`.../cross_graph_move.py`, offset 1240) contained an empty parameter slot between `offset` and `limit` — `{"offset": 1240, , "limit": 1}` — which InputValidationError rejected as unparseable, echoing the sent input back. On the next attempt, 3 turns later, the agent reissued the identical malformed structure with only `limit` changed (1 → 260); the stray comma and offset were untouched. The retry therefore encoded a misdiagnosis — the agent read the validation failure as a bad `limit` value rather than a JSON syntax defect — so the visible "retry" was not a real self-correction.

**Relation to the codebook (observation, not a merge):** this is the third sighting of one defect shape currently split across two pending candidates: `cand-20260806-12` ("Agent generates malformed JSON parameters in tool calls", session cb414c37, `"offset": 255, , "limit": 255`) and `cand-20260812-19` ("Read tool parameter construction leaves double comma in malformed JSON", session 0f36bfb9, `"offset": 3500, , "limit": 340`). All three instances place the empty slot between `offset` and `limit` in a Read call. What this sighting adds beyond count is the retry facet: the error is stable under retry because the agent's diagnosis targets the wrong parameter. The defect is upstream of the repo (tool-call generation/serialization); no in-repo code path produces or can intercept it.

### 1.2 Sub-agent resumed after a weekly usage-limit kill has nothing persisted to resume from (1 sighting, session 4396db7a)

A harness Agent-tool sub-agent tasked with "Find prior legibility census fixes" was terminated early ("Agent terminated early due to an API error: You've hit your weekly limit"). Its entire captured `<result>` was one sentence of stated intent — the git-log and repo-structure searches it was about to start. The harness then auto-injected the standard resume prompt: "Your previous run was interrupted by a usage limit. Continue where you left off and complete your task." There was no partial-analysis artifact anywhere for that instruction to refer to; "where you left off" was, verifiably, nowhere.

**Relation to the codebook (observation, not a merge):** a direct sighting of the `subagent-runner-protocol-defects` entry's "usage-limit resume storms" clause, whose existing sighting notes already name the same substructure ("no visible partial-output checkpoint", "no persisted watcher-progress state", "falsy-[] predictions never persisted"). Two extensions of the evidence class, both observational: the limit here is the **weekly** cap rather than the session-window limit in all prior sightings, and the interrupted party is a harness Agent-tool sub-agent rather than a one-shot CLI runner or a watcher rotation. The entry already carries filed task 2561; whether that task's scope (runner-side persistence protocol) reaches Agent-tool sub-agents is not established by this sighting.

### 1.3 Digest "User Corrections" bucket captures the usage-limit resume boilerplate and a lone `# Context` block; the existing filter does not cover either shape (1 sighting, session b976febe)

The digest for session b976febe lists three User Corrections: the injected usage-limit resume message, captured twice (turns 199 and 273), and the turn-3 `# Context` memory block — while every one of the five `signal_counts` (tool_error / self_correct / not_found / df_guard / interrupt) reads 0. None of the captured turns is corrective user feedback; the corrections bucket — the digest's gold, trimmed-last section (PRD §7.2) — is filled entirely with harness-injected content in a zero-signal session.

Verified against current main, refining the finding's cause statement: the extractor is *not* unfiltered. `scripts/legibility/digest.py` excludes sidechain, `isMeta`, tool-result-only, and harness-injected turns, with `is_harness_injected_turn` (digest.py:580) as the single filter for both the gold section and the score's `n_user_turns` component. Its coverage, however, is exactly two shapes: the full orchestrator briefing — which requires `# context`, `## agent identity`, and `# task` to **co-occur** as line-anchored headings, a deliberate false-positive guard — and the one literal in `HARNESS_PROMPT_MARKERS` (digest.py:570), the trickle-coder prompt. The resume boilerplate matches neither marker set, and a `# Context` block arriving without its two sibling headings passes the co-occurrence guard by design. The markers docstring explicitly invites the fix shape: "Extend with future harness prompt literals as one-line additions."

**Relation to the codebook (observation, not a merge):** a sighting of `entry-cand-20260722-10` ("Trickle-coder digest pipeline recursively re-ingests its own prompts as session content, mislabeled as User Corrections"), whose 12 accrued sightings already generalized the defect from self-recursion to "any large injected turn near session start"; one 08-07 sighting note already mentions retry-continuation boilerplate. Adjacent: `cand-20260813-2` (same bucket capturing skill invocations and interrupts). This sighting is the first to pin the residual gap against the *current* filter implementation on main — the earlier sightings predate or don't reference `is_harness_injected_turn`.

## 2. One-off sightings

None beyond the clusters above (each cluster is itself single-sighting this cycle).

## 3. Cross-cutting observations

1. **One harness artifact appears on both sides of the observation chain.** The usage-limit resume injection is the confusing *input* in 1.2 (a continuity claim with no substrate) and the misclassified *content* in 1.3 (harvested as gold user signal by the measurement instrument). These are distinct defects with distinct owners — the injection wording is harness-owned; the misclassification is `scripts/legibility/digest.py` — but they are the same string, and a fix to either leaves the other standing.
2. **Zero novel causes.** All three findings accrue to catalogued codebook items. Alongside the near-empty 08-05 and 08-10 corpora, this is consistent with the PRD's saturating-head premise — the nightly trickle absorbing the distribution's head — though three findings over one cycle is a small base for that reading.
3. **The retry-is-not-correction facet of 1.1 changes the cost shape of an already-catalogued defect.** Prior sightings recorded a rejected call; this one records a rejected call *plus* a burned retry that preserved the defect. The sighting establishes one wasted retry; whether the agent recovered afterwards is not established.

## 4. Remediation candidates

To be filed via the curator path (plain `submit_task`; curator dedup is the protection — PRD §6.9).

| # | Candidate | Cluster | Size |
|---|---|---|---|
| R1 | Extend `HARNESS_PROMPT_MARKERS` in `scripts/legibility/digest.py` with the usage-limit resume literal (`'your previous run was interrupted by a usage limit'`) — the one-line addition the marker docstring itself invites — and decide whether a lone `# Context` briefing/memory block near session start warrants its own guard, given the all-three-headings co-occurrence requirement (a deliberate false-positive guard) is what admits it | 1.3 | S |

No task for 1.1: the defect is in tool-call generation upstream of the repo; the codebook action (§5) is consolidation of the split candidates, and the entry is the natural anchor for any future upstream report. No new task for 1.2: the parent entry already carries filed task 2561; whether its scope covers Agent-tool sub-agents is a question for that task, not a new filing.

## 5. Codebook dispositions (input to the merger; promote/reject in place, never delete)

| Cluster / finding | Suggested disposition |
|---|---|
| 1.1 stray-comma Read JSON, stable under retry | New sighting of the existing stray-comma class; `cand-20260806-12` and `cand-20260812-19` describe the same defect shape (empty slot between `offset` and `limit` in a Read call, three sessions, 08-06→08-16) and are consolidation candidates for the merger. Append the retry-misdiagnosis facet (limit edited, comma preserved) to the surviving item's cause |
| 1.2 no persisted progress behind "continue where you left off" | Sighting of `subagent-runner-protocol-defects` (usage-limit resume storms clause); note the two evidence-class extensions — weekly cap, Agent-tool sub-agent — in the sighting note |
| 1.3 User Corrections captures resume boilerplate + lone `# Context` | Sighting of `entry-cand-20260722-10`; accrual now spans 2026-07-22 → 2026-08-16 across three census cycles with 13 sightings, which argues for promotion to an entry. Cross-reference `cand-20260813-2` as the same bucket-defect family |

## 6. Method notes for the next census

- If R1 lands, the discriminating signal next cycle is zero User-Corrections captures of the resume boilerplate in post-fix digests; a recurrence of the lone-`# Context` shape specifically would point at the co-occurrence guard rather than the marker list, and those two should not be conflated in coding.
- The stray-comma Read class is now the corpus's only recurring tool-call-generation defect (three sightings, ten days, two candidate ids). Next cycle should check whether the merger consolidated the ids, and whether the shape recurs at all — its `origin_phase: unknown` (model-side vs harness serialization) is the open question an upstream report would need answered.
- Finding 1.2 is worth one targeted look next cycle: whether task 2561's persistence protocol, if landed, changes what a weekly-limit-killed Agent-tool sub-agent leaves behind. The entry's ~20 prior sightings were all coded before the weekly cap existed as a distinct interruption class.

---

*Synthesis note to the runner: written from the 3 verified findings supplied; codebook relations verified by reading `docs/legibility/confusion-codebook.yaml` and `scripts/legibility/digest.py` (lines 545–654) on current main at synthesis time. No tasks filed and no codebook edits made from this synthesis — filing and merger application are the runner's steps.*


## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=60, sonnet verify=5, fable synthesis=1, haiku headroom-probe=2
