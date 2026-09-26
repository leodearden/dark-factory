# confusion census 2026-09-15

Project: dark_factory

## Saturation

- batches: 3
- stop reason: saturated
  - batch 0: dup_rate=0.89 (total=20, succeeded=18, failed=2, saturated=False)
  - batch 1: dup_rate=1.00 (total=20, succeeded=18, failed=2, saturated=True)
  - batch 2: dup_rate=0.94 (total=20, succeeded=18, failed=2, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | implement |
| --- | --- |
| unknown | 1 |

## Synthesis

Synthesis complete. Everything below was checked against the archived transcripts, the plan-tools source, the codebook and the prior census reports; where the verifier's framing disagrees with that evidence, the correction is stated explicitly.

**Date:** 2026-09-15
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests, per-finding verification against current main (Sonnet), then this synthesis (Fable). One finding reached synthesis. This document adds a read of the archived transcript, the plan-tools source and tests, the architect briefing source, a fleet-wide grep of every archived transcript, and the codebook. Every mechanism claim below names the evidence it rests on.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml`. Dispositions in §4 are inputs to the merger.
**Run notes:** eleventh completed periodic census, at the PRD's 5-day hard floor after 2026-09-10. Previous corpora: 09-10 (2 verified findings), 09-05 (1), 08-31 (1), 08-26 (1), 08-21 (2), 08-16 (3), 08-10 (1), 08-05 (0), 07-31 (15 findings / 4 clusters + 1 one-off), 07-24 (52). No report dated today pre-existed on disk and `census-state.json` still reads 09-10. The codebook file changed under this synthesis: the pending candidate discussed below moved from line 20494 to line 20652 between two reads a few minutes apart, so this report cites codebook records by id only. Saturation statistics and filed-task ids are appended by the runner outside this synthesis.

### Corpus

- **1 verified finding, 1 session, 1 sighting.** Session `6983eff4` is task 5032's architect session in `.worktrees/5032` on branch `task/5032`, 2026-09-12 10:29:58Z to 10:47:37Z, `claude-opus-5` at effort max, CLI 2.1.269, budget $15. The turn prompt is the architect briefing built by `orchestrator/src/orchestrator/agents/briefing.py::BriefingAssembler.build_architect_prompt`; the session's own title is "Merge-lane test group γ9 migration to injected ports".
- **The verifier's framing needs two corrections.** It labelled the agent an implementer and stamped the sighting unknown × implement. The transcript shows an architect: the briefing instructs it to build the plan with plan-tools calls, it does so, and it calls `confirm_plan` 29 seconds before the probe. The verifier also called the field "undocumented" and quoted the plan's top-level keys as the schema that lacks it. The probe crashed on a per-step key, not a top-level one, and the per-step key is documented in code. §1.1 gives the mechanism.
- **Composition:** one sighting of a shape the codebook already holds as a pending candidate with one prior sighting, `cand-20260822-3`, whose cause text guesses at three explanations that the source contradicts. The fleet grep in §2 finds eleven further unreported sightings of the identical crash.
- **Phase-stamp coverage:** the verifier's stamps carry 1 unknown of 2. The transcript resolves both to architect. The refinement is reported alongside the runner's matrix rather than overwriting it.
- Session `6983eff4` has no codebook presence.

### Executive summary (observations)

1. **The agent read a plan step by the tool's parameter name, and the plan stores it under a different key.** The architect briefing names `step_type` three times as the parameter of `add_plan_step`, and the tool's own description says only that it "must be test or impl". The plan document writes that value under `type`. The agent's readback probe indexed `s['step_type']`, crashed, corrected itself to `s.get('type')` four seconds later, and finished normally. No agent-visible surface, briefing, tool description or user doc, states the stored key.
2. **This is not a one-off.** Across the archived fleet, 13 architect sessions since 2026-08-06 crashed on exactly this KeyError, four of them in the five days before this census, and a further 45 non-architect sessions hedged the same uncertainty by reading both names. The codebook records one of the 13.

### Origin × manifestation matrix

The runner's matrix renders the verifier's stamps, unknown × implement. The table below carries the refined stamps this synthesis establishes in §1.1.

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| architect | · | 1 | · | · | · | · | · | · | · | **1** |
| **total** | **0** | **1** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **1** |

Readings, observational. The sighting originates and manifests in the same session, within one tool call. The `merge` and `verify` columns are zero for a tenth consecutive cycle. The PRD's motivating architect/implement→merge hypothesis remains untested by the eleven post-07-24 corpora, which total 13 findings.

### 1. Verified clusters

#### 1.1 Plan-tools parameter vocabulary differs from the plan document's key vocabulary; architects reading their own plan back index by the parameter name (1 sighting, session `6983eff4`; 12 further sightings in §2)

**The trace, from the archived transcript** at `data/orchestrator/agent-transcripts/5032/…/6983eff4-17fc-4258-a6b3-948f1cb97846.jsonl`, 504 records. Times are UTC on 2026-09-12.

| time | record | action | result |
|---|---|---|---|
| 10:46:47.9 | · | `confirm_plan` stamps `_finalized_at` | plan complete: 26 steps, 2 prerequisites, 15 files |
| 10:47:16.7 | 490 | Bash "Verify plan.json structure": prints top-level keys, counts, a completion-marker filter, then `[s['step_type'] for s in p['steps']]` | `KeyError: 'step_type'` at 10:47:17.3 |
| 10:47:20.6 | 497 | Bash "Confirm plan finalized and step alternation": `p['_finalized_at']`, `list(p['steps'][0].keys())`, `s.get('type')` | step keys `id, type, description, status, commit`; 26 steps alternating test/impl |
| 10:47:37.1 | 501 | closing report, `end_turn` | session ends |

| probe block | value |
|---|---|
| gap from crash to corrected probe | 3.9 s |
| budget before failed probe | $11.007 |
| cost of the failed turn | ≈ $0.12 |
| cost of the corrected turn | ≈ $0.11 |
| session total | $11.23 of $15 |

**Two key-name mismatches in one probe, not one.** The same failed probe also filtered top-level keys for the substrings `confirm`, `complete` and `status` to find the completion marker. It printed `{}`. The marker is `_finalized_at`, which the agent had stamped 29 seconds earlier and which the probe's own `keys:` line listed. The corrected probe read `_finalized_at` directly, so the agent knew the name; the filter was a second guess at the document's vocabulary from the tool's vocabulary.

**Where the name came from.** The agent did not invent `step_type`. Record 2 of the transcript, the briefing, carries `add_plan_step(step_id, step_type, description)` and "`step_type` must be `"test"` or `"impl"`" verbatim from `orchestrator/src/orchestrator/agents/roles.py` lines 1014 to 1015 and `briefing.py` line 535. Record 213, the deferred-tools record, carries the tool schema with `step_type` as a required string property. Nothing the agent could see names the stored key.

**What the source says.** `orchestrator/src/orchestrator/mcp/plan_tools.py::_add_plan_step` appends `{'id', 'type', 'description', 'status', 'commit'}`, writing the `step_type` argument under `type`; `_replace_plan_step` does the same. The markup-repair surface in the same module documents the split explicitly: "the tool's parameter names and the plan document's key names differ", `step_type` "is written as `type`", and a lookup on `holder.get('step_type')` is described as "always-absent". `orchestrator/tests/test_plan_tools_markup_repair.py` pins that a recovered `step_type` must never land as a junk key beside `type`. The mapping is therefore deliberate, documented in code, and tested. It is documented only on the side that writes the file. `docs/task-authoring.md` does not mention plan.json; `ARCHITECTURE.md` and `OPERATIONS.md` name it only as an artifact path.

**Correction to the codebook's existing framing.** `cand-20260822-3` records the earlier sighting with the cause "either the plan generator has a bug, the schema evolved and the consumer wasn't updated, or this is a task-specific schema drift". None of the three matches the source: the generator writes `type` by design, the consumer is the agent's own ad hoc script, and the key is identical in every plan read across the fleet. That candidate also stamps the sighting decompose × implement; its session `3b6d6289` is task 4573's architect session, and its crash and self-correction have the same shape as this one, 4.1 seconds apart, on CLI 2.1.239.

### 2. Fleet census of the same shape

Grep over every file under `data/orchestrator/agent-transcripts/`, run for this synthesis.

| population | files |
|---|---|
| archived transcript files, including subagent files | 14,005 |
| files carrying the architect briefing marker | 1,588 |
| files whose Bash commands index `['step_type']` | 16 |
| files containing `KeyError: 'step_type'` | 13 |
| of those, architect-briefed | 13 |
| of those, `claude-opus-5` | 13 |
| of those, already cited anywhere in the codebook | 2 |
| files whose Bash commands read `.get('step_type')` | 58 |
| of those, hedged as `s.get('type') or s.get('step_type')` | 32 |
| of those, not architect-briefed | 45 |

The 13 crashes by task and date: 3473 (08-06), 3508 (08-07), 4274 (08-16), 4589 (08-21), 4573 (08-22), 4653 (08-23), 4702 (08-24), 4481 (09-03), 4899 (09-08), 5331 (09-10), 5283 (09-11), 5032 (09-12), 5253 (09-13). The rate against the all-time architect denominator is under one percent; the window of crashes starts 08-06, so the rate over that window is higher and was not measured here.

The 58 hedged readers are mostly implementers and amenders reading a plan they did not write, for example task 2545 probing `.worktrees/.task-meta/2488/plan.json` with `s.get('type') or s.get('step_type')` and a three-way hedge on the completion marker, and task 4917 with the same two hedges. A hedge costs no crash and no turn, so these sessions surface nowhere in the digests. They are the quiet majority of the same uncertainty.

Only sessions `3b6d6289` and `b4c79eca` appear in the codebook, and `b4c79eca` only on `entry-cand-20260728-3`, an unrelated no-such-file entry. No earlier census report mentions `step_type` or `cand-20260822-3`.

### 3. Observations the runner may act on

These are observations, not rulings. The remediation surface is agent-visible text, since the source already documents the mapping on the writer's side.

- **No agent-visible surface states the stored per-step key.** The tool description for `add_plan_step` and the briefing text at `roles.py` lines 1014 to 1015 are the two places an architect reads before writing a readback probe. A readback probe was issued in 13 of 1,588 architect sessions with the wrong key and in at least 45 other sessions with a hedge.
- **The completion marker has the same gap.** `_finalized_at` is named in `confirm_plan`'s description as "a durable completeness marker" without the key, and agents guess `confirmed`, `finalized`, `status` and substring filters.
- **The candidate's cause text should be replaced, not extended.** Its three hypotheses are each contradicted by the source read above.

### 4. Codebook dispositions (inputs to the merger)

- `cand-20260822-3`: add sighting `6983eff4` (2026-09-12, dark_factory, architect × architect). Restamp the existing `3b6d6289` sighting to architect × architect; the transcript is an architect session. Retitle to the observed mechanism, that the plan-tools parameter vocabulary differs from the plan document's key vocabulary and no agent-visible text names the stored key. Replace the cause text with §1.1's source-backed mechanism. The 11 further crash sessions in §2 are available as sightings; none was mined by the trickle.
- No new candidate. No entry retires.

Not verified here: task 5032's subsequent landing is recorded in session memory as 2026-09-14 but git was unavailable to this synthesis, so the report does not rely on it.


## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=60, sonnet verify=3, fable synthesis=1, haiku headroom-probe=3
