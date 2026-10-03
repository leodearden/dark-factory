# confusion census 2026-09-10

Project: dark_factory

## Saturation

- batches: 2
- stop reason: saturated
  - batch 0: dup_rate=0.95 (total=20, succeeded=19, failed=1, saturated=True)
  - batch 1: dup_rate=0.95 (total=20, succeeded=19, failed=1, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | implement |
| --- | --- |
| implement | 2 |

## Synthesis

**Date:** 2026-09-10
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests, per-finding verification against current main (Sonnet), then this synthesis (Fable). Two findings reached synthesis. This document adds a read of both archived transcripts, the role-prompt sources, the markup-guard journal, the fleet transcript corpus since 2026-08-05, and the codebook. Every mechanism claim below names the evidence it rests on. Where the transcript or corpus contradicts the verifier's framing, the framing is corrected explicitly rather than silently.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml`. Dispositions in §5 are inputs to the merger.
**Run notes:** tenth completed periodic census, at the PRD's 5-day hard floor after 2026-09-05. Previous corpora: 09-05 (1 verified finding), 08-31 (1), 08-26 (1), 08-21 (2), 08-16 (3), 08-10 (1), 08-05 (0), 07-31 (15 findings / 4 clusters + 1 one-off), 07-24 (52). No report dated today pre-existed on disk and `census-state.json` still reads 09-05. The trickle has minted 66 candidates since the last census. Saturation statistics and filed-task ids are appended by the runner outside this synthesis.

### Corpus

- **2 verified findings, 2 sessions, 2 sightings.** Both are ordinary orchestrator-dispatched task sessions in dark_factory worktrees on CLI 2.1.263, both `claude-opus-5` at effort max. Neither is a legibility-pipeline session.
- **Session `316ede03`** is task 3651's implementer amendment pass in `.worktrees/3651`, 2026-09-07 23:54Z to 09-08 01:12Z, budget $10. The verifier labelled the area "tooling/harness-affordance-discovery" and stamped implement × implement.
- **Session `ad12f696`** is task 4823's architect re-plan after a blocking review, in `.worktrees/4823`, 2026-09-07 03:27Z to 03:34Z, budget $15. The verifier labelled the area "plan-tools MCP / tool-call envelope leakage" and stamped implement × implement.
- **Composition:** two harness-boundary classes, both already catalogued. One is a re-sighting of the tool-call envelope leak, the most heavily documented defect in the codebook. The other is a new sighting shape whose in-repo cause is a role prompt naming a tool that does not exist in any CLI build the fleet has run.
- **Phase-stamp coverage:** the verified stamps carry 0 unknown of 4. The transcripts confirm the first and refine the second: `ad12f696` is an architect-role session, so §1.2 moves it to architect × architect and reports the refinement alongside the runner's matrix rather than overwriting it.
- Session `ad12f696` already has codebook presence in two places (a sighting on `oneoff-2026-07-09` and on `cand-20260907-18`), both dated 2026-09-07 and both minted by the trickle before this census ran. Session `316ede03` has none.

### Executive summary (observations)

1. **The agent that polled its own pytest run sixteen times was following its briefing, and the briefing named a tool that does not exist.** The amender turn-prompt told it, verbatim: "background it and poll `BashOutput` to completion before you end your turn … each poll resets that clock." The session's deferred-tool list carried Monitor, TaskOutput and TaskStop and no BashOutput. Across 8,546 archived agent transcripts spanning CLI 2.1.222 to 2.1.267 there is no `BashOutput` or `KillShell` tool call, neither name ever appears in a `deferred_tools_delta`, and 138 `ToolSearch` requests naming them were all answered with TaskOutput and TaskStop instead. The agent improvised a foreground `ps -o etime` check every 39 seconds on average. That block cost about $1.71 of a $10 budget. The suite it was waiting on never finished. The verifier's remedy, Monitor, is the wrong tool by the harness's own description of Monitor and by this repo's wait guidance, both of which say a single "tell me when this finishes" is not what Monitor is for.
2. **The envelope-markup rejection was a textbook instance of the documented over-consumption shape, contained by the guard in one turn.** The `decision` argument of an `add_design_decision` call ended with a name-dialect closer for `decision`, then the opener for `rationale`, then the rationale text: the parser had swallowed the sibling parameter. The guard rejected it, returned `repaired_call` with `rationale` recovered, and the agent resubmitted that map byte-identically eight seconds later. It succeeded. The subject was verdict precedence in `df_pytest_isolation.py`, nothing to do with markup, so this is the genuine-leak class, not the deliberate-quotation class. The verifier's note that this is "a different tool than the previously catalogued submit_task leak" is not accurate: the catalogue entry already carried ten plan-tools and add_memory sightings before this one, and task 5055's own journal census already lists this exact event.

### Origin × manifestation matrix

The runner's matrix at the top of this report renders the verifier's stamps, which place both sightings at implement × implement. The table below carries the refined stamps this synthesis establishes in §1.2. Both are reported so the difference is visible rather than reconciled silently.

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| architect | · | 1 | · | · | · | · | · | · | · | **1** |
| implement | · | · | 1 | · | · | · | · | · | · | **1** |
| **total** | **0** | **1** | **1** | **0** | **0** | **0** | **0** | **0** | **0** | **2** |

Readings, observational. Both sightings originate and manifest in the same phase, in the same session, within one tool call of each other. The `merge` and `verify` columns are zero for a ninth consecutive cycle. The PRD's motivating architect/implement→merge hypothesis remains untested by the ten post-07-24 corpora, which total 12 findings.

### 1. Verified clusters

#### 1.1 Role wait-guidance names `BashOutput`, a tool absent from every CLI build in the corpus; the agent improvises foreground `ps` polls (1 sighting, session `316ede03`)

**The trace, from the archived transcript** at `data/orchestrator/agent-transcripts/3651/…/316ede03-d04c-4c71-b04c-2aa16320227c.jsonl`, 583 records. Times are UTC.

| time (09-08) | record | action | result |
|---|---|---|---|
| 00:41:24 | 330 | Bash, `run_in_background: true`: `cd …/3651/orchestrator && uv run pytest tests -q 2>&1 \| tail -60` | "Command running in background with ID: bty7mux8k … You will be notified when it completes" |
| 00:42–01:00 | 334–491 | files a follow-up ticket and esc-3651-4, reviews its diff, edits, commits; nine `pgrep`/`ps` liveness checks interleaved with that work | output file is 0 bytes at all six checks |
| 01:01:22 | 495 | first of 16 identical foreground Bash calls: `ps -o etime= -p 2388707 2>/dev/null \|\| echo "FINISHED"` | `19:55` |
| 01:01–01:11 | 503–569 | 15 more; from record 514 to 569, 14 consecutive with nothing else between them | etime climbs to `29:46`, never `FINISHED` |
| 01:11:49 | 574–575 | "Budget is nearly exhausted. Terminating the background run cleanly" then `kill 2388668 2388672 2388707` | `TERMINATED` |
| 01:12:58 | 582 | final text: "The full `orchestrator/tests` run did not finish. I killed it at ~30 min with my USD budget exhausted. It never printed output (buffered through `tail`)" | session ends |

| poll block (records 495–569) | value |
|---|---|
| exact polls | 16 |
| wall clock spanned | 586.6 s |
| mean gap between polls | 39 s (min 21, max 67) |
| budget remaining before first poll | $1.81 |
| budget remaining after last poll | $0.10 |
| cost of the block | ≈ $1.71, ≈ $0.09–0.12 per poll turn |
| launch to kill | 1825.5 s |

**What the agent was told.** The turn prompt is built by `orchestrator/src/orchestrator/agents/briefing.py::build_amender_prompt`, which interpolates `orchestrator/src/orchestrator/agents/roles.py::WAIT_PATTERN_REMINDER`. Record 4 carries it verbatim: run verification in the foreground with an explicit timeout if it finishes inside about 25 minutes, otherwise "background it and poll `BashOutput` to completion before you end your turn — the deciding limit is the working-regime watchdog, which kills this SESSION after ~30 minutes with no new turn … and each poll resets that clock." The reminder points at "the wait rules above", which live in the implementer system prompt as `BACKGROUND_WAIT_GUIDANCE`; the system prompt is not stored in the transcript, but `orchestrator/tests/test_roles_wait_pattern.py` pins that the implementer role carries it. That block names `BashOutput` and `KillShell` in five more places and names no poll cadence.

**The tool does not exist, and the repo's own record says it was never measured.** Record 5's `deferred_tools_delta.addedNames` lists 19 built-ins including Monitor, TaskOutput and TaskStop, and no BashOutput. The session's one `ToolSearch` call was for two MCP tools; it never asked for BashOutput, Monitor or TaskOutput. Fleet-wide, over `data/orchestrator/agent-transcripts/`:

| window | files | `BashOutput` tool_use | `KillShell` tool_use | files that ToolSearched for either | results returning either | `Monitor` tool_use files | `TaskOutput` tool_use files |
|---|---|---|---|---|---|---|---|
| 2026-08-05 to 08-31 | 6,371 | 0 | 0 | 94 | 0 | 6 | 114 |
| 2026-09-01 to 09-10 | 2,175 | 0 | 0 | 44 | 0 | 145 | 54 |

Every one of the 138 `select:` requests naming BashOutput or KillShell was answered with a `matches` list containing only TaskOutput, TaskStop or Monitor; a free-text query for `BashOutput` returned an empty list. The CLI versions seen run from 2.1.222 through 2.1.267, the latter being the currently installed bundle. The comment above `WAIT_PATTERN_GUIDANCE` in `roles.py` is headed "TOOL AVAILABILITY -- measured 2026-08-05, do NOT re-derive" and records that `ToolSearch("select:Monitor,TaskOutput")` returned both schemas from inside a dispatched session, then extends the conclusion: "The same holds for the `BashOutput`/`KillShell` that BACKGROUND_TASK_WARNING has named since task 2761." The first sentence is a measurement. The second is not, and it is false for every build in the corpus. The two names entered the repo on 2026-07-18 with task 2761 and were carried into the wait guidance by task 3607 on 2026-08-05/06; the file was last touched 2026-09-04. Outside `roles.py`, the names occur only in `shared/src/shared/cli_invoke.py` (a `--disallowed-tools` deny-list and the `_BACKGROUND_REAP_TOOLS` set used by `detect_ended_awaiting_background`, which also includes TaskOutput and TaskStop and therefore still works) and in synthetic fixtures in `shared/tests/test_cli_invoke_background.py`. No document or transcript attests the tool.

**Correction to the verified framing.** The verifier attributed the polling to Monitor being a deferred tool the agent never discovered, and cited the Bash tool description's "use Monitor with an until-loop to wait on a condition." Three things stand against that. The Monitor tool's own description, loaded this session, says that for one notification ("tell me when the build finishes") the right shape is Bash with `run_in_background` and a command that exits when the condition is true, and reserves Monitor for a recurring event feed. `roles.py::WAIT_PATTERN_GUIDANCE` says the same, and that bullet was written after the 08-16 census. And the agent did background the run exactly as both sources prescribe; what it lacked was a way to stay alive while the harness's completion notification was pending, since task 2761 established that a headless `--print` session that ends its turn on a background command exits and abandons the work. The prompt's answer to that was "poll `BashOutput`". The affordances that did exist in the session for a single blocking wait were `TaskOutput` with `block=true` and a timeout of up to 600 s, whose description now reads "DEPRECATED", and a foreground Bash wait bounded by an explicit `timeout`, which the guidance sanctions for anything under about 25 minutes. Whether either would have been better is not established by this sighting; what is established is that the agent was pointed at neither.

**What the polling cost, and what it did not.** The prompt's stated purpose for polling is to reset a 1800 s idle watchdog. A poll every 39 s serves that purpose about 46 times more often than needed, and the prompt gives no cadence. The block consumed about $1.71 of the remaining $1.81, and the agent killed the run because the budget, not the watchdog, was exhausted. Whether the suite would have finished under a cheaper wait is not knowable from this session: the orchestrator full suite has been measured at 3,842 to 5,087 s under host load (task 3651's own dry-run record), the run was killed at 1,825 s, and the `| tail -60` pipe meant no output would reach the file before exit in any case. The agent's thinking blocks are empty strings, so its reasoning for the cadence is not recoverable; its Bash descriptions read "Poll pytest" and nothing in its text mentions BashOutput, Monitor, the watchdog or the clock.

**Relation to the codebook, observation not a merge.** No entry or candidate covers this shape: the codebook has zero hits for `BashOutput`, `busy-poll` or `ps -o etime`, and no item contains both `watchdog` and `poll`. It is not a sighting of `entry-cand-20260721-1` (deferred tool called before ToolSearch loads its schema, 19 sightings, latest 09-05): no deferred tool was called here at all. It is not a sighting of the watcher loop-mechanics entries, whose polls are of an escalation queue by design. The nearest relatives are the two done tasks that authored the guidance, 2761 and 3607, and the 08-16 census whose §1.1 companion sits directly under this text in `roles.py`.

#### 1.2 Envelope over-consumption swallows `rationale` into `decision`; guard rejects, agent resubmits `repaired_call` verbatim, one turn lost (1 sighting, session `ad12f696`)

**The trace, from the archived transcript** at `data/orchestrator/agent-transcripts/4823/…/ad12f696-c584-4b30-9867-3dd90b7b2d01.jsonl`, 93 records. Notation follows `docs/mcp-toolcall-xml-leak.md`: `&#60;` stands for a literal angle bracket.

| time (09-07) | record | action | result |
|---|---|---|---|
| 03:33:21 | 71 | `add_design_decision` with input keys `["decision"]` only; the 2,046-char value ends `…per-clock verdicts.&#60;/decision>\n&#60;parameter name="rationale">Fixes the confirmed masking defect found in review…` | rejected |
| 03:33:28 | 72 | error: `mcp_markup_detected`, field `decision`, matched_pattern `&#60;parameter name=`, misclose `&#60;/decision>`, `recovered_params: ["rationale"]`, `repaired_call` present, hint "Resubmit repaired_call verbatim" | |
| 03:33:36 | 76 | `add_design_decision` with keys `["decision","rationale"]`, 344 and 1,662 chars, equal to `repaired_call` key for key | `{"status":"ok","total_decisions":7}` at 03:33:51 |

**Verified mechanism.** This is the shape `plans/toolcall-markup-containment-prd.md` §2.1 and `docs/mcp-toolcall-xml-leak.md` §1 describe: the model closed `decision` in the name dialect instead of the canonical closer, the harness parser did not find the closer it expected and over-consumed to the next terminator, and the sibling parameter `rationale` was dumped into `decision`'s value and dropped from the call. The guard at `shared/src/shared/mcp_markup_middleware.py::MarkupGuardMiddleware` refused it and recovered the swallowed parameter, which is why `recovered_params` is non-empty; the plan-tools journal `data/orchestrator/markup-guard/plan-tools.jsonl` holds the matching fact at 03:33:27Z with `subject_task_id 4823`, `pattern &#60;/decision>`, `agent_id null`, `project null`. The decision's subject is `df_pytest_isolation.py::deploy_clock_change_report` verdict precedence; the session never discusses markup, so it belongs to the genuine-leak class that task 5055 §3 distinguishes from deliberate quotation. The cost is one turn and 15.5 s between the two calls. The retry followed the hint literally and succeeded first time.

**Phase refinement.** The prompt is built by `orchestrator/src/orchestrator/workflow.py::_replan` and dispatched under the architect role after a blocking review. Origin `architect`, manifested `architect`. The verifier's implement × implement names the pipeline neighbourhood, not the session role.

**Correction to the verified framing.** "A different tool than the previously catalogued submit_task leak" understates the catalogue. `oneoff-2026-07-09` carried, before this sighting, ten sightings on `add_design_decision`, `add_reuse_item`, `add_memory` and `submit_review_verdict`. Task 5055 was filed on 2026-09-03 for the general fix, records this event in its journal census ("4823 add_design_decision 1"), and had its architect route decided on 2026-09-07. The verifier's cause field restates its summary and adds no causal claim.

**Structural facts about this sighting, recorded as observations.** Task 5055's evidence block states that once a session starts leaking it keeps leaking across retries, citing 7 and 10 rejections in single sessions. This session leaked once in 14 assistant messages and was clean on the verbatim retry. It is one data point against treating per-session persistence as universal, not a refutation. The codebook's sighting on `29487354` (six consecutive rejections with quoted technical content in `rationale`) and this one bracket the class: a pure envelope over-consumption on clean prose repairs in one turn; a value that itself contains envelope-like substrings does not.

**Relation to the codebook.** Already present twice, both trickle-minted on 09-07 before this census: a sighting on `oneoff-2026-07-09` (its 12th; the sighting's manifested stamp is `unknown`) and on `cand-20260907-18` "MCP markup injection endemic to design-tool codepath", pending. Entry profile at synthesis time:

| `oneoff-2026-07-09` | value |
|---|---|
| status / filed_tasks | mined-unverified / none |
| entry-level stamps | unknown / unknown |
| sightings | 12, dated 08-07 to 09-07 |
| sighting manifested stamps | implement 7, architect 3, review 1, unknown 1 |

Twenty-nine pending candidates match "markup" or "envelope", of which about eighteen are this defect; none is promoted or rejected, and none is cross-referenced to the entry or to tasks 5055 and 5283.

### 2. One-off sightings

None beyond the two clusters above, each single-sighting this cycle.

### 3. Cross-cutting observations

1. **A comment forbidding re-derivation protected a claim that was never derived.** The `roles.py` block says "measured 2026-08-05, do NOT re-derive" and "do NOT 'fix' this block by trimming those tools out." The measured half (Monitor, TaskOutput) is true; the extended half (BashOutput, KillShell) is contradicted by 94 transcripts from the same August window. Two later edits (the 08-16 companion constant and the 09-04 touch) cite the block as settled. The prose has been read as the measurement.
2. **The verification stage confirmed surfaces, not causes, for the second cycle running.** For 1.1 it checked the Bash tool description on current main and proposed Monitor; the transcript shows the agent following a briefing that names a nonexistent tool, and the harness's own Monitor description declines the role the verifier assigned it. For 1.2 it called the tool "different from the catalogued one"; the catalogue and the owning task both already list it. The 09-05 synthesis recorded the same pattern.
3. **A census finding can duplicate a trickle sighting minted days earlier.** `ad12f696` was in the codebook in two places before verification started. The runner does not check session presence before spending a verifier call. `316ede03`, by contrast, was absent and is the cycle's only new information.
4. **The codebook was rewritten by the merger while this synthesis was reading it.** Between reads it went from 23,632 to 23,466 lines; `cand-20260909-6` disappeared and three promoted entries appeared. All figures here are from the snapshot with mtime 04:29Z. A synthesis that quotes line numbers from an earlier read would now be wrong.
5. **Prose guidance was present in both sessions and shaped both outcomes.** In 1.1 the guidance was followed and the named tool was missing. In 1.2 the guard's remedy text was followed and worked. This is recorded as evidence about what remedy text can and cannot do, not as a request for more of it.

### 4. Remediation candidates

1. **Correct the wait guidance to name tools that exist, and record the 08-05 claim as unmeasured.** Surface: `orchestrator/src/orchestrator/agents/roles.py` (`BACKGROUND_TASK_WARNING`, `WAIT_PATTERN_GUIDANCE`, `WAIT_PATTERN_REMINDER`, and the "TOOL AVAILABILITY" comment), with `orchestrator/tests/test_roles_wait_pattern.py` and `shared/src/shared/cli_invoke.py::_BACKGROUND_REAP_TOOLS` in scope. Signal: no role prompt names a tool absent from the CLI's built-in or deferred tool set, the guidance states a poll cadence relative to the idle bound it cites, and the availability claim is pinned by a fixture that exercises the named tools rather than by prose. Ownership check: two `search_tasks` queries returned only done tasks 2761 and 3607 and pending 3606, which covers watcher re-arm loops, not this. That search is the extent of the check; it is not a claim that no owner exists.
2. **No task for 1.2.** Task 5055 owns the general fix and already lists this event; task 5283 owns the containment follow-ups. One sighting adds no fix surface.

### 5. Codebook dispositions (input to the merger; promote/reject in place, never delete)

| Cluster / finding | Suggested disposition |
|---|---|
| 1.1 Wait guidance names `BashOutput`; agent polls with foreground `ps` | Mint a new pending candidate. Session `316ede03-d04c-4c71-b04c-2aa16320227c`, date 2026-09-07, origin `implement`, manifested `implement`, area `orchestrator-prompt / harness-affordance`. Evidence quote should carry the reminder sentence naming `BashOutput`, the 16-poll block with its 39 s cadence, and the `deferred_tools_delta` list lacking the name. Record three discriminators: the agent backgrounded the run as prescribed; no deferred tool was called, so this is not `entry-cand-20260721-1`; the run ended on budget, not on the watchdog. |
| 1.2 Envelope over-consumption on `add_design_decision` | Do **not** mint; do **not** append a third sighting. The session is already on `oneoff-2026-07-09` and `cand-20260907-18`. Consider refining the existing `oneoff-2026-07-09` sighting's manifested stamp from `unknown` to `architect`, and the entry's modal stamps from unknown/unknown toward its sightings (implement 7, architect 3). Consider adding `filed_tasks: 5055, 5283` to the entry, which has none despite an owned PRD. |
| `cand-20260907-18` and the ~17 pending markup candidates | Observation, no disposition requested: all are the `oneoff-2026-07-09` class and none names the entry or its owning tasks. |

### 6. Method notes for the next census

- **Carry-forwards from 09-05.** Task 3606 remains `pending`, last updated 08-21. `cand-20260806-12` and `cand-20260812-19` were not re-checked. The self-ingestion watch on `entry-cand-20260729-4` was not checked this cycle.
- **Check codebook presence before verifying.** One of two findings was already catalogued twice. A session-id grep over the codebook costs nothing and would have saved one verifier call.
- **Read the codebook once, from a recorded snapshot.** The merger can run during synthesis; record the mtime and line count at the first read and re-check before quoting line numbers.
- **Watch: whether 1.1 recurs after the guidance changes.** The fleet corpus grep is cheap: `"name":"BashOutput"` should stay at zero either way; `ps -o etime` and `pgrep` files since 09-01 stand at 20 and 170 and are the baseline.
- **Watch: self-ingestion onto the new candidate.** This session and today's verifier session both quote the `ps -o etime` command and the reminder text, and both will be in tomorrow's trickle window.
- **Codebook scale at synthesis time:** 105 entries; 752 candidates, of which 641 pending, 79 promoted, 32 rejected; 66 minted since 09-05 (13 on 09-06, 20 on 09-07, 31 on 09-08, 2 on 09-09).

*Synthesis note to the runner: written from the 2 verified findings supplied. Mechanism claims verified by reading both archived transcripts (session `316ede03`, records 4, 5, 330–582 including budget attachments and the `deferred_tools_delta`; session `ad12f696`, records 58–91 including the `repaired_call` equality check), `orchestrator/src/orchestrator/agents/roles.py` lines 355–626, `orchestrator/src/orchestrator/agents/briefing.py::build_amender_prompt`, `orchestrator/tests/test_roles_wait_pattern.py`, `shared/src/shared/cli_invoke.py` lines 355–396, 730–739 and 3055–3058, `docs/mcp-toolcall-xml-leak.md` §1, `plans/toolcall-markup-containment-prd.md` §2.1, the plan-tools markup journal, and the Monitor and TaskOutput tool descriptions loaded in this session. Fleet counts from `grep` over `data/orchestrator/agent-transcripts/` partitioned by mtime; the installed CLI bundle at `~/.local/share/claude/versions/2.1.267` was not readable from the sandbox, so its contents are inferred from transcripts only. Codebook relations verified on the 04:29Z snapshot: `oneoff-2026-07-09`, `cand-20260907-18`, `entry-cand-20260721-1`, `cand-20260721-1`, and the absence of both session ids where stated. Task status via fused-memory `get_task` for 3651, 5055 and 5283; ownership via `search_tasks`. No tasks filed and no codebook edits made from this synthesis; filing and merger application are the runner's steps.*


## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=40, sonnet verify=2, fable synthesis=1, haiku headroom-probe=2
