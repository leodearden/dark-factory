# confusion census 2026-09-15

Project: dark_factory

## Saturation

- batches: 2
- stop reason: saturated
  - batch 0: dup_rate=0.94 (total=20, succeeded=18, failed=2, saturated=True)
  - batch 1: dup_rate=0.94 (total=20, succeeded=17, failed=3, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | verify | recon |
| --- | --- | --- |
| verify | 1 | 0 |
| unknown | 0 | 1 |

## Synthesis

**Date:** 2026-09-15
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests, per-finding verification against current main (Sonnet), then this synthesis (Fable). Two findings reached synthesis. This document adds a read of both transcripts, the orchestrator's `runs.db` invocation and event rows for the second session, the reviewer-trial campaign log for the first, the role-prompt and watchdog sources, the fleet transcript archive, and the codebook. Every mechanism claim names its evidence. Where the evidence contradicts the verifier's framing, the correction is stated explicitly.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml`. Dispositions in §5 are inputs to the merger.
**Run notes:** eleventh completed periodic census, at the PRD's 5-day hard floor after 2026-09-10. No report dated today pre-existed on disk and `census-state.json` still reads 09-10 (done count 3798). Saturation statistics and filed-task ids are appended by the runner outside this synthesis. This synthesis ran in a sandbox that granted no approvals for Python, heredocs, `journalctl`, `dmesg`, or reads under `~/.claude` other than the Read tool on transcript files, so every fleet figure below is a `grep` count and the kernel log was not consulted.

### Corpus

- **2 verified findings, 2 sessions, 2 sightings.** Both on CLI 2.1.268. Neither session id was present in the codebook at synthesis time.
- **Session `738f85df`** is not an orchestrator pipeline session. It is one reviewer invocation of the Fable evidence-package reviewer-trial campaign: `plans/fable-evidence-package-2026-09-10/campaign/run-trial1.log` records its completion at 05:41:53 local as `variant_sonnet5_solo` against corpus diff `mined_1225`, `claude-sonnet-5`, effort high, budget $5, cwd the main checkout on `main`, cost $1.03, wall 999.6 s, 196 records. Its transcript exists only under `~/.claude/projects`; there is no archive copy under `data/orchestrator/agent-transcripts`. The verifier labelled it "harness/tooling — Bash tool static-analysis pre-check" and stamped unknown × recon.
- **Session `9c959b2f`** is task 4448's implementer re-dispatch after esc-4448-5 (scope_violation), in `.worktrees/4448`, 2026-09-11 17:28:19Z to 19:05:29Z, `claude-opus-5` at effort max, budget $20, 600 records. The verifier labelled it "verify" and stamped verify × verify.
- **Composition:** two harness-boundary classes. One is a re-sighting of a pending candidate minted three days earlier, whose mechanism is a user-level PreToolUse hook rather than anything in the Bash tool. The other is the orchestrator's own working-idle watchdog, observed from inside the session it killed, and it is the exact case the wait guidance in `roles.py` describes.
- **Phase-stamp coverage:** the verifier's stamps carry 1 unknown of 4. The transcripts refine both: §1.1 is a review-role session whose cause lives in operator configuration (origin `ops`, manifested `review`); §1.2 is an implementer session killed during its own test run, which by the role convention this census series adopted on 09-10 is implement × implement.

### Executive summary (observations)

1. **The "malicious input" verdict came from a command the agent never issued.** The reviewer asked for `tail -5 fused-memory/tests/test_stages.py`. A PreToolUse hook registered in `~/.claude/settings.json` (`skim-rewrite.sh`, documented in `skills/spawn/hooks/README.md` §"Absolute paths") rewrote the call to `skim fused-memory/tests/test_stages.py --mode=pseudo --last-lines 5`; the transcript carries the rewrite as a separate `hook_success` attachment with `updatedInput`, which the model does not see. The tool result was the rewritten command's exit 1 and its "Too many AST nodes: 100001 (max: 100000)" message. The agent then ran `wc -l`, which the hook left alone, got 16,021 lines, and moved on without ever reading the tail. Fleet-wide the hook has run in 9,573 of 14,004 archived transcripts and rewritten at least 1,101 commands (562 `--max-lines`, 539 `--last-lines`); 10 archived transcripts carry the AST-cap message, 17 occurrences. Nothing in the Bash tool parses the target file; the verifier's "static-analysis pre-check" is not the mechanism.
2. **The SIGKILL was the orchestrator's working-idle watchdog, and the agent had been told about it.** The implementer ran the full fused-memory suite as a blocking foreground Bash call with a 2,400,000 ms timeout. The call produced no new assistant turn; 1,832.7 s after the last turn the session was killed. `runs.db` records the invocation as `error_timeout_killed_with_progress`, `timed_out: true`, 173 transcript turns, cost 0.0. The idle bound is `max(working_idle_secs=1800, implementer ceiling=1200)` = 1800 s, polled at 60 s, so 1,832 s is the expected kill time to within one poll. The Bash tool's own timeouts render as "Command timed out after Nms" (1,066 occurrences fleet-wide); a bare "Exit code 137" is not that. The OOM reading is neither confirmed nor excluded here, because the kernel log was unreadable from this sandbox, but the orchestrator's own record names a different killer. The wait guidance in `orchestrator/src/orchestrator/agents/roles.py::WAIT_PATTERN_GUIDANCE` describes this exact shape: "a 45-minute verify sized to a 2700000 timeout sits well under the harness cap and still gets the session killed around 30 minutes."

### Origin × manifestation matrix

The runner's matrix at the top of this report renders the verifier's stamps (unknown × recon, verify × verify). The table below carries the refined stamps this synthesis establishes. Both are reported so the difference is visible rather than reconciled silently.

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| implement | · | · | 1 | · | · | · | · | · | · | **1** |
| ops | · | · | · | · | 1 | · | · | · | · | **1** |
| **total** | **0** | **0** | **1** | **0** | **1** | **0** | **0** | **0** | **0** | **2** |

Readings, observational. The `ops` origin is new to this series: the cause of §1.1 is a file in the operator's home directory, applied to every session on the host, including orchestrator-dispatched ones. The `merge` column is zero for a tenth consecutive cycle; the `verify` column is zero under the role convention and one under the verifier's. The PRD's architect/implement→merge hypothesis remains untested by the eleven post-07-24 corpora, which total 14 findings.

### 1. Verified clusters

#### 1.1 A user-level PreToolUse hook rewrites `tail`/`head` into `skim`, whose node cap fails on large test files with a "malicious input" message; the agent sees neither the rewrite nor the real output (1 sighting, session `738f85df`; same defect as `cand-20260912-21`)

**The trace, from the transcript** at `~/.claude/projects/-home-leo-src-dark-factory/738f85df-b948-46a6-b017-7292e4d85c34.jsonl`. Times are UTC, 2026-09-11.

| time | record | action | result |
|---|---|---|---|
| 04:38:28.972 | 158 | Bash `tail -5 fused-memory/tests/test_stages.py`, "Check tail of test_stages.py against diff context" | |
| 04:38:29.017 | 162 | `PreToolUse:Bash` hook `/home/leo/.claude/hooks/skim-rewrite.sh`, 39 ms, `updatedInput.command` = `skim fused-memory/tests/test_stages.py --mode=pseudo --last-lines 5` | attachment record; not model-visible |
| 04:38:46.832 | 163 | tool_result, `is_error: true` | `Exit code 1` / `Error: Failed to parse source code: Too many AST nodes: 100001 (max: 100000). Possible malicious input.` |
| 04:38:49.139 | 167 | Bash `wc -l fused-memory/tests/test_stages.py` (not rewritten) | `16021` at 04:38:59 |
| after | 172, 177, 189 | two Grep calls, then `submit_review_verdict` | verdict submitted |

The rewritten command ran for 17.8 s. The agent's thinking blocks are empty strings, so its reading of the error is not recoverable; its next action was a size check, and it did not retry the read by any other tool.

**Verified mechanism.** The hook is registered as `PreToolUse: Bash → skim-rewrite.sh` in `~/.claude/settings.json`; `skills/spawn/hooks/README.md` §"MERGE, never clobber" lists that registration as pre-existing operator configuration outside the repo. The binary is `~/.cargo/bin/skim`. The hook script, the settings file and the binary could not be read or executed from this sandbox, so where inside `skim` the cap lives and whether the message is `skim`'s own or a library's is not established. What the transcript establishes is: the command that ran was not the command the model wrote; the rewrite is recorded only in an attachment the model does not receive; and the result the model received names neither `skim` nor the rewrite. The file was 16,021 lines at the time (16,688 lines, 755 KB on main today).

**Fleet counts, from `grep` over `data/orchestrator/agent-transcripts/`:**

| measure | value |
|---|---|
| archived transcripts | 14,004 |
| transcripts in which `skim-rewrite.sh` ran | 9,573 |
| rewrites to `--mode=pseudo --max-lines` (head/cat shapes) | 562 |
| rewrites to `--mode=pseudo --last-lines` (tail shapes) | 539 |
| transcripts carrying "Too many AST nodes" | 10 (17 occurrences; 6 in task 3731's `a39b2d73`) |
| plus this session, outside the archive | 1 |

**Corrections to the verified framing.** (a) "Bash tool's source-parsing pre-check" names the wrong layer: the Bash tool ran what the hook handed it. (b) "rejects a benign read-only command" is not what happened: the command was replaced, then the replacement failed; nothing was rejected. (c) The fused-memory memory record `518379c8` (2026-07-24, "Bash text ops on very large source files fail … use Read/Grep") and its 2026-09-14 amendment from task 3731 ("harness-side RESULT RENDERING, not execution; the command still runs") both describe this hook without naming it. The 3731 transcript shows why the amendment saw a `sed -i` apply and a chained `tail -8` vanish: the hook's `updatedInput` rewrote only the `tail -8` segment of the `&&` chain to `skim … --last-lines 8`, so `sed` and `wc` ran, `skim` exited 1, and the trailing `grep` never ran. That is ordinary `&&` semantics on a rewritten command, not result rendering. (d) `cand-20260731-20` ("subagent shell inherits parent aliases … head → skim", session `6dd54f56`) attributes a `skim` usage error on `head -30 file1 file2` to bashrc aliasing; the hook is the sufficient and documented mechanism, and no alias is needed. That candidate was not re-read at transcript level this cycle.

**Relation to the codebook.** `cand-20260912-21` ("Bash tool pre-check fails on AST node limit even for commands needing only a file prefix", pending, 1 sighting, session `f8a0e99f`, `head -120 orchestrator/tests/test_merge_queue.py`) is the same defect; that transcript's `updatedInput` reads `skim orchestrator/tests/test_merge_queue.py --mode=pseudo --max-lines 120`, confirming the mechanism for the prior sighting too. Its title and cause carry the same wrong layer as the verifier's. Four codebook notes (around `cand-` items at lines ~3617 to ~3756 of the 03:42Z snapshot) describe `skim` "inserted" into multi-line commands as a "flattening bug"; they were not verified this cycle and are listed in §6 as a watch item, since a hook that rewrites the command string is a candidate explanation.

#### 1.2 A foreground Bash call sized above the working-idle watchdog: the session is killed at 1800 s without a new turn, the agent's last record is a bare `Exit code 137`, and the spend is recorded as zero (1 sighting, session `9c959b2f`)

**The trace, from the archived transcript** at `data/orchestrator/agent-transcripts/4448/…/9c959b2f-f614-4eb3-8168-4fb9e7650912.jsonl` and the orchestrator's `data/orchestrator/runs.db`. Times are UTC, 2026-09-11.

| time | source | event |
|---|---|---|
| 17:28:19.96 | `invocations` 27410 | implementer invocation starts (after steward esc-4448-5 at 17:28:58) |
| 17:28:48.78 | record 1 | first transcript record |
| 17:28 to 18:34 | records 24 to 589 | 100-odd Bash calls: RED/GREEN commits, targeted pytest runs of 5 s to 2 min, ruff |
| 18:34:53.47 | record 595 | budget attachment: $8.64 of $20 |
| 18:34:56.25 | record 596 | Bash, foreground, `timeout: 2400000`: `uv run python -m pytest fused-memory/tests/ -q -m "not integration" -p no:randomly 2>&1 \| tail -6`, "Run the full fused-memory suite" |
| 19:05:28.92 | record 598 | tool_result `Exit code 137`, `is_error: true` |
| 19:05:29.19 | record 600 | budget attachment: $8.73 of $20; last record |
| 19:05:29.82 | `events` | `invocation_end`, `subtype: error_timeout_killed_with_progress`, `timed_out: true`, `transcript_turns: 173`, `turns: 0`, `success: false` |
| 19:05:29.82 | `invocations` 27410 | `duration_ms 5829816`, `cost_usd 0.0` |
| 19:05:48.28 | `invocations` 27415 | implementer re-dispatched; completes 19:55:35, $3.11 |

| measure | value |
|---|---|
| last assistant turn to kill | 1,832.7 s |
| Bash timeout requested | 2,400 s |
| idle bound, `max(working_idle_secs, implementer ceiling)` | max(1800, 1200) = 1,800 s |
| working-regime poll cadence | 60 s |
| session wall clock | 5,829.8 s |
| absolute cap | 7,200 s |
| gate-measured duration of the fused-memory suite (`plans/verify-speed-study-df-2026-09-10.md` §module table) | 342 s for 19,485 tests |

**Verified mechanism.** `orchestrator/src/orchestrator/config.py::TimeoutsConfig.working_idle_secs` (default 1800) documents the working regime: once one assistant turn has been seen, the watchdog kills only after no new turn for `max(working_idle_secs, per-role ceiling)`, bounded by `OrchestratorConfig.invocation_timeout` (7200). The implementer ceiling is 1200 in the same class. `shared/src/shared/cli_invoke.py::_run_subprocess` polls the transcript every 60 s in this regime (`_WATCHDOG_WORKING_POLL_SECS`) and on expiry sends SIGTERM to the CLI, waits 5 s, then SIGKILLs the process group; the comment at the timeout-classification site says the SIGTERM-first order exists so the CLI flushes its result on the way out, which is consistent with the `Exit code 137` result and the final budget attachment being present in the transcript. Which process in the `pytest … | tail -6` pipeline received SIGKILL first is not recorded; 137 is what the tool reported for the child. The `| tail -6` pipe means no partial output could exist, so whether pytest was hung or merely slow under load is not recoverable from this session. The same task's later invocation 27421 ran 7,203,022 ms and was killed at the absolute cap, also with `cost_usd 0.0`; that is a sibling ceiling, not this sighting.

**Corrections to the verified framing.** (a) The premise that the tool's own timeout "would show a different exit/signal" holds on fleet evidence: bare `Exit code 137` occurs 47 times in the archive and never with timeout text, while the Bash tool's own timeouts render as `Command timed out after Nms` (775 plus 291 occurrences at 30,000 ms alone). The inference from that premise to "an OOM condition or a resource cap enforced outside the agent's visibility" is not supported: the orchestrator's own event row names the killer. Kernel OOM evidence could not be read from this sandbox, so an OOM contribution is not excluded, only unneeded. (b) "The agent has no signal" understates what the briefing carried. `roles.py::WAIT_PATTERN_GUIDANCE` says, for implementers, that the binding constraint is not the harness Bash cap but the idle watchdog, gives the 1800 s figure, and gives this exact example. The system prompt is not stored in the transcript; `orchestrator/tests/test_roles_wait_pattern.py` pins that the implementer role carries the block (per the 09-10 report). What the agent did was follow the first half of that guidance (foreground with an explicit timeout) while sizing the timeout above the ceiling the second half names. The same block still tells the agent to poll `BashOutput`, the nonexistent tool the 09-10 census reported in its §1.1; no task was filed from that report.

**Fleet context, from `runs.db` `events` since 2026-09-01:** 70 timed-out `invocation_end` rows, of which 36 are `error_timeout_killed_with_progress` (35 opus, 1 sonnet), 31 `error_empty_output`, 3 steward escalation ends. By day the with-progress kills ran 0, 0, 0, 1, 7, 1, 4, 2, 0, 0, 7, 10, 2, 2, 0 (09-01 to 09-15), against 24 to 303 invocation ends per day. Fourteen archived transcripts dated 09-11 to 09-15 carry an `Exit code 137` (tasks 3541, 3730, 4448, 4878, 5024, 5028 ×2, 5032 ×2, 5103, 5147, 5283, 5342, 5383); they were not classified this cycle and are the baseline for the next.

**Relation to the codebook.** No entry or candidate carries this session. The nearest item is `cand-20260905-3` ("Long-running process timeout leaves log state missing", session `a25e08c4`, "pytest exit 137 after 50-minute timeout", implement × implement, pending); the 50-minute figure is not the idle bound, so it may be the absolute cap or the tool's own timeout, and it was not re-read this cycle. The watcher-loop items citing exit 137 are the 120 s default Bash kill, a third ceiling.

### 2. One-off sightings

None beyond the two clusters above, each single-sighting this cycle.

### 3. Cross-cutting observations

1. **Both causes live outside the repository and outside the model's view.** §1.1's cause is a hook in `~/.claude`; §1.2's is a ceiling in orchestrator config. In both sessions the mechanism left a record the model does not receive (a `hook_success` attachment; an orchestrator event row) and handed the model a result that names something else (a parser message; a bare exit code).
2. **The verification stage confirmed surfaces, not causes, for the third cycle running.** For 1.1 it named the Bash tool; the transcript names the hook. For 1.2 it reasoned from the exit code to an external killer; the orchestrator's ledger was one query away. The 09-05 and 09-10 syntheses recorded the same pattern.
3. **Three existing memory and codebook records carried a wrong mechanism for 1.1.** The 07-24 procedural memory, its 09-14 amendment, and `cand-20260731-20` each explain a `skim` symptom without the hook. The 09-14 amendment's practical rule ("one Bash call, one operation; never chain a mutation with its verification") is sound for a different reason than it gives: the chain stops at the rewritten segment's exit 1.
4. **A session-id presence check would not have caught 1.1; a message-text check would have.** Neither session id was in the codebook, but `cand-20260912-21`'s evidence quote contains the identical error string. The 09-10 method note asked for a session-id grep before verifying; this cycle shows the grep needs the evidence text too.
5. **Eval-harness sessions enter the census corpus.** `738f85df` was produced by `orchestrator/src/orchestrator/evals/reviewer_trial/runner.py::_build_reviewer_prompt`, not by the task pipeline, and it qualified because its cwd is the project root. The finding is real and the hook fires in pipeline sessions too (9,573 archived transcripts), so the inclusion cost nothing here; it is recorded because a future finding from a trial session could describe the trial's prompt rather than production's.
6. **Killed invocations record `cost_usd 0.0` while the transcript's budget attachments show $8.73.** Observation about the ledger only; the 09-10 feedback memory on interrupted-episode spend already covers the interpretation.

### 4. Remediation candidates

1. **The `skim` rewrite hook.** Surface: `~/.claude/hooks/skim-rewrite.sh` and its `PreToolUse` registration in `~/.claude/settings.json`, both outside this repo; `skills/spawn/hooks/README.md` is the only in-repo description. Signals that would close the shape, any one of which is observable from a transcript: the rewrite does not fire on a file the tool cannot parse; or a `skim` failure falls back to the original command; or the tool result names the rewrite when it fails. Ownership check: `search_tasks` for "Bash tool Too many AST nodes static-analysis pre-check misreports large test file as malicious input" returned only AST-guard test tasks (5284, 5010, 4928, 5133, 4246, 5490), none about this. Whether a dark_factory task is the right vehicle for a change to a file in the operator's home directory is the runner's filing policy to decide; this synthesis records the surface and the signal.
2. **Wait guidance versus the idle watchdog.** Surface: `orchestrator/src/orchestrator/agents/roles.py` (`WAIT_PATTERN_GUIDANCE`, `WAIT_PATTERN_REMINDER`, `BACKGROUND_TASK_WARNING`), with `orchestrator/tests/test_roles_wait_pattern.py` in scope. This is the 09-10 §4.1 candidate with one more sighting: the guidance names a tool that does not exist and does not state that a foreground `timeout` above `working_idle_secs` is self-defeating in one sentence an agent sizing a timeout would read. Signal: no dispatched agent issues a foreground Bash call with `timeout` greater than the idle bound for its role; measurable from transcripts by comparing the `timeout` argument to the config value. Ownership check: `search_tasks` for the SIGKILL/OOM phrasing returned 1373 (done), 5138, 5147 (done), 1851 (done), 3955 (done), 5082, 1811 (done), 869 (done); none owns the guidance. That search is the extent of the check.
3. **No task for the ledger observation (§3.6) or for the orphaned `cost_usd 0.0` on killed invocations.** Existing feedback memory covers the reading; no defect is claimed.

### 5. Codebook dispositions (input to the merger; promote/reject in place, never delete)

| Cluster / finding | Suggested disposition |
|---|---|
| 1.1 `skim` rewrite hook | Do **not** mint. Append a sighting to `cand-20260912-21`: session `738f85df-b948-46a6-b017-7292e4d85c34`, date 2026-09-11, origin `ops`, manifested `review`, area `operator-hook / skim rewrite`. Evidence quote should carry the original `tail -5` command, the `updatedInput` value from the hook attachment, and the exit-1 message. Consider revising the candidate's title and cause to name the PreToolUse hook and `skim` rather than a Bash-tool pre-check, and adding a cross-reference to `cand-20260731-20` as the same mechanism under an alias attribution. Record two discriminators: the model never sees the rewrite; `&&` chains stop at the rewritten segment. |
| 1.2 Foreground Bash above the idle watchdog | Mint a new pending candidate. Session `9c959b2f-f614-4eb3-8168-4fb9e7650912`, date 2026-09-11, origin `implement`, manifested `implement`, area `orchestrator-watchdog / wait-guidance`. Evidence quote should carry the Bash call with `timeout: 2400000`, the bare `Exit code 137` result 1,832.7 s later, and the `invocation_end` row's `error_timeout_killed_with_progress`. Record three discriminators: not the Bash tool's timeout (no "Command timed out" text, and 1,832 s < 2,400 s); not the 7,200 s absolute cap; not the 120 s default Bash kill of the watcher entries. Note `cand-20260905-3` as a possible relative pending a transcript read. |
| `518379c8` memory record and its 09-14 amendment | Observation for the memory curator, not a codebook disposition: both describe this hook without naming it; the amendment's mechanism claim ("result rendering") is contradicted by the 3731 transcript's `updatedInput`. |

### 6. Method notes for the next census

- **Carry-forwards from 09-10.** Task 3606 remains pending. The 09-10 §4.1 wait-guidance candidate was not filed and has now recurred as §1.2. `cand-20260806-12`, `cand-20260812-19`, and the self-ingestion watch on `entry-cand-20260729-4` were not checked this cycle.
- **Check evidence text, not only session ids, before verifying.** One of two findings was already catalogued under a different session with the identical error string.
- **Read `runs.db` before reasoning about a kill.** `select data from events where event_type='invocation_end' and task_id=…` names the subtype; it took one query here and would have replaced the verifier's OOM inference.
- **Watch: the four "flattening bug" notes that mention `skim`** (codebook lines ~3617 to ~3756 in the 03:42Z snapshot). A hook that rewrites the command string is a candidate explanation for `skim` appearing mid-command; unverified.
- **Watch: whether 1.2 recurs after the guidance changes.** The fleet query is cheap: foreground Bash `timeout` values above 1,800,000 ms in implementer/debugger/architect transcripts, and `error_timeout_killed_with_progress` counts per day, which stood at 36 for 09-01 to 09-15.
- **Watch: self-ingestion onto both candidates.** This session and the verifier's both quote the `skim` command line and the exit-137 record, and both will be in tomorrow's trickle window.
- **Codebook scale at synthesis time** (snapshot mtime 03:42:08 +0100, 27,058 lines): 78 entries and 10 one-offs; 935 candidates, of which 825 pending, 78 promoted, 32 rejected; 143 minted since the 09-10 census (63 on 09-11, 23 on 09-12, 18 on 09-13, 39 on 09-14).

*Synthesis note to the runner: written from the 2 verified findings supplied. Mechanism claims verified by reading session `738f85df` (records 1 to 5, 157 to 168, tool-name index, and `run-trial1.log` around its completion line), session `9c959b2f` (records 594 to 600, the Bash-command and budget-attachment indexes), `data/orchestrator/runs.db` (`invocations` rows for task 4448 on 09-11; `invocation_end` events for 4448 and the per-day timed-out breakdown since 09-01), `orchestrator/src/orchestrator/config.py::TimeoutsConfig` and `::OrchestratorConfig.invocation_timeout`, `shared/src/shared/cli_invoke.py` (watchdog constants and the SIGTERM/SIGKILL timeout path in `_run_subprocess`), `orchestrator/src/orchestrator/agents/roles.py::WAIT_PATTERN_GUIDANCE`, `skills/spawn/hooks/README.md`, the 3731 and `f8a0e99f` transcripts' `updatedInput` attachments, and the fused-memory memory record `518379c8` with its amendment. Fleet counts from `grep` over `data/orchestrator/agent-transcripts/`. Not verified: the hook script and settings file contents, the `skim` binary's behaviour, and the kernel log, all of which needed approvals this session did not have. Codebook relations verified on the 03:42Z snapshot for `cand-20260912-21`, `cand-20260731-20`, `cand-20260905-3`, and the absence of both session ids. Task status via `get_task` for 4448 and 3731; ownership via `search_tasks`. No tasks filed and no codebook edits made from this synthesis; filing and merger application are the runner's steps.*


## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=40, sonnet verify=2, fable synthesis=1, haiku headroom-probe=2
