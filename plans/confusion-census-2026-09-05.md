# confusion census 2026-09-05

Project: dark_factory

## Saturation

- batches: 2
- stop reason: saturated
  - batch 0: dup_rate=1.00 (total=20, succeeded=19, failed=1, saturated=True)
  - batch 1: dup_rate=0.95 (total=20, succeeded=19, failed=1, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | verify |
| --- | --- |
| unknown | 1 |

## Synthesis

All grounding is complete. Writing the synthesis now.

**Date:** 2026-09-05
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests, per-finding verification against current main (Sonnet), then this synthesis (Fable). One finding reached synthesis. This document adds a read of the finding's archived transcript, the verify-artifact code path, and the codebook. Every mechanism claim below names the evidence it rests on. Where the transcript contradicts the verifier's framing, the transcript is quoted and the framing is corrected explicitly rather than silently.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml`. Dispositions in §5 are inputs to the merger.
**Run notes:** ninth completed periodic census, at the PRD's 5-day hard floor after 2026-08-31. Previous corpora: 08-31 (1 verified finding), 08-26 (1), 08-21 (2), 08-16 (3), 08-10 (1), 08-05 (0), 07-31 (15 findings / 4 clusters + 1 one-off), 07-24 (52). Unlike 08-31, no report dated today pre-exists on disk and `census-state.json` still reads 08-31, so there is no dated-path collision to reconcile. The trickle has minted 65 candidates since the last census. Saturation statistics and filed-task ids are appended by the runner outside this synthesis.

### Corpus

- **1 verified finding, 1 session, 1 sighting.** Session `273579c5` is task 4760's implementer session in worktree `.worktrees/4760`, branch `task/4760`, model opus at effort max on a $10 budget, matching the task's recorded implementer route. It ran on 2026-08-31 from 21:33Z. The verifier labelled the area "verify artifact discovery / compound-bash-pipeline error masking" and stamped origin `unknown`, manifested `verify`.
- **Composition:** an agent-side shell-state confusion in an orchestrated implement-phase session. It is an in-repo-observable class, not the harness-rooted class of the 08-21 and 08-31 findings, and it is a new sighting of an existing open entry rather than a novel mechanism.
- **Phase-stamp coverage:** the verified stamps carry 1 unknown of 2. The transcript pins both the origin turn and the role, so this synthesis refines both to `implement` in §1.1 and reports the refinement alongside the runner's matrix rather than overwriting it.
- Session `273579c5` has no prior codebook presence. Task 4760's earlier session `cad50082` does, on the same entry this finding belongs to.

### Executive summary (observations)

1. **The summary file the agent globbed for existed, under exactly the name it guessed. The glob missed because the agent's own previous command had moved the shell's working directory.** At 22:11:19Z the agent ran a compound command beginning `cd .task/verify && ...`. The harness's per-record `cwd` field flips from `/home/leo/src/dark-factory/.worktrees/4760` to `.../4760/.task/verify` on that command's result and stays there. At 22:12:03Z the agent ran `cat $(ls -t .task/verify/*scripts.summary.json | head -1) | python3 -c "...json.load(sys.stdin)..."`, a worktree-relative path issued from inside `.task/verify`. `ls` reported the path absent, the empty substitution left `cat` reading the tool's empty stdin, and Python's parser failed on zero bytes. Nineteen seconds later the agent re-issued the read with absolute paths and it succeeded, printing both `attempt-1.scripts.summary.json` and `attempt-2.scripts.summary.json`. The verified cost is one wasted turn. The verifier's attribution to a wrong naming or location assumption is not supported by the transcript: the name matched the real convention and the file was listed three minutes before and nineteen seconds after the failed call.

### Origin × manifestation matrix

The runner's matrix at the top of this report renders the verifier's stamps, which place the one sighting at `unknown` × `verify`. The table below carries the refined stamps this synthesis establishes in §1.1. Both are reported so the difference is visible rather than reconciled silently.

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| implement | · | · | 1 | · | · | · | · | · | · | **1** |
| **total** | **0** | **0** | **1** | **0** | **0** | **0** | **0** | **0** | **0** | **1** |

Readings, observational. Under the verifier's stamp this is the first `verify`-manifested verified sighting since 07-24, which carried one at `ops` × `verify`. Under the refined stamp the streak of no verify-manifested sighting extends to eight cycles. The `merge` column is zero for an eighth consecutive cycle either way. The PRD's motivating architect/implement→merge hypothesis remains untested by the nine post-07-24 corpora, which total 10 findings.

### 1. Verified clusters

#### 1.1 Worktree-relative glob issued from a shell the agent had already `cd`-ed into `.task/verify` (1 sighting, session `273579c5`)

**The trace, from the archived transcript** at `data/orchestrator/agent-transcripts/4760/.../273579c5-e299-40c6-ab02-e4c0ab24213b.jsonl`, records 360 to 380. Times are UTC on 2026-08-31.

| time | command (abbreviated) | result | recorded cwd after |
|---|---|---|---|
| 22:10:45 | `ls /home/leo/src/dark-factory/.worktrees/4760/.task/verify/ \| head; grep ...` | lists `attempt-1.scripts.summary.json` among 10 lines | worktree root |
| 22:11:19 | `cd .task/verify && ls -t \| head -6 && ... cat $(ls -t *orchestrator.summary.json \| head -1) \| head -40` | prints the orchestrator summary | `.task/verify` |
| 22:12:03 | `cat $(ls -t .task/verify/*scripts.summary.json \| head -1) \| python3 -c "... json.load(sys.stdin) ..."` | exit 1; `ls: cannot access '.task/verify/*scripts.summary.json': No such file or directory`, then a 20-line `JSONDecodeError: Expecting value: line 1 column 1 (char 0)` traceback | `.task/verify` |
| 22:12:22 | `ls /home/leo/src/dark-factory/.worktrees/4760/.task/verify/ \| grep scripts; cat /home/.../attempt-1.scripts.summary.json \| python3 -c ...` | lists attempt-1 and attempt-2 `scripts` logs and both summaries; prints the three commands | `.task/verify` |
| next Bash | `cd /home/leo/src/dark-factory/.worktrees/4760 && timeout 600 uv run ...` | runs | worktree root |

**Verified mechanism.** Three independent facts in the transcript agree. The harness's own `cwd` field moves to `.task/verify` on the 22:11:19 command and stays there through the failure. The same directory listed successfully by absolute path immediately before and after the failure. And the agent's next `cd` is absolute, re-basing the shell. From `.task/verify`, the relative glob resolves to `.task/verify/.task/verify/*scripts.summary.json`, which does not exist. This is the Bash tool's documented behaviour, stated in the tool description every agent receives: working directory persists between calls, prefer absolute paths. The agent used absolute `cd`s at its 16th and 22nd records and again after the failure, and a relative one at the 22:11:19 record. Practice was mixed within one session.

**What the "masking" amounts to.** The `ls` diagnostic is the first line of the tool result. The Python traceback occupies the following 20 lines. The signal was not lost, and the agent's next action was a path fix, so it was read correctly. The traceback's contribution is noise and one turn, about 15 seconds of wall clock. The compound form `cat $(...) | python3 -c 'json.load(sys.stdin)'` produces this specific shape because an empty substitution turns `cat <file>` into `cat` reading the tool's empty stdin, and the pipeline's exit status is Python's, not `ls`'s.

**Correction to the verified framing.** The verifier stated the agent assumed a naming or location convention that no file matched. The name matched the real convention exactly: `orchestrator/src/orchestrator/verify.py::_persist_attempt_logs` writes `attempt-{N}[.{prefix}].summary.json` into `<worktree>/.task/verify/`, `_make_infix` sanitises the module prefix, and `scripts` is a real module prefix from `scripts/orchestrator.yaml`. The location was right too. What was wrong was the base the relative path was resolved against. The verification stage checked the surface against current main and confirmed it. It did not read the transcript, and the transcript is where the cause lives.

**Phase refinement.** Origin `implement`: the drifting `cd` is the same session, one Bash call earlier. Manifested `implement`: the session is the implementer, and the verify run it was inspecting had completed between 15:35Z and about 16:27Z, six hours before. The trickle's `verify` stamp names the artifact being read, not the pipeline phase of the session.

**Structural facts about the artifact, verified in code, recorded as observations and not as a finding.** `_persist_attempt_logs` returns only the log paths, and `_archive_attempt_log` copies those, so the summary JSON never reaches `data/verify-logs/<task>/`. For task 4760 the archive holds the following, and the worktree that held both summaries is now pruned, so no copy of either survives anywhere.

| archive `data/verify-logs/4760/` | count |
|---|---|
| `attempt-1.orchestrator.{lint,test,type}-<ts>.log`, 4 timestamps 08-26 to 08-31 | 12 |
| `attempt-2.scripts.{lint,test,type}-20260831T135957Z.log` | 3 |
| `*.summary.json` | 0 |

`dark-factory-orchestrator.yaml` already describes the summary as a transient per-attempt artifact, overwritten by the next attempt and pruned with the worktree. The only machine-readable record of which commands a task-path verify ran has no durable form. Whether that matters is not established by this sighting.

**Relation to the codebook, observation not a merge.** This is a sighting of `entry-cand-20260729-4`, "Bash cwd persists across turns, causing relative-path git commit --only pathspec failures on PRD doc commits", status open, no `filed_tasks`. That entry's profile at synthesis time:

| `entry-cand-20260729-4` | value |
|---|---|
| sightings | 14, dated 07-29 to 09-03 |
| manifested stamps | implement 10, verify 2, prd 1, unknown 1 |
| origin stamps | implement 11, architect 1, prd 1, unknown 1 |
| entry-level modal stamps | prd / prd |

The entry already holds a sighting from task 4760's earlier session `cad50082` on 08-26, where `cd orchestrator` and then `sed ... orchestrator/tests/...` both failed at turns 55 and 60. Every prior sighting on the entry infers the mechanism in its note, with words like "suggests", "consistent with", "best explained by". This sighting is the first where the transcript shows the drifting `cd` itself and the harness's `cwd` field records the move. It is not a sighting of `entry-cand-20260719-3`, where `.task/iterations.jsonl` genuinely does not exist. It is not a sighting of `entry-cand-20260728-3`, whose profile is repeated not-found with no diagnostic and no recovery. It is not a member of the Python one-liner flattening family: the one-liner here was well formed and ran, and failed on its input.

### 2. One-off sightings

None beyond the cluster above, itself single-sighting this cycle.

### 3. Cross-cutting observations

1. **The census now measurably feeds its own sessions back into the codebook.** The 08-31 finding exists as `entry-cand-20260827-23`. It was created by the 08-31 census commit `44b07dbaf6` directly as a promoted entry, not as the pending candidate the 08-31 synthesis suggested. The 09-01 trickle commit `d9b491b12c` then added two sightings dated 08-31: sessions `4ae36458` and `6eea9f5c`. Both live in the main-checkout project directory and their first prompts read "You are the periodic-census verifier" and "You are the periodic-census synthesis writer". Their notes carry the original sighting's turn numbers 69 and 70, and one says "Corroborated sighting from session 7dae04c6". The entry therefore shows 3 sightings of which 1 is an agent session. `scripts/legibility/inventory.py` resolves corpus membership by cwd prefix only, and the main checkout is a listed prefix, so census-pipeline sessions are in-corpus by construction. The 07-24 census recommended segmenting or excluding legibility-pipeline sessions from mining. A semantic task search surfaced task 3278, done, which filters injected briefing turns out of the User Corrections section, and nothing that excludes pipeline sessions from the corpus. This synthesis and today's verifier session both quote the `ls: cannot access` evidence and will be in tomorrow's trickle window.
2. **The verification stage confirms surfaces, not causes.** This cycle's finding passed verification with a cause attribution the transcript refutes. The verifier checked the claim against current main, where the naming convention and the module prefix are real, and that check passes for the wrong reason. The correcting evidence was three transcript records away. The 08-31 synthesis also checked its finding against the live harness rather than the transcript; it happened to hold.
3. **Entry-level phase stamps lag their sightings.** `entry-cand-20260729-4` reads prd/prd at the entry level from its founding sighting while 10 of its 14 sightings manifest in implement and 11 originate there. A reader of the entry header alone would place this mechanism in the wrong phase.
4. **Prose guidance to prefer absolute paths already exists and did not prevent this.** The Bash tool description carries it. `CLAUDE.md` has no absolute-path guidance beyond the note that the venv is not derivable from cwd, and `orchestrator/src/orchestrator/agents/roles.py` has none. This is recorded as evidence about the limits of prose guidance, not as a request for more of it.

### 4. Remediation candidates

None filed this cycle. The finding is a better-evidenced sighting of an open entry with no filed task, and a single sighting adds no fix surface that the entry's 13 prior sightings did not already imply. The semantic task search for the mechanism returned only unrelated cwd tasks: 3279, 2342, 2882, 3464 and 4264. That search is the extent of the ownership check; it is not a claim that no owner exists. The entry's accrual of 14 sightings across five weeks with no filed task is recorded here for the operator and the next census, not converted into a filing on the strength of one session.

### 5. Codebook dispositions (input to the merger; promote/reject in place, never delete)

| Cluster / finding | Suggested disposition |
|---|---|
| 1.1 Worktree-relative glob from a shell already `cd`-ed into `.task/verify` | Do **not** mint a new candidate. Append a sighting to `entry-cand-20260729-4`: session `273579c5-e299-40c6-ab02-e4c0ab24213b`, date 2026-08-31, origin `implement`, manifested `implement`. Evidence quote should carry both commands, the 22:11:19 `cd .task/verify && ...` and the 22:12:03 relative glob, plus the harness `cwd` field's move, since this is the entry's first sighting that pins the drifting `cd` rather than inferring it. Record three discriminators: the target file existed at the moment of failure; the name and location matched the real convention; the `ls` diagnostic was line 1 of the result and the agent recovered in one turn. Consider refreshing the entry's modal stamps from its sightings, which read implement/implement by a wide margin. |
| `entry-cand-20260827-23` | Observation for the merger, no disposition change requested: two of its three sightings are the 08-31 census's own verifier and synthesis sessions. |

### 6. Method notes for the next census

- **Carry-forwards from 08-31.** Task 3606 remains `pending`, last updated 08-21; the watcher skills were not re-grepped this cycle. `cand-20260806-12` and `cand-20260812-19` remain separate and both `pending`. The 08-31 dated-path collision did not recur: no report dated today existed at synthesis time.
- **The 08-31 synthesis's disposition was not applied as written.** It asked for a pending candidate `cand-20260831-1`; the merger created a promoted entry under a different id, and `cand-20260831-1` now names an unrelated watcher candidate. Future syntheses should not assume a suggested id will be honoured.
- **Watch: self-ingestion onto `entry-cand-20260729-4`.** Today's verifier session and this session both quote the `ls: cannot access '.task/verify/*scripts.summary.json'` line. If tomorrow's trickle adds either as a sighting, that is the second measured instance of the pattern in §3.1.
- **Watch: verifier framing versus transcript.** For a finding whose evidence is a single tool result, the transcript neighbourhood of three records on either side is cheap and, this cycle, decisive. Whether the verification prompt should read it is a design question for the PRD's owners; this note records only that it would have changed this cycle's cause attribution.
- **Codebook scale at synthesis time:** 566 pending, 75 promoted, 32 rejected candidates; 65 minted since 08-31, with 20 on 09-03 alone.

*Synthesis note to the runner: written from the 1 verified finding supplied. Mechanism claims verified by reading the archived transcript for session 273579c5 (records 360 to 380, including the harness `cwd` field on each record), `orchestrator/src/orchestrator/verify.py::_persist_attempt_logs`, `_make_infix` and the task-path archive call site, `scripts/orchestrator.yaml`, and the contents of `data/verify-logs/4760/`. Codebook relations verified by reading `docs/legibility/confusion-codebook.yaml` on current main: `entry-cand-20260729-4` and its 14 sightings, `entry-cand-20260827-23`, `entry-cand-20260719-3`, `entry-cand-20260728-3`, `cand-20260806-12`, `cand-20260812-19`; session 273579c5 confirmed absent. Self-ingestion claim verified by `git log -S` on the codebook for commits `44b07dbaf6` and `d9b491b12c` and by reading the first prompt of sessions 4ae36458 and 6eea9f5c in the main-checkout project directory. Task status via fused-memory `get_task` for 4760 and 3606, ownership via `search_tasks`. No tasks filed and no codebook edits made from this synthesis; filing and merger application are the runner's steps.*


## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=40, sonnet verify=1, fable synthesis=1, haiku headroom-probe=2
