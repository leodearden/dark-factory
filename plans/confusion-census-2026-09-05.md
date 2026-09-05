# confusion census 2026-09-05

Project: dark_factory

## Saturation

- batches: 2
- stop reason: saturated
  - batch 0: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 1: dup_rate=1.00 (total=20, succeeded=19, failed=1, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | ops |
| --- | --- |
| ops | 1 |

## Synthesis

# Confusion census — 2026-09-05

**Date:** 2026-09-05
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests, per-finding verification against current main (Sonnet), then this synthesis (Fable). One cluster survived verification. This synthesis re-read the sighting's transcript, the harness's live tool surface, the invocation site in `scripts/legibility/coder.py`, the project permission settings, and the codebook, and reports what those show. Where the transcript contradicts a clause of the verified cluster, the transcript is quoted and the clause is marked unsupported. No diagnosis appears here that was not itself checked.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml` (dispositions in §5 are inputs to the merger, which promotes/rejects in place).
**Run notes:** ninth report in the series, at the PRD's 5-day hard floor after 2026-08-31. Previous: 08-31 (1 verified finding), 08-26 (1), 08-21 (2), 08-16 (3), 08-10 (1), 08-05 (0), 07-31 (15 findings / 4 clusters + 1 one-off). Saturation statistics and filed-task ids are appended by the census runner outside this synthesis.

## Corpus

- **1 verified finding, 1 session** (f3d0e770), one sighting. Area as coded: `harness/tooling — Bash approval heuristic`.
- Composition: a **harness-rooted** gate (the Bash tool's pre-execution approval check) meeting an **in-repo** invocation site. The session is a headless `claude -p` run built by `scripts/legibility/coder.py::_invoke_cli`. Unlike the 08-21 and 08-31 findings, which had no in-repo surface at all, this one has two: the invocation site and the project's `.claude/settings.json` allowlist.
- Session f3d0e770 already carries one codebook sighting, under `entry-cand-20260722-27` (the self-ingestion family), recorded from the same turn this census coded. §1.1 reports what the transcript shows about that earlier reading.
- Phase-stamp coverage: **0 of 2 unknown** (origin `ops`, manifested `ops`). Full coverage is restored after the 08-31 cycle's 1-of-2.
- A one-finding corpus supports verification and placement, not trend claims.

## Executive summary (observations)

1. **A census-mining invocation's JSON self-check pipe was auto-denied by the Bash approval gate; the invocation completed anyway.** Session f3d0e770 (2026-08-31, 02:37Z) is a headless coder run carrying the `coder.build_prompt` text verbatim, coding the 455-byte digest of orchestrated-task session 513d8216. Its first and only tool call was `echo '<its own complete coding record>' | python3 -m json.tool`, a self-initiated validation of the JSON it was about to return. The harness answered `This Bash command contains multiple operations. The following part requires approval: python3 -m json.tool`, recorded as `toolDenialKind: user-rejected`, with no human on the other end (`entrypoint: sdk-cli`, `permissionMode: default`). The next assistant turn emitted the identical JSON as plain text and ended the turn. **The verified cluster's consequence clause, "the call stalls rather than completing", is not supported by the transcript.** The denial returned 9.2 s after the call, the final answer followed 2.7 s later, and the whole session ran about 29 s wall-clock. The verified cost is one denied call and one extra model turn.

## Origin × manifestation matrix

Rows = `origin_phase`, columns = `manifested_phase`. Counts are verified sightings (1 total). `merge` and `verify` are kept explicitly to show their zeros.

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| ops | · | · | · | · | · | · | · | 1 | · | **1** |
| **total** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **1** | **0** | **1** |

Readings (observational): no merge- or verify-manifested sighting for an **eighth consecutive cycle**. The corpora since 08-05 have never exceeded three findings per cycle, so the absence still carries limited evidence on the PRD's motivating architect/implement→merge hypothesis.

## 1. Verified clusters

### 1.1 Bash approval gate auto-denies a headless coder's JSON self-check pipe (1 sighting, session f3d0e770)

**What the transcript shows (read at synthesis time; 19 lines).**

| time (UTC, 2026-08-31) | event |
|---|---|
| 02:37:21 | prompt enqueued: `coder.build_prompt` text; digest of 513d8216 (`.worktrees/4189`, score 1.0, one `not_found` at "turn 23", zero user turns) |
| 02:37:38 | Bash: `echo '{"matches": [{"entry_id": "entry-cand-20260728-3", "origin_phase": "unknown", "manifested_phase": "implement", …}], "candidates": []}' \| python3 -m json.tool` |
| 02:37:47 | tool_result `is_error`: "This Bash command contains multiple operations. The following part requires approval: python3 -m json.tool"; `toolDenialKind: user-rejected` |
| 02:37:50 | assistant text = the same JSON verbatim; `stop_reason: end_turn`; Stop hook runs |

Session metadata: `entrypoint: sdk-cli`, `cwd: /home/leo/src/dark-factory`, branch `main`, CLI `2.1.251`, `permissionMode: default`. The model recorded on both assistant messages is `claude-sonnet-5` (effort `xhigh`). The prompt names no tool and asks for "STRICT JSON ONLY"; the pipe was the model's own idea. There was no retry.

**Which stage this was.** `docs/legibility/legibility.yaml` pins `models.trickle: haiku` and `models.census_miner: sonnet`, and both stages call `coder.code_digests` with the same `build_prompt` text (`census.py::mine_to_saturation`, `nightly.py`), so the prompt cannot tell them apart. The recorded model is Sonnet, and the session sits 31 minutes before the first of the two 08-31 census commits (`8f2df2ab06` at 03:08:48Z, `44b07dbaf6` at 03:18:52Z). The session is therefore consistent with a **census-miner** call of the first 08-31 run, not a nightly Haiku trickle call. The cluster's "trickle-coder" label is the prompt's self-description, which both stages share.

**Verified structural facts.**

- `coder.py::_invoke_cli` spawns `[claude, "-p", "--model", <model>]` with the prompt on stdin and an optional `cwd`. It passes no `--allowedTools`, no permission-mode flag, and no bypass flag. Its docstring documents cwd sandboxing for reads outside the tree ("non-interactively there is no prompt to approve") and says nothing about the approval gate for commands inside it.
- The project's `.claude/settings.json`, which a session rooted at `project_root` reads, carries 29 allow rules. The Bash prefixes present are `git clone`, `cd`, `ls`, `sqlite3`, `uv run`, `git remote`, plus literal `ps`, `journalctl`, `systemctl`, and `nvidia-smi` lines. None matches `echo`, `python3`, or a pipeline. Whether that file was consulted for this session is not visible in the transcript.
- The harness lists a built-in skill, `fewer-permission-prompts` ("Scan your transcripts for common read-only Bash and MCP tool calls, then add a prioritized allowlist to project .claude/settings.json"), in both f3d0e770's skill listing and this synthesis session's. It is an affordance that exists; no claim is made that it applies.
- Live reproduction in this synthesis session (interactive; permission mode not visible to the writer): a four-segment read-only compound (`cat …; echo …; ls …; grep …`) drew the same "contains multiple operations … requires approval" text naming the segments. Single `ls` (with a glob), `awk`, `git log -S…`, and `git -C … log` invocations each drew "This command requires approval". Single `grep` and plain `git log` invocations ran unprompted. The gate therefore fires on some single commands as well as on compounds; which rule decides is not observable from the caller's side and is not claimed here.

**Cost, as established.** One denied tool call; one additional model turn (176 output tokens against roughly 59k cached input tokens); about 12 s between the call and the final answer. No durable state touched. **Not established:** whether the returned record was applied. Current main carries no sighting for session 513d8216 under any entry, and the first 08-31 run's own report was overwritten by the same-dated second run (recorded in the 08-31 synthesis's run note), so the writer could not trace the record from this transcript to a codebook commit. A history search for the session id was itself refused by the approval gate. That gap is reported, not attributed.

**Relation to the codebook (observation, not a merge).**

- *The existing sighting of this same turn.* `entry-cand-20260722-27` ("Trickle-coder digest pipeline ingests its own prior invocations as session digests") carries f3d0e770 with the note "turn 13 Bash error contains codebook-formatted JSON … indicating pipeline may be processing … its own prior output". The transcript shows that JSON is the coder's **own draft answer for 513d8216**, echoed into its own validation command in the same turn; nothing from a prior invocation appears in the session. The note's reading is not supported by the transcript. That sighting was recorded by a later coder from f3d0e770's *digest*, where a denied `echo '<json>' | …` call renders codebook-shaped JSON inside an error neighborhood, which is what the later coder saw.
- *Nearest catalogued mechanism:* `entry-cand-20260722-6` (open). Its 08-16 sighting (session 13ed9bd7) already records the "multiple operations" rejection on semicolon, pipe, and redirect forms. Most of that entry's sightings, though, are the *flattening* defect (multi-line `python3 -c` collapsing to one line with an IndentationError; 09-02 and 09-04 sightings included). This sighting is a single-line pipe with no flattening, in a headless session: same gate, different consequence.
- *Same gate, other strata:* `entry-cand-20260722-18` (open; census **verifier** blocked per-command on read-only forensic reads, origin `ops` / manifested `verify`; its sightings record 3 to 28 denials per session and "unable to complete") and `entry-cand-20260722-34` (open; blind transcript search hitting gates; its founding quote is the identical "contains multiple operations … requires approval" text on a `find /`). Pending relatives in other strata: `cand-20260809-9` (orchestrated tasks in worktrees, read-only diagnostics), `cand-20260801-5` (interactive worktree session, read-only forensics), `cand-20260803-5` (headless coder routed to a sibling-project path, "freezing automated analysis"). This finding is the first sighting of the gate in the **census-mining** stage, and the only one in the family that records a one-denial completion rather than a denial storm.
- *Not a sighting of* `cand-20260816-1` (pending; identical-command retry after denial): f3d0e770 did not retry.
- *Disposition precedents in the same shape:* `cand-20260825-28` ("Foreground sleep+chain Bash guard blocks orchestrated-task agent, discovered only by tripping it") was **rejected** on 08-25, and `cand-20260722-22` (headless `-p` permission configuration: `--allowedTools` and Bash allowlist prefixes) was **rejected** on 07-22. Both are the closest prior adjudications of "a harness Bash guard is enforced identically for a non-interactive agent, and nothing surfaces it up front". Recorded so the runner's filed task and the codebook merge carry them.

## 2. One-off sightings

None beyond the cluster above (itself single-sighting this cycle).

## 3. Cross-cutting observations

1. **The census is now observing its own mining stage.** This cycle's only verified finding is a session produced by the previous census's own Sonnet miners. The codebook already holds a family of eleven promoted entries about the trickle pipeline ingesting its own invocations (`entry-cand-20260722-10`, `-13`, `-16`, `-17`, `-19`, `-21`, `-24`, `-26`, `-27`, `-30`, `-31`); this is the same loop one level up. The earlier note on this very turn read a coder's draft record, rendered inside an error neighborhood, as evidence of re-ingestion. This census enumerated a session from the day of the previous census, so same-day sessions are inside the window and a census's miners are enumerable by the next census.
2. **Trickle and census-miner sessions are prompt-identical.** Every "trickle-coder" label in the codebook, the self-ingestion family's sightings included, may cover census-miner sessions. The transcript's recorded model is the only discriminator visible, and digests do not carry it.
3. **A verified cluster's consequence clause can outrun its evidence.** Verification confirmed the gate fired (the quote is verbatim); the "stalls" clause was inferred from "no human present" and did not survive a transcript read. The 08-31 method note applies again: route by mechanism and check the outcome fields (`stop_reason`, timestamps), not the error string.
4. **Read and Bash have different reach for transcript forensics.** In this synthesis session the Read tool opened `~/.claude/projects/<enc>/<uuid>.jsonl` by known path, while a Bash `ls` of the same directory was refused as outside the allowed working directories. `entry-cand-20260722-5` and `entry-cand-20260722-34` record verifiers failing on exactly this tree via Bash; the route that worked here is a direct Read by slug.
5. **The harness-rooted class recurs, this time with an in-repo construction site.** 08-21 (1.1), 08-31 (1.1), and this finding are all harness tool-contract behaviors; this one is the first with a repo-owned invocation site and a repo-owned permission file in the path.

## 4. Remediation candidates

The runner attempts one curator-path task per verified cluster (`census.py::build_task_payloads`, title prefix `[legibility census]`, `metadata.source: legibility_census`); the 08-31 report lists none filed for its one cluster, so filing is not guaranteed. This synthesis neither adds to nor removes from that step. Observations the filed task should carry, so the adjudicator can weigh them:

- The two in-repo surfaces in the path are `scripts/legibility/coder.py::_invoke_cli` (no permission configuration passed) and `.claude/settings.json` (no rule matching the denied segment). No change to either is claimed correct here; `cand-20260722-22`'s rejection is the standing adjudication on `-p` permission configuration and `cand-20260825-28`'s on guard surfacing.
- The cluster text's "stalls rather than completing" should not propagate into the task description as a premise (`unverified-task-premises`): the session completed, and the only unresolved question is whether its record was applied.
- The measured cost per occurrence in this stratum is one turn; the same family's cost in the verifier stratum (`entry-cand-20260722-18`) is session-fatal. A remediation scoped by stratum would be scoped by the evidence.

## 5. Codebook dispositions (input to the merger; promote/reject in place, never delete)

| Cluster / finding | Suggested disposition |
|---|---|
| 1.1 Bash approval gate auto-denies a headless coder's JSON self-check pipe | Promote the pending candidate the merge assigns this title (runner step) with origin `ops` / manifested `ops`, first_seen 2026-08-31, session f3d0e770. Evidence notes to record: stage = census miner (model `claude-sonnet-5`; prompt shared with the trickle); outcome = completed, one denied call, no retry; the "stalls" clause is unsupported. Cross-reference `entry-cand-20260722-6` (same message, flattening-dominated), `entry-cand-20260722-18` (same gate, verifier stratum, session-fatal), `entry-cand-20260722-34` (same message on `find /`), and the rejected precedents `cand-20260825-28` and `cand-20260722-22`. |
| f3d0e770's existing sighting under `entry-cand-20260722-27` | Append-only rules forbid editing it. Record in the new entry's evidence that the "codebook-formatted JSON" that note cites is the coder's own draft record inside the denied command, per the transcript. |

## 6. Method notes for the next census

- **From 08-31, re-checked:** the Edit `replace_all` finding was catalogued under `cand-20260827-23` and promoted to `entry-cand-20260827-23`. The stray-comma pair `cand-20260806-12` / `cand-20260812-19` and the retry-loop facet `cand-20260819-1` all remain `pending` and separate. Task 3606 was not re-checked this cycle.
- **New watches:** (a) whether the next window again enumerates this census's own miner and verifier sessions; if so, the self-ingestion family will keep accruing sightings whose "evidence" is a coder's draft output. (b) Whether any census-miner or trickle session hits the gate more than once, which would be the first denial storm in the headless coder stratum. (c) Whether the digest of a denied `echo '<json>' | …` call can be told apart from re-ingested content; today the error-neighborhood renderer gives a coder no way to do so.
- **For verifiers:** when a cluster's consequence is "stalls", "hangs", or "unable to complete", read the transcript tail (`stop_reason`, last timestamp) before confirming. This cycle the mechanism was real and the consequence was not.

---

*Synthesis note to the runner: written from the 1 verified cluster supplied. Transcript claims verified by reading `~/.claude/projects/-home-leo-src-dark-factory/f3d0e770-b056-4b62-be67-d37958a6db28.jsonl` (19 lines) at synthesis time. Invocation and prompt claims from `scripts/legibility/coder.py` (`_invoke_cli`, `build_prompt`), `scripts/legibility/census.py` (`mine_to_saturation`, `build_task_payloads`, `_find_pending_candidate_id`), `scripts/legibility/nightly.py`, and `docs/legibility/legibility.yaml`. Permission-file claims from `.claude/settings.json`. Commit times from `git log` on the 08-31 census artifacts. Codebook relations from `docs/legibility/confusion-codebook.yaml` on current main. Not verifiable by the writer: whether f3d0e770's record was applied (the history search was refused by the approval gate), and this synthesis session's own permission mode. No tasks filed and no codebook edits made from this synthesis.*


## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=40, sonnet verify=1, fable synthesis=1, haiku headroom-probe=2
