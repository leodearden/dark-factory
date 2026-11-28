# confusion census 2026-11-24

Project: dark_factory

## Method

```yaml
run_id: census-dark_factory-20261124
as_of_sha: d575b86d667efce5b6fe883f20c6c4810729e9c0
since: 173059bd55507fdc8f527a7bb0aa435af2c2bf78
evidence:
  window:
  - '2026-10-25T00:00:00+00:00'
  - '2026-11-24T00:00:00+00:00'
  sessions_enumerated: 24858
  skipped_coded: 986
  skipped_zero_signal: 17502
  mined: 40
  ledger_rows: 1056
verification:
  confirmed: 3
  weakened: 0
  refuted: 8
  unverified: 0
cost:
  miner_calls: 40
  verify_calls: 22
  synthesis_calls: 1
  probe_calls: 5
  embedding_calls: 1
  wall_clock_secs: 1204.745
inputs_consumed:
- review-dark_factory-20261009
- hotspot-survey-dark_factory-20261009
- review-all-dark_factory-20261008
extra:
  inputs_consumed_note: null
  screened:
    title: 0
    session: 0
    key: 0
    cross_instrument: 0
  attached: 0
  adjudicated: 18
  ledger_created_this_run: false
  ledger_state: ok
  ledger_pruned: 484
  ledger_error: null
  verify_normalisations:
    anchor_rewritten: 0
    anchor_from_remediation: 0
    anchor_from_title: 0
    tag_dropped: 0
    severity_missing: 0
    severity_substituted: 0
    severity_reason_missing: 0
    route_defaulted: 0
    remediation_rejected: 0
    reason_missing: 0
  reverify:
    due: 11
    offered: 11
    gone: 5
    present: 6
    inconclusive: 0
    deferred: 0
  slug_rejections:
    count: 0
    values: []
    values_omitted: 0
```

## Saturation

- batches: 2
- stop reason: saturated
- operator batch cap: 50 batch(es) (not reached -- mining stopped by: saturated)
  - batch 0: dup_rate=1.00 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 1: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)

## Screened

- per rule: title 0, session 0, key 0, cross_instrument 0
- inputs consumed: review-dark_factory-20261009, hotspot-survey-dark_factory-20261009, review-all-dark_factory-20261008

## Adjudication

- model text-embedding-3-small; attach at cosine >= 0.9, cluster at >= 0.85; at most 10 queue cluster(s) per run
- embeddings: 1 request(s), 54 embedded, 1072 cached, 21 pruned
- queue depth: 1013 pending candidate(s) in 1005 cluster(s)
- processed this run: 10 cluster(s), 18 candidate(s)
- verified: "Recon agent repeatedly calls count_memories_by_metadata with undeclared 'metadata' argument instead of declared 'filters'" (cand-20261122-2) for 3 candidate(s) across 3 session(s)
- rejected: "Recon agent repeats get_task calls with an undeclared project_id after an explicit unknown_argument rejection and inline schema" (cand-20261120-1) for 2 candidate(s) across 2 session(s)
- verified: "Agent shell cwd left pointing at a worktree another session removed, failing subsequent commands until recovery" (cand-20261118-2) for 2 candidate(s) across 2 session(s)
- rejected: "Task worktree agents unaware of working directory context for repo-relative paths" (cand-20260818-15) for 2 candidate(s) across 2 session(s)
- rejected: "Path-scope guard misconfig in worktree context" (cand-20260815-2) for 2 candidate(s) across 2 session(s)
- rejected: "Supervisor rotation prompt mislabeled as User Correction in digest" (cand-20260812-8) for 2 candidate(s) across 2 session(s)
- rejected: "Trickle-coder ReportFindings output produces double-wrapped findings with missing schema field" (cand-20260809-14) for 2 candidate(s) across 2 session(s)
- rejected: "Plan asserts markup-check behavior on redo tasks without verifying the override-removal ordering, so the plan drops redos it should keep" (cand-20261123-2) for 1 candidate(s) across 1 session(s)
- rejected: "Ratchet baseline for gates.py growth is not surfaced to planners, so plans assume no baseline update is needed" (cand-20261123-1) for 1 candidate(s) across 1 session(s)
- rejected: "Lease-claim reports decision=acquired yet exits 2, so exit status cannot signal success" (cand-20261122-1) for 1 candidate(s) across 1 session(s)

## Verification

- handed all 1 novel cluster(s) to the verifier; operator verify cap: 150 (not reached).

## Origin x Manifestation Matrix

| origin \ manifested | recon | unknown |
| --- | --- | --- |
| implement | 0 | 1 |
| recon | 2 | 0 |
| unknown | 1 | 2 |

## Synthesis

- applied: 1 phase refinement(s) to 1 cluster(s) and 1 new entr(ies); 0 correction(s) applied, 0 skipped
- standing, no new evidence: 90
- invariant violations this run: none
- unkeyed entries re-observed: 5

### New

- fk-bb99b8dda638 "Escalation resolve_issue rejects project_root, a parameter the repo instructions teach for every task operation": In the current tree `escalation/src/escalation/server.py::resolve_issue` declares no `project_root` parameter, while sibling tools on the same server accept it: `get_task_escalation_history`, `get_escalation` and others take it as optional and pass it through `ProjectRootGuard.check`, and `claim_warm_worktree` and `release_warm_worktree` require it. One session passed `project_root` to `resolve_issue`, got a pydantic unexpected_keyword_argument error, and spent a turn recovering the real signature. Under heuristic 10 (clear invariants, informatively, redundantly, uniformly enforced), the project-root guard is not applied uniformly across this server's tools. Under heuristic 3 (carefully factored orthogonal dimensions of variability), the which-project axis is carried per tool rather than by one mechanism, so a caller cannot predict from one tool whether the next accepts it. The failure is loud and costs one turn. Whether any doc says which servers accept `project_root` was not checked by the verdict.
- fk-f1449a2f494f "Recon agent repeatedly calls count_memories_by_metadata with undeclared 'metadata' argument instead of declared 'filters'": Three recon sessions called `count_memories_by_metadata` with a `metadata` argument. The tool declares only `project_id` and `filters`, and the server rejected each call before execution as unknown_argument, with an error listing the declared parameters. Nothing was written; the cost is wasted turns. The rejection is the informative half of heuristic 10 (clear invariants, informatively, redundantly, uniformly enforced) working as intended, yet the same wrong shape recurred across three task IDs. The recon stage prompts (`stage1.py`, `stage2.py`, `stage3.py` under `fused-memory/src/fused_memory/reconciliation/prompts/`) already write the correct `filters={...}` shape, so the verdict found no file in this tree that teaches the wrong one.
- fk-7c4ee0eb5759 "Agent shell cwd left pointing at a worktree another session removed, failing subsequent commands until recovery": Two sessions received harness-level messages that the shell's working directory no longer existed: one a recovery of the cwd to the repo root, the other a bash getcwd failure. Each had to re-issue a command, so the cost is one retry. Both messages are consistent with a directory under `.claude/worktrees` being deleted while a shell was inside it. The verdict found no code in this tree that creates or removes `.claude/worktrees/mut-*` or `carry-*` directories, and it could not confirm which worktrees currently exist. The finding is tagged heuristic 10 (clear invariants, informatively, redundantly, uniformly enforced): the getcwd error does not name which directory vanished or what removed it, and no enforcement point for 'a worktree in use is not removed' was located in this tree.

### Re-observed

- fk-a69ec55ddc08 "Digest 'User Corrections' bucket mislabels injected task context/briefing as a correction" (gained 43c1b34c-efed-4f8b-b27e-7d5102169ca1): no change stated
- fk-8dabfa972cc0 "Orchestrated-task sub-agent denied read-only Bash inspection in don't-ask mode, forcing blind guesses while diagnosing a verify failure" (gained 0b81f333-731f-4ae5-965c-fde8cf269661/subagents/agent-a89bf01277cbf5f41): no change stated

### Phase Refinements

- fk-bb99b8dda638: origin implement, manifested unknown (The verdict confirms the cause lives in the current tree as the `resolve_issue` signature in `escalation/src/escalation/server.py`, which lacks the `project_root` parameter its sibling tools carry. That places the origin in implemented code rather than unknown. The brief does not identify the role or phase of the subagent session that hit the error, so the manifested phase stays unknown.)

### Notes

- unverified: The verdict's explanation that the recon agent guessed `metadata` because the deferred tool schema was not loaded before the call is an inference, not a verified cause. (evidence: The fk-f1449a2f494f verdict says the wrong shape 'points to' the deferred schema not being loaded and attributes it to harness tooling. No session evidence in the brief shows the schema's load state at call time.)
- unverified: `docs/flag-marker-sweep-recurring.md` may show a third call shape for `count_memories_by_metadata` (a `source=` kwarg) that differs from the declared `project_id` and `filters` parameters. (evidence: The fk-f1449a2f494f verdict cites line 767 of that operator doc and notes it is not part of the recon prompts. Whether any recon session read it is not established.)
- unverified: The fk-f1449a2f494f finding may duplicate existing codebook entries. (evidence: The verdict states the codebook already holds many near-duplicate entries, but the brief lists none and its counts show attached: 0.)
- unverified: The part of the fk-bb99b8dda638 cause saying no guidance tells agents which servers accept `project_root` is unchecked. The repo instructions' Task Routing wording appears to scope `project_root` to task operations through fused-memory MCP tools, without addressing the escalation server either way. (evidence: The verdict says it did not search the docs for this. The CLAUDE.md text supplied with this request says to use `project_root` 'for all task operations on this repo' under a section about fused-memory MCP tools.)
- unverified: The fk-7c4ee0eb5759 attribution of the removal to another session or process, and the specific directories named (mut-10723 removed while the session worked in carry-10723), rest on the finding's cause text rather than on anything the verdict confirmed. (evidence: The verdict found only two harness messages 'consistent with' a deleted worktree, found no in-tree code managing those directories, and could not run shell checks.)
- unverified: The proposed fix for fk-bb99b8dda638 would make the guard uniform on `resolve_issue` but would not by itself tell agents which other escalation tools accept `project_root`. (evidence: The verdict's proposal covers `resolve_issue` only. The brief does not enumerate the full set of escalation tools with and without the parameter.)

## Findings

- area map: briefing review/briefing.yaml; areas dashboard, escalation, fused-memory, orchestrator, repo, sampler, shared; overlay .claude/skills/review-all/project.md; rules `scripts/legibility/**` → `shared`, `scripts/**` → `repo`
- 2 re-observed finding(s) are unverified: the census does not re-verify a catalogued finding, so a sighting this run only appends to its last_seen.

| id | key | area | sub_area | anchor | tags | severity | evidence_source | statement | proposal | verdict | disposition | first_seen | last_seen | supersedes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| entry-cand-20261122-3 | fk-bb99b8dda638 | escalation |  | escalation/src/escalation/server.py::resolve_issue | h10, h3, kind:confusion | low | agent-transcripts, present-tree | CLAUDE.md tells agents to pass project_root on task operations, which are fused-memory tools. The escalation MCP server's resolve_issue takes no such keyword, so an agent carrying that habit across servers gets a pydantic unexpected_keyword_argument error. The call fails and the agent spends a turn working out the real signature. No guidance says which servers accept project_root. Verifier: Confirmed in the current tree. `escalation/src/escalation/server.py::resolve_issue` (line 2441) has no `project_root` parameter. Its parameters are escalation_id, resolution, action, resolved_by, resolution_turns, resolution_class, granted_files, acknowledge_declared_pins, escalate_model, recut and terminate. Other tools on the same server take an optional `project_root` and pass it through `ProjectRootGuard.check`: the tools at lines 2136, 2249, 3026 and 3206, `get_task_escalation_history` and `get_escalation`. `claim_warm_worktree` and `release_warm_worktree` take a required `project_root`. An agent that follows the repo habit of passing `project_root` can therefore succeed on sibling escalation tools and get an unexpected-keyword error on `resolve_issue`. I did not search the docs for guidance on which servers accept `project_root`, so that part of the claim is unchecked. | Add an optional `project_root: str \| None = None` parameter to `resolve_issue` and enforce it with `project_root_guard.check_off_loop`, as the sibling escalation tools do. | confirmed | filed:11553 | census-dark_factory-20261124 | census-dark_factory-20261124 |  |
| entry-cand-20261122-2 | fk-f1449a2f494f | repo |  | slug:recon-count-memories-by-metadata-undeclared-arg | h10, kind:confusion | low | agent-transcripts, present-tree | The agent passes a filter object under an argument name the fused-memory tool does not declare. The server rejects each call before execution with an error that lists the declared parameters, but the same wrong call shape recurs across three task IDs with no evidence of the correct shape being learned. Verifier: The wrong call shape is a real, recurring failure. The tool declares only project_id and filters, and the server rejects `metadata` as unknown_argument. The cause is not in this tree. The stage prompts already teach the correct shape: fused-memory/src/fused_memory/reconciliation/prompts/stage2.py (lines 227, 238, 336) and stage1.py and stage3.py all write `count_memories_by_metadata(project_id=..., filters={...})`. The agent still guessed `metadata`, which points to the deferred tool schema not being loaded before the call. That is harness tooling, not a file here. The rejection is loud and no state is written. docs/flag-marker-sweep-recurring.md line 767 shows a `source=` kwarg shape, but it is an operator doc, not part of the recon prompts. The codebook already holds many near-duplicate entries. |  | confirmed | filed:5968 | census-dark_factory-20261124 | census-dark_factory-20261124 |  |
| entry-cand-20261118-2 | fk-7c4ee0eb5759 | repo |  | slug:shell-cwd-removed-worktree | h10, kind:confusion | low | agent-transcripts, present-tree | A worktree directory that the session's shell was using was removed by another process (the worktree named mut-10723 was deleted while the session worked in carry-10723). The agent's shell cwd then refers to a nonexistent directory, so every later Bash call fails with getcwd or no-such-directory errors, and the agent has to re-issue commands after the shell recovers. Verifier: The evidence is two harness-level messages. One says the working directory no longer exists and the shell cwd was recovered to the repo root. The other is a bash getcwd failure. Both are consistent with a worktree under .claude/worktrees (interactive-session worktrees) being deleted while a session's shell was inside it. The tree has no code that creates or removes .claude/worktrees/mut-* or carry-* directories. The only mention is a config.py comment about skipping nested checkouts when discovering modules. So the cause is in tooling outside this tree, and no specific existing file produces, teaches or fails to guard against it. I could not run shell checks, so I did not confirm which worktrees exist right now. |  | confirmed | filed:11554 | census-dark_factory-20261124 | census-dark_factory-20261124 |  |
| cand-20261120-1 | fk-17a8a0365263 | fused-memory |  | fused-memory/src/fused_memory/server/tools.py::get_task | h10, kind:confusion | low | agent-transcripts | The agent passes a project_id argument that get_task does not declare, even though the rejection names the declared parameters and the inline schema lists only id, project_root and tag. It repeats the same rejected call across several turns instead of correcting the argument. Each rejection is loud and self-describing, so the failure comes from the agent's call shape, not from a silent or misleading tool contract. Verifier: The tool contract is accurate and loud. `fused-memory/src/fused_memory/server/tools.py::get_task` declares only id, project_root and tag. The rejection names the declared parameters and the closest match, project_root. The cluster itself says the failure comes from the agent's call shape. No file in this tree produces, teaches or fails to guard against it. Many sibling tools do take project_id, so the agent's habit is understandable, but the guard works as designed. |  | refuted | refuted | census-dark_factory-20261124 | census-dark_factory-20261124 |  |
| cand-20260818-15 | fk-1b24c833a286 | orchestrator |  | orchestrator/src/orchestrator/agents/bash_cwd_guidance.py::BASH_CWD_ANCHOR_GUIDANCE | h10, kind:confusion | low | agent-transcripts | Orchestrated task agents run within `.worktrees/<id>` with cwd set to worktree root, but attempt to use repo-relative paths (e.g., `orchestrator/src/orchestrator/`) without adjusting for depth or incomplete checkout. No briefing context supplied regarding worktree structure, available sources, or required path adjustments. Verifier: The current tree already guards against this. `orchestrator/src/orchestrator/agents/bash_cwd_guidance.py::BASH_CWD_ANCHOR_GUIDANCE` is spliced into every Bash-capable role preamble at `harness_guidance.py` line 466. It states that the agent starts at the ROOT of its checkout, that a relative path can fail with 'No such file or directory' after a lingering `cd`, and that the fix is to anchor with `git rev-parse --show-toplevel` and to give Read, Grep and Glob absolute paths. `invoke.py::apply_bash_cwd_reset_env` adds a mechanical cwd reset, and `path_not_found_guidance.py` covers recovering a path that was not found. The root CLAUDE.md has the same guidance. Both sightings are from 2026-08-18. This guidance came later, from task 5971 and codebook entry cand-20260729-4, and the current tree shows it as in place. I could not confirm the sightings' cause, which may be a `cd` into a subdirectory. I did not check whether an incomplete worktree checkout contributes. |  | refuted | refuted | census-dark_factory-20261124 | census-dark_factory-20261124 |  |
| cand-20260815-2 | fk-48903ce02ba5 | fused-memory |  | fused-memory/src/fused_memory/middleware/path_scope_guard.py | h10, kind:confusion | low | agent-transcripts | Scope guards designed/tested against main checkout paths do not account for ephemeral worktree subdirectories (.worktrees/<id>), causing systematic false-positive blocks on legitimate worktree operations early in task execution Verifier: The cluster's cause, that scope guards ignore .worktrees/<id> subdirectories, is not supported by the current tree. The guard behind DarkFactoryPathScopeViolation is fused-memory/src/fused_memory/middleware/path_scope_guard.py. It scans task or submission text for path tokens and matches them against a project prefix registry built from known_project_roots. It never reads the caller's cwd, and the file does not mention 'worktree' at all. So a worktree-path classification error is not a mechanism it can have. The trips themselves are already recorded. docs/legibility/confusion-codebook.yaml has the entry oneoff-2026-07-01 ('Path-scope guard prose-token matching ... Over-firing scope guards', fix task 2434). It already holds the 2026-08-15 session 78186d80 sighting (turns 3, 21, 33, 57, 87) and the 3106 worktree session. The cause there is prose-token over-firing, not worktree paths, so this cluster is a duplicate with an unverifiable root-cause claim. I did not check the guard's trip behaviour with worktree-prefixed paths in the text. |  | refuted | refuted | census-dark_factory-20261124 | census-dark_factory-20261124 |  |
| cand-20260812-8 | fk-a69ec55ddc08 | shared |  | scripts/legibility/reingestion.py::HARNESS_PROMPT_MARKERS | h11, kind:confusion | low | agent-transcripts | Digest generator captures the supervisor-injected rotation limits prompt and categorizes it as a User Correction instead of recognizing it as harness-injected context. Verifier: The current tree no longer shows this confusion. scripts/legibility/reingestion.py::HARNESS_PROMPT_MARKERS contains 'you are running as an autonomous escalation watcher'. is_harness_injected_turn matches it by substring. The digest's gold-turn filter and dialogue carriers consult that function; scripts/legibility/digest.py documents this as rule 7, task 7441. scripts/tests/legibility_carrier_fixtures.py has a WATCHER_ROTATION_PROMPT fixture for this exact prompt. It also has a human-turn param that names the watcher without being injected. I did not run the digest or the tests. I also did not trace the exact filter call inside digest.py's user-corrections iterator, so the verdict rests on the marker, the fixture and digest.py's own description. The sightings are from 2026-08-01 and 2026-08-12, before the marker was added. |  | refuted | refuted | census-dark_factory-20261124 | census-dark_factory-20261124 |  |
| cand-20260809-14 | fk-75991adc483b | fused-memory |  | fused-memory/src/fused_memory/reconciliation/judge.py::JUDGE_VERDICT_SCHEMA | h10, kind:confusion | low | agent-transcripts | Trickle-coder invocation of ReportFindings generates {"findings": {"findings": [...]}} structure (parameter name matches schema key pattern) and omits required 'severity' field Verifier: The cluster is attributed to the trickle-coder, and I could not confirm that in the tree. `scripts/legibility/coder.py` has no `ReportFindings` or `StructuredOutput` use and no `--json-schema` call. Its output is validated by `codebook.validate_coding_record`, and a schema-invalid record is skipped and counted. The quoted error ('root: must have required property severity, /findings: must be array') matches `JUDGE_VERDICT_SCHEMA` in `fused-memory/src/fused_memory/reconciliation/judge.py`. That schema requires a root `severity` and a `findings` array, and the sessions were in the recon phase, so the sessions were more likely the reconciliation judge. In the tree the schema is correct and is derived from `VerdictSeverity`. The double-wrapped `{findings: {findings: [...]}}` output is a model-side mistake. I found no file in the tree that teaches or fails to guard against it. The tree does not show whether the schema-validation error is retried or surfaced. |  | refuted | refuted | census-dark_factory-20261124 | census-dark_factory-20261124 |  |
| cand-20261123-2 | fk-bfb605e0d3d7 | repo |  | slug:redo-markup-override-ordering | h10, kind:confusion | low | agent-transcripts | The planning agent reasoned that the markup check could not reject a redo because the original's text had already passed it. It did not account for the allow_mcp_markup override being removed before the task is saved, so the premise was wrong and the agent had to correct it. Verifier: The underlying fact is true: the markup guard forwards allow_mcp_markup, and the tool bodies strip it before persistence (fused-memory/src/fused_memory/server/tools.py, submit_task and update_task docstrings and strip_markup_override calls). A redo of an original filed with the override can therefore be rejected. This is documented in tools.py, though. I found no redo-task pipeline in the tree that makes or relies on the claim that the markup check cannot reject a redo. The orchestrator's auto_eval_redo.py has no markup logic. The error was in the planning agent's own premise, and I found no existing file that teaches it or fails to guard against it. |  | refuted | refuted | census-dark_factory-20261124 | census-dark_factory-20261124 |  |
| cand-20261123-1 | fk-a228fbac9480 | orchestrator |  | orchestrator/tests/merge_lane_ratchet_baseline.json | h11, kind:confusion | low | agent-transcripts | The plan assumed that growth in orchestrator/src/orchestrator/merge_lane/gates.py needed no change to the ratchet baseline. The agent had to correct this mid-session, and a Bash probe of merge_lane_ratchet_baseline.json then failed with exit 1. The ratchet baseline's coupling to file-size and complexity metrics is not visible from the plan's own text. Verifier: The ratchet's coupling to gates.py is documented in the tree. CONTRIBUTING.md section 4 describes the ratchet, the baseline path and the --write-baseline and --authorize-raise flow. The `_README` that scripts/merge_lane_ratchet.py freezes into the baseline spells out the same rules. The baseline lists gates.py with explicit lines, prose_lines and cognitive metrics, so growth there is visibly measured. The cluster's only evidence is one plan that misjudged this and was corrected mid-session. I found no existing file that produces the confusion or fails to guard against it. The Bash exit 1 is not tied to anything in the tree. |  | refuted | refuted | census-dark_factory-20261124 | census-dark_factory-20261124 |  |
| cand-20261122-1 | fk-9cfea71d75c0 | orchestrator |  | orchestrator/src/orchestrator/session_registry.py::main | h10, kind:confusion | low | agent-transcripts | session_registry.py lease-claim prints a success decision (decision=acquired, lease acquired) but returns exit code 2, so an agent that trusts the exit code will misread a successful claim as a failure and may retry or abandon it. Verifier: In the current tree, `main()` in orchestrator/src/orchestrator/session_registry.py runs `_run_lease_claim`, which prints `decision=acquired` and returns normally. `main()` then returns 0. The only exit 2 on the lease verbs is `parser.error` for an underivable slug, and that fires before anything is printed. The `except Exception` branches also return 0. So lease-claim cannot print `decision=acquired` and then exit 2. The sighting's exit 2 most likely came from another command in the same compound Bash call, since the evidence starts with a `.worktrees/6701` path. That cause is outside this tree. |  | refuted | refuted | census-dark_factory-20261124 | census-dark_factory-20261124 |  |
| entry-cand-20260918-18 | fk-a69ec55ddc08 | shared |  | scripts/legibility/reingestion.py::HARNESS_PROMPT_MARKERS | h11, h12, h10, kind:confusion | medium | agent-transcripts | Digest pipeline confuses task-instruction boilerplate (adjudication rubric, prior-round reference list, context) with actual user-provided feedback corrections |  | unverified | open | census-dark_factory-20261024 | census-dark_factory-20261024, census-dark_factory-20261031, census-dark_factory-20261107, census-dark_factory-20261124 |  |
| entry-cand-20261030-3 | fk-8dabfa972cc0 | orchestrator |  | orchestrator/src/orchestrator/dry_run_unblock.py::_ALLOWED_TOOLS | h10, h6, kind:confusion | low | agent-transcripts | The orchestrated task runs with Claude Code in don't-ask permission mode, so the agent's Bash calls for read-only inspection (cat/head/grep on the suppression key and baseline files) are denied. The agent cannot read the gate's own baseline and key files while diagnosing a post-merge verify failure, and the one failed lookup is unrecoverable from inside the session. |  | unverified | open | census-dark_factory-20261031 | census-dark_factory-20261031, census-dark_factory-20261124 |  |

## Filed Tasks

| finding_key | title | step | resolution | task_id | ticket | disposition | project_root | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fk-bb99b8dda638 | Escalation resolve_issue rejects project_root, a parameter the repo instructions teach for every task operation | curator | created | 11553 | tkt_0RXBESWAC6KZRNAPJM1FQMF63B | filed:11553 |  |  |
| fk-f1449a2f494f | Recon agent repeatedly calls count_memories_by_metadata with undeclared 'metadata' argument instead of declared 'filters' | semantic | existing | 5968 |  | filed:5968 |  |  |
| fk-7c4ee0eb5759 | Agent shell cwd left pointing at a worktree another session removed, failing subsequent commands until recovery | curator | created | 11554 | tkt_0RXBESWRHS50VR4C793YPRE9PY | filed:11554 |  |  |
| fk-c1df06e83aa5 | Spawned /team session hands the human a brief without enough context to follow the questions | curator | created | 11555 | tkt_0RXBESWYBADD907JBVD0V8N0CW | filed:11555 |  | §8 step 1c: done task 7073 carries the key and the finding is still present |
| fk-e18c7f482a29 | Interactive agent session with zero user turns contradicts session class | curator | created | 11556 | tkt_0RXBESX4BCWVYFAAGYHXMZCV7N | filed:11556 |  | §8 step 1c: done task 7440 carries the key and the finding is still present |
| fk-c2e12c9e2b12 | Digest 'User Corrections' bucket captures skill invocations and task notifications | curator | created | 11557 | tkt_0RXBESXC06EWRRDVJ9WR7FMX81 | filed:11557 |  | §8 step 1c: done task 7442 carries the key and the finding is still present |
| fk-a6083fb7ed86 | Task worktree .task/metadata.json unavailable during post-merge verification | curator | created | 11558 | tkt_0RXBESXHE29FQ7N49GP45ZGH8Y | filed:11558 |  | §8 step 1c: done task 8705 carries the key and the finding is still present |
| fk-cef39957cd50 | MCP tool parameter contamination with XML envelope markup | curator | combined | 5055 | tkt_0RXBESXK1BWDA061S4ZSDC4TQS | filed:5055 |  | §8 step 2 could not run: search_tasks: the call raised McpCallError: MCP tool search_tasks reply refused (IS_ERROR): {'content': [{'type': 'text', 'text': '{"error": "Tool call refused: \'query\' carries raw MCP envelope markup whose own boundary cannot be determined, so no repair was attempted.", "error_type": "mcp |

- Tickets from earlier runs resolved this run:
  - tkt_0RWPNGV8JKNXD6N0J024SV0ANK (entry entry-cand-20261030-3): created task 8704 (create: No existing task covers adding granted-tools guidance to the dry_run_unblock investigation prompt. Task 6203 is about git rules, not prompt wording. Create.)

## Structural

_no structural findings this run._

## Dispositions

- Re-verify at as_of_sha (D8): 11 due, 11 offered, 5 gone, 6 present, 0 inconclusive, 0 deferred by the verify cap.

| entry_id | finding_key | ticket | task_id | task_status | outcome | disposition | project_root |
| --- | --- | --- | --- | --- | --- | --- | --- |
| entry-cand-20260809-16 |  |  | 3959 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260815-17 |  |  | 4273 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260811-6 |  |  | 4274 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260812-20 |  |  | 4275 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260817-1 |  |  | 4578 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260824-28 |  |  | 4751 | done | done -> re-verified present at cd48063adec0e9f824e16e9577d1609b854a7e18 in run census-dark_factory-20261024, but structural (contract §6: not re-filed); not re-verified while this task decides the entry | open |  |
| entry-cand-20260827-23 |  |  | 4964 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260831-16 |  |  | 5120 | cancelled | abandoned (eligible to re-file; not sighted this run) | open |  |
| entry-cand-20260907-21 |  |  | 5332 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260912-25 |  |  | 5520 | cancelled | abandoned (eligible to re-file; not sighted this run) | open |  |
| entry-cand-20260918-19 |  |  | 5683 | done | done -> entry already fixed | open |  |
| entry-cand-20260915-20 |  |  | 5684 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260915-21 |  |  | 5685 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260919-1 |  |  | 5686 | done | done -> entry already fixed | open |  |
| entry-cand-20260919-2 |  |  | 5687 | done | corrected the entry's record, not its defect (decides no status) | open |  |
| entry-cand-20260919-3 |  |  | 5688 | done | done -> entry already fixed | open |  |
| entry-cand-20260916-17 |  |  | 5689 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260916-18 |  |  | 5690 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260921-25 |  |  | 5876 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260921-26 | fk-c1df06e83aa5 |  | 7073 | done | done -> re-verified present (re-filed, supersedes 7073) | filed:11555 |  |
| entry-cand-20260929-1 | fk-282c6e761d54 |  | 7074 | done | done -> re-verified gone (fixed at d575b86d667efce5b6fe883f20c6c4810729e9c0) | fixed:d575b86d667efce5b6fe883f20c6c4810729e9c0 |  |
| entry-cand-20261008-5 | fk-9c36e1e93a30 | tkt_0RW5FRBDGRGS9ZC4KJCHG6JN0B | 7438 | done | done -> re-verified gone (fixed at d575b86d667efce5b6fe883f20c6c4810729e9c0) | fixed:d575b86d667efce5b6fe883f20c6c4810729e9c0 |  |
| entry-cand-20260922-3 | fk-69414b6740fe | tkt_0RW5FRBTQM864TQ7K3NFHM6H5K | 7439 | done | done -> re-verified gone (fixed at d575b86d667efce5b6fe883f20c6c4810729e9c0) | fixed:d575b86d667efce5b6fe883f20c6c4810729e9c0 |  |
| entry-cand-20260911-28 | fk-e18c7f482a29 | tkt_0RW5FRC08TTDX8HFPEYMBKZKDV | 7440 | done | done -> re-verified present (re-filed, supersedes 7440) | filed:11556 |  |
| entry-cand-20260918-18 | fk-a69ec55ddc08 | tkt_0RW5FRC7ET51RZA763BVXH7ZAA | 7441 | done | done (task 7442 decides the entry) | open |  |
| entry-cand-20260918-18 | fk-a69ec55ddc08 |  | 7442 | done | done -> re-verified present but structural (contract §6: not re-filed) | open |  |
| entry-cand-20260916-1 | fk-c2e12c9e2b12 | tkt_0RW5FRCC10XP47EPHA3X9HRK50 | 7442 | done | done -> re-verified present (re-filed, supersedes 7442) | filed:11557 |  |
| entry-cand-20261030-3 | fk-8dabfa972cc0 |  | 6203 | done | done (not acted on: ticket tkt_0RWPNGV8JKNXD6N0J024SV0ANK awaits the curator and may become the live task) | open |  |
| entry-cand-20261030-3 | fk-8dabfa972cc0 | tkt_0RWPNGV8JKNXD6N0J024SV0ANK |  |  | ticket awaiting the curator | open |  |
| entry-cand-20261002-10 | fk-a6083fb7ed86 |  | 3160 | done | done (task 8705 decides the entry) | filed:11558 |  |
| entry-cand-20261002-10 | fk-a6083fb7ed86 | tkt_0RWPNGVF759WQH288G1WG14GA3 | 8705 | done | done -> re-verified present (re-filed, supersedes 8705) | filed:11558 |  |
| entry-cand-20260921-15 | fk-3335098ac5e9 |  | 6123 | pending | live | filed:6123 |  |
| entry-cand-20260912-7 | fk-f9b718ee0766 | tkt_0RWPNGTE5V8SDX0JVH1Z4EZ55W | 5683 | done | done -> re-verified gone (fixed at d575b86d667efce5b6fe883f20c6c4810729e9c0) | fixed:d575b86d667efce5b6fe883f20c6c4810729e9c0 |  |
| entry-cand-20260828-22 | fk-534838662cfc | tkt_0RWPNGTRS39ZWPVWQQHMDH0MXA | 2761 | done | done -> re-verified gone (fixed at d575b86d667efce5b6fe883f20c6c4810729e9c0) | fixed:d575b86d667efce5b6fe883f20c6c4810729e9c0 |  |
| entry-cand-20260825-12 | fk-cef39957cd50 | tkt_0RWPNGTWB2092RMRM1JP3SM9FK | 4696 | done | done -> re-verified present (re-filed, supersedes 4696) | filed:5055 |  |

- entry-cand-20260929-1: open -> fixed, fixed at d575b86d667efce5b6fe883f20c6c4810729e9c0
- entry-cand-20261008-5: open -> fixed, fixed at d575b86d667efce5b6fe883f20c6c4810729e9c0
- entry-cand-20260922-3: open -> fixed, fixed at d575b86d667efce5b6fe883f20c6c4810729e9c0
- entry-cand-20260912-7: open -> fixed, fixed at d575b86d667efce5b6fe883f20c6c4810729e9c0
- entry-cand-20260828-22: open -> fixed, fixed at d575b86d667efce5b6fe883f20c6c4810729e9c0

## Cost

invoke calls: sonnet miner=40, sonnet verify=22, fable synthesis=1, haiku headroom-probe=5
