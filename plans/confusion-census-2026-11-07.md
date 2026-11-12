# confusion census 2026-11-07

Project: dark_factory

## Method

```yaml
run_id: census-dark_factory-20261107
as_of_sha: 173059bd55507fdc8f527a7bb0aa435af2c2bf78
since: 2cc9ab386a56bb4a08409c605c6e35bae75ee181
evidence:
  window:
  - '2026-10-08T00:00:00+00:00'
  - '2026-11-07T00:00:00+00:00'
  sessions_enumerated: 19572
  skipped_coded: 951
  skipped_zero_signal: 14837
  mined: 40
  ledger_rows: 885
verification:
  confirmed: 6
  weakened: 0
  refuted: 4
  unverified: 0
cost:
  miner_calls: 40
  verify_calls: 12
  synthesis_calls: 1
  probe_calls: 3
  embedding_calls: 1
  wall_clock_secs: 6023.126
inputs_consumed:
- review-dark_factory-20261009
- hotspot-survey-dark_factory-20261009
- review-all-dark_factory-20261008
extra:
  inputs_consumed_note: null
  screened:
    title: 0
    session: 0
    key: 1
    cross_instrument: 0
  attached: 0
  adjudicated: 20
  ledger_created_this_run: false
  ledger_state: ok
  ledger_pruned: 0
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
    due: 2
    offered: 2
    gone: 0
    present: 2
    inconclusive: 0
    deferred: 0
```

## Saturation

- batches: 2
- stop reason: saturated
- operator batch cap: 50 batch(es) (not reached -- mining stopped by: saturated)
  - batch 0: dup_rate=1.00 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 1: dup_rate=1.00 (total=20, succeeded=20, failed=0, saturated=True)

## Screened

- per rule: title 0, session 0, key 1, cross_instrument 0
- inputs consumed: review-dark_factory-20261009, hotspot-survey-dark_factory-20261009, review-all-dark_factory-20261008
- attached by key: "Interactive session classified with zero user turns contradicts definition" (cand-20260912-20) → entry-cand-20260911-28 under fk-e18c7f482a29; primary tag disagreement: standing h1, verifier h10

## Adjudication

- model text-embedding-3-small; attach at cosine >= 0.9, cluster at >= 0.85; at most 10 queue cluster(s) per run
- embeddings: 1 request(s), 16 embedded, 1077 cached, 27 pruned
- queue depth: 986 pending candidate(s) in 972 cluster(s)
- processed this run: 10 cluster(s), 20 candidate(s)
- verified: "Capability manifest names a script that exists only in the task worktree, never git-added, so main lacks it and post-merge verify fails" (cand-20261101-3) for 2 candidate(s) across 2 session(s)
- rejected: "Watcher-rearm.sh timeout ceiling (exit 124) followed by process SIGKILL (exit 137), conflating designed stop with harness force-termination" (cand-20260925-14) for 2 candidate(s) across 2 session(s)
- verified: "Post-merge verify diagnostic harness unable to locate .task/ metadata files in task worktree" (cand-20260921-15) for 2 candidate(s) across 2 session(s)
- rejected: "Bash tool's static-analysis pre-check rejects simple read-only commands on large files via AST node count" (cand-20260915-3) for 2 candidate(s) across 2 session(s)
- verified: "Interactive session classified with zero user turns contradicts definition" (cand-20260912-20) for 2 candidate(s) across 2 session(s)
- rejected: "Digest pipeline turn-reference validation missing in worktree sessions" (cand-20260912-15) for 2 candidate(s) across 2 session(s)
- verified: "Agent constructs malformed Python one-liners in bash -c with improper statement separation" (cand-20260912-7) for 2 candidate(s) across 2 session(s)
- rejected: "ReportFindings short_summary field validation constraint not enforced before tool invocation" (cand-20260911-20) for 2 candidate(s) across 2 session(s)
- verified: "Orchestrated-task planning session abandons work mid-turn awaiting backgrounded task" (cand-20260828-22) for 2 candidate(s) across 2 session(s)
- verified: "MCP tool parameter contamination with XML envelope markup" (cand-20260825-12) for 2 candidate(s) across 2 session(s)

## Verification

- handed all 0 novel cluster(s) to the verifier; operator verify cap: 150 (not reached).

## Origin x Manifestation Matrix

| origin \ manifested | architect | implement | verify | unknown |
| --- | --- | --- | --- | --- |
| architect | 3 | 0 | 0 | 0 |
| implement | 0 | 0 | 3 | 0 |
| verify | 0 | 0 | 2 | 0 |
| ops | 0 | 0 | 0 | 1 |
| unknown | 0 | 1 | 0 | 2 |

## Synthesis

- applied: 1 phase refinement(s) to 1 cluster(s) and 1 new entr(ies); 1 correction(s) applied, 0 skipped
- standing, no new evidence: 81
- invariant violations this run: none
- unkeyed entries re-observed: 14

### New

- fk-656e5cc784ed "Capability manifest names a script that exists only in the task worktree, never git-added, so main lacks it and post-merge verify fails": The manifest at plans/mcp-network-exposure-prd.capability-manifest.yaml names scripts/check_network_exposure.py as a script-check target while itself marking that path 'absent on main', and the file is neither tracked nor present under scripts/. This rests on heuristic 10, clear invariants redundantly enforced: the invariant 'every manifest script target resolves on main' is checked at only one point, the post-merge capability-manifest tests, so a target that resolved only in the implementer's working tree reaches merge unchallenged. It also rests on heuristic 7, prefer stateless interactions between modules: the manifest's correctness depends on uncommitted working-tree state rather than on what the commit carries. The 'never git-added' step of the cause is consistent with the absent file but was not itself observed, since the worktree is gone from this view.
- fk-3335098ac5e9 "Post-merge verify diagnostic harness unable to locate .task/ metadata files in task worktree": Two sessions read .task/metadata.json inside a task worktree and hit file-not-found. orchestrator/src/orchestrator/artifacts.py::ARTIFACT_ROOTS routes metadata.json to the per-dispatch lane root under .task-meta/<lane>/, and LANE_SYMLINKED_ARTIFACTS exposes only plan.json and iterations.jsonl in the lane's .task/, so the read fails by layout, not by cleanup or a missing sync; the brief's cause wording on that point is not supported by the tree. This rests on heuristic 11, SPOT: the layout has one home in ARTIFACT_ROOTS, yet the diagnosing agent's expectation of the path came from somewhere else, and no prompt in the tree teaching .task/metadata.json was found. It also rests on heuristic 13, files make internal sense in isolation: a worktree's .task/ listing cannot be interpreted without knowing the lane-root convention held in another module.
- fk-f9b718ee0766 "Agent constructs malformed Python one-liners in bash -c with improper statement separation": Both sessions produced IndentationError from multi-statement python -c one-liners whose statements lacked ';' or newlines, matching an already-populous codebook pattern. The only Python-in-Bash preamble guidance, orchestrator/src/orchestrator/agents/python_literal_guidance.py, addresses backslash escapes in pasted text and recommends a quoted heredoc only for that case. This rests on heuristic 6, well-defined purpose for each entity: the guidance module's purpose is literal escaping, and statement separation falls outside it, so no entity owns the rule. It also rests on heuristic 11, SPOT: the quoted-heredoc recipe exists in the repo CLAUDE.md anchoring section but is not the rule the role preamble states, so the lesson has two partial homes and neither names the separator requirement.
- fk-534838662cfc "Orchestrated-task planning session abandons work mid-turn awaiting backgrounded task": Two architect sessions ended their turn with a run_in_background task still pending; shared/src/shared/cli_invoke.py classifies this as AgentFailureKind.ENDED_AWAITING_BACKGROUND with an exact summary, and orchestrator/src/orchestrator/agents/harness_guidance.py already carries the 'never end your turn while a backgrounded command is still running' rule that the ARCHITECT role embeds via BASH_CAPABLE_ROLE_PREAMBLE. This rests on heuristic 10, clear invariants informatively, redundantly, uniformly enforced: the invariant is stated to the agent and detected after the fact, so the breach is named and recoverable, but there is no enforcement point between the two that prevents the turn from ending. It also rests on heuristic 11, SPOT: the rule has a single home in harness_guidance.py, which is what makes it possible to say the guard exists and was not followed, rather than that it was missing.
- fk-cef39957cd50 "MCP tool parameter contamination with XML envelope markup": Both sessions carried '</invoke>' or '</rationale>' envelope markup inside MCP parameter values and were rejected with error_type mcp_markup_detected. shared/src/shared/toolcall_markup.py, shared/src/shared/mcp_markup_middleware.py, orchestrator/src/orchestrator/mcp/plan_markup_stamp.py and fused-memory/src/fused_memory/server/markup_guard.py all detect this shape, and docs/mcp-toolcall-xml-leak.md plus plans/toolcall-markup-containment-prd.md document it. This rests on heuristic 10, clear invariants informatively, redundantly, uniformly enforced: the rejection names the invariant via a single error_type, is applied at more than one boundary, and recovers the intended parameters, which is what contains the cost to retry turns. The leak originates in tool-call serialization outside this tree, so nothing here produces it; the finding records a contained recurrence of a known pattern rather than a gap.

### Re-observed

- fk-282c6e761d54 "Escalation watcher emits get_task_escalations calls with an empty-valued optional parameter (`"status": ,`), producing unparseable JSON that recurs across tasks" (gained f9802ab5-f4dc-4088-9a72-f5ffcb3b4074, e2a2a80c-a139-4b6b-b7b9-47f2ee321c46, d160a231-f09d-42d7-a9ee-c452220dd92c): no change stated
- fk-e18c7f482a29 "Interactive agent session with zero user turns contradicts session class" (gained 05c46d31-e961-422f-8baa-5881ba2d5311, 9f891804-fa8b-4429-89d5-4f8e514acee1): no change stated
- fk-a69ec55ddc08 "Digest 'User Corrections' bucket mislabels injected task context/briefing as a correction" (gained f12ca583-3ed5-4554-a173-2e37cdd5d436, f36d9c46-539b-4333-8d9d-9348799370d8): no change stated

### Phase Refinements

- fk-656e5cc784ed: origin implement, manifested verify (The manifest that names the script was authored during implementation, and the verdict observes that the manifest itself marks the target 'absent on main', so the defective artifact was produced in the implement phase; the stamp's 'unknown' origin is less precise than that evidence. The failure surfaced in the post-merge capability-manifest tests, which is the verify phase as stamped.)

### Corrections

- entry-cand-20260911-28: cause "AgentClassEvidence.agent_class treats INTERACTIVE as the unguarded catch-all: when records exist but no agent-class marker matches, it returns INTERACTIVE regardless of whether a primary user turn was observed, while render_digest derives n_user_turns independently. A transcript of only assistant, system, isMeta or sidechain records therefore renders as agent_class: interactive with n_user_turns: 0." (This run's verdict traced the output to scripts/legibility/detectors.py::AgentClassEvidence.agent_class, which returns UNKNOWN only for an empty record set and falls through to AgentClass.INTERACTIVE whenever no marker matches, without checking whether a primary user turn exists; scripts/legibility/digest.py::render_digest counts n_user_turns separately and never reconciles the two. The standing cause's alternative of a bootstrap or initialization failure is not what the code path shows, so the cause is narrowed to the classification fall-through. The two cited transcripts were not themselves read.)

### Notes

- unverified: Four of this run's five new findings (fk-f9b718ee0766, fk-534838662cfc, fk-cef39957cd50 and, by its own verdict, fk-3335098ac5e9) describe recurrences for which a guard, classifier or documented layout already exists in the tree; only fk-656e5cc784ed names an enforcement gap. The pattern to weigh is agents not acting on standing guidance rather than missing guidance. (evidence: Each verdict's reason cites the existing containment: python_literal_guidance.py and harness_guidance.py for the one-liner and background-task findings, the four markup-guard modules for the XML-envelope finding, and ARTIFACT_ROOTS for the .task/metadata.json finding.)
- unverified: The brief's cause text for fk-3335098ac5e9 ('likely cleaned up after task completion or never synced') is contradicted by its own verdict, which shows metadata.json is routed elsewhere by design; since the finding is new rather than re-observed, no correction was issued here and the created entry's cause should be read against that verdict. (evidence: The verdict reason for fk-3335098ac5e9 in this brief, describing ARTIFACT_ROOTS and LANE_SYMLINKED_ARTIFACTS.)
- unverified: fk-282c6e761d54 gained three sessions this run with no verdict attached, which is the largest session gain in this brief for an open finding; the recurrence across tasks that its cause describes continues without a confirming or refuting read of the tree. (evidence: The brief lists three gained_sessions and a null verdict for this key.)
- unverified: The brief reports 14 unkeyed re-observations alongside 3 keyed ones, so most of this run's recurrence evidence attached to no finding key and is not reflected in the observations above. (evidence: unkeyed_re_observed: 14 in the brief; the keyed re_observed list holds three entries.)

## Findings

- area map: briefing review/briefing.yaml; areas dashboard, escalation, fused-memory, orchestrator, repo, sampler, shared; overlay .claude/skills/review-all/project.md; rules `scripts/legibility/**` → `shared`, `scripts/**` → `repo`
- 2 re-observed finding(s) are unverified: the census does not re-verify a catalogued finding, so a sighting this run only appends to its last_seen.

| id | key | area | sub_area | anchor | tags | severity | evidence_source | statement | proposal | verdict | disposition | first_seen | last_seen | supersedes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| entry-cand-20261101-3 | fk-656e5cc784ed | repo |  | plans/mcp-network-exposure-prd.capability-manifest.yaml | h10, h7, kind:confusion | medium | agent-transcripts, present-tree | The implementer's working-tree file was never committed. Local runs see it, but main does not, so the manifest's script-check target ERRORs after merge. The gap is only caught post-merge by the capability-manifest tests. Verifier: The capability manifest plans/mcp-network-exposure-prd.capability-manifest.yaml still lists scripts/check_network_exposure.py as a script-check target. The script is neither tracked by git nor present in /home/leo/src/dark-factory/scripts, so a post-merge script check has nothing to run. The manifest itself marks the path as 'absent on main', so the missing file is observable. The 'never git-added' cause is not directly checkable, because the task worktree is gone from this view. The gap is only caught post-merge, because nothing in the tree compares worktree-untracked files against the manifest's script targets before merge. |  | confirmed | open | census-dark_factory-20261107 | census-dark_factory-20261107 |  |
| entry-cand-20260921-15 | fk-3335098ac5e9 | orchestrator |  | orchestrator/src/orchestrator/artifacts.py::ARTIFACT_ROOTS | h11, h13, kind:confusion | low | agent-transcripts, present-tree | When post-merge verification fails and the diagnostic agent attempts to reconstruct task execution state by reading .task/metadata.json and .task/iterations.jsonl from the original task worktree, these files don't exist (likely cleaned up after task completion or never synced to that location). The worktree is orphaned from its iteration metadata, breaking the post-merge diagnostics feedback loop. Verifier: In the current tree, `.task/metadata.json` is not expected to exist in a task worktree. `orchestrator/src/orchestrator/artifacts.py::ARTIFACT_ROOTS` routes `metadata.json` to the per-dispatch LANE root under `.task-meta/<lane>/`. `LANE_SYMLINKED_ARTIFACTS` exposes only `plan.json` and `iterations.jsonl` inside the lane's `.task/`. `iterations.jsonl` is a symlink into durable task storage, and the briefings say it is absent until the first iteration is logged. A diagnostic agent that reads `.task/metadata.json` will therefore always hit file-not-found. The `.task/` listing in the sighting (`plan.json`, `verify`, `reviews`, no `metadata.json`) matches this layout. I did not find any prompt in the tree that tells an agent to read `.task/metadata.json`, so I cannot name a specific file that teaches the wrong path. The post-merge diagnostic prompt may live in tooling outside this tree. |  | confirmed | filed:6123 | census-dark_factory-20261107 | census-dark_factory-20261107 |  |
| entry-cand-20260911-28 | fk-e18c7f482a29 | shared |  | scripts/legibility/detectors.py::AgentClassEvidence | h1, h12, kind:confusion | low | agent-transcripts, present-tree | Session digest pipeline marks sessions as 'interactive' despite n_user_turns=0, indicating either void/aborted sessions being captured, misclassified automated runs, or trickle-coder's own meta-session (analysis-in-progress with no user interaction yet) being ingested Verifier: Verified in the current tree. `scripts/legibility/detectors.py::AgentClassEvidence.agent_class` returns UNKNOWN only when there are no records at all. When records exist and no marker matches, it falls through to `AgentClass.INTERACTIVE`. It never checks whether a primary user turn exists (`first_turn_text` stays None). `scripts/legibility/digest.py::render_digest` counts `n_user_turns` separately and takes `agent_class` as an input, so nothing reconciles the two. A transcript made only of assistant, system or isMeta records, or only sidechain records, therefore renders as `agent_class: interactive` with `n_user_turns: 0` and score 0.0. The `AgentClass` docstring says UNKNOWN is for an empty transcript, so INTERACTIVE is the unguarded catch-all. I did not check the two cited sessions' transcripts. I confirmed only that the code path can produce this output. | In AgentClassEvidence.agent_class, return AgentClass.UNKNOWN instead of INTERACTIVE when no marker matched and first_turn_text is None (no primary user turn was observed), and update the AgentClass docstring to say UNKNOWN covers sessions with no user turn. | confirmed | filed:7440 | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| entry-cand-20260912-7 | fk-f9b718ee0766 | orchestrator |  | orchestrator/src/orchestrator/agents/python_literal_guidance.py | h6, h11, kind:confusion | low | agent-transcripts, present-tree | When embedding Python code in `bash -c 'python3 -c ...'` commands, the agent concatenates multiple statements without proper separators (semicolons or newlines), causing Python IndentationError on parse Verifier: The failure is still observable. The codebook at docs/legibility/confusion-codebook.yaml already holds many sightings of IndentationError from multi-statement `python -c` commands collapsed onto one line, so this cluster repeats a known pattern. The only Python-in-Bash guidance in the role preamble is orchestrator/src/orchestrator/agents/python_literal_guidance.py. It covers backslash escapes in pasted text and tells agents to use a quoted heredoc for scripts that carry pasted text. It does not say to avoid multi-statement `python -c` one-liners, and neither it nor harness_guidance.py states the general rule that statements need `;` or newlines. The cause of the missing separators sits in the model's generation, which I cannot verify from this tree. | Add a short rule that any multi-statement Python in Bash goes through a quoted heredoc (`python3 - <<'PY'`) rather than a `python -c` one-liner, because statements on one line need `;` or real newlines. | confirmed | filed:5683 | census-dark_factory-20261107 | census-dark_factory-20261107 |  |
| entry-cand-20260828-22 | fk-534838662cfc | orchestrator |  | orchestrator/src/orchestrator/agents/harness_guidance.py | h10, h11, kind:confusion | low | agent-transcripts, present-tree | Agent launched a background task (run_in_background=true) but ended its turn before the task completed, or the harness reaped the task before completion, leaving work unfinished and no recovery path Verifier: The failure is real and recurring. shared/src/shared/cli_invoke.py classifies it as AgentFailureKind.ENDED_AWAITING_BACKGROUND with the exact summary 'agent ended its turn awaiting a still-pending backgrounded task (work abandoned mid-turn)'. plans/agent-capacity-study-2026-09-09/S4-reify.md and plans/portfolio-review-2026-11-01/followups/model-routing-seats/U-unplannable.md cite it as 'architect failed (ended_awaiting_background)'. The architect role (orchestrator/src/orchestrator/agents/roles.py::ARCHITECT) already embeds BASH_CAPABLE_ROLE_PREAMBLE. That preamble (orchestrator/src/orchestrator/agents/harness_guidance.py) carries an explicit 'never end your turn while a backgrounded command is still running' rule and polling guidance. So the guard exists in the tree and the agents still occasionally ignore it. I found no remaining gap in an existing file that I can name. |  | confirmed | filed:2761 | census-dark_factory-20261107 | census-dark_factory-20261107 |  |
| entry-cand-20260825-12 | fk-cef39957cd50 | shared |  | shared/src/shared/mcp_markup_middleware.py | h10, kind:confusion | low | agent-transcripts, present-tree | Agent serializes XML tags (</invoke>, </rationale>) from tool-call envelope into MCP parameter field values, causing systematic tool rejection Verifier: The containment for this confusion is present in the tree. shared/src/shared/toolcall_markup.py, shared/src/shared/mcp_markup_middleware.py, orchestrator/src/orchestrator/mcp/plan_markup_stamp.py and fused-memory/src/fused_memory/server/markup_guard.py detect envelope markup such as '</invoke>' and '<parameter name=' and reject the call with error_type mcp_markup_detected. docs/mcp-toolcall-xml-leak.md and plans/toolcall-markup-containment-prd.md document the defect. docs/legibility/confusion-codebook.yaml already holds several entries for it, including add_design_decision with '</invoke>' or '</rationale>' in 'rationale' or 'decision' (around lines 3931, 3937, 22871 and 22900). Both sightings match that existing pattern, so the cluster is not novel. The cause is the model's own tool-call serialization, which this tree does not contain. No existing file here produces, teaches or fails to guard against it. |  | confirmed | filed:4696 | census-dark_factory-20261107 | census-dark_factory-20261107 |  |
| cand-20260925-14 | fk-5a0f1665d9b2 | repo |  | skills/escalation-watcher/SKILL.md | h11, kind:confusion | low | agent-transcripts | Rotation supervisor or harness forcibly kills watcher-rearm.sh process after it emits designed-timeout exit code (124), creating ambiguous exit semantics and spurious tool_error signal Verifier: This cluster is already documented and guarded in the current tree. `skills/escalation-watcher/SKILL.md` has an outcome table that separates `WATCHER_REARM_OUTCOME: CEILING` (exit 124, the designed hourly heartbeat that the cap exempts) from `KILLED` (exit 137, 143 or 144). It tells the watcher not to re-issue a killed foreground arm, and a diagnostic section covers the case where a missing marker means an external SIGKILL to the process group. The codebook `docs/legibility/confusion-codebook.yaml` already holds many near-identical sightings: repeated exit-124 CEILING outcomes, exit-137 kills of the long wait, and the digest scorer counting 124 as `tool_error` (entry-cand-20260729-2). The claim that a rotation supervisor or harness force-kills the process after it exits 124 is a root-cause diagnosis. I could not verify it from this tree, because the digest does not show who sent the kill and the harness is not in this tree. So there is no novel, observable defect, and no existing file in this tree that fails to guard against it. |  | refuted | refuted | census-dark_factory-20261107 | census-dark_factory-20261107 |  |
| cand-20260915-3 | fk-ddeb06e64a70 | repo |  | slug:bash-ast-node-limit-precheck | h10, kind:confusion | low | agent-transcripts | Pre-check enforces a hard limit of 100000 AST nodes and rejects benign shell commands when target files exceed this threshold, misreporting the limit breach as 'malicious input' Verifier: The 'Too many AST nodes' error comes from the Bash tool's harness-side parser, which this tree doesn't contain. The only grep hits are codebook entries, calibration data and survey query files that record the sightings. The codebook (docs/legibility/confusion-codebook.yaml, around line 1362) says the architect could not reproduce it on CLI 2.1.293 and that task 5518's premise of a live hazard was stale. The sightings are from 2026-09-12 and 2026-09-15. I did not re-run the failing commands. So the current-state claim is not confirmed, and no file in this tree produces, teaches or fails to guard against it. |  | refuted | refuted | census-dark_factory-20261107 | census-dark_factory-20261107 |  |
| cand-20260912-15 | fk-fdfcc91230c2 | shared |  | scripts/legibility/digest.py::iter_not_found | kind:confusion | low | agent-transcripts | Digest extraction attempts to report on turns (28, 50) that exceed or fall outside the session's actual transcript bounds, generating bare 'does not exist' errors instead of diagnostics about index validation, session truncation, or transcript loading failures. Verifier: The claim doesn't match the current tree. In scripts/legibility/digest.py, `iter_not_found` only collects not-found literals from tool_result content, and the output line `- (turn N) ...` is just the index of the matching record. The 'does not exist' text is ordinary tool output (for example a Read of a missing path) quoted from the transcript. It is not the digest reporting on turns outside the transcript bounds. I found no code that validates or reports on turn references beyond a session's range. The evidence reads like a miner misreading a not_found count (not_found: 2) alongside tiny session sizes, not a digest defect I can observe. |  | refuted | refuted | census-dark_factory-20261107 | census-dark_factory-20261107 |  |
| cand-20260911-20 | fk-f611b1591f4a | repo |  | slug:reportfindings-short-summary-limit | h10, kind:confusion | low | agent-transcripts | Agent invoked ReportFindings with short_summary values exceeding 60-character maximum on two attempts (turns 107, 120), suggesting the constraint is either undocumented at the point of tool invocation or agent was not constraining values before submission. Verifier: The 60-character short_summary limit is defined in the ReportFindings tool schema, which belongs to the harness and not to this tree. The only matches in the tree are the codebook entries recording this confusion, two test files, and an audit script. No existing prompt, skill or doc in the tree defines or teaches this limit, so I could not observe a file here that causes or fails to guard against it. The evidence is two transcript errors that the schema already rejects with a clear message and that the agent can retry past. The cause lies in tooling this tree does not contain. |  | refuted | refuted | census-dark_factory-20261107 | census-dark_factory-20261107 |  |
| entry-cand-20260929-1 | fk-282c6e761d54 | repo |  | skills/escalation-watcher-auto/SKILL.md | h12, h11, kind:confusion | low | agent-transcripts | When the watcher wants to omit the optional status filter, it writes the key with no value instead of dropping it. The harness then rejects the whole call with InputValidationError, wasting a turn. The same malformed shape recurred for tasks 6060 and 5813, 63 turns apart, so nothing carries the lesson forward. The tool's parameter docs or the watcher prompt could state that optional filters must be omitted, not left blank. |  | unverified | filed:7074 | census-dark_factory-20261017 | census-dark_factory-20261017, census-dark_factory-20261024, census-dark_factory-20261031, census-dark_factory-20261107 |  |
| entry-cand-20260918-18 | fk-a69ec55ddc08 | shared |  | scripts/legibility/reingestion.py::HARNESS_PROMPT_MARKERS | h11, h12, h10, kind:confusion | medium | agent-transcripts | Digest pipeline confuses task-instruction boilerplate (adjudication rubric, prior-round reference list, context) with actual user-provided feedback corrections |  | unverified | filed:7442 | census-dark_factory-20261024 | census-dark_factory-20261024, census-dark_factory-20261031, census-dark_factory-20261107 |  |

## Filed Tasks

| finding_key | title | step | resolution | task_id | ticket | disposition | project_root | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fk-3335098ac5e9 | Post-merge verify diagnostic harness unable to locate .task/ metadata files in task worktree | semantic | existing | 6123 |  | filed:6123 |  |  |
| fk-e18c7f482a29 | Interactive session classified with zero user turns contradicts definition | key | existing | 7440 |  | filed:7440 |  |  |
| fk-f9b718ee0766 | Agent constructs malformed Python one-liners in bash -c with improper statement separation | curator | combined | 5683 | tkt_0RWPNGTE5V8SDX0JVH1Z4EZ55W | filed:5683 |  |  |
| fk-534838662cfc | Orchestrated-task planning session abandons work mid-turn awaiting backgrounded task | curator | combined | 2761 | tkt_0RWPNGTRS39ZWPVWQQHMDH0MXA | filed:2761 |  |  |
| fk-cef39957cd50 | MCP tool parameter contamination with XML envelope markup | curator | combined | 4696 | tkt_0RWPNGTWB2092RMRM1JP3SM9FK | filed:4696 |  | §8 step 2 could not run: search_tasks: the call raised McpCallError: MCP tool search_tasks reply refused (IS_ERROR): {'content': [{'type': 'text', 'text': '{"error": "Tool call refused: \'query\' carries raw MCP envelope markup whose own boundary cannot be determined, so no repair was attempted.", "error_type": "mcp |
| fk-8dabfa972cc0 | Orchestrated-task sub-agent denied read-only Bash inspection in don't-ask mode, forcing blind guesses while diagnosing a verify failure | curator |  |  | tkt_0RWPNGV8JKNXD6N0J024SV0ANK | open (ticket tkt_0RWPNGV8JKNXD6N0J024SV0ANK pending) |  | pending with the curator; resolves on the next run |
| fk-a6083fb7ed86 | Task worktree .task/metadata.json unavailable during post-merge verification | curator | created | 8705 | tkt_0RWPNGVF759WQH288G1WG14GA3 | filed:8705 |  |  |

## Structural

These verified findings are structural (contract §6, D6): recorded in the codebook and NOT filed; each goes to deliberation with the human, then a program doc and /prd.

| finding_key | entry_id | title | anchor |
| --- | --- | --- | --- |
| fk-656e5cc784ed | entry-cand-20261101-3 | Capability manifest names a script that exists only in the task worktree, never git-added, so main lacks it and post-merge verify fails | plans/mcp-network-exposure-prd.capability-manifest.yaml |

## Dispositions

- Re-verify at as_of_sha (D8): 2 due, 2 offered, 0 gone, 2 present, 0 inconclusive, 0 deferred by the verify cap.

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
| entry-cand-20260918-19 |  |  | 5683 | done | done -> entry already fixed | filed:5683 |  |
| entry-cand-20260915-20 |  |  | 5684 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260915-21 |  |  | 5685 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260919-1 |  |  | 5686 | done | done -> entry already fixed | open |  |
| entry-cand-20260919-2 |  |  | 5687 | done | corrected the entry's record, not its defect (decides no status) | open |  |
| entry-cand-20260919-3 |  |  | 5688 | done | done -> entry already fixed | open |  |
| entry-cand-20260916-17 |  |  | 5689 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260916-18 |  |  | 5690 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260921-25 |  |  | 5876 | done | done -> entry already fixed | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260921-26 | fk-c1df06e83aa5 |  | 7073 | pending | live | filed:7073 |  |
| entry-cand-20260929-1 | fk-282c6e761d54 |  | 7074 | pending | live | filed:7074 |  |
| entry-cand-20261008-5 | fk-9c36e1e93a30 | tkt_0RW5FRBDGRGS9ZC4KJCHG6JN0B | 7438 | pending | live | filed:7438 |  |
| entry-cand-20260922-3 | fk-69414b6740fe | tkt_0RW5FRBTQM864TQ7K3NFHM6H5K | 7439 | pending | live | filed:7439 |  |
| entry-cand-20260911-28 | fk-e18c7f482a29 | tkt_0RW5FRC08TTDX8HFPEYMBKZKDV | 7440 | pending | live | filed:7440 |  |
| entry-cand-20260918-18 | fk-a69ec55ddc08 | tkt_0RW5FRC7ET51RZA763BVXH7ZAA | 7441 | done | done (task 7442 decides the entry) | filed:7442 |  |
| entry-cand-20260918-18 | fk-a69ec55ddc08 |  | 7442 | pending | live | filed:7442 |  |
| entry-cand-20260916-1 | fk-c2e12c9e2b12 | tkt_0RW5FRCC10XP47EPHA3X9HRK50 | 7442 | pending | live | filed:7442 |  |
| entry-cand-20261030-3 | fk-8dabfa972cc0 |  | 6203 | done | done -> re-verified present (re-filed, supersedes 6203) | open |  |
| entry-cand-20261002-10 | fk-a6083fb7ed86 |  | 3160 | done | done -> re-verified present (re-filed, supersedes 3160) | filed:8705 |  |

_no outcome changed this run._

## Cost

invoke calls: sonnet miner=40, sonnet verify=12, fable synthesis=1, haiku headroom-probe=3
