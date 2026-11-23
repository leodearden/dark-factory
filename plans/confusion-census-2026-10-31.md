# confusion census 2026-10-31

Project: dark_factory

## Method

```yaml
run_id: census-dark_factory-20261031
as_of_sha: 2cc9ab386a56bb4a08409c605c6e35bae75ee181
since: cd48063adec0e9f824e16e9577d1609b854a7e18
evidence:
  window:
  - '2026-10-01T00:00:00+00:00'
  - '2026-10-31T00:00:00+00:00'
  sessions_enumerated: 18266
  skipped_coded: 781
  skipped_zero_signal: 13800
  mined: 40
  ledger_rows: 666
verification:
  confirmed: 3
  weakened: 0
  refuted: 7
  unverified: 0
cost:
  miner_calls: 40
  verify_calls: 11
  synthesis_calls: 1
  probe_calls: 3
  embedding_calls: 1
  wall_clock_secs: 4667.008
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
  adjudicated: 27
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
    due: 1
    offered: 1
    gone: 0
    present: 1
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
- attached by key: "Digest 'User Corrections' bucket captures task failure/block information from post-merge verification" (cand-20260925-18) → entry-cand-20260918-18 under fk-a69ec55ddc08

## Adjudication

- model text-embedding-3-small; attach at cosine >= 0.9, cluster at >= 0.85; at most 10 queue cluster(s) per run
- embeddings: 1 request(s), 27 embedded, 1077 cached, 73 pruned
- queue depth: 999 pending candidate(s) in 969 cluster(s)
- processed this run: 10 cluster(s), 27 candidate(s)
- verified: "Orchestrated-task sub-agent denied read-only Bash inspection in don't-ask mode, forcing blind guesses while diagnosing a verify failure" (cand-20261030-3) for 6 candidate(s) across 6 session(s)
- rejected: "Digest Retry Loops designed-outcome count contradicts Designed Outcomes section" (cand-20260912-22) for 3 candidate(s) across 3 session(s)
- rejected: "Digest pipeline includes zero-signal sessions in codebook analysis" (cand-20260911-45) for 3 candidate(s) across 3 session(s)
- rejected: "Digest mislabels supervisor-injected watcher rotation configuration as User Corrections" (cand-20260908-14) for 3 candidate(s) across 3 session(s)
- rejected: "Capability manifest declares a script target that exists only as an untracked working-tree file, so the check errors on main" (cand-20261019-3) for 2 candidate(s) across 2 session(s)
- rejected: "Wall-clock p99 latency assertion flakes under xdist load and blocks an unrelated merge as a 'new failing test'" (cand-20261018-1) for 2 candidate(s) across 2 session(s)
- rejected: "Orchestrated-task worktree session produces zero output and registers no diagnostic signals" (cand-20261002-15) for 2 candidate(s) across 2 session(s)
- verified: "Task worktree .task/metadata.json unavailable during post-merge verification" (cand-20261002-10) for 2 candidate(s) across 2 session(s)
- rejected: "Host `skim rewrite` hook turns a multi-file `cat a.py b.py` into a single-file `skim` call, which rejects the second path" (cand-20260927-19) for 2 candidate(s) across 2 session(s)
- verified: "Digest 'User Corrections' bucket captures task failure/block information from post-merge verification" (cand-20260925-18) for 2 candidate(s) across 2 session(s)

## Verification

- handed all 0 novel cluster(s) to the verifier; operator verify cap: 150 (not reached).

## Origin x Manifestation Matrix

| origin \ manifested | verify | ops |
| --- | --- | --- |
| architect | 6 | 0 |
| implement | 2 | 2 |

## Synthesis

- applied: 3 phase refinement(s) to 3 cluster(s) and 2 new entr(ies); 0 correction(s) applied, 0 skipped
- standing, no new evidence: 76
- invariant violations this run: contracts-machine-checked 2
- unkeyed entries re-observed: 17

### New

- fk-8dabfa972cc0 "Orchestrated-task sub-agent denied read-only Bash inspection in don't-ask mode, forcing blind guesses while diagnosing a verify failure": The envelope in `orchestrator/src/orchestrator/dry_run_unblock.py::_ALLOWED_TOOLS` grants Read, Glob and Grep but not Bash, so a `cat`/`head`/`grep` from an agent under `permission_mode='dontAsk'` is denied by design while the same inspection is available through the granted tools. Heuristic 10, clear invariants informatively enforced: the denial states that the call is refused, not which granted tool performs the same read-only inspection, so the agent cannot recover from the message alone and guesses instead. Heuristic 6, well-defined purpose for each entity: the allow-list's purpose (read-only inspection) is stated only in a comment addressed to reviewers, and no agent-facing guidance names Read/Grep as the sanctioned substitute, so the capability exists but is not discoverable from inside the session. The verdict could not tie the six sessions to a specific dispatch path; what is reproducible is the symptom under either `dontAsk` path (dry_run_unblock, reconciliation verify CLI).
- fk-a6083fb7ed86 "Task worktree .task/metadata.json unavailable during post-merge verification": Task metadata is written to `<worktree_base>/.task-meta/<name>/metadata.json` via `TaskArtifacts.meta_root_for`, while the module docstring of `orchestrator/src/orchestrator/artifacts.py`, the `TaskArtifacts` class docstring and several method docstrings still say `.task/` inside the worktree, so a read of `<worktree>/.task/metadata.json` fails by design. Heuristic 1, informative names: the docstrings name a location the code does not use, which is the lying-name class of defect. Heuristic 11, SPOT: the artifact root is defined once in `meta_root_for` and restated, differently, in prose across the same file. Comments stance: this is drift in prose that outlived the code change it described, and the verdict found no agent-facing text in `skills/` that points at the `.task-meta` location either.

### Re-observed

- fk-282c6e761d54 "Escalation watcher emits get_task_escalations calls with an empty-valued optional parameter (`"status": ,`), producing unparseable JSON that recurs across tasks" (gained 29949602-2812-4a56-8a06-bbc1f0d82635, 99ad7280-9724-4039-acac-43ef62fcb780): no change stated
- fk-a69ec55ddc08 "Digest 'User Corrections' bucket mislabels injected task context/briefing as a correction" (gained 8f523090-51b1-4272-892a-931e06adf81c, 36dd7939-aead-4582-8955-35ff8027be76, 3a0018de-2b55-46ee-b5ec-3726fa6ca136, aac8ba6d-4af2-4969-b612-a76a15f4a74e, 41aaa54b-a6b2-4466-87b0-acce7eda41ab, 9407614a-7d48-456a-8a17-e62f5944fb91, 6498d18b-6a3a-4f6c-bd04-637c92b9a268): no change stated
- fk-c2e12c9e2b12 "Digest 'User Corrections' bucket captures skill invocations and task notifications" (gained 36dd7939-aead-4582-8955-35ff8027be76): no change stated

### Phase Refinements

- fk-8dabfa972cc0: origin architect, manifested verify (The verdict traces the denial to a deliberate design of the dispatch envelope (`permission_mode='dontAsk'` with a narrow read-only `_ALLOWED_TOOLS`), not to a defect in any one implementation; the gap is in what that envelope tells the agent, which places origin at the design of the path rather than unknown. Manifestation stays at verify, where the sessions were diagnosing a post-merge verify failure.)
- fk-a6083fb7ed86: origin implement, manifested verify (The verdict shows the `.task/` docstrings were left standing when callers moved to `meta_root_for` and `.task-meta/`; that omission is an implementation-phase change that did not carry its prose with it. The sessions hit it while investigating a post-merge verify failure, so manifestation stays at verify.)
- fk-a69ec55ddc08: origin implement, manifested ops (The verdict locates the defect in `scripts/legibility/reingestion.py::is_harness_injected_turn`, whose marker sets miss the `dry_run_unblock` user prompt (origin: an implementation omission in the legibility scripts). The mislabel surfaces in the digest consumed by the trickle coder and census, which is operator legibility tooling, not a review of a task's change; ops places it better than the standing review stamp.)

### Notes

- unverified: fk-a69ec55ddc08 and fk-c2e12c9e2b12 may be one gap in `is_harness_injected_turn` manifesting on two injector shapes rather than two independent findings. (evidence: Both gained session 36dd7939-aead-4582-8955-35ff8027be76 this run, both concern the digest's 'User Corrections' bucket, and the fk-a69ec55ddc08 verdict itself frames the sighting as the known harness-injected-prompt-as-User-Correction defect for one more injector shape.)
- unverified: The six fk-8dabfa972cc0 sessions were probably dispatched through one of the two `dontAsk` paths (dry_run_unblock or the reconciliation verify CLI), since the verdict states regular task agents default to `bypassPermissions`. (evidence: The verdict reports it could not tie the sightings to a dispatch path, but the denial is only reproducible under `dontAsk`; the inference is the synthesis's, not the verdict's.)
- unverified: The fk-282c6e761d54 cause's claim that the malformed `"status": ,` shape recurs across tasks with nothing carrying the lesson forward is not re-verified by this run. (evidence: The finding gained two sessions this run with a null verdict; the brief supplies no new evidence about the watcher prompt or tool parameter docs.)

## Findings

- area map: briefing review/briefing.yaml; areas dashboard, escalation, fused-memory, orchestrator, repo, sampler, shared; overlay .claude/skills/review-all/project.md; rules `scripts/legibility/**` → `shared`, `scripts/**` → `repo`
- 2 re-observed finding(s) are unverified: the census does not re-verify a catalogued finding, so a sighting this run only appends to its last_seen.

| id | key | area | sub_area | anchor | tags | severity | evidence_source | statement | proposal | verdict | disposition | first_seen | last_seen | supersedes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| entry-cand-20261030-3 | fk-8dabfa972cc0 | orchestrator |  | orchestrator/src/orchestrator/dry_run_unblock.py::_ALLOWED_TOOLS | h10, h6, kind:confusion | low | agent-transcripts, present-tree | The orchestrated task runs with Claude Code in don't-ask permission mode, so the agent's Bash calls for read-only inspection (cat/head/grep on the suppression key and baseline files) are denied. The agent cannot read the gate's own baseline and key files while diagnosing a post-merge verify failure, and the one failed lookup is unrecoverable from inside the session. Verifier: Partly confirmed. The tree deliberately runs some agents under `--permission-mode dontAsk` with a narrow allow list. `orchestrator/src/orchestrator/dry_run_unblock.py` passes `permission_mode='dontAsk'` and its `_ALLOWED_TOOLS` grants only Read, Glob, Grep, a few read-only `git` subcommands and a few fused-memory tools. The reconciliation verify CLI also uses `dontAsk` (asserted in `fused-memory/tests/reconciliation/test_verify_cli_invocation.py`). Under that envelope a Bash `cat`, `head` or `grep` is denied with the quoted message, so the symptom is reproducible by design. I could not tie the six sightings to one of these dispatch paths. Regular orchestrated task agents default to `bypassPermissions` in `orchestrator/src/orchestrator/agents/invoke.py`. The envelope comment says only that every entry is a grant and must be read-only. It does not tell the agent to use Read or Grep for file inspection, so the denial and the blind guessing are a prompt-guidance gap rather than a missing capability. I found no single existing file that clearly produces or fails to guard against it. |  | confirmed | filed:6203 | census-dark_factory-20261031 | census-dark_factory-20261031 |  |
| entry-cand-20261002-10 | fk-a6083fb7ed86 | orchestrator |  | orchestrator/src/orchestrator/artifacts.py::TaskArtifacts | h1, h11, comments, kind:confusion | low | agent-transcripts, present-tree | Agent investigating post-merge test failure attempts to read task metadata file which does not exist in the worktree .task/ directory Verifier: The sighting still describes the current tree. Task metadata is no longer written to `<worktree>/.task/metadata.json`. Production callers pass `meta_root=TaskArtifacts.meta_root_for(...)`, so it lives at `<worktree_base>/.task-meta/<name>/metadata.json`, a sibling of the worktree (for example under `.worktrees/.task-meta/`). A read of `.worktrees/4555/.task/metadata.json` therefore fails by design. The same file still tells readers otherwise: the module docstring says it 'Manages the .task/ directory structure in each worktree', and the `TaskArtifacts` class docstring says 'Manages .task/ directory in a worktree'. Many method docstrings also still say '.task/...'. I did not find any agent-facing text in skills/ pointing to the `.task-meta` location. So an agent that goes by these names looks in the wrong place. | Reword the module and `TaskArtifacts` docstrings (and the per-artifact '.task/...' mentions) to say artifacts live under the `.task-meta/<worktree_name>` root returned by `meta_root_for`, with `<worktree>/.task` named only as the legacy fallback. | confirmed | filed:3160 | census-dark_factory-20261031 | census-dark_factory-20261031 |  |
| entry-cand-20260918-18 | fk-a69ec55ddc08 | shared |  | scripts/legibility/reingestion.py::HARNESS_PROMPT_MARKERS | h11, h12, h10, kind:confusion | medium | agent-transcripts, present-tree | Digest generator mislabels task failure and block reasons (Post-merge verification failed) as 'User Corrections' instead of a task-failure bucket Verifier: The 'Task ID / Worktree / Block reason' text is the user prompt built in orchestrator/src/orchestrator/dry_run_unblock.py. It reads 'Task ID: {task_id}\nWorktree: {worktree}\nBlock reason: {reason}\nDetail: ...\n\n{task_context}\n\nInvestigate and emit your structured proposal.' scripts/legibility/reingestion.py::is_harness_injected_turn does not recognise it. Its briefing heading rule, HARNESS_HEADING_SETS, HARNESS_PROMPT_MARKERS and the '# context' memory-block rule all miss it. The only place that knows this shape is scripts/legibility/detectors.py::AGENT_CLASS_MARKERS, which uses it to classify the agent class, not to exclude the turn. As a result the digest bucket records the turn as a human 'User Correction'. This repeats the already-known harness-injected-prompt-as-User-Correction defect for one more injector shape. The 'mislabel as a task-failure bucket' framing is a symptom: the failure text is the harness's own prompt, not a user turn. | Add the stable closing sentence of the dry_run_unblock user prompt ('investigate and emit your structured proposal.') to HARNESS_PROMPT_MARKERS, with a lockstep test that builds the real prompt and asserts is_harness_injected_turn is True. | confirmed | filed:7442 | census-dark_factory-20261024 | census-dark_factory-20261024, census-dark_factory-20261031 |  |
| cand-20260912-22 | fk-11d0515bf84b | shared |  | scripts/legibility/digest.py::annotate_retry_loops | h11, kind:confusion | low | agent-transcripts | Retry Loops section reports 'Bash x6 (3 designed-outcome results)' for watcher-rearm.sh calls, but Designed Outcomes section displays 4 entries (all watcher-rearm.sh exit 124), and signal_counts.designed_outcome: 4. A 3 vs 4 mismatch suggests digest extraction is reading inconsistent counters or conflating separate invocation counts. Verifier: The current digest code explains the 3-vs-4 mismatch, and it is not a counter inconsistency. In scripts/legibility/digest.py, the retry-loop annotation (task 4751) groups tool_use blocks by (tool name, exact input signature). It counts only the designed outcomes whose tool_use_id belongs to that one group, so `designed_outcome_count` is a per-group join against the same neighborhood scan. The 'Designed Outcomes' section and `signal_counts.designed_outcome` count every designed result in the session through `iter_designed_outcomes`, whatever the call's input signature. Four watcher-rearm.sh exit-124 results can therefore sit beside a 'Bash x6 (3 designed-outcome results)' group when one rearm call has a slightly different command string and falls into another group. Both layers read the one classification (`_neighborhoods_to_partition`), and the docstring states annotation is never suppression. The cluster's own evidence is inconsistent too: 3, 1 and 2 designed entries across the sightings. I found no code path in the current tree that reads divergent counters. |  | refuted | refuted | census-dark_factory-20261031 | census-dark_factory-20261031 |  |
| cand-20260911-45 | fk-5cc471a143e5 | shared |  | scripts/legibility/sampling.py | h10, kind:confusion | low | agent-transcripts | Digest generator lacks filtering for sessions with n_user_turns=0 and all-zero signal_counts, submitting them to trickle-coder for pattern matching despite containing no actionable confusion evidence Verifier: The claim doesn't hold for the current tree. scripts/legibility/sampling.py already drops zero-signal sessions before anything reaches trickle-coder. In its per-stratum selection, `nonzero = [r for r in stratum_records if r.score > 0]`, and the dropped count is reported as `zero_signal_dropped`. Because score_signals adds a positive weight per user turn, a session with n_user_turns=0 and all-zero signal_counts scores 0.0 and is filtered out. The census path also skips zero-signal sessions (`skipped_zero_signal`), and test_census_ledger_boundary.py::test_row4_a_zero_signal_session_is_never_mined covers that. render_digest in digest.py only renders and scores, so it is not meant to filter. The three sightings are 377 to 378 byte digests with score 0.0. I could not tell from the tree how those digests reached the miner, but the sampler and census gates are in place. |  | refuted | refuted | census-dark_factory-20261031 | census-dark_factory-20261031 |  |
| cand-20260908-14 | fk-6866758fec5a | shared |  | scripts/legibility/reingestion.py::HARNESS_PROMPT_MARKERS | h12, kind:confusion | low | agent-transcripts | Digest extraction categorizes supervisor-injected parameters (ROTATION_ESCALATIONS, ROTATION_HOURS, DARK_FACTORY_ROOT) marked explicitly as '(injected by supervisor)' under 'User Corrections' instead of system-context or configuration section Verifier: The current tree already guards against this. `scripts/legibility/reingestion.py::HARNESS_PROMPT_MARKERS` contains 'you are running as an autonomous escalation watcher'. `is_harness_injected_turn` matches that marker, and `scripts/legibility/digest.py::iter_user_turns` drops such turns through `is_non_dialogue_carrier`, so the watcher rotation prompt can no longer reach the 'User Corrections' section. `orchestrator/src/orchestrator/harness.py` still opens its prompt with the same sentence, so the marker still matches. `scripts/tests/test_legibility_reingestion.py::test_watcher_rotation_prompt_is_excluded` and the `WATCHER_ROTATION_PROMPT` fixture in `scripts/tests/legibility_carrier_fixtures.py` cover the shape. I did not run the tests or the digest on the 2026-08-26, 2026-09-01 and 2026-09-08 sessions. The sightings predate the marker, which is consistent with a fix that has landed, but I did not check the git history or the marker's commit date to confirm that. |  | refuted | refuted | census-dark_factory-20261031 | census-dark_factory-20261031 |  |
| cand-20261019-3 | fk-7b82175d5f61 | repo |  | plans/review-all-program-tooling-prd.capability-manifest.yaml | h10, kind:confusion | low | agent-transcripts | An implementer created scripts/check_review_all_program_fixture.py and referenced it from the capability manifest but never git-added it. The file passes in the worktree, then the post-merge verify fails on main because the script is not tracked and _run_script_check cannot exec it. Verifier: The background grep that just finished only lists files that mention the script name. It doesn't change the earlier finding. scripts/check_review_all_program_fixture.py is not tracked by git and is not in the working tree. plans/review-all-program-tooling-prd.capability-manifest.yaml still lists it in a `kind: path`, `expect: present` check (around line 717), under a capability whose binding is 'built by this leaf'. That makes it a planned deliverable, not an untracked leftover. I could not confirm the claimed state of an untracked file that main lacks. The sightings also describe past worktree and merge events, which this tree does not show. |  | refuted | refuted | census-dark_factory-20261031 | census-dark_factory-20261031 |  |
| cand-20261018-1 | fk-58b02450e2a4 | orchestrator |  | orchestrator/tests/test_merge_status_boundary_b10.py | h10, kind:confusion | low | agent-transcripts | A merge-verified test asserts a hard 50 ms p99 wall-clock bound over 200 replays. Under parallel gw3 load the measured p99 was 50.78 ms (p50 2.24 ms, max 225.49 ms), so the assertion failed nondeterministically. The post-merge verifier reported it as a new regression not present on main, which blocks the task. Agents and reviewers cannot distinguish load-induced timing noise from a real regression, and the gate charges the failure to the task. Verifier: The p99 wall-clock assertion no longer exists. In orchestrator/tests/test_merge_status_boundary_b10.py, test_b10_cross_run_lookup_median_is_under_50_ms now asserts that the median of 200 replays is under 50 ms. The reported p50 was about 2.2 ms, so that bound has roughly 20x headroom. The deterministic assertion is that each read's query plan SEARCHes a partial idx_mf_* index. The module docstring says the PRD's B10 row records why these bounds replaced a p99 bound. The failure text in the cluster, 'train_id_hit: p50=... p99=...', no longer appears in any test. The test also sits in an xdist_group, and its failure message now includes the load average. Grep over *.py found no other test using that p99 wording. |  | refuted | refuted | census-dark_factory-20261031 | census-dark_factory-20261031 |  |
| cand-20261002-15 | fk-2d8c80449849 | repo |  | slug:zero-signal-orchestrated-session-digest | h10, kind:confusion | low | agent-transcripts | Task in .worktrees executes with no user interaction, generates minimal output (416 bytes), and no diagnostic signals (tool_error, self_correct, not_found, df_guard, interrupt, designed_outcome all zero), becoming invisible to downstream analysis Verifier: I only searched the digest and sampling code in scripts/legibility (sampling.py, digest.py, detectors.py). I did not open the two cited sessions or the .worktrees/5209 and .worktrees/4308 directories. The cluster describes a 416-byte transcript with no user turns and every signal count at zero. That is what a trivial or stub session produces: the digest scores it 0.0 and the sampler ranks it last. The cluster names no confusion, wrong action or misleading file, so there is nothing in the current tree to anchor it to. The evidence is an absence of signals, and that is not a defect in an existing file. |  | refuted | refuted | census-dark_factory-20261031 | census-dark_factory-20261031 |  |
| cand-20260927-19 | fk-3f9acf853ca1 | repo |  | slug:skim-rewrite-hook-cat-arity | h10, kind:confusion | low | agent-transcripts | The PreToolUse skim rewrite hook substitutes `skim` for `cat` on a source-file path without checking arity. `skim` accepts only one [FILE], so `cat family_driver.py fm.py` fails with 'unexpected argument' and exit 2. The agent never wrote `skim` and gets no hint that a rewrite happened. This is a different failure from the newline-flattening in entry-cand-20260722-6, though it comes from the same hook family. Verifier: I could not confirm this as a fact about the current tree. The skim rewrite hook is host-level PreToolUse tooling, and nothing in this tree defines it. A grep for 'skim' in the tree's docs, scripts, config and Python files finds it only in the confusion codebook, a census script, plans and evidence data. The codebook (docs/legibility/confusion-codebook.yaml) already records the same hook family in an entry about newline flattening. That entry's title also lists `cat`/`head -N` on a source-file path as a rewrite target. The multi-file arity failure comes from the host hook and the external `skim` CLI, not from any file here. |  | refuted | refuted | census-dark_factory-20261031 | census-dark_factory-20261031 |  |
| entry-cand-20260929-1 | fk-282c6e761d54 | repo |  | skills/escalation-watcher-auto/SKILL.md | h12, h11, kind:confusion | low | agent-transcripts | When the watcher wants to omit the optional status filter, it writes the key with no value instead of dropping it. The harness then rejects the whole call with InputValidationError, wasting a turn. The same malformed shape recurred for tasks 6060 and 5813, 63 turns apart, so nothing carries the lesson forward. The tool's parameter docs or the watcher prompt could state that optional filters must be omitted, not left blank. |  | unverified | filed:7074 | census-dark_factory-20261017 | census-dark_factory-20261017, census-dark_factory-20261024, census-dark_factory-20261031 |  |
| entry-cand-20260916-1 | fk-c2e12c9e2b12 | shared |  | scripts/legibility/digest.py::iter_user_turns | h1, h6, h12, kind:confusion | low | agent-transcripts | Digest extraction treats /unblock skill command and task-completion notifications as user corrections when they are neither corrections nor user-originated feedback. |  | unverified | filed:7442 | census-dark_factory-20261024 | census-dark_factory-20261024, census-dark_factory-20261031 |  |

## Filed Tasks

| finding_key | title | step | resolution | task_id | ticket | disposition | project_root | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fk-8dabfa972cc0 | Orchestrated-task sub-agent denied read-only Bash inspection in don't-ask mode, forcing blind guesses while diagnosing a verify failure | semantic | existing | 6203 |  | filed:6203 |  |  |
| fk-a6083fb7ed86 | Task worktree .task/metadata.json unavailable during post-merge verification | semantic | existing | 3160 |  | filed:3160 |  |  |
| fk-a69ec55ddc08 | Digest 'User Corrections' bucket captures task failure/block information from post-merge verification | semantic | existing | 7442 |  | filed:7442 |  | §8 step 1c: done task 7441 carries the key and the finding is still present |

## Structural

_no structural findings this run._

## Dispositions

- Re-verify at as_of_sha (D8): 1 due, 1 offered, 0 gone, 1 present, 0 inconclusive, 0 deferred by the verify cap.

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
| entry-cand-20260921-26 | fk-c1df06e83aa5 |  | 7073 | pending | live | filed:7073 |  |
| entry-cand-20260929-1 | fk-282c6e761d54 |  | 7074 | pending | live | filed:7074 |  |
| entry-cand-20261008-5 | fk-9c36e1e93a30 | tkt_0RW5FRBDGRGS9ZC4KJCHG6JN0B | 7438 | pending | live | filed:7438 |  |
| entry-cand-20260922-3 | fk-69414b6740fe | tkt_0RW5FRBTQM864TQ7K3NFHM6H5K | 7439 | pending | live | filed:7439 |  |
| entry-cand-20260911-28 | fk-e18c7f482a29 | tkt_0RW5FRC08TTDX8HFPEYMBKZKDV | 7440 | pending | live | filed:7440 |  |
| entry-cand-20260918-18 | fk-a69ec55ddc08 | tkt_0RW5FRC7ET51RZA763BVXH7ZAA | 7441 | done | done -> re-verified present (re-filed, supersedes 7441) | filed:7442 |  |
| entry-cand-20260916-1 | fk-c2e12c9e2b12 | tkt_0RW5FRCC10XP47EPHA3X9HRK50 | 7442 | pending | live | filed:7442 |  |

_no outcome changed this run._

## Cost

invoke calls: sonnet miner=40, sonnet verify=11, fable synthesis=1, haiku headroom-probe=3
