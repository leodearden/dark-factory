# confusion census 2026-10-24

Project: dark_factory

## Method

```yaml
run_id: census-dark_factory-20261024
as_of_sha: cd48063adec0e9f824e16e9577d1609b854a7e18
since: c674ec95191828f1c069835ee17ae65b770b450d
evidence:
  window:
  - '2026-09-24T00:00:00+00:00'
  - '2026-10-24T00:00:00+00:00'
  sessions_enumerated: 11714
  skipped_coded: 426
  skipped_zero_signal: 7855
  mined: 99
  ledger_rows: 386
verification:
  confirmed: 6
  weakened: 0
  refuted: 9
  unverified: 4
cost:
  miner_calls: 100
  verify_calls: 29
  synthesis_calls: 1
  probe_calls: 5
  embedding_calls: 5
  wall_clock_secs: 10880.503
inputs_consumed:
- review-dark_factory-20261009
- hotspot-survey-dark_factory-20261009
- review-all-dark_factory-20261008
extra:
  inputs_consumed_note: null
  screened:
    title: 0
    session: 4
    key: 1
    cross_instrument: 0
  attached: 0
  adjudicated: 61
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
    due: 14
    offered: 14
    gone: 13
    present: 1
    inconclusive: 0
    deferred: 0
```

## Saturation

- batches: 5
- stop reason: saturated
- operator batch cap: 50 batch(es) (not reached -- mining stopped by: saturated)
  - batch 0: dup_rate=0.85 (total=20, succeeded=20, failed=0, saturated=False)
  - batch 1: dup_rate=0.90 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 2: dup_rate=0.89 (total=20, succeeded=19, failed=1, saturated=False)
  - batch 3: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 4: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)

## Screened

- per rule: title 0, session 4, key 1, cross_instrument 0
- inputs consumed: review-dark_factory-20261009, hotspot-survey-dark_factory-20261009, review-all-dark_factory-20261008
- screened by session: "Host `skim rewrite` hook turns a multi-file `cat a.py b.py` into a single-file `skim` call, which rejects the second path" → cand-20260927-6, entry-cand-20260722-6
- screened by session: "Ambiguous design jargon ('re-base Γ2') read as a git rebase led to an escalation being closed with mistaken resolution text" → cand-20260927-4, entry-cand-20260714-2
- screened by session: "Chained slash-command skill invocations at message start are expanded into repeated invocations of the whole message" → cand-20260928-1, entry-cand-20260725-3
- screened by session: "Host skim rewrite hook turns a multi-file `cat` into `skim`, which accepts only one FILE, so the command exits 2" → cand-20260926-2, entry-cand-20260722-6
- attached by key: "Watcher supervisor-injected context mislabeled as User Corrections" (cand-20260922-5) → entry-cand-20260918-18 under fk-a69ec55ddc08

## Adjudication

- model text-embedding-3-small; attach at cosine >= 0.9, cluster at >= 0.85; at most 10 queue cluster(s) per run
- embeddings: 5 request(s), 1155 embedded, 0 cached, 0 pruned
- queue depth: 1038 pending candidate(s) in 962 cluster(s)
- processed this run: 10 cluster(s), 61 candidate(s)
- verified: "Digest not_found at turns exceeding session turn count" (cand-20260922-3) for 10 candidate(s) across 10 session(s)
- rejected: "Digest User Corrections bucket captures orchestration task-completion notifications as user input" (cand-20260918-10) for 8 candidate(s) across 8 session(s)
- rejected: "Bash tool mangles multi-line git commands by injecting spurious tokens into heredoc payloads" (cand-20260915-16) for 8 candidate(s) across 8 session(s)
- verified: "Interactive agent session with zero user turns contradicts session class" (cand-20260911-28) for 8 candidate(s) across 8 session(s)
- verified: "Digest 'User Corrections' bucket mislabels injected task context/briefing as a correction" (cand-20260918-18) for 7 candidate(s) across 7 session(s)
- verified: "Watcher supervisor-injected context mislabeled as User Corrections" (cand-20260922-5) for 6 candidate(s) across 6 session(s)
- rejected: "Bash tool loses newlines when passing multi-line Python to bash -c" (cand-20260913-1) for 5 candidate(s) across 5 session(s)
- rejected: "Digest extraction tries to navigate absent turns in zero-user-turn sessions" (cand-20260920-11) for 3 candidate(s) across 3 session(s)
- rejected: "Orchestrated task worktree provisioning omits .task directory initialization" (cand-20260918-8) for 3 candidate(s) across 3 session(s)
- verified: "Digest 'User Corrections' bucket captures skill invocations and task notifications" (cand-20260916-1) for 3 candidate(s) across 3 session(s)

## Verification

- handed all 5 novel cluster(s) to the verifier; operator verify cap: 150 (not reached).

## Origin x Manifestation Matrix

| origin \ manifested | review | ops |
| --- | --- | --- |
| implement | 13 | 21 |
| ops | 0 | 1 |

## Synthesis

- applied: 4 phase refinement(s) to 5 cluster(s) and 4 new entr(ies); 0 correction(s) applied, 0 skipped
- standing, no new evidence: 64
- invariant violations this run: StructuredOutput schema validation: root must have 'severity', /findings must be array; received {'findings': {'findings': [...]}} 1, Task dispatched with unverified premise that api/tasks.py and data/task_snapshot.py exist in the orchestrated worktree 1, contracts-machine-checked 3
- unkeyed entries re-observed: 26

### New

- fk-9c36e1e93a30 "Agent guesses the reconciliation tickets.db location and schema by trial and error": The reconciliation tickets store has a path defined in source (main_checkout_defaults.py, tools.py) and one relative-path sqlite3 example in OPERATIONS.md, but nothing an agent doing a forensic read would reach first points at it. That is a heuristic 11, SPOT, gap of the derivation kind: the fact has one home, but no reader-facing pointer derives from it, so each agent re-derives it by probing candidate paths. The errors it met ('no such table: tickets', 'no such column: id') are the same shape CLAUDE.md already warns about for the tasks store and its 0-byte decoy, which is why they cost turns rather than a single refusal. Heuristic 1, informative names, applies to the error surface: a 'no such table' answer from a wrong file names neither the file the agent should have opened nor the store's real columns. The verdict's proposal to extend the existing 'Forensic reads' section is the smallest change that gives the fact a pointer in the place the tasks-store pointer already lives.
- fk-69414b6740fe "Digest not_found at turns exceeding session turn count": The verdict's evidence places this squarely under heuristic 1, informative names: the digest renders '(turn N)' where N is an enumerate() position over all transcript records, including assistant messages and tool_results, while n_user_turns counts only user turns. The label says one thing while the value is another, and a reader who trusts the label concludes a turn is missing or a session boundary slipped. Heuristic 12, structured data instead of meaningful strings, is the second leg: the record index and the user-turn count are different quantities that reach the reader only as the prose word 'turn', so the distinction is lost at the rendering boundary. The brief's cause (session-boundary confusion, turn misattribution, off-by-one) is not supported by the verified code path; the phantom numbers are fully explained by the mislabel, and the 'does not exist' text is a not_found literal from a tool_result, not the digest reporting a missing turn. The verdict did not check whether other renderers share the label, so the fix should be sought wherever index is rendered, not only in _render_user_corrections.
- fk-e18c7f482a29 "Interactive agent session with zero user turns contradicts session class": Per the verdict, AgentClassEvidence.agent_class returns INTERACTIVE as the residual when a session has records, is not in a worktree dir, and matches no marker; it never consults user turns. So 'interactive' is a default, not a positive finding, which fails heuristic 1, informative names: the class name claims a human was present when the classifier only knows that nothing else matched. Heuristic 12, structured data instead of meaningful strings, is engaged because the enum value INTERACTIVE is doing double duty as 'observed a user' and 'no marker matched', two states that a typed classification should keep apart (the verdict's UNKNOWN or a new class). The co-occurrence with n_user_turns: 0 is therefore a product of two independent computations in detectors.py and digest.py, not evidence of a bootstrap failure; that reading in the brief's cause was not verifiable. Worktree sessions whose encoded dir matches no WORKTREE_DIR_MARKERS substring fall into the same residual, so the mislabel is wider than zero-turn sessions.
- fk-a69ec55ddc08 "Digest 'User Corrections' bucket mislabels injected task context/briefing as a correction": The verdict confirms one shape in the cluster is still unguarded: the prior-round adjudication prompt built in orchestrator/src/orchestrator/review_suggestions/prior_round.py passes reingestion.is_harness_injected_turn because no entry in HARNESS_PROMPT_MARKERS, the briefing headings, or the memory-context caveats covers its opener. This is heuristic 11, SPOT, in its failure mode: the set of harness-injected prompt shapes is authored in the orchestrator and recognised in scripts/legibility by an independently maintained marker list, so each new prompt shape silently joins the gold turns until someone adds a marker. Heuristic 10, clear invariants, informatively, redundantly, uniformly enforced, names the cost: the invariant 'a gold user turn is human-typed' is enforced by one string-match list rather than by anything the injector itself stamps, so a miss is invisible. Heuristic 12, structured data instead of meaningful strings, is the root: origin is being inferred from prose markers rather than carried as a typed field. The verdict notes the '# Context' and briefing shapes are probably already filtered by instrument versions 2-6, and that two other wordings in the cluster were not checked, so the proposed marker addition closes the confirmed shape only.
- fk-c2e12c9e2b12 "Digest 'User Corrections' bucket captures skill invocations and task notifications": Only part of this cluster survives in the current tree. Task-notification turns are already dropped: iter_user_turns calls detectors.is_non_dialogue_carrier and reingestion.has_non_human_origin filters records whose origin.kind is not 'human' (instrument version 6). What remains is slash-command wrapper turns (<command-name>/unblock</command-name>, <command-message>reflect</command-message>) and '[Request interrupted by user]' turns, which no filter in the digest path excludes. Those are human-typed, so the brief's 'not user-originated' framing is wrong for them; the verified defect is heuristic 1, informative names, at the section level: 'User Corrections' promises corrections and delivers every surviving user turn. Heuristic 6, well-defined purpose for each entity, is the shape of the fix: the user_turns bucket either filters to corrections or is renamed to what it holds, but not both at once. Heuristic 12, structured data instead of meaningful strings, applies to the command wrappers themselves, which arrive as markup inside the turn text and would need yet another string match to drop. Interrupts are additionally double-counted via the separate Interrupts bucket.

### Re-observed

- fk-282c6e761d54 "Escalation watcher emits get_task_escalations calls with an empty-valued optional parameter (`"status": ,`), producing unparseable JSON that recurs across tasks" (gained ccbc6c04-884e-4a08-b500-008bf6b1f43b, cd6542fd-0976-4e0a-97a6-b4173c56f50e, 03ac9dd5-4ac9-4f49-900f-d234d377caa4, 67c9e6e8-7331-4d22-9d0f-3718cfc7a7f1, 612a6eda-b230-4f35-8958-2e798d4417fe, 5f45aa08-079c-4e48-8c0b-d2918b8116f2, c5b48cce-f430-4e26-958c-b86bf802fc92): no change stated

### Phase Refinements

- fk-69414b6740fe: origin implement, manifested ops (The verdict locates the defect in scripts/legibility/digest.py rendering code, an implementation choice of the digest instrument, not in memory reconciliation; it surfaces when a census or operator reads the rendered digest. 'recon' in the phase vocabulary names the fused-memory reconciliation pipeline, which this code is not part of.)
- fk-e18c7f482a29: origin implement, manifested ops (The stamp is unknown/unknown; the verdict pins the cause to the residual-default branch of AgentClassEvidence.agent_class in scripts/legibility/detectors.py, an implementation of the legibility classifier, and the mislabel shows up in digest and census triage metadata read during operations.)
- fk-a69ec55ddc08: origin implement, manifested review (Origin was stamped unknown; the verdict traces it to the gap between the prompt authored in review_suggestions/prior_round.py and the marker list in scripts/legibility/reingestion.py, both implementation artefacts. The manifested stamp of review matches the evidence: the mislabelled turns are the prior-round adjudication prompts of review-phase sessions.)
- fk-c2e12c9e2b12: origin implement, manifested ops (The stamp is unknown/unknown; the verdict places the surviving defect in scripts/legibility/digest.py::iter_user_turns lacking a slash-command and interrupt filter, an implementation gap, and it manifests when the rendered 'User Corrections' section is read by the trickle coder or an operator.)

### Notes

- unverified: The brief's cause text for fk-69414b6740fe (session-boundary confusion, turn misattribution, off-by-one) and the 'bootstrap/initialization failure' clause in fk-e18c7f482a29's cause are both displaced by their verdicts, which explain the symptom from the current code without them; as new entries they cannot be corrected through this run's corrections list, so their codebook cause lines may read as endorsing a diagnosis the verdict declined to verify. (evidence: Verdict reason fields for fk-69414b6740fe ('need no session-boundary confusion or off-by-one indexing') and fk-e18c7f482a29 ('a guess I could not verify').)
- unverified: fk-69414b6740fe, fk-a69ec55ddc08 and fk-c2e12c9e2b12 all land in the digest's 'User Corrections' section and all propose a DIGEST_INSTRUMENT_VERSION bump; one task could carry the three mechanical fixes under a single bump rather than three separate re-renders. (evidence: Each finding's anchor is in scripts/legibility (digest.py::_render_user_corrections, reingestion.py::HARNESS_PROMPT_MARKERS, digest.py::iter_user_turns) and two of the three proposals name the instrument version bump explicitly.)
- unverified: fk-c2e12c9e2b12's cause ('neither corrections nor user-originated feedback') is half-refuted by its own verdict: slash-command and interrupt turns are human-typed, so the entry's cause overstates the defect at creation. (evidence: Verdict reason: 'These are human-typed, so the user-originated half of the claim is wrong.')
- unverified: fk-282c6e761d54 gained seven sessions this run with no change in shape; the standing cause says nothing carries the lesson forward, and this run's breadth is consistent with that. Whether the recurrence rate now warrants a severity review above low is a judgement the brief's session list alone cannot settle. (evidence: gained_sessions lists seven session ids; the brief supplies no verdict or new evidence detail for this key.)
- unverified: Two invariant hits in this run are single occurrences worth watching rather than filing: a StructuredOutput payload wrapped as {'findings': {'findings': [...]}} failing root schema validation, and a task dispatched on the unverified premise that api/tasks.py and data/task_snapshot.py exist in the orchestrated worktree. (evidence: invariant_counts in the brief records each at 1; contracts-machine-checked at 3.)

## Findings

- area map: briefing review/briefing.yaml; areas dashboard, escalation, fused-memory, orchestrator, repo, sampler, shared; overlay .claude/skills/review-all/project.md; rules `scripts/legibility/**` → `shared`, `scripts/**` → `repo`
- 1 re-observed finding(s) are unverified: the census does not re-verify a catalogued finding, so a sighting this run only appends to its last_seen.

| id | key | area | sub_area | anchor | tags | severity | evidence_source | statement | proposal | verdict | disposition | first_seen | last_seen | supersedes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| entry-cand-20261008-5 | fk-9c36e1e93a30 | repo |  | CLAUDE.md | h11, h1, kind:confusion | low | agent-transcripts, present-tree | No discoverable pointer gives the path and schema of the reconciliation tickets store. The agent probed several candidate paths (fused-memory/data/tickets.db, data/reconciliation/tickets.db, data/queue/tickets.db). It hit 'no such table: tickets' on one file and 'no such column: id' on another, which costs several turns per forensic read. CLAUDE.md offers scripts/tasks_db_schema.py only for the tasks store. Verifier: The tree still has no pointer to the reconciliation tickets store's path or schema. The only schema-introspection script in scripts/ is tasks_db_schema.py, which covers the tasks store. CLAUDE.md's 'Forensic reads of tasks.db' section covers only the tasks store and never mentions tickets.db. The only operator-facing mention of the tickets store is a single sqlite3 example in OPERATIONS.md at line 812, which uses the relative path data/reconciliation/tickets.db and lists no columns. The path is otherwise defined only in source files such as main_checkout_defaults.py and tools.py, where an agent doing a forensic read would not look. | Extend the 'Forensic reads of tasks.db' section to name the reconciliation tickets store's canonical path and say how to get its schema, for example by pointing at scripts/tasks_db_schema.py with --db if it accepts that store, or at a small schema-dump script. | confirmed | filed:7438 | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| entry-cand-20260922-3 | fk-69414b6740fe | shared |  | scripts/legibility/digest.py::_render_user_corrections | h1, h12, kind:confusion | low | agent-transcripts, present-tree | Digest extraction is indexing into turn numbers (67, 148) that do not exist in a 1-user-turn session, suggesting either session-boundary confusion, turn-number misattribution from a prior/different session, or off-by-one indexing in turn reconstruction Verifier: Confirmed in the current tree. scripts/legibility/digest.py renders each digest item as "(turn {item['index']})" (for example _render_user_corrections, line 773). That `index` comes from enumerate() over ALL transcript records: _signal_items enumerates the per-record hits, and _error_neighborhoods enumerates the per-record errors. Records include assistant messages and tool_results. n_user_turns is a separate count, len(iter_user_turns(records)), which includes only user turns. So "turn 67" or "turn 148" is a record position, not a user-turn number, and it can be far above n_user_turns, or appear when n_user_turns is 0. The "turn" label is a name that says one thing while the value is another. The "does not exist" text is a not_found hit literal that a tool_result contained. It is not the digest reporting a missing turn. I did not check whether other digest renderers use the same label. The reported phantom turn numbers are explained by this label and need no session-boundary confusion or off-by-one indexing. | Change the rendered '(turn N)' labels to '(record N)', or an equivalent name that says N is a transcript record index, so they cannot be confused with the n_user_turns count. | confirmed | filed:7439 | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| entry-cand-20260911-28 | fk-e18c7f482a29 | shared |  | scripts/legibility/detectors.py::AgentClassEvidence | h1, h12, kind:confusion | low | agent-transcripts, present-tree | Session classified as 'interactive' but n_user_turns: 0 indicates no user engagement occurred; combined with early termination (size_bytes: 412) and single incomplete guard signal, suggests bootstrap/initialization failure or session-classification misattribution Verifier: The pairing is still produced by the current code. In scripts/legibility/detectors.py, `AgentClassEvidence.agent_class` returns `AgentClass.INTERACTIVE` whenever the session has records, is not in a worktree dir, and matches no marker. Its docstring says 'No marker is INTERACTIVE', so interactive is the residual default, not a positive finding. It does not look at user turns. A session with no primary user turn therefore still gets `agent_class: interactive`, and `n_user_turns` is computed separately in digest.py, so a digest can carry `agent_class: interactive` with `n_user_turns: 0`. I did not check the individual sessions. Worktree-run sessions are labelled orchestrated-task only when the encoded dir contains a `WORKTREE_DIR_MARKERS` substring. A session in a worktree path that matches none of them also falls back to interactive. The 'bootstrap failure' reading in the cluster is a guess I could not verify; the verifiable fact is that the class name overstates what the classifier knows. | In AgentClassEvidence.agent_class, return AgentClass.UNKNOWN (or a new distinct class) instead of INTERACTIVE when no primary user turn was observed, so INTERACTIVE only labels sessions with a real user turn. | confirmed | filed:7440 | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| entry-cand-20260918-18 | fk-a69ec55ddc08 | shared |  | scripts/legibility/reingestion.py::HARNESS_PROMPT_MARKERS | h11, h12, h10, kind:confusion | medium | agent-transcripts, present-tree | Digest pipeline confuses task-instruction boilerplate (adjudication rubric, prior-round reference list, context) with actual user-provided feedback corrections Verifier: The gold-turn filter `scripts/legibility/digest.py::iter_user_turns` drops injected turns through `reingestion.is_harness_injected_turn`. That predicate matches only briefing headings, heading sets, the prose markers in `HARNESS_PROMPT_MARKERS`, and memory-context caveat markers. The prior-round adjudication prompt is built in `orchestrator/src/orchestrator/review_suggestions/prior_round.py` and opens 'You are adjudicating whether each CURRENT review suggestion merely re-flags...'. No marker in `scripts/legibility/` covers that text, so it still passes as a human turn and renders under '## User Corrections'. Other shapes in the cluster, such as '# Context' and the briefing, are probably already filtered by the existing rules (instrument versions 2-6). I did not check the '(turn 3) The implementation was reviewed and blocking issues were found' wording or the '/prd decompose' command-args turn. The adjudication-prompt shape is confirmed still unguarded. | Add the prior-round adjudication prompt opener ('you are adjudicating whether each current review suggestion merely re-flags') to HARNESS_PROMPT_MARKERS, and bump DIGEST_INSTRUMENT_VERSION in scripts/legibility/digest.py. | confirmed | filed:7441 | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| entry-cand-20260916-1 | fk-c2e12c9e2b12 | shared |  | scripts/legibility/digest.py::iter_user_turns | h1, h6, h12, kind:confusion | low | agent-transcripts, present-tree | Digest extraction treats /unblock skill command and task-completion notifications as user corrections when they are neither corrections nor user-originated feedback. Verifier: Only part of the cluster still holds. The task-notification part is already fixed: scripts/legibility/digest.py::iter_user_turns now calls detectors.is_non_dialogue_carrier, and reingestion.has_non_human_origin drops records whose origin.kind is not 'human'. That filter is part of instrument version 6. The part that remains is slash-command wrapper turns (<command-name>/unblock</command-name>, <command-message>reflect</command-message>) and '[Request interrupted by user]' turns. Neither digest.py nor detectors.py::user_turn_text excludes them, and a grep of scripts/legibility finds no command-name filter in the digest path. So they still reach the 'User Corrections' section, labeled as corrections, and interrupts are also counted in the separate Interrupts bucket. These are human-typed, so the user-originated half of the claim is wrong. What remains is that the section label ('User Corrections', from digest.py's section-title table) promises more than the user_turns bucket delivers. The cost is added noise for the trickle coder. | In iter_user_turns, skip user turns whose text is only a slash-command wrapper or an '[Request interrupted by user' marker. Alternatively, rename the 'user_corrections' section label to match what it actually holds, and bump DIGEST_INSTRUMENT_VERSION. | confirmed | filed:7442 | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| cand-20261010-2 | fk-11eb08c8e0c7 | repo |  | slug:read-tool-hallucinated-line-range | comments, kind:confusion | low | agent-transcripts | The agent supplied an invented 'line_range' argument alongside the real offset and limit parameters to Read. The tool schema rejected the call as an unexpected parameter, which cost a turn. The redundant range was most likely the agent's own guess at a range syntax. Nothing in the call path steers the agent to the real offset/limit contract before it guesses. Verifier: The failed call is real, but the cause is the agent's own invented argument and the Read tool's schema. The Read tool is built into the harness, and this tree does not contain it. The tree uses 'line_range' as an identifier in unrelated code, such as workflow.py, merge_lane/worker.py and tests. I found no file in the tree that teaches Read a 'line_range' parameter or that could guard against this. The confusion is not an observable property of the current tree's state, and no existing file produces it. |  | refuted | refuted | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| cand-20260928-3 | fk-cc02c45f2d77 | repo |  | slug:playwright-mcp-allowed-roots-vs-scratchpad | h11, kind:confusion | low | agent-transcripts | mcp__playwright__browser_evaluate rejects a filename under the session scratchpad (/tmp/claude-1000/<project>/<session>/scratchpad) because only the project root and .playwright-mcp are allowed roots. The agent used the scratchpad as its scratch area, so the call failed and a turn was wasted. Nothing tells the agent that the browser tool's allowed roots differ from the Bash scratchpad. Verifier: I can't confirm a cause in this tree. The quoted error is a Playwright MCP / Claude Code harness restriction on file-output roots, and the tree doesn't contain that tooling. I searched the *.md, *.yaml and *.json files for 'playwright-mcp' or 'scratchpad'. I saw no hits that define or teach a rule about those roots. The search output was capped at 20 files, and I opened only the codebook, where 'playwright' has no entry. The tree doesn't tell agents to use the scratchpad for browser output. |  | refuted | refuted | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| cand-20260928-4 | fk-8e25d0bdf19a | fused-memory |  | fused-memory/src/fused_memory/server/tools.py::add_memory | h11, kind:confusion | low | agent-transcripts | The entities-declaration check rejects a declaration whose referent appears in the content only in a non-canonical form such as 'task-173-...' or 'gate 184'. The workaround of omitting entities, which always succeeds, then gets recorded as durable guidance ('drop entities entirely'). Those memories defeat the structured-entity purpose of the feature, which is to stop entities being inferred from free text. The gate's accepted surface forms are not documented where authors write, so each author rediscovers the rejection by failure. Verifier: The claim is not borne out by the current tree. The `add_memory` docstring in fused-memory/src/fused_memory/server/tools.py does document the accepted surface form. It says a conflicting declaration is rejected with DeclaredReferentConflictRejected and to "fix the id, or write `task NNN` in the content". It also says absence is never rejected and that `id` is digits, never a label. `_LOCAL_MENTION_PATTERN` in fused-memory/src/fused_memory/utils/canonical_labels.py accepts `task` followed by `#`, `:`, whitespace, `-` or `/` and then digits. That means 'task-173-...' is recognised unless it is glued to a preceding word character, colon or hyphen. 'gate 184' is not a task mention, and the docstring directs authors to write `task NNN`. The remaining claim, that agents record 'drop entities' as durable guidance, concerns memory contents. I could not check that from the code, and there is one sighting. |  | refuted | refuted | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| cand-20261012-1 | fk-4fa6a20cb517 | repo |  | skills/merge-queue/SKILL.md | h11, kind:confusion | low | agent-transcripts | A tool-use guard rejects a short foreground sleep chained with echo and directs the agent to Monitor with an until-loop or run_in_background. The agent used a plain sleep to wait for a merge request to finalize, and the rule is not in the agent's guidance, so it is learned by failure, costing a turn. The same session then polled merge_status x3 and get_merge_queue x5, which suggests no blocking wait primitive for merge finalization is advertised. Verifier: The guard itself is Claude Code harness tooling, which this tree does not contain. The tree already teaches the rule that the cluster says is missing. skills/merge-queue/SKILL.md has a 'Waiting between polls' section (anchor waiting-between-polls). It names the 2026-09-27 and 2026-10-07 harness blocks of a bare `sleep 60`. It prescribes a bounded `timeout ... bash -c 'while ...; do sleep 5; done'` loop that wakes early on git ancestry, and it explains why Monitor is not the right tool. It also covers 15s to 60s backoff polling with `merge_status`, and `merge_request(wait_secs=100)` is the blocking wait it advertises. The polling of `merge_status` x3 and `get_merge_queue` x5 is the documented design, not a missing primitive. I did not check whether the session that hit this loaded that skill, but that is not a defect in any existing file here. |  | refuted | refuted | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| cand-20260918-10 | fk-1f0bf55fa1cb | shared |  | scripts/legibility/digest.py::iter_user_turns | h11, kind:confusion | low | agent-transcripts | Digest extraction conflates task-notification event objects (background agent completions) with actual user corrections, treating completion summary reports as user feedback Verifier: Not observable in the current tree. scripts/legibility/digest.py::iter_user_turns is the single source for the 'User Corrections' bucket and the user-turn score. It drops any record that detectors.py::is_non_dialogue_carrier flags, which is true when reingestion.py::has_non_human_origin sees a non-human origin stamp, such as a background task-notification, or when is_reingested_content matches the text. scripts/tests/test_legibility_digest.py has a regression test built from a task-notification fixture (TASK_NOTIFICATION_TEXT). The sightings are dated 2026-08-15 to 2026-09-18 and the codebook already holds several entries for this pattern. I did not run the tests or replay a transcript. I also did not check whether an unstamped notification (text only, no origin) is caught by is_reingested_content. |  | refuted | refuted | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| cand-20260915-16 | fk-57678ad5f308 | repo |  | docs/legibility/confusion-codebook.yaml | h11, kind:confusion | low | agent-transcripts | Bash tool escaping or command construction corrupts compound git commands containing heredoc commit messages, injecting spurious tokens ('skim' before 'git status') that break shell syntax Verifier: The codebook already tracks this confusion at docs/legibility/confusion-codebook.yaml line 4467 as an existing entry: a Bash command whose first line has a skim-rewrite target reaches the shell with its newlines collapsed, which breaks heredocs. Several sightings are already attached to it, so the cluster is not novel. The cause is a command-rewriting layer in the Bash tool or harness, not a file in this tree. The only other 'skim' hits I opened were prose and census data, not code that rewrites commands, and I read none of them for a rewrite hook. No existing file here produces, teaches or fails to guard against it, so there is nothing to remediate. |  | refuted | refuted | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| cand-20260913-1 | fk-a7c2d2626b80 | repo |  | slug:bash-tool-newline-flattening | h10, kind:confusion | low | agent-transcripts | JSON marshaling of inline Python code loses or collapses escaped newlines during bash -c argument passing, causing IndentationError in Python parser Verifier: The newline-flattening happens in the Bash tool's marshaling of command arguments. That tooling is outside this tree. The tree's own skills use only single-line `python -c` one-liners (skills/review-all/SKILL.md, skills/factory-init/references/recon-registration.md, skills/hotspot-survey/references/orchestration.md) or a quoted multi-line form in skills/escalation-watcher/SKILL.md. No file in the tree teaches or guards against the multi-line `python3 -c` pattern, and I could not check the session transcripts, so I can't confirm the failure is reproducible from this tree's current state. |  | refuted | refuted | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| cand-20260920-11 | fk-17d88123ba94 | repo |  | scripts/tests/test_legibility_digest.py | h1, kind:confusion | low | agent-transcripts | Digest signal scraper attempts to extract content from turn indices (16, 29, 39) that don't exist, potentially due to mismatched assumptions about session structure or navigating a truncated/empty transcript Verifier: The claim does not hold up in the code I read. The digest does not navigate to turn indices or extract content from them. Its `iter_not_found` detector scans tool-result text for literals such as 'does not exist' and 'no such file or directory'. It ignores assistant text and Write-tool inputs; `scripts/tests/test_legibility_digest.py` pins this. The '(turn N)' label is only the position of the record that contained the literal, so the turn exists and the literal is a real tool error. A session with n_user_turns=0 can still have high record indices, because n_user_turns counts only user records, not tool or assistant records. A small score such as 3.0 from three not_found hits is therefore consistent with the scoring. I could not locate the digest source script itself, only its tests, so this rests on the tests and not on reading the script. The cluster misreads a 'file missing' tool-error signal as an extraction attempt on absent turns. |  | refuted | refuted | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| cand-20260918-8 | fk-7ac0cfcdad69 | orchestrator |  | orchestrator/src/orchestrator/artifacts.py::TaskArtifacts | h10, kind:confusion | low | agent-transcripts | Verify-cold provisioning or task-dispatch initialization fails to create or populate .task/metadata.json and .task/iterations.jsonl in the worktree before task execution begins. Agent expects these files to exist and fails silently when they don't. Verifier: The current tree does initialize `.task/`. `TaskWorkflow._setup` calls `artifacts.init()` (workflow.py ~2739). That method creates `.task/`, `reviews/` and `verdicts/`, and writes `metadata.json` (artifacts.py::TaskArtifacts.init). `iterations.jsonl` is not created at init by design. It is appended lazily on first iteration, and reads go through `_read_path`, which tolerates a missing file. So its absence in a fresh worktree is expected, not a provisioning omission. The sightings come from the steward or unblock flows and from sessions where setup had not run. For example, the `.task/` listing held only `plan.json`, `verify` and the claude-config directories. The tree does not show verify-cold provisioning or dispatch skipping `.task` init. The claim that provisioning omits `.task` initialization is therefore not an observable fact about the current tree. |  | refuted | refuted | census-dark_factory-20261024 | census-dark_factory-20261024 |  |
| entry-cand-20260929-1 | fk-282c6e761d54 | repo |  | skills/escalation-watcher-auto/SKILL.md | h12, h11, kind:confusion | low | agent-transcripts | When the watcher wants to omit the optional status filter, it writes the key with no value instead of dropping it. The harness then rejects the whole call with InputValidationError, wasting a turn. The same malformed shape recurred for tasks 6060 and 5813, 63 turns apart, so nothing carries the lesson forward. The tool's parameter docs or the watcher prompt could state that optional filters must be omitted, not left blank. |  | unverified | filed:7074 | census-dark_factory-20261017 | census-dark_factory-20261017, census-dark_factory-20261024 |  |

## Filed Tasks

| finding_key | title | step | resolution | task_id | ticket | disposition | project_root | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fk-9c36e1e93a30 | Agent guesses the reconciliation tickets.db location and schema by trial and error | curator | created | 7438 | tkt_0RW5FRBDGRGS9ZC4KJCHG6JN0B | filed:7438 |  |  |
| fk-69414b6740fe | Digest not_found at turns exceeding session turn count | curator | created | 7439 | tkt_0RW5FRBTQM864TQ7K3NFHM6H5K | filed:7439 |  |  |
| fk-e18c7f482a29 | Interactive agent session with zero user turns contradicts session class | curator | created | 7440 | tkt_0RW5FRC08TTDX8HFPEYMBKZKDV | filed:7440 |  |  |
| fk-a69ec55ddc08 | Digest 'User Corrections' bucket mislabels injected task context/briefing as a correction | curator | created | 7441 | tkt_0RW5FRC7ET51RZA763BVXH7ZAA | filed:7441 |  |  |
| fk-a69ec55ddc08 | Watcher supervisor-injected context mislabeled as User Corrections | key | created | 7441 | tkt_0RW5FRC7ET51RZA763BVXH7ZAA | filed:7441 |  | same finding key as "Digest 'User Corrections' bucket mislabels injected task context/briefing as a correction", routed earlier this run |
| fk-c2e12c9e2b12 | Digest 'User Corrections' bucket captures skill invocations and task notifications | curator | created | 7442 | tkt_0RW5FRCC10XP47EPHA3X9HRK50 | filed:7442 |  |  |

- Backfill linked 21 historical census task(s): 3959 -> entry-cand-20260809-16, 4273 -> entry-cand-20260815-17, 4274 -> entry-cand-20260811-6, 4275 -> entry-cand-20260812-20, 4578 -> entry-cand-20260817-1, 4751 -> entry-cand-20260824-28, 4964 -> entry-cand-20260827-23, 5120 -> entry-cand-20260831-16, 5332 -> entry-cand-20260907-21, 5520 -> entry-cand-20260912-25, 5683 -> entry-cand-20260918-19, 5684 -> entry-cand-20260915-20, 5685 -> entry-cand-20260915-21, 5686 -> entry-cand-20260919-1, 5687 -> entry-cand-20260919-2, 5688 -> entry-cand-20260919-3, 5689 -> entry-cand-20260916-17, 5690 -> entry-cand-20260916-18, 5876 -> entry-cand-20260921-25, 7073 -> entry-cand-20260921-26, 7074 -> entry-cand-20260929-1
- Backfill matched no entry for 4 census task(s): 5330 [legibility census] Ad-hoc forensic sqlite queries against tasks.db assume unverified column/type shapes; 5331 [legibility census] Grep tool rejects lookahead/lookbehind regex patterns without exposing a PCRE2 option; 5518 [legibility census] Bash tool's source-parsing pre-check rejects a benign read-only command against a large file with a false 'malicious input' verdict; 5519 [legibility census] Full-suite pytest verify run killed by SIGKILL (exit 137) well under its own timeout budget

## Structural

_no structural findings this run._

## Dispositions

- Re-verify at as_of_sha (D8): 14 due, 14 offered, 13 gone, 1 present, 0 inconclusive, 0 deferred by the verify cap.

| entry_id | finding_key | ticket | task_id | task_status | outcome | disposition | project_root |
| --- | --- | --- | --- | --- | --- | --- | --- |
| entry-cand-20260809-16 |  |  | 3959 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260815-17 |  |  | 4273 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260811-6 |  |  | 4274 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260812-20 |  |  | 4275 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260817-1 |  |  | 4578 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260824-28 |  |  | 4751 | done | done -> re-verified present but structural (contract §6: not re-filed) | open |  |
| entry-cand-20260827-23 |  |  | 4964 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260831-16 |  |  | 5120 | cancelled | abandoned (eligible to re-file; not sighted this run) | open |  |
| entry-cand-20260907-21 |  |  | 5332 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260912-25 |  |  | 5520 | cancelled | abandoned (eligible to re-file; not sighted this run) | open |  |
| entry-cand-20260918-19 |  |  | 5683 | done | done -> entry already fixed | open |  |
| entry-cand-20260915-20 |  |  | 5684 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260915-21 |  |  | 5685 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260919-1 |  |  | 5686 | done | done -> entry already fixed | open |  |
| entry-cand-20260919-2 |  |  | 5687 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260919-3 |  |  | 5688 | done | done -> entry already fixed | open |  |
| entry-cand-20260916-17 |  |  | 5689 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260916-18 |  |  | 5690 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260921-25 |  |  | 5876 | done | done -> re-verified gone (fixed at cd48063adec0e9f824e16e9577d1609b854a7e18) | fixed:cd48063adec0e9f824e16e9577d1609b854a7e18 |  |
| entry-cand-20260921-26 | fk-c1df06e83aa5 |  | 7073 | pending | live | filed:7073 |  |
| entry-cand-20260929-1 | fk-282c6e761d54 |  | 7074 | pending | live | filed:7074 |  |

- entry-cand-20260809-16: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18
- entry-cand-20260815-17: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18
- entry-cand-20260811-6: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18
- entry-cand-20260812-20: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18
- entry-cand-20260817-1: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18
- entry-cand-20260827-23: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18
- entry-cand-20260907-21: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18
- entry-cand-20260915-20: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18
- entry-cand-20260915-21: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18
- entry-cand-20260919-2: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18
- entry-cand-20260916-17: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18
- entry-cand-20260916-18: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18
- entry-cand-20260921-25: open -> fixed, fixed at cd48063adec0e9f824e16e9577d1609b854a7e18

## Cost

invoke calls: sonnet miner=100, sonnet verify=29, fable synthesis=1, haiku headroom-probe=5
