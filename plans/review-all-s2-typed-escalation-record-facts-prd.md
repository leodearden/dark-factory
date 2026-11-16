# S2 — Typed escalation record facts (PRD)

**Status:** draft, authored 2026-11-10 by the S2 author seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S2, wave 3, size M, risk medium); critique
folded 2026-11-11; sitting rulings of 2026-11-12 (residual answers 2026-11-13) folded 2026-11-13
(§Rulings); fold-verified 2026-11-13; aligned with the sibling drafts and the lead's later rulings
2026-11-13. B+H: contracts C1–C7 and a two-way boundary-test sketch. Seventeen leaves filed at decompose
plus two the lead files later (§Not filed); the integration gate ι mints the escalation-vocabulary
invariant (R22 item 2). Implements ruling **R9** without relitigating it. No question is open.

**Code anchors** re-verified at main `e59840cdaf` (2026-11-13). The draft's anchors were verified at
`1f240e6da2` (2026-11-10) and `ba17934861` (2026-11-11); every path this PRD names was re-checked with
`git log ba17934861..main`, and every count a done-criterion or a delivered check rests on is labelled
`measured at e59840cdaf` (or says which earlier sha it still holds at); main then moved to `d538279068`;
tasks 6629 and 5370 changed `dashboard/src/dashboard/data/memory_evals.py` and fused-memory's
`reconciliation/harness.py` in between, and every symbol and count this plan rests on in them was re-read
there unchanged (the five fingerprint reads, the status constants, 54 literal sites); the sidecar was
re-linted at `ce6673c7e1`. The fold-verifier (2026-11-13) re-read every symbol the rulings touch at
`fceb951e7e` (the fingerprint writers and readers, `authority.py`'s identity sets, `classify.py`'s human set,
`mcp_lifecycle.py::McpLifecycle.mcp_config_json`, `agents/invoke.py`, the client configurations on this host)
and labels what it re-measured `measured at fceb951e7e`.
The live-store figures were read
on 2026-11-10, read-only, from the eleven `~/src/*/data/escalations` queue directories on this host;
pruning moves them daily. Main moves fast: cite by symbol; re-locate at implementation time.

Edge notation everywhere: "X → Y" means X depends on Y (Y lands first). "Lands before" is spelled out in
words, never with an arrow.

## Goal

The escalation record says what it is in typed fields, and every reader routes on those fields. The
status, severity and resolve-action vocabularies are StrEnums in `escalation/models.py`, parsed when a
record is constructed. A cascade member names its parent L2 in a persisted field, so `resolved_by` goes
back to naming who acted. The resolve callback receives a frozen event. `ACTION_EFFECTS` says what each
action does to the record as well as to the task, so `resolve_issue` branches on the table. One function
reads the caller's identity headers for every ladder tool, every client sends an identity, and a caller
that sends none can file and read but holds no authority. One reader answers "which failure is this".
And no module outside `escalation/` compares an escalation status, action or category to a string
literal; a guard enforces it.

What a user observes when this lands:

- An archived cascade member in `data/escalations/archive/` reads
  `"resolved_by": "interactive:coordinator", "cascade_parent_id": "esc-4120-3"`. No new record
  carries `l2-cascade:`; the 1,495 dark-factory records that do (measured 2026-11-10) carry the field too
  after the startup sweep. Every resolver-tier reader — the Escalation Analytics resolver-tier chart, the
  queue's late-resolution capture, the steward's lost-race check and the shadow-ruling agreement report —
  reads the field first, so a cascade member is in the `cascade` tier. That precedence is set by β1 in
  task 8562's `classify.resolver_tier(record)`: `cascade` is answered before 8562's `human-ruled`.
- `mcp__escalation__resolve_issue(..., action='close_only', duplicate_of='esc-512-2')` persists
  `duplicate_of` on the dismissed record; the sitting tooling reads it from the record (S25 ξ, task 6770,
  released onto δ).
- `mcp__escalation__escalate_blocker(severity='critical', ...)` from an agent stores
  `severity: "blocking", original_severity: "critical"` and leaves the summary as written.
- A record carries one declared failure identity, `failure_signature`. Every reader that today matches on
  the stored `dedupe_fingerprint` matches on it: the Memory Evals panel's join, fused-memory's
  escalation-archive fingerprint set, the reconciliation harness's two re-file checks, the queue's
  open-record match and the fingerprint fold. Once γ3 lands (after S1's closing task) no filer writes
  `dedupe_fingerprint` and the field is gone; a legacy record that carries only `dedupe_fingerprint` is
  read as declaring that value, so it is found exactly as today.
- Every interactive session, watcher and spawned agent sends `X-Escalation-Identity`: the project's
  `.mcp.json` sends `${ESCALATION_IDENTITY:-interactive}`, the L2 watcher sends `escalation-watcher`
  where its launch lane passes that variable, and each agent the orchestrator spawns sends its
  dispatched agent id (`claude-task-<id>-<role>`) on its orchestrator-built entry and, through
  `ESCALATION_IDENTITY` in its environment, on the worktree `.mcp.json` entry it may also load. A header
  whose value is empty counts as no header. Once the denial lands (task ο, which the lead files after the deploy and
  its check), `mcp__escalation__resolve_issue`, `promote_to_l2`, `stamp_triage`, `amend_escalation`,
  `claim_policy_class` or `declare_pin` called on a connection with no identity header is refused
  `identity_required`, at every level, with a message that names the missing header, the tool, the
  refused action and level, the config file and the fix. `escalate_info`, `escalate_blocker` and the
  read tools stay open to such a connection, and its filings carry `filed_via: "unidentified"`. This
  ends the 2026-08-20 human-channel guarantee (esc-2087-2).
- `mcp__escalation__promote_to_l2` called on a connection that sends a levels header without an
  identity (e.g. `X-Escalation-Levels: 0,1` alone) is refused `level_forbidden`. An identity in
  `authority.py::PROMOTE_ALLOWED` (the supervised auto-watcher, the executor, the coordinator and the
  human identities) promotes whatever its level ceiling.
- `mcp__escalation__resolve_issue(..., resolved_by='claude-team-7174-carry')` from an `interactive`
  session stores `resolved_by: "interactive"` (the identity) and keeps the argument as
  `resolver_detail: "claude-team-7174-carry"`, so the analytics' human tier and the name both survive
  the deploy.
- The dashboard's Escalation Analytics, Escalations, Performance, Memory Evals and corpus readers agree
  on what is open and what is terminal, and name an unknown status the same way. No queue listing raises
  on a record it cannot parse: the record is skipped, logged and counted.
- `uv run --project shared pytest scripts/tests/test_escalation_vocabulary_literals.py` fails on a seeded
  module that writes `esc.status == 'pending'` in `orchestrator/` and passes on the tree; the invariant
  `escalation-vocabulary-imported` is registered.

## Background

The run `review-all-dark_factory-20261008` found that the escalation record carries its structural facts
as meaningful strings (heuristic 12): the cascade parent inside `resolved_by`, the filing site inside
`summary`, the failure identity derived from `summary`, the severity downgrade as a summary suffix. Its
closed vocabularies are bare `str`. Consumers in four packages re-spell them and each picks its own answer
for a value it does not know. Rulings this PRD implements: R9 (the stream, its invariant, its seam
ownership), R22 item 2 (the invariant is S2's), R23 (none of S2's 14 keys is weakened), R24 (launch).
Program conventions: the filing stamps, the freshness clause and the held-leaf form in the program doc's
Shared conventions; R54 (one per-file order, edges onto wave-2 task ids), R55 (new `merge_lane/worker.py`
tasks park behind 8528), R58 (dated checks are filed only after their prerequisite lands).

**Re-measured at `1f240e6da2`, re-checked at `e59840cdaf`.** Main moved by thousands of lines in every
anchor file since `6551b75aba`. What changed for this stream (the ruling fold's additions are marked
"since the draft"):

- Task **5257** (`a08453dba9`, done) put the category vocabulary in
  `escalation/src/escalation/categories.py::EscalationCategory` (104 members), enforced by
  `require_registered_category` at `EscalationQueue.submit`, `submit_resolved`, `has_open_l1` and the
  three filing tools, and never checked on rehydration. `server.py::CATEGORIES` no longer exists.
- Task **5516**'s amendment (`343e658060`) gave the cascade attribution one writer and one reader,
  `escalation/models.py::cascade_resolved_by` and `cascade_parent_id(resolved_by)`. The fact still travels
  inside `resolved_by`: `EscalationQueue.resolve` writes it on archived members, `EscalationQueue.park`
  writes it on pending members in memory only ("In-memory cascade signal only; no disk write"),
  `classify.py::_CASCADE_PREFIX` is a second spelling read by `classify_resolver_tier`, and
  `harness.py::Harness._resolve_escalation_action` falls through to a legacy map (`dismissed` →
  `close_only`, otherwise `resume`) when the parent lookup in `Harness._cascade_parent` returns nothing.
  Task **8777** (pending, low) would delete `_CASCADE_PREFIX`; β1 subsumes it. Task **8784** (pending,
  low) would make `_resolve_escalation_action`'s `parent` required; β2 deletes the function.
- Task **8422** (done) and the decision-stream PRD (`plans/decision-stream-prd.md` §3 decision 2) turned
  the task-2630 L2 auto-close classes into rows of `authority.py::POLICY_TABLE` with presence-only evidence
  regexes. `l2_auto_close_class` (the watcher door) and `authority.py::executor_refusal`, reached through
  the executor gate `policy_gate.py::identity_resolve_refusal`, both read the same rows; the
  `stale_task_scoped` row starts `ClassSwitch.SHADOW` and opens to the executor when the decision-stream
  gate flips it live.
- Task **6578** added `server.py::_park_on_info_refusal`, a per-action precondition written inline in
  `resolve_issue`.
- The wave-2 S3 plan (filed, tasks 9152–9210) reshapes the files this stream edits: κ 9173 puts
  `LEVELS_HEADER`, `IDENTITY_HEADER`, `WATCHER_AUTO_IDENTITY`, `CallerHeaders` and `read_caller_headers`
  in `models.py`, and its step 2 makes the five header-reading tools call `read_caller_headers` (its C6,
  "S2's `parse_caller_identity` absorbs `read_caller_headers` later"); β 9157 gives the queue
  `_mutate_pending`/`_mutate_terminal` and `set_resolve_callback(cb, *, deliver_on)` (its C4, "D7 keeps
  `l2-cascade:` for S2"); γ 9162 moves the scan helpers into `escalation/scan.py`; ι 9181 classifies
  `resolve_issue`'s steps into `ladder_gates.py` `RESOLVE_GATES` (its C8, "S2's
  `ACTION_EFFECTS.record_disposition` will branch inside the effect step"); ζ 9200 moves the ladder tools
  into `filing_tools.py`, `resolve_tools.py` and `read_tools.py`. In the store, 9181 → 9177 → 4567.
- Task **8562** (instruments-honesty δ, pending, no open prerequisite) adds `Escalation.ruled_by_human`,
  `classify.resolver_tier(record)` (answering `human-ruled` first) and a `resolution_class` requirement on
  level-2 closes, in `models.py`, `queue.py`, `classify.py`, `authority.py`, `server.py` and
  `escalation_analytics.py`.
- The wave-3 S1 draft (the one escalation filing entry point, `escalation/filing.py`, being folded
  concurrently) moves `_chokepoint_or_submit`'s gate body into `filing.py` (its α, contracts C1–C3: C1 is all of
  `filing.py` — `EscalationDraft`, `FilingPolicy`, `EscalationFiler`; C2 is the door's ordered pipeline,
  whose step 2 writes the severity downgrade; C3 is the `Sentinel` registry and `has_open` in
  `models.py`/`queue.py`), moves
  `promote_to_l2`'s create step onto the filer (its β), rewrites the 17 workflow filings through
  `TaskWorkflow._file_escalation` (its δ) and migrates the fused-memory escalators that build their own
  queue, and the ticket janitor, and the recon filers (its θ2, ι2). Its integration gate ω mints `escalation-filing-one-door`.
  S1's draft `EscalationDraft` carries `dedupe_fingerprint`, and its `OpenAtLevel(same_fingerprint=True)`
  and `Fingerprint` policies match and fold on it (S1 C1).
- Since the draft: task **6390** is done (`bbaa5f91e4`): `Escalation.review_due: ReviewDueState | None`
  is on the record, so α types the vocabulary around it rather than ahead of it. Task **8475** is done
  (`19d62b7687`, `e6178c8fe3`): `authority.py` holds reversibility in one place
  (`REVERSIBLE_POLICY_ACTIONS`, `PolicyRow.reversible`) and `PolicyAction.ABANDON` is gone. Task **4567**
  is done (`b255f96b3e`). Task **6145** moved the needs-human predicate to
  `escalation/src/escalation/pins.py::needs_human`. Task **8926** added a filer,
  `escalation.file_or_refresh`, which is S1's to migrate. Task **6753** (S25) is done. Task **9493**
  (fused-memory's double stop signal) is done; this plan does not depend on it.
- Since the draft: the package-layering plan (S19, being folded concurrently) creates
  `shared/src/shared/escalation_layout.py` in its first small leaf θ0 (its contract C9: the archive
  layout constants and the standard-library names of the terminal escalation statuses, `RESOLVED`,
  `DISMISSED`, `TERMINAL_STATUS_NAMES`). The cockpit's decision registry must stay standard-library
  only, so it reads the names there; α builds `EscalationStatus` from them (lead ruling S19/Q-C8).
- Since the draft: task **9490** (pending, high) makes the task store refuse a dependency edge that would
  close a cycle (`DependencyCycle`, the cycle path named), for both `add_dependency` and
  `update_task(dependencies=…)`. Once it lands, that refusal is the guard against the cycles this plan's
  ordering avoids; a re-check after wiring stays an early check only.

Counting rules: a "literal comparison site" is an `ast.Compare` whose non-constant operand is
`X.status` (in a module importing `escalation`), any `status` operand against `'resolved'`/`'dismissed'`,
a dict-form `status` read (`X['status']`, `X.get('status')`) compared to any escalation status in a
module that names the escalation store (`data/escalations`, `escalation_queue_dir` or an `'escalations'`
path segment), `X.category`/`['category']`/a name ending `category` against an `EscalationCategory` value,
or `X.resolution_action`/a name `action`/`resolution_action` against a resolve action; plus a module- or
class-level tuple/set literal of two or more status values containing a terminal one, or of two or more
resolve actions. Scope `dashboard/src`, `orchestrator/src`, `fused-memory/src`, `scripts`, non-test. The
probe is in the fold seat's scratch (`scan_rules_fold.py`); ι turns it into the guard. The store-anchored
dict shape is anchored on the module, not on `import escalation`: anchored on the import, the same shape
flags 8 task-dict reads (`scheduler.py` 6, `workflow.py` 1, recon `task_knowledge_sync.py` 1, measured at
`ba17934861`), which are task statuses and S10's guard's to own.

- **54 literal comparison sites in 25 modules** (measured at `e59840cdaf` with the fold probe's rule;
  55 at `ba17934861`: `deterministic_runner.py`'s `e.category == 'milestone_gate'` went in `3a607efadd`,
  the gate-fields builder that files `milestone_gate` or `review_due`): orchestrator 32 (harness 13, workflow 4, steward 4, escalation_router 3,
  digest 2, deterministic_runner 1, l0_promotion, provenance_conflict, resume_resolution,
  sentinel_alarm_wrappers, offline_lane_episodes 1 each), dashboard 11 (escalation_analytics 6,
  performance 2, escalations 1, escalation_corpus 1, memory_evals 1), scripts 8
  (`sitting/payloads.py::RESOLVE_ACTIONS`, `sitting/apply.py`, `sitting/gates.py`, `sitting/inventory.py`,
  `sitting/fleet_state.py`, `member-chain-sweep.py` ×2 including `sweep_queue`'s
  `record.get('status') == 'pending'`, `sweep_toolcall_markup.py`), fused-memory 3
  (`reconciliation/backlog_policy.py` ×2 including
  `BacklogPolicy._close_pending_halt_records`' `record.get('status') != 'pending'`,
  `reconciliation/escalation_archive.py`). Seven of them are module-level re-definitions of the terminal
  set (`TERMINAL_STATUSES`, `_CLOSED_ESCALATION_STATUSES`, `CLOSED_ESCALATION_STATES`,
  `_TERMINAL_STATUSES`, `_STATUS_KEYS`). Every one of the 17 `X.status == 'pending'` sites was read: each
  is an escalation record.
- **Live stores (2026-11-10):** 12,359 records in eleven queue directories. Every `status` is one of
  pending/resolved/dismissed, every `severity` one of info/blocking/critical/urgent, every
  `resolution_action` one of the five actions or null, every `category` registered; 1,111 records carry
  3,376 evidence entries, all with `observation`, `measured_at` and `ref`, 25 with extra numeric keys.
  dark-factory alone: 4,093 records, 1,495 with `resolved_by` starting `l2-cascade:`, 244 L2s with
  members. One record carries the string `l2-cascade:` only in its `detail` (`esc-8422-3`, no cascade
  parent), so a grep over the store is not a measure of cascade members.
- **Resolver-tier readers** (measured at `ba17934861`, unchanged at `e59840cdaf`):
  `classify.py::classify_resolver_tier(resolved_by)` has 8 non-test call sites in 4 modules, read on by
  the steward and the workflow — `queue.py::_is_late_resolution_worth_capturing` ×2 and
  `EscalationQueue._capture_late_resolution`'s `via_cascade`; `classify.py::closed_by_non_sweep_resolver`
  (read by `orchestrator/steward.py` and `orchestrator/workflow.py`) and
  `classify.py::default_resolution_class_for_resolver` (read by `EscalationQueue.resolve`);
  `dashboard/src/dashboard/data/escalation_analytics.py::_lifespan_block` and `::_workflow_block`;
  `shadow_ruling.py::agreement_report`. Each reads the cascade tier from the `l2-cascade:` prefix.
- **Resolve-callback test surface** (measured at `ba17934861`, unchanged in `.py` files at `e59840cdaf`;
  two new data fixtures mention the names and need no edit): 26 test files name
  `_on_escalation_resolved` (174 lines); 12 of them call it directly with a record (102 lines). 16 test
  sites in 10 orchestrator files register `harness._on_escalation_resolved` (or `h.…`) on a real queue,
  and 11 test files register a record-reading callable (`resolved.append`, a lambda reading `e.id`). 19
  test files reach the cascade helpers or the `l2-cascade:` string. The union is 45 test files, 27 of them
  under `orchestrator/tests/`.
- **Action tables:** `ACTION_EFFECTS` (5 rows, wildcarded on level and category), `server.py::RESOLVE_ACTIONS`
  (5, read only by the `illegal_transition` message), `server.py::_DISMISS_ACTIONS` (2), the inline
  `if action == 'park'` branch with `_park_on_info_refusal`, `scripts/sitting/payloads.py::RESOLVE_ACTIONS`
  (a copy pinned by `scripts/tests/test_sitting_payloads.py`), `authority.py::_RESOLVE_ACTIONS` (derived
  from `ACTION_EFFECTS`), and the harness's legacy inverse map.
- **Header reads:** five ladder tools read `get_http_headers()` by hand today — `resolve_issue` (levels
  header intersected with the role ceiling), `promote_to_l2` (identity only, levels header ignored),
  `stamp_triage`, `claim_policy_class` and `amend_escalation` (identity for attribution); S3 κ 9173
  replaces those reads with `read_caller_headers` before any S2 leaf runs. `declare_pin` reads no header;
  `claim_policy_class` already refuses every identity but the executor's, header-less ones included. At
  main a header-less connection keeps full authority by design (`authority.py` module docstring, the
  esc-2087-2 human-channel guarantee of 2026-08-20), which the sitting's S2-2 ruling ends; the tracked
  `.mcp.json` sends no header. `resolve_issue`, `stamp_triage` and `amend_escalation` already overwrite
  the caller's attribution argument with the identity header whenever one is sent. The supervised auto-watcher connects with
  `harness.py::_WATCHER_ESCALATION_HEADERS` (`X-Escalation-Levels: 0,1`, identity
  `orchestrator-escalation-watcher-auto`); `authority.py::ROLE_LEVEL_ALLOWLIST` gives that identity
  {0, 1} and `authority.py::PROMOTE_ALLOWED` admits it, and `promote_to_l2` (which mints a level-2 record)
  is its only route to the human.
- **Client configurations** (measured 2026-11-13 by reading every `~/src/*/.mcp.json`; no user- or
  local-scope escalation entry exists in `~/.claude.json`, per `digests/discussion/S2.md` §S2-2). Twelve
  projects carry an escalation entry and none sends a header: dark-factory (8102), reify (8100),
  autopilot-video (8101), know-live (8105), solar-challenge (8106), solar-challenge-platform (8107),
  pump-web-ui (8108), scoping-intake (8109), stream-touch (8110), market-predictor (8112), simple-algo
  (8104) are tracked; autotrade (8111) is an untracked local file. Twelve side checkouts carry a tracked
  copy and pick the change up only when pulled (dark-factory-hotfix-mainhealth, df-interactive-5467,
  reify-fix10/12/13/14/15, reify-recovery, reify-task-7370, reify-unblock-5481/5602,
  reify-analysis-spec-coverage). dark-factory's reconciliation queue (8103) is reached through the
  tracked `recon-watch/mcp.json`, which `skills/recon-escalation-watcher/SKILL.md` §Launching prints;
  it sends no header either. The laptop's checkouts were not measured. Re-read by the fold-verifier at
  `fceb951e7e`: the same twelve entries and twelve side checkouts, none with `headers`; nine
  orchestrator units run on this host (autopilot-video, dark-factory, know-live, solar-challenge,
  pump-web-ui, reify, scoping-intake, solar-challenge-platform, stream-touch). HTTP callers of an
  authority-bearing tool outside `.mcp.json` (grep of non-test code in the twelve repositories for the six
  tool names): `scripts/sitting/apply.py` (already identified as `interactive:coordinator`),
  `fused-memory/scripts/cgl_eta_finalize_gate.py` (header-less `resolve_issue`; its gate 2273 is done) and
  `reify/scripts/docs-truth-sweep.py` (header-less `promote_to_l2`, reading only the URL from
  `--mcp-config`; a task in reify's own store gives it an identity, §ν). The five dark-factory timer
  scripts only file.
- **Spawned agents' escalation entries** (measured at `e59840cdaf`):
  `orchestrator/src/orchestrator/mcp_lifecycle.py::McpLifecycle.mcp_config_json` adds headers only when
  passed `escalation_headers`, and only the auto-watcher rotation passes them
  (`harness.py::_WATCHER_ESCALATION_HEADERS`). Its five other production call sites pass none: the
  steward's continuation (`steward.py`) and its triage config, the task workflow's role config builder
  (`workflow.py::compose_invocation_mcp_config`), the review checkpoint (`review_checkpoint.py`) and the dry-run unblock
  (`dry_run_unblock.py`). Three of them pass an escalation URL and so build a header-less escalation
  entry; the steward's triage config and the dry-run unblock pass none and build no escalation entry.
  The steward is the only spawned role holding an authority-bearing escalation tool (`resolve_issue`,
  `agents/roles.py::STEWARD`). Each dispatched agent already has one identity string,
  `orchestrator/src/orchestrator/agents/briefing.py::_caller_agent_id` (`claude-task-<id>-<role>`), and
  `classify.py::classify_resolver_tier` classes `claude-task-<id>-steward` as the `steward` tier. The
  eval cells' lifecycle stub, `orchestrator/src/orchestrator/evals/runner.py::_EvalMcpStub`, returns
  `{}`: an eval agent gets no escalation server at all.
- **Stored-fingerprint readers** (measured at `e59840cdaf`). `Escalation.dedupe_fingerprint` is written
  by about twenty production filers (the orchestrator's workflow, merge lane and provenance conflict;
  fused-memory's five escalators, reconciliation harness, backlog policy, stall detector and consolidation
  executor; the memory-eval runner) and read by five readers, none of which this plan's draft moved:
  `dashboard/src/dashboard/data/memory_evals.py`'s join of open `eval_regression` escalations;
  `fused-memory/src/fused_memory/reconciliation/escalation_archive.py`'s terminal fingerprint set; the
  reconciliation harness's two re-file checks (`fused-memory/src/fused_memory/reconciliation/harness.py`,
  the pending-fingerprint set and its per-finding fallback); `queue.py::EscalationQueue.has_open_l1`'s
  fingerprint filter (S1 α's `has_open`); and `dedupe.py::content_fingerprint_key` and
  `gate_backlog_fingerprint_key`, the fold keys. Each would stop matching a record that declares only a
  `failure_signature`.
- **Prose identities in the harness:** `harness.py::_is_scope_divergence_orphan` matches the summary
  substring `'plan.files/metadata.files divergence detected'`, which `workflow.py::TaskWorkflow._escalate_scope_invariant_violation`
  keeps "verbatim"; `classify.py::is_done_step_commit_orphan` keys on `suggested_action ==
  'verify_wip_reconciliation'`; `harness.py::Harness._reblock_signature` is `category + ':' +
  normalised(summary)[:120]`.
- **Fold identity:** `dedupe.py::summary_dedupe_key` (first three casefolded summary tokens),
  `canonical.canonical_root_cause`, `dedupe.py::gate_backlog_fingerprint_key` (parses `project_id:` out of
  `detail`); `_chokepoint_or_submit` appends `[downgraded:<severity>]` to the summary;
  `Escalation.dedupe_fingerprint` is the recon filers' declared identity.
- **Evidence:** `escalate_info` and `escalate_blocker` `cast(list[EvidenceEntry], evidence or [])`; there
  is no hypothesis field (4 hand-written `Hypothesis` lines in `server.py`'s own reports).
- **Reader error path:** `queue.py::_SCAN_PARSE_ERRORS` is `(json.JSONDecodeError, KeyError, TypeError)`;
  `EscalationQueue.get` and 7 read-modify-write sites in `queue.py` catch that tuple (measured at
  `ba17934861`, unchanged at `e59840cdaf`). Only `sweep.py::_SWEEP_PARSE_ERRORS` and `shadow_ruling.py::_SWEEP_PARSE_ERRORS` add a
  bare `ValueError`.
- **`DedupeConfig`:** a mutable `@dataclass`; 133 `infra_dedupe_` references in 32 `.py` files
  (escalation 77 in 11, fused-memory 37 in 14, orchestrator 18 in 6, scripts 1 in 1; tests included);
  162 references in 43 files across all file types (measured at `ba17934861`, unchanged at
  `e59840cdaf`).
- **Role prompt:** `agents/roles.py::ESCALATION_LADDER_CORE` still says the dedupe gate "covers only
  `category='infra_issue'` inside a short (600s) window" and that "a `critical`/`urgent` filing bypasses
  dedupe entirely — … a NEW page to the human", while the server downgrades every non-sentinel
  critical/urgent filing to `blocking` before dedupe.

## Findings disposition

Every key of the stream appears once. Step 1 of the contract §8 dedup (`find_tasks_by_metadata(key=
"x_finding_key", value=<key>)`, run 2026-11-10 by the author and 2026-11-11 by the critic for all 14 keys)
returns no task; re-run by the ruling fold on 2026-11-13 for all 14 keys, still none. Step 2
(`search_tasks` on each leaf's remedy, threshold 0.6, confirmed with `get_task`/a read-only store read)
finds one live task, 8777 (subsumed by β1), and done tasks this PRD builds on (1617, 2041, 2204, 4192).
Task 8784 scored 0.52, below the bar, but β2 deletes the function it would change, so it is cancelled
with a reason (§Cross-PRD). The ruling fold's new leaves (γ2, ε2, μ, ν, and ξ and ο filed later) score
below 0.6 against every filed task (best: done tasks 2041 at 0.597 and 2042 at 0.592, the 2026-07
header guard this plan builds on).

| Key | Sev | Disposition | Re-verified at `1f240e6da2`, re-checked at `e59840cdaf` |
|---|---|---|---|
| `fk-8c48269c7363` | medium | leaves β1 (field, readers), β2 (attribution, event) | one writer/reader pair since `343e658060`, but the fact is still in `resolved_by`; `park` stamps pending members in memory; `classify._CASCADE_PREFIX` second spelling; the harness's parent lookup still falls through silently to the legacy map |
| `fk-5332d25b1702` | medium | leaves β1/β2 (cascade half), γ (fields), θ1 (readers) | cascade read via `cascade_parent_id(resolved_by)` ×2; summary-substring orphan class; `suggested_action` discriminator; summary-derived reblock signature; no `filing_site`/`failure_signature` field |
| `fk-e3b7459c681d` | medium | leaf δ (α adds the column) | `ACTION_EFFECTS` carries only the task side; `_DISMISS_ACTIONS`, the park branch and the sitting copy are separate |
| `fk-d0e1094c99cb` | high | leaf δ (α adds the precondition column) | the "reconstructed independently" half is gone (`effect_for` consumed by the harness since `0197fe8969`); the legality half holds: every row wildcarded, the one per-action precondition (`_park_on_info_refusal`) inline |
| `fk-bbb0a7b3c8a8` | medium | leaf α | `status`, `severity`, `resolution_action` bare `str`, no `__post_init__`; the category half moved to `categories.py` (`a08453dba9`) |
| `fk-60b7c33ea0ad` | medium | leaves η, θ1, θ2, ι (consumer half); the vocabulary half is `fixed:a08453dba9` | `CATEGORIES` gone, registry enforced at submit; 8 category literal comparisons remain outside `escalation/` (measured at `e59840cdaf`; 9 at `ba17934861`) |
| `fk-638cca2f65bb` | medium | leaves γ (the record facts and the one failure identity), γ2 (the five stored-fingerprint readers read it) and γ3 (the stored fingerprint is retired, after S1 ω) | three prose normalisers unchanged; downgrade suffix still written into the key-bearing summary; the stored fingerprint is a second spelling of the failure identity with five direct readers |
| `fk-479375564c2f` | medium | leaves α (evidence parsed), γ (hypothesis field) | `cast` ×2 unchanged; no hypothesis field |
| `fk-755f786c32be` | medium | leaf ε | five hand-written header reads with four meanings (S3 κ 9173 unifies the read, not the meaning); `promote_to_l2` ignores the levels header |
| `fk-48a76ee9f746` | high | leaf ε2 | `L2_AUTO_CLOSE_ALLOWLIST` rows (`superseded_main_sweep`, `self_cleared_infra`, `stale_task_scoped`) still authorise on presence-only regexes over the watcher's own text, and the executor gate reads the same rows; the server never asks the task store, though `create_server` takes `task_status_lookup` |
| `fk-ece88b7661b7` | high | closed by ο, the denial (filed by the lead after ξ is done, §Not filed), on top of ε (parser, channel stamp, self-explaining refusal, no denial), μ and ν (identity deployed to every client) and ξ (the check, filed by the lead once ν is done and the fleet runs μ); the key is stamped on ο when it is filed | header-less callers keep full authority (the esc-2087-2 guarantee) and the server cannot tell an in-process call from an unidentified HTTP client |
| `fk-9c7ebc2b50b2` | low | leaf κ | 133 references in 32 `.py` files (measured at `ba17934861`), still mutable, two "historical" disclaimers (three at as_of) |
| `fk-a65731424439` | medium | leaf η | 11 dashboard sites in 5 modules by the guard rule (the run's grep rule counted 12, including `memory_evals._OPEN_ESCALATION_STATUS`), four unknown-status behaviours unchanged |
| `fk-89388166d3ca` | medium | leaf ζ | the 600 s / bypass / new-page claims unchanged |

## Pre-conditions for activating

- **S3's non-held leaves** on the shared files (all pending on 2026-11-13): κ 9173 (C6 header reader,
  `models.py`), β 9157 (C4 mutation primitives and callback delivery, `queue.py`, `harness.py`), γ 9162
  (`scan.py`, `queue.py`, `models.py`), ι 9181 (C8 gate tables, `server.py`). None of them waits on S6's
  native-port chain or on task 5040; ι waits on θ 9177, whose open prerequisites are 9162, 9173 and 9199
  (task 4567 is done, `b255f96b3e`).
- **Task 8562** (instruments-honesty δ, pending, no open prerequisite) lands before α; S2 extends 8562's
  `classify.resolver_tier(record)`. Task 6390 (the typed `review_due` payload) has landed (`bbaa5f91e4`).
- **S19 θ0** (the package-layering plan's `shared/escalation_layout.py`, its C9: the archive layout and
  the standard-library terminal status names; about 120 lines, no prerequisites) lands before α, which
  builds `EscalationStatus` from those names (lead ruling S19/Q-C8).
- **The S1 filing entry point, in a per-file order** (sitting ruling S2-3 (a)). S2's typed record lands
  first and S1's filer is typed from its first commit:
  - `models.py`, `queue.py`: S2 α, β1, β2, then S1 α (S1 α → S2 β2 holds through 9181 → 9177 → β2; S1
    lands its `Sentinel` registry after S2 β2, lead ruling S2/OQ-6).
  - The escalate tools and the chokepoint (`server.py`, `filing.py`, `dedupe.py`): S2 γ → S1 β (both
    edit the escalate tools in `server.py`; δ follows through γ). γ adds
    `original_severity`, `hypothesis`, `failure_signature` and `filing_site` to S1's `EscalationDraft` and
    its parity test, and writes `original_severity` in S1 C2 step 2.
  - The workflow filers: S2 θ1 → S1 δ (the 17 workflow filings through `TaskWorkflow._file_escalation`)
    and → S1 η2 (S1's last `harness.py` leaf).
  - Orchestrator readers: S2 θ2 → S1 γ, κ1, κ3, κ4; recon: S1 ι1 → S2 η.
  - `promote_to_l2` and the other ladder filers: S2 ε → S1 β (every other escalation-package filer
    through the door, including `promote_to_l2`'s create step). The caller-identity task follows the
    door's other-filers task, so the door carries `filed_via` on every record it writes from ε on.
  - `dedupe.py` and the fused-memory escalators: S2 κ → S1 θ2 (the fused-memory escalators that build
    their own queue, and the ticket janitor), S1 ι2 (the other recon filers; `dedupe.file_or_fold_l1`
    deleted) and S1 ω (the `escalation-filing-one-door` integration gate). κ lands last on those files.
  - The stored fingerprint (the lead's "FINGERPRINT" ruling of 2026-11-13, which amends lead S2/OQ-3;
    D12): S1 keeps writing `dedupe_fingerprint` until its closing task ω. This plan carries the
    retirement: γ2 moves the five stored-fingerprint readers onto the one declared failure identity (it
    needs only γ), then γ3 deletes the field and S1's fingerprint arguments, depending on γ2 and S1 ω.
    No S1 leaf depends on either.
  No S1 leaf is a prerequisite of S2 α, β1 or β2: an edge from either of them onto S1 α would close the
  cycle S2 α → S1 α → 9181 → 9177 → S2 β2 → S2 α. Once task 9490 lands the task store refuses that edge
  (`DependencyCycle`, the path named); until then the lead re-checks the store after wiring.
- **The identity deploy precedes the denial** (sitting ruling S2-2, Leo: "deploy the MCP configuration
  to all existing projects before denying"). ε lands the parser, the channel stamp and the refusal's
  message with no denial; μ (dark-factory's own clients and every spawned agent) and ν (the other
  projects, an operator-sweep gate) deploy the identity; ξ, filed by the lead once ν is done and every
  orchestrator and fused-memory runs μ, shows three days with no unidentified authority-bearing action; ο, the denial, is filed by the
  lead only after ξ is done (R58's policy-flip rule). The escalation server runs inside each project's
  orchestrator from the shared package, so the denial reaches a project when that orchestrator restarts.
  A client neither μ nor ν found loses its authority-bearing
  act with ε's self-explaining `identity_required` message, never an escalation: filings and reads stay open.

## Sketch of approach

**α — the typed vocabulary.** `escalation/models.py` gains `EscalationStatus` (with `OPEN_STATUSES`,
`TERMINAL_STATUSES`; its `RESOLVED` and `DISMISSED` values come from `shared.escalation_layout`'s
standard-library names, S19 C9), `Severity` (with `BORN_AT_L2_SEVERITIES` and `SEVERITY_RANK` derived from it,
replacing the hand-kept frozenset, set and dict), `ResolveAction`, and one error type
`UnknownEscalationValue(field, value, legal, record_id)`. `Escalation.__post_init__` parses `status`,
`severity` and `resolution_action` into members and raises with an INV-2 message on anything else; it
coerces a registered `category` to its `EscalationCategory` member and keeps an unregistered one verbatim
(5257's rule: rehydration is never checked); it validates each evidence entry (`observation`,
`measured_at`, `ref` present as strings, extra keys kept). Every stored record on this host parses
(measured). A record that fails construction is an unparseable record, exactly like malformed JSON:
`UnknownEscalationValue` joins `_SCAN_PARSE_ERRORS` (in `scan.py` after S3 γ 9162) and
`EscalationQueue.get`'s tuple, so every listing skips it per record with a WARNING naming the file and the
field and counts it, and a mutation of it is S3 C4's `unparseable` outcome. `action_effects.py` keys
`ACTION_EFFECTS` by `ResolveAction`, gains `RecordDisposition` and two columns, `record_disposition` and
`preconditions`, and exports `RESOLVE_ACTIONS = tuple(ResolveAction)`. `classify.effective_benign` reads
the status sets.

**β1 — the cascade parent is a record field, and every resolver-tier reader reads it.**
`Escalation.cascade_parent_id: str | None = None`. `EscalationQueue.resolve`'s member cascade writes it on
each archived member; `EscalationQueue.park` writes it on each pending member through S3's
`_mutate_pending`, durably. In β1 the queue still writes today's `l2-cascade:<id>` attribution beside the
field, so the harness's two prefix reads keep working unchanged until β2. Every `classify_resolver_tier`
caller moves to a record-based answer that reads `cascade_parent_id` first:
`_is_late_resolution_worth_capturing` and `_capture_late_resolution` take an explicit `via_cascade: bool`
from the cascade call instead of inferring it from `resolved_by`; `closed_by_non_sweep_resolver(record)`
and `default_resolution_class_for_resolver` count a record with `cascade_parent_id` as the cascade tier;
the dashboard's `_lifespan_block`/`_workflow_block` and `shadow_ruling.agreement_report` use
`resolver_tier(record)` (8562's), in which `cascade_parent_id is not None` → `cascade` is answered before
`ruled_by_human`. A startup-sweep pass, `backfill_record_facts`, sets the field on legacy records
— from the `l2-cascade:` prefix on archived members (the only place the prefix is parsed after β2, inside
the migration, through `_mutate_terminal`) and from the `members` list of a pending parked L2 for its
pending members — and is idempotent. `classify._CASCADE_PREFIX` is deleted.

**β2 — the resolve callback receives a frozen event; `resolved_by` is attribution only.** The queue builds
one frozen `ResolutionEvent` per resolved, parked or cascaded record — carrying the effective action,
decided in the queue where the parent is in hand, and the recut facts the harness needs
(`recut`, `ruling_record_id`, `ruling_resolved_at`: the record whose resolution issued the action, the
member itself or its cascade parent) — and delivers it through S3 C4's `deliver_on` contract. A cascade
member's `resolved_by` becomes the parent's resolver; a parked member's stays `None`. The
harness's `_on_escalation_resolved` takes the event and builds the recut ruling from it alone;
`_cascade_parent`, `_resolve_escalation_action` and its legacy map, and `_restart_recut_ruling`'s `parent`
parameter go. `cascade_resolved_by`, `cascade_parent_id(resolved_by)`, `_L2_CASCADE_RESOLVER_PREFIX` and
park's in-memory stamp are deleted. Tests that drove the private callback move onto public seams:
`escalation/tests/_resolution_events.py::event_for(record, …)` builds an event from a record for the
escalation tests, and `orchestrator/tests/_escalation_events.py::deliver(harness, record_or_event)` drives
the harness callback for the 12 orchestrator test files that call `_on_escalation_resolved` directly. The
16 sites that register `harness._on_escalation_resolved` on a real queue need no edit: the method takes
the event the queue now delivers. The L2 watcher's skill describes today's cascade
(`skills/escalation-watcher/SKILL.md`, its "L2 cluster cascade" row and the cascade paragraph that says
`queue.resolve()` stamps `resolved_by='l2-cascade:<L2-id>'` and the harness reads the parent through the
queue's read API); β2 rewrites both to name `cascade_parent_id` and the event in the same commit.

Why two leaves, and why the callback change is β2's: a callback type change without its consumer turns
main red — `Harness._start_escalation_server` and those 16 test sites hand the queue a callable that reads
a record. So the producer and consumer sides of the callback change land together in β2, and β1 is the
part that is green alone: the field, the migration and the eight readers. Rollout (INV-13): each project's
queue has two long-running writers, its orchestrator and the system-wide fused-memory service, restarting
on separate clocks; a legacy-format write from a not-yet-restarted writer is backfilled by that queue's
next startup sweep, and after β2 nothing parses `resolved_by` outside the migration.

**γ — filer-declared facts.** `Escalation` gains `filing_site: FilingSite | None`, `failure_signature:
str | None`, `original_severity: str | None` (a `Severity` member at runtime, D2) and `hypothesis:
str | None`. One reader answers "which failure is this" (lead ruling S2/OQ-3):
`models.declared_failure_identity(esc)` is the filer's declaration, `failure_signature` else the stored
`dedupe_fingerprint` else `None`, and `models.failure_identity(esc)` is that declaration else the legacy
`category:normalised-summary` form, so it is total. The infra fold's default key
(`dedupe.py::_default_summary_key`) uses a declared `failure_signature` when present and the summary key
otherwise, so no persisted cluster re-partitions. γ lands after S1's filing entry point (sitting ruling
S2-3 (a)): S1's `EscalationDraft` (S1 C1) gains the four fields and its
parity test covers them, `escalate_info` and `escalate_blocker` take `hypothesis` and
`failure_signature` and drop the evidence `cast`, and the downgrade S1 moved into `filing.py` writes
`original_severity` instead of the summary suffix. `backfill_record_facts` stamps `filing_site` on legacy
pending records that match today's two discriminators, so the readers θ1 switches need no fallback.

**γ2 — the stored-fingerprint readers read the one declared identity.** The five readers that compare
`dedupe_fingerprint` directly (§Background) read `declared_failure_identity` instead: the Memory Evals
join, the escalation-archive fingerprint set, the reconciliation harness's pending-fingerprint set and its
per-finding fallback, the queue's open-record filter (`has_open`, S1 C3, compared on both the draft and
the record) and the two fold keys in `dedupe.py`. A legacy record matches exactly as today, and a record
that declares only a `failure_signature` matches too. γ2 is the first half of the retirement the lead
ruled (D12), and it can land before S1 ω because it changes no write.

**γ3 — the stored fingerprint is retired.** After S1 ω every filer builds an `EscalationDraft`, so one
change finishes the retirement: every filer that passes `dedupe_fingerprint=X` passes
`failure_signature=X`; `Escalation.dedupe_fingerprint` and `EscalationDraft.dedupe_fingerprint` go; S1's
`OpenAtLevel(same_fingerprint=True)`, `EscalationQueue.has_open(dedupe_fingerprint=)` and the
`Fingerprint` fold key key on `failure_signature` (exact names tactical); `declared_failure_identity`
becomes `failure_signature` alone. Records on disk keep the old key, and `Escalation.from_dict` drops
keys that are not fields, so the deletion would silently strip every legacy identity: γ3 therefore
extends `backfill_record_facts` (D7) first, which copies a raw `dedupe_fingerprint` into
`failure_signature` on every pending and archived record that lacks one, in every queue both harnesses
serve (the eleven project queues and the reconciliation queue).

**δ — `resolve_issue` runs on the table; `duplicate_of` (C5).** `resolve_issue` selects its branch from
`effect.record_disposition` (`KEEP_OPEN_AT_L2` → `queue.park`; `ARCHIVE_RESOLVED`/`ARCHIVE_DISMISSED` →
`queue.resolve(dismiss=…)`); `_DISMISS_ACTIONS`, `server.RESOLVE_ACTIONS` and the sitting copy go, and
`authority.py`'s module-docstring pointer at "the existing RESOLVE_ACTIONS / _DISMISS_ACTIONS convention"
is rewritten to name `ACTION_EFFECTS`. Per-action legality is declared on the row: `preconditions` holds
named predicates (today one, `park` refuses an info record, from `_park_on_info_refusal`), and one gate in
S3's `RESOLVE_GATES` evaluates the row's preconditions, so a new action's legality is one row.
`resolve_issue(..., duplicate_of=…)` is accepted only on a row whose disposition is `ARCHIVE_DISMISSED`;
the survivor must exist (any status) and differ from the record; `queue.resolve` persists it; a cascade
member never inherits it.

**ε — one caller identity, stamped, with a self-explaining refusal and no denial yet.**
`models.parse_caller_identity(headers, *, in_request)` absorbs S3 κ's `read_caller_headers` and
`CallerHeaders` and returns a frozen `CallerIdentity(identity, levels, channel)`; a call outside an HTTP
request is `IN_PROCESS` (tests' `tool.fn()`; both harnesses serve the server over HTTP only, through
`http_app()`), an HTTP call without an identity header is `UNIDENTIFIED`. `authority.effective_levels(caller)`
is the one ceiling computation. The five tools that call `read_caller_headers` after S3 κ, and
`declare_pin`, call the pair instead: every tool rejects a malformed header the same way.
`authority.AUTHORITY_BEARING` names the ladder's authority-bearing acts, one row per tool and action:
every `resolve_issue` action (park included) at every level, `promote_to_l2`, `stamp_triage`,
`amend_escalation`, `claim_policy_class` and `declare_pin`; filings and reads are not in it.
`authority.identity_required_refusal(caller, *, tool, action, level)` is the one refusal for an
`UNIDENTIFIED` caller on such an act: code `identity_required`, a message that names the missing
`X-Escalation-Identity` header, the tool, the refused action and level, the config file (the project's
`.mcp.json`, or the `--mcp-config` file the session was launched with) and the fix (the `"headers"` line
to add), pinned word for word by a boundary test. Until ο lands the server does not refuse: an
`UNIDENTIFIED` authority-bearing call succeeds, its response carries the same text under
`identity_warning`, and one WARNING line names the tool and action. The channel is stamped on
authority-bearing writes: `filed_via` at submit (the door writes it on every record, S1's draft carries
it, in-process filers default to `IN_PROCESS`) and `resolved_via` at resolve, park and promote, so
neither overwrites the other. `scripts/check_unidentified_escalation_actions.py` counts records whose
`resolved_via` is `unidentified` since a given time in every discovered queue and exits non-zero when
there is any; it is ξ's predicate. Its queues are each known project root's `data/escalations` plus
every `--queue-dir` it is given: the reconciliation queue (`data/reconciliation/escalations` under
dark-factory, where the recon watcher resolves) is under no project's `data/escalations` and the host's
`DASHBOARD_KNOWN_PROJECT_ROOTS` does not reach it (measured at `fceb951e7e`), so ξ passes it explicitly. Attribution (D11): `authority.HUMAN_IDENTITIES` (`interactive`,
`escalation-watcher`, `interactive:coordinator`) is the one human set, which `classify.py`'s human tier
reads; `PROMOTE_ALLOWED` gains it, so the deploy does not take promotion away from a human session; for a
human identity the tool's `resolved_by` argument is kept as `resolver_detail`. `promote_to_l2` authorises
by identity first (lead ruling S2/OQ-9 as amended by the sitting's S2-2): an identity in `PROMOTE_ALLOWED`
may mint regardless of its level ceiling (the supervised watcher keeps its only L1→L2 path); an identity
outside it is refused (today's rule); with no identity, a levels header that lacks 2 is refused
`level_forbidden`; with no header at all the call is an `UNIDENTIFIED` authority-bearing act, warned
until ο and refused `identity_required` by ο.

**ε2 — corroborated L2 auto-close.** Corroboration is a predicate on the policy row,
`PolicyRow.corroborate: Callable[[Escalation, TaskStatusLookup], bool] | None`, evaluated by both doors
that read the row — `l2_auto_close_class` (the watcher) and `executor_refusal` (the executor, through
`policy_gate.identity_resolve_refusal`). For `stale_task_scoped` the only authorising fact is "the subject
task's status, read through the server's task-status lookup, is `done` or `cancelled`" (sitting ruling
S2-1 (a′): the task store decides); re-scoped and re-dispatched citations and a `cited_task_id` argument
are audit annotations; a lookup that is unwired, times out or raises refuses. The lookup is called from
the threadpool tool with `anyio.from_thread.run` and a bounded timeout. The two classes the server
cannot corroborate (`superseded_main_sweep`, `self_cleared_infra`) keep presence-only evidence under the
decision-stream audit; a refusal leaves the L2 pending for the human (D10).

**μ — every dark-factory client and every spawned agent sends an identity.** `McpLifecycle.mcp_config_json`
takes `escalation_identity` and refuses to build an escalation entry without one (`ValueError` naming the
caller's missing argument), so no future caller can build a header-less escalation connection. The three
call sites that pass an escalation URL (the steward's continuation, `workflow.py::compose_invocation_mcp_config`,
the review checkpoint) pass the dispatched agent id from `briefing.py::_caller_agent_id`, made public;
the steward's triage config and the dry-run unblock pass no URL and are unchanged; the eval stub accepts
the argument and still returns `{}`. dark-factory's `.mcp.json` escalation entry gains
`"headers": {"X-Escalation-Identity": "${ESCALATION_IDENTITY:-interactive}"}`, `recon-watch/mcp.json`
and the copy `skills/recon-escalation-watcher/SKILL.md` §Launching prints gain the same, and
`skills/escalation-watcher/SKILL.md` tells the L2 watcher's launcher to set `ESCALATION_IDENTITY=escalation-watcher`
(the terminal and tmux lanes of `skills/spawn/spawn-claude.sh` pass the launcher's environment; a T3-hosted
watcher thread does not, so it sends `interactive` and its tool argument survives as `resolver_detail`;
both identities are in `HUMAN_IDENTITIES`, so its authority is the same).

Two facts read by the fold-verifier, measured at `fceb951e7e` with Claude Code 2.1.322 installed on this
host:
- Header expansion. The CLI's MCP-config loader passes every value of an `http` entry's `headers` through
  its environment expander, whose pattern is `${NAME}` or `${NAME:-default}`: an unset variable takes the
  default, a variable set to the empty string expands to the empty string (not the default), and a
  credential-shaped name is blanked for a remote server (`ESCALATION_IDENTITY` is not credential-shaped).
  So the expansion is supported, read from the shipped code rather than tried live. μ's first step still
  tries it on one live session; if a future CLI does not expand it, the entry carries the literal
  `interactive`, and the watcher launcher passes `--strict-mcp-config --mcp-config <file>` whose header is
  `escalation-watcher` (strict, because two entries named `escalation` would otherwise collide). The
  server treats an empty header value as no header.
- Spawned agents load two escalation entries. Production runs with `strict_mcp_config: false`
  (`orchestrator/src/orchestrator/config.py`), so a task agent in a worktree merges the worktree's
  `.mcp.json` with the orchestrator-built `--mcp-config` (measured: 10 of 60 task-agent sessions of the last
  24 h list the `.mcp.json`-only `playwright` tools). Both define a server named `escalation`, and which
  one the CLI keeps was not established. Once `.mcp.json` sends `${ESCALATION_IDENTITY:-interactive}`, an
  agent whose kept entry is the ambient one would claim the human identity. So μ also puts
  `ESCALATION_IDENTITY=<the dispatched agent id>` in every spawned agent's environment (the
  `env_overrides` of the invocation sites it edits, and the dry-run unblock's, which builds no escalation
  entry of its own but still loads the ambient one): whichever entry the CLI keeps, it sends the agent's
  id. The orchestrator must be restarted on μ before any worktree branched from μ dispatches an agent; μ's
  signal checks it.

μ also gives the one other dark-factory HTTP caller of an authority-bearing tool an identity:
`fused-memory/scripts/cgl_eta_finalize_gate.py` calls `resolve_issue` header-less (its gate, task 2273, is
done, so it is a no-op today, but a rerun after ο would be refused). `scripts/sitting/apply.py` already
sends `interactive:coordinator`; `scripts/migrate_open_consolidation_gates.py` only prints the calls it
would make (measured by grepping non-test code for the six tools' names, at `fceb951e7e`).

**ν — every other project's escalation entry sends an identity (operator-sweep gate).** A dark-factory
task cannot commit to another repository, so the eleven other projects' `.mcp.json` files change through
one human-gate task in the program's operator-sweep shape (the cross-stream consequence shared with S16's
yaml sweep): `task_kind='deterministic'`, `always_escalates=true`, no `before_done`, filing a
born-at-L2 escalation whose description is a `/team` brief and quotes Leo's pre-authorisation of
2026-11-12. One `.mcp.json`-only `git commit --only` per tracked repository (autotrade's untracked file
is edited in place), each sha or skip reason recorded on the escalation; a resume marks the gate done.
The twelve side checkouts that carry a tracked copy of a `.mcp.json` are fast-forwarded or edited in
place (an uncommitted one-line change in a scratch checkout), each recorded, so no checkout on this host
is left header-less. The session changes configuration only. A repository's own non-test code that
calls an authority-bearing tool over HTTP with its own client would also be refused by ο, but it is not
edited by this gate (lead ruling 2026-11-13): the one measured at `fceb951e7e` across the eleven
repositories, `reify/scripts/docs-truth-sweep.py` (it calls `promote_to_l2` to book its human sitting and
reads only the URL from `--mcp-config`; its systemd unit is in `reify/deploy/systemd/`, not installed on
this host today), gets its identity header through a task the lead files in reify's own store at
decompose, and ν depends on that task as a cross-project dependency (`"reify:<id>"`), so the check ξ and
the denial ο, which follow ν, come after it. Any further such caller the session's grep finds is recorded
on the escalation, not edited, and the lead files it the same way before ξ.

**ξ and ο — the check, then the denial (filed by the lead, §Not filed).** ξ is a deterministic predicate
task: three days after μ and ν are done, `scripts/check_unidentified_escalation_actions.py` must find no
record whose `resolved_via` is `unidentified`; a non-zero exit files `milestone_check_failed` and a human
finds the client the deploy missed. ο turns ε's warning into the refusal: every `UNIDENTIFIED`
authority-bearing act is refused `identity_required`, a header-less `promote_to_l2` included; the five
esc-2087-2 test pins change, and the prose that promises the human channel full authority goes.

**ζ — the ladder prompt states live facts.** `ESCALATION_LADDER_CORE`'s dedupe and downgrade sentences
are rendered at import from `DedupeConfig`'s defaults, `BORN_AT_L2_SEVERITIES`,
`HARNESS_SENTINEL_ROLE_PREFIXES` and the `original_severity` field, and the prompt names the
`hypothesis` argument. A test changes the default and sees the prompt change.

**η — dashboard and fused-memory readers.** The five dashboard modules and two fused-memory
reconciliation modules compare through C1, including the store-anchored dict read
`backlog_policy.py::BacklogPolicy._close_pending_halt_records` (`record.get('status') != 'pending'` →
`EscalationStatus.PENDING`, which also keeps S10's task-status guard, task 9285, from flagging an
escalation record). Object readers never see an unknown status: α makes such a record unparseable, and
the dashboard reports the skip count under the one name `unknown_escalation_status`. Raw-dict readers
(`performance.py`, `memory_evals.py`) apply the one policy `memory_evals` already argues for: named
`unknown_escalation_status`, excluded from both open and terminal counts.

**θ1 / θ2 — orchestrator readers.** θ1: the workflow's two filers set `filing_site` and
`failure_signature`; `_is_scope_divergence_orphan` and `classify.is_done_step_commit_orphan` read
`filing_site`; `_reblock_signature` becomes a call to `failure_identity`, and
`test_harness_reblock_signature_guard.py` moves onto `models.failure_identity`; the remaining harness and
workflow literal sites compare through C1. θ2: the nine other orchestrator modules.

**ι — the guard and the invariant.** The literal-comparison guard
(`scripts/tests/test_escalation_vocabulary_literals.py`), its seeded fixtures, the scripts' sites (incl.
`member-chain-sweep.py::sweep_queue`'s dict read), C6's rule that no module but
`models.parse_caller_identity` reads `LEVELS_HEADER` or `IDENTITY_HEADER`, and the four-site lockstep
edit registering `escalation-vocabulary-imported`, whose rule names the vocabulary's three homes:
`escalation.models`, `escalation.categories` and, for the standard-library terminal names,
`shared.escalation_layout`. The guard lives in `scripts/tests/` because the full
`test_command`'s last segment, `uv run --project shared pytest tests/scripts/ scripts/tests/`, which every
merge runs (`merge_verify_breadth: full`), runs it, and there `escalation` is importable (14 `scripts/tests` modules
import it, measured at `e59840cdaf`; 13 at `ba17934861`); a whole-tree guard in `escalation/tests/` would run only in
escalation-scoped gates, so a dashboard-only task could land a violation its own gate never checks.
S10's sibling guard sits in `shared/tests/`, but the `shared` project declares no `escalation` dependency
and this guard imports the vocabulary.

**κ — `DedupeConfig` says what it is.** Fields `enabled`, `window_secs`, `categories`; `frozen=True`;
the disclaimers go.

## Resolved design decisions

- **D1 — the ruled vocabulary, read for this tree.** R9's "Status/Severity/Category/ResolveAction
  StrEnums" are three new enums in `models.py` plus `EscalationCategory`, which task 5257 already put in
  `categories.py` with its census test; the vocabulary is imported from those two modules, with no
  re-export (lead ruling S2/OQ-4, recorded as a clarification of R9's wording). The two terminal status
  names also live, as standard-library strings, in `shared.escalation_layout` (S19 C9), because the
  cockpit's decision registry cannot import the escalation package; `EscalationStatus` takes its values
  from them, and the invariant names that third home (lead ruling S19/Q-C8; R9's invariant text gains it).
- **D2 — the record stays a mutable dataclass; the event is the frozen value.** S3 C4's `mutate(esc)`
  primitives mutate hydrated records, and 8562/S3 β are built on that. `Escalation.__post_init__` parses
  at construction; field annotations keep the wire type `str` because 595 `Escalation(` calls (109 non-test
  outside `escalation/`, 475 in tests, 11 in `escalation/src`; AST count measured at `e59840cdaf`, where the
  draft's text-grep rule had given 624) pass literals and pyright `basic` would reject them under
  enum annotations. The runtime value is always a member after construction. This applies to every new
  closed field too (`original_severity: str | None` holds a `Severity` member). Writers inside
  `escalation/` assign members in the leaves that touch them.
- **D3 — strict closed fields, open category on read, and a reader that never raises.** `status`,
  `severity`, `resolution_action` reject unknown values at construction (every stored record parses);
  `category` stays checked at filing only, so a retired category in an archive never makes a record
  unreadable. A record that fails `__post_init__` is an unparseable record, treated like malformed JSON:
  every reader skips it per record, logs a WARNING naming the file and the `UnknownEscalationValue`
  fields, and counts it. No listing raises. A mutation of it is refused (S3 C4's `unparseable` outcome).
- **D4 — park persists the parent on pending members.** R9's consequence ("a holder/notice fact that
  survives restarts") applies to parked clusters most of all; β1 writes the field durably and β2 deletes
  the in-memory stamp. `resolved_by` on a cascaded record is the parent's resolver after β2 (lead ruling
  S2/OQ-8, flagged: what operators read on cascaded members changes); on a parked member it stays `None`.
- **D5 — the effective action is decided once, in the queue.** The event carries it and the recut facts;
  the harness never re-derives either from status or from the parent record. The legacy `status → action`
  map survives only inside the queue, for a record resolved with no action (in-process callers), as
  `action_effects.default_action(status)`.
- **D6 — legality on the row, enforcement in S3's gate table.** `ACTION_EFFECTS` rows declare
  `preconditions`; S3 ι's `RESOLVE_GATES` gains one gate that evaluates them. Level and category still do
  not narrow legality (the G6 archive verification in `action_effects.py` stands); the column exists so the
  next action's legality is one row.
- **D7 — migrations are one startup-sweep pass.** `sweep.py::backfill_record_facts` (β1 creates, γ
  extends) is idempotent, runs at every server start in both harnesses beside S3 θ's pass 0, and is the
  only code that reads the legacy prefix (after β2) or the two legacy discriminators (after θ1).
- **D8 — the guard is receiver-anchored and derives its exclusions.** It flags only comparisons whose
  operand is an escalation field by the counting rule above; the terminal-only values are
  `TERMINAL_STATUSES`; there are no exemption entries (INV-12). The dict-form shape is anchored on a
  module that names the escalation store, never on `import escalation`, which would flag task-status
  reads. It is complementary to S10's task-status guard (S10 D13, task 9285), which ignores attribute
  operands: a dict-form escalation status read compared to `'pending'` satisfies both only once it uses
  `EscalationStatus`, so η and ι convert those sites.
- **D9 — prose stance.** Every docstring a leaf rewrites follows the Comments stance; rationale moves to
  the commit message; Greek labels never appear in code.
- **D10 — who decides each row** (Leo's design principle of 2026-11-12, the invariant
  `one-decision-partition-per-area`, INV-23, which S17's invariant task mints in
  `docs/legibility/design-invariants.md`; cited here by slug, not restated). Every
  table this plan builds or changes names who decides each row, and code decides every row facts settle:
  | Table | Rows | Who decides |
  |---|---|---|
  | `ACTION_EFFECTS` (α, δ) | each action's record disposition, task effect and preconditions | code |
  | Unparseable record (α) | skip, log, count | code |
  | Resolver tier (β1) | cascade, human, auto-watcher, steward, reaper, other | code |
  | `AUTHORITY_BEARING` and the identity refusal (ε, ο) | every authority-bearing act by channel | code |
  | `POLICY_TABLE` row `stale_task_scoped` (ε2) | subject task done or cancelled | code, from the task store (sitting ruling S2-1) |
  | `POLICY_TABLE` rows `superseded_main_sweep`, `self_cleared_infra` | the server cannot read the fact | the auto-watcher (an LLM) judges under the decision-stream audit; a refusal leaves the L2 to the human |
  | The identity check ξ | any unidentified authority-bearing act in three days | code; a failure goes to a human, who finds the client |
  The other `POLICY_TABLE` rows are the decision-stream plan's, unchanged here.
- **D11 — attribution under one identity per channel** (the proposal in `digests/discussion/S2.md` §S2-2
  "Consequences he may not have seen", adopted as the plan's design; a discussion proposal, not a Leo
  ruling, CONFIRMED by the lead on 2026-11-13 and told to Leo). The header names the channel, not the person: `.mcp.json`
  sends `${ESCALATION_IDENTITY:-interactive}` and the L2 watcher's launcher sets `escalation-watcher` where
  its lane passes its environment (a T3-hosted watcher sends `interactive`, keeping `escalation-watcher` as
  `resolver_detail`; the same authority either way);
  `interactive:coordinator` stays the sitting tooling's own identity and is not reused, because
  `authority.py::coordinator_denied` refuses it on any record with a fresh executor claim (still an open
  question in pending task 8477). The server keeps writing the identity into `resolved_by` (today's rule
  since task 2041) and, for a human identity, keeps the tool's `resolved_by` argument as `resolver_detail`,
  so names such as `claude-team-7174-carry` survive and the analytics' human tier still reads human.
  `authority.HUMAN_IDENTITIES` is that one set; `classify.py`'s human tier and `PROMOTE_ALLOWED` read it.
  Without the `PROMOTE_ALLOWED` change the deploy itself would cost every human session its promote:
  today a header-less session may promote and an identified one outside the set may not. Spawned agents
  send their dispatched agent id, on the orchestrator-built entry and, through `ESCALATION_IDENTITY` in
  their environment, on the ambient `.mcp.json` entry, so the steward's `claude-task-<id>-steward` keeps
  its `steward` tier and no agent claims `interactive`. Checked against the code at `fceb951e7e`:
  `HUMAN_IDENTITIES` equals today's `classify.py::_HUMAN_RESOLVERS`; neither `interactive` nor
  `escalation-watcher` is a key of `authority.py::ROLE_LEVEL_ALLOWLIST`, so both keep the unrestricted
  level ceiling a header-less session has today; `authority.py::coordinator_denied` and
  `policy_gate.py::identity_resolve_refusal` bind only the coordinator and executor identities, and task
  8562's `ruled_by_human_forbidden` binds only the executor and the auto-watcher, so neither human
  identity gains or loses an act except promotion staying open (the `PROMOTE_ALLOWED` change).
  One shift is accepted: `skills/unblock-low-risk/SKILL.md` resolves with `resolved_by="unblock-low-risk"`
  inside the L2 watcher's session, so after μ its records read `escalation-watcher` (human tier) with
  `resolver_detail="unblock-low-risk"`, where today they fall in `other-auto`. The identity is
  self-asserted, not authenticated: the denial catches a forgotten or misconfigured client, never one that
  deliberately claims to be human.
- **D12 — the stored fingerprint and the failure signature are one fact** (lead ruling S2/OQ-3 (a), as
  amended by the lead's "FINGERPRINT" ruling of 2026-11-13 in `digests/wave3-lead-rulings.md`, which
  resolves the block "TO RESOLVE AT THE RULING FOLD"). One reader answers (`declared_failure_identity`,
  `failure_identity`, C4). S1 KEEPS writing `dedupe_fingerprint` until its closing task ω: its draft
  carries it (S1 C1 `EscalationDraft.dedupe_fingerprint`), its door folds and matches on it (S1 C1
  `OpenAtLevel.same_fingerprint`, S1 C3 `has_open(dedupe_fingerprint=)`, the `Fingerprint` fold key), and
  five readers compare it directly, so stopping the writes first would silently stop all five matching new
  records. This plan carries the retirement, in two tasks because of size: γ2 moves the five readers onto
  `declared_failure_identity` (about 10 files; it needs only γ, so it can land early and makes θ1's
  signature-only filings visible to the readers at once); γ3 then deletes the field and S1's fingerprint
  arguments (about 22 production files and every test that names the field), depending on γ2 and on S1 ω,
  after which every filer builds a draft. γ3's migration copies a stored `dedupe_fingerprint` into
  `failure_signature` before the field goes, because `Escalation.from_dict` drops unknown keys. No S1
  leaf depends on γ2 or γ3; an edge from S1 α or S1 β onto either would close a cycle through γ (checked
  on the folded DAG with the store's edges; task 9490's refusal is the guard once it lands). S1's draft
  states the same (its §Cross-PRD row "one failure identity").
- **D13 — evals stay hermetic** (Leo, under S1-2). This plan adds no store, sink, filer, queue or archive
  that an eval cell can reach. The new record fields ride the existing queue; the identity change reaches
  agents through `McpLifecycle.mcp_config_json`, and eval cells use `evals/runner.py::_EvalMcpStub`,
  which returns no escalation entry; μ keeps that and pins it with a test. The check script ξ runs reads
  only.

## Contracts

```python
# C1 — escalation/src/escalation/models.py (α)
class EscalationStatus(StrEnum): PENDING = 'pending'; RESOLVED = RESOLVED; DISMISSED = DISMISSED
                                                  # the last two values from shared.escalation_layout (S19 C9)
OPEN_STATUSES: frozenset[EscalationStatus]        # {PENDING}
TERMINAL_STATUSES: frozenset[EscalationStatus]    # {RESOLVED, DISMISSED}
class Severity(StrEnum): INFO = 'info'; BLOCKING = 'blocking'; CRITICAL = 'critical'; URGENT = 'urgent'
BORN_AT_L2_SEVERITIES: frozenset[Severity]        # {CRITICAL, URGENT}; KNOWN_SEVERITIES and SEVERITY_RANK derive
class ResolveAction(StrEnum): RESUME = 'resume'; RESTART = 'restart'; PARK = 'park'; ABANDON = 'abandon'; CLOSE_ONLY = 'close_only'
class UnknownEscalationValue(ValueError):         # message names field, value, legal values, record id
    field: str; value: object; legal: tuple[str, ...]; record_id: str | None
def parse_status(raw: object) -> EscalationStatus: ...   # raises UnknownEscalationValue
def parse_severity(raw: object) -> Severity: ...
def parse_action(raw: object) -> ResolveAction: ...
# Escalation.__post_init__: parses status/severity/resolution_action; coerces a registered category;
# validates evidence entries (observation, measured_at, ref: str; extras kept).
# scan.py (after S3 γ 9162) / queue.py: UnknownEscalationValue is in _SCAN_PARSE_ERRORS and
# EscalationQueue.get's tuple — an unparseable record is skipped, logged and counted, never raised.

# C2 — escalation/src/escalation/action_effects.py (α columns; δ consumer)
class RecordDisposition(StrEnum):
    KEEP_OPEN_AT_L2 = 'keep_open_at_l2'; ARCHIVE_RESOLVED = 'archive_resolved'; ARCHIVE_DISMISSED = 'archive_dismissed'
class ActionPrecondition(StrEnum): NOT_INFO_SEVERITY = 'not_info_severity'
@dataclass(frozen=True)
class TaskEffect:
    target_status: str | None; workflow_disposition: str
    record_disposition: RecordDisposition; preconditions: frozenset[ActionPrecondition] = frozenset()
ACTION_EFFECTS: dict[tuple[ResolveAction, object, object], TaskEffect]
RESOLVE_ACTIONS: tuple[ResolveAction, ...]           # derived; the only exported action list
def effect_for(action: str, level: int, category: str) -> TaskEffect | None: ...   # unchanged
def default_action(status: EscalationStatus) -> ResolveAction: ...  # resolved -> RESUME, dismissed -> CLOSE_ONLY

# C3 — models.py, queue.py, sweep.py, classify.py (β1); models.py, queue.py (β2); extends S3 C4
cascade_parent_id: str | None = None                 # new Escalation field (β1)
def backfill_record_facts(queue: EscalationQueue) -> BackfillCounts: ...   # sweep.py; idempotent (β1)
def resolver_tier(record: Escalation) -> str: ...    # classify.py, task 8562's; cascade_parent_id first (β1)
@dataclass(frozen=True)
class ResolutionEvent:                               # β2
    record: Escalation                               # a copy taken under the lock; mutating it changes nothing
    effective_action: ResolveAction
    kind: Literal['resolved', 'parked']
    recut: bool                                      # the effective recut qualifier, from the record or its parent
    ruling_record_id: str                            # the record whose resolution issued the action
    ruling_resolved_at: str | None                   # that record's resolved_at
EscalationQueue.set_resolve_callback(cb: Callable[[ResolutionEvent], None], *, deliver_on=None)   # β2

# C4 — models.py, dedupe.py, filing.py (γ)
class FilingSite(StrEnum): SCOPE_DIVERGENCE = 'scope_divergence'; DONE_STEP_COMMIT_ORPHAN = 'done_step_commit_orphan'
filing_site: FilingSite | None = None; failure_signature: str | None = None
original_severity: str | None = None; hypothesis: str | None = None      # new Escalation fields (D2)
def declared_failure_identity(esc: Escalation | EscalationDraft) -> str | None: ...   # failure_signature, else dedupe_fingerprint
def failure_identity(esc: Escalation) -> str: ...    # declared_failure_identity, else the legacy category:summary form
# S1 C1 EscalationDraft gains the same four fields; its parity test covers them.
# γ2: the five stored-fingerprint readers (memory_evals join, escalation_archive set, the recon harness's
# two re-file checks, EscalationQueue.has_open's filter, dedupe.py's two fold keys) read declared_failure_identity.
# γ3 (after S1 ω): Escalation.dedupe_fingerprint and EscalationDraft.dedupe_fingerprint are deleted;
# declared_failure_identity(esc) is esc.failure_signature; S1's OpenAtLevel/has_open/Fingerprint key on it;
# backfill_record_facts copies a stored raw 'dedupe_fingerprint' into failure_signature (idempotent).

# C5 — models.py, queue.py, server.py (δ); requested by S25 (task 6770)
duplicate_of: str | None = None                      # new Escalation field
resolve_issue(..., duplicate_of: str | None = None)  # accepted only on an ARCHIVE_DISMISSED row
EscalationQueue.resolve(..., duplicate_of: str | None = None)

# C6 — models.py, authority.py, policy_gate.py, filing.py (ε); mcp_lifecycle.py (μ); supersedes S3 C6's read_caller_headers
class CallerChannel(StrEnum): IDENTIFIED = 'identified'; UNIDENTIFIED = 'unidentified'; IN_PROCESS = 'in_process'
@dataclass(frozen=True)
class CallerIdentity: identity: str | None; levels: frozenset[int] | None; channel: CallerChannel
class BadCapabilityHeader(ValueError): ...
def parse_caller_identity(headers: Mapping[str, str], *, in_request: bool) -> CallerIdentity: ...
def effective_levels(caller: CallerIdentity) -> frozenset[int] | None: ...   # authority.py; None = unrestricted
HUMAN_IDENTITIES: frozenset[str]                     # authority.py: interactive, escalation-watcher, interactive:coordinator
PROMOTE_ALLOWED: frozenset[str]                      # watcher-auto, executor, plus HUMAN_IDENTITIES
AUTHORITY_BEARING: frozenset[tuple[str, str | None]] # authority.py: (tool, action); every resolve_issue action, promote_to_l2,
                                                     # stamp_triage, amend_escalation, claim_policy_class, declare_pin
@dataclass(frozen=True)
class IdentityRequired:                              # code 'identity_required'
    tool: str; action: str | None; level: int | None
    def message(self) -> str: ...                    # names the header, tool, action, level, config file and the fix
def identity_required_refusal(caller: CallerIdentity, *, tool: str, action: str | None,
                              level: int | None) -> IdentityRequired | None: ...   # None unless UNIDENTIFIED and authority-bearing
filed_via: CallerChannel | None = None               # new Escalation field, written by the door at submit
resolved_via: CallerChannel | None = None            # stamped at resolve, park and promote
resolver_detail: str | None = None                   # a human identity's own resolved_by argument
EscalationDraft.filed_via: CallerChannel = CallerChannel.IN_PROCESS   # S1 C1's draft; the MCP tools pass the parsed channel
# before ο: an UNIDENTIFIED authority-bearing call succeeds and its response carries {'identity_warning': message()};
# ο: it returns {'error': message(), 'code': 'identity_required'} and changes nothing.
McpLifecycle.mcp_config_json(escalation_url: str | None = None, escalation_headers: dict | None = None,
                             *, escalation_identity: str | None = None) -> dict
                                                     # μ: an escalation_url without an identity (from either argument) raises ValueError

# C7 — authority.py, policy_gate.py, server.py (ε2)
PolicyRow.corroborate: Callable[[Escalation, TaskStatusLookup], bool] | None = None
```

Invariants. C1: after construction an `Escalation`'s `status`, `severity` and `resolution_action` are
members or the constructor raised `UnknownEscalationValue`; the status sets partition the enum; no queue
listing raises on an unparseable record, it skips, logs and counts it; no module outside `escalation/`
compares an escalation status, action or category to a literal (ι). C2: `RESOLVE_ACTIONS` equals the set
of row actions; every row has a `record_disposition`; exactly the `PARK` row keeps the record open; a
precondition refusal names the precondition, the action and the record (INV-2). C3: every cascaded or
parked member carries `cascade_parent_id` (β1); every resolver-tier reader answers `cascade` for a record
with `cascade_parent_id`, ahead of `human-ruled` (β1); no record written after β2 carries
`l2-cascade:` in `resolved_by`; the event's `effective_action` equals what the harness acts on, and the
harness builds the recut ruling from the event alone (β2); `backfill_record_facts` run twice changes
nothing the second time; the event is delivered under S3 C4's thread contract. C4: `failure_identity` is
total; a filing with no `failure_signature` folds exactly as today; a draft and the record it becomes
carry the same four facts; after γ2 no module outside `escalation/models.py` reads `dedupe_fingerprint`
to decide whether two records are one failure (writes and the parent-preserving fold copy stay); after
γ3 no code names `dedupe_fingerprint` except the migration in `backfill_record_facts`, and every legacy
record's stored fingerprint is its `failure_signature`. C5: `duplicate_of` is written only by `queue.resolve`, only with an
`ARCHIVE_DISMISSED` disposition, only naming an existing other record (any status); it is not inherited by
cascade members; refusal code `invalid_duplicate_of`. C6: no module but `parse_caller_identity` reads
`LEVELS_HEADER` or `IDENTITY_HEADER` (ι's guard checks it); every ladder tool that gates on levels calls
`effective_levels`, except that `promote_to_l2` authorises by identity first (an identity in
`PROMOTE_ALLOWED` mints regardless of its ceiling, an identity outside it is refused, with no identity
a levels header lacking 2 is `level_forbidden`, and with no header at all the call is an unidentified
authority-bearing act); a malformed levels header is `bad_capability_header` on every tool; every record
written through the door carries `filed_via`; every resolve, park and promote stamps `resolved_via`;
`identity_required_refusal` returns a refusal exactly for an `UNIDENTIFIED` caller on an
`AUTHORITY_BEARING` act, never for a filing, a read or an `IN_PROCESS` call; its message is pinned
word for word; after ο no `UNIDENTIFIED` authority-bearing call changes a record. `HUMAN_IDENTITIES`
is defined once and read by `classify.py`'s human tier and by `PROMOTE_ALLOWED`. `mcp_config_json`
never builds an escalation entry without an identity (μ). C7: a policy row with a `corroborate`
predicate authorises only when the predicate is true, on both the watcher and the executor door, and a
lookup failure is a refusal. Exact field names beyond those above are tactical.

## Boundary-test sketch

| # | Side | Scenario | Pre | Post |
|---|---|---|---|---|
| 1 | α producer | construct with `status='resolvd'`; rehydrate fixture copies of every stored status/severity/action value | — | `UnknownEscalationValue` names `status`, the value, the legal set and the id; every fixture rehydrates to members |
| 2 | α producer | evidence entry without `observation`; one with extra numeric keys; an unregistered category on rehydrate | — | first raises naming index and key; second kept; third kept verbatim, while `submit` still raises `UnregisteredEscalationCategory` |
| 3 | α consumer | a real queue dir holding one valid pending record and one whose JSON says `"status": "superseded"`; `get_pending`, `get_by_task`, `find_terminal_by_citation`, `dismiss_all_pending`, the startup sweep, `get(<bad id>)`, a mutation of the bad id | — | each listing returns the valid record, logs one WARNING naming the bad file and field, and counts it; `get` does not raise; the mutation answers `unparseable` and leaves the file untouched |
| 4 | β1 producer | `resolve` an L2 with two pending members, real queue dir | L2 resolved by `interactive:coordinator` | each archived member carries `cascade_parent_id` = L2 id, durably |
| 5 | β1 producer | `park` the same L2; a fresh `EscalationQueue` on the same dir | — | members still pending with `cascade_parent_id`; the field survives the new instance |
| 6 | β1 migration | archive holding a legacy member (`resolved_by='l2-cascade:esc-9-1'`) and a pending parked L2 with pending members | legacy files | after `backfill_record_facts`: fields set, `resolved_by` untouched, second run changes nothing |
| 7 | β1 consumer | an L2 resolution forwarded onto a member that a reaper sweep already dismissed | member dismissed `auto-dismissed` | the forward is logged at INFO as a cascade forward and captured in `late_resolutions`; no WARNING |
| 8 | β1 consumer | an L2 closed by a reaper sweep; its member read by `closed_by_non_sweep_resolver` and `default_resolution_class_for_resolver` | — | `closed_by_non_sweep_resolver` → True; the default resolution class is None, as for any cascade |
| 9 | β1 consumer | a legacy `l2-cascade:` member after the backfill; a member of an L2 carrying `ruled_by_human` | — | `resolver_tier(record)`, both dashboard blocks, `agreement_report` and the steward predicate all give `cascade` for both records |
| 10 | β2 producer | `resolve` an L2 with two pending members, real queue dir | L2 resolved by `interactive:coordinator` | each member: `resolved_by` = `interactive:coordinator`; no record's parsed `resolved_by` starts `l2-cascade:`; one event per record, members' `effective_action` = the L2's; `park` yields `kind='parked'` events |
| 11 | β2 consumer | harness callback on a real queue: member of a parked L2, member of a `restart`+`recut` L2 | `queue.get` patched to raise | task blocked / re-cut as today; the recut ruling is built from `event.recut`, `ruling_record_id` and `ruling_resolved_at`; the harness never reads the parent record |
| 12 | γ producer | agent `escalate_blocker(severity='critical')` with a two-token summary | — | stored `blocking`, `original_severity='critical'`, summary unchanged, folds with an identical earlier filing |
| 13 | γ producer | two infra filings, same `failure_signature`, different summaries; two legacy filings | — | first pair folds into one parent; legacy pair folds exactly as before |
| 14 | γ producer (S1 seam) | an `EscalationDraft` carrying `failure_signature`, `filing_site`, `hypothesis` and severity `critical` from a non-sentinel role, filed through S1's filer | — | the stored record carries the three facts, `severity='blocking'`, `original_severity='critical'`; S1's draft parity test covers the four fields; a contained record (S1 C2 step 5) keeps the caller's severity and leaves `original_severity` unset |
| 15 | δ producer | `resolve_issue` parametrised over `ACTION_EFFECTS` rows; an injected extra row | — | each record outcome equals the row's `record_disposition`; the injected row works without editing `resolve_issue` |
| 16 | δ producer | `park` on an info record; `duplicate_of` with `close_only` naming a pending and a terminal survivor, with `resume`, naming a missing id, naming itself; a cascade of a record carrying `duplicate_of` | — | the precondition gate's refusal names `not_info_severity`; `close_only` persists the field for both survivors; the others `invalid_duplicate_of`, record unchanged; no member inherits the field |
| 17 | δ consumer (S25) | `scripts/sitting/gates.py::classify_member` on a record with `duplicate_of` and a resolution lacking the marker | — | dedup survivor (S25 B10, asserted by 6770 after release) |
| 18 | ε producer | each of the six identity-reading tools over real HTTP: watcher headers, identity `interactive`, levels only `0,1`, none, malformed; and in-process `tool.fn()` | real server | one `CallerIdentity` per shape (`IN_PROCESS` for `tool.fn()`); `promote_to_l2`: watcher headers → promotes; `interactive` → promotes; levels-only `0,1` → `level_forbidden`; none → promotes, stamps `resolved_via='unidentified'` and answers with `identity_warning` (ε, before ο); malformed → `bad_capability_header` on every tool; an identity header with an empty value is treated as none (`UNIDENTIFIED`); a header-less resolve stamps `resolved_via='unidentified'` and carries `identity_warning`; a header-less filing succeeds with `filed_via='unidentified'` and no warning; an in-process call is neither warned nor stamped unidentified |
| 18a | ε producer | `identity_required_refusal` for each `AUTHORITY_BEARING` row with an `UNIDENTIFIED` caller; for a filing and a read; for `IN_PROCESS` | — | one refusal per authority-bearing row and none otherwise; the message equals the pinned text naming `X-Escalation-Identity`, the tool, the action, the level, `.mcp.json` / `--mcp-config` and the `"headers"` line to add |
| 18b | ε producer | `resolve_issue(resolved_by='claude-team-7174-carry')` with identity `interactive`; the same with identity `claude-task-9-steward`; `resolver_tier` of each | real server | `resolved_by='interactive'`, `resolver_detail='claude-team-7174-carry'`, tier `human`; `resolved_by='claude-task-9-steward'`, no detail, tier `steward`; `classify.py`'s human set is `authority.HUMAN_IDENTITIES` |
| 19 | ε2 producer | class `stale_task_scoped`, resolution text `status: done`: a watcher `close_only` on an L2; an executor `close_only` with the class live; a `cited_task_id` naming a real `done` task while the subject is `pending` | lookup says `pending` / `done` / raises | watcher and executor alike: forbidden / closed / forbidden; the `cited_task_id` case is forbidden |
| 20 | η consumer | the five dashboard readers over one corpus holding a record whose JSON says `"status": "superseded"` | — | the object readers (`escalation_analytics`, `escalations`, `escalation_corpus`) report it through the skip count as `unknown_escalation_status`; the raw-dict readers (`performance`, `memory_evals`) name it `unknown_escalation_status`; none counts it open or terminal; all five agree on open and terminal counts |
| 21 | θ1 consumer | legacy pending scope-divergence record after the sweep; a new workflow filing; two re-blocks with different summaries and one `failure_signature` | — | reaper defers via `filing_site`; filing carries both fields; the reblock guard sees one identity |
| 22 | ζ consumer | `DedupeConfig` default window changed in a test | — | the rendered prompt states the new window and the downgrade to `blocking` |
| 23 | ι guard | seeded modules, one per counting-rule shape (incl. the store-anchored dict read), one reading `LEVELS_HEADER` outside `parse_caller_identity`, plus a clean module | fixtures | each shape flagged with path and line; clean module and the tree pass |
| 24 | γ2 consumer | a queue holding a legacy record with only `dedupe_fingerprint=F`, a new record with only `failure_signature=F`, and one with neither; each of the five readers asked about `F` | real queue dir | both F records are found by the Memory Evals join, the archive fingerprint set (when terminal), both recon re-file checks, `has_open` and both fold keys; the third is named as having no usable identity, exactly as today |
| 24a | γ3 producer (migration) | a real queue dir and archive holding a legacy record with `"dedupe_fingerprint": "F"` and no signature, one carrying both keys with different values, and one with neither; `backfill_record_facts` run twice; then each of the five readers asked about `F` | legacy JSON files | the first record's `failure_signature` is `F`, the second keeps its signature, the third is unchanged; the second run changes nothing; every reader finds the first record; no module but the migration names `dedupe_fingerprint` |
| 25 | γ2 consumer (S1 seam) | a draft declaring only `failure_signature=F` filed through S1's filer under `OpenAtLevel(same_fingerprint=True)` while a legacy open record carries `dedupe_fingerprint=F` | — | `Suppressed` naming the legacy record |
| 26 | μ producer | `mcp_config_json(escalation_url=u, escalation_identity='claude-task-9-steward')`; the same without an identity; the steward, workflow and review-checkpoint configs; the eval stub | — | the entry carries `X-Escalation-Identity: claude-task-9-steward`; without one, `ValueError` naming the argument; each spawned config carries its agent id; the eval stub returns no escalation entry |
| 26a | μ producer | a steward, task-workflow, review-checkpoint and dry-run invocation built by the orchestrator; the invocation's environment and a worktree `.mcp.json` carrying `${ESCALATION_IDENTITY:-interactive}` | — | each invocation's environment carries `ESCALATION_IDENTITY` equal to its dispatched agent id, so either escalation entry the CLI keeps sends the agent's id, never `interactive` |
| 27 | μ consumer (live) | a dark-factory interactive session started with no `ESCALATION_IDENTITY` resolves an L0 note; the L2 watcher started by its launcher does the same | live server | the first record reads `resolved_by='interactive'`, `resolved_via='identified'`; the second `resolved_by='escalation-watcher'` |
| 28 | ο producer (filed later) | each `AUTHORITY_BEARING` tool over real HTTP with no headers, at levels 0, 1 and 2; `escalate_info` and `get_escalation` with no headers | real server | every authority-bearing call returns `identity_required` with the 18a text and the record is unchanged; the filing and the read succeed |

## Design invariant (ι performs the four-site lockstep edit)

Minted here: **`escalation-vocabulary-imported`**. The slug is the stable id; the number is the next free
one when ι lands (lead ruling S2/OQ-10, which amends R51's numbering clause; INV-14 is still the last on
main at `e59840cdaf`; wave 2's R51 order for 9140, 9242 and 9154 stands). The six wave-3 minting tasks
(S1 ω, S1's eval-hermeticity task, S2 ι, S16 ι, S19 ε and S17's `one-decision-partition-per-area`) take
INV-18 to INV-23 in landing order, the decision-partition invariant last as INV-23 (lead ruling), so ι
takes one of INV-18 to INV-22; the lead wires the invariant-file chain at decompose and corrects numbers
then if the order changes; every check anchors on the slug. Sites, enforced by
`scripts/tests/test_design_invariants_consistency.py` (`PINNED_SITES`; it requires contiguous numbering):
`docs/legibility/design-invariants.md` (section, header dating paragraph, `## Fixtures` date list),
`docs/legibility/design-invariants-fixtures.md` (`INV-N-PRD`, `INV-N-CODE`, a dated addendum walk with its
rehearsal rows and the cumulative match count), `skills/prd/references/gates.md` (the family-inventory row
and the `inv-trigger-shapes` span); `CONTRIBUTING.md` is pinned as an absence and `code_quality.md`'s
partial mapping is not extended. Draft normative text:

> **Rule**: No module outside `escalation/` compares an escalation record's status, resolve action or
> category to a string literal or re-spells a set of them; it imports the vocabulary from
> `escalation.models` and `escalation.categories`, or, where it may import only the standard library,
> the terminal status names from `shared.escalation_layout`.
>
> **Checkable design question(s)**: Does this feature read an escalation record's status, action or
> category to decide something? Does it compare against an imported member or a set from
> `escalation.models`, or does it spell the value?
>
> **Evidence**: at `e59840cdaf`, 54 comparison sites in 25 modules spelled the vocabulary (the guard's own
> counting rule) and the dashboard alone had four answers for an unknown status (`fk-a65731424439`,
> `fk-60b7c33ea0ad`, run `review-all-dark_factory-20261008`).
>
> **House pattern**: `EscalationStatus`, `TERMINAL_STATUSES`, `ResolveAction`, `EscalationCategory`,
> `shared.escalation_layout.TERMINAL_STATUS_NAMES`; the guard
> `scripts/tests/test_escalation_vocabulary_literals.py`.
>
> **Family boundary**: INV-5 governs duplicated logic in general and accepts a drift test; this invariant
> governs one vocabulary and its guard is the enforcement, not a drift test. S10's task-status guard owns
> task statuses; this owns escalation records.

Trigger shape for gates.md: *an escalation status, action or category compared to a string literal, or
a set of them re-spelled, outside `escalation/` (`escalation-vocabulary-imported`)?*

The guard is exercised against hand-written fixture modules (INV-10): one module per shape is flagged
with its path and line, a clean module passes, and the tree passes. Its rule is the counting rule of
§Background, every shape of which it checks (the Rule claims nothing the guard does not); its derived sets
come from C1 and `EscalationCategory`; it carries no exemption list (INV-12).

## Cross-PRD relationship

Edge notation: "X → Y" means X depends on Y (Y lands first). Statuses read with `get_task`/`get_statuses`
on 2026-11-13. An edge that rests only on a shared test file is not wired (lead ruling after the wave-3
fold-verifiers, restating R54); none of the edges below rests only on a test file.

| Other stream / PRD / task | Direction | Seam mechanism | Owner | Edges and status |
|---|---|---|---|---|
| S3 escalation server and queue (filed 9152–9210, all pending) | consumes | C4 (`_mutate_pending`/`_mutate_terminal`, `set_resolve_callback(…, deliver_on)`) for β1/β2; C6 (`read_caller_headers`, absorbed and deleted by ε); C8 (`RESOLVE_GATES`) for δ/ε; γ 9162's `scan.py` for α; ζ 9200 moves the edited tools whole | S3 | α → κ 9173; β1 → β 9157, γ 9162; γ → ι 9181; δ → ι 9181; S3 θ 9177 → β2 (`queue.py`, `sweep.py`; lead action); S3 ε1 9188 → ε2 (`server.py`; lead action; through ε2 also ε, δ, γ); S3 λ 9205 and ζ 9200 land after ε2 through 9188 |
| S3 `server.py` per-file chain (R54: 9181, then S10 β1 9187, S10 β2 9191, S5 λ0 9256, S7 δ 9127, S5 λ 9267, S5 ξ0 9268, then S3 ε1 9188) | same file, other regions | γ edits the escalate tools, δ `resolve_issue`'s action branch, ε the header reads and the identity refusal, ε2 the task-status lookup call; those six tasks edit `update_task` call sites, `merge_request`'s retry parameter and git-runner imports | — | lead ruling S2/OQ-12 (a): γ, δ, ε and ε2 stay off R54's `server.py` chain, between S1 β and S3 ε1 9188 — a stated, measured exception to R54 (the leaves edit disjoint regions; it also departs from S1's own chaining of its `server.py` leaves). Chaining them per R54 would make 9187's 211 open dependents (39 high, 109 wave-2; measured 2026-11-13) also wait on them; through 9177 → β2 they already wait on S2 α–β2, S19 θ0 and 8562 |
| S1 one escalation filing entry point (wave 3, folded concurrently): its α delivers `escalation/filing.py` (C1: `EscalationDraft`, `FilingPolicy`, `EscalationFiler`), the door's ordered pipeline (C2, whose step 2 writes the severity downgrade) and the `Sentinel` registry with the level-parameterised `has_open` (C3); its β routes every other escalation-package filer, `promote_to_l2`'s create step included, through the door | consumes and produces | S2 produces C1 (vocabulary) and C4 (record facts, the one failure identity) to S1; S2 γ adds four fields to S1's draft and its parity test and writes `original_severity` in C2 step 2; S2 ε adds `filed_via` to the draft and the door writes it on every record (lead alignment: the door is the submit path); S1 keeps writing `dedupe_fingerprint` until ω; S2 carries its retirement (the lead's "FINGERPRINT" ruling, D12): γ2 moves the stored-fingerprint readers, including C3's `has_open` filter, and γ3, after S1 ω, deletes C1 `EscalationDraft.dedupe_fingerprint`, C1 `OpenAtLevel.same_fingerprint`'s fingerprint key, C3 `has_open(dedupe_fingerprint=)` and the `Fingerprint` fold key's field, keying them on `failure_signature` | S1 (filing; the field until ω), S2 (vocabulary, record facts, identity, the retirement) | Sitting ruling S2-3 (a), the lead's cross-stream order: S1 α (the filing entry point) → S2 α (S1's direct edge) and → S2 β2 (implied through 9181 → 9177 → β2); S2 γ → S1 β (both edit the escalate tools in `server.py`; δ follows through γ); S2 θ1 → S1 δ (the 17 workflow filings through `TaskWorkflow._file_escalation`) and → S1 η2 (the filings that stay in `harness.py`, S1's last `harness.py` leaf); S2 ε → S1 β (the caller-identity task follows the door's other-filers task); S1 ι1 (the recon escalation registry) → S2 η (`backlog_policy.py`), recorded from S1's draft; S2 θ2 → S1 γ (typed status-write outcome, `deterministic_runner.py`), S1 κ1 (router, L0 promotion), S1 κ3 (provenance conflict and others), S1 κ4 (the steward's level-1 re-escalation); S2 κ → S1 θ2 (the fused-memory escalators that build their own queue, and the ticket janitor), S1 ι2 (the other recon filers), S1 ω (the `escalation-filing-one-door` gate); S2 γ3 (the fingerprint retirement's second half) → S2 γ2 and → S1 ω; no S1 leaf depends on γ2 or γ3 (D12, matching S1's §Cross-PRD row "one failure identity"). No S2 α/β1/β2 → S1 edge (it would close a cycle; task 9490 makes the store refuse it). The `Sentinel` registry is S1's (lead ruling S2/OQ-6) |
| S19 package layering (wave 3, folded concurrently) | consumes | S19 C9, created by θ0: `shared/escalation_layout.py`'s standard-library terminal status names and archive layout; S2 α builds `EscalationStatus` from the names, and the invariant and guard name the module | S19 | S2 α → S19 θ0 (lead ruling S19/Q-C8); S2 η and κ edit `reconciliation/backlog_policy.py` and `standing_decision_constants.py` at the paths S19 β1 moves them to, and κ edits the escalators at S19 γ's `fused_memory/escalators/` paths (κ declares what it finds at dispatch); S2 η, κ → S19 β1 and S2 κ → S19 γ as S19's draft records |
| S17 scheduler seams | cites | Leo's design principle enters `docs/legibility/design-invariants.md` through S17's invariant task as `one-decision-partition-per-area`, minted last of the six wave-3 invariants as INV-23 (lead ruling); D10 applies it here by slug | S17 | xref only; no edge (S2 builds no check on the invariant) |
| S16 config shape | consumes; shares a shape | the operator-sweep gate (ν) uses the same shape as S16's yaml sweep gate σ (cross-stream consequence) and the same spawn mechanism: S16 χ teaches the escalation watcher to spawn a gate's session keyed on `metadata.x_preauthorised_spawn`, which ν carries; the six wave-3 invariant leaves share the three invariant files | S16 (χ) | ν → S16 χ (lead ruling 2026-11-13: S2's gate depends on S16's mechanism task, which has no prerequisite inside S16); invariant order per lead ruling S2/OQ-10 and the lead's INV-18–INV-23 ruling |
| S6 merge-lane lifecycle (filed) | adjacent | `harness.py` order | S6 | S6 β1 9144 → β2 (lead ruling S2/OQ-5, conditional on S2-3 (a), which Leo ruled): β2 rewrites `_on_escalation_resolved` after S3 κ and before S6's harness chain. 9144 already reaches 9177 transitively, so the edge states the `harness.py` order and adds no new wait (9144 has 183 open dependents, 35 high, 95 wave-2, gate 6698; measured 2026-11-13). S6 θ 9176 (high), which releases halt holders inside `_on_escalation_resolved`, follows β2 through 9144; lead action: amend 9176 |
| S8 harness composition root (filed) | adjacent | S8 α 9216 (`SlotRegistry`) reads the registry in `_on_escalation_resolved`; S8 ζ1a/ζ1b/ζ2 filer modules repoint onto C1; the resolution appliers stay in `harness.py` | S8 | 9216 follows β2 through 9144; lead action: amend 9216. θ1 → S8 ω 9236 (`harness.py` tail) and S8 κ1 9232 (`workflow.py` tail) (lead ruling S2/OQ-11 (a)) |
| S9 workflow spine (filed) | adjacent | `BlockReason.suggested_action` stays `str` (S9 D5): S2 replaces the one routing use of `suggested_action` with `filing_site`, and types no hint vocabulary; S9's `BlockReason.dedupe_fingerprint` can carry a `failure_signature` later | S9 | xref; S9's `workflow.py` leaves precede S8 κ1 9232, so θ1 → 9232 orders θ1 after them |
| S10 task-write contract (filed) | boundary and same files | S10 ο 9285's guard flags a dict-form status compared to a `TaskStatus` value in the fused-memory tree, which would flag `backlog_policy.py::BacklogPolicy._close_pending_halt_records` (an escalation record); η converts it to `EscalationStatus.PENDING`. Same files: `scripts/sitting/payloads.py` (S10 α 9180), `steward.py`, `deterministic_runner.py` (S10 β2 9191), `provenance_conflict.py` (S10 ξ 9283) | S10 | 9285 → η (lead action); δ → 9180; θ2 → 9191, 9283 |
| S10/S9/S8/S5 orchestrator tails on θ2's files | same files | `steward.py` (S9 ξ 9287 last of 9190, 9202, 9229, 9255, 9287), `digest.py` (S8 ι2 9231), `offline_lane_episodes.py` and `agents/roles.py` (S5 ο0 9270, a codemod) | — | θ2 → 9287, 9231, 9270; ζ → 9270 (lead ruling S2/OQ-11 (a); an import-only codemod, so the lead may drop ζ → 9270 at decompose, S19/Q-C6); κ follows θ1 and θ2 |
| μ's one-line call-site edits and skill texts | same files, other regions | μ adds one keyword argument and one environment entry at the `mcp_config_json` call sites and invocations (`steward.py`, `workflow.py`, `review_checkpoint.py`, `dry_run_unblock.py`) and edits the identity/launch passages of the two watcher skills, files whose wave-2 tails wait on 5040 (`steward.py` 9287, `workflow.py` 9232, `review_checkpoint.py` 9202/9218, `skills/escalation-watcher/SKILL.md` and `dry_run_unblock.py` 9288; measured 2026-11-13) | — | a stated, measured exception to R54, CONFIRMED by the lead (2026-11-13, "the configuration-deploy task's one-line edits … take NO per-file ordering edge"): μ takes no edge onto those tails; the file locks serialize it and the later task rebases one line. The lead also ruled (2026-11-13) that the exception covers μ's one-line edits in `evals/runner.py` and `workflow.py` against S1's eval-hermeticity tasks λ1 and λ2 and S1's corpus-sandbox task μ: no cross-plan edge |
| S22 dashboard readers (wave 3) | produces (to) | C1 and the one unknown-status policy; S22 restructures `escalation_analytics.py`, `performance.py`, `escalations.py`, `escalation_corpus.py` | S2 | S2 η → S22 θ (dashboard event readers on the read model), S22 η (Performance tallies), S22 γ2 (the escalation-URL probe), S22 δ2 (the app-scoped caches; through it S22 δ1, the TTLCache module): S22's structural edits land first and S2's operand swaps rebase onto them, as S22's draft recommends; the lead wires it at decompose |
| S25 scripts instruments (filed, wave 1) | produces (to) | C5 `duplicate_of` for ξ 6770 (held, `deferred`); ι repoints the scripts sites | S2 | 6770 → δ (release); ι → 6770 (`sitting/gates.py`, `member-chain-sweep.py`), which holds the signal leaf behind 6770 → 6760 → 6759 (6753 is done), S25's scripts chain (none of them waits on 4567, 5040 or 8528). 6770 also edits `skills/escalation-watcher/SKILL.md` (its dedup and pin steps), as μ and β2 do in other passages: no edge under μ's exception, which the lead confirmed |
| Other repositories (operator sweep) | produces (to) | the eleven other projects' `.mcp.json` escalation entries (§Background table) and the twelve side checkouts' copies, changed by ν's operator session (configuration only) | S2 (ν) | ν → ε, μ, S16 χ |
| reify's own store: `reify/scripts/docs-truth-sweep.py` (header-less `promote_to_l2`) | consumes | a reify task, filed by the lead in reify's store at decompose, gives the script an identity header (lead ruling 2026-11-13: no dark-factory task edits it) | lead (reify task) | ν → `"reify:<id>"` (cross-project dependency), so ξ and ο, which follow ν, come after the script sends an identity |
| Decision-stream follow-ups 8477, 8478 (pending, free; 8477 is an open design question); 8475 is done (`19d62b7687`) | same file | `authority.py::POLICY_TABLE` rows and the shadow codec; ε2 adds the `corroborate` column every row carries | each | 8477, 8478 → ε2 (lead actions): follow-ups that rebase onto ε2's `corroborate` column; nothing waits on them. 8475's edge is dropped (done) |
| Task 5258 (pending, low; unify the markup filers) | same file | `authority.py`; S1 amends it to build drafts and edges it after S1 β | 5258 | 5258 → ε2 (lead action) |
| Task 8562 instruments-honesty δ (pending) | consumes | `resolver_tier(record)`, `ruled_by_human`, the L2 `resolution_class` requirement; β1 sets `cascade` before `human-ruled` in `resolver_tier` (lead ruling S2/OQ-8) | 8562 | α → 8562 |
| Task 9490 (pending, high; the store refuses a cycle-closing dependency edge) | relies on | `DependencyCycle` refusal in `add_dependency` and `update_task(dependencies=…)` | lead (outside every plan) | xref only; once it lands it, not a pre-pass, is the guard against the S1/S2 cycle this plan's ordering avoids |
| Task 8777 (pending, low) | subsumed | `classify._CASCADE_PREFIX` | β1 | cancel, reason names β1 |
| Task 8784 (pending, low) | subsumed | `Harness._resolve_escalation_action(escalation, parent=None)` | β2 | cancel, reason names β2 |
| Task 6348 (pending, low) | adjacent | `DedupeConfig.for_content_fingerprint` adoption, `dedupe.py` | S1 | 6348 cancelled into S1 θ2; κ → S1 θ2 carries it |
| Tasks 7639, 7392 (pending, low) | adjacent | `roles.py` escalation constants extraction; a ladder-prompt sentence | each | 7639 → ζ, 7392 → ζ; under lead ruling S2/OQ-11 (a) both then wait on task 5040 with ζ |
| Tasks 4727, 4749 (deferred) | covered | category REFACTOR TRIGGER; `already_landed` registration — both done by 5257 | 5257 | cancel as covered (not S2 leaves) |
| Task 6390 (done, `bbaa5f91e4`) | landed | `Escalation.review_due: ReviewDueState` | — | the draft's 6390 → α edge is dropped; α types around the landed payload |
| Task 5040 / chain 8525–8528 (8525 done) | context | κ declares `merge_lane/worker.py`, so it is parked behind 8528 at filing (R55); θ1, θ2 and ζ wait on 5040 through their end-of-chain edges (lead ruling S2/OQ-11 (a)) | — | κ → 8528 |
| Wave-3 invariant leaves (S1 ω `escalation-filing-one-door`, S16 ι `config-default-one-home`, S17's `one-decision-partition-per-area`, S19 ε `fused-memory-subpackage-dag`, S1's eval-hermeticity invariant) and R51's chain ending at S7 ζ 9154 | same docs | the three invariant files | lead | lead ruling S2/OQ-10 (b): no edge onto 9154 for a number; the six wave-3 minting tasks take INV-18 to INV-23 in landing order, `one-decision-partition-per-area` last as INV-23 (lead ruling); S2 ι → S1 ω as both drafts have it, and the lead wires the rest of the invariant-file chain at decompose. S2 ι also reaches 9154 through S1 ε1's file edge (via S1 ω or θ1 → S1 η2 → S1 ε3), so it waits on 9154 for a file, never for a number; that costs nothing under lead ruling S2/OQ-11 (a) |

## Decomposition plan

Overlay bands: α, β1, γ, γ2, δ, ε, ε2, ζ, η, θ1, θ2, ι and μ are within 300–1,500 LOC (μ and γ2 near the
floor, ε near the top) and ≤ 13 declared files; ν is a human gate with no files. Three leaves cross the
15-file review trigger, each a mechanical migration examined and kept whole: β2 (33 files: 3 production
modules, 2 new test seams, the L2 watcher's skill text, 27 test files whose edits are a one-call
substitution onto the seams or a `resolved_by` assertion; splitting it would land the callback's
producer and consumer in different commits), κ (36 files: a one-token rename; splitting it would
stitch one rename across leaves) and γ3 (about 66 files: one field deleted and its value renamed at every
writer; the retirement is two tasks, γ2 and γ3, for size, as the lead ruled). Same-file leaves are serialized by real edges, in depends-on form:
`models.py` γ3 → γ2 → ε → δ → γ → S1 α → β2 → β1 → α; `queue.py` κ → γ3 → γ2 → ε → δ → S1 α → β2 → β1 → α (δ reaches
S1 α through γ); `server.py` κ → ε2 → ε → δ → γ → {9181, S1 β}, and 9188 → ε2; `filing.py` γ3 → γ2 → ε → γ → S1 β;
`sweep.py` γ3 → γ2 → γ → β2 → β1; `classify.py` θ1 → ε → β2 → β1 → α (ε reaches β2 through δ and γ); `dedupe.py`
κ → γ3 → γ2 → γ → S1 α; `roles.py` κ → ζ → 9270; `ladder_gates.py` ε → δ; `authority.py` 8477, 8478, 5258 →
ε2 → ε → δ; `policy_gate.py` ε2 → ε; `escalation_analytics.py` η → β1 → α; `memory_evals.py` and
`escalation_archive.py` γ3 → γ2 → η; fused-memory `reconciliation/harness.py` κ → γ3 → γ2 → 9254 (the end
of the wave-2 chain; γ2's order against S1 ι1/ι2 and S12's tasks on that file is the lead's at decompose;
γ3 follows S1 ι2 through S1 ω); `harness.py` 9144 → β2 and θ1 → {β2, 9236, S1 η2},
κ → θ1; `workflow.py` θ1 → {9232, S1 δ, μ}, κ → γ3 → θ1; `steward.py` θ2 → {9287, S1 κ4, μ};
`skills/escalation-watcher/SKILL.md` μ → β2 (through ε); `provenance_conflict.py` κ → γ3 → θ2 → {9283, S1
κ3}; `merge_lane/worker.py` κ → γ3 → {8528, S1 ω}; `deterministic_runner.py` θ2 → {9191, S1 γ}; `escalation_router.py`/`l0_promotion.py` θ2 → S1 κ1;
`backlog_policy.py` κ → γ3 → η (through γ2). μ's one-line edits in `steward.py`, `workflow.py`,
`review_checkpoint.py`, `dry_run_unblock.py`, `mcp_lifecycle.py`, `agents/briefing.py`, `evals/runner.py`
and the two watcher skills take no edge onto those files' wave-2 tails (a stated, measured exception to
R54, confirmed by the lead on 2026-11-13; §Cross-PRD). No
edge here rests only on a shared test file. `high` leaves (δ, ε, ε2, μ, ν) become dependencies of gate
6698. Signal leaf: ι.

**α — Typed escalation vocabulary: status, severity and resolve-action enums parsed at construction**
(`task_kind=normal`, priority medium, est. ~700 LOC)
- Files: escalation/src/escalation/models.py, escalation/src/escalation/action_effects.py,
  escalation/src/escalation/classify.py, escalation/src/escalation/queue.py,
  escalation/src/escalation/scan.py (created by S3 γ 9162), escalation/tests/test_models.py,
  escalation/tests/test_action_effects.py, escalation/tests/test_classify.py,
  escalation/tests/test_status_authority_gate.py, escalation/tests/test_sweep.py,
  escalation/tests/test_queue.py
- Keys: `fk-bbb0a7b3c8a8`, `fk-479375564c2f` (evidence half).
- Deps: S3 κ 9173 (`models.py`; through it S3 γ 9162's `scan.py`), 8562 (`models.py`, `classify.py`),
  S19 θ0 (`shared/escalation_layout.py`'s terminal status names, S19 C9; lead ruling S19/Q-C8). Task
  6390 has landed (`bbaa5f91e4`); α types around `Escalation.review_due`.
- Work: C1 (`EscalationStatus`'s terminal values from `shared.escalation_layout`), including the reader-side skip (D3: `UnknownEscalationValue` in `_SCAN_PARSE_ERRORS` and
  `EscalationQueue.get`'s tuple); C2's columns (`record_disposition`, `preconditions`, `RESOLVE_ACTIONS`,
  `default_action`); `effective_benign` reads the sets; `test_sweep.py::…::test_unknown_status_value_skipped`
  writes the raw JSON instead of constructing `Escalation(status='weird')` (the only one of 595
  `Escalation(` calls, AST count measured at `e59840cdaf` (581 at `ba17934861`), whose literal α rejects). Deletes: `KNOWN_SEVERITIES`/`SEVERITY_RANK`
  as hand-kept literals (both derive from `Severity`).
- Signal: boundary rows 1–3; `uv run --project escalation pytest escalation/tests/test_models.py` green;
  the live dark-factory orchestrator and fused-memory read every pending and archived record after
  restart (no `UnknownEscalationValue` WARNING in either journal).
- G7: INV-2 (the error names field, value, legal set, id), INV-10 (fixture records from every stored
  shape), INV-11 (an unparseable record is counted and logged, never silent, never a crash).

**β1 — Persist the cascade parent on escalation records; every resolver-tier reader reads it**
(`task_kind=normal`, priority medium, est. ~700 LOC)
- Files: escalation/src/escalation/models.py, escalation/src/escalation/queue.py,
  escalation/src/escalation/sweep.py, escalation/src/escalation/classify.py,
  escalation/src/escalation/shadow_ruling.py, dashboard/src/dashboard/data/escalation_analytics.py,
  escalation/tests/test_cascade_attribution.py, escalation/tests/test_queue.py,
  escalation/tests/test_sweep.py, escalation/tests/test_classify.py,
  escalation/tests/test_shadow_ruling.py, dashboard/tests/test_escalation_analytics.py,
  orchestrator/tests/test_steward.py
- Keys: `fk-8c48269c7363` (field, readers), `fk-5332d25b1702` (cascade half, field).
- Deps: α; S3 β 9157 (`queue.py`, C4); S3 γ 9162 (`queue.py`, `shadow_ruling.py`). Cancels 8777.
- Work: C3's β1 part — the field written by `resolve` and `park` (beside today's attribution, so the
  harness is untouched), `backfill_record_facts`, and the eight resolver-tier call sites moved to the
  record (explicit `via_cascade` in the late-resolution capture; `closed_by_non_sweep_resolver` and
  `default_resolution_class_for_resolver` count `cascade_parent_id` as cascade; the dashboard blocks and
  `agreement_report` call `resolver_tier(record)` with `cascade` before `human-ruled`, lead ruling S2/OQ-8).
  Deletes: `classify._CASCADE_PREFIX`; the string-tier calls outside `classify.py`.
- Signal: boundary rows 4–9; after the dark-factory orchestrator and fused-memory restart on β1, every
  record whose parsed `resolved_by` starts with `l2-cascade:` carries `cascade_parent_id` (read by
  parsing each file, not by grep: `esc-8422-3` carries the string in `detail` only), and the Escalation
  Analytics resolver-tier chart's cascade count is unchanged across the restart.
- G7: INV-13 (the backfill decision per record kind; the reader proves its producer — the live sweep),
  INV-11 (a migration failure is counted and logged per record, never silent), INV-5 (one tier rule).

**β2 — The resolve callback receives a frozen event; cascade attribution leaves `resolved_by`**
(`task_kind=normal`, priority medium, est. ~1,100 LOC, most of it mechanical test migration)
- Files: escalation/src/escalation/models.py, escalation/src/escalation/queue.py,
  orchestrator/src/orchestrator/harness.py, escalation/tests/_resolution_events.py (new),
  orchestrator/tests/_escalation_events.py (new), escalation/tests/test_queue.py,
  escalation/tests/test_cascade_attribution.py, escalation/tests/test_capability_guard_http.py,
  escalation/tests/test_resolve_resolution_action.py, escalation/tests/test_server.py,
  escalation/tests/test_category_contract.py, escalation/tests/test_note_to_file.py,
  escalation/tests/test_resolve_issue_park_on_info.py, escalation/tests/test_resolve_issue_recut.py,
  escalation/tests/test_server_chokepoint.py, escalation/tests/test_server_eval_lane.py,
  dashboard/tests/test_escalation_lifecycle_gate.py, orchestrator/tests/test_cascade_unblock.py,
  orchestrator/tests/test_harness_action_dispatch.py, orchestrator/tests/test_harness_infra_hold_repend.py,
  orchestrator/tests/test_harness_infra_resume_truthful.py, orchestrator/tests/test_harness_park_stop.py,
  orchestrator/tests/test_harness_pause_visibility.py, orchestrator/tests/test_harness_reblock_signature_guard.py,
  orchestrator/tests/test_harness_resume_scheduler.py, orchestrator/tests/test_repend_state_machine.py,
  orchestrator/tests/test_resume_claimant_liveness.py, orchestrator/tests/test_status_authority_gate.py,
  orchestrator/tests/test_stranded_blocked_sweep.py, orchestrator/tests/test_workflow_escalated_steward_stall.py,
  orchestrator/tests/test_workflow_merge_gating_strand.py,
  orchestrator/tests/test_harness_escalation_callbacks.py (created by S3 β 9157),
  skills/escalation-watcher/SKILL.md (the cascade row and paragraph)
- Keys: `fk-8c48269c7363` (closes), `fk-5332d25b1702` (cascade half, closes).
- Deps: β1. Edges onto filed tasks: S3 θ 9177 → β2 (sitting ruling S2-3 (a)); S6 β1 9144 → β2 (lead
  ruling S2/OQ-5). Cancels 8784.
- Work: C3's β2 part — `ResolutionEvent` with the recut facts; the queue builds and delivers it;
  `set_resolve_callback`'s type; a cascade member's `resolved_by` is the parent's resolver;
  `_on_escalation_resolved(event)`; the two public test seams; the L2 watcher skill's cascade passages
  name `cascade_parent_id` and the event. The test list was measured at
  `ba17934861` (direct callers, record-reading registered callbacks, `resolved_by` prefix assertions);
  re-measure at dispatch and declare any new file. Deletes: `cascade_resolved_by`,
  `cascade_parent_id(resolved_by)`, `_L2_CASCADE_RESOLVER_PREFIX`, park's in-memory stamp,
  `Harness._cascade_parent`, `Harness._resolve_escalation_action` and its legacy map,
  `Harness._restart_recut_ruling`'s `parent` parameter, the harness's two `ResolveAction` literal
  comparisons, and the 102 direct `_on_escalation_resolved(record)` test calls (onto
  `_escalation_events.deliver`).
- Signal: boundary rows 10–11; after the dark-factory orchestrator and fused-memory restart on β2, a
  live L2 resolution's archived members carry the resolver's name in `resolved_by` and the parent in
  `cascade_parent_id`, and the task behind each member takes the L2's action.
- G7: INV-3 (the parent is read under the lock, never re-fetched by the harness), INV-7 (the event is a
  value; the harness holds no queue reference to act on it), tests-stance (the private callback calls
  move onto two named public seams).

**γ — Filer-declared escalation facts: filing site, failure signature, original severity, hypothesis**
(`task_kind=normal`, priority medium, est. ~750 LOC)
- Files: escalation/src/escalation/models.py, escalation/src/escalation/dedupe.py,
  escalation/src/escalation/server.py, escalation/src/escalation/sweep.py,
  escalation/src/escalation/filing.py (created by S1 α), escalation/tests/test_models.py,
  escalation/tests/test_dedupe.py, escalation/tests/test_server.py,
  escalation/tests/test_server_chokepoint.py, escalation/tests/test_sweep.py,
  escalation/tests/test_filing.py (created by S1 α)
- Keys: `fk-638cca2f65bb`, `fk-479375564c2f` (hypothesis half), `fk-5332d25b1702` (fields).
- Deps: β2 (`models.py`, `sweep.py`); S3 ι 9181 (`server.py`); S1 β (every other escalation-package
  filer through the entry point; both edit the escalate tools in `server.py`; through it S1 α, which
  creates `escalation/filing.py`) (sitting ruling S2-3 (a)).
- Work: C4, including `declared_failure_identity` and `failure_identity` (lead ruling S2/OQ-3);
  `EscalationDraft` gains the four fields and S1's parity test covers them;
  `escalate_info`/`escalate_blocker` take `hypothesis` and `failure_signature`; the evidence `cast` ×2 go;
  writes `original_severity` in S1 C2 step 2 (the downgrade); a contained record (S1 C2 step 5) keeps the
  caller's severity and leaves `original_severity` unset; `backfill_record_facts` stamps `filing_site` on
  legacy pending records. Deletes: the `[downgraded:…]` suffix and its two explanatory comments.
- Signal: boundary rows 12–14; a live `escalate_info(hypothesis=…)` on dark-factory returns a record whose
  `hypothesis` is a field and whose summary has no `Hypothesis:` line.
- G7: INV-2, INV-11 (an unknown `failure_signature` is not an error: the field is optional by contract).

**γ2 — The five readers of the stored dedupe fingerprint read the one declared failure identity**
(`task_kind=normal`, priority medium, est. ~350 LOC)
- Files: escalation/src/escalation/dedupe.py, escalation/src/escalation/queue.py,
  dashboard/src/dashboard/data/memory_evals.py,
  fused-memory/src/fused_memory/reconciliation/escalation_archive.py,
  fused-memory/src/fused_memory/reconciliation/harness.py, escalation/tests/test_dedupe.py,
  escalation/tests/test_queue.py, dashboard/tests/test_memory_evals_data.py,
  fused-memory/tests/reconciliation/test_escalation_archive.py, fused-memory/tests/test_harness.py
- Keys: `fk-638cca2f65bb` (readers half; γ carries the record half).
- Deps: γ (`dedupe.py`; `declared_failure_identity` and `failure_signature`); ε (`queue.py`); η
  (`memory_evals.py`, `escalation_archive.py`); task 9254 (S10 λ, the end of the wave-2 chain on
  fused-memory `reconciliation/harness.py`). γ3 (the second half of the retirement, D12) depends on γ2;
  the lead orders γ2 against S1 ι1/ι2 and S12's tasks on the recon harness at decompose (γ2 changes no
  write, so either order is safe).
- Work: the Memory Evals join, the archive fingerprint set, the recon harness's pending-fingerprint set
  and its per-finding fallback, `EscalationQueue.has_open`'s filter (S1 C3; the draft side reads the same
  function) and `dedupe.py::content_fingerprint_key`/`gate_backlog_fingerprint_key`'s fast path read
  `declared_failure_identity`. Memory Evals keeps naming a record with no usable identity. Deletes: every
  direct `.dedupe_fingerprint` comparison in the three consumer modules.
- Signal: boundary rows 24–25; the live Memory Evals panel links the same open `eval_regression`
  escalations before and after the restart on γ2.
- G7: INV-5 (one reader for one fact), INV-13 (each reader is shown matching a record from each producer
  shape), INV-11 (a record with no identity is named, never dropped).

**γ3 — Retire the stored dedupe fingerprint: every filer declares the failure signature and the field is deleted**
(`task_kind=normal`, priority low, est. ~1,000 LOC, mechanical)
- Files: escalation/src/escalation/models.py,
  escalation/src/escalation/filing.py (created by S1 α), escalation/src/escalation/dedupe.py,
  escalation/src/escalation/queue.py, escalation/src/escalation/sweep.py,
  dashboard/src/dashboard/data/memory_evals.py, orchestrator/src/orchestrator/workflow.py,
  orchestrator/src/orchestrator/merge_lane/worker.py, orchestrator/src/orchestrator/merge_lane/types.py,
  orchestrator/src/orchestrator/provenance_conflict.py,
  fused-memory/src/fused_memory/reconciliation/harness.py,
  fused-memory/src/fused_memory/reconciliation/backlog_policy.py,
  fused-memory/src/fused_memory/reconciliation/consolidation_auto_executor.py,
  fused-memory/src/fused_memory/reconciliation/escalation_archive.py,
  fused-memory/src/fused_memory/reconciliation/finding_task_escalation.py,
  fused-memory/src/fused_memory/reconciliation/stage1_stall_detector.py,
  fused-memory/src/fused_memory/middleware/curator_escalator.py,
  fused-memory/src/fused_memory/middleware/dead_letter_escalator.py,
  fused-memory/src/fused_memory/middleware/entity_mint_storm_escalator.py,
  fused-memory/src/fused_memory/middleware/mem0_update_storm_escalator.py,
  fused-memory/src/fused_memory/middleware/scope_violation_escalator.py,
  fused-memory/scripts/backfill_gate_backlog_summaries.py, fused-memory/scripts/backfill_recon_escalations.py,
  fused-memory/scripts/memory_eval_runner.py, OPERATIONS.md, skills/recon-escalation-watcher/SKILL.md, and
  the 40 test files that name the field (dashboard 6, escalation 5, fused-memory 18, orchestrator 11).
  Measured at `fceb951e7e`; re-measure `git grep -l dedupe_fingerprint` at dispatch and declare what it
  finds, including the paths S19 β1 and γ moved (three of the escalators move to `fused_memory/escalators/`).
- Keys: `fk-638cca2f65bb` (closes; γ and γ2 carry the earlier halves).
- Deps: γ2 (and through it γ, the failure-signature task); S1 ω (S1's closing task: every filer builds
  a draft; lead "FINGERPRINT" ruling); θ1 (`workflow.py`), θ2 (`provenance_conflict.py`), 8528 (it declares
  `merge_lane/worker.py`; parked behind 8528 at filing, R55). No S1 leaf depends on γ3.
- Work: C4's γ3 lines. First `backfill_record_facts` copies a raw `dedupe_fingerprint` into
  `failure_signature` on every pending and archived record lacking one (the deletion would otherwise drop
  it through `Escalation.from_dict`'s field filter); then every `dedupe_fingerprint=X` becomes
  `failure_signature=X`; `Escalation.dedupe_fingerprint`, `EscalationDraft.dedupe_fingerprint` and the
  fingerprint arguments of S1's `OpenAtLevel`, `has_open` and `Fingerprint` go, keyed on
  `failure_signature`. Deletes: every `dedupe_fingerprint` outside the migration.
- Signal: boundary row 24a; `git grep -c dedupe_fingerprint -- '*.py'` names only `sweep.py`'s migration;
  after the dark-factory orchestrator and fused-memory restart on γ3, the live Memory Evals panel links the
  same open `eval_regression` escalations as before.
- Size: crosses the 15-file review trigger like κ: a one-field rename whose files cannot be split without
  stitching one deletion across leaves (the lead ruled the retirement two tasks, γ2 and γ3, for size;
  γ3 is the deletion half and stays whole).
- G7: INV-13 (the migration runs before any reader loses the old key), INV-5, INV-11.

**δ — resolve_issue branches on ACTION_EFFECTS record disposition; persisted duplicate_of (C5)**
(`task_kind=normal`, priority **high**, est. ~700 LOC)
- Files: escalation/src/escalation/server.py, escalation/src/escalation/ladder_gates.py (created by
  S3 ι 9181), escalation/src/escalation/action_effects.py, escalation/src/escalation/queue.py,
  escalation/src/escalation/models.py, escalation/src/escalation/authority.py,
  escalation/src/escalation/shadow_ruling.py, scripts/sitting/payloads.py,
  escalation/tests/test_action_effects.py, escalation/tests/test_resolve_resolution_action.py,
  escalation/tests/test_ladder_gates.py (created by S3 ι 9181), escalation/tests/test_duplicate_of.py (new),
  scripts/tests/test_sitting_payloads.py
- Keys: `fk-e3b7459c681d`, `fk-d0e1094c99cb`.
- Deps: γ (`server.py`, `models.py`, `queue.py`); S3 ι 9181 (`ladder_gates.py`); S10 α 9180
  (`scripts/sitting/payloads.py`, end of that file's chain). Release: 6770 → δ.
- Work: the disposition branch; the precondition gate; C5 (survivor of any status; never inherited by a
  cascade member); `authority.py`'s module-docstring pointer at the old convention is rewritten to name
  `ACTION_EFFECTS`. Deletes: `_DISMISS_ACTIONS`, `server.RESOLVE_ACTIONS`,
  `scripts/sitting/payloads.py::RESOLVE_ACTIONS`, `_park_on_info_refusal` as an inline step, the
  `if action == 'park'` branch.
- Signal: boundary rows 15–17; `mcp__escalation__resolve_issue(action='close_only', duplicate_of=…)` on a
  live dark-factory record persists the field; task 6770's B10 passes once released.
- G7: INV-1 (one table both sides read), INV-5 (the sitting imports), INV-2 (refusals name the
  precondition).

**ε — One parse of escalation caller headers for every ladder tool, the channel stamped, and a self-explaining identity refusal (no denial yet)**
(`task_kind=normal`, priority **high**, est. ~1,200 LOC)
- Files: escalation/src/escalation/models.py, escalation/src/escalation/authority.py,
  escalation/src/escalation/policy_gate.py, escalation/src/escalation/server.py,
  escalation/src/escalation/ladder_gates.py (created by S3 ι 9181), escalation/src/escalation/queue.py
  (`resolve`/`park` take the channel), escalation/src/escalation/filing.py (created by S1 α; the draft
  carries `filed_via` and the door writes it), escalation/src/escalation/classify.py (the human tier
  reads `HUMAN_IDENTITIES`), scripts/check_unidentified_escalation_actions.py (new),
  escalation/tests/test_caller_identity.py (new), escalation/tests/test_authority.py,
  escalation/tests/test_capability_guard_http.py, scripts/tests/test_check_unidentified_escalation_actions.py (new)
- Keys: `fk-755f786c32be`. `fk-ece88b7661b7` is named in the description, not stamped: ο closes it.
- Deps: δ (`server.py`, `ladder_gates.py`, `models.py`, `authority.py`, `queue.py`; through γ,
  `filing.py`); S3 κ 9173 (C6, through α); S1 β (every other escalation-package filer, incl.
  `promote_to_l2`'s create step; the caller-identity task follows the door's other-filers task, lead
  alignment). Edges onto filed tasks: none directly; 9188, 5258, 8477 and 8478 follow ε through ε2.
- Work: C6 except `mcp_config_json`. `parse_caller_identity` and `effective_levels` for the five
  header-reading tools and `declare_pin`; `CallerChannel` including `IN_PROCESS` for a call outside an HTTP
  request; `AUTHORITY_BEARING`; `identity_required_refusal` and its message, returned under
  `identity_warning` on a successful `UNIDENTIFIED` authority-bearing call, with one WARNING line per call,
  and not yet as an error; `filed_via` written by the door on every record and `resolved_via` stamped at
  resolve, park and promote; `HUMAN_IDENTITIES` defined once and read by `classify.py`'s human tier and
  `PROMOTE_ALLOWED`; `resolver_detail` kept for a human identity (D11, confirmed by the lead);
  `promote_to_l2` identity-first (lead ruling S2/OQ-9 as amended by S2-2); the check script
  `scripts/check_unidentified_escalation_actions.py` (args `--window-secs` and repeatable `--queue-dir`,
  the queues from `shared/src/shared/project_layout.py::known_project_roots` plus each `--queue-dir`; exit 0 when no record's `resolved_via` is
  `unidentified` inside the window, 1 with a per-queue listing otherwise; read-only). Deletes:
  `read_caller_headers` and `CallerHeaders` (S3 κ's; their five call sites become
  `parse_caller_identity`), the inline `_parse_levels` use, `classify.py`'s own copy of the human set.
- Signal: boundary rows 18, 18a and 18b; on dark-factory, a header-less HTTP `resolve_issue` on an L0 note
  succeeds, answers with the pinned `identity_warning` text and archives the record with
  `resolved_via: "unidentified"`, while the auto-watcher's own L1→L2 promotion still succeeds; the check
  script run by hand over dark-factory's queue lists that record.
- G7: INV-2 (the refusal names header, tool, action, level, config file and fix), INV-11 (an unidentified
  authority-bearing act is stamped and logged, never silent), INV-5 (one human set), INV-13 (the check
  script reads what the stamp writes).

**ε2 — Corroborated L2 auto-close: the task store decides whether a stale-task L2 may be closed automatically**
(`task_kind=normal`, priority **high**, est. ~450 LOC)
- Files: escalation/src/escalation/authority.py, escalation/src/escalation/policy_gate.py,
  escalation/src/escalation/server.py, skills/escalation-watcher-auto/SKILL.md,
  escalation/tests/test_corroborated_auto_close.py (new), escalation/tests/test_authority.py
- Keys: `fk-48a76ee9f746`.
- Deps: ε (`authority.py`, `policy_gate.py`, `server.py`). Edges onto filed tasks (lead actions): S3 ε1
  9188 → ε2 (`server.py`); 5258 → ε2; 8477, 8478 → ε2 (`authority.py`).
- Work: C7: `PolicyRow.corroborate`, evaluated by `l2_auto_close_class` and `executor_refusal`, set on
  `stale_task_scoped` to "subject task `done` or `cancelled` per the lookup" through
  `anyio.from_thread.run` with a bounded timeout, unavailable → refuse (sitting ruling S2-1 (a′)); the
  watcher-auto skill may pass `cited_task_id` as an audit annotation; the two uncorroborable rows keep
  presence-only evidence under the decision-stream audit (D10). Deletes: the `stale_task_scoped` row's
  authority over text on both doors.
- Signal: boundary row 19; on dark-factory, a watcher auto-close of an L2 whose subject task is `pending`
  is refused although its text says `status: done`.
- G7: INV-3 (the row acts on a corroborated read), INV-11 (lookup failure is a refusal, not a pass),
  `corroborate-before-acting`.

**ζ — The escalation ladder role prompt renders dedupe and downgrade facts from escalation code**
(`task_kind=normal`, priority medium, est. ~250 LOC)
- Files: orchestrator/src/orchestrator/agents/roles.py, orchestrator/tests/test_roles_escalation_ladder.py
- Keys: `fk-89388166d3ca`.
- Deps: γ (the fields the prompt names); S5 ο0 9270 (`roles.py`, a codemod; end of that file's wave-2
  chain; lead ruling S2/OQ-11 (a)). κ follows ζ and renames the `DedupeConfig` attributes the prompt reads. Edges onto
  existing tasks: 7639 → ζ, 7392 → ζ. If 7639 is cancelled into 6874 before ζ dispatches, ζ edits the
  module that then holds `ESCALATION_LADDER_CORE` (re-measure at dispatch).
- Signal: boundary row 22; the rendered architect/implementer/steward prompts no longer say "bypasses
  dedupe entirely".
- G7: INV-9 (the prompt points at the one home by rendering it), INV-10.

**η — Dashboard and fused-memory readers compare escalation status through the shared vocabulary**
(`task_kind=normal`, priority medium, est. ~500 LOC)
- Files: dashboard/src/dashboard/data/escalation_analytics.py, dashboard/src/dashboard/data/performance.py,
  dashboard/src/dashboard/data/escalations.py, dashboard/src/dashboard/data/escalation_corpus.py,
  dashboard/src/dashboard/data/memory_evals.py,
  fused-memory/src/fused_memory/reconciliation/backlog_policy.py,
  fused-memory/src/fused_memory/reconciliation/escalation_archive.py,
  dashboard/tests/test_escalation_analytics.py, dashboard/tests/test_escalations_data.py,
  dashboard/tests/test_escalation_status_policy.py (new), fused-memory/tests/test_backlog_policy.py
- Keys: `fk-a65731424439`, `fk-60b7c33ea0ad` (part).
- Deps: α; β1 (`escalation_analytics.py`); S19 β1 (wave 3; moves `reconciliation/backlog_policy.py`, which η
  edits at the new path, as S19's draft records). Edge onto a filed task: S10 ο 9285 → η. Cross-stream: η →
  S22 θ, η, γ2, δ2 (§Cross-PRD; the lead wires it).
- Work: the 11 dashboard sites, the fused-memory sites including
  `backlog_policy.py::BacklogPolicy._close_pending_halt_records`' dict read (→
  `EscalationStatus.PENDING`), the one unknown-status policy (object readers report the skip count;
  raw-dict readers name the value).
- Deletes: 11 dashboard sites and their four unknown-status behaviours; `_STATUS_KEYS`,
  `_OPEN_ESCALATION_STATUS`, `_CLOSED_ESCALATION_STATUSES`, `escalation_archive._TERMINAL_STATUSES`.
- Signal: boundary row 20; the live dashboard's five escalation panels load against today's stores with
  no `unknown_escalation_status` issue (every stored status parses, measured), and the probe rule reports
  zero sites under `dashboard/src` and `fused-memory/src`.
- G7: INV-5, INV-11 (unknown is named, never silently dropped), INV-13.

**θ1 — Harness and workflow read escalation record facts instead of summary text**
(`task_kind=normal`, priority medium, est. ~600 LOC)
- Files: orchestrator/src/orchestrator/harness.py, orchestrator/src/orchestrator/workflow.py,
  escalation/src/escalation/classify.py, orchestrator/tests/test_harness_reblock_signature_guard.py,
  orchestrator/tests/test_orphan_l0_reaper.py, escalation/tests/test_classify.py,
  orchestrator/tests/test_workflow_record_facts.py (new)
- Keys: `fk-5332d25b1702` (readers), `fk-60b7c33ea0ad` (part).
- Deps: β2 (`harness.py`, `classify.py`), γ (C4 and the legacy stamping); S8 ω 9236 (`harness.py` tail),
  S8 κ1 9232 (`workflow.py` tail), S1 δ (the 17 workflow filings through `TaskWorkflow._file_escalation`),
  S1 η2 (the filings that stay in `harness.py`, S1's last `harness.py` leaf) (sitting ruling S2-3 (a));
  ε (`classify.py`); μ (`workflow.py`). Under lead ruling S2/OQ-11 (a) θ1 waits on task 5040 through 9236 and 9232.
- Work: the workflow's two filers set `filing_site` and `failure_signature` (on the drafts S1 δ builds) and extends S1 C7's keyword list (`TaskWorkflow._file_escalation`) with them;
  `test_harness_reblock_signature_guard.py` calls the public `models.failure_identity` instead of
  `Harness._reblock_signature`.
- Deletes: the summary-substring match, the workflow's "Keep it verbatim" docstring paragraph, the
  `suggested_action` discriminator, `_reblock_signature`, the remaining harness and workflow literal
  sites.
- Signal: boundary row 21; a live scope-divergence escalation on dark-factory carries `filing_site`.
- G7: INV-2, INV-5.

**θ2 — Orchestrator modules compare escalation status, action and category through the vocabulary**
(`task_kind=normal`, `complexity=simple`, priority medium, est. ~200 LOC)
- Files: orchestrator/src/orchestrator/steward.py, orchestrator/src/orchestrator/digest.py,
  orchestrator/src/orchestrator/escalation_router.py, orchestrator/src/orchestrator/l0_promotion.py,
  orchestrator/src/orchestrator/provenance_conflict.py, orchestrator/src/orchestrator/resume_resolution.py,
  orchestrator/src/orchestrator/sentinel_alarm_wrappers.py,
  orchestrator/src/orchestrator/deterministic_runner.py,
  orchestrator/src/orchestrator/offline_lane_episodes.py
- Keys: `fk-60b7c33ea0ad` (part).
- Deps: α; S9 ξ 9287 (`steward.py`), S8 ι2 9231 (`digest.py`), S10 β2 9191 (`deterministic_runner.py`),
  S10 ξ 9283 (`provenance_conflict.py`), S5 ο0 9270 (`offline_lane_episodes.py`) (lead ruling S2/OQ-11
  (a)); μ (`steward.py`); S1 γ
  (`deterministic_runner.py`), S1 κ1 (`escalation_router.py`, `l0_promotion.py`), S1 κ3
  (`provenance_conflict.py`), S1 κ4 (`steward.py`) (sitting ruling S2-3 (a)). Under lead ruling S2/OQ-11 (a) θ2 waits on task 5040
  through 9287, 9231 and 9270, so these edges add no 5040 wait. They add S1 γ, S1 κ1, and
  through S1 κ3 → ζ1 the low task 7718 (7105 is done), which priority inheritance raises.
- Signal: the probe rule reports zero sites in these nine modules; their suites green.

**ι — Escalation-vocabulary guard, the scripts' sites and invariant `escalation-vocabulary-imported`**
(`task_kind=normal`, priority medium, est. ~650 LOC) — integration gate, signal leaf
- Files: scripts/tests/test_escalation_vocabulary_literals.py (new),
  scripts/tests/fixtures/escalation_vocabulary/violations.py (new),
  scripts/tests/fixtures/escalation_vocabulary/clean.py (new), docs/legibility/design-invariants.md,
  docs/legibility/design-invariants-fixtures.md, skills/prd/references/gates.md,
  scripts/sitting/gates.py, scripts/sitting/apply.py, scripts/sitting/inventory.py,
  scripts/sitting/fleet_state.py, scripts/member-chain-sweep.py, scripts/sweep_toolcall_markup.py
- Keys: `fk-60b7c33ea0ad` (gate), `fk-a65731424439` (gate).
- Deps: β1, β2, γ, γ2, δ, ε, ε2, ζ, η, θ1, θ2; task 6770 (`sitting/gates.py`, `member-chain-sweep.py`;
  through it 6760 → 6759, S25's scripts chain; 6753 is done); the wave-3 invariant chain in the lead's
  order (lead ruling S2/OQ-10 (b); S1 ω first, as both drafts have it; INV-18 to INV-23 in landing order). Under lead ruling S2/OQ-11 (a) ι waits on task 5040 through θ1 and θ2. ι does not
  wait on μ, ν, ξ or ο: the guard checks the vocabulary, and the denial is a policy flip with its own check.
- Work: the guard over every counting-rule shape, including the store-anchored dict read and C6's
  header-reader rule; the scripts' sites, including `member-chain-sweep.py::sweep_queue`'s
  `record.get('status') == 'pending'`; the four-site lockstep edit.
- Signal: boundary row 23 and rows 1–22 green; `uv run --project shared pytest
  scripts/tests/test_design_invariants_consistency.py` passes with the slug in every pinned site; the
  guard reports zero sites on the tree.
- G7: INV-10 (the guard runs over code against fixtures), INV-12 (no exemption list).

**κ — DedupeConfig fields named for what they configure, and the config frozen**
(`task_kind=normal`, `complexity=simple`, priority low, est. ~450 LOC mechanical)
- Files: escalation/src/escalation/dedupe.py, escalation/src/escalation/queue.py,
  escalation/src/escalation/server.py, escalation/src/escalation/filing_tools.py (created by S3 ζ 9200),
  escalation/src/escalation/resolve_tools.py (created by S3 ζ 9200),
  escalation/src/escalation/read_tools.py (created by S3 ζ 9200),
  escalation/tests/conftest.py,
  escalation/tests/test_amend_escalation.py, escalation/tests/test_capability_guard_http.py,
  escalation/tests/test_dedupe.py, escalation/tests/test_queue.py, escalation/tests/test_server.py,
  escalation/tests/test_server_dedupe.py, escalation/tests/test_status_authority_gate.py,
  fused-memory/scripts/backfill_recon_escalations.py,
  fused-memory/src/fused_memory/middleware/curator_escalator.py,
  fused-memory/src/fused_memory/middleware/dead_letter_escalator.py,
  fused-memory/src/fused_memory/middleware/entity_mint_storm_escalator.py,
  fused-memory/src/fused_memory/middleware/mem0_update_storm_escalator.py,
  fused-memory/src/fused_memory/middleware/scope_violation_escalator.py,
  fused-memory/src/fused_memory/reconciliation/backlog_policy.py,
  fused-memory/src/fused_memory/reconciliation/harness.py,
  fused-memory/src/fused_memory/reconciliation/preservation_specimen_guard.py,
  fused-memory/src/fused_memory/reconciliation/standing_decision_constants.py,
  fused-memory/src/fused_memory/reconciliation/task_count_snapshot_cadence.py,
  fused-memory/tests/middleware/test_entity_mint_storm_escalator.py,
  fused-memory/tests/reconciliation/test_stage1.py, fused-memory/tests/test_harness.py,
  orchestrator/src/orchestrator/harness.py, orchestrator/src/orchestrator/merge_lane/worker.py,
  orchestrator/src/orchestrator/provenance_conflict.py, orchestrator/src/orchestrator/workflow.py,
  orchestrator/src/orchestrator/agents/roles.py, orchestrator/tests/test_digest.py,
  orchestrator/tests/test_workflow_preexisting_main_break.py, scripts/tests/sitting_escalation_servers.py
- Keys: `fk-9c7ebc2b50b2`.
- Deps: γ (`dedupe.py`); ε (`server.py`, `queue.py`; through it δ); ζ (`roles.py` reads the renamed
  attributes); η (`backlog_policy.py`); θ1 (`harness.py`, `workflow.py`); θ2 (`provenance_conflict.py`);
  8528 (it declares
  `merge_lane/worker.py`, so it is parked behind 8528 at filing, R55); S1 θ2, S1 ι2, S1 ω (the
  fused-memory escalators, the recon filers and the one-door gate) (sitting ruling S2-3 (a)); γ2 (`queue.py`,
  `dedupe.py`, fused-memory `reconciliation/harness.py`); ε2 (`server.py`); S19 β1 and S19 γ (wave 3; κ edits
  the moved modules at their new paths, as S19's draft records). 6348 cancelled into S1 θ2;
  κ → S1 θ2 carries it.
- Work: the rename and `frozen=True`. The 32 `.py` files above holding `infra_dedupe_` at `ba17934861`
  (unchanged at `e59840cdaf`) plus `roles.py` (ζ's reads). `server.py`'s tools move to `filing_tools.py`, `resolve_tools.py` and
  `read_tools.py` with S3 ζ 9200, which lands before κ under S2-3 (a); κ declares both. S1's new modules
  (`escalation/filing.py`, recon `recon_escalations.py`) may add `infra_dedupe_` uses: κ re-measures
  `git grep -l infra_dedupe_ -- '*.py'` at dispatch and declares every file it finds. S1 θ2 depends on
  S19 γ, which moves three of κ's fused-memory escalators to `fused_memory/escalators/` (lead ruling S19/OQ-4 (b)); κ declares the paths
  it finds at dispatch.
- Signal: `git grep -c infra_dedupe_ -- '*.py'` is empty; `DedupeConfig` is frozen; data files under
  `fused-memory/calibration/`, `docs/` and `plans/` keep their historical spellings.

**μ — Every dark-factory escalation client and every spawned agent sends an identity header**
(`task_kind=normal`, priority **high**, est. ~350 LOC)
- Files: orchestrator/src/orchestrator/mcp_lifecycle.py, orchestrator/src/orchestrator/steward.py,
  orchestrator/src/orchestrator/workflow.py, orchestrator/src/orchestrator/review_checkpoint.py,
  orchestrator/src/orchestrator/agents/briefing.py, orchestrator/src/orchestrator/evals/runner.py,
  orchestrator/src/orchestrator/dry_run_unblock.py, fused-memory/scripts/cgl_eta_finalize_gate.py,
  .mcp.json, recon-watch/mcp.json, skills/escalation-watcher/SKILL.md,
  skills/recon-escalation-watcher/SKILL.md, orchestrator/tests/test_mcp_lifecycle_identity.py (new)
- Keys: none stamped; its description names `fk-ece88b7661b7` (the deploy before the denial).
- Deps: ε (the server must parse identities, keep `resolver_detail` and admit human identities to
  `PROMOTE_ALLOWED` before clients start sending them). Per-file order: the stated exception to R54 in
  §Cross-PRD, confirmed by the lead on 2026-11-13 (no edge onto the wave-2 tails of its orchestrator
  files or the two skill texts); θ1 and θ2 follow μ on `workflow.py` and `steward.py`.
- Work: C6's `mcp_config_json` (an escalation entry needs an identity; the three call sites that pass an
  escalation URL pass the dispatched agent id; `briefing.py::_caller_agent_id` becomes public; the
  `McpConfigSource`/`_McpLike` protocols and the eval stub accept the argument and the stub still returns
  no escalation entry); dark-factory's `.mcp.json`, `recon-watch/mcp.json` and the copy the recon
  watcher's skill prints send `X-Escalation-Identity: ${ESCALATION_IDENTITY:-interactive}`; the L2
  watcher's skill tells its launcher to set `ESCALATION_IDENTITY=escalation-watcher`; every spawned
  agent's environment carries `ESCALATION_IDENTITY=<its dispatched agent id>` (the `env_overrides` of the
  steward, workflow, review-checkpoint and dry-run invocations), so whichever of its two `escalation`
  entries the CLI keeps sends its id; `cgl_eta_finalize_gate.py` sends an identity header. First step:
  confirm on one live session that the CLI expands `${ESCALATION_IDENTITY:-interactive}` (read in the CLI
  2.1.322 code, not yet tried); if it does not, ship the literal `interactive` and give the watcher launcher
  `--strict-mcp-config --mcp-config <file>` whose header is `escalation-watcher` (a step of this task, not
  a hope). The five timer scripts only file, so they do not change. Deletes: the `mcp_config_json`
  docstring's promise that every other caller keeps "full-authority, header-less connections".
- Signal: boundary rows 26, 26a and 27; after the dark-factory orchestrator restarts on μ, a steward's L0
  close is archived with `resolved_by: "claude-task-<id>-steward"` and `resolved_via: "identified"` (never
  `interactive`), and an interactive session's resolve reads `resolved_by: "interactive"`.
- G7: INV-5 (one identity format, `_caller_agent_id`), INV-11 (a missing identity is a construction
  error, never a silent header-less entry), D13 (the eval stub stays escalation-less, pinned by a test).

**ν — Operator gate: deploy the escalation identity header to every other project's MCP configuration (pre-authorised session)**
(`task_kind='deterministic'`, `metadata.always_escalates=true`, no `before_done`, `files: []`,
`metadata.x_preauthorised_spawn` (S16 χ), priority **high**)
- Files: none (`files: []`; every edit is in another repository).
- Keys: none stamped; its description names `fk-ece88b7661b7`.
- Deps: ε, μ (dark-factory's `.mcp.json` is the template the session copies); S16 χ (the
  escalation-watcher rule that spawns a pre-authorised gate's session, keyed on
  `metadata.x_preauthorised_spawn`; lead ruling 2026-11-13: S2's gate uses S16's mechanism and depends on
  the task that builds it, which has no prerequisite inside S16); the reify task the lead files at
  decompose for `reify/scripts/docs-truth-sweep.py`'s identity header, as the cross-project dependency
  `"reify:<id>"` (lead ruling 2026-11-13).
- Shape: the program's operator-sweep gate (cross-stream consequence, shared with S16's yaml sweep). It
  files one born-at-L2 escalation whose description is the `/team` brief: for each of the eleven projects
  in §Background's table, add `"headers": {"X-Escalation-Identity": "${ESCALATION_IDENTITY:-interactive}"}`
  (or the literal μ shipped) to the escalation entry of `<project>/.mcp.json`; one `.mcp.json`-only
  `git commit --only .mcp.json` per tracked repository, autotrade's untracked file edited in place;
  record each sha or skip reason on the escalation; in each repository, grep non-test code for an HTTP
  caller of `resolve_issue`, `promote_to_l2`, `stamp_triage`, `amend_escalation`, `claim_policy_class` or
  `declare_pin` and record each one found on the escalation without editing it (the measured one,
  `reify/scripts/docs-truth-sweep.py`, is a reify task, §Deps above); fast-forward or edit in place each of the twelve side checkouts' `.mcp.json`, recording
  which; check and record the laptop's dark-factory and reify checkouts. The description quotes
  Leo, 2026-11-12: "deploy the MCP configuration to all existing projects before denying, make the denial
  self-explaining via the error message, especially for the no identity header case", and says in so
  many words that the spawn is pre-authorised and may run while he is away. The task carries
  `metadata.x_preauthorised_spawn = {ruled_by: 'human', ruled_on: '2026-11-12', ruling_ref:
  'plans/review-all-s2-typed-escalation-record-facts-prd.md::Rulings (sitting of 2026-11-12)',
  prompt_from: 'description'}`, the key S16 χ's watcher rule reads, so the watcher spawns the session in
  attended and AFK posture alike. A resume marks the gate done and releases ξ.
- Signal: every project's escalation entry and every side checkout's sends the header (the escalation's
  per-repository sha list and side-checkout list); any other-repository HTTP caller of an
  authority-bearing tool the grep found is listed on the escalation.

DAG (depends-on): α → {9173, 8562, S19 θ0}; β1 → {α, 9157, 9162}; β2 → β1; γ → {β2, 9181, S1 β};
γ2 → {γ, ε, η, 9254}; γ3 → {γ2, S1 ω, θ1, θ2, 8528}; δ → {γ, 9181, 9180}; ε → {δ, S1 β}; ε2 → ε; μ → ε; ν → {ε, μ, S16 χ, reify:<id>};
ζ → {γ, 9270}; η → {α, β1, S19 β1, S22 θ, S22 η, S22 γ2, S22 δ2}; θ1 → {β2, γ, ε, μ, 9236, 9232, S1 δ, S1 η2}; θ2 → {α, μ, 9287, 9231, 9191,
9283, 9270, S1 γ, S1 κ1, S1 κ3, S1 κ4}; ι → {β1, β2, γ, γ2, δ, ε, ε2, ζ, η, θ1, θ2, 6770, S1 ω};
κ → {γ, γ2, γ3, ε, ε2, ζ, η, θ1, θ2, 8528, S1 θ2, S1 ι2, S1 ω, S19 β1, S19 γ}. Edges onto existing tasks (lead actions):
9177 → β2, 9144 → β2, 9188 → ε2, 5258 → ε2, 8477 → ε2, 8478 → ε2, 9285 → η, 6770 → δ (release),
7639 → ζ, 7392 → ζ. The fingerprint retirement is this plan's γ2 then γ3 (γ3 → S1 ω; D12). What waits on what: α, β1,
β2 wait on none of 4567, 5040 or 8528 (α adds S19 θ0, about 120 lines with no prerequisites); γ, γ2, δ, ε,
ε2, μ and ν wait on none of 5040 or 8528 (the lead confirmed μ's stated exception); θ1, θ2, ζ, ι, κ and γ3
wait on 5040 (lead ruling S2/OQ-11 (a)); κ and γ3 also on 8528. The cost of the S2-3 (a) order is now real: task
4567 is done (`b255f96b3e`), so S19 θ0, α, β1 and β2 (about 2,600 lines) sit strictly between 9173/9162
and 9177, and the 215 open dependents of 9177 (40 high, 111 wave-2, gate 6698; measured 2026-11-13) wait
roughly the serial time of those tasks, unless 9199's chain finishes later; priority inheritance raises
them to high, which shortens their queue wait, not their build and merge time.

### G7 walk (advisory at author time)

`contracts-machine-checked`: C1–C7 each have boundary rows; the guard is code. `structured-facts-at-failure`:
`UnknownEscalationValue`, precondition refusals, `invalid_duplicate_of`, `identity_required` (its
message names header, tool, action, level, config file and fix). `corroborate-before-acting`: ε2 on both
doors. `no-lockstep-duplication`: the 7 terminal-set copies, 3 action tables, the sitting copy and
`classify.py`'s second human set go. `one-fact-one-home`: `cascade_parent_id` (the event carries no copy
of it), `failure_identity` and `declared_failure_identity` (one reader for the stored fingerprint and the
signature), `HUMAN_IDENTITIES`, `_caller_agent_id`. `no-silent-fail-soft`: the harness's silent legacy
fallthrough goes; unknown status named; an unparseable record is counted, not raised and not dropped; an
unidentified authority-bearing act is stamped and warned before the denial and refused after it.
`readers-prove-their-producer`: β1's signal reads the live sweep's output; γ2's readers are shown
matching both producer shapes; ξ reads the stamp ε writes. `exceptions-owned-or-ratified`: no exemption
list. `storm-escape-required`: ν and ξ file one escalation each by construction; nothing else new files
escalations. `holds-owned-and-bounded`: 6770's hold is released by δ. `one-decision-partition-per-area`
(INV-23, D10): every table names who decides each row.

## Not filed (the lead files each once its trigger holds; R58)

The dated deny-check the draft listed here is withdrawn (sitting ruling S2-2): the high finding closes by
code when ο lands.

- **ξ — Check: no escalation was resolved, parked or promoted by an unidentified caller for three days
  after the identity deploy.** Trigger: ν is done (so ε's script exists, which `submit_task` validates
  for a deterministic task's `before_done.script`, R58, and μ is merged) and every project's orchestrator
  and fused-memory runs μ (`scripts/orchestrator-watchdog.py --report`), so the window never counts an
  agent spawned by pre-μ code. `task_kind='deterministic'`,
  `before_done: {kind: predicate, script: scripts/check_unidentified_escalation_actions.py,
  args: ['--window-secs', '259200', '--queue-dir', '<dark-factory>/data/reconciliation/escalations'],
  env: {DASHBOARD_KNOWN_PROJECT_ROOTS: <the host's project roots>},
  timeout_secs: 120}`, `metadata.milestone: {mode: delayed, after_secs: 259200}`, priority high,
  `source: review-all`, `x_finding_run`, `prd_path`, `prd_task_label: ξ`, no finding key. Deps: μ, ν.
  Leo's "a few days" read as three, confirmed by the lead (2026-11-13). Exit 0 → done; non-zero → a born-at-L2
  `milestone_check_failed` names the queues and records, and a human finds the client the deploy missed,
  fixes it and resumes (the check re-runs). The proxy is stated: it counts resolves, parks and promotes,
  which every client that can stamp, amend or pin also performs; those three acts are warned and logged
  by ε but not counted. Joins gate 6698.
- **ο — Refuse escalation authority to a caller that sends no identity header, with a self-explaining
  error.** Trigger: ξ is done (the policy-flip rule, R58; ruled order step 4). `task_kind=normal`,
  priority high, `x_finding_key: [fk-ece88b7661b7]` (it closes the key), joins gate 6698. Work: every
  `UNIDENTIFIED` `AUTHORITY_BEARING` call returns `identity_required` with ε's message and changes nothing,
  at every level, a header-less `promote_to_l2` included (this supersedes lead ruling S2/OQ-9's "no
  header → allowed"); filings and reads stay open; the five esc-2087-2 test pins change
  (`escalation/tests/test_capability_guard_http.py` 2, `escalation/tests/test_status_authority_gate.py` 3,
  measured at `e59840cdaf`); `authority.py`'s module docstring, `docs/escalation-standing-policy.md`'s
  esc-2087-2 passage and the recon watcher skill's paragraph that says a header-less `promote_to_l2`
  "actually succeeds" are rewritten; the change records that it ends the 2026-08-20 human-channel
  guarantee (esc-2087-2). Files re-measured at filing (S3 ζ 9200 may have moved the tools into
  `filing_tools.py`, `resolve_tools.py` and `read_tools.py` by then). Signal: boundary row 28; after every
  project's orchestrator restarts on ο, a header-less `resolve_issue` on any project answers
  `identity_required` with the pinned text. The escalation server runs inside each project's
  orchestrator from the shared package, so ο reaches a project when that orchestrator restarts.

## Out of scope

- The filing entry point, `FilingPolicy`, and the dunder sentinel task-id literals (S1, R8; S1 measures
  them; lead ruling S2/OQ-6).
- Moving tools between modules and the queue's mutation primitives (S3).
- Typing `suggested_action` or `agent_role` (open vocabularies; S9 D5 keeps `suggested_action` a hint).
- Retiring `dedupe_fingerprint` before S1 ω: S1 keeps writing it until its closing task, and γ3 retires
  it after (the lead's "FINGERPRINT" ruling, D12).
- Refusing filings or reads from a caller without an identity (the residual's option (c), not ruled):
  a client the deploy missed loses an action, never an escalation.
- Authenticating identities: the header is self-asserted (D11).
- The merge-lane control tools on the same server (`merge_request`, `merge_cancel`, `halt_merge_queue`,
  `release_workflow` and the rest) keep today's rules; the ruling covers the escalation ladder's acts.
- Replacing the decision-stream evidence model for the classes the server cannot corroborate (the
  sitting's S2-1 option (c)).
- Rewriting `resolved_by` on legacy records.

## Rulings (sitting of 2026-11-12)

Leo answered on 2026-11-12 and his residual answers on 2026-11-13 (`digests/wave3-rulings-applied.md`
§S2 and its "Cross-stream consequences"; the measured effects in `digests/discussion/S2.md` are part of
the rulings). A question Leo did not list stands as recommended.

- **OQ-1** (sitting key S2-1, who may close a human-facing escalation automatically) — option (a′): one
  corroboration predicate on the policy row, checked by both automated doors, true only when the task
  store says the subject task is done or cancelled; carried by ε2 (split out of ε by this fold). Ruled by
  Leo (as recommended).
- **OQ-2** (sitting key S2-2, what a caller with no identity header may do) — option B widened by
  residual R-6 = B: such a caller loses every authority-bearing act at every level (resolve, park,
  promote, triage stamps, amendments, policy-class claims, pin declarations); filings and reads stay open
  and are stamped `unidentified`; every spawned agent sends its role identity; the MCP configuration is
  deployed to every project before the denial; the denial explains itself, above all for the missing
  header; the header-less channel is not a source of authority. Order: ε (parser, stamp, message, no
  denial), μ and ν (deploy), ξ (check), ο (denial, filed by the lead). Ends the 2026-08-20 human-channel
  guarantee (esc-2087-2). Ruled by Leo (differs from the recommendation (a)).
- **OQ-3** (one failure identity) — (a): one reader (`declared_failure_identity`, `failure_identity`),
  AMENDED by the lead's "FINGERPRINT" ruling of 2026-11-13 (which resolves the block "TO RESOLVE AT THE
  RULING FOLD"): S1 keeps writing `dedupe_fingerprint`; this plan carries the retirement, γ2 (the five
  readers move) then γ3 (the field and S1's fingerprint arguments go, after S2 γ and S1 ω) (D12). Ruled by
  the lead.
- **OQ-4** (where the category enum lives) — (a): `escalation/categories.py`, no re-export; a
  clarification of R9's wording. Ruled by the lead.
- **OQ-5** (where the callback leaf sits on `harness.py`) — (a): 9144 waits on β2 (already implied
  through 9177); 9176 and 9216 are amended to read the event. Ruled by the lead.
- **OQ-6** (who adds the sentinel registry) — (a): S1, after S2 β2. Ruled by the lead.
- **OQ-7** (sitting key S2-3, order of S1's and S2's edits) — (a): S2's vocabulary, cascade-parent and
  callback tasks, then S1's door, then S2's later leaves; the stated cost is updated (4567 done, 215
  open dependents behind 9177). Leo's added question became task 9490 (the store refuses a cycle-closing
  edge), outside every plan; this plan's two cycle sentences name that refusal. Ruled by Leo (as
  recommended).
- **OQ-8** (what `resolved_by` says on a cascade member) — (a): the parent's resolver; `cascade` before
  `human-ruled`; flagged. Ruled by the lead.
- **OQ-9** (how `promote_to_l2` applies the level ceiling) — (a), identity first, AMENDED by OQ-2: its
  clause "a connection that sends no header is allowed" is superseded; such a promote is warned by ε and
  refused by ο. Ruled by the lead, amended by Leo's OQ-2 ruling.
- **OQ-10** (wave-3 invariant order) — (b): no wait on 9154 for a number; each invariant takes the next
  free number when it lands; amends R51's numbering clause. Ruled by the lead.
- **OQ-11** (small edits at the end of wave-2 chains) — (a): end of chain. Ruled by the lead.
- **OQ-12** (the record-facts, resolve-table and identity leaves on `server.py`) — (a): off R54's chain,
  between S1 β and 9188, a stated, measured exception to R54; ε2 joins them. Ruled by the lead.
- Lead rulings after the fold-verifiers (2026-11-13): the order (build the refusal and its message but
  only warn, deploy everywhere, measure for 3 days, then refuse) is confirmed; μ's one-line edits take no
  per-file ordering edge (a stated exception to R54); test-file-only edges are not wired. Ruled by the lead.
- Cross-stream (lead): S2 α waits on S19 θ0 and the invariant names `shared.escalation_layout` (S19/Q-C8;
  R9's invariant text gains that home). Evals stay hermetic (D13). Leo's design principle, cited as
  `one-decision-partition-per-area` (INV-23; D10). Test-file edges are not wired (restates R54).
- Lead rulings at the alignment pass (2026-11-13): D11's attribution changes (interactive and
  escalation-watcher keep promotion through `PROMOTE_ALLOWED`; `unblock-low-risk` closes move to the
  human tier with the name kept as `resolver_detail`; a T3-hosted watcher reads `interactive`; identities
  are self-declared, not authenticated) are confirmed by the lead and told to Leo. The fingerprint
  retirement is two tasks, γ2 then γ3, for size (the lead's "one task" wording is amended). μ's one-line
  edits in `evals/runner.py` and `workflow.py` are covered by the one-line exception. The edit to
  `reify/scripts/docs-truth-sweep.py` is a task the lead files in reify's store, on which ν depends. ν
  uses S16 χ's spawn mechanism and depends on it. The six wave-3 invariant tasks take INV-18 to INV-23
  in landing order, the decision-partition invariant last. Ruled by the lead.

Tactical (decide in the leaf): the event's copy mechanism (β2); the backfill pass's counter names (β1);
whether the compact projection shows `hypothesis` (γ); the corroboration timeout (ε2); the dashboard
policy function's module (η); the fixture module layout (ι); the exact signatures of the two test seams
(β2); the exact words of the `identity_required` message within C6's required content (ε); the module
that holds the check script's queue discovery (ε).
