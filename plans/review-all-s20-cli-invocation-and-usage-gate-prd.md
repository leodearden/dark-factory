# S20 — CLI invocation and usage gate: one classification, a decision-driven retry loop, and leaf modules

**Status:** draft (author pass + critic fold + ruling fold) — authored 2026-11-11 by the S20 seat of the
`/review-all` remediation program (`plans/review-all-program-dark_factory-2026-10-10.md`, stream S20,
wave 3, ruling **R18**); critique `S20.critic.md` folded the same day; the sitting's rulings of
2026-11-12 folded on 2026-11-12 (see `## Rulings (sitting of 2026-11-12)`). Remediates run
`review-all-dark_factory-20261008`. Size L, risk medium. Ruled and fold-verified; aligned with the
sibling wave-3 drafts on 2026-11-13 (`critique/S20.align.md`); not yet decomposed.
Leo's ruling on task 8253 (2026-11-12, `digests/leo-rulings-8253.md`) is applied in the text; see
`## Dependence on task 8253`.

**Code anchors** verified against main `1f240e6da2` (author pass), re-verified at `97be1d71fa`
(2026-11-11, fold pass), at `1b106418da` (2026-11-12, ruling fold) and at `0f21f5e457` (2026-11-12,
fold-verify): `git log 97be1d71fa..1b106418da` touches 23 of the plan's paths in 57 commits, and
`1b106418da..0f21f5e457` touches 5 in 73 commits, none at an anchor; what changed is recorded in
`### Re-measurement (ruling fold, measured at 1b106418da)` and `### Re-measurement (fold-verify,
measured at 0f21f5e457)`. Anchors are cited by symbol; re-locate at implementation time. Every
count is labelled with the sha it was measured at.

## Goal

The next agent who changes how a Claude CLI invocation is classified, retried, accounted or
attributed to an account makes the change in one place, and a wrong change fails a test rather than
silently benching a healthy account or under-reporting spend. After this PRD lands:

- **One classification per attempt.** Every dispatched attempt is classified exactly once, by
  `shared.invocation_outcome.classify_invocation`: inside the usage-gate slot that leased the account
  (`InvokeSlot.observe(result) -> Decision`, R18 option 1), or by the loop itself on the ungated path.
  The gate's synthetic-result second classification (`UsageGate.detect_cap_hit`) and the loop's
  zero-cost heuristic cap net are gone.
- **The retry loop is a thin state machine.** `invoke_with_cap_retry` is attempt → classify → decide (a
  pure, table-tested policy over the `Decision`) → settle. Precedence lives in the classifier and the
  decision table, not in statement order defended by prose.
- **Every attempt is accounted.** Each dispatched attempt writes one `invocations` row, and the retry
  loop is the one writer of those rows for every caller that routes through it (the task-workflow
  roles and the deep reviewer included); a discarded attempt (a cap failover, an auth
  failover, a rejected resume, a wedge) no longer spends invisibly.
- **One vocabulary for CLI result subtypes.** `CliSubtype`, a `StrEnum` beside `AgentResult`, owns
  every subtype spelling and marks which values come from the CLI wire and which this repo mints.
- **Leaf modules instead of one 4,945-line file** (measured at `1b106418da`). `shared/cli_invoke.py`
  is partitioned along the measured heuristic-14 partition into result, failure-projection,
  transcript, argv and runner leaves, carrying the public runner names the package-layering plan
  (S19, its contract C7: the public CLI runner in `shared.cli_invoke`) gives the runner; the
  `cli_invoke` ↔ `invocation_outcome` ↔ `usage_gate` import cycle and its function-local imports go.
  The transcript nesting rule has one home with a public accessor.
- **The usage gate's account machine has one writer.** `AccountState`'s four phase setters go;
  `_transition` writes `phase` and `resets_at`; one method writes the `cap_hit` account event and its
  `resumed` dual; the probe returns a verdict and the gate applies it; an invocation on a backend
  outside the account pool (codex, gemini and, since tasks 5399 and 7993, zcode and mimo) no longer leases or benches a
  Claude account; tests reach the gate through a public seam.
- **Small seams other streams wait on:** the capability-manifest sidecar surface S25 is held on
  (contract C3), the invocation cost floor with an enforcement point (C6), a scratch root threaded
  as a parameter, and delivered-check kinds as a discriminated union.

## Consumers and user-observable surfaces

| Mechanism | Consumer (named surface) |
|---|---|
| `InvokeSlot.observe -> Decision` (C2) | `shared/src/shared/cli_invoke.py::invoke_with_cap_retry`, the one production caller of the slot; through it every orchestrator role, the steward, the fused-memory curator/judge/recon stages and the legibility session runner. Observable: `cap_hit`/`failover`/`resumed` rows in `runs.db` `account_events` and the dashboard's cap history. |
| `classify_invocation` as the one classifier (C1) | the slot (C2), the loop's ungated path (C7), `shared/src/shared/account_probe.py` (new, the probe verdict), `shared/src/shared/agent_failure.py::classify_agent_failure` (projection), `orchestrator/src/orchestrator/routing.py::classify_probe_outcome`, `orchestrator/src/orchestrator/evals/measurement.py::_transport_refusal` (S18) |
| Decision table + per-attempt `invocations` rows (C7) | `runs.db` `invocations` read by `dashboard/src/dashboard/data/costs.py`, `dashboard/src/dashboard/data/model_role.py`, `orchestrator/src/orchestrator/digest.py`, `scripts/audit_model_admission.py`, `scripts/review_model_admission.py`, `shared/src/shared/quality_series/rows.py`; the session budget (`SessionBudgetExhausted`); after ο2, the task workflow's and deep reviewer's cost rows (`orchestrator/src/orchestrator/workflow.py::TaskWorkflow._invoke`, `orchestrator/src/orchestrator/review_checkpoint.py`) |
| `CliSubtype` (C4) | `shared/src/shared/agent_failure.py`, `shared/src/shared/usage_gate.py`, `orchestrator/src/orchestrator/steward.py::_is_empty_output`, `orchestrator/src/orchestrator/dry_run_unblock.py::_is_budget_exhausted`, `orchestrator/src/orchestrator/routing.py::classify_probe_outcome`, `orchestrator/src/orchestrator/reviewer_failure.py`, `fused-memory/src/fused_memory/middleware/path_scope_adjudicator.py`, `orchestrator/src/orchestrator/agents/invoke.py` and `orchestrator/src/orchestrator/agents/zcode_backend.py` (mint sites; the second added by task 5399), `scripts/review_model_admission.py` |
| `shared/transcript.py` (C5) | `shared/src/shared/claude_runner.py` (new), `shared/src/shared/transcript_archive.py`, `orchestrator/src/orchestrator/workflow.py::_capture_zero_output_evidence`, `orchestrator/src/orchestrator/harness.py`, `fused-memory/src/fused_memory/middleware/task_curator.py` |
| Sidecar surface `sidecar_for` / `tracked_sidecar_paths` (C3) | held task **6764** (S25 ι): the `commit_planning` stamper, `scripts/audit_manifest_descriptor_drift.py`, `scripts/audit_delivered_checks.py`, `scripts/audit_combine_gate_marker_loss.py`; then task **7724** |
| `INVOCATION_COST_FLOOR_USD` + `InvocationBudgetBelowFloor` (C6) | `shared/src/shared/claude_argv.py::build_claude_argv` (the enforcement point) and every caller of it (all claude-backend agent invocations); S16's config-load budget check imports the constant (S16 C5: the shared routing policy, whose `validate_role_models` refuses a Claude role whose budget is below the floor at load and at hot reload) |
| Scratch root parameter | `UsageGate`, `fused_memory.middleware.task_curator`, `scripts/legibility/session_runner.py`, `df_pytest_isolation.py`'s suite-wide confinement |
| Delivered-check discriminated union | `commit_planning` (sidecar validation), `orchestrator/src/orchestrator/delivered_checks.py` (the dispatch gate), `orchestrator/src/orchestrator/agents/briefing.py` (agent briefings), `fused-memory/src/fused_memory/server/manifest_stamping.py`, `scripts/audit_manifest_descriptor_drift.py` |

## Finding keys and freshness

All 19 keys of the S20 row (R18). Freshness: `git diff 6551b75aba..main` touched every anchor file
(`cli_invoke.py` +957/−505 in 48 commits, `usage_gate.py` +299/−215, `invocation_outcome.py`
+111/−37, `capability_manifest.py` +59/−18, `config_dir.py` +17/−2, `transcript_archive.py` +44/−17,
`cap_markers.py` +4/−4). Every symbol below was re-read at `1f240e6da2`. Contract §8 step 1
(`find_tasks_by_metadata(key="x_finding_key", value=<key>)`) returned no task for any of the 19 keys
on 2026-11-11; step 2 (`search_tasks` ≥ 0.5, confirmed with `get_task`) found the related live tasks
named in `## Related tasks`, extended by the fold (6996 at 0.645 for the module split; 8253's cap
ledger; the 44 open tasks declaring `cli_invoke.py`, `usage_gate.py` or `transcript_archive.py`).

### Re-measurement (author pass, measured at `1f240e6da2`)

- **`shared/cli_invoke.py`** is 4,911 lines: 1,443 docstring + 1,214 comment-only lines (54.1% prose).
  `invoke_with_cap_retry` spans 1,043 lines at cognitive complexity 144 (`complexipy`), the file
  maximum; `_run_subprocess` 557 lines / 102; `classify_agent_failure` 288 lines / 26. The gated
  path is still one `while True` body with ten ordered arms (AuthFailed, ModelNotFound, CLI-reject
  retry, wedge, non-pool cap, pool cap via `slot.detect_cap_hit`, heuristic zero-cost net, ServerError,
  progress-timeout, resume rejection), each defended by placement prose.
- **Second classification.** `invoke_with_cap_retry` calls `classify_invocation(result,
  strict_confirm=True, backend=backend)` once, then for the pool-cap arm calls
  `slot.detect_cap_hit(result.stderr, result.output)`, whose `UsageGate.detect_cap_hit` builds a
  synthetic `AgentResult(success=False, output=..., stderr=...)` and classifies it again.
- **Heuristic net.** The zero-cost / ≤1-turn / <5 s net is still present. It has not fired in 100+
  days: across the ten project `data/orchestrator/runs.db` stores on this host, `account_events`
  rows whose details carry `Heuristic cap` number 51 in dark-factory (last 2026-07-19), 390 in reify
  (last 2026-07-30), 9 + 2 elsewhere, and **0 since 2026-07-30**, while **1,170** unscoped `cap_hit`
  rows were written since 2026-10-11. dark-factory's last five heuristic rows were mislabelled (an
  HTTP 500 and "Agent produced no output"). Since task 6213 a genuine cap under
  `--output-format json` arrives as a failed result carrying cap text (measured there), which the
  exact tier classifies.
- **Function-local first-party imports:** 7 in `cli_invoke.py` (6 of `shared.invocation_outcome` —
  task 5579 added two `RESUME_REJECTED_MARKERS` imports since as_of — and 1 of `shared.usage_gate`),
  1 in `usage_gate.py` (`CAP_HIT_PREFIXES`/`NEAR_CAP_PREFIXES` inside `UsageGate.detect_cap_hit`), 1 in
  `transcript_archive.py` (the private `_resolve_transcript_path`). `invocation_outcome.py` imports
  `is_server_error_status, is_zero_output_timeout` from `cli_invoke` at module top; `usage_gate.py`
  imports `BUDGET_EXHAUSTED_SUBTYPE, AgentResult, claude_binary_spec, parse_claude_cli_output` from
  it. `cli_invoke` reads `getattr(usage_gate, '_config', None)` to derive the cap scope.
- **Cost accounting.** One `cost_store.save_invocation` call, after the loop, for the returned
  attempt only, and only when the caller passed `cost_store`. `_DispatchedAttempt` (task 6627) is
  rebuilt per dispatch but only the last is written. `slot.confirm(cost)` (which feeds
  `on_agent_complete`, the session budget) runs on the confirming arms only; `slot.report` arms drop
  the attempt's cost.
- **Subtype vocabulary.** `AgentResult.subtype` is a bare `str`. `'error_max_budget_usd'` now has 2
  production homes — `shared.cli_invoke.BUDGET_EXHAUSTED_SUBTYPE` (task 5521 pointed
  `path_scope_adjudicator`, `dry_run_unblock`, `routing` and `usage_gate` at it) and a new copy,
  `orchestrator/src/orchestrator/reviewer_failure.py::BUDGET_CAP_SUBTYPE` — plus 3 literal uses in
  `scripts/mint_hard_v2_fixtures.py`. `'error_max_turns'` is spelled in `cli_invoke`
  (`_ceiling_reported_by_subtype`), `scripts/review_model_admission.py::TURN_CAP_SUBTYPE` and
  `fused-memory/scripts/probe_schema_max_turns.py` (3). Minted values: `error_empty_output` (1 site
  in `cli_invoke._parse_claude_output`, 3 in `orchestrator/agents/invoke.py`), `text_output` (1 + 2),
  `error_cli_input_rejected`, `error_timeout_killed_with_progress` (1 each), `success`/`error` minted
  at 3 sites in `invoke.py`. `runs.db` `invocation_end` subtypes since 2026-10-01: `success` 9,140,
  none 1,028, `error_max_turns` 104, `error_timeout_killed_with_progress` 39.
- **Transcript nesting rule:** 3 restatements in production — `cli_invoke._record_field` (documented
  "SOLE expression"), the inline walk in `cli_invoke.detect_resumable_progress`, and
  `orchestrator/src/orchestrator/workflow.py::TaskWorkflow._capture_zero_output_evidence` (takes
  `message['content']` whenever `message` is a dict, no list check, no flat fallback inside that
  branch). The transcript part of `cli_invoke` (locator through `transcript_evidence_for_session`)
  is about 950 lines and stdlib-only.
- **`usage_gate.py`** is 3,136 lines (924 docstring + 627 comment, 49.5% prose). `UsageGate` holds the
  phase machine (`_transition` cognitive 49, `before_invoke` 40, `try_lease` 33), the per-scope cap
  overlay (`_scope_cap_for` … `scope_status`), the probe subprocess and its verdict
  (`_run_probe`, `_probe_verdict`, which calls `_transition` and writes `acct.resets_at`), the
  resume and auth re-probe loops, SIGHUP reload and cost events. `AccountState` keeps 4 public
  phase setters (`capped`, `probing`, `probe_in_flight`, `auth_failed`); production writes none
  (only `ScopeCap.capped`, twice), tests write them 288 times in 14 files. `acct.resets_at` is written
  outside `_transition` in `_handle_cap_detected` and `_probe_verdict`.
- **Tests reaching gate internals:** 39 distinct `gate._*` names in 30 test files — `gate._accounts`
  807, `_run_probe` 159, `_open` 149, `_handle_cap_detected` 89, `_transition` 73; module patch
  targets `shared.usage_gate.load_dotenv` 31, `.asyncio` 24, `.sweep_stale_pid_dirs` 11,
  `.CREDENTIALS_PATH` 8. Production accommodations for those fixtures: the class-level
  `_park_started_at`/`_park_last_logged_at`/`_park_waiters` defaults (declared for `__new__`-built
  gates in `orchestrator/tests/_orch_helpers.py::build_usage_gate`), the `_probe_config_dir`
  back-compat alias, `getattr`-default reads in `_config_dir_for` and `_start_auth_reprobe`.
  `.detect_cap_hit(` is called 124 times in 15 test files.
- **Tests reaching `shared.cli_invoke` by dotted path:** 33 test files; `asyncio.create_subprocess_exec`
  70, `asyncio.sleep` 66, `invoke_claude_agent` 75, `terminate_process_group` 23,
  `count_transcript_turns` 19, `_WATCHDOG_POLL_SECS` 14, `_run_subprocess` 10. The task-3925 block
  in `_run_subprocess` still says "Do NOT hoist a functools.partial … that silently defeats every
  patch('shared.cli_invoke.count_transcript_turns')".
- **`cap_hit` writers:** 2 call sites, both in `usage_gate.py` (`_transition` on entry to CAPPED;
  the scoped branch of `_handle_cap_detected`). The loop's own `save_account_event('cap_hit')` was
  deleted by `6ec56f30e0` (task 6690). Scoped `cap_hit` rows ever written in dark-factory: 4. A
  `resumed` event has one writer, the probe-confirmed CAPPED → PROBING edge in
  `UsageGate._account_resume_probe_loop`; the clock-driven exit, the SIGHUP reset and CAPPED →
  AUTH_FAILED write none (task 6729's measurement).
- **Cross-process cap state:** of 96 distinct (account, reset) caps recorded since 2026-10-11, 87
  were recorded by more than one project's process (median 6, max 9), i.e. each process rediscovers
  a cap on its own next dispatch. `UsageGate.check_at_startup` is still a logging no-op.
- **Scratch root:** `config_dir.sweep_stale_pid_dirs_once` still requires a keyword-only `sweep`
  callable (22 lines of rejected-alternative docstring); 3 production callers pass their own module's
  `sweep_stale_pid_dirs`; 2 tests exist only to pin that interception point
  (`test_the_module_level_sweep_name_is_still_the_interception_point` in `shared/tests/test_usage_gate.py`
  and `shared/tests/test_startup_completion_probe.py`). `UsageGate` builds probe `TaskConfigDir`s
  without `base_dir`. Task 8234 (pending) exists because the curator and session-runner sweeps are
  confined only by module-local patches.
- **Delivered-check validator:** `capability_manifest._check_kind_conditional_fields` is a 4-arm
  ladder with 26 raises, 19 of them "must not be set", ending in `else:  # manual`. Kind dispatch is
  restated outside: `orchestrator/src/orchestrator/delivered_checks.py` (`meta.kind == 'grep'`,
  `== 'path'`) and `orchestrator/src/orchestrator/agents/briefing.py::_DELIVERED_CHECK_FIELDS_BY_KIND`,
  which lacks `path` (task 7376: every well-formed path check renders "UNRECOGNISED").
- **Cost floor:** no constant exists in `shared`; `build_claude_argv` emits `--max-budget-usd` for any
  value. The floor is stated only in prose in
  `fused-memory/src/fused_memory/config/schema.py::PathScopeAdjudicatorConfig` ("~0.105 USD",
  "Do NOT lower below 0.25"). The lowest budget a production caller passes is 0.50
  (`orchestrator/src/orchestrator/evals/reviewer_trial/scorer.py`); `routing.DEFAULT_PROBE_BUDGET_USD`
  is 1.0; the usage-gate probe passes 0.01 through its own argv by design.

### Re-measurement (fold pass, measured at `97be1d71fa`)

- **Loop callers and the cost rows (B1).** 15 production calls of `invoke_with_cap_retry` in 14 files
  (AST scan). 5 pass `cost_store` (`dry_run_unblock.py`, `harness.py` ×2, `steward.py` ×2). Two
  callers write their own row after the loop instead: `workflow.py::TaskWorkflow._invoke` and
  `review_checkpoint.py` (deep reviewer) — the only `save_invocation(` calls outside
  `shared/cli_invoke.py`. The critic measured those two at about $14.0k of $14.8k in dark-factory
  `invocations` since 2026-10-11 (implementer $6,337, architect $4,757, reviewer_comprehensive
  $2,587). The remaining 8 calls (fused-memory curator, judge, recon stages, path-scope adjudicator,
  evals, legibility runner, nightly preparer) write no `invocations` row at all.
- **Ungated callers (M3).** `fused-memory/src/fused_memory/maintenance/link_adjudicator.py` passes
  `usage_gate=None`; `reconciliation/agent_loop.py` and `reconciliation/judge.py` default it to `None`;
  `orchestrator/src/orchestrator/evals/measurement.py` defaults it to `None` ("deliberately ungated",
  `orchestrator/src/orchestrator/evals/eval_gate.py`). Today's ungated path (`if not usage_gate:` in
  `invoke_with_cap_retry`) runs only the CLI-rejected retry. No ungated production caller presets
  `resume_session_id` (`judge.py` names it in comments only).
- **`invoke_slot` call shapes (M2).** `invoke_slot()` with no arguments: 60 lines in 22 files (tests
  plus `shared/src/shared/testing.py`); with a `scope=` argument: 10 lines in 4 test files
  (`test_make_gate_mock_scope.py` 5, `test_usage_gate_scope_attribution.py` 3,
  `test_park_free_invocation.py` 1, `test_usage_gate_scope_selection.py` 1). Suites configuring
  `detect_cap_hit` on a gate or `make_gate_mock`: shared 22 files (`test_cap_retry.py` 92,
  `test_usage_gate_exhaustive.py` 74, `test_cli_invoke.py` 25, `test_usage_gate_scope_attribution.py`
  20, `test_concurrency.py` 12 …), orchestrator 9 (`test_usage_gate.py` 28, `test_steward.py` 18,
  `_orch_helpers.py` 3, `test_invoke.py` 3, `test_steward_prompt_builder.py`,
  `test_watcher_rotation_cap_retry.py`, `test_dry_run_unblock.py` 2 each, `test_workflow_prompt_builder.py`,
  `test_eval_architect.py` 1 each), fused-memory 2 (`test_judge.py`, `test_main_curator_usage_gate.py`).
- **AuthFailed is backend-blind (M4).** `classify_invocation` tests 401/403 above its backend arms, so
  a codex or gemini 401 classifies `AuthFailed`, and the loop then calls `slot.report(outcome)` on the
  leased Claude account. `ACCOUNT_POOL_BACKENDS = {'claude', 'pi'}` lives in `cli_invoke`.
- **Subtype literals (M5).** Single-quoted `'error'` is spelled on 717 lines in 140 non-test `.py`
  files under `shared/src`, `orchestrator/src`, `fused-memory/src`, `dashboard/src` and `scripts`;
  `'success'` on 89 lines in 33 files — mostly status, log and JSON keys, not subtypes. The
  `ERROR_*`/`TEXT_OUTPUT` values are spelled in 7 production files (`cli_invoke.py` 14,
  `agents/invoke.py` 5, `mint_hard_v2_fixtures.py` 5, `agents/roles.py` 3 in prompt prose,
  `steward.py` 2, `reviewer_failure.py` 1, `review_model_admission.py` 1).
- **Sub-floor test budgets (M6).** `max_budget_usd=0.01` at 10 test call sites in 8 files:
  `orchestrator/tests/test_invoke_agent_credentials.py`, `test_invoke_agent_python_env.py`,
  `test_invoke_bash_cwd_reset.py`, `test_landlock.py`, `test_mcp_lifecycle.py` (2),
  `test_sandbox_git_config.py`; `shared/tests/test_cli_invoke_agent_env.py` (2),
  `test_wildcard_deny_live_inventory.py`. No production default is below 0.25 (critic's scan of 20
  project yamls, `~/.config/orchestrator/*.yaml` and non-test `.py`).
- **Delivered-check construction (M7).** `DeliveredCheckMeta(` is constructed in production at
  `orchestrator/src/orchestrator/delivered_checks.py`, `fused-memory/src/fused_memory/server/manifest_stamping.py`
  and `scripts/audit_manifest_descriptor_drift.py` (2); in tests at `shared/tests/test_capability_manifest.py`
  (19) and `fused-memory/tests/server/test_write_triage_flip_gate_invariants.py` (2). The slice is
  registered by `register_metadata_submodel('delivered_checks', DeliveredCheckMeta, cardinality='list')`,
  which requires a `BaseModel` class.
- **Importers of moved names (m13).** Count rule: a file counts when an AST scan finds
  `from shared.cli_invoke import …` naming a name that leaves `cli_invoke`, `from shared.usage_gate
  import …` naming a name that leaves `usage_gate`, or `import shared.cli_invoke`/`shared.usage_gate`;
  patch-target strings alone do not count. Test files: shared 51, orchestrator 62 (34 of them declared by
  open wave-2 tasks), fused-memory 23 (6 declared by wave-2 tasks), scripts 4, root `tests/` 1.
  Under the critic's regex rule (`from shared\.cli_invoke import|import shared\.cli_invoke`, any name)
  the counts are 46/65/25/6, which this fold reproduces. Production files: orchestrator 17, fused-memory
  8, scripts 4, `fused-memory/scripts` 1, shared 4 besides the two modules themselves (34 in all, 12 of
  them declared by open wave-2 tasks).
- **Patch strings of moved names (m10).** `'shared.cli_invoke.<name>'` strings for names that leave
  the module: outside `shared/tests`, `orchestrator/tests/test_liveness_boundary_gate.py` patches
  `terminate_process_group` (3), 4 orchestrator files patch `_run_subprocess` (S19's runner leaf
  removes those), and fused-memory `test_agent_loop.py`, `test_judge.py`, `test_task_curator.py` patch
  `invoke_claude_agent`. (`'shared.cli_invoke.asyncio.…'` patches, in 20 test files, reach the shared
  `asyncio` module object and keep working after the move; θ moves the shared ones onto `clock=`.)
- **A live cap ledger (fold discovery).** Task **8253** (pending, high; Leo's 2026-11-02 requirement for
  the nightly preparer) gives `UsageGate` an opt-in on-disk cap ledger (`cap_ledger_path`, default
  `None`, orchestrators unchanged unless set) that seeds CAPPED accounts at construction. **8252**
  (pending, high, after 8253) routes the preparer through the shared failover. Both declare
  `shared/src/shared/usage_gate.py`. (Ruling-fold note: task 6214, done 2026-11-10 `3e5b2bafd3`,
  already routes the preparer through `invoke_with_cap_retry` with whole-pool failover; what 8252 and
  8253 still own is the 8253 investigation's question, see `## Dependence on task 8253`.) **3986** (pending, high, no dependencies) records resumes in
  `cost_store.py` and `cli_invoke.py`. **8563** (pending, high, no open ancestors) declares
  `orchestrator/src/orchestrator/digest.py` and `orchestrator/tests/test_digest.py`.

### Re-measurement (ruling fold, measured at `1b106418da`)

`git log --oneline 97be1d71fa..1b106418da` over the plan's 240 declared paths: 57 commits touching 23
paths. Contract §8 step 1 still returns no task for any of the 19 keys (re-run 2026-11-12). What moved:

- **`shared/cli_invoke.py`** is 4,945 lines (was 4,911); `invoke_with_cap_retry` spans 1,062 lines (was
  1,043); `_run_subprocess` 557 and `classify_agent_failure` 288 (unchanged). Task **3985** (done,
  `d92a0434b9`) added `PROGRESS_TIMEOUT_RESUME_PROMPT` and a `resume_prompt` keyword (the continuation a
  caller-preset resume sends; mutually exclusive with `resume_delivers_prompt`). The loop now holds three
  resume prompts; the third stays in `cli_invoke` with the other two (D7), and the decision table (C7) is
  unaffected: `resume_prompt` only chooses the first dispatch's prompt. Function-local first-party
  imports: still 6 of `shared.invocation_outcome` and 1 of `shared.usage_gate`.
- **`shared/usage_gate.py`** is 3,200 lines (was 3,136). Task **9327** (done, `813cb4d8ed`,
  `65284b3e96`) bounds a probe claim's hold: `AccountState.probe_hold_timer`, armed by `_transition` on
  entering PROBE_IN_FLIGHT and disarmed on leaving it; `UsageGate._expire_probe_hold` reopens the account
  through `_transition` after `UsageCapConfig.probe_slot_hold_secs`; `probe_in_flight_account_names`
  joins the park heartbeat. It is a backstop for a holder that never reports, and it already writes
  `phase` only through `_transition`, so it fits D11; `_expire_probe_hold`'s generation guard cites the
  legacy setters ι deletes, and ι rewords that comment. `getattr(self, '_…')` default reads: 12 (was 11;
  `_arm_probe_hold` adds one), which ο3 deletes with the rest.
- **Loop callers that pass a cost store.** Still 15 production `invoke_with_cap_retry(` calls in 13
  files (AST scan); 5 pass `cost_store=` by name. Two more reach the loop's row writer through a kwargs
  helper: `fused_memory/middleware/task_curator.py::TaskCurator._ledger_kwargs` (its injected `invoke`,
  which the fold's scan missed) and, since task **6541** (done, `439fec35e0`),
  `fused_memory/middleware/path_scope_adjudicator.py::PathScopeAdjudicator._ledger_kwargs`. Both get
  per-attempt rows from η with no change of their own. The judge, recon stages, evals, legibility runner
  and nightly preparer still write no row.
- **A fifth backend.** Task **5399** (done) added `invoke_agent(backend='zcode')`
  (`orchestrator/src/orchestrator/agents/zcode_backend.py`), routed through `invoke_with_cap_retry` and
  outside `ACCOUNT_POOL_BACKENDS = {'claude', 'pi'}`, so today it leases a Claude account like codex and
  gemini; D1's account-less slot covers it without a new rule. `zcode_backend.py` imports
  `AgentResult` and the private `_SubprocessResult` from `cli_invoke` and mints `'error_empty_output'`,
  `'success'` and a default `'error'` subtype; `orchestrator/tests/test_zcode_backend.py` imports
  `_SubprocessResult`. Neither file is declared by an open task.
- **Importers of moved names** (count rule as in the fold pass): test files shared 51, orchestrator 63,
  fused-memory 23, scripts 4, root `tests/` 1 (142); production files orchestrator 18, fused-memory 8,
  `fused-memory/scripts` 1, scripts 4, shared 4 besides the two modules (35). 177 in all, 53 of them
  (41 test, 12 production) declared by open wave-2 tasks. Orchestrator changes since the fold:
  `orchestrator/tests/test_reviewer_note_dispositions.py` no longer imports a moved name (task 5227 moved
  its fixtures to `orchestrator/tests/_info_l0_router_fixtures.py`, which imports `AgentResult` and is
  declared only by pending low task 9340); `test_zcode_backend.py` and `zcode_backend.py` are new.
- **Patch strings of moved names.** Task 6541 added two `'shared.cli_invoke.invoke_claude_agent'`
  patches to `fused-memory/tests/test_path_scope_adjudicator.py` (a ξ2 file). Other counts unchanged:
  `shared.cli_invoke.invoke_claude_agent` 70 sites in 6 shared test files, `shared.cli_invoke.asyncio.sleep`
  55 in 13, `shared.cli_invoke.count_transcript_turns` 19, `read_transcript_records` 8.
- **Slot verdict methods called directly by tests.** `slot.report(` / `.confirm(` / `.settle(` in 63
  places in 16 test files, `.detect_cap_hit(` 121 in 14 (unchanged). Re-checked against every leaf's file
  list, four files call `slot.report` on a real gate and were in no leaf: `shared/tests/test_auth_reprobe_opt_out.py`,
  `shared/tests/test_usage_gate_account_names.py`, `shared/tests/test_usage_gate_probe_hold.py` (new,
  task 9327) and `fused-memory/tests/test_curation_deferral_acceptance.py`; `shared/tests/test_usage_gate_park_visibility.py`
  was only in ι, whose work is the setters. ο3 deletes `InvokeSlot.report`, so the first three and
  park-visibility join ξ1 and the fourth joins ο1.
- **Subtype literals.** `ERROR_*`/`TEXT_OUTPUT` values spelled in 8 production files (zcode_backend.py
  added); bare `'error'` on 719 lines in 142 files, `'success'` on 90 lines in 34 files.
- **Unchanged and re-confirmed:** test reach counts (`gate._*` 384/187/137/358/118/228 in the six largest
  suites; 288 setter writes in 14 files; 41 `_probe_config_dir` references), the ten sub-floor test budgets
  in eight files (`orchestrator/tests/test_landlock.py` still at its old path: S19 ζ has not moved it),
  `invoke_slot(scope=…)` 11 lines (10 test calls plus `shared/testing.py`), `invoke_slot()` 72 lines in 22
  files (+7 from task 9327's new suite), the `config_dir.sweep_stale_pid_dirs_once` `sweep` parameter and
  both interception-point tests (task 6615 replaced only the private `_pid_alive` copy), both
  `save_invocation` writers outside the loop (`workflow.py::TaskWorkflow._invoke`,
  `review_checkpoint.py::ReviewCheckpoint._run_review`), and `workflow.py::_capture_zero_output_evidence`'s
  nesting walk.

### Re-measurement (fold-verify, measured at `0f21f5e457`)

- `git log 1b106418da..0f21f5e457` (73 commits) touches five of the plan's paths: `workflow.py`,
  `scheduler.py`, `event_store.py`, `agents/briefing.py` (task 3090, scope requests) and one line pair
  of `orchestrator/tests/test_workflow_prompt_builder.py`. No anchor moved: `workflow.py` and
  `review_checkpoint.py` still hold the only two `cost_store.save_invocation(` calls outside the loop,
  `briefing._DELIVERED_CHECK_FIELDS_BY_KIND` is untouched, and the prompt-builder suite still configures
  `detect_cap_hit` once. `cli_invoke.py` and `usage_gate.py` are unchanged.
- Re-counted by grep: the four hand-written gate fakes declaring `invoke_slot(…scope=None…)` / `_cm(scope=None…)`
  (D12) are exactly `shared/tests/test_cli_invoke_sandbox_wrap.py`, `orchestrator/tests/test_dry_run_unblock.py`,
  `orchestrator/tests/test_steward.py` and `orchestrator/tests/test_invoke.py`; no other test defines
  `invoke_slot` or `_cm`. The five files whose direct `slot.report(` calls no leaf owned before the ruling fold
  hold 6 calls (4 shared files 5, `test_curation_deferral_acceptance.py` 1). All
  `slot.report|confirm|settle(` calls in test `.py` files: 63 in 16 files, 12 of them shared test files
  (the set ξ1's check names), 2 fused-memory (ξ2, ο1) and 2 orchestrator (κ2/ξ3, ξ4).
- Lead ruling of 2026-11-12 (test-file edges, restating R54): an edge that rests only on a shared test file
  is not wired. Applied in this pass to every edge of the plan whose only shared file is a test file (see
  the DAG paragraph); every pair of S20 leaves that shares a non-test file stays ordered (re-checked).
- Task store (read 2026-11-12): **8252 cancelled** and **8253 rescoped** to the return-brief trial windows
  (priority low, depends only on 8563, no longer declares `usage_gate.py`) by Leo's ruling on the 8253
  investigation; 3808, 3987, 3993, 4277 and 5062 (repath list) are deferred, not pending; 9170 is done; every
  other task the plan names has the status stated where it is cited.

### Weakened keys (R23), re-verified under G3/G6

- **fk-734bc1fc18bf (scratch root) — kept, low.** The skeptic's points hold for UsageGate-built
  suites (`df_pytest_isolation.py::arm_usage_gate_sweep_confinement` now confines them in four test
  roots, task 6275). The cost the finding names is still present for every other owner: the
  required `sweep` parameter, the two interception-point tests, three callers importing
  `sweep_stale_pid_dirs` only to hand it over, and live task 8234, filed because the curator and
  session-runner sweeps reach the real tempdir unless a module-local patch is remembered. R18 rules
  the remedy ("scratch root threaded as a parameter"). Leaf λ.
- **fk-240df0521f83 (check kinds as a discriminated union) — kept, low.** The rejection matrix in
  `shared/tests/test_capability_manifest.py` does catch a forgotten prohibition for an existing field,
  so the "silently permissive" half is weak. The remaining cost is real and measured: the kind set is
  restated in two orchestrator modules and one copy has already drifted (task 7376). The union makes
  each kind's fields the matrix and lets both consumers dispatch on the model type. Leaf μ, after
  S10 η (task 9240) moves the validator to `shared/delivered_check.py`.

## Findings disposition

One closing leaf per key — the last leaf that touches it; every other leaf listed contributes.

| Key | Sev | Disposition | Note |
|---|---|---|---|
| `fk-1778eb482209` | medium | leaf ο2 (contributors β, ξ2, ο3) | β: `CliSubtype` beside `AgentResult`, parse at the boundary, shared consumers and the spelling guard; ξ2 scripts adopt members; ο2: the orchestrator mint and compare sites; ο3 empties the guard's allow-list. |
| `fk-e59c30a25d27` | low | leaf ο2 (contributor γ) | γ: one nesting rule (`transcript.content_blocks`) and a public locator; ο2 moves `workflow._capture_zero_output_evidence` onto it. |
| `fk-734bc1fc18bf` | low | leaf λ | Weakened; re-verified above. |
| `fk-240df0521f83` | low | leaf μ | Weakened; re-verified above; behind S10 η (9240) and θ (9242). |
| `fk-dcc5dce60bb6` | high | leaf η | One classification per attempt; the heuristic net and the side predicates become rows of one decision table. |
| `fk-b8cba955b358` | medium | leaf ο2 (contributors ζ, η) | ζ: per-attempt row shape and reader filters; η: one `cost_store` write and one session-cost accumulation per attempt for callers that pass a cost store; ο2: the task workflow and the deep reviewer stop writing their own row and pass theirs into the loop (about 95% of spend). |
| `fk-2a5d97aeb64d` | high | leaf η (contributor θ) | η: attempt → classify → decide → settle; θ moves the runner out. |
| `fk-8805d20c0c4c` | high | leaf ο3 (contributors ε, η) | ε: the probe classifies once and returns a verdict the gate applies; η: the loop stops calling `UsageGate.detect_cap_hit`, the heuristic net and the `getattr(usage_gate, '_config')` reach go; ο3 deletes `UsageGate.detect_cap_hit` (the ruled "second classify goes"). Already fixed in part: the probe's second CLI-JSON parse (task 6112, `a04ae1d162`). |
| `fk-005378a1b462` | medium | leaf η (contributor ε) | Fixed in part before as_of: `_parse_resets_at` returns `None` instead of now+1 h (task 4357, `cb8e52bde9`). ε removes the remaining invented one-hour sleep target; η deletes the heuristic net. The prose tables stay: the reset time exists only in the banner text. The task-8253 investigation (2026-11-12) found that the CLI's result JSON does carry `api_error: "usage_limit_reached"`, which `parse_claude_cli_output` drops (it keeps `api_error_status` only); the new low task π (after η) adds the field and lets the classifier read it (heuristic 12, structured data instead of meaningful strings), with no key of its own (a bare 429 still classifies `Failure`, corpus row `auth_status_429_alone_is_failure`). `shared.cap_markers` remains the documented separate loose contract. |
| `fk-49b55c1270ae` | medium | accepted (ω), ruled by Leo 2026-11-12 (S20-1, option A) | Per-process cap state for the long-running orchestrator and fused-memory processes. Basis: the measured cost, one instant failed dispatch and a 5 s cooldown per process per cap (87 of 96 caps rediscovered, median 6 processes; `_CAP_HIT_COOLDOWN_SECS = 5.0` at `1b106418da`). Acceptance record ω, filed and cancelled with `x_acceptance_reason`. The record also states that no process, one-shot included, has a cap file: Leo's ruling on task 8253 (2026-11-12) rejected one, and one-shot processes rediscover caps too, at one free failed start per capped seat. |
| `fk-377f5a42e8be` | medium | leaf ε | Purpose split: vocabulary, probe and scope overlay leave `UsageGate`; the probe returns a verdict. |
| `fk-25c96d016239` | high | leaf ο3 (contributors ι, κ1, κ2) | ι: public test seam, setters and alias go; κ1/κ2: the six largest suites move onto the seam, ratchet on the rest; ο3: the `__new__` builder and the class-level defaults go. |
| `fk-f7f1747302dc` | low | leaf ε | Heuristic-14 partition of `usage_gate.py`, measured at landing. |
| `fk-db060c262575` | medium | leaf ι (contributors ε, η) | ε: `_transition` writes `resets_at`; η: probe-slot release in one place (`observe`); ι: the four setters go. |
| `fk-ae6cd7600d0e` | medium | leaf ε | Loop half fixed by `6ec56f30e0` (task 6690). ε routes the account-level and scoped writes through one cap-event writer and gives `resumed` the same writer (task 6729). |
| `fk-aacd5e9381f6` | medium | leaf ε (contributor β) | β: markers move down, `classify_agent_failure` moves up, the loop's 6 function-local imports go; ε: `scope_for`/`PoolFrozen` import from the vocabulary leaf at module level, the last `cli_invoke` and `usage_gate` function-local imports go. |
| `fk-3ae32b390a81` | medium | leaf ο3 (contributors β, γ, δ, θ, ξ1–ξ4, ο1) | The heuristic-14 partition of `cli_invoke.py`; after θ the file is the state machine plus the transitional re-export block, which ο3 deletes. |
| `fk-fede65204150` | medium | leaf θ (contributor γ) | γ: transcript patches move onto real fixture transcripts and the "do not hoist" block goes; θ: loop suites move onto `invoke_fn`/`clock`. |
| `fk-2f4b8655fb3c` | medium | leaf δ | Floor constant and enforcement in `build_claude_argv`. |

## Resolved design decisions

- **D1 — The slot owns cap detection (R18 option 1).** `InvokeSlot.observe(result) -> Decision` is
  the one place a gated attempt is classified and the gate informed. It calls `classify_invocation`
  once with the slot's backend and `strict_confirm=True`, applies the account transition for the
  outcome, accumulates the attempt's cost into the session budget, settles the slot, and returns
  the decision. The loop never calls a gate handler. `AUTH_FAILED` and `CAPPED` are account-pool
  verdicts: for a backend outside `ACCOUNT_POOL_BACKENDS` (which moves into the account vocabulary in ε;
  today codex, gemini, zcode and mimo, the last landed by task 7993 after the fold-verify) `observe` returns `NOT_ACCOUNT` for every outcome, and `invoke_slot`
  yields an account-less slot — no lease, no park on a frozen Claude pool (lead ruling S20/OQ-8). The
  probe-hold timer task 9327 added stays as the backstop for a holder that never reports; `observe` on
  `SERVED` leaves PROBE_IN_FLIGHT through `_transition`, which disarms it.
- **D2 — The loop is a pure decision table plus an executor.** `decide(decision, attempt, policy) ->
  Step` is a pure function over the `Decision`, the attempt facts (armed resume, retries used,
  backend, gated) and the caller's policy (resume opt-out, patience bounds). `Step` is one of
  `Return`, `RetryFresh(reason)`, `Wait(seconds, then)` (then = fresh or resume-if-eligible) or
  `GiveUp(AllAccountsCappedException)`. The executor runs it. The table is in C7. The two-part resume
  rule (transcript reachable, progress recorded) is one function the executor calls. With
  `usage_gate=None` the executor classifies each attempt itself (C7, the ungated path); that and the
  slot are the only two `classify_invocation` call sites for a dispatched attempt. The prompt
  builder's use for the first dispatch (task 6657) is not absorbed: it changes caller contracts in
  `workflow.py` and `steward.py`, so 6657 → θ and lands on the finished executor.
- **D3 — The heuristic zero-cost net is deleted, not re-homed** (R18). Basis: 0 firings since
  2026-07-30 across ten stores against 1,170 exact cap rows since 2026-10-11, and its last
  dark-factory firings were false positives. An unclassified zero-cost failure is returned to the
  caller as an ordinary failure (`classify_agent_failure` already projects it).
- **D4 — A cap the slot cannot attribute is a returned failure with `capped_reason='account'`.**
  Today only the heuristic net set the unattributed flag; an exact cap whose token resolved to no
  account fell through and was recorded uncapped. One rule now covers both.
- **D5 — A near-cap verdict on a failed result is not a cap.** `observe` annotates the account
  (`near_cap`, `near_cap` cost event) and returns `NOT_ACCOUNT`; the loop returns the failure. This
  closes task 7334's defect. A successful result records no near-cap annotation (unchanged since
  task 6213; nothing reads `near_cap` for a decision).
- **D6 — One `invocations` row per attempt, one writer** (lead rulings S20/OQ-4 and S20/OQ-6). `save_invocation` gains
  `invocation_id` (one id per `invoke_with_cap_retry` call), `attempt` (1-based) and `returned` (true
  only for the attempt whose result the call returns). Additive migration; legacy rows read as
  `returned = 1` with a null `invocation_id`. Readers that sum cost or duration need no change and
  become correct; readers that count invocations or treat a row as "the invocation" filter
  `returned = 1` (leaf ζ, landed before η). The session budget accumulates every attempt's cost
  through `observe` (D1). The loop writes the rows whenever its caller passes `cost_store` (η); ο2
  makes it the only writer for loop-routed calls: `workflow.py::TaskWorkflow._invoke` and
  `review_checkpoint.py` pass `cost_store`, `run_id`, `task_id`, `project_id` and `role` into
  `invoke_with_cap_retry` and delete their own `save_invocation` blocks. Each row carries every field
  those callers write today (resolved `model`, `model_id`, `role`, tokens, `duration_ms`,
  `capped`/`capped_reason` per attempt, `started_at`/`completed_at` per attempt). The curator and the
  path-scope adjudicator already pass their cost store into the loop (`_ledger_kwargs`, measured at
  `1b106418da`), so they get per-attempt rows at η with no change. Evals stay hermetic (the cross-stream
  rule of 2026-11-12): an eval cell's workflow is built by `evals/runner.py::_cell_workflow` with no
  cost store, the eval gate (`evals/eval_gate.py`) holds none, and `evals/measurement.py` and the
  reviewer-trial runner pass none, so moving the writer into the loop opens no path from an eval run to
  `runs.db`; ο2's test pins it.
- **D7 — Module partition of `cli_invoke.py`** (lead ruling S20/OQ-1; order by S20/OQ-7 = S19/Q-C2 (a)),
  following the run's measured heuristic-14 partition, with one change: the loop keeps the `cli_invoke`
  name and the runner leaves, so the loop's 22 production importers do not move and `cli_invoke` stays
  an honest name ("invoke an agent, retrying on caps"). The runner's public names are the
  package-layering plan's (S19 C7, the public CLI runner in `shared.cli_invoke`: `SubprocessResult`,
  `parse_claude_output`, `run_cli_subprocess`, `CliRunner`, `ArgvWrapper`, `TranscriptWatch`, landed in
  S19 ι before β): this plan moves them, unchanged, into its leaves.
  - `shared/cli_result.py` — `AgentResult`, `CliSubtype`, the public `SubprocessResult` and
    `parse_claude_output` (S19 C7), `parse_claude_cli_output`, the outcome predicates
    (`is_zero_output_timeout`, `is_timed_out_with_progress`, `is_server_error_status`,
    `is_cli_invocation_rejected`, `is_resume_rejected`), their stderr marker tables
    (`CLI_INPUT_REQUIRED_MARKERS`, `RESUME_REJECTED_MARKERS`, moved down from
    `invocation_outcome`), `classify_cap_kill`.
  - `shared/invocation_outcome.py` — unchanged role; imports from `cli_result` only.
  - `shared/agent_failure.py` — `AgentFailureKind`, `AgentFailureClass`, `classify_agent_failure`,
    `build_failure_message` (the steward/escalation projection of an outcome; imports
    `invocation_outcome` and `cli_result`).
  - `shared/transcript.py` — the transcript part (C5), stdlib-only.
  - `shared/claude_argv.py` — binary resolution, `build_claude_argv`, spawn env, CPU prefixes,
    stdin materialisation, `require_non_blank_prompt`, the cost floor (C6).
  - `shared/claude_runner.py` — `run_cli_subprocess`, `CliRunner`, `ArgvWrapper` and `TranscriptWatch`
    with S19 C7's signatures, `invoke_claude_agent`, `_invoke_claude`, the watchdog constants,
    `VllmBridge` wiring.
  - `shared/cli_invoke.py` — the state machine (C7), `AllAccountsCappedException`, `PromptBuilder`,
    the three resume prompts (`CAP_HIT_RESUME_PROMPT`, `CRASH_RECOVERY_RESUME_PROMPT`, and
    `PROGRESS_TIMEOUT_RESUME_PROMPT`, added by task 3985).

  Layering — imports flow downward: `invocation_outcome` imports `cli_result`; `agent_failure` imports
  `invocation_outcome` and `cli_result`; `claude_argv` and `transcript` import neither; `account_probe`
  and `scope_caps` import `account_state`; `usage_gate` imports `account_state`, `account_probe`,
  `scope_caps`, `invocation_outcome`, `cli_result` and `claude_argv`; `claude_runner` imports
  `cli_result`, `claude_argv` and `transcript`; `cli_invoke` imports all of them. No module of this
  set imports another function-locally.
- **D8 — Transitional re-exports, governed, deleted by ο3** (lead ruling S20/OQ-3). Moving names out of
  `cli_invoke` and `usage_gate` touches 177 importer files besides the two modules (142 test files,
  35 production files; measured at `1b106418da`), 53 of them (41 test, 12 production) declared by open
  wave-2 tasks that edit them in a fixed per-file order (R54). Each
  move leaf therefore repoints every importer under `shared/src` and leaves a declared re-export block
  in the old module for importers elsewhere. A guard test (`shared/tests/test_cli_invoke_layering.py`,
  `shared/tests/test_usage_gate_layering.py`) pins (a) the re-exported name set, (b) an allow-list of
  the files still importing those names through the old module, and (c) an allow-list of files whose
  patch-target strings name `shared.cli_invoke.<moved name>` or `shared.usage_gate.<moved name>` (such
  a patch silently no-ops once the name moves). The allow-lists may only shrink and each entry names
  its owning leaf. ξ1, ξ2 and ξ3 shrink them in files no wave-2 task holds; ξ4 (test files wave-2 tasks also
  hold, rebase-only) and ο1/ο2 repoint the held files; ο3 empties them and deletes both blocks. This is a
  governed exception (INV-12), not a permanent shim (heuristic 13).
- **D9 — `CliSubtype` keeps unknown values.** `AgentResult.subtype: CliSubtype | str`. `CliSubtype.parse`
  returns the member for a known spelling and the raw string otherwise, so a new CLI subtype is
  carried through untouched and is visible in logs. A `StrEnum` member compares equal to its string,
  so stored `runs.db` rows and JSON events keep their spelling. Members carry their origin
  (`CliSubtype.minted`), so an agent adding a mint site or a wire value knows which it is.
- **D10 — The usage gate keeps its account machine; three parts leave** (heuristic 6 and 14,
  measured partition): the vocabulary (`shared/account_state.py`: `AccountPhase`,
  `_LEGAL_TRANSITIONS`, `IllegalTransitionError`, `ProbeSpawnError`, `ScopeCap`, `AccountState`,
  `AccountLease`, `SessionBudgetExhausted`, `PoolFrozen`, `scope_for`, `ACCOUNT_POOL_BACKENDS`), the
  probe subprocess and its verdict (`shared/account_probe.py`: `run_probe` and the pure
  `probe_verdict` returning a `ProbeVerdict`; it never touches the gate), and the per-scope cap
  overlay as a class (`shared/scope_caps.py::ScopeCaps`, owning the per-scope waiters and last-account
  map, handed the account list). `InvokeSlot` stays in `usage_gate.py` (as a satellite it would reach
  the gate's privates, heuristic 13). The gate keeps `_run_probe(acct) -> bool` as a thin adapter
  (run the probe, apply the verdict through `_transition`, return `verdict.served`) so the ~60
  existing test stubs of that method stay valid until κ1/κ2 move them onto the seam. The probe's
  stale-dir sweep stays in `usage_gate.py` until λ, because the suite-wide confinement intercepts it
  there.
- **D11 — `_transition` is the sole writer of `phase` and `resets_at`, and one method writes the
  `cap_hit` event and its `resumed` dual.** `_transition(acct, CAPPED, resets_at=…)` stores
  `resets_at`; a same-phase repeat detection goes through `_refresh_resets_at(acct, resets_at)` (no
  edge, no event). The scoped branch and the account-level entry call one
  `_record_cap_event(acct, reason, resets_at, scope)`; scoped rows keep their `scope` detail, which
  `dashboard/src/dashboard/data/cap_history.py` already ignores. `_transition` writes one `resumed`
  event on every edge out of CAPPED (probe-confirmed, clock-driven and SIGHUP; CAPPED → AUTH_FAILED
  counts), and the probe loop's own write goes (task 6729, absorbed).
- **D12 — The gate derives the cap scope.** `invoke_slot(*, model: str | None = None,
  backend: str = 'claude', park: bool = True, max_wait_secs: float | None = None)` replaces
  `invoke_slot(scope=None, *, park, max_wait_secs)`; `model=None` means the general scope (today's
  `scope=None`), so the argument-less test calls (72 lines in 22 files at `1b106418da`) keep working,
  and η rewrites the 10 `scope=` test calls. The gate computes `scope_for(model, self._config)` for the
  claude backend. The loop's `getattr(usage_gate, '_config', None)` reach goes. Four hand-written gate
  fakes declare today's signature explicitly (`invoke_slot(self, scope=None, *, park, max_wait_secs)`)
  and are driven through the real loop, so the loop's new call raises `TypeError` in them unless they
  change in the same commit: `shared/tests/test_cli_invoke_sandbox_wrap.py::_FakeGate`,
  `orchestrator/tests/test_dry_run_unblock.py::_ForcedCapUsageGate`, and the `_cm` fakes in
  `orchestrator/tests/test_steward.py` and `orchestrator/tests/test_invoke.py` (found by the ruling
  fold; the fold-pass claim that no file outside η's list changes in η did not hold). η makes each a
  one-line signature edit (keyword-tolerant); ξ1, ξ4 and ο3 later move them onto the C2 shape.
- **D13 — Tests move onto public seams, never onto new private names.** The loop suites pass
  `invoke_fn=` (already public) and a `clock=` (new, C7) instead of patching
  `shared.cli_invoke.invoke_claude_agent` and `asyncio.sleep`; transcript-reading tests write real
  JSONL transcripts under a tmp config dir instead of patching `count_transcript_turns`; gate tests
  use `shared/testing.py::drive_to` and `UsageGate.account_snapshot()` (ι). The runner's
  `create_subprocess_exec` patches have moved onto S19's `runner` seam before θ (S19 ι). Between η
  and ο3, `shared/testing.py::make_gate_mock`'s slot carries an `observe` whose verdict is derived from
  the mock's configured `detect_cap_hit`/`report` behaviour and which records `confirm(cost)`; each
  suite still configuring `detect_cap_hit`, or calling a real slot's `report`/`confirm`/`settle`
  directly, moves to `observe` verdicts or `drive_to` in the leaf that already edits it (θ, κ2, ξ1–ξ4,
  ο1, ο3), and ο3 deletes the bridge with `detect_cap_hit`. The ruling-fold re-check found four direct
  `slot.report` callers no leaf owned and a fifth owned only by ι, whose work is the setters; ξ1 takes `test_auth_reprobe_opt_out.py`,
  `test_usage_gate_account_names.py`, `test_usage_gate_probe_hold.py` and `test_usage_gate_park_visibility.py`,
  and ο1 takes `fused-memory/tests/test_curation_deferral_acceptance.py`.
- **D14 — The cost floor is one constant with one hard enforcement point.**
  `shared.claude_argv.INVOCATION_COST_FLOOR_USD = 0.25` and `build_claude_argv` raises
  `InvocationBudgetBelowFloor` (a `ValueError` naming the floor, the value and the model) for
  `max_budget_usd` below it. Basis (G6): the measured minimal call cost was ~0.105 USD (task 1849,
  Sonnet, base-context cache creation) and has grown since (task 1983); the schema's own guidance is
  "do not lower below 0.25"; the lowest budget any production caller passes is 0.50. The usage-gate
  probe is exempt by construction (its own argv, `account_probe.PROBE_BUDGET_USD = 0.01`, where a
  budget abort means "served"). δ raises the ten sub-floor test budgets to the constant. S16 adds a
  config-load check against the constant (S16 C5, the shared routing policy's `validate_role_models`,
  which refuses at load and hot reload a Claude role whose budget is below the floor; S16's leaf that
  carries it depends on δ).
- **D15 — Scratch root is a parameter with one public default.** `shared.config_dir.default_scratch_root()`
  returns `Path(tempfile.gettempdir())`; `UsageGate(..., scratch_root=None)`, the curator and the
  session runner resolve `None` through it at call time (never at import) and pass the root to
  `TaskConfigDir(base_dir=…)` and `sweep_stale_pid_dirs_once(prefix, base_dir=…)`, whose `sweep`
  parameter goes (it calls `sweep_stale_pid_dirs` itself). `df_pytest_isolation.py` confines every
  owner at once with `monkeypatch.setattr(tempfile, 'tempdir', <per-test root>)`, the standard
  library's own knob — no module-name patch; a test that needs a specific root passes `scratch_root=`.
- **D16 — Delivered-check kinds are one model per kind.** `GrepCheck`, `ScriptCheck`, `PathCheck`
  (mechanical) and `ManualCheck`, each `extra='forbid'`. `DeliveredCheckMeta` stays a `BaseModel`: a
  `RootModel[Annotated[GrepCheck | ScriptCheck | PathCheck, Field(discriminator='kind')]]`, registered
  in S10's `SLICES` table under its slice-model rule (INV-16, task 9242). A public
  `parse_delivered_check_meta(mapping)` returns the member model, and the four production
  construction sites and the tests call it. `DeliveredCheck` is the same shape over the meta members
  plus `ManualCheck`; `MECHANICAL_CHECK_KINDS` derives from the meta union. The path-entry rule stays
  a field validator on `PathCheck`. The dispatch gate and the briefing renderer dispatch on the model
  type and render the model's declared fields.

## Contracts (B+H)

G5 fires on every count: the stream touches `shared`, `orchestrator`, `fused-memory`, `scripts` and
`dashboard`; it carries more than eight mechanisms; it changes load-bearing seams (account
attribution, cost accounting); and S25, S16, S18, S19 and S22 consume or reshape its seams.

**C1 — `classify_invocation` is the one classifier (S20 owns; signature unchanged).**
```python
def classify_invocation(result: AgentResult, *, strict_confirm: bool,
                        backend: str = 'claude') -> InvocationOutcome   # total, pure, never raises
```
Invariants: precedence is the function's own (AuthFailed > ModelNotFound(404) > OK > ModelNotFound
(marker) > CliLocalError > CapHit/NearCap > ServerError > ZeroOutputWedge > Failure). For a
dispatched attempt it is called exactly once: by `InvokeSlot.observe` when gated, by the loop's
executor when ungated (`strict_confirm=True` both); for a probe exactly once, by
`account_probe.probe_verdict` (`strict_confirm=False`, success withheld, the DD-2 regime). The
remaining callers classify a result they already hold and change no gate state
(`agent_failure.classify_agent_failure`, `routing.classify_probe_outcome`, S18's
`measurement._transport_refusal`). `invocation_outcome` imports nothing from `cli_invoke`.

**C2 — `InvokeSlot.observe` (S20 owns; `shared/src/shared/usage_gate.py`).**
```python
class AccountVerdict(StrEnum):
    SERVED = 'served'                    # OK: account confirmed (PROBE_IN_FLIGHT -> AVAILABLE)
    CAPPED = 'capped'                    # CapHit attributed to the leased account (scope set: that model only)
    UNATTRIBUTED_CAP = 'unattributed_cap'  # CapHit, token resolves to no account: no state change
    AUTH_FAILED = 'auth_failed'          # 401/403: AUTH_FAILED (a CAPPED account stays CAPPED)
    NOT_ACCOUNT = 'not_account'          # every other outcome, NearCap included: account untouched
# AUTH_FAILED and CAPPED are account-pool verdicts: for a backend outside ACCOUNT_POOL_BACKENDS,
# observe returns NOT_ACCOUNT for every outcome.

@dataclass(frozen=True)
class Decision:
    outcome: InvocationOutcome           # the attempt's single classification
    verdict: AccountVerdict
    account_name: str                    # the leased account ('' when none was leased)
    scope: str | None                    # the cap scope the slot was opened for

class InvokeSlot:
    def observe(self, result: AgentResult) -> Decision: ...
class UsageGate:
    def invoke_slot(self, *, model: str | None = None, backend: str = 'claude', park: bool = True,
                    max_wait_secs: float | None = None) -> AsyncContextManager[InvokeSlot]: ...
```
Invariants: `observe` classifies once, applies at most one account transition, adds
`result.cost_usd` to the session cost for every verdict, releases a PROBE_IN_FLIGHT claim on every
path, and settles the slot. A second `observe` on the same slot raises `SlotAlreadySettled` naming the
account and the first verdict (INV-2). A slot exited without `observe` (the prompt builder declined
the retry) releases the claim and accumulates nothing, as today; the probe-hold timer (task 9327)
remains the backstop for a claim whose holder never returns. The stale-lease fail-safe is kept
(warning plus `lease_stale` event, then proceed). `model=None` is the general scope. For a backend
outside `ACCOUNT_POOL_BACKENDS` (codex, gemini, zcode at `1b106418da`) the slot is account-less: no
lease, no park on a frozen or all-capped Claude pool, `account_name=''`. Error semantics: `observe` never raises for any
`AgentResult`. After ο3, `InvokeSlot.report`, `.confirm`, `.settle`, `.detect_cap_hit` and
`UsageGate.detect_cap_hit` do not exist; between η and ο3 they have no production caller.

**C3 — `shared.capability_manifest` sidecar surface (S20 owns; requested by S25, delivered by α).**
```python
SIDECAR_SUFFIX: Final = '.capability-manifest.yaml'     # the one production spelling
def sidecar_path(prd_key: str) -> str: ...               # unchanged: the pure PRD key -> sidecar rule
def sidecar_for(prd_path: object, project_root: str | os.PathLike[str]) -> str | None: ...
    # root-relative POSIX path of the sidecar commit_planning would open, else None:
    # None when prd_path is not a non-empty str, cannot be resolved (embedded NUL -> ValueError),
    # or resolves outside project_root; derivation = sidecar_path(prd_path), then
    # (root / derived).resolve() containment-checked against Path(project_root).resolve()
class ManifestDiscoveryUnavailable(RuntimeError): ...
def tracked_sidecar_paths(project_root: str | os.PathLike[str], *, timeout: float = 30) -> list[str]: ...
    # sorted root-relative paths of every git-TRACKED sidecar (git -C root ls-files -z -- '*'+SIDECAR_SUFFIX);
    # raises ManifestDiscoveryUnavailable on OSError, timeout or rc != 0 -- never [] for a failure
```
Invariants: `sidecar_for` agrees with today's stamper (`manifest_stamping._prd_bound_tasks` step 1
plus `_select_sidecar`'s containment) and today's drift audit
(`audit_manifest_descriptor_drift._manifest_binding` plus `_contained_sidecar_relpath`) on every row
of B1; `sidecar_for` composes `sidecar_path`, so the derivation rule is spelled once; discovery
failure is an exception. Adopting it in the stamper and audits is task 6764's work (S25 ι).

**C4 — `CliSubtype` and the result leaf (S20 owns; `shared/src/shared/cli_result.py`).**
```python
class CliSubtype(StrEnum):
    # CLI wire values (the claude CLI's result.subtype), measured
    SUCCESS = 'success'; ERROR_MAX_TURNS = 'error_max_turns'; ERROR_MAX_BUDGET_USD = 'error_max_budget_usd'
    # minted by this repo
    ERROR = 'error'; ERROR_EMPTY_OUTPUT = 'error_empty_output'
    ERROR_CLI_INPUT_REJECTED = 'error_cli_input_rejected'
    ERROR_TIMEOUT_KILLED_WITH_PROGRESS = 'error_timeout_killed_with_progress'; TEXT_OUTPUT = 'text_output'
    @property
    def minted(self) -> bool: ...
    @classmethod
    def parse(cls, raw: object) -> 'CliSubtype | str': ...   # member, else the raw str, '' for a non-str
@dataclass
class AgentResult: ...                      # unchanged fields; subtype: CliSubtype | str = ''
# also hosted here, moved unchanged from S19 C7 (S19 ι): SubprocessResult, parse_claude_output
```
Invariants: `parse_claude_output` and every mint site produce members. The AST spelling guard
(`shared/tests/test_cli_subtype_spelling.py`) fails on (a) a string literal anywhere in production
code equal to the value of a member whose name starts `ERROR_` (other than `ERROR` itself) or is
`TEXT_OUTPUT`, and (b) any literal equal to any member's value that is compared with, or assigned to,
a `.subtype` attribute or a `subtype=` keyword. Bare `'success'` and `'error'` outside a subtype
position are exempt (they are mostly status, log and JSON keys: 719 and 90 lines measured at
`1b106418da`), and the guard's docstring says so. The guard keeps a shrinking allow-list emptied by
ξ2, ξ3 (`agents/zcode_backend.py`), ο2 and ο3. `CLI_INPUT_REQUIRED_MARKERS` and `RESUME_REJECTED_MARKERS` are defined here and spliced
into `invocation_outcome.NON_CAP_CLI_ERROR_MARKERS`. Consumers: dry_run_unblock, routing, steward,
path_scope_adjudicator (seam table), plus reviewer_failure, usage_gate, invoke.py, zcode_backend.py, scripts.

**C5 — The transcript leaf (S20 owns; `shared/src/shared/transcript.py`, stdlib-only).**
```python
def transcript_path(config_dir: Path, session_id: str) -> Path | None: ...  # public locator; never raises
def transcript_exists(config_dir: Path, session_id: str) -> bool: ...
def read_transcript_records(config_dir: Path, session_id: str) -> list[dict] | None: ...
def content_blocks(record: object) -> list | None: ...
    # THE nesting rule: record['message']['content'] when it is a list, else record['content'] when
    # it is a list, else None (unrecognised shape -- callers decide what None means)
# unchanged public detectors: count_transcript_turns, detect_ended_awaiting_background,
# detect_transcript_model_id, detect_resumable_progress, resumable_progress_for_session,
# ended_awaiting_background_for_session, transcript_model_id_for_session, TranscriptEvidence,
# transcript_evidence, transcript_evidence_for_session, note_unreadable_transcript
```
Invariants: `content_blocks` is the only place in `shared`, `orchestrator` and `fused-memory` that
knows the nesting; `detect_resumable_progress` treats `None` as ambiguous (resume); the module is in
`PURE_STDLIB_LEAVES`; `transcript_archive` imports `transcript_path` at module level.
`scripts/legibility/detectors.py::message_content` is S21's legibility detector layer and out of scope.

**C6 — Invocation cost floor (S20 owns; `shared/src/shared/claude_argv.py`).**
```python
INVOCATION_COST_FLOOR_USD: Final[float] = 0.25
class InvocationBudgetBelowFloor(ValueError):   # message names floor, value and model
    floor_usd: float; budget_usd: float; model: str
```
Invariant: `build_claude_argv(max_budget_usd=b, …)` raises for `b < INVOCATION_COST_FLOOR_USD`
before building anything. The account probe does not pass through `build_claude_argv`. Consumer:
`build_claude_argv` and its callers (every claude-backend invocation). S16 imports the constant for its
config-load budget check (S16 C5, the shared routing policy's `validate_role_models`); the S16 leaf
that carries it depends on δ.

**C7 — The cap-retry state machine and per-attempt rows (S20 owns; `shared/src/shared/cli_invoke.py`, `shared/src/shared/cost_store.py`).**
```python
async def invoke_with_cap_retry(usage_gate, label, *, …existing keywords unchanged…,
                                clock: RetryClock | None = None, **invoke_kwargs) -> AgentResult
@dataclass(frozen=True)
class RetryClock:
    monotonic: Callable[[], float]; sleep: Callable[[float], Awaitable[None]]
    # default: time.monotonic, asyncio.sleep
def decide(decision: Decision, attempt: AttemptFacts, policy: RetryPolicy) -> Step: ...   # pure
async def save_invocation(…existing…, invocation_id: str | None = None, attempt: int = 1,
                          returned: bool = True) -> None     # cost_store; columns added additively
```
Decision table (`decide`, first matching row; `armed` = a resume is armed for this attempt; `gated`
= a usage gate was passed). Under the design invariant `one-decision-partition-per-area` (INV-23,
minted by the scheduler plan S17; Leo's principle of 2026-11-12), this decision area is
wholly deterministic: every row is decided by code, in this one pure function, from facts the
classifier and the slot already hold, and no row hands anything to an LLM. The human escape is
unchanged: `GiveUp` raises `AllAccountsCappedException` (or the patience bound fires), which each caller
turns into a blocked task or an escalation as today.

| # | Condition | Step | Decided by |
|---|---|---|---|
| 0 | outcome `AuthFailed`, backend outside the account pool | `Return` | code (`decide`) |
| 1 | verdict `AUTH_FAILED` | `RetryFresh('auth failover')` — no cooldown, not a cap hit | code (`decide`) |
| 2 | outcome `ModelNotFound` | `Return` | code (`decide`) |
| 3 | `is_cli_invocation_rejected(result)` and retries used < 1 | `RetryFresh('pre-turn CLI rejection')` | code (`decide`) |
| 4 | outcome `ZeroOutputWedge` and `armed` | `RetryFresh('wedged resume cleared')` | code (`decide`) |
| 5 | `gated`, outcome `CapHit`, backend outside the account pool | `Wait(window wait, then=fresh)`; counts a cap hit; no Claude account touched | code (`decide`) |
| 6 | verdict `CAPPED` | `Wait(exponential cooldown, then=resume-if-eligible)`; counts a cap hit; patience bounds checked (past them: `GiveUp`, the human escape) | code (`decide`) |
| 7 | outcome `ServerError` | `Return` | code (`decide`) |
| 8 | `armed` and `is_timed_out_with_progress(result)` | `Return` | code (`decide`) |
| 9 | `armed` and `is_resume_rejected(result)` | `RetryFresh('resume rejected')`; records the dropped session id | code (`decide`) |
| 10 | otherwise (`SERVED`, `UNATTRIBUTED_CAP`, `NOT_ACCOUNT` incl. NearCap) | `Return` | code (`decide`) |

The account verdicts that feed it (C2's `AccountVerdict`) are likewise decided by code, in
`InvokeSlot.observe`, from the single classification.

The ungated path: with `usage_gate=None` the executor classifies once per attempt via
`classify_invocation(result, strict_confirm=True, backend=…)` and builds
`Decision(outcome, verdict=NOT_ACCOUNT, account_name='', scope=None)`, so rows 0, 2–4 and 7–10 apply
and rows 1, 5 and 6 cannot match. No ungated production caller arms a resume (re-measured at
`1b106418da`), so rows 4, 8 and 9 add no behaviour on today's ungated paths. The ungated path writes
per-attempt rows like the gated one.

Invariants: one `classify_invocation` call and one `save_invocation` call (when `cost_store` is
passed) per dispatched attempt; all attempts of one call share `invocation_id`; exactly the returned
attempt has `returned = True`; `capped_reason` is per attempt (`account` for `UNATTRIBUTED_CAP`, else
`classify_cap_kill`). After ο2 no caller of `invoke_with_cap_retry` writes an `invocations` row of its
own (lead ruling S20/OQ-6). No eval path passes a cost store into the loop (D6). The public signature
is otherwise unchanged, `resume_prompt` (task 3985) included, or as pending task 9353 leaves it if it
lands first (S18's `measured_call` and every caller keep working). `AllAccountsCappedException` and `PoolFrozen` semantics are unchanged.

## Boundary-test sketch

| # | Scenario | Preconditions | Postconditions | Leaf |
|---|---|---|---|---|
| B1 | Sidecar path agrees with stamper and drift audit | `prd_path` table: plain, `./`-prefixed, `..`-bearing in-root, escaping root, absolute in-root, `.prd.md` stem, non-str, NUL-bearing | `sidecar_for` equals the value today's `_contained_sidecar_relpath(root, _manifest_binding-derived)` returns, row by row (expected values pinned from main) | α (6764 re-runs it against the consumers) |
| B2 | Discovery failure is an exception | a tmp dir that is not a git checkout; a git fixture with two tracked and one untracked sidecar | raises `ManifestDiscoveryUnavailable`; the fixture returns the two tracked paths, sorted | α |
| B3 | The cycle is gone | fresh interpreter | `import shared.invocation_outcome` does not import `shared.cli_invoke`; an AST scan finds no function-local first-party import in `cli_result`, `invocation_outcome`, `agent_failure` and `cli_invoke` (β); the scan widens to the whole D7 set, `usage_gate` included, at ε | β (widened by ε, re-asserted by θ) |
| B4 | Unknown subtype survives; the guard bites only subtype spellings | CLI JSON with `"subtype": "error_new_kind"`, and one with `error_max_budget_usd`; guard fixtures spelling `'error_max_budget_usd'` in a dict, `subtype='error'` in a call, `status == 'error'` in a comparison | the first `AgentResult.subtype == 'error_new_kind'` (a `str`), the second `is CliSubtype.ERROR_MAX_BUDGET_USD`; the guard fails on the first two fixtures and passes the third | β |
| B5 | One nesting rule | records: nested list, flat list, nested non-list plus flat list, neither | `content_blocks` returns the list, the list, the flat list, `None`; `detect_resumable_progress` resumes on the `None` record | γ |
| B6 | Producer side: one classification per attempt | real gate, `invoke_fn` fake scripting cap → resume-rejected → success; a spy wrapping `classify_invocation` | exactly 3 classify calls, one per attempt, all from `observe`; no call from any other site | η |
| B6b | The ungated path classifies once per attempt too | `usage_gate=None`, real `CostStore`, `invoke_fn` scripting a CLI-rejected result then success | 2 classify calls (from the executor) and 2 `invocations` rows sharing one `invocation_id` | η |
| B6c | An unmigrated mock suite survives η | `orchestrator/tests/test_steward_prompt_builder.py`, unchanged | passes at η through `make_gate_mock`'s `observe` bridge | η |
| B7 | Producer side: `observe` verdict table | gate with two accounts; one result per outcome variant, scoped and unscoped CapHit, unresolvable token; a codex 401, a codex cap and a zcode failure | verdicts per C2; account phases and `cap_hit` rows as listed; session cost grew by every result's cost; the codex and zcode results give `NOT_ACCOUNT` with no lease and no Claude account touched; a second `observe` raises `SlotAlreadySettled` | η |
| B8 | Consumer side: decision table | each row of C7 as a `(Decision, AttemptFacts, RetryPolicy)` value | `decide` returns the row's `Step`; rows 0–9 are mutually exclusive except one synthetic overlap (stderr carrying both the input-required and the no-conversation marker), which row 3 takes, pinned by its own case; row 10 is the fallback | η |
| B9 | Every attempt is accounted | real `CostStore` on a tmp db; the B6 script with costs 0.0, 0.0, 1.25, called with `cost_store` | 3 `invocations` rows, one `invocation_id`, attempts 1..3, `returned` only on the third, `SUM(cost_usd) == 1.25`; the dashboard invocation count for the window is 1 | ζ (readers), η (writer) |
| B10 | No heuristic cap | zero-cost, 1-turn, 300 ms failure with no cap text on a healthy account | returned to the caller; no `cap_hit` row; the account stays AVAILABLE | η |
| B11 | Probe verdict is pure and the gate applies it | `probe_verdict` over: cap text, budget abort, success, 403 body, empty | `ProbeVerdict` per row; an AUTH_FAILED account whose probe sees cap text moves to CAPPED through `_transition` with the parsed `resets_at` stored there | ε |
| B12 | One cap-event writer and its dual | an unscoped cap, a repeat detection on the CAPPED account, a scoped cap; the CAPPED account then leaves CAPPED by the clock, and another by SIGHUP | `cap_hit` rows: one account-level (no repeat row), one with `scope`; one `resumed` row per exit from CAPPED; `cap_history` pairs every `cap_hit` with a `resumed` | ε |
| B13 | Cost floor | `build_claude_argv(max_budget_usd=0.10, model='sonnet', …)`; `0.25`; the probe | `InvocationBudgetBelowFloor` naming 0.25, 0.10 and `sonnet`; 0.25 builds; the probe argv still carries `--max-budget-usd 0.01` | δ |
| B14 | Public gate seam | `drive_to(gate, 'a', AccountPhase.CAPPED)` vs observing a real cap result | identical `account_snapshot()`; `AccountState` has no `capped` setter; a `__new__`-built gate fails loudly | ι (the `__new__` half in ο3) |
| B15 | Scratch root is honoured | `UsageGate(config, scratch_root=tmp_path)` with a planted dead-pid probe dir under the real tempdir and one under `tmp_path`; a second gate with `scratch_root=None` under `df_pytest_isolation`'s `tempfile.tempdir` confinement | probe dirs created under `tmp_path`; only the `tmp_path` dead dir is removed; the second gate's dirs land under the confined root | λ |
| B16 | One model per check kind | each kind with each foreign field; a path check in a briefing; `parse_delivered_check_meta` on each kind | rejected by the kind model's own `extra='forbid'`; the parser returns the member model; the briefing renders `expect` and `paths` | μ |
| B17 | Transitional re-exports are governed | a fixture importer of `AgentResult` via `shared.cli_invoke` outside the allow-list; a fixture test patching `'shared.cli_invoke.terminate_process_group'` outside the patch allow-list | the layering guard fails naming the file, the name and its home; after ο3 the allow-lists and re-export blocks do not exist | β, ε, θ (emptied by ο3) |
| B18 | One writer of loop-routed cost rows | a `TaskWorkflow._invoke` run whose first implementer attempt caps and whose resume succeeds, real `CostStore` | 2 `invocations` rows, both from the loop, sharing an `invocation_id`, with the resolved model and `role='implementer'`; no row written by `workflow.py` | ο2 |
| B20 | The structured usage-limit field is a cap | a CLI result JSON carrying `"api_error": "usage_limit_reached"` with banner text that matches no `CAP_HIT_PREFIXES` entry; the same with a parseable reset banner; one without the field and without cap text | `AgentResult.api_error == 'usage_limit_reached'`; the first classifies `CapHit` with `resets_at=None`, the second `CapHit` with the banner's reset time, the third unchanged (`Failure`) | π |
| B19 | Evals stay hermetic after the writer moves (cross-stream rule of 2026-11-12) | a workflow built by `evals/runner.py::_cell_workflow` with a scripted `invoke_fn` and the gate `evals/eval_gate.py` builds | `invoke_with_cap_retry` receives `cost_store=None`; the eval gate holds no cost store; no `invocations` or `account_events` row is written anywhere | ο2 |

## Pre-conditions for activating

- β waits on the package-layering plan's public CLI runner (S19 ι, S19 C7: the public runner in
  `shared.cli_invoke`) (S20/OQ-7 = S19/Q-C2 (a)), on task 3986 (resume observability, same files
  `cli_invoke.py` and `test_cli_invoke.py`); both are free of task 5040 (the
  verify-extraction work most merge-lane tasks wait on) and of 8528. ζ waits on 3986 and 8563.
- No other wave-2 dependency blocks α–θ, ι, κ1, κ2 or ζ: they touch `shared/` and a few
  uncontended files. Where a wave-2 task holds one of their non-test files it now waits on the S20
  leaf instead (lead edges 9162, 9231 → ζ; 9218 → λ), measured cycle-free. Where a wave-2 task holds
  only a test file an S20 leaf edits (9232, 9233, 9234 with δ's budget raises; 9165 with θ's patch
  renames; 9187, 9217, 9232, 9288, 9190, 9229, 9287, 9233 with η's one-line fake edits; the holders
  behind 9236 with ξ4), no edge is wired: test files are rebase-only under the scheduler's file locks
  (lead ruling of 2026-11-12, restating R54). None declares
  `orchestrator/src/orchestrator/merge_lane/worker.py`, so none is parked behind 8528 (R55).
- **S10 η (task 9240)** moves the delivered-check validator into `shared/delivered_check.py` and θ
  (task 9242) mints INV-16; μ depends on both (S10 C3). 9240 also edits `capability_manifest.py`; it
  waits on α (lead edge), which is small and free.
- **S10 δ (task 9204)** edits `fused-memory/src/fused_memory/middleware/task_curator.py`; λ depends
  on it.
- **S16's `Backend` StrEnum** (S16 C1: one `Backend` StrEnum and its one decoder `parse_backend` in
  `shared/backends.py`, landing in S16's role-and-backend vocabulary leaf γ0) does not exist yet: ν is a
  held leaf (`x_seam_wait: {stream: S16, contract: "C1"}`), released by S16's decompose session. If S16
  γ0 already has a task id at S20's decompose, ν is filed as an ordinary dependency on that id, with no
  `x_seam_wait`. S16 is folded and fold-verified (2026-11-13), so its labels are final: C1 is widened in
  place with `ZCODE` and `MIMO` (tasks 5399 and 7993), both outside the account pool, and
  `AGENT_CLI_BACKENDS` is the six CLI members.
- **ο1, ο2 and ο3** wait on each held non-test file's last wave-2 task (see `## Cross-PRD
  relationship`); several of those wait on task 5040 (R31) and on 8528, so ο2 and ο3 land late by
  design (D8). ξ4's files are all test files, so it no longer waits on S8's integration gate 9236.

## Cross-PRD relationship

Edge notation: "X → Y" means X depends on Y (Y lands first). Test files take no edge: an edge that
would rest only on a shared test file is not wired, and test files are rebase-only under the
scheduler's file locks (lead ruling of 2026-11-12, restating R54). For a non-test file two wave-3
plans both edit, check whether a dependency path already exists between the two leaves in either
direction, through the explicit cross-plan edges below. If one does, add nothing. Otherwise the leaf
of the plan decomposed second depends on the last leaf of the plan decomposed first that edits the
file, except where a row below states a direction. The test files listed in the rows below are
recorded for the file locks, not for edges.

| Other stream / PRD | Direction | Seam mechanism / shared files | Owner | Edge / status |
|---|---|---|---|---|
| S25 `plans/review-all-s25-scripts-instruments-prd.md` | produces → | C3 sidecar surface | S20 (α) | release 6764 → α, and 6764 → 9240 (same files as S10 η); 7724 → 6764 (exists); μ → 7724 |
| S25 | shares files | `scripts/audit_model_admission.py`, `scripts/review_model_admission.py` (ζ readers); `scripts/review_model_admission.py`, `scripts/legibility/session_runner.py`, `scripts/sitting/nightly_prepare.py` (ξ2; the last two are also S25 ε 6760's, which follows S25 δ 6759) | — | 6765 → ζ and 6766 → ζ (lead actions); ξ2 → 6760, 6765, 6766 (fold-verify: was 6759, which declares none of ξ2's files; 6760 declares two and depends on 6759); ο2 → 6766 (`routing.py`) |
| S19 package layering (wave 3, concurrent; its C7 is "the public CLI runner in `shared.cli_invoke`", landed by S19 ι) | consumes (S20/OQ-7 = S19/Q-C2 (a), lead) | `SubprocessResult`, `parse_claude_output`, `run_cli_subprocess`, `CliRunner`, `ArgvWrapper`, `TranscriptWatch` move unchanged into `cli_result`/`claude_runner`; same files `shared/cli_invoke.py`, `shared/tests/test_cli_invoke.py` and its test siblings | S19 owns C7 | S20 β → S19 ι (S19/C7); no S19 leaf depends on any S20 leaf; S19 ι deletes `invoke.py`'s four private imports, so ο2's `invoke.py` repoint imports public names only. Since S19's draft, task 5399 added two more private `_SubprocessResult` importers (`agents/zcode_backend.py`, `orchestrator/tests/test_zcode_backend.py`); S19 ι's public rename must cover them (xref), and S20 ξ3 then repoints them to `cli_result` |
| S19 | shares files | `shared/tests/test_landlock.py` (moved there by S19 ζ; δ raises its budget); `shared/tests/test_pure_stdlib_leaves.py` (S19 ζ, θ1, θ2; S20 γ); fused-memory `cli_stage_runner.py`, `task_curator.py`, `judge.py`, their tests (S19 α, γ; S20 ξ2, λ, ο1); `test_liveness_boundary_gate.py` and orchestrator test files (S19 ι, θ1; S20 θ, ξ3, ξ4) | — | Test files take no edge (lead ruling of 2026-11-12): (a) S20's earlier request for S19 θ0, θ1, θ2 → S20 γ on `shared/tests/test_pure_stdlib_leaves.py` is DROPPED, as the lead ruled; (b) δ → S19 ζ is dropped too: δ shares only `test_landlock.py` with ζ, and δ raises its budget at whichever path the file has when δ lands (both paths are in δ's Files); (c) `test_liveness_boundary_gate.py`, `orchestrator/tests/test_invoke.py`, the other S19 ι test files and S19 θ1 ↔ S20 ξ3/ξ4: no edge. Non-test files: S20 β → S19 ι above (`shared/cli_invoke.py`), which also orders θ, ξ3 (`agents/zcode_backend.py`), ο2 (`agents/invoke.py`) and ο3 after ι; S19 α, γ ↔ S20 ξ2, λ, ο1 (`judge.py`, `probe_schema_max_turns.py`, `task_curator.py`, `cli_stage_runner.py`): the rule above; either direction is cycle-free |
| S16 config shape, defaults and routing (wave 3, concurrent, folded and fold-verified 2026-11-13; its C1 is one `Backend` StrEnum with its one decoder `parse_backend` in `shared/backends.py`, landed by its role-and-backend vocabulary leaf γ0) | consumes | C1 `Backend` for `invocation_outcome`'s backend arms, `cli_invoke`'s two `backend` comparisons and `ACCOUNT_POOL_BACKENDS` | S16 owns; S20 converts its own sites | ν → S16 γ0 (held on S16/C1, or an ordinary edge if γ0 has an id). C1 is widened in place: `Backend` carries `ZCODE` and `MIMO` (tasks 5399, 7993), both outside the account pool, and `AGENT_CLI_BACKENDS` is the six CLI members |
| S16 | produces → | C6 floor constant for S16's config-load budget check (S16 C5: the shared routing policy `shared/route_policy.py`, whose `validate_role_models` refuses a Claude role budget below the floor at load and hot reload) | S20 (δ) | S16 δ (route resolver and allowlist move to shared; it carries the check) → S20 δ (S16 records it) |
| S16 | produces → | S16 C9 `shared/model_names.py` (`FABLE_MODEL`, `MODEL_ALIASES`, `canonical_model(name) -> str`), the scoped-cap model's one spelling; S16 π makes `scope_for` (moved to `shared/account_state.py` by S20 ε) match `canonical_model(model)`, editing `scope_for`'s body | S16 owns C9 and π | S16 π → S20 ι (`shared/account_state.py`: ε creates it, ι rewrites it, π edits it after both; S16 records it). S20 ν also edits `account_state.py`; it already follows S16 π through ν → ο3 → ο2 → S16 π (row below, `routing.py`), so no further edge |
| S16 | shares files | S16 ζ6a (workflow iteration caps and review cadence become the workflow config submodel): `shared/cli_result.py` (`is_zero_output_timeout`'s docstring) → S20 β; S16 ε1 (recon agent and judge providers typed `Backend`), ε2a (recon LLM calls receive a validated route), ε2b (curator, adjudicators, write-triage judge and memory LLM take their route from the role table): `reconciliation/judge.py`, `agent_loop.py`, `maintenance/link_adjudicator.py`, `middleware/path_scope_adjudicator.py` and their tests (S20 β, ξ2), `cli_stage_runner.py`, `task_curator.py`, `test_stages.py`, `test_task_curator.py` (S20 λ, ο1); S16 γ1 (orchestrator backend fields typed `Backend`), γ2 (one role-keyed settings table), δ, ζ1–ζ6b (harness, verify, workflow and steward knobs become submodels), π (scoped-cap model spelling): `agents/invoke.py`, `workflow.py`, `harness.py`, `steward.py`, `routing.py`, `review_checkpoint.py`, `dry_run_unblock.py`, `scheduler.py`, `evals/architect_eval.py`, `agents/briefing.py`, `_orch_helpers.py`, `test_invoke.py`, `test_steward.py` and other orchestrator test files (S20 μ, ο2, ο3, ξ3, ξ4); S16 π: `shared/tests/test_usage_gate_scope.py`, `orchestrator/tests/test_routing.py` (S20 ξ1, ξ3). Labels are S16's folded draft (fold-verified 2026-11-13) | — | (a) `shared/cli_result.py`: S16 ζ6a → S20 β only (S16 records it); ζ6a no longer declares `shared/src/shared/cli_invoke.py` (S16's fold dropped it, as S20 asked: its only reference moves to `cli_result.py` in S20 β), so no S16 leaf shares `cli_invoke.py` with S20. S20 π (the structured usage-limit field, added after S16's draft) also edits `shared/cli_result.py` and has no path to or from ζ6a: the rule above applies (either direction is cycle-free). (b) `path_scope_adjudicator.py`, S20 β ↔ S16 ε2b: no edge; ε2b already follows β through S16 δ → S20 δ. (c) fused-memory `judge.py`, `agent_loop.py`, `link_adjudicator.py`, `path_scope_adjudicator.py` and their tests (S16 ε1, ε2a, ε2b ↔ S20 ξ2): S16 ε1, ε2a, ε2b → S20 ξ2; ε2a and ε2b already wait on 5040 through S16 δ; ε1 waits on S20's chain to ξ2. (d) `cli_stage_runner.py`, `task_curator.py`, `test_stages.py`, `test_task_curator.py` (S16 ε1, ε2a, ε2b ↔ S20 λ, ο1): S16 ε2b → S20 λ (ε2b already waits on 5040 through S16 δ; λ and the follow-ups edged onto it stay free of 5040), and S20 ο1 → S16 ε2a (`cli_stage_runner.py`) and ε2b (`task_curator.py`); ο1 shares only a test file with ε1, so no edge; both S20 leaves are late or low. (e) Orchestrator files: S20 μ → S16 ζ1 (`agents/briefing.py`, S16's only leaf on it); S20 ο2 → S16 π (`routing.py`, whose last S16 leaf is π; through π → κ → ζ5b, ο2 also follows S16's last leaf on every other file they share: ζ5a on `workflow.py` and `harness.py`, ζ6b on `steward.py`, ζ3 on `scheduler.py`, δ on `review_checkpoint.py` and `dry_run_unblock.py`, γ2 on `agents/invoke.py` and `evals/architect_eval.py`; S16 no longer edits `workflow_types.py`). ξ3, ξ4 and ο3 share only test files with S16 leaves, so no edge (lead ruling of 2026-11-12) |
| S16 | owns (not built here) | the cap-scope model-name mismatch the task-8253 investigation found (the nightly preparer passes `fable`, the one-shot pool's `scoped_cap_models` default is `['claude-fable-5']` and the orchestrator config lists `[claude-fable-5, claude-fable-5-1]`, so `scope_for('fable')` is none: three spellings of one fact) is S16's C9 and leaf π (row above) | S16 | D12 moves scope derivation into the gate without changing which names `scope_for` matches; S16 π then changes the match to `canonical_model(model)` (edge in the C9 row above) |
| S22 dashboard readers (wave 3, concurrent) | shares files | S22 ι "cost, cap, model-role and load readers typed" moves the `invocations` SQL from `dashboard/data/costs.py` and `model_role.py` into `shared/cost_store.py` and edits `test_cost_store.py` and `dashboard/tests/test_costs_data.py`; S22 ρ1 `quality_series/rows.py`; S22 ρ3 moves `dashboard/data/quality_markers.py`'s reads (one reads `invocations.model_id`) into `runs_read/markers.py` and states each query's C7 class (contract only, no shared file); S22 τ `audit_model_admission.py`, `review_model_admission.py`, `mint_hard_v2_fixtures.py`; S22 υ `digest.py`; S22 α/υ `event_store.py` | — | **S22 ι → S20 ζ: S22 ι's `cost_store` query values keep `returned = 1` on every count and row-as-invocation query; B9's reader half is the regression test S22 ι keeps green.** S22 ρ1, ρ3, τ, υ → ζ (S22 records all five); S22 τ ↔ ξ2/ο3 and S22 α/υ ↔ ο2 by the rule above |
| S24 fleet watchdog | shares files | `df_pytest_isolation.py` (S24 α, ω; S20 λ); `orchestrator/harness.py` (S24 γ; S20 ο2) | — | S24 α ↔ λ by the rule above (8294 and 8831 already → λ) |
| S12 reconciliation run lifecycle | shares files | `reconciliation/judge.py`, `test_judge.py`, `test_stage_agent_outcome.py`, `test_recon_amend_tool_advertisement.py` (S12 α, γ0, ι2; S20 ξ2); `test_stages.py` (S12 γ0–κ2; S20 ο1) | — | rule above |
| S13 recon sweeps | shares files | `reconciliation/cli_stage_runner.py` (S13 ε; S20 ο1); `test_stages.py` (S13 ι, κ1, κ2, ν; S20 ο1) | — | rule above |
| S1 one escalation filing entry point | shares files | `workflow.py`, `harness.py`, `scheduler.py`, `steward.py`, `review_checkpoint.py` (S1 δ, ε3, ζ1, ζ2, η2, κ3, κ4, and λ2, the second eval-hermeticity task, on `workflow.py`; S20 ο2); `test_steward.py` (S1 κ4; S20 ο3) | — | rule above. S1 λ1 and λ2 (eval hermeticity, ruled by Leo 2026-11-13) own the factory-level refusal of production stores; ο2's B19 pins only that the loop gets no cost store from an eval cell, and ο2 takes no edge onto λ1 |
| S2 typed escalation record facts | shares files | `digest.py`, `test_digest.py` (S2 θ2, κ; S20 ζ); `workflow.py`, `harness.py`, `steward.py` (S2 β2, θ1, θ2, κ; S20 ο2); `test_steward.py` (S2 β1; S20 ο3) | — | rule above (ζ is free and early, so S2 θ2 and κ → ζ when S2 files second) |
| S17 scheduler seams | shares files | `agents/briefing.py`, `delivered_checks.py`, `test_briefing.py`, `test_delivered_check_gate.py` (S17 α, γ, δ1, δ2, ε, ω; S20 μ); `scheduler.py`, `harness.py`, `workflow.py`, `event_store.py`, `dry_run_unblock.py` (S20 ο2); `_orch_helpers.py` (S20 ο3); 8 orchestrator test files (S20 ξ4) | — | rule above; S17 δ1's one-evaluator move and μ's kind dispatch edit the same functions |
| S10 `plans/review-all-s10-task-write-contract-metadata-modes-success-signal-and-registry-prd.md` | consumes | `shared/delivered_check.py` (S10 C3, η 9240); INV-16 (θ 9242) | S10 | μ → 9240, 9242; 9240 → α (same file, lead action) |
| S10 | shares files | `task_curator.py` (δ 9204), fused-memory `reconciliation/harness.py` (λ 9254, not edited by S20 after the fold) | — | λ → 9204 |
| S8 `plans/review-all-s8-harness-composition-root-and-slot-registry-prd.md` | shares files | `usage_gate.py` + `transcript_archive.py` (β 9218); `harness.py`, `invoke.py`, `steward.py`, `workflow_types.py`, `scheduler.py`, `_orch_helpers.py`, `test_steward.py`, `test_invoke.py`, 30 orchestrator test files | — | 9218 → λ (lead action); ο2 → 9236 (S8's integration gate, which every open wave-2 holder of ο2's non-test files precedes), 9232, 9233, 9229, 9234. No edge for test files (lead ruling of 2026-11-12): δ's budget raises in 9232's, 9233's and 9234's test files, η's one-line fake edits in `test_steward.py` and `test_invoke.py`, and ξ4's held test files are rebase-only (the earlier lead edge 9232 → δ and ξ4 → 9236 are withdrawn) |
| S9 `plans/review-all-s9-workflow-phase-spine-and-recovery-ledger-prd.md` | shares files | `workflow.py`, `steward.py`, `_orch_helpers.py`, `test_steward.py`, `test_liveness_boundary_gate.py` | — | ο2 → 9281, 9190, 9287 (`workflow.py`, `steward.py`); `test_liveness_boundary_gate.py`, `test_steward.py`: no edge (test files) |
| S6 merge-lane lifecycle | shares a file | `test_liveness_boundary_gate.py` (γ2 9165) | — | no edge: the shared file is a test file (the earlier lead edge 9165 → θ is withdrawn under the lead ruling of 2026-11-12) |
| S4 `plans/review-all-s4-verify-run-context-and-runner-contract-prd.md` | shares files | `review_checkpoint.py` | — | ο2 → 9150 (μ0) |
| S3 `plans/review-all-s3-escalation-server-and-queue-boundary-prd.md` | shares files | `digest.py`, `test_digest.py` (γ 9162); fused-memory `cli_stage_runner.py`, `test_stages.py` (ε1 9188, ε2 9193) | — | 9162 → ζ; ο1 → 9188 (`cli_stage_runner.py`; fold-verify: was ο1 → 9193, which shares only `test_stages.py`) |
| S5 `plans/review-all-s5-gitops-ports-query-contract-and-the-git-runner-prd.md` | shares files | `scripts/mint_hard_v2_fixtures.py` (ρ0 9272); `delivered_checks.py` (ε 9263) | — | ο3 → 9272; μ → 9263 |
| S18 `plans/review-all-s18-evals-measurement-prd.md` | consumes | `classify_invocation` (C1), `invoke_with_cap_retry` (C7) signatures | S20 | no edge: both signatures are kept; S18's leaves are done or low |
| S21 (done/low) | none | `scripts/legibility/detectors.py::message_content` stays S21's | S21 | out of scope |

## Related tasks (contract §8)

- **6764** (deferred at `1b106418da`, S25 ι, held on S20/C3; its dependency 6037 is done, 6763
  pending): released onto α (lead action carries the contract as delivered; the release binds), and
  ordered after 9240 (same files `manifest_stamping.py`, `audit_manifest_descriptor_drift.py`,
  `audit_combine_gate_marker_loss.py`). Task 5861 (done) has split
  `scripts/tests/test_audit_manifest_descriptor_drift.py` into `manifest_drift_fixtures.py` and four
  `test_audit_manifest_descriptor_drift_*.py` modules, so 6764's files widen at release (its own note
  says so). **7724** (pending) already depends on 6764; it keeps `sidecar_path` as the pure rule and adds
  the inverse beside it; μ → 7724.
- Cancelled into a leaf (lead actions carry `x_acceptance_reason`): **7084** → γ (transcript
  predicates), **6930** → ε (usage_gate partition), **6996** → θ (the `cli_invoke.py` partition; 0.645
  twin of 6930), **7334** → η (NearCap on a failure), **6605** → η (the 126-line per-caller cap-wait
  prose), **6729** → ε (one `resumed` writer, D11), **6994** → η (a non-Claude call leases a Claude
  account; lead ruling S20/OQ-8, which also covers the zcode backend task 5399 added), **8234** → λ
  (sweep confinement), **7376** → μ (path checks render UNRECOGNISED). All nine pending at `1b106418da`.
- **5727** (pending, medium) discarded attempts invisible to cost accounting and budget containment:
  its cost/duration half (defect 1) is delivered by η plus ο2 (D6). Amend to the residual (the
  `--max-budget-usd` re-granted per retry; the steward timeout-kill $0 amendment), and 5727 → ο2 (it
  also edits `workflow.py`, which ο2 edits).
- Ordered against a leaf (lead actions): **7684** → ε (it decides on `ProbeVerdict`); **4605** was
  cancelled on 2026-11-11 as a duplicate of **4542** (pending, low; it now carries 4605's scope and
  depends on 9218), so the 4605 → ε edge is dropped and 4542 reaches ε through 9218 → λ with no new
  edge; **8294**, **8831**, **7353** → λ (same files); **9218** → λ; **6657** → θ (prompt builder
  for the first dispatch: lands on the finished executor; it also edits
  `test_cap_retry_prompt_builder.py`, which θ edits); **7478** → θ (one alias for an injected
  `invoke_with_cap_retry`); **5433** → β (the `agent API error: HTTP <status>` marker's producers move in β).
  ο2 → 6657 (same method `TaskWorkflow._invoke`). 7478 and 5433 share later files with ξ2/λ/ο1/ο2 and
  γ/δ/ο2 respectively; rebase-only, no edge.
- A leaf waits on them: **3986** (high, resume observability) ← β, ζ, η; **8563** (high, digest
  landings) ← ζ; **9240**, **9242**, **9235**, **9263**, **9288**, **7724** ← μ. All pending at
  `0f21f5e457`. No edge onto **8252** (cancelled 2026-11-12: its failover half was delivered by task
  6214, its persistence half rejected) or **8253** (pending, low, rescoped to the return brief's trial
  windows, depends only on 8563, no longer edits `usage_gate.py`).
- Rebase-level only, no edge (each lands long before the S20 leaf that shares its file, and file locks
  serialize the rest): **9374** (pending, low, simple: hoist usage-gate test helpers; shares
  `_usage_gate_test_helpers.py`, `test_auth_failed.py` and `test_usage_gate_probe_hold.py` with ξ1),
  **9340** (pending, low; shares `orchestrator/tests/_info_l0_router_fixtures.py` with ξ3), **8176**
  (pending, low; shares `fused-memory/tests/test_curation_deferral_acceptance.py` with ο1, whose holder
  9278 ο1 already follows).
- **6641** (pending, low, mechanical `fk-bb04ae68c0b1`) deletes `orchestrator/usage_gate.py`: ο3 → 6641.
- **6520** (done, `b900b34486`): the probe already uses `claude_binary_spec()` (R18 "task 6520
  absorbs" is complete). **6213**, **6627**, **6690**, **5521**, **6112**, **4357** (done): the
  partial fixes cited in the disposition table. Landed since the fold and folded into the text above:
  **3985** (third resume prompt and `resume_prompt`), **9327** (probe-hold timer), **6541** (adjudicator
  cost rows), **5399** (zcode backend), **6214** (nightly preparer through the shared loop).
- **Every other open task declaring `cli_invoke.py`, `usage_gate.py` or `transcript_archive.py`**
  gets a `repath` lead action at the landing of the leaf that moves its symbol; until then it rebases
  onto the re-export block (D8); a deferred one (3808, 3987, 3993, 4277, 5062 at `0f21f5e457`) gets
  its repath only if it is re-pended. Measured at `97be1d71fa` and re-checked at `1b106418da`: 3319, 3326,
  3327, 3768, 3808, 5062, 5287, 5378 (watchdog, startup grace and turn counting → `claude_runner.py`
  at θ); 3882 (high; an `AgentResult` field → `cli_result.py` at β); 7017, 4277 (argv → `claude_argv.py`
  at δ); 7127 (background detector → `transcript.py` at γ); 3987, 3993 (rotations; stay in
  `cli_invoke.py`; 3985 is done); 6062, 7139, 7607 (several homes; repath per symbol at landing);
  6681, 7243, 4420, 4542 (stay in `usage_gate.py`; rebase onto ε's partition); 3180, 3856, 4330, 4906,
  4950, 7429 (`transcript_archive.py`, which γ edits only for its import). Filed since the fold (all
  pending, low unless stated; 46 declaring tasks at `1b106418da`): 9338 and 9343 (resume handling in the
  loop and the workflow; stay in `cli_invoke.py`, rebase onto η's executor, whose per-attempt rows carry
  each attempt's session id); 9353 (folds `resume_delivers_prompt` into `resume_prompt`; C7's keyword
  list follows whichever lands first); 9501 (rewords one line of the per-caller cap-wait prose block η
  deletes; moot if η lands first); 5654 (medium; moves repo-wide guard tests, declaring `usage_gate.py`,
  `transcript_archive.py` and `invocation_outcome.py`; rebase). Same-file, rebase-level only:
  3394 (`test_cli_invoke.py`), 8558 (`test_pure_stdlib_leaves.py`; in progress at `1b106418da`), 5366
  (`audit_model_admission.py`).

## Decomposition plan

Greek labels; ids at decompose. Every task carries `metadata.x_finding_key` (the keys it closes or
contributes to), `metadata.x_finding_run = "review-all-dark_factory-20261008"`, `metadata.source =
"review-all"`, `prd_path`, `prd_task_label`. LOC estimates count moved lines. File lists were measured
at `97be1d71fa` and re-checked at `1b106418da` (changes: ξ1 +4 files, ξ3 −1 +3, ο1 +1, ο2's test file
gains B19) and at `0f21f5e457` (δ +1: both `test_landlock.py` paths); the importer lists use the count rule stated in the fold re-measurement. Edge notation:
"X → Y" means X depends on Y.

**α — Shared sidecar surface for the commit_planning stamper and manifest audits: SIDECAR_SUFFIX, sidecar_for, tracked_sidecar_paths** (priority medium, ~250 LOC)
- Files: (2 files) `shared/src/shared/capability_manifest.py`, `shared/tests/test_capability_manifest_sidecar_surface.py` (new)
- Deps: none. Lead edges 9240 → α and 6764 → α (release).
- Keys: none of S20's (it is the C3 seam for S25 ι's `fk-b16470a6e0d2`, `fk-bb8a584a8237`).
- Work: C3. `_SIDECAR_SUFFIX` becomes public `SIDECAR_SUFFIX`; `sidecar_for` composes `sidecar_path`.
- Signal: B1, B2 in CI; `python -c "from shared.capability_manifest import sidecar_for; print(sidecar_for('plans/../plans/x-prd.md', '.'))"` prints `plans/x-prd.capability-manifest.yaml`.
- Consumer: 6764 (released onto α), then 7724.
- Deletes: nothing (the consumers' copies go in 6764/7724).

**β — Result leaf: AgentResult and CliSubtype in shared/cli_result.py; failure projection to shared/agent_failure.py; the import cycle goes** (priority medium, ~1,100 LOC)
- Files: (14 files) `shared/src/shared/cli_result.py` (new), `shared/src/shared/agent_failure.py` (new), `shared/src/shared/cli_invoke.py`, `shared/src/shared/invocation_outcome.py`, `shared/src/shared/usage_gate.py`, `shared/src/shared/testing_resume_rejection.py`, `shared/src/shared/__init__.py`, `orchestrator/src/orchestrator/reviewer_failure.py`, `fused-memory/src/fused_memory/middleware/path_scope_adjudicator.py`, `shared/tests/test_cli_result.py` (new), `shared/tests/test_cli_invoke_layering.py` (new), `shared/tests/test_cli_subtype_spelling.py` (new), `shared/tests/test_invocation_outcome.py`, `shared/tests/test_public_api.py`
- Deps: S19 ι (S19/C7, the public CLI runner in `shared.cli_invoke`; S20/OQ-7 = S19/Q-C2 (a)); 3986.
  Lead edge 5433 → β.
- Keys: contributes `fk-1778eb482209`, `fk-aacd5e9381f6`, `fk-3ae32b390a81`.
- Work: C4; D7 (result and failure parts, carrying S19 C7's public `SubprocessResult` and
  `parse_claude_output`); D8 (re-export block, layering guard, import and patch-string allow-lists); D9.
  The spelling guard's allow-list starts with every production file that spells a subtype literal at
  landing, `orchestrator/src/orchestrator/agents/zcode_backend.py` (owned by ξ3) included.
  `CLI_INPUT_REQUIRED_MARKERS` and `RESUME_REJECTED_MARKERS` move down into `cli_result`;
  `invocation_outcome` imports from `cli_result` only. The loop's function-local imports of
  `invocation_outcome` become module-level. `TestSingleSourceOwnership` follows the moved tables.
- Signal: B3, B4, B17 in CI; `python -c "import sys, shared.invocation_outcome; print('shared.cli_invoke' in sys.modules)"` prints `False`.
- Consumer: γ, δ, η; S25/S16 via `CliSubtype`.
- Deletes: 6 function-local `shared.invocation_outcome` imports in `cli_invoke`, `invocation_outcome`'s
  module-top import from `cli_invoke` and its "no import cycle" paragraph,
  `BUDGET_EXHAUSTED_SUBTYPE`'s definition (a transitional alias until ο3),
  `reviewer_failure.BUDGET_CAP_SUBTYPE`, the `'error_max_turns'` literal in
  `_ceiling_reported_by_subtype`.

**γ — Transcript leaf shared/transcript.py: public locator, content_blocks, fixtures instead of patches** (priority medium, ~1,400 LOC)
- Files: (13 files) `shared/src/shared/transcript.py` (new), `shared/src/shared/cli_invoke.py`, `shared/src/shared/transcript_archive.py`, `shared/src/shared/__init__.py`, `shared/tests/test_pure_stdlib_leaves.py`, `shared/tests/test_transcript.py` (new), `shared/tests/test_cli_invoke_transcript_offloop.py`, `shared/tests/test_cli_invoke.py`, `shared/tests/test_resumable_progress.py`, `shared/tests/test_transcript_evidence.py`, `shared/tests/test_cli_invoke_background.py`, `shared/tests/test_cli_invoke_layering.py`, `shared/tests/test_public_api.py`
- Deps: β.
- Keys: contributes `fk-e59c30a25d27`, `fk-fede65204150`, `fk-3ae32b390a81`.
- Work: C5. `_record_field`'s rule becomes `content_blocks`; `detect_resumable_progress` uses it;
  `transcript_archive` imports `transcript_path` at module level. The 19 `count_transcript_turns` and
  5 `read_transcript_records` patch sites write real JSONL transcripts under a tmp config dir.
- Signal: B5 in CI; `git grep -n "shared.cli_invoke.count_transcript_turns" -- shared/tests` is empty.
- Consumer: θ (runner), ο2 (workflow), the curator and harness through the re-exports.
- Deletes: `_record_field`'s "SOLE expression" claim, the inline nesting walk, the private
  cross-module import and its defence, the task-3925 "Do NOT hoist a functools.partial" paragraph
  (the off-loop rule itself stays).

**δ — Argv leaf shared/claude_argv.py and the invocation cost floor** (priority medium, ~650 LOC)
- Files: (18 files) `shared/src/shared/claude_argv.py` (new), `shared/src/shared/cli_invoke.py`, `shared/src/shared/usage_gate.py`, `shared/src/shared/__init__.py`, `shared/tests/test_build_claude_argv.py`, `shared/tests/test_claude_argv_cost_floor.py` (new), `shared/tests/test_spawn_temp_files.py`, `shared/tests/test_cli_invoke_layering.py`, `shared/tests/test_public_api.py`, `orchestrator/tests/test_invoke_agent_credentials.py`, `orchestrator/tests/test_invoke_agent_python_env.py`, `orchestrator/tests/test_invoke_bash_cwd_reset.py`, `orchestrator/tests/test_landlock.py`, `shared/tests/test_landlock.py`, `orchestrator/tests/test_mcp_lifecycle.py`, `orchestrator/tests/test_sandbox_git_config.py`, `shared/tests/test_cli_invoke_agent_env.py`, `shared/tests/test_wildcard_deny_live_inventory.py`
- Deps: γ. No edge onto S19 ζ or onto the wave-2 holders of its test files (lead ruling of 2026-11-12:
  test files are rebase-only). S19 ζ moves `orchestrator/tests/test_landlock.py` to
  `shared/tests/test_landlock.py`; δ declares both paths and edits the one that exists when it lands.
  S8 κ1 9232, κ2 9233 and 9234 hold `test_sandbox_git_config.py`, `test_invoke_agent_credentials.py`,
  `test_invoke_agent_python_env.py` and `test_mcp_lifecycle.py` and rebase over δ's one-line raises.
- Keys: closes `fk-2f4b8655fb3c`; contributes `fk-3ae32b390a81`.
- Work: C6; D7 (argv part); D14. The ten sub-floor test budgets (`max_budget_usd=0.01`, eight files)
  become `INVOCATION_COST_FLOOR_USD`. Exempt from the 15-file review trigger: nine of the 18 files are
  one-line budget raises that must land atomically with the floor (a later split would leave main red).
- Signal: B13 in CI; an orchestrator role configured with `budgets.<role>: 0.1` fails its first
  dispatch with `InvocationBudgetBelowFloor` naming the floor (before: a silent
  `error_max_budget_usd` no-op).
- Consumer: every claude-backend invocation; S16's config-load check.
- Deletes: `usage_gate`'s import of `cli_invoke` (with β's repoint, `usage_gate` no longer imports it).

**ε — UsageGate partition: vocabulary, probe verdict, scope overlay; one resets_at writer; one cap-event writer and its resumed dual** (priority medium, ~1,500 LOC)
- Files: (12 files) `shared/src/shared/account_state.py` (new), `shared/src/shared/account_probe.py` (new), `shared/src/shared/scope_caps.py` (new), `shared/src/shared/usage_gate.py`, `shared/src/shared/cli_invoke.py`, `shared/src/shared/__init__.py`, `shared/tests/test_account_probe.py` (new), `shared/tests/test_usage_gate_cap_events.py` (new), `shared/tests/test_usage_gate_layering.py` (new), `shared/tests/test_probe_loop.py`, `shared/tests/test_auth_failed.py`, `shared/tests/test_public_api.py`
- Deps: δ. Lead edge 7684 → ε (4605, once also edged here, was cancelled into 4542, which reaches ε
  through 9218 → λ). Cancels 6930 and 6729.
- Keys: closes `fk-377f5a42e8be`, `fk-f7f1747302dc`, `fk-ae6cd7600d0e`, `fk-aacd5e9381f6`; contributes
  `fk-8805d20c0c4c`, `fk-005378a1b462`, `fk-db060c262575`.
- Work: D10, D11, D8 (re-export block for the vocabulary). `account_probe.run_probe` builds its argv
  from `claude_argv.claude_binary_spec()` and returns a `ProbeVerdict`; `probe_verdict` is pure.
  The resume loop sleeps on the plain backoff when `resets_at` is unknown (the invented one-hour
  target goes). `cli_invoke` imports `scope_for`/`PoolFrozen`/`ACCOUNT_POOL_BACKENDS` from
  `account_state` and `UsageGate` from `usage_gate` at module level (no `TYPE_CHECKING` or
  function-local import of `usage_gate` remains; `usage_gate` stopped importing `cli_invoke` in δ).
  `UsageGate.detect_cap_hit`'s function-local `CAP_HIT_PREFIXES`/`NEAR_CAP_PREFIXES` import is hoisted
  to module level, and B3's scan widens to the whole D7 set. `AccountState.probe_hold_timer` (task 9327)
  moves with `AccountState`.
  The leaf records `usage_gate.py`'s heuristic-14 measurement (line count, prose share, the partition it
  kept; 3,200 lines at `1b106418da`) in its commit message.
- Signal: B11, B12 in CI; a CAPPED account whose `resets_at` is unknown logs no "defaulting to 1h";
  the dashboard cap history shows no open-ended cap interval for a cap ended by the clock.
- Consumer: η (observe uses the vocabulary), 7684, 4542 (which absorbed 4605).
- Deletes: `UsageGate._probe_verdict`, the probe's direct `_transition` and `resets_at` write,
  `_handle_cap_detected`'s own `resets_at` write and scoped `_fire_cost_event('cap_hit')`, the probe
  loop's own `resumed` write, the "defaulting to 1h" branch, `cli_invoke`'s function-local
  `shared.usage_gate` import, `usage_gate`'s function-local `invocation_outcome` import.

**ζ — invocations rows per attempt: invocation_id, attempt, returned; count readers filter returned** (priority medium, ~450 LOC)
- Files: (10 files) `shared/src/shared/cost_store.py`, `dashboard/src/dashboard/data/costs.py`, `dashboard/src/dashboard/data/model_role.py`, `orchestrator/src/orchestrator/digest.py`, `shared/src/shared/quality_series/rows.py`, `scripts/audit_model_admission.py`, `scripts/review_model_admission.py`, `shared/tests/test_cost_store.py`, `dashboard/tests/test_costs_data.py`, `orchestrator/tests/test_digest.py`
- Deps: 3986 (it edits `cost_store.py` and `test_cost_store.py`); 8563 (it edits `digest.py` and
  `test_digest.py`). Lead edges 6765 → ζ, 6766 → ζ, 9162 → ζ, 9231 → ζ.
- Keys: contributes `fk-b8cba955b358`.
- Work: D6's schema half. Additive migration through `shared.sqlite_column_migration`; every reader
  of `invocations` is classified sum-only (unchanged) or row-as-invocation (filters `returned = 1`);
  the classification is a table in the task record. Re-measure the reader list at start:
  `git grep -n -i "from invocations\|join invocations" -- '*.py' ':!*tests*'`.
- Signal: B9's reader half in CI (a fixture db with 3 attempt rows of one invocation counts 1
  invocation and sums all 3 costs on the dashboard costs panel and in `digest`).
- Consumer: η (writer); S22 ι keeps the filters when it moves the SQL.

**η — Slot observe returns a Decision; the cap-retry state machine; one classification and one cost row per attempt for callers that pass a cost store** (priority **high**, ~1,500 LOC)
- Files: (20 files) `shared/src/shared/usage_gate.py`, `shared/src/shared/cli_invoke.py`, `shared/src/shared/cost_store.py`, `shared/src/shared/testing.py`, `shared/tests/test_cap_retry_decision_table.py` (new), `shared/tests/test_slot_observe.py` (new), `shared/tests/test_cap_retry.py`, `shared/tests/test_cap_retry_attempt_verdict.py`, `shared/tests/test_failover_integration.py`, `shared/tests/test_invocation_outcome_boundary.py`, `shared/tests/test_cap_scan_successful_output.py`, `shared/tests/test_make_gate_mock_scope.py`, `shared/tests/test_park_free_invocation.py`, `shared/tests/test_usage_gate_scope_attribution.py`, `shared/tests/test_usage_gate_scope_selection.py`, `scripts/tests/test_legibility_session_runner.py`, `shared/tests/test_cli_invoke_sandbox_wrap.py`, `orchestrator/tests/test_dry_run_unblock.py`, `orchestrator/tests/test_steward.py`, `orchestrator/tests/test_invoke.py` (the last four: one-line fake-signature edits, D12)
- Deps: ε, ζ; 3986. Cancels 7334, 6605 and 6994 (lead ruling S20/OQ-8). The three orchestrator test
  files are held by open wave-2 tasks (9187, 9217, 9232, 9288 on `test_dry_run_unblock.py`; 9190, 9229,
  9287 on `test_steward.py`; 9233 on `test_invoke.py`), all behind 5040. Its three one-line edits there
  are rebase-only under file locks, with no ordering edge, so η does not wait behind 5040 (lead ruling
  of 2026-11-12). Exempt from the 15-file review trigger: four of the 20 files are one-line fake
  signature edits that must land in the same commit as the new slot call (D12).
- Keys: closes `fk-dcc5dce60bb6`, `fk-2a5d97aeb64d`, `fk-005378a1b462`; contributes `fk-8805d20c0c4c`,
  `fk-b8cba955b358`, `fk-db060c262575`.
- Work: C2, C7 (the decision table, the ungated path and `RetryClock`), D1–D6, D12. `make_gate_mock`'s
  slot gets an `observe` whose verdict is derived from the mock's configured `detect_cap_hit`/`report`
  behaviour and which records `confirm(cost)`; suites not yet migrated stay green until θ, κ2, ξ1–ξ4
  and ο3 move them, and no file outside η's list changes in η. The 10 `invoke_slot(scope=…)` test calls
  become `model=`. Behaviour changes, each pinned by a test: the heuristic net is gone (B10); NearCap on
  a failure returns (D5, task 7334's
  `test_a_near_cap_pool_never_claims_a_cap_that_did_not_happen` becomes a positive assertion);
  discarded attempts count toward the session budget and get a row (B9); an unattributed exact cap
  records `capped_reason='account'` (D4); a codex, gemini or zcode dispatch leases no Claude account and a
  codex 401 benches none (B7).
- Signal: B6–B10 in CI. Deploy order: restart the dashboard after ζ lands and before η lands (rows
  with `returned = 0` must meet ζ's filters). After deploy: a steward invocation that hit a cap and
  resumed shows two `invocations` rows sharing an `invocation_id` in `runs.db`
  (`orchestrator/src/orchestrator/steward.py` passes `cost_store` today), and the dashboard's per-task
  cost for it includes the discarded attempt's cost.
- Consumer: every caller of `invoke_with_cap_retry`; the dashboard and `digest` through ζ; ο2.
- Deletes: the ten-arm `while True` body and its placement prose, the zero-cost heuristic net,
  `_DispatchedAttempt`, the after-loop `save_invocation`, `getattr(usage_gate, '_config', None)`,
  the per-caller cap-wait policy prose block above `_DEFAULT_CAP_WAIT_SANITY_SECS` (6605), the loop's
  calls to `slot.detect_cap_hit`/`report`/`confirm` (the methods themselves go in ο3), the
  cross-account resume measurement comment block (its home is
  `plans/cross-account-resume-measurement.md`, which a one-line pointer names).
- G7: `corroborate-before-acting` holds — a cap acts only on the slot's own lease
  (`lease_is_current`); `no-silent-fail-soft` — B10 makes "unclassified" a returned failure, not a
  silent cap.

**θ — Runner leaf shared/claude_runner.py; cli_invoke is the state machine; loop suites onto invoke_fn and clock** (priority medium, ~1,600 LOC, mostly moved)
- Files: (23 files) `shared/src/shared/claude_runner.py` (new), `shared/src/shared/cli_invoke.py`, `shared/src/shared/__init__.py`, `shared/tests/test_claude_runner.py` (new), `shared/tests/test_cli_invoke.py`, `shared/tests/test_cap_retry.py`, `shared/tests/test_failover_integration.py`, `shared/tests/test_cli_invoke_agent_env.py`, `shared/tests/test_cap_retry_prompt_builder.py`, `shared/tests/test_cap_retry_resume_opt_out.py`, `shared/tests/test_resume_rejection.py`, `shared/tests/test_server_error.py`, `shared/tests/test_model_not_found.py`, `shared/tests/test_cli_input_rejected.py`, `shared/tests/test_park_free_invocation.py`, `shared/tests/test_cli_invoke_session_cost.py`, `shared/tests/test_cap_scan_successful_output.py`, `shared/tests/test_invocation_outcome_boundary.py`, `shared/tests/test_usage_gate_scope_boundary.py`, `shared/tests/test_cli_invoke_transcript_offloop.py`, `shared/tests/test_cli_invoke_layering.py`, `shared/tests/test_public_api.py`, `orchestrator/tests/test_liveness_boundary_gate.py`
- Deps: η. Lead edges 6657 → θ, 7478 → θ. `orchestrator/tests/test_liveness_boundary_gate.py`, held
  by S6 γ2 9165 and the S9 tasks 9196, 9241, 9279, 9281, takes no edge (test file; lead ruling of
  2026-11-12): θ's patch renames there are rebase-only. Cancels 6996.
  Exempt from the 15-file review trigger: one mechanical rewrite per test file.
- Keys: closes `fk-fede65204150`; contributes `fk-3ae32b390a81`, `fk-2a5d97aeb64d`.
- Work: D7 (runner part: S19 C7's `run_cli_subprocess`, `CliRunner`, `ArgvWrapper`, `TranscriptWatch`
  moved unchanged), D13. The loop suites stop patching `shared.cli_invoke.invoke_claude_agent` (6
  shared test files) and `asyncio.sleep` (13 shared test files) and pass `invoke_fn=` and `clock=`;
  they move from configuring `detect_cap_hit` to configuring `observe` verdicts. Runner tests move to
  `test_claude_runner.py` with their patch targets renamed, `test_liveness_boundary_gate.py`'s
  `terminate_process_group` patches included. The layering guard's patch-string allow-list (D8) is
  extended to every runner name θ moves.
- Signal: `wc -l shared/src/shared/cli_invoke.py` ≤ 1,000 (measured at landing; recorded in the
  task); `git grep -n "shared.cli_invoke.invoke_claude_agent" -- shared/tests` is empty; B3 green.
- Consumer: ν, ξ1–ξ4, ο2, ο3.
- Deletes: the runner from `cli_invoke.py`.

**ι — UsageGate public test seam; AccountState setters and the probe-dir alias go** (priority **high**, ~1,200 LOC)
- Files: (18 files) `shared/src/shared/usage_gate.py`, `shared/src/shared/account_state.py`, `shared/src/shared/testing.py`, `shared/tests/test_usage_gate_test_seam.py` (new), `shared/tests/test_usage_gate_exhaustive.py`, `shared/tests/test_usage_gate.py`, `shared/tests/test_concurrency.py`, `shared/tests/test_probe_loop.py`, `shared/tests/test_auth_failed.py`, `shared/tests/test_failover_integration.py`, `shared/tests/test_usage_gate_try_lease.py`, `shared/tests/test_usage_gate_scope_attribution.py`, `shared/tests/test_usage_gate_park_visibility.py`, `shared/tests/test_cap_retry.py`, `shared/tests/test_usage_gate_scope_selection.py`, `orchestrator/tests/test_usage_gate.py`, `orchestrator/tests/test_reify_multi_account.py`, `fused-memory/tests/test_curator_state_tool.py`
- Deps: η (`usage_gate.py`, `testing.py`). It shares only test files with θ (`test_failover_integration.py`,
  `test_cap_retry.py`), so no edge (lead ruling of 2026-11-12). Exempt from the
  15-file trigger: the setter writes are a mechanical rewrite onto `drive_to`.
- Keys: closes `fk-db060c262575`; contributes `fk-25c96d016239`.
- Work: `UsageGate.account_snapshot() -> tuple[AccountSnapshot, ...]` (frozen: name, phase,
  resets_at, near_cap, scope caps); `shared/testing.py::drive_to(gate, name, phase, *, resets_at=None)`
  reaching each phase through public calls (`observe` on a scripted result, `try_lease`, the clock);
  the 288 setter writes in 14 files (`test_cap_retry.py`'s three and
  `test_usage_gate_scope_selection.py`'s one included) and the 19 `_probe_config_dir` uses move onto
  them. Re-measure at start:
  `git grep -c "\.\(capped\|probing\|probe_in_flight\|auth_failed\) = \(True\|False\)" -- '*test*'`.
  `UsageGate._expire_probe_hold`'s guard comment, which blames the setters ι deletes, is reworded to the
  generation check it still needs (task 9327).
- Signal: B14 in CI; `git grep -n "@capped.setter" -- shared/src` is empty.
- Consumer: κ1, κ2, every new gate test.
- Deletes: the four `AccountState` setters and the "Legacy-compat boolean shim" block, the
  `_probe_config_dir` alias and its `getattr` fallback in `_config_dir_for`.

**κ1 — Move the probe and exhaustive UsageGate suites onto the public seam; private-reach ratchet** (priority **high**, ~1,500 LOC; lead ruling S20/OQ-5)
- Files: (5 files) `shared/tests/test_usage_gate_exhaustive.py`, `shared/tests/test_probe_loop.py`, `shared/tests/test_auth_failed.py`, `shared/tests/test_usage_gate_private_reach_ratchet.py` (new), `shared/tests/usage_gate_private_reach_baseline.json` (new)
- Deps: ι.
- Keys: contributes `fk-25c96d016239`.
- Work: the three suites (384 + 187 + 137 `gate._*` references measured at `1b106418da`) assert on
  `account_snapshot()`, `observe` decisions, `ProbeVerdict` and cost events instead of `gate._*`;
  probe tests drive `account_probe` directly or a fake binary via `CLAUDE_BINARY`. The ratchet test
  counts `gate._<name>` references per test file against the baseline and fails on any rise.
- Signal: `git grep -c "gate\._accounts" -- shared/tests/test_usage_gate_exhaustive.py` is 0; the
  ratchet test is green and its baseline total is below the 1,412 `gate._*` references measured across
  the six largest suites at `1b106418da`.
- Consumer: κ2, future gate changes.

**κ2 — Move the remaining large UsageGate suites onto the public seam and lower the private-reach ratchet** (priority **high**, ~1,300 LOC)
- Files: (7 files) `shared/src/shared/testing.py`, `shared/tests/test_usage_gate.py`, `shared/tests/test_concurrency.py`, `orchestrator/tests/test_usage_gate.py`, `shared/tests/test_usage_gate_scope_attribution.py`, `shared/tests/test_usage_gate_private_reach_ratchet.py`, `shared/tests/usage_gate_private_reach_baseline.json`
- Deps: κ1.
- Keys: contributes `fk-25c96d016239`.
- Work: the four suites (358 + 118 + 228 `gate._*` references at `1b106418da`, plus `test_usage_gate_scope_attribution.py`)
  move onto the seam; their `detect_cap_hit` calls become `observe`; the ratchet baseline is lowered.
  `orchestrator/tests/test_steward.py` and `test_invoke.py` and the deletion of the slot verdict
  methods move to ο3, which waits on their wave-2 holders.
- Signal: `git grep -n "gate\._accounts" -- orchestrator/tests/test_usage_gate.py` is empty; the
  ratchet baseline drops by the migrated count.
- Consumer: ξ1, ο3.

**ξ1 — Repoint shared test importers onto shared.cli_result, shared.transcript, shared.claude_argv, shared.claude_runner and shared.account_state; shared tests stop calling the old slot verdict methods** (priority medium, ~1,000 LOC, mechanical)
- Files: (58 files) `shared/tests/_capacity_skip.py`, `shared/tests/_cross_account_evidence.py`, `shared/tests/_usage_gate_test_helpers.py`, `shared/tests/startup_completion_fixtures.py`, `shared/tests/startup_completion_probe.py`, `shared/tests/test_api_refusal.py`, `shared/tests/test_auth_failed.py`, `shared/tests/test_auth_reprobe_opt_out.py`, `shared/tests/test_usage_gate_account_names.py`, `shared/tests/test_usage_gate_probe_hold.py`, `shared/tests/test_usage_gate_park_visibility.py`, `shared/tests/test_build_claude_argv.py`, `shared/tests/test_cap_retry.py`, `shared/tests/test_cap_retry_attempt_verdict.py`, `shared/tests/test_cap_retry_prompt_builder.py`, `shared/tests/test_cap_retry_resume_opt_out.py`, `shared/tests/test_cap_scan_successful_output.py`, `shared/tests/test_capacity_skip.py`, `shared/tests/test_cli_input_rejected.py`, `shared/tests/test_cli_invoke.py`, `shared/tests/test_cli_invoke_agent_env.py`, `shared/tests/test_cli_invoke_background.py`, `shared/tests/test_cli_invoke_integration.py`, `shared/tests/test_cli_invoke_layering.py`, `shared/tests/test_cli_invoke_mcp_scope_forwarding.py`, `shared/tests/test_cli_invoke_parse_output.py`, `shared/tests/test_cli_invoke_sandbox_wrap.py`, `shared/tests/test_cli_invoke_session_cost.py`, `shared/tests/test_cli_invoke_transcript_offloop.py`, `shared/tests/test_concurrency.py`, `shared/tests/test_config_dir_credentials_live.py`, `shared/tests/test_cross_account_evidence.py`, `shared/tests/test_failover_integration.py`, `shared/tests/test_invocation_outcome.py`, `shared/tests/test_invocation_outcome_boundary.py`, `shared/tests/test_liveness_boundary_gate.py`, `shared/tests/test_make_gate_mock_scope.py`, `shared/tests/test_model_not_found.py`, `shared/tests/test_park_free_invocation.py`, `shared/tests/test_probe_loop.py`, `shared/tests/test_public_api.py`, `shared/tests/test_resumable_progress.py`, `shared/tests/test_resume_rejection.py`, `shared/tests/test_server_error.py`, `shared/tests/test_spawn_temp_files.py`, `shared/tests/test_transcript_archive.py`, `shared/tests/test_transcript_evidence.py`, `shared/tests/test_transcript_unreadable_escape.py`, `shared/tests/test_usage_gate.py`, `shared/tests/test_usage_gate_exhaustive.py`, `shared/tests/test_usage_gate_layering.py`, `shared/tests/test_usage_gate_scope.py`, `shared/tests/test_usage_gate_scope_attribution.py`, `shared/tests/test_usage_gate_scope_boundary.py`, `shared/tests/test_usage_gate_scope_selection.py`, `shared/tests/test_usage_gate_test_helpers.py`, `shared/tests/test_usage_gate_try_lease.py`, `shared/tests/test_wildcard_deny_live_inventory.py`
- Deps: θ, κ2. Exempt from the 15-file trigger (one import rewrite per file).
- Keys: contributes `fk-3ae32b390a81`.
- Work: every shared test file importing a moved name (51, count rule above, re-measured at
  `1b106418da`) imports it from its home; suites still configuring `detect_cap_hit` move to `observe`
  verdicts; the four files that call a real slot's `report` directly and no earlier leaf edits
  (`test_auth_reprobe_opt_out.py`, `test_usage_gate_account_names.py`, `test_usage_gate_probe_hold.py`,
  `test_usage_gate_park_visibility.py`; 5 calls at `1b106418da`) drive the gate through `drive_to` or
  `observe` instead. ξ1 is the backstop for shared tests: any direct `slot.report`/`confirm`/`settle`
  call left in a shared test file after η, θ, ι, κ1 and κ2 (63 such calls in 16 test files at
  `1b106418da`, most in `test_cap_retry.py` and `test_usage_gate_scope_attribution.py`) moves onto
  `observe`, and a test of `report` itself is deleted where B7 already covers the verdict, so ο3 deletes
  the methods with no shared test caller left. Their entries leave both allow-lists. Task 9374 (pending,
  low) edits three of these files first or rebases. ~1,000 LOC with the backstop.
- Signal: the layering guards' allow-lists contain no `shared/tests` entry; no `shared/tests/*.py` file
  calls `slot.report(`, `slot.confirm(` or `slot.settle(`.

**ξ2 — Repoint fused-memory and scripts importers in files no wave-2 task holds; scripts adopt CliSubtype** (priority medium, ~700 LOC, mechanical)
- Files: (36 files) `fused-memory/src/fused_memory/reconciliation/agent_loop.py`, `fused-memory/src/fused_memory/reconciliation/judge.py`, `fused-memory/src/fused_memory/maintenance/link_adjudicator.py`, `fused-memory/src/fused_memory/middleware/pack_judge.py`, `fused-memory/src/fused_memory/middleware/curation_deferral.py`, `fused-memory/src/fused_memory/middleware/path_scope_adjudicator.py`, `fused-memory/scripts/probe_schema_max_turns.py`, `scripts/legibility/session_runner.py`, `scripts/sitting/nightly_prepare.py`, `scripts/probe_dontask_git_write_bypass.py`, `scripts/review_model_admission.py`, `tests/scripts/test_claude_binary_isolation.py`, `fused-memory/tests/test_main_curator_usage_gate.py`, `shared/tests/test_cli_invoke_layering.py`, `shared/tests/test_cli_subtype_spelling.py`, `fused-memory/tests/reconciliation/stubbed_stage_cli.py`, `fused-memory/tests/reconciliation/test_cli_stage_runner_cutover.py`, `fused-memory/tests/reconciliation/test_cli_stage_runner_sandbox.py`, `fused-memory/tests/reconciliation/test_stage_agent_outcome.py`, `fused-memory/tests/reconciliation/test_stage_startup_grace.py`, `fused-memory/tests/reconciliation/test_verify_agent_failure.py`, `fused-memory/tests/reconciliation/test_verify_cli_invocation.py`, `fused-memory/tests/test_agent_loop.py`, `fused-memory/tests/test_curation_deferral.py`, `fused-memory/tests/test_curator_outage_deferral.py`, `fused-memory/tests/test_curator_pool_deferral.py`, `fused-memory/tests/test_judge.py`, `fused-memory/tests/test_link_adjudicator.py`, `fused-memory/tests/test_pack_judge.py`, `fused-memory/tests/test_path_scope_adjudicator.py`, `fused-memory/tests/test_recon_amend_tool_advertisement.py`, `fused-memory/tests/test_task_curator_timeout_evidence.py`, `scripts/tests/conftest.py`, `scripts/tests/test_legibility_nightly.py`, `scripts/tests/test_legibility_session_runner.py`, `scripts/tests/test_sitting_nightly_prepare.py`
- Deps: θ; S25 6760 (`session_runner.py`, `nightly_prepare.py`; it follows 6759), 6765, 6766
  (`review_model_admission.py`). ξ1 shares only `test_cli_invoke_layering.py`: no edge.
- Keys: contributes `fk-1778eb482209`, `fk-3ae32b390a81`.
- Work: as ξ1 for the 17 fused-memory and 4 scripts test files no wave-2 task holds, the 6 fused-memory
  production importers outside `task_curator.py`/`cli_stage_runner.py`, the 4 scripts importers,
  `fused-memory/scripts/probe_schema_max_turns.py` and `tests/scripts/test_claude_binary_isolation.py`.
  `scripts/review_model_admission.py` counts kills by `CliSubtype` members. The two
  `'shared.cli_invoke.invoke_claude_agent'` patches task 6541 added to `test_path_scope_adjudicator.py`
  move onto `invoke_fn` or the runner's home with the rest; `test_curator_pool_deferral.py`'s direct
  `slot.report` call moves onto `observe`.
- Signal: `fused-memory/src/fused_memory/reconciliation/judge.py` imports the failure projection from
  `shared.agent_failure`; `scripts/review_model_admission.py` has no `TURN_CAP_SUBTYPE` copy.

**ξ3 — Repoint orchestrator importers in files no wave-2 task holds (the tests and the zcode backend) onto shared.cli_result, shared.transcript, shared.claude_argv, shared.claude_runner and shared.account_state** (priority medium, ~650 LOC, mechanical)
- Files: (32 files) `orchestrator/src/orchestrator/agents/zcode_backend.py`, `orchestrator/tests/_eval_refusal.py`, `orchestrator/tests/_info_l0_router_fixtures.py`, `orchestrator/tests/test_zcode_backend.py`, `orchestrator/tests/test_eval_architect.py`, `orchestrator/tests/test_eval_architect_steps.py`, `orchestrator/tests/test_eval_compare_unmeasured.py`, `orchestrator/tests/test_eval_diff_threading.py`, `orchestrator/tests/test_eval_elo_unmeasured.py`, `orchestrator/tests/test_eval_measurement.py`, `orchestrator/tests/test_eval_refusal_is_unmeasured.py`, `orchestrator/tests/test_eval_spawn_identity.py`, `orchestrator/tests/test_invoke_bash_cwd_reset.py`, `orchestrator/tests/test_invoke_parsers.py`, `orchestrator/tests/test_merger_disposition_verdict.py`, `orchestrator/tests/test_review_checkpoint_cap.py`, `orchestrator/tests/test_review_checkpoint_degraded_outcomes.py`, `orchestrator/tests/test_review_checkpoint_phase1_spawn.py`, `orchestrator/tests/test_review_checkpoint_pre_test.py`, `orchestrator/tests/test_review_checkpoint_working_tree.py`, `orchestrator/tests/test_reviewer_failure.py`, `orchestrator/tests/test_reviewer_trial_main.py`, `orchestrator/tests/test_reviewer_trial_mining.py`, `orchestrator/tests/test_reviewer_trial_scorer.py`, `orchestrator/tests/test_routing.py`, `orchestrator/tests/test_scope_capacity_gate_integration.py`, `orchestrator/tests/test_shared_output_schema_intact.py`, `orchestrator/tests/test_steward_prompt_builder.py`, `orchestrator/tests/test_usage_gate.py`, `orchestrator/tests/test_zero_work_failure.py`, `shared/tests/test_cli_invoke_layering.py`, `shared/tests/test_usage_gate_layering.py`
- Deps: θ (the moved names exist). It shares only test files with κ2 (`orchestrator/tests/test_usage_gate.py`)
  and with ξ1, ξ2 and ξ4 (the layering guards): no edge. Exempt from the 15-file trigger.
- Keys: contributes `fk-3ae32b390a81`.
- Work: as ξ1 for the 29 orchestrator test files no open wave-2 task declares (re-measured at
  `1b106418da`: `test_reviewer_note_dispositions.py` left the set when task 5227 moved its fixtures to
  `_info_l0_router_fixtures.py`, which pending low task 9340 also declares, rebase-level), and for
  `agents/zcode_backend.py`, the one orchestrator production importer no open task holds (task 5399,
  done): it imports `AgentResult` and S19 C7's `SubprocessResult` from `cli_result` and spells its three
  subtypes as `CliSubtype` members, which removes its spelling-guard allow-list entry.
- Signal: the allow-lists contain no entry for a ξ3 file; `zcode_backend.py` spells no subtype literal.

**ξ4 — Repoint the orchestrator test importers that wave-2 tasks also hold, rebase-only under file locks** (priority medium, ~700 LOC, mechanical)
- Files: (34 files) `orchestrator/tests/test_architect_all_committed_plan.py`, `orchestrator/tests/test_block_disposition.py`, `orchestrator/tests/test_crash_recovery.py`, `orchestrator/tests/test_dry_run_unblock.py`, `orchestrator/tests/test_eval_boundary_suite.py`, `orchestrator/tests/test_eval_mcp_isolation.py`, `orchestrator/tests/test_execute_server_error_requeue.py`, `orchestrator/tests/test_harness_watcher_supervisor.py`, `orchestrator/tests/test_invoke_agent_credentials.py`, `orchestrator/tests/test_invoke_agent_python_env.py`, `orchestrator/tests/test_invoke_argv_parity.py`, `orchestrator/tests/test_judge_completion_verdict.py`, `orchestrator/tests/test_liveness_boundary_gate.py`, `orchestrator/tests/test_out_of_band_routing.py`, `orchestrator/tests/test_plan_revalidation_skip.py`, `orchestrator/tests/test_project_layout_adoption.py`, `orchestrator/tests/test_retry_cap.py`, `orchestrator/tests/test_reviewer_trial_runner.py`, `orchestrator/tests/test_reviewer_verdict_routing.py`, `orchestrator/tests/test_session_resume_integration_gate.py`, `orchestrator/tests/test_shadow_invocation_sandbox.py`, `orchestrator/tests/test_simple_task_workflow.py`, `orchestrator/tests/test_transcript_archive_startup_sweeper.py`, `orchestrator/tests/test_verdict_servers_integration_gate.py`, `orchestrator/tests/test_watcher_rotation_cap_retry.py`, `orchestrator/tests/test_workflow_architect_l0_promotion.py`, `orchestrator/tests/test_workflow_architect_transient_retry.py`, `orchestrator/tests/test_workflow_false_premise.py`, `orchestrator/tests/test_workflow_plan_repair_replan.py`, `orchestrator/tests/test_workflow_ready_to_merge.py`, `orchestrator/tests/test_workflow_state_machine_boundary.py`, `orchestrator/tests/test_workflow_prompt_builder.py`, `shared/tests/test_cli_invoke_layering.py`, `shared/tests/test_usage_gate_layering.py`
- Deps: θ (the moved names exist, and η's `observe` for the `_ForcedCapUsageGate` move). Every file is a
  test file, so no edge onto 9236 or any wave-2 holder and none onto ξ3 (lead ruling of 2026-11-12):
  the import repoints are rebase-only under file locks, and the holders rebase over them. Exempt from
  the 15-file trigger; hand-carried if lock-starved for more than 24 h (the R57 rule).
- Keys: contributes `fk-3ae32b390a81`.
- Work: as ξ1 for the 32 held orchestrator test files (`_orch_helpers.py`, `test_steward.py` and
  `test_invoke.py` are ο3's), the `detect_cap_hit` configurations in
  `test_watcher_rotation_cap_retry.py`, `test_dry_run_unblock.py` and `test_workflow_prompt_builder.py`
  included (the last imports no moved name and is in the list for that reason), and
  `test_dry_run_unblock.py`'s hand-written `_ForcedCapUsageGate` fake (`invoke_slot(scope=…)`,
  `detect_cap_hit`, `confirm`, `settle`), which moves to the C2 shape.
- Signal: the allow-lists contain no `orchestrator/tests` entry other than ο3's three files.

**λ — Thread the scratch root as a parameter; the sweep interception point goes** (priority low, ~500 LOC)
- Files: (12 files) `shared/src/shared/config_dir.py`, `shared/src/shared/usage_gate.py`, `fused-memory/src/fused_memory/middleware/task_curator.py`, `scripts/legibility/session_runner.py`, `df_pytest_isolation.py`, `shared/tests/test_config_dir.py`, `shared/tests/test_usage_gate.py`, `shared/tests/test_startup_completion_probe.py`, `shared/tests/startup_completion_probe.py`, `fused-memory/tests/test_task_curator_config_dir_archival.py`, `fused-memory/tests/test_task_curator_timeout_evidence.py`, `shared/tests/config_dir_archival_allowlist.py`
- Deps: ι (`usage_gate.py`), ξ2 (`session_runner.py`); S10 δ 9204 (`task_curator.py`). It shares only
  test files with ξ1 and κ2: no edge. Lead edges 8294, 8831, 7353, 9218 → λ. Cancels 8234.
- Keys: closes `fk-734bc1fc18bf`.
- Work: D15 (`tempfile.tempdir` confinement, no module-name patch).
- Signal: B15 in CI; `git grep -n "def test_the_module_level_sweep_name_is_still_the_interception_point" -- shared/tests` is empty.
- Deletes: the `sweep` parameter and its 22-line rejected-alternatives docstring, the two
  interception-point tests, the per-caller `sweep_stale_pid_dirs` imports and their 11+ patch sites.

**μ — Delivered-check kinds as a discriminated union; consumers dispatch on the model** (priority low, ~700 LOC)
- Files: (12 files) `shared/src/shared/delivered_check.py`, `shared/src/shared/capability_manifest.py`, `orchestrator/src/orchestrator/delivered_checks.py`, `orchestrator/src/orchestrator/agents/briefing.py`, `fused-memory/src/fused_memory/server/manifest_stamping.py`, `scripts/audit_manifest_descriptor_drift.py`, `shared/tests/test_capability_manifest.py`, `orchestrator/tests/test_briefing.py`, `orchestrator/tests/test_delivered_check_gate.py`, `fused-memory/tests/test_manifest_stamping.py`, `scripts/tests/test_audit_manifest_descriptor_drift.py`, `fused-memory/tests/server/test_write_triage_flip_gate_invariants.py`
- Deps: α; S10 η 9240 (S10 C3: `shared/delivered_check.py` exists) and θ 9242 (INV-16, the `SLICES`
  table); 7724 (same files `capability_manifest.py`, `test_capability_manifest.py`,
  `manifest_stamping.py`); wave-2 9235, 9263, 9288 (same files `agents/briefing.py`, `test_briefing.py`,
  `delivered_checks.py`; 9170, 9178, 9255 precede them); S16 ζ1 (`agents/briefing.py`, S16's only leaf
  on it). Cancels 7376.
- Keys: closes `fk-240df0521f83`.
- Work: D16. `CHECK_SUBJECT_FIELD` derives from the models; the four production construction sites
  and the tests call `parse_delivered_check_meta`.
- Signal: B16 in CI; an architect briefing for a task carrying a `kind: path` check renders its
  `expect` and `paths` (before: "UNRECOGNISED descriptor").
- Deletes: `_check_kind_conditional_fields`, its load-bearing-order note,
  `briefing._DELIVERED_CHECK_FIELDS_BY_KIND`.

**ν — Type the invocation backend with the shared Backend enum (S16's role-and-backend vocabulary, S16/C1)** (priority low, ~200 LOC) — **held on S16**
- Files: (5 files) `shared/src/shared/cli_invoke.py`, `shared/src/shared/invocation_outcome.py`, `shared/src/shared/account_state.py`, `shared/tests/test_cap_retry_decision_table.py`, `shared/tests/test_invocation_outcome.py`
- Deps: ο3 (same file); π (`invocation_outcome.py`); S16 γ0, which delivers S16 C1 (one `Backend` StrEnum and its one decoder
  `parse_backend` in `shared/backends.py`) (held: `x_seam_wait = {stream:
  "S16", contract: "C1", release_owner: "S16"}`; an ordinary dependency if γ0 has an id at S20's
  decompose). S16 π also edits `account_state.py`; ν follows it through ο3 → ο2 → S16 π, so no
  further edge.
- Keys: none (seam integration for S16's `Backend`).
- Work: `ACCOUNT_POOL_BACKENDS: frozenset[Backend]` in `account_state`; `backend: Backend` on
  `invoke_with_cap_retry`, `invoke_slot` and `classify_invocation`, decoded once by S16's
  `parse_backend`; the codex/gemini arms keyed on members; the claude cap branch, which `pi` (an
  account-pool backend) reaches implicitly today, names both members explicitly; `zcode` and `mimo` are
  members outside the pool (S16 C1 carries both, tasks 5399 and 7993).
- Signal: pyright rejects `invoke_with_cap_retry(…, backend='Claude')` in a fixture (typecheck leg).

**ο1 — Fused-memory files that wave-2 tasks hold adopt the new shared homes** (priority medium, ~350 LOC)
- Files: (9 files) `fused-memory/src/fused_memory/middleware/task_curator.py`, `fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py`, `fused-memory/tests/_curator_helpers.py`, `fused-memory/tests/test_curation_pipeline.py`, `fused-memory/tests/test_curator_corpus_streams.py`, `fused-memory/tests/test_stages.py`, `fused-memory/tests/test_task_curator.py`, `fused-memory/tests/test_ticket_parking.py`, `fused-memory/tests/test_curation_deferral_acceptance.py`
- Deps: λ (same file `task_curator.py`, whose wave-2 holder 9204 λ follows); 9188 (S3 ε1,
  `cli_stage_runner.py`); S16 ε2a (`cli_stage_runner.py`) and ε2b (`task_curator.py`). The wave-2 holders of its test files (9193, 9278) take no edge (lead ruling of
  2026-11-12; fold-verify: ο1 → 9193 and → 9278 rested only on test files).
- Keys: contributes `fk-3ae32b390a81`.
- Work: `task_curator.py` and `cli_stage_runner.py` import from `cli_result`, `transcript`,
  `claude_argv` and `agent_failure`; the six held fused-memory test files repoint and move off
  `detect_cap_hit` and `shared.cli_invoke.invoke_claude_agent` patches;
  `test_curation_deferral_acceptance.py` (held by 9278 and pending low task 8176) moves its direct
  `slot.report` call onto `observe` or `drive_to`. (`reconciliation/harness.py` imports only
  `AllAccountsCappedException` and `UsageGate`, which stay; it is not edited.)
- Signal: `cli_stage_runner.py` imports `AgentResult` from `shared.cli_result`; no fused-memory entry
  remains on the allow-lists.

**ο2 — The orchestrator adopts CliSubtype, the transcript leaf and the retry loop as the one writer of invocation cost rows** (priority **high**, ~1,250 LOC; lead ruling S20/OQ-6)
- Files: (17 files) `orchestrator/src/orchestrator/workflow.py`, `orchestrator/src/orchestrator/harness.py`, `orchestrator/src/orchestrator/steward.py`, `orchestrator/src/orchestrator/dry_run_unblock.py`, `orchestrator/src/orchestrator/review_checkpoint.py`, `orchestrator/src/orchestrator/workflow_types.py`, `orchestrator/src/orchestrator/scheduler.py`, `orchestrator/src/orchestrator/routing.py`, `orchestrator/src/orchestrator/agents/invoke.py`, `orchestrator/src/orchestrator/zero_work_failure.py`, `orchestrator/src/orchestrator/event_store.py`, `orchestrator/src/orchestrator/evals/architect_eval.py`, `orchestrator/src/orchestrator/evals/measurement.py`, `orchestrator/src/orchestrator/evals/compare.py`, `orchestrator/src/orchestrator/evals/judge.py`, `orchestrator/src/orchestrator/evals/reviewer_trial/scorer.py`, `orchestrator/tests/test_invocation_rows_one_writer.py` (new)
- Deps: θ; wave-2 9232, 9233, 9229, 9234, 9150, 9236, 9281, 9190, 9287; S25 6766 (`routing.py`); 6657
  (same method `TaskWorkflow._invoke`); S16 π (`routing.py`, S16's last leaf on it; through π → κ →
  ζ5b, ο2 also follows S16's last leaf on each other shared file, `## Cross-PRD relationship` S16 row (e)).
  Every open wave-2 task declaring one of ο2's files is an ancestor of these (measured). Hand-carried if
  lock-starved for more than 24 h (R57 rule). Lead edge 5727 → ο2. Over the 15-file review trigger by
  two (17 files): one coherent change, the orchestrator's adoption of S20's homes, with no file that
  could move to another leaf without a second pass over `workflow.py`.
- Keys: closes `fk-1778eb482209`, `fk-e59c30a25d27`, `fk-b8cba955b358`.
- Work: D6's writer consolidation — `TaskWorkflow._invoke` and `review_checkpoint.py` pass
  `cost_store`, `run_id`, `task_id`, `project_id` and `role` into `invoke_with_cap_retry` and delete
  their own `save_invocation` blocks (B18). `CliSubtype` members at `invoke.py`'s mint sites,
  `steward._EMPTY_OUTPUT_SUBTYPES`, `dry_run_unblock`, `routing`; `_capture_zero_output_evidence` takes
  the last tool name from `transcript.transcript_evidence(records).other_tool_uses`; the production
  importers repoint (`invoke.py` imports public names only, S19 ι having removed its private ones);
  `event_store.py`'s comment citation of `is_resume_rejected` follows the move. Evals stay hermetic
  (cross-stream rule of 2026-11-12): `test_invocation_rows_one_writer.py` also carries B19, pinning that
  an eval cell's workflow passes no cost store into the loop and the eval gate holds none; the S1
  plan's eval-hermeticity tasks (S1 λ1, a factory that refuses production stores, and S1 λ2, the
  invariant; ruled by Leo 2026-11-13) own the factory-level refusal, and ο2 neither waits on λ1 nor
  duplicates it.
- Signal: B18 and B19 in CI. After deploy: a task whose implementer hit a cap and resumed shows two
  `invocations` rows sharing an `invocation_id`, and its dashboard cost includes the discarded
  attempt; `git grep -n "BUDGET_EXHAUSTED_SUBTYPE" -- orchestrator/src` is empty; a zero-output
  evidence file for a flat-shaped transcript names its last tool.
- Deletes: the two callers' own `save_invocation` blocks, the orchestrator's `BUDGET_EXHAUSTED_SUBTYPE`
  uses, the workflow nesting walk, `invoke.py`'s minted literals.

**ο3 — Transitional re-exports, the old slot verdict methods and the __new__ gate builder go** (priority **high**, ~900 LOC)
- Files: (10 files) `shared/src/shared/cli_invoke.py`, `shared/src/shared/usage_gate.py`, `shared/src/shared/testing.py`, `orchestrator/tests/_orch_helpers.py`, `orchestrator/tests/test_steward.py`, `orchestrator/tests/test_invoke.py`, `scripts/mint_hard_v2_fixtures.py`, `shared/tests/test_cli_invoke_layering.py`, `shared/tests/test_usage_gate_layering.py`, `shared/tests/test_cli_subtype_spelling.py`
- Deps: ο1, ο2, ξ1, ξ2, ξ3, ξ4, λ, κ2; existing 6641; wave-2 9272 (`mint_hard_v2_fixtures.py`).
  `test_steward.py` and `test_invoke.py`'s wave-2 holders (9190, 9229, 9287, 9233) precede it through ο2.
- Keys: closes `fk-3ae32b390a81`, `fk-25c96d016239`, `fk-8805d20c0c4c`; contributes `fk-1778eb482209`.
- Work: D8's final step; the 21 `detect_cap_hit` uses in `test_steward.py`, `test_invoke.py` and
  `_orch_helpers.py` become `observe`, and the `_cm` fakes in the first two move to the C2 shape; then `UsageGate.detect_cap_hit` and `InvokeSlot.report`,
  `.confirm`, `.settle`, `.detect_cap_hit` are deleted with `make_gate_mock`'s bridge;
  `_orch_helpers.build_usage_gate` constructs through `UsageGate(config, scratch_root=…)`;
  `scripts/mint_hard_v2_fixtures.py` spells subtypes as members.
- Signal: B17's end state in CI (no re-export block, no allow-list); `git grep -n "def detect_cap_hit"
  -- shared/src` is empty.
- Deletes: both transitional re-export blocks and allow-lists, `BUDGET_EXHAUSTED_SUBTYPE`, the second
  classification's last code path, the four slot verdict methods, the class-level `_park_*` defaults
  and their 30-line defence, the `getattr`-default reads that served `__new__`-built gates.

**π — The cap classifier also recognises the CLI's structured usage-limit field, not only banner text** (priority low, ~200 LOC)
- Files: (4 files) `shared/src/shared/cli_result.py`, `shared/src/shared/invocation_outcome.py`, `shared/tests/test_cli_result.py`, `shared/tests/test_invocation_outcome.py`
- Deps: η (the task that reshapes cap detection to one classification of the whole result; π adds one
  input to that classification). ν → π (both edit `invocation_outcome.py`).
- Keys: none (side finding of the task-8253 investigation, 2026-11-12; heuristic 12, structured data
  instead of meaningful strings, in this plan's own files).
- Why its own task: η is at the top of the size band (~1,500 LOC, 20 files) and declares neither
  `cli_result.py` nor `invocation_outcome.py`; β declares both but only moves code.
- Work: `AgentResult` gains `api_error: str | None`, set by `parse_claude_output` from the CLI's result
  JSON; `classify_invocation`'s cap arm treats `api_error == 'usage_limit_reached'` as `CapHit` even when
  no banner substring matches. The reset time and the cap scope still come from the banner text
  (`resets_at=None` when the banner gives none; ε's plain backoff then applies). No misclassification
  was observed in the investigation, hence priority low.
- Signal: B20 in CI.
- Consumer: `InvokeSlot.observe` and the loop's ungated path (C1's two call sites for an attempt).

**ω — Acceptance record: each long-running orchestrator and fused-memory process keeps its own record of capped accounts** (task_kind deterministic,
filed and cancelled at decompose; ruled by Leo 2026-11-12, S20-1 option A)
- Keys: `fk-49b55c1270ae`.
- Files: none.
- Deps: none.
- `x_acceptance_reason` (basis, then one sentence from Leo's ruling on task 8253):
  - Basis: "Each long-running orchestrator and fused-memory process holds its own cap state over the
    shared account pool. Measured 2026-11-11: 87 of 96 caps since 2026-10-11 were rediscovered by more
    than one process (median 6), each rediscovery costing one instant failed dispatch and a 5 s
    cooldown (`shared/cli_invoke.py::_CAP_HIT_COOLDOWN_SECS`), and each process probing capped accounts
    with haiku probes (~$0.001). The shared credentials-file half was fixed by per-(account, pid) probe
    dirs. Accepted because that is the whole cost: no work is lost. Re-open if a rediscovery is shown to
    lose work. Ruled by Leo, 2026-11-12."
  - No cap file: "There is no cap file for any process (Leo, 2026-11-12, task 8253); one-shot processes
    rediscover caps too, and the measured cost is one free failed start per capped seat."

DAG (X → Y = X depends on Y): β → S19 ι, 3986; γ → β; δ → γ; ε → δ; ζ → 3986, 8563;
η → ε, ζ, 3986; θ → η; ι → η; κ1 → ι; κ2 → κ1; ξ1 → θ, κ2; ξ2 → θ, 6760, 6765, 6766;
ξ3 → θ; ξ4 → θ; λ → ι, ξ2, 9204; μ → α, 9240, 9242, 7724, 9235, 9263, 9288, S16 ζ1;
ν → ο3, π, S16 γ0; π → η; ο1 → λ, 9188, S16 ε2a, S16 ε2b; ο2 → θ, 9232, 9233, 9229, 9234, 9150, 9236, 9281, 9190, 9287,
6766, 6657, S16 π; ο3 → ο1, ο2, ξ1, ξ2, ξ3, ξ4, λ, κ2, 6641, 9272. α and ω have no S20 dependency; π (low) adds the structured usage-limit field after η. Lead edges
(existing task → S20 leaf): 9240, 6764 → α; 5433 → β; 7684 → ε (4605's edge dropped: it was
cancelled into 4542); 6765, 6766, 9162, 9231 → ζ; 6657, 7478 → θ; 8294, 8831, 7353, 9218 → λ;
5727 → ο2; plus 6764 → 9240. Other wave-3 plans' edges onto S20 leaves (each recorded by the plan
named, except where `## Cross-PRD relationship` says S20 states it): S16 δ → δ; S16 ζ6a → β; S16 π → ι;
S16 ε1, ε2a, ε2b → ξ2 and S16 ε2b → λ (stated by S20); S22 ι, ρ1, ρ3, τ, υ → ζ. No edge rests only on a test file (lead ruling of 2026-11-12, restating
R54); the fold-verify withdrew δ → S19 ζ, ι → θ, ξ2 → ξ1, ξ3 → ξ2, ξ3 → κ2, ξ4 → ξ3, ξ4 → 9236,
λ → ξ1, λ → κ2, ο1 → 9193, ο1 → 9278 and the lead edges 9232 → δ and 9165 → θ on that ground, added
λ → ι (`usage_gate.py`), ο1 → 9188 (`cli_stage_runner.py`) and ξ4 → θ, ξ3 keeps θ, and corrected ξ2's
6759 to 6760. Every pair of S20 leaves that shares a non-test file is ordered by a path in this DAG
(re-checked at `0f21f5e457`); eleven pairs share only test files and are left to file locks (θ/ι,
ι/ξ3, κ2/ξ3, κ2/λ, ξ1/λ and the layering-guard pairs among ξ1–ξ4). The DAG with the live store's
open edges is acyclic (walked at `0f21f5e457`). β's former file-order edge onto 8252 is gone (8252 is
cancelled; Leo's ruling on 8253, 2026-11-12). High leaves η, ι, κ1, κ2, ο2, ο3 become dependencies of the trigger-chain gate
6698 at decompose. Signal leaves: η (the per-attempt behaviour for rows the loop writes), ο2 (the
task-workflow cost rows) and ο3 (the end state of the partition).

### G7 walk (advisory at author time; re-run at decompose)

- `contracts-machine-checked`: C1–C7 each have boundary rows (B1–B20); the decision table is a pure,
  table-tested function; the layering and spelling guards are tests, not prose.
- `no-lockstep-duplication` / `one-fact-one-home`: β (subtypes), γ (nesting rule), ε (cap event and its
  dual, `resets_at`), α (sidecar rule), μ (kind table), δ (floor), ο2 (one `invocations` writer).
- `exceptions-owned-or-ratified`: the transitional re-exports, the patch-string allow-list and the
  spelling allow-list are governed: each entry names its owning leaf (ξ1–ξ4/ο1–ο3) and the lists only
  shrink; the private-reach ratchet has a committed baseline owned by κ1/κ2.
- `no-silent-fail-soft`: D3 and D4 (an unclassified or unattributed failure is returned and recorded,
  never a silent cap or an uncapped row); C3's discovery failure is an exception; C6 replaces a
  silent budget no-op with a raise; a stale patch target of a moved name fails the layering guard
  instead of silently no-opping.
- `readers-prove-their-producer`: ζ's readers treat legacy rows (`invocation_id` null) as returned;
  B9 seeds rows through the real writer once η lands (ζ's own test uses the migration path, and η's
  B9 re-runs the reader half against rows η wrote).
- `corroborate-before-acting`: `observe` acts only on the slot's own lease (stale-lease fail-safe kept).
- `structured-facts-at-failure`: `InvocationBudgetBelowFloor`, `SlotAlreadySettled` and the guards name
  values.
- `holds-owned-and-bounded`: ν is held with `x_seam_wait` naming S16 and a release owner.
- The two rules Leo stated on 2026-11-12: `one-decision-partition-per-area` (INV-23; C7's table and C2's
  verdicts are code-decided, one function each, human escape unchanged) and hermetic evals (B19).
- No hits on `storm-escape-required` (no detector added; one deleted), `loop-thread-occupancy-bounded`,
  `status-matches-liveness`, `guards-exercise-behaviour` (every guard runs code or an AST over the
  live tree), `sampler-vocabulary-one-table`.

## Out of scope

- Cross-process cap state for the long-running processes (accepted in ω, ruled by Leo 2026-11-12).
- The runner's public API with an argv-wrapper hook and moving its `create_subprocess_exec` patches
  onto the runner seam (S19 ι, R17), which lands before β.
- Config-shape and role-budget validation at load (S16), dashboard reader redesign (S22), the
  delivered-check evaluator move (S17), the legibility detectors' own `message_content` (S21).
- Budget containment across retries (`--max-budget-usd` re-granted per attempt) and the steward
  timeout-kill $0 — task 5727's residual.
- The prompt builder for the first dispatch (task 6657, after θ).
- Replacing the prose cap tables: the CLI's result JSON carries `api_error: "usage_limit_reached"` but no
  reset time, so π reads the field for detection only and the banner text stays the source of
  `resets_at` and of the cap scope.
- Line-pin and `ruff format` cleanup (standing rulings).

## Rulings (sitting of 2026-11-12)

Sources: `digests/wave3-rulings-applied.md` §S20 and its cross-stream block, `digests/discussion/S20.md`
§S20-1, `digests/wave3-lead-rulings.md` §S20 and its cross-stream alignments.

- **OQ-1** (module names) — as drafted: `cli_result`, `agent_failure`, `transcript`, `claude_argv`,
  `claude_runner`, `account_state`, `account_probe`, `scope_caps`, carrying the package-layering plan's
  public runner names. Ruled by the lead.
- **OQ-2** (sitting key S20-1; cap rediscovery by the long-running processes) — option A: accept, with
  acceptance record ω whose reason rests on the measured cost; plus Leo's request for a separate
  investigate-and-discuss session on task 8253, spawned by the lead, whose ruling is applied (see the
  task-8253 line below and `## Dependence on task 8253`). Ruled by Leo. Applies R18 ("per-process cap state" was not part of
  R18's remedy) and amends no program ruling.
- **OQ-3** (moving names out of the two modules) — governed transitional re-exports, four mechanical
  repoint tasks for the free files and a final task that deletes the re-exports. Ruled by the lead.
- **OQ-4** (shape of per-attempt cost rows) — one row per attempt in `invocations` with
  `invocation_id`, `attempt` and `returned`. Ruled by the lead.
- **OQ-5** (test migration) — migrate the six largest UsageGate suites in two tasks (κ1, κ2) and add a
  ratchet against new private reach. Ruled by the lead.
- **OQ-6** (who writes the per-attempt rows) — the retry loop is the only writer; the task workflow
  and the deep reviewer pass their cost store and row fields into it. Ruled by the lead; it is R18's
  "one `cost_store` write per attempt" read as one writer.
- **OQ-7** (which plan lands first on the CLI runner) — the package-layering plan's public runner
  (S19 ι) first, and this plan's split moves its public names unchanged; same decision as S19/Q-C2 (a).
  Ruled by the lead.
- **OQ-8** (should a non-Claude call lease a Claude account) — no: an account-less slot, and task 6994
  is cancelled into η; flagged to Leo as a behaviour change (non-Claude calls stop waiting on Claude
  caps). Ruled by the lead. The fold extends the same rule to the zcode backend task 5399 added, with no
  new decision: the rule is keyed on `ACCOUNT_POOL_BACKENDS`.
- Cross-stream rules applied (Leo, 2026-11-12): the design invariant `one-decision-partition-per-area`
  (INV-23 under the lead's numbering ruling; checks anchor on the slug) — C7's
  decision table and C2's verdicts are wholly code-decided, with the human escape unchanged (C7); evals
  hermetic — moving the cost-row writer into the loop opens no path from an eval run, pinned by B19 in
  ο2 (D6).
- Lead sequencing rulings after the fold-verifiers (2026-11-12): (1) an edge that rests only on a shared
  test file is not wired (restating R54); S20's earlier request for edges from the package-layering plan's
  tasks (S19 θ0, θ1, θ2) onto γ on `shared/tests/test_pure_stdlib_leaves.py` is dropped, and the
  fold-verify withdrew every other test-file-only edge of this plan (DAG paragraph), intra-plan ones
  included, a reading the lead confirmed (the rule covers intra-plan edges too); (2) η's three
  one-line edits in test files held by wave-2 tasks are rebase-only, with no ordering edge, so η does not
  wait behind task 5040. Ruled by the lead; both read R54 and amend no program ruling.
- Task 8253 (Leo, 2026-11-12, after the investigation he asked for under OQ-2): no process reads or
  writes a cap file; task 8252 is cancelled and 8253 is rescoped away from `usage_gate.py`. Applied:
  β's edge onto 8252 is gone, the cap-ledger sentences left D10 and ε, ω's reason says no process has a
  cap file, and the out-of-scope line naming 8253 is deleted. Ruled by Leo; amends no program ruling.
- Alignment pass (2026-11-13): the decision-partition invariant is cited by its slug
  `one-decision-partition-per-area` and as INV-23 (the lead's numbering ruling: the six wave-3
  invariant tasks take INV-18 to INV-23 in landing order, this one last); S16's verified fold is
  reflected (C1 with `ZCODE` and `MIMO`, ζ6a without `shared/cli_invoke.py`, C9 and S16 π → ι, and the
  S16 edges made concrete by file); the eval-hermeticity work is the S1 plan's λ1 and λ2, not a lead
  follow-up outside every plan.

## Dependence on task 8253

On 2026-11-12 Leo ruled, after the investigation he asked for under OQ-2 (`digests/leo-rulings-8253.md`),
that no process, long-running or one-shot, reads or writes a cap file: task 8252 is cancelled and task
8253 (pending, low) is rescoped to the return brief's trial windows and no longer edits
`shared/src/shared/usage_gate.py`. This plan therefore has no edge onto either task and no text that
depends on a cap file, and ω's reason records that one-shot processes rediscover caps too, at one free
failed start per capped seat.

## Open questions (tactical)

1. **Floor value per model.** One constant (0.25) is enforced. If a cheaper model's measured
   minimal call is shown to need a lower floor for a real caller, the constant becomes a per-model
   table in the same module. Decide in δ only if a caller fails.
2. **`routing.probe_models` with a budget below the floor** now records `invoke_error` instead of
   `budget_too_low`. If an operator relies on that status for sub-floor budgets, map the new
   exception to `budget_too_low` in ο2. Decide in ο2.
3. **Test-file split of `shared/tests/test_cli_invoke.py`** (6,305 lines) follows the module split
   in θ; finer splits are θ's call, under heuristic 13.
