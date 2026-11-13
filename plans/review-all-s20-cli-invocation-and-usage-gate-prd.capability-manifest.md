# Capability manifest — S20 CLI invocation and usage gate

PRD: `plans/review-all-s20-cli-invocation-and-usage-gate-prd.md`. The machine-readable twin is
`plans/review-all-s20-cli-invocation-and-usage-gate-prd.capability-manifest.yaml` (every `task_id`
is null until decompose).

**Code anchors** verified against main `97be1d71fa` (2026-11-11), re-verified at `1b106418da`
(2026-11-12, ruling fold) and at `0f21f5e457` (2026-11-12, fold-verify) and cited by symbol.

Every grep and path `delivered_check` was run at main `1b106418da`, and again at `0f21f5e457` by the fold-verify, with `git grep -E` over its declared paths: 85 mechanical checks (82 producer checks and 3 substrate checks), loaded through `shared.capability_manifest.CapabilityManifestDoc` and passed through `shared.delivered_check_polarity.lint_delivered_checks` against each leaf's declared files with no finding. Each producer check fails at `1b106418da` and passes once its leaf lands (an `expect: present` pattern or path is absent there; an `expect: absent` pattern is present there). The 3 `*-substrate` checks pass today by design (`vacuous_present`) and are never copied to a task as producer checks. The ruling fold added five producer checks (η, ξ1, ξ3, ο1, ο2) and dropped one stale path from ξ3's. The second fold-verify added π's two producer checks.

## Verdict summary

- No binding is FAIL.
- One binding is OPEN: ν `backend-typed`, whose producer is S16 C1 (one `Backend` StrEnum with its one decoder `parse_backend` in `shared/backends.py`, landed by S16's role-and-backend vocabulary leaf γ0, not yet filed; S16 is folded and fold-verified, and C1 carries `ZCODE` and `MIMO`, tasks 5399 and 7993, both outside the account pool). ν is a held leaf (`x_seam_wait` S16/C1) unless S16 γ0 has an id at decompose; the row carries a mechanical check that flips when ν lands.
- One binding is a cross-stream producer: μ `delivered-check-module-upstream` → task 9240 (S10 η, S10 C3), upstream of μ by dependency; it carries a manual check naming that edge.
- ω (the acceptance record for `fk-49b55c1270ae`, ruled by Leo 2026-11-12, S20-1 option A) carries no capability and has no block. Its reason also records Leo's ruling on task 8253 (2026-11-12): no process has a cap file; no capability here depends on 8253.
- Signal leaves: η, ο2, ο3 (η: per-attempt behaviour for rows the loop writes; ο2: task-workflow cost rows; ο3: the partition's end state).

## Per-leaf bindings

**α — Shared sidecar surface for the commit_planning stamper and manifest audits: SIDECAR_SUFFIX, sidecar_for, tracked_sidecar_paths**
- Note: Delivers C3, the seam held task 6764 (S25 ι) waits on; closes no S20 key.
- `sidecar-path-rule-substrate`: `grep:shared/src/shared/capability_manifest.py::sidecar_path` substrate: the pure PRD-key rule sidecar_for composes (task 6582). Check: grep `def sidecar_path\(` in `shared/src/shared/capability_manifest.py` present. **PASS**
- `sidecar-for-defined`: built here — C3 sidecar_for, containment-checked. Check: grep `def sidecar_for\(` in `shared/src/shared/capability_manifest.py` present. **PASS**
- `tracked-sidecar-paths-defined`: built here — C3 tracked_sidecar_paths over git ls-files. Check: grep `def tracked_sidecar_paths\(` in `shared/src/shared/capability_manifest.py` present. **PASS**
- `discovery-failure-is-an-exception`: built here — C3 ManifestDiscoveryUnavailable. Check: grep `class ManifestDiscoveryUnavailable` in `shared/src/shared/capability_manifest.py` present. **PASS**
- `suffix-public`: built here — SIDECAR_SUFFIX replaces the private _SIDECAR_SUFFIX. Check: grep `^SIDECAR_SUFFIX` in `shared/src/shared/capability_manifest.py` present. **PASS**
- `b1-table-agrees-with-stamper-and-audit`: built here — B1 pins the eight prd_path rows against the values the stamper and drift audit produce at main. Check: manual — B1 expected values are pinned from main; a reviewer confirms the table covers the eight shapes C3 lists. **PASS**

**β — Result leaf: AgentResult and CliSubtype in shared/cli_result.py; failure projection in shared/agent_failure.py; the import cycle goes**
- Note: Contributes fk-1778eb482209, fk-aacd5e9381f6, fk-3ae32b390a81 (closers: ο2, ε, ο3). Depends on S19 ι (S19/C7, the public CLI runner in `shared.cli_invoke`; lead ruling S20/OQ-7 = S19/Q-C2 (a)).
- `classify-invocation-substrate`: `grep:shared/src/shared/invocation_outcome.py::classify_invocation` substrate: C1 signature kept. Check: grep `def classify_invocation\(` in `shared/src/shared/invocation_outcome.py` present. **PASS**
- `cli-result-module`: built here — shared/cli_result.py. Check: path `shared/src/shared/cli_result.py` present. **PASS**
- `clisubtype-defined`: built here — C4 CliSubtype beside AgentResult. Check: grep `class CliSubtype\(StrEnum\)` in `shared/src/shared/` present. **PASS**
- `agent-failure-module`: built here — shared/agent_failure.py. Check: path `shared/src/shared/agent_failure.py` present. **PASS**
- `outcome-no-longer-imports-cli-invoke`: built here (deletion) — invocation_outcome imports cli_result only. Check: grep `^from shared\.cli_invoke import` in `shared/src/shared/invocation_outcome.py` absent. **PASS**
- `cli-invoke-local-outcome-imports-gone`: built here (deletion) — the function-local invocation_outcome imports. Check: grep `^ +from shared\.invocation_outcome import` in `shared/src/shared/cli_invoke.py` absent. **PASS**
- `reviewer-failure-budget-copy-gone`: built here (deletion) — reviewer_failure uses CliSubtype. Check: grep `BUDGET_CAP_SUBTYPE = ` in `orchestrator/src/orchestrator/reviewer_failure.py` absent. **PASS**
- `subtype-spelling-guard`: built here — the AST spelling guard with its shrinking allow-list. Check: path `shared/tests/test_cli_subtype_spelling.py` present. **PASS**

**γ — Transcript leaf shared/transcript.py: public locator, content_blocks, fixtures instead of patches**
- Note: Contributes fk-e59c30a25d27, fk-fede65204150, fk-3ae32b390a81 (closers: ο2, θ, ο3).
- `transcript-module`: built here — shared/transcript.py (C5). Check: path `shared/src/shared/transcript.py` present. **PASS**
- `content-blocks-public`: built here — the one nesting rule. Check: grep `def content_blocks\(` in `shared/src/shared/` present. **PASS**
- `archive-private-reach-gone`: built here (deletion) — transcript_archive imports transcript_path at module level. Check: grep `import _resolve_transcript_path` in `shared/src/shared/transcript_archive.py` absent. **PASS**
- `hoist-paragraph-gone`: built here (deletion) — no test patches count_transcript_turns, so the paragraph has no reason. Check: grep `Do NOT hoist a functools\.partial` in `shared/src/shared/` absent. **PASS**
- `transcript-patch-sites-gone`: built here (deletion) — tests write real transcripts. Check: grep `shared\.cli_invoke\.count_transcript_turns` in `shared/tests/` absent. **PASS**

**δ — Argv leaf shared/claude_argv.py and the invocation cost floor**
- Note: Closes fk-2f4b8655fb3c; contributes fk-3ae32b390a81. Raises the ten sub-floor test budgets (eight files).
- `argv-module`: built here — shared/claude_argv.py. Check: path `shared/src/shared/claude_argv.py` present. **PASS**
- `cost-floor-home`: built here — C6 INVOCATION_COST_FLOOR_USD. Check: grep `^INVOCATION_COST_FLOOR_USD` in `shared/src/shared/` present. **PASS**
- `cost-floor-enforced`: built here — C6 InvocationBudgetBelowFloor raised by build_claude_argv. Check: grep `class InvocationBudgetBelowFloor` in `shared/src/shared/` present. **PASS**
- `usage-gate-stops-importing-cli-invoke`: built here (deletion) — usage_gate imports cli_result and claude_argv only. Check: grep `^from shared\.cli_invoke import` in `shared/src/shared/usage_gate.py` absent. **PASS**
- `sub-floor-test-budgets-gone`: built here (deletion) — the ten test call sites passing max_budget_usd=0.01 pass the floor constant. Check: grep `max_budget_usd=0\.01` in 9 paths (listed in the yaml; `test_landlock.py` at both its current and its post-S19-ζ path) absent. **PASS**

**ε — UsageGate partition: vocabulary, probe verdict, scope overlay; one resets_at writer; one cap-event writer and its resumed dual**
- Note: Closes fk-377f5a42e8be, fk-f7f1747302dc, fk-ae6cd7600d0e, fk-aacd5e9381f6; contributes fk-8805d20c0c4c, fk-005378a1b462, fk-db060c262575. Absorbs tasks 6930 and 6729.
- `account-state-module`: built here — shared/account_state.py. Check: path `shared/src/shared/account_state.py` present. **PASS**
- `account-probe-module`: built here — shared/account_probe.py. Check: path `shared/src/shared/account_probe.py` present. **PASS**
- `probe-verdict-type`: built here — ProbeVerdict returned by the probe. Check: grep `class ProbeVerdict` in `shared/src/shared/` present. **PASS**
- `scope-caps-module`: built here — shared/scope_caps.py. Check: path `shared/src/shared/scope_caps.py` present. **PASS**
- `probe-no-longer-transitions`: built here (deletion) — the gate applies the verdict. Check: grep `def _probe_verdict` in `shared/src/shared/usage_gate.py` absent. **PASS**
- `invented-reset-gone`: built here (deletion) — unknown resets_at sleeps on the plain backoff. Check: grep `defaulting to 1h` in `shared/src/shared/` absent. **PASS**
- `one-cap-event-writer`: built here — _record_cap_event. Check: grep `def _record_cap_event` in `shared/src/shared/usage_gate.py` present. **PASS**
- `cli-invoke-imports-usage-gate-at-module-level`: built here (deletion) — no indented import of shared.usage_gate in cli_invoke. Check: grep `^ +from shared\.usage_gate import` in `shared/src/shared/cli_invoke.py` absent. **PASS**
- `usage-gate-local-outcome-import-gone`: built here (deletion) — the function-local CAP_HIT_PREFIXES import in UsageGate.detect_cap_hit is hoisted. Check: grep `^ +from shared\.invocation_outcome import` in `shared/src/shared/usage_gate.py` absent. **PASS**
- `pool-backends-in-vocabulary`: built here — ACCOUNT_POOL_BACKENDS moves into the account vocabulary so the slot can read it. Check: grep `^ACCOUNT_POOL_BACKENDS` in `shared/src/shared/account_state.py` present. **PASS**

**ζ — invocations rows per attempt: invocation_id, attempt, returned; count readers filter returned**
- Note: Contributes fk-b8cba955b358 (closer ο2). Row shape per lead ruling S20/OQ-4.
- `attempt-columns`: built here — additive invocation_id/attempt/returned columns. Check: grep `invocation_id` in `shared/src/shared/cost_store.py` present. **PASS**
- `count-readers-filter-returned`: built here — every row-as-invocation reader filters returned = 1. Check: manual — a reviewer confirms each reader in the re-measured list is classified sum-only or row-as-invocation and the latter filter returned = 1. **PASS**

**η — Slot observe returns a Decision; the cap-retry state machine; one classification and one cost row per attempt for callers that pass a cost store**
- Note: Closes fk-dcc5dce60bb6, fk-2a5d97aeb64d, fk-005378a1b462; contributes fk-8805d20c0c4c, fk-b8cba955b358, fk-db060c262575. Signal leaf. Absorbs 7334, 6605 and 6994 (lead ruling S20/OQ-8).
- `stale-lease-failsafe-substrate`: `grep:shared/src/shared/usage_gate.py::UsageGate.lease_is_current` substrate: observe keeps the stale-lease fail-safe. Check: grep `def lease_is_current` in `shared/src/shared/usage_gate.py` present. **PASS**
- `observe-defined`: built here — C2 InvokeSlot.observe. Check: grep `def observe\(` in `shared/src/shared/usage_gate.py` present. **PASS**
- `account-verdict-defined`: built here — C2 AccountVerdict. Check: grep `class AccountVerdict` in `shared/src/shared/` present. **PASS**
- `decision-table-defined`: built here — C7 decide. Check: grep `^def decide\(` in `shared/src/shared/cli_invoke.py` present. **PASS**
- `clock-seam`: built here — C7 RetryClock, used by the Wait-row tests. Check: grep `class RetryClock` in `shared/src/shared/cli_invoke.py` present. **PASS**
- `heuristic-net-gone`: built here (deletion) — D3. Check: grep `Heuristic cap: zero-cost instant exit` in `shared/src/shared/` absent. **PASS**
- `private-config-reach-gone`: built here (deletion) — D12, the gate derives scope. Check: grep `getattr\(usage_gate, '_config'` in `shared/src/shared/` absent. **PASS**
- `loop-stops-calling-detect-cap-hit`: built here (deletion) — the second classification leaves the production path. Check: grep `slot\.detect_cap_hit\(` in `shared/src/shared/cli_invoke.py` absent. **PASS**
- `loop-lifetime-attempt-record-gone`: built here (deletion) — one AttemptRecord written per attempt. Check: grep `class _DispatchedAttempt` in `shared/src/shared/` absent. **PASS**
- `every-attempt-accounted`: built here — B9: three attempts write three rows sharing invocation_id. Check: manual — B9 runs against a real CostStore in CI; a reviewer confirms the after-deploy runs.db read of a steward invocation named in the signal. **PASS**
- `hand-written-gate-fakes-accept-new-slot-call`: built here (deletion) — D12, the four hand-written gate fakes driven through the real loop stop declaring today's `invoke_slot(scope=None, …)` signature (found by the ruling fold). Check: grep `invoke_slot\(self, scope=None|def _cm\(scope=None` in `shared/tests/test_cli_invoke_sandbox_wrap.py`, `orchestrator/tests/test_dry_run_unblock.py`, `orchestrator/tests/test_steward.py`, `orchestrator/tests/test_invoke.py` absent. **PASS**

**θ — Runner leaf shared/claude_runner.py; cli_invoke is the state machine; loop suites onto invoke_fn and clock**
- Note: Closes fk-fede65204150; contributes fk-3ae32b390a81, fk-2a5d97aeb64d. Carries S19/C7 public runner names unchanged (lead ruling S20/OQ-7). Absorbs task 6996.
- `runner-module`: built here — shared/claude_runner.py. Check: path `shared/src/shared/claude_runner.py` present. **PASS**
- `runner-left-cli-invoke`: built here (deletion) — the runner lives in claude_runner. Check: grep `^async def (_run_subprocess|run_cli_subprocess)\(` in `shared/src/shared/cli_invoke.py` absent. **PASS**
- `loop-sleep-patches-gone`: built here (deletion) — loop suites pass clock= instead of patching asyncio.sleep through cli_invoke (the 13 shared test files patching it at 97be1d71fa). Check: grep `shared\.cli_invoke\.asyncio\.sleep` in 13 files (listed in the yaml) absent. **PASS**
- `loop-patch-sites-gone`: built here (deletion) — loop suites pass invoke_fn (the 6 shared test files patching it at 97be1d71fa). Check: grep `shared\.cli_invoke\.invoke_claude_agent` in 6 files (listed in the yaml) absent. **PASS**
- `runner-patch-targets-renamed`: built here (deletion) — a patch of a moved runner name through cli_invoke would silently no-op. Check: grep `shared\.cli_invoke\.terminate_process_group` in `orchestrator/tests/test_liveness_boundary_gate.py`, `shared/tests/test_cli_invoke.py`, `shared/tests/test_cli_invoke_transcript_offloop.py` absent. **PASS**
- `cli-invoke-size-measured`: built here — heuristic-14 measurement recorded at landing. Check: manual — wc -l shared/src/shared/cli_invoke.py ≤ 1,000, recorded in the task record. **PASS**

**ι — UsageGate public test seam; AccountState setters and the probe-dir alias go**
- Note: Closes fk-db060c262575; contributes fk-25c96d016239.
- `setters-gone`: built here (deletion) — the four AccountState phase setters. Check: grep `@capped\.setter` in `shared/src/shared/` absent. **PASS**
- `probe-dir-alias-gone`: built here (deletion) — the _probe_config_dir alias. Check: grep `self\._probe_config_dir = ` in `shared/src/shared/` absent. **PASS**
- `drive-to-seam`: built here — shared/testing.py::drive_to. Check: grep `def drive_to\(` in `shared/src/shared/testing.py` present. **PASS**
- `account-snapshot-seam`: built here — UsageGate.account_snapshot. Check: grep `def account_snapshot\(` in `shared/src/shared/usage_gate.py` present. **PASS**

**κ1 — Move the probe and exhaustive UsageGate suites onto the public seam; private-reach ratchet**
- Note: Contributes fk-25c96d016239. Lead ruling S20/OQ-5.
- `exhaustive-suite-off-private-state`: built here (deletion) — gate._accounts reads in the exhaustive suite. Check: grep `gate\._accounts` in `shared/tests/test_usage_gate_exhaustive.py` absent. **PASS**
- `private-reach-ratchet`: built here — the ratchet test. Check: path `shared/tests/test_usage_gate_private_reach_ratchet.py` present. **PASS**

**κ2 — Move the remaining large UsageGate suites onto the public seam and lower the private-reach ratchet**
- Note: Contributes fk-25c96d016239. The slot verdict methods are deleted later, in ο3.
- `orchestrator-gate-suite-off-private-state`: built here (deletion) — gate._accounts reads. Check: grep `gate\._accounts` in `orchestrator/tests/test_usage_gate.py` absent. **PASS**
- `kappa2-suites-observe-not-detect`: built here (deletion) — the four suites drive observe instead of configuring detect_cap_hit. Check: grep `detect_cap_hit` in `shared/tests/test_usage_gate.py`, `shared/tests/test_concurrency.py`, `orchestrator/tests/test_usage_gate.py`, `shared/tests/test_usage_gate_scope_attribution.py` absent. **PASS**

**ξ1 — Repoint shared test importers onto shared.cli_result, shared.transcript, shared.claude_argv, shared.claude_runner and shared.account_state; shared tests stop calling the old slot verdict methods**
- Note: Contributes fk-3ae32b390a81. Files enumerated at 97be1d71fa, re-checked at 1b106418da (count rule in the PRD).
- `shared-tests-stop-calling-slot-verdict-methods`: built here (deletion) — the 12 shared test files calling a slot's `report`/`confirm`/`settle` directly at `1b106418da` move onto `observe` or `drive_to`, so ο3 can delete those methods. Check: grep `slot\.(report|confirm|settle)\(` in 12 files (listed in the yaml) absent. **PASS**
- `shared-tests-import-agent-result-from-home`: built here (deletion) — the 18 shared test files importing AgentResult through cli_invoke at 97be1d71fa import it from its home (single-line form; the layering guard covers parenthesised imports). Check: grep `from shared\.cli_invoke import .*\bAgentResult\b` in 18 files (listed in the yaml) absent. **PASS**

**ξ2 — Repoint fused-memory and scripts importers in files no wave-2 task holds; scripts adopt CliSubtype**
- Note: Contributes fk-1778eb482209, fk-3ae32b390a81.
- `judge-imports-from-homes`: built here — judge.py imports the failure projection from shared.agent_failure (it keeps importing invoke_with_cap_retry from cli_invoke). Check: grep `from shared\.agent_failure import` in `fused-memory/src/fused_memory/reconciliation/judge.py` present. **PASS**
- `review-admission-turn-cap-copy-gone`: built here (deletion) — counts by CliSubtype members. Check: grep `TURN_CAP_SUBTYPE = ` in `scripts/review_model_admission.py` absent. **PASS**

**ξ3 — Repoint orchestrator importers in files no wave-2 task holds (the tests and the zcode backend) onto shared.cli_result, shared.transcript, shared.claude_argv, shared.claude_runner and shared.account_state**
- Note: Contributes fk-3ae32b390a81. Files enumerated at 97be1d71fa, re-checked at 1b106418da (`test_reviewer_note_dispositions.py` left the set; `_info_l0_router_fixtures.py`, `test_zcode_backend.py` and `agents/zcode_backend.py` joined).
- `zcode-backend-spells-subtypes-as-members`: built here (deletion) — `agents/zcode_backend.py` (task 5399) mints `CliSubtype` members, leaving the spelling guard's allow-list. Check: grep `'error_empty_output'|subtype='success'|subtype: str = 'error'` in `orchestrator/src/orchestrator/agents/zcode_backend.py` absent. **PASS**
- `orchestrator-free-importers-import-agent-result-from-home`: built here (deletion) — no ξ3 file imports AgentResult through cli_invoke (single-line form; the layering guard covers the rest). Check: grep `from shared\.cli_invoke import .*\bAgentResult\b` in 23 files (listed in the yaml) absent. **PASS**

**ξ4 — Repoint the orchestrator test importers that wave-2 tasks also hold, rebase-only under file locks**
- Note: Contributes fk-3ae32b390a81. Every file is a test file, so no edge onto 9236 or any wave-2 holder (lead ruling of 2026-11-12).
- `orchestrator-held-tests-import-agent-result-from-home`: built here (deletion) — no ξ4 file imports AgentResult through cli_invoke (single-line form). Check: grep `from shared\.cli_invoke import .*\bAgentResult\b` in 19 files (listed in the yaml) absent. **PASS**

**λ — Thread the scratch root as a parameter; the sweep interception point goes**
- Note: Closes fk-734bc1fc18bf (weakened, re-verified). Absorbs task 8234.
- `default-scratch-root`: built here — config_dir.default_scratch_root. Check: grep `def default_scratch_root\(` in `shared/src/shared/config_dir.py` present. **PASS**
- `sweep-parameter-gone`: built here (deletion) — the required sweep callable. Check: grep `sweep: Callable` in `shared/src/shared/config_dir.py` absent. **PASS**
- `interception-tests-gone`: built here (deletion) — the two interception-point tests. Check: grep `def test_the_module_level_sweep_name_is_still_the_interception_point` in `shared/tests/` absent. **PASS**

**μ — Delivered-check kinds as a discriminated union; consumers dispatch on the model**
- Note: Closes fk-240df0521f83 (weakened, re-verified). Absorbs task 7376. Depends on S10 η 9240 (S10 C3) and θ 9242 (INV-16), and on S16 ζ1 (`agents/briefing.py`).
- `delivered-check-module-upstream`: producer:task-9240 (S10 η moves the validator into shared/delivered_check.py; upstream of μ). Check: manual — dependency edge μ → 9240, wired at decompose. **PASS**
- `kind-models`: built here — GrepCheck/ScriptCheck/PathCheck/ManualCheck. Check: grep `class GrepCheck\(` in `shared/src/shared/` present. **PASS**
- `meta-parser-public`: built here — parse_delivered_check_meta, called by the four construction sites. Check: grep `def parse_delivered_check_meta\(` in `shared/src/shared/` present. **PASS**
- `ladder-gone`: built here (deletion) — _check_kind_conditional_fields. Check: grep `def _check_kind_conditional_fields` in `shared/src/shared/` absent. **PASS**
- `briefing-kind-table-gone`: built here (deletion) — the briefing renders model fields. Check: grep `_DELIVERED_CHECK_FIELDS_BY_KIND` in `orchestrator/src/` absent. **PASS**

**ν — Type the invocation backend with the shared Backend enum (S16's role-and-backend vocabulary, S16/C1)**
- Note: HELD on S16/C1 (one `Backend` StrEnum and its decoder `parse_backend` in `shared/backends.py`, S16 γ0; x_seam_wait), unless S16 γ0 already has a task id at decompose. Closes no key. Follows S16 π on `shared/account_state.py` through ο3 → ο2 → S16 π.
- `backend-typed`: producer: S16 C1, one `Backend` StrEnum with its one decoder `parse_backend` in `shared/backends.py` (S16's shared role-and-backend vocabulary leaf γ0, not yet filed; S16 is folded and C1 carries `ZCODE` and `MIMO`); ν held on it; built here once released. Check: grep `backend: Backend` in `shared/src/shared/cli_invoke.py` present. **OPEN**

**ο1 — Fused-memory files that wave-2 tasks hold adopt the new shared homes**
- Note: Contributes fk-3ae32b390a81. Waits on λ and 9188 (the non-test files' wave-2 holders) and on S16 ε2a and ε2b (`cli_stage_runner.py`, `task_curator.py`); test files take no edge (lead ruling of 2026-11-12).
- `deferral-acceptance-test-stops-calling-slot-report`: built here (deletion) — `fused-memory/tests/test_curation_deferral_acceptance.py`, which no leaf owned at the fold, drives the gate through `observe` or `drive_to` so ο3 can delete `InvokeSlot.report`. Check: grep `slot\.report\(` in `fused-memory/tests/test_curation_deferral_acceptance.py` absent. **PASS**
- `stage-runner-imports-from-homes`: built here — cli_stage_runner imports AgentResult and is_zero_output_timeout from shared.cli_result. Check: grep `from shared\.cli_result import` in `fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py` present. **PASS**

**ο2 — The orchestrator adopts CliSubtype, the transcript leaf and the retry loop as the one writer of invocation cost rows**
- Note: Closes fk-1778eb482209, fk-e59c30a25d27, fk-b8cba955b358. Signal leaf (task-workflow cost rows). Lead ruling S20/OQ-6. Waits on S16 π (`routing.py`, S16's last leaf on it). The factory-level refusal of production stores in eval cells is the S1 plan's λ1/λ2 (eval hermeticity), not this leaf's.
- `eval-cells-write-no-cost-rows`: built here — B19, evals stay hermetic after the writer moves into the loop (cross-stream rule, Leo 2026-11-12); the one-writer test drives a workflow built by `evals/runner.py::_cell_workflow`. Check: grep `_cell_workflow` in `orchestrator/tests/test_invocation_rows_one_writer.py` present. **PASS**
- `budget-alias-gone-from-orchestrator`: built here (deletion) — consumers use CliSubtype members. Check: grep `BUDGET_EXHAUSTED_SUBTYPE` in `orchestrator/src/` absent. **PASS**
- `steward-minted-literal-gone`: built here (deletion) — steward._EMPTY_OUTPUT_SUBTYPES uses CliSubtype members. Check: grep `frozenset\(\{'error_empty_output'` in `orchestrator/src/orchestrator/steward.py` absent. **PASS**
- `workflow-nesting-copy-gone`: built here (deletion) — zero-output evidence reads transcript_evidence. Check: grep `msg = rec\.get\('message'\)` in `orchestrator/src/orchestrator/workflow.py` absent. **PASS**
- `callers-stop-writing-their-own-row`: built here (deletion) — the task workflow and the deep reviewer pass cost_store into the loop (lead ruling S20/OQ-6). Check: grep `cost_store\.save_invocation\(` in `orchestrator/src/orchestrator/workflow.py`, `orchestrator/src/orchestrator/review_checkpoint.py` absent. **PASS**

**ο3 — Transitional re-exports, the old slot verdict methods and the __new__ gate builder go**
- Note: Closes fk-3ae32b390a81, fk-25c96d016239, fk-8805d20c0c4c; contributes fk-1778eb482209 (the spelling allow-list's last entries). Signal leaf (end state).
- `second-classify-gone`: built here (deletion) — UsageGate.detect_cap_hit and InvokeSlot.detect_cap_hit. Check: grep `def detect_cap_hit` in `shared/src/shared/` absent. **PASS**
- `slot-report-gone`: built here (deletion) — InvokeSlot.report. Check: grep `def report\(self, outcome` in `shared/src/shared/usage_gate.py` absent. **PASS**
- `new-built-gate-gone`: built here (deletion) — _orch_helpers builds through the constructor. Check: grep `UsageGate\.__new__` in `orchestrator/tests/_orch_helpers.py` absent. **PASS**
- `class-level-park-defaults-gone`: built here (deletion) — no __new__-built gate remains. Check: grep `^    _park_waiters: int = 0` in `shared/src/shared/usage_gate.py` absent. **PASS**

**π — The cap classifier also recognises the CLI's structured usage-limit field, not only banner text**
- Note: No S20 key (side finding of the task-8253 investigation, 2026-11-12; heuristic 12). Priority low; after η. The reset time and cap scope still come from the banner text.
- `classifier-reads-usage-limit-field`: built here — classify_invocation's cap arm recognises api_error usage_limit_reached. Check: grep `usage_limit_reached` in `shared/src/shared/invocation_outcome.py` present. **PASS**
- `result-carries-api-error`: built here — AgentResult.api_error set from the CLI result JSON. Check: grep `^    api_error: str` in `shared/src/shared/` present. **PASS**
