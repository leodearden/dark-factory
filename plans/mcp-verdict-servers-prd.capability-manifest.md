# Capability manifest — plans/mcp-verdict-servers-prd.md

Binds each leaf task's asserted capabilities to evidence (G3+G6 mechanized).
Verified 2026-07-12 against main @ `7b6c1f829c`. **No FAIL bindings.**

Empty-value sentinel for I-FAIL-SAFE (the "absent verdict" state): a **missing**
`verdicts/<role>.json` after a pre-invocation clear — established as a rejection
mechanism, not a placeholder.

## task α — verdict-tools server + artifact contract (intermediate; roped into θ)

| Capability | Evidence | Verdict |
|---|---|---|
| FastMCP stdio server spawned per-invocation with `--worktree`/`--meta-root`, all writes via `TaskArtifacts` | grep:orchestrator/src/orchestrator/mcp/plan_tools.py:499-526 (`create_server(artifacts)→FastMCP`), :815-859 (`_artifacts_from_args`+`main()`, stdio) — the pattern α mirrors | PASS wired |
| Artifact write/read primitives to mirror (`write_verdict`/`read_verdict`/`clear_verdict`) | grep:orchestrator/src/orchestrator/artifacts.py:492-518 (`write_review`/`read_reviews` = `_write_json`/`_read_path` under `self.root/<subdir>/<name>.json`), :397-398 (unlink helper for clear), :219 (`reviews/` mkdir in `init()` — `verdicts/` added the same way) | PASS wired |
| Agent-side and orchestrator-side resolve the IDENTICAL artifact root | grep:orchestrator/src/orchestrator/mcp/plan_tools.py:815-848 (`_artifacts_from_args` builds `TaskArtifacts(worktree, meta_root)`) mirrors workflow's `_meta_root_for_worktree` (workflow.py:128-139) — verified same-root by the plan-tools contract | PASS wired |
| The four existing schemas the envelope wraps exist unchanged | grep:orchestrator/src/orchestrator/agents/briefing.py:32-44 (`COMPLETION_JUDGE_SCHEMA`), agents/triage.py:184-231 (`TRIAGE_OUTPUT_SCHEMA`), workflow.py:5533-5555 (reviewer `review_schema`); merger schema `{blocked,reason}` is NEW-in-α (replaces the substring, no prior schema) | PASS wired |

## task γ — migrate merger (leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| `submit_merge_disposition` tool | producer:task-α (upstream) | PASS producer-upstream |
| verdict-server injection + `_VERDICT_TOOLS_ROLES` gate | producer:task-β (upstream) | PASS producer-upstream |
| Merger invoked with the worktree as `cwd` (so `_inject_verdict_tools_mcp` fires — needs `cwd`) | grep:orchestrator/src/orchestrator/workflow.py:6332 (`_invoke(MERGER, prompt, self.worktree)`) | PASS wired |
| The substring grep to delete exists exactly where the swap lands | grep:orchestrator/src/orchestrator/workflow.py:6334 (`'BLOCKED' in merger_result.output.upper()`) | PASS wired |
| Merger `allowed_tools` + BLOCKED prose to edit | grep:orchestrator/src/orchestrator/agents/roles.py:590 (merger `allowed_tools`), :543,:569,:588 (BLOCKED prose that produces the grepped word) | PASS wired |
| Rejection/field-population: `read_verdict('merger')` yields a real `blocked` bool the workflow branches on; absent ⇒ blocked-equivalent | rejection-check: the swap reads the α-written envelope's `verdict.blocked`; absent-slot ⇒ `_mark_blocked` (I-FAIL-SAFE). Producer α populates a non-sentinel `blocked` bool. Signal (false-trip removal) OBSERVED in θ scenario 1 | PASS rejection-fires |

## task δ — migrate reviewer panel (leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| `submit_review_verdict` tool | producer:task-α (upstream) | PASS producer-upstream |
| verdict-server injection + gate | producer:task-β (upstream) | PASS producer-upstream |
| Reviewer invoked with worktree `cwd`; the parse cascade to replace | grep:orchestrator/src/orchestrator/workflow.py:5557-5559 (`_invoke(role, prompt, self.worktree, output_schema=review_schema)`), :5561-5577 (`structured_output`→`json.loads`→synth ERROR — the cascade δ replaces) | PASS wired |
| Reviewer payload shape (envelope `verdict`) | grep:orchestrator/src/orchestrator/workflow.py:5533-5555 (`review_schema`: `verdict∈{PASS,ISSUES_FOUND}`, `issues[]`, `summary`) | PASS wired |
| Downstream retry/aggregate path unchanged by the swap | grep:orchestrator/src/orchestrator/workflow.py:5490-5525 (`_review` retries ERROR, `write_review`, `aggregate_reviews`) | PASS wired |
| Inert `mcp__jcodemunch__*` grant to remove (fold-in κ) | grep:orchestrator/src/orchestrator/agents/roles.py:426 (`allowed_tools=[*_READ_ONLY_TOOLS,*_JCODEMUNCH_TOOLS]`), :22 (`_JCODEMUNCH_TOOLS`) — inert (reviewer in no MCP-config gate) | PASS wired |
| Rejection: absent reviewer verdict ⇒ ERROR (retried) | rejection-check: absent-slot ⇒ synth `verdict:'ERROR'` (workflow.py:5516-5521, existing) → retry (:5490-5507). OBSERVED in θ scenario 5 | PASS rejection-fires |

## task ε — migrate triage (leaf; steward path)

| Capability | Evidence | Verdict |
|---|---|---|
| `submit_triage` tool | producer:task-α (upstream) | PASS producer-upstream |
| Triage invocation is in the steward (NOT `workflow._invoke`) — where ε injects + reads | grep:orchestrator/src/orchestrator/steward.py:599-668 (`_pre_triage_escalation`), :622-627 (inlined `allowed_tools` + `output_schema=TRIAGE_OUTPUT_SCHEMA` invocation ε extends) | PASS wired |
| Consumer to replace + its schema | grep:orchestrator/src/orchestrator/agents/triage.py:270-282 (`parse_triage_result` reads `structured_output`, checks 3 keys), :184-231 (`TRIAGE_OUTPUT_SCHEMA`) | PASS wired |
| Rejection/inline fallback: absent triage verdict ⇒ steward inline | rejection-check: grep:orchestrator/src/orchestrator/steward.py:644-650 (`parse_triage_result(...) is None` → return escalation unchanged → inline triage). Absent-slot maps to the same `None` path. OBSERVED in θ scenario 11 | PASS rejection-fires |
| Triage prompt to edit (instruct the tool call) | grep:orchestrator/src/orchestrator/agents/triage.py:234 (`build_triage_prompt`) | PASS wired |

## task ζ — migrate judge, transition window (leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| `submit_completion_verdict` tool | producer:task-α (upstream) | PASS producer-upstream |
| verdict-server injection + gate; judge already in `_MCP_CONFIG_ROLES` (an http `mcp_config` exists to inject into) | producer:task-β (upstream); grep:orchestrator/src/orchestrator/workflow.py:116-118 (judge ∈ `_MCP_CONFIG_ROLES`) | PASS producer-upstream / wired |
| Judge invoked with worktree `cwd`; the consumption + fallback sites | grep:orchestrator/src/orchestrator/workflow.py:4991-4994 (`_invoke(JUDGE,…,output_schema=COMPLETION_JUDGE_SCHEMA)`), :5014-5030 (`result.structured_output` + required-keys check — the transition fallback ζ keeps) | PASS wired |
| The completion gate the verdict feeds (`complete` is load-bearing, not decorative) | grep:orchestrator/src/orchestrator/workflow.py:4775-4801 (gate on `judge_verdict.get('complete') is True` + `substantive_work`) | PASS wired |
| Judge `--json-schema` prompt sentence + `allowed_tools` to edit | grep:orchestrator/src/orchestrator/agents/roles.py:518-519 (the `--json-schema` sentence), :521 (judge `allowed_tools`) | PASS wired |
| Rejection: absent-both (no tool + no structured output) ⇒ `None` ⇒ keep iterating (never false-complete) | rejection-check: grep:orchestrator/src/orchestrator/workflow.py:5007-5028 (every failure mode → `None`). OBSERVED in θ scenarios 8 (transition) + 9 (absent-both) | PASS rejection-fires |

## task η — terminal cleanup + shared-machinery regression guard (leaf)

| Capability | Evidence | Verdict |
|---|---|---|
| The four roles' `output_schema`/substring usages to remove | producers γ/δ/ε/ζ (upstream): merger substring (γ), reviewer `json.loads` (δ), triage `output_schema` (ε), judge transition scaffold (ζ) | PASS producer-upstream |
| Shared `cli_invoke` `--json-schema` path is a DISTINCT surface η must NOT touch | grep:shared/src/shared/cli_invoke.py:174-217 (`_SCHEMA_OUTPUT_TOOL`/deny-list), :1256-1278 (`output_schema`→`--json-schema`; wildcard-expansion only for `['*']` callers — orchestrator roles pass real lists, so removing their `output_schema` does not perturb this block) | PASS wired |
| Fused-memory recon/curator `output_schema` callers = the regression-guard target (still ride the shared path) | grep:fused-memory/src/fused_memory/middleware/task_curator.py:2002,2093; reconciliation/stages/base.py:210-226; reconciliation/cli_stage_runner.py:262,315 (all via shared `cli_invoke`) | PASS wired |
| Rejection (negative assertion): after η, no orchestrator verdict role passes `output_schema` / greps transcript; recon/curator still function | rejection-check: `git grep` for the removed patterns returns 0 in the four roles + a regression test exercises the recon/curator `output_schema` path green (I-SHARED-INTACT). OBSERVED in θ scenario 14 | PASS rejection-fires |

## task θ — integration gate (leaf; B+H boundary-test signal)

| Capability | Evidence | Verdict |
|---|---|---|
| Tools + swaps under test | producers α,β,γ,δ,ε,ζ,η (upstream) | PASS producer-upstream |
| Workflow test rig with a fake agent runner (`invoke_fn`) that can write verdict artifacts | grep:orchestrator/src/orchestrator/workflow.py:7439-7452 (`invoke_with_cap_retry(..., invoke_fn=invoke_agent, ...)` — `invoke_fn` is the injectable seam the rig substitutes); the orchestrator TDD pipeline is test-covered under orchestrator/tests | PASS wired |
| Freshness (I-FRESH) is testable end-to-end | grep:orchestrator/src/orchestrator/artifacts.py:179-204 (`meta_root_for` documents the pooled-lane staleness the pre-invocation clear defends — θ scenarios 12/13 assert it) | PASS wired |

## task ι — done_provenance briefing fix (leaf; independent)

| Capability | Evidence | Verdict |
|---|---|---|
| The stale kind-less prompt to fix | grep:orchestrator/src/orchestrator/agents/briefing.py:863 (`done_provenance={"commit":…}` / `{"note":…}` — both omit `kind`) | PASS wired |
| The validator that REJECTS it (rejection mechanism exists + fires) | rejection-check: grep:fused-memory/src/fused_memory/middleware/task_interceptor.py:4098-4234 (`_validate_done_provenance`), :4199-4213 (missing/empty `kind` hard-errors), :4227-4234 (note-alone rejected post-3092) — the current briefing shape provably fails validation today | PASS rejection-fires |
| The CORRECT shape to align to (the reference) | grep:orchestrator/src/orchestrator/agents/roles.py:885-941 (steward system prompt: `kind="merged"`+`commit` or `kind="found_on_main"`+`commit`+`note` — consistent with the validator) | PASS wired |

## Intermediates (for completeness)

- **task β** (wiring): consumes α; mirrors the plan-tools injection template —
  `_PLAN_TOOLS_ROLES` (workflow.py:125), `_inject_plan_tools_mcp`
  (workflow.py:142-175), `plan_tools_mcp_server` (mcp_lifecycle.py:69-144),
  `_PLAN_CREATOR_TOOLS` grant family (roles.py:80-104). Consumers: γ/δ/ζ.
  Injection creates a `{'mcpServers':{}}` skeleton when `mcp_config is None`
  (workflow.py:169-170), so reviewer (in no MCP-config gate) is reachable
  without joining `_MCP_CONFIG_ROLES`.
