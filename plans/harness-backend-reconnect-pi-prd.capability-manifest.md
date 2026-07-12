# Capability manifest — harness-backend-reconnect-pi-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized). Evidence verified on `main`
`7b6c1f829c`, 2026-07-12. Line refs drift; symbols are canonical. Sub-checks applied:
anti-orphan/wired, anti-inversion (no owner depends on its consumer), field-population,
numeric-floor. No grammar-fixture sub-check applies (no parser/grammar substrate in this PRD).

## T1 — Reconnect: forward `backend` through the dispatch seam (`shared/cli_invoke.py`)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `invoke_with_cap_retry` receives a `backend` param it can forward | grep:`cli_invoke.py:715` `backend: str = 'claude'`; `**invoke_kwargs` :716 | PASS wired |
| The dispatch call site to inject the forward exists | grep:`cli_invoke.py:884` `invoke = invoke_fn or invoke_claude_agent`; `await invoke(**invoke_kwargs)` :896, :917 | PASS wired |
| `invoke_agent` (the custom `invoke_fn`) accepts + acts on `backend` | grep:`invoke.py:66` `backend: str = 'claude'`; dispatch `elif backend == 'codex'/'gemini'` :108/:115; `Unknown backend` guard :123 | PASS wired |
| The default `invoke_claude_agent` path must stay backend-kwarg-free (recon safety) | grep:`cli_invoke.py:884` default branch; forward gated on `invoke_fn is not None` (§5 decision 1) — Invariant 3 | PASS (rejection-backed: recon path never receives `backend`) |
| `backend` currently dies at cap-classification only (the defect being fixed) | grep:`cli_invoke.py:936` `classify_invocation(..., backend=backend)`; :1007 `detect_cap_hit(..., backend=backend)`; prose `steward.py:556-561` | PASS (defect present, one wire to reconnect) |
| Consumers route through the seam (no per-call-site backend plumbing) | consumers T3/T4 pass `backend=<b>` via existing `invoke_with_cap_retry(..., invoke_fn=invoke_agent, backend=...)` calls (workflow.py:7456, steward.py:575) | PASS wired |
| DAG-direction (anti-inversion) | T1 is upstream of T3/T4/T-dedupe; no owner depends on its consumer | PASS producer upstream |

## T2 — pi empirical spike (`plans/pi-spike-findings.md` + throwaway `scripts/` probe)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| pi CLI is installable + runnable (external substrate) | brief research: `@earendil-works/pi-coding-agent` v0.80.6 (github.com/earendil-works/pi), `pi-mcp-adapter` v2.11.0 (github.com/nicobailon), MIT; sources dated 2026-07-12 | PASS (external, brief-verified; the spike RE-verifies empirically) |
| The seven questions are answerable by observation, not decode | Appendix B enumerates each with the observation method (exit code, JSONL, RPC, tool-name string) | PASS (empirical, producible) |
| Deliverable is a committed artifact (not a bare go/no-go decision) | signal = `plans/pi-spike-findings.md` exists and answers all 7 with evidence; may commit a `scripts/` probe | PASS (concrete artifact; not a churning no-code decision) |
| No numeric/exactness claim asserted | the spike REPORTS observed values; asserts no floor/rate | PASS (floor branch n/a) |
| DAG-direction | T2 is upstream of T4; independent of T1/T-price | PASS producer upstream |

## T-price — Config-driven model price table (`config.py` + `defaults.yaml` + `invoke.py`)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Hardcoded price table exists to replace | grep:`invoke.py:42-49` `_MODEL_COSTS`; silent fallback `{'input':2.0,'output':8.0}` :393 | PASS wired |
| Config layer accepts a new `prices` map | grep:`config.py` `OrchestratorConfig` model + `defaults.yaml` top-level sections (e.g. `backends:` :210) — additive field | PASS wired (additive) |
| Cost estimation site to route through config | grep:`invoke.py:393-394` `rates = _MODEL_COSTS.get(model, ...)`; `cost = (in*rate + out*rate)/1e6` | PASS wired |
| **Fallback is loud, not silent** (rejection/negative assertion) | signal: un-listed model ⇒ **logged warning** + defined fallback, replacing the silent `$2/$8` (§5 decision 5) — the absence of a silent default is the observable | PASS (rejection-backed) |
| Numeric-floor sub-check | no accuracy/throughput floor asserted; prices are config values, not a bound | PASS (floor branch n/a) |
| Consumer named (anti-orphan) | consumed by T3 (codex cost), T4 (pi cost), and eval-revival price table (G4) | PASS wired |

## T3 — Codex hardening to trialable (`invoke.py` + `config.py` + `defaults.yaml`; deps T1, T-price)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `AGENTS.md` is written into the worktree today (the leak to close) | grep:`invoke.py:274-276` `agents_md = cwd / 'AGENTS.md'`; unlink only in `finally` :304-306 | PASS wired |
| The staging path would commit it (rejection mechanism to satisfy) | grep:`git_ops.py:591` `git add -A`; excludes only `.task/` (`:!.task/` :5150,:5216; `.gitignore` :40) → `AGENTS.md` is stage-eligible | PASS (leak real; signal = never in committed diff) |
| Codex caps are droppable/enforceable at the call site | grep:`invoke.py:259-306` `_invoke_codex` passes no `max_turns`; `max_budget_usd` only to `_run_subprocess_local` | PASS wired |
| Codex cost accounting can read T-price | producer:T-price upstream; `_parse_codex_output` cost calc invoke.py:392-394 | PASS producer upstream |
| Codex MCP serializer exists (no new substrate for MCP) | grep:`invoke.py:409` `_write_codex_mcp_config` | PASS wired |
| Anti-orphan/wired | codex backend is dispatchable only after T1 (dep declared); consumer = eval-revival codex bundle | PASS wired |
| DAG-direction | T3 depends on T1/T-price (upstream); no inversion | PASS producer upstream |

## T4 — pi backend (`invoke.py` + `roles.py` + `config.py` + `defaults.yaml`; deps T1, T2, T-price)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| pi runtime contracts (exit codes, cost source, tool-name format, liveness path) | producer:T2 (spike) upstream — **the G3 explicit-prerequisite**; not assumed from main | PASS (prerequisite queued, not guessed) |
| Dispatch hook to add `elif backend == 'pi'` | grep:`invoke.py:91-123` dispatch chain + `Unknown backend` guard | PASS wired |
| `AgentResult` shape to populate | grep:`invoke.py:396-406` `_parse_codex_output` returns `AgentResult(success,cost_usd,duration_ms,turns,session_id,...)` — same target for `_parse_pi_output` | PASS wired |
| Role allowed/disallowed lists to map to `--tools` | grep:`roles.py` per-role `allowed_tools`/`disallowed_tools` (e.g. reviewer grant roles.py:426) | PASS wired |
| BackendsConfig admits `'pi'` (no validator blocks it) | grep:`config.py:255-267` fields are freeform `str` (no enum) — only `invoke_agent`'s `else: raise` guards unknown; T4 adds the `pi` branch | PASS wired (additive) |
| **Field-population** — `_parse_pi_output` populates cost/turns/session from real events, not sentinels | signal binds cost aggregation to the spike-chosen source (Usage events / JSONL / RPC), asserting non-zero real values on a successful run; empty-output ⇒ defined error result (mirrors `_parse_codex_output` :317-324) | PASS (populated from observed events; sentinel only on genuine empty output) |
| Liveness watchdog wiring (seam owned here) | producer:T2 identifies the session-JSONL path; analog of Claude transcript glob `_run_subprocess` (cli_invoke.py) | PASS producer upstream |
| DAG-direction | T4 depends on T1/T2/T-price; no inversion | PASS producer upstream |

## T5 — Per-role env forwarding (`config.py` + `workflow.py` + `defaults.yaml`)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `_build_agent_env` is the single env-merge seam | grep:`workflow.py:7304-7333` `_build_agent_env(role)`; consumed `env_overrides=self._build_agent_env(role)` :7475 | PASS wired |
| Today it is role-gated (the gate to widen) | grep:`workflow.py:7323` `if role.name not in ('architect','implementer','debugger'): return None` | PASS wired |
| Config accepts a per-role env map (additive) | grep:`config.py:2505` global `env_overrides: dict[str,str]` — precedent for the map type; new `role_env_overrides` is additive | PASS wired (additive) |
| **The judge-vLLM safety is preserved by construction** (rejection/negative assertion) | grep:`workflow.py:7470-7474` the exclusion comment (3cd380a079); §3 — per-role opt-in means judge gets env **iff** `role_env_overrides['judge']` is set; signal asserts the judge receives NO endpoint env when its key is absent | PASS (rejection-backed: default-empty ⇒ no off-Claude routing) |
| Consumer named (anti-orphan) | eval-revival Claude-endpoint candidates consume this (G4); operator config sets per-role endpoints | PASS wired |
| No numeric floor asserted | env forwarding is a mapping, not a bound | PASS (floor branch n/a) |

## T6 — CostStore role telemetry (`workflow.py` + `steward.py`)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `save_invocation` already accepts `role` (no schema change) | grep:`cost_store.py:102` `async def save_invocation(... role ...)`; guarded call `cli_invoke.py:1177-1195` `if cost_store:` | PASS wired |
| The recording guard is skipped for judge/steward/triage today (the gap) | grep:`workflow.py:4776` `_run_completion_judge`; steward main `steward.py:570` + pre-triage/triage `steward.py:612` — all call `invoke_with_cap_retry` **without** `cost_store=` | PASS (gap present, empirically brief-verified vs runs.db) |
| Threading `cost_store` is sufficient (fix mechanism exists) | the guarded `save_invocation` fires whenever `cost_store` is passed (cli_invoke.py:1177); `self.cost_store` is in scope on the workflow/steward | PASS wired |
| **Field-population / premise** — the fix's observable is role rows *appearing* | signal: after a run, `runs.db` has rows for `role in {judge, steward, triage}` (currently absent) — presence is the rejection-backed observable | PASS (populated; premise = current absence) |
| No numeric floor asserted | presence-of-row is boolean, not a bound | PASS (floor branch n/a) |
| DAG-direction | independent root; consumes only existing CostStore substrate | PASS |

## T-dedupe — Claude-argv dedupe (`invoke.py`; dep T1; LOW; blocks nothing)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Two claude-argv builders exist to unify | grep:`invoke.py:174-208` sandbox fork (`cmd = ['claude','--print',...]`); non-sandbox builder in `cli_invoke.py` (invoke_claude_agent argv assembly) | PASS wired |
| Parity is testable (the signal) | both builders produce a `claude ...` argv from the same inputs; a parity test asserts equality modulo the sandbox `wrap_command` prefix (invoke.py:213) | PASS (producible) |
| Pure refactor — no behaviour claim, no numeric floor | signal is argv-equality only; no user-observable behaviour change | PASS (floor branch n/a) |
| DAG-direction | depends on T1; blocks nothing (anti-inversion trivially holds) | PASS |

No FAIL bindings. The single not-on-main capability (pi runtime contracts) is queued as an
explicit prerequisite (T2), satisfying G3's prerequisite branch rather than failing it. Batch
clear to queue.
