# Capability Manifest — W4 invocation-outcome

Mechanizes G3 (assumed-substrate verified) + G6 (premise validity) for
`plans/invocation-outcome-prd.md`. Each task's observable signal is decomposed into the
capabilities it asserts; each capability is bound to evidence. Any `FAIL`-class binding
blocks the batch. **All bindings PASS.**

Substrate greps re-anchored on `main @ dd73359c81` (2026-07-06); M2's landing did not
touch `usage_gate.py` / `cli_invoke.py` / `steward.py` / `config_dir.py`.

**Legend:** `producer:<label> upstream` = capability delivered by an upstream batch task
in the transitive dependency closure (DAG-direction verified). `grep:<file>:<line>` =
pre-existing substrate on main. `rejection-check` = negative/precedence assertion the
implementer must **observe firing**, not assume.

---

## α — `invocation_outcome.py` + `classify_invocation` + golden corpus
Signal (B3): each historical cap/error string → the expected `InvocationOutcome` variant,
per backend, `strict_confirm` honoured.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `AgentResult` fields the classifier reads (`is_error`, `subtype`, `api_error_status`, `returncode`, `stderr`, `output`) | `grep:shared/src/shared/cli_invoke.py:1141` `_parse_claude_output` produces `AgentResult`; fields consumed today at `usage_gate.py:366`/`cli_invoke.py:905` | PASS |
| Backend string tables to consolidate (`CAP_HIT_PREFIXES`, `CAP_CONFIRM_KEYWORDS`, `NEAR_CAP_PREFIXES`, `NON_CAP_CLI_ERROR_MARKERS`, `CODEX_CAP_PATTERNS`, `GEMINI_CAP_PATTERNS`) | `grep:usage_gate.py:48,78,86,90` + `cli_invoke.py:120` | PASS |
| Real `backend` values (`claude`/`codex`/`gemini`) | `grep:usage_gate.py:386` (`if backend == 'codex'`), `:392` (`elif backend == 'gemini'`) | PASS |
| Golden-corpus strings exist | historical fix-commit strings cited in finding 1/4 histories (ba38ce4ee1, b88b4625d5, e3df395c9f, 77d1d18c49, b5f6b04ac1, 1e8a9b2dd0, 66daedbc76, 7d1fa90075/reify-3604) → materialised as a checked-in fixture (Q5) | PASS |
| `InvocationOutcome` sum type | **this task creates it** (new file) | PASS |

## γ — `AccountPhase` + `_transition` sole-writer
Signal (B1/B2/B8/B10): illegal transitions raise; `_open` equivalence property; SIGHUP
uncaps; shutdown refuses probes.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| The 6 flags to fold into one phase (`capped/near_cap/probing/probe_in_flight/auth_failed` + timestamps) | `grep:usage_gate.py:114` `AccountState` (fields :119-136) | PASS |
| The ~10 `_open` recompute sites to unify | `grep:usage_gate.py:303,335,465,535,601,701,834,897,1177,1211` | PASS |
| Handlers to migrate onto `_transition` (`_handle_cap_detected`, `_handle_auth_failure`, `confirm_account_ok`, `release_probe_slot`, `_refresh_capped_accounts`, probe loops, `_on_sighup_async`) | `grep:usage_gate.py:430,493,1156,1179,816,910,663` | PASS |
| Background-task handles to own (`resume_task`, `auth_reprobe_task`) | `grep:usage_gate.py:122,136` (`AccountState` fields) | PASS |
| Illegal-transition **rejection** fires | `rejection-check`: `_transition` builds the legal-table assertion **in this same task**; the property test authors an illegal edge and observes the raise (B1). Rejection mechanism co-located with its test ⇒ no cross-task inversion. | PASS |
| `AccountPhase`, `_transition`, `_shutting_down` guard | **this task creates them** | PASS |

## β — rewire the 5 classification sites onto `classify_invocation`
Signal (B4): single-source grep; reify-3604 non-cap regression.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `classify_invocation` | `producer:α upstream` (β→α) | PASS |
| `_transition` (probe/cap sides route through it) | `producer:γ upstream` (β→γ) | PASS |
| The 5 sites to collapse (`detect_cap_hit`, `_run_probe`, `invoke_with_cap_retry` heuristic, `_parse_claude_output`, `classify_agent_failure`) | `grep:usage_gate.py:366,910` + `cli_invoke.py:660,1141,453` | PASS |
| reify-3604 non-cap **rejection** (`CliLocalError`, never `CapHit`) | `rejection-check`: authored a `NON_CAP_CLI_ERROR_MARKERS` stderr, `classify_invocation` returns `CliLocalError` and **no** cap transition fires (B4). `NON_CAP_CLI_ERROR_MARKERS` exists `grep:cli_invoke.py:120`. | PASS |

## δ — `AccountLease` + `before_invoke` returns lease
Signal (B5): attribution reports the invoked account under PROBE_IN_FLIGHT skew.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| Selection predicate to source the lease from | `grep:usage_gate.py:296` (`before_invoke` predicate `not (capped or probe_in_flight or auth_failed)`) | PASS |
| `active_account_name` re-derivation to delete | `grep:usage_gate.py:1135` (skew source — predicate omits `probe_in_flight`, :1137-1139) | PASS |
| `InvokeSlot` to carry the lease | `grep:usage_gate.py:147` (`InvokeSlot`, `account_name` set at :171) | PASS |
| cost attribution consumer reads `lease.name` | `grep:cli_invoke.py` `save_invocation`/failover logs (finding 3 sites 782/863/979) — **field-population:** δ writes the *selected* account's name/token into the lease on the production `before_invoke` path (non-sentinel), test B5 samples `lease.name` | PASS |
| `AccountLease` + `generation` | **this task creates it**; `generation` bumped by `_transition` (`producer:γ upstream`) | PASS |
| `_transition` for lease generation bump | `producer:γ upstream` (δ→γ) | PASS |

## ε — `InvokeSlot.report` atomic transition+settle; delete reach-ins
Signal (B6): zero reach-ins in `cli_invoke`; settled-iff-informed; probe-slot-leak-free.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `InvocationOutcome` (report's argument) | `producer:α upstream` (via β; ε→β→α) | PASS |
| `classify_invocation` producing the outcome in `cli_invoke` | `producer:β upstream` (ε→β) | PASS |
| `_transition` (report maps outcome→phase) | `producer:γ upstream` (via β/δ; ε→β→γ, ε→δ→γ) | PASS |
| `AccountLease` on the slot | `producer:δ upstream` (ε→δ) | PASS |
| reach-ins to delete (`_handle_auth_failure`/`_handle_cap_detected` + `slot.settle()`) | `grep:cli_invoke.py:806,932,944` | PASS |
| probe-slot leak on exception (the 4× class) | `grep:usage_gate.py:1179` `release_probe_slot`; B6 asserts release on **every** exit path incl. exception | PASS |
| `InvokeSlot.report` | **this task creates it** | PASS |

## ζ — extend `invoke_with_cap_retry` with `rebuild_prompt` + `max_cap_retries`
Signal: bound raises after N; `rebuild_prompt` invoked on unresumable cap-retry (stubbed).

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `invoke_with_cap_retry` to extend (post-report shape) | `producer:ε upstream` (ζ→ε) + `grep:cli_invoke.py:660` | PASS |
| the loop's cap-retry / resume decision point where hooks fire | `grep:cli_invoke.py:660-1001` (the policy loop) | PASS |
| `rebuild_prompt`, `max_cap_retries` params | **this task adds them** | PASS |

## η — delete `steward._invoke_steward` fork; route through shared loop
Signal (B7): steward inherits wedge guard (zero-turn → fresh-session, not infinite
resume); no direct `invoke_slot`/`detect_cap_hit` in `steward.py`.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| the forked loop to delete | `grep:orchestrator/src/orchestrator/steward.py:536-607` (`for _ in range(_MAX_CAP_RETRIES)` :536, `invoke_slot()` :537, `detect_cap_hit` :563) | PASS |
| `invoke_with_cap_retry` + the two hooks | `producer:ζ upstream` (η→ζ) | PASS |
| steward already imports the shared loop (partial adoption exists at :634) | `grep:steward.py:28` (import), `:634` (existing call site) | PASS |
| zero-output wedge guard the fork lacks | `grep:cli_invoke.py` wedge guard (finding 2: shared loop has it at the 842 region); η inherits it by routing through the shared loop | PASS |
| `max_cap_retries` bound value | `= 16` reused (`grep:steward.py:54` `_MAX_CAP_RETRIES = 16`; Q3) | PASS |

## θ — probe config dir unique per `(account, pid)` + env-precedence test
Signal (B9): distinct dirs per `(account,pid)`; env token precedence observed.

| Capability asserted | Evidence | Verdict |
|---|---|---|
| the shared-fixed probe dir to make unique | `grep:usage_gate.py:227` (`TaskConfigDir('usage-gate-probe')`) → `grep:config_dir.py:34-37` (fixed `/tmp/claude-config-usage-gate-probe`) | PASS |
| `os.getpid()` + `acct.name` available at construction | stdlib `os`; `AccountState.name` `grep:usage_gate.py:117` | PASS |
| env-token **precedence** over config-dir cred | `rejection-check`/`must-observe`: finding 5 flags this as "asserted nowhere". θ **authors** the conflict (env token + differing config-dir `.credentials.json`) and **observes** which wins (B9). If config-dir wins ⇒ larger latent bug ⇒ **escalate**, do not silently pass. | PASS (as a must-observe gate) |

## ι — `UsageGate.shutdown()` refuses new probe tasks
Signal (B10): post-shutdown `CapHit` transition spawns no probe task (gate-enforced).

| Capability asserted | Evidence | Verdict |
|---|---|---|
| `_shutting_down` guard checked by `_transition` | `producer:γ upstream` (ι→γ; γ introduces the guard, ι wires `shutdown()` to set it + asserts the invariant) | PASS |
| `shutdown()` exists to guard | `grep:usage_gate.py` `shutdown` (harness calls `usage_gate.shutdown()` `grep:harness.py:1680`) | PASS |
| the respawn hazard being closed | finding cross-system note: `harness.py:1618-1636` cancel-before-shutdown workaround; ι enforces the invariant in the gate (harness edit deferred, §10) | PASS |

## κ — integration gate: the §8 two-way boundary suite (LEAF)
Signal: full B1–B10 boundary suite green facing both producer and consumer sides.

| Capability asserted | Evidence (all upstream in κ's closure via ε, η, θ, ι) | Verdict |
|---|---|---|
| `classify_invocation` + `InvocationOutcome` (B3/B4) | `producer:α` (via ε→β→α) | PASS |
| `AccountPhase` + `_transition` legal table + `_open` invariant (B1/B2/B8) | `producer:γ` (via ε, θ, ι) | PASS |
| `AccountLease` attribution (B5) | `producer:δ` (via ε→δ) | PASS |
| `InvokeSlot.report` atomicity + no reach-ins (B6) | `producer:ε` (κ→ε) | PASS |
| steward wedge-guard inheritance (B7) | `producer:η` (κ→η) | PASS |
| probe-dir isolation + env precedence (B9) | `producer:θ` (κ→θ) | PASS |
| shutdown refuses probes (B10) | `producer:ι` (κ→ι) | PASS |

**Every capability a κ signal requires is delivered by κ's transitive prerequisites
(never by a task that depends on κ).** DAG-direction: no inversion. Field-population:
δ's lease is written non-sentinel on the production path. Numeric-floor: N/A (no numeric
bounds). Grammar-fixture: N/A (Python, no DSL). Rejection-mechanisms (B1 illegal-raise,
B4 non-cap, B9 env-precedence) are each **observed to fire**, not assumed.

**Gate result: no FAIL bindings — batch clears for queueing.**
