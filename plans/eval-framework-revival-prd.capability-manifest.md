# Capability Manifest — eval-framework-revival-prd

Mechanizes G3 (substrate exists/**wired**) + G6 (premise valid) per task. One
block per task carrying an asserted capability/signal; each capability bound
to on-main evidence. Any **FAIL** binding blocks queueing.

**Domain flags:** tooling/infra domain — **no grammar/DSL** → grammar-fixture
checks **N/A**; **no numeric accuracy bounds/method floors** → numeric-floor &
closed-form-exactness checks **N/A**. The framework's numbers (≥3 trials, CI95)
are *methodology* assertions (run N times), not accuracy bounds — no floor
applies. Live checks here: **capability→producer (wired)**, **DAG-direction**,
**field-population**, **rejection-mechanism**.

All `file:line` evidence re-confirmed on current main during the authoring
session (2026-07-12); line numbers are current-main anchors.

---

## α — `build_workflow()` factory  *(intermediate; single-point ownership)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Two divergent `TaskWorkflow(` construction sites exist to unify | capability→producer (wired) | **confirmed** — `harness.py:5604` (production) + `evals/runner.py:266` (eval), both hand-wire the ctor today | PASS |
| A factory can construct the production `TaskWorkflow` | capability→producer (wired) | producer = α; extraction target is the existing ctor call — no new substrate | PASS |
| Grep-guard tripwire: no `TaskWorkflow(` outside the factory (P2) | rejection-mechanism | producer = α — α's tripwire test authors the guard and observes it fire on a synthetic direct-construction; rejection is α's deliverable | PASS (built+bound by α) |

## β — eval profile via `model_copy` + parity tripwire (D5/D3/D4)  *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `OrchestratorConfig.model_copy(update=…)` derives an eval config from `load_config()` | capability→producer (wired) | **substrate exists** — `OrchestratorConfig` is pydantic-settings (config.py); `model_copy` is standard pydantic v2; `build_eval_orch_config` (runner.py:88) is the replacement site | PASS |
| `rebase_before_verify` / `inter_iteration_rebase` are real config fields set False in profile (D3) | capability→producer (wired) | **confirmed** — `config.py:1544` / `config.py:1538`, both `default=True` today (the leak) | PASS |
| `unblock_auto.enabled` is a real field set False in profile (D4) | capability→producer (wired) | **confirmed** — gated at `workflow.py:8928` (`self.config.unblock_auto.enabled`) | PASS |
| Parity tripwire: eval-config field-diff vs `load_config()` **equals** `EVAL_PROFILE` (P1) — fails on undocumented divergence | rejection-mechanism | producer = β — β's RED test enumerates the diff and asserts equality; the D5 root-cause guard is β's deliverable, not assumed substrate | PASS (built+bound by β) |
| `auto_eval_enabled` / `simple_task_enabled` are real fields | capability→producer | **exists** — referenced in config + workflow (auto-eval gate; SIMPLE_TASK fast path per CLAUDE.md `metadata.complexity`) | PASS |

*G6 note (tactical, → PRD Open-Q 1):* `model_copy(update=)` skips validators
in pydantic v2 — covered by `base=load_config()` being pre-validated + P1;
not a FAIL (profile leaves are bools).

## γ — thread committed base into `get_diff` (D1)  *(leaf; the worst bug)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `task['pre_task_commit']` is the authoritative base commit | capability→producer (wired) | **confirmed present** on the task record — `pre_task_commit` referenced in artifacts/workflow (the brief's D1 fix target); grep-anchored this session | PASS |
| `get_diff` reaches the committed diff for a `.task-meta`-relocated worktree (P3) | capability→producer (wired) | producer = γ; **substrate exists** — `meta_root_for` (`artifacts.py:180`) owns the `.task-meta/<name>` path shape; γ threads `base_commit` instead of the metadata.json read | PASS |
| Judge/compare **grade a non-empty diff** end-to-end (B4) | field-population | producer = γ writes the real committed diff into the value `judge.py:75-76` / `compare.py:390` consume (today they inherit the empty-diff sentinel `{}` — the exact false-premise this fixes) | PASS (γ populates the non-sentinel diff) |
| `--worktree` resume reads the same diff | capability→producer (wired) | producer = γ; `--worktree` resume path (cli.py `--reuse-worktree` option, cli.py:622) reads via the same `get_diff` | PASS |

## δ — `_StubMcpSession` completion + dispatch-literal tripwire (D2)  *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `set_task_claimant` is a real dispatched tool the stub must answer | capability→producer (wired) | **confirmed** — `scheduler.py:1970` `dispatch_tool('set_task_claimant', …)`, heartbeat every 60 s; `_StubMcpSession` raises on unknown tool at `runner.py:607` (the warning-spam source) | PASS |
| `add_dependency` likewise | capability→producer (wired) | **exists** — dispatched from scheduler/workflow; stub lacks the branch | PASS |
| Dispatch-literal tripwire: every `dispatch_tool(...)` string literal in scheduler.py has a stub branch (B3) — fails on a new literal | rejection-mechanism | producer = δ — δ's tripwire enumerates the literals (`scheduler.py` `dispatch_tool` call sites at :1741,:1837,:1970,…) and asserts stub coverage; the guard is δ's deliverable | PASS (built+bound by δ) |
| No `set_task_claimant` warning spam in a ≥90 s run (B9) | field-population | producer = δ populates the stub response so no warning is logged | PASS |

## ε — isolate eval memory writes (D8)  *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Eval config's fused-memory endpoint is redirectable to a null/recording sink | capability→producer (wired) | **confirmed** — the real fused-memory URL is passed at `runner.py:231`; producer = ε overrides it in the eval profile (C1.mem) | PASS |
| Production `dark_factory` store write-count delta = 0 after a green run (B5) | field-population | producer = ε; the recording sink captures intended writes; assertion reads the production store's own count via its read path (not by peeking) | PASS |

## ι — Phase-1 integration gate (boundary suite B1–B9)  *(leaf — C-as-integration-gate)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Every B1–B9 capability | DAG-direction | all produced by α/β/γ/δ/ε, **upstream** of ι (ι depends α,β,γ,δ,ε) | PASS |
| One refreshed fixture exists to drive the suite | DAG-direction | ι runs against an existing April fixture (`df_task_12|13|18`, present on `evals/*` branches per brief); the ~10–14 re-cut is ζ, **downstream** of ι — ι does **not** need ζ | PASS |
| A real `TaskWorkflow` runs end-to-end in eval mode | capability→producer (wired) | **exists** — eval-mode seams on main: `initial_plan` skips PLAN, `_worktree_external` skips MERGE, eval branch in `_wait_for_resolution` (brief G3, re-confirmed) | PASS |

## ζ — task-set sampler + re-cut ~10–14 fixtures  *(intermediate → Phase 3 substrate)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Completed tasks of the last ~6 weeks are queryable, both repos | capability→producer (wired) | **exists** — task store (`get_tasks`/`get_task`, sqlite backend) + git history on both repos; CostStore/`runs.db` for the economics strata | PASS |
| Landed diff / verify-gate-at-SHA / frozen plan are capturable per fixture | capability→producer (wired) | **exists** — landed diff via git at the pinned SHA; repo verify gates are the project verify suites at that SHA; frozen plan is the existing frozen-plan fixture shape (implementer eval already consumes frozen plans) | PASS |
| `evals/<task_id>` pin convention | capability→producer (wired) | **exists** — April fixtures already pinned this way (`df_task_12|13|18`, `reify_task_12|27`) | PASS |
| Reify fixture refs worded generically (DF path-scope guard) | rejection-mechanism | **known guard** — `DarkFactoryPathScopeViolation` scans descriptions for reify-exclusive prefixes; ζ's task text must not cite `crates/…` literally (authoring constraint, honored in decompose) | PASS (constraint noted) |

## η — adversarial fixtures + recovery rubric  *(intermediate → μ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| A `recovery_score` column can be emitted separately from composite | field-population | producer = η defines the rubric; λ's report schema (C4) carries `recovery_score | null`; η populates it for adversarial fixtures | PASS |
| Fixtures with a planted regression / wrong plan-step / misleading verify-failure are constructible | capability→producer | producer = η authors them against ζ's real fixtures (mutate a real landed diff/plan) — no novel substrate | PASS |
| DAG-direction: depends on ζ's fixtures | DAG-direction | ζ **upstream** of η | PASS |

## θ — architect eval harness (live planning)  *(intermediate → μ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| The architect **can be invoked live** in eval (not frozen) | capability→producer (wired) | **substrate exists** — the eval reuses the real `TaskWorkflow`; today `initial_plan` **skips** PLAN to freeze the plan, so the architect seam is present and toggleable; producer = θ drives the un-frozen path against fixtures | PASS |
| A real landed plan exists as the judge reference | DAG-direction | producer = ζ captures the frozen/landed plan per fixture, **upstream** of θ | PASS |
| Plan-quality rubric score is populated per fixture | field-population | producer = θ writes a non-sentinel plan-quality score into the report row | PASS |

## κ — reviewer_trial refresh (low priority)  *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| reviewer_trial corpus + runner exist to re-run | capability→producer (wired) | **confirmed** — `evals/reviewer_trial/{corpus,runner.py,scorer.py,variants.py}` present on main | PASS |
| Cost ranking (Opus vs candidates) populated | field-population | producer = κ; reuses λ's price table (κ depends ι; price table is λ — see note) | PASS-with-dep |

*κ dep note:* κ's cost column needs λ's price table; if κ is scheduled before
λ lands, its quality ranking is still valid and cost is deferred. Decompose
may add `κ depends λ` to make the cost column non-sentinel. Not a FAIL.

## λ — composite + price table + statistics  *(intermediate → μ)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Composite/report machinery to extend | capability→producer (wired) | **confirmed** — `evals/{metrics,report,compare,elo}.py` present on main | PASS |
| `tests_pass` hard gate exists to keep | capability→producer (wired) | **exists** — the existing composite gates on tests (April framework); λ preserves it | PASS |
| Price-table-derived `cost_usd` + `latency_secs` populated (P5) | field-population | producer = λ writes non-sentinel cost from the price table (NOT CLI `cost_usd`, which is wrong for proxied endpoints); latency measured per run | PASS |
| CI95 populated from ≥3 trials | field-population | producer = λ; ≥3-trial variance is a methodology run (no numeric floor) — populated, not sentinel | PASS |
| Aggregation replaces the all-tasks-intersection collapse + noisy Elo | capability→producer | producer = λ; the collapse/Elo are existing `elo.py`/`compare.py` behavior being replaced (Open-Q 3) | PASS |

## μ — OFAT→matrix→confirm driver + contract-agnostic scoring  *(leaf; G2 top signal)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `orchestrator eval --matrix` CLI surface exists | capability→producer (wired) | **confirmed** — `@main.command('eval')` + `--matrix` flag at `cli.py:592,597`; producer = μ adds OFAT/confirm sub-modes (Open-Q 4) | PASS |
| Runs ≥2 candidate configs end-to-end via cloud endpoints | end-to-end capability | trace: candidate configs ← ν/ξ (Phase 4) **but** incumbents (Opus/Sonnet) need **no** Phase-4 substrate → μ's ≥2-config signal is satisfiable with **incumbent-only** cloud configs (Opus vs Sonnet OFAT) that exist today; non-incumbent bundles are additive | PASS (incumbent configs satisfy the ≥2 floor without Phase 4) |
| Report carries composite/judge/cost/latency/CI, non-empty diffs graded | end-to-end capability | every field traces to an **upstream** producer: non-empty diff ← γ; composite/cost/latency/CI ← λ; architect score ← θ; recovery ← η — all upstream of μ (no inversion) | PASS |
| Contract-agnostic scoring off persisted artifacts (P4, B8) | capability→producer (wired) | producer = μ reads persisted verdict/plan/diff artifacts, not transcripts; substrate = artifacts already persisted by the review/judge path (`rereview.py`, artifacts.py). Tracks whichever contract mcp-verdict-servers made live — **no coupling to a specific contract** | PASS |

## ν — Claude-format-endpoint candidate bundles  *(leaf; cross-PRD-gated)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Per-role `ANTHROPIC_BASE_URL`/`AUTH_TOKEN` forwarding to a candidate endpoint (C5) | capability→producer (wired) | producer = **harness-reconnect-pi's per-role `env_overrides` forwarding task** — **NOT on main today**. Resolution (G3-b): cross-PRD dep on that task; ν filed but **held `deferred` until the dep target exists**, then wired + flipped | **PASS-pending-cross-PRD** (dep must exist before ν → pending) |
| Non-incumbent Claude-format endpoints (MiniMax/GLM/DeepSeek/Kimi) are reachable behind Claude Code | capability→producer | external endpoints; reachability is the bundle config's own concern, exercised only once the env-forwarding dep lands | PASS-pending-cross-PRD |
| Cost recorded from the price table | field-population | producer = λ's price table, **upstream** (ν depends μ→λ) | PASS |

## ξ — codex/pi candidate bundles  *(leaf; cross-PRD-gated)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Codex CLI backend hardened for dispatch | capability→producer (wired) | producer = **harness-reconnect-pi's codex-hardening task** — **NOT on main today**. Resolution (G3-b): cross-PRD dep; ξ held `deferred` until it exists | **PASS-pending-cross-PRD** |
| pi backend + spike landed (incl. pi+Sonnet control) | capability→producer (wired) | producer = **harness-reconnect-pi's pi-backend + spike tasks** — **NOT on main today**; same cross-PRD resolution | **PASS-pending-cross-PRD** |
| Rust implementer dispatch via codex records a result | end-to-end capability | trace: needs codex backend (above, cross-PRD upstream) + μ's driver (upstream). No inversion. Rust caution (decision 14): evaluate only, no prod change | PASS-pending-cross-PRD |

---

## Manifest verdict

- **Phase 1–3 (α, β, γ, δ, ε, ι, ζ, η, θ, κ, λ, μ): all bindings PASS.** No
  novel substrate — every capability is wired on main today, or produced
  upstream in-batch. The G2 top signal (μ) is satisfiable with **incumbent
  cloud configs** (Opus vs Sonnet), so Phase 1–3 can decompose, wire
  intra-batch deps, commit this manifest, and flip to **pending** immediately.
- **Phase 4 (ν, ξ): PASS-pending-cross-PRD.** Their only FAIL-risk
  capabilities (per-role env forwarding, codex/pi backends) are **owned by
  harness-reconnect-pi** and are **not on main today**. Per G3 resolution (b),
  they are filed as tracked tasks with a **cross-PRD dependency** on the
  relevant harness-reconnect-pi tasks; they are held **`deferred`** (not
  flipped to pending) until those task IDs exist and the dep is wired. Flipping
  ν/ξ to pending before then would dispatch a task whose substrate is absent —
  the exact stall G3 prevents.

**Decompose-time action:** file α–μ, wire intra-batch deps, commit manifest,
`commit_planning` α–μ → pending. File ν, ξ (deferred) with intra-batch deps to
μ wired; add the cross-PRD deps to harness-reconnect-pi's env-forwarding /
codex / pi tasks **once those exist**, then flip ν/ξ to pending in a follow-up.
Surface this split in the decompose hand-back.
