# PRD: Offline eval framework revival & refresh

**Status:** active — authored 2026-07-12 (design session; user AFK. Design
input is the verified decision brief
`~/.claude/spawn-briefs/2026-07-12-eval-revival.md`, which pre-answers G1–G6
and records Leo's explicit decisions; those are honored here, not
re-litigated).
**Project:** dark_factory. **Approach:** **B+H** (contract + boundary-test
sketch). G5 heuristic hit on all four axes: cross-module blast radius ≥ 3
(`orchestrator/evals/*`, `config.py`, `harness.py`, `workflow.py`), mechanism
count ≫ 8, the **load-bearing eval↔production `TaskWorkflow` seam** (the exact
seam that drifted silently-wrong), and **2 cross-PRD consumers**
(harness-reconnect-pi, mcp-verdict-servers).

## Goal

Restore the offline fixed-target eval framework
(`orchestrator/src/orchestrator/evals/`) to **correctness against the current
production orchestrator**, then refresh it so the operator can compare
per-role **(harness, model) candidate bundles** on a representative,
near-HEAD task set via **cloud APIs only** (no RunPod/Docker/vLLM this round).

**User-observable surface (the G2 top signal):** one CLI invocation —
`orchestrator eval --matrix` (or a named successor subcommand) — runs the
refreshed task set against **≥ 2 candidate configs end-to-end via cloud
endpoints** and emits a report carrying, **per config**: composite score,
judge results, **cost, latency, and confidence intervals** — with **non-empty
diffs actually graded** (today the framework grades **empty diffs** — the D1
bug below).

The framework is a *fixed target*: one corpus, many candidates compared
against it. This round evaluates the deployable configuration
(harness + model together), not a bare model.

## Background — why this exists

The eval framework last executed **2026-04-14/15** (297 result JSONs: 176
blocked / 88 timeout / 33 done). Core runner/judge/config code is untouched
since April (only a June-18 `_git_diff_stats` hardening). Meanwhile the
production orchestrator moved under it, and the framework has drifted
**silently-wrong**: all 247 eval-adjacent tests pass (verified 2026-07-12)
**because none of them drive a real `TaskWorkflow` in eval mode** — so the
drift is invisible to CI. Evals reuse the *real production* `TaskWorkflow`
(runner.py:266), which is exactly why every new production default silently
leaks into eval runs.

Economics motivating the refresh (CostStore, `data/orchestrator/runs.db`,
since 2026-06-01): **architect 42.7% / 39.2%** of spend (DF / reify),
**implementer 33.2% / 36.6%**, reviewer ~11–13%, debugger 4–7% —
architect+implementer+debugger ≈ **80–83% of all spend**. The eval today has
**zero architect coverage** (plans are frozen, the architect is never
invoked) yet the architect is the **#1 cost line** — the single biggest gap
this PRD closes.

## Drift inventory (verified 2026-07-12, refs re-confirmed this session)

Each is a concrete, on-main defect. Refs are current-main line anchors
(± a few dozen lines from the brief's April-anchored refs; mechanisms
re-confirmed present this session).

| ID | Defect | On-main evidence | Fix |
|---|---|---|---|
| **D1** (worst) | `.task` → sibling `.task-meta/` relocation (task 2258/W11, `artifacts.py:180` `meta_root_for`) not followed by the eval diff path. `get_diff(worktree_path)` (snapshots.py:103) falls back to uncommitted-only; `judge.py:75-76` & `compare.py:390` then grade **empty diffs**; review artifacts read `{}`; `--worktree` resume broken. | Thread `task['pre_task_commit']` into `get_diff` (or route through `TaskArtifacts.meta_root_for`); drop the metadata.json read. Fix all four callers (runner, snapshots, compare, rereview). |
| **D2** | `_StubMcpSession` (runner.py:533, raises on unknown tool at :607) lacks `set_task_claimant` (a real dispatch tool — scheduler.py:1970 heartbeats every 60 s → warning spam) and `add_dependency`. | Add the missing branches **+ a tripwire test** asserting every string literal passed to `dispatch_tool(...)` in scheduler.py has a stub branch. |
| **D3** | `rebase_before_verify=True` **default** (config.py:1544, landed 2026-04-27) + `inter_iteration_rebase=True` (config.py:1538) → eval worktrees rebase onto **live main mid-eval** for main-ancestor fixtures. | `False` in the eval profile. |
| **D4** | `unblock_auto.enabled=True` default → every blocked/timeout eval run spawns an unmetered ~$5 Sonnet dry-run investigation (workflow.py:8928). | `False` in the eval profile. |
| **D5** (root cause) | `build_eval_orch_config` (runner.py:88) builds `OrchestratorConfig(...)` **by constructor** (runner.py:154) — **every new production field silently lands at its pydantic default**, which is precisely how D3/D4 leaked in. | Derive via `base_config.model_copy(update=EVAL_PROFILE)` with an explicit documented profile. |
| **D8** | Eval passes the **real fused-memory URL** (runner.py:231) → green eval runs write observations into **production dark_factory memory**. | Point at a null/recording endpoint. |

## Sketch of approach

Three seam-level mechanisms carry the whole refresh; everything else is
downstream of them.

1. **Explicit eval profile derived from the live base config
   (`model_copy`).** Closes D5 at the root: the eval config becomes
   `load_config().model_copy(update=EVAL_PROFILE)` where `EVAL_PROFILE` is a
   small documented, code-owned dict. A **parity tripwire test** asserts the
   eval config differs from `load_config()` **only** in the documented
   `EVAL_PROFILE` fields — so the *next* production field to land can no
   longer silently change eval behavior (it either matches production, or it
   is a deliberate, documented profile entry). D3 and D4 become two entries
   in that profile rather than latent defaults.

2. **Shared `build_workflow()` factory.** The eval runner (runner.py:266) and
   production harness (harness.py:5604) each hand-wire a `TaskWorkflow(...)`.
   They drift. Extract one factory both call, so a new **mandatory** workflow
   dependency is acquired by evals automatically (or breaks both call sites
   at once — the tripwire).

3. **Contract-agnostic scoring off persisted artifacts.** Scoring reads
   **persisted verdict/plan/diff artifacts**, not agent transcripts — so it
   tracks whichever reviewer/judge output contract is live (the
   mcp-verdict-servers PRD is moving those from `--json-schema`/substring to
   MCP verdict tools; see §Cross-PRD).

On top of the restored framework: a **near-HEAD task-set re-cut** (real
completed tasks give ground truth for free), **new architect eval coverage**,
**adversarial fixtures**, an **OFAT→matrix→confirm** methodology with real
statistics, and **cloud-API candidate bundles**.

## Resolved design decisions

Decisions marked **(Leo)** were made explicitly in the brief's originating
session.

1. **(Leo) Keep an offline *fixed* eval** — a fixed target multiple
   candidates compare against — refreshed with a larger, near-HEAD task set.
   **Cloud APIs only** this round (no pods, no Docker, no vLLM). Retire the
   `hardware_time_seconds` GPU-imputation machinery.
2. **D5 root fix = `model_copy`-derived profile + parity tripwire**, not a
   longer constructor. The profile is the single source of truth for how eval
   differs from production; the tripwire makes silent divergence a test
   failure. `EVAL_PROFILE = {rebase_before_verify: False,
   inter_iteration_rebase: False, unblock_auto.enabled: False,
   auto_eval_enabled: False, simple_task_enabled: False}` (see §Contract for
   the exact leaf-set + the D8 memory-endpoint override).
3. **Shared `build_workflow()` factory** extracted from harness.py + runner.py
   wiring; both construction sites route through it; a tripwire guards
   single-point ownership.
4. **D1 fixed by threading the committed base commit**, not by patching each
   hand-joined path — `task['pre_task_commit']` (or `TaskArtifacts`) is the
   authoritative base; the metadata.json read is dropped.
5. **(Leo) Task-set re-cut from real completed tasks of the last ~6 weeks**,
   both repos, **stratified by repo (DF/reify) × kind (bugfix/feature/
   refactor) × path (simple/full)**. Target **~10–14 fixtures** (up from 5).
   Real tasks give ground truth for free: the **actual landed diff** becomes
   the judge's `reference` contender; the **repo verify gates at that SHA**
   are the deterministic gate; the **frozen plan** seeds the implementer
   eval. Keep the pinned-branch convention (`evals/<task_id>`); after the
   D5/D3 fix, fixtures **may** be main ancestors.
6. **2–3 adversarial fixtures** — a plan with one wrong step; a diff with a
   planted regression a reviewer must catch; a misleading verify failure —
   each scored with a **separate recovery-behavior rubric**. Rationale: frozen
   perfect inputs reward obedient instruction-followers; production rewards
   models that *notice and repair*.
7. **NEW: architect eval coverage** — **live** planning against fixtures (the
   architect *is* invoked), judged against the real landed plan/diff plus a
   plan-quality rubric. Built in the **same phase** as the implementer
   refresh, not deferred — it is the #1 cost line.
8. **Generalize the frozen-corpus pattern per role**: plans → implementer
   eval (exists), diffs → reviewer eval (reviewer_trial corpus exists),
   verify-failures → debugger eval (the D6 adversarial fixture seeds this).
   **(Leo)** Freeze upstream-role outputs when evaluating a downstream role
   (noise isolation + token savings).
9. **(Leo, design ratified) Methodology = OFAT screen → targeted matrix →
   end-to-end confirm.** OFAT: each candidate in **one** role, all other roles
   pinned to incumbents (attribution + cheap elimination). Then a small
   **architect × implementer matrix** of survivors **including same-family
   diagonals** (the plan-style/implementer coupling hypothesis). Then **one
   end-to-end confirmation batch** of the winner before any production config
   change. With frozen plans, implementer eval is architect-decoupled by
   construction; the coupling question exists only in the end-to-end runs.
10. **Statistics done properly.** ≥ 3 trials per cell; variance/CIs in the
    report. **Fix aggregation** — the April all-tasks-intersection rule
    collapsed the leaderboard to 2 entries and Elo noise was a self-admitted
    32%; retire/replace it.
11. **Composite adds cost + latency**, keeps the `tests_pass` **hard gate**,
    and adds a **per-config price table** for cloud endpoints (CLI-reported
    `cost_usd` is wrong for proxied endpoints — compute cost from the price
    table, not the CLI).
12. **Contract-agnostic scoring** — read persisted artifacts, not
    transcripts, so eval scoring tracks whichever reviewer/judge output
    contract mcp-verdict-servers has made live.
13. **Reviewer stays Opus this round** — only ~11–13% of spend and the Apr-8
    trial picked 1×Opus on both quality and cost. A reviewer_trial refresh
    (Sonnet 5 + one cross-family candidate) is **in scope but low priority**.
14. **(Leo) Rust caution** — do **not** cost-optimize the reify implementer
    yet; Claude leads the only Rust-bearing public benchmarks and open-model
    Rust evidence is thin. Codex+GPT-5.6 Sol is evaluated for the Rust
    implementer, but no production change on thin evidence.

## Pre-conditions for activating

- **Phase 1–3 have no external pre-conditions** — all substrate is on main
  today (G3 verified this session): the eval framework + CLI
  (`orchestrator eval --matrix`, cli.py:592), the real `TaskWorkflow` reuse
  (runner.py:266) and its eval-mode seams (`initial_plan` skips PLAN,
  `_worktree_external` skips MERGE, the eval branch in `_wait_for_resolution`),
  `build_eval_orch_config`/`OrchestratorConfig` (`model_copy` is standard
  pydantic-v2), `meta_root_for` (artifacts.py:180), `pre_task_commit` on the
  task record, `CostStore`/`runs.db` economics. **No novel syntax, schema,
  endpoint, or flag.**
- **Phase 4 (candidate bundles) gates on the harness-reconnect-pi PRD**:
  - Claude-format-endpoint candidates (MiniMax M2.5, GLM-5.2, DeepSeek V4,
    Kimi via official Anthropic-format endpoints behind Claude Code) need
    **only** that PRD's **per-role `env_overrides` forwarding**
    (`ANTHROPIC_BASE_URL`/`AUTH_TOKEN` per role).
  - **codex/pi bundles** (Codex CLI + GPT-5.6 Sol; pi+model incl. a
    pi+Sonnet harness-isolating control) gate on that PRD's codex hardening
    and pi backend + spike.

## Cross-PRD relationship

Two sibling PRDs are being authored concurrently in this same checkout.

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `~/.claude/spawn-briefs/2026-07-12-harness-reconnect-pi.md` (PRD in flight) | **consumes** | Per-role `env_overrides` forwarding (`ANTHROPIC_BASE_URL`/`AUTH_TOKEN`); codex backend hardening; pi backend + spike; **CostStore judge/steward/triage telemetry** | **harness-reconnect-pi** | queued (this PRD's Phase-4 tasks `depends_on` its env-forwarding / codex / pi tasks) |
| `~/.claude/spawn-briefs/2026-07-12-mcp-verdict-servers.md` (PRD in flight) | **consumes** | Reviewer/judge/triage/merger output contract (moving `--json-schema`/substring → MCP verdict tools) | **mcp-verdict-servers** (owns the contract) | this PRD's scoring is **contract-agnostic** (reads persisted artifacts) so it tracks whichever contract is live — no integration task owed either way |

**Seam-ownership resolution (G4):** harness-reconnect-pi **owns** all backend
env-forwarding, codex, pi, and telemetry substrate; this eval PRD **consumes**
it via normal `dark_factory` task deps (bare-integer intra-project deps — both
PRDs decompose into the same `dark_factory` project). mcp-verdict-servers
**owns** the output contract; this PRD deliberately **does not** couple to a
specific contract (decision 12), so there is **no reciprocal ambiguity** and
no integration task owed in either direction. The eval composite consumes the
judge/steward/triage CostStore telemetry harness-reconnect-pi adds, but treats
its **absence** as a documented gap (those roles are simply uncosted until
that PRD lands) rather than a hard dependency — so Phase 3 does not gate on it.

## Contract (H)

### C1 — Eval config profile

```
EVAL_PROFILE = {                      # the ONLY documented divergences from load_config()
  "rebase_before_verify":       False,   # D3 — no mid-eval rebase onto live main
  "inter_iteration_rebase":     False,   # D3 — same
  "unblock_auto.enabled":       False,   # D4 — no unmetered $5 dry-run per block
  "auto_eval_enabled":          False,   # eval never re-triggers itself
  "simple_task_enabled":        False,   # fixtures route through the full path deterministically
  # D8: memory endpoint overridden to a null/recording sink (see C1.mem)
}
build_eval_orch_config(base=None, **overrides) -> OrchestratorConfig:
    base = base or load_config()
    return base.model_copy(update={**EVAL_PROFILE_resolved, **overrides})
```

- **Invariant P1 (parity).** For the resolved config `c` and a fresh
  `load_config()` `L`: `{path : (L[path], c[path]) for path where L[path] !=
  c[path]}` **equals** the documented `EVAL_PROFILE` leaf-set (plus the D8
  memory endpoint). A production field that changes eval behavior without a
  profile entry **fails this test**. This is the D5 root-cause guard.
- **Note (tactical, → Open questions):** `model_copy(update=…)` does **not**
  re-run validators in pydantic v2. `EVAL_PROFILE` values are simple
  bools/leaves; P1 + the existing `load_config()` validation of `base` cover
  correctness. If a future profile entry needs cross-field validation, wrap
  the copy in a re-validation (the config-hot-reload PRD's hybrid-revalidate
  pattern is the precedent).

### C2 — `build_workflow()` factory

```
build_workflow(*, task, config, mcp_session, cost_store, ...deps) -> TaskWorkflow
```

- Single construction site for `TaskWorkflow`. Both **harness.py:5604**
  (production) and **runner.py:266** (eval) call it.
- **Invariant P2 (single-point).** A new **mandatory** `TaskWorkflow`
  constructor parameter is added in exactly one place (the factory) and is
  therefore acquired by eval runs automatically; a tripwire test asserts
  there is no direct `TaskWorkflow(` construction outside the factory
  (grep-guard), so the two sites cannot silently diverge again.

### C3 — Diff-threading (`get_diff`)

```
get_diff(worktree_path, base_commit) -> str   # base_commit := task['pre_task_commit']
```

- **Invariant P3.** For a fixture whose landed change is **committed** on its
  `evals/<id>` branch, `get_diff` returns the **full committed diff** against
  `base_commit`, never the uncommitted-only fallback; `judge`/`compare` grade
  that non-empty diff; `--worktree` resume reads the same. No metadata.json
  read.

### C4 — Composite & report schema

```
per-config report row := {
  config_id, role_under_test,          # OFAT: which single role varies
  composite, tests_pass (HARD GATE),   # composite gated to 0 if tests fail
  quality, cost_usd, latency_secs,     # cost from the price table, NOT CLI cost_usd
  trials: [...], ci95: {composite, cost, latency},   # ≥3 trials → variance/CIs
  judge: {...}, recovery_score | null, # recovery_score only for adversarial fixtures
}
+ price_table: {config_id: {role: {input_per_mtok, output_per_mtok}}}
```

- **Invariant P4 (contract-agnostic scoring).** Quality/judge/recovery scores
  are computed from **persisted artifacts** (verdict files, plan files, diff
  files) — never by parsing agent transcripts — so scoring is invariant to
  whether the reviewer/judge emitted via `--json-schema` or an MCP verdict
  tool (tracks mcp-verdict-servers).
- **Invariant P5 (cost honesty).** `cost_usd` is derived from `price_table`,
  not the CLI's `cost_usd` (which is wrong for proxied endpoints). The report
  states, per config, which cost source was used.

### C5 — Candidate bundle

```
CandidateBundle := { id, harness: "claude-code"|"codex"|"pi", model,
                     per_role_env: {role: {ANTHROPIC_BASE_URL, ANTHROPIC_AUTH_TOKEN}} }
```

- A candidate is a **(harness, model) bundle** — the deployable config, not a
  bare model. `per_role_env` is forwarded by harness-reconnect-pi's
  env-forwarding mechanism (Phase-4 dependency).

## Boundary-test sketch (H) — the Phase-1 integration-gate signal

Two-way tests facing **both** the eval side and the production side of each
seam. This table is task **ι**'s observable signal (§Decomposition).

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | **Parity tripwire (P1, D5)** | live `load_config()` + eval profile | field-diff set **equals** `EVAL_PROFILE` (+D8 endpoint) exactly; test **fails** if a new production field diverges or an eval field is undocumented |
| B2 | **Factory single-point (P2)** | add a synthetic required arg to `TaskWorkflow` | **both** harness + eval construction fail to build (proving single ownership); grep-guard finds no `TaskWorkflow(` outside the factory |
| B3 | **Dispatch-literal tripwire (D2)** | scheduler.py `dispatch_tool` literals enumerated | every literal has a `_StubMcpSession` branch; test **fails** when a new literal (e.g. a future tool) lacks one |
| B4 | **Non-empty diff graded end-to-end (P3, D1)** | one fixture with a committed landed diff | eval run's judge + compare grade the **full committed diff** (non-empty); a run that would have graded `{}` **fails** |
| B5 | **Memory isolation (D8)** | green eval run against a recording endpoint | production `dark_factory` store write-count delta **= 0**; the recording endpoint captured the intended writes |
| B6 | **No mid-eval rebase (D3)** | a **main-ancestor** fixture, live main advanced | the eval worktree is **not** rebased onto live main during the run (`rebase_before_verify=False` observed) |
| B7 | **No unblock_auto spawn (D4)** | a fixture that blocks | **zero** dry-run investigation subprocesses spawned |
| B8 | **Contract-agnostic scoring (P4, cross-PRD)** | persisted verdict artifact in **both** the legacy `--json-schema` shape and the MCP-verdict-tool shape | scoring yields the same quality score from either artifact shape — no transcript parsing |
| B9 | **No `set_task_claimant` warning spam (D2)** | 90 s+ eval run (≥1 heartbeat) | **zero** "unknown tool `set_task_claimant`" warnings in the run log |

## Decomposition plan

Greek labels; task IDs assigned at decompose. **Phase 1** restores
correctness; **Phase 2** re-cuts the corpus + adds architect/adversarial
coverage; **Phase 3** is methodology + reporting; **Phase 4** is the candidate
bundles (cross-PRD-gated). Phase 1 tasks mostly touch `evals/runner.py`, so
they are **chained** (α→β→γ→δ→ε) — under the narrow module lock they cannot
run concurrently anyway, and chaining makes ordering deterministic and avoids
rebase churn.

**Phase 1 — restore correctness (self-contained; no external deps)**

- **α — `build_workflow()` factory** *(intermediate → unlocks the eval-runner
  path; consumers: production harness dispatch + eval runner)*. Extract the
  shared `TaskWorkflow` construction from harness.py:5604 + runner.py:266 into
  one factory both call (C2). **Signal (intermediate):** production and eval
  both build via the factory; a grep-guard tripwire (B2) proves no other
  `TaskWorkflow(` site exists. Modules: orchestrator/harness.py,
  orchestrator/evals/runner.py.
- **β — eval profile via `model_copy` + parity tripwire (D5, D3, D4)** *(leaf;
  parity test is its signal)*. Replace the constructor build (runner.py:154)
  with `base_config.model_copy(update=EVAL_PROFILE)` (C1); the parity tripwire
  (B1, P1) enumerates the field-diff and asserts it equals `EVAL_PROFILE`.
  D3/D4 become profile entries. **Signal:** B1 green; the enumerated diff
  equals the documented set. Depends α. Modules: orchestrator/config.py (or a
  new `evals/profile.py`), orchestrator/evals/runner.py.
- **γ — thread committed base into `get_diff` (D1, the worst bug)** *(leaf)*.
  Thread `task['pre_task_commit']` into `get_diff` (C3); fix all four callers
  (snapshots, runner, compare, rereview); drop the metadata.json read.
  **Signal:** B4 — an eval run over a committed-diff fixture grades the full
  committed diff, not `{}`; a RED test asserts `get_diff` on a
  `.task-meta`-relocated worktree returns the committed diff. Depends β.
  Modules: orchestrator/evals/{snapshots,runner,compare,rereview}.py.
- **δ — complete `_StubMcpSession` + dispatch-literal tripwire (D2)** *(leaf)*.
  Add `set_task_claimant` + `add_dependency` branches; add a tripwire test
  (B3) over scheduler.py `dispatch_tool(...)` literals. **Signal:** B9 (no
  heartbeat warning spam) + B3 (tripwire fails on a new literal). Depends γ.
  Modules: orchestrator/evals/runner.py, orchestrator/tests.
- **ε — isolate eval memory writes (D8)** *(leaf)*. Point the eval config's
  fused-memory endpoint at a null/recording sink (C1.mem). **Signal:** B5 —
  a green eval run leaves the production `dark_factory` store write-count
  delta at 0; the recording sink captured the intended writes. Depends δ.
  Modules: orchestrator/evals/runner.py.
- **ι — Phase-1 integration gate (H boundary-test suite B1–B9)** *(leaf —
  C-as-integration-gate; the two-way eval↔production seam test)*. Implements
  the boundary-test sketch end-to-end through a real eval run. **Signal:** the
  B1–B9 suite passes against one refreshed fixture. Depends α, β, γ, δ, ε.
  Modules: orchestrator/tests, orchestrator/evals.

**Phase 2 — task-set re-cut + new coverage**

- **ζ — task-set sampler + re-cut ~10–14 fixtures** *(leaf/tooling → substrate
  for Phase 3)*. A tool that samples real completed tasks (last ~6 weeks, both
  repos), **stratified by repo × kind × path**, pins each on `evals/<task_id>`,
  and captures per fixture: the **landed diff** (judge `reference`), the
  **repo verify gates at that SHA** (deterministic gate), the **frozen plan**
  (implementer eval). Word reify fixture refs **generically** (DF path-scope
  guard rejects `crates/…` example tokens). **Signal:** the eval `tasks/` dir
  holds ~10–14 fixtures each with a pinned branch + reference diff + recorded
  verify outcome; a listing command prints the stratification counts. Depends
  ι (fixtures validated against the *fixed* framework).
- **η — 2–3 adversarial fixtures + recovery rubric** *(leaf)*. A plan with one
  wrong step; a diff with a planted regression; a misleading verify failure
  (the latter also seeds the debugger-eval corpus, decision 8). Each tagged
  with its adversarial type + a **separate recovery-behavior rubric**.
  **Signal:** 3 adversarial fixtures present; an eval run over them emits a
  distinct `recovery_score` column (C4). Depends ζ.
- **θ — architect eval harness (NEW live-planning coverage)** *(leaf)*. The
  architect **is invoked** (live planning) against fixtures, judged vs the
  real landed plan/diff + a plan-quality rubric. Freeze downstream roles when
  scoring the architect (decision 8). **Signal:** `orchestrator eval` with an
  architect-role candidate emits a per-fixture plan-quality score judged
  against the real landed plan. Depends ζ.
- **κ — reviewer_trial refresh (low priority)** *(leaf)*. Re-run the
  reviewer_trial corpus with Sonnet 5 + one cross-family candidate; reviewer
  **stays Opus** absent a clear win (decision 13). **Signal:** a refreshed
  reviewer_trial report ranks Opus vs the candidates on quality **and** cost.
  Depends ι. **Priority: low.**

**Phase 3 — methodology + reporting**

- **λ — composite + price table + statistics aggregation** *(leaf)*. Add cost
  + latency to the composite; keep the `tests_pass` hard gate; add the
  per-config **price table** (compute cost from it, not CLI `cost_usd`, P5);
  ≥ 3 trials/cell with variance/CIs (decision 10); **fix aggregation** (retire
  the all-tasks-intersection collapse + the noisy Elo); retire
  `hardware_time_seconds`. **Signal:** the report shows per-config composite
  with cost, latency, CI95, and a price table; a multi-trial run reports
  cross-trial variance. Depends ι. Modules:
  orchestrator/evals/{metrics,report,compare,elo}.py.
- **μ — OFAT→matrix→confirm driver + contract-agnostic scoring** *(leaf; the
  G2 top signal)*. The methodology runner: OFAT screen (each candidate in one
  role, others pinned), then architect×implementer matrix of survivors incl.
  same-family diagonals, then one end-to-end confirmation batch; freeze
  upstream-role outputs when scoring a downstream role (decision 9);
  contract-agnostic scoring off persisted artifacts (P4, B8). **Signal (G2
  top):** `orchestrator eval --matrix` (or successor) runs the refreshed task
  set against **≥ 2 candidate configs end-to-end via cloud endpoints** and
  emits the report with per-config composite, judge results, **cost, latency,
  CIs** — **non-empty diffs graded**. Depends λ, θ, η, ζ (and ι transitively).
  Modules: orchestrator/evals/{runner,compare,configs,report}.py, cli.py.

**Phase 4 — candidate bundles (cross-PRD-gated)**

- **ν — Claude-format-endpoint candidate bundles** *(leaf)*. Bundle configs
  for incumbents (Opus 4.8 / Sonnet) + MiniMax M2.5, GLM-5.2, DeepSeek V4,
  Kimi via official Anthropic-format endpoints behind Claude Code (per-role
  `ANTHROPIC_BASE_URL`/`AUTH_TOKEN`, C5). **Signal:** a bundle config lists
  each (harness, model) with its per-role endpoint; an eval run dispatches to
  a non-incumbent Claude-format endpoint and records its price-table cost.
  Depends μ + **harness-reconnect-pi's per-role env-forwarding task**
  (cross-PRD, bare-int dep once that task exists).
- **ξ — codex/pi candidate bundles** *(leaf)*. Codex CLI + GPT-5.6 Sol for the
  **Rust implementer** role; pi+model bundles once the pi spike lands, incl. a
  **pi+Sonnet harness-isolating control** (same model, different harness).
  Rust caution (decision 14): evaluate, do not cost-optimize reify implementer
  on thin evidence. **Signal:** codex/pi bundle configs present; an eval run
  dispatches the Rust implementer via codex and records a result; the
  pi+Sonnet control isolates harness effect. Depends μ + **harness-reconnect-pi's
  codex-hardening + pi-backend tasks** (cross-PRD).

## Out of scope

- **RunPod pods / Docker images / vLLM / any self-hosted GPU** this round —
  cloud APIs only; the `hardware_time_seconds` GPU-imputation machinery is
  retired, not maintained.
- **Any production per-role config change** — this PRD *measures*; adopting a
  winner is a separate deterministic-deploy task after the confirmation batch.
- **CostStore judge/steward/triage telemetry** — owned by harness-reconnect-pi;
  this PRD consumes it if present and documents its absence otherwise.
- **The reviewer/judge output *contract*** — owned by mcp-verdict-servers;
  this PRD stays contract-agnostic and owns no migration.
- **A full standalone debugger-eval harness** — decision 8's verify-failure
  corpus is seeded by η's adversarial fixture; a dedicated debugger harness is
  a follow-up (debugger is 4–7% of spend).
- **Cost-optimizing the reify (Rust) implementer** — evaluated only; no change
  on thin open-model Rust evidence (decision 14).

## Open questions (tactical — surfaced, not blocking)

1. **`model_copy` validation.** pydantic v2 `model_copy(update=)` skips
   validators. **Suggested resolution:** rely on `base=load_config()` being
   pre-validated + the P1 parity test; wrap in a re-validate only if a future
   profile entry needs cross-field checks (config-hot-reload's
   hybrid-revalidate precedent). Decide in **β**.
2. **Null vs recording memory endpoint (D8).** A pure `/dev/null` sink vs a
   recording sink that captures intended writes for assertion. **Suggested:**
   recording sink — B5 needs to assert *what* would have been written. Decide
   in **ε**.
3. **Aggregation replacement for the all-tasks-intersection rule.**
   Per-fixture normalized rank-mean vs trimmed-mean vs a Bradley-Terry fit
   replacing Elo. **Suggested:** per-fixture normalized score → mean with CI;
   drop Elo unless pairwise judge data is dense. Decide in **λ**.
4. **Successor CLI surface.** Keep `orchestrator eval --matrix` vs a new
   `eval ofat` / `eval matrix` / `eval confirm` triplet. **Suggested:**
   sub-modes under `eval` mirroring the three methodology stages; keep
   `--matrix` as an alias. Decide in **μ**.
5. **Fixture count within the 10–14 band + exact strata cell sizes.** Decide
   at cut time in **ζ** from what the last-6-weeks corpus actually offers per
   cell (some cells may be thin — e.g. reify refactors).
