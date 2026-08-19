# Capability manifest — deep-merge-ahead-prd

Mechanizes G3 (assumed-substrate verified) + G6 (premise validity) for the
deep merge-ahead batch (α–κ). One block per task; each capability the task's
signal asserts is bound to evidence. A binding resolving to a FAIL value
(`declared-only` / `test-only` / `producer-absent` / `producer-downstream` /
`producer-extent-short` / `bound≤floor` / `rejection-absent`) blocks the
batch. **All bindings PASS** — every novel capability is produced by a task
in this batch and is upstream of its consumer (DAG-direction correct), and
every assumed substrate was verified live on DF `main` this session.

Substrate verified on DF `main` (the orchestrator package is DF-owned;
reify runs it): `merge_spec_warm_lane_pool` + `_spec-` lane routing
(`merge_liveness.py:690+`), `advance_main` / frozen-prefix CAS
(`merge_queue.py`), `verify_cancel.py`, `reset_persistent_merge_worktree`
(`git_ops.py`), `PermitLedger` + `speculation_accounting_violations`
(`merge_queue.py:7182,7380`), `analyze_speculation_depth.py`,
`RELOADABLE_FIELDS` (`config.py`). Out-of-batch prereq **task 3003**
(typed contended-DEFER on the reset path) exists and is in-progress; γ/δ
carry a hard dep on it.

The three contract-fixed literals the PRD pins (decisions #6/#8, §Contract)
carry a `delivered_check` grep against `orchestrator/src/` — the
dispatch-time twin that withholds each producer's dependents until the
literal actually lands on `main`:

| Producer | Literal | Grep (expect present, `orchestrator/src/`) |
|---|---|---|
| α | `merge_deep.chain_cap` config knob | `merge_deep` |
| γ | `merge_verify.chain_items` event field | `chain_items` |
| δ | `merge_finalized.landed_via_chain` field | `landed_via_chain` |

All three are confirmed **absent** on `main` today (verified this session) —
they are what this batch builds; the checks flip to satisfied the tick each
lands.

---

## α — config: `merge_deep.chain_cap` knob + kill-switch default

- **`merge_deep.chain_cap` config field (default 0)** → producer:task-α (this
  task builds it in `config.py` + `defaults.yaml`). **PASS.**
  `delivered_check`: grep `merge_deep` present in `orchestrator/src/`.
- **knob ∈ RELOADABLE_FIELDS (green-tier) + cap=0 byte-identity** →
  producer:task-α; the kill-switch/hot-reload property is asserted by α's own
  byte-identity + reload tests. **PASS** (`manual` — a test property, not a
  grep; `speculation_probe.probe_fraction=0` is the live precedent).

## β — chain builder (build-on-dispatch)

- **`build_chain(queue_snapshot, head_merge_commit, cap, target_depth) ->
  ChainResult{links, tip, truncated_at}`** — sequential submission-order
  merges in one scratch worktree, truncate at first conflict, never emit
  per-item outcomes → producer:task-β. **PASS.** Signal is an integration
  test (queue fixture incl. a conflicting item yields the clean prefix,
  conflicted item untouched, exactly one worktree used).
- **substrate: `acquire_spec_lane` / `_spec-` pool** → capability→producer
  (wired) grep `merge_liveness.py:690+` on `main`. **PASS** (live since reify
  4941; the pool owns worktree lifecycle/locking — no bespoke scratch dir).

## γ — deep-tip verify dispatch + halving state

- **verify dispatch onto an arbitrary built (chain-tip) commit** — the one
  genuinely novel runtime capability (the probe never redirected dispatch) →
  producer:task-γ. **PASS.**
- **`merge_verify.chain_items` (1-indexed count on every merge verify)** →
  producer:task-γ. **PASS.** `delivered_check`: grep `chain_items` present in
  `orchestrator/src/`. Consumers δ (landing) and ε (report reader) depend on
  γ — DAG-direction correct.
- **halving/reset state machine** (`fail(d)→max(1,⌊d/2⌋)`, `pass→min(queue,
  cap)`; d=1 floor byte-identical to today's adjacent verify) →
  producer:task-γ. **PASS** (integration signal: dispatch depths across
  scripted pass/fail sequences match the policy).
- **substrate dep: task 3003 typed contended-DEFER** — producer:task-3003
  upstream (out-of-batch, exists, in-progress); γ wires the dep. **PASS.**

## δ — prefix landing on tip pass

- **`merge_finalized.landed_via_chain = k`** → producer:task-δ. **PASS.**
  `delivered_check`: grep `landed_via_chain` present in `orchestrator/src/`.
  Consumers ι/ε/κ depend on δ — DAG-direction correct.
- **in-order CAS landing of I0..Ik through the existing terminal finalize
  path; first CAS failure aborts the walk (decision #9)** — reuses
  `advance_main` / frozen-prefix CAS (substrate on `main`) → capability→
  producer (wired) grep `merge_queue.py` `advance_main`. **PASS.**
- **head-verify cancellation with clean verify-lease release** — reuses
  `verify_cancel.py` (substrate on `main`); inherits 3003's contended→DEFER
  classification → producer:task-δ + producer:task-3003 upstream. **PASS.**

## ι — two-way boundary/integration gate (B+H)

- **the §Boundary-test sketch, green in CI, facing both the worker and the
  CAS/ledger sides** → producer:task-ι; all legs produced upstream (β build,
  γ dispatch, δ landing). **PASS** (this task IS the integration gate — the
  C-as-integration-gate leaf for the foundation slice; its signal is the
  CI-green boundary matrix).
- **conservation audits (`speculation_accounting_violations`, PermitLedger
  identities) stay green under chain landing** — substrate on `main`
  (`merge_queue.py:7182,7380`); ι asserts they hold → capability→producer
  (wired). **PASS.**

## ε — telemetry + report

- **merge report renders depth histogram + items-per-verify + deep-fail rate,
  reader keys on `chain_items`** → producer:task-ε; reads δ/γ's structured
  fields (upstream). **PASS.** `manual` — signal is "report renders from live
  events; reader output matches git-derived truth on fixtures".
- **substrate: `analyze_speculation_depth.py` event-store reader** →
  capability→producer (wired) on `main`. **PASS.**

## ζ — reify canary enable at cap=6 (deterministic deploy)

- **target orchestrator reload disposition shows `merge_deep.chain_cap=6`
  live; first `chain_items>=2` verify observed** → producer:task-ζ via the
  committed `before_done` script `scripts/merge-deep-set-cap.sh` (exists +
  executable on `main` `fbaf4dc526`). All chain machinery (α–δ) + gate (ι) +
  telemetry (ε) produced upstream — DAG-direction correct. **PASS** (`manual`
  — an operator deploy signal, not a grep).

## η1 — 7-day canary predicate (deterministic, milestone delayed 604800s)

- **predicate reads the target runs.db and gates promotion by exit code:
  deep-fail-rate ≤ 0.35, items-per-verify ≥ 1.3, verify-p90 ≤ 5400s**
  (numeric bounds, G6 branch 1) → producer:task-η1 via
  `scripts/merge-deep-canary-predicate.sh` (exists + executable on `main`).
  **Achievability basis (not a guess):** the replay study passed 57/59 (≈2/59
  ≈ 0.034 ≪ 0.35 fail rate), 3–6-item chains land ≫1.3 items/verify, and
  projected wall at depth 16 ≈ 2.1 ks ≪ 5400 s. Thresholds marked
  **provisional** in the script header; task θ retunes. **PASS.**

## η2 — promote cap 6 → 32 (deterministic deploy)

- **target reload disposition shows `merge_deep.chain_cap=32`** →
  producer:task-η2 via the same `scripts/merge-deep-set-cap.sh` (single
  script, args differ — no lock-step duplication). Gated on η1's predicate
  passing. **PASS** (`manual`).

## θ — week-after aggressiveness assessment (normal, milestone delayed)

- **written assessment committed to `plans/`; follow-up tasks filed or
  explicitly declined** → producer:task-θ (leaf). **PASS** (`manual` — a
  committed-doc + filed-tasks signal, observed via git + the task tree).

## κ — docs correction pass (normal)

- **`skills/orchestrate/SKILL.md` + OPERATIONS.md name the `merge_deep.chain_cap`
  knob and the halving policy; analysis doc §11.5 marked implemented** →
  producer:task-κ (leaf). **PASS** (`manual`/greppable docs diff — the
  knob-name string appears in the docs).

---

## G7 (design invariants) — walked, no waivers

- **INV-1 `contracts-machine-checked`**: `chain_cap` is a schema field +
  RELOADABLE_FIELDS entry; the dispatch invariant and `chain_items` are code /
  structured fields — no prose contract. PASS.
- **INV-2 `structured-facts-at-failure`**: `chain_items` / `landed_via_chain`
  are truthful structured emitter-known fields that *replace* the broken
  `depth` log-label (decision #8) — an INV-2 improvement. PASS.
- **INV-3 `corroborate-before-acting`**: stale-CAS abort (decision #9)
  re-corroborates that `main` did not move externally before each CAS land;
  dispatch re-checks live deps. PASS.
- **INV-4 `storm-escape-required`**: halving is a *bounded* degradation
  (≤log₂(cap) rounds) that bottoms out at the d=1 floor — byte-identical to
  today's adjacent verify, whose existing `consecutive_merge_thrash`
  escalation fires for a genuinely bad item — and deep-fail rate is surfaced
  in the ε report and rate-gated weekly by the η1 predicate. No new *unbounded
  silent* suppressor. PASS (satisfied, not waived).
- **INV-5 `no-lockstep-duplication`**: `build_chain` is a single new builder
  (not a copy of the spec path); δ reuses the existing CAS/finalize path; ε
  reads the single `chain_items` source; ζ/η2 call the *same*
  `merge-deep-set-cap.sh`. PASS.
