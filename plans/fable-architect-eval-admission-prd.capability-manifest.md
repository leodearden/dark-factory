# Capability manifest — fable-architect-eval-admission-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized), authored at decompose
2026-07-20. Line refs are as-of main `935e185d11` and drift; the YAML sidecar
twin carries the delivered_checks. Task π lives in eval-revival's lane — its
bindings are in `plans/eval-framework-revival-prd.capability-manifest.yaml`
(sidecar paths are strictly PRD-derived).

## τ1 — Hard-subset campaign gate (deterministic pure gate)
- `eval-ofat-tasks-dir` → **PASS (wired)** — `eval-ofat` CLI with `--tasks-dir`
  / `--trials` (`orchestrator/src/orchestrator/cli.py:1297-1310`).
- `eval-confirm-arch-flag` → **PASS (wired)** — `eval-confirm --arch/--impl`
  (`cli.py:1360+`), ≥3-trial default.
- `hard-fixtures-on-disk` → **PASS** — all six present in
  `orchestrator/src/orchestrator/evals/tasks/`: `df_task_2284_adv_regression`,
  `df_task_2339_adv_verify`, `df_task_2430_adv_plan`, `reify_task_12` (high),
  `reify_task_27` (high), `df_task_18`.
- `eval-bootstrap-smoke-gate` → **PASS (wired)** —
  `scripts/eval_bootstrap_smoke.sh` on main (task 2847 done, found_on_main
  `8c8eeef745`).
- `architect-fable-candidate` → **producer: task π (upstream dep)** —
  eval-revival lane.

## τ2 — Committed decision record
- `campaign-artifacts-readable` → **producer: τ1 (upstream dep)** — result
  JSONs + stdout composite tables under `evals/results/` per the τ1 run
  instructions.
- `plans-record-convention` → **PASS** — decision-record precedent (ι's signal
  shape, adaptive-routing PRD §ι).

## τ3 — Admission ratification gate (leaf; deterministic pure gate)
- `decision-record` → **producer: τ2 (upstream dep)**.
- `scope-safe-failover` → **producer: usage-gate-model-scoped-caps ε (upstream
  dep, wired cross-PRD in-project)**.
- `born-at-l2-pure-gate` → **PASS (wired)** — deterministic pure-gate preset
  (`always_escalates`, no `before_done`) is live runner behavior (CLAUDE.md
  "Deterministic task kind"; 2848 precedent).

No FAIL bindings; batch clear to queue.
