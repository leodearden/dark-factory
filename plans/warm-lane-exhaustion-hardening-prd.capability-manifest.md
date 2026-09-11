# Capability manifest — warm-lane-exhaustion-hardening-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized). Evidence verified on
main @ 1d96aacb75, 2026-07-23. Machine-readable twin:
`warm-lane-exhaustion-hardening-prd.capability-manifest.yaml` (DF labels only;
θ is cross-project and excluded from the sidecar/dispatch gate).

Contract literals prescribed by this batch (greppable, fixed at authoring):
census field `n_pinned_non_dispatched` (α); drift escalation root cause
`lane_record_drift` (γ); digest divergence anchor `lane_record_divergence` (δ);
structural-exhaustion escalation root cause
`warm_lane_pool_structurally_exhausted` (ε); unknown-key escalation root cause
`config_unknown_keys` (ζ); MCP tool name `get_warm_lane_status` (β); test name
`test_warm_lane_reclaim_on_exhaustion_default_true` (η).

## α — Typed pool census on the exhaustion path (intermediate → β, ε)

- `dispatched-predicate-seam` → wired: `orchestrator/src/orchestrator/git_ops.py:1447`
  (`warm_lane_dispatched_predicate` declare-on-callee attr; installed harness-side,
  1933 pattern). Predicate-unwired sites degrade to `n_unknown_dispatch` by design. PASS
- `exhaustion-chokepoint` → wired: `git_ops.py:4577-4583` (acquire_for None →
  EXHAUSTED) + `git_ops.py:2452-2455` (WarmLanePoolExhausted raise). PASS
- `assignment-snapshot` → wired: `warm_lane_pool.py:252` (`assignments_snapshot`),
  `:243` (`state`). PASS

## β — MCP tool `get_warm_lane_status` (leaf; deps α)

- `pool-census-type` → producer: task-α (upstream). PASS
- `escalation-server-tool-registration` → wired:
  `escalation/src/escalation/server.py:1604` (`get_task_runtime_state` — the
  in-process orchestrator-state MCP tool pattern this imitates). PASS
- `harness-state-handle` → wired: escalation server runs in-process with the
  orchestrator (same pattern as above tool's task-runtime access). PASS

## γ — Single-writer assignment store, loud drift (intermediate → δ)

- `durable-assignment-writer` → wired: `warm_lane_pool.py:326`
  (`_note_assigned_durable` → `LaneLifecycle.note_assigned`); release-side durable
  transition exists at `git_ops.py:5190-5200` (to be consolidated behind the pool
  per the PRD contract I2). PASS
- `record-store` → wired: `lane_lifecycle.py:42` (`.lane-state/<lane>.json`,
  LANE_STATE_DIRNAME). PASS
- `l2-dedup` → wired: `escalation/src/escalation/queue.py:821`
  (`find_pending_l2_by_root_cause`). PASS

## δ — Digest map↔record divergence cross-check (leaf; deps γ)

- `record-reader-seam` → wired: `harness.py` `_assigned_durable_records_with_statuses`
  (landed with task 2891, merge 642bfc9b6c). PASS
- `digest-inputs-pattern` → wired: `digest.py:621` (`stale_lane_census` — the
  2891 exemplar this extends). PASS
- `single-writer-premise` → producer: task-γ (upstream; divergence is a defect,
  not ambient, only once γ lands). PASS

## ε — EXHAUSTED cap flip + structural-exhaustion L2 (leaf; deps α)

- `disposition-row` → wired: `workflow_types.py:306-313` (WarmLanePoolExhausted
  BlockDisposition, declared-once) consumed at `workflow.py:2763-2779`. PASS
- `requeue-counting-seam` → wired: `scheduler.py:7010-7050` (genuine vs transient
  requeue buckets — precedent for classification-driven counting). PASS
- `pool-census` → producer: task-α (upstream). PASS
- `callback-install-pattern` → wired: `harness.py:1237`
  (`_on_pool_storage_absent` declare-on-callee/install-in-harness exemplar). PASS
- `l2-dedup` → wired: `queue.py:821`. PASS
- Negative assertion "requeue cap NOT incremented" → rejection-mechanism-backed:
  counting is driven solely by the declared-once disposition row (no second
  counting site); asserted by the ε test against the scheduler's
  `_requeue_counts`. kind: manual in sidecar. PASS

## ζ — Unknown-config-key census: L2 + reload + --check-config (leaf)

- `raw-yaml-source` → wired: `config.py:3970-3973` (`YamlSettingsSource` in
  `settings_customise_sources`). PASS
- `model-schema-walk` → pydantic v2 `model_fields` (library capability; BaseModel
  submodels throughout config.py). PASS
- `born-at-l2-filer` → wired: `harness.py:10552` (`_file_reblock_guard_l2`
  pattern) + `queue.py:821` dedup. PASS
- `reload-seam` → wired: `config.py:4064` (`RELOADABLE_FIELDS`) + apply machinery
  `config.py:4243,4305`; escalation `reload_config` tool. PASS
- `cli-entry` → wired: `cli.py:200-202` (click group `main`). PASS

## η — Valve default flip (leaf)

- `valve-field` → wired: `config.py:1832` (`warm_lane_reclaim_on_exhaustion`,
  currently `default=False`). PASS
- `no-defaults-yaml-shadow` → verified absent: `defaults.yaml` does not list the
  knob — the Field default is the single source (mechanical `expect: absent`
  check in sidecar). PASS
- `steal-path-wired` → wired: `git_ops.py:4081` (`_try_reclaim_lane_for`) called
  from `:4581`; `warm_lane_pool.py:176` (`reclaim_victim`). PASS
- `reseed-integrity-guard` → wired: `RESEED_CONTAMINATED` fault-and-reacquire on
  the shared reseed tail (task 2854, merge 0c8137d560) — protects the steal
  path's fresh-reset route. PASS

## θ — Reify audit honest columns (leaf, reify project; deps dark_factory:β)

Cross-project: documented here, excluded from the DF sidecar/dispatch gate
(external deps gate on status via `get_external_statuses`, not delivered_checks).

- `mcp-census-surface` → producer: dark_factory task-β (upstream, cross-project). PASS
- `durable-records-on-host` → verified: reify `.worktrees/.lane-state/` exists and
  is populated (checked 2026-07-23). PASS
- `flock-probe` → wired: reify `scripts/warm-lane-audit.sh` `_probe_assigned`
  (~:212-246) — retained as the LIVE column. PASS
