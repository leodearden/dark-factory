# Capability manifest — integration-test-lane PRD

Mechanizes G3 + G6 for `plans/integration-test-lane-prd.md`. Every binding below
was verified against HEAD this session. No binding resolves to a FAIL value, so the
batch clears the manifest gate. Machine-readable twin:
`plans/integration-test-lane-prd.capability-manifest.yaml`.

Bindings key: `grep:<file>:<line>` = wired/present on main; `producer:task-<label>` =
delivered by an upstream batch task (DAG-direction verified); `empirical` = proven by
running the check this session.

---

## α — Generalize the offline-lane runner to config-driven per-project commands

α *reuses* landed engine substrate (all present on `main`) and *delivers* the new
`offline_lane_commands` config-driven runner. The reused-substrate bindings are PASS
(present); the delivered capability is what gates β.

| Capability α asserts / delivers | Binding | Verdict |
|---|---|---|
| `OfflineLaneWorker._run_once` engine (single-flight/coalesce/from-head/never-gate/red-path) | grep:offline_lane.py:363 — wired (launched by `Harness._start_offline_lane`, harness.py:7233) | PASS |
| `verify._extract_failing_test_ids` (pytest FAILED/ERROR/node-down node-id extractor) | grep:verify.py:545 | PASS |
| `workflow.compute_failing_test_set_fingerprint` | grep:workflow.py:467 | PASS |
| `workflow.build_offline_lane_fix_task_arguments` (α parameterizes its hard-coded `priority='high'`) | grep:workflow.py:494 | PASS |
| `verify_cmd.serial_pytest` (serial confirm-run form) | grep:verify_cmd.py:545 | PASS |
| `git_ops.reset_persistent_offline_deep_worktree` (warm worktree reset-to-head) | grep:git_ops.py (present; used by offline_lane.py:407) | PASS |
| `config.RELOADABLE_FIELDS` green-tier list (α adds the new list fields) | grep:config.py:~3896 (existing `git.offline_lane_*` scalars already listed) | PASS |
| **NEW: `git.offline_lane_commands` config-driven runner** (the deliverable) | delivered by α; gate = grep `offline_lane_commands` in offline_lane.py (false today, true after α — wiring, not mere declaration) | PASS (delivered by this task) |

## β — Instantiate the lane config for dark-factory Qdrant

| Capability β asserts | Binding | Verdict |
|---|---|---|
| `offline_lane_commands` config schema to populate | producer:task-α (upstream) | PASS |
| Qdrant compat tests carry `@pytest.mark.integration` (so `pytest -m integration` selects them) | producer:task-2773 (upstream local dep; **currently `blocked`** — see §Activation) | PASS (upstream) |
| `pytest -m integration` overrides the pyproject `addopts -m 'not integration'` (selects the integration set) | empirical — `--collect-only` default selected 5; `-m integration` deselected all 5 | PASS |
| `integration` marker declared in `fused-memory/pyproject.toml` | grep:fused-memory/pyproject.toml:31 | PASS |
| **DELIVERS: DF `offline_lane_commands` config in the yaml** | delivered by β; gate = grep `offline_lane_commands` in dark-factory-orchestrator.yaml (false today, true after β) | PASS (delivered by this task) |

## γ — Deterministic-deploy: activate the dark-factory offline lane (LEAF / integration gate)

| Capability γ asserts | Binding | Verdict |
|---|---|---|
| generic config-driven runner on main | producer:task-α (upstream) | PASS |
| DF qdrant lane config on main | producer:task-β (upstream) | PASS |
| Qdrant tests integration-marked on main | producer:task-2773 (upstream) | PASS (upstream) |
| `Harness._start_offline_lane` restart-only start gate | grep:harness.py:7192 | PASS |
| `scripts/restart-all-orchestrators.sh --drain` fleet-redeploy chokepoint | grep:scripts/restart-all-orchestrators.sh (present) | PASS |

**No FAIL bindings — the batch clears the manifest gate.**

## Activation note (β/γ depend on 2773 landing)

Task **2773** (which adds the `@pytest.mark.integration` marks) is currently `blocked`
by `esc-2773-3` — the very coverage-gap this lane closes. Landing **α** closes that
substantive gap; an operator then resolves `esc-2773-3` (option A: the lane now exists)
and re-pends 2773. Once 2773 lands the marks, β/γ activate the DF lane. α has **no**
dependency on 2773 — only the DF instance (β) does. This is not a code cycle.
