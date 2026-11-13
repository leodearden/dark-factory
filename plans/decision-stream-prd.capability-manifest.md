# Capability manifest — decision stream (`plans/decision-stream-prd.md`)

Decompose session prd-df-3475198, 2026-11-04, against main `a26991efc9`. Machine-readable
twin: `plans/decision-stream-prd.capability-manifest.yaml`. Bindings cite symbols; every
existing symbol named in PRD §4 re-verified present on this tree.

## Resolved at decompose (no FAIL binding remains)

| leaf | asserted capability | first verdict | resolution |
|---|---|---|---|
| η (→ η1/η2) | "the state files change" on Leo's `live` answer | producer-absent (§5.1 names no `shadow → live` writer) | δ1 gains a `policy-state` verb; η1/η2 depend on δ1 |
| η | 3346 gates new classes only (decision 7) | producer-extent-short (one gate wired to 3346 holds the moved classes too) | split η1 (moved) / η2 (new, + 3346) |
| ε | `systemctl --user list-timers` shows the timer | producer-absent (agents cannot write `~/.config/systemd/user/`, `orchestrator/src/orchestrator/agents/write_set.py::compute_write_set`) | ε ships an idempotent installer; κ (operational gate) runs it |
| γ | "a dry rotation on a fixture queue produces the expected calls" | producer-absent (no test harness runs an LLM over a skill) | rewritten (INV-10): marker-delimited class blocks mirrored against `POLICY_TABLE`, their call sequences run against the fixture fake server; behaviour proven by η's agreement and ι |
| δ2 | "zero direct `resolve_issue` calls" by transcript check | producer-absent (same) | rewritten (INV-10): marker-delimited verb span mirrored against δ1's CLI; 'no direct resolve_issue' enforced by δ1/α behaviour tests, not text |
| ζ | rows 19, 20 "as skill tests on fixture L1s" | producer-absent (same) | rewritten (INV-10): marker-delimited sentinel span mirrored against `recovery_emission.py::RECOVERY_VETO_STREAK_SENTINEL_PREFIX` and harness's `_STALE_L0_STRAND_SENTINEL`; behaviour measured by θ2 over 14 days |
| γ, δ2, ζ | tests under `tests/skills/` | declared-only (no `test_command` collects `tests/skills/`) | tests go in `tests/scripts/` (collected by `tests/scripts/orchestrator.yaml`) |
| θ | the 14-day Q6 flip inside θ | producer-downstream (θ hangs off η; the check measures ζ) | θ2 depends on ζ; λ carries reify's yaml (reify task, external dep) |
| θ | `lease-reap` by an agent | PASS — `~/.claude/fleet/` is in the agent write-set (`write_set.py`), `session_registry.py lease-reap` exists | — |
| ι | "deterministic gate: the gate's own predicate" | producer-absent (no predicate script exists to bind at submit) | `task_kind=normal` + delayed milestone, record appended to PRD §10 (the 8274 shape) |

## Per-leaf bindings

### α — policy table, identities, class claim, shadow-vocabulary lockstep
- `POLICY_TABLE`, `EXECUTOR_IDENTITY`, `COORDINATOR_IDENTITY` → new in `escalation/src/escalation/authority.py`; extends `ROLE_LEVEL_ALLOWLIST`, `l2_auto_close_class`, `PROMOTE_ALLOWED`, `L2_AUTO_CLOSE_DENY_CATEGORIES`/`_ROLES` (all present). PASS.
- `claim_policy_class` tool → new in `escalation/src/escalation/server.py`, beside `resolve_issue`/`stamp_triage`/`promote_to_l2` (present). PASS.
- policy-state file `data/escalations/policy-table-state.json`, reader in α. PASS.
- `FIRST_TRANCHE_CLASSES` / `REVERSIBLE_ACTIONS` gain the table's classes and `done_flip` (`escalation/src/escalation/shadow_ruling.py`, present); `classify._HUMAN_RESOLVERS` gains the coordinator (present). PASS.
- Doc contract: `tests/scripts/test_shadow_ruling_doc_contract.py` (present). PASS.

### β — rotation supervisor, executor rotation, events, inflight, config
- Generalises `harness.py::_start_watcher_supervisor`, `::_run_watcher_rotation`, `::_WATCHER_ALLOWED_TOOLS`, `::_WATCHER_ESCALATION_HEADERS` (present) into `RotationSpec` + `_rotation_supervisor_loop`. PASS (producer α upstream for identity constants).
- `rotation_started`/`rotation_exited` events, `executor-inflight.json`, `executor_*` config keys. New. PASS.
- `%watcher%` ceiling ignores `escalation-executor`: `shared/src/shared/cost_store.py` `LIKE '%watcher%'` does not match the string. PASS.

### γ — executor skill
- `skills/escalation-executor/SKILL.md` new; contract test `tests/scripts/test_executor_skill_policy_contract.py` holds its class list equal to `POLICY_TABLE` (α upstream). PASS.
- Red main via `orchestrator/src/orchestrator/recover_main.py` (present); B3 via `skills/unblock-low-risk` + `orchestrator/src/orchestrator/b3_gate.py` verbs `check`/`record-launch` (present). PASS.
- DecisionRecord per close via `session_registry.py write-decision` / `close-decision` CLI verbs (present). PASS.

### δ1 — sitting tooling
- `apply`, `audit`, `policy-state` verbs over `payloads.py::ApplyPayload`, `prepare_sitting.py::_cmd_resolve_answers`, `inventory.py::queue_project_roots`, `dashboard/src/dashboard/config.py::_discover_escalation_urls` (present); `scripts/sitting/ports.py` new. PASS (α upstream for the coordinator identity and state-file shape).
- `DecisionRecord.audit_outcome` → `orchestrator/src/orchestrator/session_registry.py::DecisionRecord` (present, field new). PASS.
- Trigger-state line: δ1 owns the reader and the state-file shape and renders "evaluator has not run" while it is absent; ε (downstream) writes it. PASS (reader proves absence explicitly).

### δ2 — sitting skill
- `skills/sitting/SKILL.md` new, extracted from `skills/escalation-watcher/SKILL.md` (present); verbs from δ1 upstream. PASS.

### ε — criteria-triggered sitting
- Presence probe over `~/.t3/userdata/statev2.sqlite` `orchestration_v2_projection_messages` (verified in PRD §4); lanes via `skills/spawn/spawn-claude.sh` `t3|tmux|wm` and `spawn_t3_lane.py::_start_argv` (present). PASS.
- Timer live: via κ. PASS after resolution.

### ζ — Q6 detector rules in the L1 skill
- `skills/escalation-watcher-auto/SKILL.md` (present); 7716 upstream for the cluster-L2 member visibility. PASS.

### η1 / η2 — flip gates
- Evidence: `shadow_ruling.py::agreement_report`, `return_brief.py::_agreement_lines` (present); writer: δ1 `policy-state` (upstream). PASS. η2 adds 3346 upstream.

### θ1 — retirement
- `recon-watch/run.sh`, `scripts/recon-watch.sh` present (deletion target); `session_registry.py lease-reap` present. PASS.

### θ2 / λ — dated Q6 check and flips
- `streak_escalation_enabled` in df `dark-factory-orchestrator.yaml` and reify's (both present, `false`). PASS. λ is reify-side: `external_task_id`.

### κ — timer deploy gate
- Installer from ε (upstream). PASS.

### ι — observation
- Reads `runs.db`, `~/.claude/fleet/decisions/`, `data/return-brief.md`, `~/.claude/fleet/leases/`, the trigger state file — all produced by upstream leaves. PASS.
