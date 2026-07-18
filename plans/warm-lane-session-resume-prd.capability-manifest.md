# Capability manifest — `warm-lane-session-resume-prd.md`

Mechanizes G3 + G6 for the session-resume decomposition. Each Greek-label
block lists the capabilities that label's user-observable signal asserts,
bound to evidence: `producer:task-<label>` when a sibling in this batch
delivers it (DAG-direction: producer is **upstream**), or `grep:<path>`
when the capability is pre-existing substrate already wired on `main`.

Substrate verified 2026-07-18 against `main@8cf05ca9cf` (double-nested
layout `orchestrator/src/orchestrator/…`). Every binding PASSes; no FAIL
binding blocks the queue. The machine-readable twin (with optional
dispatch-time `delivered_check`s) is
`warm-lane-session-resume-prd.capability-manifest.yaml`.

DAG: α → β → γ → ω → ε; γ → ψ.

---

## α — Sidecar v2 + preserve-on-cancellation (intermediate → β, γ)

- **sidecar-writer-present** → `grep:orchestrator/src/orchestrator/artifacts.py`
  — `def write_agent_session(` / `def read_agent_session(` exist on main;
  v1 payload is `session_id`/`role`/`started_at`/`owner_pid`. α adds the
  `task_id`/`resume_count`/`schema_version` fields (correctly ABSENT today).
  Evidence: contract §7 fixes the field names. **PASS**
- **clear-on-cancel-seam-present** → `grep:orchestrator/src/orchestrator/workflow.py`
  — `self.artifacts.clear_agent_session()` runs unconditionally in `_invoke`'s
  bare `finally` today (the inversion α fixes: clear only on completion,
  preserve + emit `agent_session_preserved` on `CancelledError`). **PASS**

## β — Recovery adopts sessions alongside plans, lanes included (intermediate → γ, ω)

- **recovery-entry-present** → `grep:orchestrator/src/orchestrator/harness.py`
  — `async def _recover_crashed_tasks` + `self._recovered_sessions[…] = …`
  exist on main; today the sidecar is read only in the no-plan/non-lane
  branch. β widens the plan-present + lane branches to adopt too. **PASS**
- **lane-affinity-substrate-present** → `grep:orchestrator/src/orchestrator/git_ops.py`
  — `AcquireRoute.DISK_BACKSTOP_REUSE` + `_reuse_warm_lane` already re-acquire
  the same lane without reseed and preserve `.task/`; `_reset_warm_lane` is the
  foreign-task reseed path. β relies on this unchanged (I5). **PASS**

## γ — Guarded injection + config + events + prompt note (intermediate → ω, ψ)

- **injection-point-present** → `grep:orchestrator/src/orchestrator/harness.py`
  — `_run_slot` already injects `resume_session_id=recovered_session`; γ adds
  the eligibility predicate (corroborate transcript + freshness + cap) in
  front of it. **PASS**
- **green-tier-config-pattern-present** → `grep:orchestrator/src/orchestrator/config.py`
  — the `RoutingConfig`/`DeliveredChecksConfig`/`UnblockAutoConfig` submodel +
  `_submodel_leaf_paths(...)` idiom is the template γ's `SessionResumeConfig` /
  `session_resume:` block follows (correctly ABSENT today). **PASS**
- **resume-prompt-owner-present** → `grep:shared/src/shared/cli_invoke.py`
  — `CRASH_RECOVERY_RESUME_PROMPT` + `resume_delivers_prompt: bool = False`
  own the prompt swap and the resume-failure → fresh-fallback; γ appends one
  L0-dismissal sentence and does **not** flip the flag (I4). **PASS**
- **stale-escalation-dismiss-present** → `grep:orchestrator/src/orchestrator/harness.py`
  — `async def _dismiss_stale_escalations` exists; the prompt sentence tells a
  resumed agent its pre-restart escalations may have been auto-dismissed. **PASS**

## ω — Integration gate (leaf; the G2 user-observable, two-way)

- **all-legs-upstream** → `producer:task-α, task-β, task-γ` (all **upstream**
  of ω; DAG-direction PASS) — schema-v2 survival (α), plan-present/lane adopt
  (β), guarded injection + fallback (γ). **PASS**
- **restart-sim + resume-seam substrate** →
  `grep:orchestrator/tests/test_crash_recovery.py` (restart-simulation idiom)
  + `grep:shared/src/shared/cli_invoke.py` (`--resume`) /
  `grep:orchestrator/src/orchestrator/agents/invoke.py` (`--session`). The
  end-to-end resume seam is wired on main; ω drives B1–B11 over it. **PASS**

## ε — Deploy capstone (leaf, deterministic auto-deploy)

- **fleet-restart-script-present** → `grep:scripts/restart-all-orchestrators.sh`
  — committed, executable (`-rwxrwxr-x`), `--drain` arm present; `SELF_UNIT`
  defers `orchestrator-dark-factory.service` to last. DeterministicRunner
  auto-deploy preset (`before_done` present, `always_escalates=false`) is the
  proven pattern (task 2065). **PASS**
- **resume-observable-post-deploy** → `producer:task-ω` upstream — the
  `resuming prior session` line the next scheduled redeploy produces requires
  α+β+γ+ω landed (all upstream). Open-question 5: the deploy's own sidecars are
  v1 → plan.json fallback (B11); no false premise. **PASS**

## ψ — Companion cross-PRD prose update (leaf, docs-only, simple)

- **target-prds-present** → `grep:plans/worktree-lane-lifecycle-prd.md` +
  `grep:plans/agent-transcript-archival-prd.md` — both docs exist on main; ψ
  adds this PRD's seam name to their cross-PRD tables. Text settles only after
  γ names the mechanisms (ψ depends on γ). **PASS**
