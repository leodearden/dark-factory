# Capability Manifest — deterministic-task-kind-prd

Mechanizes G3 (substrate exists/wired) + G6 (premise valid) per leaf. One block per leaf signal; each asserted capability bound to evidence. Any **FAIL** binding blocks queueing.

**Domain flags:** no grammar/DSL → grammar-fixture checks **N/A**; no numeric bounds/thresholds anywhere → numeric-floor checks **N/A**. The live checks here are **capability→producer (wired)**, **DAG-direction**, **field-population**, and **rejection-mechanism**.

All `file:line` evidence verified during the study that produced this PRD.

---

## α — `task_kind` param + structured fields + validation  *(intermediate; carries an observable rejection signal)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `submit_task` rejects `deterministic ∧ before_done=None ∧ always_escalates=false` | rejection-mechanism | **producer = α** — α's RED test authors the corner and observes the rejection diagnostic; rejection is α's deliverable, not assumed substrate | PASS (built+bound by α) |
| `submit_task` rejects `before_done` on a `normal` task | rejection-mechanism | producer = α; same RED test pattern | PASS |
| `task_kind` + `metadata.before_done`/`always_escalates` persist & round-trip | capability→producer | producer = α; observable via existing `get_task` read path (`fused-memory tools.py` get_task) | PASS |
| Task store accepts the new param without migration | capability→producer (wired) | `sqlite_task_backend.py:54-67` — 9 cols + untyped JSON `metadata`; `task_kind` already lives in metadata (task 1680). No schema change | PASS |

## β — Deterministic dispatch + runner + pure-gate end-to-end  *(leaf — gate integration slice)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Scheduler routes a `deterministic` task to the runner at dispatch | capability→producer (wired) | producer = β — new branch at the harness dispatch point that today builds `TaskWorkflow` (`harness.py:3173-3201`); gated by existing `_eligible_for_dispatch` (`scheduler.py:2372-2441`) | PASS (β wires into the production dispatch path) |
| `task_kind` field available to read at dispatch | DAG-direction | producer = α, **upstream** of β (β depends α) | PASS |
| Programmatic **born-at-L2** submit from sentinel | capability→producer (wired) | **exists** — `EscalationQueue.submit(Escalation(...))` used ~15× incl. `harness.py:2381,5293`; `level=2` stamp + sentinel exemption `server.py:45-52,215-276` | PASS |
| Born-at-L2 carries real summary/detail (not sentinel) | field-population | producer = β writes task title+description+dep IDs into the L2 `summary`/`detail` | PASS |
| `blocked` + open-L2 quiescence (no churn) | capability→producer (wired) | **exists** — `harness.py:2289,2344-2352` | PASS |
| `resolve_issue(resume)` → `blocked`→`pending` re-dispatch | capability→producer (wired) | **exists** — `server.py:450-538`; `harness.py:5218-5345`. β adds the deterministic re-dispatch interpretation (`gate_escalated_at`→`done`) | PASS |

## γ — Cross-unit blocking deploy + verify  *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Run committed script blocking, once | capability→producer (wired) | producer = γ (runner subprocess); payload `reify/scripts/orchestrator-redeploy-restart.sh --exec-restart` **exists** (read in study); `before_done_ran_at` once-only stamp = γ | PASS |
| Verify fresh `MainPID` + `ActiveEnterTimestamp` | capability→producer | **exists** — `systemctl --user show -p MainPID,ActiveState,ActiveEnterTimestamp` (the exact probe used by deploy records 1793/1800/1863) | PASS |
| `done_provenance.kind='deterministic-deploy'` populated with **non-sentinel** PID | field-population | producer = γ writes the verified live MainPID (not a placeholder); RED test asserts a real integer PID | PASS |
| Escalate-on-fail (rc≠0 / stale PID) → born-at-L2 | DAG-direction | born-at-L2 producer = β, **upstream** (γ depends β) | PASS |

## δ — `escalation submit` CLI  *(intermediate; observable submit signal)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Script-callable submit to the escalation queue | capability→producer (wired) | **substrate exists** — `EscalationQueue(queue_dir)` is file-backed (`harness.py:4293`), `submit()` writes a file with no MCP server required; producer = δ adds the CLI/entrypoint wrapper | PASS |
| Submitted L2 is readable | capability→producer (wired) | **exists** — `get_pending_escalations` read path | PASS |

## ε — Runner detached self-restart + OnFailure  *(leaf)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| `systemd-run --user --on-active --on-failure=<unit>` | capability→producer (wired) | **exists** — `orchestrator-redeploy-restart.sh:155-167` already uses `systemd-run --user --on-active --unit --collect`; `--on-failure` is a standard `systemd-run` flag; producer = ε adds it | PASS |
| `OnFailure` target files a born-at-L2 | DAG-direction | producer = δ, **upstream** (ε depends δ) | PASS |
| Dispatching orchestrator not killed by the task itself | end-to-end premise | detached transient unit is a child of the **user** systemd manager and fires on `--on-active` delay **after** the runner returns (`orchestrator-redeploy-restart.sh:9-21`); the task reaches `done` before the unit fires | PASS |
| `done_provenance.kind='deterministic-deploy-scheduled'` populated | field-population | producer = ε writes the scheduled transient-unit name + fire time | PASS |

## θ — Boundary-test suite B1–B12  *(leaf — C-as-integration-gate)*

| Capability asserted | Check | Evidence | Verdict |
|---|---|---|---|
| Every B*n* capability | DAG-direction | all produced by α/β/γ/δ/ε, **upstream** of θ (θ depends β, γ, ε) | PASS |
| Restart-window replay (B11) | capability→producer | stamp persistence via metadata (durable in the task store); θ simulates restart by re-running the runner against a stamped task | PASS |

## ζ — Docs + convention correction  *(leaf — sanctioned companion correction-task)*

No substrate assertion (prose-only). Per author-mode Stage 7, a companion correction task is roped to the batch via its dependency on θ. Signal is a `CLAUDE.md` content change. **G2 docs-only escape-hatch applies.**

---

**Manifest verdict:** all bindings **PASS**. No FAIL → does not block queueing. The single new substrate deliverable (δ's `escalation submit` CLI) is filed as a tracked task with the file-backed queue confirmed present.
