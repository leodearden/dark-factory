# Capability manifest — os-sandbox-worktree-containment-prd.md

Decompose-time G3+G6 bindings for the OS-sandbox worktree-containment batch
(decomposed 2026-07-22 against main `03edc85d3b`). Machine-readable twin:
`os-sandbox-worktree-containment-prd.capability-manifest.yaml` (strictly
PRD-derived path; stamped by `commit_planning`).

All G3 substrate claims re-verified live this session: `SandboxConfig`
(`orchestrator/src/orchestrator/config.py:966`), dispatcher + both backends +
`writable_extras` plumbing (`agents/sandbox_dispatch.py:84,106,113`,
`agents/sandbox.py:60`, `agents/landlock.py:73`), role gating
(`roles.py:480,544`), eval runner pin (`evals/runner.py:244`
`enabled=False`), `EventType` enum (`event_store.py:44`),
`DeterministicRunner` deploy/predicate/self-unit paths
(`deterministic_runner.py` phase γ/ε), submit-boundary predicate guard
(`fused-memory/.../middleware/deterministic_task_guard.py`), external-dep
gate, `restart-orchestrator.sh` (executable; self-unit intended caller),
reify `scripts/orchestrator-redeploy-restart.sh` (executable; schedule mode).
**All PASS — no unbuilt substrate.**

## Decompose-time refinements vs PRD text (recorded, not drift)

1. **Probe-report path pinned** to `docs/sandbox-containment-probe-report.md`
   (PRD wrote `<wt>/probe-report.md`, which would land at repo root). γ4
   writes it, γ1's predicate checks it, the γ4 delivered-check greps it —
   one agreed path across all three.
2. **γ6 split into γ6a (config flip) + γ6b (deploy)** per the PRD's own "as
   reify tasks" plural; γ7's external dep targets γ6b (the terminal one).
3. **γ7 external deps widened** to include the five sibling-registry flips
   (γ7a–γ7e) alongside `reify:<γ6b>` — γ7's census signal ("no
   `enabled: false` factory target") must be producible from its own
   dependency set (G6).
4. **Sibling flips are config-only** (no per-project deploy tasks): sibling
   wording avoids repo-specific mechanics, and
   `scripts/restart-all-orchestrators.sh` dynamically restarts every
   `orchestrator-*.service`, so the scheduled fleet redeploy (≤8h cadence)
   activates a landed flip. DF (γ3) and reify (γ6b) get explicit deterministic
   deploys because canary/first-follower promptness + verification matter.
5. **Fail-safe predicate stub committed at decompose**
   (`scripts/check_sandbox_soak.sh`, exits 2, marker `STUB-NOT-IMPLEMENTED`)
   — the submit-time guard requires `before_done.script` to exist; γ1
   replaces it, pinned by an `expect: absent` delivered-check on the marker.
6. **Fleet census enumerated 2026-07-22** (registry =
   `DASHBOARD_KNOWN_PROJECT_ROOTS` + primary): factory targets with
   orchestrator configs = dark-factory (γ2), reify (γ6a/b),
   autopilot-video (γ7a), know-live (γ7b), pump-web-ui (γ7c),
   solar-challenge (γ7d — runs under the legacy-named unit
   `orchestrator-my-solar-challenge.service`), solar-challenge-platform
   (γ7e), plus the dashboard module config (`dashboard/orchestrator.yaml`,
   γ7 itself, edited in place per D13). autotrade + mission-control are
   recon-only registrations with no orchestrator config — not factory
   targets. Evals stay `enabled=False` (D12). solar-challenge and
   solar-challenge-platform have **no** sandbox block today (flip = add);
   the other three have `enabled: false` (flip = edit).

## Per-leaf bindings

| Label | Capability | Binding | Verdict |
|---|---|---|---|
| α1 | writer-audit findings committed | producer:α1; check = `FINAL-WRITABLE-LIST` marker in `plans/` (dir-safe pathspec) | PASS |
| α2 | single-source write-set fn | producer:α2; `def compute_write_set` under `agents/`; consumes existing `writable_extras` params (wired, no live caller today — the PRD's natural vehicle) | PASS |
| α3 | call-site consumes WriteSet; SIMPLE_TASK sandboxed | producer:α3; `compute_write_set` in `workflow.py`; SIMPLE_TASK flag judged by wiring-pin tests (manual) | PASS |
| α4 | 12-row real-kernel matrix suite | producer:α4; module name mandated `test_sandbox_enforcement_matrix` (pattern-anchored, not file:line); prior art `test_landlock.py:106-148` | PASS |
| α5 | maintenance discipline applied idempotently | producer:α5; `maintenance\.auto` in orchestrator src (absent today) | PASS |
| β1 | fail-closed refusal + deduped escalation | producer:β1; exception name mandated `SandboxUnavailable` (mirror of recon's `RemediationSandboxUnavailable`, prior art task 1935) | PASS |
| β2 | sandbox_applied / sandbox_unavailable events | producer:β2; `sandbox_applied` in `event_store.py` (absent today) | PASS |
| γ1 | real soak predicate replaces stub | producer:γ1; `STUB-NOT-IMPLEMENTED` **absent** from `scripts/check_sandbox_soak.sh` | PASS |
| γ2 | DF config flip | producer:γ2; `^ +backend: landlock` in `dark-factory-orchestrator.yaml` (value-anchored; absent today — no sandbox block exists) | PASS |
| γ3 | DF restart deploy | runner-verified (`done_provenance='deterministic-deploy-scheduled'`, self-unit phase ε) — manual | PASS |
| γ4 | committed containment-probe report | producer:γ4; `CONTAINMENT-PROBE-RESULT` marker in `docs/` (dir-safe) | PASS |
| γ5 | predicate-verified soak done | the task IS the check (`before_done.kind='predicate'` + 3-day delayed milestone) — manual | PASS |
| γ6a/γ6b | reify flip + deploy | reify-registry tasks (DF stamper can't reach; hand-stamped ids) — manual | PASS |
| γ7 | dashboard flip + docs + census | producer:γ7; `^ +backend: landlock` in `dashboard/orchestrator.yaml` (the line-64 comment does not match the value-anchored pattern — verified absent today) | PASS |
| γ7a–γ7e | sibling-registry flips | sibling registries (unstamped by DF `commit_planning`; hand-stamped ids) — manual | PASS |
| δ1 | plan-target-granularity gate escalation | pure gate (`always_escalates`, no `before_done`) — the esc-2508-1 pattern — manual | PASS |

G6 numeric/exactness premises: soak floor (≥10 done tasks / 3 days) is well
under DF throughput and producible from γ5's own dep set (β2 events × γ3
enablement × γ4 report); denial exactness (EACCES/EROFS, rows 3–6/9) is
strace- + `test_landlock.py`-backed with an active rejection mechanism
(landlock ruleset). No FAIL bindings — batch clear to queue.
