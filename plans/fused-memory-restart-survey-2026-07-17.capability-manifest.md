# Capability manifest — fused-memory restart-safety batch (phases 0–2)

PRD: `plans/fused-memory-restart-survey-2026-07-17.md` §Decomposition plan.
Substrate evidence below was adversarially verified by the 2026-07-17 survey
(33 confirmed findings, file:line-cited); bindings reference those citations.
Machine-readable twin: `fused-memory-restart-survey-2026-07-17.capability-manifest.yaml`.

| L | Capability the signal asserts | Evidence binding | Verdict |
|---|---|---|---|
| α | uvicorn accepts `timeout_graceful_shutdown`; fm's Config call site exists | grep: server/main.py:980-987 (uvicorn.Config, no kwarg today); .venv uvicorn/config.py:218 (param exists, default None) | PASS |
| β | close budget lives in `_graceful_shutdown` step table | grep: server/main.py:1738-1823 (_graceful_shutdown per-step budgets); journal 2-for-2 "memory_service.close timed out after 5.0s" | PASS |
| γ | `_draining` flag exists; `_project_loop` is the un-gated site | grep: harness.py:1580 (only spawn-gate read), harness.py:1639-1786 (loop body, zero `_draining` reads) | PASS |
| δ | /health endpoint exists to carry `recon_busy`; restart script exists | grep: server/tools.py:700-721 (/health); scripts/restart-fused-memory.sh:12-48 | PASS |
| ε | fail-empty site + guard bypass are as cited | grep: scheduler.py:2004-2027 (except→[]), :5964-5973 (empty-map drain), :4597-4619 (unknown-id→False) | PASS |
| ζ | ASGI shield is the 500 emitter during shutdown | grep: server/main.py:329-354 (_ASGIExceptionShield 500 fallback) | PASS |
| η | client retry knobs exist to resize; 503 already retryable | grep: mcp_lifecycle.py:432-434 (_RETRYABLE_STATUS {502,503,504}, _MCP_MAX_RETRIES), deterministic_runner.py:278-295 (writeback budget) | PASS |
| θ | plain write site + house atomic pattern both exist | grep: manifest_stamping.py:246-250 (write_text); curator_escalator.py:182-224 (temp+os.replace exemplar) | PASS |
| ι | untracked judge spawn sites + shutdown finally as cited | grep: harness.py:1952, :2911 (bare create_task), :1628-1636 (finally iterates only _project_tasks) | PASS |
| κ | in-memory queue + SQLite durability substrate + DL read path exist | grep: event_queue.py:128,229-252 (asyncio.Queue enqueue); shared/async_sqlite_base.py:57-86 (durability pragmas); tools.py:2529-2608 (DL read-only today) | PASS |
| λ | dual-write branch + post-hoc journal + durable-queue pattern exist | grep: memory_service.py:2085-2146 (dual_write branch), :831-866 (journal-after-await); durable_queue.py:194-216 (sync enqueue pattern) | PASS |
| μ | age-based cutoff + project-scoped restore are the sites cited | grep: journal.py:629-638 (started_at cutoff); harness.py:1083-1213 (_recover_stale_runs); event_buffer.py:593-604 (project-scoped restore) | PASS |
| ν | write_ops journal + R4 idempotency precedent exist | grep: write_journal.py:18-33 (schema); task_interceptor.py:528 (server-side uuid4), ~2003-2070 (R4 escalation-idempotency precedent) | PASS |
| ξ | watchdog WATCHED/paths tuples + parity gate exist to extend | grep: orchestrator-watchdog.py:41-49, :77-83 (no fused entries); check_fused_memory_unit_parity.py (2-entry allowlist) | PASS |
| ο | staleness_pass + shared-clock pattern exist; gate script from δ | grep: orchestrator-watchdog.py staleness_pass; producer:task-δ upstream (defer-if-busy chokepoint) | PASS |
| π | CLI session plumbing exists; recon discards it today | grep: cli_invoke.py:1315-1328 (--session-id/--resume), :1555 (AgentResult.session_id); cli_stage_runner.py:316-361 (session_id never read); journal.py:40-52 (runs schema, no column) | PASS |
| ρ | ReconReportState exists in-process; SQLite base available | grep: stages/base.py:56 (recon-report port), cli_stage_runner.py:66-70; shared/async_sqlite_base.py | PASS |
| σ | resume+prompt plumbing, per-stage report persistence, cancel handler all exist | grep: cli_invoke.py:824-833 (resume_delivers_prompt), :1098-1106 (in-flight resume); harness.py:1913-1917/:2037 (stage_reports persisted), :1970-2001 (CancelledError handler) | PASS |
| τ | green/red-tier reload precedent exists to mirror | grep: escalation reload_config (CLAUDE.md contract, applied/restart_required dispositions); fm config schema fused_memory/config/schema.py | PASS |

G6 numeric bases: δ's 35-min defer cap ← measured full cycle 29:58 (run
97b49a64, schema.py:519-536); η's ~120s retry window ← TimeoutStopSec=90 +
observed 14-15s restart (journal 07-16/07-17); σ's 1h freshness window ←
stage_timeout_seconds=3600 (config default; shipped as a tunable knob, not a
test assertion). No rejection-style signals assert unbuilt mechanisms; no
FAIL bindings.
