# Capability manifest — cpu-load-robust-verify-prd

Per-leaf capability→evidence bindings (G3+G6 mechanized). Substrate verified on `main`
`25e03d4db0`, 2026-07-18. Line refs drift; symbols are canonical. No numeric-accuracy floor is
asserted by any leaf (α's storm threshold is a safety valve, not a correctness bound). No
intra-batch dependencies — the four leaves touch disjoint file sets. `delivered_check`s are
informational for this batch; the sidecar is hand-stamped at decompose (per decompose-mode 5.5).

## α — Merge-gate single flake-retry (must-have core; B+H)

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Failing node-ids extractable from a merge verify (incl. `node down`) | grep:`verify.py:544` `_extract_failing_test_ids`; `_XDIST_NODE_DOWN_PRECEDING_NODEID_RE`/`_XDIST_CRASH_NODEID_RE` `verify.py:534-542` | PASS wired |
| Serial isolated re-run command builder | grep:`verify_cmd.py` `def serial_pytest`; `verify.py:1063` `_serial_pytest_str` | PASS wired |
| main-sweep isolated-rerun-suppress precedent to mirror (INV-5 reuse) | grep:`verify.py:5201` `run_main_tip_sweep` retry-on-flake + `_extract_failing_test_ids` confirm gate | PASS wired |
| Merge-role verify result handler to hook | grep:`verify.py` `run_scoped_verification` + `role='merge'` branch (`verify.py:981` main-probe, merge fan-out `verify.py:4557-4623`) | PASS wired |
| Structured `merge_flake_suppressed` fact emittable (INV-2) | grep:`event_store.py` `EventType` (existing event bus); observation/hypothesis split precedent | PASS wired |
| Storm-streak escalation on the fail-soft path (INV-4) | built by α; consecutive-streak precedent `merge_liveness.py`; boundary test **B4** | PASS (bound as B4) |
| Never suppress a genuine failure (rejection) | built by α; boundary tests **B2** (re-run still fails → red) + **B3** (collection ERROR → red) | PASS (bound as B2/B3) |

## β — Raise merge-path per-test timeout to 300s

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| Per-module merge `test_command`s currently omit `--timeout` (inherit 60s) | grep:`orchestrator/orchestrator.yaml:5` / `fused-memory/orchestrator.yaml:9` `pytest tests/ --tb=short -q` | PASS wired (gap confirmed) |
| `--timeout=N` is an honoured pytest-timeout flag overriding pyproject | existing fallback command uses `--timeout=300` (`dark-factory-orchestrator.yaml:41`) | PASS wired |
| A config-drift guard test exists to extend | grep:`tests/scripts/test_fallback_verify_config.py::test_fallback_verify_raises_per_test_timeout` | PASS wired |

## γ — orchestrator + scripts test-hardening

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `@pytest.mark.timeout(N)` opt-up is sanctioned | grep:`orchestrator/pyproject.toml` `--max-worker-restart=0` comment sanctions `@pytest.mark.timeout` opt-ups | PASS wired |
| Load-scaled grace helper to reuse | grep:`tests/scripts/test_spawn_claude.py:882` `_load_scaled_grace` | PASS wired |
| Shared config-mock factory to set numeric `claimant_heartbeat_interval_secs` | grep:`orchestrator/tests/conftest.py:616` `mock_orch_config` | PASS wired |
| Target tests exist | grep:`test_coalesce_integration_gate.py`, `test_cli.py::test_verify_merge_cancel_end_to_end`, `test_spawn_claude.py::test_window_close_129_robust_to_delayed_trap_install` | PASS wired |

## δ — fused-memory test-hardening

| Capability asserted by the signal | Evidence | Verdict |
|---|---|---|
| `integration` marker declared (so `-m 'not integration'` excludes) | grep:`fused-memory/pyproject.toml` markers `integration:` | PASS wired |
| `QdrantClient.collection_exists` for idempotent create | qdrant-client API (installed) | PASS wired |
| Target tests exist | grep:`test_ticket_worker.py::test_threshold_parks_oversize_ticket_as_lookahead`, `test_harness.py::test_timeout_marks_run_failed`, `test_mem0_qdrant_integration.py`, `test_mem0_client.py::TestMem0BackendAddSystemRecordIntegration` | PASS wired |
| Coverage-lane gap flagged, not silent (INV-4) | PRD §9 records: no `-m integration` lane exists → compat coverage lapses until a lane is added | PASS (flagged in PRD) |

No FAIL bindings. Batch clear to queue.
