# Provenance — `warm_lane_golden` fixtures

Canonical golden records for reify's warm-lane-audit. Reify vendors a
byte-for-byte copy in `tests/infra/fixtures/warm_lane_golden/`.

## The rule

> **These bytes belong to their producers. Never hand-edit them.** A red
> pinning test means a producer changed: refresh the golden from the produced
> bytes the failure names, never edit it to match. The refresh procedure lives
> in the pinning test's module docstring and only there.

| File | Producing code site | Pinning test | Consumer |
|---|---|---|---|
| `lane-state.json` | `orchestrator/src/orchestrator/lane_lifecycle.py::LaneRecord.to_json` | `orchestrator/tests/test_warm_lane_golden_records.py` | reify `tests/infra/fixtures/warm_lane_golden/`, driven by reify `tests/infra/test_warm_lane_audit.sh` |
| `lane-states.json` | `orchestrator/src/orchestrator/lane_lifecycle.py::LaneState` | `orchestrator/tests/test_warm_lane_golden_records.py` | reify `tests/infra/fixtures/warm_lane_golden/`, driven by reify `tests/infra/test_warm_lane_audit.sh` |
| `plan.json` | `orchestrator/src/orchestrator/mcp/plan_tools.py::create_server` through `orchestrator/src/orchestrator/artifacts.py::TaskArtifacts` | `orchestrator/tests/test_warm_lane_golden_records.py` | reify `tests/infra/fixtures/warm_lane_golden/`, driven by reify `tests/infra/test_warm_lane_audit.sh` |
