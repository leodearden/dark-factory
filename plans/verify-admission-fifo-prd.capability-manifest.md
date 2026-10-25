# Capability manifest — verify-admission-fifo-prd

Bindings checked against main `984016b697` (2026-09-18). Cited by symbol; re-locate at
implementation time. Machine-readable twin: `verify-admission-fifo-prd.capability-manifest.yaml`.

## α — Ordered task-verify admission gate with recorded wait

| capability the signal asserts | evidence | verdict |
|---|---|---|
| The gate sits on the only gated acquire path | `orchestrator/src/orchestrator/verify.py::_admission_slot` is the sole production caller of `shared.verify_admission.acquire_task_slot` (`git grep`: `offline_lane.py` imports `nice_prefix` only; `merge_queue.py` and `merge_lane/` have zero references) | PASS — wired |
| A plan-level event can be emitted without new plumbing | `verify.py::run_scoped_verification` already takes `event_store` and already aggregates every leg's `VerifyResult`; the two `workflow.py` task-verify call sites do not pass it — α wires both | PASS — producer is α |
| A per-leg record exists to carry the grant | `verify.py::CheckRun` (task 2133) — flat, JSON-native, `to_dict()` written to the attempt summary; new field goes last per its own comment | PASS — wired |
| New knobs are hot-reloadable | `config.py::RELOADABLE_FIELDS` lists all seven existing `verify_admission_*` keys | PASS — wired |
| `EventType` accepts new members | `event_store.py::EventType` is a `StrEnum` | PASS |
| Ungated roles bypass | `shared/verify_admission.py::is_gated_role`; task 5424's inline path in `_admission_slot` | PASS — wired |
| The sweep-yields boundary property is already pinned | `orchestrator/tests/test_verify_admission_integration_gate.py::TestSweepYieldsAndInterleaves` | PASS |
| Ordering / cancellation semantics | built and bound by α: boundary scenarios 1–10, 12 against the real flock | PASS — manual |

No numeric bound is asserted as a pass criterion. The PRD's predicted effect of the flip
(mean ±0, p90 −30 %, max −66 %) is an expectation recorded for β's comment, not a signal.

## β — Flip Dark Factory to arrival order

| capability | evidence | verdict |
|---|---|---|
| `verify_admission_order: arrival` validates and reloads | producer: α, upstream | PASS |
| "no same-class overtake in the rows" is checkable | producer: α's `verify_admission_plan` payload (`arrived_at`, `granted_at`, `role`, `order`), upstream | PASS |

DAG direction: α → β. β is inert until a process running α's code reloads or restarts; the
signal keys on the event's `order` field for that reason.

## γ — Flip reify to arrival order (filed in reify's task store)

Same bindings as β; external dependency `dark_factory:<α>`. Not in the YAML sidecar — the
stamper keys on this project's batch.

## δ — Decide cancel-on-red

| capability | evidence | verdict |
|---|---|---|
| Legs of one plan can be joined, with outcome and grant time | producer: α (`plan_id`, per-leg `granted_at`, `held_secs`, `rc`, `timed_out` in one row) | PASS — upstream |
| Rows are tagged with the policy in force | producer: α (`order`), flipped by β | PASS — upstream |
| The 3 % decision rule | a decision threshold, not an achievability claim: ≈ 0.7 h/day of a saturated slot, the size of lever the 2026-09-11 verify-speed synthesis treated as worth an owner (0.4–1.0 h/day) | n/a |
