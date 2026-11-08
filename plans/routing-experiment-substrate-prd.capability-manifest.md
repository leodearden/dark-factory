# Capability manifest — routing-experiment-substrate-prd

Twin of `plans/routing-experiment-substrate-prd.capability-manifest.yaml` (the machine-readable
sidecar `commit_planning` stamps). Bindings verified against main `6c16d8cdd8` (2026-11-03);
every mechanical check below is absent on that commit, so it fails until its producer lands
(polarity rule, `docs/task-authoring.md` §3.3). No check here invokes a committed program: the
only kinds used are grep, path and manual.

| label | capability | binding | verdict |
|---|---|---|---|
| α | `split` condition accepted by the rule vocabulary | producer: α adds `SplitMatch` to `orchestrator/src/orchestrator/config.py` and `_rule_matches` consumes it | PASS (producer upstream in batch) |
| α | `split_bucket` has one home in `shared/` | producer: α creates `shared/src/shared/routing_split.py::split_bucket`; consumers β, ε import it | PASS |
| α | plan shape on every `routing_decision` event | producer: α adds `plan_steps`/`plan_modules` in `workflow.py::_record_routing_decision` and `routing_dispatch.py::resolve_and_record_route` | PASS |
| α | malformed split rejected at load | rejection mechanism: `RuleMatch` is `extra='forbid'` and `SplitMatch` validates `buckets >= 2`, `0 <= bucket < buckets`; α's boundary test observes the structured error | PASS |
| β | readout computes per-task, per-arm metrics across every ledger | producer: β creates `scripts/routing_experiment_readout.py` importing `shared.routing_split.split_bucket` | PASS |
| β | A/A mode exits non-zero on skew | producer: β; the test fixture forces a skew and asserts the exit code | PASS |
| γ | the two split rules sit in the stated order in `defaults.yaml` | producer: γ edits `orchestrator/src/orchestrator/defaults.yaml` and `dark-factory-orchestrator.yaml` | PASS |
| δ | every running unit reloaded with the split rules after a passing A/A check | producer: δ's `before_done` is `scripts/routing_experiment_arm.sh` (committed with this manifest); the reload reports are read from each unit's `reload_config` dispositions, which no grep can see | PASS (manual) |
| ε | curator returns `complexity_hint` and the interceptor applies it on the treated arm only | producer: ε edits `task_curator.py`, `curator_types.py`, `task_interceptor.py` | PASS |
| ε | a filer's own `complexity` always wins | rejection mechanism: ε's test files a candidate with `complexity` set and a `simple` hint on the treated arm and asserts the filer's value survives | PASS (manual) |
| ζ | OPERATIONS.md §7 no longer claims the substrate is inactive | producer: ζ removes the four stale claims; the check asserts the stale phrase is absent | PASS |
| η | the readout gate escalates to Leo 28 days after δ, ε and β are done | producer: η is a deterministic `always_escalates` gate with a `delayed` milestone; the escalation is observed on the queue, not in the tree | PASS (manual) |

Resolutions made to clear the gate: δ's arming program is committed to main before the batch is
filed, because a deterministic task's `before_done` must name an existing executable at
`submit_task`; it fails until β lands, which δ's dependency on β already guarantees.
