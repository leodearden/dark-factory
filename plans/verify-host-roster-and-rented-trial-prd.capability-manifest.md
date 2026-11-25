# Capability manifest: a rented verify host, a live host roster, and a throughput trial

**PRD:** `plans/verify-host-roster-and-rented-trial-prd.md`. **Machine twin:** `plans/verify-host-roster-and-rented-trial-prd.capability-manifest.yaml`, the single home of the capability list and its verdicts.
**Substrate verified** on main `17fe85747c` (2026-11-25). Every pointer below was resolved against `main` by symbol.

**Polarity.** Every `path` and `grep` check below fails on that tree: none of the named files exists and none of the named strings appears in the named file.

## Decompose-time contracts (not in the PRD's words; the checks depend on them)

1. **File names are fixed.** The checks name `orchestrator/src/orchestrator/verify_host_roster_file.py`, `scripts/deploy/provision_verify_host.sh`, `scripts/deploy/check_verify_host.py`, `scripts/deploy/verify-host-wrapper.template.sh`, `config/verify-hosts.example.yaml`, `orchestrator/src/orchestrator/verify_host_roster.py`, `orchestrator/src/orchestrator/merge_lane/host_trust.py` and `scripts/verify_host_trial_report.py`. Each producing task's text states its paths.
2. **α2's hook is in `config.py`.** The check looks for the string `verify_host_roster_file` there, so the call into the parser sits in or beside `config.py::load_config`, not in a caller.
3. **γ names the planner function `plan_host_changes`** and calls it from `harness.py`.
4. **The gates (ζ to λ) carry no mechanical check.** Each is resolved by a human or the standing watcher citing the evidence its row names.

## Bindings

### α1: roster file parser, provisioning script, contract checker, known host-tied tests

| capability | binding | verdict | delivered check |
|---|---|---|---|
| the gate wrapper contract exists | `grep:scripts/deploy/install_laptop_gate_shim.sh` the ORCH_GATE_BIN wrapper contract the template must keep | PASS | — |
| the per-sha gate builder exists | `grep:orchestrator/src/orchestrator/verify_runner.py::RemoteRunner._dispatch` calls ensure_gate before every remote verify | PASS | — |
| the known host-tied tests exist | `grep:scripts/run_vllm_eval.py` `grep:fused-memory/tests/test_main_ticket_janitor_wiring.py` `grep:shared/tests/test_invocation_outcome.py` the three files the task fixes | PASS | — |
| roster file parser | built here | PASS | path present |
| provisioning script and checker | built here | PASS | path present |
| wrapper template and example roster | built here | PASS | path present |
| script and checker share one requirement table | built here | PASS | manual |

### α2: the orchestrator reads the roster file

| capability | binding | verdict | delivered check |
|---|---|---|---|
| environment ranks above the yaml | `grep:orchestrator/src/orchestrator/config.py::OrchestratorConfig.settings_customise_sources` returns init, env, yaml, dotenv in that order | PASS | — |
| config load entry point | `grep:orchestrator/src/orchestrator/config.py::load_config` the one loader every entry point uses | PASS | — |
| roster parser available | producer:task-α1 upstream | PASS | — |
| roster hook in config loading | built here | PASS | grep present |
| both sources rejected | built here; rejection mechanism added and exercised by the task's tests | PASS | manual |

### β: `orchestrator verify-host-preflight`

| capability | binding | verdict | delivered check |
|---|---|---|---|
| merge-verify spec builder wired | `grep:orchestrator/src/orchestrator/merge_lane/post_merge_verify.py::_PostMergeVerify.run` builds the spec with build_merge_verify_spec for every lane verify | PASS | — |
| remote dispatch path wired | `grep:orchestrator/src/orchestrator/verify_runner.py::RemoteRunner._dispatch` publish, ensure_gate, verify-merge | PASS | — |
| per-module failing ids and skips available | `grep:orchestrator/src/orchestrator/verify.py::_extract_failing_test_ids_from_junit` failing ids and the skipped vector are parsed from junit | PASS | — |
| CLI group exists | `grep:orchestrator/src/orchestrator/cli.py::main` the click group verify-merge and verify-task hang from | PASS | — |
| preflight command | built here | PASS | grep present |
| skip-parity exceptions are owned | built here; task 7030 owns the reify-dependent guards | PASS | manual |

### γ: live verify-host roster

| capability | binding | verdict | delivered check |
|---|---|---|---|
| reload entry point | `grep:orchestrator/src/orchestrator/harness.py::Harness.reload_config` the only caller of apply_reload on a running orchestrator | PASS | — |
| reload allowlist | `grep:orchestrator/src/orchestrator/config.py::RELOADABLE_FIELDS` `grep:orchestrator/src/orchestrator/config.py::apply_reload` | PASS | — |
| merge-lane allocator | `grep:orchestrator/src/orchestrator/verify_runner.py::HostAllocator.cancel_and_release` the park path a leaving host must not take | PASS | — |
| task-side pool | `grep:orchestrator/src/orchestrator/verify_hosts.py::HostPool.policy` one slot value for every host today | PASS | — |
| remote runner buildable at reload | `grep:orchestrator/src/orchestrator/verify_hosts.py::_remote_runner` needs the deployed sha, the checkout resolver and git.main_branch only | PASS | — |
| quarantine single writer | `grep:orchestrator/src/orchestrator/merge_lane/host_quarantine.py::HostQuarantine` | PASS | — |
| roster source is the private file | producer:task-α2 upstream | PASS | — |
| host change planner | built here | PASS | path present |
| reload applies the plan | built here | PASS | grep present |
| host lifecycle events | built here | PASS | grep present |

### δ: per-host trust

| capability | binding | verdict | delivered check |
|---|---|---|---|
| local re-run of a remote green wired | `grep:orchestrator/src/orchestrator/merge_lane/post_merge_verify.py::_PostMergeVerify.run` calls cross_check_remote_green with the runner in hand | PASS | — |
| cross-check selection | `grep:orchestrator/src/orchestrator/merge_lane/post_merge_verify.py::_PostMergeVerify._cross_check_remote` | PASS | — |
| live roster and admission probe | producer:task-γ upstream | PASS | — |
| host trust module | built here | PASS | path present |
| graduation number is config | built here | PASS | grep present |
| local re-run of a remote red | built here | PASS | manual |

### ε: `verify_host_trial_report.py`

| capability | binding | verdict | delivered check |
|---|---|---|---|
| event vocabulary | `grep:orchestrator/src/orchestrator/event_store.py::EventType` merge_verify, merge_finalized, dispatch_deferred and task_started are members | PASS | — |
| sampler vocabulary | `grep:sampler/src/sampler/metrics.py::METRIC_GROUPS` | PASS | — |
| trial report script | built here | PASS | path present |
| a block with no host row is not a zero | built here | PASS | manual |

### Gates

| label | what it needs | binding | verdict |
|---|---|---|---|
| ζ | the orchestrators read the roster file | producer:task-α2 upstream, and deployed to the running fleet | PASS |
| η | the tools exist before rent starts | producer:task-α1 upstream; producer:task-β upstream | PASS |
| θ | the host is rented and reachable | producer:task-η upstream | PASS |
| ι | the host is serving the Python projects | producer:task-θ upstream | PASS |
| κ1 | live disable and a trusted host | producer:task-γ upstream; producer:task-δ upstream; producer:task-θ upstream | PASS |
| κ2 | the previous block is closed | producer:task-κ1 upstream | PASS |
| κ3 | the previous block is closed | producer:task-κ2 upstream | PASS |
| κ4 | the previous block is closed | producer:task-κ3 upstream | PASS |
| λ | two on-blocks and two off-blocks; the report script | producer:task-κ4 upstream; producer:task-ε upstream | PASS |

## Gate re-walk at decompose (2026-11-25)

- **G1, G3, G4:** unchanged from the PRD's §4, §8 and §10. One correction made before commit: the preflight does not take the orchestrator's host slot (PRD §6.2), because a separate process cannot reach the running allocator.
- **G2:** α1 and α2 are intermediates (they unlock η, θ, ζ and γ). β, γ, δ and ε carry operator-visible signals. λ is the PRD's signal leaf.
- **G6:** the 18% threshold is a decision rule read by Leo at λ, not a pass criterion. The number 3 is Leo's. No leaf asserts a numeric bound.
- **G7:** walked every task against INV-1 to INV-15. No unresolved hit and no waiver. Recorded in the task texts: γ and δ declare restart dispositions (INV-15); the trust rebuild runs off the loop thread and reads through a raising call (INV-8, INV-11); a roster file that fails to parse is never read as "no hosts" (INV-11); ε names sampler metrics only through `METRIC_GROUPS` (INV-14) and renders a block with no host row as such (INV-13); β's exception list names an owning task per entry (INV-12); each OFF block is bounded by the next toggle gate's delayed milestone (INV-7).
