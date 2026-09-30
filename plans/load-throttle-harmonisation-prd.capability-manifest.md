# Capability manifest — load-throttle-harmonisation-prd

Mechanizes G3 (assumed-substrate verified) + G6 (premise validity) for the
harmonised-load-throttle batch (α, β, γ, δ, δ′, ε1, ε2 in `dark_factory`;
ρ1, ρ2, ρ3 in `reify`). One block per leaf; every capability the leaf's signal
asserts is bound to evidence. A binding resolving to a FAIL value
(`declared-only` / `test-only` / `producer-absent` / `producer-downstream` /
`producer-extent-short` / `bound≤floor` / `rejection-absent`) blocks the batch.

**All bindings PASS.** Five premises in the PRD prose were found stale, absent,
loosely sourced or incomplete during this walk and are corrected below
(§Corrections); each correction is carried into the filed task's own text, so no
leaf is dispatched against a fiction. **One G7 waiver** was recorded (β,
`storm-escape-required`); every other invariant hit resolved by redesign inside
the owning leaf's declared scope.

Corrections (1), (2) and (5) were rewritten or added after an adversarial
re-walk of the *filed* batch, before release. That pass also found and fixed
four `delivered_check`s that were unfalsifiable or self-defeating — most
sharply ε2's, whose `own_cpu_some_avg10` absent-check would have been broken
forever by γ's own required comment line, leaving ε2 unmarkable-done. Every
`grep` check in the sidecar was then executed against both repos: each
`expect: present` check currently fails and each `expect: absent` check on a
not-yet-delivered capability currently fails, so none is a false green. The
four that legitimately match today are must-survive guards (green-tier
registration, the sampler unit's `WorkingDirectory`, reify's
`PartOf=orchestrator-reify.service`, and the `fleet_load_detector` block that
must NOT be deleted).

Substrate verified live on dark-factory `main` `b6282847f9` and reify `main`
`840f87779c` (both 2026-09-08), by symbol. Every anchor below is
`path::symbol` or `path` — no line pins.

## dark_factory substrate

| Capability | Evidence |
|---|---|
| `PsiSample` — frozen dataclass, 5 fields (`cpu_some10`, `mem_some10`, `mem_full10`, `io_some10`, `read_ok`) | `shared/src/shared/psi.py::PsiSample` |
| `saturated()` does a bare `>=` on four thresholds (no `None` tolerance) | `shared/src/shared/psi.py::PsiSample.saturated` |
| `tripping_metric` | **absent repo-wide** — α builds it |
| `read_psi_sample(*, read=read_pressure)` — the `read` injection seam; no `project_id` today | `shared/src/shared/psi.py::read_psi_sample`, `::read_pressure` |
| every `PsiSample(` construction is keyword-only | census n=11: `shared/src/shared/psi.py::_FAIL_OPEN`, `::read_psi_sample`, `shared/tests/test_psi.py` (×7), `orchestrator/tests/{test_flake_discriminator,test_scheduler_dispatch_admission,_orch_helpers,test_scheduler_tick_phases,test_scheduler_hermetic_psi,test_scheduler_psi_saturation_transition}.py` |
| `PsiAdmissionConfig` — 6 fields; sole validator is `min_inflight_floor ge=1` | `orchestrator/src/orchestrator/config.py::PsiAdmissionConfig` |
| new leaves become green-tier automatically | `orchestrator/src/orchestrator/config.py::_submodel_leaf_paths` (`frozenset(f'{field_name}.{leaf}' for leaf in submodel_cls.model_fields)`), registered as `_submodel_leaf_paths('psi_admission', PsiAdmissionConfig)` in `RELOADABLE_FIELDS` |
| the reload test is parametrised over `model_fields` | `orchestrator/tests/test_config_psi_admission_reload.py::TestPsiAdmissionReloadDisposition.test_every_leaf_is_reloadable` |
| default-85 pins | 3 sites in `orchestrator/tests/test_config_psi_admission_reload.py` (`TestPsiAdmissionConfigDefaults.test_defaults_attached_on_orchestrator_config`, and two `PsiAdmissionConfig(cpu_some_avg10=85.0)` reload fixtures); 2 in `shared/tests/test_psi.py` |
| the dispatch gate + deferral emitter | `orchestrator/src/orchestrator/scheduler.py::Scheduler._phase_psi_gate`, `::Scheduler._note_heavy_deferral` |
| current reporting rank `cpu > mem_some > mem_full > io`, pinned | `_note_heavy_deferral`'s if/elif cascade; `orchestrator/tests/test_scheduler_dispatch_admission.py::TestDispatchAdmissionMetricRanking` |
| `dispatch_deferred` payload today: `{metric, value, in_flight, floor}` | `_note_heavy_deferral`'s `event_store.emit` block |
| yaml-leak construction census | 19 `OrchestratorConfig(` across `orchestrator/tests/{test_scheduler_dispatch_admission,test_scheduler_psi_saturation_transition,test_scheduler_hermetic_psi}.py`; 3 already pass `psi_admission=`; **16 bare** — matches the PRD §4 figure |
| sampler package + long-format store | `sampler/src/sampler/{metrics,store,__main__,sampler}.py`; `sampler/src/sampler/store.py` `CREATE TABLE IF NOT EXISTS samples (ts, metric, value, window_mean, window_max)` |
| retention today = 24 h, a keyword default, not config | `sampler/src/sampler/store.py::LoadSampleStore.cleanup_old` (`retain_seconds: int = 86400`), called with no override from `sampler/src/sampler/sampler.py` |
| the sampler already imports the shared parser (no second parser to build) | `sampler/src/sampler/metrics.py` — `from shared.psi import parse_pressure_file, read_pressure` |
| unit templates committed; **nothing installs them** | `dashboard/dark-factory-load-sampler.service`, `dashboard/dark-factory-load-sampler.timer`; repo-wide grep for `load-sampler` finds no installer |
| the `--commit` report shape to imitate | `scripts/merge-pytest-n-ab-analysis.py` — argparse `--commit` / `--no-report` / `--report-dir`, writing under `plans/` and landing it with `git commit --only` |
| `/proc/stat procs_running` — sole in-repo reader is diagnostic | `scripts/cgroup-stall-ratio.py`; production reader is new (α) |
| `psi_admission:` absent from the dark-factory yaml | grep over `dark-factory-orchestrator.yaml`: zero hits |
| an unknown submodel key is accepted and silently dropped | probe re-run 2026-09-08: `PsiAdmissionConfig(runqueue_ratio=4.0)` constructs; the model declares no `extra`, so pydantic's `ignore` applies. This is why γ/ρ1 depend on β — `orchestrator check-config` is an unknown-key linter and would flag the leaf before the schema exists |
| `check-config` / `diff_config` | `orchestrator/src/orchestrator/cli.py::check_config`; `orchestrator/src/orchestrator/config.py::diff_config` |
| `Milestone` (`mode` iff `at` / `after_secs`) | `shared/src/shared/task_metadata.py::Milestone`; `docs/task-authoring.md` §6 |
| deterministic deploy/gate presets | `docs/task-authoring.md` §5; worked examples tasks 5203 / 5204 / 5208 |
| `execution_class='operational'` DELETES `before_done` at submit | `fused-memory/src/fused_memory/middleware/task_interceptor.py::TaskInterceptor._inject_deterministic_pure_gate` — "DELETES any `before_done` key". δ′/ε1/ε2/ρ3 therefore carry `task_kind='deterministic'` only |
| loud stubs committed at the exact `before_done.script` paths | `scripts/install-load-sampler.sh`, `scripts/load-threshold-calibration.py` — mode `100755`, exit 64, commit `b6282847f9`. The dark-factory pre-commit is cheap for a script-only change (`pyright skipped (no Python changes)`), so both landed in seconds; reify's is not, which is why its stub was backed out instead |
| host premise for α's signal (live probe 2026-09-08) | orchestrator unit cgroup `…/app.slice/orchestrator-dark-factory.service`, its `cpu.pressure` readable (`some avg10=35.03`); `procs_running 101`, `len(os.sched_getaffinity(0))=32` ⇒ ratio 3.16 |

## reify substrate

| Capability | Evidence |
|---|---|
| `cpu-admit.sh` takes its PSI file from env with a format-generic `avg10=` parser | `scripts/cpu-admit.sh::cpu_admit_read_avg10` (awk over `$1 == want`, extracting `avg10=`), path from `REIFY_CPU_ADMIT_PROC_PATH` (default `/proc/pressure/cpu`) |
| `verify.sh`'s two gates are thin wrappers over `cpu-admit.sh` — one PSI reader, not two | `scripts/verify.sh::psi_gate` (`REIFY_PSI_GATE_PROC_PATH`, threshold 50), `::compile_gate` (`REIFY_COMPILE_GATE_PROC_PATH`, threshold 85) |
| balancer PSI knob + hold/release thresholds + reservoir | `scripts/jobserver-balancer.py` — `PSI_PROC_PATH` from `REIFY_JOBSERVER_PSI_PROC_PATH` (default `/proc/pressure/cpu`); `::read_pressure`; `::pressure_decide` (hold 50.0 / release 40.0, `release < hold` validated at import); `held_back` published via `::write_held_back` to `REIFY_JOBSERVER_HELD_BACK_FILE` (default `/tmp/reify-jobserver-held-back`) |
| the `reify-jobserver.service` heredoc, its `PartOf=`, and the canary | `scripts/setup-dev.sh::install_build_services` — heredoc carries `PartOf=orchestrator-reify.service`; `reify-jobserver-canary.{service,timer}` already exist and are enabled |
| `render_jobserver_unit` | **absent repo-wide** — ρ2 extracts it |
| `scripts/redeploy-jobserver-unit.sh` | **absent** — ρ2 writes it (declared file). A loud stub was committed at decompose so ρ3's `before_done` could validate at submit, then **backed out**: reify's pre-commit runs `verify.sh --include-infra`, which held `.git/index.lock` in the machine-operated checkout for 65 min on a saturated host, contending with the merge worker, for a 9-line stub. ρ3 depends on ρ2, so the script exists before ρ3 can dispatch |
| reify's current `psi_admission` block (what ρ1 replaces) | `dark-factory-orchestrator.yaml` — `cpu_some_avg10: 70.0`, `min_inflight_floor: 3` |
| reify `fused_memory.project_id` | `dark-factory-orchestrator.yaml` — `project_id: "reify"` |
| the detector and its census entry stay untouched | `scripts/fleet-load-detector.sh` (ratio arm only; the PSI arm was dropped by reify 5985), `tests/infra/test_fleet_load_detector.sh`, `cpu_governance.fleet_load_detector` block (`enabled: true`, `ratio_threshold: 4.0`) |
| the `@own-slice` fixture harness already exists on the shell side | `tests/infra/test_cpu_admit.sh` — `make_psi_fixture <avg10>` writes a `/proc/pressure/cpu`-shaped temp file and drives it through the PROC_PATH override, including a missing-file fail-open case |
| **the admission-knob parity guard** (see §Corrections (1)) | `tests/infra/test_verify_admission_knob_parity.sh` — section (B) asserts `verify_env` value == the owning script's single `${KNOB:-` default; section (C1) requires every `verify_env` key matching `^REIFY_(TEST_SEMAPHORE\|PSI_GATE)_` to carry a `KNOB_TABLE` row; (C2) is its vacuity guard |
| reify's design-invariant family for the G7 walk | `docs/legibility/design-invariants.md` — INV-SF-1 … INV-SF-7 |

---

## Corrections applied during this walk

**(1) ρ2 as written would turn reify's admission-knob parity guard red (G3, blocking).**
The PRD §8 ρ1/ρ2 rows say only that reify's `verify_env` "sets
`REIFY_PSI_GATE_PROC_PATH` and `REIFY_COMPILE_GATE_PROC_PATH` to `@own-slice`".
But `tests/infra/test_verify_admission_knob_parity.sh` (task 6393) enforces two
rules over that block: section (C1) matches every `verify_env` key against
`^REIFY_(TEST_SEMAPHORE|PSI_GATE)_` and fails on any without a `KNOB_TABLE`
row — `REIFY_PSI_GATE_PROC_PATH` matches, so adding it alone reds C1 — and
section (B) then requires the yaml value to **equal** the owning script's
single `${KNOB:-…}` default. (`REIFY_COMPILE_GATE_PROC_PATH` does not match
C1's regex, but leaving the two knobs asymmetric is worse, not better.) The
guard's rationale is exactly the hazard here: `verify_env` reaches only
orchestrator-spawned verifies, so a yaml value diverging from the script
default splits the fleet into two populations with different admission
behaviour on one host-global lock.

ρ2 must therefore do all three, not one: set both knobs to `@own-slice` in
`verify_env`; change **both** `${…:-…}` defaults in `scripts/verify.sh`
(`psi_gate`, `compile_gate`) from `/proc/pressure/cpu` to `@own-slice`; and add
a `KNOB_TABLE` row for each. This is safe precisely because §6.3's fallback is
value-preserving — no `df-*.slice` ancestor ⇒ read `/proc/pressure/cpu` and warn
once — so a hand-run `verify.sh` on a host without slices behaves exactly as it
does today. Binding held as PASS: the primitives ρ2 needs all exist; only the
declared file set and step list were short. `scripts/verify.sh` and
`tests/infra/test_verify_admission_knob_parity.sh` are added to ρ2's declared
files, and the three-part step is carried into ρ2's task text.

**(2) The single-tenant premise IS stated in `cpu-load-admission-control.md` —
just never in those words, so it must be amended in four named places (G6
branch 3).**
PRD D7 and the §8 ρ1 row instruct ρ1 to amend "the single-tenant premise of
`cpu-load-admission-control.md` §2/§6". A literal search is misleading:
`single-tenant`, `single-host`, `multi-tenant` and `multi-host` occur nowhere in
that 419-line document, and §6 is a G3 substrate table rather than a tenancy
claim. This manifest's first draft concluded from that that there was nothing to
amend, and told ρ1 not to edit prose. **That was wrong**, and the adversarial
re-walk caught it: the premise is carried implicitly, and four assertions are
false on this host —

1. the title, "…over **ALL load sources**";
2. §1's goal, to put "every significant CPU source on the build host … under a
   single work-conserving governance regime";
3. §2's source enumeration, which is closed and reify-only and attributes the
   whole measured host load (42–89 on 32 cores) to reify's own agents;
4. §3's "A lone source must reach all 32 cores."

The document's own repo already refutes them — reify's
`dark-factory-orchestrator.yaml` says "seven orchestrator services run here" and
`scripts/verify.sh` calls it a "busy multi-tenant box". So ρ1 **amends those
four**, rather than appending a block beside them and leaving four false
statements standing.

D7's substance is untouched: C-A1's *source* is what changes, and its contract
text ("host-portable %, no `nproc`-derived constant", work-conserving, never
requeue on timeout) survives verbatim. Carried into ρ1's task text.

**(3) "matching reify's detector" is not corroboration for `runqueue_ratio: 4.0`
— the two ratios are different quantities (G6 branch 1).**
PRD D9 backs the provisional 4.0 partly with "matching reify's detector".
`scripts/fleet-load-detector.sh` computes **`load1/nproc`** — a 1-minute
exponentially-decayed average that counts uninterruptible-sleep tasks — while
this PRD's `runqueue_ratio` is **`procs_running/len(os.sched_getaffinity(0))`**,
an instantaneous runnable count. The numeral 4.0 is shared; the calibration is
not. On this host right now the two are far apart: `procs_running 101/32 = 3.16`
while the same window's `load1` is several times higher.

The number's real basis is unaffected and stands on its own: 3590's own two
samples (`procs_running` med 76/32 = 2.4× quiet vs med 227/32 = 7.1× busy) and
this session's 1.1–1.9× vs 4.25×, both `procs_running`-derived, with loadavg
deliberately not used. D9 already ships 4.0 as provisional with ε1 re-deriving
it. Binding held as PASS; the correction is that γ's yaml comment and β's
docs must not cite the detector as corroboration. Carried into γ's and β's
task text.

**(4) 3590 carries a "MUST BE DONE AS PART OF THIS TASK" clause that this PRD
rules does not fire, plus one clause already delivered elsewhere.**
Task 3590's `details` contain a block headed "MUST BE DONE AS PART OF THIS TASK
— REMOVE THE NOW-SELF-REFUTING CENSUS ENTRY", requiring deletion of reify's
`config_key_census.ignore: ['cpu_governance.fleet_load_detector', …]` entry. Its
own stated antecedent is "if this task adds a `fleet_load_detector` config
surface on the dark-factory side". PRD §2 item 6 decides it does not: the
`runqueue_ratio` arm reads `/proc/stat` directly and consumes nothing reify
produces. The clause therefore **does not fire** and the entry stays. Separately,
3590 SCOPE item 3 ("retune or drop the detector's `avg10 >= 80` arm") was
already delivered by reify 5985, which dropped that arm; and item 4 ("decide
what happens to `cpu_some_avg10`") is settled by D1 — default `None`, still
configurable. All three are stated in the amendment appended to 3590.

**(5) ρ2 collides with three further reify guards, one of which reads green
while the config is unloadable (G3, blocking).**
Correction (1) named the admission-knob parity guard. The adversarial re-walk
found three more, all verified directly:

- **`tests/infra/test_host_global_unit_pinning.sh`.** It extracts
  `install_build_services()` with column-0 `sed` anchors, asserts the extracted
  snippet carries `^ExecStart=.*jobserver-balancer[.]py` (assert **B0c**), then
  `source`s only that snippet and calls the function. Moving the heredoc into
  `render_jobserver_unit` — the INV-5 move ρ2 exists to make — fails B0c
  immediately and the B1b/B2a run-time asserts after it. The guard's own comment
  anticipates this ("a future refactor of setup-dev.sh must fail this suite
  loudly"), so it is a tripwire to co-change, not a prohibition. The manifest's
  first draft bound `render_jobserver_unit` as "absent repo-wide … PASS" and
  missed the guard entirely.
- **`tests/infra/test_run_all_ambient_isolation.sh`.** It enforces **set
  equality** in both directions between the live `verify_env` keys and
  `tests/infra/run-all-ambient-vars.manifest` (task 5152: "equality, not mere
  subset"). Two new `verify_env` keys with no ledger rows are two failed
  asserts. So Correction (1)'s remedy is **four** parts, not three: the fourth
  is a ledger row per knob.
- **YAML quoting.** `@` is a reserved YAML indicator: `k: @own-slice` raises a
  `ScannerError`, so reify's orchestrator would fail to load its config
  entirely. The parity guard's own pure-awk parser accepts the unquoted form, so
  that guard reads **green while the config is unloadable** — the worst shape a
  check can have. ρ2 must write `"@own-slice"`, quoted, in `verify_env`.

ρ2's declared file set was widened accordingly (from 9 to 16), which crosses the
overlay's >15-file review trigger. That is recorded deliberately rather than
split: eleven of the sixteen are the guards and fixtures that must move with the
one behavioural change, and splitting them from it would land a red main.

---

## Per-leaf bindings

### α — `PsiSample` v2 and readers (`dark_factory`, new)

*Signal:* boundary rows 1–5 green; on this host
`read_psi_sample(project_id='dark_factory')` returns `runqueue_ratio > 0` and
`own_read_ok=True` with `own_cgroup` naming the orchestrator's cgroup.

| Capability | Binding | Verdict |
|---|---|---|
| defaulted fields are non-breaking | capability→producer (wired) — every `PsiSample(` construction repo-wide is keyword-only, census n=11 | PASS |
| `saturated()` learns `None` | producer:α — today a bare `>=` on four thresholds (`shared/src/shared/psi.py::PsiSample.saturated`); α owns the change | PASS |
| `tripping_metric` with the D10 rank | producer:α — absent repo-wide today; β (downstream of α) is the consumer | PASS |
| `read_psi_sample` gains `project_id` | producer:α — signature is `(*, read=read_pressure)` today; the `read` seam is unchanged | PASS |
| `procs_running` readable, affinity-denominated | grep — `/proc/stat` `procs_running`; sole in-repo reader is `scripts/cgroup-stall-ratio.py` (diagnostic). Live probe: 101 / 32 = 3.16 | PASS |
| own cgroup resolvable and its `cpu.pressure` readable | live probe 2026-09-08 — `/proc/self/cgroup` `0::` readable; the orchestrator unit's `cpu.pressure` reads `some avg10=35.03` | PASS |
| a slice's pressure aggregates its descendants | manual — measured `app.slice` 62 ≥ children 46 / 6 / 54 (PRD §1) | PASS |
| loop-thread cost is bounded (INV-8) | manual — three small procfs reads per gate tick (~150 s cadence); the cgroup resolution is cached per process and re-resolved only after a read failure | PASS |

### β — gate, config vocabulary, payload, leak fix, boundary test (`dark_factory`, **re-scopes 3590**; absorbs 4063)

*Signal:* boundary rows 6–8 and 10 green; `test_every_leaf_is_reloadable` covers
the new leaves.

| Capability | Binding | Verdict |
|---|---|---|
| `tripping_metric()` available to the emitter | producer:α upstream (real `add_dependency` edge β→α) | PASS |
| new leaves auto-register green-tier | grep — `config.py::_submodel_leaf_paths` enumerates `model_fields`; `test_every_leaf_is_reloadable` is parametrised over them | PASS |
| range rejection at load **and** at hot reload | producer:β — absent today (`PsiAdmissionConfig`'s four avg10 fields carry no `ge`/`le`); 4063 absorbed | PASS |
| the 85.0 default pins move with the default | grep — 3 sites in `test_config_psi_admission_reload.py`, 2 in `shared/tests/test_psi.py` (the latter α's) | PASS |
| the live-yaml leak is fixable without touching `conftest.py` | grep — 16 bare `OrchestratorConfig(` across the three scheduler test files (19 total, 3 already explicit) | PASS |
| the ranking test pins behaviour, not prose | grep — `TestDispatchAdmissionMetricRanking` asserts on `dispatch_deferred`'s emitted `data['metric']` | PASS |
| the deferral event can carry the whole sample | grep — `_note_heavy_deferral`'s `event_store.emit` already carries a `data` dict | PASS |
| 4063's fix has no orphaned dependents | manual — full 5178-task scan 2026-09-08: zero tasks list 4063 in `dependencies`; 4063 itself has none | PASS |

**G7 waiver — `storm-escape-required` (INV-4).** β extends the existing
per-sample fail-open (`read_ok=False` ⇒ non-saturated, a rate-limited scheduler
WARNING, no counter) to two further components without adding a streak
escalation. Waived because: the two new arms are **off by code default**, so an
unreadable component makes an arm inert rather than wrong; inertness degrades to
exactly today's shipped DA-D6 behaviour, not to a new risk; the WARNING names
the failing component (INV-2); and δ gives the rate a real counter —
`own_read_ok:<cgroup>` 0/1 rows in `load-samples.db` — with ε1/ε2 as the
supervised consumer on a bounded 14-day clock. A dedicated streak escalation
would page a human about a gate declining to throttle, which is the safe
direction.

### γ — dark-factory yaml block (`dark_factory`, new, `complexity='simple'`)

*Signal:* boundary row 9 for dark-factory; the next `config_reload` / restart
shows every leaf `applied`.

| Capability | Binding | Verdict |
|---|---|---|
| `check-config` reports zero unknown keys | grep — `orchestrator/src/orchestrator/cli.py::check_config` is an unknown-key linter that exits 1 on a genuinely unknown key | PASS |
| every leaf classifies `applied` | grep — `orchestrator/src/orchestrator/config.py::diff_config` | PASS |
| the block's leaves exist on the schema | producer:β upstream (edge γ→β); an unknown key is otherwise silently dropped (probe) | PASS |
| the block is landable at all | producer:β upstream — the yaml leak reds 9 of 18 tests today (re-reproduced on current main 2026-09-08) | PASS |
| the comment is a dated pointer, not a restatement (INV-9) | manual — §6.2 is the home; the comment cites the PRD path + date, and must not cite reify's detector as corroboration (§Corrections (3)) | PASS |

### δ — sampler metrics, retention, install script, calibration script (`dark_factory`, **re-scopes 3592**)

*Signal:* boundary rows 11 and 16 green.

| Capability | Binding | Verdict |
|---|---|---|
| the readers exist, with no second parser | producer:α upstream (edge δ→α); `sampler/src/sampler/metrics.py` already imports `shared.psi` | PASS |
| long-format store accepts new metric stems without migration | grep — `sampler/src/sampler/store.py` `samples (ts, metric, value, window_mean, window_max)` | PASS |
| retention is a knob to widen | producer:δ — today a keyword default `retain_seconds=86400` on `LoadSampleStore.cleanup_old`, called with no override | PASS |
| unit templates exist and nothing installs them | grep — `dashboard/dark-factory-load-sampler.{service,timer}` committed; repo-wide grep finds no installer (so δ's install script is not duplicating one) | PASS |
| the `--commit` report shape exists to imitate | grep — `scripts/merge-pytest-n-ab-analysis.py` `--commit` / `--no-report` / `--report-dir` + `git commit --only` | PASS |
| the two stub paths are δ's to replace | grep — `scripts/install-load-sampler.sh` and `scripts/load-threshold-calibration.py` are committed loud stubs (exit 64) naming 3592 | PASS |
| the drift check compares parsed mappings (INV-10) | manual — §6.4; a text diff of two yaml blocks would pin wording, not configuration | PASS |
| the sampler↔gate parity test derives from the live artifact (INV-10) | manual — the test enumerates `PsiSample`'s trippable fields rather than a hand-written list | PASS |

### δ′ — sampler deploy (`dark_factory`, new, `task_kind='deterministic'`)

*Signal:* boundary row 12 — `data/load-samples.db` gains `runqueue_ratio` rows
every 5 s; the timer is active.

| Capability | Binding | Verdict |
|---|---|---|
| `before_done.script` exists and is executable at submit | grep — `scripts/install-load-sampler.sh`, mode `100755`, commit `b6282847f9` | PASS |
| the stub can never run for real | producer:δ upstream (edge δ′→δ), and the stub exits 64 | PASS |
| deterministic auto-deploy preset | manual — `before_done` present + `always_escalates=false`; `target_unit: null` ⇒ blocking subprocess (`docs/task-authoring.md` §5) | PASS |
| the units run main's code, so δ must land first | manual — `WorkingDirectory=%h/src/dark-factory` in the committed unit | PASS |

### ε1 — host-arm calibration gate (`dark_factory`, new, `task_kind='deterministic'`)

*Signal:* a born-at-L2 escalation to Leo carrying the recommended
`runqueue_ratio`, its hold fraction against D11, and the drift check.

| Capability | Binding | Verdict |
|---|---|---|
| `before_done.script` exists and is executable at submit | grep — `scripts/load-threshold-calibration.py`, mode `100755` | PASS |
| `--commit --arm runqueue_ratio` is produced upstream | producer:δ in the transitive closure (ε1→δ′→δ) | PASS |
| ≥14 d of samples exist when it fires | producer:δ′ + γ + `reify:ρ1` upstream; `milestone {mode: delayed, after_secs: 1209600}` starts only once all deps are satisfied | PASS |
| act-then-ask preset reaches a human | manual — `before_done` present + `always_escalates=true` ⇒ run, then born-at-L2, `done` after `resume` | PASS |
| the gate is bounded and not held behind 3394 (INV-7) | manual — ε1 deliberately carries no 3394 edge; that is what the ε1/ε2 split buys | PASS |

### ε2 — own-arm calibration gate (`dark_factory`, new, `task_kind='deterministic'`)

*Signal:* an escalation recommending an `own_cpu_some_avg10` value (or "leave
off") measured on ≥14 d of `df-*.slice` samples, with the hold fraction against
D11.

| Capability | Binding | Verdict |
|---|---|---|
| `before_done.script` exists and is executable at submit | grep — `scripts/load-threshold-calibration.py`, mode `100755` | PASS |
| `--arm own_cpu_some_avg10` is produced upstream | producer:δ in the transitive closure (ε2→ε1→δ′→δ) | PASS |
| the slice population it measures exists | producer:3394 upstream (real edge); the 14-day clock starts only after 3394 is `done` | PASS |
| the own arm has no basis before this gate | manual — D9; a live unit read measured `some avg10=36.6 / avg300=40.9` at moderate load, so 40 is **not** separated from ordinary load. The arm ships `None` | PASS |

### ρ1 — reify yaml: harmonised block + companion correction (`reify`, new, `simple`)

*Signal:* boundary row 9 for reify.

| Capability | Binding | Verdict |
|---|---|---|
| the block's leaves exist on the schema | producer:`dark_factory:β` upstream (qualified external dep) | PASS |
| the block it replaces is present | grep — reify `dark-factory-orchestrator.yaml` `psi_admission: {cpu_some_avg10: 70.0, min_inflight_floor: 3}` | PASS |
| C-A1 exists to re-source | grep — `docs/prds/cpu-load-admission-control.md` C-A1: "Admits immediately while `avg10 < THRESHOLD` (`/proc/pressure/cpu`, host-portable %, no `nproc`-derived constant)" | PASS |
| the amendment adds rather than edits a premise | manual — §Corrections (2): the document states no single-tenant premise; ρ1 appends a dated amendment block | PASS |
| the detector block and its census entry are untouched | manual — §2 item 6 and §Corrections (4) | PASS |

### ρ2 — `@own-slice` for reify's throttles (`reify`, new)

*Signal:* boundary rows 13–14.

| Capability | Binding | Verdict |
|---|---|---|
| both scripts already take the PSI file from env with a cgroup-compatible parser | grep — `scripts/cpu-admit.sh::cpu_admit_read_avg10`; `scripts/jobserver-balancer.py::read_pressure` (scans for the `some` line, returns the float after `avg10=`) | PASS |
| the two knobs `verify.sh` reads | grep — `scripts/verify.sh::psi_gate` (`REIFY_PSI_GATE_PROC_PATH`), `::compile_gate` (`REIFY_COMPILE_GATE_PROC_PATH`) | PASS |
| the fixture harness for §6.3's parity rows exists | grep — `tests/infra/test_cpu_admit.sh::make_psi_fixture` drives a PROC_PATH override, including a missing-file case | PASS |
| `render_jobserver_unit` — one rendering site (INV-5) | producer:ρ2 — absent today; the heredoc lives inline in `scripts/setup-dev.sh::install_build_services` | PASS |
| the slice name to place the unit under | producer:`dark_factory:3394` upstream (qualified external dep); name from `fused_memory.project_id` = `reify` | PASS |
| the parity guard is satisfied, not tripped | producer:ρ2 — see §Corrections (1): both knobs in `verify_env`, **both** `verify.sh` defaults moved to `@own-slice`, and a `KNOB_TABLE` row per knob | PASS |
| the fallback is value-preserving, not a silent low reading (INV-11) | manual — no `df-*.slice` ancestor ⇒ read `/proc/pressure/cpu` and warn **once** per process; the balancer's existing `None` path is untouched | PASS |
| the `reify-jobserver.service` `PartOf=` and the canary timer survive | grep — both already present in `install_build_services`; ρ2 changes only `Slice=` and one `Environment=` | PASS |

### ρ3 — balancer unit re-deploy (`reify`, new, `task_kind='deterministic'`)

*Signal:* boundary row 15 — `systemctl --user show reify-jobserver.service -p
Slice,Environment` reads `df-reify.slice` and the sentinel;
`/tmp/reify-jobserver-held-back` reads 0 while only dark-factory loads the host.

| Capability | Binding | Verdict |
|---|---|---|
| `before_done.script` exists and is executable at submit | grep — `scripts/redeploy-jobserver-unit.sh`, a committed loud stub (exit 64) | PASS |
| the stub can never run for real | producer:ρ2 upstream (edge ρ3→ρ2) | PASS |
| one rendering site to source (INV-5) | producer:ρ2 upstream — `render_jobserver_unit` | PASS |
| it corroborates the live unit rather than the render (INV-3) | manual — asserts `systemctl show`'s actual `Slice` / `Environment` after the restart, and exits nonzero on mismatch (INV-SF-2) | PASS |
| `held_back` is observable | grep — `scripts/jobserver-balancer.py::write_held_back` → `/tmp/reify-jobserver-held-back` | PASS |

---

## G7 walk

dark-factory family (`docs/legibility/design-invariants.md`, INV-1…INV-11) for
α/β/γ/δ/δ′/ε1/ε2; reify's INV-SF-1…INV-SF-7 for ρ1/ρ2/ρ3, plus the df family
where the mechanism is shared across the seam. Walked per task, not only per
leaf.

Resolved by design, not waived:

- **INV-7 `holds-owned-and-bounded`** — ε was split into ε1 (host arm) and ε2
  (own arm) so the arm that ships hot is reviewed on a clock nothing else can
  stall; only ε2 carries the 3394 edge. Shape preserved from the authoring
  review.
- **INV-5 `no-lockstep-duplication`** — ρ3 sources `render_jobserver_unit` from
  `setup-dev.sh` rather than re-rendering the unit, so there is one rendering
  site. Shape preserved from the authoring review. §6.3's parity table is the
  single home for the two-implementation fixtures, and §6.4's parity test keeps
  the gate and the recorder from disagreeing on vocabulary.
- **INV-9 `one-fact-one-home`** — the `psi_admission` block necessarily exists
  in two yamls. The home is §6.2; both copies carry a dated pointer, and the
  named reconciliation mechanism is the calibration script's **parsed-block**
  drift check, run on ε1/ε2's clock.
- **INV-11 `no-silent-fail-soft`** — every component failure is carried in
  `PsiSample` (`runqueue_read_ok`, `own_read_ok`, `own_cgroup` = the path
  attempted); `read_psi_sample` never raises. On the reify side the
  `@own-slice` fallback returns a real host reading plus one WARNING, so a
  resolution failure never reads as "low pressure".
- **INV-10 `guards-exercise-behaviour`** — row 9 runs `check-config`; row 12
  runs the install script and asserts a row lands; row 15 inspects the live
  unit; the sampler↔gate parity test enumerates `PsiSample`'s fields rather
  than a transcribed list; the drift check parses rather than diffs text.
- **INV-3 `corroborate-before-acting`** — ρ3 asserts the deployed unit's own
  `Slice`/`Environment`, not the rendered text.
- **INV-SF-3 `declared-intent-consumed-or-diagnosed`** / **INV-SF-5
  `placeholders-owned-and-loud`** — `@own-slice` with no matching ancestor
  diagnoses rather than silently degrading; the committed stubs name their
  owning task and exit 64.

Waived: **β — `storm-escape-required`**, rationale in β's block above.

---

## Not in this batch

Task **3394** (per-project slice topology) is a **prerequisite**, not a batch
member: it is filed, unclaimed since 2026-07-31, and this decompose only
appends the PRD's amendment to it and raises its priority. It carries no
`prd_task_label` and is absent from the sidecar. Tasks **5205 / 5206 / 5208**
take no new edges (PRD §2 item 3, §5). Task **4063** is absorbed by β and
cancelled with a pointer; a full 5178-task scan found no dependents.
