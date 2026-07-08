# Capability manifest — `dispatch-admission-load-cap`

Mechanizes G3 + G6 per leaf for `docs/prds/dispatch-admission-load-cap.md` (L3b). Each binding ties a
task's asserted capability to **evidence** (grep/command/task-producer). Any **FAIL** binding blocks
queueing until resolved. Verified against dark-factory `main @ 961040ade7`, 2026-07-08.

**Domain notes.** This is a **scheduler control-flow** change: add a load-adaptive admission gate to the
dispatch decision. It asserts **no numeric premise that must be turned green** — the only numbers are
operator-tunable PSI thresholds (green-tier config, not RED-test bounds). Every task signal is
**structural** (a parser returns correct avg10; a reload disposition classifies a leaf `applied`; a
`dispatch_deferred` event is emitted / not emitted; dispatch counts stay bounded; the ≥1 floor holds).
The reify FEA numeric-floor and result-field-population sentinels are **N/A by construction**. The live
gates are: **G3 substrate-exists/wired** (the reader reuse, the scheduler insertion point, the config
plumbing, the event enum) and **G6 the anti-deadlock floor** (`min_inflight_floor ≥ 1 > 0`, an
absolute lower bound that must exceed the "0 = wedged" floor). There are **no cross-project deps**
(reify L0 landed `d2651f0d`; L3b consumes no reify primitive). DA1/DA2/DA3 are intermediates (each with
its own signal); **DA4 is the batch leaf** (the end-to-end integration gate, C-as-integration-gate).

---

## DA1 — PSI reader primitive (reuse via `shared`)

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3) — reuse, not new** | **PASS (on main)** | A tested pure `/proc/pressure` parser already exists: `grep:sampler/src/sampler/metrics.py:60` (`def parse_pressure_file`) + `:112` (`collect_psi`), avg10-only (`:57 _AVG10_RE`), returns `None` on garbage (= DA-D6 fail-open sentinel), handles the CPU-no-`full`-line asymmetry (`:82`). Pure stdlib (`re`, `pathlib`) — re-homeable. |
| **Reuse seam is importable both ways** | **PASS (on main)** | Both packages depend on the `dark-factory-shared` workspace: `grep:orchestrator/pyproject.toml:21` + `sampler/pyproject.toml:15` (`dark-factory-shared`). Re-homing the helper to `shared/src/shared/psi.py` is consumable by both; no backwards orchestrator→sampler-service dependency (DA-D9). |
| **"sampler tests stay green" is a real signal** | **PASS (on main)** | The re-home must not change sampler behavior — guarded by the existing suite: `grep:sampler/tests/test_load_metrics.py` (23 refs to `parse_pressure_file`/`collect_psi`). The signal is load-bearing, not synthetic. |
| **Fail-open sentinel (DA-D6)** | **PASS (self-produced)** | `read_psi_sample()` returns `read_ok=False` + `saturated()=False` on unreadable/garbage — produced by DA1's own deliverable; asserted by a malformed-fixture unit test. |
| Numeric floor / field-population | **N/A** | Parser returns opaque floats; asserts no numeric bound, populates no result field. |

## DA2 — `PsiAdmissionConfig` submodel + green-tier reload registration

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS (on main)** | Submodel-attach pattern: `grep:config.py:2202` (`fairness: FairnessConfig = Field(default_factory=…)`), the exact shape for `psi_admission: PsiAdmissionConfig`. Green-tier allowlist + generator: `grep:config.py:2621` (`RELOADABLE_FIELDS`) + `:2600` (`_submodel_leaf_paths`) — adding `_submodel_leaf_paths('psi_admission', PsiAdmissionConfig)` makes every threshold hot-reloadable. |
| **Reload disposition is observable** | **PASS (on main)** | `grep:config.py:2693` (`diff_config`) + `:2755` (`apply_reload`) bucket each differing leaf into `applied` vs `restart_required`. The signal ("a threshold edit classifies `applied`, not `restart_required`") is produced by this existing machinery. |
| **Test harness to mirror exists** | **PASS (on main)** | `grep:orchestrator/tests/test_config_reload_integration_gate.py` (the scenario-based reload gate test; its Scenario 3 already asserts `max_concurrent_tasks` is restart-only) — DA2's green-tier assertion mirrors it. |
| **Floor validation (DA-D3)** | **PASS (self-produced)** | `min_inflight_floor` validated `>= 1` at load via a pydantic field validator (DA2's own deliverable); a `floor < 1` config is rejected — asserted by a unit test. |
| Numeric floor / field-population | **N/A** | Config schema; thresholds are operator-tunable, not RED-test premises. |

## DA3 — dispatch-admission gate in `acquire_next` (the load-bearing seam)

| Check | Verdict | Evidence |
|---|---|---|
| **Insertion point exists + WIRED into production dispatch (G3, anti-orphan)** | **PASS (on main)** | The gate lives in the **production** dispatch path, not a test: `grep:scheduler.py:4059` (the `for _score, task_id, task, pri in scored:` loop that `try_acquire`s and `self._dispatched.add`s → returns a `TaskAssignment`), plus the pinned loop `grep:scheduler.py:~3985`. This is the sole dispatch chokepoint — wired by construction. |
| **In-flight counter exists (floor input)** | **PASS (on main)** | `grep:scheduler.py:1029` (`self._dispatched: set[str]`), `.add` at `:4062`, `.discard` at `:4576`. `len(self._dispatched)` is the per-orchestrator floor input (DA-D3). |
| **Deterministic-exemption predicate exists** | **PASS (on main)** | `grep:scheduler.py:1495` (`def is_deterministic`); deterministic tasks already hold no module lock (`:4805`), so `if psi_hold and not is_deterministic(task): continue` cleanly exempts them (DA-D4). |
| **Event type extensible + emit wired** | **PASS (on main)** | `grep:event_store.py:44` (`class EventType(StrEnum)`) — `dispatch_deferred` is a one-member add; emit call-site pattern proven at `grep:scheduler.py:4014` (`self.event_store.emit(EventType.lock_acquired, …)`). |
| **PSI reader capability (from DA1)** | **PASS (producer upstream)** | `shared.psi.read_psi_sample` / `saturated(cfg)` = `producer:DA1`, wired upstream via `DA3 → DA1`. DAG-direction correct (owner is upstream, not downstream). |
| **Config thresholds (from DA2)** | **PASS (producer upstream)** | `config.psi_admission.*` = `producer:DA2`, wired upstream via `DA3 → DA2`. |
| **Numeric floor (G6 anti-deadlock, DA-D3)** | **PASS** | `floor: min_inflight_floor ≥ 1 > 0` (the "0 in-flight = wedged" floor). The hold predicate `psi_hold = enabled AND saturated AND len(_dispatched) ≥ floor` makes "hold with 0 in flight" **unreachable** — the bound strictly exceeds the deadlock floor. Asserted by the floor test (saturated PSI + 0 in-flight ⇒ one heavy still dispatches). |
| **Two-way boundary (G5 B+H, self-produced)** | **PASS — DA3's own tests** | The gate's own scheduler unit tests prove **both directions**: throttle-under-pressure (saturated ⇒ heavy deferred, event emitted) AND full-dispatch-when-idle (idle ⇒ all heavy dispatch). Produced by DA3; the *end-to-end* version is DA4. |

## DA4 — end-to-end scheduler boundary test (the batch leaf, C-as-integration-gate)

| Check | Verdict | Evidence |
|---|---|---|
| **Substrate exists (G3)** | **PASS (producer upstream)** | Pure integration of the DA3 gate against the real `acquire_next` selection + lock table + event store; no new production module. Consumes DA3 = `producer:DA3`, wired upstream via `DA4 → DA3`. |
| **Injectable PSI seam for the transition** | **PASS (producer upstream)** | The saturation transition (idle→saturated→idle) is driven by injecting `read_psi_sample` — DA1 exposes it as an injectable reader (`producer:DA1`, transitively upstream via DA3→DA1). |
| **Executable integration (anti-tabulation)** | **PASS — this leaf's whole job** | The transition scenarios (bounded heavy concurrency + `dispatch_deferred` events while saturated; deterministic never deferred; full dispatch restored with no residual hold on recovery; ≥`min_inflight_floor` heavy throughout) are **run** against the live scheduler, not tabulated. Blocks the batch if any scenario fails. Binds the work-conserving + deadlock-free invariants (§6) executably. |
| Numeric floor / field-population | **N/A** | Integration assertions are structural (event stream present/absent, dispatch counts bounded, floor held). |

---

## Summary

| Task | Role | Blocking verdict |
|---|---|---|
| DA1 PSI reader (reuse via `shared`) | intermediate | **PASS** (reuse `parse_pressure_file` on main; re-home to `shared`; sampler tests `test_load_metrics.py` guard behavior) |
| DA2 `PsiAdmissionConfig` + reload reg | intermediate | **PASS** (submodel/allowlist/diff_config all on main; mirrors `test_config_reload_integration_gate.py`) |
| DA3 dispatch-admission gate | intermediate | **PASS** (insertion point `scheduler.py:4059` wired into production dispatch; `_dispatched`/`is_deterministic`/`EventType` on main; floor `≥1>0`; two-way test) |
| DA4 end-to-end boundary test | **leaf** | **PASS** (executable saturation-transition gate over the live scheduler; consumes DA3/DA1 upstream) |

**No FAIL bindings.** The batch is clear to queue. All substrate is on `main` today (the PSI parser to
reuse, the `acquire_next` scored loop, `self._dispatched`, `is_deterministic`, the `RELOADABLE_FIELDS`
allowlist + `_submodel_leaf_paths`/`diff_config`/`apply_reload` plumbing, the extensible `EventType`,
and the `test_config_reload_integration_gate.py` harness) or a named **upstream** producer in the
transitive dependency closure (DA1 reader → DA3 → DA4; DA2 config → DA3). **No cross-project deps.** The
one load-bearing G6 premise — the **anti-deadlock floor** — is bound as `min_inflight_floor ≥ 1 > 0`,
strictly above the "0 in-flight = wedged" floor, making a permanently-starved dispatch **unreachable by
construction** (DA-D3) and asserted executably by DA3's floor test and DA4's transition test. The G3
reuse finding (the PSI parser already exists in `sampler`, re-homed to `shared` rather than
reimplemented — DA-D9) removes the only "new reader" risk the narrative PRD originally carried.
