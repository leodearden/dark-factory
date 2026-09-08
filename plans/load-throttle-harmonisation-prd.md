# PRD — Harmonised load signal and dispatch throttling across dark-factory and reify

**Status:** active (authored 2026-09-08, load-tuning session with Leo; adversarially reviewed once) — for
`/prd` decompose.
**Slug:** `load-throttle-harmonisation`. **Path:** `plans/load-throttle-harmonisation-prd.md` (the
overlay's convention; the parent predates it and lives under `docs/prds/`).
**Type:** extension of the shipped `docs/prds/dispatch-admission-load-cap.md` (L3b, DA-D1..D9) plus a
companion amendment to reify's `docs/prds/cpu-load-admission-control.md` (C-A1's PSI source and its
single-tenant premise). Adopts tasks **3590** (runqueue axis + yaml-leak fix + re-land), **3592**
(record the runqueue ratio for calibration) and **4063** (PSI threshold range validation).
**Prerequisite (already filed, unclaimed since 2026-07-31):** **3394** per-project cgroup slice
topology (`df-<project_id>.slice`) — required only by the own-slice reading (D2) and by reify's
re-key (ρ2); every other leaf lands without it.

> **Code anchors** verified against main `a24b1fb665` (`2026-09-08`). Main moves fast — cite-by-symbol;
> re-locate at implementation time. Figures marked *(session)* are this session's own measurements
> (memory `d11e6525-f1d7-42c9-944f-9a25351a828b`); figures marked *(3590)* are from that task's record.

---

## 0. Goal and user-observable surface (G1)

**Goal.** Every load-keyed throttle on this shared 32-thread host reads the same load sample and uses
the same vocabulary; the host CPU arm becomes a signal that discriminates (runqueue ratio, not CPU
PSI `some`); a per-project "my own work is stalling" arm exists and is calibrated before it is
switched on; and reify's userspace throttles stop holding reify's tokens back for dark-factory's load.

**What an operator observes when this lands:**

1. `dark-factory-orchestrator.yaml` and reify's `dark-factory-orchestrator.yaml` carry the same
   `psi_admission:` block (§6.2) and `orchestrator check-config` reports zero unknown keys on both.
   The five idle orchestrators carry no block and are unaffected: every new arm is **off by code
   default** and only the two heavy projects switch the runqueue arm on.
2. A `dispatch_deferred` event names the arm that tripped from `{mem_full, runqueue_ratio,
   own_cpu_some, mem_some, io_some, cpu_some}` and carries the whole sample it was decided on
   (host PSI, runqueue ratio, own-slice pressure, which cgroup was read, per-component `read_ok`).
   `cpu_some_avg10` is off by default in both projects: it never fired on this host at its shipped
   85 (0 of 178 dark-factory deferrals in 14d *(session)*) and blips at reify's 70 (86% of its holds
   are single-tick *(session)*).
3. reify's verify gates and jobserver balancer read `df-reify.slice/cpu.pressure`, not
   `/proc/pressure/cpu`; with dark-factory saturating the host and reify idle, reify's task-pool
   reservoir reads 0 (today the balancer sat at its 8-token reservoir for 51+ minutes with nothing
   running that could use them *(session)*).
4. `data/load-samples.db` holds ≥30 days of 5-second samples of host PSI, runqueue ratio and
   per-project own pressure; `scripts/load-threshold-calibration.py` turns it into per-threshold
   hold fractions and a recommendation; two delayed deterministic gates escalate it to Leo: the host
   arm's after 14 d of samples, the own arm's after 14 d of slice samples.

**Consumers per mechanism (no orphans):**

| Mechanism | Consumer |
|---|---|
| `shared.psi.PsiSample` v2 (`runqueue_ratio`, `own_cpu_some10`, `own_cgroup`, per-component `read_ok`) + readers | the dispatch gate (`scheduler.py::_phase_psi_gate` via `saturated()`), the sampler, the `dispatch_deferred` payload |
| `PsiAdmissionConfig` v2 (new nullable arms, `cpu_some_avg10` nullable, range validators) | `saturated()` / `tripping_metric()`; operators via `reload_config` (green-tier, self-registering) |
| Harmonised yaml blocks (both heavy projects) | each orchestrator's own gate; the calibration gate's drift check |
| Per-project slice pressure (`df-<project_id>.slice/cpu.pressure`, from 3394) | the own-slice reader; reify's `cpu-admit.sh` and `jobserver-balancer.py` via `@own-slice` |
| Sampler deployment, new metrics, 30 d retention, install script | the calibration script and gate; operators (`sqlite3 data/load-samples.db`) |
| Calibration script + two delayed gates (ε1 host arm, ε2 own arm) | Leo (escalations carrying recommended thresholds, hold fractions against D11, and the two-block drift check) |
| reify `@own-slice` resolution in `cpu-admit.sh` / `jobserver-balancer.py` + unit placement | reify's verify gates and token reservoir |

## 1. Background and measured premise (G6)

*(session, 2026-09-07/08: PSI, runqueue, per-cgroup `cpu.stat`/`cpu.pressure`, per-process
`schedstat`; loadavg deliberately not used)*

- 32 threads at 99% busy; `procs_running` p50 136 / p90 380 / max 1080; ~1100 forks/s.
- Host `cpu-some avg10` stayed inside **49–72** for the whole saturated window and does not
  discriminate; the runqueue ratio does: 2.4× quiet vs 7.1× busy *(3590)*; 4.25× median saturated
  vs 1.1–1.9× after this session's fanout cuts *(session)*.
- Per-cgroup pressure at the same instants: host 62–65, `app.slice` 62–63, dark-factory unit 46–48,
  reify unit 6–10, reify verify scopes 50–54. A parent's pressure covers its whole subtree
  (`app.slice` ≥ every child), so a per-project slice's `cpu.pressure` is that project's own stall.
- **What own pressure means.** A cgroup's `cpu.pressure` says *my runnable work is waiting*, not
  *I caused the contention*: under fair share, a project whose verifies are running stalls whenever
  the host is contended, whoever contends it. That is the right input for bounding a project's
  **own parallelism** (more rustc inside a stalled share only makes each job slower) and for
  holding *new* dispatch behind work that is already waiting. It is the wrong input for deciding
  blame, which nothing here does. The cross-project subsidy this PRD removes is the case where a
  throttle holds while the project's own work is **not** stalled: today's balancer reservoir keyed
  on host PSI (reify unit 2–10 while the host read 21–64).
- Every reify throttle keys on the host file: balancer hold 50 / release 40, `verify.sh` task psi
  gate 50, compile gate 85, agent shim 50, `psi_admission` 70. Dark-factory's dispatch gate keys on
  the same file at 85. Whichever threshold is lowest subsidises the other project.
- Dispatch-gate holds are cheap and short: tick cadence ~150 s in both projects; reify CPU holds are
  86% single-tick (values 70–73), longest 5 ticks; dark-factory `mem_full` holds 82% single-tick
  *(session, from `runs.db` `dispatch_deferred` events cut at 2.5 ticks)*. A hold costs one tick of
  deferred *new* dispatch; ~60 agents cost ~1 core in total. **Hysteresis on the dispatch gate is
  therefore not adopted** (D4): the costly latching is reify's token reservoir, which already has
  hysteresis and the wrong signal.
- Task 3590's item-5 leak re-reproduced on current main *(session)*: a `psi_admission:` block in
  the live yaml fails 9 of 18 tests in three scheduler files (control 18/18). Nothing can be tuned
  via yaml until that is fixed; `reload_config` re-reads the same file.
- A live unit read *(reviewer, 2026-09-08)*: `orchestrator-dark-factory.service` `cpu.pressure`
  `some avg10=36.6 avg300=40.9` at moderate load, and `procs_running` 42–82 (ratio 1.3–2.6). So a
  40 own-arm threshold is **not** separated from ordinary load on the unit population, and the slice
  population (unit + agents + verify scopes) is a different one again. This is why D2 ships the own
  arm **off** and D9 leaves its value to the calibration gate.

## 2. Sketch of approach

1. **One sample.** `shared/src/shared/psi.py::PsiSample` grows defaulted fields (every existing
   construction is keyword-only, so nothing breaks): `runqueue_ratio` (`procs_running /
   len(os.sched_getaffinity(0))` from `/proc/stat`), `own_cpu_some10` (the `some avg10` of the
   nearest `df-<project_id>.slice` ancestor of the calling process, else the process's own cgroup),
   `own_cgroup` (the path actually read) and per-component `read_ok`. `saturated(cfg)` gains two
   OR'd arms and tolerates `None` thresholds (today it does a bare `>=` on every threshold).
2. **One vocabulary.** `PsiAdmissionConfig` gains `runqueue_ratio: float | None` and
   `own_cpu_some_avg10: float | None`, **both `None` (off) by code default**; `cpu_some_avg10`
   becomes `float | None` with default `None` (off, still configurable); every PSI threshold is
   validated to `[0, 100]`, the ratio to `> 0`, the floor to `≥ 1` (4063's fix, absorbed).
   Registration is automatic (`config.py::_submodel_leaf_paths('psi_admission', …)` enumerates
   `model_fields`; `test_config_psi_admission_reload.py::test_every_leaf_is_reloadable` is
   parametrised over them). The yaml blocks (γ, ρ1) are where the two heavy projects switch the
   runqueue arm on and raise the floor to 3.
3. **One topology, owned by 3394.** `df-<project_id>.slice` per project holding the orchestrator
   unit and its verify scopes (and, per 3394, its agent scopes). This PRD adds no dependency edge
   from 5205/5206/5208 to 3394: 5205's per-role `-p CPUWeight=` at the scope spawn is meaningful
   today among `app.slice` siblings and stays meaningful inside a slice; 3394 must **preserve** it
   when it adds `--slice=`, and must set `CPUWeight=100` on the slice at creation so systemd enables
   the cpu controller for the subtree (the value is the default; the presence of the property is
   what vivifies `cpu.weight` — probe on this host: a scope under `df-prd-probe.slice` had no
   `cpu.weight` until `set-property CPUWeight=` was applied). Two consequences 3394 owns: (a) slice
   weights, not scope weights, then set the **cross-project** split, so 5205's 43/43/14 arithmetic
   becomes intra-slice and 3394 (or its follow-up) re-derives the cross-project values; (b) `df.slice`
   is a sibling of `app.slice` under `user@1000.service` (systemd derives the parent from the name),
   so all factory work shares one weight against desktop and interactive scopes — consistent with
   Leo's 2026-08-03 ruling on 3394 that interactive work is privileged deliberately, but a policy
   point 3394 must state rather than inherit.
4. **Same signal for reify's userspace throttles.** `cpu-admit.sh` and `jobserver-balancer.py`
   already take their PSI file from env with a format-generic `some avg10=` parser that reads a
   cgroup `cpu.pressure` unchanged. They gain one sentinel value, `@own-slice` (§6.3). The two knobs
   `verify.sh` actually reads, `REIFY_PSI_GATE_PROC_PATH` and `REIFY_COMPILE_GATE_PROC_PATH`, are
   set in reify's `verify_env:` (verified reachable: `verify.py::_resolve_verify_env` →
   `_target_subprocess_env` overlays `extra` last → `verify.sh` sources `cpu-admit.sh`);
   `REIFY_CPU_ADMIT_PROC_PATH` is the direct-exec knob of the agent cargo shim, which is spawned with
   the orchestrator's `os.environ` and is inert today (reify 5908), so it is **not** set here.
   The balancer's `REIFY_JOBSERVER_PSI_PROC_PATH=@own-slice` and `Slice=df-reify.slice` go into the
   `reify-jobserver.service` heredoc in `setup-dev.sh`, and a reify deterministic deploy leaf
   re-renders, reloads and restarts that unit (its `PartOf=orchestrator-reify.service` and the canary
   timer are unchanged). Thresholds 50/40, 85 and 50 stay — they now measure reify's own stall.
5. **Calibration, not guesses.** The existing `sampler` package (undeployed; long-format
   `samples(ts, metric, value, window_mean, window_max)` store; 24 h retention; `sampler/metrics.py`
   already imports the PSI parser from `shared.psi`) is installed from the committed
   `dashboard/dark-factory-load-sampler.{service,timer}` templates by a committed install script
   (not by `setup-host.sh`, which re-renders every unit and is the known clobber hazard), records
   `runqueue_ratio` and own pressure for every `df-*.slice` (or orchestrator unit before 3394),
   retains 30 days, and `scripts/load-threshold-calibration.py` reports per-candidate-threshold hold
   fractions and the two yamls' **parsed** `psi_admission` drift. Two delayed deterministic gates
   escalate it: **ε1** for the host arm, +14 d after {sampler deployed, both blocks live} — the
   threshold that ships hot is reviewed on a clock nothing else can stall; **ε2** for the own arm,
   +14 d after {ε1, 3394}, so its recommendation is measured on the slice population it will govern.
   The script's `--commit` flag commits its report under `plans/` (the shape of
   `scripts/merge-pytest-n-ab-analysis.py`).
6. **The detector stays.** reify's `fleet-load-detector.sh` is single-axis (ratio ≥ 4.0; its PSI arm
   was dropped by reify 5985) and has no invoker in either repo. The gate's `runqueue_ratio` arm
   reads `/proc/stat` directly and does not consume it, so 3590's "delete the self-refuting census
   entry" clause does **not** fire (it is conditioned on dark-factory adding a `fleet_load_detector`
   config surface, which this PRD does not); reify 5908 marks the block KEEP and
   `reify/tests/infra/test_fleet_load_detector.sh` asserts it exists. Both stay.

## 3. Resolved design decisions

- **D1 — Runqueue ratio is the host CPU arm; `cpu_some_avg10` defaults to off.** Amends DA-D1's
  "primary signal: CPU `some avg10`" for CPU only; the memory-tighter-than-io ordering, the memory
  `full` hard trip and OR semantics are unchanged. Code default `None`; the heavy projects' yaml
  sets `runqueue_ratio: 4.0` (provisional, D9).
- **D2 — Own-project pressure is a second, per-project CPU arm, shipped off.** Read from the
  project slice once 3394 lands (before that, from the orchestrator's own unit cgroup, and
  `own_cgroup` says which). Semantics per §1: "my work is stalling". Switched on only by the
  calibration gate's recommendation (D9). Fail-open per component: an unreadable cgroup sets
  `own_read_ok=False`, the arm is inert, the scheduler's existing rate-limited "PSI unreadable"
  WARNING names the component, and the sampler records the failure so it is visible over time.
- **D3 — Equal floors: `min_inflight_floor: 3` in both heavy projects** (yaml); the five idle
  orchestrators keep the code default 1. DA-D3's per-orchestrator semantics are unchanged.
- **D4 — No hysteresis on the dispatch gate.** Per-tick evaluation stays (DA-D2). Basis §1.
- **D5 — Per-project slice, owned by 3394; name from `fused_memory.project_id`.** Slice name
  `df-<project_id>.slice` where `<project_id>` is `OrchestratorConfig.fused_memory.project_id`
  (`dark_factory`, `reify`; fused-memory derives ids with underscores), **never**
  `verify.py::_scope_tag_for` (a dash-sanitised basename plus hash: `df-dark-factory-…` would make
  systemd derive parents `df-dark-factory.slice` → `df-dark.slice`). Interactive agents stay outside
  project slices (Leo, 2026-08-03, on 3394).
- **D6 — Re-key reify's throttles; do not retire them yet.** Config-only, reversible, and it keeps
  reify's storm protection (its verify scopes stall ~50% under fair share). Retiring the balancer
  reservoir once slice weights exist is decided on the calibration data (§10 Q3).
- **D7 — Amend reify C-A1's source, keep its contract.** "host-portable %, no nproc-derived constant,
  work-conserving, never requeue on timeout" survive; only the *file* changes. The single-tenant
  premise of `cpu-load-admission-control.md` §2/§6 is amended to multi-tenant (ρ1 carries the text).
- **D8 — The sampler is the long-sample recorder.** Leo's choice over an event-store emitter. The
  oneshot+timer shape stays (warm `uv run` start ~0.1 s wall *(session)*).
- **D9 — Thresholds are provisional, dated, and only one is switched on.** `runqueue_ratio 4.0`
  (3590's ratio arm, matching reify's detector; 2.4× quiet vs 7.1× busy *(3590)*; 1.1–1.9× vs 4.25×
  *(session)*) goes into the two heavy yamls. `own_cpu_some_avg10` has **no basis on the population
  it will govern** and stays `None` until ε2 reports on ≥14 d of slice samples.
- **D10 — Reporting rank of `dispatch_deferred.metric`.** Today `scheduler.py::_note_heavy_deferral`
  ranks `cpu > mem_some > mem_full > io` (pinned by
  `test_scheduler_dispatch_admission.py::TestDispatchAdmissionMetricRanking`). Amended to
  `mem_full > runqueue_ratio > own_cpu_some > mem_some > io > cpu_some`: the swap-thrash arm first
  (it is the catastrophic one, DA-D1), then the live CPU arms, then the legacy arm. This is a
  DA-D1 reporting-rank amendment and the pinned test moves with it.
- **D11 — Intended steady state.** In the calibrated regime the gate holds on a **minority of
  ticks** (target ≤ 20%) and never sits at the floor for hours; ε1/ε2 report the
  actual hold fraction per candidate threshold, and a threshold whose hold fraction exceeds the
  target in ordinary load is not recommended. The host arm belongs only on projects that produce
  host load; the five idle projects never enable it.
- **D12 — Task adoption, not duplication.** 3590 → leaf β (re-scoped); 3592 → leaf δ (re-scoped;
  its dashboard half is out of scope, §9); 4063 → absorbed by β (its claimant heartbeat is stale
  since 2026-09-04; decompose cancels it with a pointer); 3394 is the prerequisite for D2/ρ2 only;
  5205/5206/5208 unchanged.

## 4. Pre-conditions for activating (G3 — verified or queued)

| Assumed capability | Evidence |
|---|---|
| `PsiSample` keyword-only construction everywhere | census: every construction repo-wide is keyword (`_orch_helpers.py::idle_psi_sample`, `shared/tests/test_psi.py`, the scheduler tests) |
| `saturated()` must learn `None` | today `shared/psi.py::PsiSample.saturated` does bare `>=` on four thresholds — **α changes it** |
| New `PsiAdmissionConfig` fields become green-tier automatically | `config.py::_submodel_leaf_paths('psi_admission', …)`; `test_config_psi_admission_reload.py::test_every_leaf_is_reloadable` |
| Default-85 pins | `test_config_psi_admission_reload.py` asserts `cpu_some_avg10 == 85.0` and uses 85.0 in reload fixtures; `shared/tests/test_psi.py` uses 85.0 in its truth table — **β / α update them** |
| Range-rejection mechanism | **does not exist today** (4063): β adds `ge/le` validation and the load + hot-reload rejection tests |
| Unknown submodel keys are ignored, not fatal | `PsiAdmissionConfig(runqueue_ratio=4.0)` accepted on main (probe); the unknown-key census surfaces them — so yaml blocks land **after** the schema (γ, ρ1 depend on β) |
| Own cgroup readable from inside the orchestrator | `/proc/<MainPID>/cgroup` → `…/app.slice/orchestrator-dark-factory.service`; its `cpu.pressure` readable (census) |
| Slice-level pressure aggregates descendants | measured: `app.slice` 62 ≥ children 46 / 6 / 54 |
| Scope under a named slice; slice `CPUWeight` settable | probe: `systemd-run --user --scope --slice=df-prd-probe.slice` → `df.slice/df-prd.slice/df-prd-probe.slice`; `set-property CPUWeight=250` read back; `cpu.weight` absent until a CPU property is set |
| `/proc/stat procs_running` | exists; the only in-repo reader is the diagnostic `scripts/cgroup-stall-ratio.py` — production reader is **new** (α) |
| Sampler package, store, unit templates | `sampler/src/sampler/{metrics,store,__main__}.py`; `dashboard/dark-factory-load-sampler.{service,timer}`; **not installed** by anything (δ + δ′) |
| reify PSI-path overrides reachable from verify | `verify.sh` reads `REIFY_PSI_GATE_PROC_PATH` / `REIFY_COMPILE_GATE_PROC_PATH`; `tests/infra/test_cpu_admit.sh` exercises the path override with a fixture file |
| `@own-slice` sentinel | **new** (ρ2, both scripts) |
| Slice topology | **queued**: task 3394 — prerequisite for D2's slice reading and ρ2 only |
| Cross-project deps | `docs/task-authoring.md` §3.2 `"reify:<id>"` / `"dark_factory:<id>"`; reify's project_id is `reify` (its yaml `fused_memory.project_id`) |
| The yaml-leak fix | 16 bare `OrchestratorConfig(` constructions across the three scheduler files (3 already pass `psi_admission=`); β passes an explicit `psi_admission=PsiAdmissionConfig()` at each (keeps the operational values those tests run under, which `conftest.py::code_default_config` would drop) |

No other novel substrate.

## 5. Cross-PRD relationship and seam owners (G4)

| Other PRD / task | Direction | Seam mechanism | Owner |
|---|---|---|---|
| `docs/prds/dispatch-admission-load-cap.md` (parent) | this extends | `PsiSample`, `PsiAdmissionConfig`, `_phase_psi_gate`, `dispatch_deferred` | this PRD (D1, D3, D10 amend DA-D1/DA-D3; β adds a dated pointer to the parent's header) |
| reify `docs/prds/cpu-load-admission-control.md` | this amends | C-A1's PSI source; single-tenant premise | ρ1/ρ2 carry the companion correction |
| task 3394 (slice topology) | this consumes | `df-<project_id>.slice`; `cpu.weight` vivified; preserve 5205's `-p CPUWeight` | 3394 (naming contract owner for both repos, its item 3). Its items 2 (agent placement) and 4 (govtest reaping) are not on this PRD's path; its 17 declared files exceed the overlay's review trigger — trim when it is picked up |
| tasks 5205 → 5206 → 5208 | sibling, no edge | scope weights; dark-factory scopes on; stall gate | as filed |
| reify 5908 (`cpu_governance` never ran) | sibling, blocked on 3394 | governed slices under `df-reify.slice` (C-G2) | reify 5908 |
| task 4063 (range validation) | absorbed | `PsiAdmissionConfig` validators | β |
| tasks 5203/5204 (merge `-n` A/B) | orthogonal | none | — |
| `plans/merge-lane-quality-prd.md` (5036) | none | β touches `scheduler.py`, `config.py`, `event_store.py` — none in its Appendix A | — |
| tasks 3593/3595 (pytest jobserver) | orthogonal | none | — |

## 6. Contract section (B+H)

### 6.1 `shared.psi.PsiSample` v2

```python
@dataclass(frozen=True)
class PsiSample:
    cpu_some10: float; mem_some10: float; mem_full10: float; io_some10: float; read_ok: bool
    runqueue_ratio: float = 0.0        # procs_running / len(os.sched_getaffinity(0))
    runqueue_read_ok: bool = False
    own_cpu_some10: float = 0.0        # `some avg10` of own_cgroup's cpu.pressure
    own_cgroup: str = ''               # cgroup path actually read ('' when none)
    own_read_ok: bool = False
    def saturated(self, cfg) -> bool
    def tripping_metric(self, cfg) -> str
```

- `saturated`: OR over arms. An arm never trips when its `cfg` threshold is `None` or its component
  `read_ok` is `False`. `read_ok=False` (host PSI unreadable) still makes the whole sample
  non-saturated (DA-D6), including the new arms.
- `tripping_metric`: precondition `saturated(cfg)` is true; returns the highest-ranked tripping
  arm per D10; raises if called on a non-saturated sample (the emitter only calls it after
  `saturated`; a violation is a programming error, not a fail-open case).
- `read_psi_sample(*, read=read_pressure, project_id: str | None = None)` — the `read` seam is
  unchanged; `project_id` selects the slice name; the own-cgroup resolver is separately injectable:
  `resolve_own_cgroup(project_id, *, proc_cgroup_path='/proc/self/cgroup', cgroup_root='/sys/fs/cgroup')`.
- Resolution: read the `0::` path from `/proc/self/cgroup`; walk from the leaf upward; the first
  segment equal to `df-<project_id>.slice` wins; else the leaf itself. Cached per process;
  re-resolved after any read failure. Any failure ⇒ `own_read_ok=False` with `own_cgroup` = the path
  attempted. `read_psi_sample` never raises; every component failure is carried in the result.

### 6.2 `PsiAdmissionConfig` v2

| Field | Type | Code default | Validation |
|---|---|---|---|
| `enabled` | bool | `True` | — |
| `cpu_some_avg10` | `float \| None` | **`None`** (was 85.0) | `[0, 100]` when set |
| `mem_some_avg10` | `float \| None` | 15.0 | `[0, 100]` |
| `mem_full_avg10` | `float \| None` | 3.0 | `[0, 100]` |
| `io_some_avg10` | `float \| None` | 40.0 | `[0, 100]` |
| `runqueue_ratio` | `float \| None` | **`None`** | `> 0` |
| `own_cpu_some_avg10` | `float \| None` | **`None`** | `[0, 100]` |
| `min_inflight_floor` | int | 1 | `≥ 1` |

Harmonised operator block, both heavy projects (γ / ρ1), identical text; the comment is a dated
pointer, not a restatement (INV-9):

```yaml
psi_admission:            # PRD plans/load-throttle-harmonisation-prd.md §6.2 (2026-09-08)
  mem_some_avg10: 15.0
  mem_full_avg10: 3.0
  io_some_avg10: 40.0
  runqueue_ratio: 4.0     # provisional (D9); the calibration gate re-derives it
  min_inflight_floor: 3
  # cpu_some_avg10 and own_cpu_some_avg10 deliberately absent: off (D1, D2)
```

### 6.3 Slice naming and `@own-slice`

- Name: `df-<project_id>.slice`, `<project_id>` = `fused_memory.project_id`. Owner 3394; both repos
  cite it, neither restates it.
- `@own-slice` (reify shell + python, and `shared.psi.resolve_own_cgroup` on the dark-factory side):
  resolve `/proc/self/cgroup` → nearest `df-*.slice` ancestor → `/sys/fs/cgroup<path>/cpu.pressure`.
  No ancestor ⇒ fall back to `/proc/pressure/cpu` and emit **one** WARNING per process naming the
  fallback (the balancer once at startup; `cpu-admit.sh` once per verify invocation, not per poll)
  — C-A4 fail-open preserved, INV-4 bounded.
- Two implementations exist by construction (bash in reify, python in `shared`). This table is the
  single home of the parity fixtures; each side's boundary test constructs exactly these strings and
  cites this section (INV-5 accepted with executed parity as the mirror; no waiver):

  | shape | `/proc/self/cgroup` `0::` path | sysfs state | expected read |
  |---|---|---|---|
  | under a project slice | `/user.slice/user-1000.slice/user@1000.service/df.slice/df-x.slice/df-verify-x-0123456789ab.scope` | `df-x.slice/cpu.pressure` present | the slice file |
  | not under one | `/user.slice/user-1000.slice/user@1000.service/app.slice/orchestrator-x.service` | that unit's `cpu.pressure` present | the unit file (python); `/proc/pressure/cpu` + one WARNING (bash `@own-slice`) |
  | unreadable | as row 1 | `df-x.slice/cpu.pressure` absent | `own_read_ok=False` (python); `/proc/pressure/cpu` + one WARNING (bash) |

### 6.4 Sampler rows and parity

Long-format `samples(ts, metric, value, window_mean, window_max)` gains metrics `runqueue_ratio`,
`own_cpu_some10:<cgroup-leaf-name>` (one row per `df-*.slice` present, else per
`orchestrator-*.service`) and `own_read_ok:<…>` (0/1). Retention `retain_seconds` becomes
configurable with a 30-day default. A parity test asserts every `PsiSample` field the gate can trip
on has a sampler metric of the same stem (INV-5: the gate and the recorder cannot disagree on
vocabulary). The calibration script compares the two yamls' `psi_admission` blocks as **parsed
mappings**, not text (INV-10).

## 7. Boundary-test sketch (B+H)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Reader under a project slice | fixture `/proc/self/cgroup` = `…/df.slice/df-x.slice/df-verify-x-….scope`; fixture sysfs with `cpu.pressure` at the slice | `own_cgroup` ends in `df-x.slice`, `own_read_ok=True`, value = the slice file's `some avg10` |
| 2 | Reader without a project slice | cgroup = `…/app.slice/orchestrator-x.service` | `own_cgroup` = that unit path, `own_read_ok=True` |
| 3 | Reader with unreadable slice file | slice exists, `cpu.pressure` missing | `own_read_ok=False`; the own arm cannot trip; host arms still evaluate |
| 4 | `None` thresholds never trip | `cpu_some10=99`, `cpu_some_avg10=None`; `runqueue_ratio=9`, cfg `runqueue_ratio=None` | `saturated()` False |
| 5 | Runqueue arm trips and ranks | `runqueue_ratio=4.3` with `runqueue_ratio=4.0`; `mem_some10` also over | `tripping_metric()=='runqueue_ratio'`; with `mem_full10` over too, `'mem_full_avg10'` |
| 6 | Gate transition idle → runqueue-saturated → idle (real scheduler, injected reader, real fixture cgroup tree) | burst of heavy + one deterministic task, in-flight ≥ floor | `dispatch_deferred` with `metric='runqueue_ratio'` and the full sample payload; deterministic still dispatches; full dispatch restored after |
| 7 | Own arm trips while host arms idle | `own_cpu_some10=55`, `own_cpu_some_avg10=50`, host values 0 | hold; `metric='own_cpu_some_avg10'`; payload `own_cgroup` set |
| 8 | Range rejection | `mem_full_avg10=101`, `runqueue_ratio=0` | rejected at construction and at `apply_reload` |
| 9 | Yaml block round-trip | both yamls' blocks | `orchestrator check-config` on each: zero unknown keys; `diff_config` classifies every leaf `applied` |
| 10 | Live yaml with a non-default block | block present in the repo yaml | the three scheduler test files pass (16 constructions carry explicit `psi_admission=`) |
| 11 | Sampler ↔ gate vocabulary | `PsiSample` trippable fields | each has a sampler metric of the same stem |
| 12 | Sampler deployed | install script run | `data/load-samples.db` gains `runqueue_ratio` rows every 5 s; timer active |
| 13 | reify `@own-slice` resolution | rows 1–3's shapes, shell side | `cpu-admit.sh` and the balancer read the slice file; fallback emits one WARNING and reads the host file |
| 14 | Balancer re-keyed | `REIFY_JOBSERVER_PSI_PROC_PATH=@own-slice`, host PSI ≥ 50 via fixture, own-slice file < 40 | `held_back` stays 0 |
| 15 | Balancer unit re-deployed | reify deploy leaf ran | `systemctl --user show reify-jobserver.service -p Slice,Environment` reads `df-reify.slice` and the sentinel; `/tmp/reify-jobserver-held-back` reads 0 while only dark-factory loads the host |
| 16 | Calibration cut | seeded `load-samples.db` + two yaml blocks | hold fractions per candidate threshold; drift check on parsed blocks; trailing JSON; exit 0 |

## 8. Decomposition plan (G2 signals; overlay sizing)

- **α — `PsiSample` v2 and readers** (shared). Defaulted fields, `read_runqueue_ratio`,
  `resolve_own_cgroup`, `read_own_cgroup_pressure`, `None`-tolerant `saturated`, `tripping_metric`
  with the D10 rank, extended `read_psi_sample`; update `shared/tests/test_psi.py`'s truth table.
  *Signal:* rows 1–5 green; on this host `read_psi_sample(project_id='dark_factory')` returns
  `runqueue_ratio > 0` and `own_read_ok=True` with `own_cgroup` naming the orchestrator's cgroup.
  *Files:* `shared/src/shared/psi.py`, `shared/tests/test_psi.py`. *Deps:* none. ~300–450 LOC.
- **β — gate, config vocabulary, payload, leak fix, boundary test** (orchestrator; **re-scopes
  3590; absorbs 4063; folds the integration gate**). `PsiAdmissionConfig` v2 with validators and
  the load + hot-reload rejection tests; `_phase_psi_gate` passes `project_id`;
  `_note_heavy_deferral` uses `tripping_metric()` and emits the full sample; the ranking test moves
  to D10; the 85.0 default pins updated; the 16 bare constructions in the three scheduler files gain
  explicit `psi_admission=PsiAdmissionConfig()`; a composed transition test (row 6, injected reader
  walking a fixture cgroup tree); OPERATIONS.md's `psi_admission` reference; a dated pointer in the
  parent PRD's header. *Signal:* rows 6–8 and 10 green; `test_every_leaf_is_reloadable` covers the
  new leaves. *Files:* `orchestrator/src/orchestrator/{config,scheduler,event_store}.py`,
  `orchestrator/tests/{test_scheduler_dispatch_admission,test_scheduler_psi_saturation_transition,
  test_scheduler_hermetic_psi,test_config_psi_admission_reload}.py`, `OPERATIONS.md`,
  `docs/prds/dispatch-admission-load-cap.md`. *Deps:* α. ~600–1,000 LOC.
- **γ — dark-factory yaml block** (`complexity='simple'`). §6.2 verbatim; prove row 10 in a
  throwaway worktree before committing. *Signal:* row 9 for dark-factory; the next
  `config_reload`/restart shows every leaf `applied`. *Files:* `dark-factory-orchestrator.yaml`.
  *Deps:* β.
- **δ — sampler metrics, retention, install script, calibration script** (**re-scopes 3592**,
  dashboard half dropped). Metrics per §6.4 via α's readers (no second parser); `retain_seconds`
  default 30 d; `scripts/install-load-sampler.sh` (copies the two committed units into
  `~/.config/systemd/user`, `daemon-reload`, `enable --now`, verifies a row lands; idempotent);
  `scripts/load-threshold-calibration.py` (per-signal percentiles, per-candidate-threshold hold
  fraction against the D11 target, parsed-block drift, trailing JSON; `--commit` writes
  `plans/load-threshold-calibration-<date>.md` and commits only that file; `--no-report` for
  interactive use). *Signal:* rows 11 and 16 green. *Files:* `sampler/src/sampler/{metrics,store,__main__,sampler}.py`,
  `sampler/tests/test_load_metrics.py`, `dashboard/dark-factory-load-sampler.{service,timer}`,
  `scripts/install-load-sampler.sh`, `scripts/load-threshold-calibration.py`,
  `scripts/tests/test_load_threshold_calibration.py`. *Deps:* α. ~500–900 LOC.
- **δ′ — sampler deploy** (`task_kind='deterministic'`, `before_done` = `scripts/
  install-load-sampler.sh`, `kind='deploy'`, `target_unit` null, `timeout_secs` 120). The units'
  `WorkingDirectory=%h/src/dark-factory` means the timer runs main's code, which is why δ′ follows δ.
  *Signal:* row 12. *Deps:* δ.
- **ε1 — host-arm calibration gate** (`task_kind='deterministic'`, `milestone: {mode: delayed,
  after_secs: 1209600}`, `before_done` = `scripts/load-threshold-calibration.py --commit --arm
  runqueue_ratio`, `always_escalates`). *Signal:* a born-at-L2 escalation to Leo carrying the
  recommended `runqueue_ratio`, its hold fraction against D11, and the drift check. *Deps:* δ′, γ,
  `reify:ρ1`. Bounded: fires on the clock regardless of 3394 (INV-7).
- **ε2 — own-arm calibration gate** (same shape, `--arm own_cpu_some_avg10`). *Signal:* an
  escalation recommending an `own_cpu_some_avg10` value (or "leave off") measured on ≥14 d of
  `df-*.slice` samples, with the hold fraction against D11. *Deps:* ε1, 3394.
- **3394 — slice topology** (existing, prerequisite for D2/ρ2). Amendment recorded on the task:
  set `CPUWeight=100` on `df-<project_id>.slice` at creation; preserve 5205's `-p CPUWeight`; name
  from `fused_memory.project_id`; state the `df.slice`-vs-`app.slice` split as a policy point; its
  items 2 and 4 may be split off. Priority raised to high (operator).

Reify batch (project `reify`; cross-project deps as `dark_factory:<id>`):

- **ρ1 — reify yaml: harmonised block + companion correction** (`simple`). §6.2 replaces
  `cpu_some_avg10: 70 / min_inflight_floor: 3`; `docs/prds/cpu-load-admission-control.md` gains a
  dated amendment block (C-A1 source; §2/§6 multi-tenant premise) pointing here. The
  `fleet_load_detector` block and its census entry are **not** touched (§2 item 6). *Signal:* row 9
  for reify. *Deps:* `dark_factory:β`.
- **ρ2 — `@own-slice` for reify's throttles.** `cpu-admit.sh` and `jobserver-balancer.py` accept the
  sentinel (§6.3, one-WARNING fallback); reify yaml `verify_env` sets `REIFY_PSI_GATE_PROC_PATH` and
  `REIFY_COMPILE_GATE_PROC_PATH` to `@own-slice`; the `reify-jobserver.service` heredoc in
  `setup-dev.sh` is extracted into one sourceable function (`render_jobserver_unit`) that
  `install_build_services` calls, and it gains `Slice=df-reify.slice` and
  `Environment=REIFY_JOBSERVER_PSI_PROC_PATH=@own-slice`; parity fixtures per §6.3's table in
  `tests/infra/test_cpu_admit.sh` and `tests/infra/test_jobserver_balancer.sh`.
  *Signal:* rows 13–14. *Files:* `scripts/cpu-admit.sh`, `scripts/jobserver-balancer.py`,
  `scripts/setup-dev.sh`, `dark-factory-orchestrator.yaml`, the two infra tests. *Deps:*
  `dark_factory:3394`. ~300–600 LOC.
- **ρ3 — balancer unit re-deploy** (reify `task_kind='deterministic'`, `before_done` = a committed
  `scripts/redeploy-jobserver-unit.sh` that sources `setup-dev.sh`'s `render_jobserver_unit` (one
  rendering site, INV-5), `daemon-reload`s and restarts `reify-jobserver.service`, then asserts the
  unit's `Slice` and `Environment`; `timeout_secs` 120). *Signal:* row 15. *Deps:* ρ2.

Edges: β→α; γ→β; δ→α; δ′→δ; ε1→{δ′, γ, reify:ρ1}; ε2→{ε1, 3394}; ρ1→dark_factory:β;
ρ2→dark_factory:3394; ρ3→ρ2. No new edges on 5205/5206/5208.

## 9. Out of scope

- Retiring the balancer reservoir or reify's verify psi gates (D6; §10 Q3); reify 5908.
- The `fleet_load_detector` block, script and census entry (§2 item 6).
- Interactive-terminal agents (3394's ruling); 3394's items 2 and 4.
- 3592's dashboard half (a panel over `load-samples.db`); the calibration script is the consumer.
- Slice `CPUWeight` *values* beyond the vivifying default (3394's exclusion) and the re-derivation of
  5205's cross-project split after slices exist (3394 or its follow-up).
- Memory/IO arm values (DA-D1); `max_concurrent_tasks`; the merge queue (DA-D8).

## 10. Open questions (tactical)

1. **Sampler metric naming for own pressure** — `own_cpu_some10:<leaf>` stem vs a `cgroup` column.
   The stem avoids a schema migration. Decide in δ.
2. **Calibration regime split** — percentiles + hold fractions alone (the floor), or additionally
   "saturated when a verify timeout occurred in the window". Decide in δ.
3. **Balancer reservoir retirement** once 3394 + 5205 exist and ε reports own-slice pressure rarely
   above 50 while tokens are held. Decide on ε2's data; file then.
4. **`min_inflight_floor` for the five idle orchestrators** — leave at 1 (proposed).
5. **Own-arm value after ε2** — ε2 recommends; γ/ρ1's successor commit switches it on. Decide then.
