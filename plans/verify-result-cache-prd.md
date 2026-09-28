# PRD — Verify-result cache: per-unit green replay, audited by the offline lane

**Status:** author-complete — 2026-09-19 (Leo + claude-interactive). Scope: DF
implementation plus the project-agnostic tree tier for every hosted project; the
Reify per-step tier is a separate reify PRD triggered by milestone ι against the
seam contract fixed here. **Type:** extension of shipped verify /
merge-lane machinery. **Approach:** B+H — the merge gate is the single path through
which code reaches `main`, and this PRD lets that gate replay evidence instead of
regenerating it.
**Code anchors** verified against main `6cb5f4a2b6` (2026-09-19). Main moves fast —
cite-by-symbol; re-locate at implementation time. Measurements below are dated
provenance of the studies they cite, not live counts.

## Goal

A verify replays a recorded green verdict for every **unit** whose inputs are
byte-identical to a run that already proved it green, and runs everything else.
Operator-observable when the batch lands:

- A merge gate on a tree whose nine modules all match earlier greens finishes in
  the time of the un-cacheable floor (git, plan, lint, one pyright per changed
  module, collection), and its `merge_verify` event lists every unit as
  `cache_hit` with the evidence it replayed.
- Merge-gate wall on DF falls by 5 to 6 h/day at the per-module tier and by a
  further 0.5 to 2 h/day at the per-test tier (against 13.0 h/day measured 08-11 to
  09-11); task-leg post-rebase re-verify demand (5.1 h/day, `rebase_verify_cost`)
  falls by 3 to 5 h/day at the per-test tier.
- Every landing records which cached entries it consumed; an offline-lane audit of
  each landed head runs the same command set uncached, and an audit red on a unit
  that was served from cache evicts the entries, files the fix-forward task through
  the lane's existing red path, and after two attributable leaks in seven days
  disables caching hot and raises an L2 blocker.
- Three days of log mode precede `on`: every would-hit is recorded and the unit is
  run anyway, so the first enable is made against a measured leak count.

## Background

- The 09-10 verify-speed study (`plans/verify-speed-study-df-2026-09-10.md` §B8,
  ratified 09-11 as do-not-build) priced a tree-hash verify cache at 0.31 h/day and
  rejected it on a 38% identical-tree disagreement. The 09-14 re-examination
  (memory `project_verify_cache_determinism_study_2026_09_14`) found: the 38% is a
  rate over re-runs conditioned on a red first run (12 of 15 pairs); all seven
  disagreements were spurious REDS, none a false green; a cache that stores greens is
  not exposed to any of them; the coarse ceiling counted repeats the cache could not
  serve (0.09 h/day realisable); and the per-module form was never computed. The
  gate runs its modules serially (`merge_verify_max_concurrent_modules: 1`,
  `verify.py::run_full_verification`), so skipped module seconds come off the wall.
- Measured per-module hit fractions against an earlier PASSED gate (641 gates):
  orchestrator 0.46, shared 0.75, sampler 0.75, escalation 0.46, cockpit 0.45,
  dashboard 0.41, tests_scripts 0.39, scripts 0.32, fused-memory 0.21.
- Measured false-skip of that key against every red gate in the window (session
  scratchpad `falseskip.py`, 09-16): 22 real escapes in 684 gates. 18 are tests that
  read outside their module (14 = one incident: `shared/tests/test_capability_manifest.py`
  reads `plans/*.capability-manifest.yaml`); 4 are known load flakes. Shared would
  escape 16 of its 22 reds — its key is unsound until the repo-wide guards leave it.
- Same tree ≠ same verification: the 26 sub-minute gate greens in that window were
  task 5063 (restart-rehydrated requests verified file-scoped; done 09-10). The
  `merge_verify` payload carries no plan, so a verdict's coverage is not recoverable
  from the event today.
- Existing special cases of this cache, each a separate mechanism: the task-leg tip
  checkpoint (`verify_checkpoint.py::green_checkpoint_at_tip`, 2752), the gate's
  disjoint-delta re-verify skip (`merge_gates.py::_reverify_rebased_tree`, ruled
  keep), the task-leg disjoint-rebase skip (task 5410, pending, log mode 14 d),
  Reify's `tree_oid` retry-narrowing sidecar (reify 5287) and infra content-skip.
- Task 1724 removed a tree-equality skip cascade (commit `5811bed48b`) after reify
  4502 landed red through it. Its adjudicated ground is scope-SUBSET substitution: a
  narrower task-leg plan standing in for the full gate. This PRD never narrows a
  plan; it replays a verdict recorded under an identical plan for identical inputs.
  That distinction needs Leo's explicit ruling (R-cache-1) because the multihost PRD's
  scope guard reads "never from narrowing or skipping a gate".

## Sketch of approach

### Units, keys, tiers

A **unit** is the smallest thing that gets its own verdict and its own key. Tiers
differ only in unit granularity; the registry, policy, audit and event shape are
shared.

| Tier | Unit | Key inputs | Unit source |
|---|---|---|---|
| tree | the whole verify | merge tree OID + spec hash + toolchain fp | orchestrator (project-agnostic) |
| module | one `ModuleConfig` × tool leg (test / lint / type) | own subtree OIDs + declared dep-closure subtree OIDs + root inputs + spec hash + toolchain fp | `VerifyPlan.runs` (DF) |
| test | one pytest node id | test file + conftest chain + root plugin + covered source files + observed repo reads + module tool spec + toolchain fp | coverage contexts + observer plugin (DF) |

- **Root inputs** (every DF key): `pyproject.toml`, `uv.lock`, `conftest.py`,
  `df_pytest_isolation.py`, `dark-factory-orchestrator.yaml`,
  `fused-memory/scripts/check_bare_magicmock_config.py`, the module's own
  `orchestrator.yaml` / `pyproject.toml`. Derived, not hand-listed: dep closures come
  from each workspace member's `pyproject.toml` (`dependencies` + `dependency-groups.dev`),
  root inputs from what the verify commands actually read (the lint helper is named
  in `lint_command`; the plugin is imported by every conftest).
- **Spec hash** = hash of the rendered `PlannedRun` (tool, uv_project, cwd_rel,
  base_flags, targets, env). A task-role FILE_SCOPED run and a merge-role FULL_SUITE
  run of the same module have different spec hashes and never serve each other —
  this is the machine-checked form of 1724's ground (INV-1).
- **Toolchain fingerprint** = interpreter version, `uv.lock` blob, pyright version
  (from `node_modules/.package-lock.json` or `package-lock.json`), ruff version,
  `verify_env` overlay. A remote runner without a registry always misses.
- Keys are git object ids (content hashes) computed from the tree at dispatch, never
  from a snapshot (INV-3).

### Registry (`orchestrator/src/orchestrator/verify_cache/`)

- `keys.py` — unit key derivation for the three tiers from a tree OID + plan; pure.
- `registry.py` — sqlite `data/orchestrator/verify_cache.db`: `entries(unit_id,
  key, tier, spec_hash, evidence_event_id, run_id, sha, recorded_at)`; `consumed(
  landing_sha, entry_id)`; `policy_state(disabled_until, reason, streak…)`. Greens
  only. Rebuildable from events + junit archives; deleting the file is a cold start,
  never an error.
- `policy.py` — mode `off | log | on` (config, green-tier), never-cache classes,
  admission rules, the streak kill switch. A degraded registry (locked, corrupt,
  missing) is a **miss with a named reason on the result**, never a silent run
  (INV-11).
- `audit.py` — consumes the offline lane's confirmed-red set for a head, joins it
  against `consumed` for that head's landing, and decides attributable / not.

### Admission — when a green may be recorded

Only a unit whose run finished `rc=0`, not timed out, not TRIVIAL, not produced under
`flake_suppression`, whose junit `tests=` equals its collected count (the truncation
hole 5492 closes at the gate; the recorder checks the same property), under mode
`log` or `on`. Units of a module flagged `verify_cache: never` (the repo-wide guard
module, § δ) are never recorded and never hit. Tests marked `repo_state` are never
recorded (§ per-test).

### Decision point

Both roles reach `verify.py::run_full_verification` / `run_scoped_verification`,
which iterate `VerifyPlan.runs` per module under the fan-out semaphore. Before a
module's runs execute, the policy is asked once per unit with the freshly computed
key: `hit` → synthesise the leg result from the registry entry (passed, with
`cache_hit` provenance: entry id, evidence event, recorded_at); `miss` → run and,
on green, record; `log` → run, record, and stamp `would_hit`. The unscoped
typecheck leg (`merge_queue.py::_run_unscoped_typechecks`, moving under 5036)
consults the executed-or-replayed plan and does not re-run pyright for a module the
scoped leg replayed or ran — this closes task 5294 (absorbed, § Decomposition θ).

### Events (INV-2)

`merge_verify` and `workflow_verify` gain: `plan_hash`, `cache_mode`, `units:
[{unit, tier, key, outcome: ran_green | ran_red | cache_hit | would_hit |
unavailable, evidence_event_id?}]`, and `workflow_verify` gains `duration_ms`
(today it carries none — the same gap that made the task-leg population
unmeasurable). The registry's `consumed` table is written from the merge-role
result at verdict time, not at landing, so nothing in 5036's Appendix A is touched.

### Per-test tier (DF)

- **Recording runs on the audit lane only**, never on the gate or the task leg.
  The audit executes every landed head uncached at idle nice, so it is the free
  place to pay the tracer: pytest-cov 7.1 / coverage 7.13 with `--cov-context=test`
  on the C tracer (verified 2026-09-19: the `sysmon` core leaves
  `should_start_context` / `switch_context` unimplemented, so dynamic contexts need
  `ctrace`, whose overhead the 09-10 study measured at roughly a third of suite
  wall) yields test → source-file sets; a small observer plugin (new module beside
  `df_pytest_isolation.py`, not inside it — that file is 2,414 lines) records
  repo-relative paths opened via `open` / `Path` / `os.open` during each test, and
  classifies `subprocess` calls: cwd under the test's tmp → ignored; cwd under the
  repo → the test is **unkeyable**. A test whose recording has not yet been
  produced by an audit run simply runs; recordings follow landings by one lane cycle.
- **Unkeyable inputs must be declared**: a test that reads outside the repo tree
  (home, `~/.config/systemd`, `data/`, the network) or runs a subprocess in the
  repo carries `@pytest.mark.repo_state`; the observer fails the test under
  `role='task'` when it observes such a read without the marker. Keyable inputs
  are observed, so keys self-heal on every miss; unkeyable ones are enforced, so
  they cannot leak (INV-10: the guard runs the behaviour).
- **Replay**: at collection, the plugin computes each item's key from the recorded
  input set, deselects hits, and writes a per-test verdict file that the verify
  layer merges with the fresh junit into the module verdict. A test with no
  recorded input set (new, renamed, or its recording expired) always runs.
- Task role and merge role share the plugin and the registry; spec hash keeps their
  entries apart.

### Audit (the slow lane)

- A `LaneCommand` entry `verify-cache-audit` on `git.offline_lane_commands` runs
  the merge-role command set uncached (`DF_VERIFY_CACHE=off` in its env) against
  every landed head; the lane already coalesces to the latest head and runs at idle
  nice (`OfflineLaneWorker._run_once`). Its red path is unchanged: confirm re-run →
  fix task → blocker after three confirmed-red advances.
- `audit.py` is called from the lane's red handler with the confirmed failing set
  and the head. Attributable = a failing unit (module or test) that appears in
  `consumed` for that head's landing with outcome `cache_hit`. Attributable leak →
  evict every entry with that unit's key, attach the leak record to the fix task,
  bump the streak. Streak ≥ 2 in 7 days → `policy_state.disabled_until` set (14 d,
  reason, evidence), every decision is a miss, and an L2 blocker is filed through
  the lane's `_file_blocker_escalation` pattern naming the two leaks. Re-enable is an
  operator action (config or `resolve_issue`), recorded.
- Non-attributable audit reds (main was red anyway) take the existing path, untouched.

### Modes and rollout

`off` (default in code) → `log` for three days on DF (Leo's ruling) → `on`. The
flip to `on` is made unless log mode recorded a `would_hit` unit that ran red for a
cause the branch's own diff explains (a leak that would have landed). Spurious reds
(infra kill, worker crash, deploy clock, known load flake) do not block the flip;
they are the classes the 09-14 study measured at ~4% of runs. After `on`, the
audit is the net.

## Resolved design decisions

1. **Greens only.** The registry never stores or serves a red. Every measured
   disagreement mechanism manufactures reds, so a green-only cache is not exposed
   to them; a red always runs.
2. **Replay, never narrow.** The plan is computed exactly as today; the cache only
   decides whether a planned unit executes or replays. Spec hash in every key.
3. **Per-module first, as the module leg of the per-test system.** Registry, policy,
   events, audit and kill switch are tier-agnostic and land once; the per-test
   plugin adds the fine key for the pytest leg only. Lint and type legs stay
   module-keyed permanently.
4. **Repo-wide guards leave their host modules** (δ) rather than widening
   `shared`'s key to the whole tree, which would forfeit its 0.75 hit fraction.
5. **Observed keyable inputs, declared unkeyable inputs.** Coverage + open
   observation for what can be hashed; a mandatory marker for what cannot.
   Observation runs on the audit lane, so the gate and the task leg never pay the
   tracer.
6. **Audit every landed head** (Leo, 09-19). CPU is idle-nice and off the critical
   path; detection latency is one lane cycle.
7. **Leak response** (Leo, 09-19): evict + fix task, hot-disable on a streak of two
   attributable leaks in seven days, and raise an L2 blocker whenever disabling.
8. **Log mode three days, then rely on the audit** (Leo, 09-19). 5410's 14-day log
   mode is its own; this PRD does not change it.
9. **Keep 5410; absorb 5294** (Leo, 09-19). 5410 lands sooner and is retired when
   the per-test tier makes its predicate a special case; 5294's dedupe is the
   unscoped-typecheck consult in γ, and 5294 closes as superseded.
10. **Consumption is recorded at verdict time on the verify result**, not at
    landing, so this PRD touches nothing in the merge-lane quality PRD's Appendix A
    and does not gate behind 5036 except through 4226 (audit red-path).
11. **Never cache** flake-suppressed greens, truncated sessions, TRIVIAL runs,
    `repo_state` tests, or the guard module; a remote runner without a registry
    misses.
12. **Registry is derived state** in its own sqlite file; deleting it is a cold
    start. Verdict provenance on the event points at the entry (INV-9: one home).

## Pre-conditions for activating

- **R-cache-1 (RULED, Leo, 2026-09-19):** a per-unit green replay under an
  identical spec hash and identical keyed inputs is evidence, not a skipped gate,
  and does not fall under 1724's scope-subset ground or the multihost PRD's "never
  from narrowing or skipping a gate". The plan is never narrowed; a replayed
  verdict carries the evidence event it replays; an entry under a different spec
  hash never serves. θ adds a dated pointer to this ruling beside the multihost
  PRD's scope guard and in `OPERATIONS.md`.
- Task **4226** (offline-lane red path misclassifies timeouts/crashes as failing
  tests; pending, depends on 5036) — the audit's attribution consumes the lane's
  confirmed failing set and must not inherit that misclassification. Blocks ε only;
  `log` mode and the per-module tier do not depend on it.
- Task **5492** (crash-truncated suppression) is not a dependency: admission rule
  (§ Admission) refuses truncated sessions independently, and 5492 closes the same
  hole on the un-cached path.
- Coverage tooling present in the workspace venv (verified: coverage 7.13.5,
  pytest-cov 7.1.0, xdist 3.8.0, CPython 3.13.9). No novel substrate otherwise.

## Cross-PRD relationship (G4)

| Other PRD / task | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/merge-lane-quality-prd.md` (5036 ζ2 relocation) | consumes | `_run_unscoped_typechecks` consult (γ); cite by symbol, land on the package if 5036 has landed | this PRD (γ) | queued |
| `plans/integration-test-lane-prd.md` (offline lane) | consumes | `LaneCommand` entry + red-handler hook into `audit.py` | this PRD (ε); lane engine untouched | queued |
| Task 4226 | consumes | confirmed-red set parse | 4226 | pending (blocks ε) |
| Task 5410 (task-leg disjoint-rebase skip) | parallel | both read `rebase_verify_cost`; per-test tier subsumes 5410's predicate | 5410 stays; retirement note in θ | wired |
| Task 5294 (duplicate pyright leg) | absorbed | executed-or-replayed plan consult | this PRD (γ) | close as superseded at γ landing |
| Task 5409 (NULL tip_sha) / 2752 checkpoint | parallel | tip checkpoint remains for task-leg redispatch; registry is a superset | 5409 stays | wired |
| `plans/flake-ledger-prd.md` / 5492 | consumes | `flake_suppression` on `VerifyResult` read by admission | flake ledger owns; this PRD reads | wired |
| `plans/verify-scope-inversion-prd.md` | consumes | `VerifyPlan.runs`, `merge_verify_breadth` unchanged | verify-plan owns | wired |
| `plans/merge-throughput-multihost-verify-prd.md` | constrained by | scope guard; remote runners miss | R-cache-1 | pending ruling |
| Reify per-step tier (`scripts/verify.sh` steps, reify 7424 stamps) | produces seam | § Reify seam contract: skip-set env + per-step verdict sidecar | **reify PRD**, triggered by milestone ι; this PRD owns the contract only | queued (ι) |

## Contract section (B+H)

Signatures are the seam an architect implements against; names may move, shapes do not.

```python
# verify_cache/keys.py
@dataclass(frozen=True)
class UnitKey:
    unit_id: str          # 'module:orchestrator:test' | 'test:<nodeid>' | 'tree'
    tier: Literal['tree', 'module', 'test']
    key: str              # sha256 over sorted (input_name, object_id) pairs + spec_hash + toolchain_fp
    inputs: tuple[tuple[str, str], ...]   # (path-or-name, git object id) — recorded for audit
def module_unit_keys(tree_oid: str, plan: VerifyPlan, toolchain: ToolchainFingerprint) -> list[UnitKey]
def test_unit_key(tree_oid: str, nodeid: str, recorded_inputs: RecordedInputs, run: PlannedRun, toolchain) -> UnitKey | None  # None = no recording → must run

# verify_cache/registry.py
class Registry:
    def lookup(self, key: UnitKey) -> Entry | None                 # greens only
    def record_green(self, key: UnitKey, evidence_event_id: int, run_id: str, sha: str) -> Entry
    def consume(self, landing_sha: str, entries: Iterable[Entry]) -> None
    def consumed_for(self, landing_sha: str) -> list[Entry]
    def evict_key(self, key: str, reason: str) -> int
    def policy_state(self) -> PolicyState                          # disabled_until, reason, streak

# verify_cache/policy.py
class Decision(StrEnum): HIT, MISS, WOULD_HIT, UNAVAILABLE, DISABLED
def decide(mode: Mode, key: UnitKey, registry: Registry | None, never_cache: bool) -> tuple[Decision, Entry | None, str]  # reason always set
def admissible(result: LegResult) -> tuple[bool, str]             # rc, timeout, trivial, suppression, truncation

# verify_cache/audit.py
def attribute(head: str, confirmed_failures: list[str], registry: Registry) -> AuditVerdict  # attributable entries, unattributable failures
def apply(verdict: AuditVerdict, registry: Registry, now: datetime) -> LeakResponse   # evictions, streak, disabled?, l2_required
```

Invariants:
- I1 A `HIT` carries `evidence_event_id`; the synthesised leg result is
  `passed=True`, `category='cache_hit'`, `cause_hint` names the entry. Nothing
  downstream can mistake it for a run (INV-11).
- I2 `decide` never raises; registry failure → `UNAVAILABLE` with reason; the unit
  runs.
- I3 `record_green` is idempotent on `(unit_id, key)`; a later green refreshes
  evidence, never duplicates.
- I4 Spec hash inequality ⇒ different key. Task-role and merge-role entries never
  serve each other.
- I5 `apply` is the only writer of `policy_state`; `disabled_until` always carries a
  reason and the two leak records; re-enable is an explicit operator write.
- I6 Ordering: consumption is written before the verdict is returned to the caller,
  so an audit can never find a landing without its consumption record.
- I7 Registry I/O never runs unbounded on the event-loop thread: lookups are one
  indexed read per unit, batched per verify, and executed via the same
  thread-offload the verify layer uses for git (INV-8).

## Gates walked (author mode, 2026-09-19)

- **G1** every mechanism has a consumer on main or in this batch: registry →
  γ's decision point → `merge_verify`/`workflow_verify` events (operator-visible,
  dashboard-readable); `audit.py` → offline-lane red handler → fix task and L2
  blocker (escalation queue); plugin → both pytest legs; ι → a reify escalation.
- **G3** no novel substrate: `VerifyPlan.runs` / `PlannedRun` and
  `VerifyResult.plan`, `VerifyResult.flake_suppression`, merge-role junit under
  `.df-verify-junit/`, the fan-out semaphore in `run_full_verification`,
  `LaneCommand` + `OfflineLaneWorker._handle_red_run` / `_file_blocker_escalation`,
  `metadata.milestone` (`delayed`) + cross-project external deps
  (`docs/task-authoring.md` §3.2, §6), pytest's `pytest_collection_modifyitems`,
  coverage 7.13.5 / pytest-cov 7.1.0 / xdist 3.8.0 in the workspace venv (CPython
  3.13.9) — all verified present. Per-test dynamic contexts require the C tracer
  (the `sysmon` core does not implement them), which is why recording lives on the
  audit lane.
- **G4** table above; no reciprocal ownership. The only contested seam (Reify
  per-step) is resolved: contract here, implementation in the reify PRD ι triggers.
- **G5** B+H (merge gate = load-bearing seam; blast radius: orchestrator verify
  layer, root pytest plugin, offline lane, config, escalation events).
- **G2/G6** every leaf names a mechanism-observable signal; h/day figures are
  projections and are labelled so; the one rejection assertion (row 17) is bound
  by η executing it.
- **G7** (advisory walk): `contracts-machine-checked` — mode and never-cache live
  in the config schema and the registry, spec hash in the key;
  `structured-facts-at-failure` — units and reasons on the events;
  `corroborate-before-acting` — keys computed from the dispatched tree;
  `storm-escape-required` — fix-task filing inherits the lane's dedup fingerprint
  and 3-red rule, eviction is idempotent, the streak cap ends in `DISABLED`;
  `no-lockstep-duplication` — one key derivation for both roles, 5294 closed by
  consult not by a second module set; `status-matches-liveness` /
  `holds-owned-and-bounded` — `disabled_until` carries owner, reason, deadline;
  `loop-thread-occupancy-bounded` — I7; `one-fact-one-home` — verdict home is the
  registry, events carry pointers, R-cache-1 is pointed to not restated;
  `guards-exercise-behaviour` — η and the observer execute; `no-silent-fail-soft`
  — `UNAVAILABLE` / `DISABLED` are result values. No waivers.
- **META** — no open design questions remain; tactical ones are listed below.

### Reify seam contract (owned here; implemented by the reify PRD that ι triggers)

The tree tier needs nothing from Reify: the orchestrator keys the whole verify on
the merge tree OID, the rendered spec and the toolchain fingerprint, and replays
or runs. The per-step tier is Reify code and is out of scope here; what this PRD
fixes is the seam, so neither side can drift (Leo, 2026-09-19, option 2):

- **Step verdict sidecar.** `scripts/verify.sh` reports, per plan step, `{step,
  started_at, ended_at, rc, skipped: bool}` in the per-run sidecar that reify task
  7424 introduces and that is archived on pass as well as fail. Step names are the
  eight plan steps 7424 stamps (preflight, gui vitest, clippy, release pre-builds,
  infra pool, debug nextest, release nextest, gui-feature nextest); the orchestrator
  treats the name string as the unit id and stores nothing it did not see reported.
- **Step inputs declaration.** One machine-readable table in Reify (the reify PRD
  decides its file) mapping each step to the repo paths it reads; the orchestrator
  derives each step's key from those subtree OIDs plus the tree-tier inputs. A step
  absent from the table is unkeyable and never cached.
- **Skip set.** The orchestrator passes `DF_VERIFY_SKIP_STEPS=<comma-separated step
  names>` in the verify environment; `verify.sh` honours it only when
  `DF_VERIFY_ROLE=merge` and marks each honoured step `skipped: true` in the
  sidecar with the entry id it was given. Any step not honoured runs, and the
  orchestrator records the run's verdict, not its own expectation.
- **Rule.** The orchestrator never emits a skip for a step whose verdict the
  sidecar did not report on the recording run, and never records a green for a
  step reported `skipped`. Log mode on Reify = the skip set is computed and
  recorded on the event but not passed.
- **Ownership.** This PRD's γ implements the generic side (env injection, sidecar
  read, per-step units for any project that reports steps); the reify PRD owns
  the table, the `verify.sh` change, and the infra tests that execute both.

## Boundary-test sketch (B+H — task η's signal)

Fixture: a two-module workspace repo (`alpha`, `beta`; `beta` depends on `alpha`),
a root plugin, a repo-wide guard test, and a driver that runs the real verify entry
points with an injected registry path.

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | first gate records | empty registry, mode `on`, green tree | all units `ran_green`; registry holds one entry per unit; event lists units |
| 2 | disjoint change replays | entry for `alpha`; change only `beta/` | `alpha` legs `cache_hit` with evidence id; `beta` `ran_green`; wall < run 1 |
| 3 | dep change misses | change `alpha/src` | `beta` misses (closure), `alpha` misses |
| 4 | root input change misses all | edit `uv.lock` | every unit misses |
| 5 | toolchain change misses all | pyright version bump in lockfile | every unit misses |
| 6 | spec hash separates roles | task-role FILE_SCOPED green for `alpha` | merge-role `alpha` misses |
| 7 | flake-suppressed green not recorded | green produced via `flake_suppression` | no entry |
| 8 | truncated session not recorded | junit `tests=` < collected | no entry; result carries reason |
| 9 | log mode runs anyway | mode `log`, entry present | unit `would_hit`, still executed, verdict from the run |
| 10 | registry unavailable | registry file locked | `UNAVAILABLE` reason on the unit, unit runs |
| 11 | guard module never hits | `verify_cache: never` | runs every time, never recorded |
| 12 | audit leak → evict + fix task | landing consumed entry E; audit red on E's unit | E evicted; fix task carries leak record; streak 1 |
| 13 | streak disables + L2 | second attributable leak within 7 d | `disabled_until` set; every decision `DISABLED`; blocker escalation filed |
| 14 | non-attributable audit red | audit red on a unit that `ran_green` | no eviction, existing red path only |
| 15 | per-test: covered-file change reruns only dependents | recordings for t1 (reads a.py), t2 (reads b.py); change b.py | t1 `cache_hit`, t2 runs |
| 16 | per-test: observed repo read keyed | t3 reads `plans/x.yaml`; change it | t3 runs |
| 17 | per-test: unkeyable read without marker fails | t4 runs `git` with cwd in repo, no marker, role task | t4 fails with the marker instruction |
| 18 | per-test: new test always runs | no recording for t5 | t5 runs, recording created |
| 19 | kill switch honoured everywhere | `disabled_until` in future | task and merge roles both miss |

## Decomposition plan

Labels are PRD-local; task ids are assigned at decompose. Sizing per the overlay's
bands (300 to 1,500 LOC, ≤ 12 files); the B+H integration gate η is the leaf that
proves the seam.

| # | Task | Modules | Signal | Prereqs |
|---|---|---|---|---|
| α | Verify spec hash + per-unit outcomes on `merge_verify` / `workflow_verify`; `duration_ms` on `workflow_verify` | orchestrator: `verify.py`, `verify_runner.py`, `workflow.py`, `event_store.py`, tests | intermediate → unlocks β, γ, ε; observable now: every verify event carries `plan_hash` and `units` with `ran_green/ran_red` | — |
| β | `verify_cache/` registry, keys, policy (modes, never-cache, kill state), toolchain fingerprint | orchestrator: new package + `config.py` keys + tests on a fixture repo | intermediate → unlocks γ, ε, ζ; observable: `orchestrator verify-cache status` CLI prints mode, entry count, policy state | α |
| γ | Per-module decision at the fan-out seam, both roles; hit synthesis with provenance; log mode; unscoped-typecheck consult (absorbs 5294) | orchestrator: `verify.py`, `merge_queue.py`/`merge_lane` (by symbol), tests | leaf: after 3 days of `log`, `merge_verify` events carry `would_hit` units; after `on`, a gate whose modules all hit completes under 10 min with units listed as `cache_hit` | α, β |
| δ | Repo-wide guard module: relocate whole-tree tests out of `shared`, `scripts`, `tests/scripts` into a `repo_guards` module flagged `verify_cache: never`; keep `TestWholeTreeGate`, manifest corpus, drift guards, whole-tree greps there | shared/, scripts/, tests/scripts/, root yaml, `orchestrator.yaml` per module | leaf: the 08-20 shape — a `plans/*.capability-manifest.yaml` change with `shared/` untouched — runs the manifest guard at the gate while `shared`'s pytest leg replays | β (flag) |
| ε | Audit: `verify-cache-audit` lane command, `audit.py` attribution, eviction, streak kill switch, L2 blocker | orchestrator: `offline_lane.py` hook, `verify_cache/audit.py`, yaml, tests | leaf: a seeded leak on the fixture yields eviction + a fix task carrying the leak record; a second within 7 d sets `disabled_until` and files the blocker | α, β, 4226 |
| ζ1 | Per-test recorder: coverage contexts + observer plugin + `repo_state` marker enforcement; per-test keys recorded in log mode | new plugin module beside `df_pytest_isolation.py`, `verify_cache/keys.py`, conftests, tests | intermediate → unlocks ζ2; observable: recordings exist for every executed test after one gate; an unmarked repo-state read fails at task role with the instruction | β, γ |
| ζ2 | Per-test replay: collection-time deselection of hits, per-test verdict merge into the module verdict, task-role wiring | plugin, `verify.py`, tests | leaf: a task-leg post-rebase re-verify whose main delta touches files no recorded test of the module reads replays every test and runs none, with the `workflow_verify` event listing them as `cache_hit`; expected effect (projection, not the signal): `rebase_verify_cost` wall/day falls by 3 to 5 h of 5.1 | ζ1 |
| η | Integration gate: the boundary-test sketch rows 1–19 executed end-to-end on the fixture repo | orchestrator/tests/ | leaf: all 19 rows green on main | γ, δ, ε, ζ2 |
| θ | Companion: `OPERATIONS.md` config reference (modes, audit, kill switch, re-enable), close 5294 as superseded, retirement note on 5410, memory | docs | leaf: docs-only commit; 5294 status | γ |
| ι | **Milestone (human gate, filed in reify):** author the Reify per-step verify-cache PRD against § Reify seam contract. `task_kind='deterministic'`, `always_escalates=True`, no `before_done`; `metadata.milestone = {mode: delayed, after_secs: 604800}` so it fires seven days after its deps are done, once the tree tier has audit data on Reify | reify task store only | leaf: a born-at-L2 escalation on reify naming this PRD's seam section and the Reify tree-tier audit figures to date; Leo authors the reify PRD via `/prd` | γ (external dep `dark_factory:<γ>`), reify 7424 |

Phase 1 = α, β. Phase 2 (vertical slice, user-observable) = γ, δ, ε, θ. Phase 3 =
ζ1, ζ2, η. ι is filed with the batch but lives in reify and fires on its own clock.

## Out of scope for this PRD

- Reify per-step tier implementation — a reify PRD, triggered by milestone ι,
  built against § Reify seam contract. (Reify tree-level replay is IN scope: it is
  the tree tier of γ and needs no Reify code.)
- Per-test caching for Rust (needs instrumented builds; capped by the 91% closure).
- Changing the plan: breadth, module set, `-n`, speculation, chains, trains.
- Moving the repo-wide guards' content or fixing their reds; δ moves files only.
- Dashboard surfaces for cache stats (events suffice; a later dashboard task reads them).
- WIP, admission slots, restart cadence (other programmes).

## Open questions (surfaced but not decided in this session)

1. **Registry file vs a table in `runs.db`.** Suggested: own file (derived,
   evictable, cold-start safe). Decide in β.
2. **Recording expiry.** Suggested: a test recording older than 30 days or older
   than the last toolchain change is treated as absent. Decide in ζ1.
3. **Deselect mechanism.** `pytest_collection_modifyitems` deselect vs an `-k`
   expression; the former is the only one that scales to thousands of ids. Decide
   in ζ2.
4. **Audit lane wall with the tracer on.** Recording adds roughly a third to the
   audit's own wall; measure in ζ1 and, if the lane falls behind landings, record
   on every second head rather than every head.
5. **Audit CPU budget.** Every landed head was chosen; if the lane's own queue
   starves the qdrant/warm-lane buckets, add a per-command min-interval in a
   follow-up (config schema change).
