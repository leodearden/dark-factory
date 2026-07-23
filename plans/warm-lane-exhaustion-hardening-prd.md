# Warm-lane exhaustion hardening — config honesty, lane-state coherence, cap semantics, valve default

**Status:** active — 2026-07-23. Leo-ratified design (discussion session 2026-07-23; decisions recorded in fused-memory `decisions_and_rationale`, agent `claude-interactive`).

## Goal

After this PRD lands, the failure composition behind the 2026-07-22 reify warm-lane
pool-exhaustion incident is structurally impossible to repeat silently:

- A config YAML key the orchestrator does not recognize files a **born-at-L2
  escalation at startup** naming the key (and, when the key shadows a nested-model
  field, naming the correct placement) — a phantom key can no longer sit inert for
  weeks. `orchestrator --check-config` gives the same verdict offline, exit-code'd.
- The warm-lane pool's **assignment state has one canonical store** (the durable
  `.lane-state` record) with one writer (`WarmLanePool`); mirror failure is loud
  (drift counter → deduped L2), and a digest-time cross-check alarms on divergence.
  The flock is liveness-only and labelled as such.
- **Pool exhaustion is an infra signal, not a per-task failure**: exhausted acquires
  requeue without burning the per-task requeue cap, and sustained exhaustion files
  exactly ONE deduped born-at-L2 carrying a pool census
  (size / free / assigned-dispatched / pinned-non-dispatched), instead of a
  per-task retry-cap + reblock-guard L2 storm.
- The **1933 reclaim-on-exhaustion valve defaults on** fleet-wide
  (`git.warm_lane_reclaim_on_exhaustion: True`), so structural pinning self-relieves
  by stealing the oldest non-dispatched non-terminal lane (WIP preserved on the
  branch ref) before exhaustion is ever reported.

## Background

Incident (2026-07-22, reify orchestrator): a re-pend burst hit a pool that was
structurally exhausted while looking healthy from every external view. Three
stacked causes, all verified against code 2026-07-23:

1. **Phantom spare** — reify's `dark-factory-orchestrator.yaml` declared
   `spare_warm_lanes: 8` at top level; the field lives on `GitConfig`
   (config.py `spare_warm_lanes`, read at harness pool sizing).
   `OrchestratorConfig` uses `extra='ignore'`, so the key silently dropped for
   3+ weeks; live pool was 48, not 56. `GitConfig` (plain `BaseModel`) also
   default-ignores extras, so the same class of typo *inside* `git:` drops too.
   DF task 2879 was investigated and closed on the false premise
   "effective_N=56 configured".
2. **Restart-recovery pinning** — recovery re-pins lanes to non-running tasks by
   identity (not status); only terminal-task lanes are ever auto-freed
   (`_reconcile_terminal_lanes`; extended to durable records + a stale census by
   task 2891). 40/48 lanes ASSIGNED at restart, ~4 circulating.
3. **Unwired valve** — the 1933 reclaim valve, built for exactly this, has been
   default-False since 2026-06-30 with no enablement owner. The config field's own
   description defers enablement on the strength of the phantom margin from (1).

Effects amplified by two DF-side semantics: `WarmLanePoolExhausted` counts against
the per-task requeue cap (default 3) regardless of cause, and the reblock-guard's
cumulative same-signature counter (threshold 3) converts the resulting retry-cap
escalations into a born-at-L2 per task — a day-long L2 storm from one infra
condition.

Companion SOP ratified alongside this PRD (recorded in fused-memory
`preferences_and_norms`): feature flags for new features generally default
**true**; a justified default-False knob must ship with a dated
enable-or-escalate predicate task.

Prior work this PRD extends: 1933 (valve), 2879 (`warm_lane_prewarm`, effective_N
materialization), 2891 (terminal-record reclaim + stale-lane digest census,
`plans/stranding-remediation-scheduler-ergonomics-prd.md` leaf γ), 2854
(reseed-verify guard protecting the shared reseed tail, incl. the steal path).

## Sketch of approach

Four workstreams, one package (`orchestrator/`), plus one MCP surface and one
cross-project consumer task:

- **W1 — config honesty (P1).** A recursive raw-YAML-vs-model walk at
  `load_config` time (pydantic discards extras before validation, so this is a
  separate pass over the parsed YAML tree against `model_fields`, descending
  nested submodels). Output: an unknown-key census `[(dotted_path, shadow_hint)]`
  where `shadow_hint` names any nested-model field with the same name (the
  `spare_warm_lanes` shape). Consumers: (a) Harness startup files a born-at-L2
  once the escalation queue is up, deduped by (project, key-set signature);
  (b) `reload_config` re-runs the walk and returns unknowns in its response;
  (c) a `--check-config` CLI mode prints the census and exits non-zero on any
  unknown key. **Always-on; the dedup signature is the storm escape** (one L2 per
  distinct key-set). Startup proceeds (fail-open, loud).
- **W2 — lane-state coherence (P2).** (a) A typed pool census
  (`WarmLanePoolCensus`: size, n_free, n_assigned_dispatched,
  n_pinned_non_dispatched, n_unknown_dispatch) assembled at the acquire
  chokepoint using the harness-installed dispatched-predicate; carried in the
  `WarmLanePoolExhausted` message, a WARNING log line, and a new escalation-server
  MCP tool (per-lane rows: lane, in-memory state, assigned task, durable-record
  state). (b) **Single-writer assignment store**: all assignment mutations
  (fresh acquire, release, reclaim re-key, note/restore) write through
  `WarmLanePool` → `LaneLifecycle` to the durable record; GitOps-side duplicate
  assignment writes removed (GitOps keeps non-assignment lifecycle transitions:
  seed/register/quarantine). Mirror failure: WARNING + drift counter; threshold →
  deduped born-at-L2. (c) A digest-time map↔record cross-check (extends 2891's
  census pass) with an alarm on persistent divergence.
- **W3 — cap semantics (P3).** `WarmLanePoolExhausted.counts_against_requeue_cap`
  flips to False (joining DISK_PRESSURE / HARD_DOWN / SOFT_PRESSURE); a
  consecutive-EXHAUSTED counter at the GitOps acquire chokepoint (reset on any
  successful fresh acquire) fires a harness-installed callback at threshold →
  ONE deduped born-at-L2 (`warm_lane_pool_structurally_exhausted`) carrying the
  W2 census. Closes both failure poles: no per-task cap burn, no silent infinite
  requeue.
- **W4 — valve default (P4).** `git.warm_lane_reclaim_on_exhaustion` Field default
  flips `False → True`. `defaults.yaml` does not list the knob, so the Field
  default is the single source. Fleet units adopt on their next (≤8h cadence)
  restart. No enablement predicate needed — default-True removes the
  dark-knob risk the SOP targets; no new default-False knobs are introduced by
  this PRD (W1 is always-on).
- **Reify-side consumer** — `scripts/warm-lane-audit.sh` relabelled to three
  honest columns: LIVE (flock probe, liveness only), ASSIGNED (durable record),
  PINNED (assigned ∧ ¬live — the state that hid the incident); filed as a reify
  task cross-project-dep'd on the MCP surface.

## Resolved design decisions

1. **L2 (not WARNING, not load-failure) for unknown keys; always-on.** Fail-open
   keeps a mis-keyed fleet unit serving; the L2 guarantees a human sees it. Dedup
   by key-set signature = storm escape; a changed key set re-files. (Leo,
   2026-07-23.)
2. **Shadow detection is a hint, not a distinct severity** — every unknown key
   L2s; keys matching a nested-model field get the placement hint in the detail.
3. **Two meanings, not three**: assignment (canonical = durable record, cached in
   RAM, single writer = pool) vs liveness (flock; the only signal that
   auto-releases on process death — kept, relabelled). The audit script's defect
   was presenting liveness under assignment's labels.
4. **Collapse-lite, not full collapse**: RAM map stays as the hot-path cache
   (single-loop coherence is trivial under a single writer); the record is
   canonical for recovery/census/audit. Full read-through-disk rejected (hot-path
   cost, no correctness gain).
5. **EXHAUSTED is not per-task backpressure.** The original "genuine backpressure
   and counts" rationale assumed exhaustion ⇒ demand > capacity; the incident
   shows it can mean capacity leak, and per-task caps punish the wrong party.
   Pool-level loudness replaces per-task cap burn. `WarmLaneReseedContaminated`
   keeps `counts=True` (data-integrity, correctly per-task).
6. **Valve default True.** Investigation found no surviving reason for False
   (blanket rollout convention + the phantom-margin belief verbatim in the field
   description); 2854's reseed-verify guard now protects the steal path's shared
   reseed tail. Residual accepted risk: untracked victim files are not preserved
   (tracked WIP is committed to the branch); steal-churn under sustained
   over-demand is WARNING-logged per steal.
7. **Census assembled once, consumed four ways** (exception message, WARNING log,
   MCP tool, structural-exhaustion L2) — no lock-step duplication.
8. **Counter placement at the GitOps acquire chokepoint** (not scheduler/workflow):
   exhaustion is observed exactly where it occurs; the harness installs the
   escalation callback (mirrors the 1933 candidate-provider / `_on_pool_storage_absent`
   install-in-harness pattern).

## Pre-conditions for activating

All substrate verified on main 2026-07-23 (see capability manifest beside this
PRD):

- pydantic v2 `model_fields` walk + `YamlSettingsSource` raw tree — exist
  (config.py).
- Born-at-L2 filer + `find_pending_l2_by_root_cause` dedup — exist
  (harness `_file_reblock_guard_l2` pattern).
- Escalation-server MCP tool registration — exists (`get_task_runtime_state` et al.).
- Durable-record reader seam — exists (2891's
  `_assigned_durable_records_with_statuses`, on main).
- Callback install-in-harness pattern — exists (1933 wiring).
- Not blocking, but noted: W1 will (correctly) fire on reify's still-present
  top-level `spare_warm_lanes` until reify's config-fix task lands.

## Cross-PRD relationship

| Other PRD / project | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/stranding-remediation-scheduler-ergonomics-prd.md` (leaf γ = 2891, landed) | extends | digest census pass / record-reader seam | this-prd (W2c extends the landed pass) | wired |
| reify `scripts/warm-lane-audit.sh` (+ runbook) | produces (MCP census surface) | escalation-server warm-lane status tool | this-prd owns the surface (β); reify task θ owns the consumer relabel | queued (θ, cross-project dep on β) |
| reify config-fix task (move `spare_warm_lanes` under `git:`; prewarm decision) | independent / interacts | W1's L2 fires on the phantom key until it lands | reify-side, pre-existing | external |
| `plans/config-hot-reload-prd.md` | consumes | `reload_config` response carries unknown-key census | this-prd | wired at ζ |

## Contract — assignment-store seam (B+H-lite, W2b)

**Writer contract.** `WarmLanePool` is the sole writer of durable **assignment**
state: on fresh acquire / `reclaim_victim` re-key / `note_assignment` /
`restore_assignment` it writes `task_id` + ASSIGNED; on `release` it writes the
release transition. `GitOps` retains all **non-assignment** lifecycle
transitions (SEED / REGISTERED / QUARANTINED, reseed bookkeeping) and stops
writing assignment fields directly.

**Invariants.**
- **I1 (write-through, loud):** when any pool assignment mutation returns, the
  lane's durable record reflects it, OR a WARNING was logged and the drift
  counter incremented — never a silent skip.
- **I2 (sole writer, grep-checkable):** zero call sites outside
  `warm_lane_pool.py` invoke `LaneLifecycle.note_assigned` (or otherwise write
  assignment fields).
- **I3 (fail-open dispatch):** a durable-write failure never fails or blocks the
  acquire/release itself; degradation is cache-only + loud.
- **I4 (bounded loudness):** drift counter ≥ threshold → one born-at-L2, deduped
  by root cause; counter resets on resolution.
- **I5 (recovery equivalence):** the restart adoption path rebuilds the RAM map
  from records such that map ≡ records for every adopted lane.

**Boundary-test sketch** (integration-gate observable for γ/δ):

| Scenario | Preconditions | Postconditions |
|---|---|---|
| Fresh acquire writes through | FREE lane, record RELEASED | record ASSIGNED + task_id set, map entry present |
| Release writes through | ASSIGNED lane | record released, map entry gone |
| Reclaim steal re-keys | valve on, victim non-dispatched | record task_id = thief; victim's WIP committed on its branch |
| Durable write fails | `.lane-state` unwritable | acquire still succeeds; WARNING + drift counter; L2 at threshold, exactly one pending |
| Restart adoption | records ASSIGNED for tasks T1..Tn | map rebuilt ≡ records (I5) |
| Seeded divergence | record manually desynced from map | digest divergence line + alarm; clean run reports none |

## Decomposition plan

Intra-batch deps by letter; θ files into the **reify** project with a
cross-project dep. Leaves: β, δ, ε, ζ, η, θ. Intermediates: α (unlocks β, ε),
γ (unlocks δ).

- **α — Typed warm-lane pool census, carried on the exhaustion path**
  (`warm_lane_pool.py`, `git_ops.py`). Intermediate → β, ε. Observable: under
  forced exhaustion, the `WarmLanePoolExhausted` message and a WARNING log line
  carry `size/n_free/n_assigned_dispatched/n_pinned_non_dispatched`
  (predicate-unwired construction sites degrade to `n_unknown_dispatch`).
- **β — Escalation-server MCP tool: warm-lane status** (escalation server,
  harness wiring). Leaf; deps: α. Observable: an operator MCP call returns the
  census + per-lane rows (lane, in-memory state, assigned task, durable-record
  state) matching a concurrently-inspected pool.
- **γ — Single-writer assignment store with loud drift** (`warm_lane_pool.py`,
  `git_ops.py`, `lane_lifecycle.py`). Intermediate → δ; contract above.
  Observable: boundary rows 1–4 — incl. unwritable `.lane-state` ⇒ acquire
  succeeds + WARNING + drift counter + exactly one deduped L2 at threshold.
- **δ — Digest-time map↔record divergence cross-check** (`harness.py`,
  `digest.py`). Leaf; deps: γ. Observable: seeded divergence produces a digest
  divergence section + alarm on persistence; clean run renders `_none_`.
- **ε — EXHAUSTED requeue-cap flip + structural-exhaustion L2**
  (`workflow_types.py`, `git_ops.py`, `harness.py`). Leaf; deps: α. Observable:
  N tasks requeuing through forced structural exhaustion leave per-task genuine
  requeue counts at 0 (no retry-cap escalations, no reblock-guard arming) and
  exactly ONE pending born-at-L2 `warm_lane_pool_structurally_exhausted`
  carrying the census; counter resets on a successful fresh acquire.
- **ζ — Unknown-config-key census: startup L2 + reload surfacing +
  `--check-config`** (`config.py`, `harness.py`, CLI). Leaf; independent.
  Observable: a config carrying a bogus top-level key (and a shadow key like
  top-level `spare_warm_lanes`) → startup files one deduped L2 naming both, the
  shadow with a placement hint; `--check-config` exits 1 listing them; clean
  config exits 0, no L2; `reload_config` response includes the census.
- **η — Valve default flip** (`config.py`, tests). Leaf; independent.
  Observable: stock `OrchestratorConfig().git.warm_lane_reclaim_on_exhaustion`
  is True; valve-off tests set False explicitly; under exhaustion with an
  eligible victim the steal WARNING fires instead of EXHAUSTED.
- **θ (reify) — Audit script honest columns** (reify `scripts/warm-lane-audit.sh`,
  runbook doc). Leaf; deps: `dark_factory:<β>` (cross-project). Observable: an
  audit run during a RAM-pinned/no-live-consumer state shows those lanes as
  PINNED (assigned ∧ ¬live) instead of FREE; LIVE column documented as
  liveness-only.

## Out of scope

- Pending-pin TTL / expiry-to-reclaimable (P4 option (b) from the discussion) —
  revisit only if valve-on + 2891 census prove insufficient.
- Status/age-aware restart adoption — most invasive option, explicitly held.
- Reverse config check (defaults.yaml keys without model fields) — noted, cheap
  add-on to ζ if wanted later (Open Q3).
- Reify enablement of `warm_lane_prewarm` and the `spare_warm_lanes` key move —
  reify's pre-existing config-fix task.
- Automated fleet-wide `--check-config` CI sweep — ζ ships the tool; wiring it
  into verify pipelines is per-project follow-up.

## Open questions (surfaced but not decided in this session)

1. **Structural-exhaustion L2 threshold default.** Suggested: 5 consecutive
   EXHAUSTED acquires (green-tier knob). Decide during ε.
2. **Drift-L2 threshold default.** Suggested: 3 (green-tier knob). Decide during γ.
3. **ζ scope: also walk `defaults.yaml` for keys without model fields?**
   Suggested: yes if <~20 LOC extra, else defer. Decide during ζ.
4. **MCP tool name.** Suggested: `get_warm_lane_status`. Decide during β.
5. **Census field for quarantined lanes** (durable QUARANTINED records are not
   pool members): include as `n_quarantined` from records? Suggested: yes —
   operators counted them by hand during the incident. Decide during α.
