# Capability manifest — `plans/flake-ledger-prd.md`

Built at decompose, 2026-08-06. Mechanizes **G3** (assumed substrate verified) and
**G6** (premise validity) per task, so the substrate check is paid **once, here**,
rather than once per task at dispatch. Machine-readable twin:
`plans/flake-ledger-prd.capability-manifest.yaml` (schema
`shared/src/shared/capability_manifest.py`; validated clean, 11 tasks / 30 capability
bindings / 22 mechanical + 8 manual `delivered_check`s).

## Vacuity audit — the part that makes these gates real

A `delivered_check` that already passes on `main` is a **no-op gate**. Before binding,
every pattern in the sidecar was measured against `main` at commit `b970b6b520`:

| Pattern | `expect` | Hits on main | Non-vacuous |
|---|---|---|---|
| `CREATE TABLE IF NOT EXISTS flake_occurrence` | present | 0 | ✅ |
| `CREATE TABLE IF NOT EXISTS flake_debt` | present | 0 | ✅ |
| `def record_flake_occurrence` | present | 0 | ✅ |
| `psi_cpu_some10` | present | 0 | ✅ |
| `async def confirm_isolated_rerun_verdict` | present | 0 | ✅ |
| `class FlakeVerdict` | present | 0 | ✅ |
| `effective_merge_module_configs\(` (merge_queue.py) | present | 0 | ✅ |
| `def effective_merge_module_configs` | present | 0 | ✅ |
| `['"]merge_gate['"]` | present | 0 | ✅ |
| `['"]main_probe['"]` | present | 0 | ✅ |
| `flake_suppression['"]?\s*[:=]` (verify_runner.py) | present | 0 | ✅ |
| `flake_suppression['"]?\s*[:=]` (merge_queue.py) | present | 0 | ✅ |
| `async def open_debt` | present | 0 | ✅ |
| `owner_task_id\s*[:=]` | present | 0 | ✅ |
| `regressed_after_resolution` | present | 0 | ✅ |
| `prior_resolving_commit` | present | 0 | ✅ |
| `_submodel_leaf_paths\(['"]flake_ledger['"]` (config.py) | present | 0 | ✅ |
| `class FlakeLedgerConfig` (config.py) | present | 0 | ✅ |
| `flake.ledger` (cli.py) | present | 0 | ✅ |
| `owner_task_id` (cli.py) | present | 0 | ✅ |
| `['"]chronic_marker['"]` | present | 0 | ✅ |
| `ledger_relpath` (chronic_flake.py) | **absent** | **2** | ✅ |

The one `expect: absent` check is the mechanical form of κ's "stop reading the JSONL"
deliverable (G6 branch 4): it **fails today** (2 hits at `chronic_flake.py:454,478`) and
passes only once the read is gone. Its counterpart — that the `ChronicFlakeConfig` field
itself *survives*, per §11 Q4's deprecate-in-place resolution — is deliberately scoped
out of the pattern by restricting `paths` to `chronic_flake.py` alone.

**Two checks were deliberately rejected as vacuous** and replaced:

- `_main_probe_failure_is_isolated_flake\(` for δ — the gate is already **called** at
  `verify.py:6537`; it is the *call site* that is unreachable, not the symbol. A grep for
  the call would have passed on `main` and gated nothing.
- a bare `flake_suppression` for ε — the substring already occurs inside
  `apply_merge_flake_suppression(`. Anchoring on `['"]?\s*[:=]` excludes the call form
  and measures 0.

## Name-drift policy

PRD §8 permits the contract's symbol names to be adjusted at implementation time *"only
if all call sites move together"*. A grep failing on such a rename is therefore a
**correct** signal that the contract drifted un-recorded — re-stamp the sidecar in the
same change. Do not loosen the pattern to accommodate a rename.

## Per-task bindings

Full evidence strings live in the sidecar; this section records the load-bearing ones
and the judgment calls.

### α — ledger tables + writer/reader API

| Capability | Evidence |
|---|---|
| additive `CREATE TABLE IF NOT EXISTS` schema | `event_store.py:24`, `run_store.py:19/34/70`; `runs.db` already hosts 6 tables from 4 owners |
| WAL + 5 s busy timeout | `sqlite_sync_base.py:26` → called at `event_store.py:512`, `run_store.py:90` |
| PSI sampling, not load average | `shared/src/shared/psi.py:133` `read_psi_sample` → `PsiSample.read_ok` at `:105`, fail-open sentinel `:129` |
| **ledger-owned timestamps** | `tasks.db` has **no** `created_at` [INV]; `planning_mode` bypasses the curator ticket store (`middleware/task_interceptor.py:2611`, re-resolved exact). Without ledger-owned `opened_at`, the age backstop has **no clock** — bound `manual`, since a grep can prove the column exists but not that it is *populated* |

### β — one discriminator, two thin wrappers

The premise "both wrappers' existing tests pass unchanged" was checked rather than
assumed: `confirm_merge_verify_flake_suppressible` has 4 references in
`orchestrator/tests/test_verify_merge_flake_suppression.py`, and
`_main_probe_failure_is_isolated_flake` has 16 in `test_verify_preexisting_main_break.py`.
Both wrappers have real, non-empty test coverage, so the assertion is substantive on
both halves.

Contract invariants 1 (never raises → fail **closed**) and 4 (serial + isolated +
the explicit `_MERGE_FLAKE_CONFIRM_TIMEOUT_SECS = 300` override at `verify.py:7374`,
applied `:7468`) are behavioural and bound `manual`.

### γ — effective module configs at the merge boundary

Anti-orphan: the check is that the helper is **called from `merge_queue.py`**, not merely
defined — `effective_merge_module_configs\(` scoped to `merge_queue.py`. A defined-but-
unwired helper would leave §3.1's inversion in place while looking delivered.

Floor: `EventType.merge_flake_suppressed` is emitted at `verify.py:7727`, and its
**lifetime** count is **0** across 311,758 `runs.db` event rows spanning 2026-04-09 →
2026-08-06 [RV]. γ's signal cannot close vacuously.

### δ — gating the task-μ fork *(the weakest mechanical binding; stated, not hidden)*

Line anchors re-resolved exact on 2026-08-06: the fork is `verify.py:6472`, its early
return `:6479`, and `dark-factory-orchestrator.yaml:694` sets `merge_verify_breadth: "full"`.

δ's deliverable is an **ordering** property — the discriminator must run *before* the
`:6479` return — which no single-line ERE expresses. The mechanical twin bound here
(`['"]main_probe['"]`, 0 on `main`) is honest but weak: β's own wrapper work could
satisfy it. The ordering itself is pinned by **boundary rows B4/B5 in task λ**, and the
gap is recorded as a `manual` check rather than papered over with a pattern that would
imply more than it proves.

### ε — recorder/discriminator split

The pre-existing silent hole is re-verified exact: `verify_runner.py:562-571` builds
`LocalRunner` with **no** `event_store` and **no** `escalation_queue` (defaults `None` at
`:649-650`). So today the remote/CLI path drops the event *and* **resets the storm
streak** — INV-4's escape hatch disarmed precisely where load is highest. Two mechanical
checks (`verify_runner.py` for the carrier field, `merge_queue.py` for dispatcher-side
recording) plus a `manual` for the streak bump, which B3 asserts alongside the other two.

### ζ — the debt invariant at write time

The `task_client` §8.3 requires **already exists**: `chronic_flake.py:357` / `:617`
`async def submit_task`, the de-flake argument builder at `:233`, live call at `:507` —
the facility task 2358 landed. ζ reuses it; it invents nothing.

`owner_task_id\s*[:=]` is the anti-orphan check (the loose `owner_task_id` alone would
match an unrelated comment at `scheduler.py:825` and be vacuous). §5.9's coupling rule —
the ledger never writes task status except the initial filing — is a **negative**
property bound `manual`; it is what keeps INV-6 N/A by construction.

### η, θ, ι — escalation ladder, sweep, operator surface

θ's green-tier binding is the sharpest in the batch:
`_submodel_leaf_paths\(['"]flake_ledger['"]` scoped to `config.py` is *exactly* the
property θ's signal asserts (`applied`, not `restart_required`), copied from the
`chronic_flake` idiom at `config.py:4924`.

**T = 3 d is recorded as provisional and survivorship-filtered**, not as a validated
bound: "0/35 ever exceeded 3 d" is computed over **closed** de-flake tasks, which by
construction excludes exactly the cases an age trigger fires on — task 3552 is over T
right now. The bound is green-tier hot-reloadable and re-derived after ~30 days live;
recurrence, the trigger the measurement actually supports, carries the load meanwhile.

ι discharges **INV-7** for every debt row: machine-readable owner (`owner_task_id`) plus
operator-visible age, both checked mechanically against `cli.py`.

### κ — chronic_flake migration *(one G6 finding, resolved by wiring)*

**G6 branch-3 finding.** κ's PRD signal reads *"an occurrence appears in **the report**"* —
and the report is **ι's** deliverable. PRD §10 listed κ's prereqs as **ζ only**, so ι was
absent from κ's dependency closure and the signal was unobservable in κ's own dispatch.

**Resolution: the κ → ι edge was wired at Step 4** (add the prerequisite upstream, the
first of `gates.md`'s sanctioned resolutions). DAG-direction verified: ι depends only on
α and does **not** depend on κ, so this is a missing-prereq fix, not an inversion, and it
does not disturb §10's shape — ι and κ were already siblings under λ.

### λ — integration gate

Every §9 leg traces to an upstream producer: B1/B2←γ, B3/B13←ε, B4/B5←δ, B6←β+γ,
B7/B8←ζ, B9←η, B10/B11←θ, B12←α, B14←κ. All nine are in λ's transitive closure (direct
γ δ ε η θ ι κ; transitive α β ζ). No leg is owned by a task that depends on λ. Bound
`manual` — the task *is* the check suite.

## Gate dispositions

| Gate | Verdict |
|---|---|
| **G1** consumer named | **PASS** — 5 live consumers (§1), all re-resolved on `main` |
| **G2** user-observable leaf | **PASS** — λ is the only graph-leaf and names §9; γ/δ/ε each additionally carry a *currently-absent* observable signal, so none can close vacuously |
| **G3** substrate verified | **PASS** — §4's table spot-checked; no novel substrate. Two off-by-one drifts found and corrected (`run_store.py:89`→`:90`, `verify.py:7026`→`:7027`); all load-bearing anchors (`verify.py:6472`/`:6479`, `verify_runner.py:537`/`:562-571`/`:649-650`, `merge_queue.py:2543`/`:2591`, `dark-factory-orchestrator.yaml:694`, `config.py:4924`, `task_interceptor.py:2611`) exact |
| **G4** seam ownership | **PASS with one note** — see below |
| **G5** B+H | **PASS** — §8 contract + §9 14-row sketch; λ is the integration gate naming it |
| **G6** premise validity | **PASS after one resolution** — κ's report capability wired to ι |
| **G7** design invariants | **PASS, no waivers** — walked over all 11 tasks (not only leaves); three enforcement requirements carried into task text |

### G4 note — the 3774 seam holds, but for a narrower reason than §7 states

§7 justifies "no dependency edge" with *"3774 edits `merge_queue.py` and `workflow.py`;
this PRD's δ edits `verify.py:6472`. Different files."* That is true of **δ**, but **γ**
also edits `merge_queue.py` (§8.2's ordering invariant puts the helper call at the
merge-request boundary, ahead of `:2543` and `:2591`).

The conclusion still holds, on the narrower ground that the two edits occupy **disjoint
regions** of that file — 3774 works in `_file_main_health_escalation` (`:1458-1463`),
γ at the merge-request boundary (`:2543`/`:2591`) — so there is no merge conflict, and
either landing order is correct exactly as §7 argues. What it does mean is that γ and
3774 contend for the **same narrow file lock** and will serialize rather than run
concurrently. That is a scheduling cost, not a correctness problem, and inventing a
dependency edge would serialize them *permanently* for no gain. Recorded on γ's task.

### G7 walk — no waivers, three requirements carried into task text

| Inv | Disposition across all 11 tasks |
|---|---|
| **INV-1** `contracts-machine-checked` | Satisfied. `FlakeVerdict` StrEnum + frozen `FlakeSuppression` + a SQL schema + a pydantic `FlakeLedgerConfig` — the contract is typed everywhere, and B2 asserts local/remote agreement as a **value** property, not a comment asking two sites to stay in step |
| **INV-2** `structured-facts-at-failure` | The PRD's core move. `unconfirmable` goes from a fact held in a variable and dropped to INFO (`verify.py:7027`) to a typed verdict and a counted row; the occurrence row separates raw observation (`observed_at`, `psi_cpu_some10`, `runner`) from hypothesis |
| **INV-3** `corroborate-before-acting` | Satisfied, **with a requirement**: ζ re-corroborates `owner_task_id` against live task status (B8). Carried into **η** and **θ** as well — both act on a debt row that may have changed since it was read, and must re-read live status before stamping a resolution or firing a class-2 L2 |
| **INV-4** `storm-escape-required` | Satisfied, and it repairs an existing breach (ε re-arms the streak on the remote path). **Requirement carried into δ and ζ**: each introduces a fail-soft path that becomes live *before* θ's three ledger-health counters land, so both must route through the existing `_bump_suppression_streak_and_maybe_escalate` sentinel (`verify.py:7791`) — never escape-less in the interval |
| **INV-5** `no-lockstep-duplication` | Central. Two gates → one discriminator; two module-set derivations → one helper. Both are extractions, not documented conventions. κ's two unioned input paths are genuinely different *sources*, not duplicated logic |
| **INV-6** `status-matches-liveness` | N/A by construction — §5.9's coupling rule forbids the ledger from writing task status at all (except the initial filing), so no new exit path can strand a task |
| **INV-7** `holds-owned-and-bounded` | Satisfied. Every debt row is a hold: owner `owner_task_id` (machine-readable, re-corroborated), bound `debt_age_escalate_days` **and** the recurrence trigger, surfaced with its age by ι |

### Sequencing exposure recorded rather than designed away

γ makes suppression **land merges** at a non-zero rate for the first time (lifetime count
today: 0), while ζ — the debt invariant that keeps a landed flake visible — sits on a
different branch (ζ←ε←α,β) and can land later. §5.7 is explicit that *"the invariant is
what makes landing safe — not a companion to it."*

Note the exposure is bounded, not open: α is in γ's transitive closure (γ←β←α), so
suppressions write occurrence rows from the moment γ lands, and the existing storm
counter still fires. No dependency edge was added — that would serialize the three
"independent vertical slices" §10 deliberately kept parallel. Instead **ζ is filed at the
same `high` priority as γ/δ/ε** so the scheduler does not starve the invariant behind the
mechanism that needs it, and the coupling is stated on both tasks.
