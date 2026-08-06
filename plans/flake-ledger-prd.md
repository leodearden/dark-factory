# Flake ledger — PRD

**Status:** authored 2026-08-06. Approach **B + H** (contract + two-way boundary tests) per G5.
**Origin:** the `/deb` investigation of false blocking critical `esc-3650-2` (2026-08-05), continued
2026-08-06 after `esc-3666-3` reproduced it in under 12h from the same already-owned bug (task 3552).
**Owns:** the **isolated-rerun flake discriminator** (one implementation, all call sites), the
**flake ledger** (`flake_occurrence` + `flake_debt` in `runs.db`), the **debt invariant** and its
write-time enforcement, and **ledger-health escalation**. Explicitly does **not** own the main-health
escalation's *instruction* — that is task **3774** (§7).

Goal, in Leo's words: *not to accept flaky tests, but to efficiently detect them, track them, limit
the damage for as long as they exist, and de-flake them conclusively* — as a **mostly-unattended
orchestrator subsystem** that escalates only on exceptional difficulty.

---

## 1. Consumer and user-observable surface (G1)

Five named consumers, all live today, none speculative:

| # | Consumer | What it consumes | Where |
|---|---|---|---|
| 1 | **Merge queue** (`merge_queue.py` `_do_merge` / train / post-merge) | the discriminator's verdict → land-or-red; the recorder writes the ledger row | `orchestrator/src/orchestrator/merge_queue.py:2591`, `:2543` |
| 2 | **Main-health classifier** `verify_failure_is_preexisting_on_main` | the same discriminator, applied before it can return *wholly-preexisting* | `verify.py:6337`, task-μ fork at `:6472` |
| 3 | **Escalation queue → L2 human** | three ledger-health classes (§5.6) | `escalation` package, born-at-L2 path |
| 4 | **Task tree** (`get_tasks` / `get_task`) | the auto-filed de-flake task that the debt invariant guarantees exists | fused-memory task store |
| 5 | **`chronic_flake.py`** | the ledger replaces its JSONL reader as the evidence source | `orchestrator/src/orchestrator/chronic_flake.py` |

**User-observable surface.** Four, each observable through a product read path, none by peeking at SQLite:

1. **A load-flaked merge lands instead of blocking**, and says so: a `merge_flake_suppressed` event
   appears in `runs.db` for a failing test in a module the task never touched. Today that event has
   **never been emitted** (§2).
2. **A load flake stops producing a false `preexisting_main_break` critical.** The escalation is not
   filed; a `flake_occurrence` row with verdict `passes_in_isolation` is written instead.
3. **`orchestrator flake-ledger` (new read-only CLI report, task ι)** prints open debt with its owning
   task id and age, the recurrence chain per test, and the three health counters.
4. **Every suppressed test has a non-terminal de-flake task**, visible via `get_task` and named on the
   ledger row (`owner_task_id`). This is the invariant, and it is what makes landing safe.

---

## 2. Premise — measured, with each figure's provenance marked

Two provenance classes are kept distinct on purpose (a PRD's "verified" table is a claim, not evidence):
**[RV]** = re-verified from primary sources during this authoring session, 2026-08-06;
**[INV]** = measured by the originating `/deb` investigation on 2026-08-05/06 and carried forward.

| Measurement | Value | Prov. | Source |
|---|---|---|---|
| `merge_flake_suppressed` events, **lifetime** | **0** | [RV] | `runs.db` `events`, 311,758 rows spanning 2026-04-09 → 2026-08-06 |
| `main_health_red` `merge_attempt` rows | **35** | [RV] | same |
| …distinct **episodes** within those 35 rows | **12**, of which **8 FALSE** = **67% FP** | [INV] | one genuine episode contributes 19 rows — quote the **episode** rate, never the 28.6% row rate |
| journald lines naming `confirm_merge_verify_flake_suppressible`, since 08-01 | **20** | [RV] | `journalctl --user -u orchestrator-dark-factory.service` |
| journald lines naming `_main_probe_failure_is_isolated_flake`, ever | **0** | [RV] | same — the gate has never executed |
| `…did not map to any discovered subproject … — unconfirmable` lines, since 08-01 | **6** | [RV] | same — logged at **INFO**, consumed by nobody |
| `_spawn_main_health_fix_task` firings, ever | **0** | [INV] | the escalation-only path is the live one |
| DF de-flake cycle time | P98 **2.6–2.7d**, max 2.7d, **0/35 ever exceeded 3d** | [INV] | closed de-flake tasks |
| reify de-flake cycle time | P98 **2.8–2.9d** | [INV] | reify task store |
| Distinct **recurrence chains** | **12** — e.g. `test_spawn_claude.py` **7 de-flake tasks in 7 weeks**; reify `test_verify_semaphore_e2e.sh` **4 rounds in 6 days** | [INV] | |

**Reading of the numbers.** 20 gate executions, 0 suppressions, 6 unconfirmables: the gate runs and
never fires. Its safety net has been structurally absent for its whole life and nothing said so,
because "unconfirmable" is an INFO log line rather than a counted fact.

**Epistemic caution, carried forward deliberately.** "0/35 ever exceeded 3d" is a **survivorship
artifact** — it is computed over *closed* de-flake tasks, which by construction excludes the ones an
age trigger would fire on. Task **3552** is over T *right now*. Do not read the age table as evidence
that an age backstop would never fire; read it as the distribution of the cases that *did* close.
This is also why **age is the backstop and recurrence is the primary trigger** (§5.6).

---

## 3. The two prerequisite gate bugs — a ledger fed by dead gates records nothing

Dark-factory has **two** isolated-rerun flake gates. Both are structurally dead. Fixing them is inside
this PRD, not a precondition of it, because the fixes and the ledger share one discriminator.

### 3.1 Gate 1 — `confirm_merge_verify_flake_suppressible` is blind to untouched modules

`verify.py:7377`, reached via `apply_merge_flake_suppression` (`verify.py:7866`) from
`LocalRunner.run_merge_verify` (`verify_runner.py:730`). It receives the **task-scoped**
`module_configs`.

The `merge_verify_breadth: "full"` expansion —

```python
# verify.py:5916-5917, inside run_verification
if role == 'merge' and verify_plan._merge_breadth_is_full(config):
    module_configs = list(config.module_configs_or_empty.values()) or module_configs
```

— **rebinds a local variable** inside `run_verification` and never propagates out. So the gate maps
node-ids against only the modules this task touched, and `_group_node_ids_by_subproject`
(`verify.py:6956`) returns `None` — *unconfirmable* — for exactly the failures most likely to be
unrelated load flakes. **The gate is inverted:** the more clearly unrelated the failure, the less able
it is to say so.

**Fix at the merge-path boundary, not at either consumer.** Both consumers take the same scoped set:
`merge_queue.py:2591` constructs `LocalRunner(merge_wt, req.config, req.module_configs, …)` and
`merge_queue.py:2543` calls `build_merge_verify_spec(req.config, req.module_configs, …)`, which
projects it into `spec.verify_commands`; `run_merge_verify_on_worktree` then reconstructs
`module_configs` from those commands (`verify_runner.py:537`). Resolving the effective set **once**,
at the boundary, ahead of both, makes local and remote identical **by construction** and leaves
`verify.py:5916` as a call to the same helper (INV-5).

### 3.2 Gate 2 — `_main_probe_failure_is_isolated_flake` is unreachable

`verify.py:7538` (task 3597). `dark-factory-orchestrator.yaml:694` sets `merge_verify_breadth: "full"`,
which makes `failing_result.failing_test_ids` non-`None`, so the **task-μ fork at `verify.py:6472`
returns at `:6479`** — before the legacy probe path where the gate lives. It has never logged once
[RV].

The μ fork's own failure mode is precisely what the gate was built for: a load flake can starve the
**same** timing-sensitive test on the branch **and** on the main baseline, so the id-sets match, `wholly`
is `True`, and a green main is declared broken.

**Fix:** `verify_failure_is_preexisting_on_main` already receives the branch `worktree` as its first
parameter (`verify.py:6337`) and is documented as local-only on the dispatching host. So the μ fork can
run the **same** same-tree isolated re-run before returning `(True, main_sha)`. Both gates converge on
one discriminator with two call sites.

---

## 4. Pre-conditions (G3 — every capability verified against main, 2026-08-06; none assumed)

| Assumed capability | Verified evidence |
|---|---|
| `runs.db` is WAL with a busy timeout | `event_store.py:512` / `run_store.py:89` → `apply_full_durability_pragmas_sync(conn, busy_timeout_ms=5000)`, `shared/src/shared/sqlite_sync_base.py:26` |
| Schema is additive `CREATE TABLE IF NOT EXISTS` | `event_store.py:24`, `run_store.py:19/34/70` |
| `runs.db` already hosts multiple owners' tables | live DB: `events`, `invocations`, `account_events`, `runs`, `task_results`, `scheduler_state` |
| No event retention or pruning exists anywhere | no `DELETE FROM events` / prune / retention in `event_store.py` or `run_store.py` — **so ledger rows accumulate; §5.2 bounds them** |
| All 8 client projects have their own `data/orchestrator/runs.db` | `orchestrator` is the shared per-project package; DB path derived per project_root |
| `VerifyResult` round-trips structurally | `result_to_json` / `result_from_json`, `verify_runner.py:2066/2071`; `failing_test_ids` is the precedent (task μ) |
| `MergeVerifySpec` already carries merge-gate profile fields over the wire | `merge_verify_workspace`, `merge_verify_breadth`, `global_verify_command` — `verify_runner.py:300-316` |
| Remote/CLI path drops suppression side-effects today | `run_merge_verify_on_worktree` builds `LocalRunner` with **no** `event_store`, **no** `escalation_queue` (`verify_runner.py:562-570`; defaults `None` at `:649-650`) |
| A fixed-sentinel L2 dedup pattern exists to model on | `_bump_suppression_streak_and_maybe_escalate`, `verify.py:7791`, sentinel `merge-flake-suppression-storm` |
| A registered periodic-sweep seam exists | `BackgroundService` registry, `harness.py:1955-2025` (`main-tip-sweep`, `stranded-reconcile`, `deterministic-recon-sweep`, …) |
| Green-tier hot-reload for a config submodel | `_submodel_leaf_paths('chronic_flake', ChronicFlakeConfig)`, `config.py:4924` — the exact idiom to copy |
| PSI sampling, not load average | `shared/src/shared/psi.py` — `read_psi_sample()` → `PsiSample`, fail-open `read_ok=False` |
| `chronic_flake` has two unioned input paths, one storage-independent | `_merge_evidence_by_test`, `chronic_flake.py:371-397` — ledger reader **and** `parse_chronic_flaky_markers` (parses verify **stdout**) |
| `.taskmaster/tasks/tasks.db` has **no** `created_at` | verified [INV]; and `planning_mode=True` bypasses the curator ticket store (`task_interceptor.py:2611`) — **so the ledger must own its own timestamps** |
| The shared node-id mapper already knows "unconfirmable" | `_group_node_ids_by_subproject`, `verify.py:6956`, logs the literal word at `:7026` and returns `None` |

**No novel substrate.** Every mechanism below is composed from capabilities that exist on main today.

---

## 5. Resolved design decisions

### 5.1 Storage: a table in `data/orchestrator/runs.db`, not the JSONL `ledger_relpath` names

Leo's call. JSONL leaves an **unclosed concurrency bug inside the flake detector itself** — multiple
merge lanes and a remote host appending to one file — and a lockfile means hand-rolling what SQLite
already provides. `runs.db` is WAL with a 5s busy timeout, its schema is additive, and it already hosts
six tables from four owners (§4). `ChronicFlakeConfig.ledger_relpath` becomes **vestigial** (§5.8).

### 5.2 A general facility, not a dark-factory feature

The table plus a **writer API in the `orchestrator` package** — `record_flake_occurrence()`,
`open_debt()`, `resolve_debt()` — not raw SQL at four call sites. `orchestrator` is the shared
per-project package and every client project already has its own `runs.db`, so this hands the facility
to all 8 projects with **zero provisioning**.

**Retention.** §4 establishes there is no pruning anywhere in `runs.db`, so debt rows cannot be reaped
by any existing mechanism. `flake_debt` is bounded by construction — **one row per test**, resolved
rows retained for the recurrence trigger (§5.6), which is exactly what needs them. `flake_occurrence`
is append-only and unbounded; it is bounded the same way `events` is — it isn't, and that is a known,
named, accepted debt consistent with the rest of `runs.db`, not an oversight. Recorded in §11.

### 5.3 Data model: two tables

```sql
CREATE TABLE IF NOT EXISTS flake_occurrence (   -- append-only; many rows per test; the evidence trail
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    observed_at     TEXT    NOT NULL,   -- ISO-8601 UTC, LEDGER-OWNED (§5.4)
    test_id         TEXT    NOT NULL,   -- pytest node-id, or script-suite test name
    project_id      TEXT    NOT NULL,   -- denormalised: the dashboard aggregates across runs.db files
    verdict         TEXT    NOT NULL,   -- passes_in_isolation | fails_in_isolation | unconfirmable
    call_site       TEXT    NOT NULL,   -- merge_gate | main_probe | chronic_marker
    runner          TEXT,               -- 'local' | remote host name — WHERE the discriminator ran
    merge_sha       TEXT,
    task_id         TEXT,
    psi_cpu_some10  REAL,               -- host pressure AT observation (shared.psi), NULL if read_ok=False
    detail          TEXT    DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS flake_debt (         -- ONE row per test; the set the invariant governs
    test_id                 TEXT PRIMARY KEY,   -- runs.db is per-project, so test_id alone is the key
    project_id              TEXT NOT NULL,
    opened_at               TEXT NOT NULL,      -- LEDGER-OWNED
    resolved_at             TEXT,               -- NULL while open
    owner_task_id           TEXT,               -- the non-terminal de-flake task (§5.5)
    open_count              INTEGER NOT NULL DEFAULT 1,
    prior_resolved_at       TEXT,               -- previous cycle — feeds the recurrence trigger
    prior_resolving_commit  TEXT,               -- cited verbatim in the regressed_after_resolution L2
    last_occurrence_at      TEXT NOT NULL
);
```

**Debt opens on first suppression.** Re-entry after a resolution **updates** the single row
(`open_count += 1`, prior-cycle fields carried forward) rather than inserting a second — which is what
makes the recurrence trigger a primary-key lookup rather than a scan.

### 5.4 The ledger carries its own timestamps

Non-negotiable and load-bearing: `.taskmaster/tasks/tasks.db` has **no `created_at`** column, and
`planning_mode=True` bypasses the curator ticket store entirely (`task_interceptor.py:2611`) — so the
task store cannot answer "how long has this debt been open?" for exactly the tasks this subsystem
files. `opened_at` / `resolved_at` are ledger-owned or the age backstop has no clock.

### 5.5 Record the observation, not the remedy

The verdict vocabulary is **`passes_in_isolation`**, never `flaky_test: true`.

This is not cosmetic. "De-flake" vocabulary biases an agent toward **widening a timeout**, which is the
task-**1836** failure mode: a 10s→30s widening that **masked** a real SIGHUP bug for a day until task
1841 found it. And the `esc-3650-2` flake was itself a **real production bug** — SIGPIPE under
`pipefail` at `scripts/install-flag-marker-sweep-timer.sh:39` (task 3552) — not a bad test. A test that
passes in isolation and fails under load is *evidence about the system*, and the ledger records the
evidence, leaving the diagnosis to the de-flake task.

**Corollary, binding on every task in this batch:** do not widen a timeout as a fix for anything.

The third verdict, **`unconfirmable`**, is the one that has been missing. `_group_node_ids_by_subproject`
already computes it and already logs the literal word at INFO (`verify.py:7026`) — a textbook INV-2
violation, since the emitter holds the fact in a variable and the only consumer would have to scrape a
log to get it. Promoting it to a structured, counted fact is what makes gate-blindness detectable
(§5.6 class 1) instead of invisible for a month.

### 5.6 Escalate on LEDGER HEALTH, not individual flakes

This is what makes the subsystem unattended. **An individual suppression never escalates.** Three
classes, nothing else:

| Class | Predicate | Why it exists |
|---|---|---|
| **1 — gate blind** | `unconfirmable` rate over a window exceeds threshold (with a minimum-observations floor) | The safety net itself is broken. This is exactly what went undetected: 6 unconfirmable lines sat at INFO for a month while the gate suppressed nothing, ever. |
| **2 — non-convergence** | **(a) RECURRENCE, primary:** a test re-enters debt after a resolution. **(b) AGE, backstop:** debt open > T. | Fixing flakes is expected; *failing to fix them* is the pathology. |
| **3 — systemic** | suppression-rate spike across many **distinct** tests within a short window, correlated with `shared.psi` saturation | The host is sick, not the suite. Distinct-test count is the discriminator: one test suppressing repeatedly is class 2; six tests suppressing at once is class 3. **PSI, not load average.** |

**Why recurrence is primary.** It makes resolution cheap and trusting — no expensive proof-of-fix gate
— while making a *failed* fix loud immediately. Class 2(a) fires to **L2 at once**, flagged
`regressed_after_resolution` and citing `prior_resolving_commit`, so the human sees the fix that did
not hold rather than a fresh mystery. Measurement says this is the dominant pathology: **12 recurrence
chains**; `test_spawn_claude.py` took **7 de-flake tasks in 7 weeks**.

**Why age is only the backstop.** **T = 3 days (dark-factory) / 4 days (reify)**, provisional,
green-tier hot-reloadable, **re-derived after ~30 days live**. Basis: measured de-flake cycle time (DF
P98 2.6–2.7d; reify P98 2.8–2.9d). Do **not** inherit the fix-class P98 of 7.5–8.4d — that tail is
83–100% *scheduler queue latency*, not fix difficulty, and pinning T to it would make the backstop
fire on the scheduler's backlog rather than on stuck work. And per §2, do not treat "0/35 exceeded 3d"
as proof the backstop is inert: that sample is survivorship-filtered.

### 5.7 Suppression LANDS the merge

Leo's call — the merge queue is already a throughput bottleneck, and a confirmed
`passes_in_isolation` verdict is positive evidence the tree is fine. The consequence is the whole
reason the invariant exists: once suppression lands the merge, **the flaky test stops blocking
anything**, so nothing except the debt invariant is keeping it visible. The invariant is what makes
landing safe — not a companion to it.

### 5.8 Remote/CLI verify is first-class: split the discriminator from the recorder

**Fail-closed on remote verify was proposed and rejected** — remote verify hosts are critical
infrastructure, and a subsystem that degrades merge safety when a host is remote is worse than the
problem.

The split follows the topology, not a preference:

- The **discriminator** re-runs tests, so it must run **where the worktree is** — remote or local.
- The **recorder** writes `runs.db` and files tasks, so it must run **on the dispatcher**.

The carrier between them is a typed **`flake_suppression`** field on `VerifyResult`, round-tripped by
the existing `result_to_json` / `result_from_json` codec — the identical precedent task μ set with
`failing_test_ids`.

This also closes a **pre-existing silent hole**: `run_merge_verify_on_worktree` builds `LocalRunner`
with `event_store=None` and `escalation_queue=None` (`verify_runner.py:562`), so on the remote/CLI path
today `merge_flake_suppressed` is silently dropped **and the storm streak silently resets** — the
INV-4 escape hatch is disarmed on exactly the path most likely to be under load. Moving the
side-effects to the dispatcher makes them unconditional by construction rather than dependent on which
host happened to run the verify.

### 5.9 The invariant, enforced at write time

> **Any test in the flaky ledger has a non-terminal de-flake task explicitly responsible both for
> fixing the root defect and for removing the test from the ledger.**

Enforced **at write time, inside `open_debt()`** — if no non-terminal task exists for this test, file
one right there — so the invariant is **self-maintaining** rather than audited after the fact. Dedup
follows `_bump_suppression_streak_and_maybe_escalate` (`verify.py:7791`), which already dedupes on a
fixed sentinel: here the sentinel is `owner_task_id` on the debt row, re-corroborated against live task
status before it is trusted (INV-3) rather than assumed still-open from the stored value.

**Coupling rule — binding:** *the ledger reads task status but never writes it, except the initial
filing.* The ledger never marks a task done, never blocks one, never reprioritises one. A de-flake
task's lifecycle belongs to the orchestrator; the ledger only observes it.

### 5.10 Chronic 3-in-20 is retired as a gate, repurposed as severity

The `threshold: 3` / `window: 20` numbers came **verbatim from reify's `run_all.sh`** and were never
derived for dark-factory. More decisively: once *every* suppressed test must have a task, the question
those numbers answer — "is this worth filing yet?" — is already answered. So the chronic computation
stops deciding *whether to file* and becomes a **priority/severity input** on the debt row.

`chronic_flake.py`'s two input paths are unioned at `_merge_evidence_by_test` (`chronic_flake.py:371`).
Migration replaces **only** the JSONL reader with a ledger read and **keeps
`parse_chronic_flaky_markers()`**, which parses verify **stdout** and is therefore storage-independent
— that is reify's own authoritative "this is chronic" trigger and it keeps working untouched. A parsed
marker becomes an **occurrence producer** (`call_site='chronic_marker'`, verdict
`passes_in_isolation` — reify's serial-retry pass *is* an isolation pass), which gives reify a
first-class feed into the ledger without dark-factory reading reify's JSONL at all.

---

## 6. Out of scope

| Excluded | Why (each is a decision, not an omission) |
|---|---|
| **Infra-class failures** — e.g. the 2763/2826 merge-worktree venv-sync failure producing 431 phantom `Import "pytest" could not be resolved` | Leo's call: they need **system-configuration** changes, not TDD tasks, so auto-filing a fix task is the wrong response; and they are commonly urgent, so escalating to L2 is the **correct** status quo. The ledger must not swallow them. |
| **The rc=5 "no tests ran" class** | Already fixed at source by task **1852** (`_is_collectable_test_file`, `verify.py:2516`); no recurrence since June. |
| **`await_preexisting_main_hotfix`'s wait-state instruction** | Owned by task **3774** (pending). See §7 — the fence is reciprocal and already written on 3774's side. |
| **A quarantine lane** (known-flaky tests excluded from the merge gate entirely) | Considered and **held**: highest damage-limitation, highest rot risk. Revisit only with ledger data in hand — which this PRD produces. |
| **A dashboard surface** | The operator read path here is the CLI report (task ι) plus the task tree and L2 queue. A dashboard panel would be a second consumer of the same tables and belongs to a dashboard PRD. |
| **Pruning `flake_occurrence`** | Consistent with `events`, which is also unpruned (§4). Named as accepted debt in §11, not silently inherited. |

---

## 7. Cross-PRD relationship and seam ownership (G4)

| Other work | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| **Task 3774** (pending) — main-health escalation instruction | disjoint halves of one audit | `_file_main_health_escalation`'s `suggested_action` (`merge_queue.py:1463`, `workflow.py:6847/:10087/:10131`) | **3774** | wired, **no dependency edge** |
| **Task 3552** (in-progress) — the SIGPIPE bug whose flake produced both false criticals | this PRD consumes it as the live test case | `scripts/install-flag-marker-sweep-timer.sh:39` | 3552 | independent |
| **Task 3597** — authored `_main_probe_failure_is_isolated_flake` | this PRD makes it reachable | `verify.py:7538` | **this PRD** | superseded-in-place by task β |
| **Task 2358** — `chronic_flake.py` and `ChronicFlakeConfig` | this PRD migrates its storage and retires its gate role | `chronic_flake.py:371`, `config.py:2224` | **this PRD** | task κ |
| **reify** `tests/infra/run_all.sh` (cross-project runtime) | produces `CHRONIC-FLAKY` markers | `parse_chronic_flaky_markers()` | reify | **unchanged** — deliberately kept as the storage-independent path |

**The 3774 seam, resolved explicitly.** 3774's own ANTI-SCOPE §2 already fences off
`verify.py:6472-6480`, `_main_probe_failure_is_isolated_flake` (`verify.py:7538`), and
`confirm_merge_verify_flake_suppressible`'s `module_configs` threading as belonging to *this* PRD. The
fence is therefore reciprocal and pre-agreed. Division: **3774 owns what the escalation tells a human;
this PRD owns whether the escalation is filed at all.**

**No dependency edge, and the reason is checked, not assumed.** 3774 edits `merge_queue.py` and
`workflow.py`; this PRD's δ edits `verify.py:6472`. Different files — so no narrow-file-lock contention
and no merge conflict. Either order is correct: if δ lands first, 3774's improved instruction applies
to a smaller (truer) set of escalations; if 3774 lands first, its instruction is simply right sooner.
A named relationship is not a dependency edge, and inventing one here would serialize two independent
fixes for no gain.

---

## 8. Contract (H) — the seam every call site shares

The point of the contract is that **one discriminator serves both gates**, so the two cannot drift
into different notions of "this test passes in isolation" (INV-5). Signatures are the specification;
names may be adjusted at implementation time only if all call sites move together.

```python
# orchestrator/src/orchestrator/flake_ledger.py  (new module)

class FlakeVerdict(StrEnum):
    passes_in_isolation = 'passes_in_isolation'   # re-ran clean, isolated + serial → suppressible
    fails_in_isolation  = 'fails_in_isolation'    # re-ran and failed → a real red
    unconfirmable       = 'unconfirmable'         # could not map node-ids / could not re-run

@dataclass(frozen=True)
class FlakeSuppression:
    """Discriminator output. Produced WHEREVER the worktree is (local or remote);
    consumed ONLY on the dispatcher. Rides VerifyResult across the wire."""
    verdict: FlakeVerdict
    test_ids: tuple[str, ...]      # node-ids examined — EMPTY is legal only for `unconfirmable`
    observed_at: str               # ISO-8601 UTC, stamped by the DISCRIMINATOR at observation
    call_site: str                 # 'merge_gate' | 'main_probe' | 'chronic_marker'
    runner: str                    # 'local' | remote host name — WHERE the re-run ran
    psi_cpu_some10: float | None   # shared.psi at observation; None when read_ok is False
    unconfirmable_reason: str | None  # populated iff verdict is unconfirmable
```

### 8.1 The discriminator — one implementation, three call sites

```python
async def confirm_isolated_rerun_verdict(
    worktree: Path,                        # SAME-TREE (INV-3): the exact tree being judged
    config: 'OrchestratorConfig',
    module_configs: list[ModuleConfig],    # the EFFECTIVE set — see 8.2
    failing_result: VerifyResult,
    *, call_site: str,
) -> FlakeSuppression: ...
```

**Invariants.**
1. **Never raises.** The merge path has no `VerifyInfraError` handler; an uncaught raise stalls the
   merge queue. Any unexpected exception ⇒ `fails_in_isolation` (fail **closed**: merge stays red).
2. **Total.** Every path returns a `FlakeSuppression`. The `None` that
   `_group_node_ids_by_subproject` returns today becomes `unconfirmable` with a reason string — it is
   never silently conflated with "not a flake". *This distinction is the entire gate-blind signal.*
3. **Same-tree, no worktree churn.** Re-runs in the given `worktree`; no `git worktree add`/`remove`.
4. **Serial + isolated + generous timeout.** Preserves `-p no:xdist -o addopts=` and the explicit
   `_MERGE_FLAKE_CONFIRM_TIMEOUT_SECS = 300` override — `-o addopts=` clears pyproject `addopts` but
   **not** `[tool.pytest.ini_options] timeout=60`, so without the explicit override the confirm re-run
   can itself starve under residual load into a false non-suppression.
5. **Pure.** No event emission, no ledger write, no escalation, no task filing. Side-effects belong to
   the recorder (8.3). This is what makes it safe to run on a remote host.

`confirm_merge_verify_flake_suppressible` (`verify.py:7377`) and
`_main_probe_failure_is_isolated_flake` (`verify.py:7538`) become thin wrappers over this — their
existing signatures and tests are preserved deliberately, so the change is provably behaviour-preserving
where it must be and behaviour-changing only where §3 says it should be.

### 8.2 The effective-module-configs helper — resolves §3.1 in one place

```python
# orchestrator/src/orchestrator/verify_plan.py
def effective_merge_module_configs(
    config: 'OrchestratorConfig', module_configs: list[ModuleConfig],
) -> list[ModuleConfig]:
    """The module set a MERGE-role verify actually covers.

    Under merge_verify_breadth='full', the FULL registry
    (config.module_configs_or_empty); otherwise the passed set unchanged.
    An empty registry falls back to the passed set — degrades safely
    rather than silently verifying nothing.
    """
```

**Ordering invariant:** called at the **merge-request boundary** in `merge_queue.py`, ahead of *both*
`LocalRunner(...)` (`:2591`, `:2806`, `:16631`) and `build_merge_verify_spec(...)` (`:2543`).
`verify.py:5916`'s inline rebinding is replaced by a call to this helper, making it idempotent and
value-preserving. Result: local and remote receive the identical set **by construction**, which is the
property §3.1 needs — not an assertion that two sites agree.

### 8.3 The recorder — dispatcher-only

```python
# orchestrator/src/orchestrator/flake_ledger.py
def record_flake_occurrence(db_path: Path, project_id: str, s: FlakeSuppression, *,
                            merge_sha: str | None, task_id: str | None) -> None: ...

async def open_debt(db_path, project_id, test_id, *, task_client, now) -> DebtRow: ...
    # ENFORCES the invariant: re-corroborates owner_task_id's live status (INV-3);
    # files a de-flake task if none is non-terminal. Files ONLY — never writes status.

async def resolve_debt(db_path, project_id, test_id, *, resolving_commit, now) -> None: ...
    # Stamps resolved_at + prior_resolving_commit. Called when the owning task goes terminal.
```

**Invariants.** Writes only via the API, never raw SQL at a call site. Never raises into the merge path
(mirrors `chronic_flake`'s catch-all-defensive contract) — a ledger failure must never fail a verify or
a merge. Idempotent per `(test_id, observed_at, call_site)`.

### 8.4 Wire compatibility

`flake_suppression` is an **optional** field on `VerifyResult` defaulting to `None`, serialised by the
existing codec. A dispatcher running new code against a remote host running old code receives `None`
and behaves exactly as today (no suppression, no ledger row) — degraded, never wrong. The reverse
(old dispatcher, new remote) ignores an unknown key. Version skew across a fleet redeploy is therefore
safe in both directions.

---

## 9. Boundary-test sketch (H) — both sides of the seam

Task λ's observable signal is this table passing end-to-end.

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Untouched-module flake is now confirmable | `merge_verify_breadth='full'`; failing test in a module **not** in `task_files` | Discriminator receives the **full** registry; verdict `passes_in_isolation`; `merge_flake_suppressed` emitted; merge lands. (Today: `unconfirmable`, no event, merge red.) |
| B2 | Local and remote resolve the identical module set | one `MergeVerifyRequest`, both paths | The set fed to `LocalRunner` and the set reconstructed from `spec.verify_commands` are **equal** — asserted on the value, not on both call sites' source text |
| B3 | Remote suppression reaches the dispatcher's stores | discriminator runs on a `RemoteRunner`; `LocalRunner` there has no stores | `flake_suppression` survives `result_to_json`→`result_from_json`; dispatcher writes the ledger row, emits the event, **and bumps the storm streak** |
| B4 | μ fork is gated | `failing_test_ids` non-`None`; baseline present; ids wholly match | Discriminator runs **before** the `(True, main_sha)` return; on `passes_in_isolation` the result is **not** `preexisting_main_break`; an occurrence row is written |
| B5 | μ fork still reports a genuine break | same, but the tests fail on isolated re-run too | `wholly_preexisting` returned unchanged; **no** ledger row; byte-identical to today |
| B6 | Unconfirmable is counted, not conflated | node-id maps to no discovered subproject | Verdict `unconfirmable` with a reason; occurrence row written; merge stays **red**; class-1 counter advances |
| B7 | Invariant holds at first suppression | test T suppressed, no prior debt | `flake_debt` row for T with `opened_at` set and `owner_task_id` naming a **non-terminal** task that `get_task` returns |
| B8 | Invariant re-corroborates rather than trusting the row | debt row's `owner_task_id` names a now-**terminal** task | A fresh de-flake task is filed and the row re-pointed; no duplicate filed while one is genuinely open |
| B9 | Recurrence fires at L2 | T resolved, then re-enters debt | `open_count == 2`; L2 escalation flagged `regressed_after_resolution` **citing `prior_resolving_commit`** |
| B10 | Age backstop fires | debt open longer than `T = 3d` | Exactly one L2 per debt row per window (sentinel dedup); hot-reloading T re-evaluates without a restart |
| B11 | Systemic beats per-test | N distinct tests suppressed inside the window, PSI saturated | **One** class-3 escalation, not N class-2 ones |
| B12 | Ledger failure never fails a merge | ledger DB unwritable | Verify/merge outcome byte-identical to a ledger-less run; the failure is logged loudly |
| B13 | Wire skew is safe | dispatcher new, remote old (field absent) | `flake_suppression is None`; no suppression, no row, no crash |
| B14 | Marker path feeds the ledger without JSONL | `CHRONIC-FLAKY` marker in verify stdout; **no** JSONL file present | Occurrence row with `call_site='chronic_marker'`; severity reflected in the report; **no** second de-flake task filed |

---

## 10. Decomposition plan

Greek labels; real ids assigned at decompose. Every task is `task_kind='normal'`,
`execution_class` unset (ordinary TDD code work) unless noted. **Constraint binding on all tasks: never
widen a timeout as a fix** (§5.5).

| # | Title | Modules | Observable signal | Prereqs |
|---|---|---|---|---|
| **α** | Add `flake_occurrence` + `flake_debt` to `runs.db` with the `flake_ledger` writer/reader API | `orchestrator` | *Intermediate* — unlocks β, ε, ζ, ι. Tables created additively on an existing `runs.db` with no migration step; API round-trips a row | — |
| **β** | Extract one isolated-rerun discriminator (`confirm_isolated_rerun_verdict`); make both existing gates thin wrappers | `orchestrator` | *Intermediate* — unlocks γ, δ, ε. `unconfirmable` becomes a returned verdict instead of a bare `None`+INFO line; both wrappers' existing tests pass unchanged | α |
| **γ** | Resolve effective module_configs once at the merge boundary — kill the task-scoped blindness on **both** paths | `orchestrator` | A merge whose only failing tests live in modules the task never touched emits **`merge_flake_suppressed`** and lands. Lifetime count of that event is **0** today [RV] | β |
| **δ** | Gate the task-μ fork: run the discriminator before `verify_failure_is_preexisting_on_main` returns wholly-preexisting | `orchestrator` | A load-flaked main probe under `merge_verify_breadth: "full"` no longer files `preexisting_main_break`; `_main_probe_failure_is_isolated_flake`'s path executes for the first time (**0** log lines lifetime today [RV]) | β |
| **ε** | Split recorder from discriminator: typed `flake_suppression` on `VerifyResult`, recorded on the dispatcher | `orchestrator` | A merge-verify suppression that happened on a **remote/CLI** host produces a `merge_flake_suppressed` event, a ledger row, **and** a storm-streak bump on the dispatcher — all three silently dropped today | α, β |
| **ζ** | Enforce the debt invariant inside `open_debt()` — file the de-flake task at write time | `orchestrator` | After a first suppression of test T, `get_task` returns a **non-terminal** de-flake task naming T, and the `flake_debt` row's `owner_task_id` points at it | α, ε |
| **η** | Resolution + recurrence: `resolve_debt`, prior-cycle carry-forward, `regressed_after_resolution` L2 | `orchestrator` | A test that re-enters debt after resolution produces an L2 whose detail **cites the prior resolving commit** | ζ |
| **θ** | Ledger-health sweep as a `BackgroundService` + `FlakeLedgerConfig` (green-tier) + the three classes | `orchestrator` | Each of the three classes files a distinct, deduped L2; `T` and the thresholds change under `reload_config` with `applied` (not `restart_required`) | η |
| **ι** | `orchestrator flake-ledger` — read-only operator report | `orchestrator` | **Leaf.** The command prints open debt with owning task id + age, per-test recurrence chains, and the three health counters | α |
| **κ** | Migrate `chronic_flake.py` off JSONL; retire 3-in-20 as a gate; markers become occurrence producers | `orchestrator` | With a `CHRONIC-FLAKY` marker in verify stdout and **no** JSONL file present, an occurrence appears in the report and **no** duplicate de-flake task is filed | ζ |
| **λ** | Integration gate — the §9 boundary suite, both sides of the seam | `orchestrator` | **Leaf.** All 14 rows of §9 pass end-to-end | γ, δ, ε, η, θ, ι, κ |

**Shape.** α+β are the foundation; γ, δ, ε are the three independent vertical slices that each fix one
structural defect and each carry their own observable signal; ζ→η→θ build the invariant and its
escalation ladder; ι is the operator read path; κ is the migration; λ is the C-as-integration-gate leaf
(G2's escape hatch) that α and β discharge into.

**Config shipped by θ** (`FlakeLedgerConfig`, green-tier via
`_submodel_leaf_paths('flake_ledger', FlakeLedgerConfig)`, the `chronic_flake` idiom at `config.py:4924`):

```yaml
flake_ledger:
  enabled: true                    # ships ENABLED, unlike chronic_flake: it depends on no un-landed
                                   # cross-project substrate, and shipping it off leaves the gap open
  debt_age_escalate_days: 3        # T — dark-factory; reify's own config sets 4 (§5.6)
  gate_blind_rate_threshold: 0.25  # unconfirmable / (unconfirmable + confirmed), over the window
  gate_blind_min_observations: 8   # floor — never fire on 1-of-2
  gate_blind_window_hours: 168
  systemic_distinct_tests: 4       # >= N DISTINCT tests suppressed in the window => host, not suite
  systemic_window_minutes: 60
  health_sweep_interval_secs: 900
```

---

## 11. Open questions (tactical — none design-blocking)

1. **`flake_occurrence` growth.** Unbounded, exactly like `events` (§4: nothing in `runs.db` is
   pruned). **Suggested resolution:** ship unbounded and consistent; revisit repo-wide when `events`
   retention is addressed, not as a bespoke policy for one table. Decide during θ if the row rate
   surprises.
2. **`test_id` normalisation across producers.** A pytest node-id and a reify script-suite test name
   are both `test_id`s. **Suggested resolution:** store verbatim as observed; normalise only if the
   recurrence trigger is observed to miss a chain. Decide during κ.
3. **Exact gate-blind window/threshold.** The §10 defaults are first estimates with a
   minimum-observations floor. **Suggested resolution:** ship, then re-derive from real
   `unconfirmable` counts after ~30 days, alongside T's own re-derivation (§5.6).
4. **Whether `ChronicFlakeConfig.ledger_relpath` is deleted or deprecated-in-place.** **Suggested
   resolution:** deprecate in place (leave the field, stop reading it) so no client project's config
   fails validation on upgrade. Decide during κ.

---

## 12. Design-invariant walk (G7 — `docs/legibility/design-invariants.md`)

Advisory walk at author time; the binding walk is at decompose, over **every** task.

| Inv | Assessment |
|---|---|
| **INV-1** `contracts-machine-checked` | **Satisfied by design.** §8 puts the seam in typed signatures + a schema, not prose. §8.2's helper makes local/remote agreement a *value* property (B2 asserts equality of the resolved sets) rather than a comment asking two sites to stay in step. |
| **INV-2** `structured-facts-at-failure` | **This is the PRD's core move.** `unconfirmable` is today a fact the emitter holds in a variable and drops to an INFO log (`verify.py:7026`); §5.5 promotes it to a typed verdict and a counted row. The occurrence row separates raw observation (`observed_at`, `psi_cpu_some10`, `runner`) from any hypothesis. |
| **INV-3** `corroborate-before-acting` | **Satisfied.** Discriminator is SAME-TREE by contract (8.1 inv. 3). `open_debt` re-corroborates `owner_task_id` against **live** task status rather than trusting the stored value (B8 pins it). |
| **INV-4** `storm-escape-required` | **Satisfied, and it repairs an existing breach.** Suppression is a fail-soft path; §5.6 gives it three counters. ε fixes the case where the streak silently **reset** on the remote path (§5.8) — today's escape hatch is disarmed exactly where load is highest. |
| **INV-5** `no-lockstep-duplication` | **Central.** Two gates → one discriminator (8.1). Two module-set derivations → one helper (8.2). Both are extractions, not documented conventions. |
| **INV-6** `status-matches-liveness` | **N/A by construction.** §5.9's coupling rule forbids the ledger from writing task status at all (except the initial filing), so it introduces no exit path that can strand a task. |
| **INV-7** `holds-owned-and-bounded` | **Satisfied.** Every debt row is a hold: its owner is `owner_task_id` (machine-readable, re-corroborated), its bound is `debt_age_escalate_days` **and** the recurrence trigger, and an operator sees it with its age via task ι. |

No waivers required.

---

## 13. META gate

> If I decompose and queue this PRD without further oversight, will the architecture of what gets
> implemented be complete, coherent, cohesive, and good?

**Yes.**

- **Complete** — the two dead gates, the storage, the invariant, both triggers, all three health
  classes, the remote path, the operator read surface, and the `chronic_flake` migration each have an
  owning task; §6 states what is excluded and why each exclusion is a decision.
- **Coherent** — one discriminator, one effective-module-set helper, one ledger API, one carrier
  field. The recurring failure this PRD fixes is *two sites that were supposed to agree and didn't*;
  the design answers it with extraction everywhere rather than with more agreement.
- **Cohesive** — every task depends on capabilities verified on main (§4), and each of γ/δ/ε carries a
  signal that is currently **provably absent** (0 events, 0 log lines, silently-dropped side-effects),
  so none can be closed vacuously.
- **Good** — it escalates on ledger health rather than on individual flakes, which is what makes it
  unattended; it records the observation rather than the remedy, which is what stops it manufacturing
  timeout-widening; and it lands merges rather than blocking them, with the invariant — not the merge
  gate — carrying the safety.

**Residual risk, stated rather than hidden:** T = 3d is provisional and derived from a
survivorship-filtered sample (§2). If it proves noisy it is a green-tier hot-reload away, and the
recurrence trigger — the one the measurement actually supports — carries the load meanwhile.
