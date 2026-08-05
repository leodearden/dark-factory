# Session-resume eligibility seam — PRD

**Status:** authored 2026-08-05. Approach **B + H** (contract + two-way boundary tests) per G5.
**Origin:** investigation of reify `esc-__session_resume_storm__-9` (reify queue, L1, 9th filing), 2026-08-05.
**Owns:** the resume **eligibility predicate's ordering and reason semantics**, the **sidecar lifecycle**, and the
**storm predicate's feeder** — a seam that `plans/transcript-preservation-seam-prd.md` §6 puts explicitly out of
scope, that task 3256 scopes to a single reason string, and that task 3578 scopes to restore-only. No PRD owns it.

---

## 1. Consumer and user-observable surface

**Primary consumer — reify's orchestrator dispatch path.** Every restart-driven re-dispatch. reify restarts on the
8h fleet cadence with 5–8 sessions typically in flight; all of them currently restart fresh.

**Second consumer — task 3578** (`pending`, critical, restore-from-archive). It needs to *locate* an archived
transcript before it can restore one. This PRD produces that lookup (§8 contract); 3578 consumes it. Naming the
seam here is what stops 3578's architect inventing a second, divergent glob (INV-5).

**Third consumer — operators** triaging the storm L1. Nine filings to date, currently held open deliberately
because resolving it re-files it.

**User-observable surface.** Two, both measurable by queries already run in §2:

1. After a restart-driven re-dispatch **into a different warm lane**, a task whose transcript is in the durable
   archive resumes its prior session instead of starting fresh — an `EventType.session_resume` in reify's
   `runs.db` where today only `session_resume_fallback` appears.
2. The storm L1 stops firing on by-design fallbacks, and fires when restore actually breaks.

## 2. Premise (MEASURED 2026-08-05 against **reify's own** `data/orchestrator/runs.db` + filesystem)

Scope note: every figure below is reify-scoped. dark-factory's numbers differ and cannot substitute
(the trap gate `esc-3555-1` existed for exactly this).

| Measurement (reify, 2026-08-05) | Value |
|---|---|
| `session_resume` (success), lifetime | **1** |
| `session_resume_fallback`, lifetime | **197** |
| Fallback reasons, lifetime | `no_transcript` 158, `stale` **39** |
| Fallback reasons, **current boot `run-0eee46b06cf7`** | `stale` **25**, `no_transcript` **0** |
| Storm streak this boot vs `fallback_storm_threshold` | **25 vs 5** (suppressed only by open `esc-…-9`) |
| Archived transcripts recoverable by cwd-agnostic glob | **125 / 197 = 63.5%** |
| Surviving `agent_session.json` sidecars | 16, of which **13 already stale**; oldest **176h** (task 5730) |
| Surviving `claude-config-*` dirs across 81 worktrees | **1**, and it holds no transcript |
| reify `session_resume` config overrides | **none** — pure defaults (86400 / 3 / 5, enabled) |

Four findings, each independently checkable:

**P1 — the reason population FLIPPED on 2026-08-05.** Through 08-04 the fallbacks were ~90% `no_transcript`. The
current boot is **25 of 25 `stale`, zero `no_transcript`**. Task 3256 keys its downgrade on
`reason == 'no_transcript'`, calling it "uniform" and "the clean, machine-checkable discriminator"; that
discriminator now selects **zero** events in reify. Task 3221's bounding claim ("only ~14 were `stale`, which
bounds how much corroboration work could ever recover") is false by the same measurement.

**P2 — `stale` short-circuits before the transcript check, and masks it.** `_session_resume_eligible`
(`harness.py:3060-3110`) returns on the *first* matching predicate in the order `disabled → stale → capped →
no_transcript`. As reify's backlog grew, tasks began waiting >24h between the invocation that wrote the sidecar and
re-dispatch, so `stale` fires first. The transcript is still missing underneath — only the *reported reason*
changed. Traced on five tasks (5848, 5893, 5766, 5344, 5238): each shows the identical `session_id` failing
`no_transcript` repeatedly, then flipping to `stale` at the 86400s boundary.

**P3 — nothing reaps sidecars.** `owner_pid` is recorded on the sidecar (`artifacts.py:50`) but is never
liveness-checked anywhere. `clear_agent_session` runs on normal completion only. So a sidecar whose invocation died
and whose task returned to the backlog is re-adopted **every boot, forever**, aging into `stale`. This is the
storm's fuel supply: it manufactures P1 and P2.

**P4 — the storm predicate has no genuine-breakage feeder.** `no_transcript` is documented as the anticipated
by-design case (`harness.py:3075-3076`, "B4 reseed/wipe") yet classified as "genuine corroboration fail"
(`harness.py:7662`) — the self-contradiction 3256 fixes. `stale` on a long-backlogged task is equally by-design.
Once both are recognised as such, the streak has no remaining feeder that indicates anything is broken.

**Cost.** 197 degraded dispatches across 127 distinct sessions since 2026-07-19, at a measured mean reify
invocation cost of **$3.658** (n=6,379) ⇒ **~$465** of re-run work, ~14/day recently. The cost is silent: the
fallback is safe, so nothing surfaces except a storm L1 that points the operator at NTP.

## 3. Sketch of approach

**The inversion.** Eligibility today asks *"is the sidecar young?"* before *"is the transcript reachable?"*.
Reachability is the question that decides whether resume can work; age is a backstop against resuming into a world
that has moved. The order is backwards, and P1/P2 are what that costs.

**The cheap half is a lookup, not a restore.** 63.5% of the transcripts reify needs are already sitting in
`data/orchestrator/agent-transcripts/`. The archive path is
`<archive_root>/<task_id>/<encoded-cwd>/<session_id>.jsonl[.gz]` — and although the path *encodes* the cwd, a
lookup keyed on `(task_id, session_id)` can glob the cwd component and is therefore **cwd-agnostic by
construction**. MEASURED: tasks 5848 and 5766 each have archives under two *different* lanes
(`_lane-49`/`_lane-0`, `_lane-33`/`_lane-13`), so this is the reify case, not a hypothetical. Nothing in the tree
performs that lookup today — `transcript_archive.py` is write-only, its sole public entry being
`archive_task_transcripts`.

**Sequencing is instrument-then-act.** α adds the lookup and reports its answer on the existing fallback event, so
the recoverable population is measured *in production* before any predicate changes behaviour (INV-3). β then
fixes the reason semantics and stops the false storm. γ drains the zombie sidecars. Only δ/ε — which need 3578's
restore — change what actually resumes.

## 4. Pre-conditions (G3 — all verified 2026-08-05, none assumed)

| Assumed capability | Verification |
|---|---|
| An archive lookup/predicate helper exists | **IT DOES NOT.** `transcript_archive.py`'s only public entry is `archive_task_transcripts` (`:128`); the module is write-only. Resolved per G3(b): **queued as leaf α**, upstream of every consumer. The `archive_root` hits in `verify_runner.py` are a *different* archive (merge-verify logs). |
| The archive path is derivable and cwd-globbable | `_archive_one` builds `archive_root / task_id / rel.parent / (rel.name + '.gz')` where `rel` is relative to `projects/` (`transcript_archive.py:110`). VERIFIED empirically: the glob `<root>/<task_id>/*/<session_id>.jsonl*` returns 125/197 hits on reify's live archive. |
| A pid-liveness helper exists to reuse | `shared/src/shared/config_dir.py:51` `_pid_alive`. Its docstring records that it is the **fourth deliberate copy** ("copied, not imported… to keep `shared` at the bottom of the dependency stack"). `shared` is importable from `orchestrator`, so γ must import this one. VERIFIED — and see D7: adding a fifth copy is the failure mode to avoid. |
| The sidecar carries `owner_pid` | `artifacts.py:50`, populated at `:1086`. VERIFIED. |
| `session_resume.*` is hot-reloadable | Green-tier via the `session_resume` whole-submodel group in `RELOADABLE_FIELDS` (`config.py:840`). VERIFIED — so the outer bound in D3 is tunable without a restart. |
| A sidecar-clearing primitive exists | `Harness._clear_recovery_artifact` (`harness.py:2967`) clears from **both** the `.task-meta` and legacy roots. VERIFIED — γ reuses it rather than unlinking directly. |
| reify runs pure `session_resume` defaults | No `session_resume` key in `dark-factory-orchestrator.yaml`. VERIFIED. |

**Live-work hazard.** Task **3256** is `in-progress` right now (claimant `run-5566f72d2f49/3256-c655cbdf`,
heartbeat 2026-08-05T09:48Z) editing `harness.py`, `event_store.py`, `config.py` — the exact seam β and γ modify.
Both **must** depend on 3256. α touches only `shared/` and is clear.

## 5. Resolved design decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | **Cross-lane resume is archive-mediated and task-keyed.** Task 3161's boundary test **B6 is narrowed** to its defensible half — no sidecar *artifact* travels laterally between lane dirs — and its "no resume is attempted against lane A's session" clause is **dropped**. | B6 as written and 3578's signal are mutually exclusive; whichever landed second would break the other's gate. B6's real content is credential/state isolation between lane occupants, which archive-mediated restore preserves (the archive is task-keyed and outside every lane). Ruled by Leo 2026-08-05. |
| D2 | **Reachability outranks age.** Corroborate the transcript (live on disk, or restorable from the durable archive) *before* the freshness check; freshness applies only when no durable archive exists. | A durable archive does not decay with wall-clock, so a 24h bound ahead of the reachability check is measuring the wrong thing. This is the direct fix for P2. Ruled by Leo 2026-08-05. |
| D3 | **Keep an absolute outer bound as a backstop**, distinct from `freshness_window_secs`. "Archive outranks age" must not become "no age limit". | Resuming a week-old session onto a branch that has moved is a different hazard from resuming a 25h-old one. The *value* is deliberately not fixed here — see Open question 1; it must be **derived** from observed task in-flight duration, mirroring task 3621's derived-bound precedent, not picked. |
| D4 | **The storm streak fires only on eligible-but-FAILED.** Every by-design outcome (`disabled`, `capped`, `stale`, `no_transcript`) is carved out. | P4. `capped` already has exactly this carve-out (`harness.py:7653-7661`); this extends the existing precedent rather than inventing a mechanism. |
| D5 | **Report ALL true reasons, not first-match.** | First-match is why `stale` masked `no_transcript` and why 3256's discriminator went empty (P1). A composite reason set makes the classification robust to which predicate happens to fire first, and is the machine-checkable form INV-1 wants. |
| D6 | The lookup returns **`Path \| None`, not `bool`**, and is the **sole** archive-locating call site. | If it returned a bool, 3578's restore would need its own finder — two globs that must agree byte-for-byte. Returning the path makes "does it exist" simply `is not None` (INV-5). |
| D7 | Reap a sidecar when `owner_pid` is **dead** AND the task is **not in-progress**. Import `shared.config_dir`'s `_pid_alive` (promoted to public); **do not add a fifth copy**. | Both conjuncts are load-bearing: pid-dead alone would reap a sidecar whose task the orchestrator is legitimately re-dispatching after its own restart. The no-fifth-copy rule is explicit because the existing four are a *documented deliberate* pattern that reads as licence to add another. |
| D8 | **Instrument before acting.** α ships `archive_available` on the existing fallback event and changes no behaviour. | INV-3. It converts §2's 63.5% from a one-off measurement into a live production signal, so β/δ act on corroborated ground truth — and it is the operator-visible rate signal that keeps INV-4 satisfied during the window where the streak has no valid feeder (D9). |
| D9 | This PRD owns the **predicate**; task 3578 owns the **restore**. | Clean seam, and it decouples latency: α lands immediately against `shared/`, while 3578 waits behind 3619 ← 3618 ← 3256. |

## 6. Out of scope

- **Making `--resume` actually restore** — task **3578**, this PRD's consumer, not its replacement.
- **Transcript preservation at teardown** — `plans/transcript-preservation-seam-prd.md` leaves 1–4 (3618–3621).
- **Warm-lane acquire/reseed semantics.** `acquire_lane` always re-seeding from base is load-bearing and has been
  rejected as a change target three times (3256, 3578, and again here). This PRD routes *around* the reseed via the
  durable archive rather than fighting it.
- **The `--resume`-against-moved-cwd HARD GATE** — 3578's, and genuinely unanswered. This PRD makes the archive
  *findable* cross-lane (§8 I-B, MEASURED); whether the CLI *accepts* it once placed is 3578's empirical question.
  See Open question 2 for the coverage risk that D10 of the prior PRD introduces.
- **Deduplicating the four `_pid_alive` copies.** Documented and deliberate; D7 only forbids a fifth.
- **reify-side config change.** All work is dark-factory code; reify consumes it via the normal fleet redeploy.

## 7. Cross-PRD relationship and seam ownership (G4)

| Other PRD / task | Relationship | Seam mechanism | Owner |
|---|---|---|---|
| `plans/transcript-preservation-seam-prd.md` | **Adjacent, not overlapping.** Its §6 puts "fallback classification / storm-streak reset" out of scope and assigns it to 3256; 3256's own text scopes itself to one reason string and defers "making resume work" to 3578. The *ordering*, the *reason semantics*, the *sidecar lifecycle*, and the *storm feeder* fall between all three. | `_session_resume_eligible` predicate order; sidecar lifecycle; streak feeder | **This PRD** |
| Task **3578** (restore-from-archive) | **Downstream consumer.** Consumes α's `durable_archive_path` for its restore; δ and ε depend on its restore + CLI-failure instrumentation. | `durable_archive_path()` (§8) | α (producer) / 3578 (consumer) |
| Task **3161** (`.task-meta` ω, boundary matrix B1–B9) | **Direct contradiction, resolved by D1.** B6 asserts no cross-lane resume; 3578 asserts cross-lane resume. Amended, not overridden — see the amendment below. | Boundary test B6 | **This PRD** (amendment); 3161 keeps the test |
| Task **3256** (fallback classification, in-progress) | **File-level collision and premise drift.** Its `no_transcript` discriminator now selects zero reify events (P1). β supersedes its classification approach with D5's composite. | `harness.py:7649-7674` | 3256 lands first; β builds on it |
| Task **3221** (0-successes investigation) | **Overlapping and now partly stale.** Filed pre-PRD; its "~14 stale bounds the recoverable set" premise is refuted by §2, and its root-cause scope is now covered by 3619 + 3578. | — | Flagged for reconciliation at decompose, not silently duplicated |
| Task **3180** (07-29 storm: sidecar `session_id` a generation behind the on-disk transcript) | **Distinct sub-case, still unowned elsewhere.** Not folded in — a generation-skewed sidecar is a *write-ordering* bug, not an eligibility one. | — | 3180 |

**Amendment to task 3161** *(not a new leaf)* — narrow B6 per D1: keep "lane B sees no sidecar" verbatim as the
credential/state-isolation assertion; replace "and no resume is attempted against lane A's session" with "and any
resume that does occur is mediated by the task-keyed durable archive, never by the lane's sidecar". Record that the
original clause encoded a then-true status quo (resume was believed unwired for lanes as of the 2026-07-18 note),
not a designed invariant.

## 8. Contract (H) — the seam 3578 consumes

```python
# shared/src/shared/transcript_archive.py

def durable_archive_path(
    archive_root: Path, task_id: str, session_id: str
) -> Path | None:
    """Return the archived transcript for (task_id, session_id), or None."""
```

| # | Invariant | Why it is load-bearing |
|---|---|---|
| I-A | **Total** — never raises; any glob/OS error yields `None`. | Mirrors `transcript_exists`'s I3 totality contract, which `_session_resume_eligible` relies on to stay total. A raising lookup would break the fail-safe guarantee of the whole predicate. |
| I-B | **Cwd-agnostic** — globs `<archive_root>/<task_id>/*/<session_id>.jsonl*`, so a session archived under one lane's encoded-cwd component is found after re-dispatch into a *different* lane. | This is the reify case. MEASURED: tasks 5848 and 5766 each hold archives under two distinct lane dirs. |
| I-C | **Format-agnostic** — matches `.jsonl` and `.jsonl.gz`. | Spans task 3618's gzip drop with no flag day, and removes any α→3618 dependency. |
| I-D | **Read-only** — never creates, moves, deletes, or decompresses. | Keeps α free of the credential-lifetime and held-state hazards that govern 3619 (INV-7). Restoration is 3578's, under 3619's archive-before-delete guard. |
| I-E | **Sole locator** — no other site globs the archive by session id. | INV-5. 3578's restore calls this; a second finder is the lock-step-duplication failure D6 exists to prevent. |
| I-F | **Deterministic on multiple matches** — newest `mtime` wins. | Observed in the live data (a task re-dispatched across lanes archives under each). An arbitrary pick would make resume non-reproducible. |

## 9. Boundary-test sketch (H) — both sides of the seam

| # | Scenario | Precondition | Postcondition |
|---|---|---|---|
| B1 | Same-lane lookup | archive under lane A; task re-dispatched into lane A | returns that path |
| B2 | **Cross-lane lookup** (the reify case) | archive under lane A; task re-dispatched into lane B | returns the **same** path — cwd component globbed (I-B) |
| B3 | Format span | one session archived `.jsonl.gz` (pre-3618), another `.jsonl` (post-3618) | both found (I-C) |
| B4 | Absent | no archive for the session | `None`, no raise |
| B5 | Hostile root | `archive_root` missing / unreadable / not a dir | `None`, no raise (I-A) |
| B6 | Multiple matches | same `(task_id, session_id)` archived under two encoded-cwd dirs | newest-mtime path, deterministically (I-F) |
| B7 | **Consumer side** — no second locator | 3578's restore path landed | `grep` finds exactly one site globbing `<session_id>.jsonl*` under the archive root (I-E) |
| B8 | **Eligibility side** — reachable-but-old | sidecar age > `freshness_window_secs`, durable archive present | reaches the transcript/restore check; does **not** short-circuit on `stale` (D2) |
| B9 | Outer bound still binds | sidecar age > the D3 absolute bound, durable archive present | still rejected — reachability outranks freshness, not the backstop |
| B10 | Reaper conjunction | sidecar `owner_pid` dead **but** task `in-progress` | **not** reaped (D7's second conjunct) |

## 10. Decomposition plan

Each leaf names its user-observable signal (G2) and its validated premise (G6).

**α — `durable_archive_path` lookup + `archive_available` on the fallback event.** *(no intra-batch deps)*
Ships the §8 contract in `shared/src/shared/transcript_archive.py` with B1–B6 as its tests, and reports its answer
as a structured field on the existing `session_resume_fallback` event. **Changes no eligibility behaviour** (D8).
*Signal:* every `session_resume_fallback` in reify's `runs.db` carries `archive_available: true|false`; querying
that field reproduces §2's 63.5% as a live production rate rather than a one-off filesystem scan.
*Premise:* MEASURED — 125/197 recoverable by this exact glob; the helper does not exist today (§4).
*Modules:* `shared/src/shared`, `orchestrator/src/orchestrator` (event field only).

**β — composite reason reporting + by-design carve-out from the storm streak.** *(depends on 3256)*
`_session_resume_eligible` evaluates **all** predicates and returns the full true-reason set (D5); the streak stops
counting every by-design outcome (D4). Supersedes 3256's single-string discriminator, which §2/P1 shows now selects
zero reify events.
*Signal:* reify's storm L1 stops firing across a boot that produces >5 fallbacks; `session_resume_fallback` events
carry a reason **set** in which `stale` and `no_transcript` co-occur for the aged-sidecar population — the
co-occurrence that P2 shows is real and that first-match reporting hides.
*Premise:* MEASURED — 25/25 `stale` this boot vs 0 `no_transcript`, on a population whose transcripts are also
absent; five tasks traced through the flip.
*Modules:* `orchestrator/src/orchestrator`.

**γ — sidecar reaper: clear a sidecar whose owner is dead and whose task is not in-progress.** *(depends on 3256)*
Per D7, using `shared.config_dir`'s promoted `pid_alive` and the existing `_clear_recovery_artifact`. Sited at the
boot recovery scan, adjacent to `_adopt_recovered_session`.
*Signal:* after one orchestrator boot, reify's count of surviving `agent_session.json` files whose `owner_pid` is
dead **and** whose task is not `in-progress` is **0**; the >24h sidecar population (13 of 16 today, oldest 176h)
drains instead of being re-adopted every boot. B10 pins the second conjunct.
*Premise:* MEASURED — 16 sidecars, 13 stale, oldest 176h (task 5730, `started_at` 2026-07-29); `owner_pid` is
recorded but never liveness-checked anywhere in the tree (§4).
*Modules:* `orchestrator/src/orchestrator`, `shared/src/shared`.
*Note:* task 3619's startup sweeper hooks the same `_recover_crashed_tasks` scan — adjacent, not conflicting, but
decompose should confirm the edge if 3619 is still open when γ dispatches.

**δ — reachability outranks freshness, with a derived outer bound.** *(depends on α, β, 3578)*
Reorders the predicate per D2 and adds D3's absolute backstop. Depends on **3578** because eligibility must be
gated on *restore success*, not merely archive existence — arming `--resume` against a config dir the transcript
was never restored into produces the "No conversation found" failure rather than a clean fallback.
*Signal:* an `EventType.session_resume` appears in reify's `runs.db` for a task re-dispatched into a **different**
warm lane than the one its session ran in — the event that has occurred once in reify's history and never for the
cross-lane case. B8/B9 pin the ordering and the backstop.
*Premise:* MEASURED — 63.5% archive coverage; cwd genuinely varies across re-dispatch (5848, 5766).

**ε — re-point the storm streak at eligible-but-failed.** *(depends on β, 3578)*
The streak's only feeder becomes the population that indicates real breakage: archive-restore failure, and the
CLI-level "No conversation found" path that 3578 instruments and that emits no event today.
*Signal:* a fault injected into the restore path (unreadable archive) files the storm L1 within `threshold`
dispatches, while a boot of pure by-design fallbacks files nothing.
*Premise:* the feeder population is 3578's in-scope deliverable ("every CLI-level resume failure emits an event"),
so it is upstream — no DAG inversion.

**Amendment to task 3161** *(not a leaf)* — narrow B6 per D1 and §7.

## 11. Open questions (tactical, not design-blocking)

1. **The numeric value of D3's outer bound.** Deliberately unfixed — a guessed threshold is exactly what G6
   branch 1 rejects. Derive it at implementation from the observed distribution of legitimate task in-flight
   duration in `runs.db` (mirroring 3621's derived-bound test), and pin the derivation, not the number. Decide in δ.
2. **Whether 3578's HARD GATE gets exercised cross-lane.** 3578 is a dark-factory task and prior-PRD decision D10
   makes DF the validation case *precisely because the cwd-move question is vacuous there*. Nothing currently forces
   the reify exercise, so 3578 could close having validated only same-path restore. §9 B2 covers the *lookup* half
   cross-lane; the *CLI-accepts-it* half remains 3578's. Raise as an explicit acceptance criterion when δ dispatches.
3. Whether `resume_count` should reset when a resume is archive-mediated rather than live-dir-mediated. Affects only
   the `capped` path, which is already correctly carved out. Decide in δ.

## 12. Design-invariant walk (G7 — `docs/legibility/design-invariants.md`)

| Invariant | Disposition |
|---|---|
| INV-1 `contracts-machine-checked` | **Satisfied, and this is much of the point.** The eligibility contract currently lives in predicate *ordering* plus a docstring — and P1/P2 show the prose and the behaviour diverged without anything noticing. D5's composite reason set moves it to an enumerable, asserted value; §8 pins the seam signature where its consumer sees it. |
| INV-2 `structured-facts-at-failure` | **Satisfied.** α emits `archive_available` as a structured field, replacing the operator inference "was this recoverable?" — which today is unanswerable without a manual filesystem scan. β emits the full reason set rather than a first-match string that misdirects toward NTP. |
| INV-3 `corroborate-before-acting` | **Satisfied.** D2 is literally this: corroborate reachability against the durable archive instead of acting on a wall-clock proxy. D8 applies it to the rollout — measure the recoverable population in production before changing what resumes. |
| INV-4 `storm-escape-required` | **Satisfied, with the interim window named.** β's carve-out removes the streak's only current feeder, and ε's real feeder cannot exist until 3578 lands. During that window the loud signal is α's `archive_available` rate on every fallback event — a structured, queryable rate on the fail-soft path, which is what INV-4 asks for; the *escalation* returns with ε. Recorded rather than waived because the fallback path stays observable throughout. |
| INV-5 `no-lockstep-duplication` | **Satisfied, explicitly.** D6 (one archive locator, `Path \| None` so 3578 has nothing to re-implement; pinned by B7) and D7 (no fifth `_pid_alive` — the four existing copies are a documented deliberate pattern that reads as licence to add another). |
| INV-6 `status-matches-liveness` | **Satisfied.** γ is this invariant applied to an artifact rather than a status row: a sidecar asserts "an invocation is in flight" (`artifacts.py:39`, presence ⇔ in-flight) while its `owner_pid` has been dead for 176h. D7's second conjunct is what keeps the reaper from racing a legitimate re-dispatch. |
| INV-7 `holds-owned-and-bounded` | **Satisfied — the sharpest instance here.** A zombie sidecar is an unowned, unbounded hold on a resume attempt: nothing exits it, nothing ages it, and it silently degrades every future dispatch of its task. γ gives it an owner (the boot reaper) and a bound (the next process start). |

## 13. META gate

> If I decompose and queue this PRD without further oversight, will the architecture of what gets implemented be
> complete, coherent, cohesive, and good?

**Yes.** Every leaf names a consumer (§1), a signal measurable by a query already run in §2, and a premise validated
against reify's own event store or filesystem rather than assumed — including the one that matters most, that the
transcripts are already on disk and merely unlooked-for. The seam all three prior owners disclaimed has a named
owner (§7), and the one live contradiction (3161 B6 vs 3578) is resolved by ruling rather than left to whichever
task lands second.

The ordering is load-bearing and encoded in the dependency graph, not left to chance: α is pure `shared/` and can
land against the live 3256 claimant; β and γ wait on 3256 because they edit the seam it holds; δ and ε wait on 3578
because eligibility must be gated on restore *succeeding*, not on an archive merely existing — the distinction
between a resumed session and a "No conversation found" that emits no event at all.

The value is front-loaded on purpose. α, β and γ close the escalation, drain its fuel supply, and make the
recoverable population visible, without waiting on the 3618 → 3619 → 3578 chain that δ and ε need.
