# Transcript preservation seam — PRD

**Status:** authored 2026-08-04. Approach **B + H** (contracts + two-way boundary tests) per G5.
**Origin:** investigation of `esc-__session_resume_storm__-4` (dark-factory queue), 2026-08-04.
**Owns:** the seam that `plans/agent-transcript-archival-prd.md` and
`plans/warm-lane-session-resume-prd.md` each declared "disjoint" and neither owns.

---

## 1. Consumer and user-observable surface

**Primary consumer — the legibility toolchain.** `scripts/legibility/inventory.py`,
`scripts/legibility/digest.py`, and the confusion census read
`data/orchestrator/agent-transcripts/`. Their corpus is currently missing every agent
session that was in flight when an orchestrator restarted — roughly 26/day.

**Second consumer — task 3578** (`pending`, critical, ruled by Leo 2026-08-03), which
restores a transcript from that archive to make `--resume` work. It assumes archive
coverage it does not have.

**Third consumer — operators** grepping the archive during incident forensics, and
reading the new ambiguity signal in §7 leaf 3.

**User-observable surface:** after an orchestrator restart with agent sessions in
flight, every one of those sessions has a readable, greppable transcript in the durable
archive. Today, measurably, none of them do.

## 2. Premise (MEASURED 2026-08-04, dark-factory `data/orchestrator/runs.db` + filesystem)

The orchestrator **preserves the resume sidecar at shutdown and destroys the transcript
in the same teardown**, via two code paths that do not know about each other:

- On SIGTERM the cancelled invocation sets `session_preserved = True`
  (`workflow.py:11789`) and keeps `agent_session.json` so recovery can `--resume`.
- `session_preserved` feeds exactly one decision — whether to clear the sidecar
  (`workflow.py:11806`). It has **no** connection to `_preserve_config_dir`, which only
  the zero-output-hang / progress-churn breakers set (`workflow.py:7556`, `:7615`).
- So `run()`'s finally runs the unconditional `cleanup_config_dir` teardown step
  (`workflow.py:3130` → `_cleanup_config_dir` → `TaskConfigDir.cleanup()` →
  `shutil.rmtree`), deleting `.task/claude-config-<id>/` including the JSONL transcript.

The archival PRD's own safety argument has a matching hole. Its producer hook
deliberately re-raises `CancelledError` (`workflow.py:11830-11841`), justified because
"the abandoned-in-flight tail is the explicit job of β/task 2729's idempotent teardown
backstop". That backstop lives in `cleanup_worktree` and fires only at worktree
**removal** — but on this path the worktree is deliberately **retained** for resume.
Producer skips it; backstop never fires; the config dir is deleted anyway.

| Measurement (2026-08-04) | Value |
|---|---|
| `session_resume_fallback`, dark-factory lifetime | 197 |
| `session_resume` (success), dark-factory lifetime | **0** |
| Fleet-wide successes / attempts | 1 / 374 |
| Fallback reasons | `no_transcript` 182 (92%), `stale` 15 (8%) |
| Sessions preserved at the 03:15 shutdown with an archive | **0 of 5** |
| Archive coverage of adopted sessions — lifetime | 153/184 (83%) |
| Archive coverage of adopted sessions — **today's cohort** | **2/16 (12.5%)** |
| Preserved sessions in the 08-01→08-04 journal window | 84 |

**Cost.** ~197 degraded dispatches over 15 days at $2.40 mean invocation cost ≈
$200–470 for dark-factory (~$25–60/day). The larger cost is the legibility hole.

**Illustrative of the stakes:** the recorded rationale for `max_resumes_per_task=3`
survives *only* inside `data/orchestrator/agent-transcripts/2774/` — the PRD deferred the
decision to task γ, and γ's architect `plan.json` exists nowhere else. That rationale
would have been destroyed by the bug this PRD fixes.

## 3. Sketch of approach

**The inversion.** No shutdown hook is reliable — `TimeoutStopUSec=90s` then SIGKILL, and
the watchdog kills wedged units outright. Hardening "archive at shutdown" chases an
asymptote. Instead: **make a durable archive a precondition of deletion, not a step
before it.** Dying early then becomes harmless — nothing was deleted.

**Dropping gzip is what makes that cheap, so it lands first.** `transcript_archive.py:32`
states the producer "offloads the whole call to a worker thread so the CPU-bound
compression never stalls the event loop" — that `await asyncio.to_thread(...)` is
precisely what `CancelledError` kills. Without compression the archive step is a
same-filesystem `os.rename`: O(1) metadata, microseconds, **synchronous and
uncancellable**. No staging area, no deferred compressor, no held state (INV-7).

Measured compression trade (2026-08-04):

| | compressed | uncompressed |
|---|---|---|
| Archive, 4,369 files | 446 MB | 1,771 MB |
| Growth | 36 MB/day | 149 MB/day |
| Steady state @ 90-day retention | ~3.2 GB | ~13.4 GB |

Ratio 3.97x. Free space on the volume: **2.1 TB of 6.8 TB**. The uncompressed
steady-state corpus is 0.6% of free space; the compression buys back 0.5%. Against that
it costs the async-cancellation coupling above, dual-path reader complexity (including
`inventory.py:101-160`, a block that exists solely to normalize `gzip.BadGzipFile` and
truncated-stream errors so they do not misdirect operators), and greppability — the
direct concern for the legibility work.

## 4. Pre-conditions (G3 — all verified 2026-08-04, none assumed)

| Assumed capability | Verification |
|---|---|
| `os.rename` works archive-ward | `.worktrees` and `data/orchestrator` are both device **66312** (`stat -c '%d'`), one filesystem `/dev/nvme2n1p5`. VERIFIED. |
| Readers already accept plain `.jsonl` | `inventory.py:409-421` walks a single `rglob('*.jsonl*')` filtered by suffix — a strict superset covering both forms. `digest.py:86` branches on `.gz`. VERIFIED. |
| `CONFIG_DIR_PREFIX` is importable and is the sole name source | `shared/src/shared/config_dir.py:34`; `TaskConfigDir.__init__` at `:232` is the **only** mkdir site in the tree. VERIFIED. |
| Config dir name derives from `task_id`, not branch | Every production creator passes a `task_id` stem. The comment at `harness.py:3024-3026` claiming the name "embeds the branch" is **factually wrong**; the full branch is `task/<id>`, not a single path component. VERIFIED. |
| Retention knobs are hot-reloadable | `TranscriptArchiveConfig` is green-tier via `RELOADABLE_FIELDS`; `retention` reloads as one atomic leaf. VERIFIED. |
| Complete `.gz` consumer set | `transcript_archive.py` (writer); `scripts/legibility/inventory.py`; `scripts/legibility/digest.py`; `scripts/gc_agent_transcripts.py`. Enumerated by grep over production `*.py`. VERIFIED. |

**Live-work hazard.** Task **3256** is `in-progress` right now (claimant
`run-3248b9e51fe1/3256-04d3f475`, heartbeat 2026-08-04T12:12Z) editing `harness.py`,
`event_store.py`, `config.py`, `test_config.py`, `test_crash_recovery.py`,
`test_session_resume_integration_gate.py`. Leaves 3 and 4 touch those files and
**must** depend on 3256. Leaves 1 and 2 do not and may proceed immediately.

## 5. Resolved design decisions

| # | Decision | Rationale |
|---|---|---|
| D1 | Preserve the **transcript**, not the config dir | The config dir is three things with three lifetimes: credentials container (must die fast), CLI session store, transcript source. One policy currently governs all three. Preserving the whole dir keeps `.credentials.json` on disk — the ground on which task 3578 rejected live-dir preservation twice. |
| D2 | Archive is a **precondition of deletion**, not a preceding step | Robust to every kill path. Failing before archiving simply means nothing was deleted. |
| D3 | **Drop gzip.** Write plain `.jsonl` | Saving is 0.5% of free disk; cost is the async coupling that loses transcripts, dual-path reader complexity, and greppability. Reliability and greppability are the goal. |
| D4 | **Bulk-gunzip the existing 4,369 archives now**; delete `.gz` reader branches immediately | One consistent, fully greppable corpus today. +1.3 GB. Chosen over age-out (which keeps dual-path code live ~90 days). Ruled by Leo 2026-08-04. |
| D5 | Deterministic config-dir resolution; **never stash a non-matching dir** | Stashing the wrong dir converts "no candidate" into "wrong candidate" for no benefit — both end at `no_transcript`. |
| D6 | Detect-and-report at **adoption**; degrade at **eligibility** | The I3 totality contract binds `_session_resume_eligible`, not `_adopt_recovered_session` (separate method, weaker best-effort contract). This is the seam that permits a loud signal without breaking fail-safety. |
| D7 | Ambiguity gets its **own** event + deduped L1, **excluded** from the storm streak | The storm L1's prose tells the operator to check clock skew and reseeds (`harness.py:6096-6105`) — actively misdirecting for this cause. Mirrors the existing `capped` carve-out. |
| D8 | `max_task_dirs` **5,000 → 50,000** | It is a soft cap pruning **oldest-first**, so it silently truncates the 90-day window when it binds. MEASURED: 600 dirs/17 days; mean 35/day → 3,150 @ 90d; recent 7-day 46.6/day → 4,194 (84% of cap); peak 71/day → 6,390, which **bites at ~day 70**. 50,000 is ~15x the recent rate. Ruled by Leo 2026-08-04. |
| D9 | **Amend** task 3578 rather than fold or ignore | Keeps Leo's 08-03 ruling intact while making the archive-coverage gap an explicit blocking dependency instead of an unstated assumption. |
| D10 | dark-factory is the **low-risk validation case** for 3578 | 3578's HARD GATE — does the CLI accept `--resume` against a moved cwd? — is **vacuous here**: no `warm_lane_pool` key in `dark-factory-orchestrator.yaml`, worktrees are task-id-named and retained, so the encoded-cwd component is identical across restarts. Same-path restore. |

## 6. Out of scope

- Making `--resume` actually work — that is task **3578**, amended by this PRD, not replaced.
- Changing warm-lane acquire/reseed semantics. `acquire_lane` always re-seeding from base
  is load-bearing and was rejected as a change target twice (3256, 3578).
- Fallback classification / storm-streak reset — task **3256**, in flight.
- Cross-lane resume (transcript project-dir surgery) — an explicit non-goal of
  `warm-lane-session-resume-prd.md:316-318`.
- A byte-based retention cap (`max_total_bytes`). Considered; deferred as scope. Noted in
  §9 as the more principled backstop should dir-count prove a poor proxy.
- Documenting `session_resume.*` in `OPERATIONS.md`. Real gap (the whole block is absent
  from every operator doc, which is why the resume cap was invisible) but a separate
  docs task.

## 7. Cross-PRD relationship and seam ownership (G4)

| Other PRD / task | Relationship | Seam owner |
|---|---|---|
| `plans/agent-transcript-archival-prd.md` | **Overlapping, not disjoint** — it declares at `:230` that it "keeps the live dir in place — no relocation — so session-resume's read path … unaffected". It does not keep it in place; it documents at `:49-56` that `_cleanup_config_dir` destroys the transcript "at every terminal state". Its `CancelledError` gap (`workflow.py:11830-11841`) is unclosed for retained worktrees. | **This PRD** (leaves 1, 2, 4) |
| `plans/warm-lane-session-resume-prd.md` | **Overlapping, not disjoint** — D5 at `:160` asserts resume "needs the live uncompressed config dir, which this PRD keeps in place". Neither PRD keeps it. | **This PRD** (leaf 3); resume behaviour stays with 3578 |
| Task **3578** (restore-from-archive) | Downstream consumer. Assumes archive coverage that does not exist for its target population. | 3578, amended; `depends_on += leaf 1` |
| Task **3256** (fallback classification) | File-level collision on `harness.py`/`event_store.py`/`config.py`. | 3256; leaves 3 and 4 depend on it |

Both prior PRDs asserted disjointness reciprocally, which is exactly the G4 pattern
("the other owns it") that leaves a seam unowned. This PRD resolves it by claiming it.

## 8. Decomposition plan

Each leaf names its user-observable signal (G2) and its validated premise (G6).

**Leaf 1 — drop gzip; one corpus, plain and greppable.** *(no dependency; land first)*
Writer emits `<sid>.jsonl`. One-off bulk-gunzip of the 4,369 existing archives. Delete
the `.gz` branches from `inventory.py` (incl. the `:101-160` error-normalization block),
`digest.py`, and `gc_agent_transcripts.py`.
*Signal:* `find data/orchestrator/agent-transcripts -name '*.gz' | wc -l` returns **0**;
`rg <string> data/orchestrator/agent-transcripts` matches transcript content directly
with no `zcat`; neither `inventory.py` nor `digest.py` imports `gzip`.
*Premise:* MEASURED 446 MB → 1,771 MB, +1.3 GB one-off, 2.1 TB free.

**Leaf 2 — transcript preservation as a precondition of config-dir deletion.**
*(depends on leaf 1)*
`_cleanup_config_dir` and the `cleanup_worktree` backstop must not delete a transcript
lacking a current durable archive. With gzip gone the archive step is a synchronous
same-filesystem `os.rename` — uncancellable. Add a startup sweeper for the SIGKILL tail
(nothing else can cover it). Per INV-4, archival failure carries a rate/streak escalation.
*Signal:* after a SIGTERM restart with sessions in flight, **every** preserved session's
transcript is present at `data/orchestrator/agent-transcripts/<task_id>/<enc>/<sid>.jsonl`.
Re-run the §2 coverage query: today's-cohort coverage goes 12.5% → 100%.
*Premise:* MEASURED 0-of-5 today; same-device rename VERIFIED.

**Leaf 3 — deterministic config-dir resolution + ambiguity signal.** *(depends on 3256)*
Derive `expected = entry/'.task'/f'{CONFIG_DIR_PREFIX}{key}'`, importing the constant so
creator and resolver provably share one string (INV-5). Stash iff it exists; never stash a
non-matching dir (D5). Emit a structured `session_config_dir_ambiguous` event carrying
`{expected, found[], session_id, task_id}` (INV-2) plus a separately-deduped L1 (INV-4),
excluded from the storm streak (D7).
*Signal:* worktree `3464` — which holds **only** `claude-config-3464-unblock` — today
stashes that wrong dir silently (`len==1`, so the `>1` warning at `harness.py:3032` never
fires) and yields a guaranteed `no_transcript` with zero operator signal. After this leaf
it emits the ambiguity event and stashes nothing.
*Premise:* MEASURED on disk — `.worktrees/3464/.task/claude-config-3464-unblock` exists
with no sibling; `.worktrees/2971/.task/claude-config-df_task_13` is a real foreign-task dir.
*Note:* `test_crash_recovery.py:144` and `test_session_resume_integration_gate.py:166` name
fixtures `claude-config-<session_id>` and will break. That breakage is the signal the
fixtures encode the wrong contract — fix the fixtures, not the resolver.

**Leaf 4 — retention sizing so age is the only binding policy.** *(depends on 3256)*
`max_task_dirs` 5,000 → 50,000, with a test asserting the bound is derived
(`max_task_dirs ≥ 90 × observed_peak_daily_rate × safety_factor`) rather than a magic
number, so the next throughput increase re-derives it instead of silently truncating.
*Signal:* with the archive at steady state the GC prunes on **age**, never on dir count;
the derived-bound test fails if the cap could bite inside 90 days at the measured peak rate.
*Premise:* MEASURED 600 dirs/17 days; peak 71/day → 6,390 @ 90d, exceeding the current
5,000 cap at ~day 70.

**Amendment to task 3578** *(not a new leaf)* — record that archive coverage is a hard
precondition it currently assumes (12.5% for its target population), wire
`depends_on += leaf 2`, and note D10 (its cwd HARD GATE is vacuous for dark-factory, making
DF the low-risk validation case).

## 9. Open questions (tactical, not design-blocking)

1. Whether `--resume` needs anything beyond `projects/<enc>/<sid>.jsonl` — `sessions/` was
   empty in the one populated config dir inspected. Encouraging, not proof. Owned by 3578's
   HARD GATE; does not block leaves 1–4.
2. Whether today's 12.5% archive coverage partly reflects cohort youth (older days reached
   ~90% once tasks completed and the `cleanup_worktree` backstop fired). Leaf 2 makes the
   question moot rather than answering it.
3. Whether a byte cap (`max_total_bytes`) should eventually replace dir-count as the
   runaway backstop — bytes are what actually run out. Deferred (§6).
4. Exact staging semantics if a future deployment ever puts `worktree_base` on a different
   filesystem from `data/` — `os.rename` would become `EXDEV`. Not reachable today
   (device 66312 for both); the implementation should fall back to copy+unlink rather than
   assume.

## 10. Design-invariant walk (G7)

| Invariant | Disposition |
|---|---|
| INV-1 `contracts-machine-checked` | **Satisfied, and this is the point.** The config-dir naming contract currently lives in prose — and the prose is wrong (`harness.py:3024-3026`). Leaf 3 moves it to a shared constant plus derivation. Leaf 2's "never delete un-archived data" is enforced in code, not documented. |
| INV-2 `structured-facts-at-failure` | **Satisfied.** Leaf 3 emits `{expected, found[], session_id, task_id}` as structured fields rather than the current prose WARNING an operator would have to scrape. |
| INV-3 `corroborate-before-acting` | **Satisfied.** Leaf 2 is literally this: before deleting, corroborate that a current durable archive exists. |
| INV-4 `storm-escape-required` | **Satisfied by design.** Leaf 2's archival-failure path carries a rate/streak escalation; leaf 3's ambiguity carries its own deduped L1, deliberately separate from the storm streak whose prose would misdirect (D7). |
| INV-5 `no-lockstep-duplication` | **Satisfied.** `CONFIG_DIR_PREFIX` is imported, not restated. Precedent to mirror: `fused-memory/.../cli_stage_runner.py:357-381` (`recon_config_base_dir`/`gc_run_config_dir`), pinned by a test that asserts the creator builds exactly that path. |
| INV-6 `status-matches-liveness` | **N/A.** No task status transitions introduced. |
| INV-7 `holds-owned-and-bounded` | **Satisfied by construction.** The naive design (rename-aside → compress later) would create a held state needing an owner and a bound. Dropping gzip (leaf 1) collapses it: the rename *is* the archive. The only remaining held state is the SIGKILL tail, owned by the startup sweeper and bounded by the next process start. |

## 11. META gate

> If I decompose and queue this PRD without further oversight, will the architecture of
> what gets implemented be complete, coherent, cohesive, and good?

**Yes.** Every leaf has a named consumer (§1), a user-observable signal measurable by a
query already run in §2, and a premise validated against the filesystem or the event store
rather than assumed. The seam both prior PRDs disclaimed has a named owner (§7). The two
live-work collisions (3256) and the one downstream consumer (3578) are wired as explicit
dependencies. No open design questions remain — §9 is tactical.

The ordering is load-bearing and is encoded in the dependency graph rather than left to
chance: leaf 1 (drop gzip) simplifies leaf 2 from a staging-plus-compressor design into a
single atomic rename, and leaves 3 and 4 wait on 3256 to avoid a file-lock collision with
work in flight right now.
