# PRD: Durable archival of orchestrator fleet-agent transcripts (+ legibility mining integration)

**Date:** 2026-07-17 · **Status:** approved for decomposition · **Scope:** all
load-bearing code is dark-factory `orchestrator` / `shared` / `scripts/legibility`.
**Approach:** B+H (mechanism count ≥ 5; touches the agent-invocation `finally` seam and
the orchestrator→legibility cross-tool seam).

Cite by symbol; line refs are as-of `main` `d19b3645df` and drift.

## 1. Consumer + user-observable surface (G1, G2)

**Consumer (the code that changes, all in `/home/leo/src/dark-factory`):**
- `shared/src/shared/transcript_archive.py` — **new** module: `archive_task_transcripts(...)`
  primitive (the one helper both the producer hook and the teardown backstop call).
- `orchestrator/src/orchestrator/workflow.py` — `_invoke` `finally` block (currently
  `clear_agent_session()` at ~`workflow.py:8422`): add the producer-side archive call.
- `orchestrator/src/orchestrator/git_ops.py` — `cleanup_worktree` (`git_ops.py:8738`): add the
  idempotent teardown backstop call before `git worktree remove`.
- `orchestrator/src/orchestrator/config.py` — a small `transcript_archive.*` config block
  (enable flag, archive root, GC caps).
- `scripts/legibility/` — `inventory.py` (`enumerate_sessions` / `iter_project_dirs`),
  `config.py` (`load_config`), and `docs/legibility/legibility.yaml`: an additional
  configurable `agent_transcript_roots` the miner enumerates alongside `~/.claude/projects`,
  gz-aware.
- `scripts/` — a retention-GC sweep over the archive dir.

**User-observable surface (what an operator / the legibility program sees after this lands):**
1. After **any** fleet-agent role invocation completes (architect, implementer, reviewers,
   steward, and their subagents), that role's transcript exists at
   `data/orchestrator/agent-transcripts/<task_id>/<session_id>.jsonl.gz` and decompresses to
   the real transcript — and it **survives the task's worktree teardown** (merge finalize,
   crash-recovery reclaim, lane release). Verifiable: run a task to `done`, let the worktree be
   removed, confirm the gz transcripts remain.
2. A confusion-mining census/inventory run (`scripts/legibility`) over a date range with
   archived fleet transcripts **enumerates them** — the fleet's own agents, previously
   invisible, now appear in the mined corpus (classified `orchestrated-task`). With the new
   root knob unset, mining behaves byte-identically to today.
3. `.credentials.json` (the per-task OAuth token) is **never** written to the archive — only
   `projects/**/*.jsonl` transcript files are.
4. Disk stays bounded: a retention GC prunes the archive by age/count cap and **logs what it
   dropped**; at current volume (~2 MB gz/task) the default caps keep everything.

## 2. Motivation / premise validation (G6 — established by code + filesystem, 2026-07-17)

The premise is that today the fleet's own agent transcripts — the highest-volume LLM activity
in the system — are **both destroyed on task completion and invisible to confusion mining**.
Both halves are verified:

1. **Destroyed on completion.** The per-task Claude config dir (holding the CLI transcript
   JSONL) is created **inside** the worktree: `TaskConfigDir(self.task_id, base_dir=self.worktree
   / '.task')` (`workflow.py:1986`) → `<worktree>/.task/claude-config-<id>/projects/*/<sid>.jsonl`
   (`config_dir.py:34-37`; `projects/` kept per-task, `config_dir.py:23`). The `.task-meta`
   relocation (`artifacts.meta_root_for`, `artifacts.py:203-228`) covers only
   metadata/plan/reviews/verdicts — **never** the transcript. The transcript is destroyed by
   `_cleanup_config_dir` → `TaskConfigDir.cleanup()` `rmtree` at every terminal state
   (`workflow.py:5171-5198`, teardown step `cleanup_config_dir` at `workflow.py:2710`), and by
   worktree removal (`git_ops.cleanup_worktree`, `git_ops.py:8738`, which the 12 harness
   reconcile/crash paths delegate to). **Filesystem ground truth:** 425 `.task-meta` entries and
   107 surviving worktrees, but only **4** surviving `claude-config-*` transcript dirs — and
   those 4 are live/forensically-preserved. >99% already gone. No archival exists anywhere (the
   only durable `_archive_*` machinery archives *verify* stdout to `data/verify-logs/`, not the
   transcript).
2. **Invisible to mining.** Every legibility entry point defaults its root to `~/.claude/projects`
   (`inventory.py`, `census.py:974` `DEFAULT_PROJECTS_ROOT`, `nightly.py`, `sampling.py`) and
   enumerates `enumerate_sessions(projects_root, cfg.cwd_prefixes, date)` → `iter_project_dirs`
   → `project_dir.glob('*.jsonl')` (`inventory.py:43-64,188-238`), filtered by each transcript's
   own `cwd` line against `cwd_prefixes` (`legibility.yaml:13-14` = `[/home/leo/src/dark-factory]`).
   Fleet agents run under a **per-task** `CLAUDE_CONFIG_DIR` (`invoke.py` sets
   `env['CLAUDE_CONFIG_DIR']`), so their JSONLs land in the worktree, never `~/.claude/projects`
   — structurally unreachable by the miner. (The `orchestrated-task` stratum is a red herring: it
   only catches interactive `/unblock` sessions run under the *default* config.) Notably the
   miner's `cwd_prefixes` **already** treats `.worktrees` children as members
   (`legibility.yaml` comment; `inventory.is_member`, `inventory.py:67-77`) — so an archived
   fleet transcript, which keeps its worktree `cwd`, passes membership the moment it lands in an
   enumerated root. The only missing piece is a root the miner reads.

No false premise: the fix asserts only capabilities that exist (below), and the mining-visibility
claim is backed by an active enumerate change + a cwd filter that already admits worktree cwds.

## 3. Approach

Archive at the **producer** (session completion), not the destroyers — so archival never
depends on, or races, worktree teardown. Five components, foundation-first.

- **α — Archiver primitive + producer hook (the must-have core).** New shared
  `archive_task_transcripts(config_dir, task_id, session_id=None, *, archive_root)`: resolve the
  session's transcript(s) by **glob-by-session-id** (`<config_dir>/projects/*/<sid>.jsonl` plus
  its sibling `.../subagents/*.jsonl`; when `session_id is None`, all `projects/**/*.jsonl`),
  gzip each to `<archive_root>/<task_id>/<relpath>.jsonl.gz`, **idempotent** (skip when the
  archived copy's size/mtime is current), **best-effort** (any I/O error → logged structured
  fact, never raised), and **credential-safe** (only `*.jsonl` under `projects/`; never
  `.credentials.json` or any non-transcript file). Wire the call into `workflow._invoke`'s
  `finally`, right after `clear_agent_session()`, keyed on the session id just used
  (`self._last_invoke_session_id`). **Signal:** after a role invocation completes, its
  `<archive_root>/<task_id>/<sid>.jsonl.gz` exists and round-trips; a resumed session re-archives
  the grown transcript (last-write-wins); `.credentials.json` is absent from the archive; an
  archive failure is logged and does not fail the task. **Consumer:** β (reuses the helper), γ
  (mining reads the archive), δ (GC prunes it), forensics.
- **β — Teardown backstop at the single chokepoint.** Call the **same**
  `archive_task_transcripts(config_dir, task_id)` (session_id=None → archive-any-unarchived) once
  inside `git_ops.cleanup_worktree`, before `git worktree remove`, best-effort + idempotent.
  Closes the narrow tail where a role was in-flight when the orchestrator died and the task is
  reaped without a completed resume. **Signal:** a worktree removed via `cleanup_worktree` whose
  config dir holds an un-archived transcript has it archived first; an already-archived transcript
  is not re-copied. **Consumer:** forensics, γ. **Depends:** α.
- **γ — Legibility mining: second transcript root, gz-aware, turned ON.** Teach `scripts/legibility`
  to enumerate an additional configurable list of roots (`agent_transcript_roots`, added to
  `legibility.yaml` and `config.load_config`) alongside `~/.claude/projects`, with **gz-aware**
  transcript reading, filtered by the existing `cwd_prefixes` (fleet transcripts carry a worktree
  cwd already admitted by `is_member`). Generalize `enumerate_sessions`/`iter_project_dirs` to
  iterate a root list and accept the archive layout (`<task_id>/<sid>.jsonl.gz`). **This task also
  SETS `agent_transcript_roots: [data/orchestrator/agent-transcripts]` in the committed
  `legibility.yaml`** — live, not empty — so the confusion census reads archived fleet transcripts
  as soon as they exist, with **no operator flip** (Leo's explicit ask: fleet corpus visible ASAP).
  The empty *code default* is retained only as the parity/test baseline. **Signal:** a census/inventory
  run over a date with archived fleet transcripts enumerates them (they enter the corpus, classified
  `orchestrated-task`); the shipped `legibility.yaml` has the archive root set; the empty-list code
  path is byte-identical to today. **Consumer:** the confusion-reduction census/digest; the
  legibility program. **Depends:** α (needs archives to read). ∥ β, δ.
- **δ — Retention GC sweep.** A `scripts/` GC over `<archive_root>` pruning by age and/or count
  cap (`transcript_archive.retention_*` config), best-effort, **loud** (logs each dropped task
  dir + a summary count — INV-4). Runnable standalone and wireable into the existing operator
  cadence. **Signal:** with a low cap, a GC run removes the oldest task dirs beyond the cap and
  logs what it dropped; default caps keep everything at current volume; a run over an empty/absent
  archive is a no-op. **Consumer:** disk hygiene / operator. **Depends:** α (the layout it prunes).
- **ε — End-to-end boundary gate (B+H integration tests)** spanning `shared` + `orchestrator` +
  `scripts/legibility`. The Appendix B table. **Depends:** α, β, γ, δ.

### The load-bearing design decision: hook the producer, not the destroyers

Worktree destruction happens at one real chokepoint (`cleanup_worktree`, delegated to by the 12
harness paths) plus a few genuinely-distinct removals at other granularities/lifecycles
(`config_dir.cleanup` config-dir-only; warm-lane-reuse `rmtree`; `_abort_lane_acquisition`; the
`_iact-*` reaper). Hooking archival onto that set is fragile — a future removal seam silently
drops transcripts (the classic missed-integration-seam failure). Archiving at the **producer**
(`_invoke`'s `finally`, where each session is born-complete) removes the dependency on teardown
entirely: one call site, and the transcript is safe long before any destroyer runs. β adds a
single idempotent backstop at the one true chokepoint only to close the abandoned-in-flight tail
— **1 producer + 1 backstop**, both calling one shared helper (INV-5), never five teardown hooks.

### Rejected alternatives

| Alternative | Why rejected |
|---|---|
| Archive before each of the ~5 teardown/removal sites | Fragile: must enumerate every current *and future* removal seam; races the DONE path (`cleanup_done_worktree` removes the worktree before `_cleanup_config_dir`); crash paths bypass `workflow.run()`. Duplicated call sites (INV-5). Producer-side archival makes teardown coverage moot. |
| Archive the **whole** config dir | Leaks `.credentials.json` (OAuth token) into a durable store and bloats the archive with plugins/backups/shell-snapshots. Archive only `projects/**/*.jsonl`. |
| Redirect fleet agents to write into `~/.claude/projects` (so the existing miner just finds them) | Pollutes the interactive tree with fleet volume, breaks per-task credential isolation (`config_dir.py` rationale), and couples correctness to the CLI's cwd-slug convention. A separate archive root + a config knob keeps the corpora cleanly separable. |
| Compute the transcript path from the cwd-slug formula | The slug is an undocumented CLI-internal convention that drifts across versions; glob-by-session-id (a unique UUID) is version-robust — same rationale as the liveness-resume PRD. |
| Make archival mandatory / raise on failure | Archival must never break a task; best-effort + structured-fact logging + a failure counter (INV-4) is the correct fail-soft. |

## 4. Pre-conditions (G3 — verified on `main` `d19b3645df` this session)

No novel substrate is introduced:

- **Session id + config dir are orchestrator-known.** `session_id_val = str(uuid.uuid4())`
  stashed as `self._last_invoke_session_id` (`workflow.py:8348-8352`); `TaskConfigDir` =
  `<worktree>/.task/claude-config-<task_id>` (`config_dir.py:34-37`), exposed as `.path`;
  `projects/` kept per-task (`config_dir.py:23`). The `finally` that already runs
  `clear_agent_session()` (`workflow.py:8421-8423`) is the producer hook point.
- **Transcript layout on disk** (verified empirically): `<config>/projects/<enc>/<sid>.jsonl`
  plus `<config>/projects/<enc>/<sid>/subagents/agent-*.jsonl`.
- **Teardown chokepoint** for β: `git_ops.cleanup_worktree(worktree, branch)` (`git_ops.py:8738`),
  the single function the harness reconcile/crash family delegates to.
- **Legibility enumerate is already root-parameterized:** `enumerate_sessions(projects_root,
  cwd_prefixes, date)` and `iter_project_dirs(projects_root, cwd_prefixes)` (`inventory.py:43-64,
  188-238`); `is_member` already admits `.worktrees`/`.claude-worktrees` descendants of a prefix
  (`inventory.py:67-77`); config via `scripts/legibility/config.load_config` reading
  `docs/legibility/legibility.yaml`. γ generalizes the single `projects_root` into a root list +
  gz reading — additive.
- **Durable store exists:** `data/orchestrator/` is a live, git-ignored orchestrator data dir
  (holds `runs.db`, `scheduler_state.json`, …); `<...>/agent-transcripts/` is a new subdir.
- **Config plumbing:** `OrchestratorConfig` is the home for the new hot-reloadable
  `transcript_archive.*` block (mirrors the existing green-tier leaf tunables).

**New mechanisms:** the `archive_task_transcripts` primitive (α); the producer hook (α) + backstop
(β); the multi-root gz-aware miner (γ); the retention GC (δ). All wire into existing seams; no new
dispatch path is introduced.

## 5. Resolved design decisions

1. **Archive at session completion (producer), in `_invoke`'s `finally`.** Not at teardown. β adds
   one idempotent backstop at the single `cleanup_worktree` chokepoint for the abandoned-in-flight
   tail. §3 covers the rejected teardown-hook model.
2. **Transcript located by glob-by-session-id** (`projects/*/<sid>.jsonl` + its `subagents/`), not
   a computed cwd-slug — version-robust.
3. **Archive only `projects/**/*.jsonl`; never `.credentials.json`** or other config-dir files.
   Enforced in the helper and asserted in ε.
4. **Best-effort, idempotent, credential-safe, never raises.** Failures emit a structured fact
   (path/task_id/errno) and increment a failure counter/log so a systemic breakage (e.g. disk
   full → every archive fails) is loud, not silent (INV-2, INV-4).
5. **One shared helper** (`archive_task_transcripts`) is the sole archiver; producer (α) and
   backstop (β) both call it — no duplicated archive logic (INV-5).
6. **Destination `data/orchestrator/agent-transcripts/<task_id>/<relpath>.jsonl.gz`**, gzip
   (~3.3× on transcripts). One dir per task; role/session distinguished by filename.
7. **Mining reads the archive via a config-driven root list** (`agent_transcript_roots` in
   `legibility.yaml`), gz-aware, filtered by the existing `cwd_prefixes`, machine-read (INV-1).
   The *code* default is an empty list (byte-parity with today, for tests), but **this batch ships
   `legibility.yaml` with the archive root SET** so the confusion census reads the fleet corpus as
   soon as transcripts exist — no operator action (Leo's ask: visible ASAP). γ owns both the
   enumerate change and turning the root on.
8. **Retention by age and/or count cap, config-driven, loud.** Defaults sized to keep everything
   at current volume; the cap is the disk-safety valve, not an aggressive pruner.
9. **`transcript_archive.*` and `retention.*` are green-tier hot-reloadable config**; the archive
   *root* path is effectively restart-stable (changing it mid-run just starts writing elsewhere —
   documented, not enforced).

## 6. Out of scope

- **Consolidating the worktree-removal paths into one teardown function** — they remove distinct
  things at distinct lifecycle points; not byte-duplication, and producer-side archival makes it
  unnecessary. A separate hygiene refactor if ever wanted; not coupled here.
- **Mining/analysis of the newly-visible fleet corpus** — this PRD makes it *reachable*; the
  confusion-reduction program owns what to *do* with it (codebook, sampling, digests).
- **Backfilling the already-destroyed transcripts** — unrecoverable; archival is forward-only.
  (The 4 surviving live/preserved dirs are incidental.)
- **The Q3a crash-recovery resume-prompt fix (task 2723)** and the fused-memory restart-safety
  batch (tasks 2700–2718) — separate, already filed.
- **Changing reify** — reify agents run through the same orchestrator; nothing reify-specific
  changes, and reify may opt its own worktree roots into `agent_transcript_roots` later.

## 7. Cross-PRD seams (G4)

| Other PRD / seam | Direction | Mechanism | Owner | Status |
|---|---|---|---|---|
| Confusion-reduction / agent-legibility program (`plans/confusion-reduction-prd.md`, `scripts/legibility/*`, `docs/legibility/legibility.yaml`) | this PRD **produces** the archived corpus + the `agent_transcript_roots` knob and enumerate change; the legibility program **consumes** the enlarged corpus | `agent_transcript_roots` config + gz-aware `enumerate_sessions` | this PRD owns the writer + the enumerate/knob (γ); legibility program owns downstream sampling/coding | additive, opt-in; reciprocal consumer |
| Intra-DF `_invoke` `finally` seam (`workflow.py`) | this PRD adds the producer archive call beside `clear_agent_session()` | `archive_task_transcripts(config_dir, task_id, sid)` | this PRD (α owns) | wired by this batch |
| `git_ops.cleanup_worktree` chokepoint | this PRD adds an idempotent backstop | same helper (β) | this PRD (β owns) | wired by this batch |
| FM restart-safety batch σ2717 "session-resume gate" (tasks 2700–2718) | complementary — that batch evaluates reusing the *resume* mechanism for FM recon; this PRD archives *orchestrator* transcripts | none shared (FM recon vs orchestrator invocation) | separate session/batch | independent; do not absorb |
| Existing `data/` GC follow-up (Cockpit C10 18k-record GC ticket) | sibling disk-hygiene concern | this PRD's δ is a *new* sweep over a *new* dir; it does not touch the C10 GC | δ owns the archive-dir GC | independent |

## 8. Decomposition (G5: B+H — contract = §5 + Appendix A; boundary tests = ε)

- **α — Archiver primitive + producer hook (must-have core)**
  (`shared/src/shared/transcript_archive.py` **new**; `orchestrator/workflow.py` `_invoke`
  `finally`; `orchestrator/config.py` `transcript_archive.*`). **Signal:** after a role invocation
  completes, `data/orchestrator/agent-transcripts/<task_id>/<sid>.jsonl.gz` exists and
  decompresses to the role transcript; a resumed session re-archives the grown file;
  `.credentials.json` never appears in the archive; an induced archive I/O error is logged
  (structured) and the task still completes. **Consumer:** β, γ, δ, forensics. **Depends:** —.
- **β — Teardown backstop at `cleanup_worktree`** (`orchestrator/git_ops.py`). Idempotent
  archive-any-unarchived before `git worktree remove`. **Signal:** a `cleanup_worktree` on a
  config dir with an un-archived transcript archives it first; an already-archived transcript is
  not re-copied (size/mtime skip). **Consumer:** forensics, γ. **Depends:** α.
- **γ — Legibility multi-root gz-aware enumerate + turn the root ON**
  (`scripts/legibility/inventory.py`, `config.py`; `docs/legibility/legibility.yaml`). **Signal:**
  a census/inventory run over a date with archived fleet transcripts enumerates them (in-corpus,
  classified `orchestrated-task`); the shipped `legibility.yaml` has
  `agent_transcript_roots: [data/orchestrator/agent-transcripts]` set (census reads the fleet
  corpus with no operator flip); a `.jsonl.gz` under the archive root is read as transcript; the
  empty-list code path is byte-identical to today. **Consumer:** confusion-reduction census/digest.
  **Depends:** α. ∥ β, δ.
- **δ — Retention GC sweep** (`scripts/` + `orchestrator/config.py` `transcript_archive.retention_*`).
  **Signal:** with a low cap, oldest task dirs beyond the cap are removed and each removal + a
  summary count is logged; default caps keep everything; empty/absent archive → no-op. **Consumer:**
  operator / disk hygiene. **Depends:** α. ∥ β, γ.
- **ε — End-to-end boundary gate (B+H)** (tests spanning `shared` + `orchestrator` +
  `scripts/legibility`). The Appendix B table. **Signal:** all rows green — notably credential
  exclusion (E4), teardown survival (E2), backstop-idempotency (E3), and mining enumeration of an
  archived gz transcript (E5). **Consumer:** the user-observable outcome (fleet transcripts durable
  + mine-able). **Depends:** α, β, γ, δ.

**DAG:** α → β; α → γ; α → δ; {β, γ, δ} → ε. (α is the foundation — it defines the archive layout
and the helper the others consume.)

## 9. Open questions (tactical — deferred, not blocking)

1. **Exact GC caps.** Default (e.g. keep 90 days or 5,000 task-dirs, whichever larger). Calibrate
   during δ against real volume; trivial at ~2 MB gz/task on 3.5 TB free.
2. **Incremental vs full re-archive per `finally`.** Default: archive only the current session id's
   file(s) (`session_id` passed), so sibling roles aren't re-copied each invocation. Decide during α.
3. **GC scheduling cadence** — standalone script vs wired into the orchestrator watchdog/operator
   cadence. Default: standalone script + a documented cron/watchdog hook. Decide during δ.
4. **Should the archive carry a tiny per-task index** (role→session id map) for easier forensic
   navigation? The `.task-meta/<id>/agent_session.json` history already covers role↔session; skip
   unless a consumer needs it. Out of scope for correctness.
5. **reify worktree roots in `agent_transcript_roots`** — reify can opt in later via its own
   `legibility.yaml`; no cross-repo change here.

---

## Appendix A — Contract (B+H)

**Archiver primitive (shared, produced by α; the one helper β reuses):**
```
def archive_task_transcripts(
    config_dir: Path,
    task_id: str,
    session_id: str | None = None,
    *,
    archive_root: Path,
) -> int:
    """Copy the task's CLI transcript(s) to a durable, gzipped archive.

    Resolves transcripts by glob-by-session-id:
        <config_dir>/projects/*/<session_id>.jsonl        (+ its subagents/*.jsonl)
    or, when session_id is None, every <config_dir>/projects/**/*.jsonl.
    Writes <archive_root>/<task_id>/<relpath-under-projects>.jsonl.gz.

    - credential-safe: ONLY *.jsonl under projects/ (never .credentials.json).
    - idempotent: skip when the archived copy's size/mtime is already current.
    - best-effort: any I/O / gzip error is logged as a structured fact
      (path, task_id, errno) and increments an archival-failure counter;
      NEVER raises (archival must not break a task).
    Returns the number of transcript files newly archived.
    """
```

**Producer hook (orchestrator, α):** in `workflow._invoke`'s `finally`, after
`clear_agent_session()`: `archive_task_transcripts(self._config_dir.path, self.task_id,
self._last_invoke_session_id, archive_root=<config.transcript_archive.root>)`, guarded by the
enable flag, best-effort.

**Teardown backstop (orchestrator, β):** in `git_ops.cleanup_worktree`, before `git worktree
remove`: `archive_task_transcripts(<worktree>/.task/claude-config-<task_id>, task_id,
archive_root=...)` (session_id=None), best-effort + idempotent (a no-op when α already archived).

**Archive layout:** `data/orchestrator/agent-transcripts/<task_id>/<enc>/<session_id>.jsonl.gz`
(+ `.../subagents/agent-*.jsonl.gz`). One dir per task; roles distinguished by session-id filename.

**Mining root config (scripts/legibility, γ):** `legibility.yaml` is shipped with
`agent_transcript_roots: [data/orchestrator/agent-transcripts]` **set** (the empty list is only
the code default, for parity/tests) — so the census reads the fleet corpus as soon as archives
exist, no operator flip.
`enumerate_sessions(projects_roots: list[Path], cwd_prefixes, date)` iterates each root;
`iter_project_dirs` accepts both the `~/.claude/projects/<enc>/*.jsonl` and archive
`<task_id>/**/*.jsonl.gz` layouts; a `.gz` file is transparently gunzipped for the `cwd`/turn
read. Membership still by `is_member(cwd, cwd_prefixes)` (unchanged — fleet cwds already pass).

**Config block (orchestrator, α+δ; green-tier hot-reloadable):**
```yaml
transcript_archive:
  enabled: true
  root: data/orchestrator/agent-transcripts      # restart-stable in practice
  retention:
    max_age_days: 90
    max_task_dirs: 5000                            # whichever bound hits first prunes
```

## Appendix B — Boundary-test sketch (B+H; ε's observable signal)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| E1 | Session archived at completion | a role `_invoke` completes with a transcript on disk | `<archive_root>/<task_id>/<sid>.jsonl.gz` exists; gunzip round-trips to the transcript; `archive_task_transcripts` returned ≥ 1 |
| E2 | Survives teardown | E1, then the worktree is removed via `cleanup_worktree` | the gz transcript still exists under `<archive_root>` after removal |
| E3 | Backstop idempotent | a config dir with one already-archived and one un-archived transcript, then `cleanup_worktree` | the un-archived one is archived; the already-archived one is byte-unchanged (skipped by size/mtime) |
| E4 | Credential-safe | a config dir containing `.credentials.json` + `projects/*/<sid>.jsonl` | archive contains only `*.jsonl.gz`; `.credentials.json` is absent anywhere under `<archive_root>` |
| E5 | Mining enumerates the archive | an archived `<sid>.jsonl.gz` whose `cwd` is a `.worktrees/<id>` path; shipped `legibility.yaml` has `agent_transcript_roots` set to the archive dir | `enumerate_sessions` yields it (is_member true); classified `orchestrated-task`; the empty-list code default → not yielded (parity with today) |
| E6 | Resumed session re-archives | a session archived, then its JSONL grows (resume) and `_invoke` finishes again | the archived gz reflects the grown transcript (last-write-wins), not the stale one |
| E7 | Archive failure is soft + loud | archive root made unwritable during `_invoke` | the task still completes; a structured archival-failure fact is logged; the failure counter increments |
| E8 | GC prunes by cap, loudly | archive with N > cap task dirs | GC removes oldest (N − cap) dirs; logs each removed dir + a summary count; default caps → no-op |
