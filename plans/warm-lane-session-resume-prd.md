# PRD: Session-resume for warm-lane tasks across orchestrator restarts

Status: **active** — authored 2026-07-18 from the reify-side investigation brief
(12 old in-progress tasks, 6-agent log-mining sweep) + a fresh substrate
verification pass against `main@8a250e33f3`. Approach: **B + H** (contract +
boundary-test sketch — the restart path is a load-bearing seam).

## 1. Goal

After an orchestrator restart — scheduled 8h fleet redeploy **or** crash
revive — a warm-lane task whose agent was in flight resumes its prior Claude
session (`--resume <session_id>`) on re-dispatch instead of paying a fresh
agent spin-up: the journal shows `resuming prior session <id>` for the task,
the recovered plan skips the architect, and the resumed agent continues from
its own transcript instead of re-deriving orientation.

**Measured cost this removes** (reify runs.db, pre/post-throttle): 12
in-progress tasks at 4.5% utilization (43 h active in 953 h wall); 14–34
kill/re-dispatch cycles per long-lived task pre-throttle; 1–8.5 min
re-orientation tax per fresh session (worst observed: 163 s hunting for
`.task-meta`); in-flight verify runs (15–71 min) always lost. Post-throttle
the dominant cost is phase-per-8h-window pipelining — resume lets a task
continue mid-attempt instead of waiting out a fresh-dispatch cycle.

## 2. Background — what exists, what actually breaks

Substrate verified 2026-07-18 (all claims re-checked against `main`):

- **The resume primitive exists and is proven.** `_invoke` pre-mints a
  session UUID, writes the `agent_session.json` sidecar before the subprocess
  starts, and on the next invocation of the same role passes
  `resume_session_id` → `--resume` via `invoke_with_cap_retry`
  (`workflow.py:8337-8380`, `cli_invoke.py`). cli_invoke owns the
  resume-continuation prompt (`CRASH_RECOVERY_RESUME_PROMPT`, enriched with
  reassess-state framing by task 2723, merged `c52ab15946`) and owns the
  resume-failure → fresh-invocation fallback that restores the real task
  prompt (task 1462 contract). Cold worktrees already use this across
  restarts via `_recovered_sessions` → `TaskWorkflow(resume_session_id=…)`
  (`harness.py:2614-2627`, `:5771`, `:5849`).
- **The session state already survives restart + same-task re-acquire.** The
  per-task `CLAUDE_CONFIG_DIR` (`<lane>/.task/claude-config-<task_id>/`) and
  the session JSONL live under the git-ignored `<lane>/.task/`. On
  re-dispatch after a restart, `acquire_warm_lane`'s `DISK_BACKSTOP_REUSE`
  route (git_ops.py ~4414-4460) matches the lane's
  `.task-meta/_lane-N/plan.json` `task_id`, routes to `_reuse_warm_lane`,
  and **deliberately preserves `.task/`** (its same-task contract) — no
  reseed, no `git clean`. The reseed (`_reset_warm_lane`:
  `checkout -f -B` + `git clean -xfd` + CoW re-seed) runs only for a
  *different* task acquiring the lane, which is exactly when destroying the
  previous occupant's session state is correct.
- **What actually breaks is the binding, in two places:**
  1. **The sidecar is cleared on cancellation.** `_invoke`'s `finally` runs
     `clear_agent_session()` unconditionally (`workflow.py:8443-8445`). A
     SIGTERM restart (`cli.py:194` `main_task.cancel()` → CancelledError
     propagates; `TimeoutStopSec=90` gives the async cleanup room —
     `orchestrator-*.service:50`) therefore **destroys the resume binding on
     every clean restart**, while a SIGKILL crash preserves it — inverted
     from the intent.
  2. **Recovery consults the sidecar only in the no-plan branch.**
     `_recover_crashed_tasks` checks `plan.json` first; any worktree (lane
     *or* cold) with a plan gets plan-resume with a **fresh session** — the
     sidecar is never read (`harness.py:2589-2635`). For lanes without a
     plan, the sidecar is explicitly given up on ("carries … NO task_id …
     Release it back to the pool", `harness.py:2597-2612`) even though the
     sibling `plan.json` in the same `.task-meta/_lane-N/` carries the
     `task_id` when present.

  So today's "session resume across restart" fires only for a **cold
  worktree that crashed before the architect wrote a plan** — the rarest,
  cheapest case. The dominant measured cost (mid-implementer restart of a
  warm-lane task, plan present) never resumes.

- **The restart trigger is out of scope and unchanged.** The drain gate
  (`restart-all-orchestrators.sh --drain`, fleet-redeploy PRD) is
  merge-drain only; in-flight agents are SIGTERM-cancelled by design. The
  in-flight slot persists a synthetic `TaskReport(outcome=CANCELLED)` and
  deliberately does **not** release the warm lane (`harness.py:5927-5958`).
  Branch/WIP retention across the restart is already owned by the
  branch-lifecycle-decouple PRD (γ-reattach, `_branch_has_commits_beyond_main`
  guards) — committed work survives; `_reuse_warm_lane` commits WIP.

## 3. Sketch of approach

Make the binding as durable as the state it points to. Four small mechanisms,
all on existing seams — no new dispatch machinery, no lane-lifecycle changes,
no reseed-policy changes (reify D10 "always-re-seed-at-acquire" and all 11
pool invariants untouched):

1. **Sidecar schema v2** (`agent_session.json`): add `task_id`,
   `resume_count`, and `schema_version` to the payload written by
   `write_agent_session` (artifacts.py:728-738); read side accepts v1
   (missing fields → legacy semantics). A typed model, not ad-hoc dict keys
   (INV-1).
2. **Preserve-on-cancellation**: `_invoke`'s `finally` clears the sidecar
   only on completion; on `CancelledError` it is left in place (its presence
   ⇔ "agent was in flight when the orchestrator exited" — the comment at
   workflow.py:8343 already states this contract; today's code violates it
   for the clean-SIGTERM case). Emit a structured `agent_session_preserved`
   fact (task_id, session_id, role) at the preserve point (INV-2).
3. **Recovery adopts sessions with or without a plan, lanes included.**
   `_recover_crashed_tasks`: read the sidecar alongside `plan.json`; map a
   lane to its task via sidecar `task_id` (v2) or the sibling `plan.json`'s
   `task_id` (v1 fallback); populate `_recovered_sessions[task_id]`
   together with `_recovered_plans[task_id]`. Lane disposition is
   unchanged (plan recovered, lane left FREE — the W11 "never silently
   re-pin" decision stands); the existing `DISK_BACKSTOP_REUSE` route
   already gives same-task lane affinity at re-dispatch.
4. **Guarded injection.** `_run_slot` already injects
   `resume_session_id=recovered_session` (`harness.py:5849`). Add the guard
   rails, mirroring the ratified fm-restart-survey σ decisions:
   - **Corroborate before acting** (INV-3): before injecting, verify the
     session transcript actually exists under the lane's
     `CLAUDE_CONFIG_DIR` (glob-by-session-id, the version-robust idiom) and
     the sidecar is within the freshness window; otherwise fresh dispatch.
   - **One-knob kill switch + caps**: `session_resume.*` config block
     (green-tier hot-reloadable): `enabled` (default true),
     `freshness_window_secs` (default 86400), `max_resumes_per_task`
     (default 3 — an 8h cadence means a 30h task legitimately resumes ~3×).
     Cap exceeded → fresh dispatch + `session_resume_capped` event.
   - **Storm escape** (INV-4): count resume-fallbacks per boot; a streak
     above threshold files an L1 escalation instead of silently degrading
     to fresh dispatches forever.
   - **L0-dismissal note**: extend `CRASH_RECOVERY_RESUME_PROMPT`
     (cli_invoke, one string — same seam 2723 used) with one sentence: any
     escalation filed before the restart may have been auto-dismissed
     (`_dismiss_stale_escalations`, harness.py:7507-7530) — re-raise if
     still relevant. Do **not** flip `resume_delivers_prompt` for this
     caller (task 1462 regression class).

**Crash and scheduled restarts are covered uniformly.** The mechanism keys
on sidecar presence: a crash already leaves it (finally never ran); after
mechanism 2, a clean SIGTERM leaves it too. Mid-tool-call crash semantics
are exactly what the existing cold-worktree no-plan resume already accepts,
now with the 2723 reassess-state framing plus the corroboration/fallback
guards. No drain-window changes; suspension is implicit (synthetic
CANCELLED report + preserved sidecar), not a new outcome enum value.

**What resume does NOT restore:** in-flight verify runs always re-run
(stateless by design); uncommitted WIP handling is unchanged (owned by the
reuse/reattach routes — `_reuse_warm_lane` commits it); a resumed session
runs under the pre-restart system prompt by construction (`--resume` skips
it) — a deploy that changes agent prompts/tooling can set
`session_resume.enabled=false` for one cycle (the σ2717
`resume_after_restart` precedent).

**Lane-collision fallback:** if a *different* task acquired the lane before
re-dispatch (reseed destroyed the session state), the corroboration check
fails and the task takes today's path — recovered plan, fresh session. At
4.5% utilization collisions are rare; we measure before strengthening
affinity (open question 3).

## 4. Resolved design decisions

| # | Question (from the brief) | Decision |
|---|---|---|
| D1 | Where does the durable session↔task binding live? | Sidecar `agent_session.json` v2 gains `task_id` (+ `resume_count`, `schema_version`); the sibling `plan.json`'s `task_id` is the v1-compat mapping for plan-bearing lanes. No new files; the sidecar already lives in reseed-surviving `.task-meta/<name>/`. |
| D2 | Reseed policy on recovery-acquire? | **Neither** brief option. No reseed change and no CLAUDE_CONFIG_DIR relocation: the existing `DISK_BACKSTOP_REUSE` route already re-acquires the same lane without reseed and preserves `.task/`. D10 always-re-seed applies to foreign-task acquisition, where destroying session state is correct. (Relocation to `.task-meta` buys nothing: `--resume` resolves the transcript under the *cwd-keyed* project dir, so a resumed session needs the same lane path anyway.) |
| D3 | Scope: scheduled only vs also crash? | Uniform. The mechanism keys on sidecar presence, which covers both after preserve-on-cancellation lands. Guards (corroborate, freshness, cap, fallback) make a bad resume degrade to today's behavior. |
| D4 | SIGTERM drain interaction? | None needed. `TimeoutStopSec=90` already gives `finally` blocks room; mechanism 2 rides inside the existing shutdown. No new "suspend" outcome — synthetic CANCELLED stays (a typed INTERRUPTED outcome is W9's seam, see §6). |
| D5 | Relationship to transcript archival? | Non-overlapping, both named in §6: archival gz-**copies** transcripts to `data/orchestrator/agent-transcripts/` for forensics/mining; resume needs the **live** uncompressed config dir, which this PRD keeps in place (no relocation → archival's producer hook and teardown backstop are untouched). |
| D6 | Verify runs in flight at restart? | Always re-run. Verify is stateless-restartable; resuming it is complexity with no payoff. |
| D7 | L0 auto-dismissal interaction? | One sentence added to the shared `CRASH_RECOVERY_RESUME_PROMPT` telling the resumed agent its pre-restart escalations may have been auto-dismissed. |
| D8 | Resume prompt / fallback semantics? | Inherit wholesale: cli_invoke owns the prompt swap (2723's enriched constant) and the resume-failure → fresh-fallback with real-prompt restore (1462). `resume_delivers_prompt` stays False for this caller. |
| D9 | Backend scope? | Backend-agnostic: `resume_session_id` flows through `invoke_agent` to both the claude (`--resume`) and pi (`--session`) backends; pi session state lives in the same preserved `<lane>/.task/`. Tested primarily on the claude backend (open question 4). |

## 5. Pre-conditions

All landed on `main` — verified 2026-07-18:

- Resume lifecycle + sidecar (`agent-liveness-telemetry-resume-prd.md`, task 1780 done).
- Enriched crash-recovery resume prompt (task 2723, done, `c52ab15946`).
- Branch retention across restart/hard-cancel (`warm-lane-branch-lifecycle-decouple-prd.md`, tasks 1912-1915, 1923 done).
- `.task-meta` sibling relocation + `DISK_BACKSTOP_REUSE` route (W11 ε1 lineage, deployed via 2424/2263).
- 8h restart throttle (tasks 2371/2396/2398) — bounds resume frequency.

No novel substrate. **G3: no unverified capability assumptions remain.**

## 6. Cross-PRD relationship (G4)

| Other PRD / doc | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/agent-liveness-telemetry-resume-prd.md` | consumes | `--resume` lifecycle, sidecar write/read, glob-by-session-id | other (landed) | wired |
| `plans/warm-lane-branch-lifecycle-decouple-prd.md` | consumes | branch/WIP survival across restart (γ-reattach, commit-WIP) | other (landed) | wired |
| `plans/worktree-lane-lifecycle-prd.md` (W11) | produces-into | `_recover_crashed_tasks` sidecar-adopt behavior must carry into W11's record-driven adopt/quarantine rewrite | this PRD owns the adopt logic; W11 owns the rewrite; companion task ψ updates the W11 PRD prose | queued (ψ) |
| `plans/agent-transcript-archival-prd.md` (batch live, α refile 2742 in pipeline) | disjoint | gz archive = forensics copy; live config dir = resume substrate. No shared code; no relocation → producer hook + teardown backstop unaffected | each its own | wired |
| `plans/fused-memory-restart-survey-2026-07-17.md` (σ 2717) | pattern-sibling | shared `cli_invoke` resume seam; σ's ratified guard-rail decisions mirrored here. Independent batches — do not absorb (reciprocal note already in that PRD) | each its own | wired |
| `plans/orchestrator-fleet-redeploy-throughput-prd.md` | consumes | SIGTERM/drain trigger + `TimeoutStopSec=90` window | other (landed) | wired |
| `plans/workflow-state-machine-prd.md` (W9, unfiled batch) | future consumer | typed `StewardOutcome.INTERRUPTED` → resume-plan path would subsume the implicit CANCELLED+sidecar convention | other (future) | noted |
| reify `docs/prds/warm-lane-pool-cow-seeding.md` | consumes | D10 always-re-seed + 11 pool invariants — deliberately untouched | other (landed) | wired |

## 7. Contract (B+H) — seam signatures + invariants

**Sidecar v2** (`TaskArtifacts.write_agent_session` / `read_agent_session`):

```
agent_session.json = {
  schema_version: 2,
  session_id: str,      # the CLI session UUID (pre-minted, --session-id)
  role: str,            # dispatch role name; resume fires when this role is next invoked
  started_at: iso8601,
  owner_pid: int,
  task_id: str,         # NEW — the durable lane→task binding
  resume_count: int,    # NEW — incremented on each adopted resume
}
```

Invariants:
- **I1 (presence ⇔ in-flight):** the sidecar exists iff an agent invocation
  was started and has not *completed*. Cancellation is not completion.
- **I2 (binding durability):** the sidecar and the session transcript it
  points to survive every same-task path (SIGTERM restart, crash, disk-backstop
  reuse) and are destroyed together on every foreign-task path
  (`_clear_foreign_meta_root` + reseed). Never one without the other
  observable-by-recovery (corroboration closes the race window).
- **I3 (resume-or-fallback totality):** every dispatch of a task with a
  recovered session either injects a corroborated `resume_session_id` or
  falls back to fresh dispatch with the recovered plan — never a stall, never
  an error surfaced to the scheduler. (`resolve → inject` is fail-safe like
  the routing resolver: mis-state degrades, never blocks.)
- **I4 (no prompt-ownership drift):** `resume_delivers_prompt` remains False
  from this caller; cli_invoke's swap + fresh-fallback restore stay the
  single owner of resume-prompt semantics.
- **I5 (lane invariants untouched):** no change to acquire/reseed/release
  semantics; reify pool invariants 1-11 hold byte-for-byte.

Resume-eligibility predicate (evaluated in `_run_slot` before injection):

```
eligible(task) :=
  session_resume.enabled
  ∧ sidecar(task) fresh (now - started_at < freshness_window_secs)
  ∧ sidecar.resume_count < max_resumes_per_task
  ∧ transcript_exists(config_dir(task), sidecar.session_id)   # glob-by-session-id
```

## 8. Boundary-test sketch (B+H)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | Clean SIGTERM mid-implementer, warm lane | plan.json + sidecar v2 present; lane assigned | sidecar survives shutdown; boot adopts session+plan; re-dispatch takes DISK_BACKSTOP_REUSE; journal `resuming prior session <id>`; architect not invoked |
| B2 | SIGKILL crash mid-implementer, warm lane | same | identical to B1 (uniform-scope proof) |
| B3 | Restart mid-architect, warm lane (no plan yet) | sidecar v2 only | lane mapped via sidecar task_id; session adopted; lane released only if sidecar unreadable/stale |
| B4 | Foreign task acquires the lane first | recovered session for task A; task B acquires lane | B's acquire reseeds + clears meta root; A's corroboration fails → fresh dispatch + recovered plan; `session_resume_fallback` event with reason |
| B5 | Stale sidecar (beyond freshness window) | sidecar older than window | fresh dispatch; fallback event; no `--resume` flag in invocation record |
| B6 | Kill switch | `session_resume.enabled=false` | fresh dispatch everywhere; no adoption at boot |
| B7 | Resume cap | `resume_count == max_resumes_per_task` | fresh dispatch + `session_resume_capped` event |
| B8 | Resume-failure fallback storm | N consecutive fallbacks in one boot | L1 escalation filed (streak, not per-event) |
| B9 | Cold worktree, plan present | today: fresh session | after: session adopted (β widens the cold path too) |
| B10 | Completion still clears | invocation completes normally | sidecar removed; no adoption on next boot |
| B11 | v1 sidecar compat | pre-deploy sidecar (no task_id) on a plan-bearing lane | mapped via plan.json task_id; adopted; rewritten as v2 on next write |

## 9. Decomposition plan (G2 signals; ids at decompose time)

- **α — Sidecar v2 + preserve-on-cancellation** (intermediate → unlocks β, γ).
  Modules: `orchestrator/artifacts.py`, `orchestrator/workflow.py`.
  Typed sidecar model (v2 fields, v1-tolerant reader); `_invoke` finally
  clears only on completion, preserves + emits `agent_session_preserved` on
  CancelledError. Signal: after a SIGTERM restart of a live invocation, the
  lane's `.task-meta/_lane-N/agent_session.json` exists with
  `task_id`/`schema_version:2` and the structured preserve fact is in the
  journal; after normal completion it is absent (B10).
- **β — Recovery adopts sessions alongside plans, lanes included**
  (intermediate → unlocks γ, ω). Modules: `orchestrator/harness.py`.
  Consult sidecar in the plan-present branch; lane→task mapping via sidecar
  v2 / plan.json fallback; populate `_recovered_sessions` + keep existing
  lane disposition. Signal: boot journal logs session adoption for a
  plan-bearing lane (B1/B3/B9/B11 shapes); a sidecar-less lane behaves
  exactly as today.
- **γ — Guarded injection + config + events + prompt note** (intermediate →
  unlocks ω). Modules: `orchestrator/harness.py` (`_run_slot`),
  `orchestrator/config.py`, `shared/cli_invoke.py` (one string).
  Eligibility predicate (corroborate transcript, freshness, cap),
  `session_resume.*` green-tier config block, `session_resume_*` events,
  fallback-storm L1 streak escalation, L0-dismissal sentence in
  `CRASH_RECOVERY_RESUME_PROMPT`. Signal: with the knob off, no `--resume`
  is ever injected (B6); with it on, an eligible task's invocation record
  carries the pre-restart session id; an ineligible one emits a
  reason-carrying fallback event (B4/B5/B7).
- **ω — Integration gate** (leaf; the G2 user-observable, two-way).
  Modules: `orchestrator/tests/`. Drives B1-B11 end-to-end style (restart
  simulation per existing recovery-test idiom): restart mid-implementer →
  same session id in the resumed invocation, architect skipped, WIP commit
  present; foreign-acquire and stale paths fall back cleanly. Depends: α, β, γ.
- **ε — Deploy capstone** (leaf, `task_kind='deterministic'` auto-deploy
  preset, `before_done.script=scripts/restart-all-orchestrators.sh`
  `args=['--drain']`). Depends: ω. Signal: fleet units restart onto the new
  code (fresh `ActiveEnterTimestamp`); the *next* scheduled redeploy after
  this one produces `resuming prior session` lines for in-flight warm-lane
  tasks (the self-demonstrating deploy).
- **ψ — Companion cross-PRD prose update** (leaf, docs-only,
  `complexity=simple`). Update `plans/worktree-lane-lifecycle-prd.md`
  (record-driven adopt must preserve the sidecar-adopt behavior; pointer to
  this PRD) and `plans/agent-transcript-archival-prd.md` (note the live-dir
  non-overlap). Signal: both PRDs name this PRD's seam in their cross-PRD
  tables; `git log` shows the docs commit. Depends: γ (text final only after
  mechanism names settle).

DAG: α → β → γ → ω → ε; γ → ψ.

G7 walk (advisory, author mode): INV-1 — sidecar contract is a typed
versioned model, not prose (α). INV-2 — preserve/adopt/inject/fallback all
emit structured facts with fields at the decision point (α, γ). INV-3 — the
eligibility predicate corroborates snapshot state (sidecar) against ground
truth (fs transcript, freshness) before acting (γ). INV-4 — kill switch +
cap + fallback-storm streak escalation; no silent-degradation path lacks an
escape (γ). INV-5 — resume prompt/fallback logic stays single-owned in
cli_invoke; recovery reuses `_resolve_recovery_artifact`; no duplicated
resume machinery.

## 10. Out of scope

- A `SUSPENDED`/`INTERRUPTED` outcome enum value — W9's typed-outcome seam;
  the synthetic CANCELLED report + preserved sidecar convention stands until
  then.
- Relocating `CLAUDE_CONFIG_DIR` out of the lane — pointless without
  cross-lane resume, which the cwd-keyed transcript layout precludes cheaply
  (D2).
- Cross-lane resume (transcript project-dir surgery to move a session to a
  different lane path).
- Resuming verify runs, drain-window/queue changes, restart-trigger changes.
- FM reconciliation resume — σ2717's batch (independent, pattern-sibling).
- Strengthened lane-affinity scheduling (measure collision rate first — open
  question 3).

## 11. Open questions (tactical)

1. **Freshness window default.** 86400s proposed (invocation absolute cap +
   slack). Decide during γ against `invocation_timeout` config reality.
2. **Resume-cap default.** 3 proposed (8h cadence × multi-day task). Decide
   during γ; per-boot storm threshold likewise.
3. **Lane-collision rate.** If fallback events show foreign-acquire evictions
   are common, a follow-up may teach lane selection to prefer non-bound lanes
   for new tasks. Measure via `session_resume_fallback` reasons post-deploy.
4. **pi-backend resume parity.** `--session` reuse is plumbed; whether pi
   sessions resume as cleanly is untested. ω tests the claude backend;
   file a follow-up if pi-backed roles are ever warm-lane-dispatched.
5. **Should ε's deploy self-observe?** The deploy that activates resume is
   itself a restart that cancels in-flight work one last time (its sidecars
   were written by old code — v1, no task_id; plan-bearing lanes still adopt
   via the plan.json fallback, B11). No action needed; noted so the deploy
   verifier doesn't misread partial resume coverage on the first cycle.

## 12. Amendment 2026-08-03 — the reseeded-lane fallback is EXPECTED (task 3256)

Recorded after the fact; the sections above are left as authored.

**Ruling (2026-07-30, reify escalation `esc-__session_resume_storm__-4`).**
Option A — suppress at the classification seam — was chosen. Option C —
change warm-lane acquire semantics to preserve `.task/` across a reseed —
was REJECTED: always-reseed-from-base is load-bearing per
`docs/prds/warm-lane-pool-cow-seeding.md` §9.3/§9.5, so the transcript
store's destruction is a consequence of a deliberate invariant, not a bug
to fix in acquire.

**Mechanism.** Transcripts live INSIDE the lane
(`<lane>/.task/claude-config-*/projects/<slug>/<sid>.jsonl`). A session
adopted at boot has its config dir stashed; the lane is then released and
the next `acquire_lane` re-seeds it, destroying `.task/` — `git clean -xfd`
on the RECYCLE route (`.task/` is gitignored and not in
`reap_build_artifact_dirs`), `rmtree(lane/'.task')` on
RESET_IN_PLACE_REATTACH, `rmtree(lane)` on CREATE_ONCE_FRESH. The
dispatch-time re-glob then legitimately fails. That is INV-3 working
correctly against a by-design wipe, not a corroboration failure.

**New reason: `reseeded`.** `_session_resume_eligible` splits a failed
corroboration on whether the stashed config dir still exists on disk. Gone
⇒ `reseeded` (the whole store was wiped); survives with only this session's
transcript missing ⇒ `no_transcript`. A never-stashed config dir also stays
`no_transcript`: a reseed clears the out-of-lane meta root together with the
lane (I2), so it destroys the sidecar WITH the transcript and yields no
adoption at all — an adopted sidecar without a config dir is pathological
and stays loud.

- **B4 (§8)** splits. Its existing row is the never-stashed shape
  (`with_transcript=False`) and keeps `reason='no_transcript'`. A new B4'
  covers the dominant case: the transcript DID corroborate at boot, the lane
  was then reseeded, and the postcondition is `reason='reseeded'` + fresh
  dispatch with the recovered plan + **no escalation**.
- **INV-4** is narrowed and made rolling. The streak counts only UNEXPLAINED
  fallbacks (`reason ∈ {stale, no_transcript}`; `reseeded` is excluded by
  construction, as `capped` already was), and a run must now be CHAINED —
  each fallback within `session_resume.storm_window_secs` (new knob, default
  3600s) of the previous, else the streak decays to 0 first. Without the
  decay the counter was cumulative-per-boot, not consecutive: its only reset
  was an eligible resume, so a slow drip of isolated `stale` events would
  have accumulated into a false storm. The window is measured on the
  MONOTONIC clock because `stale` is itself produced by clock skew. A
  per-dispatch reset was rejected — with several concurrent slots an ordinary
  dispatch interleaves between virtually any two recovered-session
  dispatches, which would have made the threshold unreachable and INV-4 dead
  code.

**Measured impact.** 149 `session_resume*` events since 2026-07-19: 142
`no_transcript`, 7 `stale`, exactly 1 eligible resume. ~14/day and bursty
(17 inside one hour = one boot's worth of recovered tasks re-dispatched onto
reseeded lanes), spread thin across many tasks. The cost was escalation
noise — an L1 every ~5 unlucky dispatches — not a throughput loss; every
fallback still dispatched fresh WITH its recovered plan (I3).

**Out of scope here.** Making resume actually SUCCEED (1/149) is the sibling
transcript-rehydration task, not this one. This amendment only reclassifies
the outcome so the escalation channel stops firing on a by-design event; it
changes no acquire semantics and no eligibility outcome — only the reason a
fallback reports and whether that reason feeds the storm streak.
