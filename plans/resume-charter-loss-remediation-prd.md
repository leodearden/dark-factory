# PRD: Resume charter-loss remediation — restore, harden, observe, then unlock the substrate

**Status**: active · authored 2026-08-10 · programme contract absorbing the
incident batch **3983-3994** (filed 2026-08-10 pre-PRD under incident
pressure) and filling its gaps with nine new leaves (α-ι).
**Author**: /prd session from brief
`~/.claude/spawn-briefs/resume-sysprompt-prd-brief.md` (investigation
2026-08-10, reviewed and partly ratified by Leo — his decisions are marked
**[LEO APPROVED]** and are load-bearing).
**Companion artifacts**: `resume-charter-loss-remediation-prd.capability-manifest.md`
(+ `.capability-manifest.yaml` sidecar) beside this file.

## 1. Goal

Every `claude` CLI invocation the factory makes — fresh or resumed —
runs under its **current role charter**, every resume is **observable**
after the fact, the constraints that matter are **enforced by the CLI or
a server rather than by prose**, and only then are the two currently-dead
resume substrates (crash-recovery R3, recon adopt R6) brought back to
life. Plus the economic follow-on Leo approved: the escalation watcher
stops paying for empty polling (exit-on-drain), safely.

## 2. Background — the defect and the proven premise (G6)

`build_claude_argv` (`shared/src/shared/cli_invoke.py`) skipped
`--system-prompt-file` whenever `resume_session_id` was set, on the
strength of a comment claiming the flags are "(incompatible)". **False**
— proven by controlled experiment (CLI 2.1.226, canary prompt with an
unguessable token, `--fork-session` arms):

| Arm | Flags | first-turn `input_tokens` | Charter honoured |
|---|---|---|---|
| fresh | `--system-prompt-file` | 245 | yes |
| pre-fix code | `--resume` only | **2066 (stock)** | **NO** |
| Fix A | `--resume --system-prompt-file` | 446 | **yes** |
| append variant | `--resume --append-system-prompt-file` | 2132 | layered over stock |

The system prompt is a per-process invocation parameter, never persisted
in the transcript; a resumed agent ran under the stock Claude Code
prompt with no role charter. Measured blast radius (2×2 over all 3,607
escalation-watcher-auto rotations, two independent implementations):
resumed **362/1681 breached (21.5%)**, never-resumed **0/1926 (0.0%)**.
96 unauthorised merges landed. Re-passing the prompt is a prompt-cache
HIT; omitting it was a total miss — **the fix is cheaper than the bug**.

**Organising principle** (predicts the blast radius exactly, and drives
this PRD's thrust 2):

> Anything enforced by the CLI or a server SURVIVES resume. Anything
> enforced by prose does NOT.

**Fix A landed** as task 3983 (merge `03ff70c5dd`, 2026-08-10):
`--system-prompt-file` is now emitted unconditionally, the three pinning
tests are inverted (the :159 tripwire consciously retired citing 3067),
and the in-file false docstrings/comment are corrected. This PRD's
remaining work is everything Fix A does **not** cover.

### The resume-site inventory (verified 2026-08-10)

| # | Site | Status post-Fix-A |
|---|---|---|
| R1 | `cli_invoke.py` cap-hit auto-resume (all 13 callers, 2,608 cap events) | charter restored by 3983; observability still absent (3986); structural de-resume for watcher/unblock_auto still wanted (3987) |
| R7 | recon `agent_loop.py` every turn ≥ 2 | charter restored; **tool results still destroyed** — needs `resume_delivers_prompt=True` (3984) |
| R4 | gamma ceiling-kill (`workflow.py`) | charter restored; still delivers a **false** "orchestrator restart" story (3985) |
| R5 | steward | charter restored (user prompt was already safe) |
| R3 | crash-recovery resume | **DEAD** (0 successes / 260 fallbacks, ever) — substrate owned by task 3578's PRD; gated here via θ |
| R6 | recon adopt-and-resume | **DEAD — Landlock**: recon's config dir sits outside the sandbox writable set, so no recon transcript has been written since 2026-07-18 (task 2744); fixed by η |
| new | curator/adjudicator/judge/deep_reviewer/module_tagger/watcher-auto/unblock_auto via R1 | charter restored; 3986/3987/ζ harden |

### Why the dead paths matter (G5 — the key sequencing insight)

R3 and R6 being dead is what made them *safe* pre-Fix-A: a path that
never resumes never strips a charter. Reviving either **before** Fix A
would have switched charter-stripping ON for task roles at every 8h
fleet redeploy (R3) and for 56-66k-char recon stage charters (R6).
**[LEO APPROVED]: Fix A gates both substrate fixes — as dependency
edges, not prose.** Fix A is now merged, so the gates are satisfied, but
the edges are wired anyway (3578 already carries `depends_on 3983`; η
carries it too) so the ordering is machine-visible in the task graph and
survives any revert/reland of 3983.

## 3. Sketch of approach — five thrusts

1. **Restore delivery** (landed + pending): Fix A (3983, done) restored
   the charter on all seven sites; 3984 restores R7's per-turn tool
   results; 3985 gives the gamma ceiling-kill an honest resume prompt.
2. **Enforce, don't exhort**: convert the constraints the breaches
   actually violated from prose to CLI/server enforcement — stop
   resuming the two roles where resume buys nothing and costs the whole
   charter (3987), deny `Skill` to the auto-watcher (3993), make role
   leases real locks (3994), make `start_report` idempotent (3988),
   give unblock_auto a CLI-enforced read-only envelope + a governing
   cap-wait bound (ζ).
3. **Observe**: record that a resume happened and detect a
   system-prompt swap from the first assistant turn (3986); then audit
   the damage with human-approved scope (3992).
4. **Watcher economics [LEO APPROVED: exit-on-drain]**: first make the
   drain protocol durable so accumulated context stops substituting for
   the durable channels (α, β, γ), then exit-on-drain **with the
   <120s degenerate-exit guard change in the same task** (δ), plus
   quiet-project cost containment (ε).
5. **Unlock the substrate + fix the beliefs**: verify 3727 is actually
   deployed (θ) → 3578 (transcript restore, other PRD) → reassess
   3730/3733 (ι); recon Landlock fix (η) activates R6 and recon
   transcripts; correct the 9 false-belief doc sites and settle both
   kill-switches (3991, amended).

## 4. Resolved design decisions

- **D1 [LEO APPROVED]** — Fix A gates both substrate fixes (R3 revival
  via 3578, R6 revival via η). Wired as dependency edges on 3983.
- **D2** — Replace, don't append: `--system-prompt-file` (replace)
  is the correct form. Roles are restrictive charters; layering over
  the stock prompt keeps a contradicting general-purpose identity in
  force. Replace makes resumed and fresh invocations byte-identical in
  contract. (Landed with 3983.)
- **D3** — A resumed session receives the **current** role prompt, not
  the original bytes. No role prompt is templated with per-invocation
  task context, so this is safe; the four caveats (Stage 2 branches on
  `project_id`; reviewer/curator are model-keyed artifacts; Stage 1/3
  introspect live FastMCP signatures; recon-verify is tool-list
  templated) are documented in 3983. Consequence: both kill-switches
  (`session_resume.enabled`, `resume_after_restart`) lose their stated
  purpose — their fate is decided in 3991.
- **D4 [LEO APPROVED]** — R3 is **not** retired. Order: Fix A (done) →
  verify 3727 actually deployed and writing archives (θ) → 3578 →
  reassess 3730/3733 (ι). Task 3989 (the decide-task) was cancelled
  2026-08-10 20:59Z once this ruling superseded it; its architect
  dry-run's polarity dispute (does `reseeded=0` prove the config dir
  survives or was deleted?) is folded into θ as an explicit
  question-to-settle, since it determines what 3578's restore actually
  has to fix. The "pause 3732" directive in the brief is **moot**: 3732
  went done 2026-08-10 16:09Z, and it is a recon-reliability task (Stage
  2 cycle-summary backstop), not an eligibility-seam leaf as the brief
  assumed.
- **D5 [LEO APPROVED]** — Exit-on-drain, not rotation-length tuning.
  Rotation length is a ±3.5% knob (2-6h plateau); 58.2% of all watcher
  spend happens after the last unit of work, and 88.4% of turns are a
  poll loop `_watcher_has_actionable_l1()` already runs for free at 60s
  cadence between rotations. Prerequisites first (α, β, γ): the
  session-context dependence is an artifact of calling the API wrong
  (compact drain drops `root_cause`/`members`; `add_members_to_l2`
  discards 24.3% of authored framing; the archive is never queried), so
  make the durable channels sufficient **before** shortening the
  effective window. `watcher_rotation_hours` is left alone.
- **D6** — The exit-on-drain landmine: 5 clean exits under
  `watcher_misconfigured_min_rotation_secs` (120s) in 10 minutes trips
  `pause_scheduler('watcher_misconfigured')` — pausing the **scheduler**,
  not just the watcher. Exit-on-drain is behaviourally identical to the
  failure this guard catches, so **the exit reason must become explicit
  and machine-readable, and the guard change lands in the same task as
  exit-on-drain (δ), never separately**. The guard's storm-escape
  purpose is retained for exits *without* the drained marker (INV-4).
- **D7** — Recon Landlock fix shape (η): grant the recon config root to
  the sandbox writable set (`sandbox_recon_writable_extras` default, or
  relocate `recon_config_base_dir` under `<explore_codebase_root>/.task/`
  which is already granted unconditionally). This *activates* R6 and
  recon transcripts — acceptable only because Fix A is on main (D1).
  Side benefits: the recon cap-retry veto stops force-fresh re-work
  (~110-150k tokens/day floor), `count_transcript_turns` starts
  returning real values so the progress-based timeout extension engages
  for recon, and task 3972 (recon transcript archival) becomes
  satisfiable — it gains a dependency edge on η.
- **D8** — Observability detector: the `cache_read_tokens == 0` rule
  does **not** work against `runs.db` (run-level aggregate; 0 of 20,644
  rows have it). The detector reads the **first assistant message** of
  the transcript; sharpest discriminator is that message's
  `input_tokens` (~2066 stock vs 245-446 custom). Owned by 3986; 3992's
  audit depends on it. A *generalized* "substrate liveness gate"
  (assert a recovery path has executed at least once, fleet-wide) is
  out of scope (§8); the specific instances land via 3986's events and
  θ's deployment check.

## 5. Pre-conditions for activating

All satisfied at authoring time:
- Fix A on main (`03ff70c5dd`), tests inverted — verified by reading
  `main` (not the working tree).
- Substrate facts for every new leaf verified against main 2026-08-10
  (see the capability manifest beside this PRD).
- 3727 (`durable_archive_path` + `archive_available` instrumentation)
  is `done` — but its **deployment** to the running fleet is exactly
  what θ exists to verify before 3578 proceeds
  (`data.archive_available` is NULL on all 260 historical fallbacks,
  consistent with known fleet-deploy staleness).

## 6. Cross-PRD relationship (G4)

| Other PRD / owner | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/session-resume-eligibility-seam-prd.md` (3727 done; 3728/3729/3730/3731/3733 pending) | this PRD gates | ordering: eligibility tuning must not outrun a working substrate | eligibility PRD owns its leaves; **this PRD owns the gates** (θ before 3578; ι reassesses 3730/3733 after) | θ, ι queued |
| `plans/transcript-preservation-seam-prd.md` (3578, 3618, 3619) | this PRD gates | R3 revival = charter-stripping switch-on hazard (D1) | that PRD owns the restore; this PRD owns the Fix-A gate (wired: 3578 `depends_on` 3983) + the θ deployment gate (wired: 3578 `depends_on` θ) | wired |
| `plans/warm-lane-session-resume-prd.md` | consumes correction | false-belief §9 site 3 ("resumed session runs under the pre-restart system prompt by construction") | 3991 (this programme) | queued |
| `plans/agent-liveness-telemetry-resume-prd.md` | adjacent | `invocations` schema additions (`resumed`, `session_id`, `sysprompt_sha`) | 3986 (this programme) produces; liveness PRD consumes | noted in 3986 |
| escalation-l2-tiering machinery (tasks 1499/1504, landed) | this PRD amends | compact projection fields; `add_members_to_l2` write path; `find_pending_l2_by_root_cause` matching | α, β (this PRD) | queued |
| task 3089 (pending, `/tmp` hygiene) | referenced | orphaned `/tmp/sysprompt_*.txt` producer-side fix (step-0 leftover) | 3089 | already filed — not duplicated here |
| task 3972 (pending, recon transcript archival) | this PRD unblocks | recon transcripts must exist before they can be archived | 3972 owns archival; η owns existence (wired: 3972 `depends_on` η) | wired |
| `plans/recon-reliability-prd.md` (3732) | none (moot) | brief assumed 3732 built on R3 — it does not, and it is done | — | closed |

## 7. Decomposition plan

### 7a. Absorbed batch (already filed 2026-08-10; listed for the programme record — do NOT re-file)

| Task | Step (brief §12) | Status | Deps | Signal (as filed) |
|---|---|---|---|---|
| 3983 | 0 — Fix A | **done** (merged `03ff70c5dd`) | — | resumed argv contains `--system-prompt-file`; first-turn `input_tokens` custom-sized |
| 3984 | 1b — R7 `resume_delivers_prompt=True` | pending | — | serialized tool results reach the CLI on turn ≥ 2 (not `CRASH_RECOVERY_RESUME_PROMPT`) |
| 3985 | 1a — gamma-specific resume prompt | pending | — | ceiling-kill resume delivers a truthful prompt; new test (do not edit `test_crash_recovery.py:2427-2461`) |
| 3986 | 2 — resume observability | pending | — | `invocations` carries `resumed`/`session_id`; cap-hit resume emits an event; first-turn detector |
| 3987 | 3 — stop resuming watcher + unblock_auto; bound cap retries | pending | 3983 | those two roles retry FRESH on cap hit |
| 3988 | 5 — `start_report` idempotency guard | pending | — | second `start_report` no-ops/errors loudly; findings survive |
| 3989 | (R3 decision) | **cancelled** — superseded by D4 | — | — |
| 3990 | (recon transcript question) | **to cancel at decompose** — answered: Landlock (§2, D7); resolution recorded in-task before the flip | — | — |
| 3991 | 6 — false-belief + kill-switch remediation | pending | 3983 | all §9 sites corrected; kill-switches decided + documented in OPERATIONS.md; **amended at decompose to add §9 item 9** (escalation-watcher SKILL.md's inverse-differential incident note) |
| 3992 | 8 — damage audit (scope sign-off first) | pending | 3986 | scoping proposal → human sign-off → audit of ~30 recent breaches first |
| 3993 | (defence-in-depth) deny `Skill` to watcher-auto | pending | — | `Skill` in `_WATCHER_DISALLOWED_TOOLS` + test |
| 3994 | (defence-in-depth) lease ownership checks | pending | — | `lease-release`/`heartbeat` require `--slug`; contention message legible |

### 7b. New leaves (filed by this decompose; Greek labels stamped into the sidecar)

| Label | Title | Modules | Prio | Deps | User-observable signal (RED today) | Consumer |
|---|---|---|---|---|---|---|
| α | Escalation drain durability: compact projection carries `root_cause`+`member_ids`; `add_members_to_l2` stops discarding authored framing | `escalation/src/escalation/server.py`, `escalation/src/escalation/queue.py` | high | — | `get_pending_escalations(compact=True)` rows carry `root_cause` and `member_ids` (today: dropped); a promote that folds into an existing L2 preserves the incoming `root_cause`/`evidence`/`options` text on the record (today: only `members` survive — 24.3% of authored L2 framing discarded at write time); pinning tests `test_compact_projection_carries_root_cause`, `test_add_members_preserves_incoming_framing`. En route: fix the tool docstring drift (it lists 9 compact fields; the tuple returns 13 — the 4 triage-ack fields are undocumented) | γ, δ; every watcher rotation (can rebuild `already_promoted` from the queue instead of memory) |
| β | Root-cause canonicalisation in `find_pending_l2_by_root_cause` (~155 duplicate L2s = 30% of all created) | `escalation/src/escalation/server.py` | high | α | two promotes whose `root_cause` differ only in case/whitespace/punctuation fold into ONE L2 (today: exact string match only); pinning test `test_root_cause_match_is_canonicalised` | δ; the human L2 queue (fewer duplicates) |
| γ | Auto-watcher drain loop: drain non-compact (or the α-enriched projection) + query the escalation archive | `skills/escalation-watcher-auto/SKILL.md` | high | α | the drain loop's documented queries include an archive-inclusive read (`get_task_escalations(status=None)`; `get_pending_escalations` is pending-only **by design**, server.py) and rebuild `already_promoted` from projection fields, not session memory. Sharper defect fixed en route (verified 2026-08-10): today's step 3 unions `members` of *pending* L2s only, so a **resolved** L2's members fall straight back into `work_batch` — the archive read closes that hole. The 17 window-dependent finding classes become recoverable at any window length | δ; watcher rotations |
| δ | **[LEO APPROVED]** Exit-on-drain + degenerate-exit guard change **in the same task** | `skills/escalation-watcher-auto/SKILL.md`, `orchestrator/src/orchestrator/harness.py`, orchestrator defaults config | high | α, β, γ | a rotation that finds the queue drained exits early with a machine-readable `exit_reason=drained` marker the supervisor reads; drained exits do NOT count toward `watcher_max_misconfigured_clean_exits`; a <120s clean exit *without* the marker still trips the guard (pinning tests `test_drained_exit_does_not_trip_misconfigured_guard`, `test_unmarked_fast_exit_still_trips_guard`); watcher spend after last-work drops from 58.2% | fleet cost budget; scheduler safety (guard keeps its storm escape) |
| ε | Quiet-project watcher cost containment + fix the daily-ceiling under-count | orchestrator config for the 4 quiet projects; `orchestrator/src/orchestrator/harness.py` (ceiling enforcement) | medium | — | zero-work rotations stop launching on know-live / autopilot-video / my-solar-challenge / solar-challenge (verify the 2629 empty-queue launch gate is deployed & firing there — deployed-code check, not source check); `watcher_daily_cost_ceiling_usd` enforcement compensates the measured ~40% runs.db under-count; quiet-project watcher spend (was $1,960 / 28% of fleet for 138 dispositions) drops measurably | fleet cost budget |
| ζ | unblock_auto: CLI-enforced read-only envelope + a cap-wait bound that actually governs | `orchestrator/src/orchestrator/dry_run_unblock.py`, `shared/src/shared/cli_invoke.py` / `usage_gate.py`, orchestrator config | high | 3987 | the dry-run invocation argv carries an explicit permission mode + a write-tool denylist (today: `permission_mode` never passed at the call site — inherits `bypassPermissions` from `invoke_claude_agent`'s default; read-only is prose + advisory allowlist); the 1800s `_DRY_RUN_CAP_WAIT_SANITY_SECS` bounds the **real** wait — today `_check_cap_wait` is a `cli_invoke.py` closure called only inside the cap-hit retry loop (`:1896`/`:1988`), while the actual wait sits in `usage_gate.invoke_slot`, bounded by nothing (overshot 11× at 19,805s); pinning tests `test_dry_run_argv_denies_write_tools`, `test_cap_wait_sanity_bound_governs_slot_wait`. Dep on 3987: same cap-retry seam in `cli_invoke.py`, and the bound must be designed against 3987's fresh-retry semantics for this role | `/unblock-low-risk` (its `risk_label` gates autonomous merge-to-main); the fleet cost budget |
| η | Recon Landlock fix: put the recon config dir inside the sandbox writable set (activates R6 + recon transcripts) | `fused-memory/src/fused_memory/config/schema.py` or `.../reconciliation/cli_stage_runner.py`, `orchestrator/src/orchestrator/agents/landlock_exec.py` (stale comment) | high | 3983 | a recon stage session writes a transcript into its per-run config dir (today: ZERO transcripts anywhere since 2026-07-18 — probe: `CLAUDE_CONFIG_DIR/projects` → `PermissionError`); a capped recon stage RESUMES instead of retrying fresh; `count_transcript_turns` returns non-None for recon; pinning test `test_recon_config_dir_is_landlock_writable` | recon stages (~9 cap hits/day stop paying full re-work); R6; 3972 (archival — dep wired); 3990's answer made durable |
| θ | Verify 3727 is deployed & writing `archive_available`; settle the `no_transcript` polarity dispute — gate for 3578 | operational: fleet units, `runs.db`, `data/orchestrator/agent-transcripts/` | high | — | a fresh `session_resume_fallback` event carries non-NULL `data.archive_available` (today: NULL on ALL 260, despite 3727 landing 2026-08-06/07 — the discriminator between "not deployed" and "deployed but broken"); plus a written answer to the 3989-architect polarity dispute (`reseeded=0`: does the config dir survive or not in the `no_transcript` population?) with per-event evidence; `ExecMainStart`-style deployed-code check, not a source check | 3578 (dep wired); the transcript-preservation PRD |
| ι | Reassess 3728/3729/3730/3733 against a working resume baseline | decision (no code) | medium | 3578 | each of 3728/3729/3730/3733 is either re-affirmed (note recorded on the task) or cancelled/re-scoped with rationale — decided against MEASURED post-3578 resume data, not the pre-fix baseline | the eligibility-seam PRD's remaining leaves |

### 7c. Amendments to existing records (performed at decompose, not new tasks)

1. **3990** — record the answer (Landlock, §2/D7; R6 dead; breach rates
   re-derived: resumed 73.8% vs non-resumed 1.26%, direction unambiguous,
   exact historical percentages unsupported) in the task's `details`,
   then `cancelled`. No dependents exist (verified across the batch), so
   the cancel arms nothing.
2. **3991** — append to `details` (verified against main `03ff70c5dd`):
   (i) §9 item 9: the `skills/escalation-watcher/SKILL.md` task-2796
   note at ~:1048-1059 (section "Recognizing the supervised
   auto-watcher's resolutions") frames this failure class as its
   **inverse** ("Capped identity leaking into an interactive session")
   and closes the differential with "the supervised rotation is no
   longer the likely culprit — suspect a hand-injected `--mcp-config`"
   — the real third cause (a resumed rotation that lost its charter and
   believes it is interactive) is absent; a 2026-08-08 rotation read it
   and certified the wrong self-diagnosis. Add the third differential.
   (ii) Post-Fix-A reframing: the sites that stated the drop mechanism
   CORRECTLY as in-tree evidence (`judge.py:36-37/:212-213/:605-606`,
   `test_judge.py:1774/:3036`) are now **stale present-tense** — they
   describe a drop that no longer exists on main. Recast them as
   historical ("dropped until task 3983, 2026-08-10") preserving the
   incident record, and refresh the rotted `cli_invoke.py` citations
   (`1501-1503`→ current `~2202-2218`, `1552-1553`→`~2220`,
   `1270-1272`→`~1870-1871`). (iii) Confirmed still-false sites beyond
   the task's existing list: both plans (`warm-lane-session-resume-prd.md
   :138-144`, `fused-memory-restart-survey-2026-07-17.md:197-200`) were
   NOT touched by 3983's docs commit (it changed only `cli_invoke.py`),
   and `reconciliation/harness.py`'s `InterruptedRunResumeDisabled`
   branch is now at `:1656-1673`.
3. **3578** — `add_dependency` on θ (deployment gate before the restore
   work dispatches). Its 3983 edge already exists.
4. **3972** — `add_dependency` on η (no recon transcripts exist to
   archive until Landlock admits them), plus a one-line `details` note.
5. **3089** — one-line `details` note: Fix A **doubled** the
   `/tmp/sysprompt_*.txt` creation rate (the resume path now mkstemps
   one per invocation too), and cleanup remains purely per-call
   `finally` — no sweeper exists (verified: repo-wide grep finds only
   the mkstemp + per-call unlinks). The SIGKILL-orphan population now
   grows faster than when 3089 was filed.

## 8. Out of scope

- **Performing** the damage audit (3992 delivers scope first; execution
  needs human sign-off — the brief's step 8 says scope it, do not do it).
- Retiring R3 (D4 rules it stays) or redesigning warm-lane semantics —
  owned by `transcript-preservation-seam-prd`.
- A generalized substrate-liveness gate ("every recovery path must
  prove it has executed at least once") — a real idea the R3/R6 pattern
  motivates (two subsystems built, tested and refined on never-executed
  substrate), but a fleet-wide mechanism deserves its own PRD once
  3986's events exist to build it on. The specific instances are
  covered (3986, θ).
- Changing `watcher_rotation_hours` (D5: ±3.5% knob, wrong lever).
- The `/tmp/sysprompt_*.txt` leak — owned by existing task 3089.
- `--agent`-based reinforcement of charters (appends over stock;
  ARG_MAX hazard; explicitly NOT a substitute for Fix A).

## 9. Contract section (B+H — new seams only)

**C1. Compact drain projection (α).** `get_pending_escalations(compact=True)`
returns, per escalation: today's compact fields **plus** `root_cause: str`
and `member_ids: list[str]`. `detail` stays dropped (the size rationale
survives). Consumers may rebuild `already_promoted` as
`{canonical(root_cause) for L2s} ∪ {member_ids}` with no session memory.

**C2. Promote-fold preservation (α).** When `promote_to_l2` folds into an
existing L2 (`status:'updated'`), the incoming `root_cause`/`evidence`/
`options` text is appended to the record (amendment list or appended
evidence block with timestamp + submitting session), never discarded.
Storage growth is bounded (cap the amendment list; log on truncation —
no silent loss).

**C3. Canonical root-cause matching (β).** `find_pending_l2_by_root_cause`
matches on a canonicalised form (case/whitespace/punctuation-normalised;
optionally similarity-thresholded). Canonicalisation lives in exactly
one function used by both match and store paths (INV-5). Two L2s created
before β with equivalent root causes are NOT retro-merged (out of scope).

**C4. Drained-exit marker (δ).** A rotation exiting because
`get_pending_escalations` (per the γ protocol, archive-inclusive where
specified) returns no actionable work emits a machine-readable marker —
a structured final-output field or sentinel the supervisor parses
(`exit_reason=drained`), not free prose. Supervisor contract: an exit
WITH the marker bypasses the `watcher_max_misconfigured_clean_exits`
counter (its own bounded counter/telemetry instead); an exit WITHOUT it
keeps today's guard semantics unchanged. Today's supervisor splits at
`duration < watcher_misconfigured_min_rotation_secs` (harness.py:11492):
degenerate-clean feeds the guard, and **only healthy-clean clears the
watcher-outage L2** (`:11538-11548`) — δ must define the drained exit's
relation to BOTH: it must not feed the misconfigured guard, and it must
count as a liveness signal for outage-clearing purposes (the watcher
demonstrably ran and drained). `_watcher_has_actionable_l1()` remains
the relaunch gate, so drain-exit cannot thrash (it already subtracts
pending-L2 members).

**C5. unblock_auto envelope (ζ).** The dry-run invocation passes an
explicit permission mode and a denylist covering write tools (`Edit`,
`Write`, `NotebookEdit`, plus mutation-bearing MCP tools it must not
call) — flag-enforced, resume-surviving per the organising principle.
The cap-wait sanity bound is enforced where the wait actually happens
(`usage_gate.invoke_slot` path), not only in the cap-hit branch.

**C6. Recon writable set (η).** The recon per-run config dir root is a
member of the Landlock writable set for recon stage invocations, via
schema default or relocation under the already-granted
`<explore_codebase_root>/.task/`. The stale claim in `landlock_exec.py`
(~:20-23) is corrected in the same change. Credential isolation is
preserved (the parent writes `.credentials.json` pre-spawn today;
nothing new is shared across runs — per-run dirs stay per-run).

## 10. Boundary-test sketch (B+H)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 | producer side, α: compact projection carries the new fields | pending L2 with root_cause + members | `compact=True` row has `root_cause`, `member_ids`; no `detail` |
| B2 | consumer side, γ→α: rebuild `already_promoted` cold | fresh session, 2 pending L2s, 1 archived | dedup set correct with zero session memory |
| B3 | producer side, α: fold preserves framing | existing L2; promote with new evidence text | record carries both framings; nothing dropped |
| B4 | β both sides: near-duplicate root causes fold | L2 "Watcher lease stolen."; promote "watcher lease STOLEN" | `status:'updated'`, one L2, member linked |
| B5 | δ supervisor side: drained exit is benign | rotation exits <120s WITH marker, 6× in 10 min | no `pause_scheduler('watcher_misconfigured')`; drained counter increments |
| B6 | δ guard side: storm escape retained | rotation exits <120s WITHOUT marker, 5× in 10 min | guard trips exactly as today |
| B7 | ζ CLI side: denylist survives resume | dry-run resumed (pre-3987) or fresh | write-tool call fails at the CLI, not by prose |
| B8 | η sandbox side: transcript exists | recon stage runs under the real ruleset | `<config_dir>/projects/**/*.jsonl` exists; `count_transcript_turns` ≥ 1 |
| B9 | η veto side: cap-hit resume proceeds | capped recon stage, transcript present | resume (not force-fresh); charter present post-3983 |
| B10 | θ operational: instrumentation live | fresh fallback event on a restarted unit | `data.archive_available` non-NULL either way; polarity question answered from the same event's config-dir stat |

## 11. Open questions (tactical only)

1. **α field naming** — `member_ids` vs reusing `members` with ids-only
   shape. Suggested: new key `member_ids`, leaving `members` semantics
   untouched for non-compact consumers. Decide in α.
2. **β similarity mechanism** — pure canonicalisation vs embedding/fuzzy
   threshold. Suggested: start with deterministic canonicalisation
   (measurable, no tuning); revisit if the duplicate rate stays high.
   Decide in β.
3. **δ marker transport** — structured stdout sentinel vs a file the
   supervisor stats vs exit-code repurposing. Constraint: the supervisor
   currently reads only two booleans (3986 §evidence), so whichever
   transport is chosen must be read there. Decide in δ with 3986's hook
   points in view.
4. **ε mechanism split** — how much of the quiet-project saving δ
   already captures; ε's description directs re-measuring after δ lands
   before adding config. Decide in ε.
5. **η placement** — schema-default grant vs `recon_config_base_dir`
   relocation. Relocation keeps the Landlock ruleset untouched (smaller
   blast radius); the schema default documents the contract. Decide in η.

## 12. G7 walk (advisory record; full walk at decompose)

- α/C2: INV-2 applied (stop discarding emitter-known facts at write
  time); bounded amendment list satisfies INV-7's bound requirement.
- δ/C4: INV-1 (exit reason machine-checked at the consuming supervisor,
  not prose); INV-4 (the misconfigured guard's storm escape retained for
  unmarked exits; drained exits get their own bounded counter).
- ζ/C5: INV-1 (read-only becomes a CLI-enforced envelope);
  INV-7 (the unbounded `invoke_slot` wait gains its governing bound).
- ε: INV-4 repair (a cost ceiling enforced against an under-counted
  store is a silent fail-soft).
- η/C6: INV-3 (the landlock_exec.py comment asserting an unverified
  grant is corrected at the seam it describes).
- θ: INV-2 (structured per-event evidence, observation separated from
  hypothesis — required by the polarity dispute it settles).
- No leaf introduces an unbounded hold, a new prose contract, a
  log-scrape, lock-step duplication, or loop-thread work (INV-8: α/β
  are synchronous server-side dict shaping, bounded by queue size which
  is already paginated upstream).
