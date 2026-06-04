# PRD: B3 low-risk auto-unblock hardening (PRD-2)

Origin: Brief 2 of `plans/escalation-flow-2026-06-04-prd-briefs.md` (16-agent verified audit,
2026-06-04). Goal frame: increase throughput by reducing latency of issue handling, without
sacrificing final correctness of code reaching main. All anchors below re-verified against the
working tree on 2026-06-04 during PRD authoring.

## 1. Consumer + user-observable surface (G1)

- **AFK L2 watcher sessions** — the existing B3 consumer: low-risk fixes merge unattended instead
  of stranding; stale proposals abort with a *git-anchored* reason instead of a heuristic guess.
- **Attended L2 watcher sessions** (new, behind config flag + session override) — low-risk fixes
  merge without a human round-trip, with mandatory immediate in-session report (summary + merge
  sha + diff pointer).
- **The returning human** — reads one digest: per-B3 line showing merged / aborted-with-reason /
  drift-reinvestigated; cap state that survived restarts (no silently-spent phantom slots, no
  stranded backlog after slot 3).
- **Operators/auditors** — `metadata.dry_run_proposals[-1]` now carries `head_sha`/`main_sha`;
  `data/escalations/b3-state.json` is the single durable record of launches and merge charges.

Mechanism→consumer wiring (no orphan producers):
| Mechanism | Consumer |
|---|---|
| sha stamping in `dry_run_unblock.py` | `b3_gate check` (freshness verdicts); digest lines |
| `orchestrator/b3_gate.py` CLI | escalation-watcher B3 launch gate; unblock-low-risk defensive re-check + charge |
| `UnblockAutoConfig` new fields | b3_gate (cap, keep-last); watcher skill session start (attended flag) |
| proposal-list trim | metadata readers (humans, gate reading `[-1]`) — bounded blob growth |

## 2. Sketch of approach

Reify the B3 gate from prose into a small testable CLI module, anchor proposals to git shas at
investigation time, move cap state into a durable rolling-24h store, and make B3
posture-configurable instead of AFK-only.

1. **Sha anchoring (producer side).** `run_dry_run_unblock` captures
   `git -C <worktree> rev-parse HEAD` and `rev-parse main` **before** invoking the agent, and
   stamps `head_sha`/`main_sha` into **every** entry shape (`ok`, `investigation_failed`,
   `budget_exhausted`). The agent output schema is untouched — shas never round-trip through the
   agent (`additionalProperties: False` stays, by design: the agent cannot forge an anchor).
2. **`orchestrator/b3_gate.py`** — CLI-invocable (`.venv/bin/python -m orchestrator.b3_gate`),
   three verbs:
   - `check --task-id --worktree --project-root [--config] [--category]` → JSON verdict
     `fresh | drift | abort` + reason + cap remaining + already-attempted flag. Validates the
     latest proposal mechanically (risk label, no `status` key, category allowlist, sha anchors
     present, P1/P2 below).
   - `record-launch` → durable per-proposal attempt record keyed
     `(task_id, head_sha, investigated_at)` — a restart cannot re-launch a spent proposal.
   - `charge` → rolling-24h merge-slot charge, atomic; over cap → refused (caller ABORTs).
   - State: `<project_root>/data/escalations/b3-state.json`, all writes flock + tmp+rename,
     owned exclusively by b3_gate. **No task-metadata writes from the gate** (sidesteps the
     `set_task_status` metadata-replace hazard entirely).
3. **Skill prose becomes "call the gate, obey the verdict".** unblock-low-risk's heuristic
   precondition 6 → mechanical `b3_gate check`; new `b3_gate charge` immediately before
   `merge_request` (refused → ABORT). escalation-watcher's B3 subsection: gate via
   `b3_gate check`, drift → one re-investigation, cap prose deleted.
4. **Drift path.** On `drift`, the watcher spawns ONE read-only background sub-agent running the
   existing `unblock-auto` skill in the worktree, appends the fresh proposal (with shas captured
   at re-investigation start) via `update_task(append=true)`, then re-gates once. A second drift
   in the same handling cycle → leave pending + digest.
5. **Attended mode.** `UnblockAutoConfig.attended_b3_enabled` (default `false`) sets the standing
   posture; the human can override per session in either direction at watcher session start.
   Attended completions: immediate in-session report + digest entry; existing revert path
   unchanged.
6. **Proposal lifecycle.** Keep-last-N trim (default 5) at append time in `run_dry_run_unblock`,
   with a regression test that sibling metadata keys (`memory_hints`, `files`) survive.

## 3. Pre-conditions (G3 — substrate verified 2026-06-04)

- `dry_run_unblock.py:25-44` output schema, `additionalProperties: False` at `:43`; entries built
  parent-side at `:232-278`; append-only at `:198-203`; no pruning.
- `workflow.py:5736` region — `_spawn_dry_run_unblock` dedupes in-flight investigations; fresh
  investigation only on actual re-block (`:5375` region). **Not modified by this PRD** (the trim
  lives in `dry_run_unblock.py`).
- `skills/unblock-low-risk/SKILL.md:35-38` heuristic freshness; `:120-122` "future hardening" sha
  note; merge step `:69-74` (merge-queue-only).
- `skills/escalation-watcher/SKILL.md:197-242` B3 subsection (cap prose `:206-210`, AFK-only
  framing `:240-242`); AFK shift 2 `:181-188`.
- `UnblockAutoConfig` at `orchestrator/config.py:232` — natural home for new fields.
- Merge queue post-merge re-rebase+verify confirmed as the correctness backstop (register:
  existing invariant, owned by nobody, treated as read-only).
- Worktrees share refs with the primary checkout → `git -C <worktree> rev-parse main` resolves
  the local main the merge queue targets.

## 4. Resolved design decisions

### 4.1 The freshness invariant — precisely what we depend on

The proposal has two premises with very different volatility; the gate treats them differently.

- **P1 — branch HEAD (zero tolerance, hard ABORT).** The diagnosis was made against the worktree
  at `head_sha`. A blocked task's branch must not move: the workflow is parked and
  `release_workflow` re-parks it. Movement = an unknown actor touched the branch → hard ABORT,
  never overridden. Expected false-abort rate ≈ 0 *because* the sha shouldn't move; when it
  fires, it is signal, not noise.
- **P2 — main premise validity (wide tolerance, file-scoped).** Main is busy; `main_sha` equality
  would abort nearly always and is the wrong check. The fix's premise drifts only if main moved
  **in the proposal's footprint**: `git diff <recorded_main_sha>..main -- <files_referenced>`
  non-empty → verdict `drift`. Main movement elsewhere is tolerated mechanically and covered
  semantically by three existing rails: the rebase-conflict ABORT (skill step 4), the full verify
  suite (step 5), and the merge queue's authoritative post-merge re-rebase+verify.
  File-level granularity (not hunk-level) is accepted: a false positive costs one bounded
  re-investigation (~600 s / ≤$5), not a human round-trip.
- **Window.** Investigation → the sub-agent's own defensive re-check. The gate runs twice:
  watcher launch gate and unblock-low-risk precondition. The sub-agent's re-check closes the
  launch-window race; after it starts editing, the branch is its own and main movement belongs to
  the rebase/verify/merge-queue rails.
- **No wall-clock TTL.** Sha equality *is* freshness: an old proposal whose shas still match is
  valid by construction. Proposal age is reported in the digest for human awareness; it never
  auto-aborts.
- **On deterministic failure, LLM judgement never waives the gate — it regenerates the
  proposal.** Watcher-side `drift` → one read-only re-investigation producing a *new*
  sha-anchored proposal → re-gate once; second drift in the same handling cycle → pending +
  digest (main is moving under this task's feet; a human should look). Sub-agent-side re-check
  failure → hard ABORT (escalation stays pending); the watcher may route the same one-shot
  re-investigation on receiving a drift-reason abort, if not already used this cycle.
  LLM judgement is applied only at investigation time, full re-derivation — never as an override
  of a mechanical verdict.
- **Legacy degradation.** Pre-existing proposals without shas fail `check` as
  `drift: no sha anchor` → drift path → one re-investigation. No grandfathering into the old
  heuristic gate; no migration step needed.

### 4.2 Cap shape

- **Charge at merge-submit, default 6/24h rolling** (`b3_merge_cap_per_24h`, config). The slot is
  charged when the sub-agent calls `b3_gate charge` immediately before `merge_request`; refusal →
  ABORT. This caps the actual risk axis — unattended merges — at the choke point itself.
  Precondition/scope/verify aborts cannot merge and are free: no stranding from false aborts (the
  brief's 8-block AFK weekend strands 0–2 instead of ≥5).
- **Per-proposal cap = 1, durable.** `record-launch` at launch time, keyed
  `(task_id, head_sha, investigated_at)`. A session restart cannot re-launch a spent proposal; a
  genuine re-block + fresh investigation (new key) re-arms B3 for that task.
- The watcher's launch gate consults remaining capacity via `check` only to avoid pointless
  launches; enforcement is at `charge`. Concurrent sub-agents serialize on the state-file lock.

### 4.3 Attended mode

`attended_b3_enabled: bool = False` on `UnblockAutoConfig` — standing posture, versioned,
committed (per track-referenced-config convention). Session override in either direction at
watcher session start ("attended B3 on/off"). The field is **skill-facing config**: orchestrator
code never reads it; documented as such on the field. Every safety rail in unblock-low-risk is
already posture-agnostic; attended mode adds only the mandatory immediate in-session report
(one-line summary + merge sha + diff pointer) on completion, alongside the digest entry.

### 4.4 Gate contract enforcement

`b3_gate check` is the single shape validator for proposal entries (required keys incl. shas,
risk label, no `status` key, category in `{task_failure, review_issues}`). Watcher-side
re-investigation appends follow an entry template in the SKILL.md subsection mirroring
`_build_entry`; a malformed prose-appended entry simply fails the next `check` — fail-safe by
construction.

### 4.5 Config discovery

`b3_gate --config <path>` optional; built-in defaults (cap 6, keep-last 5) apply when absent.
The watcher knows the config path of the orchestrator it watches (config paths vary per project:
`orchestrator/config.yaml` here, `orchestrator.yaml`/`orchestrator-config.yaml` elsewhere).

## 5. Out of scope

- **L1 execution of B3** — explicitly rejected by the audit (breaches the L1 skill's hard safety
  envelope for one queue-hop of latency). Do not revisit.
- **`merge_queue.py` verify path** — existing invariant, read-only for all three PRDs.
- **New escalation categories / re-pend semantics** — PRD-3 territory.
- **`watcher.py`, `queue.py`, `sweep.py`, `archive.py`** — PRD-1 territory.
- Hunk-level overlap detection (future refinement; file-level + bounded re-investigation is the
  accepted trade).
- Orchestrator consumption of the new metadata fields (substrate for future tracking infra).

## 6. Cross-PRD relationships + seam ownership (G4)

Owned by this PRD (per the static register in `plans/escalation-flow-2026-06-04-prd-briefs.md`):
`orchestrator/.../dry_run_unblock.py`; `skills/unblock-low-risk/SKILL.md`;
`skills/escalation-watcher/SKILL.md` **B3 subsection + AFK shift 2 only**.

Additions not in the register, claimed here as natural extensions of the owned surface:
`orchestrator/b3_gate.py` (new module), `UnblockAutoConfig` fields in `orchestrator/config.py`
(additive, distinct class).

**Newly discovered seams** — logged append-only in `plans/escalation-flow-gaps-prd2.md`:
1. `b3-state.json` lives in the queue root that PRD-1's reaper extension will sweep — reaper must
   match `esc-*.json` only / allowlist non-escalation files (`afk-digest.md` precedent).
2. Attended-mode B3 makes the hardcoded "In AFK mode: try the low-risk auto-unblock gate first"
   lines in PRD-3-owned Handling-by-Category sections (escalation-watcher/SKILL.md:378, :386)
   stale — PRD-3 asked to defer applicability to the B3 subsection.

Sibling gaps files globbed at finalization time: none present yet (this session is first to
file); decompose mode must re-glob `plans/escalation-flow-gaps-prd*.md` before queueing.

## 7. Decomposition plan

Four tasks, each single-package (per protocol: cross-package tasks exceed the architect budget);
each SKILL.md edited by exactly one task. T1 → T2 → {T3, T4} (T3/T4 parallel, different files).

- **T1 (orchestrator): sha stamping + proposal trim + config fields** in `dry_run_unblock.py` /
  `config.py`. Capture `head_sha`/`main_sha` before `invoke_agent`; stamp ALL entry shapes; trim
  `dry_run_proposals` to `b3_proposal_keep_last` (default 5) at append; add
  `attended_b3_enabled=False`, `b3_merge_cap_per_24h=6`, `b3_proposal_keep_last=5`.
  *Signal:* unit tests — every entry shape carries both shas matching a fixture repo's actual
  shas; agent output schema still rejects sha injection (`additionalProperties: False`, sha keys
  absent from `properties`); after 6 appends the list holds 5 entries AND sibling metadata keys
  (`memory_hints`, `files`) are intact.
- **T2 (orchestrator): `b3_gate` module + tests.** `check`/`record-launch`/`charge`; state file
  with flock + tmp+rename; config-driven cap/keep-last; verdicts per §4.1–4.2. Includes the
  **two-way boundary test**: entries produced by the real `_build_entry` (+ T1 stamping) fed to
  `check` (producer/consumer contract, G5-lite B+H on this PRD's one real seam).
  *Signal:* synthetic stale proposal (HEAD moved) → `abort` with git-anchored reason; main
  overlap in `files_referenced` → `drift`; clean → `fresh`; sha-less legacy entry → `drift: no
  sha anchor`; cap state survives a process restart; two concurrent `charge` calls serialize and
  never exceed the cap; spent `(task_id, head_sha, investigated_at)` key → already-attempted.
- **T3 (skills): rewrite `unblock-low-risk/SKILL.md`.** Precondition 6 → mechanical
  `b3_gate check` (exact runnable command); add `b3_gate charge` immediately before
  `merge_request` (refused → ABORT); delete the `:120-122` "future hardening" note; abort-reason
  wording carries the gate's JSON reason verbatim into the return value.
  *Signal:* the documented gate commands execute successfully against a fixture worktree; no
  heuristic-freshness or future-hardening prose remains; charge step precedes the merge step in
  the procedure ordering.
- **T4 (skills): rewrite `escalation-watcher/SKILL.md` B3 subsection + AFK shift 2** (owned
  sections only). Gate via `b3_gate check` + `record-launch`; drift → one background read-only
  re-investigation (unblock-auto skill) → append (entry template w/ shas) → re-gate once, second
  drift → pending + digest; delete session-counted cap prose; applicability self-defined in the
  subsection (AFK always; attended when `attended_b3_enabled` or session override), with the
  mandatory attended in-session report; digest line format for merged/aborted/drift outcomes.
  *Signal:* subsection contains the exact runnable gate commands; the session-counted cap text
  and the AFK-only framing are gone; the subsection states its own applicability rule (so
  PRD-3-owned category-handler lines can defer to it); AFK shift 2 routing matches.

Decompose-mode metadata per leaf: `user_observable_signal`, `consumer_ref`, substrate-confirmed
flag; capability manifest committed beside this PRD at decompose time.

## 8. Open questions (tactical, implementation-time)

- Lock primitive: `fcntl.flock` on the state file vs separate lockfile; stale-lock timeout.
- Whether `check` should also probe `release_workflow` grip state or leave that rail purely in
  prose (currently prose, step 1 of the procedure).
- Exact digest line format for B3 entries.
- Whether `record-launch` should carry an advisory in-flight marker to dedupe across *concurrent*
  watcher sessions (the add_members_to_l2 clobber precedent says multi-session is real; the
  per-proposal key already makes the race harmless — worst case two launches of the same fresh
  proposal, serialized at `charge`).
- Entry-template wording for watcher-side re-investigation appends (mirror `_build_entry` keys).
