# PRD — Continuous confusion reduction (nightly trickle + periodic census + persistent codebook)

**Status:** active — authored 2026-07-13 from the agent-legibility session brief (`~/.claude/spawn-briefs/prd-confusion-reduction-2026-07-13.md`); every load-bearing decision below was ratified by the owner in that session.
**Predecessor artifacts:** `plans/agent-legibility-survey-2026-07-13.md` (incl. Addendum) and `docs/legibility/confusion-codebook.yaml` (both on main, commit 0691d13263).
**Approach:** B+H-lite — contracts (§7) + boundary-test sketch (§8); the mechanism count (~10) and the cross-repo blast radius (dark-factory scripts, per-project repos, systemd host state) trip the G5 heuristic.

## 1. Goal

Make confusion-source detection a standing, per-project Dark Factory capability instead of a one-off survey:

- A **nightly trickle** (per project, systemd user timer): deterministically digest yesterday's session transcripts, sample the highest-signal ones under a token budget, and have a Haiku coder code each digest against the project's persistent cause codebook — duplicates increment dated sightings, novel shapes become candidate entries.
- A **periodic census** (per project, auto-triggered): stratified-random saturation mining (Sonnet), verification of novel clusters against current main (Sonnet), synthesis (Fable) into a dated report in `plans/` with an **origin×manifestation matrix**, remediation tasks filed through the normal curator path, and codebook updates.
- A **persistent codebook contract**: `docs/legibility/confusion-codebook.yaml` in each opted-in project's repo is the append-only cause registry (never delete; mark retired), extended with `origin_phase` / `manifested_phase` stamps.

Rollout: dark_factory first, reify second.

## 2. Background

The 2026-07-13 agent-legibility survey (336 incidents → 16 verified clusters from 8,468 sessions) validated the premise empirically and exposed the cost structure: big-bang mining is expensive (~7.3M subagent tokens) and the cause distribution's head is saturating (12/16 clusters matched the existing taxonomy). The ratified replacement is exactly this PRD: deterministic digests (~20× cheaper downstream reads), cheap nightly coding, and censuses that stop at saturation. Survey §5 ("survey infrastructure") is the agreed sketch; this PRD is its design.

## 3. Consumers (G1)

| Mechanism | Consumer |
|---|---|
| Confusion digests (α) | Trickle coder (δ) and census miners (η) |
| Inventory/scorer/sampler (β) | Nightly trickle (ε) and census (η) |
| Codebook v2 + merger (γ) | Trickle coder (δ), census (η), Leo (reads the registry), design-invariants PRD (consumes `invariant_violated` coding — sibling session, see §10) |
| Dated sightings + candidates | Census trigger (ζ) — novelty spike; census (η) — saturation baseline |
| Census reports + origin×manifestation matrix | **Leo** (the matrix tests the hypothesis that merge/verify-manifested confusion originates in architect/implement phases, and points the reify rollout) |
| Census-filed remediation tasks | The task pipeline (curator-deduped `submit_task`) |
| Installed timers + liveness predicate | Leo's ops surface (`systemctl --user list-timers`; born-at-L2 escalation on silent death) |

## 4. User-observable surface (G2 seed)

- After a nightly run: `git log <project>/docs/legibility/confusion-codebook.yaml` shows a dated trickle commit; the codebook diff shows new dated sightings and/or candidate entries.
- After a census: a dated `plans/confusion-census-<date>.md` report exists with the matrix and filed task ids; the codebook is updated; `census-state.json` moves.
- A novelty spike (≥4 new candidates in 72h) triggers a census with no human action.
- `systemctl --user list-timers` shows `legibility-trickle-<project>.timer`; 7 days after deploy a liveness milestone auto-completes (or escalates `milestone_check_failed`).

## 5. Sketch of approach

All generic code lives in dark-factory at `scripts/legibility/` (tested under `scripts/tests/`, the existing convention), parameterized by a per-project config committed in the **target** repo at `docs/legibility/legibility.yaml`. Opt-in = that config file exists + a timer installed for the project.

1. **Digest extractor (α, zero LLM).** Transcript JSONL → 5–15KB markdown digest with YAML frontmatter: non-sidechain user turns (user corrections are gold), `tool_result` blocks with `is_error` plus the preceding attempt, assistant self-correction markers with context, retry loops. Rewrite of the survey's proven scorer logic, properly, with tests.
2. **Inventory + signal scorer + stratified sampler (β, zero LLM).** Enumerate yesterday's sessions across the project's transcript dirs — `~/.claude/projects/<enc>` where `<enc>` is the cwd encoding mirrored by `session_registry.transcript_path_for_cwd` (`/` and `.` → `-`); a project's agents span many encodings (57 for dark-factory, 275 for reify warm-lanes/worktrees), so the config lists **cwd prefixes** and the inventory matches encoded dir names by prefix. Then: signal-score (zero-token pass: tool errors, self-corrections, not-found, guard trips, interrupts) → stratify by agent class (recon / curator-classifier / watcher / orchestrated-task / interactive) → drop zero-signal → dedupe near-identical shapes (recon clones) → top ~10–15% per stratum with a per-stratum minimum of 2 → fill the daily digest-byte budget in score order. **Budget cap, not count cap; cap applied AFTER stratification** (session sizes vary ~100×).
3. **Trickle coder (δ, Haiku).** One headless `claude -p --model <haiku>` call per digest, given the digest + a compact codebook index (ids, titles, one-line causes). Emits a strict-JSON **coding record**: matches (entry id + phase stamps) and candidates (novel shapes). Parse failure → skip + log + count, never fabricate (codebook lesson `one-shot-subagent-contract`). >50% failures in one night → loud escalation.
4. **Deterministic codebook merger (γ).** The only writer of the codebook YAML. Applies coding records: match → append a dated sighting `{date, project, session, origin_phase, manifested_phase}`; candidate → append under `candidates:` with `first_seen`. Idempotent (dedup key: session × entry), append-only, refuses deletions; schema validator runs on every merge.
5. **Nightly trickle unit (ε).** Orchestrates β→α→δ→γ, commits the codebook diff docs-only (`git commit --only docs/legibility/...`, ref-lock retry, never stash — the machine-operated-main-checkout rules), evaluates the census trigger (ζ), and logs a one-line decision. Shipped as `legibility-trickle@.service`/`.timer` systemd user units (precedent: `orchestrator-watchdog.timer`) + an install script.
6. **Census trigger (ζ).** Fires at the earliest of: (a) 10 calendar days since last census; (b) 7 days AND ≥120 tasks landed since last census (done-count delta via fused-memory `get_statuses`); (c) novelty spike ≥4 new candidates within 72h (derived from `first_seen` dates). **Hard floor: 5 days** (fix latency — earlier censuses mostly re-observe pre-fix traces). State in `docs/legibility/census-state.json`.
7. **Census runner (η).** Usage-headroom preflight (one tiny probe call; usage-limit banner → defer to next night, log loudly) → stratified-random saturation batches of Sonnet miners coding digests against the codebook until ≥90% duplicates for 2 consecutive batches → Sonnet verification of novel clusters against current main → Fable synthesis ONLY for the final clustering/report → dated report in `plans/` incl. the origin×manifestation matrix → remediation tasks filed via curator-path `submit_task` (never planning_mode — curator dedup is the protection) → codebook update via the merger (promote/reject candidates in place, retire fixed entries). `--force` flag for operator-initiated runs.
8. **Deploys + liveness (ε′, θ′, κ).** Timer installation is host state, so it ships as `task_kind='deterministic'` deploy tasks running the committed install script; a delayed-milestone **predicate** task 7 days after the dark_factory deploy checks the timer actually ran recently (via `systemctl --user show`, not git — quiet nights commit nothing) and escalates born-at-L2 on failure.

### Model routing (ratified budget policy)

Haiku for trickle coding; Sonnet for census mining/verification; **Fable ONLY for census synthesis**. Check usage-window headroom before launching census fleets. The nightly Haiku trickle is accepted standing spend. (Static routing per this policy — deliberately NOT via the adaptive-model-routing `resolve_route` ladder; see §12.)

## 6. Resolved design decisions

1. **The LLM never edits the codebook YAML.** Coders emit coding records; the deterministic merger is the sole writer (validation, idempotency, append-only invariants live in one place). Prevents the entire "LLM mangles YAML / deletes entries" class.
2. **Idempotency via sighting identity**, not a run ledger: sightings carry the session basename; the merger dedupes on (session, entry). Re-running a night is safe; no separate trickle ledger to leak (lesson: `recon-lifecycle-state-gaps`).
3. **Candidates live in the codebook** under `candidates:` with `first_seen` dates; the census promotes to `entries` or marks `rejected` **in place** (never delete). The novelty-spike trigger derives from `first_seen` — no side channel.
4. **Per-project codebook in the target repo** (`<root>/docs/legibility/confusion-codebook.yaml`). Harness-rooted causes observed in a hosted project's sessions may carry an `upstream: dark_factory:<entry-id>` link; a hosted project's census may file remediation tasks into dark_factory (same fused-memory, different `project_root`).
5. **Headroom check = cheap preflight probe**, not a usage API (none exists as substrate): one tiny Haiku call; a usage-limit/auth banner defers the census to the next trigger evaluation with a loud log + info escalation.
6. **Phase stamps are per-sighting and may be `unknown`.** Enum: `prd | decompose | architect | implement | verify | review | merge | recon | ops | unknown`. The trickle coder stamps best-effort; census synthesis refines; the matrix reports unknowns explicitly rather than guessing (lesson: `guards-assert-unverified-diagnoses`).
7. **Trickle commits are docs-only `git commit --only`** with ref-lock retry; nothing is committed on a no-change night. Liveness is therefore probed from systemd unit state, not commit presence.
8. **Fail-loud contract:** extractor crash, >50% coder parse failures, or git-commit failure files an info escalation to the project's escalation server (port from config) and exits non-zero — degradation never silent (standing directive).
9. **Census files tasks through the curator** (normal `submit_task`), never `planning_mode` — dedup against already-filed remediation is the point. Filed tasks follow the routing rules (`task_kind`, no prose-routing intent — lesson: `prose-routing-intent`).
10. **Deploy/liveness via the deterministic task kind** (install script as `before_done` deploy; liveness as delayed-milestone predicate) — no LLM pipeline for host operations. Stub scripts are committed with this PRD (executable, fail-loud) because `submit_task` validates `before_done.script` existence at filing; task ε replaces them with the real implementations before any deploy task can run (deps gate it).

## 7. Contracts

### 7.1 Codebook v2 (extends the committed v1 in place; migration keeps all v1 fields)

```yaml
version: 2
updated: <date>
entries:
  - id: <slug>                      # immutable
    title: ...
    severity: high|medium|low
    area: ...
    cause: ...
    status: open|partially|fixed|retired|mined-unverified   # never delete; retire
    origin_phase: <phase|unknown>       # modal attribution, refined by census
    manifested_phase: <phase|unknown>
    sightings_2026_06: 17               # v1 historical aggregate, retained
    sightings:                          # v2 append-only ledger (merger-owned)
      - {date: 2026-07-14, project: dark_factory, session: <uuid-or-basename>,
         origin_phase: implement, manifested_phase: merge,
         invariant_violated: <optional-id>}     # field owned here; consumed by design-invariants PRD
    upstream: dark_factory:<entry-id>   # optional, hosted-project → harness cause link
    fix: / fix_where: / fix_effort: / filed_tasks: / known_cause_match:  # as v1
candidates:
  - id: cand-<yyyymmdd>-<n>
    title: ...
    cause: ...
    area: ...
    first_seen: <date>
    disposition: pending|promoted|rejected   # census-stamped; promoted names the entry id
    sightings: [...]                         # same shape as above
```

Phase enum everywhere: `prd | decompose | architect | implement | verify | review | merge | recon | ops | unknown`.

### 7.2 Digest (extractor output)

One markdown file per session: YAML frontmatter `{session, cwd, encoded_dir, agent_class, date, size_bytes, score, signal_counts: {tool_error, self_correct, not_found, df_guard, interrupt}}` + sections: user turns (non-sidechain), error neighborhoods (is_error tool_result + preceding attempt), self-corrections with context, retry loops. Soft cap 15KB (truncate lowest-signal sections last).

### 7.3 Coding record (coder output → merger input)

JSONL, one object per digest, strict schema:
```json
{"session": "...", "date": "...", "project": "...", "agent_class": "...",
 "matches": [{"entry_id": "...", "origin_phase": "...", "manifested_phase": "...",
              "invariant_violated": null, "note": "..."}],
 "candidates": [{"title": "...", "cause": "...", "area": "...",
                 "origin_phase": "...", "manifested_phase": "...", "evidence_quote": "..."}]}
```
Schema violation ⇒ record skipped + counted; never partially applied.

### 7.4 Per-project config (`<root>/docs/legibility/legibility.yaml`)

```yaml
project_id: dark_factory
project_root: /home/leo/src/dark-factory
escalation_port: <int>
cwd_prefixes: [/home/leo/src/dark-factory]        # reify adds warm-lane + worktree roots
budgets: {max_daily_digest_bytes: 300000}
sampling: {top_fraction: 0.12, per_stratum_min: 2}
census:
  max_interval_days: 10
  tasks_landed_threshold: 120
  tasks_landed_min_days: 7
  novelty_spike: {count: 4, window_hours: 72}
  floor_days: 5
  saturation: {dup_rate: 0.9, consecutive_batches: 2}
models: {trickle: haiku, census_miner: sonnet, census_verify: sonnet, census_synthesis: fable}
```

### 7.5 Census state (`<root>/docs/legibility/census-state.json`)

`{"last_census_at": <iso>, "last_census_report": "plans/confusion-census-<date>.md"}` — committed by the census; the trigger reads it plus fused-memory done-counts plus candidate `first_seen` dates.

## 8. Boundary-test sketch

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Extractor→coder round trip | Fixture transcript with planted user-correction, is_error result, self-correction, retry loop | Digest contains all four signal classes, ≤15KB; coder (mocked LLM in CI) given digest + micro-codebook emits a record matching the planted duplicate and raising the planted novel |
| 2 | Merger idempotency | Same coding record applied twice | Exactly one sighting; second apply is a no-op |
| 3 | Never-delete invariant | Coding record / census disposition implying entry removal | Merger refuses; validator green; entry marked retired instead |
| 4 | Sampler budget-after-stratification | Synthetic inventory, 100× size variance, clone shapes, one tiny stratum | Zero-signal dropped; clones deduped; per-stratum min 2 honored; total digest bytes ≤ budget; big sessions can't evict a whole stratum |
| 5 | Trigger conditions | Fixture states: day 9 no spike / day 7 + 130 landed / day 6 + 4 candidates in 72h / day 4 + spike | no-fire / fire / fire / **no-fire (5-day floor)** |
| 6 | Coder failure storm | >50% of a night's codings fail schema validation | Info escalation filed, non-zero exit, codebook untouched by failed records |
| 7 | Census saturation stop | Fixture digest stream reaching 90% duplicates two consecutive batches | Mining stops; matrix + report emitted; state advanced |
| 8 | Deploy verify | Install script run for dark_factory | `systemctl --user list-timers` lists the trickle timer; deterministic runner stamps `deployed-and-verified` |

## 9. Pre-conditions for activating

All verified on main 2026-07-13:
- Codebook v1 committed (`docs/legibility/confusion-codebook.yaml`, commit 0691d13263).
- Transcript-dir encoding mirrored at `orchestrator/src/orchestrator/session_registry.py:451` (`transcript_path_for_cwd`); encodings enumerable by prefix (57 DF, 275 reify).
- systemd user timer precedent (`scripts/orchestrator-watchdog.timer`); per-project service templates.
- `claude` CLI 2.1.207 with headless `-p --model` at `/home/leo/.local/bin/claude`.
- `scripts/tests/` pytest convention.
- Deterministic task kind + delayed-milestone predicate machinery (CLAUDE.md, in production).
- Seam-owner tasks 2549/2558 filed and pending (emission-time telemetry — consumed, not owned; §10).

## 10. Cross-PRD relationship (G4)

| Other PRD / artifact | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| Tasks 2549/2558 (emission-time reason codes / structured evidence, from the survey batch) | consumes | Better-structured failure facts eventually make digests richer; no interface change needed here | **2549/2558** | wired (independent; trickle reads transcripts either way) |
| Design-invariants gate PRD (sibling session, authoring 2026-07-13) | produces | Optional `invariant_violated` field on sightings/candidates (§7.1/§7.3) | **this PRD** owns the schema field; the sibling owns its gate/consumption | queued (field ships in γ) |
| `/hotspot-survey` skill | none | Disjoint: hotspot-survey mines *bug history* (git/fix-tasks/postmortems); this mines *agent confusion* (transcripts). Neither replaces the other | — | n/a |
| Adaptive-model-routing PRD (tasks 2531-2546) | none | Model choice here is static ratified policy (§5), not `resolve_route`; no shared code | — | n/a |
| verify-plan-prd (W7) | none | Census may *observe* verify-manifested confusion; it files tasks, never edits verify code | — | n/a |

## 11. Decomposition plan

Greek labels; deps in brackets. All dark_factory unless marked. Modules: `scripts/legibility/`, `scripts/tests/`, `docs/legibility/`, systemd unit templates in `scripts/`.

- **α — Deterministic confusion-digest extractor** (`scripts/legibility/digest.py` + tests). Intermediate → unlocks δ, ε, η. Observable: CLI run on a real session JSONL emits a ≤15KB digest containing frontmatter + the four signal-class sections; fixture tests per §8.1.
- **β — Session inventory, signal scorer, stratified budget sampler + config schema** (`scripts/legibility/inventory.py`, `sampling.py`, config loader; commits dark_factory's own `docs/legibility/legibility.yaml`). Intermediate → ε, η. Observable: CLI run against live `~/.claude/projects` for dark_factory prints a sampling manifest (per-stratum counts, zero-signal drops, budget accounting); §8.4 tests.
- **γ — Codebook v2 schema + validator + deterministic merger** (`scripts/legibility/codebook.py`; migrates the committed codebook to v2 in place). Intermediate → δ, ε, η, design-invariants PRD. Observable: validator green on the live codebook post-migration; fixture coding record applied → dated sighting + candidate appear; §8.2/8.3 tests.
- **δ — Haiku trickle coder** (`scripts/legibility/coder.py`; headless `claude -p`, strict-JSON output, fail-loud). [α, γ]. Intermediate → ε, η. Observable: run on one real digest against the live codebook produces a schema-valid coding record; §8.1 consumer side + §8.6 storm test (LLM mocked in CI).
- **ζ — Census trigger evaluator + state** (`scripts/legibility/census_trigger.py`, `census-state.json`; done-count delta via fused-memory `get_statuses`). [γ]. Intermediate → ε, η. Observable: §8.5 fixture matrix passes; evaluator CLI prints the decision + reasons for the live project.
- **ε — Nightly trickle assembly + systemd units + real install/liveness scripts** (`scripts/legibility/nightly.py`, `legibility-trickle@.service/.timer`, real `scripts/legibility/install-trickle-timer.sh` + `check_trickle_liveness.sh` replacing the PRD-committed stubs). [α, β, γ, δ, ζ]. **Integration-gate leaf.** Observable: a manual end-to-end run on dark_factory produces a dated docs-only codebook commit with ≥1 new dated sighting and logs the census-trigger decision line.
- **ε′ — Deploy dark_factory trickle timer** — `task_kind='deterministic'`, `before_done={script: scripts/legibility/install-trickle-timer.sh, args: [dark_factory], timeout_secs: 120}`. [ε]. Leaf. Observable: `systemctl --user list-timers` shows `legibility-trickle-dark-factory.timer`; `done_provenance.kind='deterministic-deploy'`.
- **η — Census runner** (`scripts/legibility/census.py`: headroom preflight, Sonnet saturation mining, Sonnet verification vs main, Fable synthesis, plans/ report + matrix, curator-path task filing, codebook update, `--force`). [α, β, γ, δ, ζ]. Leaf. Observable: a `--force` census on dark_factory produces `plans/confusion-census-<date>.md` with the origin×manifestation matrix + saturation stats + filed task ids; codebook commit; state advanced.
- **θ — Reify enablement** (**reify project**: commits reify's `docs/legibility/legibility.yaml` — cwd prefixes incl. warm-lane + worktree roots — and a seeded v2 codebook). [external: `dark_factory:ε`]. Leaf. Observable: both files committed in the reify repo; DF validator passes on reify's codebook.
- **θ′ — Deploy reify trickle timer** — deterministic, `before_done={script: scripts/legibility/install-trickle-timer.sh, args: [reify], timeout_secs: 120}`. [ε′; external: `reify:θ`]. Leaf. Observable: `legibility-trickle-reify.timer` listed; deterministic-deploy provenance.
- **κ — Trickle liveness milestone** — deterministic predicate, `metadata.milestone={mode: delayed, after_secs: 604800}`, `before_done={kind: predicate, script: scripts/legibility/check_trickle_liveness.sh, args: [dark_factory, "72"], timeout_secs: 120}`. [ε′]. Leaf. Observable: 7 days after the DF deploy the task auto-completes with `done_provenance.kind='deterministic-milestone'` if the timer ran within 72h, else a born-at-L2 `milestone_check_failed` escalation fires.

## 12. Out of scope

- **Emission-time telemetry reason codes** — owned by tasks 2549/2558; the census consumes better telemetry but does not build it.
- **The design-invariants enforcement gate** — sibling PRD; this PRD only ships the `invariant_violated` codebook field.
- **`/hotspot-survey` territory** — bug-history mining stays in that skill; no duplication (explicitly: this PRD mines *confusion*, not *bugs*).
- **Adaptive model routing** for legibility agents — routing here is static ratified policy.
- **Fixing the confusion causes themselves** — censuses file remediation tasks; the fixes are those tasks' work.
- **Non-DF-hosted projects / multi-host rollout** — single-host, per-project opt-in only for now.

## 13. Open questions (tactical)

1. **Sightings-ledger compaction.** Censuses may compact old per-sighting records into monthly aggregates once entries accumulate hundreds. **Suggested:** defer until a codebook exceeds ~100KB; decide at that census.
2. **Decoy-FAIL suppression detail** in the extractor (fixture-emitted "FAIL:" strings). Decide during α.
3. **Census batch size** (digests per Sonnet miner batch). **Suggested:** start 20/batch; tune from the first census's saturation curve. Decide during η.
4. **Stratum definitions for non-DF projects** (reify agent classes differ). Decide during θ from reify's session mix.
5. **Same-night vs T-1 window** for the trickle (currently T-1 calendar day). Decide during ε if late-running sessions get missed.
