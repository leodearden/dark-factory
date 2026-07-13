# PRD: Capability-manifest delivered-checks — dep `done` ⇒ capability delivered

**Status**: active — authored 2026-07-13 from the ratified spawn brief
(`~/.claude/spawn-briefs/prd-capability-delivered-checks-2026-07-13.md`; agent-legibility
survey Q1 finding; owner reviewed the design shape and said LGTM).
**Mode**: B+H — four-module blast radius (shared, fused-memory, orchestrator, /prd skill)
touching a load-bearing seam (the scheduler dispatch gate). §Contract + §Boundary-test
sketch inline.

## Goal

A dependency counts as **satisfied** only when its promised capability is
**verifiably delivered on main** — not merely when its status label says `done`.

User-observable:

1. A dependent task whose dependency closed amended / scope-cut / feature-gated
   does **not** dispatch into a false premise. It is withheld at the dispatch
   gate, and after a short grace window a pending **born-at-L2 escalation names
   the exact failed `delivered_check`** (capability name, pattern/script, dep
   task id, main SHA) instead of an architect burning a dispatch discovering
   the premise is false.
2. When the promised capability genuinely lands on main, the withheld dependent
   dispatches normally with **zero operator involvement** (the per-main-SHA
   cache re-evaluates on the new SHA inside the grace window).
3. After `commit_planning`, the PRD's capability-manifest **sidecar carries the
   real filed task ids** (not just Greek letters), and each producer task's
   record carries its `metadata.delivered_checks` — visible via `get_task`.

Mined failure class this kills: `unverified-task-premises` (survey §1.5,
15 incidents) — dep closed scope-cut → dependent dispatches on a false premise
→ architect churn or wrong implementation.

## Background

- Every recent PRD ships `<prd>.capability-manifest.md` (≈15 exemplars in
  `plans/`), built at decompose Step 2.5 to mechanize G3+G6 **at authoring
  time**. It is prose-tabular, `file:line`-anchored, and consumed by nobody at
  dispatch: `skills/prd/references/decompose-mode.md:42` explicitly anticipated
  a dispatch-time consumer, and `:143` concedes the orchestrator reads none of
  it today. This PRD closes that loop.
- The scheduler already has the exact pattern to mirror: the **cross-project
  external-dep gate** — a pure predicate parameter (`external_status_cache` in
  `_deps_satisfied`), a per-tick side-effect sweep with streak counters,
  fail-safe wait on resolver error (no streak bump), grace-then-escalate via
  the `on_external_dep_block` callback, and dependent → `blocked` with manual
  re-pend (see CLAUDE.md "Cross-project task dependencies").
- Born-at-L2 contract (escalation server): severity ∈ `BORN_AT_L2_SEVERITIES`
  ({critical, urgent}) + a harness sentinel `agent_role` (`orchestrator-*` /
  `harness-*` per `_is_harness_sentinel_role`, `escalation/server.py:303`) →
  record stamped `level=2`, bypassing the auto-watcher.
- Survey cross-cutting cause #1 (§3): *contracts must live where they're
  consumed, machine-checked* — not in description prose or an .md table only a
  human reads. This PRD is that rule applied to the manifest itself.

## Sketch of approach

Three mechanisms, one per subsystem:

1. **Machine-readable sidecar** (`/prd` skill + shared schema). At decompose,
   alongside `<prd>.capability-manifest.md`, the skill emits
   `<prd-path-minus-.md>.capability-manifest.yaml` with the same bindings. Each
   capability may carry a **`delivered_check`** — a pattern-anchored check
   (`git grep` pattern or short committed script, **never** `file:line` — line
   anchors go stale) that must PASS on main once the producer task lands.
   Distinct from the authoring-time evidence binding (which proves substrate
   existed at decompose). Capabilities that aren't mechanically expressible
   (field-population judgments, rejection-mechanism nuances) are marked
   `kind: manual` and excluded from the gate.
2. **Stamping at `commit_planning`** (fused-memory `server/tools.py:3289`) —
   the filing step is the only place the Greek-label → real-task-id mapping
   exists mechanically. For each task in the batch carrying
   `metadata.prd_path` + `metadata.prd_task_label`, the server locates the
   sidecar, stamps `task_id` onto the matching label entry (file written back;
   the decompose session commits it), and **copies that label's mechanical
   delivered_checks into the producer task's `metadata.delivered_checks`**.
   The scheduler consumes task metadata only — it never discovers or parses
   sidecar files.
3. **Scheduler dep-gate consumption** (orchestrator `scheduler.py`). At
   dispatch eligibility, a local dep in a terminal status that carries
   `metadata.delivered_checks` counts as satisfied **only if all its checks
   pass against current main** (results cached per (dep_task_id, main SHA) —
   cheap `git grep`s). Failure → withhold dispatch; after a grace streak →
   born-at-L2 escalation naming the failed check + dependent → `blocked`
   (quiescence, mirrors the external-dep contract).

**Coverage caveat (scope):** only PRD-decomposed tasks carry manifests.
Recon/escalation-filed tasks are covered separately by `premise_lint`
(task 2231, recon-reliability W5-ξ) — a G4 seam, not this PRD's work.

## Resolved design decisions

1. **Sidecar = authored provenance; task metadata = operational contract.**
   The scheduler never parses YAML files at tick time. `commit_planning` copies
   checks into `metadata.delivered_checks` on the **producer** task; the gate
   reads dep task records it already has (`tasks_by_id`). This survives PRD
   file moves and works without repo-path discovery in the scheduler.
2. **Sidecar path strictly derived**: `re.sub(r'\.md$', '', prd_path) +
   '.capability-manifest.yaml'`. (The existing `.md` manifests drifted —
   `cross-project-task-deps.capability-manifest.md` vs
   `…-task-deps-prd.md`; the sidecar convention is mechanical, no drift.)
3. **Check kinds**: `grep` | `script` | `manual`. `grep` is the primary kind:
   an ERE evaluated against the **committed main tree** via
   `git -C <project_root> grep -E <pattern> main -- <paths…>` — immune to the
   dirty machine-operated checkout (survey §1.9). `expect: present|absent`
   (`absent` supports rejection-style capabilities). `script` follows the
   `before_done` precedent: repo-relative, must exist & be executable,
   bounded `timeout_secs`, exit 0 = delivered. `manual` is recorded in the
   sidecar but **never copied to metadata** — excluded from the gate.
4. **Gate applies to every terminal-status local dep carrying checks** — both
   `done` and `cancelled`. The gate trusts main, not status labels: a
   cancelled dep whose checks pass (capability landed elsewhere) satisfies;
   a done dep whose checks fail withholds. Deps without
   `metadata.delivered_checks` behave byte-identically to today.
5. **Fail-safe runner semantics** (mirrors external resolver-degraded, tasks
   1580/1855): a check that **errors** (git failure, script timeout, budget
   exhausted this tick) → dep not satisfied this tick, **no streak bump**, no
   escalation. A check that **ran and failed** → streak bump.
6. **Grace-then-escalate**: `delivered_checks.grace_cycles` (default 3)
   consecutive failing ticks → escalation filed with `severity='critical'`,
   `agent_role='orchestrator-scheduler'` (harness sentinel → born-at-L2 per
   `server.py:303,333`), category `dependency_capability`, summary
   `DEP_CAPABILITY_NOT_DELIVERED: task <dependent> — dep <id> <status> but
   check '<name>' fails on main@<sha12>`; detail carries kind, pattern/script,
   paths, expect, observed result, and the manual re-pend recipe. Dependent →
   `blocked` (open-L2 dedupe; no re-file while open). Grace absorbs the
   done→main propagation window (merge finalize vs scheduler tick).
7. **Per-main-SHA cache**: results keyed `(dep_task_id, main_sha)`;
   `git rev-parse main` once per tick, only when some pending task has a
   terminal dep with checks. Cache persists until main advances. Per-tick
   check budget (`delivered_checks.max_checks_per_tick`, default 50) keeps
   tick latency flat; over-budget checks are deferred (fail-safe wait).
8. **Stamping is fail-soft, loud**: no sidecar on disk → no-op (every existing
   batch byte-identical). Malformed sidecar / label absent → the flip
   proceeds, but the `commit_planning` response carries a structured
   `manifest_stamping` report (`{path, stamped: [...], missing_labels: [...],
   errors: [...]}`) so the decompose session sees it and fixes. Stamping never
   strands a planned batch on a docs artifact.
9. **Who commits the stamped sidecar**: `commit_planning` writes the file;
   the decompose session commits it (`git commit --only <sidecar>`) in the
   same skill turn — same convention as the .md manifest. The skill reference
   update (β) documents this.
10. **Schema lives in `shared/`** (`shared/src/shared/capability_manifest.py`):
    pydantic v2 models (`DeliveredCheck`, `ManifestCapability`, `ManifestTask`,
    `CapabilityManifestDoc`) + loader; `metadata.delivered_checks` validated
    via `register_metadata_submodel` (`shared/task_metadata.py:305` precedent —
    same pattern as `BeforeDone`/`Milestone`/`ExternalDep`). pydantic+pyyaml
    already deps of all three packages.
11. **Scope: local deps only.** Cross-project (`metadata.external_deps`) deps
    keep today's status-only gate — evaluating a foreign project's checks
    needs that repo's checkout; out of scope (future work, §Out of scope).
12. **Default-on**: `delivered_checks.enabled: true` — safe because the gate
    is inert for every task without the metadata. Knobs (`enabled`,
    `grace_cycles`, `check_timeout_secs`, `max_checks_per_tick`) join the
    green (hot-reloadable) tier per config-hot-reload conventions.

## Contract: sidecar schema + metadata schema + gate semantics (the α seam)

### Sidecar (`<prd-stem>.capability-manifest.yaml`, schema_version 1)

```yaml
prd: plans/<slug>-prd.md        # repo-relative; MUST equal the batch's metadata.prd_path
schema_version: 1
tasks:
  - label: "α"                  # matches metadata.prd_task_label
    task_id: null               # int | null; stamped by commit_planning — never author-supplied
    title: "…"                  # human aid, not load-bearing
    capabilities:
      - name: "kebab-case-capability-name"
        binding: "capability→producer (wired) — grep:shared/src/shared/task_metadata.py register_metadata_submodel"
        verdict: PASS           # authoring-time G3/G6 verdict; PASS required to queue
        delivered_check:        # OPTIONAL — omit or kind: manual to exclude from the gate
          kind: grep            # grep | script | manual
          pattern: "register_metadata_submodel\\('delivered_checks'"   # ERE, git-grep -E
          paths: ["shared/src/shared/"]   # optional pathspec list; default whole tree
          expect: present       # present | absent
          # script kind instead:
          # script: scripts/check_x.sh    # repo-relative, committed & executable
          # args: ["--flag"]              # exit 0 = delivered
          # timeout_secs: 30
```

Validation (α, enforced by the shared loader): `label` non-empty and unique
per doc; `task_id` int|null; `kind` ∈ {grep, script, manual}; `grep` requires
`pattern` + `expect`, forbids `script`; `script` requires `script` +
`timeout_secs`, forbids `pattern`/`expect`; `manual` forbids all check fields.
Malformed docs → structured `ValidationError` naming the entry.

### Producer-task metadata (stamped by γ)

```json
"delivered_checks": [
  {"name": "…", "kind": "grep", "pattern": "…", "paths": ["…"], "expect": "present"},
  {"name": "…", "kind": "script", "script": "scripts/check_x.sh", "args": [], "timeout_secs": 30}
]
```

Validated wherever present via `register_metadata_submodel('delivered_checks', …)`;
`commit_planning` is the canonical producer, but the model validates
author-supplied values too (producer-agnostic, like `before_done`).

### Gate semantics (δ/ε)

| Dep state (local) | delivered_checks state | Gate outcome |
|---|---|---|
| terminal, no `metadata.delivered_checks` | — | satisfied (byte-identical to today) |
| terminal, all checks PASS at current main SHA | cached per (dep, SHA) | satisfied |
| terminal, ≥1 check FAIL | streak < grace_cycles | not satisfied; withhold; hold-streak visibility event |
| terminal, ≥1 check FAIL | streak ≥ grace_cycles | born-at-L2 `dependency_capability` escalation naming the check; dependent → `blocked`; manual re-pend after fix |
| terminal, check ERRORS (git/timeout/budget) | — | not satisfied this tick; **no** streak bump (fail-safe wait) |
| non-terminal | any | not satisfied (unchanged) |

Side effects (escalation, streaks, status writes) live in the per-tick sweep —
`_deps_satisfied` stays a pure predicate taking a computed
`delivered_check_cache` parameter, exactly like `external_status_cache`.

## Boundary-test sketch (the ζ integration-gate signal)

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| 1 | Stamp on commit | planning batch filed with `prd_path`+`prd_task_label`; sidecar on disk with matching labels | sidecar file carries real task ids; `get_task` shows `metadata.delivered_checks` on producers; response `manifest_stamping.stamped` lists labels |
| 2 | Legacy batch | planning batch, no sidecar on disk | `commit_planning` byte-identical to today; no stamping report errors |
| 3 | Gate transparent | dep done; its checks PASS on main | dependent dispatches normally |
| 4 | Withhold | dep done; a check FAILs (capability absent from main) | dependent NOT dispatched; hold visibility event; no escalation before grace |
| 5 | Escalate | scenario 4 persists ≥ grace_cycles ticks | pending L2 escalation names check name/pattern/dep/main-SHA; dependent `blocked`; no duplicate while open |
| 6 | Self-heal in grace | capability lands on main (new SHA) before grace expiry | cache invalidated on new SHA; checks PASS; dependent dispatches; streak cleared; no escalation |
| 7 | Runner error | git grep fails / script times out | dep not satisfied this tick; no streak bump; no escalation; recovers next tick |
| 8 | Manual-only | dep done; sidecar capabilities all `kind: manual` | no metadata stamped; gate no-op (status-only, as today) |
| 9 | Malformed sidecar | sidecar present but fails α validation | flip proceeds; `manifest_stamping.errors` populated; no metadata written; nothing gated |
| 10 | Cancelled dep with checks | dep cancelled; checks FAIL | withheld → grace → escalation (same lane as done); checks PASS → satisfied |

## Pre-conditions for activating

None novel — all substrate verified on main 2026-07-13 (this session):

- `commit_planning` at `fused-memory/src/fused_memory/server/tools.py:3289`
  (already batch-reads task records — the stamping hook point).
- External-dep gate + sweep in `orchestrator/src/orchestrator/scheduler.py`
  (`_task_external_deps`, `external_status_cache` predicate arm, streak
  counters, `on_external_dep_block` → `harness.py` L1 filer) — the pattern δ/ε mirror.
- Born-at-L2: `BORN_AT_L2_SEVERITIES` + sentinel-role check
  (`escalation/server.py:303,333`); `orchestrator-scheduler` passes the sentinel.
- `register_metadata_submodel` (`shared/src/shared/task_metadata.py:305`);
  `BeforeDone` committed-script validation precedent in `submit_task`.
- pydantic ≥2.7 + pyyaml ≥6.0 in shared/fused-memory/orchestrator pyproject.
- ~15 `.md` manifest exemplars in `plans/`; decompose Step 2.5 authors them.

## Cross-PRD relationship

| Other PRD / owner | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `/prd` skill (`skills/prd/references/{decompose-mode,gates}.md`) | this PRD extends its decompose output format | sidecar emission + delivered_check authoring rules (Step 2.5) | **this PRD (β)** | queued |
| recon-reliability W5-ξ `premise_lint` (task 2231) | complementary, disjoint | premise linting for **recon/escalation-filed** tasks (no manifest exists for those) | task 2231 | in-flight elsewhere |
| W10 harness-supervision TruthReport (task 2242) | adjacent, no collision | 2242 owns *recovery* ground truth (post-failure); this PRD owns *dispatch eligibility* (pre-dispatch) | each its own | no shared mechanism |
| Cross-project task deps (`plans/cross-project-task-deps-prd.md`, landed) | this PRD mirrors its gate pattern | `external_status_cache` predicate arm + sweep/streak/escalate shape; external deps stay status-only | landed PRD (unchanged) | wired |

## Decomposition plan

Intra-batch deps by Greek label; all filed `planning_mode=True`, flipped together.

- **α — Shared sidecar schema + delivered-check models** *(intermediate → β, γ, δ; high)*
  Modules: `shared/`. `capability_manifest.py` models + YAML loader +
  `register_metadata_submodel('delivered_checks', …)`.
  Signal: this PRD's own committed exemplar sidecar parses and validates in a
  CI test; malformed fixtures (each §Contract validation rule) are rejected
  with structured errors naming the entry.
- **β — /prd skill emits the YAML sidecar; exemplar committed** *(leaf; medium; deps α)*
  Modules: `skills/prd/references/`, `plans/`.
  Update decompose-mode Step 2.5 + gates.md manifest section: emit the YAML
  twin, delivered_check authoring rules (pattern-anchored, never file:line;
  `manual` for non-mechanical), commit-the-stamped-sidecar step after
  `commit_planning`.
  Signal: the skill references name the sidecar contract, and this PRD's
  exemplar sidecar exists on main validating against α's loader (CI fixture).
  Consumer: every future `/prd` decompose session; γ.
- **γ — `commit_planning` stamps sidecar + copies delivered_checks** *(intermediate → ζ; high; deps α)*
  Modules: `fused-memory/`.
  Locate sidecar from batch `metadata.prd_path`; stamp `task_id` per label;
  write file back; copy mechanical checks into producer
  `metadata.delivered_checks` (mind the update_task metadata merge semantics);
  structured `manifest_stamping` report in the response; fail-soft absent/
  malformed sidecar (scenarios 1, 2, 9).
  Signal: integration test drives a planning batch with a sidecar fixture
  through `commit_planning` → ids stamped on disk, metadata visible via
  `get_task`, legacy batch byte-identical.
- **δ — Scheduler delivered-check gate + runner + per-SHA cache** *(intermediate → ε, ζ; high; deps α)*
  Modules: `orchestrator/`.
  Runner (`git grep` kind + committed-script kind, timeout-bounded); per-tick
  sweep computing `delivered_check_cache`; `_deps_satisfied` gains the cache
  parameter (pure, default-None byte-identical); hold visibility events;
  fail-safe error semantics; per-tick budget.
  Signal: scheduler test — dep done with failing grep check ⇒ dependent not
  dispatched + hold event; commit making the check pass ⇒ dispatched next tick
  (scenarios 3, 4, 6, 7, 10 predicate half).
- **ε — Grace-streak escalation + config knobs + ops docs** *(intermediate → ζ; high; deps δ)*
  Modules: `orchestrator/`, CLAUDE.md.
  Streak counter; born-at-L2 filing per §Resolved 6 (open-L2 dedupe, manual
  re-pend recipe in detail); dependent → `blocked`; `delivered_checks.*` knobs
  in the green reload tier; CLAUDE.md dispatch-policy subsection.
  Signal: with a persistently failing check, a pending L2 escalation names the
  exact failed delivered_check and the dependent shows `blocked`; resolving
  after the capability lands + manual re-pend dispatches it (scenario 5).
- **ζ — End-to-end integration gate** *(leaf — G2 top signal; high; deps γ, δ, ε)*
  Modules: `orchestrator/`, `fused-memory/` (test-side).
  Synthetic e2e in CI: file producer+dependent batch with a sidecar whose
  capability is absent from main → `commit_planning` stamps → flip producer
  done (scope-cut simulation) → dependent withheld → grace → L2 escalation
  naming the check → land the capability commit → re-pend → dependent
  dispatches. The full §Boundary-test sketch table green.

## Out of scope

- **Cross-project delivered checks** — external deps keep the status-only
  gate; a foreign project's checks need its checkout (future PRD; the
  registry could publish per-task check results).
- **Backfill** of the ~15 existing `.md` manifests into sidecars — new PRDs
  only. (This PRD's own sidecar is hand-authored as the α/β exemplar.)
- **Recon/escalation-filed task coverage** — `premise_lint`, task 2231.
- **LLM judgment at dispatch time** (survey Open-Q 1's other branch) — the
  gate is mechanical-only; `manual` capabilities are deliberately excluded.
- Orchestrator consumption of `user_observable_signal` / `consumer_ref`
  metadata — still substrate for a future tracking-infra session.
- Auto-re-pend of a blocked dependent when checks later pass — keeps the
  external-dep contract's manual re-pend (a failed delivered check means the
  producer closed without delivering: a human decision, not a retry).

## Open questions (surfaced but not decided in this session)

1. **Escalation category string** — new `dependency_capability` vs reusing
   `dependency_discovered` (field is free-form `str`). **Suggested:** new
   category; watchers key urgency off severity, not category. Decide in ε.
2. **script-kind execution tree** — run the committed script from the working
   checkout (approximation: checkout ≈ main; simple) vs extracting from the
   main tree (`git show main:<script>`; exact but fiddly). **Suggested:**
   working checkout with the approximation documented; grep-kind (which reads
   the main tree exactly) is the primary kind anyway. Decide in δ.
3. **Also stamp task ids into the `.md` manifest** for human cross-reference —
   nice-to-have. **Suggested:** no (one mechanical artifact is the contract;
   the .md stays authoring provenance). Decide in γ.
4. **grace_cycles default** — 3 ticks assumed (~45 s at 15 s polls) to absorb
   merge-finalize → tick propagation. Tune in ε if the e2e shows noise.
