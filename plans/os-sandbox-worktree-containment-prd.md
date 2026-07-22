# PRD: OS filesystem sandbox — implementer/debugger/simple_task worktree containment

**Status**: active — authored 2026-07-22 under the esc-2508-1 PRD gate (born-at-L2
`milestone_gate`, dep task 2505 landed). Research basis:
`plans/os-sandbox-worktree-containment-research-2026-07-22.md` (substrate census,
host readiness, write-set gap analysis — all G3 claims verified there and
re-pinned in this session). The 6 open questions in that doc were ratified by
Leo in this session; resolutions recorded below.
**Mode**: B+H — single-package blast radius (orchestrator) but a load-bearing
seam: every code-writing agent dispatch plus the shared main-repo `.git`.
§Write-set contract + §Enforcement matrix inline.

## Goal

Code-writing agents (implementer, debugger, simple_task) are **OS-enforced**
read-only everywhere except their task's own writable scope — replacing today's
advisory prompt discipline ("Scope Boundary" text that promises an enforcement
which doesn't exist) with deterministic kernel containment (Landlock), and
eliminating the "how did a file outside this task's worktree change?" RCA class.

User-observable:

1. On a sandboxed fleet, an implementer instructed to write `<main>/CANARY`, a
   sibling worktree file, another task's `.task-meta/`, or
   `~/.claude/settings.json` observes a **deterministic permission error**
   (EACCES under landlock / EROFS under bwrap) — proven by a committed
   containment-probe report (task γ4) and pinned forever by a real-kernel CI
   matrix suite (task α4).
2. A normal task completes **end-to-end under the sandbox**: per-step
   `git commit` from the worktree, plan-tools/verdict-tools sidecar writes,
   `uv run pytest`, merge lands — nothing about the happy path changes.
3. Every sandboxed invocation emits a **`sandbox_applied` event** (backend +
   writable-set digest) in the event store; a refused dispatch (fail-closed)
   emits `sandbox_unavailable` + one deduplicated escalation — operators see
   exactly when containment was active and when it couldn't be.
4. Incident classes with prior art on record become impossible for sandboxed
   roles: production hand-patch outside the worktree (`oneoff-2026-07-04`),
   `cd`-fail amend rewriting main, sibling-worktree/`.task-meta` writes.

## Background

- **The sandbox substrate is ~90% built, tested, and wired — never enabled.**
  `SandboxConfig` (`orchestrator/src/orchestrator/config.py:966-978`, shipped
  default `enabled: false` via `defaults.yaml:454-455`), dispatcher
  (`agents/sandbox_dispatch.py`, currently fail-open), bwrap + landlock
  backends (`agents/sandbox.py`, `agents/landlock.py`, `agents/landlock_exec.py`)
  with a real kernel-enforcement test (`test_landlock.py:106-148`), role gating
  (`AgentRole.sandboxed=True` on IMPLEMENTER `roles.py:480` and DEBUGGER
  `roles.py:544`), call-site wiring (`workflow.py`: `sandbox_modules =
  self.modules` when `config.sandbox.enabled and role.sandboxed` →
  `agents/invoke.py` → `wrap_command`), and a `writable_extras` param plumbed
  through all three modules with **no live caller** — the natural vehicle for
  the carve-outs this PRD adds.
- **Why it ships off:** the writable set predates three load-bearing realities.
  (1) Per-step `git commit` (required by `agents/briefing.py:370,415-416`)
  writes the **shared main `.git`** of the linked worktree — no `.git` path is
  writable. (2) W11 moved durable task artifacts to
  `<worktree_base>/.task-meta/<name>/` outside the worktree; the plan-tools /
  verdict-tools stdio MCP servers are children of the sandboxed process and
  write there (landlock resolves the `<wt>/.task/plan.json` symlink to its
  target). (3) Module-granular writable dirs break real builds (root
  lockfiles, per-subproject `.venv`s from `uv run` in `test_command`).
- **Empirical `.git` write-set** (straced this session, linked-worktree
  `git commit`): `objects/**` (+ `objects/maintenance.lock` from auto-
  maintenance), `refs/heads/task/<id>` + `.lock`, `logs/refs/heads/task/<id>`
  (branch reflog — missed by the research doc), `worktrees/<name>/**` (index,
  HEAD.lock, logs/HEAD, COMMIT_EDITMSG, MERGE_*). A narrow set is therefore
  feasible; `main`'s ref and all non-task refs stay RO. A gc-packed task ref
  still updates fine (loose-ref shadowing); ref *deletion* would need
  `packed-refs` — agents never delete refs.
- **Host readiness:** kernel 6.14, Landlock ABI 6 live (syscall-probed).
  bwrap 0.9.0 present but effectively dead fleet-wide (Bun segfault under
  userns; reify config comment). **Landlock is the fleet backend**; `auto`
  already prefers it. `landlock.py` syscall numbers are x86_64-only — fine for
  this fleet, documented.
- **Posture model to copy:** fused-memory recon confinement
  (`reconciliation/sandbox_guard.py`, task 1935) — same backends, default-ON,
  **fail-closed** (`RemediationSandboxUnavailable`).
- **2505 synergy:** wrap is computed per-invocation and a steward
  `granted_files` grant re-invokes the implementer — grants naturally
  re-materialize into the ruleset; under whole-worktree granularity they need
  no ruleset change at all.
- **Warm lanes:** plain recycled dirs; wrap keys off the assigned lane path per
  invocation — no interaction.

## Resolved design decisions (Leo-ratified 2026-07-22 unless noted)

1. **Granularity: whole-worktree.** Writable root = the entire `<worktree>`
   (replacing per-module dirs). Survives lockfiles/.venvs/incidental sibling
   touches; matches the gate's stated goal; inert to 2505 grants. Plan-target
   tightening (writable scope = plan files + carve-outs) is **explicitly not
   in this PRD** — capstone task δ1 books the analysis/discussion.
2. **Shared `.git`: narrow set.** `objects/` + `worktrees/<name>/` +
   `refs/heads/task/` + `logs/refs/heads/task/` — probe-verified sufficient
   (§Background). `refs/heads/main`, all non-task refs, `packed-refs`, and
   `.git` root stay RO. Corollary: **auto-gc/maintenance must be disabled** in
   orchestrator-managed shared repos (`gc.auto=0`, `maintenance.auto=false`) —
   background gc under the narrow set would fail benignly but noisily; gc
   ownership moves to the orchestrator/operator out-of-band (task α5).
   Residual accepted: sibling *task* refs and the shared object store remain
   writable (landlock create-rights are directory-granular).
3. **Role scope: implementer + debugger + SIMPLE_TASK.** SIMPLE_TASK is an
   implementer with a smaller pipeline (`roles.py:1412`, same
   Edit/Write/Bash surface + per-step commit contract) — inclusion is
   `sandboxed=True` on the role. MERGER/STEWARD/architect-Bash: out of scope
   (§Out of scope).
4. **Fail posture: fail-closed.** `enabled` + `role.sandboxed` + resolved
   backend unavailable → **refuse the invocation** and escalate (mirror
   `RemediationSandboxUnavailable`). `backend: none` remains an explicit
   operator escape hatch (refusal applies to `auto`/`landlock`/`bwrap`
   resolving to nothing). Escalation is **deduplicated** — one per
   backend-state change per orchestrator process, not one per refused
   invocation (INV-4 `storm-escape-required`).
5. **`~/.claude`: narrow now.** Writable: `~/.claude/fleet/` +
   `~/.claude/hooks/state/` only (session state already lives in-worktree via
   per-task `CLAUDE_CONFIG_DIR`, `agents/invoke.py:276`). Closes the
   config/hook/skill poisoning surface flagged in
   `plans/harness-backend-reconnect-pi-prd.md:305`. Task α1 audits the actual
   writer list before the constants freeze (and may drop `hooks/state/` if
   task agents don't load `~/.claude/settings.json` at all under the
   redirected config dir).
6. **Rollout: DF canary → reify → rest of fleet.** Green = ≥10 tasks e2e under
   sandbox + 1 containment probe observing denial + 0 sandbox-attributable
   blocks, over ≥3 days (delayed-milestone soak gate, task γ5). Per-project
   flips are **filed in each project's own registry** with cross-project
   `project_id:task_id` external deps on the soak gate (the CLAUDE.md
   deterministic-deploy dep convention) — a DF task agent can't (and now
   OS-provably can't) edit a sibling repo.
7. **Shipped default stays `enabled: false`.** Enablement is explicit
   per-project config (restart-tier). Default-on is a possible future once the
   fleet has soaked — not this PRD.
8. **Probe is report-based, not escalation-based.** The probe implementer
   records denial errnos in an in-worktree report and commits it; verify greps
   the report. No `scope_violation` escalation churn for an *expected*
   denial; the escalation path stays reserved for real violations.
9. **Denial errno is backend-specific**: EACCES (landlock) / EROFS (bwrap
   RO-bind). All assertions accept either; docs say "permission error", not a
   single errno.
10. **bwrap is legacy-passthrough.** The write-set flows through the existing
    backend-agnostic params (`writable_modules`/`writable_extras`), so bwrap
    inherits it for free; no bwrap-specific work, no new bwrap tests beyond
    existing parity suites. Landlock is the supported fleet backend.
11. **Single-source write-set.** One `compute_write_set()` produces the
    writable path list consumed by both backends via the existing params —
    never duplicated per-backend (INV-5 `no-lockstep-duplication`).
12. **Eval runner keeps `enabled=False`** (`evals/runner.py:244`) — eval
    worktrees live under `/tmp` where restrictions would be nullified anyway;
    forcing it off stays correct.
13. **Dashboard config edited in place** under its legacy filename
    (`dashboard/orchestrator.yaml`) — no filename migration (task-2699
    convention: never migrate module configs).

## Write-set contract (§Contract)

Inputs available at the call-site per invocation: worktree path, worktree
admin name (`.git` file → `gitdir`), main-repo common `.git` dir, task id,
`<worktree_base>`. `compute_write_set()` is the single owner (D11).

| Path | Access | Why |
|---|---|---|
| `<worktree>/` (entire tree) | RW | D1 — whole-worktree granularity |
| `<worktree_base>/.task-meta/<name>/` | RW | plan-tools/verdict-tools sidecar writes (symlink target; landlock decides on target) |
| `<main>/.git/objects/` | RW | loose objects, tmp_obj_*, maintenance.lock |
| `<main>/.git/refs/heads/task/` | RW | branch ref + `.lock` (dir-granular create rights) |
| `<main>/.git/logs/refs/heads/task/` | RW | branch reflog |
| `<main>/.git/worktrees/<name>/` | RW | index, HEAD.lock, logs/HEAD, COMMIT_EDITMSG, MERGE_* (incl. unlink) |
| `~/.cache/uv/` | RW | uv cache locks (even warm) |
| `~/.claude/fleet/` | RW | session-registry hooks |
| `~/.claude/hooks/state/` | RW | hook state (α1 may drop — see D5) |
| `/tmp/`, `/dev/` | RW | orchestrator temp files, pytest FIFO, verify-slot semaphores, device nodes |
| **Everything else** — `<main>` tree, sibling worktrees, other tasks' `.task-meta/`, `~/.claude` remainder, `.git` root (`packed-refs`, `config`, `HEAD`), `refs/heads/main`, `~/.local/share/uv` | **RO** | the containment |

Invariants:
- The set is computed per invocation from the assigned worktree/lane — a 2505
  `granted_files` re-invoke needs no ruleset change (grants are in-worktree).
- Read access is unrestricted (whole FS) — read-containment out of scope.
- The writable-set **digest** is stamped into the `sandbox_applied` event so an
  operator can diff what a given invocation could touch (INV-2).

## Enforcement matrix (§Boundary-test sketch — task α4's suite, real kernel)

| # | Scenario | Pre | Post |
|---|---|---|---|
| 1 | write `<wt>/src/x.py` | sandbox active, whole-wt set | succeeds |
| 2 | per-step `git commit` from `<wt>` | staged change | succeeds; objects + `refs/heads/task/<id>` + reflog + worktrees/<name> written |
| 3 | write `<main>/CANARY` | — | permission error (EACCES/EROFS) |
| 4 | write sibling `<base>/<other-wt>/f` | second worktree exists | permission error |
| 5 | write `<base>/.task-meta/<other>/x` | other task's meta dir exists | permission error |
| 6 | `git update-ref refs/heads/main <sha>` from `<wt>` | — | fails (ref lock EACCES); `main` unchanged |
| 7 | child process writes `<base>/.task-meta/<name>/plan.json` via `<wt>/.task/plan.json` symlink | plan-tools shape | succeeds (ruleset inherited by child tree) |
| 8 | `uv run pytest` with warm cache | `~/.cache/uv` populated | succeeds |
| 9 | write `~/.claude/settings.json` | — | permission error |
| 10 | write `~/.claude/fleet/sessions/<x>` | — | succeeds |
| 11 | write `/tmp/<scratch>` | — | succeeds |
| 12 | `git commit` with auto-maintenance disabled | α5 config applied | succeeds with zero gc/EROFS noise |

## Pre-conditions for activating

All verified landed (G3 — research doc §2 + this session's re-pin):
`SandboxConfig` + restart-tier gating; dispatcher + both backends +
`writable_extras` plumbing; role gating + call-site wiring + capability-wiring
pin tests; landlock ABI 6 on fleet hosts; 2505 scope machinery; W11
`.task-meta` layout; recon fail-closed prior art; `DeterministicRunner`
(deploy + predicate kinds) and delayed-milestone substrate for γ3/γ5–γ7;
cross-project external-dep gate for γ6/γ7. **No unbuilt substrate — every leaf
below consumes what exists.**

## Cross-PRD relationship

| Other PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| `plans/harness-backend-reconnect-pi-prd.md` (§~305 security follow-up) | this PRD **resolves** its flagged item | `~/.claude` write-surface narrowing | this-prd | wired (D5, α1/α2) |
| `plans/capability-delivered-checks-prd.md` | consumes (process substrate) | manifest sidecar + `delivered_checks` stamping at decompose | other-prd (landed) | wired |

No contested seams. The touched⊆declared merge gate (detective complement) is a
possible future PRD — named in §Out of scope, no seam today.

## Decomposition plan

Labels are planning labels; ids assigned at decompose. All tasks
`project_id=dark_factory` except γ6 (filed in reify's registry).

**Phase α — write-set correctness**

- **α1 — Audit `~/.claude` writers for sandboxed agents** *(intermediate →
  α2)*. Static audit: every hook wired in `~/.claude/settings.json` + spawn
  hooks; whether task agents (redirected `CLAUDE_CONFIG_DIR`) load
  `~/.claude/settings.json` at all. Signal: findings appendix committed into
  this PRD (or sibling note) naming the final `~/.claude` writable list α2
  freezes. `metadata.complexity=simple` candidate.
- **α2 — `compute_write_set()` single source of truth** *(intermediate → α3,
  α4)*. New fn + `WriteSet` dataclass in `orchestrator/agents/`: whole-worktree
  root + every §Contract carve-out, derived from worktree path/admin
  name/common-dir/task id. Unit-pinned (path derivation incl. `.git`-file
  `gitdir:` parsing, symlink-target resolution for `.task-meta`). Consumed by
  both backends via existing params — no per-backend path lists (INV-5).
- **α3 — Call-site wiring + SIMPLE_TASK flag** *(intermediate → α4, γ2)*.
  `workflow.py` sandbox block: `self.modules` → `compute_write_set()`
  (writable root + `writable_extras`); `sandboxed=True` on SIMPLE_TASK
  (`roles.py:1412`); update `test_agent_capability_wiring.py` pins. Signal:
  wiring suite green; a dry `wrap_command` for each of the three roles carries
  the full contract set.
- **α4 — Kernel-enforcement matrix suite** *(leaf; CI-fixture signal)*. The 12
  §Enforcement-matrix rows as a real-kernel landlock test module beside
  `test_landlock.py:106-148` (skip-if-no-landlock like the existing test).
  Signal: suite green in CI on a landlock host; rows 3–6/9 prove denial, rows
  1–2/7–8/10–12 prove the happy path.
- **α5 — Shared-repo git maintenance discipline** *(intermediate → γ2)*.
  Orchestrator-managed shared repos get `gc.auto=0` + `maintenance.auto=false`
  applied idempotently (startup or worktree-create path); operator gc note in
  docs. Signal: `git config get gc.auto` = 0 on the DF main repo; matrix row
  12 green.

**Phase β — posture + observability** (parallel with α)

- **β1 — Fail-closed dispatch guard** *(intermediate → γ2)*. In
  `sandbox_dispatch`/call-site: `enabled` + `role.sandboxed` + no resolved
  backend → raise (refuse invocation) + escalate, mirroring
  `RemediationSandboxUnavailable`; `backend: none` stays an explicit escape;
  escalation deduped per backend-state change (INV-4). Signal: unit tests —
  refusal path raises + files exactly one escalation across N refused
  invocations; `none` + `enabled` runs unsandboxed with a WARN.
- **β2 — `sandbox_applied` / `sandbox_unavailable` events** *(intermediate →
  γ1, γ2)*. New `EventType`s (`event_store.py:44`): per-sandboxed-invocation
  `sandbox_applied` {backend, writable-set digest, role, task_id};
  `sandbox_unavailable` on fail-closed refusal. Signal: events visible in the
  event store for a simulated invocation; consumed mechanically by γ1.

**Phase γ — enablement + soak**

- **γ1 — Soak predicate script** *(intermediate → γ5)*.
  `scripts/check_sandbox_soak.sh` (+ helper): exit 0 iff ≥10 distinct tasks
  with `sandbox_applied` events reached `done` AND probe report present on
  main AND 0 sandbox-attributed blocks (structured query over event store +
  task records — never transcript-grep, INV-2). Exit-code contract per
  `before_done.kind='predicate'`. Signal: script self-test against fixture
  event-store passes; pre-soak run exits non-zero with a clear reason line.
- **γ2 — DF config flip commit** *(intermediate → γ3)*. `sandbox: {enabled:
  true, backend: landlock}` in `dark-factory-orchestrator.yaml`.
  `metadata.complexity=simple`. Deps: α3, α4, α5, β1, β2. Signal: block
  present on main; config loads (schema-valid) — inert until restart
  (restart-tier).
- **γ3 — DF restart deploy** *(intermediate → γ4, γ5)*. `task_kind=
  deterministic`, `before_done` deploy targeting DF's own orchestrator unit
  (self-unit → detached `systemd-run` scheduled path, provenance
  `deterministic-deploy-scheduled`). Dep: γ2. Signal: DF orchestrator running
  with sandbox active — next dispatched implementer emits `sandbox_applied`.
- **γ4 — Containment probe task** *(leaf)*. Normal task, dep γ3: implementer
  attempts the denied writes (matrix rows 3–6, 9), records each errno in
  `<wt>/probe-report.md`, commits; verify greps the report for denial
  evidence on every row. Signal: probe report on main proving deterministic
  denial in production (D8, D9).
- **γ5 — Soak gate** *(leaf)*. `task_kind=deterministic`,
  `before_done.kind='predicate'` → γ1's script; `metadata.milestone =
  {mode: delayed, after_secs: 259200}` (3 days from deps-satisfied anchor).
  Deps: γ1, γ3, γ4. Exit 0 → `done` (`deterministic-milestone`); non-zero →
  `milestone_check_failed` born-at-L2 (resolution `resume` **re-runs** the
  check — "wait longer" is a safe resolve).
- **γ6 — Reify flip + deploy** *(leaf; filed in reify's registry)*.
  Reify-side config edit (`sandbox: {enabled: true, backend: landlock}`) +
  drain-aware restart, as reify tasks with external dep
  `dark_factory:<γ5-id>`. Signal: reify orchestrator emits `sandbox_applied`
  on implementer dispatches. (Decompose session files via reify
  `project_root`; wording avoids DF-exclusive path tokens.)
- **γ7 — Fleet remainder + docs flip** *(leaf)*. Dashboard config
  (`dashboard/orchestrator.yaml`, edited in place per D13) + any remaining
  sibling project configs (enumerate targets at decompose from the registry
  census; sibling-repo projects get their own registry-filed flip tasks per
  D6) + flip the dashboard "when ready" comment + document landlock-only/
  x86_64 status. External dep `reify:<γ6-id>`. Signal: fleet census shows no
  `enabled: false` factory target (evals excepted, D12); dashboard comment
  gone.

**Phase δ — capstone**

- **δ1 — Plan-target-granularity analysis gate** *(leaf)*. `task_kind=
  deterministic`, `always_escalates=true`, no `before_done` (pure gate). Dep:
  γ7. Escalation brief: analyse tightening writable scope from whole-worktree
  to plan files + carve-outs ("plan target enforcement via sandbox") — 2505
  grant interplay, EROFS-vs-lock agent UX, lockfile/`.venv` carve-outs,
  expected outcome a research session → possible follow-up /prd. Signal:
  born-at-L2 escalation pending with the brief (the esc-2508-1 pattern,
  by Leo's Q1 ratification).

G2 note: α1/α2/α3/α5/β1/β2/γ1/γ2/γ3 are intermediates naming their consumers
above; leaves are α4 (CI fixture), γ4 (committed probe report), γ5
(predicate-verified done), γ6/γ7 (deployed-and-verified fleet state), δ1
(gate escalation). G7 walked: INV-4 handled in β1 (dedup), INV-5 in α2/D11,
INV-2 in β2/γ1 (structured events, no log-scrape), INV-3 via
DeterministicRunner's built-in restart verify (γ3/γ6), INV-1 via the contract
table being machine-pinned by α4's matrix. No waivers needed.

## Out of scope (explicit)

- **MERGER / STEWARD / architect-Bash sandboxing** — merger's rebase `.git`
  surface needs its own trace; steward has full authority by design.
- **Module/plan-granular tightening** — δ1 books the analysis; not designed
  here.
- **Read containment / secret scoping** — whole FS stays readable (incl.
  `~/.claude` remainder); a future concern.
- **Network scoping** — Landlock ABI 4+ could; neither backend does today.
- **touched⊆declared merge gate** — detective complement to this preventive
  control; separate PRD if pursued.
- **Default-on shipping** (D7) and **eval-runner enablement** (D12).
- **bwrap parity work** beyond free param-passthrough (D10).

## Open questions (surfaced but not decided in this session)

1. **Landlock rights breadth on carve-out dirs.** `worktrees/<name>/` needs
   unlink/rename (MERGE_* cleanup) — confirm the existing ruleset's
   access-rights mask covers FS_REMOVE_FILE/FS_REFER on writable dirs.
   **Suggested resolution:** grant the full RW rights set on all writable
   paths (matches current backend behavior). Decide in α2.
2. **Sandbox-attribution heuristic in γ1** — exact event/block-reason query
   defining "sandbox-attributable block". **Suggested resolution:** blocked
   tasks whose block window contains a `sandbox_unavailable` event or whose
   escalation detail matches EACCES/EROFS on an out-of-set path. Decide in γ1.
3. **`~/.local/share/uv` RO edge** — a `test_command` needing a
   not-yet-installed Python would make `uv` try to download → EACCES → loud
   task failure. **Suggested resolution:** accept (operator installs
   interpreters fleet-side; failure is loud and attributable); revisit if the
   soak hits it. Decide during γ soak.
4. **`hooks/state/` in the writable set** — drop if α1 shows task agents never
   load `~/.claude/settings.json` under redirected `CLAUDE_CONFIG_DIR`.
   Decide in α1→α2.
