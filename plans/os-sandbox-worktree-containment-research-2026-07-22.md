# OS Filesystem Sandbox for Implementer/Debugger Worktree Containment — Research (esc-2508-1)

Research session 2026-07-22, feeding a likely `/prd`. Escalation: **esc-2508-1**
(PRD gate, born-at-L2, `milestone_gate`, dep task 2505 landed). Do-not-resolve
until the PRD is authored/ratified.

**Goal (from the gate):** implementer/debugger agents filesystem-contained to
their worktree — whole FS read-only except the task's own writable scope —
replacing today's advisory prompt discipline with deterministic OS containment,
eliminating the "how did a file outside this task's worktree change?" RCA class.

---

## 1. Headline finding

**The sandbox subsystem is ~90% built and tested; it has never been enabled
because its writable set is stale against three load-bearing realities.** A
bare config flip today would break every implementer run within minutes (its
own `git commit` would EROFS). The PRD is therefore *not* "build a sandbox" —
it is "correct the write-set, decide granularity, flip fail-open→fail-closed,
and roll out".

## 2. What exists on main (verified)

### Substrate (all implemented, tested, wired — G3)
- **Config**: `SandboxConfig {enabled, backend: auto|bwrap|landlock|none}` —
  `orchestrator/src/orchestrator/config.py:966-978`. Shipped default
  `enabled: false` (`defaults.yaml:453-455` overrides the pydantic `True`).
  **Restart-tier** — `sandbox.*` deliberately absent from `RELOADABLE_FIELDS`;
  backend is a module-global set once in `Harness.__init__`
  (`harness.py:984-985` → `sandbox_dispatch.set_backend`).
- **Dispatcher**: `agents/sandbox_dispatch.py` — `auto` resolves
  landlock > bwrap > none; **fail-open** (unavailable backend → WARN + run
  unsandboxed, :63-70,107-115).
- **Backends**: bwrap (`agents/sandbox.py`, RO-bind `/` + selective binds) and
  landlock (`agents/landlock.py` + `landlock_exec.py` standalone
  restrict-self launcher, v1 fs ruleset, inherited by the whole child tree).
  Both fully implemented with a real kernel-enforcement test
  (`test_landlock.py:106-148`) plus dispatch/argv-parity/capability-wiring
  suites.
- **Role gating**: `AgentRole.sandboxed=True` on exactly **IMPLEMENTER**
  (`roles.py:480`) and **DEBUGGER** (`roles.py:544`); wired at
  `workflow.py:9898-9902` (`sandbox_modules = self.modules` when
  `config.sandbox.enabled and role.sandboxed`) → `invoke.py`
  `_invoke_claude_with_sandbox` → `wrap_command` per invocation (also codex/
  gemini/pi arms). Pinned by `test_agent_capability_wiring.py:207-212`.
- **`writable_extras` param** exists through all three modules but **no live
  caller passes it** — the natural vehicle for the new carve-outs.
- **Prior production art**: fused-memory recon confinement
  (`reconciliation/sandbox_guard.py`, task 1935) reuses the same backends,
  default-ON and **fail-CLOSED** — the posture model to copy.
- **Prompt already promises it**: implementer/debugger "Scope Boundary" text
  (`roles.py:458-466, 523-531`) says "you will get a permission error" —
  currently aspirational; enabling makes the prompt true.

### Host readiness (verified empirically this session)
- Kernel 6.14, **Landlock ABI 6** active in LSM stack (syscall-probed live).
- bwrap 0.9.0 present and works despite Ubuntu's
  `apparmor_restrict_unprivileged_userns=1` (shipped `/etc/apparmor.d/bwrap`
  profile) — smoke test: RO-root + writable bind OK, outside write → EROFS.
- **But bwrap is effectively dead fleet-wide**: Bun v1.3.13 segfault under
  bwrap userns on kernel 6.17 (dashboard/orchestrator.yaml:62 comment), reify
  config says "bwrap uid map broken on this kernel". **Landlock is the
  intended backend** (needs no userns at all; `auto` already prefers it).
  Caveat: `landlock.py` syscall numbers are x86_64-only (fleet is x86_64).

### Enablement census
Never enabled anywhere: defaults + dashboard + reify + all sibling project
configs are `enabled: false` (dashboard comment: "flip to `enabled: true,
backend: landlock` when ready"); DF's own yaml has no block. Eval runner
forces `enabled=False` (keep). Only production use of the backends is recon
confinement.

## 3. Why a bare flip breaks — the three write-set gaps

Current writable set (both backends): `<worktree>/<locked-module>` dirs +
`<worktree>/.task` + `~/.claude` + `/tmp` + `/dev`. Predates two structural
migrations and one contract:

1. **Main-repo `.git` (BLOCKER)** — worktrees are standard linked worktrees
   (`<wt>/.git` → `gitdir: <main>/.git/worktrees/<name>`). The implementer
   contract *requires* per-step `git commit` from inside the sandbox
   (`briefing.py:370,415-416`), which writes `<main>/.git/objects/`,
   `<main>/.git/worktrees/<name>/` (index/HEAD/logs), and
   `refs/heads/task/<id>`. No `.git` path is in the write-set; no code or doc
   even discusses it. **This gap alone explains why it ships off.**
2. **`.task-meta` sibling dir (BLOCKER)** — W11 moved durable task artifacts
   to `<worktree_base>/.task-meta/<worktree-name>/` *outside* the worktree
   (`<wt>/.task/plan.json` is a symlink into it). The plan-tools/verdict-tools
   stdio MCP servers are **children of the sandboxed claude process** and
   write there — denied under the current set. (Landlock decides on the
   symlink *target* path, so the target dir must be writable.)
3. **Module-granular ≠ reality (BLOCKER)** — agents legitimately write outside
   locked-module dirs *within* the worktree: root lockfiles, per-subproject
   `.venv`s created by `uv run` segments of `test_command`, incidental
   sibling-file touches. An April memory already flags "the PRD must settle
   writable-scope granularity."

Plus two accommodations:
- **`~/.cache/uv`** — agents run `uv run pytest`/`uv sync`; uv takes cache
  locks even warm. Needs write (or an offline/`UV_NO_SYNC` discipline).
  `~/.local/share/uv` + `~/.local/bin` need read+exec only (RO root covers).
- **`/tmp`** — stays writable (orchestrator-created sysprompt/mcp-config temp
  files, pytest jobserver FIFO, verify-slot semaphores, neutral-cwd scratch).
  Known gotcha on record: a worktree placed *under* `/tmp` nullifies
  restrictions (evals do this but evals force sandbox off).

Helpful discovery: per-task `CLAUDE_CONFIG_DIR` is **inside the worktree**
(`<wt>/.task/claude-config-<id>`, credentials + session transcripts) — session
state needs no `~/.claude` write. What *does* write `~/.claude`: the
session-start/stop hooks' fleet registry (`~/.claude/fleet/sessions/…`) and
hook state dirs. Current backends grant **all of `~/.claude` writable** —
already flagged as a security follow-up (config/hook/skill poisoning surface,
harness-backend-reconnect-pi-prd.md:305).

## 4. Current containment layers (what the sandbox replaces/complements)

| Layer | Class | Gap |
|---|---|---|
| Role-prompt "Scope Boundary" | Advisory | Pure convention; claims an enforcement that doesn't exist |
| Module locks (`LockTable`, file-granular at depth) | Preventive vs *concurrent dispatch* only | No effect on a running process's writes |
| `scope_violation` escalation | Advisory detection (agent self-reports) | Nothing detects out-of-scope writes automatically |
| 2505 scope machinery (`granted_files` → `_set_task_scope` → `handle_blast_radius_expansion`; `_check_scope_invariant` at merge entry) | Choke-pointed grants + detective tripwire | Keeps *declared* scope honest; not an enforcement |
| Merge gate `_check_plan_files_touched_in_branch` | Preventive — **wrong direction** | Checks declared⊆touched (phantom-done); **no touched⊆declared gate exists** — out-of-scope writes merge freely |
| Harness permissions | None | `--permission-mode bypassPermissions` on every agent; no `--add-dir`; path-unrestricted Bash |
| OS sandbox | Preventive by design | **Disabled everywhere** |

**Escape hatches with incidents on record (G6 evidence):** recon agent
hand-patched production `stage2.py` bypassing all gates
(confusion-codebook `oneoff-2026-07-04` — remedy was recon sandboxing, task
1935); stash-pop conflict landed markers on main (incident 13674d3c68);
`.task/` leak class (tasks 291/310/313); `cd`-fail amend rewrote main
(feedback memory); sibling worktrees/`.task-meta` of *other* tasks fully
writable today.

**2505 synergy:** wrap is computed per-invocation, and a steward
`granted_files` grant re-invokes the implementer on resume — so grants
naturally re-materialize into the sandbox ruleset with no extra machinery
(and under worktree granularity they need no ruleset change at all).

## 5. Roles NOT covered by the flip (scope decision)

`sandboxed=True` today = implementer + debugger only. Also writing code
unsandboxed: **SIMPLE_TASK** (Edit/Write/Bash — an implementer-equivalent;
recommend including), **MERGER** (resolves rebase conflicts in the task
worktree; heavier `.git` interaction), **STEWARD** (full authority by design),
**architect** (Edit/Write denied but has Bash). Escalation names
implementer/debugger; anything more is scope creep to discuss.

## 6. Proposed PRD shape (for discussion — not started)

**G1 (consumer):** `TaskWorkflow._invoke`'s sandboxed-role dispatch path, on
behalf of fleet operators — enabling on DF + reify removes the
out-of-worktree-write RCA class. Direct code consumers: `wrap_command`
call-sites (the write-set), `Harness.__init__` (the flip).

**G2 (leaf signal):** with `sandbox: {enabled: true, backend: landlock}` on
DF after restart: (a) a normal task completes end-to-end under the sandbox —
implementer commits per-step, plan-tools writes land, `uv run pytest` passes,
merge lands; (b) a containment probe (implementer instructed to write
`<main>/CANARY` / a sibling worktree file) observes deterministic EROFS and
files `scope_violation`; (c) a per-invocation `sandbox_applied`
(backend, writable-set digest) event in the event store — operator-visible.

**G3 (substrate):** verified — see §2 (all landed; host ABI 6 live; 2505
machinery landed; W11 `.task-meta` layout stable; warm lanes are plain
recycled dirs, wrap keys off assigned lane path per invocation — no
interaction).

**G6 (premise):** valid — zero OS enforcement in production today; prompt
promises an enforcement that doesn't exist; incident record above; and the
"can't just flip it" premise is verified (three concrete breakers, §3).

**Phases (sketch):**
- **α — write-set correctness** (landlock path first): worktree-granular
  writable root (whole `<wt>`, replacing per-module dirs); carve-outs via the
  existing dead `writable_extras` param: `<main>/.git/objects`,
  `<main>/.git/worktrees/<name>`, `<main>/.git/refs` (+ `packed-refs`) — or a
  narrower refs subset, see Q2; `<worktree_base>/.task-meta/<name>`;
  `~/.cache/uv`. Keep `/tmp`, `/dev`. Decide `~/.claude` narrowing (Q5).
- **β — posture + observability**: fail-closed for sandboxed roles when
  `enabled: true` (mirror recon guard's `RemediationSandboxUnavailable`),
  `sandbox_applied`/`sandbox_degraded` events, WARN→escalation on
  unavailability.
- **γ — enablement + soak**: DF canary (`enabled: true, backend: landlock` +
  restart), containment probe task, watch for EROFS-turn-burn in transcripts;
  then reify, then rest of fleet; flip the dashboard config comment.
- **δ — follow-ups (explicitly out or separate)**: SIMPLE_TASK/MERGER
  coverage, `~/.claude` narrowing if deferred, module-granular tightening,
  network scoping (Landlock ABI 4+ could, today neither backend does),
  touched⊆declared merge gate as a detective complement.

## 7. Open questions for Leo

1. **Granularity** — whole-worktree (recommended: matches the gate's stated
   goal, survives lockfiles/.venvs, inert to 2505 grants) vs per-module
   (breaks builds today; tightening can come later)?
2. **Shared `.git` write surface** — accept `objects` + `refs` writable
   (residual: an agent could still mangle other branches' refs — detectable,
   far smaller than whole-FS) vs narrower `refs/heads/task/` only (breaks
   nothing obvious but needs verification of ref-lockfile paths + pack-refs
   edge) vs routing commits through the orchestrator (rejects the per-step
   commit contract — large change, not recommended)?
3. **Scope** — implementer+debugger only (per the gate) or + SIMPLE_TASK
   (recommend: it is an implementer with a smaller pipeline)? MERGER later?
4. **Fail posture** — fail-closed when enabled but backend unavailable
   (recommend yes; landlock availability on this fleet is stable and the whole
   point is determinism)?
5. **`~/.claude`** — narrow now to `fleet/` + hook-state subdirs (session
   state already lives in-worktree) or keep fully writable for phase 1 and
   narrow as follow-up? (Poisoning surface either way until narrowed.)
6. **Rollout/soak criteria** — DF-first canary length, what counts as green
   (N tasks e2e + probe denial + no sandbox-attributable blocks), reify
   timing given its longer verifies.

## 8. Risks / tradeoffs

- **Agent behavior under EROFS**: prompt already directs "permission error →
  escalate `scope_violation`", but soak may show turn-burn on retries.
- **`~/.cache/uv` writable** → cross-project cache-poisoning channel (accept +
  note; alternative per-task `UV_CACHE_DIR` kills hardlink dedup/disk).
- **`/tmp` lateral channel** remains (jobserver FIFO, verify slots are shared
  by design).
- **x86_64-only landlock probe** — fine for this fleet; document.
- **bwrap bit-rot** — effectively deprecated on fleet hosts; `auto` prefers
  landlock; consider documenting bwrap as legacy rather than maintaining
  parity.
- **Restart-tier rollout** — each project needs config edit + drain-aware
  restart (`restart-all-orchestrators.sh --drain` chokepoint exists).

## 9. Pointers

- esc-2508-1 (port 8102) — leave pending until PRD ratified/queued.
- Task 2505 record (scope-grant machinery) — landed, `found_on_main`.
- Key files: `agents/sandbox_dispatch.py`, `agents/sandbox.py`,
  `agents/landlock.py`, `agents/landlock_exec.py`, `workflow.py:9898-9902`,
  `invoke.py:198-314`, `config.py:966-978`, `defaults.yaml:453-455`,
  `roles.py:458-466/480/523-531/544`, `shared/cli_invoke.py:1440-1964`,
  fused-memory `reconciliation/sandbox_guard.py` (fail-closed model).
