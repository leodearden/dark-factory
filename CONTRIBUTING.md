# Contributing to Dark Factory

This is guidance for changing dark-factory **itself** — the orchestrator,
fused-memory, escalation server, dashboard, shared libs, skills, hooks. If
you're instead running dark-factory *against* another project, see that
project's own docs (or `dark-factory-orchestrator.yaml` at its root) — this
file is about the factory, not its targets.

License: **AGPL-3.0** (`LICENSE`). By contributing you agree your changes
are distributed under it. No CLA or sign-off is required.

Related docs: `README.md` (what this is, quickstart), `OPERATIONS.md`
(runbook — config reload, fleet redeploy, model routing), `ARCHITECTURE.md`
(process topology and seams), `docs/task-authoring.md` (task metadata
vocabulary, deterministic/milestone task kinds), `CLAUDE.md` (the
agent-facing operating manual — session lifecycle, memory usage,
in-checkout working rules).

---

## 1. Ways to contribute

There are two legitimate paths in. Pick whichever fits the change; don't
mix them for the same piece of work.

### (a) Let the factory do it (preferred — dogfooding)

Dark Factory builds itself the same way it builds any target project: file
the work as a PRD, decompose it into tasks, let the orchestrator implement.

1. `/prd` — author (conversational drafting) or decompose (gates → filed
   tasks) mode. See §6 below for the gates that apply.
2. The orchestrator picks up pending tasks, runs PLAN → EXECUTE → VERIFY →
   [DEBUG] → REVIEW → MERGE per task, each in its own worktree on branch
   `task/<id>`.
3. `/escalation-watcher` triages anything the per-task steward and the L1
   auto-watcher couldn't resolve.
4. `/review` audits landed work end-to-end and files follow-up tasks.

This is the native path and the one most changes to this repo actually take
— including this file. Prefer it for anything a PRD can describe: new
features, refactors with a clear shape, bug fixes with a reproducible
signal.

### (b) Direct human changes

For small, mechanical, or urgent changes (a typo, a one-line config fix, an
emergency hotfix) it's fine to edit directly:

1. Create a **worktree outside `.worktrees/`** — that directory is
   reaped/recovered by the orchestrator's crash-recovery path, so a
   human-owned worktree parked there can be deleted out from under you. A
   sibling directory works well:
   ```bash
   git worktree add ../dark-factory.<short-name> -b task/<short-slug>
   ```
2. Branch from `main`. The merge queue accepts **any** branch name — the
   worker resolves the prefixed form (`task/<what-you-submitted>`) first
   and falls back to the literal name — but prefer **`task/<short-slug>`
   with a non-numeric slug** (e.g. `task/docs-user-docs`), the same
   convention `/do` and `/warm` produce: a couple of ancillary paths (the
   submit-time already-merged fast path, and `merge_status`'s
   git-authority recovery tier after an orchestrator restart) derive the
   ref by blindly prepending `task/`, so only prefixed branches get their
   full benefit. Never use a bare number as the slug: numeric `task/<id>`
   branches are orchestrator task ids, and reusing a real task's id can
   corrupt that task's merge bookkeeping.
3. Make your change, run the quality gates (§4), then merge via
   `/merge-queue` (§5) — not a direct `git merge --no-ff` into `main`
   whenever the orchestrator might be running.

---

## 2. Repo layout & conventions

This is a `uv` workspace (`pyproject.toml`); the current members are:

```
cockpit, dashboard, escalation, fused-memory, orchestrator, sampler, shared
```

**`<pkg>/src/<pkg>/` double-nesting.** Every package follows this
convention — e.g. `orchestrator/src/orchestrator/`,
`fused-memory/src/fused_memory/` (directory hyphenated, package name
underscored), `escalation/src/escalation/`, `shared/src/shared/`. Don't
flatten a package to `<pkg>/*.py` or nest it differently; the root
`conftest.py` adds each `<pkg>/src` to `sys.path` and pre-imports the
package by this exact shape so pytest's importlib collection doesn't
shadow it with a namespace package.

Other top-level dirs:

- **`skills/`** — in-repo skill source (`/prd`, `/orchestrate`, `/review`,
  `/merge-queue`, etc.), distinct from `~/.claude/skills` on any given
  machine, which symlinks to (or copies) these.
- **`plans/`** — design docs and PRDs for in-flight and past work; the
  working/scratch area (200+ files, not all landed or current).
- **`docs/prds/`** — the **committed** copy of a PRD once decomposed and
  queued, alongside its capability-manifest artifacts (`<prd-stem>.md`,
  `<prd-stem>.capability-manifest.md`, and the machine-readable
  `<prd-stem>.capability-manifest.yaml` sidecar — schema in
  `shared/src/shared/capability_manifest.py`). This is the durable record;
  `plans/` is not.
- **`docs/legibility/`** — `design-invariants.md` (INV-1..INV-5, gates
  `/prd` decompose and `/review` phase 2 — see §6) plus its calibration
  fixtures and the confusion-codebook incident taxonomy.
- **`dashboard/`** — web UI for task/escalation state.
- **`scripts/`** — operator and CI helper scripts (systemd unit templates,
  watchdog, host setup).
- **`hooks/`** — git hooks (`pre-commit`, `pre-merge-commit`, see §4-§5),
  install via `hooks/setup.sh`.

---

## 3. Environment

- **Python 3.13** (`.python-version`; `pyproject.toml` accepts `>=3.11,<4`
  but the pinned interpreter is 3.13).
- **`uv`** manages the workspace. Sync one package:
  ```bash
  cd orchestrator && uv sync
  ```
  Sync everything (needed before a full-repo test run):
  ```bash
  uv sync --all-packages
  ```
- **Tests** run per-package with `pytest`, e.g.:
  ```bash
  cd orchestrator && uv run pytest tests/ --timeout=300
  ```
  `dark-factory-orchestrator.yaml`'s `test_command` fans this out across
  every subproject in its own venv (`shared`, `escalation`, `orchestrator`,
  `fused-memory`, `dashboard`, `sampler`, optionally `cockpit`) — a bare
  repo-root `pytest` instead collects everything into one process against
  only the root `pyproject.toml`, which is slower and less isolated. Mirror
  the fan-out when running the full suite yourself.
<!-- lint-command-mirror:begin
     Mirrors the `ruff check` leg of `lint_command` in
     dark-factory-orchestrator.yaml. Pinned by
     tests/scripts/test_contributing_lint_command_drift.py — widen the yaml
     head and this line goes red until it is updated to match. -->
- **Lint**: `uv run ruff check shared escalation fused-memory orchestrator dashboard sampler cockpit conftest.py df_pytest_isolation.py skills`
<!-- lint-command-mirror:end -->
- **Type-check** (pyright, run from each configured package directory so it
  picks up that package's `[tool.pyright]` block):
  ```bash
  cd fused-memory && uv run pyright   # also: orchestrator, dashboard
  ```
  `dark-factory-orchestrator.yaml`'s `type_check_command` runs the same
  three packages via `npx pyright` (needs Node 22+) — either invocation
  works.

Treat `dark-factory-orchestrator.yaml`'s `test_command` / `lint_command` /
`type_check_command` as the source of truth if these drift.

---

## 4. Quality gates before submitting

Whether you're an orchestrator-dispatched agent or a human on a direct
branch, before any merge submission:

1. `pytest` — for every package you touched (and its dependents; `shared`
   and `escalation` are imported by most of the others, so a change there
   should trigger a broader run).
2. `uv run ruff check <touched packages>`.
3. `uv run pyright` in each touched, pyright-configured package
   (`fused-memory`, `orchestrator`, `dashboard`).

Do this **before** `merge_request`/`/merge-queue`, not after — a red
post-merge verify blocks or reverts the merge, which is more expensive than
catching it locally.

**`hooks/pre-commit`** additionally runs on every commit to `main`
(installed via `hooks/setup.sh`, which points `core.hooksPath` at `hooks/`):
it strips any staged `.task/` files (see §8), then on `main` runs `ruff
check`, the asyncmock/bare-MagicMock style checks on staged test files, and
**pyright up to 3×** (once per touched package under `PYRIGHT_PACKAGES`, or
across all three if the change touches a shared dependency like `shared` or
`escalation`). This can comfortably exceed two minutes — give commit
commands a timeout of at least `300000`ms (or run detached via `setsid` and
poll) rather than letting a default 2-minute timeout kill it mid-hook.
Never `--no-verify` this hook to skip pyright/ruff on a code change; the
two narrow documented exceptions are the *pre-merge-commit* emergency
bypass (§5) and a **docs-only** commit landing under index-lock contention
in the machine-operated main checkout (see `OPERATIONS.md` §"Working in
the main checkout").

---

## 5. Git workflow

- Branch from `main`. The merge queue merges **any** branch name (the
  worker tries `task/<submitted>` first, then the literal name), but
  prefer a `task/<short-slug>` branch with a **non-numeric** slug
  (`task/docs-user-docs`, `task/fix-merge-liveness`, …) so the
  already-merged fast path and post-restart status recovery — which
  derive the ref by prepending `task/` — work for your branch too.
  Numeric `task/<id>` names are orchestrator task ids — treat those as
  reserved.
- **Never `git stash` in the main checkout** (`/home/leo/src/dark-factory`
  or wherever `project_root` points). The merge worker's advance path
  consumes the stash stack as part of its own bookkeeping — a stash you
  push can be popped out from under you by an unrelated concurrent
  process. Park WIP as commits on a branch instead.
- If you must commit directly to `main` under contention (rare — prefer
  §1(b)'s worktree + merge-queue flow), use
  `git commit --only <path> [<path> ...]`, never a bare `git commit`, so
  you don't sweep up unrelated staged/dirty state some concurrent process
  left behind.
- **Merge via `/merge-queue`**, not raw `git merge --no-ff`, whenever the
  orchestrator might be running. `hooks/pre-merge-commit` actively blocks a
  direct merge commit on `main` outside the merge worker's own
  `_merge-*` worktrees — the queue's `merge_request` → `merge_status`
  submit/poll protocol (an explicit bounded `wait_secs`, never omitted) is
  the only way in without racing the orchestrator's own merge worker.
  Pre-rebase your branch onto `main` first to reduce the odds of a
  `conflict` result.
- If the escalation MCP isn't reachable at all (orchestrator not running),
  `/merge-queue` falls back to a direct merge for you — you don't need to
  hand-roll that fallback.

---

## 6. Design invariants & PRD gates

Every PRD authored or decomposed through `/prd` runs a fixed gate sequence
(`skills/prd/references/gates.md`):

- **G1** consumer named — no producer-orphans (a mechanism nothing wires in).
- **G2** user-observable leaf signal — every leaf task names a real,
  user-visible completion signal, not just a passing unit test.
- **G3** assumed-substrate verified — every capability the PRD assumes
  (endpoint, schema column, flag, function) is confirmed to exist or filed
  as an explicit prerequisite task.
- **G4** cross-PRD seam ownership — contested integration points get a
  named owner, recorded in a `## Cross-PRD relationship` table.
- **G5** design-first for high-stakes/complex work (contract section +
  boundary-test sketch) vs. a bare vertical slice.
- **G6** premise validity — numeric bounds, exactness claims, and rejection
  assertions must be substantiated, not guessed.
- **G7** design invariants — re-checked against
  `docs/legibility/design-invariants.md` (**INV-1..INV-5**:
  `contracts-machine-checked`, `structured-facts-at-failure`,
  `corroborate-before-acting`, `storm-escape-required`,
  `no-lockstep-duplication`). An unresolved, unwaived hit blocks the batch;
  a deliberate exception is a `G7 waiver: <slug> — <rationale>` line in the
  PRD plus `metadata.g7_waivers` on the filed task.
- **Capability manifest** — mechanizes G3/G6 per leaf task, committed
  beside the PRD in `docs/prds/` as a `.md` + `.yaml` sidecar pair.
- **META** — "would this PRD, decomposed and queued without further
  oversight, produce a complete, coherent, good design?"

`design-invariants.md` also gates `/review` phase 2's cross-module audit —
it's the single normative copy of the five invariants; don't restate them
elsewhere. If you're hand-writing a task (not going through `/prd`) for a
nontrivial design change, walk it against the same checklist yourself
before filing.

---

## 7. Commit messages

Match the observed convention (`git log --oneline -30` is the fastest way
to recalibrate — style drifts over time). Common shapes in this repo:

```
feat(<scope>): <what> (task <id> step-<n>)
test(<scope>): RED — <what this pins> (task <id> step-<n>)
feat(<scope>): GREEN — <what turned it green> (task <id> step-<n>)
docs(<scope>): <what>
fix(<scope>): <what>
amend: <small follow-up correction, no scope needed>
chore: <housekeeping>
```

- `<scope>` is usually the package or subsystem (`orchestrator`, `recon`,
  `legibility`, `prd`).
- RED/GREEN pairs come from the orchestrator's TDD steps — a failing test
  commit followed by the commit that makes it pass. Reuse this shape for
  hand-written TDD work too; it's what reviewers expect.
- Reference the task id where one exists — `(task 2904 step-6)` — so a
  commit is traceable back to its PRD/task record. Merge commits are the
  plain `Merge task/<id> into main` produced by the merge queue; don't
  hand-write those.

---

## 8. What NOT to do

- **Don't hand-edit machine-operated state.** `data/`, `.worktrees/` (and
  `.worktrees-orphaned/`, `.eval-worktrees/`), `.task/`, `.task-meta/`,
  `.taskmaster/` are runtime scratch/state owned by the orchestrator, the
  merge worker, and reconciliation — not source. `.task/` in particular must
  **never** be committed on any branch (dedicated `pre-commit` guard, plus
  post-merge scrubbing in `git_ops.py`); don't work around it.
- **Don't casually edit `evals/runner.py`'s exit-code semantics**
  (`orchestrator/src/orchestrator/evals/runner.py`). Its exit-code contract
  is load-bearing for eval scoring elsewhere — exit 0 does not mean
  "success" in the naive sense there; read `docs/plan-scoring-and-judge.md`
  first.
- **Don't add a new top-level `metadata` key without the blessing
  process.** `parse_metadata` (`shared/src/shared/task_metadata.py`) logs a
  `task_metadata.schema_warning ... code=unknown_key` line for anything not
  on the `_BLESSED_METADATA_KEYS` allowlist. A one-off key belongs under the
  `x_`-prefixed namespace or a generic `annotations` field, not as a bespoke
  top-level key — see `docs/task-authoring.md` for the vocabulary and
  promotion process.
- **Don't `--no-verify` the pre-commit hook** to skip ruff/pyright — if it's
  genuinely too slow, raise the timeout instead (§4).
- **Don't use a bare task number as a branch slug**, or reuse a
  blocked/in-flight task's id for unrelated work — either can corrupt that
  task's merge bookkeeping (non-numeric `task/<slug>` branches are the
  preferred interactive form — §5).
