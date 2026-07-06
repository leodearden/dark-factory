# PRD — verify-plan: declarative decision layer for the merge/verify gate (stream W7)

**Status:** active · wave 1 · 2026-07-06 · bug-hotspot remediation program
(`plans/bug-hotspot-remediation-program-2026-07-06.md`, verify cluster of
`plans/bug-hotspot-survey-2026-07-06.md`).
**Approach:** B + H (high stakes — merge-gate correctness; cross-module blast
radius 6; 7 mechanisms; cross-PRD consumer W9). Contract + boundary-test
sketch below.
**Owns (G4 authoritative):** `VerifyCmd`, `VerifyPlan`, `FailureCategory`,
tool-dispatched `classify_failure`, `BlockRecord`, and the merge_queue
block-path dry-run wiring.

## Goal

`verify.py` (3.6k lines) is the god-module of the merge gate: it derives scope,
executes checks, classifies failures, persists logs, and runs two copy-pasted
main-tip probes — all as interleaved procedural code over raw shell strings and
parallel scalars, with policy scattered across 5+ hand-synced string registries.
Nine distinct "wrong shell command", five "same file-classification bug fixed in
both functions", three "cargo classifier re-grounding", and one documented
"stale `timed_out`" regression have all landed here. This PRD rebuilds verify's
**decision layer** around declarative, serializable, unit-testable structures so
those regression classes become impossible by construction rather than
patched-in-two-places forever. It also closes the standing AFK pain point where
**every** merge-verify RED — including one-line lint/type fixes — falls to a
human because the merge-queue block path never produces a dry-run proposal.

**User/operator-observable outcome when this lands:**
- A merge-verify RED on a trivially-fixable scoped lint/type error produces a
  `metadata.dry_run_proposals[]` entry (today: none) that the B3 gate /
  escalation-watcher can auto-unblock — the biggest always-human escalation
  class becomes B3-gateable.
- `derive_verify_plan()` emits a serializable plan logged per merge attempt
  ("why did verify run/skip X") and unit-tested against the exact historical
  incident diffs — the "same bug fixed twice" class closes.
- An unparseable config verify command is classified `OPAQUE` and **never
  scoped** (today it is scoped anyway and produces an un-runnable argv).
- Adding a new failure category or block class is one table/enum row; a missing
  policy row is a module-import-time error, not silent default behaviour.

## Background

- Survey evidence: `plans/bug-hotspot-survey-2026-07-06-full-findings.json`
  cluster `verify` (7 confirmed findings). Bug-history themes cited per task.
- Operator memory: `feedback_b3_gate_aborts_on_merge_verify_no_proposal`
  ("trivial merge-verify type/lint fixes ALWAYS fall to human /unblock");
  the warm-lane broad-`git worktree prune` registration-wipe incident
  (2026-07-04, df 2097-2100) that the comment-only "no broad prune" invariant
  failed to prevent.
- The `has_conftest`→full-suite convention currently lives as CLAUDE.md operator
  lore **because the code cannot express it**; this PRD makes it an assertable
  property of the plan object.

## Sketch of approach

Seven mechanisms, decomposed into a linear `verify.py` spine (file-lock
serialization — `verify.py` is one lock under Contract-1), a parallel
block-record spine, and one B+H integration gate:

1. **`VerifyCmd`** (`verify_cmd.py`, new) — `@dataclass VerifyCmd(tool: ToolKind
   [PYTEST|RUFF|PYRIGHT|CARGO_TEST|CARGO_CLIPPY|NPX|OPAQUE], uv_project, cwd_rel,
   base_flags, targets, env, wrappers)`. Parse each config command **once** at
   config load via `shlex`; unparseable → `OPAQUE` (never scoped). Scoping =
   replace `targets`; reprojection = set `uv_project`; cd-strip = clear
   `cwd_rel`; cargo scoping = `targets=[-p crate…]` with `--exclude` dropped
   structurally; cpu-governance = a `wrappers` entry. Render to a shell string
   only inside `_run_cmd`. All six string-rewrite helpers
   (`_scope_command`/`_strip_directory_flag`/`_strip_leading_cd`/
   `_reproject_bare_uv_run`/`_scope_cargo_workspace`/`_force_serial_pytest`) and
   `_maybe_govern_merge_cmd`'s bash-wrap die. `render()` asserts invariants that
   are comment-lore today.
2. **`derive_verify_plan()`** (`verify_plan.py`, new) — pure function →
   `VerifyPlan = list[PlannedRun(module_prefix, VerifyCmd, scope_kind:
   FULL_SUITE|FILE_SCOPED|SKIPPED|TRIVIAL, reason: str)]` + plan flags
   (`needs_pipeline_guard_check`). File classification happens **exactly once**
   via a `FileKind` enum (CONFTEST, COLLECTABLE_TEST, TEST_DATA, STRUCTURAL,
   SOURCE, INERT), unifying `scope_module_config` + `_build_fallback_config` (the
   same conftest bug was fixed in both by task 1077 commits d7504d432d +
   cb7277926d; the same data-module bug by task 1852 commits 4fbed6c4fb +
   7c9b316260). The STRUCTURAL→unscoped-pyright widening (today only in
   `scope_module_config`, a latent gap in the fallback path) applies uniformly.
   `run_scoped_verification` shrinks to derive→execute→aggregate; the plan is
   serializable, logged, and attached to `VerifyResult` for post-hoc triage.
3. **Tool-dispatched `classify_failure(tool: ToolKind, rc, output, timed_out)`**
   with per-tool tables; prefer **structured tool output** where available
   (`pyright --outputjson`, `ruff --output-format json`,
   `cargo --message-format json` — all three verified present on the pinned
   versions: pyright 1.1.408, ruff 0.15.9, cargo present). Generic regex ladder
   survives only for `OPAQUE`. Ends the cargo re-grounding (tasks 1103/1109/1116)
   and flake-tightening arms races — a cargo token can no longer swallow a
   pytest/rustc line by construction; env_transient patterns run only under
   `PYTEST`.
4. **`FailureCategory(StrEnum)` + one `CATEGORY_POLICY` table**
   (`verify_categories.py`, new) — `CategoryPolicy(severity_rank, archive,
   preexisting_probe, is_infra_transient, retry_kind)`. Derive `_CATEGORY_
   PRIORITY`, `_ARCHIVE_DENY_LIST`, `PREEXISTING_BREAK_SKIP_CATEGORIES`, and the
   sweep infra-sentinel set from the table; delete the `endswith('_error')`
   heuristic; import-time `assert` enforces every member has a policy row.
   `StrEnum` keeps the JSON codec byte-identical (verify_runner Invariant 1); all
   in-process branching in verify.py / verify_runner.py / merge_queue.py /
   workflow.py goes through the enum.
5. **Typed `BlockRecord`** (`unblock_types.py`, new — does not exist today) —
   `block_class: BlockClass` enum {AGENT_FAILURE, REVIEW_ISSUES, MERGE_VERIFY_RED,
   POST_MERGE_RED_MAIN, …one member per existing block-reason prefix},
   `risk_label`, `head_sha`, `main_sha`, `files_referenced`, `investigated_at`,
   with `to_dict`/`from_dict`. Constructed by **workflow AND merge_queue**;
   `b3_gate` branches on `block_class` (the re-declared prefix constant + the
   `'status'`-key sniff die). **Dual-read bridge:** `b3_gate` reads `block_class`
   when present, falls back to the existing prose-prefix/status-sniff for legacy
   proposals — the coherence test stays until all producers emit `block_class`.
   **POST_MERGE_RED_MAIN keeps its task-1680 hard-abort EXACTLY** (unblock-low-
   risk depends on it): `check_proposal` returns ABORT for that `block_class`
   before any risk/git check.
6. **Close the coverage gap** — merge_queue's block path constructs
   `BlockRecord(block_class=MERGE_VERIFY_RED)` and spawns the **same** dry-run
   investigation workflow already runs (`run_dry_run_unblock`; the block site
   already holds `merge_wt`, the failing `VerifyResult`, and the scoped diff via
   `_derive_task_files_from_git` — its input is strictly richer than the
   agent-block case). Trivial merge-verify fixes stop falling to a human.
7. **`CheckRun` dataclass** — `CheckRun(label, cmd, rc, output, timed_out,
   started_at, duration_secs)` + `CheckRun.skipped()`; `VerifyAttempt(checks)`
   exposing `passed` / `any_timed_out` / `pure_timeout_failure` computed in
   **one** place. `run_verification`'s 15 parallel locals collapse to one
   `VerifyAttempt`; both copies of the 6-clause timeout-consistency formula
   (retry loop + env-recovery branch) die — the documented stale-`timed_out`
   drift (verify.py:2735-2744) becomes impossible.
8. **`git_ops.ephemeral_worktree(kind: WorktreeKind, sha)`** async context
   manager — owns naming (prefix registry per kind), retry-on-lock-contention,
   and **guaranteed scoped cleanup (`git worktree remove --force` + rmtree;
   NEVER `git worktree prune`)**. Both verify probes
   (`verify_failure_is_preexisting_on_main`, `run_main_tip_sweep`) shrink to
   consume it. Registers its kind's prefix into M1's `PROTECTED_PREFIXES` and
   routes cleanup through M1's `_prune_registrations` chokepoint (see
   Pre-conditions / Open Q1 for the M1-ordering default).

## Resolved design decisions (do not relitigate)

1. **Linear `verify.py` spine.** `metadata.files` is file-level (Contract-1);
   every task touching `verify.py` serialises on one lock. The six verify.py
   tasks (α FailureCategory, β VerifyCmd, γ derive_verify_plan, δ classifier,
   ε CheckRun, θ ephemeral_worktree) form a linear dependency chain to prevent
   rebase thrash — the proven merge-queue-refactor pattern (tasks 1985-2002).
2. **`StrEnum`, not bare `Enum`.** Python is 3.13 (`>=3.11`); `StrEnum` members
   *are* `str`, so `json.dumps` is byte-identical to today — verify_runner's
   canonical-JSON Invariant 1 and all on-the-wire category strings are preserved.
3. **`b3_gate` dual-read bridge.** New producers emit `block_class`; legacy
   in-flight proposals (written before this lands) keep routing via the prose
   prefix + `'status'` sniff. This makes the migration zero-downtime and keeps
   the coherence test meaningful during transition.
4. **POST_MERGE_RED_MAIN hard-abort is a preserved invariant, not a redesign.**
   Task 1680's defense-in-depth and unblock-low-risk's hard refusal of that class
   are load-bearing; `block_class == POST_MERGE_RED_MAIN` reproduces the exact
   hard-abort-before-risk-check semantics.
5. **`ephemeral_worktree()` never prunes.** DD5 becomes code: scoped remove only.
   The mechanical "no source file outside git_ops invokes `git worktree prune`"
   grep-guard is **M1-owned** (M1's chokepoint grep-guard test) — θ consumes it,
   does not duplicate it.
6. **Golden classifier corpus derives expected categories from historical fix
   commits** (1103/1109/1116 for cargo; 1077/1852 for file-classification), not
   invented strings (G6). Same for the plan-derivation golden tests — they run
   against the exact historical incident diffs.
7. **BlockRecord is backward-compatible metadata, not a W3 schema dependency.**
   It serialises additively into the existing `dry_run_proposals` dict shape;
   W3 registering a `DryRunProposal`/`BlockRecord` sub-model is an *optional
   future* consolidation, not a cross-batch blocker (Open Q2).

## Pre-conditions for activating

- **Substrate (all verified 2026-07-06):** `pyright --outputjson`,
  `ruff --output-format json`, `cargo --message-format json` present on pinned
  versions; `shlex` (stdlib); `StrEnum` (py3.13); `run_dry_run_unblock` present
  with a keyword API; merge_queue block site holds `merge_wt` + `VerifyResult` +
  `_derive_task_files_from_git`; `unblock_types.py` does not exist (clean new
  module). No unverified substrate remains → G3 satisfied.
- **Soft cross-batch prereq — M1 (gitops-chokepoints).** θ (ephemeral_worktree)
  registers its prefix into M1's `PROTECTED_PREFIXES` and routes cleanup through
  `_prune_registrations`. M1 is wave-`now` and not yet filed at authoring time.
  **Default (Open Q1):** the extraction is correct and safe *standalone*
  (scoped-cleanup-only), so it is not hard-blocked; at decompose time, if M1's
  chokepoint task is filed, wire a bare-integer dep on it; if not, file θ
  self-contained with the prefix-registration described in-task (fail-loud if
  the M1 registry is absent at dispatch) and record the coupling here.

## Cross-PRD relationship (G4 — authoritative for this seam)

| Other stream | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| **W9** (workflow-state-machine, wave 2) | consumes | `BlockRecord` in `_mark_blocked`; `classify_failure`→`BlockDisposition` table; `FailureCategory` | **W7** owns; W9 consumes | queued — W9 wires dep to W7's ζ/α task ids at its decompose |
| **M1** (gitops-chokepoints, wave now) | consumes | `PROTECTED_PREFIXES` + `_prune_registrations` chokepoint (θ registers/routes) | **M1** owns chokepoint+registry; W7's θ consumes | soft — wire at decompose if M1 filed (Open Q1) |
| **W1** (merge-queue-reliability, wave 1) | co-touches | merge_queue block path (η adds `BlockRecord` construct + dry-run spawn at the **existing** block site) | **W7** owns the block-path wiring; **W1 must not move that path without a dep** | coordinated — constraint on W1 recorded in program seam map |
| **W3** (task-metadata-schema, wave 1) | co-touches | `dry_run_proposals` metadata surface | **W7** keeps backward-compatible; W3 consolidation optional | decoupled (Open Q2) |
| **unblock-low-risk** skill | consumes | `metadata.dry_run_proposals[-1]` via b3_gate | **W7** preserves shape + POST_MERGE_RED_MAIN hard-abort | wired (backward-compatible) |

No reciprocal-ownership ambiguity: every seam above has exactly one owner.

## Contract section (B + H)

**`VerifyCmd` (verify_cmd.py)**
- `ToolKind = Enum(PYTEST, RUFF, PYRIGHT, CARGO_TEST, CARGO_CLIPPY, NPX, OPAQUE)`.
- `parse_config_command(raw: str) -> VerifyCmd` — `shlex.split`; classify the
  head token → `ToolKind`; anything unrecognised/unsplittable → `OPAQUE` with the
  raw string retained. **Invariant P1:** `OPAQUE` commands are never scoped,
  reprojected, cd-stripped, or cargo-scoped — every mutator is a no-op on OPAQUE.
- `render(cmd: VerifyCmd) -> str` — the *only* place a shell string is produced;
  the inverse of parse for non-OPAQUE. **Invariant P2:** `render(parse(x))` is
  argv-equivalent to `x` for a well-formed `x` (round-trip). **Invariant P3:**
  `cwd_rel is None` whenever `targets` are worktree-root-relative.
- Mutators are pure `VerifyCmd → VerifyCmd`: `scope_to(files)`,
  `reproject(project)`, `strip_cwd()`, `cargo_scope(crates)`, `serial_pytest()`,
  `govern_cpu()` — each a structured field edit, never a regex.

**`derive_verify_plan` (verify_plan.py)**
- `derive_verify_plan(existing_files, module_configs, config, worktree_reader)
  -> VerifyPlan`. Pure (no execution, no I/O beyond `worktree_reader`).
- `classify_file(path) -> FileKind` is called exactly once per file; predicates
  `_is_test_file`/`_is_collectable_test_file`/`_is_conftest` are derived from
  `FileKind`, never recombined ad hoc.
- **Invariant D1 (the CLAUDE.md convention as code):** any `CONFTEST` or
  `TEST_DATA` file in the diff ⇒ that module's `PlannedRun.scope_kind ==
  FULL_SUITE`. **Invariant D2:** a `STRUCTURAL` file ⇒ unscoped pyright, in both
  the module and fallback paths (closes the latent 1852-shaped gap). **Invariant
  D3:** `VerifyPlan` is JSON-serialisable and attaches to `VerifyResult`.

**`classify_failure`**
- `classify_failure(tool: ToolKind, rc: int, output: str, timed_out: bool)
  -> FailureCategory`. **Invariant C1:** a pattern in tool T's table is only ever
  matched against tool-T output. **Invariant C2:** where structured output is
  requested, the classifier parses it; the human log is unchanged.

**`FailureCategory` / `CATEGORY_POLICY`**
- **Invariant F1:** module-import asserts `set(CATEGORY_POLICY) ==
  set(FailureCategory)` (exhaustiveness). **Invariant F2:** `str(FailureCategory.X)
  == "<x>"` and `json.dumps` is byte-identical to the pre-change string.

**`BlockRecord` (unblock_types.py) + b3_gate**
- **Invariant B1:** `from_dict(to_dict(r)) == r`. **Invariant B2:**
  `check_proposal` on `block_class == POST_MERGE_RED_MAIN` returns ABORT *before*
  any `risk_label`/git check (task-1680 preservation). **Invariant B3:** a
  proposal with no `block_class` (legacy) routes identically to today's
  prose-prefix/status-sniff path. **Invariant B4:** a `MERGE_VERIFY_RED` proposal
  with `risk_label == 'low'` is *gateable* (not auto-aborted).

**`ephemeral_worktree` (git_ops.py)**
- **Invariant E1:** the context manager performs scoped `git worktree remove
  --force` + `rmtree` on exit and **never** invokes `git worktree prune`.
  **Invariant E2:** its `kind` prefix is present in `PROTECTED_PREFIXES` (M1) so
  reapers skip it.

## Boundary-test sketch (B + H — the integration-gate signal)

One integration test module (leaf task ι) faces **both** sides of each seam:

| # | Scenario | Preconditions | Postconditions (asserted) |
|---|---|---|---|
| 1 | VerifyCmd render round-trip (producer↔runner) | a representative config command per ToolKind | `render(parse(x))` executes an argv-equivalent process; a real scoped pytest cmd runs the same tests as today |
| 2 | OPAQUE never scoped | an unparseable config command (the historical broken layout) | classified OPAQUE; `scope_to` is a no-op; the emitted argv is the raw command unchanged |
| 3 | Plan golden — root conftest | diff = root `conftest.py` | plan = FULL_SUITE, reason names conftest (D1) |
| 4 | Plan golden — lone data module | diff = task-1852 data-module diff | plan = SKIPPED-with-reason (not silent) — the bug that was fixed twice |
| 5 | Plan golden — structural file | diff = a Protocol/TypedDict file | unscoped pyright in BOTH module and fallback paths (D2) |
| 6 | Classifier tool-isolation | cargo output containing a pytest-like token | classified as cargo category; the pytest table never sees it (C1), expected category derived from commit 1103/1109/1116 |
| 7 | Category exhaustiveness | a synthetic FailureCategory member with no policy row | import-time assert fires (F1) |
| 8 | CheckRun timeout consistency | env-recovery run that hits the wall clock | `any_timed_out` computed once; category cannot flip to infra_timeout while `timed_out=False` (the 2735-2744 drift) |
| 9 | Merge-verify block → proposal (the coverage gap) | a merge-verify RED with a trivial scoped-lint diff | `metadata.dry_run_proposals[]` gains a `MERGE_VERIFY_RED` entry (today: none); b3_gate returns non-ABORT for a low-risk one (B4) |
| 10 | POST_MERGE_RED_MAIN preserved | a proposal with `block_class=POST_MERGE_RED_MAIN` | b3_gate ABORTs before risk/git check (B2) — task-1680 semantics |
| 11 | Legacy proposal bridge | a proposal dict with no `block_class` | routes identically to pre-change behaviour (B3) |
| 12 | ephemeral_worktree no-prune | drive both verify probes through the CM | assert `git worktree prune` is never invoked; prefix is in PROTECTED_PREFIXES (E1/E2) |

## Decomposition plan

Greek labels; task ids assigned at decompose. **verify.py spine is linear**
(α→β→γ→δ→ε→θ, file-lock); **block spine** ζ→η (after α on workflow/merge_queue);
ι is the B+H integration gate over the whole batch.

- **α — `FailureCategory` enum + one policy table** (`verify_categories.py`;
  rewire verify.py, verify_runner.py, merge_queue.py, workflow.py).
  *force_full_path* (deceptively mechanical, 4-module blast).
  **Signal (leaf-ish, roped to ι):** a synthetic category missing a policy row
  raises at import; all on-the-wire category strings byte-identical (golden).
  **Consumer:** classifier δ, verify_runner sentinels, merge_queue:721/924,
  workflow:4894/4939, W9. **Prereq:** none.
- **β — `VerifyCmd` structured command model** (`verify_cmd.py`; rewire verify.py
  scoping + `_run_cmd`; delete 6 string helpers + `_maybe_govern_merge_cmd`
  bash-wrap). **Signal:** render round-trip (test 1) + OPAQUE-never-scoped
  (test 2). **Consumer:** derive_verify_plan γ, classifier δ (tool identity),
  `_run_cmd`. **Prereq:** α (verify.py lock).
- **γ — `derive_verify_plan()` + FileKind** (`verify_plan.py`; shrink
  run_scoped_verification to derive→execute→aggregate). **Signal:** plan goldens
  (tests 3-5) — conftest/data-module/structural. **Consumer:**
  run_scoped_verification, VerifyResult.plan, ι. **Prereq:** β.
- **δ — tool-dispatched `classify_failure`** (per-tool tables + structured-output
  parse; generic ladder for OPAQUE only). **Signal:** classifier tool-isolation
  golden (test 6), categories derived from historical fix commits. **Consumer:**
  `_summarize_checks`/run_verification → VerifyResult.category, merge_gates
  sentinels. **Prereq:** α (FailureCategory), γ (verify.py lock; ToolKind carried
  by plan).
- **ε — `CheckRun` / `VerifyAttempt` dataclasses** (collapse run_verification's
  15 locals; single timeout-consistency computation). **Signal:** timeout
  consistency test (test 8) — the 2735-2744 drift is unreachable. **Consumer:**
  run_verification, `_persist_attempt_logs`/`_build_summary_payload` (via
  to_dict). *force_full_path* (looks mechanical, replaces a load-bearing
  invariant). **Prereq:** δ (verify.py lock; run_verification calls classifier).
- **ζ — typed `BlockRecord` + b3_gate branches on `block_class`**
  (`unblock_types.py`; rewire b3_gate.py, dry_run_unblock.py `_build_entry`,
  workflow `_spawn_dry_run_unblock`). **Signal:** POST_MERGE_RED_MAIN still
  hard-aborts (test 10); legacy proposal bridge (test 11); MERGE_VERIFY_RED
  gateable (part of test 9). **Consumer:** b3_gate, W9 `_mark_blocked`,
  unblock-low-risk, merge_queue η. **Prereq:** α (workflow.py lock;
  FailureCategory→BlockClass mapping).
- **η — merge_queue block path spawns the dry-run investigation** (construct
  `BlockRecord(MERGE_VERIFY_RED)` + call `run_dry_run_unblock` at the existing
  block site; thread scheduler/mcp/config handles). **Signal:** the coverage-gap
  test (test 9) — a merge-verify RED now yields a `dry_run_proposals[]` entry.
  **Consumer:** b3_gate/escalation-watcher/unblock-low-risk (they can now gate
  this class), the AFK operator. **Prereq:** ζ (BlockRecord + merge_queue lock
  after α via ζ). **Constraint:** W1 must not move the block path without a dep.
- **θ — `git_ops.ephemeral_worktree()` extraction** (both verify probes consume
  it; scoped-cleanup-only; register prefix into M1 PROTECTED_PREFIXES / route via
  `_prune_registrations`). **Signal:** no-prune + protected-prefix test
  (test 12). **Consumer:** the two verify probes; M1's reaper. **Prereq:** ε
  (verify.py lock) + soft M1 (Open Q1).
- **ι — B+H integration gate: two-way boundary tests** (ONE new test module;
  boundary-test sketch rows 1-12 above). **Signal (the leaf):** the boundary-test
  module passes — argv round-trip, plan goldens, classifier isolation, timeout
  consistency, block-path end-to-end (MERGE_VERIFY_RED gateable +
  POST_MERGE_RED_MAIN hard-abort + legacy bridge), ephemeral-worktree no-prune.
  **Consumer:** the merge-gate correctness guarantee / CI. **Prereq:** θ (verify
  spine tip) + η (block spine tip) — transitively the whole batch.

## Out of scope

- **W9's** consumption of `BlockRecord` in `_mark_blocked` and its
  `BlockDisposition` table (wave 2; W9 wires the dep to W7 ids).
- **merge_queue lifecycle internals** (W1) — η's edit is limited to constructing
  `BlockRecord` + spawning the investigation at the **existing** block site; it
  does not move or restructure the block path.
- Verify **execution transport** (verify_runner multi-host, RemoteRunner) — the
  runner re-enters `run_scoped_verification`; it inherits the new plan/classifier
  for free but its dispatch is untouched.
- W3's `shared/task_metadata.py` schema versioning of `dry_run_proposals`
  (optional future consolidation; Open Q2).
- The env-transient/venv-isolation *root* cause (a missing upstream install-lock
  guarantee) — this PRD keeps the compensating forensics but re-homes them under
  `ToolKind.PYTEST`; the durable venv fix is a separate concern.

## Open questions (tactical — surfaced, not blocking; AFK defaults recorded)

1. **M1 dependency ordering for θ.** M1 (gitops-chokepoints) is not yet filed.
   **Default:** at decompose, `search_tasks` for M1's `_prune_registrations` /
   PROTECTED_PREFIXES task; if filed, wire a bare-int dep on θ; if not, file θ
   self-contained (scoped-cleanup-only is correct standalone) with the
   prefix-registration described in-task and fail-loud if M1's registry is absent
   at dispatch. Decide at decompose / θ impl.
2. **W3 schema home for the proposal surface.** **Default:** `BlockRecord` lives
   in orchestrator `unblock_types.py` and serialises backward-compatibly; do not
   hard-couple to W3. If W3 later adds a `DryRunProposal` sub-model, migrate then.
   Decide during a future W3/W7 consolidation.
3. **Classifier module home.** Inline in verify.py vs a new `verify_classify.py`.
   **Suggested:** new `verify_classify.py` to keep verify.py shrinking; δ rewires
   `_summarize_checks` either way (still on the verify.py lock). Decide at δ impl.
4. **Threading scheduler/mcp/config into the merge-worker block path (η).** The
   capability (`run_dry_run_unblock`) exists and workflow already calls it; the
   merge worker has `config` but scheduler/mcp handles must be plumbed (or a thin
   adapter built). **Suggested:** plumb the handles the harness already owns.
   Decide at η impl.
5. **`BlockClass` member set.** **Default:** one member per block-reason prefix
   that exists on main today (so the dual-read bridge is total), plus the four
   named. Decide at ζ impl by enumerating current prefixes.

## Notes for the decompose session

- Metadata fields `user_observable_signal` / `consumer_ref` / substrate-confirmed
  flag are written for a future tracking-infra session; the orchestrator does not
  read them yet.
- Every task filed `planning_mode=True`; wire all deps while deferred; flip the
  whole batch in one `commit_planning`. `metadata.files` file-level only.
- Capability manifest committed beside this PRD as
  `plans/verify-plan-prd.capability-manifest.md`.
