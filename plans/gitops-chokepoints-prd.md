# PRD: git_ops chokepoints — guarded prune, unified acquire-fault teardown, protected prefix bands

**Status:** active — 2026-07-06. Stream M1 of the bug-hotspot remediation program
(`plans/bug-hotspot-remediation-program-2026-07-06.md`). Bare-B (tests are the boundary).

## Goal

Close the two already-lived CRITICAL incident classes in the git worktree layer with
cheap structural chokepoints, so the *next* fix cannot miss a route:

1. **Registration-wipe class** (Jul-3/4 incident, tasks 2097–2100): `git worktree prune`
   during a mount-down window wipes every mount-resident lane's `.git/worktrees` admin
   entry. Task 2099 added a `pool_storage_present()` refuse-gate — but only inside
   `prune_worktrees()`. Six raw argv sites bypass it. After this PRD, the argv literal
   `['git','worktree','prune']` exists in exactly ONE place in `orchestrator/src`, behind
   the guard, and a CI guard test enforces that count forever.
2. **Stale-checkout collision class** (task 2062): a fault after a branch checkout
   succeeded releases the lane FREE while `task/<id>` stays checked out in it; the
   requeue collides with `already used by worktree at _lane-K`. Task 2062 fixed three
   named paths; the top-level `except Exception` backstop (git_ops.py:3060-3065) still
   does a bare `pool.release` with no detach. After this PRD, every fault exit routes
   through ONE teardown primitive.
3. **Foreign-band deletion hazard** (latent; proven adjacent on Jul-5 when deleting a
   lane dir killed task 4791's plan.json): destructive cleanup sweeps each carry their
   own ad-hoc name filter; nothing structurally prevents a sweep from removing a
   worktree band it does not own. A PROTECTED_PREFIXES registry + refusal helper makes
   band ownership explicit and machine-checked.

## Background / evidence (code-verified 2026-07-06)

Raw prune argv sites in `orchestrator/src` (7 total, 1 guarded):

| Site | Function | Guarded? |
|---|---|---|
| git_ops.py:3824 | `create_worktree` leftover-branch cleanup | no |
| git_ops.py:4341 | `materialize_member_solo` pre-clean | no |
| git_ops.py:4426 | `delete_solo_branch` | no |
| git_ops.py:5365 | `prune_stale_merge_worktrees` | no |
| git_ops.py:5725 | `reap_interactive_worktrees` | no |
| git_ops.py:6807 | `prune_worktrees` (task-2099 guard at 6788-6806) | **yes** |
| harness.py:4781 | substrate-gate stale-worktree cleanup (tuple argv) | no |

The merge-lane sites (4341/4426/5365) run on merge-queue activity independent of
lane-pool health — during the exact mount-down window 2099 guards against, a
solo-branch cleanup can still wipe every mount-resident lane's admin entry.

Acquire-fault teardown idioms in `acquire_warm_lane` / `_reset_and_seed_recycled_lane`
(4 divergent):

- `release_warm_lane` (detach + branch-retention + pool release): the task-2062 fix
  sites (~2751, ~2803, and via `_reset_and_seed_recycled_lane` ~3320/~3348).
- `git worktree remove --force` + **bare** `pool.release`: create-once reattach
  seed-fault (~2870-2874) and create-once fresh seed-fault (~2912-2916); plus bare
  release on `git worktree add` failure (~2890-2893).
- Degenerate-ref delete (`warm_lane_ref_is_degenerate` → `_delete_branch_if_on_main`,
  task 2112): create-once fresh seed-fault ONLY (~2919-2941) — enforced solely by
  comment.
- Top-level `except Exception` (3060-3065): **bare `pool.release`, NO detach** — the
  backstop every other route falls into. E.g. `_reuse_warm_lane` → `commit()` raising
  `RuntimeError('Commit failed: …')` (git_ops.py:3887), or a shared-tail failure after
  `_reset_warm_lane`'s `checkout -f -B`, lands here with `task/<id>` still checked out.

Worktree name bands under `worktree_base` (all existing literals):
`_lane-*` (WarmLanePool, warm_lane_pool.py:42), `_spec-*` (spec pool), `_iact-*`
(config.iact_prefix, config.py:1161), `_merge-*` + persistent `_merge-verify`
(PERSISTENT_MERGE_WORKTREE_NAME, git_ops.py:160), `_offline-deep`
(PERSISTENT_OFFLINE_DEEP_WORKTREE_NAME, git_ops.py:167), `_solo-*`
(materialize_member_solo, git_ops.py:4289), `_substrate-gate-*` (harness.py:4760).

## Consumers (G1)

- **`_prune_registrations` chokepoint**: consumed by all six converted call sites (the
  warm-lane pool, merge-lane cleanup, interactive reaper, and harness substrate-gate
  paths themselves) — plus stream W11 (worktree-lane-lifecycle), whose LaneLifecycle
  tasks will declare deps on this batch per the program G4 table.
- **Prune grep-guard test**: consumer is CI / pre-merge verify (`pytest tests/`) — a
  second argv literal introduced anywhere in `orchestrator/src` turns main red with a
  file:line offender list.
- **`_abort_lane_acquisition`**: consumed by every fault exit in
  `acquire_warm_lane` / `_reset_and_seed_recycled_lane`; behavioural consumer is the
  requeue path (no more `already used by worktree` collisions after mid-acquire faults).
- **PROTECTED_PREFIXES registry**: consumed by the destructive cleanup paths
  (`prune_stale_merge_worktrees`, `reap_interactive_worktrees`, `create_worktree` /
  `create_interactive_worktree` self-heal rmtree, harness substrate-gate cleanup) and
  by W11's LaneLifecycle as the authoritative band map.

## Sketch of approach

### Mechanism 1 — `_prune_registrations(context: str)` (α, β, γ)

- New private coroutine on `GitOps` holding the ONLY `['git','worktree','prune']` argv
  in `orchestrator/src`. Body = the current `prune_worktrees()` semantics verbatim:
  refuse when `pool_in_use()` and not `pool_storage_present()` (benign-skip via
  `_pool_storage_bootstrap_ok()`, escalate via `_note_pool_storage_absent()`),
  best-effort, never-raise. `context` is threaded into every log line so operators can
  attribute which sweep asked.
- Public `prune_worktrees(context: str = 'prune_worktrees')` becomes a thin delegate
  (keeps the existing public API; harness callers unchanged).
- Convert the 5 raw git_ops sites to `await self._prune_registrations(context='…')`
  (contexts: `create_worktree-leftover`, `materialize_member_solo`,
  `delete_solo_branch`, `prune_stale_merge_worktrees`, `reap_interactive_worktrees`).
- Convert the harness substrate-gate tuple-argv subprocess (harness.py:4779-4781) to
  `await self.git_ops.prune_worktrees(context='substrate-gate-cleanup')`. The `git
  worktree remove --force <gate_path>` half of that loop stays a direct subprocess
  (remove is path-scoped; only prune is registration-global).
- **Guard test** (new `orchestrator/tests/test_prune_chokepoint_guard.py`): AST-scan
  every `*.py` under `orchestrator/src` (rglob) for a List/Tuple literal whose elements
  are the string constants `'git','worktree','prune'` in order; assert exactly one
  occurrence and that it is inside `_prune_registrations`. Mirrors the existing
  source-scan guard pattern (`orchestrator/tests/test_event_loop_antipattern_guard.py`
  — AST-based so docstrings/comments mentioning the phrase, e.g. verify.py's DD5 notes,
  never false-positive). Fails with a file:line offender list.

Happy-path behaviour is identical at all six sites: same argv, same cwd, same
best-effort rc logging. The only behavioural delta is the *refusal* during
pool-configured + storage-absent windows — which is precisely the incident fix.

### Mechanism 2 — `_abort_lane_acquisition(lane, bare_id, *, remove_worktree: bool)` (δ)

One never-raise teardown primitive for every acquire-path fault:

1. **Commit-or-retain WIP** — best-effort `commit()` snapshot before any ref movement
   (mirrors `detach_lane_checkout`'s commit-before-detach contract; hard rule: never
   discard uncommitted WIP).
2. **`git checkout --detach`** — frees the single-checkout lock so `task/<id>` never
   stays checked out in a FREE lane.
3. **Branch-retention rule** — `_delete_branch_if_on_main` (retain any branch carrying
   commits beyond main), preceded by the task-2112 degenerate-ref check
   (`warm_lane_ref_is_degenerate(bare_id)`) so a zero-commit residue ref from a failed
   `worktree add -b` is cleaned on EVERY fault route, not just the create-once one.
4. **`remove_worktree=True`** (create-once routes where the worktree was minted this
   call and seeding failed): `git worktree remove --force` before pool release.
   `remove_worktree=False` (reuse/reset/recycle routes + top-level backstop): keep the
   lane worktree in place.
5. **`warm_lane_pool.release(lane)`** — always last.

Route EVERY fault exit through it: the bare-release sites (~2870-2874, ~2890-2893,
~2912-2916 → `remove_worktree=True`), the existing `release_warm_lane` fault sites in
`acquire_warm_lane`/`_reset_and_seed_recycled_lane` (→ `remove_worktree=False`; the
primitive subsumes what `release_warm_lane` did there plus the degenerate-ref check),
and — load-bearing — the top-level `except Exception` (3060-3065,
`remove_worktree=False`). `release_warm_lane` itself (the normal terminal-release path)
is NOT changed — this primitive is fault-exit-only.

Fault-injection test: monkeypatch so an exception is raised after the reuse-path
checkout/mapping succeeded (e.g. `commit()` raises inside `_reuse_warm_lane`); assert
post-fault the lane HEAD is detached, the pool slot is FREE, and a follow-up
`acquire_warm_lane` for the same task does NOT fail with `already used by worktree`.

### Mechanism 3 — PROTECTED_PREFIXES registry (ε)

- Module-level `PROTECTED_PREFIXES: dict[str, str]` in git_ops.py mapping band prefix →
  owner tag for the static bands (`_lane-`, `_spec-`, `_merge-`, `_solo-`,
  `_substrate-gate-`) plus exact persistent names (`_merge-verify`, `_offline-deep`);
  an instance method `protected_prefixes()` merges the config-driven `iact_prefix`
  (bands are config-shaped, so the authoritative view is per-instance).
- Refusal helper `_refuse_foreign_band(path: Path, owned: frozenset[str], context: str)
  -> bool`: True (refuse + loud WARNING naming the owning band) when `path` is a direct
  child of `worktree_base` whose name matches a protected prefix outside the caller's
  declared `owned` set. Unknown/unprefixed names are NOT refused (fail-open for
  non-band paths — task worktrees, quarantine dirs — this is a band-ownership check,
  not a general ACL).
- Wire as defense-in-depth at the destructive sites that remove worktree_base children:
  `prune_stale_merge_worktrees` (owns `_merge-`), `reap_interactive_worktrees` (owns
  iact band), `create_worktree` / `create_interactive_worktree` self-heal rmtree (own
  their target path's band or non-band), harness substrate-gate cleanup (owns
  `_substrate-gate-`). Existing positive filters stay; the helper turns a filter bug
  into a loud refusal instead of a deletion.

## Resolved design decisions

1. **Argv home**: the literal lives ONLY in `_prune_registrations`; `prune_worktrees`
   stays as the public delegate so existing harness call sites keep working. (Safe
   default; avoids a rename ripple.)
2. **Guard-test mechanics**: AST scan (List/Tuple of three string constants), not text
   grep — comments/docstrings legitimately mention the phrase (verify.py DD5,
   prune_worktrees docstring) and must not count. Precedent:
   `test_event_loop_antipattern_guard.py`.
3. **`_abort_lane_acquisition` never discards WIP**: commit-first, retain-on-doubt —
   consistent with the create_worktree fail-safe (git_ops.py:1559) and
   `detach_lane_checkout`.
4. **Degenerate-ref check runs on every fault route** (upgrade from the comment-enforced
   create-once-only scoping of task 2112). `warm_lane_ref_is_degenerate` is fail-soft
   False and `_delete_branch_if_on_main` retains commit-bearing branches, so widening
   the check cannot delete real work.
5. **`release_warm_lane` unchanged** on the normal (non-fault) release path — the
   primitive is for fault exits only; conflating the two would re-couple the branch
   lifecycle that tasks 1912/1914 deliberately decoupled.
6. **PROTECTED_PREFIXES fails open for non-band names** — it guards the named ephemeral
   bands only. Making it a general deny-list would break task-worktree cleanup
   (`agent-*`, task dirs) and is W11's LaneLifecycle territory.
7. **Prune *semantics* unchanged** — the guard/refusal semantics are exactly task
   2099's; this PRD only relocates them to a chokepoint (per the stream brief's
   "chokepointing only" scope).

## Pre-conditions for activating

All satisfied on main (verified 2026-07-06): `pool_in_use` (git_ops.py:915),
`pool_storage_present` (:930), `_note_pool_storage_absent` (:977),
`_pool_storage_bootstrap_ok` (:994), `warm_lane_ref_is_degenerate` (:2136),
`_delete_branch_if_on_main` (:3735), `release_warm_lane` (:3355), `commit` raise-shape
(:3887), source-scan guard-test precedent
(`orchestrator/tests/test_event_loop_antipattern_guard.py`). Tasks 2062, 2097–2100,
2112 all `done`.

## Cross-PRD relationship (G4)

| Other stream/PRD | Direction | Seam mechanism | Owner | Status |
|---|---|---|---|---|
| W11 worktree-lane-lifecycle | W11 consumes | `_prune_registrations`, `_abort_lane_acquisition`, PROTECTED_PREFIXES | **M1 (this PRD)** | W11 unfiled (wave 2); its session wires deps on this batch's task ids |

No contested seams; ownership is fixed in the program doc's G4 table (M1 row).

## Decomposition plan

| # | Task | Files | Prereqs | Observable signal |
|---|---|---|---|---|
| α | `_prune_registrations(context)` chokepoint + convert the 5 raw git_ops prune sites | git_ops.py, orchestrator/tests/test_prune_registrations_chokepoint.py | — | With pool configured + storage absent (simulated), a merge-lane cleanup (`delete_solo_branch`) logs `refusing to run \`git worktree prune\`` with its context tag and fires `_note_pool_storage_absent`; with storage present the prune runs identically to today (rc logged). Unlocks β, γ. |
| β | Route harness substrate-gate cleanup through the chokepoint | harness.py, orchestrator/tests/test_substrate_gate.py | α | Substrate-gate stale-worktree cleanup emits the chokepoint's context-tagged prune log (`substrate-gate-cleanup`); no raw prune argv remains in harness.py. Unlocks γ. |
| γ | CI grep-guard: exactly one prune argv in orchestrator/src | orchestrator/tests/test_prune_chokepoint_guard.py | α, β | `pytest orchestrator/tests/test_prune_chokepoint_guard.py` passes on main; planting a second `['git','worktree','prune']` literal anywhere under orchestrator/src makes it fail with a file:line offender list (CI/pre-merge verify consumer). |
| δ | `_abort_lane_acquisition` teardown primitive routed from every acquire fault exit | git_ops.py, orchestrator/tests/test_warm_lane_abort_teardown.py | — | Fault-injection: exception after the reuse-path checkout leaves lane HEAD detached + pool FREE; follow-up acquire does NOT log `already used by worktree`. Existing acquire-path tests stay green (happy paths byte-identical). |
| ε | PROTECTED_PREFIXES registry + foreign-band refusal in cleanup sweeps | git_ops.py, harness.py, orchestrator/tests/test_protected_prefixes.py | — | A sweep steered at a foreign band (simulated `_iact-*` dir reaching the merge sweep's removal step) refuses with a WARNING naming the owning band and leaves the dir intact; owned-band and non-band removals behave exactly as today. |

α is intermediate (unlocks β, γ); β is intermediate (unlocks γ); γ, δ, ε are leaves.
α/δ/ε all touch git_ops.py — the orchestrator's file locks serialize them; no semantic
dep exists (they edit disjoint regions), so no artificial ordering edges.

## Out of scope

- LaneLifecycle durable state, `.task/` relocation, per-lane records → stream W11
  (depends on this batch).
- Any change to prune/removal *semantics* — chokepointing only.
- `git worktree remove` chokepointing (path-scoped, not registration-global).
- `scripts/warm-lane-gc.sh` / shell-side sweeps (out of the Python argv surface the
  guard test covers).
- The 7-route `acquire_warm_lane` classifier/table refactor (separate survey finding;
  δ deliberately only unifies the *fault* exits).

## Open questions (tactical)

1. **Context-string vocabulary** for `_prune_registrations` — exact tags are
   implementer's choice; keep them stable and greppable. Decide in α.
2. **Whether β asserts via a mocked GitOps or a real repo fixture** in
   test_substrate_gate.py — either satisfies the signal. Decide in β.
3. **Exact placement of the foreign-band check inside each sweep** (before candidate
   enumeration vs before each removal) — before each destructive call is the safe
   default. Decide in ε.
