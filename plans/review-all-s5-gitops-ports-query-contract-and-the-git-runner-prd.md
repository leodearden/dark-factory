# S5 — GitOps ports, query contract and the git runner (PRD)

**Status:** draft, authored 2026-11-01 by the S5 author seat of the `/review-all` remediation program
(`plans/review-all-program-dark_factory-2026-10-10.md`, stream S5, agent mode, wave 2, size L, risk high,
ruling R5); critic pass and folds 2026-11-02 (the critic's B1–B4, M1–M15, m1–m18 and its delta passes
applied); post-sitting ruling fold 2026-11-10: every design question is ruled (§Rulings), the open-question
markers are gone, and every claim is re-checked against main; fold-verify 2026-11-10 at `bb1c659562`
applied the lead's sequencing rulings (λ0 second in the chain; the OQ-5 follow-up σ is filed); the alignment
pass of 2026-11-10 applied the lead's later rulings (S5 α and λ0 follow S6 α; σ depends on θ only). B+H:
contracts C1–C8 and a two-way
boundary-test sketch (G5 fired on every count: blast radius orchestrator + escalation + fused-memory +
shared, 14 mechanisms, the `run_git` / `LaneLifecycle` / `GitOpsCallbacks` / `AdvanceOutcome.detail` seams,
consumers S3, S4, S6, S7, S8, S9, S10). Twenty-three tasks: one total order of twenty-one leaves, the
independent μ, and the low-priority follow-up σ (ruling OQ-5), which π does not wait on.

**Code anchors** verified against main `e634cfe658` (2026-11-10). Main moves fast — cite by symbol;
re-locate lines at implementation time. History: authored at `ce0f71a384` (2026-11-01) and folded at
`d40ba048e9` (2026-11-02); main has since moved 2,532 commits, 32 of them on `git_ops.py` (16,439 → 16,667
lines). The ruling fold re-ran, at `e634cfe658` and by the rule stated beside each, every count that a
done-criterion or a delivered check rests on. A count labelled `measured at e634cfe658` changed since the
fold or was first measured by it; a count still labelled `measured at d40ba048e9` or `ce0f71a384` gave the
same answer at `e634cfe658` by the same rule. §Freshness at e634cfe658 lists what moved. The fold-verify
re-checked every anchor the 77 later commits touched at `bb1c659562` (2026-11-10); a count it changed is
labelled `measured at bb1c659562` (§Freshness at bb1c659562).

## Goal

`shared.git_async.run_git` is the one place the factory spawns git. `GitOps` is constructed, never finished
after the fact: its scripts, probes, limits, callbacks and durable lane record arrive through its
constructor, so a test passes a fake through the public seam instead of patching 41 private names (measured
at `e634cfe658`). A `GitOps`
read query either answers or raises `GitQueryError` — it never hands back an empty value that a caller then
reads as a fact. One table says what a warm lane's unavailability means; one primitive moves
`refs/heads/main`; one context manager leases a lane; one module owns writes to `refs/heads/task/*`; and
`git_ops.py` is partitioned along the state its methods actually touch, each piece passing heuristic 13.

What an operator observes when this lands (the user-observable surface, G1/G2):

- `git grep -nE "from orchestrator\.git_ops import[^#]*\b_run\b|^\s+_run,\s*$|(= |await )git_ops(_module)?\._run\b" -- orchestrator/src escalation/src ':(exclude)orchestrator/src/orchestrator/git_ops.py'`
  is empty (10 modules, 14 sites today, measured at `d40ba048e9`); `git_ops.py` has no `_run`; the nine
  orchestrator modules that import it call `shared.git_async.run_git` or `orchestrator.git_run.run_in_worktree`
  per α's table (C1), and `escalation/server.py::release_warm_worktree` calls `run_git` (S3 δ's repoint).
- `scripts/merge_lane_metrics.py` measures patch targets under `orchestrator.git_ops` and the S5 modules
  that take its methods (`worktrees`, `main_landing`, `task_branch`, `lane_lease`, `git_queries`; today it
  measures only `LANE_PACKAGE = 'orchestrator.merge_lane'`, measured at `e634cfe658`), and π's new
  `scripts/check_gitops_private_patch_targets.py` finds no `_`-prefixed patch target under them in
  `orchestrator/tests` or `escalation/tests`; no test names `orchestrator.git_ops._` (24 files today,
  measured at `ce0f71a384`). A test that needs git to fail passes a `ScriptedGitRunner` to
  `GitOps(runner=...)`.
- A failed `git diff` during Lever-B revalidation, inter-iteration rebase or suffix-conflict recomputation
  surfaces in the journal as `GitCommandFailed: argv=… cwd=… rc=128 stderr=…` (the command form of
  `GitQueryError`) and the task blocks with that text, instead of the architect's revalidation being skipped
  or a train stacking a member that "touches nothing" (C4). `git_ops.py::GitQueryFailed`, the message-only
  error tasks 7485 and 7332 added for seven lookups, no longer exists: it is `GitQueryError` under R5's name.
- Adding a warm-lane unavailability cause is one enum member carrying its requeue class, disposition, steal
  retryability and seed exit codes; the `workflow_types` row, the `create_worktree` raise and the FAULT
  message follow from it, and a member without a requeue class fails at import (C3).
- `merge_lane/gates.py::_map_advance_failure` matches on `outcome.detail` — a frozen union with one variant
  per detail-bearing result — so a missing fact is a pyright error; `getattr(git_ops, '_last_…')` is
  gone, deleted by η together with the writes (C5). A break-glass `recover_red_main` waits out a foreign `index.lock` and retries its
  `read-tree` exactly as `advance_main` does, because both call `_move_main`.
- `.lane-state/<lane>.json` has one owner with a public API: `Harness` holds a `LaneLifecycle` it built and
  passed to `GitOps` and `WarmLanePool`; `grep -rn "git_ops._lane_lifecycle" orchestrator/src` is empty; a
  corrupt record is reported as corrupt (`LaneRecords.corrupt`), never as "no record" (C6).
- The remote `orchestrator verify-merge` and the local `GitOps.merge_verify_lease` take the lane lock through
  the same `lane_lease` context manager; the `.merge_verify.lock` span lock keeps only its gap-closing purpose,
  stated once in `lane_lease`'s docstring, and its old-binary rollout rationale is gone (C7).
- `git_ops.py` carries a measured, named partition in its commit record (heuristic 14), and no satellite
  imports from `git_ops.py` (heuristic 13). The expected residual, re-estimated at `e634cfe658` from the
  §Sketch partition: cluster (c) ≈4,060 lines (4,034 at `ce0f71a384` plus task 3076's `lane_in_use` and
  `reset_stale_lane_in_use_records`) + ≈2,510 non-class lines (2,982 at `e634cfe658` less ≈470 moved by θ
  and ζ; the types ξ0, ο0, λ and ρ0 move come off too and are not counted) ≈6.6k lines once ρ extracts
  cluster (f). That is ≥3× the 2,000-line alarm; the remaining partition is cluster (c), the only cluster
  that owns pool state, and ρ's commit record names it.

## Background

The `/review-all` run `review-all-dark_factory-20261008` found that `orchestrator/src/orchestrator/git_ops.py`
is one 203-method class whose purpose cannot be stated without "and" (heuristic 6), that it is reached into
from four other modules and 24 test files by private name (heuristic 7, Tests stance), and that it hand-rolls
the same decisions in several places: four unified-diff parsers with three header-disambiguation rules
(heuristic 11), seven "does this tree hold work" predicates with opposite failure directions (heuristic 10),
ten read queries that discard the git return code (INV-11), three encodings of "why a warm lane could not be
provided" kept in step by hand (heuristic 3), two copies of the protocol that moves `refs/heads/main`, two
copies of the lane-lock lease, and thirteen writers of `refs/heads/task/*` with no owner of the invariant
every consumer assumes. Twenty-four finding keys; the program's R5 ruling (2026-10-10) chose option 1 for
the warm-lane encoding and accepted the rest of the remedy as proposed.

Since the run (7,689 commits), only the edges moved: `get_changed_files` now warns on `rc != 0` but still
returns `[]` (task 3493); `get_conflict_details` sits on a new byte-faithful `GitOps.unmerged_paths`
(task 7302); `is_ancestor_checked` distinguishes a git error from "no" (task 6861); `AcquireRoute` labels
the seven acquire routes after the fact (task 7863) without separating them; `lane_branch_holder.py`
(task 5565) added a fifth bare `LaneState` importer; and the test private-patch surface grew from 128 sites
in 27 files to 159 in 31 (measured at `e634cfe658`). Every structural claim re-verified; the dispositions
are in §Findings disposition.

Between the fold and the ruling fold the edges moved again (§Freshness at e634cfe658): tasks 7485 and 7332
added a raising `_git_query` helper and a message-only `GitQueryFailed` for seven lookups; task 5695 deleted
`get_diff_from_main`; task 6428 renamed `_run_thin_warm_lane` to `thin_warm_lane`; task 6219 put the requeue
rebase behind `_rebase_requeued_onto_main`, which un-stacks first; task 7332 added a ninth holds-work
predicate, `branch_tip_if_no_commits_beyond`; task 3076 landed the IN_USE spans; task 8267 split the
harness's crash-recovery and lane-checkout sweeps per entry. No finding's cost is gone.

Prior work this PRD builds on and does not redo: `shared.git_async.run_git` already exists and `git_ops._run`
is already a thin adapter over it (task 3778); `lane_lifecycle.py::LaneLifecycle` with `LEGAL_TRANSITIONS`
exists (task 2254) and the pool already accepts `lane_lifecycle` in its constructor; `merge_lane/types.py::
QueuedBranch` is the typed branch identity at the queue boundary (tasks 2176/2178); `AdvanceOutcome` is
already a frozen value (task 1997); `rebase_recovery.py::guarded_abort` already disables rerere for aborts
(task 4797); `git_ops.py::GitQueryFailed` and `_git_query` already raise on a failed lookup (tasks 7485,
7332), and α gives that type R5's name rather than adding a second one.

## Rulings (sitting of 2026-11-10)

Every design question of this stream is ruled. The text below is authored under these rulings and carries no
open-question marker.

| Id | Ruling | Ruled by |
|---|---|---|
| OQ-1 | A: `BlockDisposition` and `RequeueKind` move to the leaf `orchestrator/block_disposition.py`; the derivation from `WarmLaneUnavailable` replaces the nine `workflow_types` rows (D4, C3, δ) | the lead (shown to Leo in the uncontested block) |
| OQ-2 | A: `QueuedBranch` moves to `shared.branch_names` in the codemod λ0, every importer repointed, no re-export; S6 imports it from there (D11, C8, λ0) | the lead |
| OQ-3 | A′: task 4752 is folded into λ (`OntoRef = MainTip \| StackedOn \| FrozenPrefix`, superseded `metadata['train']` clearing, `_resolve_predecessor_tip` validation, the 3269/4405 reproductions as B23 rows); the lead adds 4752 → λ when λ is minted and cancels 4752 at decompose after a heartbeat check (D11, C8, λ) | Leo (sitting key S5-1, as recommended) |
| OQ-4 | A: γ and ι do the minimal `Harness.__init__` wiring and delete the installs and reaches; S8 β re-homes the construction behind a same-file edge on γ and ι, and S8 drops its held λ (C2, C6) | the lead (= S8/OQ-10 (a)) |
| OQ-5 | B′: the `.merge_verify.lock` co-lock stays, renamed the span lock; its gap-closing purpose is stated once in `lane_lease`'s docstring and the old-binary rollout prose goes (D8, C7, B18b, θ). The recommendation Leo accepted includes the follow-up: one low-priority task σ for the re-entrant lane lease (the A′ form), filed at this plan's decompose and depending on θ | Leo (sitting key S5-2, as recommended); the follow-up's reading is the lead's sequencing ruling of 2026-11-10 |
| OQ-6 | A: the `WorktreeMissing` adaptation is the leaf `orchestrator/git_run.py::run_in_worktree`; `git_ops._run` is deleted (D2, C1) | the lead; recorded as a clarification of R5's wording ("git_ops keeps only the `WorktreeMissing` adaptation": its intent, not its letter) |
| OQ-7 | B: `AdvanceOutcome.detail` is the frozen union `WipOverlap \| PopConflict \| StashFailed \| ParkLockContended \| NoDetail` (C5, η) | the lead |
| OQ-8 | A: η deletes the 5 `getattr(git_ops, '_last_…')` reads with the 7 writes; S6's ε stays deleted; fk-59b3df6f52a3 is disposed in η (C5) | the lead (= S6/OQ-13 (a)) |
| OQ-9 | A: ρ0 and ρ extract cluster (f) into `orchestrator/git_queries.py::GitQueries` after ο (D14) | Leo (sitting key S5-3, as recommended) |

No ruling of this stream amends Leo's 2026-10-10 ruling R5; OQ-6 is recorded as a clarification of its
wording.

Lead sequencing rulings after the ruling folds (2026-11-10; they win over anything earlier):

- **λ0 is second in the chain**, right after α and before β0 and γ: **α → λ0 → β0 → β → γ → …** λ0 is a
  mechanical move with no semantic prerequisite past α (α shares `workflow.py` and `merge_lane/worker.py`
  with it). Every S6 task other than S6 α that edits `merge_lane/types.py`, `merge_lane/worker.py` or
  `harness.py` imports is ordered after S5 λ0 by the lead's per-file order. S6's request "S5 λ0 waits on
  S6 λ and S6 μ" is declined: it would put S5's tail behind the native-port chain.
- **S6 α lands first on every file it shares with another plan** (the lead's ruling after the
  fold-verifiers, 2026-11-10; S6 α is the timing-budget validator, INV-15, which fixes a live
  misconfiguration). S6 α takes no edge on any S5 task; instead **S5 α → S6 α** (shared
  `merge_lane/worker.py`, `verify.py`) and **S5 λ0 → S6 α** (shared `merge_lane/types.py`,
  `merge_lane/worker.py`, `harness.py` and the `QueuedBranch`-importing merge-queue tests). The rest of S5's
  chain follows S6 α through α and λ0.
- **OQ-5's follow-up is filed**: one low-priority task σ for the re-entrant lane lease (A′), filed at this
  plan's decompose and depending on θ (§Decomposition plan).

Cross-stream rulings of the same sitting that bind this plan: S6-2 = B (task 5040 finishes first and S6's
lifecycle task waits on it, so S5 γ, which waits on that S6 task, now waits on 5040 too); same-file order on
`merge_lane/worker.py` and `gates.py` unchanged (S6's typed facts first: S5 γ → S6 ζ, S6 κ → S5 θ); S3-1 = A
leaves S5 β's edge on S3's operator-tool move (S3 δ, not held) unchanged; S8/OQ-10 (a) names S8 β as the leaf
that re-homes the harness construction.

## Finding keys

24 keys, all `open` at the run, none carried by any task today (contract §8 step 1 run on 2026-11-01 and
re-run by the ruling fold on 2026-11-10: `find_tasks_by_metadata(key="x_finding_key", value=<key>)` returned
no task for any of the 24).

### Findings disposition

Dispositions per the brief's vocabulary. `leaf <label>` means the leaf closes the key and stamps it.

| Key | Severity | Anchor (at main) | Disposition | Note |
|---|---|---|---|---|
| fk-b1e3368f99a1 | medium | `git_ops.py::GitOps.__init__` | leaf ι (+γ for the 4 installs) | 16 private reaches from 4 modules at `d40ba048e9`; 8 `git_ops._lane_lifecycle` reaches in `harness.py` (measured at `e634cfe658`: task 8267 moved crash recovery's per-entry body into `Harness._recover_worktree_entry`, task 2987 added `Harness._advance_lane_record_divergence_and_alarm`); the 3 `.transition` writes now sit in `Harness._recover_worktree_entry`; 2 `LaneState` enums — all re-observed |
| fk-6f7520ff1987 | medium | `git_ops.py::WarmLaneUnavailable` | leaf δ | 10 members, 8 `WarmLaneRequeue` subclasses, 9 `pool_info is` arms + residual literal list, 9 `workflow_types` rows, 46 `WarmLaneUnavailable.` refs — byte-identical to as-of; task 6594 (this run's mechanical filing for `_STEAL_RETRYABLE`, fk-04c87a1b5dd8) is cancelled into δ |
| fk-e2ac53b0d72b | medium | `git_ops.py::GitOps._seed_warm_lane` | leaf γ (runner port in β) | grown: 41 private names / 159 sites / 31 test files (measured at `e634cfe658`; 158 at `d40ba048e9` by the same rule); the three production test accommodations still present |
| fk-d727d82341b3 | medium | `git_ops.py::parse_diff_line_ranges` | leaf ζ | still unguarded; 4 parsers, 5 `worktree list` parsers, 8 `rev-list --count` sites with 6 fail encodings |
| fk-d176c9ae1cc7 | medium | `git_ops.py::GitOps.get_changed_files` | leaf ε (α gives the existing error R5's name) | 9 read queries and 2 write paths still drop the rc (measured at `e634cfe658`; task 5695 deleted `get_diff_from_main`, `09f20e85c1`); `get_changed_files` warns but still returns `[]` (task 3493); tasks 7485 and 7332 added `GitQueryFailed` and a raising `_git_query` for 7 lookups only; `get_main_sha` has 50 awaited call sites in 8 files, 13 of which guard the empty value (measured at `bb1c659562`: task 8524 folded two guarded worker sites into `SpeculativeMergeWorker._main_sha_or_empty`; 51/14 at `d40ba048e9` and `e634cfe658`; table in ε) |
| fk-7a3ee979b48a | low, weakened | `git_ops.py::GitOps.recover_red_main` | leaf η | re-verified under G3/G6: 6 hook sites (2+4), 2 CAS writers, 3 `read-tree` syncs, index-lock stand-off and `rev-parse --verify` each on one side only — cost present, the weakening was about the hook count (5 → 6) and intent, not existence |
| fk-78b757799cab | low | `artifacts.py::TaskArtifacts._read_json_object` | leaf ν | 10 raw non-plan `json.loads` readers vs 6 seam users (`read_title` joined the seam; measured at `e634cfe658`); plan-family readers excluded (S9, R2) |
| fk-a7cefe40ef4c | medium | `git_ops.py::GitOps._acquire_warm_lane_impl` | leaf κ | 879 lines (as-of 889), seven routes inlined behind an after-the-fact `AcquireRoute` label; identity-mismatch reseed still inline |
| fk-a7d433cf8308 | high | `git_ops.py::GitOps` (task-ref writers) | leaf λ (λ0 moves `QueuedBranch`) | 13 writer sites, 10 rebase entry calls of which 3 pass the post-condition wrapper (`_reuse_warm_lane`'s now through task 6219's `_rebase_requeued_onto_main`, which un-stacks first); no owner; task 4752 still pending (2026-11-10) |
| fk-ad8c0528c5e1 | medium | `git_ops.py::GitOps.has_uncommitted_work` | leaf ε | 9 predicates (task 7332 added `branch_tip_if_no_commits_beyond`, fail-safe `None`), 3 fail directions (True or `None`-retain ×5, CLEAN ×3, 0-sentinel ×1; measured at `e634cfe658`) |
| fk-eb7b9f65f412 | medium | `git_ops.py::GitOps.advance_main` | leaf η | 7 `_last_*` writes, none initialised, 5 `getattr` reads in `gates.py::_map_advance_failure`; S6's fk-59b3df6f52a3 is the same defect and is disposed with these two keys by η (ruling OQ-8) |
| fk-3ee93d431321 | medium | `git_ops.py::GitOps.advance_main` | leaf η | same evidence as fk-eb7b9f65f412 |
| fk-856541951d65 | high | `git_ops.py::_seed_rc_to_unavailable` | leaf δ (DF side); shell-side exit codes are reify's | 4-code map + 9 inline rc checks in 6 methods + `_seed_script_supports` text grep, unchanged |
| fk-c5d5769d68c6 | medium | `git_ops.py::GitOps.acquire_spec_lane` | leaf κ | duplicated 21/28-line bootstrap block; 4 `mark_pool_storage_present` writers against a docstring claiming one; `_reconcile_pool_storage_before_sweep` already encodes the predicate for one caller |
| fk-718c82fcc1a2 | medium | `git_ops.py::GitOps.resolve_queued_branch_ref` | leaf λ (λ0 moves `QueuedBranch`) | 30 inline `branch_prefix` f-strings in 23 scopes; try-both-forms resolver unchanged; `QueuedBranch` exists at the queue boundary |
| fk-739af676767e | medium | `git_ops.py` module docstring / `GitOps.commit` | leaf ξ | "No .task-specific guards remain" still stated while `merge_to_main` removes `.task/` and `artifacts.py::TaskArtifacts.ensure_lane_symlinks` (task 3160) links `<worktree>/.task/plan.json` and `<worktree>/.task/iterations.jsonl` on every dispatch (measured at `e634cfe658`) |
| fk-c00c5f01be4c | medium | `git_ops.py` (h14) | leaves ξ0, ξ, ο0, ο, ρ0, ρ (+ θ, ζ, λ extract clusters a, the parsers of f, d) | 16,667 lines: 6,737 docstring + 2,749 comment (56.9%), 5,691 code (measured at `e634cfe658`); measured partition in §Sketch |
| fk-d73aeaebb792 | medium | `git_ops.py::_run` | leaves α, β (+ the partition in ξ, ο) | 10 src modules / 14 sites import or read `_run` (measured at `d40ba048e9`): 9 orchestrator modules + `escalation/server.py` (S3 δ repoints that one); 6 function-local |
| fk-3a4cc6298c02 | medium | `git_ops.py::_run` (tests) | leaves β0, β | 135 test files import `_run` from `orchestrator.git_ops` (133 orchestrator, 2 escalation; AST), 151 with the patch/attribute forms (measured at `e634cfe658`; 131 and 147 at `d40ba048e9` by the same rules); the author's 254 refs / 32 files counted patch-form references only |
| fk-31e9856ed662 | high | `lane_lifecycle.py::LaneLifecycle` | leaf ι | reduced since W11 but harness still falls back on `None`; `restore_assignment` still called from both harness (6 code sites: `Harness._recover_worktree_entry` ×1, `Harness._reconcile_lane_checkouts` ×1, `Harness._reconcile_one_lane_checkout` ×4) and `GitOps._settle_branch_checkout` ×2 (measured at `e634cfe658`; task 8267 split both sweeps per entry) — ι routes all eight through `GitOps.adopt_lane`, so pool assignment has one mutating module |
| fk-a10d37257808 | low | `lane_lifecycle.py::LaneLifecycle.read` | leaf ι | `CorruptLaneRecord` → `None` + WARNING; `all_records` skips silently; unchanged at `e634cfe658`. Task 3076 added `LaneLifecycle._read_best_effort` (`OSError` → `None`) so the IN_USE spans and sweep never abort; ι keeps that guarantee (C6) |
| fk-54eb9f7cc0be | medium | `rebase_recovery.py::guarded_abort` | filed:5697 (dependency of λ; lead stamps the key) | task 5697 (pending, medium) carries exactly this work: `RECOVERY_GIT` on the three rebase primitives, abort rc read, preflight before rebase; its `details` re-measured it on 2026-09-21 |
| fk-20135ef54f75 | medium | `cli.py::verify_merge` | leaf θ | reshaped since as-of: the click command is 59 lines; the protocol lives in `cli.py::_merge_lane_gate` (a context manager) and `cli.py::_verify_merge_body`'s build-scoped hold; `GitOps.merge_verify_lease` (218 lines, 75 comment) is still the second copy; the compat co-lock is still held |
| fk-c04bf2045ad3 | low | `task_interceptor.py::_verify_commit_on_main` | leaf μ | 3 `create_subprocess_exec` probes, 1 of 3 reaps on timeout; `landed_on_main.py::_is_ancestor` uses `run_git`; unchanged at `e634cfe658`. Task 6236 (pending, low) asks for the same runner seam inside a module extraction; it depends on μ (§Related tasks) |

Every key appears exactly once. No key is `fixed`, `stale` or `accepted`; re-checked at `e634cfe658`, no
key's cost is gone.

### Freshness at e634cfe658 (ruling fold, 2026-11-10)

`git log d40ba048e9..main` over the 95 existing paths this plan touches: 57 changed (30 of them source or
script files). What moved, and where the plan now says so:

- **Task 7485 (done, `d37a05026d`) and task 7332 (done; `95596f1356`).** `git_ops.py::GitQueryFailed`
  (message-only `RuntimeError`), the module helper `_git_query` (raises it on rc != 0; 9 calls), the
  fail-safe wrapper `_none_if_query_failed`, `GitOps.commits_named` and the `*_checked` twins of
  `find_merge_marker` and `find_task_citation_commit`; 7 raise sites (3 over a failed rc, 2 over unparseable
  `cat-file --batch-check` output, 2 over an unknown `is_ancestor_checked` / `merge_base_with_main`); a
  `workflow_types.py::_disposition_table` row (BLOCK + escalate); importers `workflow_types.py` and
  `scripts/restamp_degenerate_branch_bases.py`. → D5, C1, C4, the placement table, α (one name), β
  (`_git_query` becomes `GitOps._query`), ε (`GitCommandFailed`). The edge "7485 depends on ε" is gone.
- **Task 5695 (done, `09f20e85c1`).** `GitOps.get_diff_from_main` deleted. → ε's query list, C4, and ρ's
  signal and delivered check, which named it.
- **Task 6428 (done, `e22f619d55`).** `GitOps._run_thin_warm_lane` is now the public `thin_warm_lane`. → D10,
  κ, B22.
- **Task 6219 (done, `ac4fec65cd`).** `_reuse_warm_lane` rebases through the new
  `GitOps._rebase_requeued_onto_main`, which un-stacks from an unlanded base first. → λ's writer list and
  κ's REUSE route; the edge "6219 depends on κ" is gone.
- **Task 7332.** A ninth holds-work predicate, `GitOps.branch_tip_if_no_commits_beyond` (fail-safe `None`),
  sole caller `workflow.py::TaskWorkflow._restamp_branch_base_if_no_own_commits`. → ε's predicate table,
  `BranchWork.tip`. A new composition root, `scripts/restamp_degenerate_branch_bases.py`, builds a `GitOps`.
- **Task 3076 (done, `1f40b7986b`).** `GitOps.lane_in_use`, `reset_stale_lane_in_use_records`,
  `LaneLifecycle.note_in_use` / `note_idle` / `reset_stale_in_use`, `_read_best_effort` and `LaneLiveness`.
  → ι no longer depends on 3076; C6 keeps 3076's never-abort guarantee.
- **Tasks 8267 and 2987.** Harness crash recovery and the lane-checkout reconcile now run per entry
  (`Harness._recover_worktree_entry`, `Harness._reconcile_one_lane_checkout`); the lane-record divergence
  digest (`Harness._advance_lane_record_divergence_and_alarm`, new module `lane_record_divergence.py`) reads
  `git_ops._lane_lifecycle`. → ι's reach and `restore_assignment` counts.
- **Task 3155 (done, `ef96d32603`) and task 3160.** `artifacts.py::ARTIFACT_ROOTS` is the one resolver;
  `TaskArtifacts.ensure_lane_symlinks` links `plan.json` and `iterations.jsonl` into the lane's `.task/`. → ν
  no longer depends on 3155; fk-739af676767e's note and D14.
- **Task 4911 (done, `68403ec55a`).** Module-level `task_citation_pattern` and `CitationPatternInvalid`,
  imported by `agents/landed_sha_searches.py`, `workflow_types.py` and
  `scripts/restamp_degenerate_branch_bases.py`. → the placement table (ρ0).
- **Task 7298 (`6cf7a271a9`, before the draft).** The ratchet's authorized raises are one file per entry under
  `orchestrator/tests/merge_lane_ratchet_raises/`; `merge_lane_ratchet_authorized_raises.json` no longer
  exists. → the INV-12 line of the G7 walk.
- **Task 6433.** `scripts/merge_lane_metrics.py` has no `LANE_PATCH_MODULES`; the lane set is
  `LANE_PACKAGE = 'orchestrator.merge_lane'` via `lane_patch_modules`. → B28, π, Pre-conditions.
- **Task 7485's test shim.** `orchestrator/tests/_git_fault_shim.py` puts a failing `git` first on `PATH`
  (one importer, `test_git_ops_rebased_citation.py`): a second git fake beside `_worktree_add_fakes.py`. → β.
- **fused-memory tests.** The probe reference in `fused-memory/tests/_fm_helpers.py` moved to
  `_fm_git_repos.py`. → μ.
- **Counts re-measured** (labelled `measured at e634cfe658` where they appear): test files importing `_run`
  131 → 135; single-line importers 120 → 124 files; `GitOps(` test constructors 184 → 200;
  `LaneLifecycle(`/`WarmLanePool(` 17 → 19; `QueuedBranch` test importers 86 → 89; `BlockDisposition` /
  `RequeueKind` test importers 5 → 6; private patch sites 158 → 159; `workflow_types` names imported from
  `git_ops` 19 → 21; `git_ops.py` 16,439 → 16,667 lines, `GitOps` 203 → 210 methods (+9: `_commit_paths`,
  `_rebase_requeued_onto_main`, `branch_tip_if_no_commits_beyond`, `commits_named`,
  `find_merge_marker_checked`, `find_task_citation_commit_checked`, `lane_in_use`,
  `reset_stale_lane_in_use_records`, `thin_warm_lane`; −2: `_run_thin_warm_lane`, `get_diff_from_main`).
- **Unchanged by the same rule at both shas:** the 14 `_run` import sites in 10 modules, the 4
  post-construction installs, every `WarmLaneUnavailable` count, `get_main_sha`'s 51 sites scope for scope,
  the lane-lease functions in `cli.py`, `advance_main` (690 lines) and `recover_red_main` (216), the 7
  `_last_*` writes and 5 `getattr` reads, the task-ref writer, deleter and prefix-construction sites, the
  porcelain and `rev-list --count` sites, and the 10 fused-memory `create_subprocess_exec` sites.
- **Task statuses changed:** 3076, 3155, 6219 and 7485 are done (edges dropped). Every other task this plan
  names is unchanged; 4752 is still pending, high, with no claimant or heartbeat.

### Freshness at bb1c659562 (fold-verify, 2026-11-10)

`git log e634cfe658..main` (77 commits): of this plan's source files only `merge_lane/worker.py`,
`merge_lane/types.py`, `workflow.py`, `artifacts.py`, `lane_lifecycle.py`, `escalation/server.py` and
`merge_lane/speculation_controller.py` changed, and only task 8524 (merge-lane re-anchor) reached an
anchor. It routed `SpeculativeMergeWorker._finalize_inflight`'s INV-3 read and `_dispatch_item`'s
`_dispatch_main` read through a new fail-open helper `SpeculativeMergeWorker._main_sha_or_empty` (4 callers),
moved `_guard_main_sha` into `SpeculativeMergeWorker._launch_verify`, and renamed `_remerge_inner` to
`SpeculativeMergeWorker._remerge_on_main`. `get_main_sha` now has 50 awaited sites in 8 files (ε's table,
measured at `bb1c659562`). It also added `orchestrator/tests/test_merge_lane_remerge_base.py`, a 90th
`QueuedBranch` test importer (λ0). Unchanged by the same rules at both shas: the 14 `_run` sites, the 135
test `_run` importers, the 200 `GitOps(` and 19 `LaneLifecycle(`/`WarmLanePool(` test constructors, the 6
`BlockDisposition`/`RequeueKind` test importers, the 3 lease `except` sites and 2 `_map_advance_failure`
callers in `worker.py`, the 8 `src` `QueuedBranch` importers, and `git_ops.py` (16,667 lines). Every
sidecar check keeps its polarity at `bb1c659562`. Task statuses unchanged; 4752 and 5040 still pending.

### Freshness and G3/G6 re-verification (measured at `ce0f71a384`; re-run at `e634cfe658`)

Rules are stated so the decompose pass and the implementer can re-run them. The ruling fold re-ran them at
`e634cfe658`; a changed value carries that label.

- `_run` importers: `from orchestrator.git_ops import … _run` (module-level): `workflow.py`,
  `merge_lane/worker.py`, `merge_lane/disposition.py`, `merge_lane/landing_evidence.py`,
  `merge_lane/gates.py` (also imports `_INDEX_LOCK_STALE_FLOOR_S`), `delivered_checks.py` (three
  `runner = git_ops._run` / `git_ops_module._run` defaults); function-local: `verify.py::_derive_task_files_from_git`,
  `verify.py::_verify_pipeline_guard_requires_full_gate`, `verify.py::confirm_main_tip_failure_is_real`,
  `merge_lane/skew_tripwire.py::_run_load_bearing_oracle`, `scheduler.py::Scheduler._resolve_main_sha` (attribute
  call), `escalation/src/escalation/server.py::release_warm_worktree`. 10 modules, 14 sites (measured at
  `d40ba048e9`; the author's "11 modules" double-counted; the rule is §Goal bullet 1's pattern, which also
  catches the two multi-line `    _run,` imports in `workflow.py` and `merge_lane/worker.py` and skips the
  docstring mentions in `delivered_checks.py`, `rebase_recovery.py` and `skew_tripwire.py`). Inside `GitOps`: 107
  methods, 197 `await _run(...)` calls, plus 9 calls of task 7485's module helper `_git_query`, itself over
  `_run` (measured at `e634cfe658`; 207 and 0 at `d40ba048e9`). `_run` is module-level and byte-identical to
  as-of; its body is the `WorktreeMissing` pre-flight and re-classification over `run_git(..., bounded=False)`.
- Test references to `_run` (measured at `e634cfe658`, B3): 135 test files import `_run` from
  `orchestrator.git_ops` as a fixture helper (AST over `ImportFrom`; 133 in `orchestrator/tests`, 2 in
  `escalation/tests`, 0 in `scripts/tests`); 151 with the patch/attribute forms. The author's 254 refs in 32
  files (`grep -rcE "git_ops\._run\b|GitOps\._run\b|['\"]_run['\"]"`) counted patch-form references only;
  that regex matches 32 files at both `d40ba048e9` and `e634cfe658`. The grep rule
  `from orchestrator\.git_ops import[^#]*\b_run\b|^\s+_run,\s*$` returns 141 files (137 at `d40ba048e9`), some
  of them false matches (the `offline_lane` tests import a different `_run`), so β0 enumerates by AST.
  `orchestrator/tests/_worktree_add_fakes.py` already provides one parameterised fake `_run` (5 refs), and
  task 7485 added a second fake, `orchestrator/tests/_git_fault_shim.py`, a failing `git` placed first on
  `PATH` (1 importer).
- Private patch sites (regexes `patch.object(X,'_n')`, `setattr(X,'_n')`, `<x>._n =`,
  `orchestrator.git_ops.GitOps._n`, `orchestrator.git_ops._n`, names classified by AST of `git_ops.py`):
  41 names / 159 sites / 31 files (measured at `e634cfe658`; 158 at `d40ba048e9`); top: `_seed_warm_lane`
  20/8, `_is_registered_worktree` 16/4, `_on_pool_storage_absent` 14/2, `_warm_lane_base_resolvable` 11/3,
  `_RESET_WARM_LANE_LOCK_WAIT_SECS` 11/3, `_on_structural_exhaustion` 7/2, `_run_warm_lane_gc_reclaim` 6/2,
  `_run_warm_lane_disk_guard` 5/1,
  `_last_recovery_branch` 5/2, `_last_overlap_files` 4/2. Module constants monkeypatched:
  `_RESET_WARM_LANE_LOCK_WAIT_SECS`, `_LANE_LOCK_HOLDER_SETTLE_SECS`, `_MERGE_VERIFY_LEASE_WAIT_SECS`,
  `_SEED_WARM_LANE_LOCK_WAIT_SECS`, `_EFFECT_PROBE_MEMO_MAX_ENTRIES`, `_MERGE_MARKER_INDEX_RECHECK_SECS`,
  `_WARM_LANE_AUDIT_CACHE_TTL_SECS`. Production test accommodations: `_DF_WARM_LANE_SCRIPT_DIR_ENV =
  'ORCH_WARM_LANE_SCRIPT_DIR'`, the `POOL_ROOT_SENTINEL  # noqa: F401  re-export shim` import, and
  `GitOps.refuse_foreign_band` wrapping `GitOps._refuse_foreign_band`.
- Post-construction installs on `GitOps` (`grep -rnE 'git_ops\.[a-zA-Z_]+ *= [^=]'`): 4, all in
  `harness.py::Harness.__init__` (`warm_lane_reclaim_candidate_provider`, `warm_lane_dispatched_predicate`,
  `_on_pool_storage_absent`, `_on_structural_exhaustion`), plus
  `git_ops.warm_lane_pool.set_on_lane_record_drift(...)` and a second `set_lane_lifecycle` in
  `Harness._recover_crashed_tasks` that `GitOps.__init__` already performed.
- `GitOps.__init__` signature: `(config, project_root, *, warm_lane_pool_size=0,
  merge_spec_warm_lane_pool_size=0, transcript_archive=None)`; it constructs `LaneLifecycle(worktree_base,
  quarantine_worktree=self.quarantine_worktree)` and both `WarmLanePool`s itself. Non-test construction
  sites (measured at `e634cfe658`): `harness.py::Harness.__init__`, `cli.py` ×2 (`_run_remote_verify`,
  `cancel_verify`), `recover_main.py`, `evals/runner.py::_cell_workflow`, the operator script
  `scripts/restamp_degenerate_branch_bases.py` (task 7332), plus `scripts/tests/test_stack_base_census.py` and
  `scripts/tests/test_restamp_degenerate_branch_bases.py` and docstrings. 200 test files construct `GitOps(`
  and 19 construct `LaneLifecycle(`/`WarmLanePool(` (AST, measured at `e634cfe658`), which is why C2/C6 keep
  `lane_lifecycle` optional.
- `WarmLaneUnavailable`: 10 members (`EXHAUSTED, FAULT, DISK_PRESSURE, SOFT_PRESSURE, BASE_ABSENT,
  RESEED_CONTAMINATED, LANE_LOCK_TIMEOUT, STEAL_FAILED, LANE_LOCK_CONTENDED, DISABLED`); `WarmLaneRequeue`
  subclasses 8; `GitOps.create_worktree` arms 9 (`DISABLED` raises `RuntimeError`) + residual `RuntimeError`
  whose f-string names the eight classes; `_seed_rc_to_unavailable` maps 75/76/124/77 else FAULT, called at
  4 sites; inline `rc == 127|75|124|77` checks 9 in 6 methods (`create_interactive_worktree`,
  `_seed_warm_lane`, `thin_warm_lane` (task 6428's public name), `prewarm_pool` ×2, `_acquire_warm_lane_impl` ×3,
  `_reset_and_seed_recycled_lane`); `_STEAL_RETRYABLE` placeholder rebound once, one consumer
  (`GitOps.acquire_warm_lane`); `workflow_types.py::_disposition_table` holds 9 `WarmLane*: BlockDisposition(`
  rows; no test iterates the enum. `BlockDisposition` and `RequeueKind` are defined in `workflow_types.py`,
  which imports 21 `git_ops.py` names *inside* `_disposition_table` (task 7332 added `GitQueryFailed` and
  `CitationPatternInvalid`; measured at `e634cfe658`) — a function-local import that breaks the would-be
  cycle today (a heuristic-13 symptom). `git_ops`' import closure does not load `workflow_types` (runtime
  probe re-run at `e634cfe658`), so once D4 moves the two types to a leaf the function-local `git_ops` import
  goes module-level. Importers of the two types: `workflow.py`, `dry_run_unblock.py` and 6 test files
  (`test_reviewer_verdict_routing.py` joined; measured at `e634cfe658`; 7 test files name them; `scheduler.py` mentions
  `BlockDisposition` only in a docstring) — measured at `d40ba048e9`. `verify_categories.py` and
  `unblock_types.py` import no orchestrator module.
- rc-dropping queries (AST: `await _run` results whose rc is `_` or discarded, per method; listed where
  discarded == total; measured at `e634cfe658`): `_reuse_warm_lane` 2/2, `get_diff_from_base`,
  `get_changed_line_ranges`, `get_new_side_changed_line_ranges`, `get_current_branch`, `get_main_sha`,
  `_detect_unmerged_paths`, `has_dirty_working_tree` 2/2, `get_conflict_details`, and two write-path
  methods (`merge_to_main` 1/1, `_create_merge_worktree` 2/2); `get_diff_from_main` is gone (task 5695). Enumerator contracts: `get_commit_changed_files`
  silent `[]`; `get_changed_files` `[]` + WARNING; `get_files_touched_in_branch` `[]` + WARNING;
  `get_merge_diff_files` and `get_branch_changed_files` `(files, err)`. Callers treating the empty value as
  fact: `workflow.py::_plan` (Lever B skip), `workflow.py::_inter_iteration_rebase`,
  `suffix_graph.py::recompute` (its `except Exception` cannot fire on rc), `task_verify_dispatch.py::_task_files`,
  `workflow.py::_maybe_form_train` and `merge_lane/worker.py::_stack_coalesce_train` (both feed `{}` to
  `merge_lane/train_selection.py::select_train_members`), `harness.py::_file_dirty_tree_escalation`
  (`''` from a failed `status` auto-resolves the dirty-tree L2). `get_main_sha` returns `''` on failure and
  has 51 awaited call sites in 8 files (AST, measured at `d40ba048e9`; the critic's grep found 52), 14 of
  which guard the empty value and skip or log (`Harness._run_main_tip_sweep` ×2,
  `OfflineLaneWorker._run_once`, `OfflineLaneWorker._poll_tick`, and 10 more — ε's table). No
  `GitQueryError` type exists; `git_ops.py::GitQueryFailed` (tasks 7485, 7332) is a message-only
  `RuntimeError` raised at 7 sites and read as "no answer" by `_none_if_query_failed` for the fail-safe twins
  (measured at `e634cfe658`).
- Predicates (fail direction on git error): `_branch_has_commits_beyond_main` True; `_orphan_has_commits`
  False on missing branch else True; `worktree_has_unsaved_work` True; `_worktree_dirty` True;
  `has_uncommitted_work` CLEAN (`rc == 0 and bool(output)`); `has_dirty_working_tree` CLEAN (rc unread);
  `worktree_head_beyond_main` None for both "nothing" and "unreadable"; `rebase_preserving_task_commits`
  inline counts → 0 (disarms the guard before, false wipe alarm after); `branch_tip_if_no_commits_beyond`
  (task 7332) `None` when `is_ancestor` reads a git error as False (retain direction: an unreadable branch
  counts as having commits). 25 call sites in 6 modules (`git grep` of the nine names outside their
  definitions, measured at `e634cfe658`), listed in ε.
- Parsers: 4 unified-diff parsers (`parse_diff_line_ranges` unguarded; `parse_diff_added_line_ranges`
  guarded; `_anchor_diff_lines`; `_batch_added_line_counts`), all in `git_ops.py`; `--unified=0` producers 4;
  `'worktree', 'list'` parsers 5 in `git_ops.py` (`_is_registered_worktree`, `_worktree_holding_branch`,
  `lane_branch_checkouts`, `_iter_merge_worktrees`, `_iter_interactive_worktrees`) + 3 elsewhere
  (`fused_memory/services/live_workflow_detector.py::worktree_index_for`,
  `fused_memory/models/scope.py::resolve_main_checkout`, `scripts/reclaim_orphaned_worktrees.py`);
  `'rev-list', '--count'` 8 in `git_ops.py` (`_freshen_main` ×2, `_branch_has_commits_beyond_main`,
  `_reseed_verified_clean`, `rebase_preserving_task_commits` ×2, `get_rebase_distance`,
  `worktree_has_unsaved_work`) + `merge_lane/gates.py::_resolve_already_landed_branch` +
  `live_workflow_detector.py::_branch_own_commit_count`, six fail encodings (local main / True / False / 0 /
  -1 / None). No `FileDelta`/`WorktreeEntry` type exists.
- `advance_main` 690 lines, `recover_red_main` 216, both AST-identical to as-of. `['sh','-c',
  self.config.main_gate_*]` sites: 2 + 4. CAS `update-ref` writers of main: 1 + 1. `read-tree -u --reset
  HEAD` syncs: 2 (incl. retry) + 1. `_await_index_lock_clear`/`_index_lock_state`: advance_main only.
  `rev-parse --verify` of inputs: recover_red_main only. `self._last_*` writes: 7 (`_last_park_lock_info` ×2,
  `_last_overlap_files`, `_last_stash_dirty_files` ×2, `_last_recovery_branch` ×2), none in `__init__`;
  `getattr(git_ops, '_last_…')` reads: 5 in `merge_lane/gates.py::_map_advance_failure`. `AdvanceOutcome`
  fields: `result, advanced_sha, rebased_from, rebased_onto`; `AdvanceResult` has 12 literals, five of which
  (`wip_overlap`, `pop_conflict`, `pop_conflict_no_advance`, `stash_failed`, `park_lock_contended`) carry
  side-channel detail.
- Lane lease: `cli.py::_merge_lane_gate` (sync context manager, ~70 comment lines) acquires the lane lock as a
  contention gate, releases it, then holds the compat `.merge_verify.lock` across the span. Its comment gives
  that hold two purposes: (i) it "closes the momentary lane-release gaps between the gate, reset, and the
  build against a concurrent waiter", and (ii) it "still mutually excludes an in-flight OLD verify-merge";
  only (ii) is rollout-scoped. `cli.py::_verify_merge_body` re-takes the lane lock around the build FAIL-OPEN
  on contention, justified by "the pre-flight gate + the compat co-lock held across the whole span", and
  writes the holder pgid;
  `GitOps.merge_verify_lease` (218 lines, 75 comment) does acquire-off-thread → self-owned-leak probe →
  `MergeVerifyLeaseContended` → holder rendezvous iff persistent lane → release; `GitOps.task_verify_lease`
  (80 lines) is the flock-only variant; `cli.py::cancel_verify` 107 lines, 45 comment. `merge_verify_lock_path`
  is referenced 3× in `cli.py` and defined in `verify_cancel.py`; nothing else holds that lock. `verify_cancel.py`
  exports `lane_lock_path`, `merge_verify_lock_path`, `acquire_merge_verify_flock`, `release_merge_verify_flock`,
  `lane_lock_holder_pids[_strict]`, `write/read/remove_lock_holder_pgid`. The module-level lane-lock helpers
  `_lane_lock_identity … _lane_lock_holder_facts` in `git_ops.py` are 6 functions, 317 lines, state
  `_HELD_LANE_LOCK_FDS`. Lease callers: `merge_lane/worker.py` ×2, `merge_lane/shadow.py`,
  `merge_lane/drift.py`, `workflow.py` ×2 (`task_verify_lease`).
- Durable lane record (AST, measured at `e634cfe658`): `git_ops._lane_lifecycle` reaches — `harness.py` 8
  code sites (1 `set_lane_lifecycle` pass in `Harness._recover_crashed_tasks`; 4 reads:
  `Harness._recover_worktree_entry` ×2, `Harness._assigned_durable_records_with_statuses`,
  `Harness._advance_lane_record_divergence_and_alarm` (task 2987); 3 `.transition` writes in
  `Harness._recover_worktree_entry`: RELEASED with the hand-copied `ASSIGNED/IN_USE` guard, QUARANTINED ×2),
  `task_runtime.py::_build_pooled_snapshot` 1, `stranded_verified_green.py::detect_verified_green` 1.
  `LaneLifecycle` is constructed only in `GitOps.__init__`. `restore_assignment` callers: `git_ops.py::
  _settle_branch_checkout` ×2 and `harness.py` ×6 (`Harness._recover_worktree_entry` ×1,
  `Harness._reconcile_lane_checkouts` ×1, `Harness._reconcile_one_lane_checkout` ×4; task 8267 split both
  sweeps per entry); `note_assignment`: `_acquire_warm_lane_impl` only. Two `LaneState` enums;
  `as DurableLaneState` in `warm_lane_pool.py` and `harness.py`, bare in `stranded_verified_green.py`,
  `task_runtime.py`, `lane_branch_holder.py`, `lane_record_divergence.py` (task 2987) and `git_ops.py`.
  `LaneLifecycle.read` maps `CorruptLaneRecord` → `None` + WARNING; `all_records` skips corrupt records;
  `_read_or_raise` exists; task 3076 added `_read_best_effort` (`OSError` → `None`) under `note_in_use`,
  `note_idle` and `reset_stale_in_use`, so an unreadable record never aborts a `GitOps.lane_in_use` span or
  the IN_USE sweep, and `LaneLiveness` (the span's refcount).
  `LaneLifecycle.quarantine` requires an injected `quarantine_worktree` callable, which is a `GitOps` method
  using `self.commit`, `self.quarantine_base`, `self.rename_worktree`.
- Pool storage: sentinel primitives live in `lane_lifecycle.py` (`POOL_ROOT_SENTINEL`,
  `pool_storage_present`, `mark_pool_storage_present`); `mark_pool_storage_present` writers 4
  (`_seed_warm_lane`, `_reconcile_pool_storage_before_sweep`, `acquire_spec_lane`, `_acquire_warm_lane_impl`);
  the `exists() and not pool_storage_present()` → `_pool_storage_bootstrap_ok()` block is duplicated in
  `acquire_spec_lane` and `_acquire_warm_lane_impl`; `thin_warm_lane`, `harness.py::_recover_crashed_tasks`
  and `harness.py::_reap_orphan_worktrees` carry no-bootstrap variants; `_run_warm_lane_gc_reclaim` mentions
  the sentinel only in prose.
- Task refs: writers of `refs/heads/task/*` 13 (`git branch` in `create_worktree`; `worktree add -b` in
  `create_interactive_worktree`, `_acquire_warm_lane_impl`, `materialize_member_solo`; `checkout -B` in
  `rebind_branch_to_head`; `update-ref` in `_swap_unlanded_back`, `_swap_ref_forward`; `branch -m` in
  `rename_worktree`, `_rename_branch_aside`; `reset --hard` of a checked-out task branch in
  `acquire_spec_lane`, `_reset_worktree`, `merge_branch_into_worktree`, `rebase_preserving_task_commits`).
  Deleters (`branch -D/-d`, `update-ref -d`): 9 — `git_ops.py` 7, `branch_stack.py` 1,
  `escalation/server.py` 1 (measured at `d40ba048e9`).
  Rebase argv sites 4 (`rebase_onto_main`, `materialize_member_solo`, `advance_main`,
  `branch_stack.py::rebase_own_delta`); rebase entry calls 10, of which 3 pass
  `rebase_preserving_task_commits` (`create_worktree`, `_reuse_warm_lane`, `workflow.py::_inter_iteration_rebase`)
  (`_reuse_warm_lane`'s pass now runs inside task 6219's `_rebase_requeued_onto_main`, after
  `unstack_from_unlanded_base`; measured at `e634cfe658`) and 6 call `rebase_onto_main` directly
  (`_stack_member`, `suffix_graph.py::bounce_conflicting_suffix_items`,
  `merge_lane/worker.py::_do_train_merge`, `workflow.py::_rebase_before_enqueue`,
  `workflow.py::_handle_blocking_dep_report`, `workflow.py::_resolve_and_resubmit`). Inline
  `f'{self.config.branch_prefix}…'` constructions: 30 in 23 scopes. `shared/branch_names.py` exports
  `canonical_queued_branch_name`, `bare_queued_branch_id`; `merge_lane/types.py::QueuedBranch.parse` has 10
  call sites. No module states the invariant "main..task/N is task N's own delta"; the nearest predicate is
  `merge_lane/gates.py::_branch_delta_survives` (per-merge).
- h14: 16,667 lines = 1,490 blank + 6,737 docstring + 2,749 comment + 5,691 code; `class GitOps` 13,685
  lines, 210 methods; 38 module-level classes besides `GitOps`, 6 module-level `Literal` aliases and 32
  module-level functions (AST, measured at `e634cfe658`; 16,439 / 13,519 / 203 / 36 / 28 at `d40ba048e9`);
  32 `orchestrator/src` modules import from `git_ops` (AST `ImportFrom`/`Import`, measured at
  `e634cfe658`; the earlier 34 used a looser rule). The self-state partition and the placement table are in
  §Sketch.
- `.task/`: module docstring still ends "No .task-specific guards remain in this module."; `GitOps.commit`'s
  docstring still says `git add -A` "has nothing under .task/ to stage"; `GitOps.merge_to_main` still removes
  `merge_wt/'.task'` (now via `tree_removal.remove_tree`); `artifacts.py::TaskArtifacts.ensure_lane_symlinks`
  (task 3160) links `<worktree>/.task/plan.json` and `<worktree>/.task/iterations.jsonl` on every dispatch
  (sole caller `workflow.py`; measured at `e634cfe658`); `artifacts.py` line 1
  still reads "Manages the .task/ directory structure in each worktree." while the root is `.task-meta`.
- Artifacts (re-measured at `d40ba048e9`, m9): 13 raw `json.loads` readers besides the seam
  `_read_json_object` — 10 non-plan (`read_lane_task_id`, `read_base_commit`, `update_base_commit`,
  `read_blocking_dependency`, `read_already_done`, `read_ready_to_merge`, `read_unactionable_task`,
  `read_false_premise`, `read_iteration_log`, `read_reviews`) and 3 plan-family (`_read_normalized_plan`,
  `validate_plan_owner`, `read_plan_lock`, S9's); `read_iteration_log` is S9 C2's
  (`TaskArtifacts.read_iteration_entries`), so ν's list is the other 9. Seam users 6 (`read_title` joined;
  measured at `e634cfe658`). Consumers of the five
  architect-exit readers: `workflow.py` 15 sites indexing the dict by string key, `evals/metrics.py` name table.
- fused-memory: `create_subprocess_exec` 10 sites in fused-memory/src; the three in
  `middleware/task_interceptor.py` (`_verify_commit_on_main`, `_resolve_commit_sha`,
  `_resolve_commit_committed_at`) spawn `git -C …` with `wait_for(communicate(), 5.0)`; only the third
  `await proc.wait()`s after `kill()`. `shared.git_async.run_git(cmd, cwd=None, *, input_text=None,
  timeout=None, bounded=True, pin_c_locale=True) -> GitResult(returncode, stdout, stderr, timed_out)`;
  fused-memory already imports it in `services/landed_on_main.py`, `services/live_workflow_detector.py`,
  `services/unlanded_branch_paths.py`. Tests reach the three probes in 2 files, `test_task_interceptor.py`
  and `_fm_git_repos.py` (which took `_fm_helpers.py`'s reference; measured at `e634cfe658`). Task 6236
  (pending, low) asks for an injectable git-runner seam on the same probes as part of moving the
  done-provenance cluster into its own module.
- rerere: `.git/config` sets `rerere.enabled=true` and `rerere.autoupdate=true`; `RECOVERY_GIT` prefixes only
  the aborts; `agents/sandbox_git_config.py` turns rerere off for sandboxed agents (task 5844); the three
  orchestrator rebase primitives run rerere-on. Task 5697 (pending, medium, 14 declared files) is the live
  carrier of exactly this fix — disposition `filed:5697`.

## Sketch of approach

Layering, lowest first (each module imports only downward; no module imports `git_ops.py`):

```
shared.git_async            run_git, GitResult, GitRunner (protocol)                        C1 (α)
orchestrator.git_run        run_in_worktree, WorktreeMissing, GitQueryError (α); query,      C1, C4 (α, ε)
                            GitCommandFailed (ε)
orchestrator.git_diff       FileDelta, LineRange, parse_unified0 — pure parsers              (ζ)
orchestrator.worktree_list  WorktreeEntry, parse_worktree_list — pure parser                 (ζ)
orchestrator.block_disposition  BlockDisposition, RequeueKind (moved from workflow_types)    C3 (δ)
orchestrator.lane_lease     lane_lease, LaneLeaseContended, lane-lock holder helpers         C7 (θ)   [cluster a]
orchestrator.warm_lane_scripts  WarmLaneScripts, SeedOutcome, WarmBaseProbe, WarmBaseHealth  C2 (γ)
orchestrator.git_queries    GitQueries — diff/query reads                                    (ρ)      [cluster f]
orchestrator.lane_lifecycle LaneLifecycle (public owner), LaneRecords                        C6 (ι)
orchestrator.task_branch    TaskBranchRefs — the one writer of refs/heads/task/*             C8 (λ)   [cluster d]
orchestrator.worktrees      WorktreeOps — add/remove/register/ephemeral/interactive/seed     (ξ)      [cluster b]
orchestrator.main_landing   MainLanding — merge worktrees, _move_main, advance/recover, park C5 (η, ο) [cluster e]
orchestrator.git_ops        GitOps — lane acquire/release + composition root                 (κ)      [cluster c]
```

Placement of `git_ops.py`'s module-level types and helpers (measured at `e634cfe658`: 38 classes besides
`GitOps`, 6 `Literal` aliases, 32 functions). B27 forbids a new module importing `git_ops`, so a type moves
*with* the first leaf whose new module needs it, and that leaf repoints every importer of the name (the
enumeration rule at decompose: AST `ImportFrom` of the name from `orchestrator.git_ops` over every `src` and
`tests` tree) — no re-export stays behind in `git_ops.py`. `workflow_types.py::_disposition_table` imports
21 of these names (measured at `e634cfe658`) and is repointed by each moving leaf in turn.

| Name(s) in `git_ops.py` | New home | Leaf | `src` importers the leaf repoints (AST `ImportFrom`, measured at `e634cfe658`; tests enumerated at decompose) |
|---|---|---|---|
| `WorktreeMissing` | `git_run` | α | `workflow_types`, `steward`, `merge_lane/worker`; prose mentions in `task_ground_truth`, `rebase_recovery`, `merge_lane/speculation_controller` |
| `GitQueryFailed` (renamed `GitQueryError`, tasks 7485/7332), `_none_if_query_failed` (→ `none_if_query_failed`) | `git_run` | α | `workflow_types` (its disposition row is re-keyed), `scripts/restamp_degenerate_branch_bases` |
| `_git_query` | deleted (`GitOps._query` over the runner port, β; over `git_run.query`, ε) | β | — |
| `_df_warm_lane_script_dir` (deleted), `_seed_script_text`, `_seed_script_supports*` → `WarmLaneScripts.supports`; `SeedLaneLock`, `WarmBaseHealth` | `warm_lane_scripts` | γ | — (`harness` reaches the probe through `GitOps`, m4) |
| `_seed_rc_to_unavailable` | deleted (`WarmLaneUnavailable.from_seed_rc`) | δ | — |
| `MergeVerifyLeaseHeld`, `MergeVerifyLeaseContended` (→ `LaneLeaseContended`), `LaneLockSelfOwnedLeak`; `_lane_lock_identity` … `_lane_lock_holder_facts` (6 functions), `_HELD_LANE_LOCK_FDS` | `lane_lease` | θ | `workflow_types`, `merge_lane/worker` (the lease handlers) |
| `parse_diff_line_ranges`, `parse_diff_added_line_ranges` | `git_diff` (new names; old names deleted, M15) | ζ | — (callers are inside `git_ops`) |
| `TrainPredecessor`, `TrainMembership`, `TrainStackResult`, `BranchResetError`, `ReplayMatch`, `is_wip_safety_commit`, `_is_train_successor` (task 6219), `_merge_marker_pattern`, `_merge_markers_by_branch` | `task_branch` | λ | `workflow_types`, `workflow` (`BranchResetError`, `TrainMembership`, `is_wip_safety_commit`) |
| `WorktreeKind`, `WorktreeClass`, `RemovalOutcome`, `_utc_stamp` (called by `quarantine_worktree` → ξ and `park_prior_incarnation` → ο; `main_landing` imports it from `worktrees`), `EphemeralWorktreeError`, `InteractiveWorktreeLimitError`, `InteractiveWorktreeInfo`, `ReapedInteractiveWorktree`, `WorktreeConflictError`, `classify_worktree_entry`, `default_protected_prefixes`, `render_protect_glob`, `_worktree_add_failure_is_retryable`, `_git_clean_failure_is_benign` | `worktrees` | ξ0 | `workflow_types`, `verify` (`WorktreeKind`, `EphemeralWorktreeError`), `workflow` (`WorktreeConflictError`), `escalation/server` (`InteractiveWorktreeLimitError`), `cli`, `harness` (`classify_worktree_entry`) |
| `AdvanceOutcome`, `AdvanceResult`, the `AdvanceDetail` variants, `MergeParkError`, `MergeParkLockContentionError`, `MergeParkContentionError`, `MergeResult`, `PushResult`, `RecoverResult`, `PERSISTENT_MERGE_WORKTREE_NAME`, `_merge_subject`, `_merge_worktree_remove_argv`, `_assert_no_conflict_markers` | `main_landing` | ο0 | `workflow_types` (`MergePark*`), `merge_lane/worker`, `merge_lane/types` (`MergeResult`), `merge_lane/liveness` (`PERSISTENT_MERGE_WORKTREE_NAME`) |
| `CommitEffectProbe`, `ConflictProbe`, `RebasedCitation`, `_git_stderr_is_unresolved_ref`, `task_citation_pattern`, `CitationPatternInvalid` (task 4911; with `find_task_citation_commit*`, whose only state is `config` and `project_root`) | `git_queries` | ρ0 | `suffix_graph` (`ConflictProbe`), `agents/landed_sha_searches`, `workflow_types`, `scripts/restamp_degenerate_branch_bases` (`task_citation_pattern`, `CitationPatternInvalid`); the two `orchestrator/tests/merge_lane_ratchet_raises/4911-*.json` entries follow |
| `WarmLaneUnavailable`, `WarmLaneRequeue` and its 8 subclasses, `PoolPrewarmResult`, `WorktreeInfo` | stay in `git_ops` (cluster c) | — | `workflow_types`, `workflow` import from `git_ops` module-level after D4 |

Every check whose symbol this table moves is dir-anchored (`paths: ["orchestrator/src/orchestrator/"]`) in
the sidecar.
The typed-facts leaves land first, each extracting its cluster as it goes (θ takes cluster a, ζ the pure
parsers of f, λ cluster d); the two h14 leaves (ξ, ο) then move clusters b and e below the pool cluster, as
R5 orders it. The self-state partition this follows (AST over `self.<attr>` per method, measured at
`ce0f71a384`; the nine methods added since are placed by the same rule — `lane_in_use` and
`reset_stale_lane_in_use_records` in (c), `_rebase_requeued_onto_main`, `branch_tip_if_no_commits_beyond`
and `find_merge_marker_checked` in (d), `commits_named`, `_commit_paths` and
`find_task_citation_commit_checked` in (f), `thin_warm_lane` in γ's script wrappers): (a) lane locks/leases 7 methods, 612 lines, state `worktree_base`; (b) worktree create/seed/
scripts 22 methods, 2,178 lines, state `project_root`, `config`, `worktree_base`, `_interactive_wt_lock`;
(c) lane acquire/release/pool storage/quarantine 64 methods, 4,034 lines — the only cluster that owns pool
state and the callbacks; (d) branch lifecycle 37 methods, 1,375 lines, state `config`, `project_root`,
`_merge_marker_index*`; (e) merge worktree + advance/recover/park 40 methods, 3,265 lines, state `config`,
`project_root`, `worktree_base`, `_last_*`, `_index_lock_path_cache`; (f) diff/query reads 32 methods, 1,636
lines, state `project_root`, `_effect_probe_memo`. Straddlers: `cleanup_worktree` (c/e), `_freshen_main`
(b/c), `get_rebase_distance` (d/f). Clusters (a), (d), (f) share no mutable state with (c) beyond `config`,
`project_root`, `worktree_base`, which is what makes them extractable without a reach-back (heuristic 13).

Tests move in the same direction: a `ScriptedGitRunner` fake passed through `GitOps(runner=…)` replaces
every `orchestrator.git_ops._run` patch (and `orchestrator/tests/_worktree_add_fakes.py`'s parameterised fake
and task 7485's `PATH` shim `orchestrator/tests/_git_fault_shim.py`, heuristic 11); the 135 test files that
import `_run` as a plain git helper move onto `orchestrator/tests/_git.py::run` in a deterministic codemod
(β0) before β deletes `_run`; `WarmLaneScripts`/`WarmBaseProbe`/`GitOpsLimits`/`GitOpsCallbacks` fakes
replace the 159 private patch sites; `lane_lifecycle` is an optional constructor argument (`None` → `GitOps`
builds one), so the 200 test files that construct `GitOps(` need no edit (all three counts measured at
`e634cfe658`); the callers of methods ξ and ο move (41 and 99 test files, 9 src modules for ο, measured at
`d40ba048e9` by the author's method lists; one wider word-match rule run at both shas grew by 1 and 3 test
files to `e634cfe658`) are repointed by their deterministic codemod leaves ξ0 and ο0, which enumerate at
decompose; nothing in `orchestrator/tests` names a `git_ops._x`.

## Resolved design decisions

- **D1 (R5, verbatim ruling).** `WarmLaneUnavailable` members carry their `WarmLaneRequeue` class and
  `BlockDisposition`; one raise site in `create_worktree`; the `workflow_types` warm-lane rows derive from the
  enum; the 9-arm if-chain, the FAULT literal class list and the `_STEAL_RETRYABLE` placeholder are deleted.
- **D2 (R5; ruling OQ-6).** `shared.git_async.run_git` is the factory-wide runner. R5 says "git_ops keeps only the
  `WorktreeMissing` adaptation"; this PRD keeps that adaptation as a *function in a leaf module*
  (`git_run.run_in_worktree`) rather than a private `git_ops` global — R5's intent rather than its letter,
  recorded as a clarification of R5 — because the worktree-cwd importers need the adaptation too and
  a leaf lets them (and the new S5 modules, B27) take it without importing `git_ops`; a private name with ten
  external importers is the finding, not the fix (heuristic 1).
  `GitOps` reaches the runner through a constructor port `runner: GitRunner = run_git` and every one of its
  spawns (197 `await _run(...)` and 9 `_git_query(...)` calls, measured at `e634cfe658`) goes through
  `self._git(...)` / `self._query(...)` (C1). This is what makes fk-3a4cc6298c02
  closable: a test injects the fake through the public constructor.
- **D3 (R5).** Constructor-injected ports `WarmLaneScripts`, `WarmBaseProbe`, `GitOpsLimits` and a
  `GitOpsCallbacks` bundle (C2) replace the 4 post-construction installs, the 7 monkeypatched module constants,
  the env override, the re-export shim and the `refuse_foreign_band` wrapper. Callbacks default to
  `GitOpsCallbacks.inert()` so `cli.py` and `recover_main.py` construct without a harness; `Harness` passes
  the real bundle. The pools are constructed with their `lane_lifecycle` and drift callback;
  `WarmLanePool.set_lane_lifecycle` and `set_on_lane_record_drift` are deleted.
- **D4 (ruling OQ-1).** `BlockDisposition` and `RequeueKind` move to a new leaf `orchestrator/block_disposition.py`
  (imports only `unblock_types.BlockClass` and `verify_categories.FailureCategory`, neither of which imports
  an orchestrator module), so `git_ops.py` can import the disposition type without a cycle through
  `workflow_types.py`, which imports the nine exception classes from `git_ops.py`. `workflow_types.py` and
  `dry_run_unblock.py`, `workflow.py` and the 6 test files that import the two types import from the leaf;
  no re-export shim. Today `workflow_types.py` imports 21 `git_ops` names *inside* `_disposition_table`, a
  function-local import that is itself the cycle-breaker (a heuristic-13 symptom); D4 removes it — with the two
  types in the leaf, `workflow_types` imports `git_ops` at module level (`git_ops`' import closure does not load
  `workflow_types`, re-probed at `e634cfe658`). The derivation `{m.requeue: m.disposition for m in
  WarmLaneUnavailable if m.requeue is not None}` replaces the nine rows in `workflow_types.py::_disposition_table`;
  S9's `BlockReason` work (R2) is untouched and reads the derived table.
- **D5 (R5).** One `GitQueryError` contract for every `GitOps` read query (C4). A query returns the parsed
  value or raises `GitQueryError`; it never returns the empty value on `rc != 0`. A failed git command raises
  the subclass `GitCommandFailed(argv, cwd, returncode, stderr)` (INV-2); a lookup that composes several
  calls and cannot answer raises the base type with its reason. The type already exists under another name:
  tasks 7485 and 7332 added `git_ops.py::GitQueryFailed` (message-only, 7 raise sites), a raising
  `_git_query` helper and a BLOCK-and-escalate disposition row. α moves it to `git_run.py` under R5's name
  with its row, importers and `_none_if_query_failed`; β makes `_git_query` the `GitOps._query` port
  adapter; ε adds `GitCommandFailed` and `git_run.query` and moves the remaining queries onto them. α also
  rewords the row's `reason_prefix` ('Git lookup on main could not answer') to 'Git query could not answer',
  since ε widens the type to worktree queries (no reader matches the old prefix, measured at `e634cfe658`). The five
  changed-file enumerators return `list[str]` and raise; the `(files, err)` tuple contract is deleted and its
  callers catch. `get_main_sha` raises; each of its 50 awaited call sites (measured at `bb1c659562`) is
  labelled in ε's table `propagate` or `catch GitQueryError → <today's skip/log>`: the 13 sites that guard the
  empty value today become the catch with identical behaviour (so `Harness._run_main_tip_sweep`, the two
  `OfflineLaneWorker` ticks and the merge worker's loop guards keep running), the rest propagate — a failure to
  read main's sha on `project_root` is a stop, not a value (INV-11). The nine "holds work" predicates (task
  7332 added `branch_tip_if_no_commits_beyond`) become two queries that raise: `worktree_work(worktree, *, untracked) -> WorkState` for a checkout (the `untracked` flag
  carries today's per-predicate untracked-file policy, e.g. `has_dirty_working_tree` excludes untracked files)
  and `branch_work(branch) -> BranchWork` for a branch that may have no worktree
  (`_branch_has_commits_beyond_main`, `_orphan_has_commits`, `branch_tip_if_no_commits_beyond`); each former caller states its own direction
  (retain on doubt vs refuse on doubt) at the call site, in ε's predicate table.
- **D6 (R5).** One typed diff reader (`git_diff.parse_unified0 -> list[FileDelta]`, hunk-state
  classification, the only header rule) behind the four parsers, which become projections that live in
  `git_diff.py` under new names — the four `git_ops.py` names are deleted and their callers repointed; one
  `worktree_list.parse_worktree_list -> list[WorktreeEntry]` behind the five porcelain parsers; one
  `GitOps.count_commits(range) -> int` (raises `GitQueryError`) behind the `rev-list --count` sites that ε does
  not already replace (`rebase_preserving_task_commits`' two inline counts and the two predicates' counts are
  ε's), each site's direction stated in ζ's table.
  The three porcelain parsers outside orchestrator (`fused_memory/…/live_workflow_detector.py`,
  `fused_memory/models/scope.py`, `scripts/reclaim_orphaned_worktrees.py`) are noted for S10/S12/S25 in §Cross-PRD;
  a shared parser would live in `shared/`, which is a later decision, not this stream's.
- **D7 (R5).** `_move_main(new_sha, expected_sha, *, gate: MainGate) -> MoveMainResult` owns the main-gate
  hooks (one `_run_main_gate_hook(name)` for `mark`/`unmark`/`bypass`/`bypass_clear`), `rev-parse --verify`
  of both inputs, the CAS `update-ref`, the `index.lock` stand-off and the guarded `read-tree` sync with
  retry; `advance_main` calls it with `MainGate.MARK`, `recover_red_main` with `MainGate.BYPASS` when
  `main_gate_bypass_command` is set, else `MARK` (as today). The two gates clean up differently and C5 states
  both: MARK's sentinel is consumed by the hook on success and unmarked only on failure; BYPASS is cleared in a
  `finally` on every exit path including success, because a bypass left engaged disables the project's non-ff
  guard for every later ref move (`git_ops.py::GitOps.recover_red_main`'s own rationale). `AdvanceOutcome.detail`
  carries what the `_last_*` attributes carried as a frozen union, one variant per detail-bearing result
  (ruling OQ-7; C5); `merge_lane/gates.py::_map_advance_failure` matches on it, and η deletes the 5 `getattr` reads
  in the same leaf as the 7 writes (ruling OQ-8). Task 6892's hook-refusal classification adds a `HookRefused`
  variant after η (edge, not folded).
- **D8 (R5).** One `lane_lease` async context manager in `orchestrator/lane_lease.py` (C7), parameterised by
  the holder rendezvous path and the self-leak probe; `GitOps.merge_verify_lease`, `GitOps.task_verify_lease`,
  `cli.py::_merge_lane_gate` and `cli.py::_verify_merge_body` are thin uses; `cancel_verify`'s identity-gated
  holder cleanup is a function of the same module. The span co-lock stays (ruling OQ-5, B′). The co-lock has
  two purposes (`cli.py::_merge_lane_gate`): (i) closing the lane-release gaps
  between gate, reset and build against a concurrent waiter — which `_verify_merge_body`'s fail-open build hold
  relies on — and (ii) excluding an in-flight old-binary `verify-merge` during the task-2830 rollout; only (ii)
  is over (task 6565 landed 2026-10-20). So the lock stays, renamed the *span lock*; purpose (i) is stated
  once in `lane_lease`'s docstring and the old-binary rationale and its prose go. Making the lane lease
  re-entrant within one span (A′) needs the self-owned-leak probe redesigned first; it is not θ's work but
  the low-priority follow-up σ, filed with this plan and run after θ (ruling OQ-5). The lock-lifetime rationale (split lifetime around `reset_persistent_merge_worktree`, the
  span lock, B12 fd invariant) is written once, in `lane_lease`'s docstring, and the ~215 comment lines across
  the three copies go. Cluster (a) — the 6 module-level lane-lock helpers and `_HELD_LANE_LOCK_FDS` — moves with it.
- **D9 (R5).** `LaneLifecycle` is the single public owner of durable lane records (C6): constructed by the
  composition root without a quarantine callable (the callable was the cycle), passed to `GitOps` and
  `WarmLanePool` through their constructors (an optional parameter: `None` means `GitOps` builds
  `LaneLifecycle(worktree_base)` and hands it to its pools, so the 200 test files that construct `GitOps(` and
  the 19 that construct `LaneLifecycle(`/`WarmLanePool(` need no edit, measured at `e634cfe658`; every
  orchestrator composition root passes one explicitly), and exposing `release(lane)` (the pool's guard, now the one
  copy), `quarantine(lane, branch, reason, *, relocate)` (the git relocation passed per call by
  `GitOps.quarantine_lane`), `read` that raises `CorruptLaneRecord` and `all_records -> LaneRecords(records,
  corrupt)`. Harness recovery's three `.transition` writes (now in `Harness._recover_worktree_entry`, task
  8267) become `release`/`quarantine_lane`; its `None`
  fallback is reached only for a genuinely absent record; `task_runtime` and `stranded_verified_green` take a
  `LaneLifecycle` argument. `warm_lane_pool.LaneState` is renamed `PoolSlotState`; `DurableLaneState` aliases
  go. Pool assignment gets one mutating module: `GitOps.adopt_lane(lane, task_id)` is
  `pool.restore_assignment(task_id, lane)` only — the pool's `WarmLanePool._note_assigned_durable` is the one
  durable mirror and fails open on conflict (it swallows `IllegalLaneTransition` into the drift counter),
  whereas a second `LaneLifecycle.note_assigned` would raise for a record naming another task and abort
  `Harness._reconcile_lane_checkouts`' deliberate re-pins mid-recovery; harness's 6 `pool.restore_assignment` calls
  (`Harness._recover_worktree_entry` ×1, `Harness._reconcile_lane_checkouts` ×1,
  `Harness._reconcile_one_lane_checkout` ×4, measured at `e634cfe658`) and
  `GitOps._settle_branch_checkout`'s 2 call it, and no module outside `git_ops`/`warm_lane_pool` calls
  `restore_assignment` — the residual fk-31e9856ed662's re-verification names is closed, not declared API.
- **D10.** `_acquire_warm_lane_impl` becomes a classifier plus one method per `AcquireRoute` member behind a
  dispatch table (`_ACQUIRE_ROUTES: Mapping[AcquireRoute, Callable]`), the identity-mismatch RECYCLE path calls
  `_reset_and_seed_recycled_lane`, and `_ensure_pool_storage(context) -> bool` (the generalisation of
  `_reconcile_pool_storage_before_sweep`) is the one bootstrap-or-refuse guard for `acquire_spec_lane`, the
  warm acquire, `thin_warm_lane` and the two harness sweeps. `mark_pool_storage_present` then has two
  writers by design — the guard's bootstrap arm and `_seed_warm_lane`'s rc=0 — and the docstring says so.
  The REUSE route keeps task 6219's un-stack (`_rebase_requeued_onto_main`).
- **D11 (rulings OQ-2, OQ-3).** `orchestrator/task_branch.py::TaskBranchRefs` is the one writer of non-delete
  `refs/heads/task/*` moves and the one deleter (C8): create, rebase, reset, rename-aside, swap, delete, each
  write followed by the own-delta post-condition (`own_delta(branch)` commit count preserved), raising
  `BranchResetError` with the values (INV-2). The rebase target is a closed sum, `onto: OntoRef = MainTip |
  StackedOn(TrainPredecessor) | FrozenPrefix(staging_sha, main_sha)`, each variant with its own validation
  (task 4752's guard at the one site): `MainTip` is main; `StackedOn` names a train predecessor whose tip is
  itself on or behind main or a verified train member; `FrozenPrefix` — the suffix-conflict bounce's
  speculative staging commit (`suffix_graph.py::SuffixConflictTracker.bounce_conflicting_suffix_items`) — is
  validated by `TaskBranchRefs.rebase` by ancestry only: `main_sha` must be an ancestor of `staging_sha`. That
  `staging_sha` is the lane's newest frozen merge commit holds by construction: the only producer of a
  `FrozenPrefix` is `bounce_conflicting_suffix_items`, which builds it from the worker's `frozen_prefix_tip`,
  and B24's AST tripwire asserts no other construction site. `frozen_prefix_tip -> str` is unchanged
  everywhere (`SuffixConflictTracker.recompute`, the worker's §5.3 base guard and
  `merge_lane/intake_buffers.py`'s `Callable[[str], str]` keep reading a sha); when the frozen prefix is empty
  (`SpeculativeMergeWorker.frozen_prefix_tip` returns `main_sha`) the caller passes `MainTip` —
  `bounce_conflicting_suffix_items` builds the target locally: `onto = MainTip() if frozen_tip == main_sha else
  FrozenPrefix(frozen_tip, main_sha)`. λ edits only `suffix_graph.py` for this. Under ruling OQ-3 (A′) λ also
  carries 4752's remaining scope: clearing a superseded `metadata['train']` and validating
  `git_ops.py::GitOps._resolve_predecessor_tip`'s resolution. The 13 writer sites, 9 deleters (git_ops 7,
  `branch_stack.py` 1, `escalation/server.py` 1) and 10 rebase entry calls route through it, including task
  6219's `_rebase_requeued_onto_main`, whose un-stack runs before the owner's rebase. `QueuedBranch`
  moves to `shared.branch_names` (its `parse` already delegates there; `escalation` imports the same module) so
  `git_ops` accepts a `QueuedBranch` without importing `merge_lane.types`; λ0 repoints every importer (7 `src`
  modules + the `merge_lane/__init__.py` lazy table + 90 test files, measured at `bb1c659562`) and leaves no
  re-export in `merge_lane/types.py` (a re-export would be a heuristic-13 symptom). The 30 inline prefix
  constructions and `resolve_queued_branch_ref`'s try-both go. Task 5144's admission-time citation
  guard stays as the independent second check (heuristic 10, redundantly enforced) and is named by λ.
- **D12.** fused-memory's three done-provenance probes are rebuilt on `run_git(timeout=…)` with
  `GitResult.timed_out` and spawn `OSError` mapped onto the existing `confirmed_absent` /
  `confirmed_off_main=False` / `None` branches; the runner is a `git_runner` parameter threaded from
  `TaskInterceptor.__init__` (default `run_git`) into the three module functions, so tests pass a fake through
  the constructor instead of patching three dotted names (a module-level injectable would still be a
  dotted-name patch seam). The seven other `create_subprocess_exec` sites in fused-memory are outside the
  finding and this stream (§Cross-PRD).
- **D13.** `TaskArtifacts`: one `_read_model(name, cls)` over `_read_json_object`; five frozen
  `ArchitectExit` dataclasses (`BlockingDependencyExit`, `AlreadyDoneExit`, `ReadyToMergeExit`,
  `UnactionableTaskExit`, `FalsePremiseExit`) with `from_mapping`, as `AgentSession` does; the nine non-plan
  readers (`read_lane_task_id`, `read_base_commit`, `update_base_commit`, the five architect-exit readers,
  `read_reviews`) go through the seam; `read_iteration_log` is S9 C2's (`TaskArtifacts.read_iteration_entries`)
  and untouched here; the plan family
  (`read_plan`, `_read_normalized_plan`, `read_plan_lock`, `validate_plan_owner`, `lock_plan`) is S9's (R2)
  and untouched; the module docstring states the `.task-meta` root. `workflow.py`'s 15 consumer sites read
  typed fields.
- **D14.** The h14 split follows the measured partition after the primitives move below the pool cluster
  (R5): ξ extracts cluster (b) into `orchestrator/worktrees.py::WorktreeOps` (add/remove/registration
  probes/ephemeral/interactive/quarantine relocation/rename; the seed and script wrappers are γ's
  `WarmLaneScripts`, not ξ's), ο extracts cluster (e) into `orchestrator/main_landing.py::MainLanding` (merge
  worktrees and the merge-worktree factory, `_move_main`, `advance_main`, `recover_red_main`, park/restore,
  interactive reap), and ρ extracts cluster (f) (ruling OQ-9), the diff/query reads (32
  methods, 1,636 lines, state `project_root`, `_effect_probe_memo`), into
  `orchestrator/git_queries.py::GitQueries`. `GitOps` keeps cluster (c) and composes the rest
  (`self.worktrees`, `self.landing`, `self.branches`, `self.queries`); callers that need a primitive call the
  collaborator (`git_ops.worktrees.ephemeral(...)`), never a delegating method on `GitOps` — a 200-method façade
  would be the stitching heuristic 14 forbids. Each move is two leaves: a deterministic codemod (ξ0, ο0) that
  creates the new module with the moved *types* and a collaborator whose methods, for exactly one chain step,
  call the not-yet-moved `GitOps` bodies through the reference it was constructed with (typed by a local
  Protocol, so no import of `git_ops`), and repoints every caller and test (41 test files for ξ, 99 test files
  and 9 `src` modules for ο, measured at `d40ba048e9`; a few more at `e634cfe658`, §Sketch; enumerated at
  decompose); then the move leaf (ξ, ο) moves the bodies and deletes
  the `GitOps` methods with no caller edit. The expected residual is stated in §Goal. Each new file passes heuristic 13: imports only downward, no import of
  `git_ops`. Module docstrings state purpose, not history; the retired-guard narrative, the "Task 4930 rewrote
  this message" note and the 4930 FAULT prose leave the code (Comments stance); the `.task/` facts are stated
  truthfully in one place (the hot path links `<worktree>/.task/plan.json` and `iterations.jsonl` through
  `TaskArtifacts.ensure_lane_symlinks`, root `.gitignore` covers `.task/`, `merge_to_main` keeps one
  belt-and-braces removal).
- **D15.** rerere is not re-decided here: task 5697 is the live carrier (filed:5697); λ depends on it so
  the branch owner's rebase runs rerere-off from the day it exists.

## Contracts (B+H)

### C1 — the factory-wide git runner (α, β)

```python
# shared/src/shared/git_async.py  (additions; run_git's semantics unchanged)
class GitRunner(Protocol):
    async def __call__(self, cmd: Sequence[str], cwd: Path | str | None = None, *,
                       input_text: str | None = None, timeout: float | None = None,
                       bounded: bool = True, pin_c_locale: bool = True) -> GitResult: ...
run_git: GitRunner   # the production implementation

# orchestrator/src/orchestrator/git_run.py  (new leaf; imports shared.git_async only)
class WorktreeMissing(FileNotFoundError): path: Path            # moved from git_ops.py (α)
class GitQueryError(Exception):                                 # α: git_ops.GitQueryFailed (tasks 7485, 7332), renamed
    # a read-only git lookup could not answer, as distinct from answering "none"; str(exc) says why
class GitCommandFailed(GitQueryError):                          # ε
    argv: tuple[str, ...]; cwd: Path | None; returncode: int; stderr: str
    # str(exc) names all four (INV-2 structured-facts-at-failure)
async def none_if_query_failed(lookup: Awaitable[str | None]) -> str | None   # α: git_ops._none_if_query_failed
async def run_in_worktree(cmd, worktree: Path | str | None, *, runner: GitRunner = run_git,
                          input_text: str | None = None) -> GitResult
    # the WorktreeMissing pre-flight + post-spawn re-classification git_ops._run carries today;
    # bounded=False (operator scripts and oracles share this path); timeout is the caller's wait_for
async def query(cmd, cwd, *, runner: GitRunner = run_git, input_text=None) -> str          # ε
    # run_in_worktree, then raise GitCommandFailed on rc != 0; returns stdout (git_ops._git_query's job)
```

α's per-importer table (the 13 orchestrator sites; measured at `d40ba048e9`). The runner is chosen per call
site by its cwd — a worktree cwd takes `run_in_worktree` (it needs the `WorktreeMissing` adaptation), a
`project_root` cwd takes `run_git` — and every repointed site keeps today's `bounded=False`, which is what
`git_ops.py::_run` passes ("Deliberately UNBOUNDED": it runs operator scripts and oracles, pinned by
`test_a_long_running_script_cannot_delay_a_concurrent_git_call`); α changes no runner semantics.

| Importer (`module::symbol`) | Runner | `bounded` |
|---|---|---|
| `workflow.py` (module-level multi-line import) | `run_in_worktree` / `run_git` by cwd | `False` |
| `merge_lane/worker.py` (module-level multi-line import) | `run_in_worktree` / `run_git` by cwd | `False` |
| `merge_lane/disposition.py` (module-level) | `run_git` / `run_in_worktree` by cwd | `False` |
| `merge_lane/landing_evidence.py` (module-level) | `run_git` / `run_in_worktree` by cwd | `False` |
| `merge_lane/gates.py` (module-level; its `_INDEX_LOCK_STALE_FLOOR_S` import moves to `GitOpsLimits` in γ, m7) | `run_in_worktree` / `run_git` by cwd | `False` |
| `merge_lane/skew_tripwire.py::_run_load_bearing_oracle` (function-local; operator oracle) | `run_in_worktree` | `False` — must stay unbounded |
| `verify.py::_derive_task_files_from_git` (function-local) | `run_in_worktree` | `False` |
| `verify.py::_verify_pipeline_guard_requires_full_gate` (function-local) | `run_in_worktree` | `False` |
| `verify.py::confirm_main_tip_failure_is_real` (function-local) | `run_in_worktree` | `False` |
| `scheduler.py::Scheduler._resolve_main_sha` (attribute call) | `run_git` (`project_root`) | `False` |
| `delivered_checks.py::run_delivered_check` and its two siblings (3 `runner` defaults; script and grep checks) | `run_in_worktree`-shaped default, `_Runner` retyped to `GitRunner` | `False` — script checks must stay unbounded |
| `escalation/src/escalation/server.py::release_warm_worktree` (function-local) | `run_git` — S3 δ's edit, not α's (all three calls run at `cwd=gops.project_root`, so no `WorktreeMissing` case) | — |

Invariants: no module outside `git_ops.py` imports a private `_run` (there is none); `GitOps.__init__` takes
`runner: GitRunner = run_git` and every subprocess it spawns goes through `self._git(cmd, cwd) ->
tuple[int, str, str]` (adapter over `run_in_worktree(..., runner=self._runner)`) or `self._query(cmd, cwd)
-> str`; `delivered_checks.run_delivered_check`'s `runner` defaults to `run_git`-shaped `GitResult`, its
`_Runner` alias retyped; tests build `GitOps(runner=ScriptedGitRunner(...))`
(`orchestrator/tests/_git_runner_fakes.py`: a fake that matches argv prefixes to scripted `GitResult`s and
delegates unmatched calls to the real `run_git`, recording every call; it replaces both
`orchestrator/tests/_worktree_add_fakes.py`'s fake `_run` and task 7485's `PATH` shim
`orchestrator/tests/_git_fault_shim.py`). Error semantics: `WorktreeMissing` for a vanished cwd,
`FileNotFoundError` for a missing binary, `GitCommandFailed` only from `query`, and `GitQueryError` from a
lookup that composes several calls (today's 4 such raises in `GitOps.commits_named` and
`GitOps.resolve_rebased_citation`, measured at `e634cfe658`).

### C2 — GitOps construction ports (γ)

```python
# orchestrator/src/orchestrator/warm_lane_scripts.py  (new; imports git_run, config)
@dataclass(frozen=True) class SeedOutcome: returncode: int; stdout: str; stderr: str; script: Path
class WarmLaneScripts:
    def __init__(self, project_root: Path, config: GitConfig, runner: GitRunner, *,
                 script_dir: Path | None = None)       # replaces the ORCH_WARM_LANE_SCRIPT_DIR env override
    def resolve(self, name: str) -> Path | None          # today's _warm_lane_script_candidates/_resolve_warm_lane_script*
    def supports(self, script: Path, flag: str) -> bool  # today's _seed_script_supports (stat-keyed text probe) — the one capability check
    async def seed(self, lane, *flags) -> SeedOutcome; disk_guard(...); soft_guard(...); audit(...);
    async def gc_reclaim(...); thin(...); degenerate_ref_check(...); refresh_base(...); provision_debug_port(...)

WarmBaseProbe = Callable[[], WarmBaseHealth]   # sync, no arguments — today's GitOps._warm_lane_base_resolvable(self),
                                                # which harness.py::Harness._probe_warm_base_health calls without await

@dataclass(frozen=True) class GitOpsLimits:   # defaults = today's module constants, verbatim
    reset_warm_lane_lock_wait_secs: float; lane_lock_holder_settle_secs: float; merge_verify_lease_wait_secs: float
    seed_warm_lane_lock_wait_secs: float; effect_probe_memo_max_entries: int; merge_marker_index_recheck_secs: float
    warm_lane_audit_cache_ttl_secs: float
    index_lock_stale_floor_s: float     # today's _INDEX_LOCK_STALE_FLOOR_S, which merge_lane/gates.py imports (m7)

@dataclass(frozen=True) class GitOpsCallbacks:
    on_pool_storage_absent: Callable[..., None] | None = None          # signatures = today's attribute types
    on_structural_exhaustion: Callable[..., None] | None = None
    warm_lane_reclaim_candidate_provider: Callable[..., Any] | None = None
    warm_lane_dispatched_predicate: Callable[..., bool] | None = None
    on_lane_record_drift: Callable[[int], None] | None = None
    @classmethod
    def inert(cls) -> GitOpsCallbacks: ...

GitOps.__init__(self, config: GitConfig, project_root: Path, *, runner: GitRunner = run_git,
                lane_lifecycle: LaneLifecycle | None = None,        # C6 (ι); None → GitOps builds
                                                                    # LaneLifecycle(worktree_base) and passes it to its pools
                scripts: WarmLaneScripts | None = None,              # None → WarmLaneScripts(project_root, config, runner)
                base_probe: WarmBaseProbe | None = None,
                limits: GitOpsLimits = GitOpsLimits(),
                callbacks: GitOpsCallbacks = GitOpsCallbacks.inert(),
                warm_lane_pool_size: int = 0, merge_spec_warm_lane_pool_size: int = 0,
                transcript_archive: TranscriptArchiveConfig | None = None)
```

Invariants: after `__init__` returns, no attribute of a `GitOps` is assigned from another module (checkable:
`git grep -nE 'git_ops\.[a-zA-Z_]+ *= [^=]' -- orchestrator/src/orchestrator/ escalation/src/escalation/` empty;
4 matches today, all in `Harness.__init__`, measured at `d40ba048e9`); the pools are built inside
`__init__` with `lane_lifecycle=` and `on_lane_record_drift=callbacks.on_lane_record_drift`; the module-level
constants named in `GitOpsLimits` no longer exist in `git_ops.py`. The real `GitOpsCallbacks` is built by the
harness (seam row): γ performs the minimal wiring edit at the existing `Harness.__init__` construction site
(ruling OQ-4), and S8 β later re-homes that construction behind a same-file edge on γ.

### C3 — the warm-lane unavailability table (δ)

```python
# orchestrator/src/orchestrator/block_disposition.py  (new leaf; imports unblock_types, verify_categories)
class RequeueKind(Enum): ...                          # moved verbatim from workflow_types
@dataclass(frozen=True) class BlockDisposition: ...   # moved verbatim from workflow_types (6 fields)

# git_ops.py
@enum.unique
class WarmLaneUnavailable(Enum):
    # value stays today's string (GitOps.acquire_warm_lane logs result.value / sentinel.value); the data rides on
    # each member as attributes set in __new__ — requeue: type[WarmLaneRequeue] | None, disposition:
    # BlockDisposition | None, steal_retryable: bool, seed_rcs: frozenset[int], summary: str
    EXHAUSTED = ('exhausted', WarmLanePoolExhausted, <row>, False, frozenset(), '...')
    ...  # one member per cause; FAULT and DISABLED carry requeue=None (RuntimeError, as today)
    @classmethod
    def from_seed_rc(cls, rc: int) -> WarmLaneUnavailable   # 75→DISK_PRESSURE, 76→BASE_ABSENT, 124→LANE_LOCK_TIMEOUT,
                                                            # 77→LANE_LOCK_CONTENDED, 127→FAULT(script absent), else FAULT
    def raise_for(self, branch_name: str, detail: str) -> NoReturn   # THE raise site; builds the FAULT message
                                                                     # from the table (names every member with a requeue)
    @classmethod
    def steal_retryable(cls) -> frozenset[WarmLaneUnavailable]      # replaces _STEAL_RETRYABLE

# workflow_types.py::_disposition_table
rows |= {m.requeue: m.disposition for m in WarmLaneUnavailable if m.requeue is not None}
rows[WarmLaneRequeue] = <base row>   # unchanged fallback
```

Invariants (import-time asserts in `git_ops.py` + a test): member values are pairwise distinct strings
(`@enum.unique`; a tuple value would log a dataclass repr and two equal tuples would silently alias); every
member has `requeue is not None` xor is in `{FAULT, DISABLED}`; every concrete subclass of `WarmLaneRequeue`
is exactly one member's `requeue`; the members' `seed_rcs` are pairwise disjoint; a member with `requeue` has a
`disposition`. `_seed_rc_to_unavailable`
and the 9 inline rc checks are deleted; `SeedOutcome.returncode` is mapped once by `from_seed_rc`.
S9 (`workflow_types` rows derive) is served by the derivation above landing in δ.

### C4 — the query contract (ε)

```python
# every GitOps read query: returns the parsed value or raises GitQueryError (never the empty value on rc != 0);
# a failed command raises its subclass GitCommandFailed
async def get_changed_files(self, base, head) -> list[str]            # was [] + WARNING
async def get_commit_changed_files(...) -> list[str]                  # was silent []
async def get_files_touched_in_branch(...) -> list[str]               # was [] + WARNING
async def get_merge_diff_files(...) -> list[str]                      # was (files, err)
async def get_branch_changed_files(...) -> list[str]                  # was (files, err); unresolved ref → GitCommandFailed with that stderr
async def get_changed_line_ranges(...) / get_new_side_changed_line_ranges(...) -> dict[str, list[LineRange]]
async def get_diff_from_base(...) -> str                              # get_diff_from_main is gone (task 5695)
async def get_current_branch(...) -> str; get_main_sha(self) -> str   # raises; never ''
# unmerged paths: GitOps.unmerged_paths (task 7302) is the one definition; _detect_unmerged_paths deleted (task 7843)

@dataclass(frozen=True) class WorkState:
    head: str; commits_beyond_main: int; dirty_paths: tuple[str, ...]
async def worktree_work(self, worktree: Path, *, untracked: bool) -> WorkState   # raises GitQueryError / WorktreeMissing
    # untracked=False reproduces has_dirty_working_tree's project_root policy (untracked files excluded)
@dataclass(frozen=True) class BranchWork:
    exists: bool; tip: str | None; commits_beyond_main: int          # tip is None iff not exists
async def branch_work(self, branch: QueuedBranch) -> BranchWork      # raises GitQueryError; a branch may have no worktree
# worktree_work replaces worktree_has_unsaved_work(worktree, branch), _worktree_dirty, has_uncommitted_work,
# has_dirty_working_tree, worktree_head_beyond_main and the inline counts in rebase_preserving_task_commits;
# branch_work replaces _branch_has_commits_beyond_main(full_branch), _orphan_has_commits(full_branch) (an
# orphan may have no worktree) and branch_tip_if_no_commits_beyond(full_branch, main) (task 7332; its one
# caller passes main); the caller chooses its direction on GitQueryError, explicitly (ε's table)
```

Error semantics: `GitCommandFailed` carries argv, cwd, rc, stderr; a composed lookup's `GitQueryError`
carries its reason; `WorktreeMissing` is unchanged; the fail-safe twins (`find_merge_marker`,
`find_task_citation_commit`) keep reading `GitQueryError` as "no answer" through `none_if_query_failed`. Consumer
obligations: `suffix_graph.py::recompute`'s `except` branch becomes live for `GitQueryError` and yields the
"unknown footprint" it was written for; `workflow.py::_plan`'s Lever-B skip and `_inter_iteration_rebase`
propagate (a failed diff blocks with the error text, it does not skip revalidation); the two
`select_train_members` feeders exclude a candidate whose ranges could not be read rather than passing `{}`;
`harness.py::_file_dirty_tree_escalation` leaves the L2 pending on a failed `status`. Retention-direction
callers of the old predicates (`_cleanup_leftover_branch`, `_delete_branch_if_on_main`, `reap_interactive_worktrees`,
`harness.py::_reap_orphan_worktrees`, `_revert_in_progress_if_no_live_claimant`,
`workflow.py::TaskWorkflow._restamp_branch_base_if_no_own_commits`) treat `GitQueryError` as
"has work" (retain); refusal-direction callers (`merge_to_main`, `classify_and_merge`, `read_branch_state`,
`_commit_amendment_wip`, `_iteration_commit_provenance`, `task_verify_dispatch._run_remote`) propagate.

### C5 — `AdvanceOutcome.detail` and `_move_main` (η)

```python
# ruling OQ-7: a frozen union, one variant per detail-bearing result; AdvanceResult's
# 12 literals are unchanged (task 5298 requires them)
@dataclass(frozen=True) class ParkLockInfo: lock_path: Path; age_seconds: float; waited_seconds: float
@dataclass(frozen=True) class WipOverlap: overlap_files: tuple[str, ...]                 # 'wip_overlap'
@dataclass(frozen=True) class PopConflict: recovery_branch: str; read_tree_synced: bool = True
                                                                  # 'pop_conflict' (advances main), 'pop_conflict_no_advance'
@dataclass(frozen=True) class StashFailed: stash_dirty_files: tuple[str, ...]            # 'stash_failed'
@dataclass(frozen=True) class ParkLockContended: park_lock: ParkLockInfo                 # 'park_lock_contended'
@dataclass(frozen=True) class NoDetail: read_tree_synced: bool = True                    # every other result
AdvanceDetail = WipOverlap | PopConflict | StashFailed | ParkLockContended | NoDetail
# task 6892 adds a HookRefused variant after η; the result→variant pairing is re-read against advance_main's
# 7 _last_* writes at implementation time (each write site becomes the variant its result carries)
@dataclass(frozen=True) class AdvanceOutcome:
    result: AdvanceResult; advanced_sha: str | None = None; rebased_from: str | None = None
    rebased_onto: str | None = None; detail: AdvanceDetail = NoDetail()
    # __post_init__ asserts the result/variant pairing above (a 'wip_overlap' outcome carries WipOverlap, …)

class MainGate(Enum): MARK; BYPASS
@dataclass(frozen=True) class MoveMainResult: moved: bool; cas_lost: bool; hook_refusal: str | None; read_tree_synced: bool
async def _move_main(self, new_sha: str, expected_sha: str | None, *, gate: MainGate) -> MoveMainResult
    # rev-parse --verify both inputs; _await_index_lock_clear; update-ref CAS; guarded read-tree -u --reset HEAD
    # with one retry after a second stand-off when project_root has main checked out. Gate hooks:
    # MainGate.MARK: run main_gate_mark_command before the CAS. On CAS failure or an exception, run
    #   main_gate_unmark_command (best-effort). On success run nothing: the hook consumes the sentinel.
    # MainGate.BYPASS: set engaged before awaiting main_gate_bypass_command. Run main_gate_bypass_clear_command
    #   in a finally on every exit path, including success and an exception raised mid-engage. If the bypass is
    #   engaged with no clear command, log ERROR naming the disabled guard (unreachable under
    #   config.py::GitConfig._reject_bypass_command_without_clear).
    # recover_red_main uses BYPASS when main_gate_bypass_command is set, else MARK.
```

Invariants: `GitOps` has no `_last_*` attribute; `merge_lane/gates.py::_map_advance_failure` matches on
`outcome.detail` (exhaustive over the union, so a missing fact is a pyright error) and never
`getattr(git_ops, …)`; η deletes the 7 writes and the 5 reads together (ruling OQ-8), so no window exists in which
`getattr(…, None)` silently drops overlap and recovery facts from a failure message (an INV-11 regression); `advance_main` and `recover_red_main` contain no `sh -c
main_gate_*`, `update-ref refs/heads/<main>` or `read-tree` argv of their own. S6 consumes `AdvanceOutcome.detail`
(R1: its "typed facts first" leaf depends on this contract — seam row).

### C6 — `LaneLifecycle` as the public owner (ι)

```python
class LaneLifecycle:
    def __init__(self, worktree_base: Path, *, escalation_queue=None)       # no quarantine callable
    def read(self, lane) -> LaneRecord | None          # None ONLY when no record file exists; corrupt → raises CorruptLaneRecord
    def all_records(self) -> LaneRecords              # LaneRecords(records: Mapping[str, LaneRecord], corrupt: tuple[str, ...])
    def release(self, lane) -> LaneRecord | None       # ASSIGNED|IN_USE → RELEASED; any other state → None, no write, DEBUG
    async def quarantine(self, lane, branch, reason, *, relocate: Callable[[Path, str, str], Awaitable[Path | None]]) -> Path | None
    def note_assigned(...); refresh_assigned(...); transition(...)   # unchanged; transition stays public for SEED/REGISTERED
GitOps.quarantine_lane(self, lane, branch, reason) -> Path | None     # lifecycle.quarantine(..., relocate=self.quarantine_worktree)
GitOps.adopt_lane(self, lane: str, task_id: str) -> None              # the one mutator of pool assignment outside the pool:
                                                                      # pool.restore_assignment(task_id, lane) only — the pool's
                                                                      # _note_assigned_durable is the one durable mirror; it fails
                                                                      # open on conflict
WarmLanePool.__init__(self, worktree_base, size, *, name_prefix='_lane-', lane_lifecycle: LaneLifecycle | None = None,
                      drift_l2_threshold=3, on_lane_record_drift: Callable[[int], None] | None = None)   # set_* deleted
warm_lane_pool.PoolSlotState   # FREE / ASSIGNED (renamed from LaneState)
```

Invariants: every orchestrator composition root (`Harness.__init__`, `cli.py` ×2, `recover_main.py`,
`evals/runner.py::_cell_workflow`) constructs a `LaneLifecycle` and passes one explicitly (the `None` default
exists for the 200 test files that construct `GitOps(`, measured at `e634cfe658`, and for the one-shot
operator script `scripts/restamp_degenerate_branch_bases.py`, task 7332; B6 asserts `Harness` passes one);
once `read` raises `CorruptLaneRecord`, `LaneLifecycle._read_best_effort` catches it beside `OSError`, so an
unreadable record still never aborts a `GitOps.lane_in_use` span or the IN_USE sweep (task 3076's guarantee); `restore_assignment` is called only by `GitOps.adopt_lane` and inside `warm_lane_pool.py`;
`grep -rn "_lane_lifecycle" orchestrator/src` matches only the two owners'
own fields (`GitOps`, `WarmLanePool`); `.transition(` to RELEASED or QUARANTINED is called only inside
`lane_lifecycle.py`; `task_runtime._build_pooled_snapshot` and `stranded_verified_green.detect_verified_green`
take `lane_lifecycle: LaneLifecycle`; one `LaneState` name in `orchestrator/src`. Pool and harness go through
`LaneLifecycle` (seam row; S8 receives it).

### C7 — the lane lease (θ)

```python
# orchestrator/src/orchestrator/lane_lease.py  (new; imports verify_cancel primitives, shared)
class LaneLeaseContended(Exception): lock_path: Path; waited_secs: float; holder_facts: ...   # MergeVerifyLeaseContended renamed/moved
class LaneLockSelfOwnedLeak(LaneLeaseContended): ...                                        # moved
@asynccontextmanager
async def lane_lease(lock_path: Path, *, wait_secs: float, holder_rendezvous: Path | None,
                     self_leak_probe: Callable[..., LaneLockSelfOwnedLeak | None] | None = None) -> AsyncIterator[None]
    # acquire off-thread with bounded wait → raise (leak subclass first); write holder pgid iff holder_rendezvous;
    # remove in finally; release fd in finally (the B12 orphaned-fd invariant)
def release_holder_if(worktree_base: Path, pgid: int) -> bool          # cancel_verify's identity-gated cleanup
@contextmanager
def span_lock(worktree_base: Path) -> Iterator[int | None]             # ruling OQ-5 (B′): today's compat co-lock, renamed;
    # today's order: acquire the lane gate; acquire span_lock while still holding the lane lock; then release the
    # lane lock; held to the end of the build so the gate→reset→build gaps stay closed against a concurrent
    # verify-merge span on the same host and lane (purpose (i)). If the span-lock wait times out, emit the
    # contention result and release the lane lock (cli.py::_merge_lane_gate arm (2)). The lock file stays
    # <worktree_base>/.merge_verify.lock — only the Python names change, which keeps θ's own rollout excluding
    # pre-θ binaries; the old-binary rationale (ii) prose is gone
# the 6 module-level lane-lock helpers and HELD_LANE_LOCK_FDS move here from git_ops.py
```

Uses: `GitOps.merge_verify_lease(lane_dir=None)` = `lane_lease(lane_lock_path(...),
wait_secs=limits.merge_verify_lease_wait_secs, holder_rendezvous=worktree_base if persistent else None,
self_leak_probe=...)`; `GitOps.task_verify_lease` = `lane_lease(..., holder_rendezvous=None)`;
`cli.py::_merge_lane_gate` = the gate form (acquire-then-release, no holder) and `_verify_merge_body`'s build
hold = `lane_lease(..., holder_rendezvous=worktree_base)`. Order, as today: the gate acquires the lane
lock, takes `span_lock` while still holding it, then releases the lane lock; on a span-lock wait timeout it
emits the contention result and releases the lane lock. The lock file stays `<worktree_base>/.merge_verify.lock`
(only Python names change). Invariants: the span lock is held continuously from before the gate's release until
the build hold is released, and
`_verify_merge_body`'s fail-open build hold stays sound because of it (B18b); `lane_lease`'s docstring is the
one statement of the lock lifetime; the merge worker's handler sites (3 `except` sites over
`MergeVerifyLeaseContended`/`MergeVerifyLeaseHeld` in `merge_lane/worker.py`, measured at `d40ba048e9`) and
`workflow_types.py::_disposition_table` catch/import the renamed classes from `lane_lease`.

The follow-up σ (ruling OQ-5, A′) is the re-entrant lease's end state: `lane_lease` yields a `LaneLeaseToken`
that `reset_persistent_merge_worktree` accepts to skip its own acquire, the CLI holds the lane lock
continuously from gate to build, and the span lock and its `.merge_verify.lock` file go. That needs the
self-owned-leak probe (`git_ops.py::_lane_lock_held_in_process`, which θ moves into `lane_lease` with the
other lane-lock helpers, and `GitOps._lane_lock_self_owned_leak`, which θ passes as `self_leak_probe`), which
today treats any in-process holder as a leak, redesigned first so it can tell a token holder from a leak. σ keeps B18b's postcondition and drops only the mechanism.

### C8 — the task-branch owner (λ)

```python
# shared/src/shared/branch_names.py: QueuedBranch (moved from merge_lane/types.py; parse(raw, prefix) unchanged)
# orchestrator/src/orchestrator/task_branch.py  (new; imports git_run, shared.branch_names, config, rebase_recovery)
@dataclass(frozen=True) class OwnDelta: base: str; commits: tuple[str, ...]
@dataclass(frozen=True) class MainTip: ...                                   # onto = main
@dataclass(frozen=True) class StackedOn: predecessor: TrainPredecessor       # its tip on/behind main or a verified member
@dataclass(frozen=True) class FrozenPrefix: staging_sha: str; main_sha: str
    # TaskBranchRefs.rebase validates FrozenPrefix by ancestry only: main_sha must be an ancestor of staging_sha.
    # That staging_sha is the lane's newest frozen merge commit holds by construction. The only producer of a
    # FrozenPrefix is suffix_graph.py::SuffixConflictTracker.bounce_conflicting_suffix_items, which builds it
    # from the worker's frozen_prefix_tip, and B24's AST tripwire asserts no other construction site.
    # frozen_prefix_tip -> str is unchanged everywhere; bounce_conflicting_suffix_items builds the target
    # locally: onto = MainTip() if frozen_tip == main_sha else FrozenPrefix(frozen_tip, main_sha)
OntoRef = MainTip | StackedOn | FrozenPrefix
@dataclass(frozen=True) class RebaseOutcome: ok: bool; before: OwnDelta; after: OwnDelta | None; abort_failed: bool
class TaskBranchRefs:
    def __init__(self, runner: GitRunner, config: GitConfig, project_root: Path)
    async def create(self, branch: QueuedBranch, start_ref: str, *, worktree: Path | None) -> None
    async def rebase(self, worktree: Path, branch: QueuedBranch, *, onto: OntoRef) -> RebaseOutcome
        # validates onto per variant (task 4752's guard, one site) else raises BranchResetError naming onto and main;
        # post-condition: own-delta count preserved else reset to the pre-rebase head and raise BranchResetError
    async def reset(self, worktree, branch, target) -> None; rename_aside(...); swap(...)
    async def delete(self, branch: QueuedBranch, *, force: bool) -> None   # the 9 branch -D/-d / update-ref -d sites
    async def own_delta(self, branch: QueuedBranch) -> OwnDelta           # merge-base with main + rev-list
GitOps.branches: TaskBranchRefs   # composed; GitOps methods accept QueuedBranch
```

Invariants: `'worktree', 'add', '-b'`, `'checkout', '-B'`, `'branch', '-m'`, `'branch', '-D'/'-d'`,
`update-ref [-d] refs/heads/task` and `'rebase'` argv for task branches appear only in `task_branch.py`
(the one deleter in `release_warm_worktree` — `escalation/server.py` today, `orchestrator/mcp/ops_tools.py`
after S3 δ — goes through `GitOps.branches.delete`; whichever of λ and S3 δ lands second adapts) (AST tripwire test, the shape
`TestGitOpsAbortUniformity` already uses); `resolve_queued_branch_ref` is deleted — callers hold a
`QueuedBranch` and ask for `full_name`; `git_ops.py` contains no `f'{self.config.branch_prefix}` construction.
Task 5144's admission guard is the redundant second check and is unchanged by λ.

## Boundary-test sketch

Two-way tests per contract; each is the done-criterion of its leaf and re-run by π.

| # | Scenario | Preconditions | Postconditions |
|---|---|---|---|
| B1 (C1) | A `ScriptedGitRunner` fails `git worktree add` once | `GitOps(runner=fake)` on a real tmp repo | `create_worktree` retries once and succeeds; no `patch` of any `orchestrator.git_ops` name in the test |
| B2 (C1) | `verify.py::_derive_task_files_from_git` on a vanished worktree | worktree dir removed before the call | raises `git_run.WorktreeMissing`; `verify.py` imports nothing from `git_ops` for it |
| B3 (C1) | `delivered_checks.run_delivered_check` with the default runner | a script check whose script sleeps 2 s under a 1 s timeout | the child is killed and reaped (no zombie), the check reports timeout — same as today, via `run_git`; the default runner passes `bounded=False` (as do `skew_tripwire::_run_load_bearing_oracle`'s calls), so `test_a_long_running_script_cannot_delay_a_concurrent_git_call` still passes |
| B4 (C2) | `GitOps(scripts=FakeScripts(seed_rc=75))` | pool enabled, one lane free | `create_worktree` raises `WarmLaneDiskPressure`; the test touched no private name |
| B5 (C2) | `GitOps(limits=GitOpsLimits(reset_warm_lane_lock_wait_secs=0.1))` | a held lane lock | the bounded wait is 0.1 s — replaces 11 monkeypatches of `_RESET_WARM_LANE_LOCK_WAIT_SECS` |
| B6 (C2) | `Harness` built with a real `GitOpsCallbacks` | pool storage absent | `on_pool_storage_absent` fires; `grep` finds no `git_ops.<attr> =` outside `__init__`; `Harness` passed its own `LaneLifecycle` to `GitOps` (the `None` default was not taken) |
| B7 (C3) | every `WarmLaneUnavailable` member | — | member with a requeue class raises exactly that class from `create_worktree`; `FAULT` raises `RuntimeError` naming every requeue class from the table; `workflow_types.classify_failure(cls(...))` returns `m.disposition` for each |
| B8 (C3) | a new member added without a requeue class in a test module | — | import-time `AssertionError` names the member |
| B9 (C3) | `from_seed_rc(124)` | — | `LANE_LOCK_TIMEOUT`; `from_seed_rc(127)` → `FAULT`; rc sets disjoint |
| B10 (C4) | `get_changed_files('main', 'task/nope')` | ref missing | raises `GitCommandFailed` (a `GitQueryError`) whose `str` names argv, cwd, rc 128 and the stderr; `suffix_graph.recompute` returns `None` footprint for that branch; `find_task_citation_commit_checked` over a failing walk raises the same `GitCommandFailed`, and `find_task_citation_commit` reads it as `None` |
| B11 (C4) | `workflow.py::_plan` Lever-B with a failing diff | `ScriptedGitRunner` fails `git diff --name-only` | revalidation is NOT skipped; the task's block reason carries the `GitCommandFailed` text under the `GitQueryError` disposition row (BLOCK, escalate) |
| B12 (C4) | `worktree_work` on a dirty tree with 2 commits beyond main | real tmp repo | `WorkState(commits_beyond_main=2, dirty_paths=('f.txt',))`; a failed `status` raises, and `_cleanup_leftover_branch` retains the branch |
| B13 (C4) | `select_train_members` feeders with one unreadable candidate | fake fails the diff for one of three | that candidate is excluded; the other two stack |
| B13b (C4) | `OfflineLaneWorker._poll_tick` while `rev-parse` of main fails | `ScriptedGitRunner` fails `rev-parse` | the tick returns as today (no crash, the worker keeps polling) and logs the `GitQueryError`; the same holds for every `catch` row of ε's `get_main_sha` table |
| B14 (D6) | `parse_unified0` on a diff whose content line is `++ b/zzz` | the as-of reproduction fixture | one `FileDelta` for `f.txt` with ranges (2,2),(6,6); `get_changed_line_ranges` agrees with `parse_diff_added_line_ranges`' old answer |
| B15 (D6) | `parse_worktree_list` on porcelain with a detached, a bare and a locked entry | — | three `WorktreeEntry`s with `head`, `branch`, `detached`, `locked`; the five former parsers are projections |
| B16 (C5) | `advance_main` on a WIP overlap | project_root dirty on an overlapping file | `AdvanceOutcome(result='wip_overlap', detail.overlap_files=('x',))`; `gates._map_advance_failure` renders the file names without `getattr` |
| B17 (C5) | `recover_red_main` while a foreign `index.lock` is held for 1 s | `ScriptedGitRunner` + a real lock file | the move waits out the lock, `read-tree` runs, `read_tree_synced=True`; the same scenario through `advance_main` behaves identically (one primitive) |
| B17b (C5) | `recover_red_main` with `main_gate_bypass_command` configured | (a) the CAS succeeds; (b) the engage hook raises | `main_gate_bypass_clear_command` runs exactly once in (a) and still runs in (b); `advance_main` (MARK) runs `unmark` only on failure |
| B18 (C7) | local `merge_verify_lease` and the remote `_verify_merge_body` build hold on the same lane | one holds | the other raises `LaneLeaseContended` naming the holder; holder pgid file present only while the persistent lane is held |
| B18b (C7) | two `_merge_lane_gate` + `_verify_merge_body` spans on one host and one lane | the second started while the first is between gate release and build re-take | the second span gets the contention result, and its reset never runs while the first span's build is live |
| B19 (C7) | `cancel_verify` for a request whose pgid is not the recorded holder | holder file names another pgid | `release_holder_if` returns False and leaves the file |
| B20 (C6) | corrupt `.lane-state/_lane-3.json` | — | `all_records().corrupt == ('_lane-3',)`; `Harness._recover_crashed_tasks` does not take the "no record" path for it; `read` raises `CorruptLaneRecord` |
| B21 (C6) | recovery of an `ASSIGNED` lane whose task is done | — | `lifecycle.release(lane)` writes RELEASED; for a `RELEASED` lane `release` returns None and writes nothing — the pool's `_note_released_durable` uses the same method; `adopt_lane` on a lane whose record names another task leaves the record unchanged, increments the drift counter and does not raise |
| B22 (D10) | `acquire_spec_lane` and the warm acquire on a mounted pool with no sentinel | bootstrap allowed | both call `_ensure_pool_storage` (spy); sentinel marked once; `thin_warm_lane` refuses with the same guard |
| B23 (C8) | `TaskBranchRefs.rebase` per `OntoRef` variant | sibling branch, train predecessor and a frozen staging commit exist | `MainTip` proceeds; a sibling tip resolved by `_resolve_predecessor_tip` without a verified `StackedOn` is refused with `BranchResetError` naming onto and main (the task-3269 graft); `StackedOn(TrainPredecessor(...))` proceeds when the predecessor's tip is on/behind main or a verified member and is refused otherwise; `FrozenPrefix(staging, main)` proceeds when `main` is an ancestor of `staging` and is refused when `main` is not an ancestor of `staging`, and the B24 tripwire covers construction of a `FrozenPrefix` outside `suffix_graph.py`; a bounce with an empty frozen prefix rebases onto main (`MainTip`) and is not refused; the own-delta count is preserved on every success; task 3269's sibling-tip graft and task 4405's reuse leak reproduce and are refused (ruling OQ-3) |
| B24 (C8) | every writer argv for task refs | AST walk over `orchestrator/src` | all `worktree add -b` / `checkout -B` / `branch -m` / task `update-ref` / `rebase` argv sit in `task_branch.py`; `FrozenPrefix(` is constructed only in `suffix_graph.py::SuffixConflictTracker.bounce_conflicting_suffix_items` |
| B25 (D12) | `_verify_commit_on_main` under a 5 s timeout with a hanging git | fake runner returns `timed_out=True` | returns `_AncestorCheckFailure(confirmed_off_main=False)`; no `create_subprocess_exec` in the module |
| B26 (D13) | corrupt `blocking_dependency.json` | — | `read_blocking_dependency()` returns `None` with one WARNING (seam behaviour) instead of raising `JSONDecodeError` into the workflow; a valid file yields `BlockingDependencyExit` |
| B27 (D14) | import graph of the new modules | — | none of `git_run`, `git_diff`, `worktree_list`, `block_disposition`, `lane_lease`, `warm_lane_scripts`, `task_branch`, `worktrees`, `main_landing`, `git_queries` imports `orchestrator.git_ops` (heuristic 13 tripwire) |
| B28 (π) | the patch-target measure over the S5 modules | `scripts/merge_lane_metrics.py` extended (it measures only `LANE_PACKAGE = 'orchestrator.merge_lane'` through `lane_patch_modules` today, measured at `e634cfe658`, so the baseline cannot see `git_ops` targets) | `scripts/source_measures.py::patch_targets` over `orchestrator.git_ops`, `orchestrator.worktrees`, `orchestrator.main_landing`, `orchestrator.task_branch`, `orchestrator.lane_lease`, `orchestrator.git_queries` across `orchestrator/tests` and `escalation/tests` returns no `_`-prefixed leaf; the instance-form sites (`patch.object(<gitops>,'_x')`, `setattr(<gitops>,'_x')`, `<gitops>._x = `; ≥19 / ≥10 / 80 at `d40ba048e9`; by an any-receiver rule over `git_ops`' private names, 37 / 37 / 81 at `e634cfe658`) are counted and are 0; 0 private reads of `GitOps` from other `src` modules |

## Pre-conditions for activating

- Substrate confirmed at `ce0f71a384`, re-checked at `d40ba048e9` and `e634cfe658` (G3): `shared.git_async.run_git`/`GitResult`
  exist with `timeout`, `bounded`, `pin_c_locale`; `lane_lifecycle.LaneLifecycle` with `LEGAL_TRANSITIONS`,
  `_read_or_raise`, `pool_storage_present`/`mark_pool_storage_present`; `WarmLanePool.__init__(lane_lifecycle=)`;
  `merge_lane/types.py::QueuedBranch`; `rebase_recovery.guarded_abort`/`RECOVERY_GIT`; `GitOps.unmerged_paths`;
  `GitOps.is_ancestor_checked`; `lane_lifecycle.AcquireRoute`; `scripts/source_measures.py::patch_targets`;
  `shared/delivered_check_polarity.py` (its `build_grep_argv` passes pathspecs verbatim, so the
  `:(exclude)` pathspec in α's check works); `git_ops.py::GitQueryFailed`, `_git_query` and
  `_none_if_query_failed` (tasks 7485, 7332), which α renames and moves; `LaneLifecycle._read_best_effort`
  and `GitOps.lane_in_use` (task 3076). **Not substrate:** the merge-lane ratchet does not measure `git_ops`
  patch targets — `scripts/merge_lane_metrics.py::LANE_PACKAGE = 'orchestrator.merge_lane'` is the only
  package `lane_patch_modules` enumerates (measured at `e634cfe658`) and the baseline `tests` section has no
  target under `git_ops` while 24 test files patch `orchestrator.git_ops._*` — so π extends the measure
  (B28) rather than reading it.
- Out-of-batch dependencies: 7923 and 3918 (ε), 7721 (κ), 4841, 7341 and 5697 (λ) — all pending with no
  claimant at `e634cfe658`; the lead wires them at decompose. 3076 (ι) and 3155 (ν) landed after authoring
  and are no longer edges. Task 4752 is folded into λ (ruling OQ-3). Cross-stream: α and λ0 depend on S6 α
  (same-file; S6 α lands first and is not held behind 5040); β depends on S3 δ; γ
  depends on S6's lifecycle task (S6 ζ), which under S6-2 = B waits on task 5040 (§Cross-PRD), so every
  leaf from γ on (17 of the chain's 21, and σ) waits on 5040; α, λ0, β0 and β do not.
- The lead's `cancel` moves (§Related tasks) are applied before the leaves they name dispatch.

## Cross-PRD relationship (G4)

| Seam | Owner | This PRD | Consumers |
|---|---|---|---|
| `shared.git_async.run_git` / `GitRunner` / `git_run.run_in_worktree` (C1) | S5 | α, β land it | verify (S4) — α repoints `verify.py`'s 3 function-local imports; merge lane (S6/S7) — α repoints `worker.py`, `gates.py`, `disposition.py`, `landing_evidence.py`, `skew_tripwire.py` (α → S6 α: S6 α lands first on `worker.py` and `verify.py`); escalation server (S3) — **S3 δ** repoints `release_warm_worktree` onto `run_git` while moving it to `orchestrator/mcp/ops_tools.py`, and α does not touch `escalation/server.py`; β (which deletes `_run`) depends on S3 δ, or repoints that one import itself if S3 δ is not filed at β's dispatch; fused-memory probes (S10) — μ |
| `GitOpsCallbacks` / ports / `LaneLifecycle` (C2, C6) | S5 | γ, ι | harness (S8) constructs — ruling OQ-4 (= S8/OQ-10 (a)): γ and ι perform the minimal wiring at today's `Harness.__init__` site and delete the 4 installs and the `git_ops._lane_lifecycle`/`git_ops._*` reaches; S8 β re-homes the construction behind a same-file edge on γ and ι, and S8 drops its held λ. |
| `AdvanceOutcome.detail` (C5) | S5 | η | merge lane (S6). Rulings OQ-7 (B) and OQ-8 (A): the type is the frozen union `AdvanceDetail` (C5) on `AdvanceOutcome.detail`; **η** deletes the 7 `_last_*` writes and the 5 `getattr(git_ops, '_last_…')` reads together and changes `_map_advance_failure` to take the outcome (its 2 callers in `merge_lane/worker.py` with it); S6's ε is deleted; fk-59b3df6f52a3 (S6) is disposed with fk-eb7b9f65f412 and fk-3ee93d431321 in S5 η. |
| `WarmLaneUnavailable` table (C3) | S5 | δ | workflow_types rows derive (S9) — δ lands the derivation; S9's `BlockReason` reads it; `BlockDisposition` home per D4 (ruling OQ-1); S9's λ edits the member data in `git_ops.py` after δ (same-file edge wired at decompose) |
| `GitQueryError` / `GitCommandFailed` (C1, C4) | S5 | α (name), ε (contract) | S7's `GateMeasurement` wraps it for gates that query git (R11); S9's phase loop sees it as a typed block cause; tasks 7485 and 7332 (landed) raise it today under the old name `GitQueryFailed` from `resolve_rebased_citation`, `commits_named` and the `*_checked` lookups, and `agents/landed_sha_searches.py` (task 4911) shares its citation pattern |
| `lane_lease` / span lock (C7) | S5 | θ | `cli.verify_merge`/`cancel_verify` (S20 owns cli invocation — θ edits only the lease spans); tasks 3077 and 3071 build on θ |
| `QueuedBranch` placement (C8) | S5 moves, S6 imports | λ0 | every importer repointed (ruling OQ-2); no re-export from `merge_lane/types.py`; escalation server already imports `shared.branch_names`; λ0 → S6 α (S6 α lands first on the shared files), and every other S6 task on `merge_lane/types.py`, `worker.py` or `harness.py` imports follows λ0 |
| `TaskArtifacts` readers (D13) | S5 (non-plan readers and the typed architect-exit reports) / S9 (plan family, `read_iteration_entries`) | ν | S9 C2 owns `read_iteration_log`'s replacement; ν does not touch it and depends on S9 γ, which lands it first in `artifacts.py`; S9 ceded the five typed architect-exit reports to ν (S9 Q-C2, 2026-11-02) and S9 δ depends on ν |
| `scripts/reclaim_orphaned_worktrees.py`, fused-memory porcelain parsers | S25 / S10 / S12 | noted, not edited | a shared `WorktreeEntry` parser is a later `shared/` decision |
| `git_ops.py` partition (ξ0/ξ, ο0/ο, ρ0/ρ) | S5 | ξ0–ρ | S6's `merge_lane/landing.py` extraction (task 5042) builds on `MainLanding`; S8 composes `WorktreeOps` nowhere directly; S3's server split consumes `git_ops.worktrees.*` — ξ0 repoints `claim_warm_worktree`/`release_warm_worktree` in whichever module holds them at ξ0's dispatch (S3 δ moves them; whichever lands second adapts) |

Cross-stream editors of S5's files, each needing an edge or an xref at decompose (B4): S4 ζ adds
`WorktreeKind.MAIN_SWEEP_CONFIRM` in `git_ops.py` while ξ0 moves `WorktreeKind` to `worktrees.py`; S8 β edits
`harness.py` after γ and ι (ruling OQ-4); S9 λ edits the `WarmLaneUnavailable` member data after δ; S6 leaves edit
`merge_lane/worker.py` and `merge_lane/gates.py` (α, γ, ε, ζ, η, θ, ο0 touch them) — lead ruling 2026-11-02:
S6's typed-facts leaves land first on those two files, so **S5 γ depends on S6 ζ** (ItemLifecycle operative,
S6's lifecycle task, which under the sitting's S6-2 = B itself waits on task 5040);
ε, ζ, η and θ follow γ through the chain, so that one edge orders them all; S6 ζ does not wait on S5 η; S6 κ
still depends on S5 θ; S6 ε is deleted and S5 η keeps the `_last_*` reads and writes; S3 δ moves
`release_warm_worktree` (β, λ, ξ0). Lead sequencing ruling of 2026-11-10: S5 λ0 runs second in this
chain (after α only), and every S6 task other than S6 α that edits `merge_lane/types.py`,
`merge_lane/worker.py` or `harness.py` imports is ordered after S5 λ0 by the lead's per-file order; S6's
request that S5 λ0 wait on S6 λ and S6 μ is declined. Lead ruling after the fold-verifiers (2026-11-10):
S6 α lands first on every file it shares with another plan and takes no edge on S5, so S5 α → S6 α
(`merge_lane/worker.py`, `verify.py`) and S5 λ0 → S6 α (`merge_lane/types.py`, `merge_lane/worker.py`,
`harness.py`, the merge-queue tests that import `QueuedBranch`). λ0 also repoints the function-local `QueuedBranch` import in
`escalation/server.py::merge_request`, which S3's merge-tool move later carries (whichever lands second
adapts).

## Related tasks (contract §8)

Step 1 (`find_tasks_by_metadata` on every key): none, re-run on 2026-11-10. Step 2 (`search_tasks ≥ 0.5`,
hits read with `get_task`; re-run by the critic, which found five more, M10, and by the ruling fold at ≥ 0.6
over each leaf's remedy, which found 6236, 7957 and 8631): the live tasks below carry part of this work;
each is a dependency, a `cancel` move or a note, never a second task. Statuses read on 2026-11-10.

| Task | Status | Relation |
|---|---|---|
| 5697 | pending, medium | carries fk-54eb9f7cc0be (rerere on the rebase primitives, abort rc, preflight) — `filed:5697`; λ depends on it; lead stamps `x_finding_key` |
| 7923 | pending, low (PRD merge-pre-verify-screen ε) | lands the canonical ref + typed error for `get_changed_line_ranges`; ε depends on it and moves it onto `GitCommandFailed`; the typed error 7923 adds should be the existing `GitQueryFailed` (or `GitQueryError` once α lands), not a third type |
| 3918 | pending, high (dependency 5036 done, so dispatchable) | rewrites `GitOps.get_files_touched_in_branch`, one of ε's five enumerators; S7's D6 orders it before ε — ε depends on 3918 |
| 5937 | pending, low | `get_changed_files` failure caller-visible — subsumed by ε (cancel → ε) |
| 7843 | pending, low | retire `_detect_unmerged_paths` — subsumed by ε (cancel → ε) |
| 7485 | done 2026-11-07 (`d37a05026d`) | landed `GitQueryFailed`, `_git_query` and `GitOps.commits_named`; `resolve_rebased_citation` now raises on a git fault; α renames the type, β and ε move its helper — no edge |
| 6594 | pending, low (this run's mechanical filing, fk-04c87a1b5dd8) | `_STEAL_RETRYABLE` bound twice — subsumed by δ (cancel → δ; retryability becomes member data, C3) |
| 7457 | pending, low | public seam for `ephemeral_worktree`'s runner and seed — subsumed by β/γ (cancel → γ) |
| 7645 | pending, low | `TestFreshenMain` stops patching `_run` — subsumed by β (cancel → β) |
| 7590 | pending, low | a test imports `_run` — subsumed by β0 (cancel → β0) |
| 3076 | done 2026-11-05 (`1f40b7986b`) | landed the IN_USE spans (`GitOps.lane_in_use`, `LaneLifecycle.note_in_use`/`note_idle`/`reset_stale_in_use`, `LaneLiveness`); ι builds on them and keeps their never-abort guarantee (C6) — no edge |
| 3077 | pending, high (needs a design pass, per its own 2026-10-30 note) | whole-assignment lane lease replacing the two `task_verify_lease` call sites, which θ rewrites — 3077 depends on θ |
| 3071 | pending, medium | merge-worker handlers on `MergeVerifyLeaseContended`/`LaneLockSelfOwnedLeak`, which θ renames and moves — 3071 depends on θ. σ redesigns the self-owned-leak probe 3071 reacts to: note, no edge (σ is low; whichever lands second adapts). The fold-verify dedup for σ (`search_tasks`, 2026-11-10) found 3071 at 0.72 and 3077 at 0.66, neither carrying the re-entrant lease |
| 7721 | pending, high | spec-lane leak on chain-build timeout; same region as κ — κ depends on it |
| 4841, 7341 | pending | non-`task/*` branch resolution and the remaining ad-hoc prefix strips — λ depends on both |
| 4752 | pending, high, no claimant or heartbeat on 2026-11-10; its only dependency 5036 is done, so it can dispatch any time | onto-ref ancestor guard across call sites, `metadata['train']` clearing, `_resolve_predecessor_tip` — folded into λ (ruling OQ-3, A′), which carries all three (C8 `OntoRef`): the lead adds 4752 → λ when λ is minted and cancels 4752 at decompose after a heartbeat check; until then, park it by adding a dependency (Leo's park-by-dependency SOP), never by deferring |
| 5144 | pending, high | admission-time foreign-commit citation guard — stays; λ names it as the redundant check |
| 3155 | done 2026-11-02 (`ef96d32603`) | `artifacts.py::ARTIFACT_ROOTS` is the one root resolver; ν builds on it — no edge |
| 6892, 5298 | pending | advance_main hook-refusal classification (adds the `HookRefused` variant); HEAD-state verification — both depend on η (edges); 3232 (deferred) is superseded by 5298 (note) |
| 5274 | pending, low | task_interceptor done_provenance claims — depends on μ (edge) |
| 6236 | pending, low | moves the done-provenance cluster out of `task_interceptor.py` into `middleware/done_provenance.py` with an injectable git-runner seam — μ carries the runner seam, so 6236 depends on μ and keeps only the extraction (edge; S10's ν partitions the same file — xref) |
| 5341, 7767, 5042 | pending | lane release on non-runnable tasks → depends on ι; `BranchCheckedOutElsewhere` requeue → depends on δ; `landing.py` extraction → depends on η (edges; S6 note for 5042) |
| 6219 | done 2026-11-03 (`ac4fec65cd`) | un-stacks a requeued branch before `_reuse_warm_lane` rebases it (`_rebase_requeued_onto_main`); κ keeps it on the REUSE route and λ routes its rebase through the owner — no edge |
| 7957, 8631 | pending, low | 7957: a refused lane WIP snapshot re-pins the lane with only a WARNING in the reconcile sweep that ι routes through `adopt_lane`; 8631: `acquire_spec_lane`'s reset-in-place strands a lane on `WorktreeMissing`, in the region κ puts on `_ensure_pool_storage` — notes, no edge (low priority; an edge onto a chain leaf would hold them for the whole chain); whichever lands second adapts |
| 6432, 7350 | pending, low | gates' probes behind a GitOps seam (S7's call after α), verify_runner env probe via `run_git` (S4) — notes |

## Decomposition plan

Labels are Greek; a `0` suffix marks a mechanical codemod sibling filed `task_kind='normal'` with
`metadata.complexity='simple'` (the mechanical fast path, which allows several files) — never
`task_kind='deterministic'`, which runs no agent and no branch (`docs/task-authoring.md` §5). Each carries a
non-empty `metadata.files` listing its enumerated files (`SimpleTaskFilesRequired` rejects an empty list), and
its description stays free of the fast-path veto tokens; ξ0, ο0 and ρ0 also create a module, so the decomposer
may set `metadata.force_full_path` on those three. `priority` is the highest
severity of the keys the leaf closes; `high` leaves become dependencies of the trigger-chain gate 6698 at
decompose. Every leaf carries `metadata.x_finding_key`, `x_finding_run = review-all-dark_factory-20261008`,
`source = review-all`, `metadata.files`, except the follow-up σ, which closes no key and carries no
`x_finding_key`. Sizes are expected changed LOC in the overlay's sense (moves counted
by moved region, not diffstat); the codemod leaves exceed the ≤10–12-file band by construction and are
mechanical for that reason (one shape each, enumerated by a stated rule at decompose). Counts
labelled `measured at d40ba048e9` were re-measured by the fold; counts labelled `measured at e634cfe658` by
the ruling fold (§Freshness at e634cfe658).

- **α — `run_git` is the factory-wide runner; `git_run` leaf; the 9 orchestrator importers repointed; one
  name for the git-query error** (C1, part). Modules: `shared/src/shared/git_async.py` (+`GitRunner`), new
  `orchestrator/src/orchestrator/git_run.py` (`run_in_worktree`; `WorktreeMissing` moved; task 7485's
  `GitQueryFailed` moved and renamed `GitQueryError`, R5's name, and `_none_if_query_failed` moved as
  `none_if_query_failed`), `workflow.py`, `merge_lane/worker.py`, `merge_lane/disposition.py`,
  `merge_lane/landing_evidence.py`, `merge_lane/gates.py` (its `_INDEX_LOCK_STALE_FLOOR_S` import stays until γ),
  `merge_lane/skew_tripwire.py`, `verify.py`, `scheduler.py`, `delivered_checks.py`, the `WorktreeMissing`
  importers (`workflow_types.py`, `steward.py`, `merge_lane/worker.py`; prose mentions in `task_ground_truth.py`,
  `rebase_recovery.py`, `merge_lane/speculation_controller.py`), the `GitQueryFailed` importers
  (`workflow_types.py`, whose disposition row is re-keyed to `GitQueryError` and its prefix reworded, D5;
  `scripts/restamp_degenerate_branch_bases.py`; tests `test_git_ops_checked_lookups.py`,
  `test_git_ops_rebased_citation.py`, `scripts/tests/test_restamp_degenerate_branch_bases.py`), and the tests
  of those modules that patch `_run` by dotted path (target: `shared.git_async.run_git` or an injected runner
  where the module already has one — `delivered_checks`, `skew_tripwire`). Per-importer runner and `bounded`
  value: C1's table — every site keeps `bounded=False`. `escalation/src/escalation/server.py` is S3 δ's (M4).
  Deletes: the 13 orchestrator sites' use of a private name; `WorktreeMissing`, `GitQueryFailed` and
  `_none_if_query_failed` from `git_ops.py`. Signal: §Goal bullet 1's pattern matches only
  `escalation/server.py` (S3 δ's) and `git_ops.py`'s own definition; `GitQueryFailed` is absent from
  `orchestrator/src` and `scripts`; B2/B3 pass. Prereqs: S6 α (cross-stream same-file edge on
  `merge_lane/worker.py` and `verify.py`: S6 α lands first, lead ruling after the fold-verifiers of
  2026-11-10); nothing in this plan. Priority medium (fk-d73aeaebb792). ~650 LOC, 17
  src and script files + tests (review trigger acknowledged: two moves of one mechanical shape).
- **λ0 — `QueuedBranch` moves to `shared.branch_names`** (`complexity='simple'`; C8, ruling OQ-2; second in
  the chain by the lead's sequencing ruling of 2026-11-10; `metadata.files` = the 90 test files + the src
  modules below). Modules:
  `shared/src/shared/branch_names.py` (`QueuedBranch` moved, `parse(raw, prefix)` unchanged),
  `merge_lane/types.py` (definition deleted, no re-export), `merge_lane/__init__.py` (lazy table points at the
  new home), every importer: 7 `src` modules (`escalation/server.py` — the function-local import in
  `merge_request`, in whichever module holds it at λ0's dispatch — `branch_mergeability.py`, `harness.py`,
  `merge_lane/queue_store.py`, `merge_lane/worker.py`, `stranded_verified_green.py`, `workflow.py`) and 90 test
  files (AST: 89 at `d40ba048e9` and `e634cfe658`, plus task 8524's `test_merge_lane_remerge_base.py`,
  measured at `bb1c659562`; enumerated at decompose), `shared/tests/test_branch_names.py`,
  `test_merge_types.py`. Deletes: `QueuedBranch` from `merge_lane/types.py`. Signal: `^class QueuedBranch`
  present in `branch_names.py`, absent in `merge_lane/types.py`. Prereqs: α (chain; shares `workflow.py` and
  `merge_lane/worker.py`); S6 α (cross-stream same-file edge on `merge_lane/types.py`,
  `merge_lane/worker.py`, `harness.py` and the merge-queue tests that import `QueuedBranch`: S6 α lands
  first, lead ruling after the fold-verifiers of 2026-11-10). It needs nothing else, so it lands before S6's
  other tasks on `merge_lane/types.py`, `merge_lane/worker.py` and `harness.py`, which the lead orders after it. Priority medium
  (fk-718c82fcc1a2 part). ~250 LOC.
- **β0 — test git helper codemod** (`complexity='simple'`; C1 tests; `metadata.files` = the 135 enumerated test files). New `orchestrator/tests/_git.py::run` (a 3-tuple
  `(rc, out, err)` helper over `git_run.run_in_worktree`, the shape the fixtures use today); every
  `from orchestrator.git_ops import … _run` in `orchestrator/tests` becomes `from _git import run as _run`
  (call sites unchanged) and the 2 `escalation/tests` importers call `orchestrator.git_run.run_in_worktree`
  directly. Enumerated by AST `ImportFrom(module='orchestrator.git_ops', names∋'_run')` at decompose — 135
  files at `e634cfe658` (133 + 2; `scripts/tests` has none). The critic's grep rule
  (`from orchestrator\.git_ops import[^#]*\b_run\b|^\s+_run,\s*$`) also matches `offline_lane` tests that
  import a different `_run`, so the AST rule is the enumerator. Deletes: 135 test imports of a private name.
  Signal: the single-line import pattern is absent from `orchestrator/tests` and `escalation/tests`; the
  multi-line form is enforced by β, whose deletion of `_run` fails collection of any remaining importer.
  Prereqs: λ0 (chain; the two codemods share test files), α. Priority medium (fk-3a4cc6298c02). Cancels folded: 7590. ~300 LOC (one import line × 135 files).
- **β — `GitOps` takes its runner; `_run` deleted; GitOps patch-form tests move onto the port** (C1). Modules:
  `git_ops.py` (constructor `runner`, `_git`/`_query` adapters over 197 `await _run(` sites; task 7485's
  module helper `_git_query`, 9 calls, becomes `GitOps._query` — measured at `e634cfe658`), new
  `orchestrator/tests/_git_runner_fakes.py::ScriptedGitRunner` (replacing `orchestrator/tests/_worktree_add_fakes.py`'s
  parameterised fake `_run` and task 7485's `PATH` shim `orchestrator/tests/_git_fault_shim.py`, heuristic 11 —
  both deleted, their importers moved, `test_git_ops_rebased_citation.py` among them), the test files that
  patch or read `orchestrator.git_ops._run` / `GitOps._run` (29 at `d40ba048e9`; the author's regex matches 32
  files at both `d40ba048e9` and `e634cfe658`; `test_git_ops.py`, `test_verify_main_tip_sweep.py`,
  `test_ephemeral_worktree.py`, `test_create_merge_worktree_retry.py`, …; enumerated at decompose).
  Deletes: module-level `_run` and `_git_query`; every patch-form `_run` reference; the two test fakes. Signal: `grep -c '^async def _run(' git_ops.py`
  = 0; B1. Prereqs: β0; S3 δ (cross-stream edge — deleting `_run` breaks `release_warm_worktree`'s
  function-local import; if S3 δ is not filed at β's dispatch, β repoints that one import onto `run_git`
  itself). Priority medium (fk-d73aeaebb792, fk-3a4cc6298c02). Cancels folded: 7645. ~900 LOC, mostly tests.
- **γ — constructor ports: `WarmLaneScripts`, `WarmBaseProbe`, `GitOpsLimits`, `GitOpsCallbacks`** (C2).
  Modules: new `orchestrator/src/orchestrator/warm_lane_scripts.py` (the 14 script-wrapper methods and the seed
  wrappers — γ is their one owner, not ξ; `SeedLaneLock`, `WarmBaseHealth`, the `_seed_script_*` helpers),
  `git_ops.py` (constructor; `_warm_lane_base_resolvable` becomes the default sync probe; the 7 constants and
  `_INDEX_LOCK_STALE_FLOOR_S` become `GitOpsLimits` fields), `merge_lane/gates.py` (`_STALE_LOCK_FLOOR_S` reads
  `git_ops.limits.index_lock_stale_floor_s`, m7), `harness.py::Harness.__init__` (constructs `GitOpsCallbacks`
  + minimal wiring (ruling OQ-4); the 4 installs and the `set_on_lane_record_drift` call deleted), `cli.py` ×2,
  `recover_main.py` and `evals/runner.py::_cell_workflow` (`GitOpsCallbacks.inert()`;
  `scripts/restamp_degenerate_branch_bases.py` takes the same default with no edit), `warm_lane_pool.py`
  (drift callback in the constructor), the test files that patch the covered names (`_seed_warm_lane`,
  `_warm_lane_base_resolvable`, `_on_pool_storage_absent`, `_on_structural_exhaustion`,
  `_run_warm_lane_gc_reclaim`, `_run_warm_lane_disk_guard`, the 7 constants — ≈14 files incl.
  `test_pool_storage_guard.py`, `test_lane_lock_leak_guard.py`, `test_warm_lane_steal_retry.py`,
  `test_ephemeral_worktree.py`). Deletes: `_DF_WARM_LANE_SCRIPT_DIR_ENV`, the `POOL_ROOT_SENTINEL` re-export
  shim, `GitOps.refuse_foreign_band` wrapper, the 4 post-construction installs, 8 module constants, ≈75
  private patch sites. Signal: B4–B6; `git grep -nE 'git_ops\.[a-zA-Z_]+ *= [^=]' -- orchestrator/src/orchestrator/
  escalation/src/escalation/` empty. Prereqs: β; S6 ζ (cross-stream same-file edge, lead ruling 2026-11-02;
  under the sitting's S6-2 = B, S6 ζ itself waits on task 5040).
  Priority medium (fk-e2ac53b0d72b; fk-b1e3368f99a1 part).
  Cancels folded: 7457. ~1,200 LOC.
- **δ — the warm-lane unavailability table** (C3; R5 option 1). Modules: new
  `orchestrator/src/orchestrator/block_disposition.py`, `git_ops.py` (`@enum.unique` `WarmLaneUnavailable`
  with string values and per-member data, `from_seed_rc`, `raise_for`; `create_worktree`'s chain → one call;
  `_seed_rc_to_unavailable`, `_STEAL_RETRYABLE`, the 9 inline rc checks deleted; `SeedOutcome` mapped once),
  `workflow_types.py` (rows derived; `BlockDisposition`/`RequeueKind` from the leaf; the function-local
  `git_ops` import goes module-level, m16), `dry_run_unblock.py` and `workflow.py` (imports), `warm_lane_scripts.py`
  (`SeedOutcome` consumers), tests: new `orchestrator/tests/test_warm_lane_unavailable_table.py` (B7–B9),
  the 6 test files that import `BlockDisposition`/`RequeueKind` (`test_block_disposition.py`,
  `test_ephemeral_worktree.py`, `test_seed_lane_lock_reentrancy.py`, `test_warm_lane_soft_floor.py`,
  `test_workflow_state_machine_boundary.py`, `test_reviewer_verdict_routing.py`; measured at `e634cfe658`), `test_warm_lane_steal_retry.py`, the 5 tests naming
  `_seed_rc_to_unavailable`. Deletes: the 9-arm chain, the FAULT literal class list and its 23 comment lines,
  `_STEAL_RETRYABLE`, `_seed_rc_to_unavailable`, 9 inline rc checks, the nine-row block in `workflow_types.py`,
  the three prose member lists. Signal: B7–B9; `grep -c 'pool_info is WarmLaneUnavailable\.' git_ops.py` = 0.
  Prereqs: γ. **Priority high** (fk-856541951d65; fk-6f7520ff1987). Cancels folded: 6594. ~750 LOC.
- **ι — `LaneLifecycle` as the public owner** (C6). Modules: `lane_lifecycle.py` (`read` raises on corrupt,
  `LaneRecords`, `release`, `quarantine(relocate=)`, constructor without the callable), `git_ops.py` (optional
  `lane_lifecycle` port, `None` → built here; `quarantine_lane`; `adopt_lane`; pools built with lifecycle),
  `warm_lane_pool.py` (`PoolSlotState`; optional `lane_lifecycle`; `set_lane_lifecycle`/`set_on_lane_record_drift`
  deleted; `_note_released_durable` → `lifecycle.release`), `harness.py` (`Harness.__init__` constructs the
  lifecycle and passes it; recovery's 3 writes in `Harness._recover_worktree_entry` → `release`/`quarantine_lane`;
  the 6 `pool.restore_assignment` calls (`_recover_worktree_entry` ×1, `_reconcile_lane_checkouts` ×1,
  `_reconcile_one_lane_checkout` ×4, measured at `e634cfe658`) → `git_ops.adopt_lane`;
  `_assigned_durable_records_with_statuses` reads `LaneRecords` and `_advance_lane_record_divergence_and_alarm`
  (task 2987) reads the lifecycle Harness holds; the second `set_lane_lifecycle` in `_recover_crashed_tasks`
  deleted), `lane_lifecycle.py::_read_best_effort` (also catches `CorruptLaneRecord`, keeping task 3076's
  never-abort spans), `task_runtime.py`, `stranded_verified_green.py`, `lane_branch_holder.py`
  (import), `cli.py` ×2, `recover_main.py`, `evals/runner.py::_cell_workflow` (each passes its own lifecycle),
  tests `test_lane_lifecycle.py`, `test_lane_lifecycle_gitops.py`, `test_lane_lifecycle_integration.py`,
  `test_crash_recovery.py`, `test_harness_warm_lane_wiring.py`, `test_warm_lane_pool.py`, `test_task_runtime*.py`,
  `test_stranded_verified_green.py`, `test_harness_lane_record_divergence.py` (B20, B21). Deletes: `git_ops._lane_lifecycle`
  reaches (10 code sites in 3 modules: `harness.py` 8, `task_runtime.py` 1, `stranded_verified_green.py` 1,
  measured at `e634cfe658`), the hand-copied RELEASED guard, the two `set_*` methods, the `DurableLaneState` aliases, the
  `None`-means-corrupt mapping, harness's direct `restore_assignment` calls. Signal: B20, B21; `grep -rn
  "git_ops._lane_lifecycle" orchestrator/src` empty; `restore_assignment` called only by `adopt_lane` and inside
  the pool. Prereqs: δ (chain), γ (task 3076, a prerequisite at authoring, landed 2026-11-05). **Priority high**
  (fk-31e9856ed662; fk-b1e3368f99a1, fk-a10d37257808). Edge: 5341 depends on ι; note 7957 (same re-pin arm).
  ~850 LOC.
- **κ — acquire routes and the pool-storage guard** (D10). Modules: `git_ops.py` (`_classify_acquire_route`,
  one `_acquire_<route>` per `AcquireRoute` member, `_ACQUIRE_ROUTES` table, `_ensure_pool_storage`;
  `acquire_spec_lane`, `thin_warm_lane` on the guard; the REUSE route keeps task 6219's
  `_rebase_requeued_onto_main` un-stack), `harness.py` (`_recover_crashed_tasks`,
  `_reap_orphan_worktrees` on the guard), `lane_lifecycle.py` (`AcquireRoute` doc comments lose their stale
  line pins), tests `test_warm_lane_pool.py`, `test_pool_storage_guard.py`, `test_warm_lane_integration_gate.py`,
  `test_spec_lane_reacquire.py`, new `test_acquire_routes.py` (B22 + one test per route through the public
  `create_worktree`). Deletes: the 879-line method body as one unit, the inline reset+rmtree+reseed copy, the
  duplicated bootstrap block, the harness no-bootstrap variants. Signal: B22; every `AcquireRoute` member has a
  method named for it (AST test). Prereqs: ι (chain), δ, task 7721. Priority medium (fk-a7cefe40ef4c,
  fk-c5d5769d68c6). Note 8631 (same `acquire_spec_lane` region; task 6219, an edge at authoring, landed
  2026-11-03). ~1,100 LOC (mostly moved).
- **ε — the query contract: `GitCommandFailed`, enumerators, `worktree_work`/`branch_work`** (C4). Modules:
  `git_run.py` (`GitCommandFailed(GitQueryError)`, `query`), `git_ops.py` (`GitOps._query` over `query`; 9
  rc-dropping read queries + `merge_to_main`/`_create_merge_worktree` rc checks, measured at `e634cfe658`; the
  hand-rolled rc checks of `find_merge_marker_checked` and `find_task_citation_commit_checked` onto `_query`;
  5 enumerators, 9 predicates → `worktree_work`/`branch_work`, `rebase_preserving_task_commits`' two
  inline counts — ε is their one owner, ζ excludes them — `_detect_unmerged_paths` deleted in favour of
  `unmerged_paths`), callers: `workflow.py`, `suffix_graph.py::recompute`, `task_verify_dispatch.py`,
  `merge_lane/worker.py`, `merge_lane/train_selection.py` (unreadable-candidate exclusion), `harness.py`,
  `offline_lane.py`, `verify.py`, `branch_mergeability.py::read_branch_state`, `service_restart.py` (if it still
  reads a `(files, err)` tuple — verify at decompose); tests `test_git_ops.py`, `test_suffix_conflict_tracker.py`,
  `test_plan_revalidation_skip.py`, new `test_git_query_contract.py` (B10–B13, B13b). Deletes: the `(files, err)`
  contract, the empty-value-on-failure branches, 9 predicates, `_detect_unmerged_paths` (task 7843).
  `get_main_sha` call sites (50 awaited sites in 8 files, AST, measured at `bb1c659562`; 51 at `d40ba048e9`
  and `e634cfe658` before task 8524 folded two guarded worker reads into one helper; "guard" = an
  `if not <var>` / `bool(<var>)` / `or None` within 4 lines of the call, or a fail-open helper; re-run the rule at dispatch):

  | Disposition | Sites (`module::scope`) |
  |---|---|
  | `catch GitQueryError →` today's skip/log, identical behaviour (13) | `git_ops.py::GitOps.describe_commit_effect_in_main`, `GitOps._lookup_merge_marker`; `harness.py::Harness._run_main_tip_sweep` ×2; `merge_lane/worker.py::_run_post_merge_verify` ×3, `SpeculativeMergeWorker._main_sha_or_empty` (task 8524's fail-open helper behind `_finalize_inflight`'s INV-3 check, `_dispatch_item`'s chain check and the two re-merge callers; its `except Exception` narrows to `GitQueryError`); `offline_lane.py::OfflineLaneWorker._run_once`, `OfflineLaneWorker._poll_tick`; `verify.py::verify_failure_is_preexisting_on_main`, `run_main_tip_sweep`; `workflow.py::TaskWorkflow._inter_iteration_rebase` |
  | `propagate` (37) | `delivered_checks.py::gate_mark_done_on_delivered_checks`; `git_ops.py::GitOps._unstack_recorded`, `GitOps.worktree_head_beyond_main` (retired by `worktree_work`), `GitOps.advance_main`; `merge_lane/worker.py::classify_and_merge`, `reconcile_landed_task`, `reconcile_landed_outbox`, `_do_train_merge` ×2, `_run_deferred_main_health_probe`, `_advance_train`, `SpeculativeMergeWorker._dissolve_coalesce_trains_if_halted`, `SpeculativeMergeWorker._remerge_on_main` ×2 (task 8524's name for `_remerge_inner`), `SpeculativeMergeWorker._launch_verify` (`_guard_main_sha`, moved from `_dispatch_item` by task 8524; already under that guard's own `except Exception: pass`, unchanged), `_merger_loop` ×2, `_advance_candidate`; `suffix_graph.py::SuffixConflictTracker.recompute`, `bounce_conflicting_suffix_items`; `workflow.py::TaskWorkflow._run_merge_phase` ×2, `_handle_already_done_report`, `_handle_ready_to_merge_report`, `_check_branch_on_main`, `_finalise_without_merge_sha`, `_resume_merge_retry_if_pending`, `_rebase_before_enqueue` ×2, `_handle_no_plan_failure`, `_stamp_merge_retry_pending`, `_handle_blocking_dep_report`, `_maybe_cleanup_done_worktree`, `_plan`, `_attribute_train_failure`, `_handle_terminal_exit_on_block`, `_drive` |

  A `propagate` site inside a long-lived loop (`_merger_loop`, `_launch_verify`, `_advance_candidate`) is
  confirmed at dispatch to sit under that loop's existing outer handler, else it moves to the catch row.
  Predicates (M8):

  | Predicate (fail direction today) | Replacement | `untracked` | Caller direction on `GitQueryError` |
  |---|---|---|---|
  | `_branch_has_commits_beyond_main(full_branch)` (True) | `branch_work(branch).commits_beyond_main > 0` | — | retain (catch → "has work") |
  | `_orphan_has_commits(full_branch)` (False if missing, else True) | `branch_work(branch)`; `exists=False` → no work | — | retain |
  | `worktree_has_unsaved_work(worktree, branch)` (True) | `worktree_work(worktree, untracked=…)` + `branch_work(branch)` | today's argv | retain |
  | `_worktree_dirty` (True) | `worktree_work(…).dirty_paths` | today's argv | retain |
  | `has_uncommitted_work` (CLEAN) | `worktree_work(…).dirty_paths` | today's argv | refusal-direction callers propagate |
  | `has_dirty_working_tree` (CLEAN, rc unread; `project_root`) | `worktree_work(project_root, untracked=False)` | `False` (untracked excluded today) | `harness.py::_file_dirty_tree_escalation` leaves the L2 pending |
  | `worktree_head_beyond_main` (None for both) | `worktree_work(…).commits_beyond_main` | today's argv | per C4's caller list |
  | `rebase_preserving_task_commits` inline counts (0) | `worktree_work(…)` | today's argv | propagate (a failed count aborts the rebase with the error, never disarms the guard) |
  | `branch_tip_if_no_commits_beyond(full_branch, main)` (task 7332; `None` when `is_ancestor` reads a git error as False) | `branch_work(branch)`; `commits_beyond_main == 0` → `tip` | — | retain: `workflow.py::TaskWorkflow._restamp_branch_base_if_no_own_commits` catches → "has commits", base left as recorded (its documented direction today) |

  "today's argv" = the predicate's current porcelain flags, read at implementation (`--untracked-files=no` ⇒
  `False`). Signal: B10–B13, B13b. Prereqs: κ (chain), β, tasks 7923, 3918. Priority medium (fk-d176c9ae1cc7,
  fk-ad8c0528c5e1). Cancels folded: 5937, 7843. (The edge "7485 depends on ε" is gone: 7485 landed
  2026-11-07, and α gave its error R5's name.) ~1,150 LOC, 16 declared files (12 src + 4 tests) — past the
  15-file review trigger, acknowledged: one contract applied at each caller of the changed queries, which a
  split would leave half-migrated.
- **ζ — typed diff, worktree-list and commit-count readers** (D6). Modules: new
  `orchestrator/src/orchestrator/git_diff.py` (`FileDelta`, `parse_unified0` and the four projections under new
  names), new `orchestrator/src/orchestrator/worktree_list.py`, `git_ops.py` (the four parser names deleted and
  their callers repointed to `git_diff`; 5 porcelain sites → `parse_worktree_list`; the count sites below →
  `count_commits`), `merge_lane/gates.py::_resolve_already_landed_branch` (`count_commits`), tests: new
  `test_git_diff.py` (B14, the as-of reproduction as a fixture), `test_worktree_list.py` (B15), `test_git_ops.py`.
  Count sites (each keeps today's direction, now explicit at the catch; ε owns the other four):

  | Site | Today on failure | After ζ |
  |---|---|---|
  | `git_ops.py::GitOps._freshen_main` ×2 | local main + WARNING | catch → local main + WARNING |
  | `git_ops.py::GitOps._reseed_verified_clean` | `False` | catch → `False` (not verified clean) |
  | `git_ops.py::GitOps.get_rebase_distance` | `-1` + WARNING | catch → `-1` + WARNING |
  | `merge_lane/gates.py::_resolve_already_landed_branch` | `None` + WARNING (default gate) | catch → `None` + WARNING |

  Deletes: 4 parsers' header logic (the module keeps no header-prefix logic), 5 porcelain loops, 5
  `int()`-and-choose-a-direction sites. Signal: B14, B15; `grep -c "startswith('+++ b/')" git_ops.py` = 0;
  `^def parse_diff_line_ranges\(` absent everywhere in `orchestrator/src`. Prereqs: ε. Priority medium
  (fk-d727d82341b3). ~500 LOC.
- **η — `_move_main` and `AdvanceOutcome.detail`** (C5). Modules: `git_ops.py` (`AdvanceDetail` union and its
  variants (ruling OQ-7), `ParkLockInfo`, `MainGate`, `MoveMainResult`, `_move_main` with C5's MARK/BYPASS cleanup
  rules, `_run_main_gate_hook`; `advance_main` and `recover_red_main` rewritten over it; 7 `_last_*` writes
  deleted), `merge_lane/gates.py::_map_advance_failure` (the 5 `getattr` reads deleted in this leaf (ruling
  OQ-8), replaced by a match on `outcome.detail`; its first parameters `(git_ops, result: str, …)` become
  `outcome: AdvanceOutcome`, the other keywords unchanged), `merge_lane/worker.py` (its 2
  `_map_advance_failure` callers), `recover_main.py` (reads `MoveMainResult`), tests `test_git_ops.py`
  (advance/recover sections), `test_merge_gates.py` for `_map_advance_failure`, `test_merge_queue_train_attribution.py` (from S6 ε's file list), new `test_move_main.py` (B16,
  B17, B17b). Deletes: `recover_red_main`'s protocol copy (6 hook sites → 1 helper, 2 CAS → 1, 3 read-tree →
  1 guarded sync), the `_last_*` side channel (writes and reads). Signal: B16, B17, B17b;
  `git grep -c "= getattr(git_ops, '_last_" gates.py` = 0. Prereqs: ζ (chain), β. Priority medium
  (fk-eb7b9f65f412, fk-3ee93d431321, fk-7a3ee979b48a; S6's fk-59b3df6f52a3 disposed here, ruling OQ-8). Edges:
  6892, 5298, 5042 and 6416 depend on η (S6's ε is deleted). ~850 LOC.
- **θ — the lane lease** (C7; cluster a). Modules: new `orchestrator/src/orchestrator/lane_lease.py`
  (`lane_lease`, `span_lock` (ruling OQ-5), `LaneLeaseContended`, `LaneLockSelfOwnedLeak`, `release_holder_if`, the 6
  lane-lock helper functions and `HELD_LANE_LOCK_FDS` moved from `git_ops.py`), `git_ops.py`
  (`merge_verify_lease`, `task_verify_lease`, `reset_persistent_merge_worktree`'s acquire → `lane_lease`),
  `cli.py` (`_merge_lane_gate`, `_verify_merge_body`, `cancel_verify` → the module; the compat co-lock becomes
  `span_lock`, its old-binary rationale and prose deleted), `verify_cancel.py`, `merge_lane/worker.py` (3
  handler sites renamed), `workflow_types.py` (the lease class imports), `merge_lane/shadow.py`,
  `merge_lane/drift.py` (exception name), tests `test_lane_lock_leak_guard.py`, `test_cli.py`,
  `test_merge_queue_concurrent_verify.py` (lease sections), new `test_lane_lease.py` (B18, B18b, B19).
  Deletes: the second and third copies of the protocol, ≈215 comment lines (`_merge_lane_gate` ≈70,
  `_verify_merge_body` ≈20, `merge_verify_lease` 75, `cancel_verify` 45 → one docstring), the rollout-only
  rationale for the span lock. Signal: B18, B18b, B19; `compat_fd` absent from `cli.py`; the `in-flight OLD` prose absent from `cli.py` and `git_ops.py`. Prereqs: η (chain),
  β. Priority medium (fk-20135ef54f75). Edges: 3077 and 3071 depend on θ. ~800 LOC incl. the 317-line move.
- **λ — the task-branch owner** (C8; cluster d). Modules: new `orchestrator/src/orchestrator/task_branch.py`
  (`TaskBranchRefs`, `OntoRef` = `MainTip | StackedOn | FrozenPrefix`, `delete`; the types the placement table
  gives it), `git_ops.py` (13 writer sites, 7 deleters and 30 prefix constructions → `self.branches`/`QueuedBranch`;
  `resolve_queued_branch_ref` deleted; `rebase_onto_main`, `rebase_preserving_task_commits`, `_stack_member`,
  `materialize_member_solo`, `rebind_branch_to_head`, `_swap_*`, `_rename_branch_aside`, `rename_worktree`
  route through the owner, and so does task 6219's `_rebase_requeued_onto_main` (its un-stack first, then the
  owner's rebase; `_is_train_successor` moves with `TrainMembership`); `_resolve_predecessor_tip` validated and
  a superseded `metadata['train']` cleared (ruling OQ-3, A′)), `branch_stack.py` (`rebase_own_delta` and its deleter → owner), `suffix_graph.py` (the frozen-tip
  bounce builds `MainTip()`/`FrozenPrefix(frozen_tip, main_sha)` locally; `frozen_prefix_tip` stays `-> str`), `merge_lane/worker.py::_do_train_merge`, `workflow.py` (3 direct
  `rebase_onto_main` callers; `BranchResetError`/`TrainMembership`/`is_wip_safety_commit` imports),
  `workflow_types.py` (`BranchResetError` import), `release_warm_worktree`'s deleter (in whichever module holds
  it, per S3 δ), tests `test_git_ops.py`, `test_rebase_branch_reset_guard.py`, `test_git_ops_unstack.py`,
  `test_suffix_conflict_tracker.py`, new `test_task_branch.py` (B23 per variant, B24), plus 4752's named test
  targets (`test_stack_onto_guard.py`, `test_merge_queue_bounce.py`, `test_workflow_train_worktree.py`).
  Deletes: try-both resolver, 30 inline prefix f-strings, the per-call-site guard copies 4752 would have added.
  Signal: B23, B24. Prereqs: θ (chain), λ0, ε, tasks 4841, 7341, 5697; 4752 is folded in (ruling OQ-3: the lead
  adds 4752 → λ, then cancels 4752). **Priority high**
  (fk-a7d433cf8308; fk-718c82fcc1a2). ~1,300 LOC.
- **ξ0 — worktree surface codemod** (`complexity='simple'`, `force_full_path` at the decomposer's call; D14; `metadata.files` = the test files naming ξ's methods — 41 at `d40ba048e9`, enumerated at decompose — + the src modules below). New `orchestrator/src/orchestrator/worktrees.py` with
  the types the placement table gives it and a `WorktreeOps` whose methods, for this one chain step, call the
  not-yet-moved `GitOps` bodies through the reference it is constructed with (typed by a local Protocol; no
  import of `git_ops`); `GitOps.worktrees` composed; every caller and test repointed to
  `git_ops.worktrees.*` and every importer of a moved type repointed (41 test files naming ξ's methods; `src`
  callers in `verify.py`, `harness.py`, `cli.py`, `workflow.py`, `workflow_types.py`, and
  `claim_warm_worktree`/`release_warm_worktree` in whichever module holds them). Enumerated at decompose by
  `git grep -l` over the moved method names plus AST `ImportFrom` for the types. Signal: `^class WorktreeOps`
  and `^class WorktreeKind` present in `worktrees.py`. Prereqs: λ (chain). Priority medium (fk-c00c5f01be4c).
  ~400 LOC.
- **ξ — h14 split I: worktree primitives below the pool cluster** (D14; cluster b). Modules: `worktrees.py`
  (the bodies move in: `worktree add/remove/prune` primitives, registration probes (`_is_registered_worktree` →
  `is_registered`), `ephemeral_worktree`, `create_interactive_worktree` + `reap_interactive_worktrees`'
  relocation half, `quarantine_worktree`, `rename_worktree`, `_clean_lane_retaining_artifacts`, `_freshen_main`;
  not the seed/script wrappers, which are γ's), `git_ops.py` (the moved methods deleted; module docstring
  rewritten per D14). No caller edit (ξ0 did them). Deletes: the retired-guard module docstring (38 lines), the
  "Task 4930 rewrote this message" note, `GitOps.commit`'s false `.task/` sentence, cluster (b) from
  `git_ops.py`. Moved size re-measured after γ at decompose (the author's 22 methods / 2,178 lines included the
  seed wrappers γ takes). Signal: B27 for `worktrees`; `git_ops.py` line count recorded in the commit message
  with the partition; `grep -c "No .task-specific guards remain" git_ops.py` = 0. Prereqs: ξ0 (chain), γ, θ, κ,
  λ. Priority medium (fk-c00c5f01be4c, fk-739af676767e, fk-d73aeaebb792 partition). ~2,000 LOC moved, ~200
  changed.
- **ο0 — main-landing surface codemod** (`complexity='simple'`, `force_full_path` at the decomposer's call; D14; `metadata.files` = the test files naming ο's methods — 99 at `d40ba048e9`, enumerated at decompose — + the 9 src modules below). New `orchestrator/src/orchestrator/main_landing.py`
  with the types the placement table gives it (`AdvanceOutcome`, the `AdvanceDetail` variants, `MergeResult`,
  the park errors, `PERSISTENT_MERGE_WORKTREE_NAME`, …) and a one-step forwarding `MainLanding`;
  `GitOps.landing` composed; every caller and test repointed to `git_ops.landing.*` — 99 test files and 9 `src`
  modules (`cli.py`, `merge_lane/{drift, gates, intake, liveness, shadow, worker}.py`, `recover_main.py`,
  `workflow.py`), measured at `d40ba048e9` — including `merge_lane/gates.py::_check_post_merge_pyright`'s reach
  into `_create_merge_worktree` (→ the public `landing.create_merge_worktree`; ο is that factory's one owner);
  every importer of a moved type repointed (`workflow_types.py`, `merge_lane/types.py`, `merge_lane/liveness.py`,
  `merge_lane/worker.py`). Signal: `^class MainLanding` and `^class AdvanceOutcome` present in
  `main_landing.py`. Prereqs: ξ (chain). Priority medium (fk-c00c5f01be4c). ~450 LOC.
- **ο — h14 split II: main landing** (D14; cluster e). Modules: `main_landing.py` (bodies move in: merge
  worktree create/cleanup/reap, `advance_main`, `_move_main`, `recover_red_main`, park/restore,
  `reset_persistent_merge_worktree`, `acquire_host_verify_worktree`, `merge_to_main`, `_index_lock_*`),
  `git_ops.py` (moved methods deleted). No caller edit (ο0 did them). Deletes: cluster (e) from `git_ops.py`.
  Signal: B27 for `main_landing`; `async def advance_main(` absent from `git_ops.py`. Prereqs: ο0 (chain), η.
  Priority medium (fk-c00c5f01be4c). ~3,300 LOC moved, ~150 changed.
- **ρ0 / ρ — h14 split III: the diff/query reads** (ruling OQ-9; D14; cluster f: 32 methods, 1,636 lines less
  ζ's parsers at `ce0f71a384`, plus `commits_named`, `_commit_paths` and `find_task_citation_commit_checked`
  since, less the deleted `get_diff_from_main`). ρ0 (`complexity='simple'`, `force_full_path` at the
  decomposer's call; `metadata.files` enumerated at decompose) creates
  `orchestrator/src/orchestrator/git_queries.py::GitQueries` with `CommitEffectProbe`, `ConflictProbe`,
  `RebasedCitation`, `task_citation_pattern`, `CitationPatternInvalid` and a one-step forwarding surface,
  composes `GitOps.queries`, and repoints callers, tests and type importers (`suffix_graph.py`,
  `agents/landed_sha_searches.py`, `workflow_types.py`, `scripts/restamp_degenerate_branch_bases.py`, the two
  `merge_lane_ratchet_raises/4911-*.json` entries), enumerated at decompose; ρ moves the bodies (state
  `project_root`, `_effect_probe_memo`). Signal: `^class GitQueries` present; `async def
  describe_commit_effect_in_main(` absent from `git_ops.py` (`get_diff_from_main`, the signal the fold chose,
  was deleted by task 5695); `git_ops.py` falls to ≈6.6k lines, the remaining partition — cluster (c) —
  named in ρ's commit record. Prereqs: ο (chain). Priority medium (fk-c00c5f01be4c). ρ0 ~400 LOC; ρ ~1,500
  LOC moved, ~100 changed.
- **μ — fused-memory done-provenance probes on `run_git`** (D12). Modules:
  `fused-memory/src/fused_memory/middleware/task_interceptor.py` (3 probes; a `git_runner` parameter threaded
  from `TaskInterceptor.__init__`, default `run_git`), `fused-memory/tests/test_task_interceptor.py` and
  `fused-memory/tests/_fm_git_repos.py`, the other test file that reaches the probes (measured at
  `e634cfe658`; tests pass a fake through the constructor) (B25). Deletes: 3 `create_subprocess_exec` spawns,
  their kill/reap drift, the dotted-name patch sites. Signal: B25; `grep -c create_subprocess_exec
  task_interceptor.py` = 0. Prereqs: none (shares no file with the chain). Priority low (fk-c04bf2045ad3).
  Edges: 5274, 6236 and S10's ν depend on μ. ~150 LOC.
- **ν — `TaskArtifacts` readers through the seam; typed architect exits** (D13). Modules: `artifacts.py`
  (`_read_model`, five `*Exit` dataclasses, the 9 non-plan readers on the seam — `read_lane_task_id`,
  `read_base_commit`, `update_base_commit`, `read_blocking_dependency`, `read_already_done`,
  `read_ready_to_merge`, `read_unactionable_task`, `read_false_premise`, `read_reviews` — and the module
  docstring, line 1 included; ν is its one owner), `workflow.py` (15 consumer sites), `evals/metrics.py` (name
  table), tests `test_artifacts.py`, `test_workflow_already_done.py`, `test_workflow_unactionable.py`,
  `test_workflow_false_premise.py`, `test_workflow_blocking_dep.py` (B26). Deletes: 9 raw `json.loads`
  readers, 5 untyped dict contracts. `read_iteration_log` is S9 C2's and untouched. Signal: B26. Prereqs: ρ
  (chain — ν shares `workflow.py` with every codemod leaf, so it runs after them; task 3155, a prerequisite
  at authoring, landed 2026-11-02); S9 γ (cross-stream same-file edge on `artifacts.py` and
  `test_artifacts.py`: S9's draft lands its `read_iteration_entries` first, D18; S9 γ waits only on S9 β and
  α, so no cycle). Priority low
  (fk-78b757799cab). ~450 LOC.
- **π — integration gate** (signal leaf). Modules: `orchestrator/tests/test_gitops_boundary_gate.py` (new;
  runs B1–B28 as one suite over a real tmp repo + `ScriptedGitRunner`), `scripts/merge_lane_metrics.py` (the
  patch-target measure extended to `orchestrator.git_ops`, `orchestrator.worktrees`, `orchestrator.main_landing`,
  `orchestrator.task_branch`, `orchestrator.lane_lease`, `orchestrator.git_queries`, or a separate measure
  beside `LANE_PACKAGE` / `lane_patch_modules`),
  new `scripts/check_gitops_private_patch_targets.py` (runs `scripts/source_measures.py::patch_targets` over
  those modules across `orchestrator/tests` and `escalation/tests` plus the instance-form count, exits non-zero
  on any `_`-prefixed leaf or instance-form site), `orchestrator/tests/merge_lane_ratchet_baseline.json`,
  `orchestrator/tests/test_merge_lane_ratchet.py` (a private-read scan over `GitOps` attributes from other
  `orchestrator/src` modules, modelled on `patch_targets`), the heuristic-13 import tripwire (B27) as a tracked
  test. Deletes: nothing; it proves the deletions. Signal: B28 and the §Goal bullets, re-measured and recorded in
  the commit message, with the heuristic-14 residual. Prereqs: every other leaf (ν last on the chain) except
  the follow-up σ. Priority medium. ~350 LOC.
- **σ — follow-up: the re-entrant lane lease** (ruling OQ-5, A′; the lead's sequencing ruling of 2026-11-10
  files it with this plan; not part of π's signal). Design first, then build: redesign the self-owned-leak
  probe so it tells a lease-token holder from a leak; `lane_lease` yields a `LaneLeaseToken`;
  `reset_persistent_merge_worktree` (in `git_ops.py`, or `main_landing.py` once ο has moved it — whichever
  module holds it at σ's dispatch) accepts it and skips its own acquire; the
  CLI holds the lane lock continuously from `_merge_lane_gate` through `_verify_merge_body`'s build; the span
  lock, `span_lock` and the `.merge_verify.lock` file (`verify_cancel.py::MERGE_VERIFY_LOCK_FILENAME`) are
  deleted, with the operator messages that name it (`merge_lane/worker.py`, `verify_runner.py`). Modules
  (enumerated at decompose): `lane_lease.py`, `main_landing.py`, `git_ops.py` (`merge_verify_lease`,
  `GitOps._lane_lock_self_owned_leak`), `cli.py`, `verify_cancel.py`, `verify_runner.py`,
  `merge_lane/worker.py`; tests `test_lane_lease.py` (B18b re-run with no span lock; a token holder is not a
  leak; a genuine leak still raises `LaneLockSelfOwnedLeak`), `test_lane_lock_leak_guard.py`, `test_cli.py`.
  Signal: B18b's postcondition holds with the span lock gone; `^class LaneLeaseToken` present in
  `lane_lease.py`; the `'.merge_verify.lock'` assignment absent. Prereqs: θ only (the lead's ruling after
  the fold-verifiers, 2026-11-10). σ is low and no leaf waits on it; against the chain leaves after θ that
  share its files (`git_ops.py`, `cli.py`, `merge_lane/worker.py`, `main_landing.py`), whichever lands second
  adapts. Notes: 3071 reacts to the probe σ redesigns and 3077 replaces `task_verify_lease` call
  sites; both depend on θ and whichever lands second adapts. Priority low (no finding key: a follow-up of
  fk-20135ef54f75, which θ closes). ~600 LOC (not measured; the sitting sized A′ as larger than θ's ~800
  with the probe redesign inside).

Dependency order (B4: one total order, so every pair of leaves that edits the same file is serialised by a
real edge — the overlay's same-file rule): **α → λ0 → β0 → β → γ → δ → ι → κ → ε → ζ → η → θ → λ → ξ0 → ξ →
ο0 → ο → ρ0 → ρ → ν → π** (λ0 second by the lead's sequencing ruling of 2026-11-10; α and λ0 each also
depend on S6 α, which lands first on the files they share with it). μ shares no file with
the chain and stays independent (π depends on it). The follow-up σ depends on θ only and nothing depends on
it. Every semantic prerequisite stated per leaf is implied by the chain. The
shared files that forced the total order (measured on the author's edges, critic B4): `git_ops.py` (γ∥ε, δ∥ε,
δ∥ζ, δ∥λ, ι∥ε, ι∥λ, κ∥ζ, κ∥λ, ζ∥λ, ζ∥ξ, ζ∥ο), `harness.py` (γ∥ε, ι∥ε), `workflow.py` (ν∥α, ν∥ε, ν∥λ),
`cli.py` (γ∥θ, ι∥θ), `merge_lane/gates.py` (η∥ζ, ζ∥ξ), `test_git_ops.py`. Cross-stream editors of these files
are in §Cross-PRD and need an edge or an xref at decompose.

Deletable list (synthesis) accounted for: 7 (now 10 modules / 14 sites) external importers of `_run` → α (9
modules) and S3 δ (`escalation/server.py`); 135 test imports → β0; 3 fused-memory git spawns → μ; 3 of 4 diff
parsers → ζ; the 9-arm chain and FAULT literal list → δ; `recover_red_main`'s protocol copy → η;
`verify_merge`'s comment lines → θ; 128 (now 159) private patch sites → β, γ, η, ι, π;
`_STEAL_RETRYABLE` → δ.

### G7 walk (advisory at author time; re-run at decompose)

- INV-2 structured-facts-at-failure: `GitCommandFailed` (argv, cwd, rc, stderr), `BranchResetError`, `LaneLeaseContended` carry values (a composed lookup's `GitQueryError` carries its reason);
  the FAULT `RuntimeError` names the producers from the table.
- INV-5 no-lockstep-duplication: the enum table replaces three encodings; `lane_lease` replaces three
  copies; `_move_main` two; `parse_unified0` four; `worktree_work`/`branch_work` nine; `_ensure_pool_storage` five; one `GitQueryError` replaces `GitQueryFailed` and the hand-rolled empty values.
- INV-7 holds-owned-and-bounded: `lane_lease`'s bounded wait (`GitOpsLimits.merge_verify_lease_wait_secs`)
  and its holder record (pgid file written iff `holder_rendezvous`, removed in `finally`) and the span lock's
  scope (gate release → build release) are the owned, bounded holds; B18b and B19 pin them.
- INV-9 one-fact-one-home: lock-lifetime rationale lives once in `lane_lease`; the partition measurement
  lives in commit records, not the module docstring; this PRD restates no heuristic.
- INV-11 no-silent-fail-soft: every former empty-value-on-failure path raises or is explicitly directional
  (ε's two tables, ζ's count table).
- INV-12 exceptions-owned-or-ratified governs the dispositions of allow-list and suppression entries: every
  entry a leaf adds under `orchestrator/tests/merge_lane_ratchet_raises/` (one file per entry since task
  7298; for `GitQueryError`/`GitCommandFailed`, `BranchResetError`, `LaneLeaseContended`) carries
  `Debt(owner)` or `Policy(ratified)`; no new suppression.
- INV-13 readers-prove-their-producer: `LaneRecords.corrupt` makes the reader honest about what it dropped.
- No waiver requested.

## Out of scope

- The reify-side warm-lane shell scripts and their exit-code vocabulary (fk-856541951d65's cross-repo half);
  δ makes the DF side a table and `WarmLaneScripts.supports` the one capability probe.
- rerere ownership (task 5697 carries it; D15); sandboxed-agent rerere (task 5844, done).
- Merge-lane decomposition (S6, R1), `GateMeasurement` (S7, R11), `VerifyContext` (S4, R4), Harness as
  composition root and `SlotRegistry` (S8, R6), `BlockReason`/typed Plan model and `read_iteration_entries`
  (S9, R2).
- The seven other `create_subprocess_exec` sites in fused-memory (`pre_done_hook.py`, `reconciliation/verify.py`,
  `task_knowledge_sync.py`) — S12/S10.
- A shared `WorktreeEntry` parser for the three non-orchestrator porcelain readers.
- Task 5144's admission-time citation guard (stays, independently).
- The re-entrant lane lease (OQ-5 A′) is not part of θ or of π's signal: it is the follow-up σ, filed at
  decompose at low priority and depending on θ (lead sequencing ruling, 2026-11-10).
- `escalation/server.py::release_warm_worktree`'s runner repoint (S3 δ).

## Open questions (tactical)

1. **`FAULT` sub-causes.** `from_seed_rc(127)` ("script absent") maps to `FAULT` today via inline checks
   with a distinct message. Decide in δ whether `FAULT` carries a `detail` string only (recommended — same
   disposition, no new member) or a `SCRIPT_ABSENT` member is added (would change the disposition table).
2. **`merge_to_main`'s `.task/` removal.** ξ keeps it as the one belt-and-braces guard and states it
   truthfully; deleting it is a separate, measured decision.
3. **`ScriptedGitRunner` matching.** Argv-prefix match vs regex; decide in β by what the patch-form test files
   need (`test_verify_main_tip_sweep.py` dispatches by ordinal call count today — prefer argv match so adding
   one internal call does not reorder a script).
4. **`GitOpsLimits` for clusters (e)/(a) after ξ/ο.** `MainLanding` and `lane_lease` take the limits they need
   as parameters; whether `GitOpsLimits` is split per collaborator is decided in ο.
