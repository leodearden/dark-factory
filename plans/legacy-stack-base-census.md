# Legacy stack-base census, 2026-10-28 (task 6125)

Task 5618 records the base a branch is stacked onto at the moment of stacking
(`orchestrator/src/orchestrator/branch_stack.py::StackBaseLedger`), and merge
admission strips it again
(`orchestrator/src/orchestrator/git_ops.py::GitOps.unstack_from_unlanded_base`).
Branches stacked before that record existed carry no record. This is the
read-only census of them, produced by `scripts/stack_base_census.py`.

## The run

- Command, from the task worktree on `task/6125` at the commit "fix(6125): a
  sibling at the very same commit is not a contained sibling". It is cited by
  subject because every rebase of the branch gives it a new SHA:
  `uv run --project shared python scripts/stack_base_census.py --project-root /home/leo/src/dark-factory`
- READ-ONLY: `--adopt-confirmed` was NOT passed. Without it the census issues
  no write command. After the run, `refs/dark-factory/stack-base/` held seven
  records, and the census reported every one of them RECORDED.
- Started 2026-10-28T18:01:03Z against `main` at `145ab68f8ea37228f87b9795ccbc825abef76fd7`.
- Exit code 0. Census elapsed 49.5s; wall clock 49.8s; max RSS 62 MB (`/usr/bin/time -v`).
- The task store was readable from the sandbox: no warning on stderr, so
  task statuses were applied. Branches whose task is done or cancelled are INERT.
- 639 unlanded task branches, 104 findings: 1 confirmed, 3 stranded_base,
  0 unadjudicated, 18 already_clean, 7 recorded, 75 inert.

### Re-run after the review amendments, 2026-10-29

The review amendments (the `amend:` commits on `task/6125`) changed three
verdict rules. A base's owner must still hold the base as its own work. A base
that git cannot read is UNADJUDICATED. So is a record that names no commit. The
same read-only command then ran again from the amended worktree:

- Finished 2026-10-29T01:40Z against `main` at `914da890003937187f46b4ce283acfdee8e844e2`.
- Exit code 0. Census elapsed 60.3s; wall clock 61.4s; max RSS 60 MB.
- `refs/dark-factory/stack-base/` held the same ten records before and after the run.
- 643 unlanded task branches, 106 findings: 1 confirmed, 3 stranded_base,
  0 unadjudicated, 16 already_clean, 10 recorded, 76 inert.
- The four actionable findings below are unchanged: same branches, bases,
  verdicts and owner. The other counts moved because main and the branch set
  did. The recorded set is now `task/4763`, `task/4847`, `task/5612`,
  `task/5758`, `task/5766`, `task/6018`, `task/6128`, `task/6163`, `task/6387`
  and `task/7502`.

## What it found

**No `task/<id>` branch of a live task is stacked without a 5618 record.**
All four actionable findings are hand-made backup snapshots. Their names are
not `task/<digits>`, so they have no task id, and the census adjudicates them
as live by design (an unknown status is the conservative reading).

| branch | verdict | measured |
|---|---|---|
| `task/3493-preserved-20261027` | stranded_base | reflog: `branch: Created from cd4d5a5db6`; tip == base; 9 commits not on main; `task/3493` no longer exists; task 3493 is done |
| `task/3508-pre-rebase-backup` | stranded_base | reflog: `branch: Created from HEAD`; tip == base; 16 commits not on main; `task/3508` no longer exists; task 3508 is done |
| `task/3580-pre-rebase-backup` | confirmed | reflog: `branch: Created from 8bee6520df`; tip == base; 13 commits not on main; `task/3580` points at the same commit (its live owner); task 3580 is done |
| `task/5033-prerebase-backup` | stranded_base | reflog: `branch: Created from HEAD`; tip == base; 23 commits not on main; `task/5033` no longer exists; task 5033 is done |

Each was created at an off-main commit and has no commit of its own, so its
cut equals its tip: every commit on it is the base's. For the three stranded
ones, that base has no other live carrier. Keeping the only copy of those
commits is what a backup branch is for.

`--adopt-confirmed` would write exactly one record, for the single CONFIRMED
finding, `task/3580-pre-rebase-backup`. `StackBaseLedger.prune_orphans` drops
that record once the branch is deleted.

Hypothesis: none of the four is ever a merge candidate, because each backs up a
done task under a name no task id maps to. If so, the one record has no effect.

The other findings need no action:

- **recorded (7):** `task/4763`, `task/4847`, `task/5612`, `task/5758`,
  `task/5766`, `task/6140`, `task/6387`. These are every record that exists
  today. The planning pass on main `d53a0e9769` counted eight, so one has been
  pruned or consumed since.
- **already_clean (18):** stacked once, but no longer carrying any of that base's
  commits.
- **inert (75):** stacked-looking branches of done or cancelled tasks. They are
  never adjudicated, because they never merge again. This includes `task/2943`,
  the task's original first hit. Its newest reflog entry reads
  `rebase (finish): refs/heads/task/2943 onto 2495ac3d36`, which is still
  `task/5441`'s tip. Both tasks are done.

## Stranded carriers (task item 3)

From the task store and refs, 2026-10-28:

- **4480**: done (merged 2026-09-22). `task/4480` still exists, with 15 commits
  not reachable from main. It is not stranded.
- **5708**: cancelled. Its two commits remain only on `task/5708`
  (`182288d5ff`, `git rev-list --count main..task/5708` = 2).
- **4763**: pending. Its dependencies 4754 and 4755 are done, and it holds
  `refs/dark-factory/stack-base/task/4763`, so this census reports it RECORDED and
  5618's admission un-stacks it.

Nothing is left to schedule. The STRANDED_BASE verdict surfaces any further
carriers on a re-run.

## What remains: gate task 7661

A dispatched agent's write-set admits only `refs/heads/task/` in the shared
`.git` (`orchestrator/src/orchestrator/agents/write_set.py`). An implementer
therefore cannot write `refs/dark-factory/stack-base/*`, so the live adoption is
the operator step, task 7661. It re-runs the census on the then-current main,
decides whether to run `--adopt-confirmed` (on today's evidence, that writes the
one inert record above), and adjudicates any STRANDED_BASE or UNADJUDICATED
finding the re-run reports.

## Limits of the evidence

- Direction comes from each branch's own reflog. A stacking event older than
  the reflog's expiry, or a branch moved with `git update-ref`, leaves no event.
  Containment of a sibling's tip then surfaces it as UNADJUDICATED instead.
- A sibling at the very same commit is not counted as contained: neither branch
  has a commit the other lacks.
- A base's owner is another branch that holds the base as its own work. Either
  it points at the base, or its reflog once held the base and its tip still
  carries every unlanded commit of it, as an ancestor or a patch-equivalent. A
  branch stacked onto the same base is no owner. With no owner, the verdict is
  STRANDED_BASE, never CONFIRMED.
- INERT trusts the task store's status. A branch with no numeric task id, or no
  row, is adjudicated as live.

## The rendered report

### Legacy stack-base census

- main: `main` at `145ab68f8ea37228f87b9795ccbc825abef76fd7`
- unlanded task branches: 639
- findings: 104
- elapsed: 49.5s

| verdict | branches |
|---|---|
| confirmed | 1 |
| stranded_base | 3 |
| unadjudicated | 0 |
| already_clean | 18 |
| recorded | 7 |
| inert | 75 |

#### Actionable findings

##### task/3493-preserved-20261027: stranded_base

- task status: unknown, adjudicated as live
- base: `cd4d5a5db60ac42dce5f117830081426f322b640`
- live owners of the base: none
- cut: `cd4d5a5db60ac42dce5f117830081426f322b640`
- contained siblings: none
- stacking events, newest first: created on `cd4d5a5db60ac42dce5f117830081426f322b640`
- remedy: no live branch holds the base any more, so un-stacking would drop the only copy of the base's commits. Decide first whether they must land; once adjudicated, record the base with `git update-ref refs/dark-factory/stack-base/task/3493-preserved-20261027 cd4d5a5db60ac42dce5f117830081426f322b640`

##### task/3508-pre-rebase-backup: stranded_base

- task status: unknown, adjudicated as live
- base: `9f61e4e8f9ac4894258c51f654d7fc93d50cc393`
- live owners of the base: none
- cut: `9f61e4e8f9ac4894258c51f654d7fc93d50cc393`
- contained siblings: none
- stacking events, newest first: created on `9f61e4e8f9ac4894258c51f654d7fc93d50cc393`
- remedy: no live branch holds the base any more, so un-stacking would drop the only copy of the base's commits. Decide first whether they must land; once adjudicated, record the base with `git update-ref refs/dark-factory/stack-base/task/3508-pre-rebase-backup 9f61e4e8f9ac4894258c51f654d7fc93d50cc393`

##### task/3580-pre-rebase-backup: confirmed

- task status: unknown, adjudicated as live
- base: `8bee6520dfd9301722d9706ffdd535c55a30f0bf`
- live owners of the base: task/3580
- cut: `8bee6520dfd9301722d9706ffdd535c55a30f0bf`
- contained siblings: none
- stacking events, newest first: created on `8bee6520dfd9301722d9706ffdd535c55a30f0bf`
- remedy: run `scripts/stack_base_census.py --adopt-confirmed` to record the base, so task 5618's merge admission un-stacks the branch; by hand: `git rebase --onto main 8bee6520dfd9301722d9706ffdd535c55a30f0bf task/3580-pre-rebase-backup`

##### task/5033-prerebase-backup: stranded_base

- task status: unknown, adjudicated as live
- base: `abeea023fc07ca7e9bee74bef96cb58797b27ae8`
- live owners of the base: none
- cut: `abeea023fc07ca7e9bee74bef96cb58797b27ae8`
- contained siblings: none
- stacking events, newest first: created on `abeea023fc07ca7e9bee74bef96cb58797b27ae8`
- remedy: no live branch holds the base any more, so un-stacking would drop the only copy of the base's commits. Decide first whether they must land; once adjudicated, record the base with `git update-ref refs/dark-factory/stack-base/task/5033-prerebase-backup abeea023fc07ca7e9bee74bef96cb58797b27ae8`

#### Not actionable

- already_clean (18): task/3155, task/3321, task/5172, task/5179, task/5385, task/5453, task/5737, task/5861, task/5925, task/5958, task/6057, task/6062, task/6094, task/6103, task/6201, task/6738, task/6741, task/7614
- recorded (7): task/4763, task/4847, task/5612, task/5758, task/5766, task/6140, task/6387
- inert (75): task/2943, task/3147, task/3168, task/3212, task/3247, task/3381, task/3420, task/3561, task/3594, task/3701, task/3736, task/3744, task/3771, task/3803, task/3815, task/3830, task/3883, task/4047, task/4176, task/4188, task/4211, task/4216, task/4223, task/4234, task/4258, task/4305, task/4347, task/4356, task/4395, task/4413, task/4488, task/4514, task/4537, task/4545, task/4547, task/4591, task/4668, task/4766, task/4935, task/4978, task/5119, task/5174, task/5237, task/5279, task/5445, task/5458, task/5546, task/5564, task/5576, task/5614, task/5624, task/5765, task/5783, task/5792, task/5989, task/5990, task/6025, task/6046, task/6405, task/6414, task/6586, task/7189, task/7218, task/7229, task/7232, task/7233, task/7257, task/7281, task/7287, task/7293, task/7328, task/7412, task/7419, task/7477, task/7480
