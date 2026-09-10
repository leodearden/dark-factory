# `tasks_hard_v2` — the fable-trial-v2 curated hard fixture pool

This directory holds the **fable-architect-trial-v2 hard pool**. It is **NOT**
the standing eval corpus. The standing corpus lives one directory over, in
`orchestrator/src/orchestrator/evals/tasks/`, and is byte-unchanged by the task
that minted this pool (β1 / task 3631).

## What is in here

| Path | What it is |
|---|---|
| `<id>.json` | Minted fixture records, canonical `evals/tasks/*.json` schema |
| `CURATION.md` | **Generated** human-readable curation table — never hand-edit |
| `_meta/curation.json` | Machine-readable single source of truth (census + candidates + ceilings + continuity) |
| `README.md` | This file |

`CURATION.md` is rendered from `_meta/curation.json` by
`scripts/mint_hard_v2_fixtures.py`'s pure `render_curation_md()`, and a test
asserts the re-render is byte-identical to the committed file — so the human
table can never silently drift from the machine manifest. Edit
`_meta/curation.json` and regenerate; do not edit `CURATION.md` by hand.

Regenerate with:

```bash
python3 scripts/mint_hard_v2_fixtures.py --render   # manifest -> CURATION.md
python3 scripts/mint_hard_v2_fixtures.py --redrive  # re-derive provenance
python3 scripts/mint_hard_v2_fixtures.py --base-distance-report
```

`--render` reads only the committed manifest and touches no database. Do NOT
use `--author` for this: it re-derives the whole manifest from the three live
`runs.db` files and refuses on any census drift — and additive drift (new
architect exhaustions, which keep landing) is expected and harmless — so it is
not a regeneration path and would overwrite a manifest edit rather than render
it.

`--redrive` is the regeneration path after a provenance-RESOLUTION fix (a
better landing-merge matcher, a corrected baseline rung). It re-derives only
`merge_sha` / `baseline_sha` / `baseline_source` / `mint_mode` on the rows the
committed manifest already carries, so by construction it cannot add or drop a
fixture or re-adjudicate one — which is exactly why `--author` is wrong for
this: `--author` re-censuses against `runs.db` files that have moved on since
the recorded census date, and would change pool membership as a side effect of
a bug fix. See `scripts/mint_hard_v2_fixtures.py::redrive_provenance`. Follow a
redrive with `--mint` and `--render`.

`--base-distance-report` prints the per-fixture table of how far each
`pre_task_commit` sits from the task's true branch point, measured as the
symmetric difference `git rev-list --count <branch-point>...<base>` (both plain
and `--first-parent`, which differ by roughly 3x on this history). It REPORTS
measured distances and asserts nothing against a threshold. See
`scripts/mint_hard_v2_fixtures.py::base_distance_rows`.

It is a **pre-redrive** tool: `after` is always what a redrive would produce
right now, so run it BEFORE `--redrive` and it previews (and records) the
movement; run it after and it compares the redriven rows with themselves and
correctly reports none. The measurement that motivated the mode therefore does
not live in a live run — it is committed at
[`_meta/base-distance-report.md`](_meta/base-distance-report.md), whose header
carries the exact two commands that reproduce it. Pass `--before-manifest
<path>` to read the `before` side from an earlier manifest (extracted from git
history with `git show <rev>:…/curation.json`) instead of the committed one;
that is what makes the artifact regenerable rather than hand-transcribed.

## Why this directory can never leak into a default eval run

`cli._load_fixture_dir` (`orchestrator/src/orchestrator/cli.py`) globs
`*.json` **non-recursively** against a single `tasks_dir`:

```python
for tp in sorted(tasks_dir.glob('*.json')):
```

Two consequences this layout depends on:

1. A default eval run resolves `tasks_dir` to `evals/tasks/`. A *sibling*
   directory is never globbed, so no fixture in here is reachable from a
   default run. Isolation is structural, not conventional — and a test pins
   it (`test_hard_v2_fixture_pool.py`: the `tasks/` id-set is still exactly
   its 22 pre-existing ids and is disjoint from this pool).
2. The same non-recursiveness is why the curation manifest lives in the
   `_meta/` **subdirectory**. A top-level `curation.json` would be globbed and
   loaded as a malformed fixture by any consumer pointed at this dir; inside
   `_meta/` it is invisible to the glob.

## Who consumes it

Only the fable-trial-v2 campaign driver (β2), via an **explicit**
`--tasks-dir orchestrator/src/orchestrator/evals/tasks_hard_v2`. Nothing
resolves this path by default.

## Fixture-side ceilings

Every fixture in here carries `max_architect_turns` and `timeout_minutes`
overlaid from `_meta/curation.json`'s `ceilings` block. Both keys are read
straight off the task record by `evals/runner.py` (`task.get(...)` with 50 /
60 defaults), so pinning them is pure data — no runner change. The derivation
of `timeout_minutes` and its headroom multiples are recorded in `CURATION.md`.

## Two mint modes

A **landing merge** is written under either of two subject spellings, and both
are accepted — `Merge task/<id> into main` and `Merge task/<id>: <subject>`.
Both are in live use (censused over reify's `main`: 2741 of the first, 74 of
the second), and the single derivation both are matched from is
`scripts/mint_hard_v2_fixtures.py::_merge_subject_patterns`, mirroring
`orchestrator/src/orchestrator/git_ops.py::_merge_subject`'s
writer-and-reader-share-one-derivation discipline.

Two scopings keep a commit that never landed from answering as a landing
merge, both in `::find_merge_sha`. The walk is over **`main`**, not `--all`, so
a merge living only on a side or remote branch cannot answer at all. And
`--grep` is used only as a coarse pre-filter, with each candidate's **subject**
re-tested against the same pattern in Python: git applies `^`/`$` per LINE
across the whole message, so a body line quoting a merge subject would
otherwise match. Both mirror
`orchestrator/src/orchestrator/git_ops.py::GitOps.find_task_citation_commit`,
which defends the same trap for the same reason. Measured over all 39 include
rows, the three matcher variants agree on every row — this is hardening, not a
correction.

- **`referenced`** — the candidate has exactly one clean landing merge under
  either spelling, so `(pre, post)` = `(M^1, M)` resolves and a real
  `reference` block is captured.

  The landed `verify_outcome` these carry is MEASURED, not assumed:
  `provenance.post_commit_reachable_from_main` records `git merge-base
  --is-ancestor` at mint time, and a `false` there downgrades the outcome to
  `{source: 'landed_branch_tip', passed: null, …}` exactly as the continuity
  path does. The reference diff is unaffected — it is captured from the
  pre/post SHAs directly — so only the gate claim is withdrawn.
- **`planrate_only`** — the candidate landed SPLIT/direct (no single landing
  merge under EITHER spelling), so no landed `post` SHA exists. These fixtures
  carry **no** `reference` key at all and instead stamp
  `provenance.reference_unavailable` naming the cause, plus
  `provenance.baseline_source` naming which rung of the baseline ladder
  produced their `pre_task_commit`. The omission is a positive recorded fact,
  not a silent empty block.

  For the same reason they also carry **no landed `verify_outcome`**. The
  `{source: 'landed', passed: true}` shape asserts "the task merged to `main`
  ⇒ its gates passed at the post commit"; with no post commit that premise
  does not hold, and for the one cancelled include (`reify_task_3586`) the
  task never landed at all. These fixtures instead carry
  `{source: 'unavailable', passed: null, commands: …, reason: …}`, with the
  candidate's adjudicated terminal status mirrored into
  `provenance.task_status` so the JSON is self-describing.

## Is the base a true branch point, or an approximation?

Every fixture carries a bool `provenance.base_is_approximated`, and every
approximated one carries a `provenance.base_approximation_reason` naming what
was measured. **Only `baseline_source: merge_first_parent` is the task's true
branch point** (`M^1` of its landing merge — the exact state of `main` the work
started from). Every weaker ladder rung is an approximation whose error is
unbounded, and unknowable without a landing merge to measure against:
`status_autocommit` anchors on a status auto-commit that can precede or follow
the real start, and `timestamp_walk` walks back to the FIRST architect
invocation, which for a re-tried task can be days and thousands of commits
before the run that produced the work.

A readout that depends on the base being the real branch point should EXCLUDE
approximated fixtures rather than average them in. Some fixtures can never be
rescued: `reify_task_3883` has no landing merge under either spelling, so its
branch point is not derivable from git at all — which is precisely why the flag
exists rather than a silent second guess.

Continuity fixtures do not go through the rung table at all: their base is
inherited verbatim from the standing corpus, so
`scripts/mint_hard_v2_fixtures.py::_mint_continuity_one` MEASURES the flag —
does the inherited `pre_task_commit` equal `M^1` of the task's landing merge? —
and records the merge it compared against in
`provenance.base_verified_against_merge` (`null` when none exists). All three
were measured to diverge. Their commits are still carried verbatim; the marking
is observational, never corrective.
