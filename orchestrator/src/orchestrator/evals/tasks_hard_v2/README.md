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

- **`referenced`** — the candidate has exactly one clean
  `Merge task/<id> into main` commit, so `(pre, post)` = `(M^1, M)` resolves
  and a real `reference` block is captured.
- **`planrate_only`** — the candidate landed SPLIT/direct (no single merge
  commit), so no landed `post` SHA exists. These fixtures carry **no**
  `reference` key at all and instead stamp `provenance.reference_unavailable`
  naming the cause, plus `provenance.baseline_source` naming which rung of the
  baseline ladder produced their `pre_task_commit`. The omission is a positive
  recorded fact, not a silent empty block.
