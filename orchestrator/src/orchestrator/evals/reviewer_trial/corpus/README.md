# reviewer_trial corpus

Evaluation corpus for the reviewer-panel trial (`orchestrator/evals/reviewer_trial`).
Originally 15 hand-authored diffs (12 synthetic mutations + 3 real-world bugs);
expanded under **task 2495** (PRD `tier1-prompt-optimization` T4 / D-6) to
≥50 diffs (aim ~100) by *mining* additional diffs from this project's own
orchestrator run history. This file documents the corpus schema, the mining
pipeline that produced the `source: "mined"` diffs, the train/selection/test
split, and the ground-truth adjudication protocol.

## Layout

```
corpus/
  manifest.json          # diff index: id, language, source, description, project, split
  annotations/<id>.json  # ground_truth issues (+ provenance, for mined diffs)
  synthetic/<id>.diff     # source == "synthetic" diff text
  real_world/<id>.diff    # source == "real_world" diff text
  mined/<id>.diff         # source == "mined" diff text
  adjudication_log.jsonl  # one AdjudicationEntry per diff_id (see below)
  README.md               # this file
```

`CorpusManifest.load()` (`corpus.py`) stitches these together: it reads
`manifest.json` for the diff index, then resolves each diff's text from
`<source>/<diff_id>.diff` and its ground truth (+ provenance) from
`annotations/<diff_id>.json`.

## Schema

**`manifest.json`** — `{"version": "1.0", "split_seed": "<seed>", "diffs": [...]}`.
Each entry: `{diff_id, language, source, description, project?, split?}`.

- `language` — `"python" | "rust" | "typescript"`.
- `source` — `"synthetic"` (planted mutation) | `"real_world"` (hand-picked
  real bug, pre-task-2495) | `"mined"` (task-2495 FN mining, see below).
- `project` — optional; selects the working directory a reviewer would see
  around the diff (`corpus._PROJECT_CWD_MAP`). All mined diffs use
  `"dark-factory"`.
- `split` — `"train" | "selection" | "test"`, assigned by
  `mining.assign_split()` (see "Split" below). Every diff has one.

**`annotations/<diff_id>.json`** — `{diff_id, ground_truth: [...], provenance?}`.
Each `ground_truth` entry is a `GroundTruthIssue`:
`{id, location, category, severity, description, mutation_type}`
(`severity` is `"blocking" | "suggestion"`). `provenance` is present only for
mined diffs (see "Mining provenance" below).

## Split (train / selection / test, 2:1:7)

Every diff carries a `split`, assigned by `mining.assign_split(diff_ids, seed)`
over the **full** diff_id set (original 15 + all mined diffs together) using a
stable hash of `f"{seed}:{diff_id}"` sliced into cumulative 2:1:7 buckets —
deterministic (same ids + seed always reproduce the same assignment) and
reproducible across machines/runs. The seed actually used to produce the
committed split is recorded at the top level as `manifest.json`'s
`split_seed`.

Per PRD §2, the **test** split should be scored once (not iterated against
during prompt/variant tuning) — use `train`/`selection` for iteration and
reserve `test` for a final read.

## Mining pipeline (task 2495 / PRD D-6)

The `mined` diffs are **false-negative (FN) candidates**: tasks whose review
phase passed but where a bug surfaced downstream. They are produced by the
`mine` CLI subcommand (`__main__.py`), which is **offline and re-runnable**
but reads from data that lives outside this repo's git history:

- `data/orchestrator/runs.db` (`task_results` + `events` tables) — the main
  repo's `.gitignore`d orchestrator run history.
- `data/escalations/*.json` — the main repo's `.gitignore`d escalation
  records (including rotated `archive/<date>/` subdirectories).

These are **build-time-only inputs**: nothing in the committed corpus depends
on them at runtime, and every unit test in `mining.py`/`adjudication.py`
exercises the machinery exclusively against synthetic fixtures
(`tests/_reviewer_trial_mining_fixtures.py`), never the real gitignored data.
Only the *output* of running `mine` — diff text, ground-truth labels,
provenance, and the adjudication log — is committed.

### FN heuristic

A task is an FN candidate (`mining.mine_fn_candidates`) when its review phase
ran (`review_cycles >= 1`) **and** at least one downstream bug signal
follows:

- `outcome` in `{"requeued", "blocked"}`, or
- `verify_attempts >= 2`, or
- an `escalation_created` event exists for the task.

Each candidate's `signal_reason` names every signal that fired.

### Diff recovery

A candidate's diff text is recovered from git:

1. `events(merge_finalized).data.merge_sha`, when present, or
2. `mining.resolve_merge_sha_by_task_id()` — a fallback grep over
   `git log --all` for this repo's `"Merge task/<id> into <branch>"` commit
   message convention, for candidates whose row predates (or otherwise
   lacks) a `merge_finalized` event.

`mining.recover_diff()` then runs `git show`/`git diff <sha>^1 <sha>` against
the recovered sha. Candidates with no resolvable sha, or no diff output, are
skipped.

### Frontier labeling

Ground truth for a mined diff is **proposed by a frontier model** (opus, via
`mining.propose_labels_frontier()` — reuses the `scorer.match_issues`
`invoke_agent` pattern with `effort="high"`), never fabricated by the mining
script itself. The model is prompted with the diff text (truncated to the
first 10K characters — see "Known limitation" below) and the FN context (task
title + `signal_reason`), and asked to list concrete issues.

A candidate is only added to the corpus if the frontier model proposes at
least one issue; candidates with an empty proposal are skipped (not added,
but eligible to be retried by a future `mine` run since they were never
recorded as "in the corpus").

**Known limitation:** many mined merge-commit diffs are far larger than the
10K-character prompt window (they bundle a task's full implementation, not
just the buggy hunk). `mine` prioritizes smaller, fully-visible diffs first
(sorts the candidate pool ascending by size before labeling) to bias toward
higher-fidelity labels, but for diffs above ~10K characters the frontier
model only sees a prefix — a real issue located later in the diff can be
missed. This is why human spot-checking (below) matters most for larger
mined diffs.

### Mining provenance

Every mined diff's annotation carries a `provenance` dict:

```jsonc
{
  "kind": "mined",
  "task_id": "...", "project_id": "dark_factory",
  "outcome": "...", "review_cycles": 1, "verify_attempts": 2,
  "signal_reason": ["verify_attempts>=2 (actual=2)", "escalation_created event"],
  "merge_sha": "...",
  "runs_db_path": "/home/leo/src/dark-factory/data/orchestrator/runs.db",
  "runs_db_query": "... see mining.mine_fn_candidates ...",
  "escalation_refs": [{"task_id": "...", "category": "...", "severity": "...",
                        "summary": "...", "level": 0, "path": "..."}]
}
```

`escalation_refs` is frequently empty even when `signal_reason` includes
`"escalation_created event"` — the events table is a durable historical log,
but not every escalation still has a live `esc-*.json` file (older/resolved
records can be pruned from the archive). The `signal_reason` entry is the
authoritative provenance for that signal either way.

## Adjudication log (`adjudication_log.jsonl`)

One JSON object per line, one entry per diff_id (`adjudication.AdjudicationLog`
/ `AdjudicationEntry`) — **every** diff in the manifest has an entry, not just
mined ones:

- Mined diffs: `frontier_model="opus"`, `frontier_proposal` = the labels
  `propose_labels_frontier` returned (i.e. exactly `ground_truth`).
- The original 15 hand-authored diffs: `frontier_model="hand-authored"`,
  `frontier_proposal` = their pre-existing (task-2495-predating) ground
  truth, `notes` states plainly that these predate frontier mining and were
  never frontier-proposed. This keeps `adjudication_log` coverage complete
  over the whole corpus without fabricating a frontier opinion that was
  never actually solicited for them.

### Human spot-check subset

A deterministic, evenly-spread ~10% sample of the **mined** diffs (floor 5,
`__main__._select_spot_check_subset`) is flagged `in_spot_check_subset: true`
for a documented human spot-check, with `spot_check_status` starting at
`"pending"`. Membership is **sticky** across re-runs of `mine`: once an entry
is flagged (or its `spot_check_status` moves past `"pending"`), it stays
flagged even as later runs grow the mined set and recompute the sample --
new ids are only ever added on top, so an operator's sign-off is never
silently dropped by a subsequent `mine` invocation.

**Protocol for an operator confirming the subset:** for each entry with
`in_spot_check_subset: true` and `spot_check_status: "pending"`, read the
corresponding diff (`corpus/mined/<diff_id>.diff`) and its `frontier_proposal`
side by side, decide whether the proposed label(s) are accurate, and update
that entry in place:

```python
from orchestrator.evals.reviewer_trial.adjudication import AdjudicationLog

log = AdjudicationLog.load(Path("corpus/adjudication_log.jsonl"))
entry = next(e for e in log.entries if e.diff_id == "<diff_id>")
entry.spot_check_status = "confirmed"  # or "rejected"
entry.spot_check_reviewer = "<name>"
log.save(Path("corpus/adjudication_log.jsonl"))
```

As of the task-2495 generation run, this repo's agents cannot perform live
human review themselves — the spot-check subset is flagged and documented
here, but sign-off is an **operator follow-on** (filed as a non-blocking
`escalate_info`, not a blocker on the corpus-expansion work itself). Until an
entry's `spot_check_status` is updated, it remains `"pending"`.

## Verifying corpus integrity

```bash
cd orchestrator && uv run python -m orchestrator.evals.reviewer_trial corpus-audit
```

Checks (`mining.audit_corpus`): diff count ≥ floor, every diff has a split,
split ratios ≈2:1:7, every mined diff has non-empty provenance, the
adjudication log covers every diff_id, and the spot-check subset is
non-empty. `orchestrator/tests/test_reviewer_trial_corpus_integrity.py`
pins this as a deterministic (no-LLM) regression gate over the committed
corpus.

## Regenerating / extending the corpus

```bash
cd orchestrator && uv run python -m orchestrator.evals.reviewer_trial mine --help
```

`mine` is idempotent and resumable: it skips `diff_id`s already present in
`manifest.json`, and re-saves (manifest + adjudication log, re-deriving the
split and spot-check subset over the current full set) after every
successfully-labeled diff, so an interrupted run only loses whatever was
still in flight. Re-running with a higher `--target-total` extends the
corpus further; the `--limit` option caps how many *new* candidates a single
invocation attempts (useful for a small validation run before a full batch).

`--runs-db`, `--escalations-dir`, and `--repo-path` default to this
deployment's on-disk paths in the main repo checkout, and can be overridden
either via the flag or via the `REVIEWER_TRIAL_RUNS_DB` /
`REVIEWER_TRIAL_ESCALATIONS_DIR` / `REVIEWER_TRIAL_REPO_ROOT` env vars. If
`--runs-db` doesn't point at a real file, `mine` fails fast with a clear
error rather than letting `sqlite3` silently create an empty database.
