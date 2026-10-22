# reviewer_trial corpus

Evaluation corpus for the reviewer-panel trial (`orchestrator/evals/reviewer_trial`).
Originally 15 hand-authored diffs (12 synthetic mutations + 3 real-world bugs);
expanded under **task 2495** (PRD `tier1-prompt-optimization` T4 / D-6) to
≥50 diffs (aim ~100) by *mining* additional diffs from this project's own
orchestrator run history. This file documents the corpus schema, the mining
pipeline that produced the `source: "mined"` diffs, the train/selection/test
split, the ground-truth adjudication protocol, and the answer-key pooling
protocol.

## Layout

```
corpus/
  manifest.json          # diff index: id, language, source, description, project, split
  annotations/<id>.json  # ground_truth issues (+ provenance, for mined diffs)
  synthetic/<id>.diff     # source == "synthetic" diff text
  real_world/<id>.diff    # source == "real_world" diff text
  mined/<id>.diff         # source == "mined" diff text
  adjudication_log.jsonl  # one AdjudicationEntry per diff_id (see below)
  pooling_log.jsonl       # one PoolingRecord per pooled answer-key issue (see below)
  README.md               # this file
```

`CorpusManifest.load()` (`corpus.py`) stitches these together: it reads
`manifest.json` for the diff index, then resolves each diff's text from
`<source>/<diff_id>.diff` and its ground truth (+ provenance) from
`annotations/<diff_id>.json`.

## Schema

**`manifest.json`** — `{"version": "2.0", "split_seed": "<seed>", "diffs": [...]}`.
Each entry: `{diff_id, language, source, description, project?, split?}`.

- `version` — bumped whenever the answer key changes. Recall, precision and
  F1 are comparable only between scores computed against the same version.
  1.0 → 2.0 is the 2026-10-22 pooling (task 7367, see "Answer-key pooling
  protocol").
- `language` — `"python" | "rust" | "typescript"`.
- `source` — `"synthetic"` (planted mutation) | `"real_world"` (hand-picked
  real bug, pre-task-2495) | `"mined"` (task-2495 FN mining, see below).
- `project` — optional; selects the working directory a reviewer would see
  around the diff (`corpus._PROJECT_CWD_MAP`). All mined diffs use
  `"dark-factory"`.
- `split` — `"train" | "selection" | "test"`, assigned by
  `mining.assign_split()` (see "Split" below). Every diff has one.

**`annotations/<diff_id>.json`** — `{diff_id, ground_truth: [...], provenance?, context?}`.
Each `ground_truth` entry is a `GroundTruthIssue`:
`{id, location, category, severity, description, mutation_type}`
(`severity` is `"blocking" | "suggestion"`). `provenance` is present only for
mined diffs (see "Mining provenance" below). `context` (`corpus.FixtureContext`)
carries the inputs the context-complete reviewer briefing (ruling D2, task
5369) needs beyond the diff, so the trial can score that briefing offline:

- `task` — `{id?, title, description, details, test_strategy, authored,
  record_source?}`: the record the implementer received. Mined fixtures read
  it from `.taskmaster/tasks/tasks.db` (`record_source` says when, and whether
  the row moved after the merge); the 15 hand-authored fixtures carry a
  plausible record reconstructed from the diff, flagged `authored: true`, that
  states the intent without naming the planted defect.
- `plan` — the architect's `plan.json` (`analysis`, `design_decisions`,
  `steps`) when a copy survives, else `null` with `plan_reason` listing the
  locations searched. As of the 2026-09-11 enrichment one mined fixture's plan
  (1157) survives, in a never-reclaimed worktree; the other 37 have none
  (task-meta dirs are reclaimed with the worktree, runs.db events carry only
  token counts).
- `base_sha` / `branch_sha` — the commit the stored diff applies to and the
  reviewed tip. For mined fixtures these are the merge commit's two parents,
  and `verification` records that `git diff base_sha merge_sha` reproduces the
  stored diff byte-for-byte and `git merge-tree` of the parents equals the
  merge commit's tree. Hand-authored fixtures have `branch_sha: null`; their
  `base_sha` is the commit the diff was cut from when `git apply --check` or
  a byte-level content match recovers one (`verification.applies_cleanly`
  says which), else `null` with `base_sha_reason`.
- `changed_files` — the paths the diff touches.

The loader is tolerant: a fixture without `context` loads with
`CorpusDiff.context is None`, and `save()` omits the key for it.

Every panel runs in its own scratch clone
(`checkout.py::fixture_checkout`). The clone is a `git clone --shared` of the
fixture's source repository, detached at `base_sha`, with the stored diff
applied and committed, so HEAD is the reviewed tree and HEAD^ is its base.
Every arm, diff-only included, runs at the clone root, and the clone is
removed when the panel ends. A context-complete (`d2`) arm needs two things:

- `context`: `runner.py::run_trial` refuses a corpus lacking it before it
  spends anything;
- a materialisable base: a fixture with `base_sha: null`, or a diff that
  `git apply` rejects, is skipped for that arm and logged, with no result. The
  diff-only arm still reviews it, without the change applied.

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

## Pooling log (`pooling_log.jsonl`)

One JSON object per line, one record per pooled answer-key issue
(`key_pooling.PoolingRecord` / `PoolingLog`):

- `diff_id`, `issue_id` — the annotation and the `ground_truth` id the record
  describes.
- `pooling_date` — the day the pooling run adjudicated the issue.
- `pooling_run` — the run's tracked directory, relative to the checkout
  root: `corpus/pooling_runs/<date>/`. It holds the run's adjudication
  standards (`pooling/BRIEF.md`, `skeptic/BRIEF.md`,
  `blocking-adjudication/BRIEF.md`) and `pooling/key/key.json`, which
  unblinds each `finding_ids` entry to arm, trial and reviewer severity.
- `provenance` — the run's `x_provenance` for the issue, verbatim:
  `finding_ids`, `real` (`yes | partly`), `adjudicated_severity`,
  `skeptic_corrections`, `corpus_severity` (the first-pass rating, before a
  `quality_block` promotion), `quality_block`, `confidence`, `evidence`.

The 2026-10-22 files are verbatim copies from that run's working directory,
`plans/fable-evidence-package-2026-09-10/campaign-d2/pooling-2026-10-22` in
the main checkout. That directory is untracked and not durable; the packets,
the per-finding verdicts (including findings ruled not real) and the rescore
outputs exist only there. Its `pooled_ground_truth.json` is not copied,
because the annotations plus this log reproduce it.

Provenance lives here rather than in the annotations for two reasons. The
adjudication log stays one entry per diff_id, and `CorpusManifest.load`
builds `GroundTruthIssue(**gt)`, which rejects unknown keys, so an
annotation carries only the six issue fields.

A key issue is pooled when its `mutation_type` starts with `pooled_`
(`key_pooling.is_pooled_issue`); `fold_pooled_issues` refuses an entry
without that prefix. `key_pooling.audit_pooling` checks the log ↔ pooled
issue bijection: no duplicate record, no record naming anything but a pooled
issue, and no pooled issue without its record, even when every record of
its run is lost. `test_reviewer_trial_corpus_integrity.py` pins it over the
committed corpus.

## Answer-key pooling protocol

An unmatched reviewer finding scores as a false alarm, so an incomplete key
penalises reviewers for finding real problems. A blind check in the D2
campaign showed the key was badly incomplete. Pooling repairs it: every
compared arm's unmatched findings are adjudicated against the code, and each
distinct real problem in the change becomes a new key issue.

**The 2026-10-22 run** (task 7367). Twelve opus adjudicators, blind to arm,
read all 851 unmatched findings of both arms (opus diff-only and opus `d2`),
3 trials, all 37 test-split fixtures, with read access to scratch clones of
the reviewed trees. The findings resolved to 271 distinct problems: 8
duplicated an existing key issue, 36 were not real, 26 were real but in code
the change left alone, and 201 were added (30 blocking, 171 suggestion). A
refute-prompted skeptic pass downgraded `mined_1169_pool_01` and `_02` to
suggestion.

**Rules.** Pooling is blind, symmetric, and drawn from the unmatched
findings of every arm being compared. A new arm's unmatched findings MUST be
pooled the same way before its scores are compared; otherwise an arm whose
real findings were never pooled is penalised for them.

**Severity mapping.** `blocking` means a correctness or security blocker, or
a maintainability problem a careful reviewer would refuse to merge
(`quality_block`). The mapping was calibrated on 57 controls: every reviewer
flag that matched an existing blocking key issue was rated one of those two
kinds, and none merely a suggestion.

**Folding a run.** Pooled entries carry `mutation_type` `pooled_<date>`
(`pooled_2026_10_22` for the run above). `pooled_ground_truth.json` maps diff_id to entries holding the six
`GroundTruthIssue` fields plus `x_provenance`. Copy the run's briefs and
unblinding key into `corpus/pooling_runs/<date>/`, then fold from
`orchestrator/` with `uv run python` and bump `version`:

```python
import dataclasses, json
from pathlib import Path

from orchestrator.evals.reviewer_trial.corpus import CorpusManifest
from orchestrator.evals.reviewer_trial.key_pooling import PoolingLog, fold_pooled_issues

corpus = Path("src/orchestrator/evals/reviewer_trial/corpus")
pooled = json.loads(Path("<run>/pooling/pooled_ground_truth.json").read_text())
manifest, log = fold_pooled_issues(
    CorpusManifest.load(corpus / "manifest.json"),
    PoolingLog.load(corpus / "pooling_log.jsonl"),
    pooled,
    pooling_date="<YYYY-MM-DD>",
    pooling_run="orchestrator/src/orchestrator/evals/reviewer_trial/corpus/pooling_runs/<date>",
)
dataclasses.replace(manifest, version="<next major>").save(corpus / "manifest.json")
log.save(corpus / "pooling_log.jsonl")
```

`fold_pooled_issues` appends each diff's pooled issues after its existing
key, in file order, and rejects an unknown diff_id, a repeated issue id, or
an entry with missing or unexpected keys. A load → save round trip
reproduces the corpus byte for byte, so only the pooled annotations and
`manifest.json` change.

### Key entries flagged for human review (not edited)

- `mined_829` `frontier_1`: a Python 3.11 probe shows the marker DOES change
  `AsyncMock` behaviour, contradicting the entry.
- `py_always_passes` `py_pass_3`: the assertion is roundabout, not
  tautological.
- `py_real_01` `py_real_cap_3`: possibly blocking, not suggestion.

### Matcher noise

Re-running the haiku matcher (`scorer.py::match_issues`) on identical reviews
and an identical key changed 7% of cell F1 values and moved an arm-difference
mean by 0.033. A single matcher run is noisy at the decision margin.

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
