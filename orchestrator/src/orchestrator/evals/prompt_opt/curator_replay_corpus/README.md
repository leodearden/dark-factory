# curator replay corpus

Evaluation corpus for the T6 curator prompt-optimization loop
(`orchestrator/evals/prompt_opt`). Built under **task 2496** (PRD
`tier1-prompt-optimization` T5 / D-6) by replaying
`data/reconciliation/tickets.db` (4,019 lifetime curator decisions: 3,014
`created` / 753 `combined` / 249 `failed`) through a frontier-adjudication +
human-spot-check labeling pass. This file documents the corpus schema, the
build pipeline, the train/selection/test split, and the ground-truth
adjudication protocol. It mirrors
`orchestrator/evals/reviewer_trial/corpus/README.md`.

**Not yet built.** Unlike the reviewer corpus (task 2495 already ran its
mining pipeline and committed the result), this directory holds only this
README until an operator runs `build-curator-corpus` (see "Building the
corpus" below) against the real, gitignored `tickets.db`. Task 2496 proves
the machinery hermetically (`test_prompt_opt_curator_corpus.py`,
`test_prompt_opt_curator_scorer.py` -- synthetic fixtures + injected fakes
only); running the real, costly build is an operator action (PRD §6/§8: "the
batch builds and smoke-proves the machinery; the operator drives the costly
runs").

## Layout (once built)

```
curator_replay_corpus/
  manifest.json           # item index: ticket_id, split
  annotations/<id>.json   # full CuratorReplayItem: candidate, recorded_*,
                           # gold_*, split, provenance
  adjudication_log.jsonl  # one AdjudicationEntry per ticket_id (see below)
  README.md               # this file
```

Unlike the reviewer corpus, there is no separate `<source>/<id>.diff` text
file per item -- a curator decision's "content" is the candidate task JSON
itself (title/description/details/files_to_modify/priority/spawned_from),
which is small enough to live inline in each annotation file rather than in
a companion text blob.

`CuratorCorpusManifest.load()` (`curator_corpus.py`) stitches these
together: it reads `manifest.json` for the item index, then resolves each
item's full record from `annotations/<ticket_id>.json`.

## Schema

**`manifest.json`** -- `{"version": "1.0", "split_seed": <seed>, "items": [...]}`.
Each entry: `{ticket_id, split?}`.

**`annotations/<ticket_id>.json`** -- a serialized `CuratorReplayItem`:

- `ticket_id`, `candidate` -- the ticket id and its candidate task dict, read
  verbatim from tickets.db's `candidate_json` column.
- `recorded_action`, `recorded_target_fingerprint`, `recorded_target_id` --
  the ticket's HISTORICAL (unverified) curator decision, recovered from
  tickets.db. **Provenance / weak signal only -- never a gold label** (see
  "Decisions != ground truth" below).
- `gold_action`, `gold_target_fingerprint`, `gold_target_id` -- the
  frontier-adjudicated (+ possibly human-spot-checked) label
  `CuratorActionScorer` actually grades against.
- `split` -- `"train" | "selection" | "test"`, assigned by
  `reviewer_trial.mining.assign_split()` (see "Split" below). Every item has
  one.
- `provenance` -- a dict recording build-time context (currently just
  `source_db`, the tickets.db path the item was read from).

## Recorded-decision recovery (the drop/combine gotcha)

tickets.db persists **both** `drop` and `combine` decisions with
`status='combined'` -- the live curator middleware folds a dropped candidate
into its duplicate target the same way it folds a combined one
(`task_interceptor.py`: "Drop: fold candidate into the existing target task
(status='combined')"). `status` alone therefore cannot distinguish them; only
the embedded `result_json['action']` can. `recover_recorded_action()`
(`curator_corpus.py`) always prefers `result_json['action']` when present and
valid, falling back to `status == 'created' -> 'create'` only when no action
is embedded. `status='failed'`/`'pending'` rows (or a missing/unparseable
`result_json` not rescued by the `'created'` fallback) are un-actionable and
skipped.

**`recorded_target_fingerprint` is always `None` on real builds.** The live
curator's persisted `result_json` for drop/combine
(`task_interceptor.py`'s `_dispatch_ticket_decision`) only ever writes `{id,
title, deduplicated, action, justification}` -- there is no
`target_fingerprint` key to recover. `recover_recorded_action()` still reads
one if present (for a hypothetical richer `result_json` shape), but on the
real `tickets.db` only `recorded_target_id` (from `result_json['id']`) comes
back populated. This is harmless: `recorded_*` fields are provenance/weak
signal only (see "Decisions != ground truth" below) and are never read by
`CuratorActionScorer` or used as a gold label.

## Split (train / selection / test, 2:1:7)

Every item carries a `split`, assigned by
`reviewer_trial.mining.assign_split(ticket_ids, seed)` over the full sampled
`ticket_id` set using a stable hash of `f"{seed}:{ticket_id}"` sliced into
cumulative 2:1:7 buckets -- deterministic and reproducible across
machines/runs. The seed used to produce a given build is recorded at the top
level as `manifest.json`'s `split_seed`.

As with the reviewer corpus, the **test** split should be scored once (not
iterated against during heuristics tuning) -- use `train`/`selection` for
iteration and reserve `test` for the final `run_optimization_loop` read
(`ArtifactProvenance.held_out_TEST_score`).

## Decisions != ground truth (PRD D-6)

A ticket's recorded action/target is what the live curator historically
decided, **unverified**. It is retained on each item only as
`recorded_*` provenance -- `CuratorActionScorer` never reads it. Every gold
label instead comes from:

1. **Frontier adjudication** -- `propose_curator_label_frontier()` (build
   time only, `invoke_agent` with `model='opus'`, `effort='high'`, no tools,
   structured JSON output `{action, target_fingerprint, target_id,
   justification}`) proposes an independent action for the candidate,
   informed only by the candidate JSON itself (title/description/details) --
   never by what the live curator actually did. An unparseable/malformed
   frontier response degrades to `action='create'` (the live curator's own
   best-effort fallback semantics) rather than raising.
2. **Human spot-check** -- a deterministic, action-stratified subset (see
   below) flagged for a human to independently confirm or reject the
   frontier proposal.

`build_curator_corpus()`'s own unit test
(`TestBuildCuratorCorpus::test_returns_manifest_with_frontier_gold_labels_not_recorded_actions`)
pins this invariant directly: every item's `gold_action` comes from the
injected proposer, and at least one item's `recorded_action` differs from
its `gold_action`.

## Build pipeline (`build-curator-corpus` CLI, task 2496)

`build_curator_corpus(db_path, *, n, seed, spot_check_size, frontier_proposer)`
(`curator_corpus.py`), wired to the real tickets.db + the real frontier
proposer by the `build-curator-corpus` CLI command (`__main__.py`):

1. `read_curator_decisions(db_path)` -- read-only stdlib `sqlite3` scan of
   `tickets`, recovering every actionable decision (see "Recorded-decision
   recovery" above). No import of `fused_memory` -- tickets.db rows are read
   as plain dicts by column name, keeping this eval package decoupled from
   the live middleware.
2. Downsample to at most `n`, round-robin across recorded-action strata (so
   a bounded `n` stays representative of drop/combine/create even when one
   action dominates the raw history).
3. Call the frontier proposer once per sampled candidate to obtain its GOLD
   label.
4. Flag the human spot-check subset (below), assign the 2:1:7 split, and
   assemble the `CuratorReplayItem`s + a companion `AdjudicationLog`
   recording one frontier-proposal entry per item.

Fully hermetic given an injected fake proposer (see
`test_prompt_opt_curator_corpus.py`); only the real CLI invocation below
touches the real tickets.db and makes real (billed) frontier calls.

## Adjudication log (`adjudication_log.jsonl`)

One JSON object per line, one entry per `ticket_id`
(`reviewer_trial.adjudication.AdjudicationLog` / `AdjudicationEntry` --
reused verbatim, not reimplemented): `frontier_model`, `frontier_proposal`
(the proposed `{action, target_fingerprint, target_id, justification}`),
`in_spot_check_subset`, and `spot_check_status` (`"pending" | "confirmed" |
"rejected"`).

### Human spot-check subset

A deterministic, action-stratified sample of the built items --
`select_spot_check_subset()` (`curator_corpus.py`), the PRD §9 tactical
sizing decision this task owns: within each recorded-action stratum, sample
`~20%` (floor 5, so a stratum smaller than the floor is taken in full), then
trim the combined subset to `spot_check_size` (default 200) if it would
otherwise exceed it -- bounding total human review effort regardless of
corpus size. The trim itself is representation-preserving: it reserves up to
`minimum` (default 5) ids from every stratum before spending any remaining
cap budget on the leftover pool, so every present action (drop/combine/
create) keeps spot-check representation whenever `spot_check_size` is at
least the number of present actions. A pathologically small
`spot_check_size` (below the number of present-action strata, e.g. `< 3`
when all of drop/combine/create appear) cannot preserve all of them -- the
cap bound still wins, and which stratum(a) survive is decided by the same
deterministic seeded shuffle. Same items + seed always yields the same
subset.

**Protocol for an operator confirming the subset:** for each entry with
`in_spot_check_subset: true` and `spot_check_status: "pending"`, read the
corresponding item (`curator_replay_corpus/annotations/<ticket_id>.json`,
which carries both `candidate` and the frontier's `gold_*` fields) and the
matching `adjudication_log.jsonl` entry's `frontier_proposal` side by side,
decide whether the proposed label is accurate, and update that entry in
place:

```python
from pathlib import Path
from orchestrator.evals.reviewer_trial.adjudication import AdjudicationLog

log = AdjudicationLog.load(Path("curator_replay_corpus/adjudication_log.jsonl"))
entry = next(e for e in log.entries if e.diff_id == "<ticket_id>")
entry.spot_check_status = "confirmed"  # or "rejected"
entry.spot_check_reviewer = "<name>"
log.save(Path("curator_replay_corpus/adjudication_log.jsonl"))
```

(`AdjudicationEntry.diff_id` is the generic field name reused verbatim from
`reviewer_trial` -- for a curator entry it holds the `ticket_id`.)

As with the reviewer corpus, this repo's agents cannot perform live human
review themselves -- the spot-check subset is flagged and documented here,
but sign-off is an **operator follow-on** (a non-blocking `escalate_info`,
not a blocker on the corpus-build work itself). Until an entry's
`spot_check_status` is updated, it remains `"pending"`.

## Building the corpus

```bash
cd orchestrator && uv run python -m orchestrator.evals.prompt_opt build-curator-corpus --help
```

```bash
cd orchestrator && uv run python -m orchestrator.evals.prompt_opt build-curator-corpus \
    --db-path /home/leo/src/dark-factory/data/reconciliation/tickets.db \
    --n 100 --seed 2496 --spot-check-size 200 \
    --out orchestrator/src/orchestrator/evals/prompt_opt/curator_replay_corpus/manifest.json
```

`--db-path` defaults to `/home/leo/src/dark-factory/data/reconciliation/tickets.db`
(override via the flag or the `CURATOR_TICKETS_DB` env var, for portability
across checkouts -- tickets.db is offline, gitignored data that lives in the
MAIN repo, not a task worktree). If `--db-path` doesn't point at a real
file, the command fails fast with a clear error rather than letting
`sqlite3` silently create an empty database.

The command prints the total frontier labeling cost (summed from each
sampled candidate's `FrontierLabel.cost_usd`, populated by
`propose_curator_label_frontier`'s real `invoke_agent` call) and an
`audit_curator_corpus` PASS/FAIL summary after writing the
manifest/annotations/adjudication log, and exits non-zero on FAIL.
Re-running with a different `--seed` reshuffles the sample and split;
this command is a **full rebuild**, not an incremental `mine`-style resume
(unlike `reviewer_trial mine`, tickets.db is a static historical replay
source rather than a growing one, so there is no "new decisions since last
run" case to resume from).

## Verifying corpus integrity

`audit_curator_corpus(manifest, adjudication_log, min_items=50,
ratios=(2, 1, 7), ratio_tolerance=0.1)` (`curator_corpus.py`) checks: item
count >= the minimum floor, every item has a split, split proportions
approximate 2:1:7 within tolerance, every item carries a gold label, the
adjudication log covers every `ticket_id`, and the human spot-check subset
is non-empty. It runs automatically at the end of `build-curator-corpus`;
to re-audit an already-built corpus without rebuilding it:

```python
from pathlib import Path
from orchestrator.evals.prompt_opt.curator_corpus import CuratorCorpusManifest, audit_curator_corpus
from orchestrator.evals.reviewer_trial.adjudication import AdjudicationLog

corpus_dir = Path("curator_replay_corpus")
manifest = CuratorCorpusManifest.load(corpus_dir / "manifest.json")
log = AdjudicationLog.load(corpus_dir / "adjudication_log.jsonl")
report = audit_curator_corpus(manifest, log)
print(report.ok, report.failures)
```
