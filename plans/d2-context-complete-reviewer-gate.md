# D2 context-complete reviewer: gate, flip, watch, kill

Operator runbook for ruling D2 (task 5369). The merged code adds a second
reviewer generation behind the config leaf `review.reviewer_briefing`. Its
default, `diff-only`, keeps every project's reviewer byte-identical to the
incumbent, so nothing changes until step 2 flips it. Each step below is one
action.

- `diff-only` is the incumbent: the diff cut at 50,000 chars, plus the
  original Rule 2.
- `d2` is the context-complete generation:
  - the task record (with its test strategy) and the plan;
  - every changed file, each to be read whole;
  - project conventions;
  - a suggestion-only scope check;
  - a Rule 2 that blocks only on breakage or on a named code-quality heuristic
    plus the concrete next change it taxes.

  It is built by
  `orchestrator/src/orchestrator/agents/review_briefing.py::render_context_complete_review`
  over inputs gathered by
  `orchestrator/src/orchestrator/review_inputs.py::gather_review_subject`.

## 1. The gate run

```bash
cd orchestrator && uv run python -m orchestrator.evals.reviewer_trial campaign \
  --results-dir <fresh dir> --split test --trials 3 \
  --arms variant_a --briefing diff-only,d2
```

To also measure the xhigh-effort arm, add it: `--arms variant_a,variant_a_xhigh`.
That runs four arms: `variant_a`, `variant_a_d2`, `variant_a_xhigh` and
`variant_a_xhigh_d2`. The incumbent is always the first arm, `variant_a`.

**Adopt only if** the `variant_a_d2` non-inferiority line in
`<results-dir>/campaign_report.md` reads **PASS**. The same verdict is in
`campaign_report.json` under `decisions.non_inferiority`. The rule is
`orchestrator/src/orchestrator/evals/reviewer_trial/campaign.py::non_inferior`:
in each of at least three trials, the arm's mean blocking recall is not below
the incumbent's and its mean F1 is not worse. Ties pass.

How to read the report:

- **The comparison is paired.** Each trial compares only the
  `(trial, fixture)` cells that both arms measured. The report line prints the
  per-trial blocking-recall and F1 deltas.
- **The d2 arm skips fixtures it cannot materialise.** Every panel runs in a
  scratch clone at the fixture's `base_sha` with its stored diff applied and
  committed. A fixture with no recoverable base, or whose diff `git apply`
  rejects, is skipped for the d2 arm: it is logged as a WARNING
  (`Skipping variant_a_d2__<fixture> ...`) and writes no score record, so it
  falls out of the pairing. The diff-only arm still runs it. On the
  2026-10-18 test split (37 fixtures) that means four skips: `py_obvious_copy`
  (base unrecoverable), and `py_real_01`, `rs_real_01` and `ts_real_01`, whose
  diffs are not in git-apply format.
- **A d2 arm fails loudly on an unenriched corpus.** If any fixture lacks
  `context`, the campaign raises `FixtureEnrichmentMissing` naming every such
  fixture, before it spends anything.
- **Unmeasured matcher cells are excluded on both sides** (task 6736). Read
  each arm's `unmeasured` count in the report. If it is large, rerun into the
  same `--results-dir`: measured cells are cached and unmeasured ones are
  re-scored.
- **Use a fresh `--results-dir`.** Score files cached before task 6736 cannot
  be told apart from a refused matcher's zero.

The campaign draws on the shared OAuth account pool by default (`--pool`).
Pass `--no-pool` to run on the CLI's default login.

## 2. The flip

1. **Deploy the merged code to the fleet first.** A running orchestrator only
   knows the leaf once it runs the merged code. That happens through the normal
   merge-landed coordinator redeploy, or by hand with
   `scripts/restart-all-orchestrators.sh --drain`. Before any manual restart,
   run `scripts/orchestrator-watchdog.py --report`, which is read-only, and read
   its `FLEET-LEASE:` line.
2. **Set the leaf in `dark-factory-orchestrator.yaml` only.** Every other
   project keeps the `diff-only` default:

   ```yaml
   review:
     reviewer_briefing: d2
   ```

   It is a direct-to-main commit, so use `git commit --only <path>`.
3. **Apply it with `reload_config`** (`mcp__escalation__reload_config`). The
   leaf is green tier. Read the returned dispositions:
   `review.reviewer_briefing` must appear under `applied` and not under
   `restart_required`.

The next review dispatch runs the d2 roster and builder
(`orchestrator/src/orchestrator/workflow.py::TaskWorkflow._review`). Each
reviewer role's prompt harness version is part of the verdict-cache
fingerprint (`TaskWorkflow._reviewer_config_fingerprint`). So a cached
diff-only verdict is never reused after the flip: in-flight tasks pay one
redundant review. That is the fail-safe direction.

## 3. The two-week watch

Compare the two weeks before the flip with the two weeks after:

- **Per-review cost**: `invocation_end` events for role
  `reviewer_comprehensive` in `data/orchestrator/runs.db`.
- **Block rate**: the share of reviews with at least one `blocking` issue.
- **Review-phase amendment spend**: amendment rounds triggered by review
  suggestions.
- **Post-merge failures**: `merge_attempt` outcomes `main_health_red` and
  `verify_failed`.

## 4. The kill rule

Revert the leaf to `diff-only` with another `reload_config` if `review_issues`
escalations per review exceed 15% without a matching fall in post-merge
failures (`main_health_red` / `verify_failed`). The revert also changes the
verdict-cache fingerprint, so it too costs one redundant review per in-flight
task.

## 5. Sequencing with sibling tasks

- **Task 5695 (stale stored review base).** The d2 path derives its review
  range from a fresh fork point at review time
  (`orchestrator/src/orchestrator/review_inputs.py::review_fork_point`). The
  `diff-only` path still diffs from the write-once stored base until 5695
  lands, so on a branch with main merged in it can still review main's churn.
- **Task 5224 (open info notes in the reviewer briefing).** It slots into the
  d2 builder as one more field on
  `orchestrator/src/orchestrator/agents/review_briefing.py::ReviewSubject` and
  one more section in `render_context_complete_review`. It needs no second
  briefing path.
