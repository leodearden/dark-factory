# Memory-eval exemplar artifacts

Committed exemplars for the M1 metric-series schema and the M2 limits
evaluator (`docs/prds/memory-eval-program.md` §3). They are laid out as the
real runtime tree — `<root>/<eval_id>/metrics-<STAMP>.json` — so the path
helper is exercised end-to-end rather than mocked.

Consumed by `shared/tests/test_memory_eval_boundary.py` (both sides of the
boundary: the producer-side schema/writer and the consumer-side evaluator plus
a dashboard-shaped `json.load`-only reader).

## Record schema (M1)

```
{
  "schema_version": 1,
  "eval_id": "<eval id, == the containing directory name>",
  "run_stamp": "<%Y%m%dT%H%M%SZ UTC stamp, == the filename stamp>",
  "corpus": {"project_id": "<str>", "counts": {"<bucket>": <int>, ...}},
  "metrics": [
    {
      "metric_id": "<non-empty, unique within the series>",
      "kind": "tripwire" | "proportion" | "count" | "scalar",
      "value": <float>,
      "n": <int >= 0>,
      "denominator": <int, REQUIRED for proportion, forbidden otherwise>,
      "direction": "higher_is_worse" | "lower_is_worse",  // REQUIRED for
                   // proportion/count, forbidden otherwise
      "items": [{"item_key": "<str>", "passed": <bool>}],   // REQUIRED for tripwire
      "details_path": "<optional relative path to a companion report>"
    }
  ]
}
```

Per-kind cross-field rules (enforced at emit time by
`shared.memory_eval_metrics`, which raises `MetricSchemaError`):

- **tripwire** — `items` non-empty; `n == len(items)`; `value` == the number of
  items with `passed == false`.
- **proportion** — `denominator` required and equal to `n`; `value` in `[0, 1]`
  and `value * denominator` a whole number (the binomial successes count).
- **count** — `value` a non-negative whole number; no `denominator`/`items`.
- **scalar** — free float; reported but never alarmed.
- **direction** — required for `proportion` and `count`, rejected for `tripwire`
  and `scalar`. The exact tests are two-sided, so without it a dramatic
  IMPROVEMENT (`canonical-in-top-5` at 30/30, `dangling-pointers` at 0) is
  indistinguishable from a regression and would alarm. It cannot be defaulted
  per kind — higher is good for `canonical-in-top-5` and bad for
  `dangling-pointers`. A surprising move the safe way reports as verdict status
  `improved` and never enters the published `alarms` feed.

## Exemplars and their provenance

All values were chosen so every verdict separates from the derived α
(`1.0 / (90 runs × 4 alarm-eligible metrics) = 1/360 ≈ 0.002778`) by three or
more orders of magnitude — no verdict here is knife-edge. The p-values quoted
below were computed with stdlib at plan time and are re-derived by the tests.

Each `e1-retrieval-health` run carries all four metric kinds, four of which are
alarm-eligible (`canonical-in-top-5`, `dangling-pointers`,
`superseded-above-successor`, `topic-canonical-present`), which is what makes
the derived α exactly `1/360`.

| File | Role | Notable content |
|---|---|---|
| `e1-retrieval-health/metrics-20260701T031500Z.json` | trailing baseline window | proportion 24/30, dangling 4, superseded 2, 2 failing topics |
| `e1-retrieval-health/metrics-20260702T031500Z.json` | trailing baseline window | proportion 24/30, dangling 5, superseded 2, same 2 failing topics |
| `e1-retrieval-health/metrics-20260703T031500Z.json` | trailing baseline window | proportion 24/30, dangling 6, superseded 2, same 2 failing topics |
| `e1-retrieval-health/metrics-20260704T031500Z.json` | current run, **regression** variant | proportion 12/30 (p≈1.84e-06 → alarm), dangling 20 vs λ=5 (p≈3.45e-07 → alarm), `t-worktree-lifecycle` **newly** fails (→ alarm), `t-routing-ladder` now passes (→ ratchet) |
| `e1-retrieval-health/metrics-20260705T031500Z.json` | current run, **quiet** variant | proportion 20/30 (p≈0.105 → ok), dangling 8 vs λ=5 (p≈0.174 → ok), failing topics identical to the grandfather snapshot (→ no alarm, idempotent re-run) |
| `e1-dual-tripwire/metrics-20260801T031500Z.json` | two-tripwire snapshot | `topic-canonical-present` + `successor-pointer-present`; `t-shared` fails in the first and passes in the second |
| `e1-dual-tripwire/metrics-20260802T031500Z.json` | unchanged re-run of the above | identical items → no alarm, grandfather hash unmoved |
| `e1-thin/metrics-20260704T031500Z.json` | `insufficient_data` path | `n = 6` below `LimitsConfig.min_samples`, and a single run so the baseline window is empty |
| `malformed/metrics-bad-kind.json` | negative | `kind: "histogram"` is outside the closed vocabulary |
| `malformed/metrics-proportion-out-of-range.json` | negative | proportion `value: 1.4` outside `[0, 1]` |
| `e1-retrieval-health/limits-current.json` | M2 limits artifact | **generated**, see below |

The baseline window pools to exactly the anchors the tests use: proportion
`72/90 = 0.8`, dangling-pointer rate `(4+5+6)/90 = 1/6` per unit of exposure —
an expected `5.0` at the current run's `n = 30` — and superseded-above-successor
rate `6/90`, an expected `2.0`. Rule (c) pools events over `Metric.n` (the
exposure a count was measured over) rather than averaging raw counts, the same
way rule (b) pools successes over trials; these anchors are unchanged either way
only because every `e1-retrieval-health` run measures `n = 30`. `e1-thin`
deliberately does not (`n = 6`), which is what makes the exposure explicit
rather than incidental. The two topics failing throughout the window
(`t-recon-watcher-triage`, `t-routing-ladder`) are the grandfather snapshot —
they stand in for the known-bad findings the 3111/3112 fix lineage already owns,
which D1 says must be reported in the initial-state report and never alarmed on.

## Grandfather keys are scoped by metric

The persisted known-bad set — `grandfather_set` in `limits-current.json`, and
`EvaluationResult.grandfather` — carries `"<metric_id>::<item_key>"` strings,
not bare item_keys. item_keys are unique only WITHIN a metric, and a series may
carry several tripwires, so a flat namespace would let one metric's pass
release another's known-bad entry (and let evaluating the second tripwire drop
the first one's entries entirely). Either way the next run alarms on data
nobody changed — the phantom alarm the ratchet exists to prevent.

`shared.memory_eval_limits.scoped_grandfather_key` / `grandfather_slice`
compose and recover the scope; `evaluate_tripwire` itself still speaks bare
item_keys for one metric, and `evaluate_series` does the composition. A
metric_id containing `::` is refused at key-composition time. `e1-dual-tripwire`
is the committed corpus for all of this — including the colliding `t-shared`
key — because a single-tripwire corpus cannot distinguish the two designs.

A run resumes from BOTH halves of the persisted state: `grandfather_set` and
`snapshotted_metric_ids`. The second records which tripwires have already been
snapshotted, because an empty grandfather slice cannot tell a metric that is
NEW from one whose every known-bad item was fixed — and those two must behave
in opposite ways. Without it, a probe wired up in month three would alarm on
every one of its pre-existing failures, which M2 says to grandfather. It is
evaluator state, not a dashboard signal.

## Regenerating — do not hand-edit

`e1-retrieval-health/limits-current.json` is **generated** by
`shared.memory_eval_limits.write_limits_artifact` over the committed baseline
window, so the exemplar the dashboard-shaped reader parses is valid by
construction rather than by hand.

Its exact recipe lives in one place — `evaluate_exemplar()` in
`tests/test_memory_eval_boundary.py` — so the artifact is reproducible rather
than a mystery blob: the **regression run** (`metrics-20260704T031500Z.json`)
judged against the three baseline runs, resuming from a grandfather set seeded
by the *first* baseline run's failures (with the matching
`snapshotted_metric_ids` ledger), under
`LimitsConfig(false_alarm_budget=1.0, runs_per_quarter=90, min_samples=10,
baseline_window=3)` — which derives α = 1/360 across 4 alarm-eligible metrics.
That run was chosen because it exercises all three alarm paths at once (a
proportion regression, a count regression and a newly-failing tripwire item)
*and* the ratchet releasing `t-routing-ladder`, leaving the known-bad list at
exactly `["t-recon-watcher-triage"]`. To regenerate:

```python
from pathlib import Path
from shared.memory_eval_limits import limits_artifact_path, write_limits_artifact
from test_memory_eval_boundary import evaluate_exemplar  # tests/ on sys.path

result = evaluate_exemplar()
write_limits_artifact(result, limits_artifact_path(Path('tests/fixtures/memory_eval'), result.eval_id))
```

`TestDashboardShapedReader` consumes it two ways that must agree: with plain
`json.load` + dict access (no `shared.memory_eval_*` import at all, exactly as a
dashboard would), and by regenerating it and asserting byte-identity. The first
pins the published contract; the second stops the committed bytes drifting from
what the writer emits.

### The null convention — one rule, both artifacts

**Stated in one place: the `shared.memory_eval_metrics` module docstring.**
Read it there. Restating it here would make the convention that exists to stop
two artifacts drifting apart into three copies of prose that can themselves
drift — so this section only says what it means *for these fixtures*:

- Both writers render through `shared.memory_eval_metrics.canonical_json_text`
  (`indent=2, sort_keys=True, ensure_ascii=False` + trailing newline), so any
  exemplar you author or regenerate must be in exactly that form.
- A field that does not apply is **absent**, not `null` — e.g. `item_key` on a
  whole-metric alarm, `denominator` on a non-proportion metric. Read them with
  `.get`, never `[...]`.
- The one field that *is* emitted as an explicit `null` is
  `limits-current.json`'s top-level `alpha`, on a run with nothing
  alarm-eligible. That is the convention's always-emit half, not an exception
  to it.

Pinned by `TestNullSerializationConvention` in `test_memory_eval_limits.py`,
which checks the rule field-by-field at every level the writer emits (artifact,
verdicts, alarms) rather than only for the fields named above.

The `metrics-*.json` series files are hand-authored, but in that exact
canonical serialization. `test_memory_eval_boundary.py` asserts byte-identity by
re-emitting each parsed exemplar through `write_metric_series`, so any drift
fails CI rather than silently diverging from what the runners actually produce.

To change an exemplar, edit it and re-run that byte-identity test; if it fails,
regenerate the file through `write_metric_series` rather than reconciling the
bytes by hand. Adding a metric to the `e1-retrieval-health` runs changes the
alarm-eligible count and therefore the derived α — update the α anchors in
`test_memory_eval_limits.py` and this README together.
