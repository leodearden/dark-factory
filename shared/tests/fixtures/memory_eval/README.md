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
| `e1-thin/metrics-20260704T031500Z.json` | `insufficient_data` path | `n = 6` below `LimitsConfig.min_samples`, and a single run so the baseline window is empty |
| `malformed/metrics-bad-kind.json` | negative | `kind: "histogram"` is outside the closed vocabulary |
| `malformed/metrics-proportion-out-of-range.json` | negative | proportion `value: 1.4` outside `[0, 1]` |
| `e1-retrieval-health/limits-current.json` | M2 limits artifact | **generated**, see below |

The baseline window pools to exactly the anchors the tests use: proportion
`72/90 = 0.8`, dangling-pointer mean `(4+5+6)/3 = 5.0`, superseded-above-successor
mean `2.0`. The two topics failing throughout the window
(`t-recon-watcher-triage`, `t-routing-ladder`) are the grandfather snapshot —
they stand in for the known-bad findings the 3111/3112 fix lineage already owns,
which D1 says must be reported in the initial-state report and never alarmed on.

## Regenerating — do not hand-edit

`e1-retrieval-health/limits-current.json` is **generated** by
`shared.memory_eval_limits.write_limits_artifact` over the committed baseline
window, so the exemplar the dashboard-shaped reader parses is valid by
construction rather than by hand.

The `metrics-*.json` series files are hand-authored, but in the writer's exact
canonical serialization: `json.dumps(..., indent=2, sort_keys=True,
ensure_ascii=False)` plus a trailing newline, with `None`-valued optional
fields omitted. `test_memory_eval_boundary.py` asserts byte-identity by
re-emitting each parsed exemplar through `write_metric_series`, so any drift
fails CI rather than silently diverging from what the runners actually produce.

To change an exemplar, edit it and re-run that byte-identity test; if it fails,
regenerate the file through `write_metric_series` rather than reconciling the
bytes by hand. Adding a metric to the `e1-retrieval-health` runs changes the
alarm-eligible count and therefore the derived α — update the α anchors in
`test_memory_eval_limits.py` and this README together.
