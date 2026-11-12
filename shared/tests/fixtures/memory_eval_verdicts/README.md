# Memory-eval verdicts exemplars

Committed exemplars of the root-level verdicts artifact the scheduled runner
(`fused-memory/scripts/memory_eval_runner.py`) writes once per run as
`verdicts-<STAMP>.json` and `verdicts-current.json`. The schema lives in
`shared.memory_eval_verdicts`; the consumer is the dashboard, which reads the
file with `json.load` only.

Consumed by `shared/tests/test_memory_eval_verdicts.py::TestCommittedVerdictsExemplars`.

| File | Recipe | Shape |
|---|---|---|
| `filed/verdicts-current.json` | regression run 20260704, storm limit K=3 | 3 alarms, storm not triggered: the runner files one escalation per alarm |
| `storm/verdicts-current.json` | the same run, K=2 | 3 alarms > K, storm triggered: the runner files only the aggregate |

## Record schema

```
{
  "schema_version": 1,
  "generator": "shared.memory_eval_verdicts",
  "run_stamp": "<%Y%m%dT%H%M%SZ>",
  "entries": [
    {
      "eval_id": "<str>", "metric_id": "<str>",
      "rule_kind": "tripwire" | "proportion" | "count" | "scalar",
      "item": "<tripwire item_key>",          // exactly one of item / window
      "window": "trailing-<baseline_window>",
      "fingerprint": "<64 hex chars>",
      "verdict": "alarm" | "no_alarm" | "insufficient_data" | "grandfathered",
      "metric_status": "<the evaluator's VerdictStatus, verbatim>",
      "value": <float>, "limit_ref": "<str>", "run_stamp": "<stamp>", "detail": "<str>"
    }
  ],
  "storm_escape": {
    "triggered": <bool>, "alarm_count": <int>, "alarm_limit": <K>,
    "aggregate_fingerprint": "<64 hex chars, present iff triggered>"
  },
  "runner_failures": [
    {"scope": "<eval_id> | program", "phase": "produce" | "evaluate" | "runner",
     "error_type": "<str>", "message": "<str>", "exit_code": <int, optional>}
  ],
  "runner_failure_fingerprint": "<64 hex chars, present iff runner_failures is non-empty>"
}
```

The null convention is stated once, in the `shared.memory_eval_metrics`
module docstring. Here it means: `item` or `window`, `aggregate_fingerprint`,
`exit_code` and `runner_failure_fingerprint` are absent when they do not
apply. Read them with `.get`.

## Verdict vocabulary

| Evaluator status | Verdict |
|---|---|
| `alarm` (whole metric) | `alarm` |
| `ok`, `improved` | `no_alarm` (`metric_status` keeps `improved`) |
| `insufficient_data` | `insufficient_data` |
| scalar `ok` | `no_alarm` (`limit_ref` is `no alarm rule`) |
| tripwire item named by an alarm | `alarm` |
| tripwire item failing, not alarmed (known-bad) | `grandfathered` |
| tripwire item passing | `no_alarm` |

A first run (`baseline_snapshot`) grandfathers every failing item and alarms
nothing.

## Entry granularity: one entry per fingerprint

A tripwire contributes one entry per item. A proportion, count or scalar
metric contributes one entry, labelled with its trailing window. The
fingerprint identity is `(eval_id, metric_id, item | window)`. It never
includes a value, a stamp or prose, so a persisting regression keeps its
fingerprint across runs and a recovered item's `no_alarm` entry carries the
same fingerprint as its still-open escalation.

`dashboard/src/dashboard/data/memory_evals.py` renders one metric row per
entry, keyed `(eval_id, metric_id, item | none)`, so each tripwire item is its
own row. Its parity suite, `dashboard/tests/test_memory_evals_parity.py`, reads
these exemplars.

## Fingerprints are opaque

Each fingerprint is a sha256 hex digest. Join on whole-string equality with an
escalation's `dedupe_fingerprint`, and never parse one. The storm aggregate is
scoped to its run. The runner-failure fingerprint is program-wide and has no
stamp, so a multi-day outage keeps one pending escalation.

## Why a sibling of `memory_eval/`

The dashboard treats every directory under the artifact root as an eval.
`dashboard/tests/test_memory_evals_data.py::TestCommittedExemplarBoundary`
reads `shared/tests/fixtures/memory_eval/` as such a root and pins
`missing_verdicts` there. A verdicts file or an extra directory inside it
would break that test, so these exemplars live beside it. This tree holds no
`metrics-*.json`, and its only directories are `filed/` and `storm/`.

## Regenerating — do not hand-edit

```python
from pathlib import Path
from shared.memory_eval_metrics import load_metric_series, metrics_artifact_path
from shared.memory_eval_verdicts import build_verdicts_artifact, serialize_verdicts_artifact
from test_memory_eval_boundary import evaluate_exemplar  # tests/ on sys.path

stamp = '20260704T031500Z'
series = load_metric_series(
    metrics_artifact_path(Path('tests/fixtures/memory_eval'), 'e1-retrieval-health', stamp))
for name, k in (('filed', 3), ('storm', 2)):
    out = Path('tests/fixtures/memory_eval_verdicts') / name / 'verdicts-current.json'
    out.write_text(serialize_verdicts_artifact(build_verdicts_artifact(
        [(series, evaluate_exemplar())], run_stamp=stamp, alarm_limit=k)), encoding='utf-8')
```

The byte-identity test fails if a committed file drifts from what the builder
emits.
