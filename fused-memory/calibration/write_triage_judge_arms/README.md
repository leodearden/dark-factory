# Write-triage judge arms: prompt by model

Four arms of `scripts/eval_write_triage_judge.py`: the judge prompt
{`shipped`, `psi`} by the judge model {`gpt-6.1-sol`, the configured one;
`gpt-6.2-sol`}. All four judge ONE retrieval, so that
`scripts/compare_write_triage_judge_arms.py` can separate the prompt effect
from the model effect. Task 9045 built the arms and task 11213 runs them.

This directory holds what the runs write: `<wording>__<model>.json`, its `.md`
render and its `.cases.jsonl` for three of the arms, and `comparison.json` with
its `comparison.md`. The fourth arm, `shipped` x `gpt-6.1-sol`, is the committed
arbiter, `calibration/write_triage_judge_accuracy_report.json`.

Run every command from the fused-memory package root.

## 0. The ChatGPT-plan proxy

The eval refuses a live judge unless `write_triage.judge_api_url` and
`write_triage.judge_api_key` are both set. `config.yaml` fills them from these
two variables (task 9946). If they are unset, the run stops before the fixture
is loaded rather than billing `OPENAI_API_KEY`.

```bash
export OPENAI_PLAN_PROXY_URL=...   # the plan proxy's OpenAI-compatible base URL
export OPENAI_PLAN_PROXY_KEY=...   # its client key
```

## 1. Smoke

Two labelled records from two clusters, judged by `gpt-6.2-sol` and written to
`/tmp`. A completed report confirms that the plan serves the model. `--limit 1`
is refused, because a single cluster leaves the distractor control an empty
slate (`EvalPlan.__post_init__`, pinned by
`tests/test_eval_write_triage_judge_live_edge.py::TestTheLimitedCli::test_a_one_record_limit_is_refused_before_anything_is_written`).

```bash
python scripts/eval_write_triage_judge.py --limit 2 --judge-model gpt-6.2-sol \
    --report-path /tmp/write_triage_judge_smoke.json
```

## 2. The arbiter arm, `shipped` x `gpt-6.1-sol`

This run writes the committed artifact. It retrieves once and uses
`--slates-out` to snapshot the slates, outside the tracked tree, for the other
three arms. Keep the snapshot until step 4 has run. `--judge-model` is refused
on this report path, so the model is the configured one.

```bash
python scripts/eval_write_triage_judge.py --slate-mode retrieved --project-id reify \
    --canonical-aliases tests/fixtures/write_triage_calibration.canonical_aliases.json \
    --report-path calibration/write_triage_judge_accuracy_report.json \
    --cases-path calibration/write_triage_judge_accuracy_report.cases.jsonl \
    --slates-out /tmp/write_triage_judge_arms.slates.json
```

## 3. The other three arms

Each run judges the snapshot through `--slates-in`, and takes its project from
the snapshot.

```bash
python scripts/eval_write_triage_judge.py --slate-mode retrieved \
    --slates-in /tmp/write_triage_judge_arms.slates.json \
    --canonical-aliases tests/fixtures/write_triage_calibration.canonical_aliases.json \
    --wording psi \
    --report-path calibration/write_triage_judge_arms/psi__gpt-6.1-sol.json \
    --cases-path calibration/write_triage_judge_arms/psi__gpt-6.1-sol.cases.jsonl
python scripts/eval_write_triage_judge.py --slate-mode retrieved \
    --slates-in /tmp/write_triage_judge_arms.slates.json \
    --canonical-aliases tests/fixtures/write_triage_calibration.canonical_aliases.json \
    --judge-model gpt-6.2-sol \
    --report-path calibration/write_triage_judge_arms/shipped__gpt-6.2-sol.json \
    --cases-path calibration/write_triage_judge_arms/shipped__gpt-6.2-sol.cases.jsonl
python scripts/eval_write_triage_judge.py --slate-mode retrieved \
    --slates-in /tmp/write_triage_judge_arms.slates.json \
    --canonical-aliases tests/fixtures/write_triage_calibration.canonical_aliases.json \
    --wording psi --judge-model gpt-6.2-sol \
    --report-path calibration/write_triage_judge_arms/psi__gpt-6.2-sol.json \
    --cases-path calibration/write_triage_judge_arms/psi__gpt-6.2-sol.cases.jsonl
```

Every arm passes the same `--canonical-aliases`. None passes `--limit`,
`--field-chars` or `--judge-candidate-count`, because the comparison refuses
arms whose population provenance differs.

## 4. The comparison

The comparison refuses arms that did not judge the same population, or that do
not fill the grid. It writes `comparison.md` beside `comparison.json`.

```bash
python scripts/compare_write_triage_judge_arms.py \
    --arm calibration/write_triage_judge_accuracy_report.json \
    --arm calibration/write_triage_judge_arms/psi__gpt-6.1-sol.json \
    --arm calibration/write_triage_judge_arms/shipped__gpt-6.2-sol.json \
    --arm calibration/write_triage_judge_arms/psi__gpt-6.2-sol.json \
    --out calibration/write_triage_judge_arms/comparison.json
```
