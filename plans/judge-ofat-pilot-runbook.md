# Judge-model OFAT pilot — decide-and-act runbook (Routing κ / task 2815)

**Goal.** Decide whether to switch the ζ completion judge from Sonnet to Haiku
(`models.judge`) by trialling `judge-haiku` against the `judge-sonnet` incumbent
over the eval framework, and — **only on a clear ADOPT verdict** — apply the
flip. This is a decide-and-**act** pilot: the tooling produces a committed,
byte-deterministic report and an exit code; a human/CI reads the verdict and
either flips the config or escalates.

This runbook is the operator surface. The pure logic lives in
`orchestrator/src/orchestrator/evals/judge_pilot.py`
(`decide_judge_adoption` / `analyze_judge_ofat` / `format_judge_ofat_report`);
the CLI is `scripts/run_judge_ofat_pilot.py`. Neither edits the eval instrument
(runner.py / configs.py / the judge / the Elo machinery) — the eval-revival ο
seam (task 2825) owns it (PRD decision 11); this pilot is a **consumer** of ο's
composite.

---

## Why OFAT, and what is measured

The judge is scored **indirectly**, through μ's end-to-end composite, not by any
new judge-verdict metric. A too-lenient judge stops iteration early → a worse
diff (lower quality); a too-strict judge burns iterations → higher cost. So a
cheaper judge that preserves the end-to-end composite is a pure win, and one that
degrades it is not — the composite already prices both failure modes in.

`JUDGE_EVAL_CONFIGS` (configs.py) varies exactly ONE role — the ζ completion
judge — while the implementer is pinned to `JUDGE_OFAT_IMPLEMENTER_PIN`
(`claude-sonnet-max`) and architect/reviewer stay fixed. That is true OFAT: the
composite delta between `judge-haiku` and `judge-sonnet` isolates the judge.

- `judge-sonnet` — `claude/sonnet/medium`, reproduces today's incumbent judge.
- `judge-haiku` — `claude/haiku/medium`, the cheaper candidate.

Both at `effort='medium'` (the production judge default) so ONLY the model
varies. `backends.judge` stays `claude` and `budgets.judge` stays `0.50` pinned
(an always-Claude, read-only quality call).

---

## 1. Run the pilot

The live run spawns full architect→implementer→debugger→reviewer→judge workflows
with a Sonnet-max implementer over fixtures × trials — **hours of wall-clock and
real API spend.** Run it as an operator/CI step, never inline in a normal task.

```bash
# From the repo root. --trials >=3 is REQUIRED: the adoption gate needs each
# composite CI to be `sufficient` (n>=3 trials/cell), else the verdict is a loud
# `marginal` (insufficient data) rather than adopt/reject.
scripts/run_judge_ofat_pilot.py --run \
  --trials 3 \
  --config dark-factory-orchestrator.yaml \
  --out plans/judge-ofat-pilot-report.md
```

`--run` fans `run_ofat_stage(task_paths, JUDGE_EVAL_CONFIGS, trials=N)` over the
packaged fixtures (`orchestrator/src/orchestrator/evals/tasks/`, override with
`--tasks-dir`), persists each `EvalResult` under `evals/results/`, then analyzes
and writes the report. Re-analyze already-persisted results without re-running:

```bash
scripts/run_judge_ofat_pilot.py --analyze-only \
  --results-dir orchestrator/src/orchestrator/evals/results \
  --out plans/judge-ofat-pilot-report.md
```

**Exit code = verdict** (so CI can gate the flip): `0` adopt · `1` marginal ·
`2` reject.

```bash
if scripts/run_judge_ofat_pilot.py --analyze-only --out plans/judge-ofat-pilot-report.md; then
  echo "ADOPT — proceed to the flip below"
else
  echo "marginal/reject — keep sonnet, escalate with the report"
fi
```

Tuning knobs: `--margin` (non-inferiority margin, default `0.05`),
`--incumbent` / `--candidate` (config names, default `judge-sonnet` /
`judge-haiku`).

---

## 2. Adoption criterion

`decide_judge_adoption` adopts iff **all three** hold (else it escalates):

1. **Non-inferior** — candidate composite CI **lower bound** ≥ incumbent
   composite **mean** − `margin`. The one-sided non-inferiority test; `margin` is
   an exposed policy knob (default `0.05` composite points), **not** an
   empirically-asserted bound. Tolerates at most a 5-composite-point regression.
2. **Cheaper** — candidate mean `cost_usd` < incumbent mean `cost_usd`.
3. **Sufficient** — **both** composite CIs are `sufficient` (n≥3 trials/cell,
   from `report.mean_ci95`).

Verdict taxonomy → `escalate` flag:

| Verdict | When | escalate |
|---|---|---|
| `adopt` | non-inferior AND cheaper AND both CIs sufficient | no |
| `reject` | NOT non-inferior (with sufficient data) | yes |
| `marginal` | non-inferior but NOT cheaper | yes |
| `marginal` | either composite CI NOT sufficient (n<3) | yes |
| `marginal` | a required config row (`judge-sonnet` / `judge-haiku`) missing | yes |

Insufficient data and a missing row are **loud** `marginal` outcomes (structured
`reasons`, escalate) — never a silent pass or an unhandled error
(loud-over-silent / structured-facts-at-failure).

**Cost is embedded in the composite — read the raw quality delta on a borderline
verdict.** ο's `build_composite_report` composite is *already* cost-normalized
per fixture (it blends quality with a cost/latency ratio score), and the judge
OFAT run contains only the two judge configs — so the cheaper candidate
mechanically gets a cost-ratio boost to its composite. "Non-inferior on
composite" is therefore partly the same cost win the separate **Cheaper** check
already requires (cost is counted on both axes), which for a *borderline* case
could let a modest genuine quality regression on the cheaper judge still read as
`adopt`. The gate is deliberately left as composite-non-inferiority AND cheaper
(PRD decision 11 forbids editing the instrument here), but the report surfaces
the **raw quality delta** — the un-cost-normalized `quality` row/axis — right next
to the composite delta. When a verdict is borderline (composite delta near zero,
CI lower bound near the margin), inspect that raw quality delta: a small negative
composite delta hiding a large negative raw quality delta means the composite is
being carried by the cost boost, not by preserved quality — treat that as a
`reject`-leaning signal and escalate rather than flipping.

---

## 3a. On ADOPT — apply the flip (staged, reversible)

Do NOT jump straight to `defaults.yaml`. Flip the **running** config first, watch
the outcome a fixed window, and only then persist.

1. **Flip the running orchestrator config.** In the operational
   `dark-factory-orchestrator.yaml`, set the judge model under `models:`:

   ```yaml
   models:
     judge: "haiku"   # was: inherits defaults.yaml -> "sonnet"
   ```

   `models` is a **green-tier** hot-reloadable knob.

2. **Hot-reload** (no restart): `mcp__escalation__reload_config`. Confirm the
   returned `applied` disposition includes the `models.judge` change (do not
   trust the top-level `reloaded` flag alone).

3. **Watch the δ / task 2534 rollup a fixed window** (e.g. 24–48h of live
   traffic) for `haiku` judge-invocation rows. δ measures **outcome** rates
   (done / blocked / cost) post-flip — confirm no regression in merge outcomes or
   iteration counts and that the judge-cost line drops as the pilot predicted.

4. **Persist to `defaults.yaml`.** Once the watch window is clean, set the
   fleet-wide default at `orchestrator/src/orchestrator/defaults.yaml:271`:

   ```yaml
   judge: "haiku"   # models.judge — was "sonnet"
   ```

   Commit it (this is the durable flip; the step-1 running-config edit was the
   canary).

**Rollback.** If the watch window regresses, revert the running-config
`models.judge` to `sonnet` and `reload_config` again — no restart, no
`defaults.yaml` change needed.

## 3b. On MARGINAL / REJECT — keep sonnet, escalate

Leave `models.judge=sonnet` unchanged. File an escalation
(`mcp__escalation__escalate_blocker`, category `design_concern`) and **attach the
generated report** (`plans/judge-ofat-pilot-report.md`). The report's structured
`reasons` field states exactly why (not non-inferior / not cheaper / insufficient
trials / missing row); include the `verdict` and the composite + cost deltas.
For a `marginal`-insufficient verdict, the remedy is usually **more trials**
(re-run `--run --trials` higher), not a config change.

---

## Report template

`format_judge_ofat_report` renders this deterministically (no wall-clock, so it
is byte-stable and diff-clean). Shape:

```
# Judge-model OFAT pilot report

verdict: <adopt|marginal|reject> (escalate: <yes|no>)
incumbent: judge-sonnet | candidate: judge-haiku
non-inferior: <yes|no> | cheaper: <yes|no> | sufficient: <yes|no> | margin: 0.0500
composite delta (judge-haiku - judge-sonnet): <±d.dddd>
raw quality delta (judge-haiku - judge-sonnet): <±d.dddd>
cost delta (judge-haiku - judge-sonnet): <±d.dddd>
judge cost delta (judge-haiku - judge-sonnet): <±d.dddd>

reasons:
  - <structured reason string>

composite report:            <-- embedded report.format_composite_table
config        composite  quality  cost_usd  ...  ci95_composite      trials  fixtures
...
price table:
...

RECOMMENDATION: <adopt — flip models.judge: sonnet -> haiku ...>
             | <keep models.judge=sonnet and escalate with this report ...>
```

Deltas are candidate-minus-incumbent (a negative `cost delta` == the candidate is
cheaper). The `raw quality delta` is the un-cost-normalized quality axis, printed
next to the cost-embedding `composite delta` so the two can be compared on a
borderline verdict (see §2's cost-in-composite note). The embedded composite table
is the same surface every other eval stage renders, so the pilot report stays
consistent with `eval-ofat` output.

---

## Cross-references

- PRD §κ-judge (decisions 10/11): decide-and-act; do not edit the eval instrument.
- ο judge-OFAT seam (task 2825): `JUDGE_EVAL_CONFIGS`, `JUDGE_OFAT_IMPLEMENTER_PIN`,
  `run_ofat_stage`'s judge branch, `build_eval_orch_config(..., judge_config=)`.
- δ / task 2534: the post-flip outcome + cost watch this runbook's step 3a.3 uses.
- Config hot-reload tiers: `CLAUDE.md` "Orchestrator Config Reload" (`models` is
  green-tier) and `plans/config-hot-reload-prd.md`.
