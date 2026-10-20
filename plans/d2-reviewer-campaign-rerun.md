# Reviewer campaign rerun under d2: rulings D2, D6 and D7

Operator runbook for task 5438. The rerun measures three things on the
context-complete (`d2`) reviewer briefing:

- D2, the d2 gate;
- D6, Fable 5.1 against the incumbent;
- D7, the cross-family (GPT-6 Astra) arm.

The decision rules were pre-registered in code before any of it runs. Each step
below is one action.

## 1. Where

1. Run every command from the MAIN checkout's `orchestrator/`, after task 5438
   has merged.
2. Pass an ABSOLUTE results dir. A relative one resolves against
   `orchestrator/`:

   ```bash
   RESULTS=/home/leo/src/dark-factory/plans/fable-evidence-package-2026-09-10/campaign-d2
   ```

The evidence package (`plans/fable-evidence-package-2026-09-10/` and
its `.md`) is untracked and lives only in the main checkout. That covers its
scripts too: `cross_family_arm.py`, `reduce_campaign.py` and
`complement_adjudication.md`.

## 2. Invocation A: the D2 gate

Run the command from
[`d2-context-complete-reviewer-gate.md`](d2-context-complete-reviewer-gate.md)
§1 into `$RESULTS`:

```bash
uv run python -m orchestrator.evals.reviewer_trial campaign \
  --results-dir "$RESULTS" --split test --trials 3 --max-parallel 6 \
  --arms variant_a --briefing diff-only,d2
```

- Keep `--trials 3` for the diff-only arm as well. The rule
  `orchestrator/src/orchestrator/evals/reviewer_trial/campaign.py::non_inferior`
  needs three paired trials, so a single diff-only trial fails D2 by
  construction.
- Afterwards, copy `$RESULTS/campaign_report.{md,json}` to
  `$RESULTS/d2-gate-report.{md,json}`. Every invocation overwrites
  `campaign_report.*`.

## 3. Invocation B: the D6 arms

Run into the SAME `$RESULTS`:

```bash
uv run python -m orchestrator.evals.reviewer_trial campaign \
  --results-dir "$RESULTS" --split test --trials 3 --max-parallel 6 \
  --arms variant_a,variant_fable51_solo,variant_a_xhigh,variant_sonnet5_solo \
  --briefing diff-only,d2
```

- **The CLI refuses `--briefing d2` on its own.**
  `orchestrator/src/orchestrator/evals/reviewer_trial/__main__.py::_parse_briefings`
  requires `diff-only` first whenever d2 is listed, so this command also runs
  every new arm's diff-only generation. That is 333 panels beyond the d2 arms
  (see §6).
- **The incumbent's panels are not paid twice.**
  `orchestrator/src/orchestrator/evals/reviewer_trial/runner.py::run_trial`
  reuses Invocation A's `variant_a__<fixture>.json` and
  `variant_a_d2__<fixture>.json` checkpoints by file name.
- Afterwards, copy `campaign_report.{md,json}` to `d6-report.{md,json}`. A
  rerun with identical arguments only reduces, so an overwritten report can
  be regenerated for free.

## 4. The Astra arm (D7)

Keep codex on the ChatGPT login and never the API key (Leo's ruling).

**If task 5403 has landed**, it adds the ChatGPT usage-window handling. Run:

```bash
uv run python -m orchestrator.evals.reviewer_trial campaign \
  --results-dir "$RESULTS" --split test --trials 3 \
  --arms variant_a,variant_cross_family --briefing diff-only,d2
```

This also runs the diff-only Astra arm, which spends ChatGPT windows too.

**If 5403 has not landed**, the CLI has no window handling. A cap-hit panel is
persisted as `ERROR` and reused on resume, so delete those result files
before you resume.

To spend windows on the d2 arm only, adapt the untracked
`cross_family_arm.py`. It predates d2: it runs the diff-only
`VARIANT_CROSS_FAMILY` into `campaign/` and writes panel results only. It needs
three changes:

- run `with_briefing(VARIANT_CROSS_FAMILY, ReviewBriefing.CONTEXT_COMPLETE)`
  (`variants.py::with_briefing`);
- write into `$RESULTS/trial-<n>`;
- score each result with `scorer.py::score_panel_run`, writing it to
  `trial-<n>/scores/` only when `unmeasured` is None. That is the CLI's rule
  in `__main__.py::campaign`. `reduce_campaign.py` caches unmeasured scores
  too, so do not point it at `$RESULTS`.

Then hand-adjudicate every `variant_cross_family_d2` complement finding, the
way `complement_adjudication.md` did. Unresolved items go to Leo.

## 5. The verdicts

Every `campaign_report` pairs each arm against `variant_a`, the diff-only
incumbent. D6 and D7 are judged against `variant_a_d2`. So re-reduce the
cached d2 score records with that incumbent; it spends nothing:

```bash
uv run python - <<'PY'
import json
from pathlib import Path
from orchestrator.evals.reviewer_trial import campaign
from orchestrator.evals.reviewer_trial.corpus import CorpusManifest

root = Path('/home/leo/src/dark-factory/plans/fable-evidence-package-2026-09-10/campaign-d2')
corpus = CorpusManifest.load(Path('src/orchestrator/evals/reviewer_trial/corpus/manifest.json'))
blocking = {d.diff_id: {g.id for g in d.blocking_issues()} for d in corpus.diffs}
records = [
    json.loads(path.read_text()) | {'trial': trial}
    for trial in (1, 2, 3)
    for path in sorted((root / f'trial-{trial}' / 'scores').glob('*_d2__*.json'))
]
arms, decisions = campaign.summarize_campaign(records, 'variant_a_d2', blocking)
(root / 'report.md').write_text(campaign.format_markdown(arms, decisions) + '\n')
(root / 'report.json').write_text(json.dumps(campaign.to_json(arms, decisions), indent=2))
PY
```

Unmeasured scores are never cached, so `report.md` undercounts them. Read the
unmeasured counts from the copied CLI reports instead. The decisions are
unaffected, because every figure is over measured cells only.

Read the three verdicts. Each rule lives in
`orchestrator/src/orchestrator/evals/reviewer_trial/campaign.py`:

- **D2**: the `variant_a_d2` line for `::non_inferior` in `d2-gate-report.md`.
- **D6**: the `variant_fable51_solo_d2` line for `::strict_improvement` in
  `report.md`. Report its blocking precision (`blocking P`) and its
  `$/blocking finding` alongside the verdict.
- **D7**: the `variant_cross_family_d2` line for `::clears_self_complement_band`
  in `report.md`, AND an adjudicated complement precision of at least 50%.
  The noise floor is the incumbent's own band, `::self_complement_band`. The
  report prints it on the "Incumbent self-complement band" line.

## 6. Budget

- Expect wall clock to bind before dollars. Spend falls on pool-hours as well
  as dollars, and the Astra arm on ChatGPT Pro windows.
- Cut arms, never fixtures, and never select fixtures on outcome.
- d2 arms skip 4 of the 37 test fixtures (gate runbook §1), so a d2 arm is
  99 panels over three trials and a diff-only arm is 111. Expected panel
  counts:

  | Invocation | Panels | What runs |
  |---|---|---|
  | A | 210 | `variant_a` diff-only and d2 |
  | B | 630 | the three new arms, d2 (297) and diff-only (333) |
  | Astra, d2 only | 99 | `variant_cross_family_d2` |

- B's 333 diff-only panels are the largest avoidable spend. D6 needs only the
  d2 arms, but the CLI cannot run them alone today.

After the run, `add_memory` the headline numbers and hand the three verdicts
to the study session.
