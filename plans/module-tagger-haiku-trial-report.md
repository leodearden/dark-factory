# module_tagger haiku replay-agreement trial

Task 2540 (Routing κ) — offline haiku-vs-fresh-sonnet replay over 30
historical done tasks carrying ground-truth `metadata.files`. The pass/fail
verdict is computed by this trial (δ's rollup is only the post-flip watch).

**Decision: FAIL**  (N = 30)

## Leaderboard — mean precision / recall / F1 vs ground truth

| Model | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| haiku | 0.401 | 0.436 | 0.370 |
| sonnet | 0.264 | 0.296 | 0.267 |

## Haiku-vs-sonnet agreement (symmetric, mean)

- Precision: 0.494
- Recall: 0.437
- F1: 0.441
- Jaccard: 0.368

## Frontier (opus) adjudication of haiku-vs-sonnet disagreements

- haiku better: 16
- sonnet better: 6
- tie: 3
- haiku_worse_fraction: 0.273

## Thresholds (design decision 5)

| Constant | Value |
|----------|-------|
| F1_PARITY_BAND | 0.05 |
| F1_FAIL_GAP | 0.15 |
| AGREEMENT_FLOOR | 0.7 |
| AGREEMENT_FAIL | 0.5 |
| ADJ_WORSE_PASS | 0.5 |
| ADJ_WORSE_FAIL | 0.6 |
| MIN_SAMPLES | 20 |

## Machine-readable summary

```json
{
  "n_samples": 30,
  "haiku": {
    "precision": 0.40138888888888885,
    "recall": 0.4355952380952381,
    "f1": 0.36987394957983194
  },
  "sonnet": {
    "precision": 0.2638888888888889,
    "recall": 0.2961111111111111,
    "f1": 0.26724867724867724
  },
  "agreement": {
    "precision": 0.49444444444444446,
    "recall": 0.43722222222222223,
    "f1": 0.4409857978279031,
    "jaccard": 0.3677046374105198
  },
  "adjudication": {
    "haiku_better": 16,
    "sonnet_better": 6,
    "tie": 3,
    "haiku_worse_fraction": 0.2727272727272727
  },
  "decision": "fail",
  "thresholds": {
    "F1_PARITY_BAND": 0.05,
    "F1_FAIL_GAP": 0.15,
    "AGREEMENT_FLOOR": 0.7,
    "AGREEMENT_FAIL": 0.5,
    "ADJ_WORSE_PASS": 0.5,
    "ADJ_WORSE_FAIL": 0.6,
    "MIN_SAMPLES": 20
  }
}
```

## Ratified outcome — flip applied + agreement-floor gate fixed (esc-2540-17)

> **This supersedes the pre-fix auto-rendered verdict above.** The live run's
> `Decision: FAIL` (and `"decision": "fail"` in the machine-readable block) was
> computed under the ORIGINAL agreement-floor gate, which hard-failed on
> `mean Jaccard 0.368 < AGREEMENT_FAIL 0.50` alone. Under the fixed gate
> (Part 2 below) those identical numbers reclassify to **MARGINAL** — the low
> agreement is no longer permitted to hard-fail a challenger that wins the
> primary quality signal.

**Decision (Leo, 2026-07-21, resolving the esc-2540-17 `design_concern`): FLIP
`module_tagger` to haiku (option a) AND fix the trial's agreement-floor gate
(option d).** The FAIL was driven SOLELY by the haiku-vs-sonnet agreement floor,
while on the primary quality signal (F1 vs ground truth) haiku BEATS the
incumbent sonnet. Both parts are implemented in this task.

### Part 1 — FLIP applied (option a)

`module_tagger`'s default model is flipped **sonnet → haiku** in both places, so
the code default and the shipped config agree:

- `orchestrator/src/orchestrator/defaults.yaml` — `models.module_tagger: "haiku"`
- `orchestrator/src/orchestrator/config.py` — `ModelsConfig.module_tagger` Field default `'haiku'`

haiku is already in `routing.allowed_models`, so no admission change was needed
(verified: 308 orchestrator config/routing/harness-tagging tests green). The
running `dark-factory-orchestrator.yaml` operator config declares **no** `models`
block, so `defaults.yaml` governs the fleet: the flip takes effect on the next
orchestrator deploy/restart. This replaces the earlier plan's
hot-reload-of-`dark-factory-orchestrator.yaml` step — the ratified home for the
flip is the fleet-wide `defaults.yaml`, not a per-instance operator override.

### Ground-truth-F1 evidence (why haiku wins)

| Signal | haiku | sonnet |
|--------|-------|--------|
| mean F1 vs ground truth | **0.370** | 0.267 |
| mean precision | 0.401 | 0.264 |
| mean recall | 0.436 | 0.296 |

- Opus frontier adjudication of the 25 disagreements: **haiku better on 16**,
  sonnet better on 6, 3 ties (`haiku_worse_fraction = 0.273`).
- Cost: haiku is ~10× cheaper per invocation than sonnet.
- Absolute F1 is low for both models because exact merge-diff file prediction
  from title+description is a demanding target; production locks at coarser
  module granularity, so the *comparative* signal is what matters.

### Part 2 — agreement-floor gate fixed (option d)

`decide()` in `scripts/trial_module_tagger_haiku.py` now gates the
agreement-floor hard-fail (`mean Jaccard < AGREEMENT_FAIL`) so it fires only when
the challenger is **also not better** than the incumbent on ground-truth mean F1.
A challenger that beats the incumbent on the primary signal can no longer be
hard-failed by low agreement alone — it lands in the **marginal** band and
escalates to a human instead of auto-failing (and still cannot auto-PASS while
Jaccard is below `AGREEMENT_FLOOR`). The other two hard-fail triggers (F1 gap,
opus majority-worse) and the PASS floor are unchanged. Covered by a RED→GREEN
test in `tests/scripts/test_trial_module_tagger_haiku.py`; confined to the trial
script + its test per PRD decision 11 (no `evals/` machinery touched).

### Post-flip watch (δ, task 2534)

δ's `digest.model_role_rollup` is the OUTCOME watch, never a pass/fail basis:
once this lands, haiku `module_tagger` invocation/cost rows (NULL `task_id`)
confirm the flip is live and quantify the F1/$ trade in production.
