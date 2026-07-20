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
