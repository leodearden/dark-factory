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

## Decide-and-act outcome (step 16)

**Action: NO FLIP — config left unchanged.** `module_tagger` remains on its
configured incumbent (sonnet). `dark-factory-orchestrator.yaml` was **not**
edited and no hot-reload was performed, per the plan's rule "do NOT flip on
anything but a clear pass."

**Why the verdict is FAIL:** the only hard-fail trigger that fired is the
haiku-vs-sonnet **agreement floor** — mean Jaccard `0.368` < `AGREEMENT_FAIL`
`0.50`. Neither F1-gap nor the adjudication trigger fired.

**Important nuance (surfaced to a human via a `design_concern` escalation):**
the fail is driven by low *agreement*, not by haiku underperforming. On the
primary quality signal (F1 vs ground truth) haiku (`0.370`) **beats** the
incumbent sonnet (`0.267`) — an F1 gap of `-0.103` in haiku's favour — and the
opus frontier judged haiku better on **16** of the 25 disagreements vs **6**
for sonnet (3 ties; `haiku_worse_fraction=0.273`). So the low agreement
reflects sonnet being the weaker model on this recent-30 sample, not haiku
being unreliable — the opposite of the divergence the agreement floor was
designed to catch. Absolute F1 is low for both models because exact merge-diff
file prediction from title+description is a demanding target; production locks
at coarser module granularity, so the *comparative* signal is what matters.

Per the plan's fail/marginal branch this is escalated (`escalate_blocker`,
category `design_concern`) for human adjudication of the threshold tension —
whether to flip anyway on haiku's ground-truth + cost edge (~10× cheaper),
re-run at larger N / higher effort to tighten the estimate, keep sonnet, or
revisit whether `AGREEMENT_FAIL` should hard-fail when the divergence favours
the challenger.

**Deferred follow-on (NOT performed):** the `defaults.yaml` bake
(module_tagger sonnet→haiku) was contingent on a PASS + the δ-rollup watch
window, neither of which occurred; it is not done.
