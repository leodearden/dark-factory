"""HARD, LLM-free action-match scorer for the curator replay corpus (T5).

See plans/tier1-prompt-optimization-prd.md T5/D-5. `CuratorActionScorer`
implements the T6 `Scorer` Protocol (orchestrator.evals.prompt_opt.scorer):
a deterministic action-match score against a gold label, with no LLM judge
call -- a low-variance "hard signal", contrasted with the reviewer's noisy
haiku matcher (D-5). Reads `gold_*` fields off *item* and the predicted
fields off *rollout* -- both dict-keyed OR attribute-shaped, since the T6
engine's injected `rollout_fn` return shape isn't fixed by this module.

Conformance to the T6 `run_optimization_loop` engine seam -- driven purely
on the executor model, no fake Scorer stand-in -- is proven hermetically by
test_prompt_opt_curator_scorer.py::TestCuratorActionScorerEngineConformance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ['CuratorActionScorer']


def _read_action(obj: Any, *, prefix: str = '') -> tuple[str | None, str | None, str | None]:
    """Extract ``(action, target_fingerprint, target_id)`` from *obj*.

    *obj* may be a plain dict (key access) or a duck-typed object (attribute
    access). *prefix* selects the gold-label fields on a corpus item
    (``'gold_'`` -> ``gold_action``/``gold_target_fingerprint``/
    ``gold_target_id``) vs. the bare predicted fields on a rollout (``''``
    -> ``action``/``target_fingerprint``/``target_id``).
    """

    def _get(key: str) -> Any:
        full_key = f'{prefix}{key}'
        if isinstance(obj, dict):
            return obj.get(full_key)
        return getattr(obj, full_key, None)

    return _get('action'), _get('target_fingerprint'), _get('target_id')


@dataclass
class CuratorActionScorer:
    """HARD action-match scorer: create/drop equality, combine-target correctness.

    ``score`` in ``[0, 1]``: an action mismatch (e.g. gold ``create`` vs
    predicted ``combine``) always scores 0.0. A create/create or drop/drop
    match scores 1.0. A combine/combine match scores 1.0 only when the
    combine TARGET also matches -- keyed on ``target_fingerprint`` (the
    stable combine key; task ids get reallocated over a tickets.db's
    lifetime) and falling back to ``target_id`` when a fingerprint isn't
    available on both sides. A combine whose action matches but target
    doesn't scores ``combine_target_partial_credit`` (default 0.0 -- a crisp
    binary by default; exposed as a constructor param so the "action right,
    target wrong" sub-signal can be tuned without a code change).
    """

    combine_target_partial_credit: float = 0.0

    async def score(self, item: Any, rollout: Any) -> float:
        gold_action, gold_fingerprint, gold_target_id = _read_action(item, prefix='gold_')
        pred_action, pred_fingerprint, pred_target_id = _read_action(rollout)

        if gold_action != pred_action:
            return 0.0
        if gold_action != 'combine':
            return 1.0

        if gold_fingerprint is not None and pred_fingerprint is not None:
            matched = gold_fingerprint == pred_fingerprint
        else:
            matched = gold_target_id == pred_target_id

        return 1.0 if matched else self.combine_target_partial_credit
