"""Scorer Protocol + injected-seam type aliases for the T6 prompt-optimization loop.

See plans/tier1-prompt-optimization-prd.md T6: `Scorer` is the pluggable
scoring contract T5 (the reviewer/curator scorer) implements against the
generic optimization engine in `engine.py`. `RolloutFn` and `ProposeFn`
describe the other two async seams the engine calls through, so every
external effect (rollout on the executor, optimizer reflection, scoring) is
dependency-injected — making the dry-run integration test in
`test_prompt_opt_engine.py` fully hermetic (no real `invoke_agent` call) and
the engine reusable for the reviewer corpus (T2/T7), the curator corpus
(T5), and Tier 2.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = ['ProposeFn', 'RolloutFn', 'ScoredItem', 'Scorer']


@runtime_checkable
class Scorer(Protocol):
    """The pluggable scoring contract T5 implements.

    ``score`` returns a scalar in ``[0, 1]``, higher is better. The engine
    calls this once per ``(item, rollout)`` pair it needs to grade — never
    with the CONTRACT or heuristics text directly — so a Scorer
    implementation only ever judges the executor's *output*, staying
    agnostic to the prompt-optimization machinery around it.
    """

    async def score(self, item: Any, rollout: Any) -> float:
        ...


@dataclass(frozen=True)
class ScoredItem:
    """One ``(item, rollout, score)`` triple — the engine's per-item scoring record."""

    item: Any
    rollout: Any
    score: float


# (composed_prompt, item, model) -> rollout. The optimization engine ALWAYS
# calls this with model=executor_model (see engine.run_optimization_loop),
# structurally guaranteeing every rollout — and therefore every acceptance
# decision — is "scored on the ACTUAL executor model" (PRD invariant).
RolloutFn = Callable[[str, Any, str], Awaitable[Any]]

# (current_heuristics, scored_minibatch, rejected_buffer, ...) -> new
# heuristics block text. Receives ONLY the heuristics text — never the
# CONTRACT — so the CONTRACT is un-editable by construction (D-3). See
# optimizer.propose_heuristics_edit for the production implementation.
ProposeFn = Callable[..., Awaitable[str]]
