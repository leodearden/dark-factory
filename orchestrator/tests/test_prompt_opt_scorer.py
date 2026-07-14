"""Tests for orchestrator.evals.prompt_opt.scorer — the T6 Scorer Protocol surface.

`Scorer` is the pluggable scoring contract T5 (the reviewer/curator scorer)
implements; `RolloutFn`/`ProposeFn` are the type aliases describing the other
two async seams the optimization engine calls through. See
plans/tier1-prompt-optimization-prd.md T6.
"""

from __future__ import annotations

import collections.abc
import typing
from dataclasses import FrozenInstanceError

import pytest

from orchestrator.evals.prompt_opt.scorer import (
    ProposeFn,
    RolloutFn,
    ScoredItem,
    Scorer,
)


class _ConformingScorer:
    """Minimal object satisfying the Scorer Protocol's structural shape."""

    async def score(self, item, rollout) -> float:
        return 1.0


class _NonConformingScorer:
    """Missing `score` entirely — must fail the isinstance check."""

    def other_method(self) -> None:
        ...


class TestScorerProtocol:
    def test_is_runtime_checkable(self) -> None:
        # A @runtime_checkable Protocol supports isinstance() checks; a
        # plain (non-runtime-checkable) Protocol raises TypeError here, so
        # the fact that this returns (rather than raises) is itself the
        # signal — assert the boolean result explicitly rather than relying
        # on the absence of an exception.
        result = isinstance(_ConformingScorer(), Scorer)
        assert result is True

    def test_conforming_fake_passes_isinstance(self) -> None:
        assert isinstance(_ConformingScorer(), Scorer)

    def test_non_conforming_object_fails_isinstance(self) -> None:
        assert not isinstance(_NonConformingScorer(), Scorer)

    def test_plain_object_fails_isinstance(self) -> None:
        assert not isinstance(object(), Scorer)

    @pytest.mark.asyncio
    async def test_conforming_fake_score_is_awaitable(self) -> None:
        conforming = _ConformingScorer()
        result = await conforming.score(item='x', rollout='y')
        assert result == 1.0


class TestScoredItem:
    def test_holds_item_rollout_score(self) -> None:
        scored = ScoredItem(item='the-item', rollout='the-rollout', score=0.75)
        assert scored.item == 'the-item'
        assert scored.rollout == 'the-rollout'
        assert scored.score == 0.75

    def test_is_frozen(self) -> None:
        scored = ScoredItem(item='i', rollout='r', score=0.5)
        with pytest.raises(FrozenInstanceError):
            scored.score = 0.9  # type: ignore[misc]


class TestTypeAliases:
    def test_rollout_fn_is_a_callable_type_alias(self) -> None:
        # RolloutFn = Callable[[str, Any, str], Awaitable[Any]] — assert it
        # is actually a `collections.abc.Callable` generic alias (the seam's
        # real shape), not just "importable and non-None".
        assert typing.get_origin(RolloutFn) is collections.abc.Callable

    def test_propose_fn_is_a_callable_type_alias(self) -> None:
        # ProposeFn = Callable[..., Awaitable[str]] — same shape check.
        assert typing.get_origin(ProposeFn) is collections.abc.Callable
