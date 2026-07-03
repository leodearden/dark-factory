"""I2 shape-validation invariants for the merge-queue message dataclasses (task 1990).

Retires the task-1928 field-drop bug class structurally: SpeculativeItem and
InflightEntry validate their own shape at construction time instead of relying
on every call site to hand-build a consistent set of fields.

step-1  RED  — SpeculativeItem XOR / merge_wt-iff-merge_result / already_delivered
step-3  RED  — InflightEntry passthrough_outcome shadow invariant
step-7  RED  — InflightStatus str-compatible Enum
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from orchestrator.git_ops import MergeResult
from orchestrator.merge_queue import MergeOutcome, SpeculativeItem

# ── step-1: SpeculativeItem shape validation ─────────────────────────────────


def _real_kwargs() -> dict:
    """Minimal well-formed REAL SpeculativeItem kwargs (merge_result+merge_wt set)."""
    return dict(
        request=MagicMock(),
        merge_result=MergeResult(success=True, merge_commit='deadbeef'),
        merge_wt=Path('/fake/merge-wt'),
        base_sha='aabbccdd',
        speculative=False,
        skip_verify=False,
    )


def _decided_kwargs() -> dict:
    """Minimal well-formed DECIDED SpeculativeItem kwargs (immediate_outcome set)."""
    return dict(
        request=MagicMock(),
        merge_result=None,
        merge_wt=None,
        base_sha='aabbccdd',
        speculative=False,
        skip_verify=False,
        immediate_outcome=MergeOutcome('blocked', reason='test'),
    )


class TestSpeculativeItemXorInvariant:
    """Exactly one of {merge_result, immediate_outcome} may be non-None."""

    def test_both_set_raises(self) -> None:
        kwargs = _real_kwargs()
        kwargs['immediate_outcome'] = MergeOutcome('conflict')
        with pytest.raises(ValueError):
            SpeculativeItem(**kwargs)

    def test_both_none_raises(self) -> None:
        kwargs = _real_kwargs()
        kwargs['merge_result'] = None
        kwargs['merge_wt'] = None
        with pytest.raises(ValueError):
            SpeculativeItem(**kwargs)

    def test_real_shape_constructs(self) -> None:
        item = SpeculativeItem(**_real_kwargs())
        assert item.merge_result is not None
        assert item.immediate_outcome is None

    def test_decided_shape_constructs(self) -> None:
        item = SpeculativeItem(**_decided_kwargs())
        assert item.immediate_outcome is not None
        assert item.merge_result is None


class TestSpeculativeItemMergeWtInvariant:
    """merge_wt is not None iff merge_result is not None."""

    def test_merge_result_without_merge_wt_raises(self) -> None:
        kwargs = _real_kwargs()
        kwargs['merge_wt'] = None
        with pytest.raises(ValueError):
            SpeculativeItem(**kwargs)

    def test_merge_wt_without_merge_result_raises(self) -> None:
        kwargs = _decided_kwargs()
        kwargs['immediate_outcome'] = None
        kwargs['merge_wt'] = Path('/fake/merge-wt')
        with pytest.raises(ValueError):
            SpeculativeItem(**kwargs)


class TestSpeculativeItemAlreadyDeliveredInvariant:
    """already_delivered=True requires immediate_outcome to be set."""

    def test_already_delivered_without_immediate_outcome_raises(self) -> None:
        kwargs = _real_kwargs()
        kwargs['already_delivered'] = True
        with pytest.raises(ValueError):
            SpeculativeItem(**kwargs)

    def test_already_delivered_with_immediate_outcome_constructs(self) -> None:
        kwargs = _decided_kwargs()
        kwargs['already_delivered'] = True
        item = SpeculativeItem(**kwargs)
        assert item.already_delivered is True
