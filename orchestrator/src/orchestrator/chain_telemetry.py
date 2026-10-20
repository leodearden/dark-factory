"""Deep merge-ahead chain telemetry: histogram, deep-fail rate and items landed per deep verify.

Pure aggregation over already-parsed ``merge_verify`` and ``merge_finalized``
payloads (task 3188, PRD ε of plans/deep-merge-ahead-prd.md).  The fields it
reads are defined by the payload contracts on
orchestrator/src/orchestrator/event_store.py::EventType (``merge_verify`` and
``merge_finalized``); this module consumes them and does not restate them.
Windowing and I/O belong to the caller
(orchestrator/src/orchestrator/digest.py::chain_telemetry_in_window).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

_DEEP_CHAIN_ITEMS = 2


@dataclass(frozen=True)
class ChainBucket:
    """Verifies that exercised one chain length, and how many of them passed."""

    passed: int = 0
    total: int = 0

    @property
    def rate(self) -> float | None:
        return self.passed / self.total if self.total else None


@dataclass(frozen=True)
class ChainTelemetry:
    """Chain telemetry for one window; ``ChainTelemetry()`` is the no-data value."""

    chain_items_histogram: Mapping[int, ChainBucket] = field(default_factory=dict)
    deep_verifies: int = 0
    deep_failures: int = 0
    deep_fail_rate: float | None = None
    items_landed_via_chain: int = 0
    items_landed_per_deep_verify: float | None = None
    excluded_legacy_depth: int = 0


def _usable_positive_int(value: object) -> int | None:
    return value if type(value) is int and value >= 1 else None


def _usable_chain_items(data: Mapping[str, Any]) -> int | None:
    """``chain_items`` when it honours its emitter contract, else None.

    The single producer is orchestrator/src/orchestrator/verify_runner.py::VerifyRunner.dispatch.
    """
    return _usable_positive_int(data.get('chain_items'))


def _is_legacy_deep_label(data: Mapping[str, Any]) -> bool:
    """True for a row whose only depth signal is a legacy ``depth >= 2`` label.

    Both the native-int and the str spelling count, because
    orchestrator/src/orchestrator/merge_lane/worker.py::_emit_speculative
    str-coerces payload values.
    """
    depth = data.get('depth')
    if _usable_chain_items(data) is not None or not isinstance(depth, (int, str)):
        return False
    try:
        return int(depth) >= _DEEP_CHAIN_ITEMS
    except ValueError:
        return False


def summarize_chain_telemetry(
    merge_verify_data: Iterable[Mapping[str, Any]],
    merge_finalized_data: Iterable[Mapping[str, Any]],
) -> ChainTelemetry:
    """Aggregate chain telemetry from ``merge_verify`` and ``merge_finalized`` payloads.

    The rules, stated once for every chain-telemetry reader in the repo:

    - Trust guard: a ``chain_items`` or ``landed_via_chain`` value counts only
      when it is a native int >= 1.  None, str, bool, float and values < 1
      are skipped, never coerced.
    - Deep threshold: a verify is deep when ``chain_items >= 2``; that is the
      denominator of both rates.  A missing ``passed`` counts as a failure.
    - No signal is None: with zero deep verifies both rates are None, never
      0.0, so a window that exercised no chains cannot read as perfect health.
    - Legacy label: a row with no usable ``chain_items`` and a ``depth >= 2``
      (int or str) is counted in ``excluded_legacy_depth`` and bucketed
      nowhere.  A row with both keys is classified by ``chain_items`` alone.

    ``landed_via_chain`` is summed as emitted; its 1-per-landed-item unit is
    pinned by orchestrator/src/orchestrator/merge_lane/types.py::MergeOutcome.

    Known divergences of other readers, deliberately not reconciled here:
    scripts/merge-deep-canary-predicate.sh admits bool ``True`` as
    ``landed_via_chain == 1`` (``isinstance``), and
    scripts/merge_lane_throughput.py::compute_chains coerces floats and sums
    ``landed_via_chain`` only on ``state == 'done'`` rows.
    scripts/tests/test_chain_telemetry_agreement.py is the mechanical lock
    that keeps every reader agreeing on realistic payloads.
    """
    counts: dict[int, list[int]] = {}
    excluded_legacy_depth = 0
    for data in merge_verify_data:
        excluded_legacy_depth += _is_legacy_deep_label(data)
        chain_items = _usable_chain_items(data)
        if chain_items is None:
            continue
        bucket = counts.setdefault(chain_items, [0, 0])
        bucket[0] += bool(data.get('passed'))
        bucket[1] += 1

    items_landed_via_chain = sum(
        _usable_positive_int(data.get('landed_via_chain')) or 0
        for data in merge_finalized_data
    )
    histogram = {
        chain_items: ChainBucket(passed=passed, total=total)
        for chain_items, (passed, total) in sorted(counts.items())
    }
    deep_buckets = [
        bucket for chain_items, bucket in histogram.items()
        if chain_items >= _DEEP_CHAIN_ITEMS
    ]
    deep_verifies = sum(bucket.total for bucket in deep_buckets)
    deep_failures = sum(bucket.total - bucket.passed for bucket in deep_buckets)
    return ChainTelemetry(
        chain_items_histogram=histogram,
        deep_verifies=deep_verifies,
        deep_failures=deep_failures,
        deep_fail_rate=deep_failures / deep_verifies if deep_verifies else None,
        items_landed_via_chain=items_landed_via_chain,
        items_landed_per_deep_verify=(
            items_landed_via_chain / deep_verifies if deep_verifies else None
        ),
        excluded_legacy_depth=excluded_legacy_depth,
    )
