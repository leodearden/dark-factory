"""Per-tick orchestration for the load sampler.

``run_tick`` is the pure core — it takes an already-opened ``LoadSampleStore``
and pre-collected metric dicts.  All I/O is done before calling it, so it is
fully unit-testable.

Tick responsibilities
---------------------
1. Hand the store one whole tick: the 6 PSI metrics to be stored with NULL
   window columns (kernel-windowed, no re-window), and every non-PSI metric to
   be stored with its DB-backed trailing window. ``LoadSampleStore.write_tick``
   does it on one connection in one transaction — see its docstring for the
   measurement, and for why a half-written tick must never reach the corpus.
2. Call cleanup_old to enforce the retention policy.

Key guard
---------
The per-tick key count is NOT fixed. The load group emits one
``own_cpu_some10`` and one ``own_read_ok`` key per cgroup leaf discovered at
collection time, so its keys cannot be enumerated in advance.
``unexpected_keys`` therefore admits a host-level key by its exact metric
name with no leaf, and a per-leaf key by a registered per-leaf metric with a
non-empty leaf. The PSI and process groups register no per-leaf metric, which
leaves their strictness exactly where it is.

Note: maybe_vacuum is NOT called here — it is called from ``__main__.main()``
after run_tick completes, to keep concerns separated.
"""

from __future__ import annotations

from collections.abc import Collection

from sampler.metrics import SampleKey, Samples
from sampler.store import LoadSampleStore

__all__ = ['run_tick', 'unexpected_keys']


def unexpected_keys(
    keys: Collection[SampleKey],
    *,
    exact: frozenset[str],
    per_leaf: frozenset[str],
) -> set[SampleKey]:
    """Return the keys that are neither an *exact* host-level key nor a *per_leaf* one.

    A host-level key names an *exact* metric and carries no leaf. A per-leaf
    key names a *per_leaf* metric and carries a NON-EMPTY leaf: without one it
    would be per-leaf evidence about no cgroup, written by a collector that
    failed to attach the leaf. The leaf itself is opaque data.

    An EMPTY *per_leaf* set degenerates to plain exact-set membership. That is
    how the PSI and process groups keep exactly the strictness they have
    today: neither has a per-leaf metric, so a typo there must still fail.
    """
    return {
        key
        for key in keys
        if not (
            (key.metric in exact and key.leaf is None)
            or (key.metric in per_leaf and bool(key.leaf))
        )
    }

# PSI metric names that carry NULL windows (kernel-windowed already)
_PSI_METRICS = frozenset([
    'psi_cpu_some_avg10',
    'psi_cpu_full_avg10',
    'psi_mem_some_avg10',
    'psi_mem_full_avg10',
    'psi_io_some_avg10',
    'psi_io_full_avg10',
])

# Non-PSI metric names that carry trailing window_mean/window_max
_PROCESS_METRICS = frozenset([
    'occt_queue_depth',
    'verify_concurrency',
    'verify_rss_total_bytes',
])

_LOAD_METRICS = frozenset([
    'runqueue_ratio',
    'runqueue_read_ok',
])

# One key per DISCOVERED cgroup leaf, so the leaves are not knowable here — see
# sampler.metrics.discover_pressure_cgroups.
_LOAD_PER_LEAF = frozenset([
    'own_cpu_some10',
    'own_read_ok',
])

# The PSI and process groups have no per-leaf metric, so the guard stays
# exactly as strict as it is today.
_NO_PER_LEAF: frozenset[str] = frozenset()


def run_tick(
    store: LoadSampleStore,
    now: int,
    *,
    psi: Samples,
    process_metrics: Samples,
    load_metrics: Samples,
) -> None:
    """Write one tick's worth of samples to the store.

    Args:
        store:           Open LoadSampleStore instance.
        now:             Unix timestamp for this tick (integer seconds).
        psi:             Dict of 6 PSI avg10 values keyed as psi_*_avg10.
        process_metrics: Dict of 3 non-PSI values (occt_queue_depth,
                         verify_concurrency, verify_rss_total_bytes).
        load_metrics:    Dict of the runqueue and per-cgroup own-pressure
                         values (runqueue_ratio, runqueue_read_ok, and one
                         own_cpu_some10 + own_read_ok pair keyed by each
                         cgroup leaf discovered this tick).

    Each group is passed separately, and a degraded one arrives as ``{}``.
    They are kept apart because they read unrelated kernel surfaces and so
    fail independently — folding the load group into process_metrics would
    put two failure domains behind one except and silently widen what a
    single failure erases.
    """
    # Guard against unexpected/misspelled metric keys that would silently
    # persist without matching any consumer. A PARTIAL dict is always legal —
    # a collection group that degrades hands us {} (see __main__.py's
    # degrade-and-continue handling), so only unrecognised names are an error.
    unexpected_psi = unexpected_keys(psi, exact=_PSI_METRICS, per_leaf=_NO_PER_LEAF)
    assert not unexpected_psi, f'unexpected PSI keys: {unexpected_psi}'
    unexpected_process = unexpected_keys(
        process_metrics, exact=_PROCESS_METRICS, per_leaf=_NO_PER_LEAF
    )
    assert not unexpected_process, f'unexpected process metric keys: {unexpected_process}'
    unexpected_load = unexpected_keys(
        load_metrics, exact=_LOAD_METRICS, per_leaf=_LOAD_PER_LEAF
    )
    assert not unexpected_load, f'unexpected load metric keys: {unexpected_load}'

    # 1. Write the whole tick in one transaction. PSI rows carry NULL windows
    #    (already kernel-windowed); the process and load groups share the
    #    sampler-windowed path, because their write MODE is the same and they
    #    differ along exactly one axis — the per-leaf set their keys are
    #    validated against, above.
    #
    #    The load group's window columns are RECORDED AHEAD OF A CONSUMER, and
    #    that is stated here so a reader does not have to grep for one: the
    #    dashboard reads window_mean/window_max only for the 9 names in
    #    dashboard/src/dashboard/data/load.py::KNOWN_METRICS, and
    #    scripts/load-threshold-calibration.py::_fetch reads (leaf, ts, value)
    #    by metric and computes its own windows. So the ~2 + 2-per-cgroup-leaf
    #    window reads this buys per tick (16 on this host) are write-only
    #    today, and NO consumer is scheduled — say so rather than implying one.
    #    Kept windowed regardless, for two reasons that survive having no
    #    reader. Each read seeks its own (metric, leaf), as
    #    sampler/tests/test_load_store.py::TestTrailingWindow::test_a_sparse_keys_window_read_does_not_walk_a_busy_neighbours_history
    #    pins, so the saving on offer is ~nothing; and
    #    routing one GROUP to the unwindowed path would give the store a
    #    second axis of variability (window mode AND provenance) where it
    #    deliberately has one. Nothing is lost either way if that is revisited:
    #    both columns are recomputable from the raw values, which are the
    #    thing this corpus exists to keep.
    store.write_tick(
        now,
        unwindowed=psi,
        windowed={**process_metrics, **load_metrics},
    )

    # 2. Enforce retention (delete-by-age, interval-gated inside the store)
    store.cleanup_old(now)
