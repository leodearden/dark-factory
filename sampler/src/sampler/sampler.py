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

Metric-name guard
-----------------
The per-tick metric count is NOT fixed. The load group emits one
``own_cpu_some10:<leaf>`` and one ``own_read_ok:<leaf>`` row per cgroup leaf
discovered at collection time, so its names cannot be enumerated in advance
and no frozenset can admit them. ``unexpected_metric_names`` therefore
validates a name against an exact-name set OR a registered STEM with a
non-empty tail; the PSI and process groups register no stem, which leaves
their strictness exactly where it is.

Note: maybe_vacuum is NOT called here — it is called from ``__main__.main()``
after run_tick completes, to keep concerns separated.
"""

from __future__ import annotations

from collections.abc import Collection

from sampler.store import LoadSampleStore

__all__ = ['run_tick', 'unexpected_metric_names']


def unexpected_metric_names(
    names: Collection[str],
    *,
    exact: frozenset[str],
    stems: frozenset[str],
) -> set[str]:
    """Return the names matching neither an *exact* name nor a *stem* with a tail.

    A name matches a stem when it splits on its FIRST ':' into a registered
    stem and a NON-EMPTY tail. Both halves of that are load-bearing: a bare
    stem is not itself a metric (nothing emits one, and admitting it would let
    a collector that failed to append a leaf name write an anonymous row), and
    an empty tail names no cgroup, so it would be per-leaf evidence about
    nothing.

    An EMPTY *stems* set degenerates to plain exact-set membership. That is
    how the PSI and process groups keep exactly the strictness they have
    today: neither has a dynamic component, so a typo there must still fail.
    """
    return {
        name
        for name in names
        if name not in exact and not _matches_a_stem(name, stems)
    }


def _matches_a_stem(name: str, stems: frozenset[str]) -> bool:
    stem, separator, tail = name.partition(':')
    return bool(separator) and bool(tail) and stem in stems

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

# One row per DISCOVERED cgroup leaf, so the tail is not knowable here — see
# sampler.metrics.discover_pressure_cgroups.
_LOAD_STEMS = frozenset([
    'own_cpu_some10',
    'own_read_ok',
])

# The PSI and process groups have no dynamic component, so they register no
# stem and the guard stays exactly as strict as it is today.
_NO_STEMS: frozenset[str] = frozenset()


def run_tick(
    store: LoadSampleStore,
    now: int,
    *,
    psi: dict[str, float],
    process_metrics: dict[str, float],
    load_metrics: dict[str, float],
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
                         own_cpu_some10:<leaf> + own_read_ok:<leaf> pair per
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
    unexpected_psi = unexpected_metric_names(psi, exact=_PSI_METRICS, stems=_NO_STEMS)
    assert not unexpected_psi, f'unexpected PSI keys: {unexpected_psi}'
    unexpected_process = unexpected_metric_names(
        process_metrics, exact=_PROCESS_METRICS, stems=_NO_STEMS
    )
    assert not unexpected_process, f'unexpected process metric keys: {unexpected_process}'
    unexpected_load = unexpected_metric_names(
        load_metrics, exact=_LOAD_METRICS, stems=_LOAD_STEMS
    )
    assert not unexpected_load, f'unexpected load metric keys: {unexpected_load}'

    # 1. Write the whole tick in one transaction. PSI rows carry NULL windows
    #    (already kernel-windowed); the process and load groups share the
    #    sampler-windowed path, because their write MODE is the same and they
    #    differ along exactly one axis — the stem set their names are validated
    #    against, above.
    #
    #    The load group's window columns are RECORDED AHEAD OF A CONSUMER, and
    #    that is stated here so a reader does not have to grep for one: the
    #    dashboard reads window_mean/window_max only for the 9 names in
    #    dashboard/src/dashboard/data/load.py::KNOWN_METRICS, and
    #    scripts/load-threshold-calibration.py::_fetch selects (metric, ts,
    #    value) and computes its own windows. So the ~2 + 2-per-cgroup-leaf
    #    window reads this buys per tick (16 on this host) are write-only
    #    today, and NO consumer is scheduled — say so rather than implying one.
    #    Kept windowed regardless, for two reasons that survive having no
    #    reader. Each read is the indexed `SEARCH samples USING INDEX
    #    idx_samples_metric_ts (metric=?)` that LoadSampleStore.cleanup_old's
    #    docstring measured at ~0 ms, so the saving on offer is ~nothing; and
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
