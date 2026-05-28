"""Per-tick orchestration for the load sampler.

``run_tick`` is the pure core — it takes an already-opened ``LoadSampleStore``
and pre-collected metric dicts.  All I/O is done before calling it, so it is
fully unit-testable.

Tick responsibilities
---------------------
1. Write 6 PSI metrics with NULL window columns (kernel-windowed, no re-window).
2. For each of 3 non-PSI metrics: compute the trailing window from the DB, then
   write the row with window_mean + window_max populated.
3. Call cleanup_old to enforce the 24-hour retention policy.

Note: maybe_vacuum is NOT called here — it is called from ``__main__.main()``
after run_tick completes, to keep concerns separated.
"""

from __future__ import annotations

from sampler.store import LoadSampleStore

__all__ = ['run_tick']

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


def run_tick(
    store: LoadSampleStore,
    now: int,
    *,
    psi: dict[str, float],
    process_metrics: dict[str, float],
) -> None:
    """Write one tick's worth of samples to the store.

    Args:
        store:           Open LoadSampleStore instance.
        now:             Unix timestamp for this tick (integer seconds).
        psi:             Dict of 6 PSI avg10 values keyed as psi_*_avg10.
        process_metrics: Dict of 3 non-PSI values (occt_queue_depth,
                         verify_concurrency, verify_rss_total_bytes).
    """
    # Guard against unexpected/misspelled metric keys that would silently
    # persist without matching any consumer.  Subset checks (<=) rather than
    # equality allow partial dicts when a collection group degrades to {} on
    # error (see __main__.py degrade-and-continue handling).
    assert set(psi) <= _PSI_METRICS, f'unexpected PSI keys: {set(psi) - _PSI_METRICS}'
    assert set(process_metrics) <= _PROCESS_METRICS, (
        f'unexpected process metric keys: {set(process_metrics) - _PROCESS_METRICS}'
    )

    # 1. Write PSI rows (NULL windows — PSI is already kernel-windowed)
    for metric, value in psi.items():
        store.insert_sample(now, metric, value, window_mean=None, window_max=None)

    # 2. Write non-PSI rows with DB-backed trailing windows
    for metric, value in process_metrics.items():
        window_mean, window_max = store.trailing_window(metric, value)
        store.insert_sample(now, metric, value, window_mean=window_mean, window_max=window_max)

    # 3. Enforce 24-hour retention (delete-by-age, cheap + index-backed)
    store.cleanup_old(now)
