"""Regression test for cross-test _PROBE_CACHE leakage (task 1662).

verify._PROBE_CACHE is a process-global TTL dict keyed by
(main_sha, category, normalized_cause_hint).  Without an autouse fixture
clearing it between tests, a test that seeds a True entry for a key can make
a later test on the same xdist worker short-circuit the real probe logic and
return (True, main_sha) instead of running to completion — producing an
order-dependent flake.

These two tests are pinned to one xdist worker via xdist_group so that
--dist loadgroup keeps them in definition order on the same process (same
_PROBE_CACHE global).  Without that guarantee the sentinel would live in
worker-A's cache while test_b runs in worker-B's fresh cache — masking the
leak and giving a false GREEN.
"""

import pytest

# Sentinel key shape matches the documented _PROBE_CACHE key:
# (main_sha: str, category: str, normalized_cause_hint: str)
SENTINEL_KEY: tuple[str, str, str] = (
    "probe-cache-isolation-sentinel",
    "compile_error",
    "error ts2769: foo.tsx:12",
)


@pytest.mark.xdist_group("probe_cache_isolation")
class TestProbeCacheIsolation:
    """Two in-order tests that verify _PROBE_CACHE is cleared between tests.

    test_a seeds a sentinel entry (the polluter).
    test_b asserts that entry is absent (the victim).

    RED before conftest adds _clear_probe_cache: test_b finds the sentinel.
    GREEN after conftest adds _clear_probe_cache: test_b finds an empty cache.

    *** Isolation note ***
    This regression test is only meaningful when BOTH of the following hold:
      1. The full suite (or at least this file) runs under ``--dist loadgroup``
         so xdist_group keeps both methods on the SAME worker process (same
         _PROBE_CACHE global).
      2. test_a executes before test_b on that worker (definition order).

    If you select ONLY test_b (e.g. ``pytest -k test_b_probe_cache``), nothing
    seeds the sentinel, so test_b trivially passes even with the fixture
    removed — that is a false GREEN, not a fix.  Run the whole class (or the
    whole file) to get meaningful coverage.
    """

    def test_a_seeds_probe_cache(self) -> None:
        """Polluter: write a sentinel entry into _PROBE_CACHE."""
        from orchestrator import verify

        # Value shape: (probe_time: float, is_preexisting: bool)
        verify._PROBE_CACHE[SENTINEL_KEY] = (1.0, True)
        assert SENTINEL_KEY in verify._PROBE_CACHE  # sanity — entry was stored

    def test_b_probe_cache_cleared_between_tests(self) -> None:
        """Victim: assert the sentinel from test_a is gone.

        Fails (RED) when conftest has no _clear_probe_cache autouse fixture.
        Passes (GREEN) once that fixture calls verify._PROBE_CACHE.clear()
        before each test.
        """
        from orchestrator import verify

        assert SENTINEL_KEY not in verify._PROBE_CACHE, (
            "verify._PROBE_CACHE was NOT cleared between tests — "
            "the sentinel seeded by test_a_seeds_probe_cache leaked into "
            "test_b.  Add an autouse _clear_probe_cache fixture to "
            "orchestrator/tests/conftest.py that calls "
            "verify._PROBE_CACHE.clear() before (and after) each test."
        )
