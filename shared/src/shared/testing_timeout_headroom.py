"""The headroom a per-test timeout ceiling keeps over its family's measured-under-load worst case, held where every package's tests can import it.

A pure-stdlib leaf like every ``shared.testing_*`` module, registered in
``shared/tests/test_pure_stdlib_leaves.py``.
"""

from __future__ import annotations

__all__ = ['UNDER_LOAD_HEADROOM_FACTOR']

# 8x rather than 2x: the breaches per-test ceilings exist to prevent (xdist
# worker deaths, in the thread era before task 6354) were observed at loadavg
# 250-423, one inflation step past the load at which any measurement has been
# taken.  Rationale and corpus:
# plans/pytest-per-test-timeout-measurement-2026-09-17.md.
UNDER_LOAD_HEADROOM_FACTOR = 8
