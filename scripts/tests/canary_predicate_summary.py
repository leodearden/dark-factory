"""Reads the summary line scripts/merge-deep-canary-predicate.sh prints on stdout.

The line looks like ``merge-deep-canary window_days=7 n_deep_verifies=2:
deep_fail_rate=0.5(FAIL <= 0.35) items_per_verify=2.0(ok >= 1.3)
verify_p90_secs=n/a(FAIL <= 5400) -> HOLD``.  This is the only parser of that
line, shared by every test that drives the shipped predicate.
"""
from __future__ import annotations

import re


def n_deep_verifies(stdout: str) -> int:
    match = re.search(r'n_deep_verifies=(\d+)', stdout)
    assert match, stdout
    return int(match.group(1))


def metric(stdout: str, name: str) -> float | None:
    """The leg's value, or None when the predicate reported ``n/a``."""
    match = re.search(rf'{name}=([^(]+)\(', stdout)
    assert match, stdout
    raw = match.group(1)
    return None if raw == 'n/a' else float(raw)


def verdict(stdout: str, name: str) -> str:
    """The leg's ``ok`` or ``FAIL``."""
    match = re.search(rf'{name}=[^(]+\((\w+) ', stdout)
    assert match, stdout
    return match.group(1)
