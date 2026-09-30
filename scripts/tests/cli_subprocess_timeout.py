"""Wall-clock budget for test suites that spawn a script's CLI as a subprocess.

A 10s budget flaked: concurrent orchestrator agents push interpreter startup
plus imports past it while the CLI under test behaves correctly (tasks 4217,
4515, 5085). 60s gives real headroom without slowing an idle-machine run.

Each consumer names its own override variable. Unset or blank means "not
overridden", so a CI template that always exports the variable cannot fail
collection of a file's pure-unit tests. A present but malformed value
(non-numeric, non-positive or non-finite) fails loudly, naming the variable,
rather than silently falling back and masking a typo.

A plain module rather than a fixture: consumers bind the result as a
def-time default argument, which resolves at import, before fixtures run.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping

DEFAULT_CLI_TIMEOUT_SECS = 60.0


def cli_timeout_from_env(env_var: str, environ: Mapping[str, str] = os.environ) -> float:
    """Return the budget in seconds from *env_var*, else the 60s default."""
    raw = environ.get(env_var, "").strip()
    if not raw:
        return DEFAULT_CLI_TIMEOUT_SECS
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        raise ValueError(
            f"{env_var} must be a positive, finite number of seconds; got {raw!r}"
        )
    return value
