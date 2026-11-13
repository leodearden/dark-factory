"""Whether a test that needs the real landlock sandbox may skip on this host.

A host whose verify_env arms ``DF_REQUIRE_SANDBOX_TESTS=1``
(dark-factory-orchestrator.yaml, task 4635) is known to be landlock-capable, so
there a missing landlock is a host or probe regression and fails the run
instead of quietly removing the test. The same rule, with its rationale, is
orchestrator/tests/test_landlock.py::_skip_landlock.
"""

from __future__ import annotations

import os

import pytest

from orchestrator.agents.landlock import is_landlock_available

ARMING_VAR = 'DF_REQUIRE_SANDBOX_TESTS'
LANDLOCK_SKIP_REASON = 'landlock not supported on this kernel'


def skip_without_landlock() -> bool:
    if is_landlock_available():
        return False
    if os.environ.get(ARMING_VAR) == '1':
        pytest.fail(
            f'{ARMING_VAR}=1 but {LANDLOCK_SKIP_REASON}: refusing to silently skip a '
            'real-kernel sandbox test.',
            pytrace=False,
        )
    return True
