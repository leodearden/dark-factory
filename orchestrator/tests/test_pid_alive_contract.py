"""Cross-package contract test: orchestrator.harness._pid_alive vs
fused_memory.services.orchestrator_detector._pid_alive.

The two copies are intentionally duplicated to avoid a cross-package import
edge (see orchestrator/src/orchestrator/harness.py:50-61).  This contract
test imports both under distinct aliases and asserts they return identical
values for every input/os.kill outcome combination — catching any future
behavioural drift without requiring extraction to a shared module.

fused_memory.services.orchestrator_detector is loaded via importlib.util so
that the heavy fused_memory.services.__init__ (which needs graphiti_core,
qdrant, etc.) is never executed — the contract only needs the ~15-line
_pid_alive function.
"""

import importlib.util
import os
import unittest.mock
from pathlib import Path

import pytest

from orchestrator.harness import _pid_alive as harness_pid_alive

# ---------------------------------------------------------------------------
# Load orchestrator_detector directly from its file to avoid importing the
# full fused_memory.services package (which drags in graphiti_core etc.)
# ---------------------------------------------------------------------------
_FM_DETECTOR_PATH = (
    Path(__file__).parent.parent.parent
    / 'fused-memory' / 'src'
    / 'fused_memory' / 'services' / 'orchestrator_detector.py'
)
assert _FM_DETECTOR_PATH.exists(), (
    f'orchestrator_detector.py not found at expected path: {_FM_DETECTOR_PATH}\n'
    f'Check that fused-memory/ is a sibling of orchestrator/ in the monorepo layout.'
)
_spec = importlib.util.spec_from_file_location('_fm_orchestrator_detector', _FM_DETECTOR_PATH)
assert _spec is not None and _spec.loader is not None, (
    f'Cannot create module spec for orchestrator_detector at {_FM_DETECTOR_PATH}'
)
_detector_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_detector_mod)  # type: ignore[union-attr]
detector_pid_alive = _detector_mod._pid_alive


# ---------------------------------------------------------------------------
# (1) Real os.kill — both copies must agree for these PID values
# ---------------------------------------------------------------------------

# Known absolute answers for deterministic PID inputs — evaluated at
# collection time so os.getpid() is stable for the full test session.
_KNOWN_PID_ANSWERS: dict[int, bool] = {
    0: False,
    -1: False,
    os.getpid(): True,
    2 ** 31 - 1: False,  # ~2.1 B — not a valid live PID on any standard Linux
}


@pytest.mark.parametrize('pid', list(_KNOWN_PID_ANSWERS))
def test_matching_return_for_real_pids(pid: int):
    """Both _pid_alive copies return the same bool for the given pid,
    and the result matches the known-correct answer for each deterministic input."""
    harness_result = harness_pid_alive(pid)
    detector_result = detector_pid_alive(pid)
    assert harness_result == detector_result, (
        f'_pid_alive({pid!r}) diverged: '
        f'harness={harness_result!r}, detector={detector_result!r}'
    )
    expected = _KNOWN_PID_ANSWERS[pid]
    assert harness_result is expected, (
        f'harness _pid_alive({pid!r}): expected {expected!r}, got {harness_result!r}'
    )
    assert detector_result is expected, (
        f'detector _pid_alive({pid!r}): expected {expected!r}, got {detector_result!r}'
    )


# ---------------------------------------------------------------------------
# (2) Mocked os.kill — each OSError branch must produce the same return value
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('exc_cls', [
    ProcessLookupError,
    PermissionError,
    OSError,
])
def test_matching_return_for_mocked_os_kill_branches(exc_cls):
    """Both copies agree when os.kill raises each OSError subclass."""
    with unittest.mock.patch('os.kill', side_effect=exc_cls('x')):
        harness_result = harness_pid_alive(12345)
        detector_result = detector_pid_alive(12345)
    assert harness_result == detector_result, (
        f'_pid_alive(12345) diverged when os.kill raises {exc_cls.__name__}: '
        f'harness={harness_result!r}, detector={detector_result!r}'
    )
