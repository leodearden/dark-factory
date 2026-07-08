"""PSI (pressure stall information) reader — re-homed pure parser + reader.

PRD ``docs/prds/dispatch-admission-load-cap.md`` task DA1 (DA-D9: reuse, do
not reimplement). ``parse_pressure_file`` and the ``/proc/pressure/<name>``
reader were originally written and tested in
``sampler/src/sampler/metrics.py``; they are re-homed here VERBATIM
(behavior-identical) so both the load-sampler and the orchestrator import the
SAME parser instead of each maintaining their own copy that could drift and
re-derive the "CPU has no ``full`` line on some kernels" asymmetry bug.
``sampler.metrics`` re-exports ``parse_pressure_file`` from this module (see
that module's ``TestParserRehomedToShared`` identity guard).

This module is a direct-import submodule — like ``shared.deploy_state`` — and
is deliberately NOT re-exported from ``shared/__init__.py``:
``shared/tests/test_public_api.py::TestInitAllCompleteness`` pins
``shared.__all__`` to a hardcoded module union, so consumers import via
``from shared.psi import ...``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    'parse_pressure_file',
    'read_pressure',
]

_AVG10_RE = re.compile(r'avg10=([0-9]+(?:\.[0-9]+)?)')


def parse_pressure_file(text: str) -> dict[str, float] | None:
    """Parse a /proc/pressure/<name> text and return {some_avg10, full_avg10}.

    If the ``full`` line is absent (e.g. CPU on some kernels), ``full_avg10``
    defaults to 0.0.

    Returns:
        A dict with ``some_avg10`` and ``full_avg10`` on success, or ``None``
        if no some/full avg10 value could be extracted (empty text, garbage
        content, or truncated read).  A *partial* miss where ``some`` is
        present but ``full`` is absent still returns a dict — that is a
        legitimate kernel behaviour, not a fault.  Only a *total* miss (neither
        key extracted) returns ``None`` so callers can distinguish a
        read/parse fault from genuine zero pressure.

    Note on asymmetry:
        The kernel **always** emits a ``some`` line for all PSI resources; the
        ``full`` line is the one that may be omitted (e.g. CPU on some kernels).
        Therefore the partial-miss case is always *some-present / full-absent*,
        and a *full-present / some-absent* result is not a legitimate kernel
        state.  If that impossible case were ever produced (e.g. by a
        kernel change or filesystem stub), the current logic would fabricate
        ``some_avg10=0.0``. This is documented here as a known asymmetry; the
        sentinel (``None``) is not triggered in that case because ``found``
        becomes ``True`` via the ``full`` branch. Should the kernel contract
        shift, a separate ``found_some`` guard should be added mirroring the
        ``full`` handling.
    """
    result: dict[str, float] = {'some_avg10': 0.0, 'full_avg10': 0.0}
    found = False
    for line in text.splitlines():
        line = line.strip()
        m = _AVG10_RE.search(line)
        if m is None:
            continue
        value = float(m.group(1))
        if line.startswith('some'):
            result['some_avg10'] = value
            found = True
        elif line.startswith('full'):
            result['full_avg10'] = value
            found = True
    if not found:
        return None
    return result


def read_pressure(name: str) -> str:
    """Read /proc/pressure/<name> from the live kernel."""
    return Path(f'/proc/pressure/{name}').read_text()
