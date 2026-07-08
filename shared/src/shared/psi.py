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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    'parse_pressure_file',
    'read_pressure',
    'PsiSample',
    'read_psi_sample',
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


@dataclass(frozen=True)
class PsiSample:
    """Immutable PSI snapshot consumed by the orchestrator's dispatch-admission gate.

    ``read_ok=False`` is the DA-D6 fail-open sentinel (see ``read_psi_sample``):
    an unreadable or unparseable /proc/pressure/* file degrades the whole
    sample rather than partially gating on incomplete data.
    """

    cpu_some10: float
    mem_some10: float
    mem_full10: float
    io_some10: float
    read_ok: bool

    def saturated(self, cfg) -> bool:
        """Return True if any PSI metric is at/over its configured avg10 threshold.

        ``cfg`` is duck-typed (only ``cpu_some_avg10``, ``mem_some_avg10``,
        ``mem_full_avg10``, and ``io_some_avg10`` are read) so this has no
        dependency on DA2's ``PsiAdmissionConfig`` submodel. Always False when
        ``read_ok`` is False (fail-open — a degraded sample must never trip
        the gate, regardless of the threshold values).
        """
        return self.read_ok and (
            self.cpu_some10 >= cfg.cpu_some_avg10
            or self.mem_some10 >= cfg.mem_some_avg10
            or self.mem_full10 >= cfg.mem_full_avg10
            or self.io_some10 >= cfg.io_some_avg10
        )


_FAIL_OPEN = PsiSample(
    cpu_some10=0.0,
    mem_some10=0.0,
    mem_full10=0.0,
    io_some10=0.0,
    read_ok=False,
)


def read_psi_sample(*, read: Callable[[str], str] = read_pressure) -> PsiSample:
    """Read and parse /proc/pressure/{cpu,memory,io} into a PsiSample.

    Maps cpu.some -> cpu_some10, mem.some -> mem_some10, mem.full ->
    mem_full10, and io.some -> io_some10.

    Fail-open (DA-D6): if any source is unreadable (``read`` raises ANY
    exception -- e.g. ``OSError``, or a ``UnicodeDecodeError`` from non-UTF-8
    content) or unparseable (``parse_pressure_file`` returns None), the WHOLE
    sample degrades to the fail-open sentinel (all metrics 0.0,
    read_ok=False) rather than gating on partial data — this must never
    wedge dispatch. The except is deliberately broad rather than scoped to
    ``OSError`` alone: a caller-injected ``read`` is untrusted, and "never
    wedge dispatch" is an absolute guarantee, not one scoped to a single
    exception type. Loud, rate-limited logging on this condition is the
    caller's (DA3) responsibility; this function stays side-effect-light (at
    most a single debug line) to avoid per-tick log spam.
    """
    try:
        cpu = parse_pressure_file(read('cpu'))
        mem = parse_pressure_file(read('memory'))
        io = parse_pressure_file(read('io'))
    except Exception:
        logger.debug('PSI read failed; failing open', exc_info=True)
        return _FAIL_OPEN

    if cpu is None or mem is None or io is None:
        logger.debug('PSI parse failed (unparseable source); failing open')
        return _FAIL_OPEN

    return PsiSample(
        cpu_some10=cpu['some_avg10'],
        mem_some10=mem['some_avg10'],
        mem_full10=mem['full_avg10'],
        io_some10=io['some_avg10'],
        read_ok=True,
    )
