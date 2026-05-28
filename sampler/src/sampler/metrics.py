"""Load-sampler metrics collection — pure parsers + thin I/O shells.

Purpose
-------
Collect 9 host-load metrics every tick and return them as a flat dict of
``{metric_name: float}``.  The 9 metrics are:

PSI metrics (6) — kernel-windowed, no re-windowing, NULL DB windows:
    psi_{cpu,mem,io}_{some,full}_avg10

Process metrics (3) — carry trailing window_mean/window_max in the DB:
    occt_queue_depth, verify_concurrency, verify_rss_total_bytes

Design decisions
----------------
* Pure parsers (parse_pressure_file, count_occt_queue_depth, count_verify_concurrency,
  sum_verify_rss) accept injected data so they are fully deterministic / unit-testable.
* Thin I/O shells (collect_psi default read, collect_process_metrics psutil calls)
  are validated only by the live integration signal (systemctl + sqlite query).
* PSI is kernel-windowed — we parse avg10 only, never re-window.

FD-9 heuristic (PRD open question Q3)
--------------------------------------
``occt_queue_depth`` counts bash processes matching ``cargo-test-occt-gated``
whose ``/proc/<pid>/fd/9`` symlink does NOT exist.  File-descriptor 9 is the
jobserver read-end of the semaphore pipe opened by the OCCT gate script; its
absence signals the process is still waiting to acquire the semaphore.
This is an implementation-detail heuristic that may break if the gate script is
refactored to use a different FD number.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

__all__ = [
    'parse_pressure_file',
    'collect_psi',
    'count_occt_queue_depth',
    'count_verify_concurrency',
    'sum_verify_rss',
    'collect_process_metrics',
]

# ---------------------------------------------------------------------------
# PSI helpers
# ---------------------------------------------------------------------------

_AVG10_RE = re.compile(r'avg10=([0-9]+(?:\.[0-9]+)?)')


def parse_pressure_file(text: str) -> dict[str, float]:
    """Parse a /proc/pressure/<name> text and return {some_avg10, full_avg10}.

    If the ``full`` line is absent (e.g. CPU on some kernels), ``full_avg10``
    defaults to 0.0.
    """
    result: dict[str, float] = {'some_avg10': 0.0, 'full_avg10': 0.0}
    for line in text.splitlines():
        line = line.strip()
        m = _AVG10_RE.search(line)
        if m is None:
            continue
        value = float(m.group(1))
        if line.startswith('some'):
            result['some_avg10'] = value
        elif line.startswith('full'):
            result['full_avg10'] = value
    return result


def _default_psi_read(name: str) -> str:
    """Read /proc/pressure/<name> from the live kernel."""
    return Path(f'/proc/pressure/{name}').read_text()


def collect_psi(
    *,
    read: Callable[[str], str] = _default_psi_read,
) -> dict[str, float]:
    """Return the 6 PSI avg10 values keyed as psi_{cpu,mem,io}_{some,full}_avg10.

    Args:
        read: Callable mapping pressure-file name ('cpu'|'memory'|'io') to text.
              Defaults to reading /proc/pressure/* from the live kernel.
    """
    sources = [
        ('cpu', 'cpu'),
        ('memory', 'mem'),
        ('io', 'io'),
    ]
    out: dict[str, float] = {}
    for file_name, key_prefix in sources:
        text = read(file_name)
        parsed = parse_pressure_file(text)
        out[f'psi_{key_prefix}_some_avg10'] = parsed['some_avg10']
        out[f'psi_{key_prefix}_full_avg10'] = parsed['full_avg10']
    return out


# ---------------------------------------------------------------------------
# Process-metric counters (pure, take injected process iterables)
# ---------------------------------------------------------------------------


def _is_occt_gated(proc: Any) -> bool:
    """Return True if proc is a bash process running cargo-test-occt-gated."""
    try:
        cmdline = proc.cmdline()
    except Exception:
        return False
    return any('cargo-test-occt-gated' in part for part in cmdline)


def count_occt_queue_depth(
    procs: Iterable[Any],
    fd9_exists: Callable[[int], bool],
) -> int:
    """Count ``cargo-test-occt-gated`` bash processes waiting on the semaphore.

    A process is *waiting* when its /proc/<pid>/fd/9 symlink does not exist
    (see FD-9 heuristic in the module docstring).
    """
    count = 0
    for proc in procs:
        if _is_occt_gated(proc) and not fd9_exists(proc.pid):
            count += 1
    return count


def _is_verify_sh(proc: Any) -> bool:
    """Return True if proc's argv[0] basename is 'verify.sh'."""
    try:
        cmdline = proc.cmdline()
    except Exception:
        return False
    if not cmdline:
        return False
    return os.path.basename(cmdline[0]) == 'verify.sh'


def count_verify_concurrency(procs: Iterable[Any]) -> int:
    """Count processes whose argv[0] basename is 'verify.sh'."""
    return sum(1 for p in procs if _is_verify_sh(p))


def sum_verify_rss(procs: Iterable[Any]) -> int:
    """Sum RSS (bytes) over all verify.sh processes and their child trees.

    Shared PIDs are de-duplicated so each process's RSS is counted only once.
    """
    seen_pids: set[int] = set()
    total = 0

    def _add(proc: Any) -> None:
        if proc.pid in seen_pids:
            return
        seen_pids.add(proc.pid)
        try:
            total_ref[0] += proc.memory_info().rss
        except Exception:
            pass
        try:
            for child in proc.children(recursive=True):
                _add(child)
        except Exception:
            pass

    total_ref = [0]
    for proc in procs:
        if _is_verify_sh(proc):
            _add(proc)
    return total_ref[0]


# ---------------------------------------------------------------------------
# Thin I/O shell — not unit-tested, validated by live integration signal
# ---------------------------------------------------------------------------


def _default_fd9_exists(pid: int) -> bool:
    return os.path.lexists(f'/proc/{pid}/fd/9')


def collect_process_metrics(
    *,
    proc_iter: Any = None,
    fd9_exists: Callable[[int], bool] = _default_fd9_exists,
) -> dict[str, float]:
    """Materialise the live process list and return the 3 non-PSI metrics.

    Returns:
        {'occt_queue_depth': float, 'verify_concurrency': float,
         'verify_rss_total_bytes': float}

    Per-process psutil.NoSuchProcess / AccessDenied are silently skipped.
    """
    import psutil

    if proc_iter is None:
        proc_iter = psutil.process_iter

    try:
        procs = list(proc_iter(['pid', 'name', 'cmdline', 'memory_info']))
    except Exception:
        procs = []

    # Guard per-process access errors
    safe_procs: list[Any] = []
    for p in procs:
        try:
            p.cmdline()  # probe; raises NoSuchProcess/AccessDenied if gone
            safe_procs.append(p)
        except Exception:
            pass

    return {
        'occt_queue_depth': float(count_occt_queue_depth(safe_procs, fd9_exists)),
        'verify_concurrency': float(count_verify_concurrency(safe_procs)),
        'verify_rss_total_bytes': float(sum_verify_rss(safe_procs)),
    }
