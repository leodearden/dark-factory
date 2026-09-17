"""Load-sampler metrics collection — pure parsers + thin I/O shells.

Purpose
-------
Collect the host-load metrics of one tick and return them as a flat dict of
``{metric_name: float}``. The count is NOT fixed: it is 11 plus two per
discovered cgroup leaf (25 on this host today, where seven leaves are
present).

PSI metrics (6) — kernel-windowed, no re-windowing, NULL DB windows:
    psi_{cpu,mem,io}_{some,full}_avg10

Process metrics (3) — carry trailing window_mean/window_max in the DB:
    occt_queue_depth, verify_concurrency, verify_rss_total_bytes

Load metrics (2 + 2 per cgroup leaf) — also carry trailing windows
(PRD ``plans/load-throttle-harmonisation-prd.md`` §6.4):
    runqueue_ratio, runqueue_read_ok,
    own_cpu_some10:<leaf>, own_read_ok:<leaf>

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

import contextlib
import logging
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from shared.psi import (
    RunqueueReading,
    parse_pressure_file,
    read_pressure,
    read_runqueue_ratio,
    resolve_own_cgroup,
)

logger = logging.getLogger(__name__)

__all__ = [
    'parse_pressure_file',
    'collect_psi',
    'count_occt_queue_depth',
    'count_verify_concurrency',
    'sum_verify_rss',
    'collect_process_metrics',
    'discover_pressure_cgroups',
    'collect_load_metrics',
    'ARM_METRIC_STEMS',
]

ARM_METRIC_STEMS = {
    'mem_full_avg10': 'psi_mem_full_avg10',
    'mem_some_avg10': 'psi_mem_some_avg10',
    'io_some_avg10': 'psi_io_some_avg10',
    'cpu_some_avg10': 'psi_cpu_some_avg10',
    'runqueue_ratio': 'runqueue_ratio',
    'own_cpu_some_avg10': 'own_cpu_some10',
}
"""Gate ARM name -> the sampler metric name (or ':' stem) recording that arm.

PRD ``plans/load-throttle-harmonisation-prd.md`` §6.4, boundary row 11. The
gate decides on arms (``shared.psi._ARMS``, whose field names are also the
``psi_admission`` config leaves and the ``tripping_metric`` vocabulary); this
module records metrics; ε1/ε2 calibrate one against the other. Those are two
vocabularies for one set of signals, so the correspondence needs a single
home (heuristic 11) — this one — and a reconciler, which is
``TestArmMetricStemParity`` in sampler/tests/test_load_metrics.py. That test
enumerates ``_ARMS`` itself rather than a copy, asserts BOTH directions so
the mapping can neither miss an arm nor keep a dead key, and asserts each
stem is actually emitted.

NO PRODUCTION CODE READS THIS TABLE, and that is its point rather than an
oversight: it is a DECLARATION, and its readers are the reconcilers — the
parity test above, and the calibration script's own copy which that test
checks against this one. Stated here so a later agent weighing its deletion
does not have to infer the purpose from an empty grep.

The stems are NOT mechanically derivable from the arm names, which is exactly
why this is an explicit table and not a rule: the four host arms gain a
``psi_`` prefix, ``own_cpu_some_avg10`` records as ``own_cpu_some10``
(``avg10`` against ``10``), and ``runqueue_ratio`` maps to itself. Neither
the identity nor a systematic rewrite covers all three.

Two of the six values are ':' STEMS rather than whole metric names —
``own_cpu_some10`` is emitted as ``own_cpu_some10:<cgroup-leaf>``, one row
per discovered leaf (PRD open question 1, resolved as the stem spelling).
``scripts/load-threshold-calibration.py`` carries the same correspondence by
necessity — it runs under the system python3 and cannot import this module —
and a lockstep test reconciles the two.
"""

# The cgroup v2 unified-hierarchy mountpoint. A DEFAULT for the injected
# ``cgroup_root`` seam, in the same shape as ``_default_fd9_exists`` below —
# not a second source of truth: every caller that cares injects its own, and
# every test does. shared.psi holds the same literal privately for its own
# defaults and exposes no public constant to borrow (it is α's file, out of
# this task's scope), so the two defaults coexist rather than one importing a
# private from the other.
_CGROUP_ROOT = Path('/sys/fs/cgroup')

_USER_MANAGER_PREFIX = 'user@'
_DF_LEAF_GLOB = '*/df-*.slice'
_ORCHESTRATOR_LEAF_GLOB = '*/orchestrator-*.service'


def _anchor_segments(own_cgroup_path: str) -> list[str] | None:
    """Return the segments of *own_cgroup_path* up to its ``user@<uid>.service``.

    ``None`` when there is no such segment — i.e. nothing to anchor at, which
    is not an error: a sampler running outside a systemd user manager (or one
    handed α's ``OwnCgroup('', None)`` failure value) simply has no per-slice
    neighbourhood to enumerate.
    """
    segments = own_cgroup_path.split('/')
    for depth, segment in enumerate(segments, start=1):
        if segment.startswith(_USER_MANAGER_PREFIX):
            return segments[:depth]
    return None


def discover_pressure_cgroups(
    *,
    own_cgroup_path: str,
    cgroup_root: Path,
) -> list[tuple[str, Path]]:
    """Return sorted ``(leaf_name, cpu_pressure_path)`` for the sibling cgroups.

    PRD ``plans/load-throttle-harmonisation-prd.md`` §6.4: one row per
    ``df-*.slice`` present, ELSE one per ``orchestrator-*.service``. The
    fallback is not a theoretical branch — measured on this host on
    2026-09-13 there are ZERO ``df-*.slice`` (task 3394 has not landed) and
    seven ``orchestrator-*.service`` leaves, so the ELSE arm is the one
    actually running today.

    The search is ANCHORED at the ``user@<uid>.service`` segment of
    *own_cgroup_path* rather than walking down from *cgroup_root*, for two
    reasons. Cost: measured here, ``rglob('orchestrator-*.service')`` from the
    cgroup root takes 32.7 ms and an ``os.walk`` capped at depth 6 takes
    19.9 ms, against 0.30 ms for the anchored glob — and this runs every 5 s,
    so an unanchored sweep would spend 0.6% of every tick walking cgroupfs for
    a result that changes only when a unit starts. Correctness: another user
    manager's slices are not ours, and an unanchored search would report them
    under our own metric names.

    The anchor costs no new /proc/self/cgroup reader: α's
    ``shared.psi.resolve_own_cgroup(None)`` already parses the ``0::`` line
    and returns that kernel path, and this takes the segments up to and
    including the ``user@`` one (INV-5 — reuse the reader, add no second
    parser).

    Never raises. Any failure — an unanchorable path, a cgroup_root that does
    not exist, a permission error mid-glob — returns ``[]``, which the caller
    reads as "no leaves discovered" and reports as such (see
    ``collect_load_metrics``); it never synthesises a leaf name.
    """
    segments = _anchor_segments(own_cgroup_path)
    if segments is None:
        return []
    anchor = Path(cgroup_root).joinpath(*(s for s in segments if s))
    try:
        leaves = sorted(anchor.glob(_DF_LEAF_GLOB))
        if not leaves:
            leaves = sorted(anchor.glob(_ORCHESTRATOR_LEAF_GLOB))
    except OSError:
        logger.warning('cgroup discovery failed under %s', anchor, exc_info=True)
        return []
    return [(leaf.name, leaf / 'cpu.pressure') for leaf in leaves]

# ---------------------------------------------------------------------------
# PSI helpers — parse_pressure_file/read_pressure are re-homed to shared.psi
# (DA-D9: reuse, do not reimplement) and re-exported below so this module's
# public API is unchanged. See shared/src/shared/psi.py for the
# implementation and TestParserRehomedToShared in this module's test suite
# for the anti-drift object-identity guard.
# ---------------------------------------------------------------------------


def collect_psi(
    *,
    read: Callable[[str], str] = read_pressure,
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
        if parsed is None:
            logger.warning(
                'PSI parse miss for %s (no avg10 lines in %d bytes); '
                'emitting no row instead of a fabricated 0.0',
                file_name,
                len(text),
            )
            continue
        out[f'psi_{key_prefix}_some_avg10'] = parsed['some_avg10']
        out[f'psi_{key_prefix}_full_avg10'] = parsed['full_avg10']
    return out


# ---------------------------------------------------------------------------
# Process-metric counters (pure, take injected process iterables)
# ---------------------------------------------------------------------------


def collect_load_metrics(
    *,
    read_runqueue: Callable[..., RunqueueReading] = read_runqueue_ratio,
    own_cgroup_path: str | None = None,
    proc_cgroup_path: str | Path | None = None,
    cgroup_root: Path = _CGROUP_ROOT,
) -> dict[str, float]:
    """Return the runqueue and per-cgroup own-pressure metrics for one tick.

    PRD ``plans/load-throttle-harmonisation-prd.md`` §6.4. Shaped exactly like
    ``collect_psi``: pure over its injected seams, with the live defaults as a
    thin shell.

    Runqueue (PRD detail A). ``runqueue_read_ok`` is emitted on EVERY tick as
    1.0/0.0, and ``runqueue_ratio`` only when the read succeeded. It is a
    metric of its own rather than a log line because after D1
    ``runqueue_ratio`` is the sole LIVE CPU arm: β's recorded G7 waiver on
    storm-escape-required rests on being able to count how often that arm was
    readable at all, and a log line is not countable from the corpus ε1/ε2
    calibrate against. α degrades an unreadable /proc/stat to
    ``RunqueueReading(0.0, False)`` — a fail-open VALUE, not a reading — so
    persisting that 0.0 as a ratio would write a fabricated "completely idle
    host" row, re-introducing the defect task 1817 fixed in this module.

    Args:
        read_runqueue: α's ``read_runqueue_ratio`` by default; no /proc/stat
            reader is written here (INV-5).
        own_cgroup_path: The sampler's own ``0::`` kernel path. ``None`` asks
            α's ``resolve_own_cgroup`` for the live one, so there is no second
            /proc/self/cgroup reader either. That ``None`` is the branch every
            real tick takes, since ``__main__`` calls this with no arguments.
        proc_cgroup_path: Where to read that ``0::`` line FROM, forwarded to α
            unchanged. Only consulted when ``own_cgroup_path`` is ``None``,
            which is the only branch that resolves anything. ``None`` here
            leaves the argument out of α's call rather than spelling
            ``/proc/self/cgroup`` a second time, so the live path keeps its
            single home in α (heuristic 11) and the live resolution keeps its
            existing cache key.
        cgroup_root: Where the unified hierarchy is mounted.
    """
    runqueue = read_runqueue()
    out: dict[str, float] = {'runqueue_read_ok': float(runqueue.read_ok)}
    if runqueue.read_ok:
        out['runqueue_ratio'] = float(runqueue.ratio)

    if own_cgroup_path is None:
        seam = {} if proc_cgroup_path is None else {'proc_cgroup_path': proc_cgroup_path}
        own_cgroup_path = resolve_own_cgroup(
            None, cgroup_root=str(cgroup_root), **seam
        ).path
    leaves = discover_pressure_cgroups(
        own_cgroup_path=own_cgroup_path, cgroup_root=cgroup_root
    )
    if not leaves:
        # Severity by CAUSE, because this fires on a 5 s oneshot: an
        # unconditional warning here is 17,280 journal lines a day for a steady
        # condition, which buries the per-tick logger.exception lines the three
        # degrade handlers exist to surface. No anchor at all is STRUCTURAL and
        # expected (a dev box, a container, any sampler outside a systemd user
        # manager) -> debug. An anchor that resolved but enumerated nothing is a
        # surprise worth a warning: we are in a user manager and it holds
        # neither a df-*.slice nor an orchestrator-*.service. Either way the
        # corpus stays the channel that matters -- absent own_read_ok:<leaf>
        # rows make the condition countable without reading the journal.
        #
        # The surviving warning branch is per-tick too, and that was weighed
        # rather than overlooked. It is a SEVERITY signal, not new volume:
        # __main__ already writes one INFO `tick ...` line per tick, so the
        # journal carries 17,280 sampler lines a day on every host regardless,
        # and what this branch adds is a line an operator's `-p warning` filter
        # can see. De-duplicating it across ticks is what would cost: the unit
        # is Type=oneshot, so each tick is a FRESH PROCESS and a module-level
        # "already warned" memo would reset every 5 s. The only cross-tick
        # state on this host is the store's meta table, and reaching it from
        # here would hand a pure collector a store handle and invert the
        # module dependency (heuristic 7) to quieten a log line. On a host
        # where this branch is steady it is steady BECAUSE the sampler is
        # collecting nothing useful there, which is worth saying loudly once
        # per tick rather than never.
        say = logger.debug if _anchor_segments(own_cgroup_path) is None else logger.warning
        say(
            'no pressure cgroups discovered under the anchor derived from %r; '
            'emitting no own_* rows rather than inventing a leaf name',
            own_cgroup_path,
        )
    for leaf_name, pressure_path in leaves:
        some_avg10 = _read_leaf_some_avg10(pressure_path)
        out[f'own_read_ok:{leaf_name}'] = float(some_avg10 is not None)
        if some_avg10 is not None:
            out[f'own_cpu_some10:{leaf_name}'] = some_avg10
    return out


def _read_leaf_some_avg10(pressure_path: Path) -> float | None:
    """Return one cgroup's ``some avg10``, or ``None`` if it could not be read.

    A cgroup cpu.pressure has the same some/full avg10 shape as
    /proc/pressure/cpu, so the text goes straight to α's
    ``parse_pressure_file`` and no second parser is written (INV-5). That
    parser's ``None`` already means "unparseable", so an unreadable file is
    collapsed onto the SAME sentinel and the two failure modes share one exit
    — the shape α's own ``read_own_cgroup_pressure`` uses.

    ``read_own_cgroup_pressure`` itself is deliberately not reused: it reads
    the READING PROCESS's own cgroup, and this reads every sibling leaf.
    """
    try:
        parsed = parse_pressure_file(pressure_path.read_text())
    except OSError:
        logger.debug('cgroup pressure unreadable: %s', pressure_path, exc_info=True)
        return None
    if parsed is None:
        return None
    return parsed['some_avg10']


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
    For each verify.sh root, ``children(recursive=True)`` is called once to
    obtain the full descendant list in a single flat pass.  This avoids the
    exponential revisiting that a per-child recursive call produces:
    ``children(recursive=True)`` already returns all descendants, so recursing
    into each returned child would visit a node at depth N up to 2^N times.
    """
    seen_pids: set[int] = set()
    total = 0
    for proc in procs:
        if not _is_verify_sh(proc):
            continue
        try:
            subtree = [proc] + proc.children(recursive=True)
        except Exception:
            subtree = [proc]
        for p in subtree:
            if p.pid in seen_pids:
                continue
            seen_pids.add(p.pid)
            with contextlib.suppress(Exception):
                total += p.memory_info().rss
    return total


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

    procs = list(proc_iter(['pid', 'name', 'cmdline', 'memory_info']))

    # No separate cmdline() probe loop: _is_occt_gated and _is_verify_sh each
    # have their own try/except that silently skips processes that died or are
    # access-denied between listing and inspection.  A separate p.cmdline()
    # probe here would add a redundant OS call per process on the 5s hot path.
    return {
        'occt_queue_depth': float(count_occt_queue_depth(procs, fd9_exists)),
        'verify_concurrency': float(count_verify_concurrency(procs)),
        'verify_rss_total_bytes': float(sum_verify_rss(procs)),
    }
