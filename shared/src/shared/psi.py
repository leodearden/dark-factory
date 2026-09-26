"""PSI (pressure stall information) reader — re-homed pure parser + reader.

PRD ``docs/prds/dispatch-admission-load-cap.md`` task DA1 (DA-D9: reuse, do
not reimplement). ``parse_pressure_file`` and the ``/proc/pressure/<name>``
reader were originally written and tested in
``sampler/src/sampler/metrics.py``; they are re-homed here VERBATIM
(behavior-identical) so both the load-sampler and the orchestrator import the
SAME parser instead of each maintaining their own copy that could drift and
re-derive the "CPU has no ``full`` line on some kernels" asymmetry bug.
``sampler.metrics`` re-exports ``parse_pressure_file`` from this module (see
that module's ``TestParserRehomedToShared`` identity guard). That re-home was
verbatim; the parser has since grown ADDITIVELY (the 60 s window, ruling D17),
which is exactly the growth having one home makes safe — both consumers get it
at once and neither can drift.

This module is a direct-import submodule — like ``shared.deploy_state`` — and
is deliberately NOT re-exported from ``shared/__init__.py``:
``shared/tests/test_public_api.py::TestInitAllCompleteness`` pins
``shared.__all__`` to a hardcoded module union, so consumers import via
``from shared.psi import ...``.

PSI v2 (PRD ``plans/load-throttle-harmonisation-prd.md`` §6.1/§6.3,
2026-09-08) adds two further load signals beside host PSI: a runqueue ratio
from /proc/stat and the reading process's own cgroup pressure. Each is an
independent component with its own read_ok flag, and ``PsiSample`` carries
all three.

Every import here is stdlib, and must stay that way: ``shared.psi`` is a
member of ``shared/tests/test_pure_stdlib_leaves.py::PURE_STDLIB_LEAVES``.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

__all__ = [
    'parse_pressure_file',
    'read_pressure',
    'PsiSample',
    'read_psi_sample',
    'RunqueueReading',
    'read_runqueue_ratio',
    'OwnCgroup',
    'resolve_own_cgroup',
    'OwnPressureReading',
    'read_own_cgroup_pressure',
]

_AVG_RE = re.compile(r'avg(10|60)=([0-9]+(?:\.[0-9]+)?)')

_PROC_STAT = '/proc/stat'
_PROC_SELF_CGROUP = '/proc/self/cgroup'
_CGROUP_ROOT = '/sys/fs/cgroup'


def parse_pressure_file(text: str) -> dict[str, float] | None:
    """Parse a /proc/pressure/<name> text into the two lines x two windows.

    Both averaging windows come off ONE scan, keyed ``<line>_avg<window>``:
    ``some_avg10``, ``some_avg60``, ``full_avg10``, ``full_avg60``. An absent
    line or an absent window defaults to 0.0, so ``full_avg10`` is 0.0 when the
    ``full`` line is absent (e.g. CPU on some kernels).

    The 60 s window is read because ruling D17 (task 3353) stamps host load on
    every verify summary and wants a window wider than the 10 s one, while
    forbidding a second PSI reader (INV-5) — so this reader grew rather than a
    parallel parser appearing in the consumer. The 300 s window the kernel also
    emits is deliberately NOT extracted: nothing reads it, and an unread key is
    a shape a consumer can come to depend on by accident.

    Returns:
        A four-key dict on success, or ``None`` if no some/full averaging value
        this parser reads could be extracted (empty text, garbage content, a
        truncated read, or a text carrying only windows this parser does not
        read).  A *partial* miss — ``some`` present but ``full`` absent, or one
        window present and the other absent — still returns a dict; that is a
        legitimate kernel behaviour, not a fault.  Only a *total* miss (no key
        extracted at all) returns ``None`` so callers can distinguish a
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
    result: dict[str, float] = {
        'some_avg10': 0.0,
        'some_avg60': 0.0,
        'full_avg10': 0.0,
        'full_avg60': 0.0,
    }
    found = False
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('some'):
            prefix = 'some'
        elif line.startswith('full'):
            prefix = 'full'
        else:
            continue
        # finditer, not search: one line carries BOTH windows, so a scan that
        # stops at the first match would silently drop avg60.
        for match in _AVG_RE.finditer(line):
            window, value = match.group(1), match.group(2)
            result[f'{prefix}_avg{window}'] = float(value)
            found = True
    if not found:
        return None
    return result


def read_pressure(name: str) -> str:
    """Read /proc/pressure/<name> from the live kernel."""
    return Path(f'/proc/pressure/{name}').read_text()


class RunqueueReading(NamedTuple):
    """The runqueue component of a PsiSample; ``read_ok`` False means degraded."""

    ratio: float
    read_ok: bool


def read_runqueue_ratio(*, proc_stat_path: str | Path = _PROC_STAT) -> RunqueueReading:
    """Read ``procs_running`` / ``len(os.sched_getaffinity(0))`` from /proc/stat.

    The ``procs_running`` extraction is the same line scan the in-repo
    diagnostic ``scripts/cgroup-stall-ratio.py`` uses, so the production reader
    and the diagnostic agree on which field they read. (Shape reuse only —
    ``scripts/`` is not importable from ``shared``, which must stay a
    ``PURE_STDLIB_LEAVES`` member.)

    Never raises. Any failure — the file missing or unreadable, no
    ``procs_running`` line, an unparseable value, an empty CPU affinity mask,
    or ``sched_getaffinity`` absent on a non-Linux host — degrades to
    ``RunqueueReading(0.0, False)``,
    carrying the failure BY VALUE so the arm goes inert rather than wrong.
    The broad except matches the module's shipped "never wedge dispatch"
    stance; loud, rate-limited operator logging is the scheduler's job, so
    this stays at a single debug line.
    """
    try:
        text = Path(proc_stat_path).read_text()
        line = next(
            line for line in text.splitlines() if line.startswith('procs_running')
        )
        running = int(line.split()[1])
        cpus = len(os.sched_getaffinity(0))
        return RunqueueReading(running / cpus, True)
    except Exception:
        logger.debug('runqueue read failed; component degraded', exc_info=True)
        return RunqueueReading(0.0, False)


class OwnCgroup(NamedTuple):
    """A resolved own-cgroup: the kernel path, and its sysfs cpu.pressure file.

    ``pressure_path`` is ``None`` exactly when ``path`` is ``''`` — i.e. when
    nothing could be resolved.
    """

    path: str
    pressure_path: Path | None


@functools.cache
def resolve_own_cgroup(
    project_id: str | None,
    *,
    proc_cgroup_path: str | Path = _PROC_SELF_CGROUP,
    cgroup_root: str | Path = _CGROUP_ROOT,
) -> OwnCgroup:
    """Resolve the reading process's own cgroup and its cpu.pressure file.

    Read the ``0::`` (unified-hierarchy) path from ``proc_cgroup_path`` and
    walk it leaf-upward: the first segment equal to ``df-<project_id>.slice``
    wins, else the leaf itself. Shapes are owned by PRD
    ``plans/load-throttle-harmonisation-prd.md`` §6.3, which is also the home
    of the parity fixtures this is tested against.

    Owning the kernel-path-to-sysfs-path join is what gives ``cgroup_root`` a
    purpose, and puts that join in exactly one home rather than duplicating it
    in ``read_own_cgroup_pressure``.

    The result is cached per process because the ~150 s gate tick must repeat
    neither the walk nor the join. The key includes both injected paths, so a
    fixture never shares an entry with the live defaults or with another
    fixture. ``resolve_own_cgroup.cache_clear()`` is the public invalidation
    seam — ``read_own_cgroup_pressure`` calls it after any failure of its own,
    a failed resolution included, and
    tests use that same seam rather than reaching into module internals.

    Never raises: any failure returns ``OwnCgroup('', None)``. A degenerate
    ``0::`` line with no path is one of those failures, not a resolution of
    the root cgroup: joining it would yield the ROOT ``cpu.pressure``, whose
    reading would then be reported under ``own_cgroup=''`` — a
    mis-attribution, and a break of this type's ``''`` / ``None``
    biconditional.
    """
    try:
        text = Path(proc_cgroup_path).read_text()
        line = next(line for line in text.splitlines() if line.startswith('0::'))
        path = line[len('0::') :]
        if not path:
            raise ValueError('0:: line carries no cgroup path')
        segments = path.split('/')
        if project_id is not None:
            slice_name = f'df-{project_id}.slice'
            for depth in range(len(segments), 0, -1):
                if segments[depth - 1] == slice_name:
                    path = '/'.join(segments[:depth])
                    break
        return OwnCgroup(path, Path(cgroup_root) / path.lstrip('/') / 'cpu.pressure')
    except Exception:
        logger.debug('own cgroup resolution failed; component degraded', exc_info=True)
        return OwnCgroup('', None)


class OwnPressureReading(NamedTuple):
    """The own-cgroup component of a PsiSample.

    ``cgroup`` is the path that was ATTEMPTED, so it names which cgroup failed
    even when ``read_ok`` is False.
    """

    cgroup: str
    some_avg10: float
    read_ok: bool


def read_own_cgroup_pressure(
    project_id: str | None,
    *,
    proc_cgroup_path: str | Path = _PROC_SELF_CGROUP,
    cgroup_root: str | Path = _CGROUP_ROOT,
) -> OwnPressureReading:
    """Read the ``some avg10`` of the reading process's own cgroup cpu.pressure.

    Resolution shapes are owned by PRD
    ``plans/load-throttle-harmonisation-prd.md`` §6.3 (see
    ``resolve_own_cgroup``). The text is handed to ``parse_pressure_file`` —
    a cgroup cpu.pressure has the same some/full avg10 shape as
    /proc/pressure/cpu, so there is no second parser (DA-D9), and that
    parser's ``None`` already means "unparseable", which maps straight onto
    ``read_ok=False`` — so an unreadable file is collapsed onto that same
    sentinel and the two failures share one exit.

    Never raises. A failure carries the attempted cgroup in the result so the
    degradation is visible by value (INV-11), and re-resolves: a
    ``df-<project_id>.slice`` that does not exist yet must be picked up when
    it appears, without restarting the orchestrator. Failing to RESOLVE is one
    of those failures — a memoized ``OwnCgroup('', None)`` from one transient
    /proc/self/cgroup read error would otherwise disable this arm for the
    process lifetime — so all three (no resolution, unreadable file,
    unparseable text) share ONE exit, which is where the cache is cleared.
    """
    own = resolve_own_cgroup(
        project_id, proc_cgroup_path=proc_cgroup_path, cgroup_root=cgroup_root
    )
    parsed = None
    if own.pressure_path is not None:
        try:
            parsed = parse_pressure_file(own.pressure_path.read_text())
        except Exception:
            logger.debug(
                'own cgroup pressure unreadable; component degraded', exc_info=True
            )
    if parsed is None:
        resolve_own_cgroup.cache_clear()
        return OwnPressureReading(own.path, 0.0, False)
    return OwnPressureReading(own.path, parsed['some_avg10'], True)


class _Arm(NamedTuple):
    """One row of the saturation truth table.

    ``field`` is simultaneously the cfg attribute holding this arm's threshold
    and the string ``tripping_metric`` reports, so the gate, the
    ``dispatch_deferred`` payload and the operator vocabulary stay one set.
    """

    field: str
    value: Callable[[PsiSample], float]
    component_ok: Callable[[PsiSample], bool]


_ARMS: tuple[_Arm, ...] = (
    _Arm('mem_full_avg10', lambda s: s.mem_full10, lambda s: s.read_ok),
    _Arm('runqueue_ratio', lambda s: s.runqueue_ratio, lambda s: s.runqueue_read_ok),
    _Arm('own_cpu_some_avg10', lambda s: s.own_cpu_some10, lambda s: s.own_read_ok),
    _Arm('mem_some_avg10', lambda s: s.mem_some10, lambda s: s.read_ok),
    _Arm('io_some_avg10', lambda s: s.io_some10, lambda s: s.read_ok),
    _Arm('cpu_some_avg10', lambda s: s.cpu_some10, lambda s: s.read_ok),
)
"""The saturation truth table, in the D10 reporting rank.

Single home (heuristic 11): ``saturated`` reads it order-insensitively
(``any``) and ``tripping_metric`` reads the SAME generator order-sensitively
(first element), so the OR set and the rank cannot drift apart. The rank
itself is owned by PRD ``plans/load-throttle-harmonisation-prd.md`` D10.

That single home is still ahead of the tree (2026-09-09): until β (task 3590,
PRD §8) switches ``scheduler.py::_note_heavy_deferral`` from its own if/elif
chain to ``tripping_metric()``, that site ranks in the older DA-D1 order
(cpu > mem_some > mem_full > io). Two live rankings, on purpose and for that
window only — they already agree on the reported DOMAIN (the cfg field
names), so β changes the order and nothing else.
"""


@dataclass(frozen=True)
class PsiSample:
    """Immutable PSI snapshot consumed by the orchestrator's dispatch-admission gate.

    ``read_ok=False`` is the DA-D6 fail-open sentinel (see ``read_psi_sample``):
    an unreadable or unparseable /proc/pressure/* file degrades the whole
    sample rather than partially gating on incomplete data.

    v2 (PRD ``plans/load-throttle-harmonisation-prd.md`` §6.1) appends five
    defaulted fields carrying two further, independently-read components:

    - ``runqueue_ratio`` — ``procs_running`` from /proc/stat divided by
      ``len(os.sched_getaffinity(0))``; ``runqueue_read_ok`` says whether that
      read succeeded (see ``read_runqueue_ratio``).
    - ``own_cpu_some10`` — the ``some avg10`` of the READING PROCESS's own
      cgroup ``cpu.pressure``; ``own_cgroup`` is the cgroup path actually read
      (``''`` when none was resolved) and ``own_read_ok`` says whether that
      read succeeded (see ``read_own_cgroup_pressure``).

    Each component carries its own ok flag so a partially-degraded sample is
    distinguishable from a healthy one BY VALUE rather than collapsing to a
    flat sentinel. The defaults are the "component absent" reading, so every
    shipped keyword construction stays valid.

    ``cpu_some60`` (ruling D17, task 3353) appends by that same rule: the host
    CPU ``some avg60``, carried for the verify-summary load stamp, and governed
    by the HOST component's ``read_ok`` like the four fields beside it. It is
    deliberately absent from ``_ARMS``: it is telemetry, and adding an arm would
    silently change ``saturated()`` / ``tripping_metric()``, whose D10 rank is
    owned by PRD ``plans/load-throttle-harmonisation-prd.md`` and tasks
    3590/3592.
    """

    cpu_some10: float
    mem_some10: float
    mem_full10: float
    io_some10: float
    read_ok: bool
    runqueue_ratio: float = 0.0
    runqueue_read_ok: bool = False
    own_cpu_some10: float = 0.0
    own_cgroup: str = ''
    own_read_ok: bool = False
    cpu_some60: float = 0.0

    def _tripping_arms(self, cfg) -> Iterator[_Arm]:
        """Yield the arms of ``_ARMS`` this sample trips, in D10 rank order."""
        for arm in _ARMS:
            # β (task 3590) must retire the ABSENCE half of this default with
            # `{a.field for a in _ARMS} <= set(PsiAdmissionConfig.model_fields)`
            # — `shared` cannot import that model, and without the guard a
            # later rename there silently disables an arm forever. `saturated`
            # documents why absence is tolerated until then.
            threshold = getattr(cfg, arm.field, None)
            if threshold is None:
                continue
            if not arm.component_ok(self):
                continue
            if arm.value(self) >= threshold:
                yield arm

    def saturated(self, cfg) -> bool:
        """Return True if any arm is at/over its configured avg10 threshold.

        ``cfg`` is duck-typed — only the ``_ARMS`` field names are read, and
        each is read with ``getattr(cfg, field, None)``, so an ABSENT attribute
        and an explicit ``None`` both mean "this arm is off". That keeps a
        v1 cfg (four thresholds, no ``runqueue_ratio`` / ``own_cpu_some_avg10``)
        working in the live dispatch gate while the config-side v2 lands
        separately, and it is the same reading the code defaults ask for.

        An arm also never trips when its own component read failed. The host
        ``read_ok`` gate stays OUTSIDE the loop, so an unreadable host PSI
        sample is non-saturated as a whole — including the two non-host arms
        (DA-D6 fail-open: a degraded sample must never trip the gate,
        regardless of the threshold values).
        """
        return self.read_ok and any(self._tripping_arms(cfg))

    def tripping_metric(self, cfg) -> str:
        """Return the name of the highest-ranked arm this sample trips.

        Precondition: ``saturated(cfg)``. The emitter only reaches this after
        the gate has already held, so a violation is a programming error, not
        a fail-open case — hence ``ValueError`` rather than a sentinel return,
        which would let a mis-wired emitter publish a metric for a sample that
        never tripped.

        The returned domain is the cfg FIELD NAMES, so the gate, the
        ``dispatch_deferred`` payload and the operator vocabulary stay one set.
        Ranking is whatever ``_ARMS`` order says; PRD
        ``plans/load-throttle-harmonisation-prd.md`` D10 owns that rank.
        """
        if not self.saturated(cfg):
            raise ValueError(
                'tripping_metric() requires a saturated sample; '
                f'saturated(cfg) is False for {self!r}'
            )
        return next(iter(self._tripping_arms(cfg))).field


class _HostReading(NamedTuple):
    """The host-PSI component: the five /proc/pressure/* fields plus its flag.

    No field is defaulted, on purpose: both construction sites are in this
    module, so a default would buy nothing but a silent 0.0 — which reads as a
    QUIET host — for a field some later edit forgets to pass.
    """

    cpu_some10: float
    cpu_some60: float
    mem_some10: float
    mem_full10: float
    io_some10: float
    read_ok: bool


_HOST_FAIL_OPEN = _HostReading(
    cpu_some10=0.0,
    cpu_some60=0.0,
    mem_some10=0.0,
    mem_full10=0.0,
    io_some10=0.0,
    read_ok=False,
)


def _read_host_pressure(read: Callable[[str], str]) -> _HostReading:
    """Read /proc/pressure/{cpu,memory,io}; degrade the host fields together.

    Fail-open (DA-D6): if any source is unreadable (``read`` raises ANY
    exception -- e.g. ``OSError``, or a ``UnicodeDecodeError`` from non-UTF-8
    content) or unparseable (``parse_pressure_file`` returns None), the whole
    HOST component degrades to all-zeros with ``read_ok=False`` rather than
    gating on partial data — this must never wedge dispatch. The except is
    deliberately broad rather than scoped to ``OSError`` alone: a
    caller-injected ``read`` is untrusted, and "never wedge dispatch" is an
    absolute guarantee, not one scoped to a single exception type. Loud,
    rate-limited logging on this condition is the caller's (DA3)
    responsibility; this stays side-effect-light (at most a single debug
    line) to avoid per-tick log spam.
    """
    try:
        cpu = parse_pressure_file(read('cpu'))
        mem = parse_pressure_file(read('memory'))
        io = parse_pressure_file(read('io'))
    except Exception:
        logger.debug('PSI read failed; failing open', exc_info=True)
        return _HOST_FAIL_OPEN

    if cpu is None or mem is None or io is None:
        logger.debug('PSI parse failed (unparseable source); failing open')
        return _HOST_FAIL_OPEN

    return _HostReading(
        cpu_some10=cpu['some_avg10'],
        cpu_some60=cpu['some_avg60'],
        mem_some10=mem['some_avg10'],
        mem_full10=mem['full_avg10'],
        io_some10=io['some_avg10'],
        read_ok=True,
    )


def read_psi_sample(
    *,
    read: Callable[[str], str] = read_pressure,
    project_id: str | None = None,
    proc_stat_path: str | Path = _PROC_STAT,
    proc_cgroup_path: str | Path = _PROC_SELF_CGROUP,
    cgroup_root: str | Path = _CGROUP_ROOT,
) -> PsiSample:
    """Read the three PSI components into one PsiSample.

    Maps cpu.some -> cpu_some10 (and its 60 s window -> cpu_some60), mem.some
    -> mem_some10, mem.full -> mem_full10, io.some -> io_some10, and adds the
    runqueue and own-cgroup components (PRD
    ``plans/load-throttle-harmonisation-prd.md`` §6.1/§6.3).
    ``project_id`` selects the slice name the own-cgroup resolver looks for.

    The three sources are orthogonal, so they are read INDEPENDENTLY and
    assembled once: a host-PSI failure zeroes only the host fields and
    sets ``read_ok=False``, while the runqueue and own-cgroup components keep
    whatever they separately obtained. Collapsing a partial degradation to a
    flat all-zero sentinel would discard a reading that actually succeeded,
    which is the silent-fail-soft shape INV-11 forbids; ``saturated``'s outer
    ``read_ok`` conjunct still makes such a sample non-saturated (DA-D6).

    Never raises — every component failure is carried in the result by value.
    """
    host = _read_host_pressure(read)
    runqueue = read_runqueue_ratio(proc_stat_path=proc_stat_path)
    own = read_own_cgroup_pressure(
        project_id, proc_cgroup_path=proc_cgroup_path, cgroup_root=cgroup_root
    )

    return PsiSample(
        cpu_some10=host.cpu_some10,
        mem_some10=host.mem_some10,
        mem_full10=host.mem_full10,
        io_some10=host.io_some10,
        read_ok=host.read_ok,
        runqueue_ratio=runqueue.ratio,
        runqueue_read_ok=runqueue.read_ok,
        own_cpu_some10=own.some_avg10,
        own_cgroup=own.cgroup,
        own_read_ok=own.read_ok,
        cpu_some60=host.cpu_some60,
    )
