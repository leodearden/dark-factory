"""The merge lane's remote verify hosts: which are benched, why, and when they come back.

``SpeculativeMergeWorker`` composes one :class:`HostQuarantine`, the lane's
only writer of the shared quarantine set and of the per-host
RunnerUnavailable tracker. Every other lane site asks it through a method:
a RunnerUnavailable goes through :meth:`HostQuarantine.bench` or
:meth:`HostQuarantine.quarantine_unreachable`, and a verdict divergence (the
land-time cross-check, the drift detective) through
:meth:`HostQuarantine.quarantine_divergent`.

Lane-side, the set leaves this class only into the ``verify_runner.HostAllocator``
it builds (:meth:`HostQuarantine.allocator_over`), which shares it BY
REFERENCE so a bench takes effect on the next acquire without a rebuild.
``HostAllocator.quarantine_and_release`` writes it under verify_runner's own
contract.  When the harness injects the set (task 6670) it also shares it
with the task path's HostPool, which is outside the lane.
"""
from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

from orchestrator.config import OrchestratorConfig
from orchestrator.deployed_sha import DeployedSha
from orchestrator.event_store import EventStore, EventType
from orchestrator.merge_lane.gate_reaper_watch import GateReaperWatch
from orchestrator.merge_lane.liveness import (
    _alarm_verify_host_skew,
    _alarm_verify_host_unreachable,
    _clear_verify_host_skew,
    _clear_verify_host_skew_of_an_earlier_process,
    _clear_verify_host_unreachable,
)
from orchestrator.merge_lane.ports import ClockPort
from orchestrator.merge_lane.types import (
    UNTRACKED_HOST_WIRE,
    RecoveryEvidence,
    _HostUnavailability,
    quarantine_class,
)
from orchestrator.remote_gate import GateIdentityPair, RunnerUnavailableKind
from orchestrator.verify_hosts import build_remote_runners
from orchestrator.verify_runner import HostAllocator, RunnerUnavailable

logger = logging.getLogger('orchestrator.merge_lane')

_Allocator = TypeVar('_Allocator', bound=HostAllocator)


def _ssh_targets_by_runner(config: OrchestratorConfig) -> Mapping[str, str]:
    """``{runner name: ssh_host}`` of *config*'s enabled verify runners."""
    return {r.name: r.ssh_host for r in config.enabled_verify_runners}


class _RunnerLookup:
    """The runner handle of a tracked host, for one reprobe sweep.

    Resolution asks ``allocator.remote_runner`` first, because a tracked host
    may NOT be in the quarantine set — the strand shape
    ``quarantined_remote_runners()`` structurally cannot report.  The
    ``quarantined_remote_runners()`` fallback keeps pre-existing allocator
    doubles (which predate ``remote_runner``) working.

    The fallback map is loop-INVARIANT, so it is built AT MOST ONCE per
    sweep, lazily on the first ``remote_runner`` miss (task 3043 amend).
    Against a real HostAllocator ``remote_runner`` always resolves, so it is
    never built and costs nothing; only the legacy-double path pays, and it
    pays O(n_quarantined) per sweep instead of per tracked host.
    """

    def __init__(self, allocator: Any) -> None:
        self._allocator = allocator
        self._resolve = getattr(allocator, 'remote_runner', None)
        self._fallback: dict[str, Any] | None = None

    def __call__(self, name: str) -> Any:
        runner = self._resolve(name) if self._resolve is not None else None
        if runner is not None:
            return runner
        if self._fallback is None:
            self._fallback = dict(self._allocator.quarantined_remote_runners())
        return self._fallback.get(name)


class HostQuarantine:
    """The lane's remote verify hosts out of the pool, and the sweep that brings them back.

    Collaborators arrive as frozen values (the clock, the deployed sha, the
    configured host names) or as zero-argument resolvers (the allocator, the
    event store, the escalation queue) that re-read the worker's live
    attribute on every call, because tests re-assign those after
    construction.
    """

    # task 3043 amend (reviewer_comprehensive, efficiency): rate limit for the
    # reprobe sweep's "still unreachable" record.  A host that is permanently
    # gone — a laptop removed from the fleet, the reify 2026-07-25 shape —
    # stays in the RU tracker forever while the worker's reprobe loop sweeps
    # every verify_host_reprobe_interval_s (120 s), so an unconditional INFO
    # per sweep is ~720 identical lines/day.  Unbounded repetition of an
    # UNCHANGING fact is what destroys the greppability the decision logging
    # exists to buy, so the record is INFO for the first few sweeps of an
    # episode, INFO again on any CHANGE (a new streak), and thereafter at most
    # one INFO per period — DEBUG in between, so nothing is ever actually
    # dropped.  Recovery / park-skip / unresolvable-runner records are
    # TRANSITIONS (at most one per episode) and are never throttled.
    # Class attributes so tests can monkeypatch them (mirrors
    # MAX_INFLIGHT_DEAD_VERIFY_ABORTS).
    REPROBE_STILL_DOWN_INFO_SWEEPS: int = 3
    REPROBE_STILL_DOWN_INFO_PERIOD_SECS: float = 1800.0
    # How often the sweep launches the remotes' detached gate reapers
    # (task 6565, design D2: at most hourly per process).
    GATE_REAP_INTERVAL_SECS: float = 3600.0

    def __init__(
        self,
        *,
        clock: ClockPort,
        allocator: Callable[[], Any],
        event_store: Callable[[], EventStore | None],
        escalation_queue: Callable[[], Any],
        deployed_sha: DeployedSha | None,
        verify_hosts: Sequence[str] = (),
        quarantine: set[str] | None = None,
        host_holder_reader: Callable[[], Mapping[str, str]] | None = None,
        on_host_readmitted: Callable[[str], None] | None = None,
    ) -> None:
        self._clock = clock
        self._allocator = allocator
        self._event_store = event_store
        self._escalation_queue = escalation_queue
        # Task 6565: handed to every RemoteRunner this lane builds.
        self._deployed_sha = deployed_sha
        # Task 6565: every verify runner named in config, enabled or not; at
        # start the worker resolves the bench L1s a previous process left
        # open for them.  A host since dropped from config is not covered.
        self._verify_hosts = tuple(verify_hosts)
        # The set of remote runner names out of the verify pool: benched for
        # a RunnerUnavailable (tracked below) or quarantined for verdict
        # divergence by the drift detective or the land-time cross-check
        # (untracked).  Additive across submissions; the dedup'd L1
        # escalation in DriftDetector is the durable cross-restart
        # protection.  None-safe: empty here so bare-worker tests that don't
        # set up remotes stay green.
        #
        # RESTART RE-TRUST WINDOW: an injected *quarantine* is owned by
        # the harness for the PROCESS lifetime (task 6670, PRD
        # plans/remote-task-leg-verify-prd.md decision 4, §12 Q4), shared by
        # reference with the task path's HostPool, so it survives an
        # in-process worker rebuild; the RU tracker below stays this worker's.
        # It still resets to empty on a process restart: a remote that
        # diverged (remote PASS / local FAIL) becomes re-eligible then.  The
        # dedup'd L1 blocking escalation (submitted by DriftDetector before
        # quarantine) is the cross-restart guard: an open
        # verify_drift_divergence L1 halts the merge queue across restart via
        # the normal EscalationQueue gate, so the re-trusted remote's verdict
        # is never authoritative for real merges until the escalation is
        # resolved.  Rely on that coupling rather than persisting the set.
        self._runner_quarantine: set[str] = set() if quarantine is None else quarantine
        # Task 6670: the harness's reader of task-path holds per remote host,
        # shown as task_held_by on the snapshot's host rows, and its hook for
        # a readmission (readmit_host), so the task pool re-offers the host.
        self._host_holder_reader = host_holder_reader
        self._on_host_readmitted = on_host_readmitted
        # Per-host RunnerUnavailable streak tracker (task 1795).  Keyed by
        # runner name; entries created on first failure, popped on recovery.
        # Used by _record_runner_unavailable / _record_runner_recovered to fire
        # a dedup'd escalation and auto-reprobe on recovery.  None-safe at call
        # sites: only populated when remote runners are configured.
        self._runner_unavailable: dict[str, _HostUnavailability] = {}
        # Alarm thresholds (copies of config knobs set by adopt_config;
        # defaults keep bare-worker tests green).
        self._unreachable_escalate_after_n: int = 3
        self._unreachable_escalate_after_secs: float = 600.0
        # {runner name: configured ssh_host}, the target the unreachability
        # alarm names (task 5290); set alongside the knobs above.
        self._runner_ssh_targets: Mapping[str, str] = {}
        # Rate-limit state for the reprobe sweep's "still unreachable" record
        # (task 3043 amend).  {host: (last_info_at, last_streak, sweeps_seen)}.
        # A host that is permanently gone stays tracked forever and the sweep
        # runs every reprobe interval (120 s), so an unconditional INFO per
        # sweep is ~720 identical lines/day — unbounded repetition of an
        # unchanging fact, which degrades exactly the greppability the
        # decision logging buys.  Written only by the reprobe sweep; popped in
        # lockstep with the tracker entry by _record_runner_recovered, so it
        # cannot outlive the episode it describes.
        self._reprobe_last_info: dict[str, tuple[float, int, int]] = {}
        # Skew benches opened per (host, deployed sha) across tracker episodes
        # (task 6565, design D6).  A readmit pops the tracker entry, so the
        # count that makes a host's second skew bench sticky lives here.
        self._skew_benches: dict[tuple[str, str | None], int] = {}
        # When _reap_idle_gates last launched the remote gate reapers.
        self._last_gate_reap_at: float | None = None
        # Task 8520: judges each launch's reap-stamp reading per host.
        self._gate_reaper_watch = GateReaperWatch(escalation_queue=escalation_queue)

    # ── the pool this lane hands out ─────────────────────────────────────

    def allocator_over(
        self, remotes: Iterable[Any], *, factory: Callable[..., _Allocator] = HostAllocator,
    ) -> _Allocator:
        """A *factory* allocator over *remotes* sharing the quarantine set BY REFERENCE.

        HostAllocator-driven (RunnerUnavailable) and DriftDetector-driven
        quarantines then share one source of truth, and a bench here stops
        ``acquire_remote`` handing the host out with no allocator rebuild.
        """
        return factory(list(remotes), quarantine=self._runner_quarantine)

    def remote_runners_from(self, config: OrchestratorConfig, cwd: Path) -> list:
        """The config's remote runners, gated on this process's deployed sha.

        A config with enabled runners needs that sha: without it there is no
        gate to dispatch through, and a remote verify must never fall back to
        a shared remote tree (task 6565, design D2).  ``cli.py::run`` refuses
        to start in that state, so this raise is the second line.

        Unfiltered by quarantine (task 6670): the allocator reads the shared
        set at acquire, and a host benched before this build must still get
        a slot, or its row is an orphan the reprobe cannot re-adopt.
        """
        enabled = [r.name for r in config.enabled_verify_runners]
        if not enabled:
            return []
        if self._deployed_sha is None:
            raise RuntimeError(
                f'merge worker has enabled verify_runners {enabled} but no deployed '
                'sha to gate them on (task 6565); start it with Harness(deployed_sha=...)'
            )
        return build_remote_runners(config, cwd, deployed_sha=self._deployed_sha)

    def adopt_config(self, config: OrchestratorConfig) -> None:
        """Adopt *config*'s alarm thresholds and the ssh targets the alarm names.

        The worker calls it when it builds its allocator, so operator config
        drives production while the ``__init__`` defaults keep bare-worker
        tests green (mirrors the ``_heartbeat_interval_s`` precedent).
        """
        self._unreachable_escalate_after_n = config.verify_host_unreachable_escalate_after_n
        self._unreachable_escalate_after_secs = (
            config.verify_host_unreachable_escalate_after_secs
        )
        self._runner_ssh_targets = _ssh_targets_by_runner(config)

    # ── reads ────────────────────────────────────────────────────────────

    def is_quarantined(self, name: str) -> bool:
        """Whether *name* is out of the verify pool, for any reason."""
        return name in self._runner_quarantine

    def quarantined_hosts(self) -> frozenset[str]:
        """Every host out of the verify pool, for any reason."""
        return frozenset(self._runner_quarantine)

    def unavailability(self, name: str) -> _HostUnavailability | None:
        """A copy of *name*'s open RunnerUnavailable episode, or None when it has none."""
        entry = self._runner_unavailable.get(name)
        return None if entry is None else copy.copy(entry)

    def unavailable_hosts(self) -> frozenset[str]:
        """Every host with an open RunnerUnavailable episode."""
        return frozenset(self._runner_unavailable)

    # ── per-host RunnerUnavailable tracker (tasks 1795, 6565) ────────────

    def _record_runner_unavailable(
        self, host: str, reason: str, now: float, *,
        kind: RunnerUnavailableKind = 'transport', identity: GateIdentityPair | None = None,
    ) -> bool:
        """Record one RunnerUnavailable of *kind* for *host* and return whether to escalate.

        Creates a new ``_HostUnavailability`` entry on the first call, then
        increments ``streak`` on each subsequent call.  ``first_unavailable_at``
        is fixed to the first-call timestamp so duration can be computed later.

        A 'skew'-class *kind* (:func:`quarantine_class`) sets the entry's kind,
        reason and identity, and when it opens a skew bench it counts toward
        the host's sticky bound (:meth:`_open_skew_bench`).  A 'transport' RU
        never downgrades a skew entry: it refreshes the reason of a transport
        entry only.

        Returns whether the caller should alarm: the entry's
        :meth:`_HostUnavailability.alarm_due` against this lane's streak and
        time thresholds, the same predicate the reprobe sweep re-asserts
        (task 5290).  The alarm helpers' ``has_open_l1`` dedup prevents
        re-submitting.
        """
        entry = self._runner_unavailable.setdefault(
            host, _HostUnavailability(streak=0, first_unavailable_at=now, reason=reason),
        )
        entry.streak += 1
        if quarantine_class(kind) == 'skew':
            if quarantine_class(entry.kind) == 'ru':
                self._open_skew_bench(host, entry)
            entry.kind, entry.reason, entry.identity = kind, reason, identity
        elif quarantine_class(entry.kind) == 'ru':
            entry.reason = reason
        return entry.alarm_due(
            now,
            after_n=self._unreachable_escalate_after_n,
            after_secs=self._unreachable_escalate_after_secs,
        )

    def _open_skew_bench(self, host: str, entry: _HostUnavailability) -> None:
        """Count a new skew bench of *host* on this deployed sha; the second is sticky.

        Every bench after the first follows a passing probe (a readmit is what
        closed the previous one), so the second bench means the cause survives
        a passing probe: re-admitting again would only spend another verify
        (design D6).  A failing reprobe opens no bench and so never counts.
        """
        key = (host, self._deployed_code_sha())
        self._skew_benches[key] = self._skew_benches.get(key, 0) + 1
        entry.sticky = self._skew_benches[key] >= 2

    def _record_runner_recovered(self, host: str) -> None:
        """Clear the unavailability tracker entry for *host* (idempotent).

        After this call, a subsequent ``_record_runner_unavailable`` starts a
        fresh episode with ``streak=1`` and a new ``first_unavailable_at``.

        The reprobe sweep's still-unreachable rate-limit state is popped in
        lockstep (task 3043 amend), so the next downtime episode for this host
        logs its first sweeps at INFO again instead of inheriting the previous
        episode's throttle.
        """
        self._runner_unavailable.pop(host, None)
        self._reprobe_last_info.pop(host, None)

    def _local_host_name(self) -> str:
        """Name of the local (trust-anchor) host, per the allocator when present.

        Reads :attr:`HostAllocator.local_name` (O(1)) rather than scanning
        ``host_states()`` for the ``is_local`` flag, which materialised one
        dict per host on every call (task 3043 amend).
        """
        allocator = self._allocator()
        if isinstance(allocator, HostAllocator):
            return allocator.local_name
        return 'local'

    def quarantine_unreachable(
        self, host: str, reason: str, now: float, *,
        kind: RunnerUnavailableKind = 'transport', identity: GateIdentityPair | None = None,
    ) -> None:
        """Quarantine *host* for a RunnerUnavailable of *kind* AND record it (tasks 3043, 6565).

        The ONE chokepoint every RunnerUnavailable bench goes through, so the
        design-D6 policy (:func:`quarantine_class`, :func:`escalation_policy`,
        no downgrade, the sticky bound) is applied in one place; a skew-class
        bench also emits ``verify_host_skew``.

        INV-A — **a host is never removed from the acquirable pool for an
        unreachability reason without a ``_runner_unavailable`` entry.**
        The reprobe sweep (:meth:`sweep`) can only re-adopt hosts it can see in
        that tracker, so a quarantine without a record is a permanent strand:
        the host stops being dispatched to and nothing can ever re-probe it,
        until an orchestrator restart rebuilds the allocator.

        Recovery from task 1795 was gated on a fragile *conjunction* — a host
        was re-adoptable only if it was in BOTH the shared quarantine set and
        the tracker — which three real paths violate: an orphaned verify task
        whose ``RunnerUnavailable`` escapes after the outer coroutine was
        cancelled, a ``cancel_and_release`` that leaves the slot PARKED against
        a down host, and any future quarantine site that forgets to record.
        Funnelling every RU-class quarantine through this ONE method makes the
        invariant checkable by reading a single function instead of auditing N
        call sites for a two-piece state update.

        Order: quarantine first (so ``acquire_remote`` stops handing the host
        out immediately — ``_runner_quarantine`` is shared BY REFERENCE with the
        allocator, so this needs no allocator rebuild and no restart), then
        record, then alarm.  Streak / ``first_unavailable_at`` / reason
        semantics are DELEGATED to :meth:`_record_runner_unavailable` so they
        are identical across this path and the existing RUNNER_UNAVAILABLE
        finalize path, and the L1 alarm keeps the existing
        ``has_open_l1`` dedup (at most one open alarm per host at a time).

        Local is a no-op *for state*: it is the trust anchor, is never
        quarantined, and has no remote to probe for recovery.  It is NOT a
        silent no-op (task 3043 amend): the previous RUNNER_UNAVAILABLE
        finalize path recorded a local lease unconditionally, so returning
        without a word here would make "the trust anchor reported itself
        unavailable" vanish entirely — the silent degradation this repo's
        loud-over-silent norm and the no-silent-fail-soft design invariant
        rule out.  A WARNING keeps the event greppable while preserving the
        never-quarantine-local rule.  None-safe throughout — with no allocator
        the tracker entry is still written, so recovery remains possible once
        one exists.
        """
        if host == self._local_host_name():
            logger.warning(
                'RU-class quarantine requested for the LOCAL trust-anchor host %r '
                '(reason: %s) — ignoring; local is never quarantined',
                host, reason,
            )
            return

        self._runner_quarantine.add(host)

        _should_escalate = self._record_runner_unavailable(
            host, reason, now, kind=kind, identity=identity,
        )
        entry = self._runner_unavailable[host]
        if quarantine_class(kind) == 'skew':
            self._emit_skew_bench(host, entry)
        if _should_escalate:
            self._alarm_host(host, entry, now - entry.first_unavailable_at)

    def bench(self, name: str, exc: RunnerUnavailable) -> None:
        """Bench remote *name* for *exc*, by its kind: the ``on_unavailable`` sink.

        Pools, drift checks and the train fallback hand a RunnerUnavailable
        here; it goes through :meth:`quarantine_unreachable`.
        """
        self.quarantine_unreachable(
            name, str(exc), self._clock.now(), kind=exc.kind, identity=exc.identity,
        )

    def quarantine_divergent(self, name: str) -> None:
        """Take *name* out of the pool for a verdict divergence: the detective-control sink.

        The land-time cross-check and the drift detective call it when a
        remote's verdict disagrees with the local trust anchor's.  It opens NO
        tracker entry, so the reprobe sweep never re-admits the host on mere
        reachability (Invariant 5): only an operator :meth:`clear` lifts it.
        """
        self._runner_quarantine.add(name)

    def _deployed_code_sha(self) -> str | None:
        return self._deployed_sha.sha if self._deployed_sha is not None else None

    def _emit_skew_bench(self, host: str, entry: _HostUnavailability) -> None:
        event_store = self._event_store()
        if event_store is None:
            return
        event_store.emit(EventType.verify_host_skew, data={
            'host': host, 'kind': entry.kind, 'reason': entry.reason,
            'sticky': entry.sticky, 'code_sha': self._deployed_code_sha(),
            'identity': entry.identity.to_wire() if entry.identity is not None else None,
        })

    def _alarm_host(self, host: str, entry: _HostUnavailability, duration_s: float) -> None:
        """File the dedup'd L1 for *host*'s bench, in its quarantine class's category."""
        if quarantine_class(entry.kind) == 'skew':
            _alarm_verify_host_skew(
                self._escalation_queue(), host, entry,
                code_sha=self._deployed_code_sha(), duration_s=duration_s,
            )
            return
        _alarm_verify_host_unreachable(
            self._escalation_queue(),
            host,
            entry.reason,
            streak=entry.streak,
            duration_s=duration_s,
            ssh_target=self._runner_ssh_targets.get(host),
            event_store=self._event_store(),
        )

    def host_states(self, now: float) -> list[dict]:
        """Per-host slot state + quarantine classification for the worker's ``snapshot()``.

        One dict per host the allocator manages, in allocator order (local
        first), then one per **orphan** RU-tracked host (see below), each with
        a **uniform schema** — every key always present, ``None`` where not
        applicable::

            {name, is_local, slot_state, quarantined, quarantine_class,
             unavailable_since, unavailable_secs, streak, reason,
             kind, sticky, identity, task_held_by}

        *now* is the snapshot's own clock (``time.time()``), passed in so
        ``unavailable_secs`` is measured against the same instant as every
        other relative age in the snapshot (``entries[].age_secs``,
        ``verify_in_progress.age_secs``) rather than against a second,
        slightly-later read of the wall clock.

        The allocator owns slot state and quarantine membership
        (:meth:`~orchestrator.verify_runner.HostAllocator.host_states`); the
        RunnerUnavailable streak tracker is owned here, so the RU enrichment
        is applied here rather than pushed down into the allocator.

        Notes
        -----
        - ``quarantine_class`` of a quarantined host is the tracked entry's
          :func:`quarantine_class` — ``'ru'`` (re-probed by ``ssh true``) or
          ``'skew'`` (re-probed by ensure-gate, task 6565) — or
          ``'divergence'`` for an untracked one, held for verdict divergence
          and cleared only by an operator.  Tracker membership is the same
          predicate the tracker-driven reprobe sweep loops on, and
          :meth:`probe_host` dispatches on the same function, so this
          snapshot and the reprobe can never disagree.  ``kind``, ``sticky``
          and ``identity`` (``GateIdentityPair.to_wire()``) come from the
          entry too, so a skew bench shows both gate identities.
        - ``unavailable_since`` / ``unavailable_secs`` / ``streak`` / ``reason``
          are populated whenever the host is RU-tracked, **independent of
          quarantine**, so a host accumulating failures below the quarantine
          threshold is visible too.  ``unavailable_since`` is the absolute
          epoch (for cross-log correlation); ``unavailable_secs`` is the
          derived *relative* downtime — ``max(0.0, now - first_unavailable_at)``
          — which is the operationally interesting form ("how long has this
          host been down") and matches how every other time-valued field in
          the snapshot is expressed.  Clamped at 0 so a clock step backwards
          cannot surface a negative duration.
        - **Orphan RU entries.** ``_runner_unavailable`` is only pruned on
          recovery (:meth:`_record_runner_recovered`), while the allocator is
          built once and cached for the worker's lifetime, so a host that
          leaves the managed set (renamed/removed remote) can keep a live RU
          streak that no allocator slot corresponds to.  Those names are
          appended after the managed hosts with ``slot_state=None``
          (``None`` = "not allocator-managed", keeping the non-null slot
          vocabulary exactly :data:`~orchestrator.verify_runner._SLOT_WIRE`)
          and ``is_local=False``, so an orphaned streak is never invisible —
          that blind spot is the same class of gap this block exists to close.
          Quarantine membership for them is read via
          :meth:`~orchestrator.verify_runner.HostAllocator.is_quarantined`,
          which answers from the shared-by-reference quarantine set and so
          works for names the allocator does not manage —
          :meth:`~orchestrator.verify_runner.HostAllocator.host_states` cannot
          answer it, since an orphan has no slot to report.  Consequence:
          ``len(hosts)`` equals ``occupancy.hosts_total`` **plus** the number
          of orphan entries; with no orphans (the steady state) the two agree.
        - The ``isinstance`` gate is a TYPE check, not a defensive fail-soft:
          the worker's ``_ensure_host_allocator`` always constructs a real
          :class:`~orchestrator.verify_runner.HostAllocator` and nothing else is
          ever assigned in production, so the ``[]`` branch is unreachable
          there.  It exists solely because the worker's ``_host_allocator``
          carries a ``MagicMock`` double at 79 assignment sites across 15 test
          files; those doubles answer every attribute, so ``hasattr``/duck-typing
          would let them crash or inject garbage into this read-only path.
        - ``task_held_by`` is the harness's task-hold reading for the host
          (``'task:<id>[,task:<id>]'``), else None.  It is independent of
          ``slot_state``: a task turn on a remote host takes no merge slot
          (task 6670, hosts are symmetric).
        - ``[]`` therefore also means "allocator not built yet (no verify
          dispatched)".  Deliberately NOT a fabricated ``local`` entry — an
          absent allocator is a distinct fact from an idle one.
        """
        allocator = self._allocator()
        if not isinstance(allocator, HostAllocator):
            return []
        holders = self._host_holder_reader() if self._host_holder_reader is not None else {}

        def _entry(name: str, *, is_local: bool, slot_state: str | None,
                   quarantined: bool) -> dict:
            ru = self._runner_unavailable.get(name)
            return {
                'name': name,
                'is_local': is_local,
                'slot_state': slot_state,
                'quarantined': quarantined,
                'quarantine_class': (
                    (quarantine_class(ru.kind) if ru is not None else 'divergence')
                    if quarantined else None
                ),
                **(ru.to_wire(now) if ru is not None else UNTRACKED_HOST_WIRE),
                'task_held_by': holders.get(name),
            }

        out: list[dict] = [
            _entry(
                st['name'],
                is_local=st['is_local'],
                slot_state=st['slot_state'],
                quarantined=bool(st['quarantined']),
            )
            for st in allocator.host_states()
        ]
        # Orphan RU entries: RU-tracked hosts with no allocator slot. Appended
        # after the managed hosts so the allocator's own ordering (local first,
        # then remotes in declaration order) is untouched; sorted for a stable
        # snapshot across calls.
        _managed = {h['name'] for h in out}
        for _orphan in sorted(set(self._runner_unavailable) - _managed):
            out.append(_entry(
                _orphan,
                is_local=False,
                slot_state=None,
                quarantined=allocator.is_quarantined(_orphan),
            ))
        return out

    # ── the reprobe sweep ────────────────────────────────────────────────

    async def sweep(self, now: float) -> None:
        """One reprobe sweep: the worker's reprobe loop runs it every reprobe interval.

        Three steps: the tracker-driven :meth:`_reprobe_quarantined_hosts`
        first, then :meth:`_reprobe_untracked_parked_slots`, the backstop for
        PARKED hosts with no tracker entry, then the at-most-hourly
        :meth:`_reap_idle_gates`, which also checks each host's reaper still
        completes its runs.

        Each step is fault-isolated: an unexpected exception is logged and
        swallowed (same pattern as the worker's heartbeat loop), so a probe
        bug can neither crash the worker nor disable the other steps.
        """
        try:
            await self._reprobe_quarantined_hosts(now)
        except Exception:
            logger.exception('reprobe_quarantined_hosts: unexpected error in loop')
        try:
            await self._reprobe_untracked_parked_slots()
        except Exception:
            logger.exception('reprobe_untracked_parked_slots: unexpected error in loop')
        try:
            await self._reap_idle_gates(self._clock.now())
        except Exception:
            logger.exception('reap_idle_gates: unexpected error in loop')

    async def _reprobe_quarantined_hosts(self, now: float) -> None:
        """Probe each RU-**tracked** remote host and re-engage it on recovery.

        Candidacy is **tracker-driven** (task 3043): the sweep iterates
        ``self._runner_unavailable`` and resolves a runner for each tracked
        host, rather than iterating the allocator's quarantine set and then
        requiring a tracker entry.

        Why the inversion.  The old conjunction ("in the quarantine set AND in
        the tracker") made recovery reachable only when two pieces of state
        happened to agree, and three real paths violate that agreement, each
        stranding a host permanently — non-acquirable, yet structurally
        invisible to this sweep:

        - an orphaned inner verify task whose ``RunnerUnavailable`` escapes
          after the outer coroutine was cancelled (in neither);
        - a ``cancel_and_release`` against a down host, which leaves the slot
          PARKED and non-acquirable *without* adding the host to the quarantine
          set (in neither, yet out of the pool);
        - any future quarantine site that forgets to record.

        A single positive condition ("the tracker says this host is
        unavailable, so probe it") covers **both** the RUNNER_UNAVAILABLE
        VerifyResult path and those escaping-exception / PARKED-strand paths.
        The INV-A chokepoint :meth:`quarantine_unreachable` guarantees the
        tracker entry exists for every RU-class removal from the pool.

        Per host (:meth:`_reprobe_host`):

        1. Probes via :meth:`probe_host` — ``runner.health()`` for an 'ru'
           bench, ensure-gate for a 'skew' one; a sticky bench is not probed.
        2. **On success** (host recovered): if the slot is PARKED and the runner
           exposes ``probe_clean``, re-admission additionally requires a clean
           probe — PARK means "the cancel RPC failed, so a stale verify may
           still be running there", and freeing the slot on mere reachability
           could double-dispatch onto a host still churning on the previous
           merge.  A dirty host stays tracked and parked so the NEXT sweep
           retries it.  A PARKED host whose runner does NOT expose the optional
           probe is re-admitted on reachability alone but **loudly** (WARNING) —
           see the inline note: refusing would make it permanently
           non-acquirable, and the arm is unreachable against a real
           ``HostAllocator``.  Otherwise the host is **re-admitted** via
           ``allocator.readmit`` — which un-quarantines *and* un-PARKs; plain
           ``clear_quarantine`` cannot recover a PARKED slot, since
           ``acquire_remote`` additionally requires ``_SLOT_FREE`` — then
           :meth:`readmit_host` resets the tracker and resolves the alarms on
           the evidence the probe named.
        3. **On failure** (host still unreachable): re-asserts the alarm while
           :meth:`_HostUnavailability.alarm_due` holds (dedup'd via
           ``has_open_l1``), so a resolve issued while the host still fails its
           probe — on an occupancy read, say — is re-opened on the next sweep:
           recovery is a measurement (the probe), never a proxy (task 5290).
           Probing before alarming avoids a spurious open→immediate-resolve L1
           churn for a host that recovers in the same cycle it would have tripped
           the time-based threshold.

        **Correctness invariant (Invariant 5)**: hosts quarantined for verdict
        divergence — by :class:`DriftDetector`'s sampled parity check OR by the
        land-time per-land cross-check in the post-merge verify (task 2822 fix
        b) — go through :meth:`quarantine_divergent`, which adds to the same
        quarantine set and never to ``self._runner_unavailable``, so
        tracker-driven candidacy excludes them **by construction** rather than
        by an explicit skip.  Clearing either on mere SSH reachability would
        bypass the verdict parity gate; both stay quarantined until an operator
        re-proves the host and clears it explicitly.

        Runner resolution (:class:`_RunnerLookup`) deliberately uses
        ``getattr`` + a ``quarantined_remote_runners()`` fallback rather than
        the ``isinstance(allocator, HostAllocator)`` gate used elsewhere in
        this class: the reprobe suites drive this method through lightweight
        allocator doubles that are not ``HostAllocator`` instances, and an
        isinstance gate would make the sweep a silent no-op for them.

        The tracker is iterated as a **snapshot** because the sweep mutates it
        (``_record_runner_recovered``).  Per-host exceptions are caught so one
        host's failure cannot abort the sweep for the remaining hosts.

        **Observability.** Every decision emits exactly one single-line record
        naming the host — recovered, still-unreachable, park-safety skip,
        unresolvable-runner (WARNING), and the allocator-missing early return
        while hosts are tracked (WARNING) — so grepping the orchestrator log
        for a host name answers "did the reprobe loop consider this host, and
        what did it decide?".  That question was unanswerable during the reify
        2026-07-25 incident, whose post-mortem had to argue from the *absence*
        of dispatch attempts.  This complements task 3275's snapshot/heartbeat
        observability, which reports per-host STATE but not this sweep's
        decisions.  Deliberately **silent** when the tracker is empty: the
        healthy steady state runs every ``verify_host_reprobe_interval_s``
        (120 s) and must not become log noise.

        For the same reason the **still-unreachable** record — the only one
        that repeats for an unchanging state — is rate-limited to
        ``REPROBE_STILL_DOWN_INFO_SWEEPS`` INFO lines per episode, plus one on
        any streak change, plus one per ``REPROBE_STILL_DOWN_INFO_PERIOD_SECS``
        thereafter; suppressed sweeps drop to DEBUG rather than vanishing (task
        3043 amend).  Every other record is a TRANSITION, at most one per
        episode, and is never throttled.
        """
        allocator = self._allocator()
        if allocator is None:
            self._warn_tracked_hosts_cannot_recover()
            return
        runner_for = _RunnerLookup(allocator)
        for name, entry in list(self._runner_unavailable.items()):
            try:
                await self._reprobe_host(name, entry, now, runner_for(name))
            except Exception:
                logger.exception(
                    'reprobe_quarantined_hosts: unexpected error probing %r; skipping',
                    name,
                )

    def _warn_tracked_hosts_cannot_recover(self) -> None:
        # Silent when nothing is tracked (the common "no verify dispatched
        # yet" case at a 120 s cadence); loud when hosts ARE tracked, since
        # then the missing allocator is the reason none of them can recover.
        if self._runner_unavailable:
            logger.warning(
                'reprobe: no host allocator built, but %d host(s) are tracked '
                'unavailable (%s) — none can be re-probed or re-admitted this sweep',
                len(self._runner_unavailable),
                ', '.join(sorted(self._runner_unavailable)),
            )

    async def _reprobe_host(
        self, name: str, entry: _HostUnavailability, now: float, runner: Any,
    ) -> None:
        downtime_s = now - entry.first_unavailable_at
        if runner is None:
            logger.warning(
                'reprobe: host %r is tracked unavailable (down %.0fs) but has '
                'no resolvable runner handle — it is stranded out of the verify '
                'pool and cannot be re-probed',
                name,
                downtime_s,
            )
            return
        # Probe first so a host that recovers in the same cycle it would
        # have tripped the time-based threshold goes directly to the
        # recovery path, avoiding a spurious open→immediate-resolve L1.
        evidence = await self.probe_host(name, runner)
        if evidence is not None:
            await self._readmit_benched_host(name, runner, now, evidence=evidence)
        else:
            self._note_host_still_out(name, entry, now)

    async def probe_host(self, name: str, runner: Any) -> RecoveryEvidence | None:
        """The evidence that tracked host *name* has recovered, or None (design D6, task 5290).

        An 'ru' bench answers ``runner.health()`` (``ssh true``), giving
        ``SSH_HEALTH_PROBE``.  A 'skew' bench answers ``runner.ensure_gate()``
        — publish, build or reuse the gate, handshake — giving ``ENSURE_GATE``,
        so a host only comes back once it can run this dispatcher's gate.  A
        sticky bench is never probed: it waits for an operator clear or a
        restart.
        """
        entry = self._runner_unavailable[name]
        if entry.sticky:
            return None
        if quarantine_class(entry.kind) == 'ru':
            return await self._ssh_health_evidence(runner)
        try:
            await runner.ensure_gate(event_store=self._event_store())
        except RunnerUnavailable as exc:
            logger.info('reprobe: ensure-gate for skew-benched host %r failed (%s): %s',
                        name, exc.kind, exc)
            return None
        return RecoveryEvidence.ENSURE_GATE

    @staticmethod
    async def _ssh_health_evidence(runner: Any) -> RecoveryEvidence | None:
        return RecoveryEvidence.SSH_HEALTH_PROBE if await runner.health() else None

    async def _readmit_benched_host(
        self, name: str, runner: Any, now: float, *, evidence: RecoveryEvidence,
    ) -> bool:
        """Re-admit *name* on *evidence* unless its PARKED slot is not yet clean.

        PARK means "the cancel RPC failed, so a stale verify may still be
        running there", so a PARKED slot is freed only on a clean
        ``probe_clean()``; a dirty host stays tracked and parked for the next
        sweep.  Returns whether the host was re-admitted.
        """
        allocator = self._allocator()
        _is_parked = getattr(allocator, 'is_parked', None)
        _was_parked = bool(_is_parked(name)) if _is_parked is not None else False
        _park_epoch = getattr(allocator, 'park_epoch', None)
        _probed_epoch = _park_epoch(name) if _park_epoch is not None else None
        entry = self._runner_unavailable.get(name)
        downtime_s = now - entry.first_unavailable_at if entry is not None else 0.0
        if _was_parked and not await self._parked_slot_is_clean(name, runner, downtime_s):
            return False
        if not self.readmit_host(name, now=now, evidence=evidence, expect_epoch=_probed_epoch):
            return False
        logger.info(
            'reprobe: host %r recovered after %.0fs down — re-admitted to '
            'the verify pool (slot un-PARKed: %s)',
            name,
            downtime_s,
            _was_parked,
        )
        return True

    async def _parked_slot_is_clean(self, name: str, runner: Any, downtime_s: float) -> bool:
        _probe_clean = getattr(runner, 'probe_clean', None)
        if _probe_clean is None:
            # FAIL-OPEN, but never SILENTLY (task 3043 amend).  Refusing would
            # make such a host PERMANENTLY non-acquirable; and against a real
            # HostAllocator it is unreachable, since cancel_and_release (the
            # only PARK writer) calls probe_clean() on that very path.
            logger.warning(
                'reprobe: host %r is reachable again after %.0fs and its '
                'slot is PARKED, but its runner (%s) exposes no '
                'probe_clean() — re-admitting on reachability ALONE; a '
                'stale verify could still be running there',
                name,
                downtime_s,
                type(runner).__name__,
            )
            return True
        if await _probe_clean():
            return True
        logger.info(
            'reprobe: host %r is reachable again after %.0fs but its '
            'slot stays PARKED pending a clean probe (a stale verify '
            'may still be running there); retrying next sweep',
            name,
            downtime_s,
        )
        return False

    def _note_host_still_out(self, name: str, entry: _HostUnavailability, now: float) -> None:
        """Log a failed reprobe (rate-limited) and re-assert the alarm while it is due."""
        downtime_s = now - entry.first_unavailable_at
        # Rate-limited (task 3043 amend): this is the one record that repeats
        # forever for a permanently-gone host.  INFO for the first
        # REPROBE_STILL_DOWN_INFO_SWEEPS sweeps of an episode, INFO on any
        # CHANGE (a new streak), then at most one INFO per
        # REPROBE_STILL_DOWN_INFO_PERIOD_SECS.  Suppressed sweeps log at DEBUG.
        _last_at, _last_streak, _sweeps = self._reprobe_last_info.get(
            name, (None, None, 0)
        )
        _at_info = (
            _last_at is None
            or _sweeps < self.REPROBE_STILL_DOWN_INFO_SWEEPS
            or entry.streak != _last_streak
            or (now - _last_at) >= self.REPROBE_STILL_DOWN_INFO_PERIOD_SECS
        )
        logger.log(
            logging.INFO if _at_info else logging.DEBUG,
            'reprobe: host %r still unavailable (%s%s) after %.0fs down '
            '(streak=%d) — staying out of the verify pool',
            name,
            quarantine_class(entry.kind),
            ', sticky' if entry.sticky else '',
            downtime_s,
            entry.streak,
        )
        self._reprobe_last_info[name] = (
            now if _at_info else (_last_at if _last_at is not None else now),
            entry.streak,
            _sweeps + 1,
        )
        if entry.alarm_due(
            now,
            after_n=self._unreachable_escalate_after_n,
            after_secs=self._unreachable_escalate_after_secs,
        ):
            self._alarm_host(name, entry, downtime_s)

    def readmit_host(
        self, name: str, *, now: float, evidence: RecoveryEvidence,
        expect_epoch: int | None = None,
    ) -> bool:
        """Return remote *name* to the verify pool and close its bench: THE readmit.

        Un-quarantines and un-PARKs it (``allocator.readmit``, refused when the
        park episode moved past *expect_epoch*), drops its tracker entry, and
        resolves its pending unreachability and skew alarms on the named
        *evidence* (task 5290).  The reprobe sweep calls
        it after a passing :meth:`probe_host`, and keeps the per-(host, sha)
        skew count that makes the next skew bench sticky;
        :meth:`clear` (the operator clear) calls it without a
        probe and then resets that count, as a restart does.  A readmission
        then calls the harness's ``on_host_readmitted`` hook, fault-isolated,
        so the task pool re-offers the host (task 6670).  Returns whether the
        host was re-admitted.
        """
        allocator = self._allocator()
        _readmit = getattr(allocator, 'readmit', None)
        if _readmit is None:
            if allocator is None:
                return False
            allocator.clear_quarantine(name)
        elif not _readmit(name, expect_epoch=expect_epoch):
            return False
        entry = self._runner_unavailable.get(name)
        downtime_s = now - entry.first_unavailable_at if entry is not None else 0.0
        self._record_runner_recovered(name)
        _clear_verify_host_unreachable(
            self._escalation_queue(), self._event_store(), name,
            downtime_s=downtime_s, evidence=evidence,
        )
        _clear_verify_host_skew(
            self._escalation_queue(), self._event_store(), name,
            downtime_s=downtime_s, evidence=evidence,
        )
        self._notify_host_readmitted(name)
        return True

    def _notify_host_readmitted(self, name: str) -> None:
        """Tell the harness *name* is back; a task-pool fault never undoes the readmission."""
        if self._on_host_readmitted is None:
            return
        try:
            self._on_host_readmitted(name)
        except Exception:
            logger.exception(
                'on_host_readmitted hook raised for re-admitted verify host %r; '
                'the merge-lane readmission stands', name,
            )

    async def clear(self, name: str) -> dict[str, Any]:
        """The operator clear (escalation tool ``clear_verify_host_quarantine``).

        Re-admits remote *name* whatever benched it — an 'ru' or 'skew' bench,
        sticky or not, or a verdict-divergence quarantine — without probing
        its health; a PARKED slot is still freed only once ``probe_clean()``
        finds no stale verify there.  Like a restart, a clear resets the
        host's skew count, so its next skew bench is a first one again
        (design D6: sticky lasts until a restart or an operator clear).
        Returns ``{host, cleared, quarantine_class, was_sticky}`` plus
        ``reason`` when nothing was cleared.
        """
        entry = self._runner_unavailable.get(name)
        quarantined = name in self._runner_quarantine
        result: dict[str, Any] = {
            'host': name,
            'cleared': False,
            'quarantine_class': (
                quarantine_class(entry.kind) if entry is not None
                else 'divergence' if quarantined else None
            ),
            'was_sticky': bool(entry is not None and entry.sticky),
        }
        allocator = self._allocator()
        runner = (
            allocator.remote_runner(name)
            if isinstance(allocator, HostAllocator) else None
        )
        if runner is None:
            return {**result, 'reason': f'{name!r} is not a remote verify host of this worker'}
        if entry is None and not quarantined:
            return {**result, 'reason': f'{name!r} is not benched'}
        if not await self._readmit_benched_host(
            name, runner, self._clock.now(), evidence=RecoveryEvidence.OPERATOR_CLEAR,
        ):
            return {
                **result,
                'reason': 'its slot is PARKED and a stale verify may still be running there',
            }
        self._skew_benches.pop((name, self._deployed_code_sha()), None)
        return {**result, 'cleared': True}

    def clear_bench_alarms_of_an_earlier_process(self) -> None:
        """Resolve the skew L1s a previous process left open for this lane's verify hosts.

        A restart clears every bench (design D6), so at start no bench of
        this process exists yet and any open skew L1 is its predecessor's.
        Covers the hosts named in config (``verify_hosts``); fault-isolated,
        since a queue error must never stop the worker starting.
        """
        for host in self._verify_hosts:
            try:
                _clear_verify_host_skew_of_an_earlier_process(
                    self._escalation_queue(), host, code_sha=self._deployed_code_sha(),
                )
            except Exception:
                logger.exception('could not resolve the earlier skew L1 of host %r', host)

    async def _reprobe_untracked_parked_slots(self) -> None:
        """Free PARKED hosts the tracker sweep cannot see; tracked ones stay its job."""
        allocator = self._allocator()
        if not isinstance(allocator, HostAllocator):
            return
        await allocator.reprobe_parked_slots(
            exclude=frozenset(self._runner_unavailable),
        )

    async def _reap_idle_gates(self, now: float) -> None:
        """At most hourly, launch the detached gate reaper of each remote in the pool (design D2).

        A benched host (tracked unavailable, or quarantined for any reason,
        verdict divergence included) is skipped.  Each launch's reap-stamp
        reading goes to the gate reaper watch (task 8520).
        """
        allocator = self._allocator()
        if not isinstance(allocator, HostAllocator):
            return
        last = self._last_gate_reap_at
        if last is not None and now - last < self.GATE_REAP_INTERVAL_SECS:
            return
        self._last_gate_reap_at = now
        for name in allocator.host_names:
            runner = allocator.remote_runner(name)
            benched = name in self._runner_unavailable or allocator.is_quarantined(name)
            if runner is not None and not benched:
                self._gate_reaper_watch.observe(
                    name, await runner.reap_idle_gates(),
                    launched_at=now, ssh_target=self._runner_ssh_targets.get(name),
                )
