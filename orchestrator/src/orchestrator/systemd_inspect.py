"""Standalone systemd unit inspector (task 2119).

Hoists the task-2091 hardened ``systemctl --user show`` subprocess call —
previously duplicated between ``DeterministicRunner._default_inspect_unit``
and harness's ``_recon_inspect_unit`` — into ONE module-level function so
exactly one ``systemctl show`` subprocess site exists in the orchestrator
package (survey: harness-cluster "Duplicated systemctl inspector diverged";
bug arc 2091/2074/2087/2090).

``DeterministicRunner._default_inspect_unit`` and harness's
``_recon_inspect_unit`` are now thin delegates to :func:`inspect_systemd_unit`
below, and each preserves its own pre-existing injectable seam
(``self._unit_inspector`` / ``self._recon_unit_inspector``) unchanged.

Deliberately has NO Harness/DeterministicRunner instance dependencies — a
standalone module-level function so stream W10's ``proc_supervision.py`` can
import/relocate it without carrying orchestrator state (program seam table:
"never a second copy").
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Mapping

from shared.deploy_state import VerifyBaseline

logger = logging.getLogger(__name__)

# Task 2091: bound the `systemctl --user show` call's `communicate()` — an
# unbounded call here would strand a caller (DeterministicRunner's baseline/
# verify inspect, or the harness recon sweep) identically to task 2087's
# signature. 10s comfortably covers a normal `systemctl show` round trip.
_INSPECT_TIMEOUT_SECS: float = 10.0


def _wedged_unit_sentinel() -> dict:
    """The MainPID=0 degradation sentinel, in ONE place (task 4157).

    Returned by :func:`inspect_systemd_unit` whenever it cannot obtain a
    trustworthy reading — the task-2091 ``communicate()`` timeout, and (task
    4157) a subprocess SPAWN failure. ``DeterministicRunner._inspect_unit_guarded``
    returns it too, for an injected inspector that raises past this module.

    A FACTORY returning a fresh dict, deliberately NOT a module-level constant:
    callers do not merely read the result, they retain and embed it —
    ``_enrich_deploy_state_baseline`` persists baseline-derived values into
    ``deploy_state``, the baseline gate interpolates it into an escalation
    detail, and ``_capturing_inspector`` stashes it into ``captured['new_state']``
    for ``done_provenance``. A shared mutable dict would let any future
    in-place mutation along one of those paths corrupt every other call site's
    view, including across tasks in the same process.
    """
    return {
        'MainPID': 0,
        'ActiveState': '',
        'ActiveEnterTimestamp': '',
        'ActiveEnterTimestampMonotonic': 0,
    }


async def inspect_systemd_unit(
    unit: str,
    *,
    timeout_secs: float,
    reap_grace_secs: float = 5.0,
    degradation_sink: dict | None = None,
) -> dict:
    """Query systemctl for unit state fields needed for fresh-PID verify / health checks.

    Returns a dict with at minimum: MainPID (int), ActiveState (str),
    ActiveEnterTimestamp (str), ActiveEnterTimestampMonotonic (int).
    Integers default to 0 on parse failure (sentinel-safe).

    Never raises on a systemd-side failure: BOTH degradation modes — a
    subprocess SPAWN failure (task 4157: missing ``systemctl``, or a fork
    failure under resource pressure) and a ``communicate()`` timeout (task
    2091: systemd busy/hung) — return :func:`_wedged_unit_sentinel`'s
    MainPID=0 / ActiveState='' dict instead. That sentinel is rejected
    downstream rather than trusted, so a degraded reading fails closed: see
    the branch comment below.

    A non-``OSError`` spawn failure (e.g. ``NotImplementedError`` from an
    event loop with no subprocess support) is deliberately NOT caught — it
    signals a broken runtime, not a degraded systemd, and a sentinel would
    lie about the unit's state.

    ``degradation_sink`` is an optional caller-supplied dict used as an OUT
    parameter: when a degradation occurs, ``sink['cause']`` is set to a short
    human-readable string naming WHICH mode fired and why (the errno/exception
    repr for a spawn failure, the elapsed budget for a timeout). The four
    sentinel fields stay byte-identical in both modes — classifiers
    (``_deterministic_deploy_health_verdict``, the baseline ``ActiveState``
    gate, ``FreshPidVerify``) must not be able to tell the modes apart — but a
    HUMAN reading the resulting escalation must, since EMFILE, a missing
    ``systemctl`` binary and a hung systemd call for three different operator
    responses. Threading the cause here keeps that fact out of the journal-only
    path that INV-2 ``structured-facts-at-failure`` forbids. Omitting the sink
    (all pre-4157 callers) changes nothing.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            'systemctl', '--user', 'show', unit,
            '-p', 'MainPID,ActiveState,ActiveEnterTimestamp,ActiveEnterTimestampMonotonic',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError as exc:
        # Task 4157: the spawn itself can fail — a missing/unresolvable
        # `systemctl` (FileNotFoundError) or a fork failure under resource
        # pressure (EMFILE/ENOMEM, PermissionError). This `await` was
        # previously OUTSIDE any try, so such an OSError escaped raw past
        # DeterministicRunner.run()'s "always returns BLOCKED, never a raw
        # exception" contract and past the escalations an operator needs.
        # Converge on the SAME sentinel the timeout branch returns below, so
        # both degradation modes are indistinguishable to every CLASSIFIER —
        # deliberately NOT to the operator: the reviewer amendment threads the
        # distinguishing cause out through `degradation_sink` instead, because
        # EMFILE, a missing `systemctl` and a hung systemd need three different
        # operator responses even though all three must fail closed identically.
        logger.warning(
            'systemctl show %s could not be spawned (%r) — returning MainPID=0 sentinel',
            unit, exc,
        )
        if degradation_sink is not None:
            degradation_sink['cause'] = (
                f'systemctl show {unit} could not be spawned: {exc!r}'
            )
        return _wedged_unit_sentinel()
    try:
        stdout, _ = await asyncio.wait_for(
            proc.communicate(), timeout=timeout_secs,
        )
    except TimeoutError:
        # Task 2091: a wedged `systemctl show` here (systemd busy/hung, or
        # a grandchild inheriting the stdout pipe) would otherwise strand
        # the caller identically to task 2087's signature. This process is
        # NOT spawned with start_new_session=True (unlike
        # DeterministicRunner._default_run_script's), so it shares the
        # orchestrator's own process group — killing via
        # `_terminate_process_tree`'s `os.killpg` would risk a
        # self-inflicted SIGKILL. A direct `proc.kill()` is sufficient: a
        # plain `systemctl show` isn't expected to fork grandchildren the
        # way a deploy script can.
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.kill()
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=reap_grace_secs)
        logger.warning(
            'systemctl show %s timed out after %ss — returning MainPID=0 sentinel',
            unit, timeout_secs,
        )
        if degradation_sink is not None:
            degradation_sink['cause'] = (
                f'systemctl show {unit} timed out after {timeout_secs}s '
                f'(systemd busy/hung)'
            )
        # On the VERIFY leg, MainPID=0 routes through the existing
        # verify-fail path: fresh-PID verify already treats MainPID=0 as
        # a sentinel failure -> born-at-L2 escalate + blocked (matching
        # the 2090 hardening pattern). On the BASELINE leg, MainPID=0
        # alone would NOT be caught there (only the verify leg checks
        # pid > 0) — run()'s baseline capture additionally checks
        # ActiveState=='' to catch a wedged baseline before the deploy
        # is even attempted.
        #
        # Task 4157: the spawn-failure branch above converges on this same
        # sentinel, so BOTH degradation modes are rejected by those two
        # downstream checks identically — one sentinel, one definition
        # (`_wedged_unit_sentinel`), one set of consumers to reason about.
        return _wedged_unit_sentinel()
    result: dict = {}
    for line in (stdout or b'').decode(errors='replace').splitlines():
        if '=' in line:
            key, _, val = line.partition('=')
            result[key.strip()] = val.strip()
    # Coerce numeric fields — a missing / unparseable value is treated as 0 (sentinel)
    for field in ('MainPID', 'ActiveEnterTimestampMonotonic'):
        try:
            result[field] = int(result.get(field, 0))
        except (TypeError, ValueError):
            result[field] = 0
    return result


# Task 2611 (esc-2584-1): terminal ActiveState values accepted as proof that
# an empty-baseline (no persistent MainPID) deploy actually activated its
# target. An ALLOWLIST, not a ``('', 'failed')`` blocklist — a
# transient/mid-transition state (``'activating'``, ``'deactivating'``,
# ``'reloading'``), the wedged empty-string sentinel, and a missing/
# malformed ``ActiveState`` (``None``) are all rejected, not just the
# failed/wedged pair. ``'active'`` covers a ``.timer`` or a
# ``RemainAfterExit=yes`` oneshot; ``'inactive'`` covers a plain
# ``Type=oneshot`` service that ran to completion and settled back down.
_EMPTY_BASELINE_TERMINAL_STATES = frozenset({'active', 'inactive'})


def _empty_baseline_fresh(
    baseline_monotonic: int, live_monotonic: object, active_state: object,
) -> bool:
    """Empty-baseline (``baseline_main_pid == 0``) freshness predicate (task 2611).

    Shared by :func:`_deterministic_deploy_health_verdict` below (the
    recon-sweep classifier) and ``proc_supervision.RestartPlan.
    _execute_cross_unit_blocking`` (the live cross-unit verify leg) so the
    empty-baseline freshness rule lives in exactly one place — task 2611
    fixed the ``pid > 0`` false-block symmetrically at both sites, and a
    single shared predicate stops them from silently diverging if the rule
    ever changes again (program seam table: "never a second copy").

    A ``.timer`` unit or a ``Type=oneshot`` service never reports a
    persistent MainPID — even once genuinely active — so when the pre-deploy
    baseline observed no main process (``baseline_main_pid == 0``),
    freshness instead requires: ``live_monotonic`` (the live
    ``ActiveEnterTimestampMonotonic``) is an int that has strictly advanced
    past ``baseline_monotonic``, AND ``active_state`` has settled into
    :data:`_EMPTY_BASELINE_TERMINAL_STATES` (``'active'`` or ``'inactive'``).
    Everything else — ``'failed'``, the wedged ``''`` sentinel, a
    transient/mid-transition state, or ``None`` (a missing ``ActiveState``
    key on a malformed/partial inspect result) — is rejected.
    """
    return (
        isinstance(live_monotonic, int)
        and live_monotonic > baseline_monotonic
        and active_state in _EMPTY_BASELINE_TERMINAL_STATES
    )


def _deterministic_deploy_health_verdict(
    inspect_result: dict | None,
    verify_baseline: VerifyBaseline | Mapping | None = None,
) -> str:
    """Classify a systemd unit-inspector result as 'healthy' or 'unconfirmed'.

    Three modes:

    - **No baseline** (``verify_baseline=None``, the default): 'healthy' iff
      MainPID is a positive int AND ActiveState == 'active' — a conservative
      liveness signal (task 2074 design decision: brittle wall-clock
      ActiveEnterTimestamp comparison is deliberately avoided).  This is the
      EXACT pre-ζ behaviour, preserved verbatim for backward compat — a
      deploy stranded from BEFORE task 2240/ζ activated never persisted a
      baseline, so it always falls into this branch (see the CAVEAT below).
    - **With a persistent-PID baseline** (``verify_baseline.main_pid > 0`` —
      ζ/task 2240, DS-3): 'healthy' iff MainPID is a positive int AND the
      live ActiveEnterTimestampMonotonic has advanced STRICTLY PAST the
      pre-deploy baseline's — real freshness, resolving the CAVEAT below for
      an always-on unit (a stale/unchanged monotonic now correctly reads
      'unconfirmed' even when the unit is currently active, because the
      restart demonstrably did not happen). MainPID>0 remains the liveness
      signal in this mode, unchanged from pre-task-2611 behaviour.
    - **With an EMPTY baseline** (``verify_baseline.main_pid == 0`` — task
      2611, esc-2584-1): no persistent main process was observed at the
      pre-deploy baseline inspect, inferred to mean the target is a
      ``.timer`` unit, a ``Type=oneshot`` service, or a unit that simply was
      not running yet — none of which can ever report a live MainPID, even
      once genuinely active. Requiring MainPID>0 here would permanently
      misclassify a healthy install as 'unconfirmed', so it is dropped in
      favor of :func:`_empty_baseline_fresh`: 'healthy' iff the live
      ActiveEnterTimestampMonotonic has advanced STRICTLY PAST the (zero)
      baseline's AND ActiveState has settled into ``'active'`` or
      ``'inactive'`` (an ALLOWLIST, not a ``('', 'failed')`` blocklist — a
      transient/mid-transition state like ``'activating'`` or
      ``'deactivating'``, and a missing/malformed ``ActiveState`` (``None``),
      are rejected too, not just the failed/wedged pair).

    ``verify_baseline`` accepts either a ``VerifyBaseline`` instance or a
    plain ``Mapping`` (the ``to_metadata()``-shaped
    ``{'active_enter_timestamp_monotonic': ..., 'main_pid': ...}`` dict) in
    all baseline modes.

    None-safe throughout: a missing/malformed *inspect_result* is
    'unconfirmed'.

    CAVEAT (task 2074 amendment; superseded by the freshness branches above
    whenever a baseline is available): the no-baseline liveness check is NOT
    a freshness check — it does not confirm that *this* deploy's restart is
    what made the unit active, only that the unit is up right now.  For a
    long-lived/always-on service unit (the common case — e.g.
    'fused-memory.service'), the no-baseline verdict is near-constant
    'healthy' regardless of whether the triggering restart actually took
    effect, because the unit was probably already active before the deploy
    ran too.  A deterministic deploy that persisted a ``verify_baseline``
    (every deploy since ζ/task 2240) gets the real freshness comparison
    instead; only a deploy stranded from BEFORE ζ activated (no baseline was
    ever captured for it) falls back to this weaker signal — still strictly
    better than the prior silent-strand status quo, and both callers
    (Source A's stranded_blocked/resume filing and Source B's auto-resolve)
    RE-FILE/resolve an escalation rather than flipping task status directly,
    so a wrong verdict surfaces via the normal escalation/watcher machinery
    rather than silently corrupting state.
    """
    if not inspect_result:
        return 'unconfirmed'
    pid = inspect_result.get('MainPID', 0)
    pid_live = isinstance(pid, int) and pid > 0
    if verify_baseline is not None:
        baseline_monotonic = (
            verify_baseline.active_enter_timestamp_monotonic
            if isinstance(verify_baseline, VerifyBaseline)
            else verify_baseline.get('active_enter_timestamp_monotonic', 0)
        )
        baseline_pid = (
            verify_baseline.main_pid
            if isinstance(verify_baseline, VerifyBaseline)
            else verify_baseline.get('main_pid', 0)
        )
        live_monotonic = inspect_result.get('ActiveEnterTimestampMonotonic', 0)
        monotonic_advanced = (
            isinstance(live_monotonic, int) and live_monotonic > baseline_monotonic
        )
        if baseline_pid == 0:
            # Empty baseline (task 2611): pid>0 can never be satisfied for a
            # .timer/Type=oneshot target, so freshness falls back to the
            # shared _empty_baseline_fresh predicate (activation-clock
            # advance + a settled, non-transient, non-failed ActiveState).
            active_state = inspect_result.get('ActiveState')
            if _empty_baseline_fresh(baseline_monotonic, live_monotonic, active_state):
                return 'healthy'
            return 'unconfirmed'
        if pid_live and monotonic_advanced:
            return 'healthy'
        return 'unconfirmed'
    if pid_live and inspect_result.get('ActiveState') == 'active':
        return 'healthy'
    return 'unconfirmed'
