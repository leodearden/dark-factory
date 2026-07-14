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


async def inspect_systemd_unit(
    unit: str,
    *,
    timeout_secs: float,
    reap_grace_secs: float = 5.0,
) -> dict:
    """Query systemctl for unit state fields needed for fresh-PID verify / health checks.

    Returns a dict with at minimum: MainPID (int), ActiveState (str),
    ActiveEnterTimestamp (str), ActiveEnterTimestampMonotonic (int).
    Integers default to 0 on parse failure (sentinel-safe).
    """
    proc = await asyncio.create_subprocess_exec(
        'systemctl', '--user', 'show', unit,
        '-p', 'MainPID,ActiveState,ActiveEnterTimestamp,ActiveEnterTimestampMonotonic',
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
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
        # On the VERIFY leg, MainPID=0 routes through the existing
        # verify-fail path: fresh-PID verify already treats MainPID=0 as
        # a sentinel failure -> born-at-L2 escalate + blocked (matching
        # the 2090 hardening pattern). On the BASELINE leg, MainPID=0
        # alone would NOT be caught there (only the verify leg checks
        # pid > 0) — run()'s baseline capture additionally checks
        # ActiveState=='' to catch a wedged baseline before the deploy
        # is even attempted.
        return {
            'MainPID': 0,
            'ActiveState': '',
            'ActiveEnterTimestamp': '',
            'ActiveEnterTimestampMonotonic': 0,
        }
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


def _deterministic_deploy_health_verdict(
    inspect_result: dict | None,
    verify_baseline: VerifyBaseline | Mapping | None = None,
) -> str:
    """Classify a systemd unit-inspector result as 'healthy' or 'unconfirmed'.

    Two modes:

    - **No baseline** (``verify_baseline=None``, the default): 'healthy' iff
      MainPID is a positive int AND ActiveState == 'active' — a conservative
      liveness signal (task 2074 design decision: brittle wall-clock
      ActiveEnterTimestamp comparison is deliberately avoided).  This is the
      EXACT pre-ζ behaviour, preserved verbatim for backward compat — a
      deploy stranded from BEFORE task 2240/ζ activated never persisted a
      baseline, so it always falls into this branch (see the CAVEAT below).
    - **With baseline** (ζ/task 2240, DS-3): 'healthy' iff MainPID is a
      positive int AND the live ActiveEnterTimestampMonotonic has advanced
      STRICTLY PAST the pre-deploy baseline's — real freshness, resolving
      the CAVEAT below for an always-on unit (a stale/unchanged monotonic
      now correctly reads 'unconfirmed' even when the unit is currently
      active, because the restart demonstrably did not happen).
      ``verify_baseline`` accepts either a ``VerifyBaseline`` instance or a
      plain ``Mapping`` (the ``to_metadata()``-shaped
      ``{'active_enter_timestamp_monotonic': ..., 'main_pid': ...}`` dict).

    None-safe throughout: a missing/malformed *inspect_result* is
    'unconfirmed'.

    CAVEAT (task 2074 amendment; superseded by the freshness branch above
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
    if not (isinstance(pid, int) and pid > 0):
        return 'unconfirmed'
    if verify_baseline is not None:
        baseline_monotonic = (
            verify_baseline.active_enter_timestamp_monotonic
            if isinstance(verify_baseline, VerifyBaseline)
            else verify_baseline.get('active_enter_timestamp_monotonic', 0)
        )
        live_monotonic = inspect_result.get('ActiveEnterTimestampMonotonic', 0)
        if isinstance(live_monotonic, int) and live_monotonic > baseline_monotonic:
            return 'healthy'
        return 'unconfirmed'
    if inspect_result.get('ActiveState') == 'active':
        return 'healthy'
    return 'unconfirmed'
