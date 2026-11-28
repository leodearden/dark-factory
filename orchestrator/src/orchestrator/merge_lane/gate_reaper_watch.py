"""Decides from successive reap-stamp readings whether each remote host's gate
reaper still completes runs, and raises or clears that host's L1 (task 8520).
"""

from __future__ import annotations

import logging
import posixpath
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from escalation.models import Escalation

from orchestrator.merge_lane.liveness import _duration_text, _resolve_pending_alarm
from orchestrator.remote_gate import ReapStamp

logger = logging.getLogger('orchestrator.merge_lane')

_SENTINEL_PREFIX = '__gate_reaper_stale__'
_REAPER_SCRIPT = 'orchestrator/src/orchestrator/remote_gate/reap-gates.sh'


def gate_reaper_stale_sentinel(host: str) -> str:
    """The task id every gate-reaper L1 for *host* is filed under."""
    return f'{_SENTINEL_PREFIX}{host}'


@dataclass(frozen=True)
class _HostTrack:
    launched_at: float
    misses: int


class GateReaperWatch:
    """Judges each host's reap-stamp reading against the launch before it.

    A reading taken just before a launch shows whether the run the previous
    launch started completed.  A miss is a reading whose last completed run is
    older than the gap between the two launches plus :attr:`LAUNCH_GRACE_SECS`;
    :attr:`MISSES_BEFORE_ALARM` consecutive misses file one dedup'd L1 for the
    host, and the first judgement that shows a completed run resolves it.
    """

    MISSES_BEFORE_ALARM = 2
    # Absorbs ssh delay before the probe and clock jitter; it must stay far
    # below HostQuarantine.GATE_REAP_INTERVAL_SECS.
    LAUNCH_GRACE_SECS = 300.0

    def __init__(self, *, escalation_queue: Callable[[], Any]) -> None:
        self._escalation_queue = escalation_queue
        self._tracks: dict[str, _HostTrack] = {}

    def observe(
        self, host: str, reading: ReapStamp | None, *, launched_at: float, ssh_target: str | None,
    ) -> None:
        """Judge *host*'s *reading*, taken by the launch at *launched_at*.

        None, a launch that failed, neither counts as a miss nor resets the
        streak.  A host's first reading after a process start only starts its
        track: there is no previous launch to judge it against.
        """
        if reading is None:
            return
        previous = self._tracks.get(host)
        if previous is None:
            self._tracks[host] = _HostTrack(launched_at, 0)
            return
        window = launched_at - previous.launched_at + self.LAUNCH_GRACE_SECS
        if reading.shows_a_run_within(window):
            self._tracks[host] = _HostTrack(launched_at, 0)
            self._resolve(host)
            return
        track = _HostTrack(launched_at, previous.misses + 1)
        self._tracks[host] = track
        if track.misses >= self.MISSES_BEFORE_ALARM:
            self._alarm(host, reading, misses=track.misses, ssh_target=ssh_target)

    def _resolve(self, host: str) -> None:
        queue = self._escalation_queue()
        if queue is None:
            return
        resolved = _resolve_pending_alarm(
            queue, gate_reaper_stale_sentinel(host),
            f'The gate reaper on {host!r} completed a run again; auto-resolved by the merge lane.',
        )
        if resolved:
            logger.info('GateReaperWatch: %r completed a reap again; resolved %s', host, list(resolved))

    def _alarm(self, host: str, reading: ReapStamp, *, misses: int, ssh_target: str | None) -> None:
        queue = self._escalation_queue()
        if queue is None:
            return
        sentinel = gate_reaper_stale_sentinel(host)
        if queue.has_open_l1(sentinel):
            return
        last_run = 'never' if reading.age_secs is None else f'{_duration_text(reading.age_secs)} ago'
        summary = (
            f'Gate reaper on remote verify host {host!r} has completed no run for '
            f'{misses} consecutive launches (last completed run: {last_run})'
        )
        queue.submit(Escalation(
            id=queue.make_id(sentinel),
            task_id=sentinel,
            agent_role='orchestrator-verify-host-monitor',
            severity='blocking',
            level=1,
            category='gate_reaper_stale',
            summary=summary,
            detail=_alarm_detail(host, reading, last_run=last_run, misses=misses, ssh_target=ssh_target),
            suggested_action=(
                'Run the foreground reproduction in the detail to see why the reaper '
                'stops, fix the cause on the host, and let the next hourly launch '
                'resolve this alarm.  Never delete sha dirs or lock files by hand.'
            ),
        ))
        logger.warning('GateReaperWatch: %s', summary)


def _alarm_detail(
    host: str, reading: ReapStamp, *, last_run: str, misses: int, ssh_target: str | None,
) -> str:
    target = ssh_target or '<ssh target>'
    path = shlex.quote(reading.path)
    gate_root = shlex.quote(posixpath.dirname(reading.path))
    return (
        f'Host: {host}\n'
        f'SSH target: {ssh_target or "unknown"}\n'
        f'Stamp: {reading.path}\n'
        f'Last completed run: {last_run}\n'
        f'Consecutive launches with no completed run: {misses}\n'
        '\n'
        f'The merge lane\'s reprobe sweep launches {_REAPER_SCRIPT} on this host '
        'over ssh at most hourly; no systemd timer or cron runs it on the host.  '
        'A run that completes touches the stamp as its last step, and each launch '
        'first reads the stamp to judge the run before it.\n'
        '\n'
        f'Check the stamp: ssh {target} stat -c %y {path}\n'
        'Reproduce in the foreground, from a dark-factory checkout: '
        f'ssh {target} env GATE_ROOT={gate_root} bash -s < {_REAPER_SCRIPT}\n'
        '\n'
        'This alarm resolves itself at the next completed run.'
    )
