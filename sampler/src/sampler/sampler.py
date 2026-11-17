"""One load-sampler tick: collect the groups that are due, then admit and write them.

``collect_tick`` gives each group an outcome: not yet due, collected (its
samples and the state its next collection starts from), or failed. It hands
each collector the state the last written tick persisted, and it is the one
degrade handler: a collector that raises writes no rows, never a fabricated
0.0, and the other groups are still collected.

``run_tick`` admits each collected group against its own ``MetricGroup``. A
group carrying any key it does not admit is dropped for the tick with an ERROR
naming the group and the keys. The admitted rows and every attempted group's
record go to ONE ``LoadSampleStore.write_tick`` transaction, so a group's
state never commits without the rows computed from it, nor they without it.
Vacuum is ``sampler.__main__``'s.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType

from sampler.metrics import Collected, MetricGroup, SampleKey, Samples
from sampler.store import GroupRecord, LoadSampleStore, stamp_is_due

__all__ = ['Failed', 'LogThrottle', 'Outcome', 'Skipped', 'collect_tick', 'run_tick']

logger = logging.getLogger(__name__)

# A failing collector logs its traceback at most once per exception class in
# this long; every tick line still names the failure.
_FAILURE_LOG_INTERVAL_SECONDS = 3600


class Skipped(Enum):
    """A group that wrote nothing this tick, without having failed."""

    NOT_DUE = 'not-due'
    DROPPED = 'dropped'


@dataclass(frozen=True)
class LogThrottle:
    """When each failure kind last had its traceback logged.

    It is persisted in the group's record, because every tick is a process of
    its own and an in-memory throttle would forget on the next tick.
    """

    logged_at: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))

    def should_log(self, kind: str, now: int) -> bool:
        return stamp_is_due(self.logged_at.get(kind), now, _FAILURE_LOG_INTERVAL_SECONDS)

    def logged(self, kind: str, now: int) -> LogThrottle:
        return LogThrottle(MappingProxyType({**self.logged_at, kind: now}))

    def to_json(self) -> str:
        return json.dumps(dict(self.logged_at), sort_keys=True)

    @classmethod
    def from_json(cls, text: str | None) -> LogThrottle:
        """The throttle *text* encodes; absent or unreadable text is an empty throttle."""
        try:
            decoded = json.loads(text) if text is not None else {}
        except ValueError:
            return cls()
        if not isinstance(decoded, dict) or not all(
            isinstance(stamp, int) and not isinstance(stamp, bool) for stamp in decoded.values()
        ):
            return cls()
        return cls(MappingProxyType(decoded))


@dataclass(frozen=True)
class Failed:
    """A collector that raised ``kind``; ``log_throttle`` is the throttle to persist, if changed."""

    kind: str
    log_throttle: LogThrottle | None = None


Outcome = Collected | Failed | Skipped


def collect_tick(
    store: LoadSampleStore, now: int, groups: Iterable[MetricGroup]
) -> dict[MetricGroup, Outcome]:
    """Each of *groups*' outcome at *now*: not due, collected, or failed.

    Every group's record is read once, up front.
    """
    groups = tuple(groups)
    records = store.read_group_records([group.name for group in groups])
    return {group: _collect(store, group, records[group.name], now) for group in groups}


def _collect(store: LoadSampleStore, group: MetricGroup, record: GroupRecord, now: int) -> Outcome:
    if group.due_seconds is not None and not store.is_group_due(
        group.name, now, group.due_seconds
    ):
        return Skipped.NOT_DUE
    try:
        return group.collect(now, record.state)
    except Exception as exc:
        return _failed(group, exc, LogThrottle.from_json(record.log_throttle), now)


def _failed(group: MetricGroup, exc: Exception, throttle: LogThrottle, now: int) -> Failed:
    kind = type(exc).__name__
    if not throttle.should_log(kind, now):
        return Failed(kind)
    logger.error(
        'Failed to collect %s metrics (%s); writing the other groups only, and logging '
        'this kind of failure again in %d s at the earliest',
        group.name, kind, _FAILURE_LOG_INTERVAL_SECONDS, exc_info=exc,
    )
    return Failed(kind, throttle.logged(kind, now))


def run_tick(
    store: LoadSampleStore,
    now: int,
    outcomes: Mapping[MetricGroup, Outcome],
) -> dict[MetricGroup, Outcome]:
    """Write one tick of *outcomes*, and return each group's disposition.

    A collected group with an unadmitted key becomes ``Skipped.DROPPED``. Every
    attempted group — collected, dropped or failed — has its attempt stamped;
    only an admitted group's state is persisted, so a dropped or failed group's
    next collection spans the whole interval since its last written one.
    """
    dispositions = {group: _admit(group, outcome) for group, outcome in outcomes.items()}
    written = {g: d.samples for g, d in dispositions.items() if isinstance(d, Collected)}
    store.write_tick(
        now,
        unwindowed=_merge(s for g, s in written.items() if not g.windowed),
        windowed=_merge(s for g, s in written.items() if g.windowed),
        records={
            group.name: record
            for group, disposition in dispositions.items()
            if (record := _record(disposition, now)) is not None
        },
    )
    store.cleanup_old(now)
    return dispositions


def _admit(group: MetricGroup, outcome: Outcome) -> Outcome:
    if not isinstance(outcome, Collected):
        return outcome
    unadmitted = sorted(key.label() for key in outcome.samples if not group.admits(key))
    if unadmitted:
        logger.error(
            'dropping the %s group this tick: unadmitted keys %s', group.name, unadmitted
        )
        return Skipped.DROPPED
    return outcome


def _record(disposition: Outcome, now: int) -> GroupRecord | None:
    match disposition:
        case Skipped.NOT_DUE:
            return None
        case Collected(state=state):
            return GroupRecord(attempted_at=now, state=state)
        case Failed(log_throttle=throttle) if throttle is not None:
            return GroupRecord(attempted_at=now, log_throttle=throttle.to_json())
        case _:
            return GroupRecord(attempted_at=now)


def _merge(groups: Iterable[Samples]) -> dict[SampleKey, float]:
    return {key: value for samples in groups for key, value in samples.items()}
