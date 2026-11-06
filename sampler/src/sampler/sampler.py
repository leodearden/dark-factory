"""One load-sampler tick: admit each group, write the admitted ones, enforce retention.

Each collected group is checked against its own ``MetricGroup``. A group
carrying any key it does not admit is dropped for the tick with an ERROR
naming the group and the keys, and the other groups are still written, all
in one ``LoadSampleStore.write_tick`` transaction. Vacuum is
``sampler.__main__``'s.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping

from sampler.metrics import MetricGroup, SampleKey, Samples
from sampler.store import LoadSampleStore

__all__ = ['run_tick']

logger = logging.getLogger(__name__)


def run_tick(
    store: LoadSampleStore,
    now: int,
    collected: Mapping[MetricGroup, Samples],
) -> dict[MetricGroup, Samples]:
    """Write one tick of *collected* samples, and return the groups written.

    A degraded group arrives as ``{}``, which is admitted and writes nothing.
    A group with an unadmitted key is absent from the result.
    """
    admitted: dict[MetricGroup, Samples] = {}
    for group, samples in collected.items():
        unadmitted = _unadmitted_labels(group, samples)
        if unadmitted:
            logger.error(
                'dropping the %s group this tick: unadmitted keys %s', group.name, unadmitted
            )
            continue
        admitted[group] = samples
    store.write_tick(
        now,
        unwindowed=_merge(s for g, s in admitted.items() if not g.windowed),
        windowed=_merge(s for g, s in admitted.items() if g.windowed),
    )
    store.cleanup_old(now)
    return admitted


def _unadmitted_labels(group: MetricGroup, samples: Samples) -> list[str]:
    return sorted(key.label() for key in samples if not group.admits(key))


def _merge(groups: Iterable[Samples]) -> dict[SampleKey, float]:
    return {key: value for samples in groups for key, value in samples.items()}
