"""The no-silent-caps render of a Stage-2 payload section's task list.

Audit sections promise COMPLETE enumeration yet carry a defensive render cap.
This module owns what a clip must say: a ``_NOTE:`` line in the section (the
token the Stage-2 prompt tells the model to read as clipped coverage) and a
WARNING log naming what was dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from fused_memory.reconciliation.task_filter import format_task_list

logger = logging.getLogger(__name__)


def render_capped_task_list(
    tasks: list[dict],
    *,
    cap: int,
    cap_name: str,
    omitted_noun: str,
    dropped_first: str,
    log_event: str,
    log_extra: Mapping[str, object],
) -> str:
    """Render the first ``cap`` of ``tasks``, announcing any clip in the text and the log.

    The caller's ordering decides which tasks survive; ``dropped_first`` names
    the end it puts last. The WARNING carries ``log_extra`` plus ``rendered``
    and ``omitted``.
    """
    omitted = len(tasks) - cap
    if omitted <= 0:
        return f'{format_task_list(tasks)}\n'
    logger.warning(log_event, extra={**log_extra, 'rendered': cap, 'omitted': omitted})
    return (
        f'{format_task_list(tasks[:cap])}\n'
        f'\n_NOTE: {omitted} additional {omitted_noun} were omitted from this render by the '
        f'{cap_name}={cap} cap. Coverage was clipped — NOT complete this cycle; '
        f'{dropped_first} were dropped first._'
    )
