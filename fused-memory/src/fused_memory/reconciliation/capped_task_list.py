"""The no-silent-caps render of a Stage-2 payload section's task list.

Audit sections promise COMPLETE enumeration yet carry a defensive render cap.
A clip here says so twice: the shared ``_NOTE:`` line in the section
(``utils/clip_disclosure.py::clip_note``, the token the Stage-2 prompt tells
the model to read as clipped coverage) and a WARNING log naming what was
dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from fused_memory.reconciliation.task_filter import format_task_list
from fused_memory.utils.clip_disclosure import clip_note

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
    note = clip_note(
        omitted=omitted,
        omitted_noun=omitted_noun,
        cap_name=cap_name,
        cap=cap,
        dropped_first=dropped_first,
    )
    return f'{format_task_list(tasks[:cap])}\n\n{note}'
