"""Flag deduplication helpers for Stage 1 (MemoryConsolidator).

This module provides deterministic, code-level deduplication of Stage 1's
``items_flagged`` output.  The LLM has no memory of prior cycles, so the same
(task_id, flag_type) pair can be emitted cycle after cycle.  Rather than
relying on prompt instructions (fragile across model versions), the harness
post-processes each report: for flags with a computable *signature* we check
Mem0 for a prior ``stage1_flag_marker`` memory and, on a hit, annotate the
flag with ``persisted_from_run`` so Stage 2 can decide whether to re-act.

Public API
----------
- ``compute_flag_signature(flag)`` — cheap, sync, no I/O.
- ``dedup_flags(memory_service, project_id, run_id, flags)`` — async, does
  Mem0 search + optional write; best-effort (exceptions are logged, not raised).
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def compute_flag_signature(flag: dict[str, Any]) -> tuple[str, str] | None:
    """Return a (task_id_str, flag_type_str) signature for *flag*, or ``None``.

    Both ``task_id`` and ``flag_type`` must be present and truthy for a
    signature to be computed.  Values are coerced to ``str`` so that an integer
    task_id (common in LLM output) and a string task_id compare equal.

    Returns ``None`` for flags without enough signal to deduplicate — these are
    passed through unchanged by :func:`dedup_flags`.
    """
    task_id = flag.get('task_id')
    flag_type = flag.get('flag_type')
    if not task_id or not flag_type:
        return None
    return (str(task_id), str(flag_type))
