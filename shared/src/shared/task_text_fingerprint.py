"""Fingerprint of the task prose a plan was authored against (task 4031).

Basis: description and details, the prose `_format_task` renders; a title-only edit is
not an amendment here. Whitespace-collapsed, NOT casefolded: this is not
fused_memory.reconciliation.flag_dedup._normalize_content_description, which casefolds
for finding-dedup where case is noise. Here a case change is an author's change. The two
must differ; do not unify them. Contract: plans/task-amendment-delivery-prd.md §9 C1.
"""

import hashlib
from enum import StrEnum

__all__ = ['TaskTextDrift', 'task_text_drift', 'task_text_fingerprint']


class TaskTextDrift(StrEnum):
    CURRENT = 'current'
    AMENDED = 'amended'
    UNSTAMPED = 'unstamped'


def task_text_fingerprint(description: str | None, details: str | None) -> str:
    basis = ' '.join((description or '').split()) + '\x00' + ' '.join((details or '').split())
    return 'tfp:' + hashlib.sha256(basis.encode('utf-8')).hexdigest()[:32]


def task_text_drift(stamped_fp: object, current_fp: str) -> TaskTextDrift:
    if not isinstance(stamped_fp, str) or not stamped_fp:
        return TaskTextDrift.UNSTAMPED
    return TaskTextDrift.CURRENT if stamped_fp == current_fp else TaskTextDrift.AMENDED
