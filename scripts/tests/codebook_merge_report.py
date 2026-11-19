"""Run one §7.3 coding record through the codebook's sole writer
(``legibility.codebook.apply_coding_records``) and compare the report it
returns.

Shared by the merger's unit suite (``test_codebook.py``) and the live-codebook
guards (``test_live_codebook.py``). A plain module rather than a conftest
fixture, like ``legibility_task_store_fake``: conftest appends this directory
to ``sys.path``, so both import it by bare name.
"""
from __future__ import annotations

from typing import Any

from legibility import codebook as mod

# The vocabulary of a project declaring no invariant slugs, for every merge
# test that is not about the slug screen.
NO_SLUGS: tuple[str, ...] = ()


def merge_one(codebook: dict, record: Any, slugs: tuple[str, ...] = NO_SLUGS) -> mod.MergeResult:
    return mod.apply_coding_records(codebook, [record], invariant_slugs=slugs)


def merge_report(merge: mod.MergeResult) -> tuple:
    """Everything a MergeResult reports besides its codebook, compared whole so
    a no-op pin cannot miss a report field."""
    return (merge.counts, merge.invariant_slugs, merge.skipped_deletions, merge.invalid_records)


def clean_merge_report(**counts: int) -> tuple:
    """The `merge_report` of a merge that skipped nothing and counted only
    *counts*. With no *counts* it is what the sole writer reports when it
    changed nothing."""
    return (mod.MergeCounts(**counts), mod.SlugTally(), (), ())
