"""Typed BlockClass/BlockRecord bridge (PRD: plans/verify-plan-prd.md task ζ).

b3_gate.check_proposal classifies a dry_run_proposals[-1] entry read straight
from tasks.db JSON via two fragile string sniffs: a ``block_reason.startswith(
POST_MERGE_RED_MAIN_REASON_PREFIX)`` prose-prefix check for the
highest-blast-radius class, and a bare ``'status' in entry`` presence check
for "is this a failure entry". This module gives producers (dry_run_unblock,
workflow) a typed ``block_class`` field to stamp onto proposal entries so
b3_gate can dual-read: prefer the typed field, fall back to the legacy prose
sniff for in-flight proposals written before this field existed.

``BlockClass`` is a ``StrEnum`` so every member IS its string value —
``json.dumps`` output and every existing str-keyed comparison/membership
check stay byte-identical (mirrors ``orchestrator.verify_categories.
FailureCategory``, task α's pattern).

Design decision (Open Q5): BlockClass has exactly the 4 named semantic
classes b3_gate actually branches on — NOT one member per raw block-reason
prefix (there are ~15 on main today, across merge_gates/merge_queue/workflow,
and enumerating them 1:1 would re-couple this module to every incidental
merge-phase string, reintroducing the exact brittle prose coupling this
module removes). ``classify_block_reason`` is TOTAL: every string reason
maps to exactly one BlockClass, defaulting to AGENT_FAILURE.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BlockClass(StrEnum):
    """The closed 4-value semantic domain a blocked-task proposal falls into.

    b3_gate.check_proposal only branches DISTINCTLY on POST_MERGE_RED_MAIN
    (hard-abort, task 1680); the other three all take the identical
    risk/status/category/git gating path. See this module's docstring and
    plans/verify-plan-prd.md task ζ design decisions for the full rationale.
    """

    AGENT_FAILURE = 'agent_failure'
    # Reserved: no current producer stamps this and classify_block_reason(...)
    # never returns it (it only yields AGENT_FAILURE/MERGE_VERIFY_RED/
    # POST_MERGE_RED_MAIN). Intended to be set explicitly by a future
    # review-issue producer (task η), mirroring how η passes MERGE_VERIFY_RED
    # explicitly rather than deriving it. Inert today — not a live code path.
    REVIEW_ISSUES = 'review_issues'
    MERGE_VERIFY_RED = 'merge_verify_red'
    POST_MERGE_RED_MAIN = 'post_merge_red_main'


def classify_block_reason(reason: str) -> BlockClass:
    """Classify a raw ``block_reason`` string into its semantic BlockClass.

    TOTAL: every input string maps to exactly one BlockClass member — any
    reason not matching one of the 4 canonical merge_gates prefixes below
    defaults to AGENT_FAILURE. This default is intentional and covers
    workflow-originated reasons (e.g. 'Workflow error: ...', 'Planning
    failed: ...') as well as any future/unrecognized prefix — a missing
    mapping must never raise here, since this runs on the fire-and-forget
    dry-run-investigation producer path.

    Imports the 4 canonical prefix constants from ``orchestrator.merge_gates``
    LAZILY (inside this function body, not at module top level): merge_gates
    imports ``orchestrator.event_store`` and ``orchestrator.git_ops``, which
    are heavy. ``orchestrator.b3_gate`` imports ``BlockClass``/``BlockRecord``
    from this module and must stay dependency-light (stdlib sqlite3 only,
    config imported lazily) — a top-level merge_gates import here would
    transitively pull event_store/git_ops into the b3_gate CLI. This function
    is only ever called from the already-heavy producer path (workflow.py /
    dry_run_unblock.py), never from b3_gate itself, which only ever reads an
    already-serialised ``block_class`` value.

    Note: merge_queue's generic post-merge-verify-blocked reason (task η's
    MERGE_VERIFY_RED source) is an un-constanted f-string and is deliberately
    NOT mapped here, to avoid re-declaring a new prefix constant — task η
    passes ``block_class`` explicitly instead of relying on this classifier.
    """
    from orchestrator.merge_gates import (
        DROPPED_PLAN_TARGETS_REASON_PREFIX,
        PLAN_FILES_NOT_TOUCHED_REASON_PREFIX,
        POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX,
        POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX,
    )

    if reason.startswith(POST_MERGE_PYRIGHT_BROKEN_REASON_PREFIX):
        return BlockClass.POST_MERGE_RED_MAIN
    if (
        reason.startswith(DROPPED_PLAN_TARGETS_REASON_PREFIX)
        or reason.startswith(POST_MERGE_EQUIVALENCE_FAILED_REASON_PREFIX)
        or reason.startswith(PLAN_FILES_NOT_TOUCHED_REASON_PREFIX)
    ):
        return BlockClass.MERGE_VERIFY_RED
    return BlockClass.AGENT_FAILURE


@dataclass(frozen=True)
class BlockRecord:
    """The gate-relevant subset of a dry-run proposal entry, typed.

    5 of these 6 fields already exist as keys on a proposal entry
    (risk_label/head_sha/main_sha/files_referenced/investigated_at);
    ``block_class`` is the only additive key. Constructed by producers at
    the point where all 6 fields are available and stamped onto the entry
    via ``entry.update(record.to_dict())`` — additive, so sibling keys
    (proposal_text/block_reason/timestamp/status/cost_usd/diagnostics) are
    left untouched.
    """

    block_class: BlockClass
    risk_label: str
    head_sha: str | None
    main_sha: str | None
    files_referenced: list[str] = field(default_factory=list)
    investigated_at: str = ''

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, json-serialisable dict of exactly these 6 fields.

        ``block_class`` is emitted as its plain str value (not the StrEnum
        member) — pins the wire shape explicitly even though StrEnum members
        already serialise identically via ``json.dumps``.
        """
        return {
            'block_class': self.block_class.value,
            'risk_label': self.risk_label,
            'head_sha': self.head_sha,
            'main_sha': self.main_sha,
            'files_referenced': list(self.files_referenced),
            'investigated_at': self.investigated_at,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BlockRecord:
        """Build a BlockRecord from a mapping, reading exactly these 6 keys.

        Tolerates a superset mapping (e.g. a full proposal entry carrying
        proposal_text/status/block_reason/timestamp/cost_usd/diagnostics
        siblings) — everything else is ignored. ``.get`` defaults on the
        optional fields mean a partial dict still round-trips (Invariant B1).
        """
        return cls(
            block_class=BlockClass(d['block_class']),
            risk_label=d.get('risk_label', ''),
            head_sha=d.get('head_sha'),
            main_sha=d.get('main_sha'),
            files_referenced=list(d.get('files_referenced', [])),
            investigated_at=d.get('investigated_at', ''),
        )
