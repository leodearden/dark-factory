#!/usr/bin/env python3
"""One-shot audit: cross-check found_on_main-provenance done tasks against
actual commit lineage on the audited ref (default: main).

Motivation: a task marked done with ``metadata.done_provenance.kind ==
'found_on_main'`` was not landed by the orchestrator's own merge pipeline —
its deliverable was independently discovered already present on main.  That
provenance kind is intentionally exempt from the stronger ``is_ancestor`` +
commit-citation checks task 2500 added for the ``merged`` kind, which leaves
two retrospective blind spots this script sweeps for:

  (a) post-hoc revert — the cited commit genuinely landed, but was later
      reverted (or its declared files quietly removed) without the task
      being reopened.
  (b) misattribution — the cited commit is real and on the ref, but its
      message cites a *different* task, so it was never actually this
      task's deliverable.

For every found_on_main task this script:
  1. Parses ``metadata.done_provenance`` (commit/note) and ``metadata.files``
     (declared deliverables) via ``shared.task_metadata.parse_metadata``.
  2. Gathers git facts about the cited commit: is it reachable from the
     audited ref, what its message says, what files it touched, whether it
     was later reverted, and whether the declared files are still present
     at the ref's HEAD.
  3. Classifies the task into exactly one verdict via a fixed precedence
     ladder: ``commit_not_on_main`` > ``misattributed`` > ``reverted`` >
     ``deliverable_absent`` > ``unverifiable`` > ``ok`` (first match wins;
     see :func:`classify`).

``--apply`` is intentionally non-destructive: it annotates every flagged
task's ``metadata.x_provenance_audit`` (an ``x_``-prefixed forward-compat
key, so it round-trips through ``parse_metadata`` with no ``SchemaWarning``)
and lists every flagged task under ``needs_human_review`` in the summary —
it NEVER reopens a done task back to pending.  That correction is a human
decision.

Usage
-----
  # Dry run (default): print JSON report, touch nothing.
  python scripts/audit_found_on_main_provenance.py --project-root /path/to/project

  # Annotate flagged tasks' metadata (still never reopens anything).
  python scripts/audit_found_on_main_provenance.py --project-root /path/to/project --apply

  # Audit lineage against a ref other than main.
  python scripts/audit_found_on_main_provenance.py --project-root /path/to/project \\
      --ref origin/main
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from shared.task_metadata import parse_metadata

logger = logging.getLogger('audit_found_on_main_provenance')

# Mirrors orchestrator.git_ops.DEFAULT_COMMIT_CITATION_PATTERN (git_ops.py:219)
# conventions. NOT imported from there — orchestrator.git_ops pulls in the
# whole GitOps stack, and a fused-memory -> orchestrator runtime import is
# architecturally backwards (see task 2645 design decisions). This local
# pattern is generalized to EXTRACT every cited task id from a message
# (named capture groups), rather than testing one already-known id via
# str.format interpolation as the orchestrator version does:
#   - conventional-commit subject: `impl(50): ...` / `fix(50): ...`
#   - `task/{id}` branch mention: `Merge task/50 into main`, `... task/50 ...`
# `\d+` is greedy, so a captured id is always the FULL digit run — `task/339`
# never yields a citation match for id '3399', nor vice versa.
CITATION_PATTERN = re.compile(
    r'^(?:merge|impl|amend|fix|test|feat|chore|docs|refactor|style|build)'
    r'\(\s*(?P<conv_tid>\d+)\s*[):]'
    r'|\btask/(?P<branch_tid>\d+)\b',
    re.MULTILINE,
)

# Verdicts that apply_audit_annotations treats as "flagged" (annotated +
# surfaced under needs_human_review). `ok` and `unverifiable` are left alone.
_FLAGGED_VERDICTS = frozenset({
    'commit_not_on_main', 'misattributed', 'reverted', 'deliverable_absent',
})


@dataclass
class TaskProvenanceAudit:
    """Per-task facts gathered for the found_on_main provenance audit.

    Task facts (``task_id``/``title``/``commit``/``note``/``declared_files``)
    are populated by :func:`select_found_on_main_tasks`. The git-fact fields
    default to benign placeholders until :func:`build_audit_report` fills
    them in from an injected git-facts dependency. ``verdict``/``reasons``
    are set by :func:`classify`.
    """

    task_id: str
    title: str
    commit: str | None
    note: str | None
    declared_files: list[str]
    is_ancestor: bool = True
    commit_subject: str = ''
    commit_message: str = ''
    commit_files: list[str] = field(default_factory=list)
    revert_commit: str | None = None
    declared_files_missing_on_main: list[str] = field(default_factory=list)
    verdict: str = ''
    reasons: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Selection (pure)
# ---------------------------------------------------------------------------

def _id_as_int(task_id: str, fallback: int = 0) -> int:
    """Numeric task id as int for deterministic sort; non-numeric -> fallback.

    Mirrors ``audit_duplicate_tasks._id_as_int``: dotted subtask ids (e.g.
    ``'1.2'``) and other non-decimal shapes fall back rather than raising
    ``ValueError`` from a bare ``int()`` call.
    """
    return int(task_id) if str(task_id).isdecimal() else fallback


def select_found_on_main_tasks(tasks: list[dict]) -> list[TaskProvenanceAudit]:
    """Filter *tasks* to found_on_main-provenance ones, parsed via ``parse_metadata``.

    Tolerates ``metadata`` given as a dict OR a JSON string (``parse_metadata``
    handles both) and never raises on malformed metadata — a blob that fails
    validation resolves ``done_provenance`` to ``None`` via ``parse_metadata``'s
    warn-and-omit path, which this function treats the same as "no
    provenance at all" and simply skips.

    Returns audits sorted by ``int(task_id)`` (see :func:`_id_as_int`) for a
    deterministic report ordering downstream.
    """
    selected: list[TaskProvenanceAudit] = []
    for task in tasks:
        meta, _warnings = parse_metadata(task.get('metadata'), direction='read')
        prov = meta.done_provenance
        if prov is None or prov.kind != 'found_on_main':
            continue
        task_id = str(task.get('id', ''))
        selected.append(TaskProvenanceAudit(
            task_id=task_id,
            title=task.get('title') or '',
            commit=prov.commit,
            note=prov.note,
            declared_files=list(meta.files),
        ))
    selected.sort(key=lambda a: _id_as_int(a.task_id))
    return selected


# ---------------------------------------------------------------------------
# Citation helpers (pure)
# ---------------------------------------------------------------------------

def extract_cited_task_ids(message: str) -> set[str]:
    raise NotImplementedError


def commit_cites_task(message: str, task_id: str) -> bool:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Classifier (pure)
# ---------------------------------------------------------------------------

def classify(audit: TaskProvenanceAudit) -> tuple[str, list[str]]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Git I/O (injected) — thin async subprocess wrappers, each degrading to a
# safe/benign default on failure rather than raising (mirrors
# invalidate_fabricated_shipping_edges.py's _git_show_files shape).
# ---------------------------------------------------------------------------

async def _git_show_files(project_root: str, commit: str) -> list[str]:
    raise NotImplementedError


async def _git_is_ancestor(project_root: str, commit: str, ref: str) -> bool:
    raise NotImplementedError


async def _git_find_revert(project_root: str, commit: str, ref: str) -> str | None:
    raise NotImplementedError


async def _git_files_missing_on_ref(
    project_root: str, files: list[str], ref: str,
) -> list[str]:
    raise NotImplementedError


async def _git_commit_message(project_root: str, commit: str) -> str:
    raise NotImplementedError


class GitFacts:
    """Live git-inspection facade over a project's working tree.

    Thin async wrapper around the module-level ``_git_*`` subprocess
    helpers above, injected into :func:`build_audit_report` so its
    orchestration logic is unit-testable against a fake with canned
    per-commit facts (no live repo required).
    """

    def __init__(self, project_root: str) -> None:
        self.project_root = project_root

    async def gather(
        self, commit: str, ref: str, declared_files: list[str],
    ) -> dict[str, Any]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Orchestration (async)
# ---------------------------------------------------------------------------

async def build_audit_report(
    tasks: list[dict], git: Any, ref: str = 'main',
) -> dict[str, Any]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Apply layer (non-destructive)
# ---------------------------------------------------------------------------

async def apply_audit_annotations(
    backend: Any,
    project_root: str,
    report: dict[str, Any],
    *,
    tag: str | None = None,
) -> dict[str, Any]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

async def _run(args: argparse.Namespace) -> int:
    raise NotImplementedError


def main() -> int:
    raise NotImplementedError


if __name__ == '__main__':
    sys.exit(main())
