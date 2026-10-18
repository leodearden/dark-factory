"""Gather the context-complete reviewer's inputs from a checkout.

Production (``workflow.py::TaskWorkflow._review``) and the reviewer trial
(``evals/reviewer_trial/runner.py``) both call this module, so the trial
scores the inputs production gathers. Callers pass the main branch and the
conventions path; nothing here reads config.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import yaml
from shared.git_async import run_git

from orchestrator.agents.review_briefing import (
    ChangedFile,
    ReviewChange,
    ReviewConvention,
    ReviewSubject,
)

_GIT_TIMEOUT_SECS = 120.0


async def review_fork_point(checkout: Path, main_branch: str) -> str | None:
    """Where HEAD forked from *main_branch*, derived now; ``None`` when git cannot say.

    Re-derived at use time rather than read from a write-once stored base,
    which goes stale once main is merged into the branch (task 5695).
    """
    result = await run_git(
        ['git', 'merge-base', 'HEAD', main_branch], cwd=checkout, timeout=_GIT_TIMEOUT_SECS,
    )
    return result.stdout if result.ok and result.stdout else None


async def collect_review_change(checkout: Path, base: str) -> ReviewChange:
    """The change ``<base>...HEAD`` in *checkout*: its diff and every path it touches."""
    span = f'{base}...HEAD'
    diff, numstat = await asyncio.gather(
        _git_output(checkout, base, ['git', 'diff', '--no-renames', span]),
        _git_output(checkout, base, ['git', 'diff', '--no-renames', '--numstat', '-z', span]),
    )
    return ReviewChange(base_sha=base, diff=diff, files=_parse_numstat(numstat))


def load_review_conventions(path: Path) -> tuple[ReviewConvention, ...]:
    """The ``conventions`` list of a review briefing yaml; ``()`` when the file or list is absent.

    Raises ``ValueError`` naming *path* (and the entry index) for any other shape.
    """
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return ()
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f'{path}: not valid YAML: {exc}') from exc
    if document is None:
        return ()
    if not isinstance(document, dict):
        raise ValueError(f'{path}: expected a mapping at the top level, got {type(document).__name__}')
    entries = document.get('conventions')
    if entries is None:
        return ()
    if not isinstance(entries, list):
        raise ValueError(f'{path}: `conventions` must be a list, got {type(entries).__name__}')
    return tuple(_convention(path, index, entry) for index, entry in enumerate(entries))


async def gather_review_subject(
    checkout: Path,
    base: str,
    *,
    task: Mapping[str, object],
    plan: Mapping[str, object] | None,
    plan_absent_reason: str,
    conventions_relpath: str,
) -> ReviewSubject:
    """Everything the context-complete reviewer is briefed with, read from *checkout*."""
    return ReviewSubject(
        task=task,
        plan=plan,
        plan_absent_reason=plan_absent_reason,
        change=await collect_review_change(checkout, base),
        conventions=load_review_conventions(checkout / conventions_relpath),
    )


async def _git_output(checkout: Path, base: str, cmd: list[str]) -> str:
    result = await run_git(cmd, cwd=checkout, timeout=_GIT_TIMEOUT_SECS)
    if not result.ok:
        raise RuntimeError(
            f'`{" ".join(cmd)}` failed in {checkout} for review base {base}: {result.stderr}',
        )
    return result.stdout


def _parse_numstat(output: str) -> tuple[ChangedFile, ...]:
    """Parse ``git diff --no-renames --numstat -z``: ``<added>\\t<deleted>\\t<path>\\0`` per file."""
    files = []
    for record in output.split('\0'):
        if not record:
            continue
        added, deleted, path = record.split('\t', 2)
        files.append(ChangedFile(path, _count(added), _count(deleted)))
    return tuple(files)


def _count(field: str) -> int | None:
    """A numstat count; git writes ``-`` for a binary file."""
    return None if field == '-' else int(field)


def _convention(path: Path, index: int, entry: object) -> ReviewConvention:
    if not isinstance(entry, dict):
        raise ValueError(f'{path}: conventions[{index}] must be a mapping, got {entry!r}')
    rule, why = entry.get('rule'), entry.get('why')
    if not isinstance(rule, str) or not isinstance(why, str):
        raise ValueError(
            f'{path}: conventions[{index}] needs string `rule` and `why`, got {entry!r}',
        )
    return ReviewConvention(rule, why)
