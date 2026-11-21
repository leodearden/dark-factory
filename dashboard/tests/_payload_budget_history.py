"""The append-only rule for payload budget history, judged against the merge base.

A request's budget IS the last entry of its history, so "history only grows"
is the whole raise rule: every raise or lowering is a new entry, and the
loader refuses an entry with no task or reason. This module decides whether a
branch kept that rule, in two layers, split as ``_cache_buster_helpers.py`` is:

* DECISION, ``history_violations``: pure, over two parsed tables, so every
  red is constructible without a repository that exhibits it.
* GIT, ``resolve_base_table``: the only layer that shells out. It reads the
  table at ``merge-base(main, HEAD)``, never at main's tip, so an unrelated
  merge landing on main cannot turn an in-flight branch red.

Removing a request or an exemption is not judged here: the completeness test
in ``test_payload_budgets.py`` requires the table to match the live routes.

Residual: a commit that reaches main without running verify is not examined.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from dashboard.payload_budgets import (
    TABLE_REPO_PATH,
    BudgetTable,
    EndpointBudget,
    parse_budget_table,
)


def history_violations(base: BudgetTable, head: BudgetTable) -> list[str]:
    """One message per request whose head history does not extend its base history."""
    recorded = {entry.request_key: entry for entry in base.endpoints}
    violations = []
    for entry in head.endpoints:
        if entry.request_key in recorded:
            violation = _first_divergence(recorded[entry.request_key], entry)
            if violation is not None:
                violations.append(violation)
    return violations


def _first_divergence(base: EndpointBudget, head: EndpointBudget) -> str | None:
    for index, recorded in enumerate(base.history):
        if index >= len(head.history):
            return (
                f'{head.url_suffix()}: history[{index}] {recorded} was deleted; '
                f'the merge base records {len(base.history)} entries and this tree '
                f'has {len(head.history)}'
            )
        if head.history[index] != recorded:
            return (
                f'{head.url_suffix()}: history[{index}] was {recorded} at the merge '
                f'base and is {head.history[index]} here'
            )
    return None


@dataclass(frozen=True)
class BaseTable:
    """The table as recorded at the merge base."""

    base_ref: str
    base_commit: str
    table: BudgetTable


@dataclass(frozen=True)
class FirstWrite:
    """The merge base has no table: this branch writes it for the first time."""

    base_ref: str
    base_commit: str


def resolve_base_table(
    repo_root: Path, base_refs: Sequence[str] = ('main', 'origin/main'),
) -> BaseTable | FirstWrite | None:
    """The table at ``merge-base(<ref>, HEAD)``, trying each ref in turn.

    Returns ``None`` when no ref resolves: an sdist or a shallow clone is not
    measurable, and failing there would blame a tree that is not broken, so the
    caller skips visibly. Once a merge base resolves this is a real checkout,
    so any later git failure RAISES with git's stderr rather than degrading.
    """
    resolved = _merge_base(repo_root, base_refs)
    if resolved is None:
        return None
    base_ref, base_commit = resolved
    listed = _git(repo_root, 'ls-tree', '--name-only', base_commit, '--', TABLE_REPO_PATH)
    _raise_on_failure(listed, f'git ls-tree at merge base {base_commit}', repo_root)
    if not listed.stdout.strip():
        return FirstWrite(base_ref=base_ref, base_commit=base_commit)
    shown = _git(repo_root, 'show', f'{base_commit}:{TABLE_REPO_PATH}')
    _raise_on_failure(shown, f'git show at merge base {base_commit}', repo_root)
    table = parse_budget_table(shown.stdout, source=f'{base_commit}:{TABLE_REPO_PATH}')
    return BaseTable(base_ref=base_ref, base_commit=base_commit, table=table)


def _merge_base(repo_root: Path, base_refs: Sequence[str]) -> tuple[str, str] | None:
    for ref in base_refs:
        proc = _git(repo_root, 'merge-base', ref, 'HEAD')
        if proc.returncode == 0 and proc.stdout.strip():
            return ref, proc.stdout.strip()
    return None


def _raise_on_failure(
    proc: subprocess.CompletedProcess[str], what: str, repo_root: Path,
) -> None:
    if proc.returncode != 0:
        raise RuntimeError(
            f'{what} failed in {repo_root} after the merge base had already '
            f'resolved, so this is a broken checkout rather than an unmeasurable '
            f'one: {proc.stderr.strip()}'
        )


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run git in *repo_root*: the verify lane's cwd is ``dashboard/``, not the root."""
    return subprocess.run(
        ['git', *args], cwd=repo_root, capture_output=True, text=True, timeout=60,
    )
