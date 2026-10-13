"""Expand stored truncated-uuid citations to the full ids they name.

PRD contract C6, ``plans/uuid-prefix-resolution-prd.md`` §4-C6, which owns the
lane table and the rules; this docstring does not restate them (INV-9). Every
token comes from ``shared.uuid_prefix`` (C1) and every answer from the
``PrefixResolver`` port (C2), resolved against the record's own project. Only
``unique`` tokens are expanded; ``ambiguous`` ones are reported by record and
left byte-identical, ``none`` ones are counted and left alone.

Run it one project per sitting, dry run first (the default), then one lane at
a time::

    uv run --project fused-memory python scripts/sweep_uuid_prefixes.py \\
        --project <id> [--project-root <checkout>] [--lane all]
    uv run --project fused-memory python scripts/sweep_uuid_prefixes.py \\
        --project <id> [--project-root <checkout>] --apply --lane <lane>

A second ``--apply`` of the same lane must report ``0 expandable remaining``
and exit 0.
"""
from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

# Bind `shared` and `fused_memory` to THIS checkout, the
# scripts/check_consolidation_closure.py idiom: the editable installs are
# ordinary .pth entries, so sys.path order decides which copy a worktree run
# imports.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _src in (_REPO_ROOT / 'fused-memory' / 'src', _REPO_ROOT / 'shared' / 'src'):
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from shared.uuid_prefix import PrefixToken, find_prefix_tokens, substitute_all  # noqa: E402
from shared.uuid_prefix_guard import (  # noqa: E402
    PrefixResolver,
    Resolution,
    ResolverUnavailable,
)

# ---------------------------------------------------------------------------
# The planner: one record, its tokens' resolutions, what to write.
# ---------------------------------------------------------------------------


class AmbiguousSite(NamedTuple):
    """A token that names more than one id, so the sweep must not choose."""

    path: tuple[str | int, ...]
    token: str
    candidates: tuple[str, ...]


class RecordPlan(NamedTuple):
    """What one record holds, and the document to write when anything expands.

    ``after`` is ``None`` exactly when ``expansions`` is empty, so "nothing to
    write" is never encoded as an unchanged copy.
    """

    expansions: tuple[tuple[PrefixToken, str], ...]
    ambiguous: tuple[AmbiguousSite, ...]
    #: Unique tokens under a field the lane's write path cannot reach.
    unwritable: tuple[PrefixToken, ...]
    untouched: int
    after: dict[str, Any] | None


def plan_record(
    fields: Mapping[str, Any],
    resolutions: Mapping[str, Resolution],
    unwritable_fields: frozenset[str],
) -> RecordPlan:
    """Classify every token in *fields* and build the expanded document.

    Pure. A token with no entry in *resolutions* raises ``KeyError``: every
    token was meant to be resolved first, and silently treating a missing
    answer as ``none`` would leave a live citation unexpanded (INV-11).
    """
    expansions: list[tuple[PrefixToken, str]] = []
    ambiguous: list[AmbiguousSite] = []
    unwritable: list[PrefixToken] = []
    untouched = 0
    for token in find_prefix_tokens(dict(fields)):
        resolution = resolutions[token.token]
        if resolution.outcome == 'unique':
            if token.path[0] in unwritable_fields:
                unwritable.append(token)
            else:
                expansions.append((token, resolution.candidates[0].id))
        elif resolution.outcome == 'ambiguous':
            ambiguous.append(
                AmbiguousSite(
                    path=token.path,
                    token=token.token,
                    candidates=tuple(candidate.id for candidate in resolution.candidates),
                )
            )
        else:
            untouched += 1
    return RecordPlan(
        expansions=tuple(expansions),
        ambiguous=tuple(ambiguous),
        unwritable=tuple(unwritable),
        untouched=untouched,
        after=substitute_all(fields, expansions) if expansions else None,
    )


# ---------------------------------------------------------------------------
# Resolution: once per distinct token, a bounded number at a time.
# ---------------------------------------------------------------------------

#: Resolutions in flight at once. Each one walks the project's whole id space
#: (D9); the plan's two-phase resolution decision sets this bound. Distinct
#: from ``uuid_prefix_guard.MAX_CONCURRENT_RESOLUTIONS``, which is a per-call
#: latency policy for the MCP guard, not a corpus-walk load policy.
RESOLVE_CONCURRENCY = 4


async def resolve_distinct(
    project_id: str,
    tokens: Iterable[str],
    resolve: PrefixResolver,
    memo: Mapping[str, Resolution],
    *,
    concurrency: int = RESOLVE_CONCURRENCY,
) -> dict[str, Resolution]:
    """*memo* extended with a resolution for every token it lacks.

    Each distinct token is resolved exactly once. *memo* is never mutated. A
    store outage on any token raises ``ResolverUnavailable`` itself, after the
    task group has cancelled every other resolution still in flight.
    """
    pending = [token for token in dict.fromkeys(tokens) if token not in memo]
    resolved: dict[str, Resolution] = {}
    slots = asyncio.Semaphore(concurrency)

    async def resolve_one(token: str) -> None:
        async with slots:
            resolved[token] = await resolve(project_id, token)

    try:
        async with asyncio.TaskGroup() as group:
            for token in pending:
                group.create_task(resolve_one(token))
    except* ResolverUnavailable as outage:
        raise outage.exceptions[0] from None
    return {**memo, **resolved}
