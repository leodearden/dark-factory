"""Shared check that an MCP server's live tool surface is fully classified.

Recon stage gating is deny-list only, so every tool a server registers must be
placed in exactly one named bucket — a deny list or a reviewed-safe set — and
every bucket must name only tools the server actually registers.  The policy
lives here once; each server's guard supplies its registered names, its buckets
and the remedy to print.

Lives outside conftest.py, under a module name unique across subprojects, for
the reason ``_ast_guard`` records.
"""

import itertools
from collections.abc import Iterable, Mapping, Set


def assert_tool_surface_classified(
    registered: Set[str],
    buckets: Mapping[str, Iterable[str]],
    *,
    remedy: str,
) -> None:
    """Every registered tool sits in exactly one bucket; every bucket names only registered tools."""
    named = {name: frozenset(tools) for name, tools in buckets.items()}
    unclassified = registered - frozenset().union(*named.values())
    assert not unclassified, (
        f'These tools are registered but in no classification bucket, so every '
        f'recon stage can call them: {sorted(unclassified)}. {remedy}'
    )
    for name, tools in named.items():
        stale = tools - registered
        assert not stale, (
            f'{name} names tools the server does not register: {sorted(stale)}. '
            f'Remove them, or fix the rename.'
        )
    assert_pairwise_disjoint(named)


def assert_pairwise_disjoint(buckets: Mapping[str, Iterable[str]]) -> None:
    """A tool in two buckets is two contradictory decisions, and whichever is read first wins."""
    for (name_a, a), (name_b, b) in itertools.combinations(buckets.items(), 2):
        shared = set(a) & set(b)
        assert not shared, f'{name_a} and {name_b} both classify {sorted(shared)}'
