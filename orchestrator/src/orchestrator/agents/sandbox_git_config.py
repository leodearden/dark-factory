"""The git configuration every process run under the agent sandbox gets.

rerere records resolutions into the common dir's ``rr-cache``, which
``orchestrator/src/orchestrator/agents/write_set.py::compute_write_set`` leaves
read-only, so a half-on rerere leaves a dangling MERGE_RR and a stale
MERGE_RR.lock. Rationale: ``docs/shared-repo-git-maintenance.md``.
"""
from __future__ import annotations

from collections.abc import Mapping

SANDBOX_GIT_CONFIG: tuple[tuple[str, str], ...] = (
    ('rerere.enabled', 'false'),
    ('rerere.autoupdate', 'false'),
)


def _inherited_count(inherited: Mapping[str, str]) -> int:
    raw = inherited.get('GIT_CONFIG_COUNT')
    if raw is None:
        return 0
    if not (raw.isascii() and raw.isdecimal()):
        raise ValueError(
            f'inherited GIT_CONFIG_COUNT={raw!r} is not a non-negative integer'
        )
    return int(raw)


def sandbox_git_config_env(inherited: Mapping[str, str]) -> dict[str, str]:
    """The GIT_CONFIG_* entries appending SANDBOX_GIT_CONFIG after *inherited*'s own."""
    base = _inherited_count(inherited)
    entries = {'GIT_CONFIG_COUNT': str(base + len(SANDBOX_GIT_CONFIG))}
    for offset, (key, value) in enumerate(SANDBOX_GIT_CONFIG):
        entries[f'GIT_CONFIG_KEY_{base + offset}'] = key
        entries[f'GIT_CONFIG_VALUE_{base + offset}'] = value
    return entries
