"""The single reading of a systemd unit's LoadState: is its unit file installed and usable?

Stdlib-only and free of import side effects, so any probe under scripts/ can consume it.
"""

from __future__ import annotations

import enum
from typing import Final

__all__ = ['LoadVerdict', 'load_verdict']

_NOT_INSTALLED_LOAD_STATES: Final = frozenset({'', 'not-found'})


class LoadVerdict(enum.Enum):
    LOADED = 'loaded'
    NOT_INSTALLED = 'not-installed'
    MASKED = 'masked'
    NOT_LOADED = 'not-loaded'


def load_verdict(load_state: str) -> LoadVerdict:
    """NOT_LOADED is every LoadState systemd answers for a unit file it found but rejected."""
    if load_state == 'loaded':
        return LoadVerdict.LOADED
    if load_state in _NOT_INSTALLED_LOAD_STATES:
        return LoadVerdict.NOT_INSTALLED
    if load_state == 'masked':
        return LoadVerdict.MASKED
    return LoadVerdict.NOT_LOADED
