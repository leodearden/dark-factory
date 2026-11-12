"""Resolve a collected test item's pytest-xdist group, so a test can pin a tree scan's consumers to one worker.

The naming rule is xdist's own, from ``xdist/remote.py``
``WorkerInteractor.pytest_collection_modifyitems``: under ``--dist loadgroup``
an item's group is the sorted, ``'_'``-joined set of every ``xdist_group``
mark's name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol

__all__ = ['CollectedItem', 'MarkLike', 'single_xdist_group', 'xdist_group_of']

_XDIST_GROUP_MARK = 'xdist_group'
_DEFAULT_GROUP_NAME = 'default'


class MarkLike(Protocol):
    @property
    def args(self) -> tuple[Any, ...]: ...

    @property
    def kwargs(self) -> Mapping[str, Any]: ...


class CollectedItem(Protocol):
    @property
    def nodeid(self) -> str: ...

    def iter_markers(self, name: str) -> Iterable[MarkLike]: ...


def xdist_group_of(item: CollectedItem) -> str | None:
    names = {
        str(mark.args[0] if mark.args else mark.kwargs.get('name', _DEFAULT_GROUP_NAME))
        for mark in item.iter_markers(_XDIST_GROUP_MARK)
    }
    return '_'.join(sorted(names)) if names else None


def single_xdist_group(items: Iterable[CollectedItem]) -> str:
    resolved = [(item.nodeid, xdist_group_of(item)) for item in items]
    if not resolved:
        raise ValueError('no items to resolve an xdist_group for')
    first_group = resolved[0][1]
    if first_group is not None and all(group == first_group for _, group in resolved):
        return first_group
    listing = '\n'.join(
        f'  {nodeid} -> {group if group is not None else "no xdist_group"}'
        for nodeid, group in resolved
    )
    raise AssertionError(
        'these tree-scan consumers must share one xdist_group so one worker '
        f'pays for the scan:\n{listing}'
    )
