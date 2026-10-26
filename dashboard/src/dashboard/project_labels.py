"""Name each configured project root by a label no other configured root shares.

The rule mirrors ``dashboard/src/dashboard/static/redux/scheduler_utils.jsx::disambiguateLabels``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


def basename_label(project_root: str | os.PathLike[str]) -> str:
    """The root's basename, or the full string for a root with none (``'/'``)."""
    root_str = str(project_root)
    return Path(root_str).name or root_str


def _as_path(project_root: str | os.PathLike[str]) -> Path:
    return Path(os.fspath(project_root))


def _shared_trailing_parts(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    shared = 0
    for left_part, right_part in zip(reversed(left), reversed(right), strict=False):
        if left_part != right_part:
            break
        shared += 1
    return shared


def _shortest_unique_suffix(root: Path, roots: tuple[Path, ...]) -> str:
    most_shared = max(
        (_shared_trailing_parts(root.parts, peer.parts) for peer in roots if peer != root),
        default=0,
    )
    length = 1 + most_shared
    return str(root) if length >= len(root.parts) else '/'.join(root.parts[-length:])


@dataclass(frozen=True, slots=True)
class ProjectLabels:
    """A label for each root it holds, no two roots sharing one."""

    _label_by_root: Mapping[Path, str]
    _root_by_label: Mapping[str, Path] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        root_by_label: dict[str, Path] = {}
        for root, label in self._label_by_root.items():
            claimed_by = root_by_label.setdefault(label, root)
            if claimed_by != root:
                raise ValueError(f'project label {label!r} names two roots: {claimed_by} and {root}')
        object.__setattr__(self, '_root_by_label', MappingProxyType(root_by_label))

    @classmethod
    def of_roots(cls, roots: Iterable[str | os.PathLike[str]]) -> ProjectLabels:
        """Label each of *roots* by its shortest trailing path suffix no other of them shares."""
        unique = tuple(dict.fromkeys(_as_path(root) for root in roots))
        return cls(MappingProxyType({root: _shortest_unique_suffix(root, unique) for root in unique}))

    def including(self, roots: Iterable[str | os.PathLike[str]]) -> ProjectLabels:
        """This table plus *roots*, each added root labelled by its shortest suffix no other root here shares.

        Every label already held is kept, so payloads keyed by it still join. The one
        root this cannot label is a relative path spelling a held label (``'app'``
        beside ``/srv/app``): it stays out, and :meth:`of` gives it its basename.
        """
        held = tuple(self._label_by_root)
        everything = tuple(dict.fromkeys((*held, *(_as_path(root) for root in roots))))
        label_by_root = dict(self._label_by_root)
        for root in everything[len(held):]:
            label = _shortest_unique_suffix(root, everything)
            if label not in self._root_by_label:
                label_by_root[root] = label
        return ProjectLabels(MappingProxyType(label_by_root))

    @property
    def roots(self) -> tuple[Path, ...]:
        """The labelled roots, in first-seen order."""
        return tuple(self._label_by_root)

    def of(self, project_root: str | os.PathLike[str]) -> str:
        """The root's label; a root this table does not hold is labelled by its basename."""
        label = self._label_by_root.get(_as_path(project_root))
        return basename_label(project_root) if label is None else label

    def root_for(self, label: str) -> Path | None:
        """The root carrying *label*, if any."""
        return self._root_by_label.get(label)
