"""The defect-tests collection map archived beside a junit report: each collected test's criterion and defects, by case id."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from client_feed.contract import first_error
from client_feed.feed_io import FeedIoError, read_tail
from client_feed.sources.archive import case_id

COLLECTION_MAP_SUFFIX: Final = '.tests.json'


class MapProblem(StrEnum):
    INVALID_MAP = 'invalid_map'
    UNREADABLE = 'unreadable'


class CollectionMapError(Exception):
    """A collection map that is there but cannot be read as one."""

    def __init__(self, *, problem: MapProblem, name: str, detail: str) -> None:
        self.problem = problem
        self.name = name
        self.detail = detail
        super().__init__(f'{name} refused ({problem}): {detail}')


def collection_map_path(junit: Path) -> Path:
    """The collection map the defect-tests plugin writes beside `junit`; client_feed's copy of orchestrator verify.py::collection_map_path, which it may not import."""
    return junit.with_name(junit.name + COLLECTION_MAP_SUFFIX)


class MapEntry(BaseModel):
    """One collected test's marks: the defects it pins and the criterion it evidences."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    defects: tuple[str, ...]
    criterion: str | None


CollectionMap: TypeAlias = Mapping[str, MapEntry]
_COLLECTION_MAP: Final = TypeAdapter(dict[str, MapEntry])


def read_collection_map(path: Path, *, name: str) -> CollectionMap | None:
    """The collection map at `path`, keyed by nodeid and never read through a symlink; None when the run recorded none."""
    try:
        tail = read_tail(path, offset=0)
    except (FeedIoError, OSError) as exc:
        detail = exc.detail if isinstance(exc, FeedIoError) else exc.strerror or type(exc).__name__
        raise CollectionMapError(problem=MapProblem.UNREADABLE, name=name, detail=detail) from exc
    if tail is None:
        return None
    try:
        return _COLLECTION_MAP.validate_json(tail.data)
    except ValidationError as exc:
        raise CollectionMapError(
            problem=MapProblem.INVALID_MAP, name=name, detail=first_error(exc, whole='<map>')
        ) from exc


def _junit_address(nodeid: str) -> tuple[str, str]:
    """The junit (classname, name) pytest records for `nodeid`; restates _pytest/junitxml.py::mangle_test_address."""
    path, bracket, params = nodeid.partition('[')
    names = path.split('::')
    names[0] = names[0].replace('/', '.').removesuffix('.py')
    names[-1] += bracket + params
    return '.'.join(names[:-1]), names[-1]


def case_entries(collection_map: CollectionMap) -> tuple[tuple[str, MapEntry], ...]:
    """Each collected test's junit case id and entry, in map order; two nodeids can redact to one case id."""
    return tuple(
        (case_id(*_junit_address(nodeid)), entry) for nodeid, entry in collection_map.items()
    )
