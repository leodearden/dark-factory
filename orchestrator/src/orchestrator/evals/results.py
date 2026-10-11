"""Owns the persisted eval cell: its type, its on-disk name and its directory."""

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shared.safe_io import atomic_write_text

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / 'results'

_SEP = '__'
_SUFFIX = '.json'
_GRAMMAR = f'<task_id>{_SEP}<config_name>{_SEP}<run_id>{_SUFFIX}'


@dataclass
class EvalResult:
    """Result of one (task, config) eval run."""

    task_id: str
    config_name: str
    outcome: str
    metrics: dict[str, Any]
    worktree_path: str
    wall_clock_ms: int = 0
    run_id: str = ''
    trial: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_component(value: str) -> bool:
    return not (
        _SEP in value or '/' in value or value.startswith('_') or value.endswith('_')
    )


@dataclass(frozen=True)
class CellName:
    """The address of one cell; every component round-trips through the filename."""

    task_id: str
    config_name: str
    run_id: str

    def __post_init__(self) -> None:
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not _is_component(value):
                raise ValueError(
                    f'CellName.{field.name}={value!r} may not contain {_SEP!r} '
                    f'or "/", nor begin or end with "_": the cell would not '
                    f'round-trip through {_GRAMMAR}'
                )

    @property
    def filename(self) -> str:
        return f'{self.task_id}{_SEP}{self.config_name}{_SEP}{self.run_id}{_SUFFIX}'


class ResultLoadError(Exception):
    """A persisted cell (or its directory) could not be read."""

    def __init__(self, path: Path, detail: str) -> None:
        super().__init__(f'{path}: {detail}')
        self.path = path
        self.detail = detail


class MissingResultsDirError(ResultLoadError):
    """The results directory to load from does not exist."""


LEGACY_METRIC_RULES: tuple[Callable[[dict[str, Any]], dict[str, Any]], ...] = ()


def normalize_legacy_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(metrics)
    for rule in LEGACY_METRIC_RULES:
        normalized = rule(normalized)
    return normalized


def _split_name(path: Path) -> CellName | None:
    if path.suffix != _SUFFIX:
        return None
    segments = path.stem.split(_SEP)
    if len(segments) != 3 or not all(map(_is_component, segments)):
        return None
    return CellName(*segments)


_EVAL_RESULT_FIELDS = frozenset(f.name for f in dataclasses.fields(EvalResult))


def _load_cell(path: Path) -> EvalResult:
    data = json.loads(path.read_text())
    result = EvalResult(**{k: v for k, v in data.items() if k in _EVAL_RESULT_FIELDS})
    return dataclasses.replace(result, metrics=normalize_legacy_metrics(result.metrics))


@dataclass(frozen=True)
class ResultStore:
    """A directory of persisted eval cells, one JSON file per cell."""

    directory: Path

    @classmethod
    def packaged(cls) -> ResultStore:
        return cls(RESULTS_DIR)

    def path_for(self, result: EvalResult) -> Path:
        name = CellName(result.task_id, result.config_name, result.run_id)
        return self.directory / name.filename

    def save(self, result: EvalResult) -> Path:
        path = self.path_for(result)
        atomic_write_text(path, json.dumps(result.to_dict(), indent=2), mkdir=True)
        logger.info(f'Saved result: {path}')
        return path

    def cell_paths(self) -> list[Path]:
        if not self.directory.is_dir():
            return []
        return sorted(p for p in self.directory.glob(f'*{_SUFFIX}') if p.is_file())

    @staticmethod
    def parse_name(path: Path) -> CellName:
        name = _split_name(path)
        if name is None:
            raise ValueError(f'{path} is not a cell name of the form {_GRAMMAR}')
        return name

    def paths_matching(self, task_id: str, config_name: str) -> list[Path]:
        return [
            path
            for path in self.cell_paths()
            if (name := _split_name(path)) is not None
            and (name.task_id, name.config_name) == (task_id, config_name)
        ]

    def exists(self, task_id: str, config_name: str) -> bool:
        return bool(self.paths_matching(task_id, config_name))

    def load_all(self) -> list[EvalResult]:
        """Every top-level cell, whatever its name.

        A missing dir or an unreadable cell raises naming it, never an empty or
        shortened list: either would read as a real, smaller measurement.
        """
        if not self.directory.is_dir():
            raise MissingResultsDirError(
                self.directory, 'results dir not found (or is not a directory)'
            )
        results = []
        for path in self.cell_paths():
            try:
                results.append(_load_cell(path))
            except (ValueError, OSError, TypeError, AttributeError) as e:
                raise ResultLoadError(path, f'{type(e).__name__}: {e}') from e
        return results
