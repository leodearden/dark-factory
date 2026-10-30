"""Owns the persisted eval cell: its type, its on-disk name and its directory."""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shared.safe_io import atomic_write_text

from .measurement import MeasurementStage, UnmeasuredCause, UnmeasuredRecord

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

    @classmethod
    def from_path(cls, path: Path) -> CellName:
        """The inverse of :attr:`filename`; a name outside the grammar raises ValueError."""
        segments = path.stem.split(_SEP)
        if path.suffix != _SUFFIX or len(segments) != 3:
            raise ValueError(f'{path} is not a cell name of the form {_GRAMMAR}')
        try:
            return cls(*segments)
        except ValueError as e:
            raise ValueError(f'{path} is not a cell name: {e}') from e


class ResultLoadError(Exception):
    """A persisted cell (or its directory) could not be read."""

    def __init__(self, path: Path, detail: str) -> None:
        super().__init__(f'{path}: {detail}')
        self.path = path
        self.detail = detail


class MissingResultsDirError(ResultLoadError):
    """The results directory to load from does not exist."""


_STAGES = frozenset(MeasurementStage)
_CAUSES = frozenset(UnmeasuredCause)


def _unclassified(reason: str) -> UnmeasuredRecord:
    return {
        'stage': MeasurementStage.ARCHITECT.value,
        'cause': UnmeasuredCause.LEGACY_UNCLASSIFIED.value,
        'reason': reason,
    }


def _record_from_marker(marker: str) -> UnmeasuredRecord:
    first = marker.split(';')[0].strip()
    stage_text, sep, rest = first.partition(':')
    cause_text = rest.strip().partition(':')[0].strip() if sep else first
    if stage_text not in _STAGES or cause_text not in _CAUSES:
        return _unclassified(marker)
    parts = marker.split(':', 2)
    reason = parts[2].removeprefix(' ') if len(parts) == 3 else ''
    return {'stage': stage_text, 'cause': cause_text, 'reason': reason}


def _invocation_errors_from_marker(metrics: dict[str, Any]) -> dict[str, Any]:
    """A pre-6734 ``invocation_error`` string as typed records (PRD §7 D3)."""
    is_legacy = 'invocation_errors' not in metrics and (
        'invocation_error' in metrics or metrics.get('cap_tainted')
    )
    if not is_legacy:
        return dict(metrics)
    normalized = dict(metrics)
    marker = normalized.pop('invocation_error', None)
    if marker:
        normalized['invocation_errors'] = [_record_from_marker(marker)]
    elif normalized.get('cap_tainted'):
        normalized['invocation_errors'] = [_unclassified('')]
    else:
        normalized['invocation_errors'] = None
    return normalized


LEGACY_METRIC_RULES: tuple[Callable[[dict[str, Any]], dict[str, Any]], ...] = (
    _invocation_errors_from_marker,
)


def normalize_legacy_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(metrics)
    for rule in LEGACY_METRIC_RULES:
        normalized = rule(normalized)
    return normalized


_EVAL_RESULT_FIELDS = frozenset(f.name for f in dataclasses.fields(EvalResult))


def _load_cell(path: Path) -> EvalResult:
    data = json.loads(path.read_text())
    result = EvalResult(**{k: v for k, v in data.items() if k in _EVAL_RESULT_FIELDS})
    return dataclasses.replace(result, metrics=normalize_legacy_metrics(result.metrics))


def load_result(path: Path) -> EvalResult:
    """The cell persisted at *path*; one that cannot be read raises ResultLoadError naming it."""
    try:
        return _load_cell(path)
    except (ValueError, OSError, TypeError, AttributeError) as e:
        raise ResultLoadError(path, f'{type(e).__name__}: {e}') from e


def _read_raw_cell(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except (ValueError, OSError) as e:
        raise ResultLoadError(path, f'{type(e).__name__}: {e}') from e
    if not isinstance(data, dict):
        raise ResultLoadError(path, f'not a JSON object but {type(data).__name__}')
    return data


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

    def paths_matching(self, task_id: str, config_name: str) -> list[Path]:
        matches = []
        for path in self.cell_paths():
            try:
                name = CellName.from_path(path)
            except ValueError:
                continue
            if (name.task_id, name.config_name) == (task_id, config_name):
                matches.append(path)
        return matches

    def exists(self, task_id: str, config_name: str) -> bool:
        return bool(self.paths_matching(task_id, config_name))

    def rewrite(
        self, path: Path, change: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> bool:
        """Apply *change* to the raw JSON of one of this store's cells, atomically.

        The raw object is handed over, so keys :class:`EvalResult` does not
        declare survive. A change that leaves the cell equal writes nothing, so
        the file's mtime still says when its content last changed. Returns
        whether the cell was written.
        """
        if path.parent != self.directory:
            raise ValueError(f'{path} is not a cell of the store at {self.directory}')
        original = _read_raw_cell(path)
        changed = change(copy.deepcopy(original))
        if changed == original:
            return False
        atomic_write_text(path, json.dumps(changed, indent=2))
        logger.info(f'Rewrote result: {path}')
        return True

    def load_all(self) -> list[EvalResult]:
        """Every top-level cell, whatever its name.

        A missing dir or an unreadable cell raises naming it, never an empty or
        shortened list: either would read as a real, smaller measurement.
        """
        if not self.directory.is_dir():
            raise MissingResultsDirError(
                self.directory, 'results dir not found (or is not a directory)'
            )
        return [load_result(path) for path in self.cell_paths()]
