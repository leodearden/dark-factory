"""The config overlay (D14): one untracked YAML file merged over the tracked base.

``OVERLAY_ENV_VAR`` names the file by absolute path. Only the sections in
``OVERLAY_ALLOWLIST`` may appear in it, each merged by its unit's
``MergeMode``. ``fused_memory.config.schema::YamlSettingsSource.__call__``
applies it after the base is loaded, so every ``FusedMemoryConfig()`` sees
the same effective config and the env layer still outranks it.

A leaf module: it imports no fused_memory module. The settings class used to
walk the schema is passed in by the caller.
"""

import copy
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

OVERLAY_ENV_VAR = 'FUSED_MEMORY_CONFIG_OVERLAY'

WRITE_TRIAGE_BAND_KEYS: frozenset[str] = frozenset({
    't_high', 't_low', 't_high_by_category', 'calibration_report_path',
})


class MergeMode(StrEnum):
    PER_KEY = 'per_key'
    REPLACE_UNIT = 'replace_unit'
    REPLACE_EACH_ENTRY = 'replace_each_entry'


@dataclass(frozen=True)
class OverlayUnit:
    """One overlayable value group: a top-level section, optionally narrowed to ``keys``.

    ``keys=None`` means any key of the section belongs to the unit.
    """

    section: str
    keys: frozenset[str] | None
    merge: MergeMode


OVERLAY_ALLOWLIST: tuple[OverlayUnit, ...] = (
    OverlayUnit('embedder', None, MergeMode.PER_KEY),
    OverlayUnit('embedder_thresholds', None, MergeMode.REPLACE_UNIT),
    OverlayUnit('write_triage', WRITE_TRIAGE_BAND_KEYS, MergeMode.REPLACE_UNIT),
    OverlayUnit('write_triage_project_calibrations', None, MergeMode.REPLACE_EACH_ENTRY),
)

_UNIT_BY_SECTION: dict[str, OverlayUnit] = {unit.section: unit for unit in OVERLAY_ALLOWLIST}


class ConfigOverlayProvenance(BaseModel):
    """Which overlay file was merged, what it set, and the file's mtime (ISO-8601 UTC)."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    path: str
    merged_keys: tuple[str, ...]
    mtime: str


@dataclass(frozen=True)
class LoadedOverlay:
    data: Mapping[str, Any]
    provenance: ConfigOverlayProvenance


def load_overlay(
    environ: Mapping[str, str], schema: type[BaseModel],
) -> LoadedOverlay | None:
    """Read the overlay ``environ`` names, or None when the variable is unset."""
    raw_path = environ.get(OVERLAY_ENV_VAR)
    if raw_path is None:
        return None
    path = Path(raw_path)
    with path.open('rb') as handle:
        mtime = os.fstat(handle.fileno()).st_mtime
        data = yaml.safe_load(handle)
    provenance = ConfigOverlayProvenance(
        path=str(path),
        merged_keys=_merged_keys(data),
        mtime=datetime.fromtimestamp(mtime, UTC).isoformat(),
    )
    return LoadedOverlay(data=data, provenance=provenance)


def merge_overlay(base: Mapping[str, Any], overlay_data: Mapping[str, Any]) -> dict[str, Any]:
    """``base`` with each overlay section merged by its unit's mode; neither input is mutated."""
    merged = copy.deepcopy(dict(base))
    for section, section_overlay in copy.deepcopy(dict(overlay_data)).items():
        unit = _unit_for(section)
        base_section = merged.get(section)
        base_section = base_section if isinstance(base_section, dict) else {}
        merged[section] = _merge_section(unit, base_section, section_overlay)
    return merged


def _unit_for(section: str) -> OverlayUnit:
    unit = _UNIT_BY_SECTION.get(section)
    if unit is None:
        raise ValueError(
            f'{section!r} is not an overlayable section; '
            f'OVERLAY_ALLOWLIST holds {sorted(_UNIT_BY_SECTION)}',
        )
    return unit


def _merge_section(
    unit: OverlayUnit, base_section: dict[str, Any], section_overlay: dict[str, Any],
) -> dict[str, Any]:
    match unit.merge:
        case MergeMode.PER_KEY:
            return _deep_merge(base_section, section_overlay)
        case MergeMode.REPLACE_UNIT if unit.keys is None:
            return section_overlay
        case MergeMode.REPLACE_UNIT:
            kept = {k: v for k, v in base_section.items() if k not in unit.keys}
            return kept | section_overlay
        case MergeMode.REPLACE_EACH_ENTRY:
            return base_section | section_overlay


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _merged_keys(data: Mapping[str, Any]) -> tuple[str, ...]:
    """The dotted paths the overlay sets: leaves at full depth for a PER_KEY
    section, ``section.key`` for every other unit."""
    paths: list[str] = []
    for section, section_overlay in data.items():
        if _unit_for(section).merge is MergeMode.PER_KEY:
            paths.extend(_leaf_paths(section, section_overlay))
        else:
            paths.extend(f'{section}.{key}' for key in section_overlay)
    return tuple(sorted(paths))


def _leaf_paths(prefix: str, value: Any) -> list[str]:
    if not isinstance(value, dict):
        return [prefix]
    return [
        path
        for key, child in value.items()
        for path in _leaf_paths(f'{prefix}.{key}', child)
    ]
