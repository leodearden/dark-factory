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
import types
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ConfigDict

OVERLAY_ENV_VAR = 'FUSED_MEMORY_CONFIG_OVERLAY'

PROVENANCE_FIELD = 'config_overlay'
"""The FusedMemoryConfig field the loader, and only the loader, sets."""

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


class OverlayRefusalReason(StrEnum):
    FILE_MISSING = 'file_missing'
    PATH_NOT_ABSOLUTE = 'path_not_absolute'
    UNPARSEABLE = 'unparseable'
    NOT_A_MAPPING = 'not_a_mapping'
    EMPTY = 'empty'
    KEY_OUTSIDE_ALLOWLIST = 'key_outside_allowlist'
    UNKNOWN_KEY = 'unknown_key'


@dataclass(frozen=True)
class OverlayRefusal:
    reason: OverlayRefusalReason
    path: str
    keys: tuple[str, ...]
    detail: str


class ConfigOverlayRefused(RuntimeError):
    """The overlay was refused as a whole; the config is never built from the base alone."""

    def __init__(self, refusal: OverlayRefusal) -> None:
        self.refusal = refusal
        super().__init__(
            'Config overlay refused (no-silent-fail-soft: an overlay applies whole '
            'or not at all, and never falls back to the base config): '
            f'reason={refusal.reason.value} path={refusal.path!r} '
            f'keys={sorted(refusal.keys)}. {refusal.detail} '
            f'Fix the file, or unset {OVERLAY_ENV_VAR}.'
        )


def _refused(
    reason: OverlayRefusalReason, path: str, detail: str, keys: tuple[str, ...] = (),
) -> ConfigOverlayRefused:
    return ConfigOverlayRefused(OverlayRefusal(reason, path, keys, detail))


def load_overlay(
    environ: Mapping[str, str], schema: type[BaseModel],
) -> LoadedOverlay | None:
    """Read and validate the overlay ``environ`` names, or None when the variable is unset.

    ``schema`` is the settings class every overlay path must name a field of.
    Raises ConfigOverlayRefused on any defect; nothing is ever partially merged.
    """
    raw_path = environ.get(OVERLAY_ENV_VAR)
    if raw_path is None:
        return None
    if not raw_path or not Path(raw_path).is_absolute():
        raise _refused(
            OverlayRefusalReason.PATH_NOT_ABSOLUTE, raw_path,
            'The overlay path must be absolute, so the effective config cannot '
            'depend on the working directory.',
        )
    mtime, data = _read(raw_path)
    _validate(data, raw_path, schema)
    provenance = ConfigOverlayProvenance(
        path=raw_path,
        merged_keys=_merged_keys(data),
        mtime=datetime.fromtimestamp(mtime, UTC).isoformat(),
    )
    return LoadedOverlay(data=data, provenance=provenance)


def _read(path: str) -> tuple[float, Any]:
    try:
        with open(path, 'rb') as handle:
            mtime = os.fstat(handle.fileno()).st_mtime
            return mtime, yaml.safe_load(handle)
    except FileNotFoundError as exc:
        raise _refused(OverlayRefusalReason.FILE_MISSING, path, str(exc)) from exc
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as exc:
        raise _refused(OverlayRefusalReason.UNPARSEABLE, path, str(exc)) from exc


def _validate(data: Any, path: str, schema: type[BaseModel]) -> None:
    if not isinstance(data, dict):
        raise _refused(
            OverlayRefusalReason.NOT_A_MAPPING, path,
            f'The overlay document must be a mapping of sections; got {type(data).__name__}.',
        )
    not_mappings = tuple(sorted(
        section for section, value in data.items()
        if section in _UNIT_BY_SECTION and not isinstance(value, dict)
    ))
    if not_mappings:
        raise _refused(
            OverlayRefusalReason.NOT_A_MAPPING, path,
            'Each overlay section must be a mapping.', not_mappings,
        )
    outside = _outside_allowlist(data)
    if outside:
        raise _refused(
            OverlayRefusalReason.KEY_OUTSIDE_ALLOWLIST, path,
            'Only the OVERLAY_ALLOWLIST units may be overlaid: '
            f'{_describe_allowlist()}.', outside,
        )
    if not _merged_keys(data):
        raise _refused(OverlayRefusalReason.EMPTY, path, 'The overlay sets no key.')
    unknown = tuple(sorted(_unknown_paths(schema, data, prefix='')))
    if unknown:
        raise _refused(
            OverlayRefusalReason.UNKNOWN_KEY, path,
            'These paths name no config field, so they would be silently ignored.',
            unknown,
        )


def _outside_allowlist(data: Mapping[str, Any]) -> tuple[str, ...]:
    offending: list[str] = []
    for section, section_overlay in data.items():
        unit = _UNIT_BY_SECTION.get(section)
        if unit is None:
            offending.append(section)
        elif unit.keys is not None:
            offending.extend(f'{section}.{key}' for key in section_overlay if key not in unit.keys)
    return tuple(sorted(offending))


def _describe_allowlist() -> str:
    return ', '.join(
        unit.section if unit.keys is None else f'{unit.section}.{{{", ".join(sorted(unit.keys))}}}'
        for unit in OVERLAY_ALLOWLIST
    )


def _unknown_paths(model: type[BaseModel], data: Mapping[str, Any], prefix: str) -> list[str]:
    unknown: list[str] = []
    for key, value in data.items():
        path = f'{prefix}{key}'
        field = model.model_fields.get(key)
        if field is None:
            unknown.append(path)
        else:
            unknown.extend(_unknown_in_value(field.annotation, value, path))
    return unknown


def _unknown_in_value(annotation: Any, value: Any, path: str) -> list[str]:
    """Descend nested models and ``dict[str, Model]`` values; any other type is a leaf."""
    annotation = _without_none(annotation)
    if not isinstance(value, Mapping):
        return []
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _unknown_paths(annotation, value, prefix=f'{path}.')
    if get_origin(annotation) is dict:
        entry_annotation = get_args(annotation)[1]
        return [
            unknown
            for entry_key, entry in value.items()
            for unknown in _unknown_in_value(entry_annotation, entry, f'{path}.{entry_key}')
        ]
    return []


def _without_none(annotation: Any) -> Any:
    if get_origin(annotation) in (Union, types.UnionType):
        members = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(members) == 1:
            return members[0]
    return annotation


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
    match unit.merge, unit.keys:
        case MergeMode.PER_KEY, _:
            return _deep_merge(base_section, section_overlay)
        case MergeMode.REPLACE_UNIT, None:
            return section_overlay
        case MergeMode.REPLACE_UNIT, unit_keys:
            kept = {k: v for k, v in base_section.items() if k not in unit_keys}
            return kept | section_overlay
        case MergeMode.REPLACE_EACH_ENTRY, _:
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
