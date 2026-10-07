"""The write-boundary contract for Stage-1 flag-family Mem0 records.

Task 4863 (#3919). One place defines, per flag-family record kind, which
metadata keys name the kind and which flag-type field spelling is canonical,
how a non-canonical spelling is reconciled (lossless conversions normalize,
lossy or contradictory input raises :class:`FlagRecordSchemaError`), and which
kinds a reconciliation stage may not write at all.

Enforcement points, all of which call this module:

- ``fused_memory/server/tools.py::add_memory`` and ``::add_system_record``
  return the refusal as a structured tool result.
- ``fused_memory/services/memory_service.py::MemoryService.add_memory`` and
  ``::MemoryService.add_system_record`` refuse and normalize every write.
- ``fused_memory/reconciliation/flag_dedup.py`` builds its marker and
  suppression payloads through the same normalizer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final, Literal

__all__ = [
    'FLAG_RECORD_SHAPES',
    'RECON_STAGE_FLAG_KIND_REFUSALS',
    'STAGE1_FLAG_KIND',
    'STAGE1_FLAG_MARKER_KIND',
    'STAGE1_FLAG_SUPPRESSION_KIND',
    'FlagKindRefusal',
    'FlagRecordSchemaError',
    'FlagRecordShape',
    'ReconStageFlagRecordWriteRefused',
    'canonical_flag_types',
    'enforce_flag_record_write_contract',
    'flag_record_kinds',
    'normalize_flag_record_metadata',
    'recon_stage_flag_kind_refusal',
]

STAGE1_FLAG_MARKER_KIND: Final = 'stage1_flag_marker'
STAGE1_FLAG_KIND: Final = 'stage1_flag'
STAGE1_FLAG_SUPPRESSION_KIND: Final = 'stage1_flag_suppression'

_RECON_STAGE_AGENT_PREFIX: Final = 'recon-stage-'

FlagTypeField = Literal['flag_type', 'flag_types']


@dataclass(frozen=True)
class FlagRecordShape:
    kind_keys: tuple[str, ...]
    flag_type_field: FlagTypeField


FLAG_RECORD_SHAPES: Final[Mapping[str, FlagRecordShape]] = MappingProxyType({
    STAGE1_FLAG_MARKER_KIND: FlagRecordShape(('kind', 'source'), 'flag_type'),
    STAGE1_FLAG_KIND: FlagRecordShape(('kind',), 'flag_type'),
    STAGE1_FLAG_SUPPRESSION_KIND: FlagRecordShape(('kind',), 'flag_types'),
})


@dataclass(frozen=True)
class FlagKindRefusal:
    error: str
    error_type: str
    hint: str


RECON_STAGE_FLAG_KIND_REFUSALS: Final[Mapping[str, FlagKindRefusal]] = MappingProxyType({
    STAGE1_FLAG_MARKER_KIND: FlagKindRefusal(
        error='flag_marker_write_blocked',
        error_type='ReconFlagMarkerWriteRejected',
        hint=(
            'stage1_flag_marker persistence is code-managed via the recon_ledger; '
            'add_memory and add_system_record are not valid write paths for it'
        ),
    ),
    STAGE1_FLAG_SUPPRESSION_KIND: FlagKindRefusal(
        error='flag_suppression_write_blocked',
        error_type='ReconFlagSuppressionWriteRejected',
        hint=(
            'stage1_flag_suppression records are operator-managed recon_ledger rows; '
            'a recon stage cannot create one, and a Mem0 record of this kind has no '
            'gate effect. Report the recurring finding (task_id + flag_type) in your '
            'cycle report so an operator can decide whether to suppress it'
        ),
    ),
})


class FlagRecordSchemaError(ValueError):
    """A flag-family record's metadata cannot be normalized without loss or contradiction."""


class ReconStageFlagRecordWriteRefused(PermissionError):
    def __init__(self, refusal: FlagKindRefusal, agent_id: str) -> None:
        self.refusal = refusal
        self.agent_id = agent_id
        super().__init__(
            f'{refusal.error_type}: {refusal.error} (agent_id={agent_id}) — {refusal.hint}'
        )


def flag_record_kinds(metadata: object) -> frozenset[str]:
    """Return the flag-family kinds *metadata* names through any of that kind's kind keys."""
    if not isinstance(metadata, Mapping):
        return frozenset()
    return frozenset(
        kind
        for kind, shape in FLAG_RECORD_SHAPES.items()
        if any(metadata.get(key) == kind for key in shape.kind_keys)
    )


def canonical_flag_types(values: Iterable[object]) -> list[str]:
    return sorted({str(value) for value in values})


def normalize_flag_record_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return a new dict holding *metadata* in its kind's canonical flag-record shape."""
    normalized = dict(metadata)
    kinds = flag_record_kinds(normalized)
    if not kinds:
        return normalized
    if len(kinds) > 1:
        raise FlagRecordSchemaError(
            'a flag-family record must name exactly one record kind; '
            f'metadata names {sorted(kinds)!r}'
        )
    (kind,) = kinds
    shape = FLAG_RECORD_SHAPES[kind]
    _reconcile_kind_keys(normalized, kind, shape.kind_keys)
    _reconcile_flag_type_field(normalized, kind, shape.flag_type_field)
    return normalized


def _reconcile_kind_keys(metadata: dict[str, Any], kind: str, kind_keys: tuple[str, ...]) -> None:
    for key in kind_keys:
        value = metadata.setdefault(key, kind)
        if value != kind:
            raise FlagRecordSchemaError(
                f'every record-kind key of a {kind!r} record must name {kind!r}; '
                f'metadata.{key} is {value!r}'
            )


def _reconcile_flag_type_field(
    metadata: dict[str, Any], kind: str, canonical_field: FlagTypeField
) -> None:
    has_singular = 'flag_type' in metadata
    has_plural = 'flag_types' in metadata
    singular = metadata.get('flag_type')
    plural = canonical_flag_types(_as_flag_type_list(metadata.get('flag_types')))
    singular_as_list = [str(singular)] if singular not in (None, '') else []
    if has_singular and has_plural and singular_as_list != plural:
        raise FlagRecordSchemaError(
            f'flag_type and flag_types of a {kind!r} record must agree; '
            f'flag_type={singular!r}, flag_types={plural!r}'
        )
    flag_types = plural if has_plural else singular_as_list
    if canonical_field == 'flag_types':
        metadata.pop('flag_type', None)
        if flag_types:
            metadata['flag_types'] = flag_types
        else:
            metadata.pop('flag_types', None)
        return
    if len(flag_types) > 1:
        raise FlagRecordSchemaError(
            f'a {kind!r} record carries exactly one flag_type; '
            f'flag_types={flag_types!r} cannot be narrowed without loss'
        )
    metadata.pop('flag_types', None)
    if not has_singular and flag_types:
        metadata['flag_type'] = flag_types[0]


def _as_flag_type_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def recon_stage_flag_kind_refusal(agent_id: object, metadata: object) -> FlagKindRefusal | None:
    """Return the refusal for a recon-stage write naming a stage-forbidden flag kind, else None."""
    if not (isinstance(agent_id, str) and agent_id.startswith(_RECON_STAGE_AGENT_PREFIX)):
        return None
    for kind in sorted(flag_record_kinds(metadata)):
        refusal = RECON_STAGE_FLAG_KIND_REFUSALS.get(kind)
        if refusal is not None:
            return refusal
    return None


def enforce_flag_record_write_contract(
    metadata: Mapping[str, Any] | None, *, agent_id: str | None
) -> dict[str, Any]:
    """Refuse a stage-forbidden recon-stage write, else return the normalized metadata copy."""
    refusal = recon_stage_flag_kind_refusal(agent_id, metadata)
    if refusal is not None:
        raise ReconStageFlagRecordWriteRefused(refusal, str(agent_id))
    return normalize_flag_record_metadata(metadata or {})
