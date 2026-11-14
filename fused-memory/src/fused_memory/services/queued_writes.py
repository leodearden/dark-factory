"""The durable write queue's vocabulary: one frozen value per queued operation.

A queue row stores ``(group_id, operation, payload)``. ``operation`` is a
:class:`QueueOperation` value and ``payload`` is the value's :meth:`to_json`,
which keeps the flat key names producers wrote before these types existed, so
an older build can still drain a newer row. :func:`decode_queued_write` turns a
row back into a value once, at the queue boundary; a row it cannot decode
raises :class:`QueueDecodeError` and is never executed.

Contract: plans/review-all-s26-memory-service-write-path-and-graph-client-prd.md
C1, D1, D2.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, ClassVar, Self, TypeGuard, get_args

from fused_memory.utils.canonical_labels import Referent
from fused_memory.utils.referent_resolution import (
    REFERENT_SOURCES,
    ReferentResolution,
    ReferentSet,
    ReferentSource,
)
from fused_memory.utils.validation import _safe_repr

logger = logging.getLogger(__name__)


class QueueOperation(StrEnum):
    """What a queue row asks the executor to do; write-journal operation strings are a separate vocabulary."""

    ADD_EPISODE = 'add_episode'
    ADD_MEMORY_GRAPHITI = 'add_memory_graphiti'
    MEM0_CLASSIFY_AND_ADD = 'mem0_classify_and_add'
    MEM0_ADD = 'mem0_add'  # decode-only: no producer writes it any more


class QueueCallback(StrEnum):
    DUAL_WRITE_EPISODE = 'dual_write_episode'
    REFRESH_ENTITY_SUMMARIES = 'refresh_entity_summaries'


class QueueDecodeError(ValueError):
    """A queue row that cannot become a value; the queue dead-letters it on its first attempt."""

    def __init__(self, reason: str, *, operation: str, row_id: int, raw_payload: str) -> None:
        super().__init__(f'row {row_id} ({operation!r}): {reason}')
        self.reason = reason
        self.operation = operation
        self.row_id = row_id
        self.raw_payload = raw_payload


class UnknownQueueOperation(QueueDecodeError):
    """A row whose ``operation`` is not a :class:`QueueOperation` value."""


def _required_str(body: Mapping[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str):
        raise ValueError(f'{key!r} must be a string, got {type(value).__name__}')
    return value


def _optional_str(body: Mapping[str, Any], key: str) -> str | None:
    value = body.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f'{key!r} must be a string or null, got {type(value).__name__}')
    return value


def _present_strs(body: Mapping[str, Any], *keys: str) -> dict[str, str]:
    """The string *keys* *body* carries; an absent or null key is left to the field's default."""
    return {key: value for key in keys if (value := _optional_str(body, key)) is not None}


def _present_flag(body: Mapping[str, Any], key: str) -> dict[str, bool]:
    value = body.get(key)
    return {} if value is None else {key: bool(value)}


def _load_body(text: str) -> dict[str, Any]:
    body = json.loads(text)
    if not isinstance(body, dict):
        raise ValueError(f'payload must be a JSON object, got {type(body).__name__}')
    return body


@dataclass(frozen=True, kw_only=True)
class WriteEnvelope:
    """Who wrote a queued value and which journal rows it belongs to; the same six wire keys on every operation."""

    project_id: str
    group_id: str
    agent_id: str | None = None
    session_id: str | None = None
    causation_id: str | None = None
    write_op_id: str | None = None

    def to_wire(self) -> dict[str, str | None]:
        return {
            'project_id': self.project_id,
            'group_id': self.group_id,
            'agent_id': self.agent_id,
            'session_id': self.session_id,
            '_causation_id': self.causation_id,
            '_write_op_id': self.write_op_id,
        }

    @classmethod
    def from_wire(
        cls, body: Mapping[str, Any], *, group_id: str, project_id_default: str | None,
    ) -> WriteEnvelope:
        """Decode the envelope keys of *body*; *group_id* is the row's own group and wins.

        *project_id_default* answers an absent ``project_id``; ``None`` makes the key required.
        """
        body_group = _optional_str(body, 'group_id')
        if body_group is not None and body_group != group_id:
            raise ValueError(
                f'payload group_id {body_group!r} disagrees with the row group {group_id!r}'
            )
        project_id = _optional_str(body, 'project_id')
        if project_id is None:
            project_id = project_id_default
        if project_id is None:
            raise ValueError("missing required key 'project_id'")
        return cls(
            project_id=project_id,
            group_id=group_id,
            agent_id=_optional_str(body, 'agent_id'),
            session_id=_optional_str(body, 'session_id'),
            causation_id=_optional_str(body, '_causation_id'),
            write_op_id=_optional_str(body, '_write_op_id'),
        )


def _referent_scalars(referents: ReferentSet) -> list[dict[str, str]]:
    return [
        {'kind': r.kind, 'project_id': r.project_id, 'number': r.number}
        for r in referents
    ]


#: The "no such key" answer for the ``'ambiguous'`` lookup, distinct from a
#: present JSON ``null`` (which is a corrupt blob, not a legacy one).
_ABSENT = object()


@dataclass(frozen=True)
class DecodedReferents:
    """The referent set a Graphiti write was resolved against, as it rides the queue.

    ``ambiguous`` is ``None`` for a blob written before it was threaded (the
    consumer re-derives it from the content) and a tuple, possibly empty, that
    the consumer believes otherwise; see the PRD's §Rationale rehomed.

    Wire shape: ``{'source': ..., 'refs': [...], 'ambiguous': [...]}``, each
    list entry ``{'kind', 'project_id', 'number'}`` as plain strings.
    """

    refs: ReferentSet
    source: ReferentSource
    ambiguous: ReferentSet | None

    def __post_init__(self) -> None:
        if not _is_referent_source(self.source):
            raise ValueError(
                f'referent source {self.source!r} is not one of {list(REFERENT_SOURCES)}'
            )

    @classmethod
    def from_resolution(cls, resolution: ReferentResolution) -> DecodedReferents:
        return cls(resolution.referents, resolution.source, resolution.ambiguous)

    def to_wire(self) -> dict[str, Any]:
        wire: dict[str, Any] = {'source': self.source, 'refs': _referent_scalars(self.refs)}
        if self.ambiguous is not None:
            wire['ambiguous'] = _referent_scalars(self.ambiguous)
        return wire

    @classmethod
    def from_wire(cls, blob: object) -> DecodedReferents:
        """Decode a referent blob; an absent blob, or any unreadable part of one, is :data:`NO_REFERENTS`.

        Never raises, so a corrupt telemetry field cannot dead-letter a memory,
        and never returns part of a set: a dropped referent would read
        downstream as a conflation. Every unreadable blob logs a WARNING; an
        absent blob and one without ``'ambiguous'`` are legacy rows and do not.
        """
        if blob is None:
            return NO_REFERENTS
        try:
            return cls._from_present_blob(blob)
        except _UnreadableReferents as exc:
            logger.warning(
                "Unreadable 'referents' payload key (%s); treating the write as "
                'having no referents. Blob: %s',
                exc, _safe_repr(blob),
            )
            return NO_REFERENTS

    @classmethod
    def _from_present_blob(cls, blob: object) -> DecodedReferents:
        if not isinstance(blob, dict):
            raise _UnreadableReferents(f'expected a dict, got {type(blob).__name__}')
        source = blob.get('source')
        if not _is_referent_source(source):
            raise _UnreadableReferents(
                f'source {source!r} is not one of {list(REFERENT_SOURCES)}'
            )
        refs = _decode_referent_list('refs', blob.get('refs'))
        raw_ambiguous = blob.get('ambiguous', _ABSENT)
        if raw_ambiguous is _ABSENT:
            return cls(refs, source, None)
        return cls(refs, source, _decode_referent_list('ambiguous', raw_ambiguous))


def _is_referent_source(value: object) -> TypeGuard[ReferentSource]:
    return value in REFERENT_SOURCES


class _UnreadableReferents(Exception):
    """Why a present referent blob cannot be read; :meth:`DecodedReferents.from_wire` degrades on it."""


def _decode_referent_list(key: str, raw: object) -> ReferentSet:
    """One wire list as referents.

    ``Referent`` validates ``kind`` only, so ``number`` and ``project_id`` are
    type-checked here: a non-string number would compare unequal to its string
    twin, and an unhashable one would raise inside a consumer's set.
    """
    if not isinstance(raw, list):
        raise _UnreadableReferents(f'{key!r} must be a list, got {type(raw).__name__}')
    decoded: list[Referent] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise _UnreadableReferents(f'{key!r} entry {_safe_repr(entry)} is not a dict')
        number = entry.get('number')
        project_id = entry.get('project_id', '')
        if not isinstance(number, str) or not isinstance(project_id, str):
            raise _UnreadableReferents(
                f'{key!r} entry {_safe_repr(entry)} has a non-string number/project_id'
            )
        try:
            decoded.append(Referent(
                kind=entry.get('kind', 'task'), project_id=project_id, number=number,
            ))
        except (KeyError, TypeError, ValueError) as e:
            raise _UnreadableReferents(
                f'{key!r} entry {_safe_repr(entry)} is not a valid Referent: {e}'
            ) from e
    return tuple(decoded)


NO_REFERENTS = DecodedReferents((), 'none', None)


def _decode_reference_time(raw: object) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
    logger.warning('Invalid reference_time %r in queue payload; treating as None', raw)
    return None


def _ignore_legacy_uuid(body: Mapping[str, Any], operation: QueueOperation, group_id: str) -> None:
    legacy_uuid = body.get('uuid')
    if legacy_uuid is not None:
        logger.warning(
            "Ignoring legacy 'uuid' key %r in queued %s payload "
            '(group_id=%r, causation_id=%r): the key predates task 3561 and '
            'naming an episode uuid upstream means "LOAD this existing '
            'episode", which raises NodeNotFoundError. Letting graphiti_core '
            'mint the uuid instead; the episode content is preserved.',
            legacy_uuid, operation.value, group_id, body.get('_causation_id'),
        )


def _graphiti_fields(
    body: Mapping[str, Any], operation: QueueOperation, group_id: str,
) -> dict[str, Any]:
    """The fields both Graphiti writes decode alike; an absent optional key takes the field's default."""
    _ignore_legacy_uuid(body, operation, group_id)
    return {
        'envelope': WriteEnvelope.from_wire(body, group_id=group_id, project_id_default=group_id),
        'content': _required_str(body, 'content'),
        'referents': DecodedReferents.from_wire(body.get('referents')),
        **_present_strs(body, 'name', 'source', 'source_description'),
        **_present_flag(body, 'unverified_claim'),
    }


@dataclass(frozen=True, kw_only=True)
class QueuedEpisodeWrite:
    operation: ClassVar[QueueOperation] = QueueOperation.ADD_EPISODE

    envelope: WriteEnvelope
    content: str
    name: str = ''
    source: str = 'text'
    source_description: str = ''
    correlation_id: str | None = None
    temporal_context: str | None = None
    unverified_claim: bool = False
    reference_time: datetime | None = None
    referents: DecodedReferents = NO_REFERENTS

    @property
    def written_text(self) -> str:
        return self.content

    def to_json(self) -> str:
        return json.dumps(self.envelope.to_wire() | {
            'name': self.name,
            'content': self.content,
            'correlation_id': self.correlation_id,
            'source': self.source,
            'source_description': self.source_description,
            'temporal_context': self.temporal_context,
            'unverified_claim': self.unverified_claim,
            'reference_time': (
                self.reference_time.isoformat() if self.reference_time is not None else None
            ),
            'referents': self.referents.to_wire(),
        })

    @classmethod
    def from_json(cls, text: str, *, group_id: str) -> Self:
        body = _load_body(text)
        return cls(
            **_graphiti_fields(body, cls.operation, group_id),
            correlation_id=_optional_str(body, 'correlation_id'),
            temporal_context=_optional_str(body, 'temporal_context'),
            reference_time=_decode_reference_time(body.get('reference_time')),
        )


@dataclass(frozen=True, kw_only=True)
class QueuedGraphitiMemoryWrite:
    """``add_memory``'s Graphiti leg, and ``replay_from_store``'s rows (no journal ids, no category)."""

    operation: ClassVar[QueueOperation] = QueueOperation.ADD_MEMORY_GRAPHITI

    envelope: WriteEnvelope
    content: str
    name: str = ''
    source: str = 'text'
    source_description: str = ''
    memory_category: str | None = None
    referents: DecodedReferents = NO_REFERENTS
    unverified_claim: bool = False

    @property
    def written_text(self) -> str:
        return self.content

    def to_json(self) -> str:
        return json.dumps(self.envelope.to_wire() | {
            'name': self.name,
            'content': self.content,
            'source': self.source,
            'source_description': self.source_description,
            'memory_category': self.memory_category,
            'referents': self.referents.to_wire(),
            'unverified_claim': self.unverified_claim,
        })

    @classmethod
    def from_json(cls, text: str, *, group_id: str) -> Self:
        body = _load_body(text)
        return cls(
            **_graphiti_fields(body, cls.operation, group_id),
            memory_category=_optional_str(body, 'memory_category'),
        )


@dataclass(frozen=True, kw_only=True)
class QueuedMem0Fact:
    """One fact extracted from an episode, bound for Mem0 classification."""

    operation: ClassVar[QueueOperation] = QueueOperation.MEM0_CLASSIFY_AND_ADD

    envelope: WriteEnvelope
    fact_text: str
    temporal_context: str | None = None
    unverified_claim: bool = False

    @property
    def written_text(self) -> str:
        return self.fact_text

    def to_json(self) -> str:
        return json.dumps(self.envelope.to_wire() | {
            'fact_text': self.fact_text,
            'temporal_context': self.temporal_context,
            'unverified_claim': self.unverified_claim,
        })

    @classmethod
    def from_json(cls, text: str, *, group_id: str) -> Self:
        body = _load_body(text)
        return cls(
            envelope=WriteEnvelope.from_wire(body, group_id=group_id, project_id_default='main'),
            fact_text=_required_str(body, 'fact_text'),
            temporal_context=_optional_str(body, 'temporal_context'),
            **_present_flag(body, 'unverified_claim'),
        )


@dataclass(frozen=True, kw_only=True)
class QueuedMem0Add:
    operation: ClassVar[QueueOperation] = QueueOperation.MEM0_ADD

    envelope: WriteEnvelope
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, 'metadata', MappingProxyType(dict(self.metadata)))

    @property
    def written_text(self) -> str:
        return self.content

    def to_json(self) -> str:
        return json.dumps(self.envelope.to_wire() | {
            'content': self.content,
            'metadata': dict(self.metadata),
        })

    @classmethod
    def from_json(cls, text: str, *, group_id: str) -> Self:
        body = _load_body(text)
        metadata = body.get('metadata', {})
        if not isinstance(metadata, dict):
            raise ValueError(f"'metadata' must be an object, got {type(metadata).__name__}")
        return cls(
            envelope=WriteEnvelope.from_wire(body, group_id=group_id, project_id_default=None),
            content=_required_str(body, 'content'),
            metadata=metadata,
        )


QueuedWrite = QueuedEpisodeWrite | QueuedGraphitiMemoryWrite | QueuedMem0Fact | QueuedMem0Add


def _value_types_by_operation() -> Mapping[QueueOperation, type[QueuedWrite]]:
    value_types: tuple[type[QueuedWrite], ...] = get_args(QueuedWrite)
    claims = Counter(value_type.operation for value_type in value_types)
    unclaimed = sorted(set(QueueOperation) - set(claims))
    shared = sorted(operation for operation, count in claims.items() if count > 1)
    if unclaimed or shared:
        raise TypeError(
            'every QueueOperation needs exactly one QueuedWrite value type; '
            f'none for {unclaimed}, several for {shared}'
        )
    return MappingProxyType({value_type.operation: value_type for value_type in value_types})


_VALUE_TYPES = _value_types_by_operation()

#: The value types each callback can be handed: the callback runs after the
#: backend write lands, so a pairing it cannot take is refused before execution.
_CALLBACK_TAKES: Mapping[QueueCallback, tuple[type[QueuedWrite], ...]] = MappingProxyType({
    QueueCallback.DUAL_WRITE_EPISODE: (QueuedEpisodeWrite,),
    QueueCallback.REFRESH_ENTITY_SUMMARIES: (QueuedEpisodeWrite, QueuedGraphitiMemoryWrite),
})
if set(_CALLBACK_TAKES) != set(QueueCallback):
    raise TypeError(
        f'callbacks without a pairing: {sorted(set(QueueCallback) - set(_CALLBACK_TAKES))}'
    )


def check_callback_pairing(value: QueuedWrite, callback: QueueCallback) -> None:
    """Raise :class:`TypeError` unless *callback* can be handed *value*."""
    takes = _CALLBACK_TAKES[callback]
    if not isinstance(value, takes):
        raise TypeError(
            f'queue callback {callback.value!r} takes '
            f'{sorted(value_type.operation.value for value_type in takes)}, '
            f'not {value.operation.value!r}'
        )


def decode_queued_write(operation: str, payload: str, *, group_id: str, row_id: int) -> QueuedWrite:
    """The value a queue row stores, or :class:`QueueDecodeError` naming the row."""
    try:
        value_type = _VALUE_TYPES[QueueOperation(operation)]
    except ValueError:
        raise UnknownQueueOperation(
            f'unknown queue operation {operation!r}',
            operation=operation, row_id=row_id, raw_payload=payload,
        ) from None
    try:
        return value_type.from_json(payload, group_id=group_id)
    except (ValueError, TypeError) as exc:
        raise QueueDecodeError(
            str(exc), operation=operation, row_id=row_id, raw_payload=payload,
        ) from exc


def decode_queue_callback(
    callback_type: str | None, value: QueuedWrite, *, row_id: int, raw_payload: str,
) -> QueueCallback | None:
    """The callback a row names for its decoded *value*, ``None`` for none, or :class:`QueueDecodeError`."""
    if callback_type is None:
        return None
    operation = value.operation.value
    try:
        callback = QueueCallback(callback_type)
    except ValueError:
        raise QueueDecodeError(
            f'unknown queue callback {callback_type!r}',
            operation=operation, row_id=row_id, raw_payload=raw_payload,
        ) from None
    try:
        check_callback_pairing(value, callback)
    except TypeError as exc:
        raise QueueDecodeError(
            str(exc), operation=operation, row_id=row_id, raw_payload=raw_payload,
        ) from None
    return callback
