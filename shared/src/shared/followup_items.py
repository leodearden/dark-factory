"""``metadata.followup_items``: the observations one follow-up task was filed to address.

Contract: plans/follow-up-filing-shape-prd.md §4.1 and §3 decisions 1, 3 and 4. Each item's
``key``, ``origin_task`` and ``origin_escalation_id`` are server-owned, and :func:`merge_items` is
the one union every writer uses. The slice registers itself at import, like ``shared.defects``, and
is not re-exported from ``shared/__init__.py``.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Annotated, Final

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, ValidationError

from shared.task_metadata import register_metadata_submodel

__all__ = [
    'FOLLOWUP_ITEMS_METADATA_KEY',
    'FOLLOWUP_ITEMS_REFUSED_ERROR',
    'MAX_FOLLOWUP_ITEMS',
    'MAX_SUMMARY_CHARS',
    'UNPARENTED_ORIGIN',
    'BundleViolation',
    'DuplicateFollowupItemKey',
    'FollowupBundleInvalid',
    'FollowupItem',
    'item_key',
    'legacy_item',
    'merge_items',
    'observed_item',
    'parse_item',
    'read_items',
    'require_unique_item_keys',
    'task_items',
    'with_derived_bundle_keys',
]

FOLLOWUP_ITEMS_METADATA_KEY: Final = 'followup_items'
# The wire ``error`` of a submit_task refused for its bundle: fused-memory reports it, filer prompts name it.
FOLLOWUP_ITEMS_REFUSED_ERROR: Final = 'followup_items_refused'
MAX_FOLLOWUP_ITEMS: Final = 5
MAX_SUMMARY_CHARS: Final = 300
UNPARENTED_ORIGIN: Final = 'unparented'

_WHITESPACE_RUN = re.compile(r'\s+')
# No repo path and no one-line summary contains NUL, so the joined key material is unambiguous.
_KEY_MATERIAL_SEPARATOR = '\x00'
_UNDERIVED_KEY = 'underived'


def _one_line(text: str) -> str:
    return _WHITESPACE_RUN.sub(' ', text).strip()


def _canonical_files(files: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(files)))


def _sha256_16(material: str) -> str:
    # A stdlib mirror of orchestrator/src/orchestrator/agents/triage.py::sha256_16's 16-char shape.
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _non_blank(text: str) -> str:
    if not text.strip():
        raise ValueError('must not be blank')
    return text


def _collapse_if_str(value: object) -> object:
    return _one_line(value) if isinstance(value, str) else value


def _bounded_summary(text: str) -> str:
    if not text:
        raise ValueError('summary must not be blank')
    if len(text) > MAX_SUMMARY_CHARS:
        raise ValueError(
            f'summary is {len(text)} chars on one line; at most {MAX_SUMMARY_CHARS} are allowed'
        )
    return text


def _declared_files(files: tuple[str, ...]) -> tuple[str, ...]:
    if not files:
        raise ValueError('files must name at least one file')
    if any(not path.strip() for path in files):
        raise ValueError(f'files must not contain a blank entry; got {list(files)!r}')
    return _canonical_files(files)


_NonBlankStr = Annotated[str, AfterValidator(_non_blank)]
_ItemSummary = Annotated[str, BeforeValidator(_collapse_if_str), AfterValidator(_bounded_summary)]
_ItemFiles = Annotated[tuple[str, ...], AfterValidator(_declared_files)]


class FollowupItem(BaseModel):
    """One observation, stored canonically: the summary on one line, the files sorted and unique."""

    model_config = ConfigDict(frozen=True, extra='forbid')

    key: _NonBlankStr
    summary: _ItemSummary
    files: _ItemFiles
    origin_task: _NonBlankStr
    origin_escalation_id: str | None = None


class BundleViolation(StrEnum):
    """Which §4.1 invariant a ``followup_items`` value breaks: the ``reason`` its refusal reports."""

    NOT_A_LIST = 'not_a_list'
    INVALID_ITEM = 'invalid_item'
    ITEM_COUNT = 'item_count'
    DUPLICATE_KEY = 'duplicate_key'
    UNDECLARED_FILES = 'undeclared_files'


class FollowupBundleInvalid(ValueError):
    """A ``followup_items`` value breaks the invariant ``reason`` names; ``detail`` gives the offending values."""

    def __init__(self, reason: BundleViolation, detail: str, *, index: int | None = None) -> None:
        self.reason = reason
        self.detail = detail
        self.index = index
        where = '' if index is None else f' at item {index}'
        super().__init__(f'metadata.{FOLLOWUP_ITEMS_METADATA_KEY} {reason.value}{where}: {detail}')


class DuplicateFollowupItemKey(FollowupBundleInvalid):
    """One list names a key more than once; ``keys`` holds each repeated key once, first seen first."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self.keys = keys
        super().__init__(
            BundleViolation.DUPLICATE_KEY,
            f'item keys must be unique; repeated: {", ".join(repr(key) for key in keys)}',
        )


def item_key(*, origin_task: str, files: Sequence[str], summary: str) -> str:
    """The item's identity: blind to file order, repeated files, and summary case and whitespace.

    Raises:
        TypeError: when ``files`` is one str, which would otherwise be keyed character by character.
    """
    if isinstance(files, str):
        raise TypeError(f'item_key files must be a sequence of paths, not one str: {files!r}')
    material = (origin_task, *_canonical_files(files), _one_line(summary).casefold())
    return _sha256_16(_KEY_MATERIAL_SEPARATOR.join(material))


def observed_item(
    *, summary: str, files: Sequence[str], origin_task: str, origin_escalation_id: str | None = None,
) -> FollowupItem:
    """One observation as an item keyed by its own stored fields.

    Raises:
        ValidationError: when the observation breaks a field invariant.
    """
    return _keyed_item({
        'summary': summary,
        'files': files,
        'origin_task': origin_task,
        'origin_escalation_id': origin_escalation_id,
    })


def _keyed_item(fields: Mapping[str, object]) -> FollowupItem:
    unkeyed = FollowupItem.model_validate({**fields, 'key': _UNDERIVED_KEY})
    derived = item_key(origin_task=unkeyed.origin_task, files=unkeyed.files, summary=unkeyed.summary)
    return unkeyed.model_copy(update={'key': derived})


def require_unique_item_keys(items: Iterable[FollowupItem]) -> None:
    """Refuse a list the registration cannot see is invalid: one that repeats a key.

    Raises:
        DuplicateFollowupItemKey: naming each repeated key.
    """
    counts = Counter(item.key for item in items)
    repeated = tuple(key for key, count in counts.items() if count > 1)
    if repeated:
        raise DuplicateFollowupItemKey(repeated)


def merge_items(base: Sequence[FollowupItem], extra: Sequence[FollowupItem]) -> tuple[FollowupItem, ...]:
    """The order-preserving union by key; a key present in both keeps ``base``'s item.

    Raises:
        DuplicateFollowupItemKey: when either input repeats a key.
    """
    require_unique_item_keys(base)
    require_unique_item_keys(extra)
    present = {item.key for item in base}
    return (*base, *(item for item in extra if item.key not in present))


def _parent_task(metadata: Mapping[str, object]) -> str | None:
    spawned_from = metadata.get('spawned_from')
    parent = '' if spawned_from is None else str(spawned_from).strip()
    return parent or None


def _followup_escalation_id(parent_task: str | None) -> str | None:
    return None if parent_task is None else f'agent-followup-{parent_task}'


def legacy_item(
    *, title: str, files: Sequence[str], metadata: Mapping[str, object],
) -> FollowupItem | None:
    """A pre-bundle single follow-up as one item keyed by its caller-sent ``suggestion_hash``.

    None when the row is not such a single: no usable ``suggestion_hash``, no files, already a
    bundle, or a title or files no item can hold.
    """
    suggestion_hash = metadata.get('suggestion_hash')
    if FOLLOWUP_ITEMS_METADATA_KEY in metadata or not files:
        return None
    if not isinstance(suggestion_hash, str) or not suggestion_hash.strip():
        return None
    escalation_id = metadata.get('escalation_id')
    return parse_item({
        'key': suggestion_hash,
        'summary': _one_line(title)[:MAX_SUMMARY_CHARS],
        'files': files,
        'origin_task': _parent_task(metadata) or UNPARENTED_ORIGIN,
        'origin_escalation_id': escalation_id if isinstance(escalation_id, str) else None,
    })


def parse_item(element: object) -> FollowupItem | None:
    """One stored element as an item, or None when it is not one: the lenient twin of :func:`read_items`."""
    try:
        return _validated_element(element, FollowupItem.model_validate)
    except FollowupBundleInvalid:
        return None


def read_items(metadata: Mapping[str, object]) -> tuple[FollowupItem, ...]:
    """The stored items, strictly: () when the key is absent.

    Raises:
        FollowupBundleInvalid: ``NOT_A_LIST``, or ``INVALID_ITEM`` naming the element's index.
    """
    if FOLLOWUP_ITEMS_METADATA_KEY not in metadata:
        return ()
    elements = _require_list(metadata[FOLLOWUP_ITEMS_METADATA_KEY])
    return tuple(
        _validated_element(element, FollowupItem.model_validate, index=index)
        for index, element in enumerate(elements)
    )


def task_items(
    *, title: str, files: Sequence[str], metadata: Mapping[str, object],
) -> tuple[FollowupItem, ...]:
    """The observations a stored task stands for: its followup_items, else its pre-bundle single as one legacy item, else ().

    Raises:
        FollowupBundleInvalid: when the stored ``followup_items`` value is malformed.
    """
    if FOLLOWUP_ITEMS_METADATA_KEY in metadata:
        return read_items(metadata)
    legacy = legacy_item(title=title, files=files, metadata=metadata)
    return () if legacy is None else (legacy,)


def _require_list(raw: object) -> list[object]:
    if not isinstance(raw, list):
        raise FollowupBundleInvalid(
            BundleViolation.NOT_A_LIST, f'expected a list of item objects, got {type(raw).__name__}',
        )
    return raw


def _validated_element(
    element: object, build: Callable[[dict[str, object]], FollowupItem], *, index: int | None = None,
) -> FollowupItem:
    if not isinstance(element, Mapping):
        raise FollowupBundleInvalid(
            BundleViolation.INVALID_ITEM, f'expected an object, got {type(element).__name__}', index=index,
        )
    try:
        return build(dict(element))
    except ValidationError as exc:
        raise FollowupBundleInvalid(BundleViolation.INVALID_ITEM, str(exc), index=index) from exc


def with_derived_bundle_keys(metadata: Mapping[str, object]) -> dict[str, object]:
    """A copy of ``metadata`` whose follow-up bundle carries server-derived R4 keys (decision 3).

    Without ``followup_items`` the copy is unchanged, so steward, review-cycle and pre-bundle
    filers keep the keys they sent. With it, each item's ``key``, ``origin_task`` and
    ``origin_escalation_id``, and the top-level ``escalation_id`` and ``suggestion_hash``, are
    derived from ``spawned_from`` and the items, and any caller value is overwritten.

    Raises:
        FollowupBundleInvalid: when the bundle breaks a create invariant; ``reason`` names which.
    """
    derived = dict(metadata)
    if FOLLOWUP_ITEMS_METADATA_KEY not in metadata:
        return derived
    parent = _parent_task(metadata)
    escalation_id = _followup_escalation_id(parent)
    items = _observed_bundle(
        metadata[FOLLOWUP_ITEMS_METADATA_KEY],
        {'origin_task': parent or UNPARENTED_ORIGIN, 'origin_escalation_id': escalation_id},
    )
    _require_declared_files(items, metadata.get('files'))
    derived[FOLLOWUP_ITEMS_METADATA_KEY] = [item.model_dump(mode='json') for item in items]
    derived['suggestion_hash'] = _bundle_suggestion_hash(items)
    if escalation_id is None:
        derived.pop('escalation_id', None)
    else:
        derived['escalation_id'] = escalation_id
    return derived


def _observed_bundle(raw: object, origin: Mapping[str, object]) -> tuple[FollowupItem, ...]:
    elements = _require_list(raw)
    if not 1 <= len(elements) <= MAX_FOLLOWUP_ITEMS:
        raise FollowupBundleInvalid(
            BundleViolation.ITEM_COUNT,
            f'a bundle holds 1..{MAX_FOLLOWUP_ITEMS} items; got {len(elements)}',
        )
    items = tuple(
        _validated_element(element, lambda fields: _keyed_item({**fields, **origin}), index=index)
        for index, element in enumerate(elements)
    )
    require_unique_item_keys(items)
    return items


def _require_declared_files(items: Iterable[FollowupItem], declared: object) -> None:
    declared_files = (
        {path for path in declared if isinstance(path, str)} if isinstance(declared, list) else set()
    )
    missing = sorted({path for item in items for path in item.files} - declared_files)
    if missing:
        raise FollowupBundleInvalid(
            BundleViolation.UNDECLARED_FILES,
            f'every item file must also be in metadata.files; missing: {", ".join(missing)}',
        )


def _bundle_suggestion_hash(items: Iterable[FollowupItem]) -> str:
    return _sha256_16(_KEY_MATERIAL_SEPARATOR.join(sorted(item.key for item in items)))


# An import side effect of THIS module, never of shared.task_metadata; why is
# shared/src/shared/deploy_state.py::DeployState's module docstring.
register_metadata_submodel(FOLLOWUP_ITEMS_METADATA_KEY, FollowupItem, cardinality='list')
