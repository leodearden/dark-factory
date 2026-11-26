"""The merge screen manifest, ``scripts/merge_screen.yaml``: one parser for every screen consumer.

Errors are per check (INV-1). A malformed entry becomes a ManifestEntryError for
that check and never takes its siblings down; only a whole-document fault, filed
under MANIFEST_SENTINEL, yields no checks. Stdlib and yaml only, so both the merge
lane and verify_plan can import it without a cycle.
Contract: plans/merge-pre-verify-screen-prd.md §6.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TypeGuard

import yaml

MANIFEST_TOP_KEYS = frozenset({'checks'})
CHECK_KEYS = frozenset({'id', 'module', 'node_ids'})
MANIFEST_SENTINEL = '<manifest>'


@dataclass(frozen=True)
class ManifestCheck:
    id: str
    module: str
    node_ids: tuple[str, ...]


@dataclass(frozen=True)
class ManifestEntryError:
    check: str
    message: str


@dataclass(frozen=True)
class ScreenManifest:
    checks: tuple[ManifestCheck, ...]
    errors: tuple[ManifestEntryError, ...]


class _ManifestFault(ValueError):
    pass


def load_screen_manifest(path: Path, registered_modules: Collection[str]) -> ScreenManifest:
    try:
        doc: object = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        return _whole_document_error(str(path), f'unreadable screen manifest: {exc}')
    return parse_screen_manifest(doc, registered_modules, source=str(path))


def parse_screen_manifest(
    doc: object, registered_modules: Collection[str], *, source: str,
) -> ScreenManifest:
    try:
        entries = _document_entries(doc)
    except _ManifestFault as fault:
        return _whole_document_error(source, str(fault))
    labels = [_check_label(index, entry) for index, entry in enumerate(entries)]
    duplicates = {label for label, count in Counter(labels).items() if count > 1}
    parsed = [
        _entry_or_error(entry, label, label in duplicates, registered_modules, source)
        for entry, label in zip(entries, labels, strict=True)
    ]
    return ScreenManifest(
        checks=tuple(item for item in parsed if isinstance(item, ManifestCheck)),
        errors=tuple(item for item in parsed if isinstance(item, ManifestEntryError)),
    )


def _whole_document_error(source: str, message: str) -> ScreenManifest:
    return ScreenManifest(checks=(), errors=(ManifestEntryError(MANIFEST_SENTINEL, f'{source}: {message}'),))


def _document_entries(doc: object) -> list[object]:
    if not isinstance(doc, dict) or set(doc) != MANIFEST_TOP_KEYS:
        found = sorted(map(str, doc)) if isinstance(doc, dict) else type(doc).__name__
        raise _ManifestFault(f'top-level keys must be exactly {sorted(MANIFEST_TOP_KEYS)}, found {found}')
    checks = doc['checks']
    if not isinstance(checks, list) or not checks:
        raise _ManifestFault(f'checks must be a non-empty list, found {checks!r}')
    return checks


def _check_label(index: int, entry: object) -> str:
    check_id = entry.get('id') if isinstance(entry, dict) else None
    return check_id if _is_valid_id(check_id) else f'#{index}'


def _is_valid_id(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and bool(value) and not any(char.isspace() for char in value)


def _entry_or_error(
    entry: object, label: str, duplicated: bool, registered_modules: Collection[str], source: str,
) -> ManifestCheck | ManifestEntryError:
    prefix = f'{source}: check {label!r}'
    if duplicated:
        return ManifestEntryError(label, f'{prefix}: duplicate id; check ids must be unique')
    try:
        return _parse_entry(entry, registered_modules)
    except _ManifestFault as fault:
        return ManifestEntryError(label, f'{prefix}: {fault}')


def _parse_entry(entry: object, registered_modules: Collection[str]) -> ManifestCheck:
    if not isinstance(entry, dict):
        raise _ManifestFault(f'a check must be a mapping, found {entry!r}')
    if set(entry) != CHECK_KEYS:
        raise _ManifestFault(f'keys must be exactly {sorted(CHECK_KEYS)}, found {sorted(map(str, entry))}')
    return ManifestCheck(
        id=_entry_id(entry['id']),
        module=_entry_module(entry['module'], registered_modules),
        node_ids=_entry_node_ids(entry['node_ids']),
    )


def _entry_id(value: object) -> str:
    if not _is_valid_id(value):
        raise _ManifestFault(f'id must be a non-empty string with no whitespace, found {value!r}')
    return value


def _entry_module(value: object, registered_modules: Collection[str]) -> str:
    if not isinstance(value, str) or value not in registered_modules:
        raise _ManifestFault(
            f'module {value!r} is not a registered module; registered: {sorted(registered_modules)}'
        )
    return value


def _entry_node_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise _ManifestFault(f'node_ids must be a non-empty list of node ids, found {value!r}')
    bad = [node for node in value if not isinstance(node, str) or not node]
    if bad:
        raise _ManifestFault(f'every node id must be a non-empty string, found {bad!r} in {value!r}')
    return tuple(value)
