"""The PRD documents a view reads: each open-work PRD's sidecar and H1 title, off disk.

ACCESS (PRD ``plans/dashboard-prd-lanes-prd.md`` §4.7, open question 6). Every
stat and every read a request makes runs in ONE ``asyncio.to_thread`` hop
(INV-8). A file is parsed again only when its ``mtime_ns`` changes. The worker
thread is handed an immutable copy of that cache and returns what it parsed
afresh; the loop thread merges that in, so no worker mutates shared state. The
keys are the PRDs with open work that the caller names, which bounds the read.

No clock read: the Datum's ``as_of`` is the caller's *now*.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Generic, TypeVar

import yaml
from pydantic import ValidationError
from shared.capability_manifest import CapabilityManifestDoc, load_capability_manifest, sidecar_path

from dashboard.data.datum import Datum, DatumState, unknown_datum
from dashboard.data.prd_view import PrdDocument, SidecarUnread, SignalCause
from dashboard.data.served_read import ServedRead

PRD_DOCUMENTS_TTL_SECONDS = 30.0
"""Age below which a root's documents are served without statting its files again."""

PRD_DOCUMENTS_FRESHNESS_BOUND_SECONDS = 90
"""The documents Datum's freshness bound, three TTLs."""

PRD_DOCUMENTS_REQUEST_WAIT_SECONDS = 4.0
"""Longest a ``/prds`` request waits on an in-flight documents read before it is
served the last good documents (or unknown); the read still lands for the next poll."""

H1_MARKER = '# '
"""A PRD's title is its first line starting with this marker, the marker stripped."""

T = TypeVar('T')
Sidecar = CapabilityManifestDoc | SidecarUnread


@dataclass(frozen=True, slots=True)
class _Parsed(Generic[T]):
    """What one file parsed to, at the ``mtime_ns`` it was parsed at."""

    mtime_ns: int
    value: T


@dataclass(frozen=True, slots=True)
class _Reading:
    """One worker hop's outcome: the documents (``None`` for an unreadable root), and
    every file it parsed afresh, keyed by absolute path."""

    documents: Mapping[str, PrdDocument] | None
    reason: str | None
    sidecars: Mapping[str, _Parsed[Sidecar]]
    titles: Mapping[str, _Parsed[str | None]]


_sidecars: dict[str, _Parsed[Sidecar]] = {}
_titles: dict[str, _Parsed[str | None]] = {}
_DocumentsKey = tuple[str, frozenset[str]]
"""A documents read's root and the PRD keys it reads."""


def _root_of(key: _DocumentsKey) -> str:
    return key[0]


_documents: ServedRead[_DocumentsKey, Mapping[str, PrdDocument]] = ServedRead(
    ttl_seconds=lambda: PRD_DOCUMENTS_TTL_SECONDS,
    freshness_bound_seconds=PRD_DOCUMENTS_FRESHNESS_BOUND_SECONDS,
    last_good_slot=_root_of,
)


async def read_prd_documents(
    project_root: str | os.PathLike[str],
    keys: Iterable[str],
    *,
    now: datetime,
    wait_seconds: float | None = None,
) -> Datum[Mapping[str, PrdDocument]]:
    """The documents of the PRDs *keys* names under *project_root*, as they read at *now*.

    A missing or unloadable sidecar is that PRD's own fact, carried in its
    document. A root that is not a readable directory is unknown, because it
    cannot be told from an unmounted volume. A read that fails or misses its
    wait serves the root's last good documents, read for whichever keys were
    asked for then, as ``stale``.
    """
    root = str(project_root)
    wanted = frozenset(keys)

    async def read() -> Datum[Mapping[str, PrdDocument]]:
        reading = await asyncio.to_thread(
            _read_documents,
            Path(root),
            sorted(wanted),
            MappingProxyType(dict(_sidecars)),
            MappingProxyType(dict(_titles)),
        )
        _sidecars.update(reading.sidecars)
        _titles.update(reading.titles)
        if reading.documents is None:
            return unknown_datum(
                reading.reason or 'documents unread', PRD_DOCUMENTS_FRESHNESS_BOUND_SECONDS,
            )
        return Datum(
            reading.documents, now, DatumState.FRESH, None, PRD_DOCUMENTS_FRESHNESS_BOUND_SECONDS,
        )

    return await _documents.serve((root, wanted), read, now=now, wait_seconds=wait_seconds)


def clear_prd_documents() -> None:
    """Forget every root's documents and every parsed file (test/admin hook)."""
    _documents.clear()
    _sidecars.clear()
    _titles.clear()


def _read_documents(
    root: Path,
    keys: Iterable[str],
    sidecars: Mapping[str, _Parsed[Sidecar]],
    titles: Mapping[str, _Parsed[str | None]],
) -> _Reading:
    """Runs in the worker thread: reads only through its arguments, writes nothing shared."""
    if not root.is_dir():
        reason = f'project root {root} is not a readable directory'
        return _Reading(None, reason, MappingProxyType({}), MappingProxyType({}))
    fresh_sidecars: dict[str, _Parsed[Sidecar]] = {}
    fresh_titles: dict[str, _Parsed[str | None]] = {}
    documents: dict[str, PrdDocument] = {}
    for key in keys:
        sidecar = _parse_if_changed(
            root / sidecar_path(key), sidecars, fresh_sidecars, _load_sidecar, _unstatable_sidecar,
        )
        title = _parse_if_changed(root / key, titles, fresh_titles, _first_h1, lambda _exc: None)
        documents[key] = PrdDocument(title=title, sidecar=sidecar)
    return _Reading(
        MappingProxyType(documents), None,
        MappingProxyType(fresh_sidecars), MappingProxyType(fresh_titles),
    )


def _parse_if_changed(
    path: Path,
    seen: Mapping[str, _Parsed[T]],
    parsed_afresh: dict[str, _Parsed[T]],
    parse: Callable[[Path], T],
    unstatable: Callable[[OSError], T],
) -> T:
    """*path*'s value: *seen*'s when its mtime is unchanged, else parsed and recorded afresh."""
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError as exc:
        return unstatable(exc)
    previous = seen.get(str(path))
    if previous is not None and previous.mtime_ns == mtime_ns:
        return previous.value
    value = parse(path)
    parsed_afresh[str(path)] = _Parsed(mtime_ns, value)
    return value


def _load_sidecar(path: Path) -> Sidecar:
    try:
        return load_capability_manifest(path)
    except FileNotFoundError:
        return SidecarUnread(SignalCause.NO_SIDECAR, None)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        return _invalid(exc)


def _unstatable_sidecar(exc: OSError) -> Sidecar:
    if isinstance(exc, FileNotFoundError):
        return SidecarUnread(SignalCause.NO_SIDECAR, None)
    return _invalid(exc)


def _invalid(exc: Exception) -> SidecarUnread:
    return SidecarUnread(SignalCause.SIDECAR_INVALID, f'{type(exc).__name__}: {exc}')


def _first_h1(path: Path) -> str | None:
    try:
        with path.open(encoding='utf-8', errors='replace') as lines:
            for line in lines:
                if line.startswith(H1_MARKER):
                    return line[len(H1_MARKER):].strip() or None
    except OSError:
        return None
    return None
