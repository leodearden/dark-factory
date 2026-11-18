"""The asset-build identity of the static/redux tree, as this server publishes it.

An identity is an opaque content hash of the files under static/redux. It
is compared only for equality and never parsed. The server publishes two on
every ``/api/v2/dashboard/`` response, through :class:`AssetBuildHeaders`:
the build it started with (``AssetBuild.started``) and the build on disk now
(``AssetBuild.on_disk()``). The page learns the build it booted with from
the ``df-asset-build`` meta that :func:`inject_page_build` fills in when
``/`` is served. It compares the three in ``static/redux/asset_skew.js``,
which holds the verdict rule and its rationale (task 10720).

The hash skips what an editor, a merge or a patch leaves beside the assets
in a working checkout: dotfiles and dot-directories (vim swap files, emacs
locks), emacs autosaves (``#name#``), and backups and leftovers by suffix
(``~``, ``.orig``, ``.rej``, ``.bak``, ``.swp``, ``.swo``, ``.tmp``). The
page never loads them, so a stray one must not announce a restart for a
build nobody shipped. The skip is a denylist on purpose: a new asset type is
hashed without anyone listing it, so it cannot leave a stale page unnoticed.
"""

from __future__ import annotations

import hashlib
import logging
import os
import stat
import time
from collections.abc import Callable
from html import escape
from pathlib import Path

from fastapi import FastAPI
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from dashboard.payload_budgets import API_PREFIX

ASSET_BUILD_STARTED_HEADER = 'X-DF-Asset-Build-Started'
ASSET_BUILD_ON_DISK_HEADER = 'X-DF-Asset-Build-On-Disk'

ON_DISK_MAX_AGE_S = 1.0
"""How long one walk of the tree answers ``on_disk()``: one poll set's burst shares it."""

_IDENTITY_HEX_CHARS = 16
_UNSHIPPED_PREFIXES = ('.', '#')
_UNSHIPPED_SUFFIXES = ('~', '.orig', '.rej', '.bak', '.swp', '.swo', '.tmp')

logger = logging.getLogger(__name__)


def _page_build_meta(identity: str) -> str:
    return f'<meta name="df-asset-build" content="{escape(identity)}" />'


PAGE_BUILD_PLACEHOLDER = _page_build_meta('')

Fingerprint = tuple[tuple[str, int, int], ...]


def _raise(error: OSError) -> None:
    raise error


def _is_shipped(name: str) -> bool:
    return not name.startswith(_UNSHIPPED_PREFIXES) and not name.endswith(_UNSHIPPED_SUFFIXES)


def _regular_files(root: Path) -> list[tuple[str, Path, os.stat_result]]:
    files: list[tuple[str, Path, os.stat_result]] = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=_raise):
        dirnames[:] = [name for name in dirnames if not name.startswith('.')]
        for name in filter(_is_shipped, filenames):
            path = Path(dirpath) / name
            status = path.stat()
            if stat.S_ISREG(status.st_mode):
                files.append((path.relative_to(root).as_posix(), path, status))
    return sorted(files, key=lambda file: file[0])


def _fingerprint(files: list[tuple[str, Path, os.stat_result]]) -> Fingerprint:
    return tuple((relpath, status.st_mtime_ns, status.st_size) for relpath, _, status in files)


def _content_hash(files: list[tuple[str, Path, os.stat_result]]) -> str:
    digest = hashlib.sha256()
    for relpath, path, _ in files:
        content = path.read_bytes()
        digest.update(os.fsencode(relpath))
        digest.update(b'\0')
        digest.update(len(content).to_bytes(8, 'big'))
        digest.update(content)
    return digest.hexdigest()[:_IDENTITY_HEX_CHARS]


def tree_identity(root: Path) -> str | None:
    try:
        return _content_hash(_regular_files(root))
    except OSError:
        return None


class AssetBuild:
    """The build a server process started with, and the build on disk now.

    ``on_disk()`` re-stats the tree at most once per ``max_age_s`` of the
    injected ``monotonic`` clock, and re-hashes it only when a file's path,
    mtime or size changed since the last walk.
    """

    def __init__(
        self,
        root: Path,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        max_age_s: float = ON_DISK_MAX_AGE_S,
    ) -> None:
        self.root = root
        self._monotonic = monotonic
        self._max_age_s = max_age_s
        self._measured: tuple[Fingerprint | None, str | None] = (None, None)
        self._last_walk: tuple[float, str | None] | None = None
        self.started = self.on_disk()

    @classmethod
    def at_start(cls, root: Path) -> AssetBuild:
        return cls(root)

    def on_disk(self) -> str | None:
        now = self._monotonic()
        if self._last_walk is not None and now - self._last_walk[0] < self._max_age_s:
            return self._last_walk[1]
        identity = self._walk()
        self._last_walk = (now, identity)
        return identity

    def _walk(self) -> str | None:
        try:
            files = _regular_files(self.root)
            fingerprint = _fingerprint(files)
            if fingerprint != self._measured[0]:
                self._measured = (fingerprint, _content_hash(files))
        except OSError:
            return None
        return self._measured[1]


def install_asset_build(app: FastAPI, build: AssetBuild) -> None:
    app.state.asset_build = build


def app_asset_build(app: FastAPI) -> AssetBuild | None:
    build = getattr(app.state, 'asset_build', None)
    return build if isinstance(build, AssetBuild) else None


class AssetBuildHeaders:
    """Pure-ASGI middleware: stamps both identities on every dashboard API response.

    A header whose identity is unknown is omitted rather than sent empty.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http' or not scope['path'].startswith(API_PREFIX):
            await self.app(scope, receive, send)
            return

        async def send_with_build(message: Message) -> None:
            if message['type'] == 'http.response.start':
                _stamp_build(message, app_asset_build(scope['app']))
            await send(message)

        await self.app(scope, receive, send_with_build)


def _stamp_build(message: Message, build: AssetBuild | None) -> None:
    if build is None:
        return
    headers = MutableHeaders(scope=message)
    for name, identity in ((ASSET_BUILD_STARTED_HEADER, build.started), (ASSET_BUILD_ON_DISK_HEADER, build.on_disk())):
        if identity is not None:
            headers.append(name, identity)


def inject_page_build(html: str, identity: str | None) -> str:
    """Fill index.html's one df-asset-build placeholder, or serve the page unmodified.

    An unmodified page carries no identity, so it stays inert: never a 500.
    """
    if identity is None:
        logger.warning('serving / with no asset-build identity: the build on disk is unknown')
        return html
    placeholders = html.count(PAGE_BUILD_PLACEHOLDER)
    if placeholders != 1:
        logger.warning(
            'serving / with no asset-build identity: index.html holds %d %s placeholders, not 1',
            placeholders,
            PAGE_BUILD_PLACEHOLDER,
        )
        return html
    return html.replace(PAGE_BUILD_PLACEHOLDER, _page_build_meta(identity))
