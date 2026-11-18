"""The asset-build identity of the static/redux tree, as this server publishes it.

An identity is an opaque content hash of every file the page loads. It is
compared only for equality and never parsed. The server publishes two on
every ``/api/v2/dashboard/`` response, through :class:`AssetBuildHeaders`:
the build it started with (``AssetBuild.started``) and the build on disk now
(``AssetBuild.on_disk()``). A page compares them with the build it booted
with, in ``static/redux/asset_skew.js``, which holds the verdict rule and
its rationale (task 10720).
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

from fastapi import FastAPI
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

ASSET_BUILD_STARTED_HEADER = 'X-DF-Asset-Build-Started'
ASSET_BUILD_ON_DISK_HEADER = 'X-DF-Asset-Build-On-Disk'

_IDENTITY_HEX_CHARS = 16
_DASHBOARD_API_PREFIX = '/api/v2/dashboard/'

Fingerprint = tuple[tuple[str, int, int], ...]


def _raise(error: OSError) -> None:
    raise error


def _regular_files(root: Path) -> list[tuple[str, Path, os.stat_result]]:
    files: list[tuple[str, Path, os.stat_result]] = []
    for dirpath, _dirnames, filenames in os.walk(root, onerror=_raise):
        for name in filenames:
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

    ``on_disk()`` re-stats the tree on every call and re-hashes it only when
    a file's path, mtime or size changed since the last call.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self._measured: tuple[Fingerprint | None, str | None] = (None, None)
        self.started = self.on_disk()

    @classmethod
    def at_start(cls, root: Path) -> AssetBuild:
        return cls(root)

    def on_disk(self) -> str | None:
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
        if scope['type'] != 'http' or not scope['path'].startswith(_DASHBOARD_API_PREFIX):
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
