"""The asset-build identity of the static/redux tree, as this server publishes it.

An identity is an opaque content hash of every file the page loads. It is
compared only for equality and never parsed. The server publishes two: the
build it started with (``AssetBuild.started``) and the build on disk now
(``AssetBuild.on_disk()``). A page compares them with the build it booted
with, in ``static/redux/asset_skew.js``, which holds the verdict rule and
its rationale (task 10720).
"""

from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

_IDENTITY_HEX_CHARS = 16

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
        digest.update(relpath.encode())
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
