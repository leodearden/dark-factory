"""A recorder of the process's file and database opens, filesystem mutations, socket connects and spawns, from every thread.

B1 (plans/client-feed-prd.md §8) confines the publisher with it: every effect of a pass lies under its project root,
its out and in dirs and its policy, or in the interpreter. B30 (§14.5) scopes the filer with it: every effect of a
filer run lies under its state dir, its policy and its inbox, or in the Python runtime and CA bundle every process reads.
"""

from __future__ import annotations

import os
import site
import ssl
import sys
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from urllib.parse import unquote, urlsplit

_OPEN: Final = 'open'
_CONNECT: Final = 'socket.connect'
_DATABASE: Final = 'sqlite3.connect'
_POPEN: Final = 'subprocess.Popen'
_MUTATIONS: Final = frozenset({
    'os.mkdir',
    'os.rename',
    'os.remove',
    'os.rmdir',
    'os.symlink',
    'os.link',
    'os.truncate',
    'os.chmod',
    'os.chown',
    'os.utime',
})
_SPAWNS: Final = frozenset({
    _POPEN,
    'os.system',
    'os.exec',
    'os.posix_spawn',
    'os.spawn',
    'os.fork',
    'os.forkpty',
})
_RECORDED: Final = frozenset({_OPEN, _CONNECT, _DATABASE, *_MUTATIONS, *_SPAWNS})
_BYTECODE_CACHE: Final = '__pycache__'
_URI_SCHEME: Final = 'file:'
_PATHLESS_DATABASES: Final = frozenset({'', ':memory:'})
_WRITE_FLAGS: Final = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

_lock = threading.RLock()
_active: tuple[list[tuple[str, tuple[Any, ...]]], ...] = ()
_installed = False


@dataclass(frozen=True)
class Opened:
    path: Path
    writes: bool


@dataclass(frozen=True)
class Mutated:
    event: str
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class Connected:
    address: object


@dataclass(frozen=True)
class DatabaseOpened:
    """A sqlite connect; `path` is None for a database with no file, such as ':memory:'."""

    path: Path | None


@dataclass(frozen=True)
class Spawned:
    event: str
    target: str | None
    cwd: Path | None = None
    args: tuple[str, ...] = ()


def _hook(event: str, args: tuple[Any, ...]) -> None:
    if event not in _RECORDED:
        return
    active = _active
    if not active:
        return
    with _lock:
        for events in active:
            events.append((event, args))


def _install() -> None:
    global _installed
    with _lock:
        if not _installed:
            sys.addaudithook(_hook)
            _installed = True


def _text(value: object) -> str | None:
    return os.fsdecode(value) if isinstance(value, str | bytes | os.PathLike) else None


def _path(value: object) -> Path | None:
    text = _text(value)
    return None if text is None else Path(os.path.abspath(text))


def _opened(args: tuple[Any, ...]) -> Opened | None:
    path, mode, flags = (*args, None, None)[:3]
    where = _path(path)
    if where is None:
        return None
    if isinstance(flags, int):
        return Opened(where, writes=bool(flags & _WRITE_FLAGS))
    return Opened(where, writes=isinstance(mode, str) and any(c in mode for c in 'wax+'))


def _mutated(event: str, args: tuple[Any, ...]) -> Mutated:
    named = args[1:2] if event == 'os.symlink' else args
    return Mutated(event, tuple(path for path in map(_path, named) if path is not None))


def _database_opened(args: tuple[Any, ...]) -> DatabaseOpened:
    text = _text(args[0]) if args else None
    if text is None or text in _PATHLESS_DATABASES:
        return DatabaseOpened(None)
    if not text.startswith(_URI_SCHEME):
        return DatabaseOpened(_path(text))
    path = unquote(urlsplit(text).path)
    return DatabaseOpened(Path(os.path.abspath(path)) if path else None)


def _argv(value: object) -> tuple[str, ...]:
    single = _text(value)
    if single is not None:
        return (single,)
    if isinstance(value, Sequence):
        return tuple(_text(item) or repr(item) for item in value)
    return ()


def _spawned(event: str, args: tuple[Any, ...]) -> Spawned:
    offset = 1 if event == 'os.spawn' else 0
    target = args[offset] if len(args) > offset else None
    spawned = Spawned(event, None if target is None else _text(target) or repr(target))
    if event != _POPEN:
        return spawned
    _executable, argv, cwd = (*args, None, None, None)[:3]
    return Spawned(spawned.event, spawned.target, cwd=_path(cwd), args=_argv(argv))


def _ca_bundles() -> list[str]:
    paths = ssl.get_default_verify_paths()
    bundles = [path for path in (paths.cafile, paths.capath) if path]
    try:
        import certifi
    except ImportError:
        return bundles
    return [*bundles, certifi.where()]


def _roots(paths: Sequence[str]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(Path(os.path.abspath(path)) for path in paths))


def interpreter_roots() -> tuple[Path, ...]:
    """Where the interpreter and its installed packages live: the sys prefixes and the site-packages dirs."""
    prefixes = [sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix]
    return _roots([*prefixes, *site.getsitepackages(), site.getusersitepackages()])


def runtime_roots() -> tuple[Path, ...]:
    """Where the Python runtime and the CA bundle live: what any process opens, so no scope rule is about them."""
    return _roots([*map(str, interpreter_roots()), *_ca_bundles()])


def _under(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(candidate.is_relative_to(root) for candidate in {path, path.resolve()} for root in roots)


def _in_runtime_cache(path: Path, roots: tuple[Path, ...]) -> bool:
    """Whether `path` lies in a bytecode cache of the runtime, the one place the importer writes there."""
    return _BYTECODE_CACHE in path.parts and _under(path, roots)


class AuditRecord:
    """The events raised while its recording() block was open, normalised on read."""

    def __init__(self, events: list[tuple[str, tuple[Any, ...]]]) -> None:
        self._events = events

    def _raw(self) -> tuple[tuple[str, tuple[Any, ...]], ...]:
        with _lock:
            return tuple(self._events)

    @property
    def opens(self) -> tuple[Opened, ...]:
        opened = (_opened(args) for event, args in self._raw() if event == _OPEN)
        return tuple(item for item in opened if item is not None)

    @property
    def mutations(self) -> tuple[Mutated, ...]:
        return tuple(_mutated(event, args) for event, args in self._raw() if event in _MUTATIONS)

    @property
    def connects(self) -> tuple[Connected, ...]:
        return tuple(Connected(args[1]) for event, args in self._raw() if event == _CONNECT)

    @property
    def database_opens(self) -> tuple[DatabaseOpened, ...]:
        return tuple(_database_opened(args) for event, args in self._raw() if event == _DATABASE)

    @property
    def spawns(self) -> tuple[Spawned, ...]:
        return tuple(_spawned(event, args) for event, args in self._raw() if event in _SPAWNS)

    def opens_outside(self, roots: tuple[Path, ...]) -> tuple[Opened, ...]:
        """Leaves out reads under `roots` and writes into a bytecode cache under them; any other write is kept."""
        resolved = _resolved(roots)
        return tuple(opened for opened in self.opens if not _runtime_open(opened, resolved))

    def mutations_outside(self, roots: tuple[Path, ...]) -> tuple[Mutated, ...]:
        """Leaves out the importer's bytecode-cache writes under `roots`; any other mutation under them is kept."""
        resolved = _resolved(roots)
        return tuple(mutated for mutated in self.mutations if not _runtime_cache_write(mutated, resolved))

    def opens_outside_runtime(self) -> tuple[Opened, ...]:
        """Leaves out reads of the runtime and the CA bundle, and the importer's bytecode-cache writes; any other write is kept."""
        return self.opens_outside(runtime_roots())

    def mutations_outside_runtime(self) -> tuple[Mutated, ...]:
        """Leaves out the importer's bytecode-cache writes, made on a module's first import from an uncompiled venv; any other mutation of the runtime is kept."""
        return self.mutations_outside(runtime_roots())


def _runtime_open(opened: Opened, roots: tuple[Path, ...]) -> bool:
    return _in_runtime_cache(opened.path, roots) if opened.writes else _under(opened.path, roots)


def _runtime_cache_write(mutated: Mutated, roots: tuple[Path, ...]) -> bool:
    return bool(mutated.paths) and all(_in_runtime_cache(path, roots) for path in mutated.paths)


def _resolved(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    return tuple({*roots, *(root.resolve() for root in roots)})


@contextmanager
def recording() -> Iterator[AuditRecord]:
    """Record every open, database open, mutation, connect and spawn raised in this process until the block exits."""
    global _active
    _install()
    events: list[tuple[str, tuple[Any, ...]]] = []
    with _lock:
        _active = (*_active, events)
    try:
        yield AuditRecord(events)
    finally:
        with _lock:
            _active = tuple(active for active in _active if active is not events)
