"""The run home's ``sessions.json``: the sessions spawned for the run, by title.

The file is ``{title: {stream, pass, item, spawned_after, record_dir}}``: ``item``
is null except on an investigate pass, and ``spawned_after`` is a ``state.now_utc``
stamp taken before the spawn. ``gen_brief.py --spawn`` creates an entry with
``record_dir`` null; ``watch_sessions.py`` fills ``record_dir`` once it resolves the
session's record. A title names one session, so an entry is never replaced.
"""
from __future__ import annotations

import json
import os
import tempfile
import types
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import home_writer
import state


@dataclass(frozen=True)
class SpawnedSession:
    title: str
    stream: str
    pass_: state.Pass
    item: str | None
    spawned_after: str
    record_dir: str | None

    def __post_init__(self) -> None:
        if self.title.splitlines() != [self.title]:
            raise state.StateError(f'title {self.title!r} is not one non-empty line')
        investigate = self.pass_ is state.Pass.INVESTIGATE
        if investigate and self.item is None:
            raise state.StateError(f'item: an {self.pass_} session names the item it investigates')
        if not investigate and self.item is not None:
            raise state.StateError(f'item {self.item!r} names only an investigate session, not {self.pass_}')


_TEXT_FIELDS: Mapping[str, bool] = types.MappingProxyType({
    'stream': False, 'item': True, 'spawned_after': False, 'record_dir': True,
})
"""Each text field of an entry, and whether it may be null."""
_FIELDS = frozenset({*_TEXT_FIELDS, 'pass'})


def _checked_fields(path: Path, title: str, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise state.StateError(f'{path}: {title} is not an object: {raw!r}')
    faults = [f'{title}.{name} is missing' for name in sorted(_FIELDS - set(raw))]
    faults += [f'{title}.{name} is not a field of a session' for name in sorted(set(raw) - _FIELDS)]
    if faults:
        raise state.StateError(f'{path}: ' + '; '.join(faults))
    for name, nullable in _TEXT_FIELDS.items():
        value = raw[name]
        if not (isinstance(value, str) or (nullable and value is None)):
            raise state.StateError(f'{path}: {title}.{name} is not a string: {value!r}')
    try:
        pass_ = state.Pass(raw['pass'])
    except ValueError:
        raise state.StateError(
            f'{path}: {title}.pass {raw["pass"]!r} is not one of {", ".join(state.Pass)}'
        ) from None
    return {**{name: raw[name] for name in _TEXT_FIELDS}, 'pass_': pass_}


def _session(path: Path, title: str, raw: Any) -> SpawnedSession:
    fields = _checked_fields(path, title, raw)
    try:
        return SpawnedSession(title=title, **fields)
    except state.StateError as bad:
        raise state.StateError(f'{path}: {title}: {bad}') from None


def read(home: state.Home) -> Mapping[str, SpawnedSession]:
    """Every recorded session by title; empty when sessions.json does not exist yet."""
    path = home.sessions_json
    try:
        text = path.read_text(encoding='utf-8')
    except FileNotFoundError:
        return types.MappingProxyType({})
    try:
        data = json.loads(text)
    except ValueError as bad:
        raise state.StateError(f'{path} is not JSON: {bad}') from None
    if not isinstance(data, Mapping):
        raise state.StateError(f'{path} is not a JSON object')
    return types.MappingProxyType({title: _session(path, title, raw) for title, raw in data.items()})


def _refuse_taken(recorded: Mapping[str, SpawnedSession], title: str) -> None:
    taken = recorded.get(title)
    if taken is not None:
        item = '' if taken.item is None else f' {taken.item}'
        raise state.StateError(
            f'session title {title!r} is taken by {taken.stream}.{taken.pass_}{item}; '
            'a title names one session'
        )


def refuse_taken(home: state.Home, title: str) -> None:
    """Refuse *title* when sessions.json already records it; ``add`` checks again under the lock."""
    _refuse_taken(read(home), title)


def _entry(session: SpawnedSession) -> dict[str, Any]:
    return {
        'stream': session.stream, 'pass': str(session.pass_), 'item': session.item,
        'spawned_after': session.spawned_after, 'record_dir': session.record_dir,
    }


def _write(home: state.Home, recorded: Mapping[str, SpawnedSession]) -> None:
    data = {title: _entry(session) for title, session in recorded.items()}
    handle, temp = tempfile.mkstemp(dir=home.root, prefix=f'.{home.sessions_json.name}.')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as out:
            out.write(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
        os.replace(temp, home.sessions_json)
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def add(home: state.Home, session: SpawnedSession) -> None:
    """Record *session* under the home's write lock; refuses a title already recorded."""
    with home_writer.locked(home):
        recorded = read(home)
        _refuse_taken(recorded, session.title)
        _write(home, {**recorded, session.title: session})
