"""What a client must never find in a feed directory: the task fields no projection carries, and caller-named needles."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

FORBIDDEN_KEYS: Final = frozenset({'description', 'details', 'test_strategy', 'metadata'})


def json_keys(value: Any) -> Iterator[str]:
    """Every object key at any depth of a decoded JSON document."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from json_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from json_keys(item)


def _json_documents(path: Path, text: str) -> list[Any]:
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(text)] if path.suffix == '.json' else []


def leaks(out_dir: Path, *, needles: tuple[str, ...]) -> list[tuple[str, str]]:
    """(file, finding) for each needle in a file's name or text, and each forbidden key in its JSON documents.

    Keys are walked, never grepped for: a tree path may well contain the word 'description'.
    """
    found: list[tuple[str, str]] = []
    for path in sorted(item for item in out_dir.rglob('*') if item.is_file()):
        name = path.relative_to(out_dir).as_posix()
        text = path.read_bytes().decode('utf-8', errors='replace')
        found += [(name, f'needle {needle!r}') for needle in needles if needle in name or needle in text]
        found += [
            (name, f'key {key!r}')
            for document in _json_documents(path, text)
            for key in sorted(FORBIDDEN_KEYS & set(json_keys(document)))
        ]
    return found
