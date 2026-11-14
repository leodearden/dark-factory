"""Replace known secret values in JSONL files with same-length placeholders.

Callers pass ``{var: value}``; which variables hold secrets is
``shared.secret_env``'s question, not this module's. A value is matched in the
JSON-string-escaped form it takes inside a JSONL line, and is replaced by
``<redacted:VAR>`` right-padded with ``x`` to the same byte length, so a file
keeps its size and line count. A line whose rewrite would turn it from valid
JSON into invalid JSON is left as it was and reported in
``FileRedaction.refused``; every other line is still redacted. No message ever
carries a value.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

__all__ = [
    'MIN_REDACTABLE_LEN',
    'FileChangedDuringRedaction',
    'FileRedaction',
    'FileStamp',
    'Needle',
    'RedactionError',
    'RedactionWouldCorruptJson',
    'RefusedLine',
    'copy_redacted',
    'file_stamp',
    'redact_in_place',
    'redaction_needles',
]

MIN_REDACTABLE_LEN = 20

_DETECT_CHUNK_BYTES = 1 << 20

StrPath = str | os.PathLike[str]


@dataclass(frozen=True)
class Needle:
    var: str
    pattern: bytes = field(repr=False)
    placeholder: bytes


@dataclass(frozen=True)
class RefusedLine:
    line_number: int
    var: str


@dataclass(frozen=True)
class FileRedaction:
    counts: Mapping[str, int]
    refused: tuple[RefusedLine, ...] = ()

    @property
    def redacted(self) -> bool:
        return any(self.counts.values())

    def raise_if_refused(self, path: StrPath) -> None:
        if self.refused:
            first = self.refused[0]
            raise RedactionWouldCorruptJson(Path(path), first.line_number, first.var)


@dataclass(frozen=True)
class FileStamp:
    st_ino: int
    st_size: int
    st_mtime_ns: int


class RedactionError(Exception):
    pass


class RedactionWouldCorruptJson(RedactionError):
    def __init__(self, path: Path, line_number: int, var: str) -> None:
        super().__init__(
            f'{path}: redacting {var} on line {line_number} would turn a valid JSON '
            'line invalid; that line was left unredacted'
        )
        self.path = path
        self.line_number = line_number
        self.var = var


class FileChangedDuringRedaction(RedactionError):
    def __init__(self, path: Path) -> None:
        super().__init__(f'{path} changed while it was being redacted; it was left as found')
        self.path = path


def redaction_needles(secrets: Mapping[str, str]) -> tuple[Needle, ...]:
    by_pattern: dict[bytes, Needle] = {}
    for var, value in sorted(secrets.items()):
        if len(value) < MIN_REDACTABLE_LEN:
            continue
        for ensure_ascii in (True, False):
            pattern = json.dumps(value, ensure_ascii=ensure_ascii)[1:-1].encode()
            by_pattern.setdefault(pattern, Needle(var, pattern, _placeholder(var, len(pattern))))
    return tuple(sorted(by_pattern.values(), key=lambda needle: -len(needle.pattern)))


def file_stamp(path: StrPath) -> FileStamp:
    return _stamp_of(os.stat(path))


def copy_redacted(src: StrPath, dest: StrPath, needles: Sequence[Needle]) -> FileRedaction:
    counts: Counter[str] = Counter()
    refused: list[RefusedLine] = []
    try:
        with open(src, 'rb') as reader, open(dest, 'wb') as writer:
            for line_number, line in enumerate(reader, start=1):
                writer.write(_redact_line(line, needles, counts, refused, line_number))
    except BaseException:
        Path(dest).unlink(missing_ok=True)
        raise
    return FileRedaction(MappingProxyType(dict(counts)), tuple(refused))


def redact_in_place(
    path: StrPath,
    needles: Sequence[Needle],
    *,
    unchanged_since: FileStamp | None = None,
) -> FileRedaction:
    path = Path(path)
    original = os.stat(path)
    stamp = unchanged_since or _stamp_of(original)
    if not needles or not _contains_any(path, needles):
        return FileRedaction(MappingProxyType({}))

    fd, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.redacting', dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        result = copy_redacted(path, temp, needles)
        if not result.redacted:
            return result
        os.chmod(temp, stat.S_IMODE(original.st_mode))
        os.utime(temp, ns=(original.st_atime_ns, original.st_mtime_ns))
        if file_stamp(path) != stamp:
            raise FileChangedDuringRedaction(path)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return result


def _placeholder(var: str, length: int) -> bytes:
    return (f'<redacted:{var}>' + 'x' * length)[:length].encode()


def _stamp_of(st: os.stat_result) -> FileStamp:
    return FileStamp(st.st_ino, st.st_size, st.st_mtime_ns)


def _contains_any(path: Path, needles: Sequence[Needle]) -> bool:
    overlap = max(len(needle.pattern) for needle in needles) - 1
    tail = b''
    with open(path, 'rb') as reader:
        while chunk := reader.read(_DETECT_CHUNK_BYTES):
            window = tail + chunk
            if any(needle.pattern in window for needle in needles):
                return True
            tail = window[len(window) - overlap:]
    return False


def _redact_line(
    line: bytes,
    needles: Sequence[Needle],
    counts: Counter[str],
    refused: list[RefusedLine],
    line_number: int,
) -> bytes:
    redacted = line
    line_counts: Counter[str] = Counter()
    for needle in needles:
        hits = redacted.count(needle.pattern)
        if hits:
            line_counts[needle.var] += hits
            redacted = redacted.replace(needle.pattern, needle.placeholder)
    if line_counts and _is_json(line) and not _is_json(redacted):
        refused.append(RefusedLine(line_number, next(iter(line_counts))))
        return line
    counts.update(line_counts)
    return redacted


def _is_json(line: bytes) -> bool:
    try:
        json.loads(line)
    except ValueError:
        return False
    return True
