"""Replace known secret values in log files with same-length placeholders.

Callers pass ``{var: value}``; which variables hold secrets is
``shared.secret_env``'s question, not this module's. A value is matched in the
escaped form it takes in the file's ``Markup``: JSON-string-escaped inside a
JSONL line, raw or JSON-string-escaped inside a plain text log, or XML-escaped
in a report's text and attribute values. It is replaced by
``<redacted:VAR>`` right-padded with ``x`` to the same byte length, so a file
keeps its size and line count; in XML the label is entity-escaped
(``&lt;redacted:VAR&gt;``) and never cut mid-entity, so a report stays
well-formed and still parses as ``<redacted:VAR>``. A ``Codec`` says how the
bytes are stored: a gzip file is decompressed, redacted and recompressed, and
one that does not decompress whole raises ``CorruptCompressedFile`` and is left
as found. A line whose rewrite would
turn it from valid JSON into invalid JSON is left as it was and reported in
``FileRedaction.refused``; every other line is still redacted. No message ever
carries a value.
"""

from __future__ import annotations

import contextlib
import enum
import gzip
import io
import json
import os
import stat
import tempfile
import zlib
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from xml.sax.saxutils import escape as xml_escape

__all__ = [
    'MIN_REDACTABLE_LEN',
    'Codec',
    'CorruptCompressedFile',
    'FileChangedDuringRedaction',
    'FileRedaction',
    'FileStamp',
    'Markup',
    'Needle',
    'RedactionError',
    'RedactionWouldCorruptJson',
    'RefusedLine',
    'file_stamp',
    'redact_in_place',
    'redaction_needles',
]

MIN_REDACTABLE_LEN = 20

_DETECT_CHUNK_BYTES = 1 << 20

StrPath = str | os.PathLike[str]

_XML_QUOTE_ENTITIES = MappingProxyType({'"': '&quot;', "'": '&apos;'})
_XML_ATTRIBUTE_ENTITIES = MappingProxyType({'"': '&quot;', '\r': '&#13;', '\n': '&#10;', '\t': '&#09;'})


class Markup(enum.Enum):
    JSON = 'json'
    TEXT = 'text'
    XML = 'xml'


class Codec(enum.Enum):
    PLAIN = 'plain'
    GZIP = 'gzip'


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


class CorruptCompressedFile(RedactionError):
    def __init__(self, path: Path) -> None:
        super().__init__(f'{path} does not decompress as whole gzip data; it was left as found')
        self.path = path


def redaction_needles(
    secrets: Mapping[str, str], *, markup: Markup = Markup.JSON
) -> tuple[Needle, ...]:
    rules = _MARKUP_RULES[markup]
    by_pattern: dict[bytes, Needle] = {}
    for var, value in sorted(secrets.items()):
        if len(value) < MIN_REDACTABLE_LEN:
            continue
        for form in rules.forms(value):
            pattern = form.encode()
            by_pattern.setdefault(pattern, Needle(var, pattern, rules.placeholder(var, len(pattern))))
    return tuple(sorted(by_pattern.values(), key=lambda needle: -len(needle.pattern)))


def file_stamp(path: StrPath) -> FileStamp:
    return _stamp_of(os.stat(path))


def redact_in_place(
    path: StrPath,
    needles: Sequence[Needle],
    *,
    codec: Codec = Codec.PLAIN,
    unchanged_since: FileStamp | None = None,
) -> FileRedaction:
    path = Path(path)
    original = os.stat(path)
    stamp = unchanged_since or _stamp_of(original)
    if not needles or not _contains_any(path, needles, codec):
        return FileRedaction(MappingProxyType({}))

    fd, temp_name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.redacting', dir=path.parent)
    os.close(fd)
    temp = Path(temp_name)
    try:
        with (
            _decoded(path, codec) as reader,
            _encoded(temp, codec, name=path.name, mtime=int(original.st_mtime)) as writer,
        ):
            result = _copy_lines(reader, writer, needles)
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


def _json_escaped_forms(value: str) -> tuple[str, ...]:
    return tuple(json.dumps(value, ensure_ascii=ascii_only)[1:-1] for ascii_only in (True, False))


def _raw_and_json_escaped_forms(value: str) -> tuple[str, ...]:
    return (value, *_json_escaped_forms(value))


def _xml_escaped_forms(value: str) -> tuple[str, ...]:
    return (
        value,
        xml_escape(value),
        xml_escape(value, _XML_QUOTE_ENTITIES),
        xml_escape(value, _XML_ATTRIBUTE_ENTITIES),
    )


def _text_placeholder(var: str, length: int) -> bytes:
    return (f'<redacted:{var}>' + 'x' * length)[:length].encode()


def _xml_placeholder(var: str, length: int) -> bytes:
    label = f'&lt;redacted:{var}&gt;'
    if len(label) > length:
        label = f'&lt;redacted:{var}'
    return (label + 'x' * length)[:length].encode()


@dataclass(frozen=True)
class _MarkupRules:
    forms: Callable[[str], tuple[str, ...]]
    placeholder: Callable[[str, int], bytes]


_MARKUP_RULES: Mapping[Markup, _MarkupRules] = MappingProxyType({
    Markup.JSON: _MarkupRules(_json_escaped_forms, _text_placeholder),
    Markup.TEXT: _MarkupRules(_raw_and_json_escaped_forms, _text_placeholder),
    Markup.XML: _MarkupRules(_xml_escaped_forms, _xml_placeholder),
})


def _stamp_of(st: os.stat_result) -> FileStamp:
    return FileStamp(st.st_ino, st.st_size, st.st_mtime_ns)


@dataclass(frozen=True)
class _CodecRules:
    open_reader: Callable[[Path], io.BufferedIOBase]
    wrap_writer: Callable[[io.BufferedIOBase, str, int], io.BufferedIOBase]
    decode_errors: tuple[type[Exception], ...]


def _gzip_writer(raw: io.BufferedIOBase, name: str, mtime: int) -> io.BufferedIOBase:
    return gzip.GzipFile(filename=name, mode='wb', fileobj=raw, mtime=mtime)


_CODEC_RULES: Mapping[Codec, _CodecRules] = MappingProxyType({
    Codec.PLAIN: _CodecRules(
        open_reader=lambda path: path.open('rb'),
        wrap_writer=lambda raw, _name, _mtime: raw,
        decode_errors=(),
    ),
    Codec.GZIP: _CodecRules(
        open_reader=lambda path: gzip.GzipFile(path, 'rb'),
        wrap_writer=_gzip_writer,
        decode_errors=(EOFError, zlib.error, gzip.BadGzipFile),
    ),
})


@contextlib.contextmanager
def _decoded(path: Path, codec: Codec) -> Iterator[io.BufferedIOBase]:
    rules = _CODEC_RULES[codec]
    try:
        with rules.open_reader(path) as reader:
            yield reader
    except rules.decode_errors as exc:
        raise CorruptCompressedFile(path) from exc


@contextlib.contextmanager
def _encoded(temp: Path, codec: Codec, *, name: str, mtime: int) -> Iterator[io.BufferedIOBase]:
    with open(temp, 'wb') as raw, _CODEC_RULES[codec].wrap_writer(raw, name, mtime) as writer:
        yield writer


def _contains_any(path: Path, needles: Sequence[Needle], codec: Codec) -> bool:
    overlap = max(len(needle.pattern) for needle in needles) - 1
    tail = b''
    with _decoded(path, codec) as reader:
        while chunk := reader.read(_DETECT_CHUNK_BYTES):
            window = tail + chunk
            if any(needle.pattern in window for needle in needles):
                return True
            tail = window[len(window) - overlap:]
    return False


def _copy_lines(
    reader: io.BufferedIOBase, writer: io.BufferedIOBase, needles: Sequence[Needle]
) -> FileRedaction:
    counts: Counter[str] = Counter()
    refused: list[RefusedLine] = []
    for line_number, line in enumerate(reader, start=1):
        writer.write(_redact_line(line, needles, counts, refused, line_number))
    return FileRedaction(MappingProxyType(dict(counts)), tuple(refused))


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
