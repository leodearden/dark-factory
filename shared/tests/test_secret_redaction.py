"""Known secret values in JSONL files are replaced by same-length placeholders.

The placeholder shape is the one the 2026-10-11 hand redaction left on disk:
``<redacted:VAR>`` right-padded with ``x`` to the replaced bytes' length.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from shared.secret_redaction import (
    FileChangedDuringRedaction,
    RedactionWouldCorruptJson,
    RefusedLine,
    copy_redacted,
    file_stamp,
    redact_in_place,
    redaction_needles,
)

VAR = 'CLAUDE_OAUTH_TOKEN_A'
VALUE = 'sk-ant-oat01-' + 'f4ke' * 9
OLD_MTIME_NS = 1_700_000_000_123_456_789
OLD_ATIME_NS = 1_700_000_100_987_654_321


def _placeholder(var: str, length: int) -> bytes:
    return (f'<redacted:{var}>' + 'x' * length)[:length].encode()


def _transcript_lines(value: str) -> list[str]:
    return [
        json.dumps({'type': 'user', 'message': {'content': 'hello'}}),
        json.dumps({
            'type': 'user',
            'message': {'content': [{'type': 'tool_result', 'content': f'the token is {value} ok'}]},
        }),
        json.dumps({
            'type': 'user',
            'message': {
                'content': [{
                    'type': 'tool_result',
                    'content': f'PATH=/usr/bin\n{VAR}={value}\nHOME=/home/x\n',
                }],
            },
        }),
        json.dumps({'type': 'assistant', 'message': {'content': 'done'}}),
    ]


def _write(path: Path, lines: list[str], *, mode: int = 0o640) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(''.join(f'{line}\n' for line in lines).encode())
    path.chmod(mode)
    os.utime(path, ns=(OLD_ATIME_NS, OLD_MTIME_NS))
    return path


def _lines(path: Path) -> list[bytes]:
    return path.read_bytes().splitlines()


def _assert_every_line_is_json(path: Path) -> None:
    for line in _lines(path):
        json.loads(line)


class TestNeedles:
    def test_placeholder_has_the_values_length_and_the_on_disk_shape(self):
        (needle,) = redaction_needles({VAR: VALUE})
        assert needle.var == VAR
        assert needle.pattern == VALUE.encode()
        assert len(needle.placeholder) == len(VALUE)
        label = f'<redacted:{VAR}>'.encode()
        assert needle.placeholder == label + b'x' * (len(VALUE) - len(label))

    def test_a_value_shorter_than_its_label_truncates_the_label(self):
        short = 'q' * 25
        (needle,) = redaction_needles({VAR: short})
        assert needle.placeholder == f'<redacted:{VAR}>'.encode()[:25]
        assert len(needle.placeholder) == len(short)

    def test_a_value_below_the_minimum_length_is_never_a_needle(self):
        assert redaction_needles({VAR: 'tok-a'}) == ()

    def test_a_value_with_json_specials_is_matched_in_its_escaped_form(self):
        value = 'abc"def\\ghi' + 'z' * 30
        escaped = json.dumps(value)[1:-1].encode()
        needles = redaction_needles({VAR: value})
        patterns = {n.pattern for n in needles}
        assert escaped in patterns
        for needle in needles:
            assert len(needle.placeholder) == len(needle.pattern)


class TestRedactInPlace:
    def test_every_occurrence_is_replaced_and_the_file_keeps_its_shape(self, tmp_path):
        path = _write(tmp_path / 's.jsonl', _transcript_lines(VALUE))
        before = path.stat()
        line_count = len(_lines(path))

        result = redact_in_place(path, redaction_needles({VAR: VALUE}))

        after = path.stat()
        data = path.read_bytes()
        assert dict(result.counts) == {VAR: 2}
        assert result.redacted
        assert VALUE.encode() not in data
        assert _placeholder(VAR, len(VALUE)) in data
        assert after.st_size == before.st_size
        assert len(_lines(path)) == line_count
        _assert_every_line_is_json(path)
        assert stat.S_IMODE(after.st_mode) == 0o640
        assert after.st_mtime_ns == OLD_MTIME_NS
        assert VALUE not in repr(result)

    def test_a_second_call_finds_nothing_and_does_not_rewrite(self, tmp_path):
        path = _write(tmp_path / 's.jsonl', _transcript_lines(VALUE))
        needles = redaction_needles({VAR: VALUE})
        redact_in_place(path, needles)
        before = path.stat()

        result = redact_in_place(path, needles)

        after = path.stat()
        assert dict(result.counts) == {}
        assert not result.redacted
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)

    def test_a_clean_file_is_not_rewritten(self, tmp_path):
        path = _write(tmp_path / 'clean.jsonl', _transcript_lines('nothing-secret-here'))
        before = path.stat()
        content = path.read_bytes()

        result = redact_in_place(path, redaction_needles({VAR: VALUE}))

        after = path.stat()
        assert dict(result.counts) == {}
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
        assert path.read_bytes() == content

    def test_a_value_straddling_any_read_boundary_is_found(self, tmp_path):
        boundaries = [2**k for k in range(12, 23)]
        body = bytearray(b'a' * (boundaries[-1] + 4096))
        for boundary in boundaries:
            start = boundary - 7
            body[start:start + len(VALUE)] = VALUE.encode()
        path = tmp_path / 'big.jsonl'
        path.write_bytes(b'{"t": "' + bytes(body) + b'"}\n')

        result = redact_in_place(path, redaction_needles({VAR: VALUE}))

        assert dict(result.counts) == {VAR: len(boundaries)}
        assert VALUE.encode() not in path.read_bytes()
        _assert_every_line_is_json(path)


def test_copy_redacted_redacts_the_destination_and_leaves_the_source(tmp_path):
    src = _write(tmp_path / 'src.jsonl', _transcript_lines(VALUE))
    dest = tmp_path / 'dest.jsonl'
    src_bytes = src.read_bytes()

    result = copy_redacted(src, dest, redaction_needles({VAR: VALUE}))

    assert dict(result.counts) == {VAR: 2}
    assert src.read_bytes() == src_bytes
    data = dest.read_bytes()
    assert VALUE.encode() not in data
    assert _placeholder(VAR, len(VALUE)) in data
    assert len(data) == len(src_bytes)
    _assert_every_line_is_json(dest)


class TestRefusedLine:
    VALUE = '00e9' + 'X' * 40

    def _lines(self) -> list[str]:
        lines = [
            json.dumps({'text': f'plain {self.VALUE} occurrence'}),
            json.dumps({'text': 'café' + 'X' * 40}),
            json.dumps({'text': f'another {self.VALUE}'}),
        ]
        assert '\\u00e9' + 'X' * 40 in lines[1], 'the value must sit inside a \\u escape'
        return lines

    def _needles(self):
        return redaction_needles({VAR: self.VALUE})

    def test_only_the_line_whose_rewrite_would_break_json_keeps_its_value(self, tmp_path):
        path = _write(tmp_path / 'held' / 's.jsonl', self._lines())
        original = _lines(path)

        result = redact_in_place(path, self._needles())

        after = _lines(path)
        assert result.refused == (RefusedLine(2, VAR),)
        assert dict(result.counts) == {VAR: 2}
        assert after[1] == original[1]
        assert self.VALUE.encode() not in after[0] + after[2]
        assert len(path.read_bytes()) == sum(len(line) + 1 for line in original)
        _assert_every_line_is_json(path)
        assert list(path.parent.iterdir()) == [path]

    def test_a_file_whose_only_hit_is_refused_is_not_rewritten(self, tmp_path):
        path = _write(tmp_path / 'held' / 's.jsonl', [self._lines()[1]])
        before = path.stat()
        content = path.read_bytes()

        result = redact_in_place(path, self._needles())

        after = path.stat()
        assert result.refused == (RefusedLine(1, VAR),)
        assert not result.redacted
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
        assert path.read_bytes() == content
        assert list(path.parent.iterdir()) == [path]

    def test_copy_redacted_keeps_the_refused_line_and_redacts_the_rest(self, tmp_path):
        src = _write(tmp_path / 'src.jsonl', self._lines())
        dest = tmp_path / 'dest.jsonl'

        result = copy_redacted(src, dest, self._needles())

        assert result.refused == (RefusedLine(2, VAR),)
        assert dict(result.counts) == {VAR: 2}
        copied = _lines(dest)
        assert copied[1] == _lines(src)[1]
        assert self.VALUE.encode() not in copied[0] + copied[2]

    def test_raise_if_refused_names_the_path_line_and_variable_but_never_the_value(self, tmp_path):
        path = _write(tmp_path / 'held' / 's.jsonl', self._lines())
        result = redact_in_place(path, self._needles())

        with pytest.raises(RedactionWouldCorruptJson) as excinfo:
            result.raise_if_refused(path)

        error = excinfo.value
        message = str(error)
        assert (error.path, error.line_number, error.var) == (path, 2, VAR)
        assert str(path) in message
        assert 'line 2' in message
        assert VAR in message
        assert self.VALUE not in message

    def test_raise_if_refused_is_silent_for_a_complete_redaction(self, tmp_path):
        path = _write(tmp_path / 's.jsonl', _transcript_lines(VALUE))

        redact_in_place(path, redaction_needles({VAR: VALUE})).raise_if_refused(path)


def test_a_file_changed_since_the_stamp_is_left_untouched(tmp_path):
    path = _write(tmp_path / 'live' / 's.jsonl', _transcript_lines(VALUE))
    stamp = file_stamp(path)
    appended = json.dumps({'type': 'assistant', 'message': {'content': 'late append'}}) + '\n'
    with path.open('ab') as handle:
        handle.write(appended.encode())
    content = path.read_bytes()

    with pytest.raises(FileChangedDuringRedaction):
        redact_in_place(path, redaction_needles({VAR: VALUE}), unchanged_since=stamp)

    assert path.read_bytes() == content
    assert content.endswith(appended.encode())
    assert list(path.parent.iterdir()) == [path]
