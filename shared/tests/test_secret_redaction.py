"""Known secret values in JSONL files are replaced by same-length placeholders.

The placeholder shape is the one the 2026-10-11 hand redaction left on disk:
``<redacted:VAR>`` right-padded with ``x`` to the replaced bytes' length.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import stat
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import pytest

from shared.secret_redaction import (
    MIN_REDACTABLE_LEN,
    MIN_TRUNCATED_PREFIX_LEN,
    Codec,
    CorruptCompressedFile,
    FileChangedDuringRedaction,
    Markup,
    RedactionWouldCorruptJson,
    RefusedLine,
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


def _junit_lines(value: str) -> list[str]:
    suite = ET.Element('testsuite', name='orchestrator', tests='1', failures='1')
    case = ET.SubElement(suite, 'testcase', classname='tests.test_env', name='test_env_is_scrubbed')
    failure = ET.SubElement(case, 'failure', message=f'AssertionError: leaked {value} here')
    failure.text = f"E   AssertionError: {{'{VAR}': '{value}', 'PATH': '/usr/bin'}}"
    ET.indent(suite)
    return ET.tostring(suite, encoding='unicode', xml_declaration=True).splitlines()


def _parsed_failure(data: bytes) -> ET.Element:
    failure = ET.fromstring(data).find('testcase/failure')
    assert failure is not None
    return failure


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


class TestXmlMarkup:
    SPECIALS = 'tok&en<"q\'>-' + 'z' * 30

    @pytest.mark.parametrize('value', [VALUE, SPECIALS])
    def test_every_placeholder_keeps_its_patterns_byte_length(self, value):
        needles = redaction_needles({VAR: value}, markup=Markup.XML)

        assert needles
        for needle in needles:
            assert len(needle.placeholder) == len(needle.pattern)

    def test_a_redacted_junit_report_still_parses_and_reads_as_the_redaction_label(self, tmp_path):
        path = _write(tmp_path / 'r.xml', _junit_lines(VALUE))
        before = path.stat()
        line_count = len(_lines(path))

        result = redact_in_place(path, redaction_needles({VAR: VALUE}, markup=Markup.XML))

        after = path.stat()
        data = path.read_bytes()
        assert dict(result.counts) == {VAR: 2}
        assert VALUE.encode() not in data
        assert after.st_size == before.st_size
        assert len(_lines(path)) == line_count
        assert stat.S_IMODE(after.st_mode) == 0o640
        assert after.st_mtime_ns == OLD_MTIME_NS
        failure = _parsed_failure(data)
        label = f'<redacted:{VAR}>'
        assert label in failure.get('message', '')
        assert label in (failure.text or '')

    @pytest.mark.parametrize(
        'length', range(MIN_REDACTABLE_LEN, len(f'&lt;redacted:{VAR}&gt;') + 1)
    )
    def test_a_value_too_short_for_the_label_never_truncates_mid_entity(self, tmp_path, length):
        value = 'q' * length
        path = _write(tmp_path / 'r.xml', _junit_lines(value))

        result = redact_in_place(path, redaction_needles({VAR: value}, markup=Markup.XML))

        assert dict(result.counts) == {VAR: 2}
        failure = _parsed_failure(path.read_bytes())
        assert value not in failure.get('message', '')
        assert value not in (failure.text or '')

    def test_a_value_with_an_xml_special_is_matched_in_its_escaped_form(self, tmp_path):
        value = 'tok&en-' + 'z' * 30
        lines = _junit_lines(value)
        escaped = value.replace('&', '&amp;')
        assert sum(line.count(escaped) for line in lines) == 2, 'ElementTree escapes & both times'
        path = _write(tmp_path / 'r.xml', lines)

        result = redact_in_place(path, redaction_needles({VAR: value}, markup=Markup.XML))

        data = path.read_bytes()
        assert dict(result.counts) == {VAR: 2}
        assert escaped.encode() not in data
        failure = _parsed_failure(data)
        assert value not in failure.get('message', '')
        assert value not in (failure.text or '')

    @pytest.mark.parametrize('value', [SPECIALS, 'tok"en' + 'z' * 30, "tok'en" + 'z' * 30])
    def test_a_value_elementtree_wrote_into_an_attribute_and_text_is_matched_in_both(self, tmp_path, value):
        path = _write(tmp_path / 'r.xml', _junit_lines(value))

        result = redact_in_place(path, redaction_needles({VAR: value}, markup=Markup.XML))

        assert dict(result.counts) == {VAR: 2}
        failure = _parsed_failure(path.read_bytes())
        assert value not in failure.get('message', '')
        assert value not in (failure.text or '')

    def test_a_value_with_whitespace_elementtree_escapes_in_an_attribute_is_matched_there(self, tmp_path):
        value = 'tok\ten\nline\r-' + 'z' * 30
        suite = ET.Element('testsuite', name='orchestrator')
        ET.SubElement(ET.SubElement(suite, 'testcase', name='t'), 'failure', message=f'leaked {value}')
        path = _write(tmp_path / 'r.xml', [ET.tostring(suite, encoding='unicode')])

        result = redact_in_place(path, redaction_needles({VAR: value}, markup=Markup.XML))

        assert dict(result.counts) == {VAR: 1}
        assert value not in _parsed_failure(path.read_bytes()).get('message', '')


class TestJsonAndTextMarkup:
    SPECIALS = 'ab"cd\\ef' + 'z' * 30

    def test_json_markup_is_the_default_and_keeps_the_raw_label(self):
        default = redaction_needles({VAR: VALUE})

        assert default == redaction_needles({VAR: VALUE}, markup=Markup.JSON)
        assert all(needle.placeholder.startswith(b'<redacted:') for needle in default)

    def test_text_markup_matches_the_raw_value_a_plain_log_holds(self, tmp_path):
        path = _write(tmp_path / 'detect.log', [f'E   AssertionError: token={self.SPECIALS}', '1 failed'])

        result = redact_in_place(path, redaction_needles({VAR: self.SPECIALS}, markup=Markup.TEXT))

        assert dict(result.counts) == {VAR: 1}
        assert self.SPECIALS.encode() not in path.read_bytes()
        assert _placeholder(VAR, len(self.SPECIALS)) in path.read_bytes()

    def test_text_markup_still_matches_a_json_escaped_value_a_log_line_embeds(self, tmp_path):
        path = _write(tmp_path / 'detect.log', [f'payload: {json.dumps({"token": self.SPECIALS})}'])

        result = redact_in_place(path, redaction_needles({VAR: self.SPECIALS}, markup=Markup.TEXT))

        assert dict(result.counts) == {VAR: 1}
        assert json.dumps(self.SPECIALS)[1:-1].encode() not in path.read_bytes()

    def test_json_markup_does_not_match_the_raw_value(self, tmp_path):
        path = _write(tmp_path / 'detect.log', [f'token={self.SPECIALS}'])

        assert dict(redact_in_place(path, redaction_needles({VAR: self.SPECIALS})).counts) == {}

    def test_a_value_without_specials_has_one_pattern_in_every_markup(self):
        for markup in Markup:
            assert [needle.pattern for needle in redaction_needles({VAR: VALUE}, markup=markup)] == [VALUE.encode()]


def _write_gzip(path: Path, lines: list[str], *, mode: int = 0o640) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(''.join(f'{line}\n' for line in lines).encode()))
    path.chmod(mode)
    os.utime(path, ns=(OLD_ATIME_NS, OLD_MTIME_NS))
    return path


def _filler_lines(count: int) -> list[str]:
    return [hashlib.sha256(str(i).encode()).hexdigest() for i in range(count)]


class TestGzip:
    def _xml_needles(self):
        return redaction_needles({VAR: VALUE}, markup=Markup.XML)

    def test_a_gzipped_report_is_redacted_through_decompress_and_recompress(self, tmp_path):
        path = _write_gzip(tmp_path / 'r.xml.gz', _junit_lines(VALUE))
        original = gzip.decompress(path.read_bytes())

        result = redact_in_place(path, self._xml_needles(), codec=Codec.GZIP)

        after = path.stat()
        data = gzip.decompress(path.read_bytes())
        assert dict(result.counts) == {VAR: 2}
        assert VALUE.encode() not in data
        assert len(data) == len(original)
        assert len(data.splitlines()) == len(original.splitlines())
        failure = _parsed_failure(data)
        assert f'<redacted:{VAR}>' in failure.get('message', '')
        assert stat.S_IMODE(after.st_mode) == 0o640
        assert after.st_mtime_ns == OLD_MTIME_NS
        assert list(path.parent.iterdir()) == [path]

    def test_a_second_gzip_call_finds_nothing_and_does_not_rewrite(self, tmp_path):
        path = _write_gzip(tmp_path / 'r.xml.gz', _junit_lines(VALUE))
        redact_in_place(path, self._xml_needles(), codec=Codec.GZIP)
        before = path.stat()

        result = redact_in_place(path, self._xml_needles(), codec=Codec.GZIP)

        after = path.stat()
        assert dict(result.counts) == {}
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)

    def test_a_clean_gzip_file_is_not_rewritten(self, tmp_path):
        path = _write_gzip(tmp_path / 'clean.xml.gz', _junit_lines('nothing-secret-here'))
        before = path.stat()
        content = path.read_bytes()

        result = redact_in_place(path, self._xml_needles(), codec=Codec.GZIP)

        after = path.stat()
        assert dict(result.counts) == {}
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
        assert path.read_bytes() == content

    def test_a_value_straddling_a_decompressed_read_boundary_is_found(self, tmp_path):
        boundaries = [2**k for k in range(12, 23)]
        body = bytearray(b'a' * (boundaries[-1] + 4096))
        for boundary in boundaries:
            start = boundary - 7
            body[start:start + len(VALUE)] = VALUE.encode()
        path = tmp_path / 'big.jsonl.gz'
        path.write_bytes(gzip.compress(b'{"t": "' + bytes(body) + b'"}\n'))

        result = redact_in_place(path, redaction_needles({VAR: VALUE}), codec=Codec.GZIP)

        data = gzip.decompress(path.read_bytes())
        assert dict(result.counts) == {VAR: len(boundaries)}
        assert VALUE.encode() not in data
        json.loads(data)

    def test_a_gzip_file_changed_since_the_stamp_is_left_untouched(self, tmp_path):
        path = _write_gzip(tmp_path / 'live' / 'r.xml.gz', _junit_lines(VALUE))
        stamp = file_stamp(path)
        path.write_bytes(gzip.compress(''.join(f'{line}\n' for line in [
            *_junit_lines(VALUE), '<!-- rewritten -->',
        ]).encode()))
        content = path.read_bytes()

        with pytest.raises(FileChangedDuringRedaction):
            redact_in_place(path, self._xml_needles(), codec=Codec.GZIP, unchanged_since=stamp)

        assert path.read_bytes() == content
        assert list(path.parent.iterdir()) == [path]

    @pytest.mark.parametrize('holds_value', [True, False], ids=['value-in-prefix', 'no-value'])
    def test_a_truncated_gzip_member_raises_corrupt_compressed_file_and_is_left_as_found(
        self, tmp_path, holds_value
    ):
        head = [f'{VAR}={VALUE}'] if holds_value else ['clean head']
        path = _write_gzip(tmp_path / 'cut' / 'r.xml.gz', [*head, *_filler_lines(2000)])
        whole = path.read_bytes()
        path.write_bytes(whole[: len(whole) // 2])
        if holds_value:
            prefix = zlib.decompressobj(wbits=31).decompress(path.read_bytes())
            assert VALUE.encode() in prefix, 'fixture premise: the value is in the decodable prefix'
        content = path.read_bytes()

        with pytest.raises(CorruptCompressedFile) as excinfo:
            redact_in_place(path, redaction_needles({VAR: VALUE}), codec=Codec.GZIP)

        message = str(excinfo.value)
        assert str(path) in message
        assert VALUE not in message
        assert path.read_bytes() == content
        assert list(path.parent.iterdir()) == [path]


class TestTruncatedPrefix:
    """A log that cut a value short (``repr`` of an environ, a pytest assertion
    diff) still leaks its head; a prefix followed by a truncation marker is a hit."""

    VAR = 'ANTHROPIC_API_KEY'
    VALUE = 'sk-ant-api03-' + hashlib.sha512(b'a live anthropic key').hexdigest()[:95]
    PREFIX = VALUE[:40]

    def _needles(self, markup: Markup = Markup.TEXT, value: str | None = None):
        return redaction_needles({self.VAR: value or self.VALUE}, markup=markup)

    def _redacted_line(self, line: str, secret: str) -> bytes:
        return line.encode().replace(secret.encode(), _placeholder(self.VAR, len(secret.encode())))

    def test_the_value_is_the_length_of_a_live_anthropic_key(self):
        assert len(self.VALUE) == 108

    def test_the_assertion_repr_a_verify_log_held_is_redacted_to_the_same_length(self, tmp_path):
        truncated = self.VALUE[:95]
        line = f"E   AssertionError: {{'{self.VAR}': '{truncated}...', 'PATH': '/usr/bin'}}"
        path = _write(tmp_path / 'remote-laptop.test-x.log', [line, '1 failed'])
        before = path.stat()

        result = redact_in_place(path, self._needles())

        after = path.stat()
        assert dict(result.counts) == {self.VAR: 1}
        assert _lines(path) == [self._redacted_line(line, truncated), b'1 failed']
        assert truncated.encode() not in path.read_bytes()
        assert after.st_size == before.st_size
        assert stat.S_IMODE(after.st_mode) == 0o640
        assert after.st_mtime_ns == OLD_MTIME_NS

    @pytest.mark.parametrize(
        'marker', ['...', '…', "'", '"', ')', ']', '}', '>', '\\u2026', '\\"'],
    )
    def test_a_text_prefix_followed_by_a_truncation_marker_is_redacted(self, tmp_path, marker):
        line = f'token={self.PREFIX}{marker} tail'
        path = _write(tmp_path / 'r.log', [line])

        result = redact_in_place(path, self._needles())

        assert dict(result.counts) == {self.VAR: 1}
        assert _lines(path) == [self._redacted_line(line, self.PREFIX)]

    @pytest.mark.parametrize(
        ('ellipsis', 'ensure_ascii'), [('...', True), ('…', True), ('…', False)],
        ids=['dots', 'escaped-ellipsis', 'raw-ellipsis'],
    )
    def test_a_jsonl_prefix_followed_by_an_ellipsis_is_redacted_and_stays_json(
        self, tmp_path, ellipsis, ensure_ascii
    ):
        line = json.dumps({'t': f"{{'K': '{self.PREFIX}{ellipsis}'}}"}, ensure_ascii=ensure_ascii)
        path = _write(tmp_path / 's.jsonl', [line])

        result = redact_in_place(path, self._needles(Markup.JSON))

        (redacted,) = _lines(path)
        assert dict(result.counts) == {self.VAR: 1}
        assert self.PREFIX.encode() not in redacted
        assert len(redacted) == len(line.encode())
        assert f'<redacted:{self.VAR}>' in json.loads(redacted)['t']

    def test_a_junit_prefix_followed_by_an_ellipsis_is_redacted_and_still_parses(self, tmp_path):
        path = _write(tmp_path / 'r.xml', _junit_lines(f'{self.PREFIX}...'))
        before = path.stat()

        result = redact_in_place(path, self._needles(Markup.XML))

        data = path.read_bytes()
        assert dict(result.counts) == {self.VAR: 2}
        assert self.PREFIX.encode() not in data
        assert path.stat().st_size == before.st_size
        failure = _parsed_failure(data)
        label = f'<redacted:{self.VAR}>'
        assert label in failure.get('message', '')
        assert label in (failure.text or '')
        assert self.PREFIX not in failure.get('message', '') + (failure.text or '')

    def test_a_prefix_of_the_minimum_length_is_redacted(self, tmp_path):
        head = self.VALUE[:MIN_TRUNCATED_PREFIX_LEN]
        line = f'x={head}... end'
        path = _write(tmp_path / 'r.log', [line])

        result = redact_in_place(path, self._needles())

        assert MIN_TRUNCATED_PREFIX_LEN == 16
        assert dict(result.counts) == {self.VAR: 1}
        assert _lines(path) == [self._redacted_line(line, head)]

    @pytest.mark.parametrize(
        'text',
        [
            f'other={VALUE[:20]}{"Z" * 30}... done',
            f'x={VALUE[:30]}Q rest',
            f'x={VALUE[:30]} rest',
            f'x={VALUE[:MIN_TRUNCATED_PREFIX_LEN - 1]}... end',
        ],
        ids=['lookalike-token', 'prefix-then-letter', 'prefix-then-space', 'below-minimum'],
    )
    def test_a_near_miss_is_left_byte_identical(self, tmp_path, text):
        path = _write(tmp_path / 'r.log', [text])
        before = path.stat()
        content = path.read_bytes()

        result = redact_in_place(path, self._needles())

        after = path.stat()
        assert dict(result.counts) == {}
        assert path.read_bytes() == content
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
        assert list(path.parent.iterdir()) == [path]

    def test_a_whole_value_followed_by_an_ellipsis_counts_once(self, tmp_path):
        line = f'x={self.VALUE}... end'
        path = _write(tmp_path / 'r.log', [line])

        result = redact_in_place(path, self._needles())

        assert dict(result.counts) == {self.VAR: 1}
        assert _lines(path) == [self._redacted_line(line, self.VALUE)]

    def test_the_longest_span_wins_where_a_values_forms_overlap(self, tmp_path):
        quoted = 'sk-ant-api03-abcde"' + hashlib.sha256(b'quoted').hexdigest()
        assert quoted.index('"') == 18
        escaped = json.dumps(quoted)[1:-1]
        whole_escaped = f'payload: {json.dumps({"token": quoted})}'
        truncated_raw = f"E   {{'K': '{quoted[:40]}...'}}"
        path = _write(tmp_path / 'r.log', [whole_escaped, truncated_raw])

        result = redact_in_place(path, self._needles(value=quoted))

        assert dict(result.counts) == {self.VAR: 2}
        assert _lines(path) == [
            self._redacted_line(whole_escaped, escaped),
            self._redacted_line(truncated_raw, quoted[:40]),
        ]
        assert quoted[:MIN_TRUNCATED_PREFIX_LEN].encode() not in path.read_bytes()

    def test_a_prefix_inside_a_gzipped_junit_report_is_redacted_and_the_archive_stays_valid(
        self, tmp_path
    ):
        path = _write_gzip(tmp_path / 'r.xml.gz', _junit_lines(f'{self.PREFIX}...'))
        original = gzip.decompress(path.read_bytes())

        result = redact_in_place(path, self._needles(Markup.XML), codec=Codec.GZIP)

        after = path.stat()
        data = gzip.decompress(path.read_bytes())
        assert dict(result.counts) == {self.VAR: 2}
        assert self.PREFIX.encode() not in data
        assert len(data) == len(original)
        assert f'<redacted:{self.VAR}>' in _parsed_failure(data).get('message', '')
        assert stat.S_IMODE(after.st_mode) == 0o640
        assert after.st_mtime_ns == OLD_MTIME_NS
        assert list(path.parent.iterdir()) == [path]

    def test_no_repr_carries_the_value_or_a_prefix_of_it(self, tmp_path):
        path = _write(tmp_path / 'r.log', [f'x={self.PREFIX}... end'])
        head = self.VALUE[:MIN_TRUNCATED_PREFIX_LEN]

        result = redact_in_place(path, self._needles())

        assert result.redacted
        assert head not in repr(result)
        for markup in Markup:
            for needle in self._needles(markup):
                assert head not in repr(needle)

    def test_a_refused_prefix_hit_names_the_variable_but_never_the_value(self, tmp_path):
        value = '00e9' + 'X' * 40
        line = json.dumps({'text': 'café' + 'X' * 20 + '...'})
        assert '\\u00e9' + 'X' * 20 + '...' in line, 'the prefix must sit inside a \\u escape'
        path = _write(tmp_path / 's.jsonl', [line])
        content = path.read_bytes()

        result = redact_in_place(path, redaction_needles({VAR: value}))

        assert result.refused == (RefusedLine(1, VAR),)
        assert not result.redacted
        assert path.read_bytes() == content
        with pytest.raises(RedactionWouldCorruptJson) as excinfo:
            result.raise_if_refused(path)
        assert value[:MIN_TRUNCATED_PREFIX_LEN] not in str(excinfo.value)
        assert VAR in str(excinfo.value)
