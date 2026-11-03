"""Tests for the single-thread dead-letter appender (task 6085, INV-8)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from _dead_letter_helpers import hold_fifo

from fused_memory.reconciliation.dead_letter_writer import DeadLetterWriter

_WRITER_LOGGER = 'fused_memory.reconciliation.dead_letter_writer'


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_appends_land_in_submission_order_and_flush_waits_for_them(tmp_path: Path):
    path = tmp_path / 'dl.jsonl'
    writer = DeadLetterWriter(path)
    records = [{'n': i, 'reason': f'r{i}'} for i in range(5)]

    futures = [writer.append(record) for record in records]
    writer.flushed().result(timeout=5)

    assert _read_records(path) == records
    assert [f.result(timeout=0) for f in futures] == [True] * 5
    assert writer.records_written == 5


def test_append_returns_before_the_write_lands(tmp_path: Path):
    held = hold_fifo(tmp_path / 'dl.jsonl')
    writer = DeadLetterWriter(held.path)
    record = {'event': {'id': 'e1'}, 'reason': 'overflow_drop'}

    written = writer.append(record)

    assert not written.done()
    assert not held.reader_opened.is_set(), (
        'append() only returned after the FIFO reader opened: the write ran '
        'on the caller thread'
    )
    assert writer.records_pending == 1
    held.release.set()
    assert held.records() == [record]
    assert written.result(timeout=5) is True
    assert writer.records_pending == 0


def test_a_failed_write_is_logged_and_resolves_false(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
):
    (tmp_path / 'blocker').write_text('a regular file, not a directory')
    writer = DeadLetterWriter(tmp_path / 'blocker' / 'dl.jsonl')

    with caplog.at_level(logging.ERROR, logger=_WRITER_LOGGER):
        written = writer.append({'event': {'id': 'e1'}})
        assert written.result(timeout=5) is False

    assert writer.records_written == 0
    assert writer.records_pending == 0
    assert any(
        rec.name == _WRITER_LOGGER and rec.levelno == logging.ERROR
        for rec in caplog.records
    ), caplog.records


def test_an_accepted_append_cannot_be_cancelled(tmp_path: Path):
    held = hold_fifo(tmp_path / 'dl.jsonl')
    writer = DeadLetterWriter(held.path)
    record = {'event': {'id': 'e1'}, 'reason': 'shutdown_timeout'}

    written = writer.append(record)

    assert written.cancel() is False
    held.release.set()
    assert held.records() == [record]
    assert written.result(timeout=5) is True


def test_an_unused_writer_is_already_flushed(tmp_path: Path):
    assert DeadLetterWriter(tmp_path / 'dl.jsonl').flushed().done()


def test_paths_list_the_current_file_then_each_rotation(tmp_path: Path):
    path = tmp_path / 'dl.jsonl'

    assert DeadLetterWriter(path, keep_rotations=2).paths() == [
        path,
        tmp_path / 'dl.jsonl.1',
        tmp_path / 'dl.jsonl.2',
    ]
    assert DeadLetterWriter(path, keep_rotations=0).paths() == [path]
