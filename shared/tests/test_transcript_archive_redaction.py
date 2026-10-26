"""Archived transcripts never hold a secret value the archiving process holds.

The archiving process's environment is the input: the values of its
secret-bearing variables are replaced by same-length ``<redacted:VAR>``
placeholders on the way into the archive, through both producers (the staged
copy and the teardown move). A transcript whose redaction is refused is
counted as an archival failure and never archived unredacted.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from shared import transcript_archive as transcript_archive_module
from shared.transcript_archive import archive_before_delete, archive_task_transcripts

ENC = '-home-leo-src-dark-factory'
SID = 'sess-redact'
TASK_ID = '6820'
POOL_A = 'sk-ant-oat01-' + 'Pa' * 17 + 'z'
OPENAI = 'sk-proj-' + 'Op' * 20
OLD_MTIME_NS = 1_700_000_000_000_000_000


def _transcript_lines() -> list[str]:
    env_dump = f'PATH=/usr/bin\nCLAUDE_OAUTH_TOKEN_A={POOL_A}\nOPENAI_API_KEY={OPENAI}\nHOME=/h\n'
    return [
        json.dumps({'type': 'user', 'message': {'content': 'run env'}}),
        json.dumps({
            'type': 'user',
            'message': {'content': [{'type': 'tool_result', 'content': env_dump}]},
        }),
        json.dumps({'type': 'assistant', 'message': {'content': 'done'}}),
    ]


def _write_transcript(config_dir: Path, lines: list[str]) -> Path:
    src = config_dir / 'projects' / ENC / f'{SID}.jsonl'
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_bytes(''.join(f'{line}\n' for line in lines).encode())
    os.utime(src, ns=(OLD_MTIME_NS, OLD_MTIME_NS))
    return src


def _archived(root: Path) -> Path:
    return root / TASK_ID / ENC / f'{SID}.jsonl'


def _archived_files(root: Path) -> list[Path]:
    return [p for p in root.rglob('*') if p.is_file()]


def _assert_redacted_copy(archived: Path, *, size: int, line_count: int) -> None:
    data = archived.read_bytes()
    assert POOL_A.encode() not in data
    assert OPENAI.encode() not in data
    assert b'<redacted:CLAUDE_OAUTH_TOKEN_A>' in data
    assert b'<redacted:OPENAI_API_KEY>' in data
    assert len(data) == size
    lines = data.splitlines()
    assert len(lines) == line_count
    for line in lines:
        json.loads(line)


@pytest.fixture
def archiving_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_A', POOL_A)
    monkeypatch.setenv('OPENAI_API_KEY', OPENAI)


@pytest.fixture
def failures() -> Iterator[list[dict]]:
    seen: list[dict] = []
    transcript_archive_module._reset_archival_failures()
    transcript_archive_module.set_archival_failure_hook(seen.append)
    yield seen
    transcript_archive_module._reset_archival_failures()


def test_the_staged_copy_is_redacted_and_the_live_source_is_not(archiving_env, tmp_path):
    config_dir = tmp_path / 'claude-config'
    root = tmp_path / 'archive'
    src = _write_transcript(config_dir, _transcript_lines())
    src_bytes = src.read_bytes()

    assert archive_task_transcripts(config_dir, TASK_ID, archive_root=root) == 1

    archived = _archived(root)
    _assert_redacted_copy(archived, size=len(src_bytes), line_count=len(_transcript_lines()))
    assert int(archived.stat().st_mtime) == int(src.stat().st_mtime)
    assert src.read_bytes() == src_bytes
    assert src.stat().st_mtime_ns == OLD_MTIME_NS
    assert archive_task_transcripts(config_dir, TASK_ID, archive_root=root) == 0


def test_the_teardown_move_is_redacted(archiving_env, tmp_path):
    config_dir = tmp_path / 'claude-config'
    root = tmp_path / 'archive'
    src = _write_transcript(config_dir, _transcript_lines())
    size = src.stat().st_size

    outcome = archive_before_delete(config_dir, TASK_ID, archive_root=root)

    archived = _archived(root)
    _assert_redacted_copy(archived, size=size, line_count=len(_transcript_lines()))
    assert archived.stat().st_mtime_ns == OLD_MTIME_NS
    assert outcome.archived == 1
    assert outcome.held == ()
    assert outcome.config_dir_removed
    assert not config_dir.exists()


def test_a_transcript_without_a_secret_is_archived_byte_identical(archiving_env, tmp_path):
    config_dir = tmp_path / 'claude-config'
    root = tmp_path / 'archive'
    clean = [json.dumps({'type': 'user', 'message': {'content': 'nothing secret'}})]
    src = _write_transcript(config_dir, clean)
    src_bytes = src.read_bytes()

    assert archive_task_transcripts(config_dir, TASK_ID, archive_root=root) == 1

    assert _archived(root).read_bytes() == src_bytes


class TestRefusedRedaction:
    VALUE = '00e9' + 'X' * 44

    def _refusable_lines(self) -> list[str]:
        return [json.dumps({'text': 'café' + 'X' * 44})]

    def test_the_staged_copy_is_counted_and_nothing_is_archived(
        self, monkeypatch, failures, tmp_path,
    ):
        monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_A', self.VALUE)
        config_dir = tmp_path / 'claude-config'
        root = tmp_path / 'archive'
        src = _write_transcript(config_dir, self._refusable_lines())

        assert archive_task_transcripts(config_dir, TASK_ID, archive_root=root) == 0

        assert transcript_archive_module._archival_failures() == 1
        assert [f['path'] for f in failures] == [str(src)]
        assert _archived_files(root) == []

    def test_the_teardown_move_holds_the_transcript(self, monkeypatch, failures, tmp_path):
        monkeypatch.setenv('CLAUDE_OAUTH_TOKEN_A', self.VALUE)
        config_dir = tmp_path / 'claude-config'
        root = tmp_path / 'archive'
        src = _write_transcript(config_dir, self._refusable_lines())
        src_bytes = src.read_bytes()

        outcome = archive_before_delete(config_dir, TASK_ID, archive_root=root)

        assert outcome.held == (src,)
        assert src.read_bytes() == src_bytes
        assert transcript_archive_module._archival_failures() == 1
        assert [f['path'] for f in failures] == [str(src)]
        assert _archived_files(root) == []
