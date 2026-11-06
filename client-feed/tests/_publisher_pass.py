"""A project the publisher answers reveal requests for, and what a pass leaves for the reveal tests to read back."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Final

from _git_repo import make_repo
from _stores import EscalationQueueDir, RunsDb, TasksDb

from client_feed.contract import FileRevealed, RevealRefused
from client_feed.feed_io import CURSOR_FILENAME, Cursor, read_feed_lines
from client_feed.policy import Policy

REVEALABLE_PATH: Final = 'src/ledger.py'
_REVEALABLE_TEXT: Final = ''.join(f'entry_{index} = {index}\n' for index in range(60))


def reveal_policy(tmp_path: Path, in_dir: Path, *, project_id: str, committed_at: datetime) -> Policy:
    """A policy over a fresh repo whose one committed file, REVEALABLE_PATH, is visible, above the reveal floor and within budget."""
    repo = make_repo(tmp_path / 'repo')
    repo.write('.gitignore', '.taskmaster/\ndata/\n')
    repo.write(REVEALABLE_PATH, _REVEALABLE_TEXT)
    repo.commit('c1', when=committed_at)
    TasksDb.create(repo.root)
    RunsDb.create(repo.root)
    EscalationQueueDir.create(repo.root)
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    return Policy.model_validate(
        {
            'project_id': project_id,
            'project_root': str(repo.root),
            'out_dir': str(out_dir),
            'in_dir': str(in_dir),
            'visible_globs': ['src/**'],
            'reveal_proportion': 1.0,
        }
    )


def reveal_answers(out_dir: Path) -> list[FileRevealed | RevealRefused]:
    return [line for line in read_feed_lines(out_dir) if isinstance(line, FileRevealed | RevealRefused)]


def inbox_offset(out_dir: Path) -> int:
    return Cursor.model_validate_json((out_dir / CURSOR_FILENAME).read_text()).inbox_offset
