"""Tests for the digest's window-landing count, read from the quality-dashboard landing datum (task 8563)."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest
from _landing_history import create_runs_db, landing_repo
from shared.quality_series.git_lines import GitUnavailable
from shared.testing_landing_history import LandingRepo, record_merge_done, record_train

import orchestrator.digest as digest

WINDOW_START = '2026-09-25T05:30:00+00:00'
WINDOW_END = '2026-09-26T05:30:00+00:00'
IN_WINDOW = '2026-09-25T18:00:00+00:00'
LATER_IN_WINDOW = '2026-09-25T19:00:00+00:00'
BEFORE_WINDOW = '2026-09-20T18:00:00+00:00'


@dataclass(frozen=True)
class Scene:
    repo: LandingRepo
    runs_db: Path


def _completed(conn: sqlite3.Connection, task_id: str, at: str) -> None:
    conn.execute(
        'INSERT INTO events (timestamp, run_id, task_id, event_type, data) '
        "VALUES (?, 'r1', ?, 'task_completed', ?)",
        (at, task_id, json.dumps({'outcome': 'done'})),
    )


@pytest.fixture
def scene(tmp_path: Path) -> Scene:
    repo = landing_repo(tmp_path / 'repo')
    before = repo.fast_forward('landed before the window')
    fast_forward = repo.fast_forward('ff')
    train = repo.merge('task/t1')
    runs_db = create_runs_db(tmp_path / 'runs.db')
    conn = sqlite3.connect(runs_db)
    try:
        record_merge_done(conn, 'old', before, BEFORE_WINDOW)
        record_merge_done(conn, 'ff', fast_forward, IN_WINDOW)
        record_train(conn, 't2', ['t1', 't2'], train, LATER_IN_WINDOW)
        _completed(conn, 'ff', IN_WINDOW)
        _completed(conn, 'old', BEFORE_WINDOW)
        conn.commit()
    finally:
        conn.close()
    return Scene(repo, runs_db)


def _query(scene: Scene, start: str = WINDOW_START, end: str = WINDOW_END) -> int:
    conn = sqlite3.connect(scene.runs_db)
    try:
        return digest.query_landings_in_window(conn, scene.repo.root, start, end)
    finally:
        conn.close()


class TestLandingsInWindow:
    def test_counts_each_landing_in_the_window_not_done_runs(self, scene: Scene) -> None:
        conn = sqlite3.connect(scene.runs_db)
        try:
            done_runs = digest.query_count_done_in_window(conn, WINDOW_START, WINDOW_END)
        finally:
            conn.close()
        assert _query(scene) == 3
        assert done_runs == 1

    def test_bounds_are_inclusive(self, scene: Scene) -> None:
        edge = scene.repo.fast_forward('landed at the window end')
        conn = sqlite3.connect(scene.runs_db)
        try:
            record_merge_done(conn, 'edge', edge, WINDOW_END)
            conn.commit()
        finally:
            conn.close()
        assert _query(scene) == 4

    def test_path_form_counts_the_same(self, scene: Scene) -> None:
        assert digest.landings_in_window(
            scene.runs_db, scene.repo.root, WINDOW_START, WINDOW_END) == 3

    def test_landings_are_read_off_the_named_main_branch(self, tmp_path: Path) -> None:
        trunk = landing_repo(tmp_path / 'repo', branch='trunk')
        runs_db = create_runs_db(tmp_path / 'runs.db')
        conn = sqlite3.connect(runs_db)
        try:
            record_merge_done(conn, 'on-trunk', trunk.fast_forward('on trunk'), IN_WINDOW)
            conn.commit()
        finally:
            conn.close()
        assert digest.landings_in_window(
            runs_db, trunk.root, WINDOW_START, WINDOW_END, main_ref='trunk') == 1
        assert digest.landings_in_window(runs_db, trunk.root, WINDOW_START, WINDOW_END) is None

    def test_path_form_is_none_for_a_root_that_is_not_a_repository(
        self, scene: Scene, tmp_path: Path,
    ) -> None:
        plain = tmp_path / 'plain'
        plain.mkdir()
        assert digest.landings_in_window(scene.runs_db, plain, WINDOW_START, WINDOW_END) is None

    def test_path_form_is_none_for_a_missing_runs_db_and_creates_no_file(
        self, scene: Scene, tmp_path: Path,
    ) -> None:
        missing = tmp_path / 'absent' / 'runs.db'
        result = digest.landings_in_window(missing, scene.repo.root, WINDOW_START, WINDOW_END)
        assert result is None
        assert not missing.parent.exists()

    def test_query_form_raises_for_a_root_that_is_not_a_repository(
        self, scene: Scene, tmp_path: Path,
    ) -> None:
        plain = tmp_path / 'plain'
        plain.mkdir()
        conn = sqlite3.connect(scene.runs_db)
        try:
            with pytest.raises(GitUnavailable):
                digest.query_landings_in_window(conn, plain, WINDOW_START, WINDOW_END)
        finally:
            conn.close()

    def test_query_form_raises_for_a_runs_db_without_events(
        self, scene: Scene, tmp_path: Path,
    ) -> None:
        bare = tmp_path / 'bare.db'
        sqlite3.connect(bare).close()
        conn = sqlite3.connect(bare)
        try:
            with pytest.raises(sqlite3.Error):
                digest.query_landings_in_window(conn, scene.repo.root, WINDOW_START, WINDOW_END)
        finally:
            conn.close()
