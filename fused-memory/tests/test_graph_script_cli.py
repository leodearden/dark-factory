"""Tests for scripts/_graph_script_cli.py: graph addressing and output files for read-only scripts."""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from _fm_helpers import load_script_module
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

SCRIPTS = Path(__file__).parent.parent / 'scripts'
cli = load_script_module(SCRIPTS / '_graph_script_cli.py', mod_name='_graph_script_cli')


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    cli.add_graph_arguments(parser)
    return parser.parse_args(argv)


class TestResolveFalkordbUri:
    def test_the_flag_wins_over_the_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('FALKORDB_URI', 'redis://from-env:1')
        assert cli.resolve_falkordb_uri(_args('--graph-uri', 'redis://from-flag:2')) == 'redis://from-flag:2'

    def test_the_env_wins_over_the_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('FALKORDB_URI', 'redis://from-env:1')
        assert cli.resolve_falkordb_uri(_args()) == 'redis://from-env:1'


class TestReadJson:
    def test_a_missing_file_is_unreadable_input(self, tmp_path: Path) -> None:
        with pytest.raises(cli.UnreadableInput, match='absent.json'):
            cli.read_json(tmp_path / 'absent.json')

    def test_a_malformed_file_is_unreadable_input(self, tmp_path: Path) -> None:
        path = tmp_path / 'bad.json'
        path.write_text('{not json')
        with pytest.raises(cli.UnreadableInput):
            cli.read_json(path)


class TestWriteFilesAtomically:
    def test_every_file_lands_and_no_staging_file_remains(self, tmp_path: Path) -> None:
        one, other = tmp_path / 'a' / 'one.json', tmp_path / 'b' / 'other.jsonl'
        cli.write_files_atomically({one: 'one\n', other: 'other\n'})
        assert (one.read_text(), other.read_text()) == ('one\n', 'other\n')
        assert sorted(p.name for p in tmp_path.rglob('*') if p.is_file()) == ['one.json', 'other.jsonl']

    def test_no_target_is_replaced_when_a_later_file_cannot_be_staged(self, tmp_path: Path) -> None:
        kept = tmp_path / 'kept.json'
        kept.write_text('before\n')
        blocker = tmp_path / 'blocker'
        blocker.write_text('a file where a directory must go')
        with pytest.raises(OSError):
            cli.write_files_atomically({kept: 'after\n', blocker / 'out.json': 'never\n'})
        assert kept.read_text() == 'before\n'
