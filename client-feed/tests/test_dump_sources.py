"""`client-feed dump-sources`: every store source read under one policy's root, printed as JSON."""

import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from _git_repo import make_repo
from _stores import RunsDb, TasksDb

from client_feed.cli import main

SHA = 'e' * 40
X_PRD = '# X plan /home/leo/secret\n\nStatus: draft.\n\n## 1. Goal\n\nThe goal paragraph.\n'
FORBIDDEN_KEYS = frozenset({'description', 'details', 'test_strategy', 'metadata'})


def keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys(item)


def write_policy(tmp_path: Path, root: Path) -> Path:
    out_dir = tmp_path / 'out'
    out_dir.mkdir(exist_ok=True)
    config = tmp_path / 'policy.yaml'
    config.write_text(
        yaml.safe_dump({
            'project_id': 'acme',
            'project_root': str(root),
            'out_dir': str(out_dir),
            'in_dir': str(tmp_path / 'in'),
            'visible_globs': ['plans/**'],
        })
    )
    return config


@pytest.fixture
def root(tmp_path: Path) -> Path:
    repo = make_repo(tmp_path / 'root')
    repo.write('plans/x-prd.md', X_PRD)
    repo.commit('prd', when=datetime(2026, 10, 1, tzinfo=UTC))
    return repo.root


@pytest.fixture
def config(tmp_path: Path, root: Path) -> Path:
    tasks = TasksDb.create(root)
    tasks.add(
        9,
        title='Unblock /home/leo/x',
        status='blocked',
        updated_at='2026-10-02T09:00:00.000Z',
        description='SECRET-DESC',
        details='SECRET-DETAILS',
        test_strategy='SECRET-TS',
        metadata={'prd_path': 'plans/x-prd.md', 'dry_run_proposals': 'SECRET'},
    )
    tasks.add(10, title='Land it', updated_at='2026-10-02T10:00:00.000Z')
    runs = RunsDb.create(root)
    paused_at = datetime.now(UTC) - timedelta(hours=3)
    earlier = paused_at - timedelta(hours=1)
    runs.emit(
        'escalation_created', at=earlier, task_id='9',
        data={'escalation_id': 'esc-9-1', 'level': 2, 'summary': 'SECRET-SUMMARY'},
    )
    runs.emit('phase_enter', at=earlier, task_id='9', phase='execute')
    runs.emit(
        'merge_finalized', at=earlier, task_id='10', phase='merge',
        data={'state': 'done', 'merge_sha': SHA, 'reason': 'SECRET-REASON'},
    )
    runs.emit('merge_blocked', at=earlier, task_id='9', data={'reason': 'SECRET-REASON /home/x'})
    runs.emit('scheduler_paused', at=paused_at, data={'reason': 'SECRET-REASON'})
    return write_policy(tmp_path, root)


def dump(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(['dump-sources', *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_dump_carries_every_source_and_no_secret(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = dump(capsys, '--config', str(config))

    assert (code, err) == (0, '')
    assert 'SECRET' not in out
    assert '/home/' not in out
    document = json.loads(out)
    assert FORBIDDEN_KEYS.isdisjoint(keys(document))
    assert document['factory']['scheduler'] == 'paused'
    assert document['factory']['idle_seconds'] >= 10800
    tasks = {row['task_id']: row for row in document['tasks']['rows']}
    assert tasks[9]['waiting_on'] == 'leo'
    assert tasks[9]['title'] == 'Unblock [path]'
    assert document['prds']['found'] == [
        {'prd_path': 'plans/x-prd.md', 'title': 'X plan [path]', 'excerpt': 'The goal paragraph.'}
    ]
    landed = [event for event in document['events']['events'] if 'task_ids' in event]
    assert [(event['commit'], event['task_ids']) for event in landed] == [(SHA, [10])]
    assert document['reviews'] == {'10': 'not_recorded'}
    assert len(document['head']) == 40


def test_the_limit_bounds_the_event_batch(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = dump(capsys, '--config', str(config), '--limit', '1')

    batch = json.loads(out)['events']
    assert code == 0
    assert len(batch['events']) == 1
    assert batch['has_more'] is True


def test_the_cursor_skips_events_already_read(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, first, _ = dump(capsys, '--config', str(config), '--limit', '1')
    cursor = json.loads(first)['events']['cursor']

    _, second, _ = dump(capsys, '--config', str(config), '--after-id', str(cursor))

    assert [event['event_id'] for event in json.loads(second)['events']['events']] == [3]


def test_a_refused_policy_exits_1_with_nothing_on_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    not_a_repo = tmp_path / 'plain'
    not_a_repo.mkdir()

    code, out, err = dump(capsys, '--config', str(write_policy(tmp_path, not_a_repo)))

    assert (code, out) == (1, '')
    assert 'project_root' in err


def test_a_missing_store_exits_1_naming_it(
    tmp_path: Path, root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    TasksDb.create(root)

    code, out, err = dump(capsys, '--config', str(write_policy(tmp_path, root)))

    assert (code, out) == (1, '')
    assert 'events' in err
    assert 'data/orchestrator/runs.db' in err
