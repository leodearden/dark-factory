"""Two sibling client projects under one tmp dir, for the §8 isolation instruments: A clean, B carrying a canary everywhere.

Each project has every source the publisher reads (plans/client-feed-prd.md §8): git history with a landed merge,
a PRD and acceptance criteria, tasks.db, runs.db with a verify window and the landing, a verify archive, an open
escalation, an engagement and a reveal request. B reuses A's task ids, so only scoping keeps them apart.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import yaml
from _archive_files import junit_xml, passed, write_junit, write_summary
from _git_repo import make_repo
from _in_dir import request_reveals, write_engagement
from _stores import EscalationQueueDir, RunsDb, TasksDb

CANARY_A_DESC: Final = 'CANARY-A-DESC'
REVEALED_PATH: Final = 'src/app.py'
DONE_TASK: Final = 7
OPEN_TASK: Final = 9
_BASE: Final = datetime(2026, 10, 21, 10, 0, tzinfo=UTC)
_APP_TEXT: Final = ''.join(f'value_{index} = {index}\n' for index in range(60))


@dataclass(frozen=True)
class Project:
    root: Path
    out_dir: Path
    in_dir: Path
    policy: Path


@dataclass(frozen=True)
class TwoProjects:
    a: Project
    b: Project
    canary_b: str


@dataclass(frozen=True)
class _Marks:
    """Where one project's marker goes: `hidden` only into fields no client sees, `shown` into everything else."""

    name: str
    hidden: str
    shown: str


def _at(minutes: float) -> datetime:
    return _BASE + timedelta(minutes=minutes)


def _git_history(root: Path, marks: _Marks) -> str:
    """main holding the PRD, the criteria, src/ and tests/, and `Merge task/7 into main`; the merge sha."""
    prd_path = _prd_path(marks)
    repo = make_repo(root)
    repo.write('.gitignore', '.taskmaster/\ndata/\n')
    repo.write(prd_path, f'# The {marks.shown} plan\n\n## 1. Goal\n\nShip the {marks.shown} app.\n')
    repo.write('acceptance/criteria.yaml', yaml.safe_dump({
        'format': 1,
        'criteria': [{
            'id': 'C-1',
            'prd': prd_path,
            'capability': f'The {marks.shown} app',
            'statement': f'The {marks.shown} app starts.',
            'kind': 'success',
        }],
    }))
    repo.write(REVEALED_PATH, _APP_TEXT)
    repo.write('tests/test_app.py', f'def test_{marks.shown.replace("-", "_")}():\n    pass\n')
    repo.commit('Initial', when=_at(-60))
    repo.branch('task/7')
    repo.checkout('task/7')
    repo.write(f'src/{marks.shown}.py', f'MARK = {marks.shown!r}\n')
    repo.commit(f'Add the {marks.shown} module', when=_at(-30))
    repo.checkout('main')
    return repo.merge_no_ff('task/7', 'Merge task/7 into main', when=_at(18))


def _prd_path(marks: _Marks) -> str:
    return f'plans/{marks.name}-prd.md'


def _tasks(root: Path, marks: _Marks) -> None:
    """Task 7 done and task 9 in progress under the PRD; `hidden` in every field the projection drops."""
    tasks = TasksDb.create(root)
    rows = (
        (DONE_TASK, f'Ship the {marks.shown} app', 'done', '2026-10-21T09:00:00.000Z', ()),
        (OPEN_TASK, f'Polish the {marks.shown} app', 'in-progress', '2026-10-21T09:30:00.000Z', (DONE_TASK,)),
    )
    for task_id, title, status, updated_at, depends_on in rows:
        tasks.add(
            task_id, title=title, status=status, updated_at=updated_at,
            description=f'description {marks.hidden}', details=f'details {marks.hidden}',
            test_strategy=f'strategy {marks.hidden}',
            metadata={'prd_path': _prd_path(marks), 'x_note': marks.hidden, 'owner': marks.shown},
            depends_on=depends_on,
        )


def _runs(root: Path, merge: str, marks: _Marks) -> None:
    """Task 7's verify window [at(0), at(10)] with its archived summary and junit, then its landing."""
    runs = RunsDb.create(root)
    runs.emit('phase_enter', at=_at(0), task_id=DONE_TASK, phase='verify')
    write_summary(root, _at(5), commands=(('test', 0), ('lint', 0), ('type', 0)), category='passed')
    write_junit(root, _at(5), junit_xml(passed(f'test_{marks.shown}'), passed('test_app')))
    runs.emit('phase_exit', at=_at(10), task_id=DONE_TASK, phase='verify')
    runs.emit(
        'merge_finalized', at=_at(20), task_id=DONE_TASK, phase='merge',
        data={'state': 'done', 'merge_sha': merge},
    )


def _project(tmp_path: Path, marks: _Marks) -> Project:
    root = tmp_path / marks.name
    out_dir, in_dir = tmp_path / f'{marks.name}-out', tmp_path / f'{marks.name}-in'
    policy = tmp_path / f'{marks.name}.yaml'
    project_id = f'project-{marks.name}'
    merge = _git_history(root, marks)
    _tasks(root, marks)
    _runs(root, merge, marks)
    EscalationQueueDir.create(root).write(
        f'esc-{OPEN_TASK}-1', task_id=str(OPEN_TASK),
        summary=f'summary {marks.shown}', detail=f'detail {marks.shown}',
    )
    write_engagement(in_dir, project_id=project_id, delivered=None)
    request_reveals(in_dir, REVEALED_PATH, at=_at(30))
    out_dir.mkdir()
    policy.write_text(yaml.safe_dump({
        'project_id': project_id,
        'project_root': str(root),
        'out_dir': str(out_dir),
        'in_dir': str(in_dir),
        'visible_globs': ['src/**', 'tests/**'],
        'reveal_proportion': 1.0,
    }))
    return Project(root=root, out_dir=out_dir, in_dir=in_dir, policy=policy)


def build_two_projects(tmp_path: Path) -> TwoProjects:
    """tmp/a, tmp/a-out, tmp/a-in and tmp/a.yaml, and the same for b."""
    canary_b = f'CANARY-B-{secrets.token_hex(4)}'
    a = _project(tmp_path, _Marks(name='a', hidden=CANARY_A_DESC, shown='alpha'))
    b = _project(tmp_path, _Marks(name='b', hidden=canary_b, shown=canary_b))
    return TwoProjects(a=a, b=b, canary_b=canary_b)
