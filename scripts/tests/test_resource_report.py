"""The resource report's command line: scripts/resource_report.py.

PRD ``plans/per-consumer-resource-recording-prd.md`` leaf 2. Every run names a
fixture samples store with ``--samples-db``, never the live one, and an
injected ``now``.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import _checkout as _checkout
import pytest
import resource_report
from resource_report_fixtures import memory_hold, project_root, samples_db
from shared.project_layout import runs_db

NOW = datetime(2026, 11, 21, 21, 0, tzinfo=UTC)
DF_UNIT = '/user.slice/user-1000.slice/user@1000.service/app.slice/orchestrator-dark-factory.service'


@pytest.fixture
def store(tmp_path: Path) -> Path:
    return samples_db(tmp_path / 'samples' / 'load-samples.db')


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    rc = resource_report.main(list(argv), now=NOW)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def holds_argv(store: Path, *roots: Path, extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    root_args = tuple(arg for root in roots for arg in ('--project-root', str(root)))
    return ('holds', *root_args, '--samples-db', str(store), *extra)


class TestArguments:
    def test_holds_without_a_project_root_exits_2_naming_it(self, store, capsys):
        with pytest.raises(SystemExit) as exc:
            resource_report.main(['holds', '--samples-db', str(store)], now=NOW)

        assert exc.value.code == 2
        assert '--project-root' in capsys.readouterr().err

    def test_a_missing_samples_db_exits_2_naming_its_path(self, tmp_path, make_runs_db, capsys):
        root = project_root(tmp_path, 'dark-factory')
        make_runs_db([], directory=runs_db(root).parent)
        absent = tmp_path / 'nowhere' / 'load-samples.db'

        rc, out, err = run(capsys, *holds_argv(absent, root))

        assert rc == 2
        assert str(absent) in err
        assert out == ''

    def test_an_unparseable_since_exits_2(self, tmp_path, store, capsys):
        root = project_root(tmp_path, 'dark-factory')

        with pytest.raises(SystemExit) as exc:
            resource_report.main(list(holds_argv(store, root, extra=('--since', 'yesterday'))), now=NOW)

        assert exc.value.code == 2
        assert 'yesterday' in capsys.readouterr().err


class TestTheRootsHeader:
    def test_every_root_is_listed_with_the_runs_db_it_read_or_missed(
        self, tmp_path, make_runs_db, store, capsys
    ):
        read_root = project_root(tmp_path, 'dark-factory')
        make_runs_db([], directory=runs_db(read_root).parent)
        missing_root = project_root(tmp_path, 'reify')

        rc, out, _err = run(capsys, *holds_argv(store, read_root, missing_root))

        assert rc == 0
        lines = out.splitlines()
        read_at = next(i for i, line in enumerate(lines) if str(runs_db(read_root)) in line)
        missing_at = next(
            i for i, line in enumerate(lines) if f'runs.db missing: {runs_db(missing_root)}' in line
        )
        count_at = next(i for i, line in enumerate(lines) if 'memory holds since' in line)
        assert str(read_root) in lines[read_at]
        assert 'missing' not in lines[read_at]
        assert str(missing_root) in lines[missing_at]
        assert read_at < missing_at < count_at

    def test_a_root_named_twice_is_read_once(self, tmp_path, make_runs_db, store, capsys):
        root = project_root(tmp_path, 'dark-factory')
        make_runs_db([], directory=runs_db(root).parent)

        _rc, out, _err = run(capsys, *holds_argv(store, root, root / 'data' / '..', extra=('--json',)))

        assert [entry['root'] for entry in json.loads(out)['roots']] == [str(root)]


class TestNoMemoryHolds:
    @pytest.fixture
    def root(self, tmp_path, make_runs_db):
        root = project_root(tmp_path, 'dark-factory')
        recent = datetime(2026, 11, 20, 12, 0, tzinfo=UTC)
        make_runs_db(
            [
                memory_hold(recent, DF_UNIT, arm='io_some_avg10'),
                memory_hold(recent, DF_UNIT, arm='runqueue_ratio'),
                memory_hold(datetime(2026, 11, 13, 23, 59, tzinfo=UTC), DF_UNIT),
            ],
            directory=runs_db(root).parent,
        )
        return root

    def test_the_text_says_zero_memory_holds_since(self, root, store, capsys):
        rc, out, _err = run(capsys, *holds_argv(store, root, extra=('--since', '2026-11-14')))

        assert rc == 0
        assert '0 memory holds since 2026-11-14T00:00:00Z' in out.splitlines()

    def test_json_is_one_object_with_the_report_keys(self, tmp_path, root, store, capsys):
        missing = project_root(tmp_path, 'reify')

        rc, out, _err = run(
            capsys, *holds_argv(store, root, missing, extra=('--since', '2026-11-14', '--json'))
        )

        assert rc == 0
        report = json.loads(out)
        assert {'roots', 'samples_db', 'since', 'window_seconds', 'holds'} <= report.keys()
        assert [(entry['root'], entry['runs_db'], entry['status']) for entry in report['roots']] == [
            (str(root), str(runs_db(root)), 'read'),
            (str(missing), str(runs_db(missing)), 'missing'),
        ]
        assert report['samples_db'] == str(store)
        assert report['since'] == '2026-11-14T00:00:00Z'
        assert report['window_seconds'] == 120
        assert report['holds'] == []


class TestParseSince:
    @pytest.mark.parametrize(
        ('text', 'expected'),
        [
            ('2026-11-14', datetime(2026, 11, 14, tzinfo=UTC)),
            ('2026-11-14T06:30:00+02:00', datetime(2026, 11, 14, 4, 30, tzinfo=UTC)),
            ('2026-11-14T06:30:00', datetime(2026, 11, 14, 6, 30, tzinfo=UTC)),
            ('7d', NOW - timedelta(days=7)),
            ('36h', NOW - timedelta(hours=36)),
            ('90m', NOW - timedelta(minutes=90)),
            ('45s', NOW - timedelta(seconds=45)),
        ],
    )
    def test_accepted_forms(self, text, expected):
        since = resource_report.parse_since(text, NOW)

        assert since == expected
        assert since.utcoffset() == timedelta(0)

    @pytest.mark.parametrize('text', ['7w', '', '-3d', 'yesterday'])
    def test_rejected_forms(self, text):
        with pytest.raises(ValueError):
            resource_report.parse_since(text, NOW)
