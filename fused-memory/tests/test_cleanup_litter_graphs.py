"""The production litter cleanup: an allowlist of graph names it may delete, and nothing else.

The offline cases drive the script through its own GraphServer Protocol with an
in-file fake, and aim ``main`` at refused ports. The one live case runs on the
scratch server only, over two graphs it creates; run it with
``uv run pytest -m integration tests/test_cleanup_litter_graphs.py``.
"""

from __future__ import annotations

import contextlib
import json
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from _fm_falkor_live import FALKOR_HOST, FALKOR_PORT, falkor_skipif
from _fm_helpers import load_script_module
from _fm_sockets import refused_port
from falkordb import FalkorDB
from redis.exceptions import RedisError

CLEANUP = load_script_module(
    Path(__file__).resolve().parents[1] / 'scripts' / 'cleanup_litter_graphs.py'
)

TARGET = 'redis://localhost:6379'


class FakeGraphServer:
    """A GraphServer over a dict of graph name to node count, recording every call.

    *on_run_id* runs at the start of each ``run_id()`` call, so a test can
    restart the server or shrink a graph mid-run. The *fail_delete_at*-th
    delete call raises RedisError and deletes nothing.
    """

    def __init__(
        self,
        counts: Mapping[str, int],
        run_id: str = 'run-a',
        *,
        on_run_id: Callable[[FakeGraphServer], None] | None = None,
        fail_delete_at: int | None = None,
    ) -> None:
        self.counts = dict(counts)
        self.current_run_id = run_id
        self.on_run_id = on_run_id
        self.fail_delete_at = fail_delete_at
        self.delete_calls = 0
        self.counted: list[str] = []
        self.deleted: list[str] = []

    def graph_names(self) -> tuple[str, ...]:
        return tuple(self.counts)

    def node_count(self, name: str) -> int:
        self.counted.append(name)
        return self.counts[name]

    def delete(self, name: str) -> None:
        self.delete_calls += 1
        if self.delete_calls == self.fail_delete_at:
            raise RedisError(f'GRAPH.DELETE {name} failed')
        self.deleted.append(name)
        del self.counts[name]

    def run_id(self) -> str:
        if self.on_run_id is not None:
            self.on_run_id(self)
        return self.current_run_id


LITTER = (
    '_probe4748b_bbd3e6b1',
    '_probe4748b_0a1b2c3d',
    '_test_3334_falkor_fulltext_1f2e3d4c',
    '_test_530_list_indices_integration_5a6b7c8d',
    '_test_4340_read_pagination_03b38704',
    '_test_3561_episode_uuid_c8c384d4',
    '_test_3769_drop_vector_indices_9e8f7a6b',
    '_test_2207_merge_entities_deadbeef',
    'probe_e1_gw0',
    'probe_e1_gw29',
    'probe_e1_master',
    'sweep_e4_gw2',
    'evalmem_test_0123abcd',
    '_probe',
)

KEPT = (
    'dark_factory',
    'reify',
    'stream_touch',
    'solar_challenge',
    'solar_challenge_platform',
    'scoping_intake',
    'know_live',
    'knowlive',
    'autopilot_video',
    'pump_web_ui',
    'market_predictor',
    'default_db',
    'evalmem_lme_ref_incumbent_a',
)

NEAR_MISSES = (
    '_probe4748b_bbd3e6b',
    '_probe4748b_BBD3E6B1',
    '_probe_extra',
    '_probe\n',
    'probe_e1_gw',
    'probe_e1_master2',
    '_test_2084_edge_dedup_33c83367',
    '_test_7793_kept_0123abcd',
    'evalmem_test_',
    'xevalmem_test_0123abcd',
    'dark_factory_test',
)


def _interleaved(first: tuple[str, ...], second: tuple[str, ...]) -> list[str]:
    longest = max(len(first), len(second))
    return [group[i] for i in range(longest) for group in (first, second) if i < len(group)]


# --- the allowlist ----------------------------------------------------------------------


def test_the_plan_deletes_exactly_the_litter_and_keeps_the_rest_in_listing_order():
    names = _interleaved(LITTER, KEPT)

    plan = CLEANUP.plan_cleanup(names)

    assert plan.delete == tuple(name for name in names if name in LITTER)
    assert plan.keep == tuple(name for name in names if name in KEPT)


@pytest.mark.parametrize('name', NEAR_MISSES)
def test_a_near_miss_is_kept(name):
    plan = CLEANUP.plan_cleanup([name])

    assert plan.delete == ()
    assert plan.keep == (name,)
    assert not CLEANUP.is_litter(name)


def test_the_plan_is_immutable():
    plan = CLEANUP.plan_cleanup(['_probe', 'dark_factory'])

    with pytest.raises(AttributeError):
        plan.delete = ()


def test_the_allowlist_is_exactly_these_seven_patterns():
    assert tuple(pattern.pattern for pattern in CLEANUP.LITTER_PATTERNS) == (
        '_probe',
        '_probe4748b_[0-9a-f]{8}',
        'probe_e1_gw[0-9]+',
        'probe_e1_master',
        'sweep_e4_gw[0-9]+',
        '_test_(?:3334_falkor_fulltext|530_list_indices_integration|4340_read_pagination'
        '|3561_episode_uuid|3769_drop_vector_indices|2207_merge_entities)_[0-9a-f]{8}',
        'evalmem_test_[0-9a-f]{8}',
    )


def test_require_litter_refuses_a_kept_graph_by_name():
    with pytest.raises(CLEANUP.NotLitter, match="'dark_factory'"):
        CLEANUP.require_litter('dark_factory')


def test_require_litter_returns_a_litter_name_unchanged():
    assert CLEANUP.require_litter('probe_e1_gw29') == 'probe_e1_gw29'


# --- the census and the reviewed listing ------------------------------------------------


def test_the_census_reads_the_run_id_and_only_the_named_graphs():
    server = FakeGraphServer({'_probe': 0, 'dark_factory': 41, 'probe_e1_gw0': 3}, 'run-a')

    census = CLEANUP.take_census(server, ['dark_factory', '_probe'])

    assert census.run_id == 'run-a'
    assert dict(census.counts) == {'dark_factory': 41, '_probe': 0}
    assert server.counted == ['dark_factory', '_probe']


def test_the_census_is_immutable():
    census = CLEANUP.take_census(FakeGraphServer({'dark_factory': 41}), ['dark_factory'])

    with pytest.raises(AttributeError):
        census.run_id = 'run-b'
    with pytest.raises(TypeError):
        census.counts['dark_factory'] = 0


def test_the_listing_splits_every_graph_into_delete_and_keep_counts():
    server = FakeGraphServer(
        {'_probe': 0, 'dark_factory': 41, 'probe_e1_gw0': 3, 'default_db': 0}, 'run-a'
    )

    listing = CLEANUP.build_listing(TARGET, server)

    assert listing.target == TARGET
    assert listing.run_id == 'run-a'
    assert datetime.fromisoformat(listing.taken_at).tzinfo is not None
    assert dict(listing.delete) == {'_probe': 0, 'probe_e1_gw0': 3}
    assert dict(listing.keep) == {'dark_factory': 41, 'default_db': 0}


def test_a_written_listing_reads_back_equal(tmp_path):
    listing = CLEANUP.build_listing(
        TARGET, FakeGraphServer({'evalmem_test_0123abcd': 2, 'reify': 7})
    )
    path = tmp_path / 'listing.json'

    CLEANUP.write_listing(path, listing)

    assert CLEANUP.read_listing(path) == listing


def _valid_listing_json() -> dict[str, object]:
    return {
        'target': TARGET,
        'run_id': 'run-a',
        'taken_at': '2026-11-21T00:00:00+00:00',
        'delete': {'_probe': 0},
        'keep': {'dark_factory': 41},
    }


@pytest.mark.parametrize(
    'content',
    [
        pytest.param('{"target": ', id='malformed-json'),
        pytest.param(
            json.dumps({k: v for k, v in _valid_listing_json().items() if k != 'delete'}),
            id='no-delete-key',
        ),
        pytest.param(
            json.dumps({k: v for k, v in _valid_listing_json().items() if k != 'target'}),
            id='no-target-key',
        ),
        pytest.param(json.dumps([_valid_listing_json()]), id='not-an-object'),
        pytest.param(json.dumps({**_valid_listing_json(), 'delete': ['_probe']}), id='delete-not-a-map'),
        pytest.param(json.dumps({**_valid_listing_json(), 'delete': {'_probe': '0'}}), id='count-not-int'),
        pytest.param(json.dumps({**_valid_listing_json(), 'target': 6379}), id='target-not-str'),
    ],
)
def test_an_unreadable_listing_is_refused_naming_its_path(tmp_path, content):
    path = tmp_path / 'listing.json'
    path.write_text(content)

    with pytest.raises(CLEANUP.UnreadableListing, match=re.escape(str(path))):
        CLEANUP.read_listing(path)


def test_a_missing_listing_is_refused_naming_its_path(tmp_path):
    path = tmp_path / 'absent.json'

    with pytest.raises(CLEANUP.UnreadableListing, match=re.escape(str(path))):
        CLEANUP.read_listing(path)


# --- the delete run ---------------------------------------------------------------------


def _reviewed(delete: tuple[str, ...], keep: tuple[str, ...] = ()) -> Any:
    return CLEANUP.Listing(
        target=TARGET,
        run_id='run-a',
        taken_at='2026-11-21T00:00:00+00:00',
        delete={name: 0 for name in delete},
        keep={name: 0 for name in keep},
    )


class RecordingSleep:
    """A ``sleep`` that records how many deletes had happened at each call, and for how long."""

    def __init__(self, server: FakeGraphServer) -> None:
        self.server = server
        self.calls: list[tuple[int, float]] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append((len(self.server.deleted), seconds))


def _execute(server: FakeGraphServer, reviewed: Any, *, pace_s: float = 0.0) -> Any:
    plan = CLEANUP.plan_cleanup(server.graph_names())
    return CLEANUP.execute(server, plan, reviewed, pace_s=pace_s, sleep=RecordingSleep(server))


def _after_deletes(count: int, change: Callable[[FakeGraphServer], None]):
    def hook(server: FakeGraphServer) -> None:
        if len(server.deleted) == count:
            change(server)

    return hook


def _restart(server: FakeGraphServer) -> None:
    server.current_run_id = 'run-b'


def test_it_deletes_the_reviewed_litter_one_at_a_time_in_plan_order_pacing_between():
    server = FakeGraphServer(
        {'probe_e1_gw0': 1, 'dark_factory': 41, '_probe': 0, 'sweep_e4_gw2': 3}
    )
    plan = CLEANUP.plan_cleanup(server.graph_names())
    sleep = RecordingSleep(server)

    record = CLEANUP.execute(
        server, plan, _reviewed(('sweep_e4_gw2', '_probe', 'probe_e1_gw0')),
        pace_s=7.5, sleep=sleep,
    )

    assert server.deleted == ['probe_e1_gw0', '_probe', 'sweep_e4_gw2']
    assert record.deleted == ('probe_e1_gw0', '_probe', 'sweep_e4_gw2')
    assert sleep.calls == [(1, 7.5), (2, 7.5)]
    assert record.target == TARGET


def test_unreviewed_litter_is_reported_not_deleted_and_vanished_reviewed_names_are_reported():
    server = FakeGraphServer({'_probe': 0, 'evalmem_test_0123abcd': 2, 'dark_factory': 41})

    record = _execute(server, _reviewed(('_probe', 'probe_e1_gw7')))

    assert server.deleted == ['_probe']
    assert record.unreviewed == ('evalmem_test_0123abcd',)
    assert record.already_absent == ('probe_e1_gw7',)
    assert 'evalmem_test_0123abcd' in server.counts


def test_a_reviewed_listing_naming_a_kept_graph_never_deletes_it():
    server = FakeGraphServer({'dark_factory': 41, '_probe': 0})

    record = _execute(server, _reviewed(('dark_factory', '_probe')))

    assert server.deleted == ['_probe']
    assert server.counts == {'dark_factory': 41}
    assert record.deleted == ('_probe',)


def test_a_plan_naming_a_kept_graph_is_refused_at_the_delete():
    server = FakeGraphServer({'dark_factory': 41})
    plan = CLEANUP.CleanupPlan(delete=('dark_factory',), keep=())

    with pytest.raises(CLEANUP.NotLitter, match="'dark_factory'"):
        CLEANUP.execute(
            server, plan, _reviewed(('dark_factory',)), pace_s=0.0, sleep=RecordingSleep(server)
        )

    assert server.deleted == []


def test_the_record_holds_every_count_before_and_each_kept_count_after():
    server = FakeGraphServer({'_probe': 0, 'probe_e1_gw0': 1, 'dark_factory': 41, 'reify': 7})

    record = _execute(server, _reviewed(('_probe', 'probe_e1_gw0')))

    assert dict(record.before) == {'_probe': 0, 'probe_e1_gw0': 1, 'dark_factory': 41, 'reify': 7}
    assert dict(record.after) == {'dark_factory': 41, 'reify': 7}
    assert record.shrank == ()
    assert record.outcome == CLEANUP.Exit.OK


def test_a_kept_graph_that_shrank_during_the_run_is_flagged():
    def shrink(server: FakeGraphServer) -> None:
        server.counts['dark_factory'] = 40

    server = FakeGraphServer(
        {'_probe': 0, 'probe_e1_gw0': 1, 'dark_factory': 41, 'reify': 7},
        on_run_id=_after_deletes(1, shrink),
    )

    record = _execute(server, _reviewed(('_probe', 'probe_e1_gw0')))

    assert record.shrank == (('dark_factory', 41, 40),)
    assert dict(record.after) == {'dark_factory': 40, 'reify': 7}
    assert record.outcome == CLEANUP.Exit.KEPT_GRAPH_SHRANK


def test_a_kept_graph_that_vanished_during_the_run_is_flagged_as_shrunk_to_zero():
    def vanish(server: FakeGraphServer) -> None:
        server.counts.pop('reify', None)

    server = FakeGraphServer(
        {'_probe': 0, 'probe_e1_gw0': 1, 'dark_factory': 41, 'reify': 7},
        on_run_id=_after_deletes(1, vanish),
    )

    record = _execute(server, _reviewed(('_probe', 'probe_e1_gw0')))

    assert record.shrank == (('reify', 7, 0),)
    assert record.outcome == CLEANUP.Exit.KEPT_GRAPH_SHRANK


def test_a_kept_graph_that_grew_is_not_flagged():
    def grow(server: FakeGraphServer) -> None:
        server.counts['dark_factory'] = 45

    server = FakeGraphServer(
        {'_probe': 0, 'probe_e1_gw0': 1, 'dark_factory': 41},
        on_run_id=_after_deletes(1, grow),
    )

    record = _execute(server, _reviewed(('_probe', 'probe_e1_gw0')))

    assert record.shrank == ()
    assert record.outcome == CLEANUP.Exit.OK


def test_a_server_restart_stops_the_run_before_the_next_delete():
    server = FakeGraphServer(
        {'_probe': 0, 'probe_e1_gw0': 1, 'sweep_e4_gw2': 3, 'dark_factory': 41},
        on_run_id=_after_deletes(2, _restart),
    )

    record = _execute(server, _reviewed(('_probe', 'probe_e1_gw0', 'sweep_e4_gw2')))

    assert server.deleted == ['_probe', 'probe_e1_gw0']
    assert record.deleted == ('_probe', 'probe_e1_gw0')
    assert record.outcome == CLEANUP.Exit.SERVER_RESTARTED
    assert (record.run_id_before, record.run_id_after) == ('run-a', 'run-b')


def test_a_server_restart_after_the_last_delete_is_still_reported():
    server = FakeGraphServer(
        {'_probe': 0, 'dark_factory': 41}, on_run_id=_after_deletes(1, _restart)
    )

    record = _execute(server, _reviewed(('_probe',)))

    assert record.deleted == ('_probe',)
    assert record.outcome == CLEANUP.Exit.SERVER_RESTARTED
    assert (record.run_id_before, record.run_id_after) == ('run-a', 'run-b')


def test_a_clean_run_sees_one_run_id_throughout():
    server = FakeGraphServer({'_probe': 0, 'dark_factory': 41}, 'run-a')

    record = _execute(server, _reviewed(('_probe',)))

    assert record.run_id_before == record.run_id_after == 'run-a'
    assert record.outcome == CLEANUP.Exit.OK



# --- the command line -------------------------------------------------------------------


@pytest.fixture
def refused_target(monkeypatch) -> Iterator[str]:
    """FALKORDB_URI on a refused port, so no test of main can reach a real server."""
    with refused_port() as port:
        uri = f'redis://127.0.0.1:{port}'
        monkeypatch.setenv('FALKORDB_URI', uri)
        yield uri


def _write_reviewed(path: Path, target: str) -> Path:
    listing = CLEANUP.Listing(
        target=target,
        run_id='run-a',
        taken_at='2026-11-21T00:00:00+00:00',
        delete={'_probe': 0},
        keep={'dark_factory': 41},
    )
    CLEANUP.write_listing(path, listing)
    return path


def test_list_against_an_unreachable_server_fails_without_writing(refused_target, tmp_path, capsys):
    out = tmp_path / 'listing.json'

    code = CLEANUP.main(['list', '--out', str(out)])

    err = capsys.readouterr().err
    assert code == CLEANUP.Exit.FAILED
    assert not out.exists()
    assert err.startswith('failed:')
    assert len(err.splitlines()) == 1


def test_delete_with_a_missing_listing_is_refused_before_connecting(refused_target, tmp_path, capsys):
    missing = tmp_path / 'absent.json'
    record = tmp_path / 'record.json'

    code = CLEANUP.main(['delete', '--reviewed', str(missing), '--record', str(record)])

    err = capsys.readouterr().err
    assert code == CLEANUP.Exit.REFUSED
    assert str(missing) in err
    assert not record.exists()


def test_delete_with_a_listing_for_another_target_is_refused_naming_both(
    refused_target, tmp_path, capsys
):
    reviewed = _write_reviewed(tmp_path / 'listing.json', 'redis://localhost:6379')
    record = tmp_path / 'record.json'

    code = CLEANUP.main(['delete', '--reviewed', str(reviewed), '--record', str(record)])

    err = capsys.readouterr().err
    assert code == CLEANUP.Exit.REFUSED
    assert 'redis://localhost:6379' in err
    assert refused_target in err
    assert not record.exists()


def test_delete_with_a_listing_for_this_target_fails_on_the_unreachable_server(
    refused_target, tmp_path, capsys
):
    reviewed = _write_reviewed(tmp_path / 'listing.json', refused_target)
    record = tmp_path / 'record.json'

    code = CLEANUP.main(['delete', '--reviewed', str(reviewed), '--record', str(record)])

    assert code == CLEANUP.Exit.FAILED
    assert capsys.readouterr().err.startswith('failed:')
    assert not record.exists()


def test_delete_without_a_record_path_is_a_usage_error(refused_target, tmp_path):
    reviewed = _write_reviewed(tmp_path / 'listing.json', refused_target)

    with pytest.raises(SystemExit) as exit_info:
        CLEANUP.main(['delete', '--reviewed', str(reviewed)])

    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    'argv',
    [
        pytest.param([], id='no-command'),
        pytest.param(['list'], id='list-without-out'),
        pytest.param(['delete', '--record', 'r.json'], id='delete-without-reviewed'),
        pytest.param(['delete', '--reviewed', 'l.json'], id='delete-without-record'),
    ],
)
def test_the_required_arguments_are_required(argv):
    with pytest.raises(SystemExit) as exit_info:
        CLEANUP.build_parser().parse_args(argv)

    assert exit_info.value.code == 2


def test_the_pace_defaults_to_ten_seconds():
    args = CLEANUP.build_parser().parse_args(['delete', '--reviewed', 'l.json', '--record', 'r.json'])

    assert args.pace_seconds == 10.0


def test_the_listing_summary_names_the_target_counts_per_pattern_and_every_kept_graph():
    listing = CLEANUP.build_listing(
        TARGET,
        FakeGraphServer(
            {'probe_e1_gw0': 1, 'probe_e1_gw1': 2, '_probe': 0, 'dark_factory': 41, 'default_db': 0},
            'run-a',
        ),
    )

    lines = CLEANUP.listing_summary(listing)

    text = '\n'.join(lines)
    assert TARGET in text
    assert 'run-a' in text
    assert any(line.split() == ['2', 'probe_e1_gw[0-9]+'] for line in lines)
    assert any(line.split() == ['1', '_probe'] for line in lines)
    assert any(line.split() == ['41', 'dark_factory'] for line in lines)
    assert any(line.split() == ['0', 'default_db'] for line in lines)


def _record_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def test_run_delete_writes_the_finished_record(tmp_path):
    server = FakeGraphServer({'_probe': 0, 'probe_e1_gw0': 1, 'dark_factory': 41})
    record_path = tmp_path / 'record.json'

    code = CLEANUP.run_delete(
        server, _reviewed(('_probe', 'probe_e1_gw0')), record_path,
        target=TARGET, pace_s=0.0, sleep=RecordingSleep(server),
    )

    record = _record_json(record_path)
    assert code == CLEANUP.Exit.OK
    assert record['before'] == {'_probe': 0, 'probe_e1_gw0': 1, 'dark_factory': 41}
    assert record['after'] == {'dark_factory': 41}
    assert record['deleted'] == ['_probe', 'probe_e1_gw0']
    assert record['outcome'] == 'ok'


def test_run_delete_returns_the_runs_outcome(tmp_path):
    server = FakeGraphServer(
        {'_probe': 0, 'probe_e1_gw0': 1, 'dark_factory': 41},
        on_run_id=_after_deletes(1, _restart),
    )
    record_path = tmp_path / 'record.json'

    code = CLEANUP.run_delete(
        server, _reviewed(('_probe', 'probe_e1_gw0')), record_path,
        target=TARGET, pace_s=0.0, sleep=RecordingSleep(server),
    )

    assert code == CLEANUP.Exit.SERVER_RESTARTED
    assert _record_json(record_path)['outcome'] == 'server_restarted'


def test_run_delete_keeps_the_record_of_a_run_a_redis_error_interrupted(tmp_path, capsys):
    server = FakeGraphServer(
        {'_probe': 0, 'probe_e1_gw0': 1, 'sweep_e4_gw2': 3, 'dark_factory': 41}, fail_delete_at=2
    )
    record_path = tmp_path / 'record.json'

    code = CLEANUP.run_delete(
        server, _reviewed(('_probe', 'probe_e1_gw0', 'sweep_e4_gw2')), record_path,
        target=TARGET, pace_s=0.0, sleep=RecordingSleep(server),
    )

    record = _record_json(record_path)
    assert code == CLEANUP.Exit.FAILED
    assert record['deleted'] == ['_probe']
    assert record['before'] == {'_probe': 0, 'probe_e1_gw0': 1, 'sweep_e4_gw2': 3, 'dark_factory': 41}
    assert record['outcome'] == 'failed'
    assert 'GRAPH.DELETE probe_e1_gw0 failed' in capsys.readouterr().err


def test_run_delete_refuses_a_listing_for_another_target(tmp_path):
    server = FakeGraphServer({'_probe': 0})
    record_path = tmp_path / 'record.json'

    code = CLEANUP.run_delete(
        server, _reviewed(('_probe',)), record_path,
        target='redis://127.0.0.1:6380', pace_s=0.0, sleep=RecordingSleep(server),
    )

    assert code == CLEANUP.Exit.REFUSED
    assert server.deleted == []
    assert not record_path.exists()


# --- the adapter ------------------------------------------------------------------------


@pytest.mark.parametrize('reply', [{}, {'run_id': ''}, b'not a map'], ids=['absent', 'empty', 'not-a-map'])
def test_a_server_that_reports_no_run_id_fails_closed(reply):
    client = SimpleNamespace(connection=SimpleNamespace(info=lambda section: reply))

    with pytest.raises(RedisError, match='no run_id'):
        CLEANUP.FalkorGraphServer(client).run_id()


# --- the adapter, live on the scratch server --------------------------------------------


@pytest.mark.integration
@falkor_skipif()
class TestFalkorGraphServerOnScratch:
    """Only the two graphs this test creates are planned: scratch holds other tests' litter."""

    def test_it_counts_lists_and_deletes_only_the_reviewed_litter(self):
        litter = f'evalmem_test_{uuid.uuid4().hex[:8]}'
        kept = f'_test_7793_kept_{uuid.uuid4().hex[:8]}'
        client = FalkorDB(host=FALKOR_HOST, port=FALKOR_PORT)
        try:
            client.select_graph(litter).query('CREATE (:Litter), (:Litter)')
            client.select_graph(kept).query('CREATE (:Kept)')
            server = CLEANUP.FalkorGraphServer(client)

            assert (server.node_count(litter), server.node_count(kept)) == (2, 1)
            assert server.run_id()
            assert server.run_id() == server.run_id()
            assert {litter, kept} <= set(server.graph_names())

            plan = CLEANUP.plan_cleanup([litter, kept])
            record = CLEANUP.execute(server, plan, _reviewed((litter,), (kept,)), pace_s=0)

            names = server.graph_names()
            assert litter not in names
            assert kept in names
            assert server.node_count(kept) == 1
            assert record.deleted == (litter,)
            assert record.outcome == CLEANUP.Exit.OK
        finally:
            for name in (litter, kept):
                with contextlib.suppress(Exception):
                    client.select_graph(name).delete()
            client.close()
