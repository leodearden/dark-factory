"""Tests for orchestrator.verify_event_channel: DF's reader and writer of reify's
REIFY_VERIFY_EVENT_LOG verify event channel (reify PRD run-all-runner-infra-harness
§4 C2/C3, as amended by reify retro-critic ruling R1).

Every present channel is composed the way production writes it: DF's own
``channel_open`` header, then real producer bytes from
``fixtures/reify_verify_event_channel`` (see its PROVENANCE.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.verify_event_channel import (
    VERIFY_EVENT_LOG_ENV,
    VerifyEventChannel,
    VerifyEventLog,
    allocate_verify_event_log,
    encode_channel_open,
    parse_verify_event_channel,
    read_verify_event_channel,
)

FIXTURES = Path(__file__).parent / 'fixtures' / 'reify_verify_event_channel'
C2_LINE_CAP = 4000


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def header(verify_id: str, role: str = 'task') -> bytes:
    return encode_channel_open(role=role, verify_id=verify_id)


def record(**fields: object) -> bytes:
    return json.dumps({'v': 1, 'ts_ns': 1, 'pid': 1, **fields}).encode() + b'\n'


def run_start() -> bytes:
    return record(event='run_start', run_id='r', scope='all', branch='pool', discovered=1)


def run_finish(**overrides: object) -> bytes:
    fields: dict[str, object] = {
        'discovered': 1, 'failures': 1, 'failed': ['x.sh'], 'flaky': [], 'skipped': [],
        'interrupted': False, 'names_omitted': 0,
    }
    fields.update(overrides)
    return record(event='run_finish', **fields)


def parse(data: bytes) -> VerifyEventChannel:
    return parse_verify_event_channel(data, verify_id='V')


POOL = 'run_all-pool.events.jsonl'
SERIAL = 'run_all-serial.events.jsonl'
SLOT_FATAL = 'slot_timeout-fatal.events.jsonl'
SLOT_SOFT = 'slot_timeout-soft.events.jsonl'


def test_env_var_name_is_reifys() -> None:
    assert VERIFY_EVENT_LOG_ENV == 'REIFY_VERIFY_EVENT_LOG'


# (a) header shape -----------------------------------------------------------


def test_channel_open_header_is_one_c2_v1_line() -> None:
    line = encode_channel_open(role='task', verify_id='abc')

    assert line.endswith(b'\n')
    assert line.count(b'\n') == 1
    assert len(line) < C2_LINE_CAP
    decoded = json.loads(line)
    assert type(decoded['v']) is int and decoded['v'] == 1
    assert decoded['event'] == 'channel_open'
    assert decoded['role'] == 'task'
    assert decoded['verify_id'] == 'abc'
    assert type(decoded['ts_ns']) is int
    assert type(decoded['pid']) is int


# (b) present captures ------------------------------------------------------


@pytest.mark.parametrize('producer', [POOL, SERIAL])
def test_header_plus_captured_run_all_channel_is_present(producer: str) -> None:
    channel = parse(header('V') + fixture_bytes(producer))

    assert channel.present is True
    assert channel.run_starts == 1
    assert channel.fatal_slot_timeouts == 0
    assert channel.soft_slot_timeouts == 0
    assert channel.skipped_lines == 0
    assert channel.failed_run_all_members() == ['test_fail.sh']


# (c) task-role shape: the header alone makes the channel present -----------


def test_header_alone_is_present_with_no_run() -> None:
    channel = parse(header('V'))

    assert channel.present is True
    assert channel.run_starts == 0
    assert channel.failed_run_all_members() == []


def test_header_plus_slot_record_without_run_start_counts_the_timeout() -> None:
    channel = parse(header('V') + fixture_bytes(SLOT_FATAL))

    assert channel.present is True
    assert channel.run_starts == 0
    assert channel.fatal_slot_timeouts == 1


# (d) C3 rule 1 absence -----------------------------------------------------


@pytest.mark.parametrize(
    'data',
    [
        pytest.param(b'', id='empty'),
        pytest.param(fixture_bytes(POOL), id='run_start-without-header'),
        pytest.param(fixture_bytes(SLOT_FATAL), id='slot-record-without-header'),
        pytest.param(header('OTHER') + fixture_bytes(POOL), id='another-verifys-header'),
        pytest.param(
            json.dumps({
                'v': 2, 'ts_ns': 1, 'pid': 1, 'event': 'channel_open',
                'role': 'task', 'verify_id': 'V',
            }).encode() + b'\n',
            id='header-v2',
        ),
    ],
)
def test_channel_without_this_verifys_header_is_absent(data: bytes) -> None:
    channel = parse(data)

    assert channel.present is False
    assert channel.failed_run_all_members() is None


def test_absent_channel_still_counts_the_facts_it_saw() -> None:
    assert parse(fixture_bytes(POOL)).run_starts == 1


def test_reading_a_missing_path_is_absent(tmp_path: Path) -> None:
    channel = read_verify_event_channel(tmp_path / 'nope.jsonl', verify_id='V')

    assert channel.present is False
    assert channel.failed_run_all_members() is None


def test_reading_a_directory_is_absent(tmp_path: Path) -> None:
    channel = read_verify_event_channel(tmp_path, verify_id='V')

    assert channel.present is False
    assert channel.failed_run_all_members() is None


def test_reading_a_real_file_parses_it(tmp_path: Path) -> None:
    path = tmp_path / 'channel.jsonl'
    path.write_bytes(header('V') + fixture_bytes(POOL))

    channel = read_verify_event_channel(path, verify_id='V')

    assert channel.present is True
    assert channel.failed_run_all_members() == ['test_fail.sh']


# (e) D4: torn and malformed lines are skipped, every other record honoured --


def test_torn_final_line_is_skipped_and_the_rest_honoured() -> None:
    pool = fixture_bytes(POOL)
    finish_line = pool.rstrip(b'\n').rsplit(b'\n', 1)[1]
    assert b'"run_finish"' in finish_line

    channel = parse(header('V') + pool + finish_line[:40])

    assert channel.skipped_lines == 1
    assert channel.present is True
    assert channel.failed_run_all_members() == ['test_fail.sh']


@pytest.mark.parametrize(
    'bad_line',
    [
        pytest.param(b'not json at all', id='garbage'),
        pytest.param(b'[1, 2, 3]', id='json-array'),
        pytest.param(json.dumps({'v': 2, 'event': 'run_start'}).encode(), id='v2'),
        pytest.param(json.dumps({'v': '1', 'event': 'run_start'}).encode(), id='v-string'),
        pytest.param(json.dumps({'event': 'run_start'}).encode(), id='no-v'),
    ],
)
def test_malformed_middle_line_is_skipped(bad_line: bytes) -> None:
    pool = fixture_bytes(POOL)

    channel = parse(header('V') + bad_line + b'\n' + pool)

    assert channel.skipped_lines == 1
    assert channel.present is True
    assert channel.run_starts == 1
    assert channel.failed_run_all_members() == ['test_fail.sh']


def test_blank_lines_are_not_skipped_lines() -> None:
    channel = parse(header('V') + b'\n\n' + fixture_bytes(POOL) + b'\n')

    assert channel.skipped_lines == 0
    assert channel.failed_run_all_members() == ['test_fail.sh']


# (f) slot_timeout counts: only disposition == 'fatal' / 'soft' --------------


def test_fatal_and_soft_slot_timeouts_are_counted_separately() -> None:
    channel = parse(
        header('V') + fixture_bytes(POOL) + fixture_bytes(SLOT_FATAL) + fixture_bytes(SLOT_SOFT)
    )

    assert channel.fatal_slot_timeouts == 1
    assert channel.soft_slot_timeouts == 1


@pytest.mark.parametrize(
    'disposition_fields',
    [
        pytest.param({'disposition': 'FATAL'}, id='upper-case'),
        pytest.param({}, id='missing'),
    ],
)
def test_slot_timeout_without_exact_disposition_counts_in_neither(
    disposition_fields: dict[str, str],
) -> None:
    slot = record(event='slot_timeout', reason='test_slot', slots=8, waited=1800, **disposition_fields)

    channel = parse(header('V') + slot)

    assert channel.fatal_slot_timeouts == 0
    assert channel.soft_slot_timeouts == 0


# (g) C3 rule 3 narrowing ----------------------------------------------------


@pytest.mark.parametrize(
    'producer',
    [
        pytest.param(run_start() + run_finish(interrupted=True), id='interrupted-true'),
        pytest.param(run_start() + run_finish(interrupted=None), id='interrupted-null'),
        pytest.param(run_start() + run_finish(names_omitted=1), id='names-omitted'),
        pytest.param(run_start() + run_finish(failed='x.sh'), id='failed-not-a-list'),
        pytest.param(run_start() + run_finish(failed=['x.sh', 3]), id='failed-not-all-str'),
        pytest.param(run_start(), id='run-start-without-finish'),
        pytest.param(run_start() + run_start() + run_finish(), id='two-starts-one-finish'),
    ],
)
def test_incomplete_run_plan_refuses_to_narrow(producer: bytes) -> None:
    assert parse(header('V') + producer).failed_run_all_members() == []


def test_two_complete_runs_union_failed_in_first_seen_order() -> None:
    producer = (
        run_start() + run_finish(failed=['b.sh', 'a.sh'])
        + run_start() + run_finish(failed=['a.sh', 'c.sh'])
    )

    assert parse(header('V') + producer).failed_run_all_members() == ['b.sh', 'a.sh', 'c.sh']


def test_complete_run_with_no_failures_gives_empty() -> None:
    producer = run_start() + run_finish(failed=[], failures=0)

    assert parse(header('V') + producer).failed_run_all_members() == []


# allocator: VerifyEventLog / allocate_verify_event_log -----------------------


def test_allocated_log_holds_only_this_verifys_header_and_is_removed_after(
    tmp_path: Path,
) -> None:
    with allocate_verify_event_log(role='task', directory=tmp_path) as log:
        assert isinstance(log, VerifyEventLog)
        assert log.path is not None
        path = log.path
        assert path.is_absolute()
        assert path.is_file()
        assert path.parent == tmp_path
        assert isinstance(log.verify_id, str) and log.verify_id

        lines = path.read_bytes().splitlines()
        assert len(lines) == 1
        written = json.loads(lines[0])
        assert written['event'] == 'channel_open'
        assert written['role'] == 'task'
        assert written['verify_id'] == log.verify_id

        channel = log.read()
        assert channel.present is True
        assert channel.run_starts == 0
        assert channel.failed_run_all_members() == []

        base = {'A': '1'}
        exported = log.export_into(base)
        assert exported is not base
        assert exported == {'A': '1', VERIFY_EVENT_LOG_ENV: str(path)}
        assert base == {'A': '1'}

    assert not path.exists()


def test_allocated_log_is_removed_when_the_block_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match='boom'):
        with allocate_verify_event_log(role='task', directory=tmp_path) as log:
            assert log.path is not None
            path = log.path
            raise RuntimeError('boom')

    assert not path.exists()


def test_producer_deleting_the_log_is_tolerated(tmp_path: Path) -> None:
    with allocate_verify_event_log(role='task', directory=tmp_path) as log:
        assert log.path is not None
        log.path.unlink()

        assert log.read().present is False


def test_export_overrides_an_operator_supplied_channel(tmp_path: Path) -> None:
    with allocate_verify_event_log(role='task', directory=tmp_path) as log:
        exported = log.export_into({VERIFY_EVENT_LOG_ENV: str(tmp_path / 'operator.jsonl')})

        assert exported[VERIFY_EVENT_LOG_ENV] == str(log.path)


def test_nested_allocations_are_distinct(tmp_path: Path) -> None:
    with allocate_verify_event_log(role='task', directory=tmp_path) as outer:
        with allocate_verify_event_log(role='task', directory=tmp_path) as inner:
            assert outer.path != inner.path
            assert outer.verify_id != inner.verify_id


def test_producer_bytes_appended_to_the_log_are_read(tmp_path: Path) -> None:
    with allocate_verify_event_log(role='task', directory=tmp_path) as log:
        assert log.path is not None
        with log.path.open('ab') as channel_file:
            channel_file.write(fixture_bytes(POOL))

        channel = log.read()

    assert channel.present is True
    assert channel.failed_run_all_members() == ['test_fail.sh']


def test_allocated_header_carries_the_role(tmp_path: Path) -> None:
    with allocate_verify_event_log(role='merge', directory=tmp_path) as log:
        assert log.path is not None
        assert json.loads(log.path.read_bytes())['role'] == 'merge'


def test_failed_allocation_fails_open_to_no_channel(tmp_path: Path) -> None:
    with allocate_verify_event_log(role='task', directory=tmp_path / 'missing') as log:
        assert log.path is None
        assert log.export_into({VERIFY_EVENT_LOG_ENV: '/tmp/x', 'A': '1'}) == {'A': '1'}
        assert log.read().present is False
