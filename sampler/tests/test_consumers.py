"""Consumer names and the D2 rule table (``sampler.consumers``), pure over path segments.

PRD ``plans/per-consumer-resource-recording-prd.md`` D2: members are leaf
cgroups named by ONE ordered rule table; every inner cgroup adds a residual.
The paths below are shaped on the dev host's live tree (2026-11-17), where
``app.slice``, ``df.slice``, ``reify.slice`` and ``session.slice`` are
siblings under ``user@1000.service``.
"""

from __future__ import annotations

import os

import pytest

from sampler.consumers import (
    LEAF_RULES,
    Consumer,
    Role,
    classify_member,
    classify_residual,
    encode_consumer,
    parse_consumer,
)

_USER = ('user.slice', 'user-1000.slice', 'user@1000.service')
_UUID = '5ea08a53433a'
_GOVERNED = 'df.slice/df-reify.slice/df-reify-governed.slice'
_GOVERNED_AGENTS = f'{_GOVERNED}/df-reify-governed-agents.slice'


def _u(path: str) -> tuple[str, ...]:
    """Segments of a path under the user manager, e.g. ``app.slice/x.service``."""
    return _USER + tuple(path.split('/'))


def _c(role: Role, key: str | None = None) -> Consumer:
    return Consumer(role, key)


class TestConsumerName:
    def test_the_roles(self):
        assert {role.value for role in Role} == {
            'service',
            'agents',
            'verify',
            'project',
            'shared',
            'interactive',
            'system',
            'other',
            'exited',
            'kernel',
        }

    @pytest.mark.parametrize(
        ('consumer', 'text'),
        [
            (_c(Role.SERVICE, 'orchestrator-reify.service'), 'service@orchestrator-reify.service'),
            (_c(Role.AGENTS, 'df-dark_factory.slice'), 'agents@df-dark_factory.slice'),
            (_c(Role.VERIFY, 'dark-factory-c889f090'), 'verify@dark-factory-c889f090'),
            (_c(Role.PROJECT, 'reify.slice'), 'project@reify.slice'),
            (_c(Role.SHARED, 'sccache.service'), 'shared@sccache.service'),
            (_c(Role.INTERACTIVE, 't3code'), 'interactive@t3code'),
            (_c(Role.SYSTEM, 'docker'), 'system@docker'),
            (_c(Role.OTHER, '/'), 'other@/'),
            (_c(Role.EXITED, 'user@1000.service'), 'exited@user@1000.service'),
            (_c(Role.KERNEL), 'kernel'),
        ],
    )
    def test_encode_and_parse_are_one_pair(self, consumer, text):
        assert encode_consumer(consumer) == text
        assert parse_consumer(text) == consumer

    def test_every_role_round_trips(self):
        for role in Role:
            consumer = _c(role) if role is Role.KERNEL else _c(role, 'k@ey')
            assert parse_consumer(encode_consumer(consumer)) == consumer

    @pytest.mark.parametrize(
        'text', ['nobody@x', 'service@', 'service', 'kernel@x', '', '@x', 'Service@x']
    )
    def test_a_malformed_name_raises_naming_the_text(self, text):
        with pytest.raises(ValueError) as raised:
            parse_consumer(text)
        assert repr(text) in str(raised.value)

    @pytest.mark.parametrize(('role', 'key'), [(Role.SERVICE, None), (Role.SHARED, ''), (Role.KERNEL, 'x')])
    def test_only_the_kernel_is_unkeyed(self, role, key):
        with pytest.raises(ValueError):
            Consumer(role, key)

    def test_consumers_sort_with_the_unkeyed_kernel_among_them(self):
        consumers = [_c(Role.SHARED, 'b'), _c(Role.KERNEL), _c(Role.SHARED, 'a')]

        assert sorted(consumers) == [_c(Role.KERNEL), _c(Role.SHARED, 'a'), _c(Role.SHARED, 'b')]


class TestLeafRules:
    def test_the_table_is_one_ordered_tuple(self):
        assert isinstance(LEAF_RULES, tuple)
        assert [rule.name for rule in LEAF_RULES] == [
            'orchestrator_service',
            'verify_scope',
            'project_slice',
            'factory_service',
            't3code',
            'terminals',
            'docker',
            'nearest_anchor',
        ]

    @pytest.mark.parametrize(
        ('path', 'consumer'),
        [
            ('app.slice/orchestrator-dark-factory.service', _c(Role.SERVICE, 'orchestrator-dark-factory.service')),
            ('app.slice/orchestrator-reify.service/daemon', _c(Role.SERVICE, 'orchestrator-reify.service')),
            ('app.slice/orchestrator-x.service/.control', _c(Role.SERVICE, 'orchestrator-x.service')),
            ('app.slice/orchestrator-x.service/agent-7', _c(Role.AGENTS, 'orchestrator-x.service')),
            (
                'df.slice/df-dark_factory.slice/orchestrator-dark-factory.service/daemon',
                _c(Role.SERVICE, 'df-dark_factory.slice'),
            ),
            (
                'df.slice/df-dark_factory.slice/orchestrator-dark-factory.service/agent-1',
                _c(Role.AGENTS, 'df-dark_factory.slice'),
            ),
            (f'{_GOVERNED}/orchestrator-reify.service/daemon', _c(Role.SERVICE, 'df-reify.slice')),
            (f'{_GOVERNED}/orchestrator-reify.service/agent-3', _c(Role.AGENTS, 'df-reify.slice')),
        ],
    )
    def test_rule_1_orchestrator_service(self, path, consumer):
        assert classify_member(_u(path)) == consumer

    def test_the_watchdog_is_shared_never_a_project_service(self):
        assert classify_member(_u('app.slice/orchestrator-watchdog.service')) == _c(
            Role.SHARED, 'orchestrator-watchdog.service'
        )

    @pytest.mark.parametrize(
        ('path', 'tag'),
        [
            (f'app.slice/df-verify-dark-factory-c889f090-{_UUID}.scope', 'dark-factory-c889f090'),
            (f'df.slice/df-x.slice/df-verify-x-1234abcd-{_UUID}.scope', 'x-1234abcd'),
        ],
    )
    def test_rule_2_both_verify_placements(self, path, tag):
        assert classify_member(_u(path)) == _c(Role.VERIFY, tag)

    @pytest.mark.parametrize(
        ('path', 'consumer'),
        [
            ('df.slice/df-x.slice/anything.scope', _c(Role.PROJECT, 'df-x.slice')),
            ('df.slice/df-x.slice', _c(Role.PROJECT, 'df-x.slice')),
            (
                'reify.slice/reify-governed.slice/reify-governed-agents.slice',
                _c(Role.PROJECT, 'reify.slice'),
            ),
            (_GOVERNED_AGENTS, _c(Role.PROJECT, 'df-reify.slice')),
            (f'{_GOVERNED_AGENTS}/x.scope', _c(Role.PROJECT, 'df-reify.slice')),
        ],
    )
    def test_rule_3_project_slice(self, path, consumer):
        assert classify_member(_u(path)) == consumer

    @pytest.mark.parametrize(
        'unit',
        [
            'sccache.service',
            'fused-memory.service',
            'dark-factory-dashboard.service',
            'reify-warm-lane-gc.service',
            'reify-jobserver.service',
            'pytest-jobserver.service',
            'cliproxyapi.service',
            'dark-factory-load-sampler.service',
        ],
    )
    def test_rule_4_factory_service(self, unit):
        assert classify_member(_u(f'app.slice/{unit}')) == _c(Role.SHARED, unit)

    def test_rule_4_the_quality_snapshot_slice(self):
        slice_name = 'app-quality\\x2dsnapshot.slice'

        assert classify_member(_u(f'app.slice/{slice_name}/run-u42.service')) == _c(
            Role.SHARED, slice_name
        )

    @pytest.mark.parametrize(
        'path',
        [
            'app.slice/t3code.service',
            'app.slice/app-t3code.slice/app-t3code-agents.slice/x.slice',
            'app.slice/app-com.t3tools.T3Code-123.scope',
        ],
    )
    def test_rule_5_t3code(self, path):
        assert classify_member(_u(path)) == _c(Role.INTERACTIVE, 't3code')

    def test_rule_6_terminals(self):
        assert classify_member(_u('app.slice/app-org.kde.konsole-42.scope')) == _c(
            Role.INTERACTIVE, 'terminals'
        )

    def test_rule_7_docker(self):
        assert classify_member(('system.slice', 'docker-3ed551ad.scope')) == _c(
            Role.SYSTEM, 'docker'
        )

    @pytest.mark.parametrize(
        ('segments', 'consumer'),
        [
            (('init.scope',), _c(Role.OTHER, '/')),
            (('dev-hugepages.mount',), _c(Role.OTHER, '/')),
            (('system.slice', 'ssh.service'), _c(Role.OTHER, 'system.slice')),
            (('user.slice', 'user-1000.slice', 'session-3.scope'), _c(Role.OTHER, 'user-1000.slice')),
            (_u('session.slice/pipewire.service'), _c(Role.OTHER, 'session.slice')),
            (_u('app.slice/app-signal-1.scope'), _c(Role.OTHER, 'app.slice')),
            (_u('df.slice'), _c(Role.OTHER, 'user@1000.service')),
            (_u('init.scope'), _c(Role.OTHER, 'user@1000.service')),
            (_u('background.slice'), _c(Role.OTHER, 'background.slice')),
        ],
    )
    def test_rule_8_nearest_anchor(self, segments, consumer):
        assert classify_member(segments) == consumer


class TestResidualRules:
    @pytest.mark.parametrize(
        ('segments', 'consumer'),
        [
            (_u('app.slice/orchestrator-x.service'), _c(Role.AGENTS, 'orchestrator-x.service')),
            (
                _u('df.slice/df-x.slice/orchestrator-x.service'),
                _c(Role.AGENTS, 'df-x.slice'),
            ),
            (_u('df.slice/df-x.slice'), _c(Role.PROJECT, 'df-x.slice')),
            (_u('reify.slice'), _c(Role.PROJECT, 'reify.slice')),
            (_u('reify.slice/reify-governed.slice'), _c(Role.PROJECT, 'reify.slice')),
            (_u(_GOVERNED), _c(Role.PROJECT, 'df-reify.slice')),
            (_u(_GOVERNED_AGENTS), _c(Role.PROJECT, 'df-reify.slice')),
            (_u(f'{_GOVERNED}/orchestrator-reify.service'), _c(Role.AGENTS, 'df-reify.slice')),
            ((), _c(Role.KERNEL)),
            (_u('app.slice/app-t3code.slice'), _c(Role.INTERACTIVE, 't3code')),
            (_u('df.slice'), _c(Role.EXITED, 'user@1000.service')),
            (_u('app.slice'), _c(Role.EXITED, 'app.slice')),
            (_u('app.slice/app-org.gnome.Terminal.slice'), _c(Role.EXITED, 'app.slice')),
            (('system.slice', 'systemd-udevd.service'), _c(Role.EXITED, 'system.slice')),
            (('system.slice',), _c(Role.EXITED, 'system.slice')),
            (_USER, _c(Role.EXITED, 'user@1000.service')),
            (('user.slice',), _c(Role.EXITED, 'user.slice')),
        ],
    )
    def test_residual_order(self, segments, consumer):
        assert classify_residual(segments) == consumer

    def test_a_governed_cgroup_keeps_its_project_key_when_it_gains_a_child(self):
        path = _u(_GOVERNED_AGENTS)
        project_depth = path.index('df-reify.slice') + 1
        prefixes = [path[:depth] for depth in range(project_depth, len(path) + 1)]

        assert len(prefixes) == 3
        for prefix in prefixes:
            assert classify_member(prefix) == _c(Role.PROJECT, 'df-reify.slice'), prefix
            assert classify_residual(prefix) == _c(Role.PROJECT, 'df-reify.slice'), prefix


# ---------------------------------------------------------------------------
# The collectors (sampler.metrics), over a fixture host
# ---------------------------------------------------------------------------

_T0 = 1_000_000
_DUE = 30
_GAUGE_NAMES = {
    'consumer_memory_bytes', 'consumer_anon_bytes', 'consumer_file_bytes', 'consumer_swap_bytes',
}
_INCREMENT_NAMES = {
    'consumer_cpu_usec', 'consumer_swapin_pages', 'consumer_reclaim_kswapd_pages',
    'consumer_reclaim_direct_pages', 'consumer_refault_anon_pages',
    'consumer_refault_file_pages', 'consumer_memory_high_events', 'consumer_memory_max_events',
    'consumer_oom_kills', 'consumer_cpu_some_stall_seconds',
    'consumer_memory_some_stall_seconds', 'consumer_memory_full_stall_seconds',
    'consumer_io_some_stall_seconds',
}
_SERVICE = encode_consumer(Consumer(Role.SERVICE, 'orchestrator-reify.service'))
_KERNEL = encode_consumer(Consumer(Role.KERNEL))


@pytest.fixture
def world(tmp_path):
    from _consumer_world import consumer_world

    return consumer_world(tmp_path)


def _by_leaf(samples) -> dict[str | None, dict[str, float]]:
    rows: dict[str | None, dict[str, float]] = {}
    for key, value in samples.items():
        rows.setdefault(key.leaf, {})[key.metric] = value
    return rows


class TestTheConsumerCollector:
    def test_a_first_collection_writes_gauges_members_and_readability_but_no_increments(
        self, world
    ):
        collected = world.collect_consumers(_T0, None)
        rows = _by_leaf(collected.samples)

        assert set(rows[_SERVICE]) == _GAUGE_NAMES | {'consumer_members', 'consumer_read_ok'}
        assert rows[_SERVICE]['consumer_members'] == 1.0
        assert rows[_SERVICE]['consumer_memory_bytes'] == 1_000.0
        for leaf, metrics in rows.items():
            assert not set(metrics) & _INCREMENT_NAMES, (leaf, metrics)

    def test_every_leaf_is_an_encoded_consumer(self, world):
        leaves = {key.leaf for key in world.collect_consumers(_T0, None).samples if key.leaf}

        assert {encode_consumer(parse_consumer(leaf)) for leaf in leaves} == leaves
        assert {
            _SERVICE,
            encode_consumer(Consumer(Role.VERIFY, 'dark-factory-c889f090')),
            encode_consumer(Consumer(Role.VERIFY, 'x-1234abcd')),
            encode_consumer(Consumer(Role.PROJECT, 'df-x.slice')),
            _KERNEL,
        } <= leaves

    def test_the_host_level_rows_of_a_first_collection(self, world):
        from _consumer_world import MEM_AVAILABLE_KIB, MEM_TOTAL_KIB

        host = _by_leaf(world.collect_consumers(_T0, None).samples)[None]

        assert host == {
            'host_anon_bytes': 7_000.0,
            'host_file_bytes': 3_000.0,
            'host_used_bytes': float((MEM_TOTAL_KIB - MEM_AVAILABLE_KIB) * 1024),
        }

    def test_the_next_collection_yields_increments_from_the_carried_state(self, world):
        from _consumer_world import DAEMON, KCOMPACTD_PID, KSWAPD_PID

        first = world.collect_consumers(_T0, None)
        world.cgroups.set(DAEMON, cpu_usec=4_000, memory_some=3_000_000, pgsteal_direct=5)
        world.thread(KSWAPD_PID, 'kswapd0', 13)
        world.thread(KCOMPACTD_PID, 'kcompactd0', 20)

        second = world.collect_consumers(_T0 + _DUE, first.state)
        rows = _by_leaf(second.samples)

        assert set(rows[_SERVICE]) == (
            _GAUGE_NAMES | _INCREMENT_NAMES | {'consumer_members', 'consumer_read_ok'}
        )
        assert rows[_SERVICE]['consumer_cpu_usec'] == 3_000.0
        assert rows[_SERVICE]['consumer_reclaim_direct_pages'] == 5.0
        assert rows[_SERVICE]['consumer_memory_some_stall_seconds'] == pytest.approx(3.0)
        assert rows[_SERVICE]['consumer_read_ok'] == 1.0
        host = rows[None]
        assert host['consumer_interval_seconds'] == float(_DUE)
        tick_usec = 1_000_000 / os.sysconf('SC_CLK_TCK')
        assert host['reclaim_kswapd_cpu_usec'] == pytest.approx(3 * tick_usec)
        assert host['reclaim_kcompactd_cpu_usec'] == 0.0

    def test_a_gap_writes_the_interval_but_no_increments(self, world):
        first = world.collect_consumers(_T0, None)

        late = _by_leaf(world.collect_consumers(_T0 + 3 * _DUE, first.state).samples)

        assert late[None]['consumer_interval_seconds'] == float(3 * _DUE)
        assert 'reclaim_kswapd_cpu_usec' not in late[None]
        assert late[_SERVICE]['consumer_read_ok'] == 0.0
        assert not set(late[_SERVICE]) & _INCREMENT_NAMES

    def test_a_corrupt_state_is_a_first_collection_not_an_exception(self, world):
        rows = _by_leaf(world.collect_consumers(_T0, '{"version": 1, "tree": 7').samples)

        assert not set(rows[_SERVICE]) & _INCREMENT_NAMES
        assert 'consumer_interval_seconds' not in rows[None]

    def test_a_consumer_that_does_not_exist_writes_no_rows(self, world):
        from _consumer_world import APP_SCOPE

        first = world.collect_consumers(_T0, None)
        world.cgroups.remove(APP_SCOPE)

        leaves = {key.leaf for key in world.collect_consumers(_T0 + _DUE, first.state).samples}

        assert encode_consumer(Consumer(Role.VERIFY, 'dark-factory-c889f090')) not in leaves
        assert encode_consumer(Consumer(Role.INTERACTIVE, 't3code')) not in leaves

    def test_the_kernel_has_no_swap_row_rather_than_a_zero_one(self, world):
        rows = _by_leaf(world.collect_consumers(_T0, None).samples)

        assert 'consumer_memory_bytes' in rows[_KERNEL]
        assert 'consumer_swap_bytes' not in rows[_KERNEL]


class TestTheVerifyScopeCollector:
    def test_it_counts_scopes_in_both_placements_and_sums_their_memory(self, world):
        collected = world.collect_verify_scopes(_T0, None)

        assert {key.metric: value for key, value in collected.samples.items()} == {
            'verify_scopes_live': 2.0,
            'verify_scope_memory_bytes': 5_000.0,
        }
        assert all(key.leaf is None for key in collected.samples)
        assert collected.state is None

    def test_only_the_scopes_own_files_are_read(self, world, monkeypatch):
        from _cgroup_reads import record_reads
        from _consumer_world import APP_SCOPE, DF_SCOPE

        read = record_reads(monkeypatch)

        world.collect_verify_scopes(_T0, None)

        assert sorted(read) == sorted(world.cgroups.path(s) for s in (APP_SCOPE, DF_SCOPE))

    def test_an_unreadable_scope_is_live_and_its_memory_is_not_written(self, world):
        from _consumer_world import APP_SCOPE

        world.cgroups.make_unreadable(APP_SCOPE, 'memory.current')

        samples = world.collect_verify_scopes(_T0, None).samples

        assert {key.metric: value for key, value in samples.items()} == {
            'verify_scopes_live': 2.0,
        }

    def test_a_scope_that_exits_after_it_was_listed_is_not_live(self, world, monkeypatch):
        from _cgroup_reads import exit_after_listing
        from _consumer_world import APP_SCOPE

        exit_after_listing(monkeypatch, world.cgroups.path(APP_SCOPE))

        samples = world.collect_verify_scopes(_T0, None).samples

        assert {key.metric: value for key, value in samples.items()} == {
            'verify_scopes_live': 1.0,
            'verify_scope_memory_bytes': 3_000.0,
        }

    def test_with_none_live_it_writes_zero_and_zero(self, world):
        from _consumer_world import APP_SCOPE, DF_SCOPE

        world.cgroups.remove(APP_SCOPE)
        world.cgroups.remove(DF_SCOPE)

        samples = world.collect_verify_scopes(_T0, None).samples

        assert {key.metric: value for key, value in samples.items()} == {
            'verify_scopes_live': 0.0,
            'verify_scope_memory_bytes': 0.0,
        }
