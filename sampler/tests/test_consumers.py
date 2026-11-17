"""Consumer names and the D2 rule table (``sampler.consumers``), pure over path segments.

PRD ``plans/per-consumer-resource-recording-prd.md`` D2: members are leaf
cgroups named by ONE ordered rule table; every inner cgroup adds a residual.
The paths below are shaped on the dev host's live tree (2026-11-17), where
``app.slice``, ``df.slice``, ``reify.slice`` and ``session.slice`` are
siblings under ``user@1000.service``.
"""

from __future__ import annotations

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
