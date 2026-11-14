"""The verify-cache audit: attributing audit reds to consumed greens, the leak response, the kill switch (task 5655).

Fixtures are kept MODULE-LOCAL (not conftest.py) for the reason stated in
test_config_verify_admission_reload.py. conftest's autouse ``_isolate_orch_config``
pins ``ORCH_PROJECT_ROOT`` to ``tmp_path``, so a CLI fixture's project IS ``tmp_path``.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from orchestrator.cli import main
from orchestrator.config import VerifyCacheConfig
from orchestrator.verify_cache import audit, keys, policy, registry
from orchestrator.verify_cache.audit_run import AuditNode
from orchestrator.verify_cache.modes import VERIFY_CACHE_ENV, CacheMode
from orchestrator.verify_evidence import UnitTier

T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)
HEAD = 'a1' * 20
LATER_HEAD = 'b2' * 20

ALPHA_NODE = str(AuditNode('alpha', 'tests/test_a.py::t1'))
ALPHA_NODE_TWO = str(AuditNode('alpha', 'tests/test_a.py::t2'))
BETA_NODE = str(AuditNode('beta', 'tests/test_b.py::t1'))
GAMMA_NODE = str(AuditNode('gamma', 'tests/test_g.py::t1'))
UNQUALIFIED_NODE = 'tests/test_a.py::t1'

_BOTH_ON = VerifyCacheConfig(task_mode=CacheMode.ON, merge_mode=CacheMode.ON)


@pytest.fixture(autouse=True)
def _no_inherited_force_off(monkeypatch):
    """The audit lane runs this suite under DF_VERIFY_CACHE=off, which forces every mode off."""
    monkeypatch.delenv(VERIFY_CACHE_ENV, raising=False)


def _key(module: str) -> keys.UnitKey:
    return keys.UnitKey.of(
        f'module:{module}:test', UnitTier.MODULE, 's', (('spec', 's'), (f'tree:{module}', module[0] * 40)),
    )


KA, KB, KG = _key('alpha'), _key('beta'), _key('gamma')


class _Seeded:
    """A real registry with greens for alpha, beta and gamma; alpha's and beta's consumed by HEAD."""

    def __init__(self, db_path: Path, t0: datetime = T0) -> None:
        self.now = t0
        self.registry = registry.Registry(db_path, clock=lambda: self.now)
        self.entry_alpha = self.green(KA)
        self.entry_beta = self.green(KB)
        self.entry_gamma = self.green(KG)
        self.consume(HEAD, self.entry_alpha, self.entry_beta)

    def green(self, key: keys.UnitKey) -> registry.Entry:
        return asyncio.run(self.registry.record_green(key, evidence_event_id=7, run_id='r1', sha='abc'))

    def consume(self, head: str, *entries: registry.Entry) -> None:
        asyncio.run(self.registry.consume(head, entries))

    def advance(self, delta: timedelta) -> datetime:
        self.now += delta
        return self.now

    def leak(self, head: str, *nodes: str) -> audit.LeakResponse:
        verdict = asyncio.run(audit.attribute(head, nodes, self.registry))
        return asyncio.run(audit.apply(verdict, self.registry, self.now))

    def policy(self) -> registry.PolicyState:
        return asyncio.run(self.registry.policy_state())

    def present(self, key: keys.UnitKey) -> bool:
        return bool(asyncio.run(self.registry.view([key])).entries)

    def fresh_leak_source(self, key: keys.UnitKey, head: str) -> None:
        """A NEW consumed entry for *key*'s unit: entry ids are never reused, so its leak is a new identity."""
        self.consume(head, self.green(key))


@pytest.fixture
def seeded(tmp_path: Path) -> _Seeded:
    return _Seeded(tmp_path / 'verify_cache.db')


def _records(state: registry.PolicyState) -> list[audit.LeakRecord]:
    return [audit.LeakRecord.parse(text) for text in state.leaks]


class TestAttribute:
    def test_a_node_on_a_consumed_unit_is_attributed_to_its_entry(self, seeded):
        verdict = asyncio.run(audit.attribute(HEAD, (ALPHA_NODE, ALPHA_NODE_TWO), seeded.registry))

        assert len(verdict.attributions) == 1
        attribution = verdict.attributions[0]
        assert attribution.entry == seeded.entry_alpha
        assert attribution.nodes == (ALPHA_NODE, ALPHA_NODE_TWO)
        record = attribution.record(T0)
        assert (record.unit_id, record.key, record.entry_id, record.evidence_event_id, record.landing_sha) == (
            'module:alpha:test', KA.key, seeded.entry_alpha.entry_id, 7, HEAD,
        )
        assert record.nodes == (ALPHA_NODE, ALPHA_NODE_TWO)
        assert verdict.unattributed == ()

    def test_a_unit_that_ran_and_an_unqualified_node_are_unattributed(self, seeded):
        verdict = asyncio.run(audit.attribute(HEAD, (GAMMA_NODE, UNQUALIFIED_NODE), seeded.registry))

        assert verdict.attributions == ()
        assert verdict.unattributed == (GAMMA_NODE, UNQUALIFIED_NODE)
        assert seeded.present(KG)
        assert seeded.policy() == registry.PolicyState()

    def test_nothing_reproduced_never_touches_the_registry(self, tmp_path):
        db_path = tmp_path / 'absent' / 'verify_cache.db'

        verdict = asyncio.run(audit.attribute(HEAD, (), registry.Registry(db_path)))

        assert (verdict.attributions, verdict.unattributed) == ((), ())
        assert not db_path.exists()


class TestApply:
    def test_row_12_one_leak_evicts_its_key_and_starts_the_streak(self, seeded):
        response = seeded.leak(HEAD, ALPHA_NODE)

        assert not seeded.present(KA)
        assert seeded.present(KB)
        assert response.evicted == {KA.key: 1}
        assert response.counted == ((KA.unit_id, KA.key, seeded.entry_alpha.entry_id),)
        assert response.policy.streak == 1
        assert not response.policy.is_disabled(seeded.now)
        assert (response.newly_disabled, response.l2_required) == (False, False)

    def test_the_same_leak_delivered_again_counts_once(self, seeded):
        for _ in range(3):
            response = seeded.leak(HEAD, ALPHA_NODE)

        assert response.policy.streak == 1
        assert response.counted == ()
        assert len(seeded.policy().leaks) == 1

    def test_row_13_a_second_leak_inside_seven_days_disables_for_fourteen(self, seeded):
        seeded.leak(HEAD, ALPHA_NODE)
        now = seeded.advance(timedelta(days=2))

        response = seeded.leak(HEAD, BETA_NODE)

        state = seeded.policy()
        assert state.disabled_until == now + timedelta(days=14)
        assert state.reason and 'module:alpha:test' in state.reason and 'module:beta:test' in state.reason
        assert {record.unit_id for record in _records(state)} == {'module:alpha:test', 'module:beta:test'}
        assert (response.newly_disabled, response.l2_required) == (True, True)
        view = asyncio.run(seeded.registry.view([KG]))
        for role in ('task', 'merge'):
            decision = policy.decide(_BOTH_ON, role, KG, view, never_cache=False, now=now)
            assert decision.decision is policy.Decision.DISABLED

    def test_a_leak_eight_days_after_the_first_starts_a_new_streak(self, seeded):
        seeded.leak(HEAD, ALPHA_NODE)
        seeded.advance(timedelta(days=8))

        response = seeded.leak(HEAD, BETA_NODE)

        assert response.policy.streak == 1
        assert response.policy.disabled_until is None
        assert [record.unit_id for record in _records(seeded.policy())] == ['module:beta:test']

    def test_two_distinct_leaks_in_one_verdict_disable_at_once(self, seeded):
        response = seeded.leak(HEAD, ALPHA_NODE, BETA_NODE)

        assert response.policy.streak == 2
        assert response.policy.is_disabled(seeded.now)
        assert response.newly_disabled

    def test_while_disabled_a_new_leak_is_recorded_and_the_disable_holds_until_it_expires(self, seeded):
        seeded.leak(HEAD, ALPHA_NODE, BETA_NODE)
        until = seeded.policy().disabled_until
        assert until is not None
        seeded.advance(timedelta(days=3))
        seeded.fresh_leak_source(KA, LATER_HEAD)

        during = seeded.leak(LATER_HEAD, ALPHA_NODE)

        assert during.policy.disabled_until == until
        assert len(during.policy.leaks) == 3
        assert (during.newly_disabled, during.l2_required) == (False, True)

        seeded.now = until + timedelta(days=1)
        seeded.fresh_leak_source(KB, 'c3' * 20)
        after = seeded.leak('c3' * 20, BETA_NODE)

        state = seeded.policy()
        assert state.disabled_until is None
        assert len(state.leaks) < registry.MIN_DISABLING_LEAKS
        assert (after.newly_disabled, after.l2_required) == (False, False)


class TestEnable:
    def test_enable_clears_the_disable_and_the_old_leaks_never_count_again(self, seeded):
        seeded.leak(HEAD, ALPHA_NODE, BETA_NODE)
        now = seeded.advance(timedelta(hours=1))

        state = asyncio.run(audit.enable(seeded.registry, 'fixed the key', now))

        assert state == seeded.policy()
        assert state.disabled_until is None
        assert state.reason and 'fixed the key' in state.reason
        assert state.streak == 0
        assert seeded.leak(HEAD, ALPHA_NODE).policy.streak == 0
        seeded.fresh_leak_source(KA, LATER_HEAD)
        assert seeded.leak(LATER_HEAD, ALPHA_NODE).policy.streak == 1

    @pytest.mark.parametrize('reason', ['', '   '])
    def test_a_blank_reason_is_refused(self, seeded, reason):
        with pytest.raises(ValueError, match='reason'):
            asyncio.run(audit.enable(seeded.registry, reason, seeded.now))


def _config(project: Path) -> Path:
    cfg = project / 'dark-factory-orchestrator.yaml'
    cfg.write_text(yaml.dump({'project_root': str(project)}))
    return cfg


class TestEnableCli:
    def test_enable_clears_a_disabled_registry_and_status_shows_it(self, tmp_path):
        seeded = _Seeded(registry.registry_db_path(tmp_path), t0=datetime.now(UTC))
        seeded.leak(HEAD, ALPHA_NODE, BETA_NODE)
        assert seeded.policy().is_disabled(datetime.now(UTC))
        cfg = _config(tmp_path)

        result = CliRunner().invoke(main, ['verify-cache', 'enable', '--reason', 'fixed X', '--config', str(cfg)])

        assert result.exit_code == 0, result.output
        assert 'policy: enabled (streak 0)' in result.output, result.output
        status = CliRunner().invoke(main, ['verify-cache', 'status', '--config', str(cfg)])
        assert 'policy: enabled' in status.output, status.output
        assert 'fixed X' in (seeded.policy().reason or '')

    def test_a_missing_reason_is_a_usage_error(self, tmp_path):
        result = CliRunner().invoke(main, ['verify-cache', 'enable', '--config', str(_config(tmp_path))])

        assert result.exit_code == 2, result.output
        assert '--reason' in result.output, result.output
