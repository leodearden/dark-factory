"""Tests for the `deterministic-*` done-provenance caller bar (PRD C5, task 5241).

Modelled clause for clause on ``tests/server/test_update_memory_authz_gate.py``
(``TestLiveRead`` / ``TestFailsClosed``), substituting a config object for that
module's ``memory_service`` stand-in — the leaf this resolver reads lives at
``reconciliation.deterministic_provenance_allowed_agent_prefixes`` rather than
under ``mem0_update``.

Same deliberate inversion those tests carry: this is a MUTATION-AUTHORIZATION
gate, so every fallback must DENY. A missing config hop, an unspecced Mock, or a
non-list leaf are all refusals, never a quiet fall-back to the schema default.

The resolver lives in its own module (``middleware/done_provenance_authz.py``)
precisely so these tests can call it DIRECTLY: ``config/reload.py``'s
reload-safety rule requires a test proving the consumer re-reads config live
before its leaf may be registered green-tier, which task 5237 already did.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import get_args
from unittest.mock import MagicMock

import pytest
from shared.task_metadata import DoneProvenance

from fused_memory.config.schema import FusedMemoryConfig
from fused_memory.middleware.done_provenance_authz import (
    DETERMINISTIC_CALLER_ERROR_TYPE,
    DETERMINISTIC_PROVENANCE_KINDS,
    DoneProvenanceAuthzDecision,
    resolve_deterministic_provenance_allowed_prefixes,
    resolve_deterministic_provenance_authorization,
)

#: The dotted config path every refusal must name so it is self-remedying.
_CONFIG_PATH = 'reconciliation.deterministic_provenance_allowed_agent_prefixes'


class TestGatedKindFamily:
    """The gated family is DERIVED from the shared Literal, never hand-listed."""

    def test_matches_the_shared_literal_in_lockstep(self):
        declared = get_args(DoneProvenance.model_fields['kind'].annotation)
        expected = {k for k in declared if k.startswith('deterministic-')}

        assert frozenset(expected) == DETERMINISTIC_PROVENANCE_KINDS, (
            'the gated family must be derived from DoneProvenance.kind, so a 5th '
            'deterministic kind added to the Literal is gated BY DEFAULT'
        )
        assert len(DETERMINISTIC_PROVENANCE_KINDS) == 4

    @pytest.mark.parametrize('kind', ['merged', 'found_on_main', 'operational-verified'])
    def test_non_deterministic_kinds_are_not_gated(self, kind):
        assert kind not in DETERMINISTIC_PROVENANCE_KINDS


class TestAllows:
    """The shipped default admits exactly the orchestrator."""

    def test_orchestrator_is_allowed_by_the_shipped_default(self):
        decision = resolve_deterministic_provenance_authorization(
            FusedMemoryConfig(), agent_id='orchestrator',
        )
        assert decision.allowed is True
        assert decision.error_type is None
        assert decision.error is None

    def test_prefix_match_not_exact_match(self):
        """'orchestrator' is a PREFIX: a suffixed client still clears the bar."""
        decision = resolve_deterministic_provenance_authorization(
            FusedMemoryConfig(), agent_id='orchestrator-merge-worker',
        )
        assert decision.allowed is True


class TestDeniesCaller:
    """A caller matching no prefix is refused with a self-remedying message."""

    @pytest.mark.parametrize('agent_id', [
        'dashboard',                        # a real MCP client's clientInfo
        'cgl-sched-gate',                   # the retired one-shot (PRD §7)
        'recon-stage-task_knowledge_sync',  # the 5156 incident's writer
        'claude-interactive',
        '',
        None,
        42,
    ])
    def test_unlisted_caller_is_refused(self, agent_id):
        decision = resolve_deterministic_provenance_authorization(
            FusedMemoryConfig(), agent_id=agent_id,
        )
        assert decision.allowed is False
        assert decision.error_type == DETERMINISTIC_CALLER_ERROR_TYPE
        assert decision.error_type == 'DeterministicProvenanceCallerNotPermitted'

    @pytest.mark.parametrize('agent_id', ['dashboard', '', None, 42])
    def test_refusal_names_the_caller_the_list_and_the_config_path(self, agent_id):
        decision = resolve_deterministic_provenance_authorization(
            FusedMemoryConfig(), agent_id=agent_id,
        )
        assert repr(agent_id) in (decision.error or ''), (
            'the refusal must name the offending agent_id verbatim'
        )
        assert "'orchestrator'" in (decision.error or ''), (
            'the refusal must name the configured prefix list'
        )
        assert _CONFIG_PATH in (decision.error or ''), (
            'the refusal must name the dotted config path so it is self-remedying'
        )

    def test_empty_list_is_the_kill_switch(self):
        cfg = FusedMemoryConfig()
        cfg.reconciliation.deterministic_provenance_allowed_agent_prefixes = []

        decision = resolve_deterministic_provenance_authorization(
            cfg, agent_id='orchestrator',
        )
        assert decision.allowed is False, (
            'an EMPTY list must deny every caller — the live incident kill switch'
        )
        assert decision.error_type == DETERMINISTIC_CALLER_ERROR_TYPE


class TestFailsClosed:
    """Unlike a soft-block guard, a missing/corrupt leaf must DENY."""

    @pytest.mark.parametrize('config', [
        None,                                    # no config object at all
        SimpleNamespace(),                       # no reconciliation section
        SimpleNamespace(reconciliation=None),
        MagicMock(),                             # unspecced: the leaf is a Mock
    ])
    def test_missing_config_hop_denies_and_does_not_raise(self, config):
        decision = resolve_deterministic_provenance_authorization(
            config, agent_id='orchestrator',
        )
        assert decision.allowed is False, (
            'a mutation-authorization gate must fail CLOSED, not fall back to '
            'the schema default'
        )
        assert decision.error_type == DETERMINISTIC_CALLER_ERROR_TYPE

    @pytest.mark.parametrize('corrupt', ['orchestrator', 42, None, {'a': 1}, ('orchestrator',)])
    def test_non_list_leaf_denies(self, corrupt):
        """A bare string is the load-bearing case: ``startswith`` would still
        accept it, silently gating on something the operator never wrote."""
        cfg = FusedMemoryConfig()
        object.__setattr__(
            cfg.reconciliation,
            'deterministic_provenance_allowed_agent_prefixes',
            corrupt,
        )
        assert resolve_deterministic_provenance_authorization(
            cfg, agent_id='orchestrator',
        ).allowed is False

    def test_non_str_and_empty_members_are_dropped(self):
        cfg = FusedMemoryConfig()
        object.__setattr__(
            cfg.reconciliation,
            'deterministic_provenance_allowed_agent_prefixes',
            ['orchestrator', '', None, 42, 'dashboard'],
        )
        assert resolve_deterministic_provenance_allowed_prefixes(cfg) == (
            'orchestrator', 'dashboard',
        )
        # An empty-string member must not become a prefix that matches everyone.
        assert resolve_deterministic_provenance_authorization(
            cfg, agent_id='stranger',
        ).allowed is False

    def test_missing_hop_resolves_to_the_empty_default(self):
        assert resolve_deterministic_provenance_allowed_prefixes(None) == ()
        assert resolve_deterministic_provenance_allowed_prefixes(SimpleNamespace()) == ()


class TestLiveRead:
    """config/reload.py's precondition for the green-tier registration (5237)."""

    def test_reads_config_live_on_every_call(self):
        cfg = FusedMemoryConfig()
        first = resolve_deterministic_provenance_authorization(cfg, agent_id='dashboard')
        assert first.allowed is False, 'dashboard is not on the default bar'

        # Mutate the SHARED config object in place, exactly as apply_reload does.
        cfg.reconciliation.deterministic_provenance_allowed_agent_prefixes.append('dashboard')

        second = resolve_deterministic_provenance_authorization(cfg, agent_id='dashboard')
        assert second.allowed is True, (
            'the resolver must re-read config on every call; a value captured at '
            'import or construction would make the leaf restart-only in disguise'
        )

    def test_kill_switch_flipped_in_place_takes_effect(self):
        cfg = FusedMemoryConfig()
        assert resolve_deterministic_provenance_authorization(
            cfg, agent_id='orchestrator',
        ).allowed is True

        cfg.reconciliation.deterministic_provenance_allowed_agent_prefixes = []

        assert resolve_deterministic_provenance_authorization(
            cfg, agent_id='orchestrator',
        ).allowed is False


class TestDecisionIsAValue:
    """The verdict is returned, never raised (INV-1)."""

    def test_decision_is_a_frozen_dataclass(self):
        assert dataclasses.is_dataclass(DoneProvenanceAuthzDecision)

        decision = resolve_deterministic_provenance_authorization(None, agent_id='x')
        assert isinstance(decision, DoneProvenanceAuthzDecision)
        # FROZEN is proved behaviourally rather than by reading
        # __dataclass_params__: the property that matters is that a verdict
        # cannot be mutated after it has been handed to a caller.
        with pytest.raises(dataclasses.FrozenInstanceError):
            decision.allowed = True  # type: ignore[misc]
