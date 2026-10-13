"""Contract suite for ``server/write_triage_harness.py``.

The harness is the one copy of the ``add_memory`` write-path config stand-in and
post-RRF search hit that the triage, near-duplicate, entities and completion-claim
tool suites share; these contracts hold for every importer, and are read back
through production's own resolvers so a misspelled leaf fails here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from fused_memory.server.near_duplicate_guard import (
    resolve_near_dup_guard_enabled,
    resolve_near_dup_threshold,
)
from fused_memory.server.write_triage import (
    OUTCOME_RESTATED,
    decide_band,
    resolve_bands,
    resolve_write_triage_enabled,
)
from fused_memory.server.write_triage_judge import resolve_judge_enabled
from server.write_triage_harness import (
    T_HIGH,
    T_LOW,
    configure_triage_config,
    search_candidate,
)


def test_the_defaults_read_as_triage_on_with_the_harness_bands() -> None:
    svc = AsyncMock()
    configure_triage_config(svc)

    assert resolve_write_triage_enabled(svc) is True
    assert resolve_bands(svc) == (T_HIGH, T_LOW)
    assert resolve_near_dup_guard_enabled(svc) is True
    assert resolve_near_dup_threshold(svc) == 0.90


def test_the_llm_judge_is_off_unless_a_test_turns_it_on() -> None:
    svc = AsyncMock()
    configure_triage_config(svc)
    assert resolve_judge_enabled(svc) is False, (
        'production defaults the judge ON when the leaf is missing or not a '
        'bool, so a dropped or misspelled leaf would let every consumer\'s '
        'middle-band write place a real, billed LLM call while still passing'
    )

    configure_triage_config(svc, judge_enabled=True)
    assert resolve_judge_enabled(svc) is True


def test_the_disabled_flags_read_as_disabled() -> None:
    svc = AsyncMock()
    configure_triage_config(svc, enabled=False, near_dup_guard_enabled=False)

    assert resolve_write_triage_enabled(svc) is False
    assert resolve_near_dup_guard_enabled(svc) is False


def test_the_default_candidate_lands_in_the_deterministic_band() -> None:
    decision = decide_band([search_candidate()], t_high=T_HIGH, t_low=T_LOW)

    assert decision.outcome == OUTCOME_RESTATED
    assert decision.canonical_id == 'm1'
    assert decision.similarity == pytest.approx(0.97)
