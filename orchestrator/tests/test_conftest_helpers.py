"""Contract tests for the ``mock_orch_config`` fixture.

These verify the *non-trivial* behavior that the fixture promises and which
wouldn't survive accidental refactoring — specifically that ``spec_set``
typo-rejection is wired up on both the top-level mock and every sub-section.

Tests of plain attribute defaults (e.g. ``mock.usage_cap.enabled is False``)
are deliberately omitted — they would just duplicate literals from
``conftest.py`` two lines away. The 7 downstream harness fixtures that
consume the fixture in earnest are the de-facto contract for those.
"""

import pytest


def test_top_level_typo_rejected(mock_orch_config):
    with pytest.raises(AttributeError):
        mock_orch_config.projcet_root = '/tmp/typo'


def test_usage_cap_typo_rejected(mock_orch_config):
    with pytest.raises(AttributeError):
        mock_orch_config.usage_cap.enabld = True


def test_review_typo_rejected(mock_orch_config):
    with pytest.raises(AttributeError):
        mock_orch_config.review.enabld = True


def test_sandbox_typo_rejected(mock_orch_config):
    with pytest.raises(AttributeError):
        mock_orch_config.sandbox.bakcend = 'auto'


def test_fused_memory_typo_rejected(mock_orch_config):
    with pytest.raises(AttributeError):
        mock_orch_config.fused_memory.projcet_id = 'oops'
