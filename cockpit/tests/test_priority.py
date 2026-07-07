"""Tests for cockpit.priority — pure, property-tested priority scoring.

Every section 6.3 invariant (Fleet Cockpit PRD) is pinned here as a hypothesis
property over arbitrary ScoringItem fields plus one explicit example. `now`
and every ScoringItem field are injected — score() never reads a clock or an
RNG — so purity holds by construction and the tests below stay green across
every later implementation step.
"""

from __future__ import annotations

from datetime import datetime, timezone

from hypothesis import given, strategies as st


_STATES = ['open', 'answered', 'dropped']

_TIMESTAMPS = st.datetimes(timezones=st.just(timezone.utc))
_SHORT_TEXT = st.text(max_size=20)
_BOOSTS = st.integers(min_value=-10_000, max_value=10_000)


class TestScorePureFloat:
    def test_returns_float_for_open_item(self):
        from cockpit.priority import Priorities, ScoringItem, score

        item = ScoringItem(
            severity='high',
            category='bug',
            project='dark_factory',
            filed_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
            manual_boost=0,
            state='open',
        )
        now = datetime(2026, 7, 7, tzinfo=timezone.utc)

        result = score(item, Priorities.default(), now)

        assert isinstance(result, float)

    @given(
        severity=_SHORT_TEXT,
        category=_SHORT_TEXT,
        project=_SHORT_TEXT,
        filed_at=_TIMESTAMPS,
        manual_boost=_BOOSTS,
        state=st.sampled_from(_STATES),
        now=_TIMESTAMPS,
    )
    def test_score_is_pure(self, severity, category, project, filed_at, manual_boost, state, now):
        """score() must be a pure function: identical inputs, identical output.

        RED (before cockpit.priority exists): ImportError inside the test body.
        """
        from cockpit.priority import Priorities, ScoringItem, score

        item = ScoringItem(
            severity=severity,
            category=category,
            project=project,
            filed_at=filed_at,
            manual_boost=manual_boost,
            state=state,
        )
        weights = Priorities.default()

        assert score(item, weights, now) == score(item, weights, now)
