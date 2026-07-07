"""Tests for cockpit.priority — pure, property-tested priority scoring.

Every section 6.3 invariant (Fleet Cockpit PRD) is pinned here as a hypothesis
property over arbitrary ScoringItem fields plus one explicit example. `now`
and every ScoringItem field are injected — score() never reads a clock or an
RNG — so purity holds by construction and the tests below stay green across
every later implementation step.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

from hypothesis import given
from hypothesis import strategies as st

_STATES = ['open', 'answered', 'dropped']

_TIMESTAMPS = st.datetimes(timezones=st.just(UTC))
_SHORT_TEXT = st.text(max_size=20)
_BOOSTS = st.integers(min_value=-10_000, max_value=10_000)
_AGES_SECONDS = st.floats(min_value=0, max_value=100_000_000, allow_nan=False, allow_infinity=False)

_NOW = datetime(2026, 7, 7, tzinfo=UTC)


def _make_item(**overrides):
    from cockpit.priority import ScoringItem

    fields = {
        'severity': 'medium',
        'category': 'unmapped-category',
        'project': 'unmapped-project',
        'filed_at': datetime(2026, 7, 1, tzinfo=UTC),
        'manual_boost': 0,
        'state': 'open',
    }
    fields.update(overrides)
    return ScoringItem(**fields)


class TestScorePureFloat:
    def test_returns_float_for_open_item(self):
        from cockpit.priority import Priorities, ScoringItem, score

        item = ScoringItem(
            severity='high',
            category='bug',
            project='dark_factory',
            filed_at=datetime(2026, 7, 1, tzinfo=UTC),
            manual_boost=0,
            state='open',
        )
        now = datetime(2026, 7, 7, tzinfo=UTC)

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


class TestCategoryProjectWeights:
    def test_raising_project_weight_increases_score(self):
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        low = replace(weights, project_weights={'proj-a': 0.5})
        high = replace(weights, project_weights={'proj-a': 5.0})
        item = _make_item(project='proj-a')

        assert score(item, high, _NOW) > score(item, low, _NOW)

    def test_raising_category_weight_increases_score(self):
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        low = replace(weights, category_weights={'cat-a': 0.5})
        high = replace(weights, category_weights={'cat-a': 5.0})
        item = _make_item(category='cat-a')

        assert score(item, high, _NOW) > score(item, low, _NOW)

    def test_severity_fallback_matches_explicit_default(self):
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        absent = _make_item(severity='unmapped-severity')
        explicit_weights = replace(
            weights, severity_weights={'mapped-severity': weights.defaults.severity}
        )
        explicit = _make_item(severity='mapped-severity')

        assert score(absent, weights, _NOW) == score(explicit, explicit_weights, _NOW)

    def test_category_fallback_matches_explicit_default(self):
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        absent = _make_item(category='unmapped-category')
        explicit_weights = replace(
            weights, category_weights={'mapped-category': weights.defaults.category}
        )
        explicit = _make_item(category='mapped-category')

        assert score(absent, weights, _NOW) == score(explicit, explicit_weights, _NOW)

    def test_project_fallback_matches_explicit_default(self):
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        absent = _make_item(project='unmapped-project')
        explicit_weights = replace(
            weights, project_weights={'mapped-project': weights.defaults.project}
        )
        explicit = _make_item(project='mapped-project')

        assert score(absent, weights, _NOW) == score(explicit, explicit_weights, _NOW)


class TestMonotonicBoost:
    @given(
        severity=_SHORT_TEXT,
        category=_SHORT_TEXT,
        project=_SHORT_TEXT,
        filed_at=_TIMESTAMPS,
        state=st.sampled_from(_STATES),
        boosts=st.tuples(_BOOSTS, _BOOSTS).map(sorted),
    )
    def test_score_nondecreasing_in_boost(
        self, severity, category, project, filed_at, state, boosts
    ):
        """Section 6.3 invariant 1: score is monotonic non-decreasing in manual_boost."""
        from cockpit.priority import Priorities, score

        b1, b2 = boosts
        weights = Priorities.default()
        common = dict(
            severity=severity, category=category, project=project, filed_at=filed_at, state=state
        )
        item_low = _make_item(manual_boost=b1, **common)
        item_high = _make_item(manual_boost=b2, **common)

        assert score(item_high, weights, _NOW) >= score(item_low, weights, _NOW)

    def test_explicit_examples_span_clamp_regions(self):
        """Below-min and above-max both flatten (clamp); in-range is strictly increasing.

        RED: manual_boost is not yet summed into raw, so score is constant
        regardless of boost — the strict in-range inequalities below fail.

        Uses severity='critical' so raw stays positive even at boost=-5 —
        otherwise the unrelated `raw = max(0.0, raw)` misconfiguration guard
        would itself flatten the low end and confound this test.
        """
        from cockpit.priority import Priorities, score

        weights = Priorities.default()

        def s(boost):
            return score(_make_item(severity='critical', manual_boost=boost), weights, _NOW)

        # Below the clamp floor (min=-5): both clamp to the same value.
        assert s(-100) == s(-50)
        # Above the clamp ceiling (max=5): both clamp to the same value.
        assert s(50) == s(100)
        # In-range (unclamped) increments are strictly increasing.
        assert s(-5) < s(-2) < s(0) < s(3) < s(5)
        # Monotonic non-decreasing across the full span, clamp regions included.
        assert s(-100) <= s(-50) <= s(-5) <= s(-2) <= s(0) <= s(3) <= s(5) <= s(50) <= s(100)


class TestMonotonicAge:
    @given(ages=st.tuples(_AGES_SECONDS, _AGES_SECONDS).map(sorted))
    def test_score_nondecreasing_in_age(self, ages):
        """Section 6.3 invariant 2: score is monotonic non-decreasing in age.

        The item is otherwise fixed (open, default fields) — only age varies.
        """
        from cockpit.priority import Priorities, score

        a1, a2 = ages
        weights = Priorities.default()
        younger = _make_item(state='open', filed_at=_NOW - timedelta(seconds=a1))
        older = _make_item(state='open', filed_at=_NOW - timedelta(seconds=a2))

        assert score(older, weights, _NOW) >= score(younger, weights, _NOW)

    def test_age_before_saturation_strictly_increases_score(self):
        """RED: age is not yet summed into raw, so this strict pin fails today."""
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        younger = _make_item(state='open', filed_at=_NOW - timedelta(days=1))
        older = _make_item(state='open', filed_at=_NOW - timedelta(days=3))

        assert score(older, weights, _NOW) > score(younger, weights, _NOW)

    def test_ages_past_saturation_give_equal_contribution(self):
        """Bounded curve: age contribution flatlines once age >= saturation_seconds."""
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        past_saturation = _make_item(state='open', filed_at=_NOW - timedelta(days=10))
        further_past_saturation = _make_item(state='open', filed_at=_NOW - timedelta(days=30))

        assert score(past_saturation, weights, _NOW) == score(
            further_past_saturation, weights, _NOW
        )

    def test_clock_skew_treated_as_zero_age_no_exception(self):
        """filed_at after now (clock skew) must not raise and must not go negative."""
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        skewed = _make_item(state='open', filed_at=_NOW + timedelta(days=5))
        zero_age = _make_item(state='open', filed_at=_NOW)

        assert score(skewed, weights, _NOW) == score(zero_age, weights, _NOW)
