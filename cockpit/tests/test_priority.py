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


_ITEM_FIELDS = st.fixed_dictionaries(
    {
        'severity': _SHORT_TEXT,
        'category': _SHORT_TEXT,
        'project': _SHORT_TEXT,
        'age_seconds': _AGES_SECONDS,
        'manual_boost': _BOOSTS,
    }
)


def _item_from_fields(fields, state):
    return _make_item(
        severity=fields['severity'],
        category=fields['category'],
        project=fields['project'],
        filed_at=_NOW - timedelta(seconds=fields['age_seconds']),
        manual_boost=fields['manual_boost'],
        state=state,
    )


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


class TestTimezoneHandling:
    def test_naive_filed_at_assumed_utc(self):
        """A naive filed_at must be treated as UTC, not raise TypeError against an aware now."""
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        naive_item = _make_item(state='open', filed_at=datetime(2026, 7, 1))
        aware_item = _make_item(state='open', filed_at=datetime(2026, 7, 1, tzinfo=UTC))

        assert score(naive_item, weights, _NOW) == score(aware_item, weights, _NOW)

    def test_naive_now_assumed_utc(self):
        """A naive `now` must be treated as UTC, not raise TypeError against an aware filed_at."""
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        item = _make_item(state='open', filed_at=datetime(2026, 7, 1, tzinfo=UTC))
        naive_now = datetime(2026, 7, 7)
        aware_now = datetime(2026, 7, 7, tzinfo=UTC)

        assert score(item, weights, naive_now) == score(item, weights, aware_now)


class TestAgeCurveSaturationGuard:
    def test_non_positive_saturation_seconds_does_not_raise(self):
        """A misconfigured saturation_seconds<=0 must not raise ZeroDivisionError."""
        from dataclasses import replace

        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        zero_saturation = replace(
            weights, age_curve=replace(weights.age_curve, saturation_seconds=0.0)
        )
        negative_saturation = replace(
            weights, age_curve=replace(weights.age_curve, saturation_seconds=-1.0)
        )
        item = _make_item(state='open', filed_at=_NOW - timedelta(days=1))

        assert isinstance(score(item, zero_saturation, _NOW), float)
        assert isinstance(score(item, negative_saturation, _NOW), float)

    def test_non_positive_saturation_seconds_treated_as_instant_saturation(self):
        """Any positive age should get the full max_bonus; zero age gets none — still monotonic."""
        from dataclasses import replace

        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        zero_saturation = replace(
            weights, age_curve=replace(weights.age_curve, saturation_seconds=0.0)
        )
        brand_new = _make_item(state='open', filed_at=_NOW)
        one_second_old = _make_item(state='open', filed_at=_NOW - timedelta(seconds=1))
        ancient = _make_item(state='open', filed_at=_NOW - timedelta(days=3650))

        # Zero age gets no bonus; any positive age instantly saturates to the same score.
        assert score(brand_new, zero_saturation, _NOW) < score(
            one_second_old, zero_saturation, _NOW
        )
        assert score(one_second_old, zero_saturation, _NOW) == score(ancient, zero_saturation, _NOW)


class TestDroppedBelowOpen:
    @given(dropped_fields=_ITEM_FIELDS, open_fields=_ITEM_FIELDS)
    def test_dropped_scores_below_open(self, dropped_fields, open_fields):
        """Section 6.3 invariant 3: any dropped item scores below any open item.

        Two fully independent, arbitrary items — only the state differs by
        construction. RED: state tiers are not yet applied, so this is only
        true by chance today.
        """
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        dropped = _item_from_fields(dropped_fields, 'dropped')
        open_item = _item_from_fields(open_fields, 'open')

        assert score(dropped, weights, _NOW) < score(open_item, weights, _NOW)

    @given(answered_fields=_ITEM_FIELDS, open_fields=_ITEM_FIELDS)
    def test_answered_scores_below_open(self, answered_fields, open_fields):
        """An answered item also scores below any open item, independent of weights."""
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        answered = _item_from_fields(answered_fields, 'answered')
        open_item = _item_from_fields(open_fields, 'open')

        assert score(answered, weights, _NOW) < score(open_item, weights, _NOW)

    def test_adversarial_huge_dropped_vs_near_zero_open(self):
        """Explicit pin: a maximally-weighted dropped item still loses to a near-zero open one."""
        from cockpit.priority import Priorities, score

        weights = Priorities.default()
        dropped = _make_item(
            severity='critical',
            category='security',
            project='huge-project',
            filed_at=_NOW - timedelta(days=3650),
            manual_boost=10_000,
            state='dropped',
        )
        near_zero_open = _make_item(
            severity='unmapped',
            category='unmapped',
            project='unmapped',
            filed_at=_NOW,
            manual_boost=0,
            state='open',
        )

        assert score(dropped, weights, _NOW) < score(near_zero_open, weights, _NOW)
