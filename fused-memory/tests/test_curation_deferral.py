"""The deferral vocabulary and policy for a submission the curator could not judge."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from shared.cli_invoke import AllAccountsCappedException
from shared.usage_gate import PoolFrozen

from fused_memory.config.schema import CuratorConfig
from fused_memory.middleware.curation_deferral import (
    DEDUPE_SKIPPED_KEY,
    POOL_UNAVAILABLE_ERRORS,
    DeferralPolicy,
    DeferReason,
    dedupe_skipped_marker,
    pool_defer_reason,
)

SUBMITTED = datetime(2026, 10, 18, 12, 0, 0, tzinfo=UTC)


def _policy(base: float = 60.0, max_backoff: float = 1800.0, bound: float = 1800.0) -> DeferralPolicy:
    return DeferralPolicy(
        backoff_base_seconds=base, backoff_max_seconds=max_backoff, max_defer_seconds=bound,
    )


class TestDeferralPolicy:
    def test_first_retry_waits_the_base(self):
        now = SUBMITTED + timedelta(seconds=5)

        assert _policy().next_attempt_at(attempts=1, submitted_at=SUBMITTED, now=now) == (
            now + timedelta(seconds=60)
        )

    def test_retries_double_up_to_the_max(self):
        policy = _policy(base=60.0, max_backoff=300.0, bound=100_000.0)
        now = SUBMITTED

        delays = [
            (policy.next_attempt_at(attempts=n, submitted_at=SUBMITTED, now=now) - now).total_seconds()
            for n in range(1, 8)
        ]

        assert delays == [60.0, 120.0, 240.0, 300.0, 300.0, 300.0, 300.0]

    def test_a_huge_attempt_count_waits_the_max(self):
        policy = _policy(base=60.0, max_backoff=300.0, bound=100_000.0)

        at = policy.next_attempt_at(attempts=5000, submitted_at=SUBMITTED, now=SUBMITTED)

        assert at == SUBMITTED + timedelta(seconds=300)

    def test_a_retry_never_lands_after_the_deadline(self):
        policy = _policy(base=60.0, max_backoff=1800.0, bound=1800.0)
        now = SUBMITTED + timedelta(seconds=1700)

        at = policy.next_attempt_at(attempts=3, submitted_at=SUBMITTED, now=now)

        assert at == policy.deadline(SUBMITTED) == SUBMITTED + timedelta(seconds=1800)

    def test_a_retry_after_the_deadline_is_due_at_the_deadline(self):
        policy = _policy()
        now = SUBMITTED + timedelta(seconds=5000)

        assert policy.next_attempt_at(attempts=2, submitted_at=SUBMITTED, now=now) == (
            policy.deadline(SUBMITTED)
        )

    def test_attempts_counts_the_deferral_being_recorded(self):
        with pytest.raises(ValueError, match='attempts'):
            _policy().next_attempt_at(attempts=0, submitted_at=SUBMITTED, now=SUBMITTED)

    def test_expired_at_and_after_the_deadline(self):
        policy = _policy(bound=1800.0)

        assert not policy.expired(SUBMITTED, SUBMITTED + timedelta(seconds=1799.999))
        assert policy.expired(SUBMITTED, SUBMITTED + timedelta(seconds=1800))
        assert policy.expired(SUBMITTED, SUBMITTED + timedelta(seconds=1801))

    def test_naive_datetimes_are_rejected(self):
        policy = _policy()
        naive = datetime(2026, 10, 18, 12, 0, 0)

        with pytest.raises(ValueError, match='timezone'):
            policy.deadline(naive)
        with pytest.raises(ValueError, match='timezone'):
            policy.expired(SUBMITTED, naive)
        with pytest.raises(ValueError, match='timezone'):
            policy.expired(naive, SUBMITTED)
        with pytest.raises(ValueError, match='timezone'):
            policy.next_attempt_at(attempts=1, submitted_at=naive, now=SUBMITTED)
        with pytest.raises(ValueError, match='timezone'):
            policy.next_attempt_at(attempts=1, submitted_at=SUBMITTED, now=naive)

    @pytest.mark.parametrize(
        ('base', 'max_backoff', 'bound', 'named'),
        [
            (0.0, 1800.0, 1800.0, ('0.0',)),
            (60.0, -1.0, 1800.0, ('-1.0',)),
            (60.0, 1800.0, 0.0, ('0.0',)),
            (float('nan'), 1800.0, 1800.0, ('nan',)),
            (120.0, 90.0, 1800.0, ('120.0', '90.0')),
        ],
        ids=['zero-base', 'negative-max', 'zero-bound', 'nan-base', 'max-below-base'],
    )
    def test_non_positive_or_inverted_bounds_are_rejected(self, base, max_backoff, bound, named):
        with pytest.raises(ValueError) as excinfo:
            _policy(base=base, max_backoff=max_backoff, bound=bound)

        for value in named:
            assert value in str(excinfo.value)

    def test_from_config_reads_the_curator_knobs(self):
        curator = CuratorConfig(
            deferral_backoff_base_seconds=7.0,
            deferral_backoff_max_seconds=70.0,
            deferral_max_seconds=700.0,
        )

        assert DeferralPolicy.from_config(curator) == _policy(base=7.0, max_backoff=70.0, bound=700.0)

    def test_give_up_justification_names_bound_and_reason(self):
        assert _policy(bound=1800.0).give_up_justification(DeferReason.ALL_ACCOUNTS_CAPPED) == (
            'curation-gave-up after 1800s: all-accounts-capped'
        )
        assert _policy(bound=90.5).give_up_justification(DeferReason.ACCOUNTS_AUTH_FAILED) == (
            'curation-gave-up after 90.5s: accounts-auth-failed'
        )


def _frozen(*, capped=(), auth_failed=(), scope=None) -> PoolFrozen:
    return PoolFrozen(
        account_count=max(len(capped) + len(auth_failed), 1),
        capped_account_names=tuple(capped),
        auth_failed_account_names=tuple(auth_failed),
        scope=scope,
    )


class TestPoolDeferReason:
    def test_pool_unavailable_errors_are_the_real_classes(self):
        assert set(POOL_UNAVAILABLE_ERRORS) == {AllAccountsCappedException, PoolFrozen}

    def test_all_accounts_capped_exception_is_capped(self):
        exc = AllAccountsCappedException(retries=4, elapsed_secs=121.0, label='curator')

        assert pool_defer_reason(exc) is DeferReason.ALL_ACCOUNTS_CAPPED

    def test_frozen_pool_with_a_capped_account_is_capped(self):
        exc = _frozen(capped=('max-a',), auth_failed=('max-b',))

        assert pool_defer_reason(exc) is DeferReason.ALL_ACCOUNTS_CAPPED

    def test_frozen_pool_of_only_auth_failed_accounts(self):
        exc = _frozen(auth_failed=('max-a', 'max-b'))

        assert pool_defer_reason(exc) is DeferReason.ACCOUNTS_AUTH_FAILED

    def test_scope_exhausted_freeze_is_capped(self):
        exc = _frozen(auth_failed=('max-a',), scope='sonnet')

        assert pool_defer_reason(exc) is DeferReason.ALL_ACCOUNTS_CAPPED

    def test_reason_values_are_the_wire_strings(self):
        assert [reason.value for reason in DeferReason] == [
            'all-accounts-capped', 'accounts-auth-failed',
        ]


class TestDedupeSkippedMarker:
    def test_marker_shape_and_utc_at(self):
        at = datetime(2026, 10, 18, 14, 30, 0, tzinfo=timezone(timedelta(hours=2)))

        marker = dedupe_skipped_marker('curation-gave-up after 1800s: all-accounts-capped', at=at)

        assert marker == {
            DEDUPE_SKIPPED_KEY: {
                'reason': 'curation-gave-up after 1800s: all-accounts-capped',
                'at': '2026-10-18T12:30:00+00:00',
            },
        }
        assert DEDUPE_SKIPPED_KEY == 'dedupe_skipped'

    def test_reason_truncated_to_500(self):
        marker = dedupe_skipped_marker('x' * 501, at=SUBMITTED)

        assert marker[DEDUPE_SKIPPED_KEY]['reason'] == 'x' * 500

    def test_naive_at_rejected(self):
        with pytest.raises(ValueError, match='timezone'):
            dedupe_skipped_marker('r', at=datetime(2026, 10, 18, 12, 0, 0))
