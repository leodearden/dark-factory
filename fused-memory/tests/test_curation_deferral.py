"""The deferral vocabulary and policy for a submission the curator could not judge."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from shared.cli_invoke import AllAccountsCappedException
from shared.usage_gate import PoolFrozen

from fused_memory.config.schema import CuratorConfig
from fused_memory.middleware.curation_deferral import (
    DEDUPE_SKIPPED_KEY,
    POOL_UNAVAILABLE_ERRORS,
    CurationRetryPolicy,
    DeferReason,
    backlog_severity,
    dedupe_skipped_marker,
    pool_defer_reason,
)

NOW = datetime(2026, 10, 18, 12, 0, 0, tzinfo=UTC)


def _policy(base: float = 60.0, max_backoff: float = 1800.0) -> CurationRetryPolicy:
    return CurationRetryPolicy(backoff_base_seconds=base, backoff_max_seconds=max_backoff)


class TestCurationRetryPolicy:
    def test_first_retry_waits_the_base(self):
        assert _policy().next_attempt_at(attempts=1, now=NOW) == NOW + timedelta(seconds=60)

    def test_retries_double_up_to_the_max(self):
        policy = _policy(base=60.0, max_backoff=300.0)

        delays = [
            (policy.next_attempt_at(attempts=n, now=NOW) - NOW).total_seconds()
            for n in range(1, 8)
        ]

        assert delays == [60.0, 120.0, 240.0, 300.0, 300.0, 300.0, 300.0]

    def test_a_huge_attempt_count_waits_the_max(self):
        policy = _policy(base=60.0, max_backoff=300.0)

        assert policy.next_attempt_at(attempts=5000, now=NOW) == NOW + timedelta(seconds=300)

    def test_retries_continue_at_the_max_however_long_the_deferral(self):
        policy = _policy(base=60.0, max_backoff=1800.0)
        days_later = NOW + timedelta(days=3)

        assert policy.next_attempt_at(attempts=200, now=days_later) == (
            days_later + timedelta(seconds=1800)
        )

    def test_attempts_counts_the_deferral_being_recorded(self):
        with pytest.raises(ValueError, match='attempts'):
            _policy().next_attempt_at(attempts=0, now=NOW)

    def test_naive_datetimes_are_rejected(self):
        with pytest.raises(ValueError, match='timezone'):
            _policy().next_attempt_at(attempts=1, now=datetime(2026, 10, 18, 12, 0, 0))

    @pytest.mark.parametrize(
        ('base', 'max_backoff', 'named'),
        [
            (0.0, 1800.0, ('0.0',)),
            (60.0, -1.0, ('-1.0',)),
            (float('nan'), 1800.0, ('nan',)),
            (120.0, 90.0, ('120.0', '90.0')),
        ],
        ids=['zero-base', 'negative-max', 'nan-base', 'max-below-base'],
    )
    def test_non_positive_or_inverted_bounds_are_rejected(self, base, max_backoff, named):
        with pytest.raises(ValueError) as excinfo:
            _policy(base=base, max_backoff=max_backoff)

        for value in named:
            assert value in str(excinfo.value)

    def test_from_config_reads_the_curator_knobs(self):
        curator = CuratorConfig(deferral_backoff_base_seconds=7.0, deferral_backoff_max_seconds=70.0)

        assert CurationRetryPolicy.from_config(curator) == _policy(base=7.0, max_backoff=70.0)


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


class TestBacklogSeverity:
    def test_a_capped_backlog_is_info_because_it_heals_at_reset(self):
        assert backlog_severity(DeferReason.ALL_ACCOUNTS_CAPPED) == 'info'

    def test_an_auth_failed_backlog_is_blocking_because_it_never_self_heals(self):
        assert backlog_severity(DeferReason.ACCOUNTS_AUTH_FAILED) == 'blocking'

    @pytest.mark.parametrize('reason', list(DeferReason))
    def test_every_reason_has_a_severity(self, reason):
        assert backlog_severity(reason) in ('info', 'blocking')


class TestDedupeSkippedMarker:
    def test_marker_shape_and_utc_at(self):
        at = datetime(2026, 10, 18, 14, 30, 0, tzinfo=timezone(timedelta(hours=2)))

        marker = dedupe_skipped_marker('create: llm-failed: TimeoutError', at=at)

        assert marker == {
            DEDUPE_SKIPPED_KEY: {
                'reason': 'create: llm-failed: TimeoutError',
                'at': '2026-10-18T12:30:00+00:00',
            },
        }
        assert DEDUPE_SKIPPED_KEY == 'dedupe_skipped'

    def test_reason_truncated_to_500(self):
        marker = dedupe_skipped_marker('x' * 501, at=NOW)

        assert marker[DEDUPE_SKIPPED_KEY]['reason'] == 'x' * 500

    def test_naive_at_rejected(self):
        with pytest.raises(ValueError, match='timezone'):
            dedupe_skipped_marker('r', at=datetime(2026, 10, 18, 12, 0, 0))
