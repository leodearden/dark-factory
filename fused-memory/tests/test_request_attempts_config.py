"""The embedder's bounded request attempts, and the rule that they fit the write budget (task 10241)."""

from __future__ import annotations

import subprocess
import sys

import pytest
from fused_memory.config.request_attempts import (
    SDK_INITIAL_RETRY_DELAY_SECONDS,
    SDK_MAX_RETRY_DELAY_SECONDS,
    RequestAttemptsConfig,
)
from pydantic import ValidationError

from fused_memory.config.schema import EmbedderConfig, FusedMemoryConfig, QueueConfig


def _config(attempts: RequestAttemptsConfig, *, write_budget_s: float) -> FusedMemoryConfig:
    return FusedMemoryConfig(
        embedder=EmbedderConfig(request_attempts=attempts),
        queue=QueueConfig(backend_write_timeout_seconds=write_budget_s),
    )


def _attempts(per_attempt_s: float, max_attempts: int) -> RequestAttemptsConfig:
    return RequestAttemptsConfig(
        per_attempt_timeout_seconds=per_attempt_s, max_attempts=max_attempts,
    )


class TestTheStalledScheduleCeiling:
    def test_the_defaults_are_three_attempts_of_eight_seconds(self):
        attempts = RequestAttemptsConfig()

        assert attempts.per_attempt_timeout_seconds == 8.0
        assert attempts.max_attempts == 3
        assert attempts.stalled_schedule_seconds == 3 * 8.0 + 0.5 + 1.0 == 25.5

    def test_a_single_attempt_has_no_backoff(self):
        assert _attempts(4.0, 1).stalled_schedule_seconds == 4.0

    def test_each_retry_adds_the_sdk_doubling_backoff(self):
        assert _attempts(2.0, 5).stalled_schedule_seconds == 5 * 2.0 + 0.5 + 1.0 + 2.0 + 4.0

    def test_the_backoff_stops_growing_at_the_sdk_cap(self):
        six = _attempts(2.0, 6).stalled_schedule_seconds
        seven = _attempts(2.0, 7).stalled_schedule_seconds
        eight = _attempts(2.0, 8).stalled_schedule_seconds

        assert six == 6 * 2.0 + 0.5 + 1.0 + 2.0 + 4.0 + 8.0
        assert seven - six == 2.0 + 8.0
        assert eight - seven == 2.0 + 8.0


class TestFieldBounds:
    @pytest.mark.parametrize('per_attempt_s', [0.0, -1.0])
    def test_a_non_positive_per_attempt_deadline_is_rejected(self, per_attempt_s):
        with pytest.raises(ValidationError):
            RequestAttemptsConfig(per_attempt_timeout_seconds=per_attempt_s)

    @pytest.mark.parametrize('max_attempts', [0, -1])
    def test_fewer_than_one_attempt_is_rejected(self, max_attempts):
        with pytest.raises(ValidationError):
            RequestAttemptsConfig(max_attempts=max_attempts)


class TestTheScheduleMustFitTheWriteBudget:
    def test_a_schedule_past_the_write_budget_is_rejected_and_named(self):
        with pytest.raises(ValidationError) as raised:
            _config(_attempts(40.0, 3), write_budget_s=120.0)

        message = str(raised.value)
        assert 'embedder.request_attempts' in message
        assert 'per_attempt_timeout_seconds=40.0' in message
        assert 'max_attempts=3' in message
        assert '121.5s' in message
        assert 'queue.backend_write_timeout_seconds (120.0s)' in message

    def test_a_schedule_just_inside_the_write_budget_is_accepted(self):
        config = _config(_attempts(39.0, 3), write_budget_s=120.0)

        assert config.embedder.request_attempts.stalled_schedule_seconds == 118.5

    def test_a_schedule_equal_to_the_write_budget_is_rejected(self):
        with pytest.raises(ValidationError, match='embedder.request_attempts'):
            _config(_attempts(39.5, 3), write_budget_s=120.0)

    def test_shrinking_only_the_write_budget_is_rejected(self):
        with pytest.raises(ValidationError, match=r'queue\.backend_write_timeout_seconds \(20\.0s\)'):
            _config(RequestAttemptsConfig(), write_budget_s=20.0)

    def test_the_backoff_term_decides(self):
        attempts = _attempts(22.6, 5)
        assert attempts.max_attempts * attempts.per_attempt_timeout_seconds < 120.0

        with pytest.raises(ValidationError, match='120.5'):
            _config(attempts, write_budget_s=120.0)

    def test_a_reload_revalidates_the_schedule(self):
        config = _config(RequestAttemptsConfig(), write_budget_s=120.0)
        config.embedder.request_attempts.per_attempt_timeout_seconds = 40.0

        with pytest.raises(ValidationError, match='embedder.request_attempts'):
            FusedMemoryConfig.model_validate(config.model_dump())


def test_the_shipped_schedule_fits_both_backend_budgets(monkeypatch):
    """A stalled search embed is retried too, not only a stalled write's."""
    monkeypatch.delenv('EMBEDDER_ATTEMPT_TIMEOUT', raising=False)
    monkeypatch.delenv('EMBEDDER_MAX_ATTEMPTS', raising=False)

    config = FusedMemoryConfig()

    ceiling = config.embedder.request_attempts.stalled_schedule_seconds
    assert ceiling < config.queue.backend_write_timeout_seconds
    assert ceiling < config.queue.backend_read_timeout_seconds


def test_the_backoff_mirrors_match_the_openai_sdk():
    from openai import _constants

    assert SDK_INITIAL_RETRY_DELAY_SECONDS == _constants.INITIAL_RETRY_DELAY
    assert SDK_MAX_RETRY_DELAY_SECONDS == _constants.MAX_RETRY_DELAY


def test_loading_the_config_schema_imports_neither_openai_nor_httpx():
    """A fresh interpreter: this process has already imported both."""
    probe = (
        'import sys; import fused_memory.config.schema; '
        "assert 'openai' not in sys.modules, sorted(k for k in sys.modules if 'openai' in k); "
        "assert 'httpx' not in sys.modules, sorted(k for k in sys.modules if 'httpx' in k)"
    )

    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
