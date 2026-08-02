"""Tests for config schema Literal type validation."""

import os
import re
import sys
import unittest.mock
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings

from fused_memory.config.schema import (
    CuratorConfig,
    EmbedderConfig,
    FusedMemoryConfig,
    GraphitiBackendConfig,
    LLMConfig,
    PathScopeAdjudicatorConfig,
    ProceduralTopicCluster,
    QueueConfig,
    ReconciliationConfig,
    ServerConfig,
    SummaryRebuildConfig,
    TaskMetadataConfig,
    TaskStatusConfig,
    YamlSettingsSource,
)
from fused_memory.server.near_duplicate_guard import find_matching_topic_cluster
from fused_memory.services.durable_queue import DEFAULT_TRANSIENT_ERROR_NAMES

# Imported from the LEAF, not from memory_metadata: this test module asserts
# the config side's behaviour, and pulling the registry here would drag in
# mem0 and mask the very import-lightness the extraction bought (task 3198).
from fused_memory.topic_slug import (
    TOPIC_SLUG_MAX_LEN,
    TOPIC_SLUG_RE,
    is_valid_topic_slug,
)


class _DummySettings(BaseSettings):
    pass


def _make_source(path):
    return YamlSettingsSource(_DummySettings, config_path=path)


class TestServerConfigTransport:
    """Tests for ServerConfig.transport Literal validation."""

    def test_default_transport_is_http(self):
        config = ServerConfig()
        assert config.transport == 'http'

    def test_valid_transport_http(self):
        config = ServerConfig(transport='http')
        assert config.transport == 'http'

    def test_valid_transport_stdio(self):
        config = ServerConfig(transport='stdio')
        assert config.transport == 'stdio'

    def test_valid_transport_sse(self):
        config = ServerConfig(transport='sse')
        assert config.transport == 'sse'

    def test_invalid_transport_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ServerConfig(transport='websocket')  # type: ignore[arg-type]

    def test_invalid_transport_grpc_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ServerConfig(transport='grpc')  # type: ignore[arg-type]


class TestServerConfigThreadWarnThreshold:
    """Tests for ServerConfig.thread_warn_threshold field."""

    def test_default_threshold_is_60(self):
        config = ServerConfig()
        assert config.thread_warn_threshold == 60

    def test_explicit_override_accepted(self):
        config = ServerConfig(thread_warn_threshold=120)
        assert config.thread_warn_threshold == 120

    def test_non_int_raises_validation_error(self):
        with pytest.raises(ValidationError):
            ServerConfig(thread_warn_threshold='not-an-int')  # type: ignore[arg-type]


class TestServerConfigGracefulShutdownTimeout:
    """Tests for ServerConfig.graceful_shutdown_timeout field (survey finding D1).

    Bounds uvicorn's internal timeout_graceful_shutdown wait so
    _shutdown_with_watchdog's force-exit timer is always reached within the
    systemd TimeoutStopSec budget.  See main._SYSTEMD_TIMEOUT_STOP_SECS.
    """

    def test_default_is_10(self):
        config = ServerConfig()
        assert config.graceful_shutdown_timeout == 10

    def test_explicit_override_accepted(self):
        config = ServerConfig(graceful_shutdown_timeout=8)
        assert config.graceful_shutdown_timeout == 8

    def test_value_is_a_positive_int(self):
        config = ServerConfig()
        assert isinstance(config.graceful_shutdown_timeout, int)
        assert config.graceful_shutdown_timeout > 0


class TestServerConfigGracefulShutdownTimeoutBudgetValidation:
    """Fail-fast guard for the D1 shutdown-budget invariant on operator overrides.

    The default (10) satisfies graceful_shutdown_timeout + _FORCE_EXIT_BUDGET (75)
    < _SYSTEMD_TIMEOUT_STOP_SECS (90) — see TestShutdownBudgetArithmetic in
    test_server_shutdown.py. That test only pins the *default*; ServerConfig is
    operator-settable, so an override that breaks the invariant must fail loudly
    at config load instead of silently at systemd SIGKILL time.
    """

    def test_rejects_non_positive_timeout(self):
        with pytest.raises(ValidationError):
            ServerConfig(graceful_shutdown_timeout=0)

    def test_rejects_negative_timeout(self):
        with pytest.raises(ValidationError):
            ServerConfig(graceful_shutdown_timeout=-1)

    def test_rejects_timeout_that_breaks_force_exit_budget_invariant(self):
        # 30 + 75 (_FORCE_EXIT_BUDGET) = 105 >= 90 (_SYSTEMD_TIMEOUT_STOP_SECS).
        with pytest.raises(ValidationError):
            ServerConfig(graceful_shutdown_timeout=30)

    def test_rejects_dead_heat_timeout(self):
        # 15 + 75 = 90 == _SYSTEMD_TIMEOUT_STOP_SECS: zero margin, not "< 90".
        with pytest.raises(ValidationError):
            ServerConfig(graceful_shutdown_timeout=15)

    def test_accepts_timeout_just_inside_the_budget(self):
        # 14 + 75 = 89 < 90: the largest value that still satisfies the invariant.
        config = ServerConfig(graceful_shutdown_timeout=14)
        assert config.graceful_shutdown_timeout == 14


class TestLLMConfigProvider:
    """Tests for LLMConfig.provider Literal validation."""

    def test_default_provider_is_openai(self):
        config = LLMConfig()
        assert config.provider == 'openai'

    def test_valid_provider_openai(self):
        config = LLMConfig(provider='openai')
        assert config.provider == 'openai'

    def test_valid_provider_anthropic(self):
        config = LLMConfig(provider='anthropic')
        assert config.provider == 'anthropic'

    def test_invalid_provider_raises_validation_error(self):
        with pytest.raises(ValidationError):
            LLMConfig(provider='gemini')  # type: ignore[arg-type]

    def test_invalid_provider_cohere_raises_validation_error(self):
        with pytest.raises(ValidationError):
            LLMConfig(provider='cohere')  # type: ignore[arg-type]


class TestEmbedderConfigProvider:
    """Tests for EmbedderConfig.provider Literal validation."""

    def test_default_provider_is_openai(self):
        config = EmbedderConfig()
        assert config.provider == 'openai'

    def test_valid_provider_openai(self):
        config = EmbedderConfig(provider='openai')
        assert config.provider == 'openai'

    def test_invalid_provider_raises_validation_error(self):
        with pytest.raises(ValidationError):
            EmbedderConfig(provider='cohere')  # type: ignore[arg-type]

    def test_invalid_provider_huggingface_raises_validation_error(self):
        with pytest.raises(ValidationError):
            EmbedderConfig(provider='huggingface')  # type: ignore[arg-type]


class TestGraphitiBackendConfigProvider:
    """Tests for GraphitiBackendConfig.provider Literal validation."""

    def test_default_provider_is_falkordb(self):
        config = GraphitiBackendConfig()
        assert config.provider == 'falkordb'

    def test_valid_provider_falkordb(self):
        config = GraphitiBackendConfig(provider='falkordb')
        assert config.provider == 'falkordb'

    def test_invalid_provider_raises_validation_error(self):
        with pytest.raises(ValidationError):
            GraphitiBackendConfig(provider='neo4j')  # type: ignore[arg-type]

    def test_invalid_provider_redis_raises_validation_error(self):
        with pytest.raises(ValidationError):
            GraphitiBackendConfig(provider='redis')  # type: ignore[arg-type]


class TestFusedMemoryConfigDefaults:
    """Tests for FusedMemoryConfig top-level defaults."""

    def test_all_defaults_load_successfully(self, tmp_path, monkeypatch):
        # Point CONFIG_PATH at a non-existent file so YamlSettingsSource returns {}
        monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'missing.yaml'))
        config = FusedMemoryConfig()
        assert config.server.transport == 'http'
        assert config.llm.provider == 'openai'
        assert config.embedder.provider == 'openai'
        assert config.graphiti.provider == 'falkordb'

    def test_valid_config_constructed_explicitly(self):
        config = FusedMemoryConfig(
            server=ServerConfig(transport='stdio'),
            llm=LLMConfig(provider='anthropic'),
            embedder=EmbedderConfig(provider='openai'),
            graphiti=GraphitiBackendConfig(provider='falkordb'),
        )
        assert config.server.transport == 'stdio'
        assert config.llm.provider == 'anthropic'

    def test_yaml_file_values_loaded(self, tmp_path, monkeypatch):
        # Write a YAML file with non-default values to exercise the full
        # YamlSettingsSource.__call__ + YAML-parsing branch (lines 61-63 of schema.py)
        config_data = {
            'server': {'port': 9999, 'transport': 'sse'},
            'llm': {'provider': 'anthropic'},
        }
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.setenv('CONFIG_PATH', str(config_file))
        config = FusedMemoryConfig()
        assert config.server.port == 9999
        assert config.server.transport == 'sse'
        assert config.llm.provider == 'anthropic'
        # Verify unmentioned config sections retain their defaults (not clobbered to null/empty)
        assert config.embedder.provider == 'openai'
        assert config.embedder.model == 'text-embedding-3-small'
        assert config.server.host == '0.0.0.0'
        assert config.graphiti.provider == 'falkordb'
        assert config.mem0.qdrant_url == 'http://localhost:6333'
        assert config.routing.confidence_threshold == 0.7

    def test_env_var_expansion_e2e(self, tmp_path, monkeypatch):
        """End-to-end: env var placeholder in YAML is expanded through full settings machinery."""
        config_file = tmp_path / 'config.yaml'
        config_file.write_text("server:\n  port: '${MY_TEST_PORT}'\n")
        monkeypatch.setenv('CONFIG_PATH', str(config_file))
        monkeypatch.setenv('MY_TEST_PORT', '4242')
        config = FusedMemoryConfig()
        assert config.server.port == 4242

    def test_env_overrides_yaml_priority(self, tmp_path, monkeypatch):
        """Env vars (via pydantic-settings __ delimiter) take priority over YAML values."""
        config_data = {'server': {'port': 9999}}
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(yaml.dump(config_data))
        monkeypatch.setenv('CONFIG_PATH', str(config_file))
        monkeypatch.setenv('SERVER__PORT', '7777')
        config = FusedMemoryConfig()
        assert config.server.port == 7777


class TestYamlSettingsSourceEnvVarExpansion:
    """Tests for YamlSettingsSource._expand_env_vars."""

    def setup_method(self):
        self.source = YamlSettingsSource(_DummySettings, config_path=None)

    def test_expands_env_var_with_value(self, monkeypatch):
        monkeypatch.setenv('MY_API_KEY', 'secret-key')
        result = self.source._expand_env_vars('${MY_API_KEY}')
        assert result == 'secret-key'

    def test_expands_env_var_with_default_when_missing(self):
        # Ensure the var is not set
        os.environ.pop('MISSING_VAR_XYZ', None)
        result = self.source._expand_env_vars('${MISSING_VAR_XYZ:default_val}')
        assert result == 'default_val'

    def test_expands_env_var_to_none_when_empty_default(self):
        os.environ.pop('MISSING_VAR_XYZ', None)
        result = self.source._expand_env_vars('${MISSING_VAR_XYZ}')
        assert result is None

    def test_expands_env_var_true_to_bool(self, monkeypatch):
        monkeypatch.setenv('FLAG_VAR', 'true')
        result = self.source._expand_env_vars('${FLAG_VAR}')
        assert result is True

    def test_expands_env_var_false_to_bool(self, monkeypatch):
        monkeypatch.setenv('FLAG_VAR', 'false')
        result = self.source._expand_env_vars('${FLAG_VAR}')
        assert result is False

    def test_expands_dict_values_recursively(self, monkeypatch):
        monkeypatch.setenv('HOST_VAR', 'localhost')
        data = {'host': '${HOST_VAR}', 'port': 8080}
        result = self.source._expand_env_vars(data)
        assert result['host'] == 'localhost'
        assert result['port'] == 8080

    def test_expands_list_values(self, monkeypatch):
        monkeypatch.setenv('ITEM_VAR', 'hello')
        result = self.source._expand_env_vars(['${ITEM_VAR}', 'static'])
        assert result[0] == 'hello'
        assert result[1] == 'static'

    def test_non_env_string_unchanged(self):
        result = self.source._expand_env_vars('plain-string')
        assert result == 'plain-string'


class TestYamlSettingsSourceErrorHandling:
    """Tests for YamlSettingsSource error handling on corrupt or unreadable files."""

    def test_corrupt_yaml_raises_runtime_error(self, tmp_path):
        """Corrupt YAML content must raise RuntimeError with the file path in the message."""
        bad_file = tmp_path / 'bad.yaml'
        bad_file.write_bytes(b': :\n  - \x00bad')
        source = _make_source(bad_file)
        with pytest.raises(RuntimeError, match=re.escape(str(bad_file))) as exc_info:
            source()
        assert exc_info.value.__cause__ is not None

    @pytest.mark.skipif(
        sys.platform == 'win32' or getattr(os, 'getuid', lambda: -1)() == 0,
        reason='chmod not reliable on Windows or when running as root',
    )
    def test_unreadable_file_raises_runtime_error(self, tmp_path):
        """An unreadable file must raise RuntimeError with the file path in the message."""
        locked_file = tmp_path / 'locked.yaml'
        locked_file.write_text('key: value')
        locked_file.chmod(0o000)
        try:
            source = _make_source(locked_file)
            with pytest.raises(RuntimeError, match=re.escape(str(locked_file))) as exc_info:
                source()
            assert exc_info.value.__cause__ is not None
        finally:
            locked_file.chmod(0o644)

    def test_expand_env_vars_error_raises_runtime_error(self, tmp_path, monkeypatch):
        """_expand_env_vars raising any exception must be wrapped in RuntimeError with config path."""
        config_file = tmp_path / 'valid.yaml'
        config_file.write_text('key: value')
        source = _make_source(config_file)

        def _raise(val):
            raise ValueError('boom')

        monkeypatch.setattr(source, '_expand_env_vars', _raise)
        with pytest.raises(RuntimeError, match=re.escape(str(config_file))):
            source()

    def test_expand_env_vars_error_includes_original_cause(self, tmp_path, monkeypatch):
        """The RuntimeError raised for _expand_env_vars failure must chain the original exception."""
        config_file = tmp_path / 'valid.yaml'
        config_file.write_text('key: value')
        source = _make_source(config_file)
        original = ValueError('original cause')

        def _raise(val):
            raise original

        monkeypatch.setattr(source, '_expand_env_vars', _raise)
        with pytest.raises(RuntimeError) as exc_info:
            source()
        assert exc_info.value.__cause__ is original

    def test_expand_env_vars_error_does_not_mask_yaml_error(self, tmp_path):
        """Corrupt YAML must still raise RuntimeError with 'Failed to load configuration' message."""
        bad_file = tmp_path / 'bad.yaml'
        bad_file.write_bytes(b': :\n  - \x00bad')
        source = _make_source(bad_file)
        with pytest.raises(RuntimeError, match='Failed to load configuration') as exc_info:
            source()
        assert 'Failed to expand' not in str(exc_info.value)


class TestYamlSettingsSourceEncoding:
    """Tests for YamlSettingsSource explicit UTF-8 encoding."""

    def test_utf8_yaml_loaded_correctly(self, tmp_path):
        """YAML files with non-ASCII UTF-8 characters must load correctly."""
        config_file = tmp_path / 'utf8.yaml'
        config_file.write_text("description: 'Ünfcödé tëst'", encoding='utf-8')
        source = _make_source(config_file)
        result = source()
        assert result.get('description') == 'Ünfcödé tëst'

    def test_utf8_open_passes_encoding_kwarg(self, tmp_path):
        """open() must be called with encoding='utf-8' when loading the YAML file."""
        config_file = tmp_path / 'utf8.yaml'
        config_file.write_text('key: value', encoding='utf-8')
        source = _make_source(config_file)
        _real_open = open
        with unittest.mock.patch('builtins.open', side_effect=_real_open) as mock_open:
            source()
        mock_open.assert_called_once()
        assert mock_open.call_args.kwargs.get('encoding') == 'utf-8'


class TestYamlSettingsSourceABCContract:
    """Tests for YamlSettingsSource ABC contract compliance."""

    def setup_method(self):
        self.source = YamlSettingsSource(_DummySettings, config_path=None)

    def test_get_field_value_returns_tuple(self):
        """get_field_value must return tuple[Any, str, bool] per PydanticBaseSettingsSource ABC."""
        field = FieldInfo(annotation=str)
        result = self.source.get_field_value(field, 'my_field')
        assert isinstance(result, tuple), f'Expected tuple, got {type(result)}'
        assert len(result) == 3, f'Expected 3-tuple, got {len(result)}-tuple'
        assert result[0] is None
        assert result[1] == 'my_field'
        assert result[2] is False


class TestConfigYamlReconciliationFlags:
    """Tests for deployment-config values in fused-memory/config/config.yaml."""

    def test_config_yaml_enables_require_done_provenance(self, monkeypatch):
        """config.yaml must enable require_done_provenance for Phase 2 enforcement.

        Step-21: loads the real deployment YAML via CONFIG_PATH env and asserts
        the gate is on. Phase 2 (6a272fd46e) shipped the validator with the
        schema default of False; this project's YAML is the source of truth for
        flipping enforcement on — see design decision 5 on task 844.
        """
        # Walk from this test file up to the repo root, then to the yaml.
        # fused-memory/tests/test_config_schema.py → ../../fused-memory/config/config.yaml
        yaml_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
        assert yaml_path.is_file(), f'expected config.yaml at {yaml_path}'
        monkeypatch.setenv('CONFIG_PATH', str(yaml_path))
        cfg = FusedMemoryConfig()
        assert cfg.reconciliation.require_done_provenance is True, (
            'fused-memory/config/config.yaml must set '
            'reconciliation.require_done_provenance: true to enable Phase 2 '
            'enforcement of the done_provenance gate.'
        )


class TestReconciliationConfigTimeouts:
    """Tests for the three dedicated CLI-timeout / claim-recovery fields on ReconciliationConfig."""

    def test_default_agent_cli_timeout_is_180(self):
        assert ReconciliationConfig().agent_cli_timeout_seconds == 180

    def test_default_judge_cli_timeout_is_600(self):
        assert ReconciliationConfig().judge_cli_timeout_seconds == 600

    def test_explicit_agent_cli_timeout_override_accepted(self):
        assert ReconciliationConfig(agent_cli_timeout_seconds=30).agent_cli_timeout_seconds == 30

    def test_explicit_judge_cli_timeout_override_accepted(self):
        assert ReconciliationConfig(judge_cli_timeout_seconds=120).judge_cli_timeout_seconds == 120

    # --- gt=0 bounds: agent_cli_timeout_seconds ---

    def test_agent_cli_timeout_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(agent_cli_timeout_seconds=0)

    def test_agent_cli_timeout_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(agent_cli_timeout_seconds=-1)

    # --- gt=0 bounds: judge_cli_timeout_seconds ---

    def test_judge_cli_timeout_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(judge_cli_timeout_seconds=0)

    def test_judge_cli_timeout_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(judge_cli_timeout_seconds=-1)

    def test_stale_claim_recovery_seconds_field_removed(self):
        """stale_claim_recovery_seconds must not exist on ReconciliationConfig.

        Task 905 made release_stale_claims(0) unconditional on startup, so the
        field no longer influences any production behaviour.  Task 909 removed it
        to prevent operators from tuning a knob that has no effect.
        """
        assert 'stale_claim_recovery_seconds' not in ReconciliationConfig.model_fields

    # --- gt=0 bounds: stage_timeout_seconds (consistency extension) ---

    def test_stage_timeout_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(stage_timeout_seconds=0)

    def test_stage_timeout_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(stage_timeout_seconds=-1)

    # --- gt=0 bounds: stale_run_recovery_seconds (consistency extension) ---

    def test_stale_run_recovery_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(stale_run_recovery_seconds=0)

    def test_stale_run_recovery_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(stale_run_recovery_seconds=-1)

    # --- cross-field inner <= outer validator ---

    def test_agent_cli_timeout_exceeds_stage_rejected(self):
        with pytest.raises(ValidationError, match='agent_cli_timeout_seconds'):
            ReconciliationConfig(agent_cli_timeout_seconds=5000, stage_timeout_seconds=3000)

    def test_judge_cli_timeout_exceeds_stage_rejected(self):
        with pytest.raises(ValidationError, match='judge_cli_timeout_seconds'):
            ReconciliationConfig(judge_cli_timeout_seconds=5000, stage_timeout_seconds=3000)

    def test_agent_cli_timeout_equal_to_stage_accepted(self):
        # inner == outer is the degenerate-but-valid co-terminal case
        cfg = ReconciliationConfig(agent_cli_timeout_seconds=3600, stage_timeout_seconds=3600)
        assert cfg.agent_cli_timeout_seconds == 3600

    def test_judge_cli_timeout_equal_to_stage_accepted(self):
        # inner == outer is the degenerate-but-valid co-terminal case (judge parallel)
        cfg = ReconciliationConfig(judge_cli_timeout_seconds=3600, stage_timeout_seconds=3600)
        assert cfg.judge_cli_timeout_seconds == 3600

    def test_defaults_pass_validator(self):
        # Shipped defaults: agent=180, judge=600, stage=3600 — all satisfy inner<=outer
        cfg = ReconciliationConfig()
        assert cfg.agent_cli_timeout_seconds <= cfg.stage_timeout_seconds
        assert cfg.judge_cli_timeout_seconds <= cfg.stage_timeout_seconds

    # --- gt=0 bounds: tool_timeout_seconds (consistency extension) ---

    def test_tool_timeout_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(tool_timeout_seconds=0)

    def test_tool_timeout_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(tool_timeout_seconds=-1.0)

    # --- gt=0 bounds: cycle_timeout_seconds (consistency extension) ---

    def test_cycle_timeout_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(cycle_timeout_seconds=0)

    def test_cycle_timeout_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(cycle_timeout_seconds=-1)

    # --- cross-field stage <= cycle validator ---

    def test_stage_timeout_exceeds_cycle_rejected(self):
        with pytest.raises(ValidationError, match='stage_timeout_seconds'):
            ReconciliationConfig(stage_timeout_seconds=86400, cycle_timeout_seconds=3600)

    def test_stage_timeout_equal_to_cycle_accepted(self):
        # stage == cycle is the degenerate-but-valid co-terminal case
        cfg = ReconciliationConfig(stage_timeout_seconds=3600, cycle_timeout_seconds=3600)
        assert cfg.stage_timeout_seconds == 3600


class TestReconciliationConfigJudgeInfraMaxConsecutiveFailures:
    """Tests for judge_infra_max_consecutive_failures (task 2947 ask a).

    Bounds consecutive judge transport/infra failures before a
    judge-unreachable halt + infra_issue escalation fires.
    """

    def test_default_is_3(self):
        assert ReconciliationConfig().judge_infra_max_consecutive_failures == 3

    def test_explicit_override_accepted(self):
        cfg = ReconciliationConfig(judge_infra_max_consecutive_failures=5)
        assert cfg.judge_infra_max_consecutive_failures == 5

    def test_value_of_one_accepted(self):
        # ge=1 lower bound: 1 is the smallest legal value (halt on first infra failure)
        cfg = ReconciliationConfig(judge_infra_max_consecutive_failures=1)
        assert cfg.judge_infra_max_consecutive_failures == 1

    def test_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(judge_infra_max_consecutive_failures=0)

    def test_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(judge_infra_max_consecutive_failures=-1)


class TestReconciliationAutoUnhaltAfterCooldown:
    """Tests for auto_unhalt_after_cooldown (task 2920 deliverable c).

    When True, a judge-halted project is auto-unhalted (with the normal
    post-unhalt grace) once its halt cooldown expires, so a transient/suspect
    halt self-heals instead of sitting forever. Default False preserves the
    legacy halt-until-manual-unhalt semantics for every existing test and
    other deployments.
    """

    def test_default_is_false(self):
        assert ReconciliationConfig().auto_unhalt_after_cooldown is False

    def test_explicit_true_accepted(self):
        assert (
            ReconciliationConfig(auto_unhalt_after_cooldown=True).auto_unhalt_after_cooldown
            is True
        )


class TestReconciliationConfigBacklogIterationBudget:
    """Tests for backlog_iteration_budget_seconds — the cumulative per-invocation
    wall-clock budget for BacklogIterator.run() (task 2040)."""

    def test_default_budget_is_1800(self):
        assert ReconciliationConfig().backlog_iteration_budget_seconds == 1800

    def test_budget_exceeding_cycle_timeout_rejected(self):
        # stage_timeout_seconds=1000 keeps the pre-existing agent/judge-cli <=
        # stage <= cycle checks satisfied (defaults 180/600) so only the new
        # backlog-budget <= cycle check is exercised.
        with pytest.raises(ValidationError, match='backlog_iteration_budget_seconds'):
            ReconciliationConfig(
                backlog_iteration_budget_seconds=1001,
                cycle_timeout_seconds=1000,
                stage_timeout_seconds=1000,
            )

    def test_budget_within_cycle_timeout_accepted(self):
        cfg = ReconciliationConfig(
            backlog_iteration_budget_seconds=500,
            cycle_timeout_seconds=1000,
            stage_timeout_seconds=1000,
        )
        assert cfg.backlog_iteration_budget_seconds == 500


class TestReconciliationConfigDeadLetterFields:
    """Tests for event_dead_letter_max_bytes and event_dead_letter_keep_rotations."""

    def test_default_max_bytes(self):
        cfg = ReconciliationConfig()
        assert cfg.event_dead_letter_max_bytes == 10 * 1024 * 1024

    def test_default_keep_rotations(self):
        cfg = ReconciliationConfig()
        assert cfg.event_dead_letter_keep_rotations == 3

    def test_max_bytes_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(event_dead_letter_max_bytes=0)

    def test_max_bytes_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(event_dead_letter_max_bytes=-1)

    def test_keep_rotations_zero_accepted(self):
        # zero means "don't keep any rotations" — valid boundary value
        cfg = ReconciliationConfig(event_dead_letter_keep_rotations=0)
        assert cfg.event_dead_letter_keep_rotations == 0

    def test_keep_rotations_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(event_dead_letter_keep_rotations=-1)


class TestReconciliationConfigBulkResetGuardFields:
    """Tests for bulk_reset_guard_write_failure_backoff_seconds (task 1032)."""

    def test_default_write_failure_backoff_seconds(self):
        cfg = ReconciliationConfig()
        assert cfg.bulk_reset_guard_write_failure_backoff_seconds == 60.0

    def test_write_failure_backoff_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(bulk_reset_guard_write_failure_backoff_seconds=-0.001)

    def test_write_failure_backoff_zero_accepted(self):
        # zero disables the backoff — valid boundary value
        cfg = ReconciliationConfig(bulk_reset_guard_write_failure_backoff_seconds=0.0)
        assert cfg.bulk_reset_guard_write_failure_backoff_seconds == 0.0

    def test_write_failure_backoff_override_accepted(self):
        cfg = ReconciliationConfig(bulk_reset_guard_write_failure_backoff_seconds=120.5)
        assert cfg.bulk_reset_guard_write_failure_backoff_seconds == 120.5


class TestReconciliationConfigStormKnobs:
    """Tests for the dead_owner_shielded suppression-storm knobs (task 1755 / PRD β).

    Two new fields on ReconciliationConfig:
      - dead_owner_suppression_storm_threshold (int, ge=1, default 6)
      - dead_owner_suppression_storm_window_seconds (float, gt=0, default 3600.0)
    """

    # --- defaults ---

    def test_default_storm_threshold_is_6(self):
        assert ReconciliationConfig().dead_owner_suppression_storm_threshold == 6

    def test_default_storm_window_is_3600(self):
        assert ReconciliationConfig().dead_owner_suppression_storm_window_seconds == 3600.0

    # --- override accepted ---

    def test_storm_threshold_override_accepted(self):
        cfg = ReconciliationConfig(dead_owner_suppression_storm_threshold=3)
        assert cfg.dead_owner_suppression_storm_threshold == 3

    def test_storm_window_override_accepted(self):
        cfg = ReconciliationConfig(dead_owner_suppression_storm_window_seconds=1800.0)
        assert cfg.dead_owner_suppression_storm_window_seconds == 1800.0

    # --- ge=1 bound: dead_owner_suppression_storm_threshold ---

    def test_storm_threshold_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(dead_owner_suppression_storm_threshold=0)

    def test_storm_threshold_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(dead_owner_suppression_storm_threshold=-1)

    # --- gt=0 bound: dead_owner_suppression_storm_window_seconds ---

    def test_storm_window_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(dead_owner_suppression_storm_window_seconds=0)

    def test_storm_window_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(dead_owner_suppression_storm_window_seconds=-1.0)


class TestReconciliationConfigResumeKnobs:
    """Tests for the interrupted-run resume knobs (task 2717 / PRD σ, rec 13).

    Five new fields on ReconciliationConfig gate the startup adopt-and-resume
    pass:
      - resume_after_restart (bool, default True)
      - resume_freshness_window_seconds (int, gt=0, default 3600)
      - resume_max_attempts_per_run (int, gt=0, default 2)
      - resume_failure_storm_threshold (int, gt=0, default 6)
      - resume_failure_storm_window_seconds (float, gt=0, default 3600.0)
    """

    # --- defaults ---

    def test_default_resume_after_restart_is_true(self):
        assert ReconciliationConfig().resume_after_restart is True

    def test_default_resume_freshness_window_is_3600(self):
        assert ReconciliationConfig().resume_freshness_window_seconds == 3600

    def test_default_resume_max_attempts_is_2(self):
        assert ReconciliationConfig().resume_max_attempts_per_run == 2

    def test_default_resume_failure_storm_threshold_is_positive(self):
        cfg = ReconciliationConfig()
        assert cfg.resume_failure_storm_threshold == 6
        assert cfg.resume_failure_storm_threshold > 0

    def test_default_resume_failure_storm_window_is_positive(self):
        cfg = ReconciliationConfig()
        assert cfg.resume_failure_storm_window_seconds == 3600.0
        assert cfg.resume_failure_storm_window_seconds > 0

    # --- override accepted ---

    def test_resume_after_restart_override_accepted(self):
        cfg = ReconciliationConfig(resume_after_restart=False)
        assert cfg.resume_after_restart is False

    def test_resume_freshness_window_override_accepted(self):
        cfg = ReconciliationConfig(resume_freshness_window_seconds=7200)
        assert cfg.resume_freshness_window_seconds == 7200

    def test_resume_max_attempts_override_accepted(self):
        cfg = ReconciliationConfig(resume_max_attempts_per_run=1)
        assert cfg.resume_max_attempts_per_run == 1

    # --- gt=0 bound: resume_freshness_window_seconds ---

    def test_resume_freshness_window_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(resume_freshness_window_seconds=0)

    def test_resume_freshness_window_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(resume_freshness_window_seconds=-1)

    # --- gt=0 bound: resume_max_attempts_per_run ---

    def test_resume_max_attempts_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(resume_max_attempts_per_run=0)

    def test_resume_max_attempts_negative_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(resume_max_attempts_per_run=-1)

    # --- gt=0 bounds: storm knobs ---

    def test_resume_failure_storm_threshold_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(resume_failure_storm_threshold=0)

    def test_resume_failure_storm_window_zero_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(resume_failure_storm_window_seconds=0)


class TestConfigYamlBacklogHardLimitOverrides:
    """Deployment test: config.yaml must carry the reify override (task 1764).

    Mirrors TestConfigYamlReconciliationFlags: loads the real YAML via CONFIG_PATH,
    constructs FusedMemoryConfig(), and asserts the deployed values.
    """

    def test_config_yaml_reify_override_in_recommended_band(self, monkeypatch):
        """config.yaml must set reconciliation.backlog_hard_limit_overrides.reify in [1000, 1500].

        reify's steady-state backlog (~500-520) sits just over the flat 500,
        causing benign recon_backlog_overflow escalations (esc-fused-memory-237).
        The recommended band is 1000-1500. Asserting the full band pins the intent
        while leaving operators room to re-tune within the range — and catches an
        edit that silently disables the guard (e.g. setting reify to 50000).
        """
        yaml_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
        assert yaml_path.is_file(), f'expected config.yaml at {yaml_path}'
        monkeypatch.setenv('CONFIG_PATH', str(yaml_path))
        cfg = FusedMemoryConfig()
        reify_limit = cfg.reconciliation.backlog_hard_limit_overrides.get('reify', 0)
        assert 1000 <= reify_limit <= 1500, (
            f'fused-memory/config/config.yaml must set '
            f'reconciliation.backlog_hard_limit_overrides.reify in [1000, 1500] '
            f'(got {reify_limit!r}) — the recommended band that clears benign '
            f'esc-fused-memory-237 noise without masking a real runaway.'
        )

    def test_config_yaml_global_default_unchanged(self, monkeypatch):
        """config.yaml must not override backlog_hard_limit from its default of 500."""
        yaml_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
        assert yaml_path.is_file(), f'expected config.yaml at {yaml_path}'
        monkeypatch.setenv('CONFIG_PATH', str(yaml_path))
        cfg = FusedMemoryConfig()
        assert cfg.reconciliation.backlog_hard_limit == 500, (
            'reconciliation.backlog_hard_limit in config.yaml must remain 500 '
            '(the per-project override is separate).'
        )


class TestReconciliationConfigBacklogHardLimitOverrides:
    """Tests for ReconciliationConfig.backlog_hard_limit_overrides (task 1764).

    New field: backlog_hard_limit_overrides: dict[str, int] = Field(default_factory=dict)
    Per-project override map keyed by project_id; empty map = flat hard_limit for all.
    """

    # --- defaults ---

    def test_default_overrides_is_empty_dict(self):
        assert ReconciliationConfig().backlog_hard_limit_overrides == {}

    def test_default_backlog_hard_limit_unchanged(self):
        """Global default 500 must remain unchanged for small projects."""
        assert ReconciliationConfig().backlog_hard_limit == 500

    # --- override accepted / round-trips ---

    def test_override_map_accepted(self):
        cfg = ReconciliationConfig(backlog_hard_limit_overrides={'reify': 1500})
        assert cfg.backlog_hard_limit_overrides == {'reify': 1500}

    def test_override_map_multi_project_round_trips(self):
        overrides = {'reify': 1500, 'autopilot_video': 800}
        cfg = ReconciliationConfig(backlog_hard_limit_overrides=overrides)
        assert cfg.backlog_hard_limit_overrides == overrides

    # --- dict[str, int] guard: non-int value rejected ---

    def test_non_int_value_rejected(self):
        with pytest.raises(ValidationError):
            ReconciliationConfig(backlog_hard_limit_overrides={'reify': 'not-an-int'})  # type: ignore[arg-type]

    # --- positivity guard: non-positive override values rejected ---

    def test_zero_override_rejected(self):
        """Zero override would make every check exceed limit — reject at config load."""
        with pytest.raises(ValidationError):
            ReconciliationConfig(backlog_hard_limit_overrides={'reify': 0})

    def test_negative_override_rejected(self):
        """Negative override must also be caught by the positivity guard."""
        with pytest.raises(ValidationError):
            ReconciliationConfig(backlog_hard_limit_overrides={'reify': -1})

    def test_mixed_valid_and_zero_rejected(self):
        """A map with one valid and one zero entry must be rejected (not partially accepted)."""
        with pytest.raises(ValidationError):
            ReconciliationConfig(backlog_hard_limit_overrides={'reify': 1500, 'bad': 0})


class TestPathScopeAdjudicatorConfigBudget:
    """Regression guard: max_budget_usd must clear the adjudicator's own call cost (task 1849).

    The adjudicator's cost floor was measured at ~0.105 USD on 2026-06-19 against
    Sonnet 4.6 — dominated by FIXED overhead: Sonnet + 3-turn json-schema flow +
    CLI base-context cache-creation (~13k tokens) + the filing project's
    CLAUDE.md/memory auto-loaded from cwd.
    The original 0.10 default was BELOW this cost, so every call returned
    error_max_budget_usd — a silent no-op that prevented any verdict=='allow'.
    That floor has since eroded as stack-wide cost-per-call rose (filing-project
    CLAUDE.md growth + opus/sonnet alias-roll tokenizer inflation), tripping the
    0.30 default the same way the sibling CuratorConfig tripped (task 1980) — so
    task 1983 re-baselines the default to a flat $2.00, matching CuratorConfig's
    durable ceiling. The 0.25 floor remains the minimum safe value below which the
    adjudicator becomes a silent no-op.
    """

    def test_default_max_budget_usd_above_cost_floor(self):
        """PathScopeAdjudicatorConfig default must clear the 0.25 cost floor.

        The adjudicator's cost floor was ~0.105 USD (fixed CLI/base-context overhead +
        3-turn json-schema flow + filing project CLAUDE.md/memory auto-loaded from cwd),
        since eroded by stack-wide cost-per-call growth (task 1983);
        a budget at or below cost means error_max_budget_usd before any verdict —
        a silent no-op.  0.25 is the minimum safe floor.
        """
        # 0.10 < 0.25 is False — RED before the fix; 2.00 >= 0.25 is True — GREEN after
        assert PathScopeAdjudicatorConfig().max_budget_usd >= 0.25, (
            'PathScopeAdjudicatorConfig.max_budget_usd must be >= 0.25 '
            '(the adjudicator cost floor was ~0.105 USD and has since risen; anything '
            'at or below cost returns error_max_budget_usd before any verdict, making '
            'it a silent no-op). Raise the default — see task 1849 / task 1983.'
        )
        assert PathScopeAdjudicatorConfig().max_budget_usd == pytest.approx(2.00), (
            'PathScopeAdjudicatorConfig.max_budget_usd must be re-baselined to a flat '
            '$2.00 (task 1983), matching CuratorConfig\'s durable ceiling (task 1980).'
        )

    def test_deployed_config_max_budget_usd_above_floor(self, monkeypatch):
        """Effective deployed value (schema default layered with config.yaml) must also clear the floor.

        Mirrors TestConfigYamlBacklogHardLimitOverrides: loads the real config.yaml via
        CONFIG_PATH, constructs FusedMemoryConfig(), and asserts the effective value.
        This catches a future YAML edit that re-pins max_budget_usd below cost — the
        exact 'silent no-op via config layering' failure class that task 1849 fixes.
        """
        yaml_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
        assert yaml_path.is_file(), f'expected config.yaml at {yaml_path}'
        monkeypatch.setenv('CONFIG_PATH', str(yaml_path))
        # config.yaml currently has no path_scope_adjudicator section, so today
        # this test exercises the default-fallback path (schema default = 2.00).
        # It will catch a future YAML edit that explicitly pins the budget below cost.
        cfg = FusedMemoryConfig()
        assert cfg.path_scope_adjudicator.max_budget_usd >= 0.25, (
            f'Effective path_scope_adjudicator.max_budget_usd (got '
            f'{cfg.path_scope_adjudicator.max_budget_usd!r}) must be >= 0.25 — '
            f'the adjudicator cost floor was ~0.105 USD and has since risen; a budget '
            f'at or below cost returns error_max_budget_usd before any verdict '
            f'(silent no-op). Check config.yaml for an override that pins the budget '
            f'below cost.'
        )


class TestCuratorConfigBudgetRaise:
    """Regression guard: TaskCurator per-call budget is a durable flat $2.00 (task 1980).

    Directive: reify L2 esc-task-curator-194 (Leo: raise budget to $2). The prior
    scale-by-batch-size attempt (reify task 2254) was CANCELLED and the cap
    recurred, so the fix here is a DURABLE flat raise rather than more per-size
    scaling. ``_scale_budget`` computes ``min(base + per_entry*size, cap)`` — for
    the ceiling to be a flat $2.00 regardless of pool/batch size, BOTH the base
    (``max_budget_usd``) and the single-call cap (``single_call_budget_cap_usd``)
    must be raised together; raising only one leaves the other clamping the
    result below $2.00.
    """

    def test_default_max_budget_usd_is_two_dollars(self):
        """CuratorConfig.max_budget_usd base floor must be raised to $2.00.

        0.30 == 2.00 is False — RED before the fix; 2.00 == 2.00 is True — GREEN after.
        """
        assert CuratorConfig().max_budget_usd == pytest.approx(2.00), (
            'CuratorConfig.max_budget_usd must be a flat $2.00 (task 1980 / '
            'esc-task-curator-194) so large-candidate curations stop tripping '
            'error_max_budget_usd. Raise the default.'
        )

    def test_single_call_cap_cannot_clamp_base_below_two_dollars(self):
        """single_call_budget_cap_usd must be >= max_budget_usd.

        The single-call effective budget is
        ``min(max_budget_usd + per_pool_entry_budget_usd*len(pool), single_call_budget_cap_usd)``.
        If the cap is left below the raised base, every single-call invocation
        clamps back down below $2.00 — silently undoing the raise. Both the base
        AND the cap must move together; that is the whole point of this task.
        """
        cfg = CuratorConfig()
        assert cfg.single_call_budget_cap_usd >= cfg.max_budget_usd, (
            f'CuratorConfig.single_call_budget_cap_usd (got '
            f'{cfg.single_call_budget_cap_usd!r}) must be >= max_budget_usd (got '
            f'{cfg.max_budget_usd!r}) — otherwise the single-call cap silently '
            f'clamps the raised base back down, defeating the durable $2.00 raise.'
        )

    def test_deployed_config_max_budget_usd_at_least_two_dollars(self, monkeypatch):
        """Effective deployed value (schema default layered with config.yaml) must also be >= $2.00.

        Mirrors TestPathScopeAdjudicatorConfigBudget.test_deployed_config_max_budget_usd_above_floor:
        loads the real config.yaml via CONFIG_PATH, constructs FusedMemoryConfig(),
        and asserts the effective value. This catches a future YAML edit that
        re-pins max_budget_usd below the durable $2.00 ceiling.
        """
        yaml_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
        assert yaml_path.is_file(), f'expected config.yaml at {yaml_path}'
        monkeypatch.setenv('CONFIG_PATH', str(yaml_path))
        # config.yaml currently has no curator.max_budget_usd override, so today
        # this test exercises the default-fallback path (schema default = 2.00).
        # It will catch a future YAML edit that explicitly pins the budget lower.
        cfg = FusedMemoryConfig()
        assert cfg.curator.max_budget_usd >= 2.00, (
            f'Effective curator.max_budget_usd (got {cfg.curator.max_budget_usd!r}) '
            f'must be >= 2.00 — task 1980 / esc-task-curator-194 raises the '
            f'TaskCurator per-call budget to a durable flat $2.00. Check '
            f'config.yaml for an override that pins the budget lower.'
        )


class TestQueueConfigTransientErrorFields:
    """Task 1936: QueueConfig exposes the error-aware retry budget knobs that
    flow into DurableWriteQueue(transient_max_attempts=..., transient_error_names=...).
    """

    def test_default_transient_max_attempts(self):
        cfg = QueueConfig()
        assert isinstance(cfg.transient_max_attempts, int)
        assert cfg.transient_max_attempts == 12
        # Must be >= the plain max_attempts default so transient errors never
        # get a SHORTER budget than non-transient ones.
        assert cfg.transient_max_attempts >= cfg.max_attempts

    def test_default_transient_error_names_contains_node_not_found(self):
        cfg = QueueConfig()
        assert isinstance(cfg.transient_error_names, list)
        assert 'NodeNotFoundError' in cfg.transient_error_names

    def test_explicit_overrides_round_trip(self):
        cfg = QueueConfig(transient_max_attempts=20, transient_error_names=['X'])
        assert cfg.transient_max_attempts == 20
        assert cfg.transient_error_names == ['X']

    def test_transient_error_names_matches_durable_queue_default(self):
        """The two default transient-error-name lists (QueueConfig's and
        DurableWriteQueue's DEFAULT_TRANSIENT_ERROR_NAMES) are documented as
        'kept in sync' but nothing enforced that — this test is the
        enforcement, so any future drift fails CI instead of silently
        denying one of the two lists' errors the extended retry budget.
        """
        assert set(QueueConfig().transient_error_names) == DEFAULT_TRANSIENT_ERROR_NAMES

    def test_transient_max_attempts_below_max_attempts_rejected(self):
        """A config that would give transient errors a SHORTER budget than
        ordinary errors is rejected at config-load time, not silently
        accepted (task 1936 review)."""
        with pytest.raises(ValidationError, match='transient_max_attempts'):
            QueueConfig(max_attempts=10, transient_max_attempts=5)


class TestSummaryRebuildConfigDefaults:
    """Task 1958: SummaryRebuildConfig is the scheduled staleness backstop
    (fix (b), follow-up to task 1949's best-effort post-ingestion refresh).

    Disabled-by-default (enabled=False, projects=[]) means the periodic
    sweep costs nothing until an operator opts in.
    """

    def test_defaults(self):
        cfg = SummaryRebuildConfig()
        assert cfg.enabled is False
        assert cfg.projects == []
        assert cfg.force is False
        assert cfg.interval_seconds == 3600.0

    def test_fused_memory_config_attaches_disabled_by_default(self):
        cfg = FusedMemoryConfig().summary_rebuild
        assert isinstance(cfg, SummaryRebuildConfig)
        assert cfg.enabled is False

    def test_interval_seconds_must_be_positive(self):
        with pytest.raises(ValidationError, match='interval_seconds'):
            SummaryRebuildConfig(interval_seconds=0)


class TestTaskMetadataConfig:
    """Task 2162 (W3-β): ``task_metadata.enforce`` — RED-TIER, restart-only.

    Warn-mode (default, ``enforce=False``): a write that violates the shared
    ``TaskMetadata`` schema (``shared.task_metadata.parse_metadata``) emits a
    ``task_metadata.schema_warning`` log line and the write proceeds. Enforce-
    mode (``True``): the same write is rejected with ``ValidationError``. Not
    hot-reloadable — see ``skills/orchestrate/SKILL.md``'s config-hot-reload
    section and PRD decision #6.
    """

    def test_defaults_to_warn_mode(self):
        assert TaskMetadataConfig().enforce is False

    def test_fused_memory_config_attaches_disabled_by_default(self, tmp_path, monkeypatch):
        # Point CONFIG_PATH at a non-existent file so YamlSettingsSource returns {}
        # and we exercise the SCHEMA default (disabled) — not the shipped config.yaml,
        # which sets task_metadata.enforce: true (the human-blessed W3-θ2 flip, task 2184).
        # This test's distinct value is proving the sub-config ATTACHES to the parent;
        # the schema default itself is covered by test_defaults_to_warn_mode above.
        monkeypatch.setenv('CONFIG_PATH', str(tmp_path / 'missing.yaml'))
        cfg = FusedMemoryConfig().task_metadata
        assert isinstance(cfg, TaskMetadataConfig)
        assert cfg.enforce is False

    def test_override_loads_enforce_true(self):
        """A config dict override for task_metadata is validated into the model."""
        cfg = FusedMemoryConfig(task_metadata={'enforce': True})  # type: ignore[arg-type]
        assert cfg.task_metadata.enforce is True


class TestTaskStatusConfig:
    """Task 2175 (rho1b): ``task_status.enforce_transitions`` — RED-TIER, restart-only.

    Log-mode (default, ``enforce_transitions=False``): an illegal
    ``(from, to, actor)`` status transition at the interceptor's
    ``_apply_status_transition`` chokepoint emits an ``illegal_transition
    would-reject`` warning and the write proceeds. Enforce-mode (``True``):
    the same write is rejected with a typed ``illegal_transition`` error.
    Flipped only after the Gamma soak proves the transition table clean; not
    hot-reloadable. See ``plans/task-status-authority-prd.md`` C3/D5/D6.
    """

    def test_defaults_to_log_mode(self):
        assert TaskStatusConfig().enforce_transitions is False

    def test_override_enables_enforce_mode(self):
        assert TaskStatusConfig(enforce_transitions=True).enforce_transitions is True

    def test_fused_memory_config_loads_enforce_flip_from_yaml(self, monkeypatch):
        # The committed config.yaml enables enforce-mode: task 2216 flipped
        # task_status.enforce_transitions -> true on 2026-07-14 (operator gate
        # esc-2216-1) after the Gamma soak proved the transition table clean.
        # FusedMemoryConfig() loads that committed config, so this pins the
        # deployed Γ-flip and tripwires an accidental revert. The pure MODEL
        # default (False) is covered by test_defaults_to_log_mode above.
        #
        # Pin CONFIG_PATH explicitly (mirrors TestConfigYamlReconciliationFlags
        # below): the ambient CONFIG_PATH-unset default resolves
        # 'config/config.yaml' relative to the process CWD, not this package,
        # so a caller running pytest from a different CWD (e.g. the repo root)
        # would silently miss the real file and fall through to the pure model
        # default instead of exercising the committed YAML.
        yaml_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
        monkeypatch.setenv('CONFIG_PATH', str(yaml_path))
        assert FusedMemoryConfig().task_status.enforce_transitions is True


class TestReconciliationRejectStaleDoneEvidence:
    """Task 2680 (PRD task gamma): ``reconciliation.reject_stale_done_evidence``
    shipped default flips ``'warn'`` -> ``'enforce'``.

    The reopen-freshness gate (task 2674, PRD task alpha) rejects a
    done-write citing PRE-reopen evidence on a task whose
    ``metadata.reopen_at`` is set (task-1175 clobber shape) only when this
    knob is ``'enforce'``; in ``'warn'`` mode the same write proceeds and
    only logs a ``task_status.done_evidence_stale_warn`` census line. Task
    alpha shipped ``'warn'`` gated on task beta (the orchestrator consumer,
    ``ProvenanceConflictSink`` + ``StaleEvidenceRejection``) landing first;
    beta is done (task 2677), so gamma flips the shipped default here. The
    committed ``fused-memory/config/config.yaml`` does not pin this key, so
    the schema default IS the effective shipped default -- both the bare
    model and ``FusedMemoryConfig()`` (which loads that committed config)
    must agree. See ``reject_stale_done_evidence``'s Field description in
    schema.py for the full mode contract.
    """

    def test_defaults_to_enforce_mode(self):
        assert ReconciliationConfig().reject_stale_done_evidence == 'enforce'

    def test_fused_memory_config_loads_enforce_default(self, monkeypatch):
        # The committed config.yaml does not pin reconciliation.
        # reject_stale_done_evidence, so FusedMemoryConfig() (which loads
        # that committed config) surfaces the schema default here -- this
        # is the interceptor's actual access path
        # (``cfg.reconciliation.reject_stale_done_evidence`` in
        # ``_reject_stale_done_evidence_mode``), not just the bare model
        # covered by test_defaults_to_enforce_mode above.
        #
        # Pin CONFIG_PATH explicitly (mirrors TestConfigYamlReconciliationFlags
        # below): the ambient CONFIG_PATH-unset default resolves
        # 'config/config.yaml' relative to the process CWD, not this package.
        # Without this, the test would pass vacuously from a different CWD
        # (schema default == fallback-on-missing-file default) without ever
        # actually loading the committed YAML it claims to exercise.
        yaml_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
        monkeypatch.setenv('CONFIG_PATH', str(yaml_path))
        assert FusedMemoryConfig().reconciliation.reject_stale_done_evidence == 'enforce'

    def test_override_opts_out_to_warn_mode(self):
        cfg = ReconciliationConfig(reject_stale_done_evidence='warn')
        assert cfg.reject_stale_done_evidence == 'warn'

    def test_invalid_value_rejected(self):
        with pytest.raises(ValidationError, match='reject_stale_done_evidence'):
            ReconciliationConfig(reject_stale_done_evidence='sometimes')  # type: ignore[arg-type]


class TestProceduralTopicClusterModel:
    """The topic-keyed cluster model that seeds the deterministic topic guard (task 2845)."""

    def test_validates_well_formed_cluster(self):
        cluster = ProceduralTopicCluster(
            topic_id='some-topic',
            phrases=['alpha', 'beta'],
            hint='route to gate task 9999',
        )
        assert cluster.topic_id == 'some-topic'
        assert cluster.phrases == ['alpha', 'beta']
        assert cluster.hint == 'route to gate task 9999'

    def test_min_phrase_hits_defaults_to_two(self):
        cluster = ProceduralTopicCluster(topic_id='t', phrases=['a', 'b', 'c'])
        assert cluster.min_phrase_hits == 2

    def test_hint_defaults_to_empty_string(self):
        cluster = ProceduralTopicCluster(topic_id='t', phrases=['a', 'b'])
        assert cluster.hint == ''

    def test_rejects_unknown_key(self):
        with pytest.raises(ValidationError):
            ProceduralTopicCluster(
                topic_id='t',
                phrases=['a', 'b'],
                unexpected_key='boom',  # type: ignore[call-arg]
            )

    def test_rejects_min_phrase_hits_below_one(self):
        with pytest.raises(ValidationError):
            ProceduralTopicCluster(topic_id='t', phrases=['a', 'b'], min_phrase_hits=0)


class TestProceduralTopicClusterTopicIdSlug:
    """``topic_id`` shares ONE namespace with ``metadata.topic`` (PRD D4, task 3198).

    Before leaf ε, ``topic_id`` was a bare ``str``: an operator could seed a
    snake_case cluster id that could never equal any validated
    ``metadata.topic``, so 3135's auto-seed invariant
    (``cluster.topic_id == canonical.metadata.topic``) was unenforceable and
    the guard would silently match nothing. The validator makes that a
    config-LOAD failure instead.
    """

    @pytest.mark.parametrize(
        ('bad_id', 'why'),
        [
            ('bad_slug', 'snake_case — the shape 98 of 352 live topics have'),
            ('Bad-Slug', 'uppercase'),
            ('bad topic-slug', 'embedded space'),
            ('-lead', 'leading separator'),
            ('', 'empty'),
            ('a' * (TOPIC_SLUG_MAX_LEN + 1), 'over the length cap'),
        ],
    )
    def test_rejects_non_slug_topic_id(self, bad_id, why):
        with pytest.raises(ValidationError) as excinfo:
            ProceduralTopicCluster(topic_id=bad_id, phrases=['a', 'b'])
        message = str(excinfo.value)
        assert repr(bad_id) in message, f'must quote the offending value ({why})'
        assert TOPIC_SLUG_RE.pattern in message, 'must name the slug rule'
        assert 'fused_memory.topic_slug' in message, 'must name the rule/home so it is findable'

    def test_over_length_id_is_rejected_by_length_not_by_the_regex(self):
        """Pins that the validator applies the CAP, not merely the pattern.

        The regex accepts an over-long run of ``a``s, so without this the
        over-length case above would pass for the wrong reason and deleting
        the length clause would go unnoticed.
        """
        over = 'a' * (TOPIC_SLUG_MAX_LEN + 1)
        assert TOPIC_SLUG_RE.match(over), 'the regex alone must NOT reject it'
        with pytest.raises(ValidationError):
            ProceduralTopicCluster(topic_id=over, phrases=['a', 'b'])

    @pytest.mark.parametrize('good_id', ['some-topic', 't', 'x1-2y'])
    def test_accepts_conforming_topic_id(self, good_id):
        """Positive control: the shapes the pre-existing tests already use.

        ``'some-topic'`` and ``'t'`` are the exact ids
        ``TestProceduralTopicClusterModel`` constructs, so this proves the
        new validator adds a rejection without regressing any existing case.
        """
        assert ProceduralTopicCluster(topic_id=good_id, phrases=['a', 'b']).topic_id == good_id

    def test_default_seeded_clusters_all_survive_the_validator(self):
        """The shipped default config must still LOAD (PRD §10's hard requirement).

        Constructed via ``ReconciliationConfig()`` rather than by copying
        the ids, so a future seed that breaks the rule fails here rather
        than at an operator's config load.
        """
        clusters = ReconciliationConfig().procedural_knowledge_topic_guard_clusters
        assert len(clusters) >= 5
        for cluster in clusters:
            assert is_valid_topic_slug(cluster.topic_id)

    def test_snake_case_cluster_id_in_yaml_fails_at_config_load(self, tmp_path, monkeypatch):
        """Operator-facing: the failure lands at LOAD, not silently at match time.

        This is the whole point of validating on the config side. A
        snake_case ``topic_id`` can never equal a validated
        ``metadata.topic``, so without this the guard would load cleanly and
        then match nothing forever — a silent no-op, the failure mode
        ``extra='forbid'`` is already on this model to prevent.
        """
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(
            yaml.dump(
                {
                    'reconciliation': {
                        'procedural_knowledge_topic_guard_clusters': [
                            {'topic_id': 'eval_worktree_plan_tools_missing', 'phrases': ['a', 'b']},
                        ],
                    },
                },
            ),
        )
        monkeypatch.setenv('CONFIG_PATH', str(config_file))
        with pytest.raises(ValidationError) as excinfo:
            FusedMemoryConfig()
        assert 'eval_worktree_plan_tools_missing' in str(excinfo.value)

    def test_conforming_cluster_id_in_yaml_loads(self, tmp_path, monkeypatch):
        """Positive control for the loader path itself.

        Without it, the failure above could be caused by an unrelated YAML
        or env-var problem rather than by the slug rule.
        """
        config_file = tmp_path / 'config.yaml'
        config_file.write_text(
            yaml.dump(
                {
                    'reconciliation': {
                        'procedural_knowledge_topic_guard_clusters': [
                            {'topic_id': 'eval-worktree-plan-tools-missing', 'phrases': ['a', 'b']},
                        ],
                    },
                },
            ),
        )
        monkeypatch.setenv('CONFIG_PATH', str(config_file))
        clusters = FusedMemoryConfig().reconciliation.procedural_knowledge_topic_guard_clusters
        assert [c.topic_id for c in clusters] == ['eval-worktree-plan-tools-missing']


class TestProceduralTopicGuardClustersDefault:
    """ReconciliationConfig seeds all known topic-guard clusters by default.

    Mix of known-contradictory (plan-tools, venv-shadowing, architect
    report_task_already_done main-reachability) and known-recurring
    (pytest-xdist, architect plan-revalidation after requeue/lock) topics --
    see the >=5 count and the per-topic-id assertions below.
    """

    def test_default_seeds_non_empty_clusters(self):
        clusters = ReconciliationConfig().procedural_knowledge_topic_guard_clusters
        assert isinstance(clusters, list)
        assert len(clusters) >= 5

    def test_default_seeds_all_known_topic_ids(self):
        clusters = ReconciliationConfig().procedural_knowledge_topic_guard_clusters
        topic_ids = {c.topic_id for c in clusters}
        assert 'eval-worktree-plan-tools-missing' in topic_ids
        assert 'eval-worktree-venv-shadowing' in topic_ids
        assert 'pytest-xdist-serial-override' in topic_ids
        assert 'architect-report-task-already-done-main-reachability' in topic_ids
        assert 'architect-plan-revalidation-requeue-lock' in topic_ids

    def test_pytest_xdist_cluster_hint_points_at_canonical_memory(self):
        clusters = ReconciliationConfig().procedural_knowledge_topic_guard_clusters
        cluster = next(c for c in clusters if c.topic_id == 'pytest-xdist-serial-override')
        assert '8bb3eb15-1133-4e7b-ac1f-5bac10329b51' in cluster.hint

    def test_every_seeded_cluster_is_well_formed(self):
        clusters = ReconciliationConfig().procedural_knowledge_topic_guard_clusters
        for cluster in clusters:
            assert isinstance(cluster, ProceduralTopicCluster)
            assert cluster.phrases, f'{cluster.topic_id} has empty phrases'
            assert cluster.min_phrase_hits >= 1


class TestReportTaskAlreadyDoneMainReachabilityCluster:
    """Topic-guard cluster for the architect report_task_already_done /
    main-reachable-commit family (gate task 3011, still open -- its 12-entry
    cluster awaits a consolidation ruling). Registered prospectively so the
    cluster stops growing while 3011 is parked; see the guard's other
    known-contradictory seeds (plan-tools, venv-shadowing) above.
    """

    def test_cluster_present_with_expected_phrases_and_hint(self):
        clusters = ReconciliationConfig().procedural_knowledge_topic_guard_clusters
        cluster = next(
            c
            for c in clusters
            if c.topic_id == 'architect-report-task-already-done-main-reachability'
        )
        assert cluster.phrases == [
            'report_task_already_done',
            'main-reachable',
            'merge-base --is-ancestor',
            '_handle_already_done_report',
        ]
        assert cluster.min_phrase_hits == 2
        assert '3011' in cluster.hint

    def test_matches_representative_near_duplicate_note(self):
        clusters = ReconciliationConfig().procedural_knowledge_topic_guard_clusters
        cluster = next(
            c
            for c in clusters
            if c.topic_id == 'architect-report-task-already-done-main-reachability'
        )
        note = (
            'The architect report_task_already_done requires a main-reachable '
            'commit, verified via git merge-base --is-ancestor by '
            '_handle_already_done_report.'
        )
        result = find_matching_topic_cluster(note, [cluster])
        assert result is not None
        assert result[0].topic_id == 'architect-report-task-already-done-main-reachability'

    def test_does_not_match_unrelated_merge_base_note(self):
        clusters = ReconciliationConfig().procedural_knowledge_topic_guard_clusters
        cluster = next(
            c
            for c in clusters
            if c.topic_id == 'architect-report-task-already-done-main-reachability'
        )
        # Only 'merge-base --is-ancestor' occurs here (1 distinct hit) -- a
        # plain git-ancestry-check note unrelated to report_task_already_done
        # must NOT reach min_phrase_hits and mis-route to gate 3011.
        unrelated_note = (
            'Use git merge-base --is-ancestor <sha> <branch> to test whether a '
            'commit is an ancestor of a branch tip before cherry-picking.'
        )
        assert find_matching_topic_cluster(unrelated_note, [cluster]) is None


class TestArchitectPlanRevalidationRequeueLockCluster:
    """Topic-guard cluster for the architect plan-revalidation after
    requeue/lock family (gate task 2973, already adjudicated). Phrases are
    drawn verbatim from the resulting canonical Mem0 entries 6a96a020
    (subcase plan_json_gitignore_wipe) and 974b0adb (subcase
    lost_plan_reconstruction).
    """

    def test_cluster_present_with_expected_phrases_and_hint(self):
        clusters = ReconciliationConfig().procedural_knowledge_topic_guard_clusters
        cluster = next(
            c for c in clusters if c.topic_id == 'architect-plan-revalidation-requeue-lock'
        )
        assert cluster.phrases == [
            '.task/plan.json',
            'plan-revalidation',
            'requeue rebase',
            'lost-plan reconstruction',
            'committed TDD steps',
        ]
        assert cluster.min_phrase_hits == 2
        assert '2973' in cluster.hint

    def test_matches_representative_near_duplicate_note(self):
        clusters = ReconciliationConfig().procedural_knowledge_topic_guard_clusters
        cluster = next(
            c for c in clusters if c.topic_id == 'architect-plan-revalidation-requeue-lock'
        )
        note = (
            'During architect plan-revalidation after a requeue rebase, check '
            'whether .task/plan.json still exists before choosing confirm vs '
            'recreate.'
        )
        result = find_matching_topic_cluster(note, [cluster])
        assert result is not None
        assert result[0].topic_id == 'architect-plan-revalidation-requeue-lock'

    def test_does_not_match_unrelated_plan_tools_note(self):
        clusters = ReconciliationConfig().procedural_knowledge_topic_guard_clusters
        cluster = next(
            c for c in clusters if c.topic_id == 'architect-plan-revalidation-requeue-lock'
        )
        # A generic plan-tools note (not about revalidation-after-requeue)
        # only hits '.task/plan.json' (1 distinct hit) -- must NOT reach
        # min_phrase_hits and mis-route to gate 2973.
        unrelated_note = (
            'Use create_plan and add_plan_step to build the plan; the '
            'plan-tools state persists in .task/plan.json.'
        )
        assert find_matching_topic_cluster(unrelated_note, [cluster]) is None

    def test_full_default_cluster_list_resolves_here_not_plan_tools_cluster(self):
        # eval-worktree-plan-tools-missing is seeded earlier in the default
        # list and find_matching_topic_cluster returns the FIRST qualifying
        # cluster, so a plan-revalidation note must not be shadowed by it.
        # The note below hits only 1 distinct phrase on that earlier cluster
        # ('plan.json', via '.task/plan.json') -- below its min_phrase_hits
        # of 2 -- so matching correctly falls through to this cluster's own
        # >=2 hits ('.task/plan.json', 'plan-revalidation', 'requeue rebase').
        clusters = ReconciliationConfig().procedural_knowledge_topic_guard_clusters
        note = (
            'During architect plan-revalidation after a requeue rebase, check '
            'whether .task/plan.json still exists before choosing confirm vs '
            'recreate.'
        )
        result = find_matching_topic_cluster(note, clusters)
        assert result is not None
        assert result[0].topic_id == 'architect-plan-revalidation-requeue-lock'


class TestWriteTriageConfig:
    """The write-triage band thresholds (task 3130, PRD leaf alpha).

    The schema's job here is to assert NO a-priori numeric threshold (PRD
    G6): both bounds default to None, meaning UNCALIBRATED, which the
    triage router must read as fail-open to ``stored``. A numeric default
    would reproduce exactly the failure this leaf corrects — the near-dup
    guard's inherited 0.92 could never fire on a genuine rediscovery pair
    measured at 0.824.
    """

    def test_section_exists_on_the_root_config(self):
        assert FusedMemoryConfig().write_triage is not None

    @pytest.mark.parametrize('field', ['t_high', 't_low', 'calibration_report_path'])
    def test_every_field_defaults_to_none(self, field):
        # Asserted on the SCHEMA CLASS, not on FusedMemoryConfig(), and the
        # distinction is the whole point. FusedMemoryConfig is a BaseSettings:
        # constructing it LOADS config.yaml, which by design now carries the
        # calibration run's measured thresholds. Asserting None there would
        # forbid the calibrated config this leaf exists to produce, while
        # saying nothing about whether a guessed number is baked into the
        # code. The invariant that actually matters is that the schema ships
        # no a-priori number, so any value the server reads had to come from
        # a calibration run.
        from fused_memory.config.schema import WriteTriageConfig  # noqa: PLC0415

        value = getattr(WriteTriageConfig(), field)
        assert value is None, (
            f'write_triage.{field} must default to None (UNCALIBRATED). '
            f'A shipped default would be an a-priori threshold; got {value!r}'
        )

    @pytest.mark.parametrize('field', ['t_high', 't_low', 'calibration_report_path'])
    def test_root_wiring_introduces_no_default(self, field):
        """The `write_triage` field itself must not smuggle in a default.

        Guards the other half of the same invariant: a
        ``Field(default_factory=lambda: WriteTriageConfig(t_high=0.9))`` would
        leave the submodel's own defaults clean while still shipping an
        a-priori threshold to every deployment.
        """
        factory = FusedMemoryConfig.model_fields['write_triage'].default_factory
        assert factory is not None, 'write_triage must be a bare submodel with a factory'
        assert getattr(factory(), field) is None  # type: ignore[call-arg]

    @pytest.mark.parametrize('field', ['t_high', 't_low'])
    @pytest.mark.parametrize('value', [0.0, 0.5, 1.0])
    def test_thresholds_accept_the_cosine_unit_range(self, field, value):
        from fused_memory.config.schema import WriteTriageConfig  # noqa: PLC0415

        assert getattr(WriteTriageConfig(**{field: value}), field) == value

    @pytest.mark.parametrize('field', ['t_high', 't_low'])
    @pytest.mark.parametrize('value', [-0.1, 1.1, 42.0])
    def test_thresholds_reject_out_of_range_values(self, field, value):
        from fused_memory.config.schema import WriteTriageConfig  # noqa: PLC0415

        with pytest.raises(ValidationError):
            WriteTriageConfig(**{field: value})

    def test_write_triage_is_a_bare_non_optional_submodel(self):
        """Required, not ``WriteTriageConfig | None``.

        reload.py's ``_iter_leaves`` descends only into a bare submodel; an
        ``X | None`` field is compared WHOLE and collapses to a single
        restart_required leaf (the esc-2718-1 lesson). That would make the
        calibration script's config write need a server restart instead of
        a hot reload.
        """
        from fused_memory.config.schema import WriteTriageConfig  # noqa: PLC0415

        annotation = FusedMemoryConfig.model_fields['write_triage'].annotation
        assert isinstance(annotation, type) and issubclass(annotation, WriteTriageConfig), (
            f'write_triage must be annotated as a bare submodel, got {annotation!r}'
        )


class TestMemoryMetadataConfig:
    """`memory_metadata` — the Mem0 metadata write-boundary section (task 3195, leaf β).

    A direct sibling of ``TaskMetadataConfig``: same top-level placement, same
    warn-by-default posture, same RED-TIER/restart-only wording on the enforce
    flags. PRD D3 names that precedent explicitly ("census first, tiers later"),
    so this section follows it rather than inventing a second shape.
    """

    def _cls(self):
        from fused_memory.config.schema import MemoryMetadataConfig  # noqa: PLC0415

        return MemoryMetadataConfig

    def test_section_is_top_level_not_nested(self):
        """Top-level, NOT under ``taskmaster``/``reconciliation``.

        It governs a vocabulary shared beyond any one backend (the registry
        lives in ``fused_memory.memory_metadata`` and leaf ι's prompt tests
        import it), exactly the rationale ``TaskMetadataConfig`` records for
        its own placement.
        """
        from fused_memory.config.schema import (  # noqa: PLC0415
            MemoryMetadataConfig,
            ReconciliationConfig,
            TaskmasterConfig,
        )

        assert 'memory_metadata' in FusedMemoryConfig.model_fields
        annotation = FusedMemoryConfig.model_fields['memory_metadata'].annotation
        assert isinstance(annotation, type) and issubclass(annotation, MemoryMetadataConfig)
        assert 'memory_metadata' not in TaskmasterConfig.model_fields
        assert 'memory_metadata' not in ReconciliationConfig.model_fields

    def test_defaults_are_warn_mode(self):
        """Both enforce flags default OFF.

        This is the census-refuted-premise safety default, not timidity: leaf
        α measured 242 of 329 live `kind` values as singletons, so a day-one
        strict reject would turn every newly invented kind into a hard
        memory-write failure on the live fleet.
        """
        cfg = FusedMemoryConfig().memory_metadata
        assert cfg.enforce is False
        assert cfg.enforce_kind_registry is False
        assert cfg.unknown_key_storm_threshold == 50
        assert cfg.unknown_key_storm_window_seconds == 300

    def test_extra_keys_are_forbidden(self):
        """A mistyped leaf must fail LOUD at config load/reload.

        With ``extra='ignore'`` an operator who typed ``enforce_kind_regsitry``
        would get a silently-ignored key and believe enforcement was on — the
        no-silent-fail-soft invariant applied to the config surface.
        """
        assert self._cls().model_config.get('extra') == 'forbid'
        with pytest.raises(ValidationError):
            # Splatted rather than written as a literal keyword: the typo is
            # the POINT of the test, and pyright rejects a misspelled keyword
            # statically, which would fail the type gate before the runtime
            # assertion could ever run.
            self._cls()(**{'enfroce': True})

    @pytest.mark.parametrize('bad', [0, -1, -50])
    def test_storm_threshold_rejects_non_positive(self, bad):
        with pytest.raises(ValidationError):
            self._cls()(unknown_key_storm_threshold=bad)

    @pytest.mark.parametrize('bad', [0, -1, -300])
    def test_storm_window_rejects_non_positive(self, bad):
        with pytest.raises(ValidationError):
            self._cls()(unknown_key_storm_window_seconds=bad)

    def test_round_trips_from_a_config_dict(self):
        # `model_validate` rather than a splatted `__init__`: this is how a
        # loaded config.yaml actually reaches the model, and a splat makes
        # pyright widen every sibling field's type to the dict's value type,
        # producing a type error per section of FusedMemoryConfig.
        cfg = FusedMemoryConfig.model_validate(
            {'memory_metadata': {'enforce': True, 'enforce_kind_registry': True}}
        )
        assert cfg.memory_metadata.enforce is True
        assert cfg.memory_metadata.enforce_kind_registry is True

    @pytest.mark.parametrize('field', ['enforce', 'enforce_kind_registry'])
    def test_enforce_flags_are_restart_only(self, field):
        """RED TIER, asserted behaviourally rather than as a doc blurb.

        The operator-facing promise is not that some `description=` string
        contains the letters "restart" — it is that `reload_config` REPORTS
        this leaf as ``restart_required`` and does not silently no-op it.
        That is a property of the reload allowlist and of `diff_config`, so
        this pins both: the section has no hot-reloadable leaf, and a real
        diff over a flipped flag buckets red, not green.

        The step-11 ``description=`` strings are retained as operator
        documentation; they are simply no longer test-pinned. Prose is
        reviewed by humans and enforced by neither pyright nor pytest, and a
        substring check over it fails open anyway (``'restarting is not
        required'`` would have passed the check this replaces).
        """
        from fused_memory.config.reload import (  # noqa: PLC0415
            RELOADABLE_FIELDS,
            diff_config,
        )

        # 1. No leaf of this section is hot-reloadable. An added-and-
        #    unreviewed allowlist entry fails here, loudly.
        assert not any(f.startswith('memory_metadata.') for f in RELOADABLE_FIELDS)

        # 2. Observable bucketing through the real diff path, mirroring the
        #    red-tier precedent `TestDiffConfig.
        #    test_non_allowlisted_leaf_lands_in_restart_required`.
        live = FusedMemoryConfig()
        fresh = FusedMemoryConfig()
        old = getattr(live.memory_metadata, field)
        # `object.__setattr__` is the established idiom in test_config_reload:
        # it bypasses the validation/assignment wrapper so the diff sees a raw
        # differing leaf.
        object.__setattr__(fresh.memory_metadata, field, not old)

        d = diff_config(live, fresh)

        assert d.restart_required[f'memory_metadata.{field}'] == {'old': old, 'new': not old}
        assert f'memory_metadata.{field}' not in d.applied_candidates
