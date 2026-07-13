"""Tests for scripts/legibility/config.py — §7.4 per-project config schema.

``config.load_config(path)`` reads a YAML file and returns a typed,
validated ``LegibilityConfig`` (pydantic). Mirrors the
``shared.task_metadata`` BeforeDone/Milestone pattern: nested pydantic
models with ``extra='allow'`` for forward-compat, ``yaml.safe_load`` +
model construction so malformed input raises ``pydantic.ValidationError``
rather than being silently discarded.

Imported as ``from legibility import config`` — ``scripts/legibility/`` is
a PEP-420 namespace package (no ``__init__.py``), resolvable because
``scripts/tests/conftest.py`` already puts ``scripts/`` on ``sys.path``
under pytest's ``--import-mode=importlib``.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from legibility import config as mod

MINIMAL_YAML = textwrap.dedent("""\
    project_id: dark_factory
    project_root: /home/leo/src/dark-factory
    escalation_port: 8103
    cwd_prefixes:
      - /home/leo/src/dark-factory
    """)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / 'legibility.yaml'
    path.write_text(text)
    return path


class TestLoadConfigMinimal:
    """A minimal §7.4 YAML (no budgets/sampling/census/models blocks) loads
    into a typed LegibilityConfig with the four required top-level fields."""

    def test_returns_typed_legibility_config(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert isinstance(cfg, mod.LegibilityConfig)

    def test_top_level_scalars(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert cfg.project_id == 'dark_factory'
        assert cfg.project_root == '/home/leo/src/dark-factory'
        assert cfg.escalation_port == 8103

    def test_cwd_prefixes_list(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert cfg.cwd_prefixes == ['/home/leo/src/dark-factory']

    def test_nested_blocks_present_via_defaults(self, tmp_path):
        # Nested blocks are entirely omitted from MINIMAL_YAML, yet each is
        # still a real nested model instance (never None) with §7.4 defaults.
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert isinstance(cfg.budgets, mod.Budgets)
        assert isinstance(cfg.sampling, mod.Sampling)
        assert isinstance(cfg.census, mod.Census)
        assert isinstance(cfg.models, mod.Models)


class TestNestedDefaults:
    """§7.4 defaults apply per-field when a nested block is omitted or
    only partially specified — the sampling block is the driving case."""

    def test_sampling_defaults_when_block_omitted_entirely(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert cfg.sampling.top_fraction == 0.12
        assert cfg.sampling.per_stratum_min == 2

    def test_sampling_defaults_when_block_present_but_empty(self, tmp_path):
        text = MINIMAL_YAML + 'sampling: {}\n'
        cfg = mod.load_config(_write(tmp_path, text))
        assert cfg.sampling.top_fraction == 0.12
        assert cfg.sampling.per_stratum_min == 2

    def test_partial_sampling_block_keeps_other_default(self, tmp_path):
        # Only top_fraction is overridden; per_stratum_min must still
        # default rather than becoming required or vanishing.
        text = MINIMAL_YAML + 'sampling: {top_fraction: 0.2}\n'
        cfg = mod.load_config(_write(tmp_path, text))
        assert cfg.sampling.top_fraction == 0.2
        assert cfg.sampling.per_stratum_min == 2

    def test_budgets_default(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert cfg.budgets.max_daily_digest_bytes == 300000

    def test_census_defaults(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert cfg.census.max_interval_days == 10
        assert cfg.census.tasks_landed_threshold == 120
        assert cfg.census.tasks_landed_min_days == 7
        assert cfg.census.novelty_spike.count == 4
        assert cfg.census.novelty_spike.window_hours == 72
        assert cfg.census.floor_days == 5
        assert cfg.census.saturation.dup_rate == 0.9
        assert cfg.census.saturation.consecutive_batches == 2

    def test_models_defaults(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert cfg.models.trickle == 'haiku'
        assert cfg.models.census_miner == 'sonnet'
        assert cfg.models.census_verify == 'sonnet'
        assert cfg.models.census_synthesis == 'fable'


class TestFullConfigOverridesDefaults:
    """A fully-populated §7.4 YAML round-trips every explicit value."""

    FULL_YAML = MINIMAL_YAML + textwrap.dedent("""\
        budgets: {max_daily_digest_bytes: 123456}
        sampling: {top_fraction: 0.2, per_stratum_min: 3}
        census:
          max_interval_days: 11
          tasks_landed_threshold: 200
          tasks_landed_min_days: 8
          novelty_spike: {count: 5, window_hours: 48}
          floor_days: 6
          saturation: {dup_rate: 0.8, consecutive_batches: 3}
        models: {trickle: haiku, census_miner: sonnet, census_verify: sonnet, census_synthesis: fable}
        """)

    def test_every_explicit_value_round_trips(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, self.FULL_YAML))
        assert cfg.budgets.max_daily_digest_bytes == 123456
        assert cfg.sampling.top_fraction == 0.2
        assert cfg.sampling.per_stratum_min == 3
        assert cfg.census.max_interval_days == 11
        assert cfg.census.tasks_landed_threshold == 200
        assert cfg.census.tasks_landed_min_days == 8
        assert cfg.census.novelty_spike.count == 5
        assert cfg.census.novelty_spike.window_hours == 48
        assert cfg.census.floor_days == 6
        assert cfg.census.saturation.dup_rate == 0.8
        assert cfg.census.saturation.consecutive_batches == 3


class TestMalformedConfigRaises:
    """Malformed §7.4 configs raise pydantic.ValidationError — never a
    silently-defaulted or partially-applied model."""

    def test_missing_project_id_raises(self, tmp_path):
        text = textwrap.dedent("""\
            project_root: /home/leo/src/dark-factory
            escalation_port: 8103
            cwd_prefixes: [/home/leo/src/dark-factory]
            """)
        with pytest.raises(ValidationError):
            mod.load_config(_write(tmp_path, text))

    def test_cwd_prefixes_not_a_list_raises(self, tmp_path):
        text = textwrap.dedent("""\
            project_id: dark_factory
            project_root: /home/leo/src/dark-factory
            escalation_port: 8103
            cwd_prefixes: /home/leo/src/dark-factory
            """)
        with pytest.raises(ValidationError):
            mod.load_config(_write(tmp_path, text))

    def test_non_int_escalation_port_raises(self, tmp_path):
        text = textwrap.dedent("""\
            project_id: dark_factory
            project_root: /home/leo/src/dark-factory
            escalation_port: not-a-port
            cwd_prefixes: [/home/leo/src/dark-factory]
            """)
        with pytest.raises(ValidationError):
            mod.load_config(_write(tmp_path, text))
