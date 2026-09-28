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
import yaml
from legibility import config as mod
from pydantic import ValidationError

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


class TestTimeouts:
    """The ``timeouts:`` block — per-census-stage claude-CLI subprocess
    budgets (census_mining_secs / census_verify_secs / census_synthesis_secs).

    An omitted block loads with all three defaults (120/900/1800), so an
    existing legibility.yaml that predates this block keeps working
    unchanged — the driving acceptance criterion of the fix that gave
    verify/synthesis their own budgets after the shared 120s coder default
    killed the first dark_factory census.
    """

    def test_timeouts_defaults_when_block_omitted_entirely(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert isinstance(cfg.timeouts, mod.Timeouts)
        assert cfg.timeouts.census_mining_secs == 120
        assert cfg.timeouts.census_verify_secs == 900
        assert cfg.timeouts.census_synthesis_secs == 1800

    def test_timeouts_defaults_when_block_present_but_empty(self, tmp_path):
        text = MINIMAL_YAML + 'timeouts: {}\n'
        cfg = mod.load_config(_write(tmp_path, text))
        assert cfg.timeouts.census_mining_secs == 120
        assert cfg.timeouts.census_verify_secs == 900
        assert cfg.timeouts.census_synthesis_secs == 1800

    def test_partial_timeouts_block_keeps_other_defaults(self, tmp_path):
        # Only census_verify_secs is overridden; mining and synthesis must
        # still default rather than becoming required or vanishing.
        text = MINIMAL_YAML + 'timeouts: {census_verify_secs: 1200}\n'
        cfg = mod.load_config(_write(tmp_path, text))
        assert cfg.timeouts.census_verify_secs == 1200
        assert cfg.timeouts.census_mining_secs == 120
        assert cfg.timeouts.census_synthesis_secs == 1800

    def test_full_timeouts_override_round_trips(self, tmp_path):
        text = MINIMAL_YAML + (
            'timeouts: {census_mining_secs: 60, census_verify_secs: 1200, '
            'census_synthesis_secs: 2400}\n'
        )
        cfg = mod.load_config(_write(tmp_path, text))
        assert cfg.timeouts.census_mining_secs == 60
        assert cfg.timeouts.census_verify_secs == 1200
        assert cfg.timeouts.census_synthesis_secs == 2400


class TestTrickleCensusCaps:
    """The ``census.trickle_caps`` block — the cost caps the nightly trickle
    forwards to the census it launches (census.py's --max-batches /
    --max-verify-clusters).

    An omitted block is BOUNDED (50/150), never uncapped: a trickle launch
    with no config opinion must not run an unattended census without a
    runaway backstop. ``null`` is the explicit uncapped opt-out. A bad cap
    fails loud at load_config rather than reaching census.py's argv, where
    it would exit 2 on every fired night.
    """

    def test_bounded_defaults_when_census_block_omitted_entirely(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert isinstance(cfg.census.trickle_caps, mod.TrickleCensusCaps)
        assert cfg.census.trickle_caps.max_batches == 50
        assert cfg.census.trickle_caps.max_verify_clusters == 150

    def test_partial_block_keeps_other_default(self, tmp_path):
        text = MINIMAL_YAML + 'census: {trickle_caps: {max_batches: 10}}\n'
        cfg = mod.load_config(_write(tmp_path, text))
        assert cfg.census.trickle_caps.max_batches == 10
        assert cfg.census.trickle_caps.max_verify_clusters == 150

    def test_null_caps_are_the_explicit_uncapped_opt_out(self, tmp_path):
        text = MINIMAL_YAML + (
            'census: {trickle_caps: {max_batches: null, max_verify_clusters: null}}\n'
        )
        cfg = mod.load_config(_write(tmp_path, text))
        assert cfg.census.trickle_caps.max_batches is None
        assert cfg.census.trickle_caps.max_verify_clusters is None

    @pytest.mark.parametrize('field', ['max_batches', 'max_verify_clusters'])
    @pytest.mark.parametrize('bad_value', ['0', '-1', 'true', "'50'", '2.5'])
    def test_non_positive_int_cap_raises(self, tmp_path, field, bad_value):
        # ``true`` would otherwise coerce to a silent 1-batch cap; 0 and
        # negatives mirror census.py::_positive_int's CLI-boundary rejection.
        text = MINIMAL_YAML + f'census: {{trickle_caps: {{{field}: {bad_value}}}}}\n'
        with pytest.raises(ValidationError):
            mod.load_config(_write(tmp_path, text))

    def test_trigger_thresholds_survive_alongside_trickle_caps(self, tmp_path):
        text = MINIMAL_YAML + textwrap.dedent("""\
            census:
              max_interval_days: 3
              trickle_caps: {max_batches: 10}
            """)
        cfg = mod.load_config(_write(tmp_path, text))
        assert cfg.census.max_interval_days == 3
        assert cfg.census.trickle_caps.max_batches == 10


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


class TestAgentTranscriptRoots:
    """agent_transcript_roots — the additional archive roots the miner
    enumerates alongside ~/.claude/projects. Defaults to [] (the parity
    baseline) when omitted, round-trips a list of strings, and rejects a
    non-list value with a pydantic.ValidationError."""

    def test_defaults_to_empty_list_when_omitted(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert cfg.agent_transcript_roots == []

    def test_present_list_round_trips(self, tmp_path):
        text = MINIMAL_YAML + textwrap.dedent("""\
            agent_transcript_roots:
              - data/orchestrator/agent-transcripts
              - /var/lib/agent-transcripts
            """)
        cfg = mod.load_config(_write(tmp_path, text))
        assert cfg.agent_transcript_roots == [
            'data/orchestrator/agent-transcripts',
            '/var/lib/agent-transcripts',
        ]

    def test_non_list_value_raises(self, tmp_path):
        text = MINIMAL_YAML + 'agent_transcript_roots: data/orchestrator/agent-transcripts\n'
        with pytest.raises(ValidationError):
            mod.load_config(_write(tmp_path, text))


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


class TestProjectRootAbsoluteness:
    """``project_root`` must be an absolute path — a relative value fails
    loudly at ``load_config`` rather than letting each consumer resolve it
    against its own process cwd (task 3702, reviewer suggestion #1 on task
    3269's ambient-cwd fix)."""

    def test_dot_project_root_raises(self, tmp_path):
        text = textwrap.dedent("""\
            project_id: dark_factory
            project_root: .
            escalation_port: 8103
            cwd_prefixes: [/home/leo/src/dark-factory]
            """)
        # Matches on 'absolute' — the token that pins the *behavior* under
        # test — not 'project_root', which pydantic prints in the error
        # header for ANY project_root-level failure (missing field, wrong
        # type, ...) and would pass even if this validator were replaced by
        # an unrelated constraint on the same field.
        with pytest.raises(ValidationError, match='absolute'):
            mod.load_config(_write(tmp_path, text))

    def test_relative_dotdot_project_root_raises(self, tmp_path):
        text = textwrap.dedent("""\
            project_id: dark_factory
            project_root: ../foo
            escalation_port: 8103
            cwd_prefixes: [/home/leo/src/dark-factory]
            """)
        with pytest.raises(ValidationError, match='absolute'):
            mod.load_config(_write(tmp_path, text))

    def test_empty_string_project_root_raises(self, tmp_path):
        # os.path.isabs('') is False — an accidentally-blanked required
        # field is the likeliest real-world malformed value, so it must
        # fail the same way as an explicit relative path rather than
        # slipping through some falsy-value special case.
        text = textwrap.dedent("""\
            project_id: dark_factory
            project_root: ''
            escalation_port: 8103
            cwd_prefixes: [/home/leo/src/dark-factory]
            """)
        with pytest.raises(ValidationError, match='absolute'):
            mod.load_config(_write(tmp_path, text))

    def test_tilde_project_root_raises(self, tmp_path):
        # os.path.isabs('~/src/foo') is False — tilde expansion is NOT
        # performed before the absoluteness check, so a tilde-prefixed
        # value is rejected today. Pinned here as a deliberate decision
        # rather than left as an untested accident.
        text = textwrap.dedent("""\
            project_id: dark_factory
            project_root: ~/src/foo
            escalation_port: 8103
            cwd_prefixes: [/home/leo/src/dark-factory]
            """)
        with pytest.raises(ValidationError, match='absolute'):
            mod.load_config(_write(tmp_path, text))

    def test_absolute_project_root_still_loads(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert cfg.project_root == '/home/leo/src/dark-factory'


class TestCwdPrefixesAbsoluteness:
    """Every ``cwd_prefixes`` entry must be an absolute path too —
    ``inventory.is_member()`` matches each prefix against an absolute
    session cwd, so a relative entry never matches anything and the
    sampler/census would silently enumerate an empty corpus rather than
    failing at config-load time (task 3702 amendment, reviewer suggestion
    #1: the same silent-degradation defect the project_root check above
    was added to prevent). ``agent_transcript_roots`` is deliberately
    project_root-relative and is exercised separately in
    TestAgentTranscriptRoots — it must stay exempt from this check.
    """

    def test_single_relative_cwd_prefix_raises(self, tmp_path):
        text = textwrap.dedent("""\
            project_id: dark_factory
            project_root: /home/leo/src/dark-factory
            escalation_port: 8103
            cwd_prefixes: [.]
            """)
        with pytest.raises(ValidationError, match='absolute'):
            mod.load_config(_write(tmp_path, text))

    def test_one_relative_entry_among_absolute_ones_raises(self, tmp_path):
        text = textwrap.dedent("""\
            project_id: dark_factory
            project_root: /home/leo/src/dark-factory
            escalation_port: 8103
            cwd_prefixes:
              - /home/leo/src/dark-factory
              - src/foo
            """)
        with pytest.raises(ValidationError, match='absolute'):
            mod.load_config(_write(tmp_path, text))

    def test_absolute_cwd_prefixes_still_load(self, tmp_path):
        cfg = mod.load_config(_write(tmp_path, MINIMAL_YAML))
        assert cfg.cwd_prefixes == ['/home/leo/src/dark-factory']


class TestShippedDarkFactoryConfig:
    """The committed docs/legibility/legibility.yaml — dark_factory's own
    per-project §7.4 config — loads and validates through config.load_config.

    Resolved relative to this test file (scripts/tests/../.. == repo root)
    so the test works regardless of cwd.
    """

    SHIPPED_CONFIG_PATH = (
        Path(__file__).resolve().parents[2] / 'docs' / 'legibility' / 'legibility.yaml'
    )

    def test_shipped_config_top_level(self):
        cfg = mod.load_config(self.SHIPPED_CONFIG_PATH)
        assert cfg.project_id == 'dark_factory'
        assert cfg.project_root == '/home/leo/src/dark-factory'
        assert cfg.cwd_prefixes == ['/home/leo/src/dark-factory']

    def test_shipped_config_budgets_and_sampling(self):
        cfg = mod.load_config(self.SHIPPED_CONFIG_PATH)
        assert cfg.budgets.max_daily_digest_bytes == 300000
        assert cfg.sampling.top_fraction == 0.12
        assert cfg.sampling.per_stratum_min == 2

    def test_shipped_config_census_and_models_nonempty(self):
        cfg = mod.load_config(self.SHIPPED_CONFIG_PATH)
        assert cfg.census.max_interval_days > 0
        assert cfg.census.tasks_landed_threshold > 0
        assert cfg.models.trickle
        assert cfg.models.census_miner
        assert cfg.models.census_verify
        assert cfg.models.census_synthesis

    def test_shipped_config_timeouts_set_live(self):
        # The per-census-stage claude-CLI budgets are pinned EXPLICITLY in the
        # shipped config, not merely riding the schema defaults, so a future
        # change to Timeouts' defaults cannot silently alter dark_factory's
        # census budgets. The raw-text assertion is load-bearing: the schema
        # defaults happen to equal these values, so without an explicit
        # ``timeouts:`` block the loaded-value asserts alone would pass on
        # defaults — the text check is what pins the block's in-file presence.
        raw = self.SHIPPED_CONFIG_PATH.read_text(encoding='utf-8')
        assert 'timeouts:' in raw, 'shipped config must carry an explicit timeouts: block'

        cfg = mod.load_config(self.SHIPPED_CONFIG_PATH)
        assert cfg.timeouts.census_mining_secs == 120
        assert cfg.timeouts.census_verify_secs == 900
        assert cfg.timeouts.census_synthesis_secs == 1800

    def test_shipped_config_trickle_caps_pinned_explicitly(self):
        # Pinned in-file like timeouts:, so a change to TrickleCensusCaps'
        # defaults cannot silently alter dark_factory's census bound. The
        # structural assert is load-bearing: the values equal the schema
        # defaults, so the loaded-value assert alone would pass without the block.
        raw = yaml.safe_load(self.SHIPPED_CONFIG_PATH.read_text(encoding='utf-8'))
        assert raw['census']['trickle_caps'] == {'max_batches': 50, 'max_verify_clusters': 150}

        cfg = mod.load_config(self.SHIPPED_CONFIG_PATH)
        assert cfg.census.trickle_caps.max_batches == 50
        assert cfg.census.trickle_caps.max_verify_clusters == 150

    def test_shipped_config_agent_transcript_roots_set_live(self):
        # The CRITICAL Leo ask (plans/agent-transcript-archival-prd.md, task γ):
        # the shipped config ships the fleet archive root SET (live), not empty,
        # so the archived fleet-transcript corpus is enumerated with no operator
        # flip. Relative to project_root; git-ignored; produced by task α's
        # shared.transcript_archive.
        cfg = mod.load_config(self.SHIPPED_CONFIG_PATH)
        assert cfg.agent_transcript_roots == ['data/orchestrator/agent-transcripts']
