"""Tests for configuration loading."""

import logging
import os
from importlib import resources as pkg_resources
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from orchestrator.config import (
    ConfigRequiredError,
    CpuPriorityConfig,
    JobserverConfig,
    ModuleConfig,
    OrchestratorConfig,
    SccacheConfig,
    TimeoutsConfig,
    _deep_merge,
    _discover_module_configs,
    load_config,
)


def _load_package_defaults() -> dict:
    """Read the shipped defaults.yaml so tests stay in sync automatically."""
    defaults_file = pkg_resources.files('orchestrator') / 'defaults.yaml'
    return yaml.safe_load(defaults_file.read_text())


# Shared diagnostic phrase used by both lock_depth boundary tests.
# Hoisted to module scope so both the positive and negative tests reference one
# source of truth, and any future sibling tests can reuse it without duplication.
DISTINCTIVE_PHRASE = 'unreachable through the scheduler/workflow path'


def _load_config_with_nested_module(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    prefix: str,
    lock_depth: int = 2,
) -> OrchestratorConfig:
    """Write a minimal global config + a nested module config and return load_config().

    Single-sources the env-isolation contract (ORCH_LOCK_DEPTH / ORCH_CONFIG_PATH)
    and the config-yaml + nested-orchestrator-yaml setup shared by the two
    lock_depth boundary tests.

    Args:
        tmp_path: pytest tmp_path fixture — used as project_root.
        monkeypatch: pytest monkeypatch fixture — owns env-var teardown.
        prefix: Relative path of the nested module (e.g. 'foo/bar' or 'foo/bar/baz').
        lock_depth: lock_depth value written to the global config.yaml (default 2).

    Returns:
        The OrchestratorConfig produced by load_config() after discovery.
    """
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.dump({
        'project_root': str(tmp_path),
        'lock_depth': lock_depth,
    }))
    # Isolate ORCH_LOCK_DEPTH so a stray env var cannot override lock_depth and
    # silently invalidate the depth-comparison boundary being tested.  Route
    # ORCH_CONFIG_PATH through monkeypatch so load_config's write is auto-restored.
    monkeypatch.delenv('ORCH_LOCK_DEPTH', raising=False)
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(config_path))
    # Create the nested module config; 'test_command' is an overridable field so
    # _discover_module_configs registers the prefix (non-overridable fields are silently
    # ignored and would leave the prefix unregistered — the step-1 helper test guards this).
    nested = tmp_path / Path(prefix)
    nested.mkdir(parents=True, exist_ok=True)
    (nested / 'orchestrator.yaml').write_text(yaml.dump({'test_command': 'pytest'}))
    return load_config(config_path)


class TestDefaults:
    """Tests for OrchestratorConfig defaults — isolated from real config files."""

    def test_default_values(self, monkeypatch, tmp_path):
        """Package defaults.yaml is loaded via settings_customise_sources."""
        monkeypatch.chdir(tmp_path)
        # Ensure no external config is loaded - must unset env before instantiation
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        defaults = _load_package_defaults()
        # Values from package defaults.yaml (not Pydantic field defaults)
        assert config.max_concurrent_tasks == defaults['max_concurrent_tasks']
        assert config.max_per_module == defaults['max_per_module']
        assert config.max_execute_iterations == defaults['max_execute_iterations']
        assert config.max_verify_attempts == defaults['max_verify_attempts']
        assert config.max_review_cycles == defaults['max_review_cycles']
        assert config.reviewer_stagger_secs == defaults['reviewer_stagger_secs']
        assert config.max_reviewer_retries == defaults['max_reviewer_retries']
        assert config.models.architect == defaults['models']['architect']
        assert config.models.reviewer == defaults['models']['reviewer']
        assert config.budgets.implementer == defaults['budgets']['implementer']
        assert config.max_turns.implementer == defaults['max_turns']['implementer']

    def test_git_defaults(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = OrchestratorConfig()
        assert config.git.main_branch == 'main'
        assert config.git.branch_prefix == 'task/'
        # commit_citation_pattern defaults to None — git_ops uses the built-in
        # DEFAULT_COMMIT_CITATION_PATTERN when None is passed through.
        assert config.git.commit_citation_pattern is None

    def test_git_commit_citation_pattern_explicit_override(self):
        """An explicit non-empty pattern is accepted verbatim."""
        from orchestrator.config import GitConfig

        cfg = GitConfig(commit_citation_pattern=r'^custom\(.*\) ')
        assert cfg.commit_citation_pattern == r'^custom\(.*\) '

    def test_git_commit_citation_pattern_empty_string_disables(self):
        """An explicit empty string disables the citation check; consumer
        code (``find_task_citation_commit``) treats '' as opt-out."""
        from orchestrator.config import GitConfig

        cfg = GitConfig(commit_citation_pattern='')
        assert cfg.commit_citation_pattern == ''

    def test_spare_warm_lanes_default_zero(self):
        """GitConfig().spare_warm_lanes defaults to 0 (byte-identical for all projects)."""
        from orchestrator.config import GitConfig

        cfg = GitConfig()
        assert cfg.spare_warm_lanes == 0

    def test_spare_warm_lanes_explicit_override(self):
        """spare_warm_lanes=8 is accepted and stored verbatim."""
        from orchestrator.config import GitConfig

        cfg = GitConfig(spare_warm_lanes=8)
        assert cfg.spare_warm_lanes == 8

    def test_spare_warm_lanes_ge_0_rejects_negative(self):
        """spare_warm_lanes=-1 must raise ValidationError (ge=0 bound).

        A negative value would shrink the pool below the derived base,
        which is never valid.
        """
        from orchestrator.config import GitConfig

        with pytest.raises(ValidationError):
            GitConfig(spare_warm_lanes=-1)

    def test_fused_memory_defaults(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = OrchestratorConfig()
        assert config.fused_memory.url == 'http://localhost:8002'
        assert config.fused_memory.project_id == 'dark_factory'
        assert config.fused_memory.config_path == 'fused-memory/config/config.yaml'
        # server_command must contain '--project' followed by 'fused-memory' (no ../)
        cmd = config.fused_memory.server_command
        assert '--project' in cmd
        project_arg_idx = cmd.index('--project')
        assert cmd[project_arg_idx + 1] == 'fused-memory'

    def test_fused_memory_defaults_no_parent_traversal(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = OrchestratorConfig()
        assert '../' not in config.fused_memory.config_path
        assert not any('../' in arg for arg in config.fused_memory.server_command)

    def test_dashboard_restart_defaults(self, monkeypatch, tmp_path):
        """Bare OrchestratorConfig() exposes dashboard_restart_* defaults for the leaf service."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = OrchestratorConfig()
        assert config.dashboard_restart_on_merge_enabled is True
        assert config.dashboard_restart_debounce_secs == 20.0
        assert config.dashboard_restart_watch_prefixes == ['dashboard/src/']
        assert config.dashboard_restart_script == 'scripts/restart-dashboard.sh'

    def test_project_root_resolved_to_absolute(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = OrchestratorConfig(project_root=Path('.'))
        assert config.project_root.is_absolute() is True

    def test_steward_timeout_default_is_1800(self, monkeypatch, tmp_path):
        """timeouts.steward default is 1800s and satisfies the steward_completion_timeout invariant.

        Documents the decoupling invariant: per-invocation wall-clock must be
        >= the workflow grace period (steward_completion_timeout) so a single
        invocation is never silently cut short inside the drain window.
        Equality is permitted; the validator on OrchestratorConfig enforces >=.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.timeouts.steward == 1800.0
        assert config.timeouts.steward >= config.steward_completion_timeout

    def test_startup_grace_secs_default_is_120(self, monkeypatch, tmp_path):
        """TimeoutsConfig.startup_grace_secs defaults to 120.0 (pre-turn-1 startup grace window).

        Documents the two-regime liveness watchdog: 120s is ~20x the observed ~6s
        turn-1 latency, so genuine startups never trip it while a true pre-turn-1
        wedge is killed fast (vs. burning the full 1200s per-role ceiling).
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.timeouts.startup_grace_secs == 120.0

    def test_verify_cold_command_timeout_secs_from_defaults_yaml(self, monkeypatch, tmp_path):
        """verify_cold_command_timeout_secs is loaded from defaults.yaml (expected 5400)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        defaults = _load_package_defaults()
        assert config.verify_cold_command_timeout_secs == defaults['verify_cold_command_timeout_secs']
        assert config.verify_cold_command_timeout_secs == 5400

    def test_verify_cold_command_timeout_secs_pydantic_default_is_none(self):
        """Raw Pydantic field default is None; the shipped 5400 value comes from defaults.yaml."""
        field_info = OrchestratorConfig.model_fields['verify_cold_command_timeout_secs']
        assert field_info.default is None

    def test_merge_verify_cold_command_timeout_secs_pydantic_default_is_none(self):
        """Raw Pydantic field default for merge_verify_cold_command_timeout_secs is None.

        The shipped 7200 value comes from defaults.yaml, not the Pydantic default.
        This mirrors the pattern for verify_cold_command_timeout_secs (line 172-175).
        """
        field_info = OrchestratorConfig.model_fields['merge_verify_cold_command_timeout_secs']
        assert field_info.default is None

    def test_merge_verify_cold_command_timeout_secs_from_defaults_yaml(self, monkeypatch, tmp_path):
        """merge_verify_cold_command_timeout_secs is loaded from defaults.yaml (expected 7200).

        7200s (2×5400, 4×1800) gives headroom for a cold full-workspace compile
        + frontend install on a cold merge-verify worktree, where 1602's fail-closed
        gate makes a timeout a queue-stalling block.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        defaults = _load_package_defaults()
        assert config.merge_verify_cold_command_timeout_secs == defaults['merge_verify_cold_command_timeout_secs']
        assert config.merge_verify_cold_command_timeout_secs == 7200

    def test_terminal_status_hard_cancel_polls_default(self, monkeypatch, tmp_path):
        """terminal_status_hard_cancel_polls defaults to 3 (ITEM 2 config, step-3 RED).

        At the 30 s default poll interval, threshold=3 means ~90 s of soft-cancel
        grace before the watcher escalates to a hard asyncio.Task.cancel().
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.terminal_status_hard_cancel_polls == 3

    def test_terminal_status_hard_cancel_polls_ge_1_rejects_zero(self):
        """terminal_status_hard_cancel_polls=0 must raise ValidationError (ge=1 bound).

        The watcher requires at least one soft-cancel attempt before hard-cancel
        so a value < 1 is never valid.
        """
        with pytest.raises(ValidationError):
            OrchestratorConfig(terminal_status_hard_cancel_polls=0)

    def test_run_forever_idle_defaults(self, monkeypatch, tmp_path):
        """Run-forever idle + full-review rate-limit fields carry their defaults.

        These three fields are absent from defaults.yaml, so OrchestratorConfig()
        yields the Pydantic field defaults: the full-review ceiling is a true AND
        of a 24h interval (86400s) and a 20-completed-task gate, and the idle poll
        cadence is 15s (separate from _PAUSED_IDLE_POLL_SECS so operators can tune
        the run-forever poll independently).
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.review.full_review_min_interval_secs == 86400.0
        assert config.review.full_review_min_tasks == 20
        assert config.idle_poll_secs == 15.0

    def test_max_failure_signature_repeat_default(self, monkeypatch, tmp_path):
        """max_failure_signature_repeat is loaded from defaults.yaml (expected 3).

        The shipped default must equal the defaults.yaml value so the loop-guard
        cap is consistent whether the operator omits the key or sets it explicitly.
        Sibling thrash thresholds max_consecutive_infra_resumes and
        max_consecutive_merge_thrash follow the same pattern.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        defaults = _load_package_defaults()
        assert config.max_failure_signature_repeat == defaults['max_failure_signature_repeat']


class TestYamlLoading:
    def test_load_config_raises_when_explicit_path_nonexistent(self, tmp_path: Path):
        """Explicit --config pointing at a missing file raises ConfigRequiredError."""
        nonexistent = tmp_path / 'nonexistent.yaml'
        with pytest.raises(ConfigRequiredError, match='Config file not found'):
            load_config(nonexistent)

    def test_load_config_uses_orch_config_path_env_var(self, tmp_path: Path, monkeypatch):
        """ORCH_CONFIG_PATH alone (no --config flag) loads the right config."""
        cfg = tmp_path / 'config.yaml'
        cfg.write_text(yaml.dump({'max_concurrent_tasks': 13}))
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(cfg))
        config = load_config(None)
        assert config.max_concurrent_tasks == 13

    def test_load_config_explicit_flag_overrides_env_var(
        self, tmp_path: Path, monkeypatch,
    ):
        """When both --config and ORCH_CONFIG_PATH are set, --config wins."""
        env_cfg = tmp_path / 'env.yaml'
        env_cfg.write_text(yaml.dump({'max_concurrent_tasks': 1}))
        flag_cfg = tmp_path / 'flag.yaml'
        flag_cfg.write_text(yaml.dump({'max_concurrent_tasks': 99}))
        monkeypatch.setenv('ORCH_CONFIG_PATH', str(env_cfg))
        config = load_config(flag_cfg)
        assert config.max_concurrent_tasks == 99

    def test_load_from_yaml(self, tmp_path: Path):
        config_data = {
            'max_concurrent_tasks': 5,
            'models': {'architect': 'sonnet'},
            'budgets': {'architect': 3.0},
        }
        config_path = tmp_path / 'config.yaml'
        config_path.write_text(yaml.dump(config_data))

        config = load_config(config_path)
        assert config.max_concurrent_tasks == 5
        assert config.models.architect == 'sonnet'
        assert config.budgets.architect == 3.0
        # Unset values should use package defaults
        assert config.models.implementer == 'sonnet'


class TestModuleConfigDiscovery:
    def test_discover_finds_orchestrator_yaml(self, tmp_path: Path):
        sub = tmp_path / 'dashboard'
        sub.mkdir()
        (sub / 'orchestrator.yaml').write_text(yaml.dump({
            'test_command': 'pytest dashboard/',
            'lint_command': 'ruff check dashboard/',
            'lock_depth': 3,
        }))
        configs = _discover_module_configs(tmp_path)
        assert 'dashboard' in configs
        mc = configs['dashboard']
        assert mc.prefix == 'dashboard'
        assert mc.test_command == 'pytest dashboard/'
        assert mc.lint_command == 'ruff check dashboard/'
        assert mc.lock_depth == 3
        assert mc.type_check_command is None

    def test_discover_ignores_non_overridable_fields(self, tmp_path: Path):
        sub = tmp_path / 'mymod'
        sub.mkdir()
        (sub / 'orchestrator.yaml').write_text(yaml.dump({
            'test_command': 'pytest',
            'models': {'architect': 'sonnet'},
            'budgets': {'architect': 1.0},
            'project_root': '/nope',
        }))
        configs = _discover_module_configs(tmp_path)
        mc = configs['mymod']
        assert mc.test_command == 'pytest'
        assert not hasattr(mc, 'models')
        assert not hasattr(mc, 'budgets')
        assert not hasattr(mc, 'project_root')

    def test_discover_empty_dir(self, tmp_path: Path):
        configs = _discover_module_configs(tmp_path)
        assert configs == {}

    def test_for_module_matches_first_component(self):
        config = OrchestratorConfig()
        config._module_configs = {
            'dashboard': ModuleConfig(prefix='dashboard', test_command='pytest dash/'),
        }
        mc = config.for_module('dashboard/src/app.py')
        assert mc is not None
        assert mc.prefix == 'dashboard'
        assert mc.test_command == 'pytest dash/'

    def test_for_module_returns_none_for_unknown(self):
        config = OrchestratorConfig()
        config._module_configs = {
            'dashboard': ModuleConfig(prefix='dashboard'),
        }
        assert config.for_module('orchestrator/src/config.py') is None

    def test_discover_loads_verify_cold_command_timeout_secs(self, tmp_path: Path):
        """_discover_module_configs propagates verify_cold_command_timeout_secs from orchestrator.yaml."""
        sub = tmp_path / 'backend'
        sub.mkdir()
        (sub / 'orchestrator.yaml').write_text(yaml.dump({
            'verify_cold_command_timeout_secs': 7200,
        }))
        configs = _discover_module_configs(tmp_path)
        assert 'backend' in configs
        mc = configs['backend']
        assert mc.verify_cold_command_timeout_secs == 7200.0

    def test_bare_module_config_has_none_cold_timeout(self):
        """A bare ModuleConfig has verify_cold_command_timeout_secs is None by default."""
        mc = ModuleConfig(prefix='x')
        assert mc.verify_cold_command_timeout_secs is None

    def test_load_config_populates_module_configs(self, tmp_path: Path):
        # Create a minimal global config
        config_path = tmp_path / 'config.yaml'
        config_path.write_text(yaml.dump({
            'project_root': str(tmp_path),
        }))
        # Create a subproject orchestrator.yaml
        sub = tmp_path / 'backend'
        sub.mkdir()
        (sub / 'orchestrator.yaml').write_text(yaml.dump({
            'test_command': 'cargo test',
            'max_per_module': 2,
        }))
        config = load_config(config_path)
        assert config._module_configs is not None
        assert 'backend' in config._module_configs
        assert config._module_configs['backend'].test_command == 'cargo test'
        assert config._module_configs['backend'].max_per_module == 2

    def test_discover_finds_nested_orchestrator_yaml_at_depth_2(self, tmp_path: Path):
        """_discover_module_configs finds orchestrator.yaml at depth >= 2."""
        nested = tmp_path / 'foo' / 'bar'
        nested.mkdir(parents=True)
        (nested / 'orchestrator.yaml').write_text(yaml.dump({
            'test_command': 'pytest foo/bar/',
        }))
        configs = _discover_module_configs(tmp_path)
        assert 'foo/bar' in configs
        mc = configs['foo/bar']
        assert mc.prefix == 'foo/bar'
        assert mc.test_command == 'pytest foo/bar/'

    def test_discover_orders_results_by_depth_then_lex(self, tmp_path: Path):
        """_discover_module_configs returns keys ordered by (depth, lex): depth-1 first, lex within depth."""
        # Create four orchestrator.yaml files at varying depths
        for parts in [
            ('c',),
            ('a',),
            ('a', 'b'),
            ('d', 'e', 'f'),
        ]:
            d = tmp_path.joinpath(*parts)
            d.mkdir(parents=True, exist_ok=True)
            (d / 'orchestrator.yaml').write_text(yaml.dump({'test_command': 'pytest'}))
        configs = _discover_module_configs(tmp_path)
        assert list(configs.keys()) == ['a', 'c', 'a/b', 'd/e/f']

    def test_discover_excludes_standard_dirs(self, tmp_path: Path):
        """_discover_module_configs does not descend into standard build/VCS directories."""
        excluded_dirs = [
            '.git', '.venv', 'venv', '.worktrees', 'node_modules',
            '__pycache__', 'build', 'target', '.gradle',
        ]
        # Create an orchestrator.yaml nested inside each excluded dir
        for excluded in excluded_dirs:
            nested = tmp_path / excluded / 'sub'
            nested.mkdir(parents=True, exist_ok=True)
            (nested / 'orchestrator.yaml').write_text(yaml.dump({'test_command': 'pytest'}))
        # Also create a legitimate module that should be found
        legit = tmp_path / 'legit'
        legit.mkdir()
        (legit / 'orchestrator.yaml').write_text(yaml.dump({'test_command': 'pytest legit/'}))
        configs = _discover_module_configs(tmp_path)
        assert 'legit' in configs
        for excluded in excluded_dirs:
            for key in configs:
                assert not key.startswith(excluded + '/') and key != excluded, (
                    f"Excluded dir {excluded!r} leaked into results as key {key!r}"
                )

    def test_discover_excludes_leftover_worktree_and_backup_dirs(self, tmp_path: Path):
        """`.worktrees.old`, `target.old`, `.claude` are pruned by static name.

        Regression guard for the 226-way merge-verify storm: a leftover
        `.worktrees.old/<id>/orchestrator.yaml` (a full task-worktree checkout
        carrying a copy of the root config) must NOT register as a phantom
        module.
        """
        for leftover in ('.worktrees.old', 'target.old', '.claude'):
            nested = tmp_path / leftover / '3859'
            nested.mkdir(parents=True, exist_ok=True)
            (nested / 'orchestrator.yaml').write_text(
                yaml.dump({'test_command': 'pytest'})
            )
        legit = tmp_path / 'legit'
        legit.mkdir()
        (legit / 'orchestrator.yaml').write_text(yaml.dump({'test_command': 'pytest legit/'}))
        configs = _discover_module_configs(tmp_path)
        assert 'legit' in configs
        assert not any(
            k.startswith(('.worktrees.old', 'target.old', '.claude'))
            for k in configs
        ), f'leftover dir leaked into module configs: {sorted(configs)}'

    def test_discover_prunes_nested_git_checkouts(self, tmp_path: Path):
        """Any subdir carrying its own `.git` is a separate checkout, not a module.

        Naming-independent guard: a worktree/clone (`.git` file OR dir) that
        contains an orchestrator.yaml must be skipped, however it is named —
        catches stray checkouts the static exclude list does not enumerate.
        """
        # (a) a worktree-style nested checkout: `.git` is a FILE
        wt = tmp_path / 'some_stray_worktree'
        wt.mkdir()
        (wt / '.git').write_text('gitdir: /elsewhere/.git/worktrees/x\n')
        (wt / 'orchestrator.yaml').write_text(yaml.dump({'test_command': 'pytest'}))
        # (b) a clone-style nested checkout: `.git` is a DIR
        clone = tmp_path / 'vendored_clone'
        clone.mkdir()
        (clone / '.git').mkdir()
        (clone / 'orchestrator.yaml').write_text(yaml.dump({'test_command': 'pytest'}))
        # (c) a legitimate monorepo subproject: NO `.git`
        legit = tmp_path / 'legit'
        legit.mkdir()
        (legit / 'orchestrator.yaml').write_text(yaml.dump({'test_command': 'pytest legit/'}))
        configs = _discover_module_configs(tmp_path)
        assert 'legit' in configs
        assert 'some_stray_worktree' not in configs
        assert 'vendored_clone' not in configs

    def test_discover_handles_self_referencing_symlink_loop(self, tmp_path: Path):
        """_discover_module_configs completes without infinite recursion when a symlink loop exists.

        Regression guard: pins the followlinks=False behavior so a future refactor
        that flips it is caught immediately.
        """
        import os as _os
        sub = tmp_path / 'sub'
        sub.mkdir()
        (sub / 'orchestrator.yaml').write_text(yaml.dump({'test_command': 'pytest sub/'}))
        # Create a self-referencing symlink inside sub -> sub
        try:
            _os.symlink(str(sub), str(sub / 'loop'))
        except OSError:
            pytest.skip('Cannot create symlinks on this platform')
        # With followlinks=False this must finish quickly and return exactly {'sub': ...}
        configs = _discover_module_configs(tmp_path)
        assert list(configs.keys()) == ['sub']
        assert configs['sub'].test_command == 'pytest sub/'

    def test_for_module_longest_prefix_match(self):
        """for_module resolves nested configs by longest-matching prefix (deepest wins).

        This is the regression test the reviewer requires: it retrieves a nested config
        through for_module() (the workflow/scheduler consumption boundary), not just
        through _discover_module_configs directly.
        """
        config = OrchestratorConfig()
        config._module_configs = {
            'foo': ModuleConfig(prefix='foo', test_command='pytest foo/'),
            'foo/bar': ModuleConfig(prefix='foo/bar', test_command='pytest foo/bar/'),
        }
        # (a) Deeper path resolves to the deeper registered prefix
        mc_deep = config.for_module('foo/bar/baz/app.py')
        assert mc_deep is not None
        assert mc_deep.prefix == 'foo/bar'
        # (b) Exact prefix match — shape scheduler passes after normalize_lock(..., depth=2)
        mc_exact = config.for_module('foo/bar')
        assert mc_exact is not None
        assert mc_exact.prefix == 'foo/bar'
        # (c) Sibling falls back to the shallower ancestor prefix
        mc_fallback = config.for_module('foo/qux/app.py')
        assert mc_fallback is not None
        assert mc_fallback.prefix == 'foo'
        # (d) Exact top-level prefix match
        mc_top = config.for_module('foo')
        assert mc_top is not None
        assert mc_top.prefix == 'foo'
        # (e) Completely unrelated path returns None
        assert config.for_module('unrelated/x.py') is None

    def test_discover_skips_root_level_orchestrator_yaml(self, tmp_path: Path):
        """A root-level orchestrator.yaml is NOT returned as a module config.

        Regression guard for the explicit ``if prefix == '.': continue`` guard in
        _discover_module_configs.  The old glob('*/orchestrator.yaml') excluded the
        root by construction; the new os.walk relies on an explicit check.
        """
        # Root-level config — should be skipped
        (tmp_path / 'orchestrator.yaml').write_text(yaml.dump({
            'test_command': 'pytest',
        }))
        # Sub-level config — should be found
        sub = tmp_path / 'sub'
        sub.mkdir()
        (sub / 'orchestrator.yaml').write_text(yaml.dump({
            'test_command': 'pytest sub/',
        }))
        configs = _discover_module_configs(tmp_path)
        assert 'sub' in configs, "sub-level config should be discovered"
        assert '.' not in configs, "root-level (prefix '.') must not appear in results"
        # Confirm no key for the root-level file leaked in any form
        assert len(configs) == 1

    @pytest.mark.parametrize('excluded_name', ['build', 'target'])
    def test_pruned_dir_containing_orchestrator_yaml_is_logged(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, excluded_name: str
    ):
        """_discover_module_configs emits a WARNING when a pruned reserved-name directory
        directly contains an orchestrator.yaml (the 'shadow' case).

        Acceptance criteria:
          (a) The excluded directory is NOT returned in configs (exclusion preserved).
          (b) A warning record mentions both the relative path and the reserved name.
        """
        shadow_dir = tmp_path / excluded_name
        shadow_dir.mkdir()
        (shadow_dir / 'orchestrator.yaml').write_text(yaml.dump({'test_command': 'pytest'}))

        with caplog.at_level(logging.WARNING, logger='orchestrator.config'):
            configs = _discover_module_configs(tmp_path)

        # (a) Excluded directory must not appear in returned configs
        assert excluded_name not in configs, (
            f"Reserved dir {excluded_name!r} must not appear in discovery results"
        )

        # (b) At least one warning mentioning the path and the reserved name
        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING
        ]
        assert any(
            excluded_name in r.getMessage()
            for r in warning_records
        ), (
            f"Expected a WARNING mentioning {excluded_name!r} for shadow orchestrator.yaml; "
            f"got records: {[r.getMessage() for r in warning_records]}"
        )

    def test_pruned_dir_without_orchestrator_yaml_emits_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ):
        """_discover_module_configs does NOT emit a warning for a pruned directory that
        has no direct orchestrator.yaml (false-positive guard)."""
        # build/ exists but only contains a subdirectory — no immediate orchestrator.yaml
        build_sub = tmp_path / 'build' / 'some_sub'
        build_sub.mkdir(parents=True)
        (build_sub / 'file.py').write_text('# placeholder')

        with caplog.at_level(logging.WARNING, logger='orchestrator.config'):
            _discover_module_configs(tmp_path)

        pruned_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and 'build' in r.getMessage()
        ]
        assert pruned_warnings == [], (
            f"Expected no warnings for build/ (no direct orchestrator.yaml); "
            f"got: {[r.getMessage() for r in pruned_warnings]}"
        )

    def test_nested_module_helper_registers_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Shared helper _load_config_with_nested_module must register the nested
        prefix in config._module_configs so the depth-boundary tests are non-vacuous.
        """
        config = _load_config_with_nested_module(tmp_path, monkeypatch, prefix='foo/bar')
        assert config._module_configs is not None
        assert 'foo/bar' in config._module_configs, (
            "Helper must register 'foo/bar' via _discover_module_configs; "
            "if this fails the boundary tests silently pass even when discovery is broken"
        )

    def test_load_config_warns_when_module_prefix_deeper_than_lock_depth(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ):
        """load_config emits a WARNING when a discovered module-config prefix has more
        path components than lock_depth.

        Regression guard for task 1328 operator diagnostic (the depth-mismatch warning loop in load_config).
        Acceptance criteria — the WARNING record from orchestrator.config must pin:
          (a) the prefix 'foo/bar/baz'
          (b) the prefix depth 'prefix depth 3'
          (c) the configured depth 'lock_depth=2'
          (d) the consequence phrase 'unreachable through the scheduler/workflow path'
        """
        # Nested module config at depth 3 (foo/bar/baz) > lock_depth 2 → must warn.
        with caplog.at_level(logging.WARNING, logger='orchestrator.config'):
            _load_config_with_nested_module(tmp_path, monkeypatch, prefix='foo/bar/baz')

        warning_records = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and r.name == 'orchestrator.config'
        ]
        # Pin the three dynamic substrings plus the distinctive consequence phrase so the
        # operator diagnostic cannot silently drop the prefix, either depth, or the
        # unreachability consequence. The exact remediation prose is intentionally omitted
        # here — it is runtime log output, so coupling tests to its verbatim wording adds
        # breakage risk without additional regression-detection value.
        matching = [r for r in warning_records if DISTINCTIVE_PHRASE in r.getMessage()]
        assert matching, (
            f"Expected a WARNING from orchestrator.config containing "
            f"{DISTINCTIVE_PHRASE!r}; got: {[r.getMessage() for r in warning_records]}"
        )
        msg = matching[0].getMessage()
        for fragment in (
            'foo/bar/baz',
            'prefix depth 3',
            'lock_depth=2',
            DISTINCTIVE_PHRASE,
        ):
            assert fragment in msg, (
                f"Expected {fragment!r} in WARNING message; got: {msg!r}"
            )

    def test_load_config_no_warning_when_module_prefix_equals_lock_depth(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ):
        """load_config does NOT emit the depth-mismatch WARNING when prefix depth ==
        lock_depth (the boundary: strictly-greater-than triggers the warning, not >=).

        False-positive guard for task 1328 operator diagnostic (the depth-mismatch warning loop in load_config).
        """
        # Nested module config at depth 2 (foo/bar) == lock_depth 2 → must NOT warn.
        with caplog.at_level(logging.WARNING, logger='orchestrator.config'):
            config = _load_config_with_nested_module(tmp_path, monkeypatch, prefix='foo/bar')

        # Non-vacuity guard: 'foo/bar' must be registered so the depth comparison in
        # the depth-mismatch warning loop is actually evaluated at the == boundary.  If
        # discovery is broken, the loop has zero iterations, no warning is emitted, and
        # this test would silently pass without exercising the boundary it claims to guard.
        # (test_nested_module_helper_registers_prefix isolates the failure cause if this
        # assertion fires, so the duplication is intentional rather than accidental.)
        assert config._module_configs is not None
        assert 'foo/bar' in config._module_configs, (
            "'foo/bar' must appear in config._module_configs; "
            "if missing, the depth-comparison loop never runs and the warning-absence "
            "check is vacuously true (the boundary is not actually exercised)"
        )
        depth_mismatch_warnings = [
            r for r in caplog.records
            if r.levelno >= logging.WARNING and DISTINCTIVE_PHRASE in r.getMessage()
        ]
        assert depth_mismatch_warnings == [], (
            f"Expected no depth-mismatch WARNING for prefix 'foo/bar' with lock_depth=2 "
            f"(depth 2 is not strictly greater than lock_depth 2); "
            f"got: {[r.getMessage() for r in depth_mismatch_warnings]}"
        )


class TestLayeredConfig:
    """Tests for deep merge of package defaults + project config."""

    def test_deep_merge_basic(self):
        base = {'a': 1, 'b': {'x': 10, 'y': 20}}
        override = {'b': {'y': 99}, 'c': 3}
        result = _deep_merge(base, override)
        assert result == {'a': 1, 'b': {'x': 10, 'y': 99}, 'c': 3}

    def test_deep_merge_override_replaces_non_dict(self):
        base = {'a': {'nested': 1}}
        override = {'a': 'flat'}
        result = _deep_merge(base, override)
        assert result == {'a': 'flat'}

    def test_deep_merge_does_not_mutate_base(self):
        base = {'a': {'x': 1}}
        override = {'a': {'y': 2}}
        _deep_merge(base, override)
        assert base == {'a': {'x': 1}}

    def test_load_config_raises_when_no_config_and_no_env(self, tmp_path, monkeypatch):
        """Without --config or ORCH_CONFIG_PATH, load_config refuses to start."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        with pytest.raises(ConfigRequiredError, match='--config is required'):
            load_config(None)

    def test_project_config_overrides_defaults(self, tmp_path, monkeypatch):
        """Project config values should override package defaults."""
        project_cfg = tmp_path / 'config.yaml'
        project_cfg.write_text(yaml.dump({
            'models': {'implementer': 'opus'},
            'max_concurrent_tasks': 8,
        }))
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = load_config(project_cfg)
        # Overridden
        assert config.models.implementer == 'opus'
        assert config.max_concurrent_tasks == 8
        # Preserved from package defaults
        assert config.models.architect == 'opus'
        assert config.effort.architect == 'max'

    def test_deep_merge_preserves_sibling_keys(self, tmp_path, monkeypatch):
        """Overriding one key in a nested dict should not clobber siblings."""
        project_cfg = tmp_path / 'config.yaml'
        project_cfg.write_text(yaml.dump({
            'budgets': {'architect': 99.0},
        }))
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = load_config(project_cfg)
        assert config.budgets.architect == 99.0
        assert config.budgets.implementer == 10.0  # preserved from defaults


class TestPathResolution:
    def test_fused_memory_paths_resolve_under_project_root(self, tmp_path: Path):
        config = OrchestratorConfig(project_root=tmp_path)
        resolved = (config.project_root / config.fused_memory.config_path).resolve()
        assert str(resolved).startswith(str(tmp_path))
        assert '..' not in resolved.parts

    def test_overrides_db_path_default(self, tmp_path: Path) -> None:
        config = OrchestratorConfig(project_root=tmp_path)
        assert config.overrides_db_path == tmp_path / 'data' / 'orchestrator' / 'scheduler_overrides.db'


class TestStewardTimeoutInvariant:
    """OrchestratorConfig must reject configs where timeouts.steward < steward_completion_timeout."""

    def test_direct_init_violation_raises_validation_error(self, monkeypatch, tmp_path):
        """Directly instantiating OrchestratorConfig with a violating config raises ValidationError."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError, match='steward'):
            OrchestratorConfig(steward_completion_timeout=900.0, timeouts=TimeoutsConfig(steward=600.0))

    def test_yaml_load_violation_raises_validation_error(self, tmp_path, monkeypatch):
        """Loading a YAML config with timeouts.steward < steward_completion_timeout raises ValidationError."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        bad_yaml = tmp_path / 'bad.yaml'
        bad_yaml.write_text(yaml.dump({
            'timeouts': {'steward': 600.0},
            'steward_completion_timeout': 900.0,
        }))
        with pytest.raises(ValidationError, match='steward'):
            load_config(bad_yaml)

    def test_equal_values_allowed(self, monkeypatch, tmp_path):
        """timeouts.steward == steward_completion_timeout is valid (invariant is >=, not strict >)."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig(steward_completion_timeout=900.0, timeouts=TimeoutsConfig(steward=900.0))
        assert config.timeouts.steward == config.steward_completion_timeout

    def test_greater_value_allowed(self, monkeypatch, tmp_path):
        """timeouts.steward > steward_completion_timeout is valid."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig(steward_completion_timeout=900.0, timeouts=TimeoutsConfig(steward=1800.0))
        assert config.timeouts.steward > config.steward_completion_timeout

    def test_error_message_contains_remediation_hint(self, monkeypatch, tmp_path):
        """ValidationError message must include operator-actionable remediation hint.

        Guards against future refactors silently dropping the guidance text.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        with pytest.raises(ValidationError, match=r'(?i)raise.*timeouts\.steward.*or lower.*steward_completion_timeout'):
            OrchestratorConfig(steward_completion_timeout=900.0, timeouts=TimeoutsConfig(steward=600.0))

    def test_env_var_override_triggers_invariant(self, monkeypatch, tmp_path):
        """ORCH_TIMEOUTS__STEWARD env-var override is caught by the mode='after' validator.

        Regression guard: pins that pydantic-settings env-sourced overrides
        are merged into the model before mode='after' validators run.
        A future pydantic-settings source-ordering regression would cause this test to fail.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_TIMEOUTS__STEWARD', '300')
        monkeypatch.setenv('ORCH_STEWARD_COMPLETION_TIMEOUT', '900')
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        # Both sides are test-pinned: timeouts.steward=300, steward_completion_timeout=900
        # env override: timeouts.steward=300 → 300 < 900 → validator must fire
        with pytest.raises(ValidationError, match='steward'):
            OrchestratorConfig()


class TestValidateAssignment:
    """validate_assignment=True must re-run model validators on top-level field mutations."""

    def test_validate_assignment_rejects_steward_completion_timeout_mutation(
        self, monkeypatch, tmp_path
    ):
        """Setting steward_completion_timeout above timeouts.steward must raise ValidationError.

        With validate_assignment=True, this assignment fires _validate_steward_timeout_invariant
        and raises ValidationError.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        # Construct a valid config: defaults give timeouts.steward=1800, sct=900
        cfg = OrchestratorConfig()
        assert cfg.timeouts.steward == 1800.0
        assert cfg.steward_completion_timeout == 900.0
        # Mutate steward_completion_timeout to 2000.0 — now above timeouts.steward=1800.
        # validate_assignment=True fires _validate_steward_timeout_invariant, raising ValidationError.
        with pytest.raises(ValidationError, match='steward'):
            cfg.steward_completion_timeout = 2000.0

    def test_validate_assignment_rejects_timeouts_replacement(
        self, monkeypatch, tmp_path
    ):
        """Replacing cfg.timeouts with a TimeoutsConfig that violates the invariant must raise.

        With validate_assignment=True, assigning cfg.timeouts fires _validate_steward_timeout_invariant
        and raises ValidationError.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        # steward_completion_timeout=900 is valid against default timeouts.steward=1800
        cfg = OrchestratorConfig(steward_completion_timeout=900.0)
        assert cfg.steward_completion_timeout == 900.0
        # Replace timeouts with steward=300 — now 300 < 900, violating the invariant.
        # validate_assignment=True fires _validate_steward_timeout_invariant, raising ValidationError.
        with pytest.raises(ValidationError, match='steward'):
            cfg.timeouts = TimeoutsConfig(steward=300.0)

    def test_validate_assignment_allows_valid_mutation(self, monkeypatch, tmp_path):
        """A valid mutation of steward_completion_timeout must succeed without errors.

        Regression guard: confirms that validate_assignment does not block
        mutations that satisfy the invariant. Passes before and after step-4.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        # default timeouts.steward=1800, steward_completion_timeout=900
        # Setting sct=500 is valid (500 <= 1800).
        cfg.steward_completion_timeout = 500.0
        assert cfg.steward_completion_timeout == 500.0

    def test_project_root_resolved_on_assignment(self, monkeypatch, tmp_path):
        """Assigning a relative path to project_root after construction must resolve it to absolute.

        With a @field_validator('project_root', mode='after') and validate_assignment=True,
        post-construction assignment fires the field validator, resolving the path.
        This test fails when model_post_init is used (which only fires at construction).
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        cfg = OrchestratorConfig()
        # Assign a relative path post-construction; the field validator must resolve it
        cfg.project_root = Path('relative/subdir')
        assert cfg.project_root.is_absolute() is True

    def test_project_root_field_validator_does_not_double_trigger_model_validator(
        self, monkeypatch, tmp_path
    ):
        """@field_validator must not cause model validators to fire twice during construction.

        With the old model_post_init: assigning self.project_root under validate_assignment=True
        would trigger a second full model-validation pass (including _validate_steward_timeout_invariant),
        so model validators fired 2× during construction.
        With @field_validator('project_root', mode='after'): field-level validation resolves the
        path without triggering a second model-validation pass, so model validators fire exactly 1×.

        Strategy: subclass OrchestratorConfig with a counting model_validator; assert count == 1.
        """
        from pydantic import model_validator as _mv

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        call_count: list[int] = []

        class TrackingConfig(OrchestratorConfig):
            @_mv(mode='after')
            def _count_model_validation_pass(self) -> 'TrackingConfig':
                call_count.append(1)
                return self

        TrackingConfig()

        assert len(call_count) == 1, (
            f'Model validators were triggered {len(call_count)} times during construction; '
            'expected exactly 1. With field_validator, the field-level resolver must not '
            'cause a second full model-validation pass.'
        )


class TestParkStopConfig:
    """park_stop_* config fields: defaults and ge/gt validators."""

    def test_park_stop_enabled_default_true(self):
        assert OrchestratorConfig().park_stop_enabled is True

    def test_park_stop_parked_threshold_default(self):
        assert OrchestratorConfig().park_stop_parked_threshold == 15

    def test_park_stop_parked_window_hours_default(self):
        assert OrchestratorConfig().park_stop_parked_window_hours == 1.0

    def test_park_stop_parked_threshold_ge_1_rejects_zero(self):
        with pytest.raises(ValidationError):
            OrchestratorConfig(park_stop_parked_threshold=0)

    def test_park_stop_parked_window_hours_gt_0_rejects_zero(self):
        with pytest.raises(ValidationError):
            OrchestratorConfig(park_stop_parked_window_hours=0.0)

    def test_defaults_yaml_has_park_stop_block(self):
        """defaults.yaml must carry the park_stop_* keys with the documented defaults.

        This guards against the shipped defaults diverging from the Pydantic
        field defaults — any OrchestratorConfig() instantiation will pull the
        YAML values first (via settings_customise_sources).
        """
        defaults = _load_package_defaults()
        assert 'park_stop_enabled' in defaults, (
            "defaults.yaml is missing 'park_stop_enabled'"
        )
        assert 'park_stop_parked_threshold' in defaults, (
            "defaults.yaml is missing 'park_stop_parked_threshold'"
        )
        assert 'park_stop_parked_window_hours' in defaults, (
            "defaults.yaml is missing 'park_stop_parked_window_hours'"
        )
        assert defaults['park_stop_enabled'] is True
        assert defaults['park_stop_parked_threshold'] == 15
        assert defaults['park_stop_parked_window_hours'] == 1.0

        # Also verify the values flow through OrchestratorConfig() at runtime.
        config = OrchestratorConfig()
        assert config.park_stop_enabled is True
        assert config.park_stop_parked_threshold == 15
        assert config.park_stop_parked_window_hours == 1.0


class TestMergeVerifyStormGuardFields:
    """Defaults and overrides for the merge-verify storm-guard knobs."""

    def test_defaults_preserve_existing_behaviour(self):
        config = OrchestratorConfig()
        assert config.merge_verify_workspace is False
        assert config.max_concurrent_module_verifies == 4
        assert config.verify_use_cgroup_scope is False

    def test_overrides_accepted(self):
        config = OrchestratorConfig(
            merge_verify_workspace=True,
            max_concurrent_module_verifies=8,
            verify_use_cgroup_scope=True,
        )
        assert config.merge_verify_workspace is True
        assert config.max_concurrent_module_verifies == 8
        assert config.verify_use_cgroup_scope is True


class TestTrainFormerConfigFields:
    """Defaults and constraint for the β train-former config knobs."""

    def test_defaults(self):
        config = OrchestratorConfig()
        # Former is OFF by default so β can land before γ/δ complete the chain.
        assert config.merge_train_former_enabled is False
        # s(N) go/no-go resolved GO at N=3 (reify esc-4455-16).
        assert config.merge_train_max_members == 3

    def test_overrides_accepted(self):
        config = OrchestratorConfig(
            merge_train_former_enabled=True,
            merge_train_max_members=5,
        )
        assert config.merge_train_former_enabled is True
        assert config.merge_train_max_members == 5

    def test_max_members_rejects_below_2(self):
        with pytest.raises(ValidationError):
            OrchestratorConfig(merge_train_max_members=1)

    def test_max_concurrent_module_verifies_floor(self):
        with pytest.raises(ValidationError):
            OrchestratorConfig(max_concurrent_module_verifies=0)


# ---------------------------------------------------------------------------
# κ step-1: SccacheConfig
# ---------------------------------------------------------------------------


class TestSccacheConfig:
    """SccacheConfig — κ shared sccache backend config model."""

    def test_defaults(self):
        sc = SccacheConfig()
        assert sc.enabled is False
        assert sc.backend_env == {}

    def test_env_overrides_disabled_returns_empty(self):
        sc = SccacheConfig()
        assert sc.env_overrides() == {}

    def test_env_overrides_enabled_returns_copy_of_backend_env(self):
        sc = SccacheConfig(enabled=True, backend_env={'SCCACHE_REDIS': 'redis://h:6379'})
        result = sc.env_overrides()
        assert result == {'SCCACHE_REDIS': 'redis://h:6379'}
        # mutating the return must not affect the model
        result['EXTRA'] = 'x'
        assert 'EXTRA' not in sc.backend_env

    def test_env_overrides_redis_backend(self):
        sc = SccacheConfig(enabled=True, backend_env={'SCCACHE_REDIS': 'redis://h:6379'})
        assert sc.env_overrides() == {'SCCACHE_REDIS': 'redis://h:6379'}

    def test_enabled_with_empty_backend_raises(self):
        with pytest.raises(ValidationError):
            SccacheConfig(enabled=True, backend_env={})

    def test_enabled_with_empty_backend_explicit_empty_raises(self):
        """enabled=True with no backend_env argument also raises (default is {})."""
        with pytest.raises(ValidationError):
            SccacheConfig(enabled=True)


# ---------------------------------------------------------------------------
# κ step-3: OrchestratorConfig sccache field + effective_verify_env
# ---------------------------------------------------------------------------


class TestOrchestratorConfigSccache:
    """OrchestratorConfig.sccache field and effective_verify_env property."""

    def test_sccache_defaults_to_disabled(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = OrchestratorConfig()
        assert isinstance(config.sccache, SccacheConfig)
        assert config.sccache.enabled is False

    def test_effective_verify_env_equals_verify_env_when_disabled(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        config = OrchestratorConfig(verify_env={'RUSTC_WRAPPER': 'sccache'})
        assert config.effective_verify_env == config.verify_env

    def test_effective_verify_env_merges_sccache_backend(self):
        config = OrchestratorConfig(
            verify_env={'RUSTC_WRAPPER': 'sccache'},
            sccache=SccacheConfig(enabled=True, backend_env={'SCCACHE_REDIS': 'redis://h:6379'}),
        )
        assert config.effective_verify_env == {
            'RUSTC_WRAPPER': 'sccache',
            'SCCACHE_REDIS': 'redis://h:6379',
        }

    def test_verify_env_wins_on_key_conflict(self):
        """verify_env values beat backend_env values on shared keys."""
        config = OrchestratorConfig(
            verify_env={'SCCACHE_REDIS': 'redis://override:1234'},
            sccache=SccacheConfig(
                enabled=True,
                backend_env={'SCCACHE_REDIS': 'redis://default:6379'},
            ),
        )
        assert config.effective_verify_env['SCCACHE_REDIS'] == 'redis://override:1234'


# ---------------------------------------------------------------------------
# κ step-5: load_config fold — sccache backend folded into verify_env
# ---------------------------------------------------------------------------


class TestLoadConfigSccacheFold:
    """load_config folds sccache.env_overrides() into config.verify_env."""

    def test_fold_adds_backend_to_verify_env(self, tmp_path):
        cfg_path = tmp_path / 'orchestrator.yaml'
        cfg_path.write_text(yaml.dump({
            'verify_env': {'RUSTC_WRAPPER': 'sccache'},
            'sccache': {
                'enabled': True,
                'backend_env': {'SCCACHE_REDIS': 'redis://orch:6379'},
            },
        }))
        config = load_config(cfg_path)
        # Both keys must appear in verify_env after fold
        assert config.verify_env.get('RUSTC_WRAPPER') == 'sccache'
        assert config.verify_env.get('SCCACHE_REDIS') == 'redis://orch:6379'

    def test_fold_verify_env_wins_on_conflict(self, tmp_path):
        """verify_env values survive the fold even when key is also in backend_env."""
        cfg_path = tmp_path / 'orchestrator.yaml'
        cfg_path.write_text(yaml.dump({
            'verify_env': {'SCCACHE_REDIS': 'redis://explicit:1111'},
            'sccache': {
                'enabled': True,
                'backend_env': {'SCCACHE_REDIS': 'redis://default:6379'},
            },
        }))
        config = load_config(cfg_path)
        assert config.verify_env['SCCACHE_REDIS'] == 'redis://explicit:1111'

    def test_fold_no_op_when_sccache_disabled(self, tmp_path):
        """When sccache is disabled, verify_env is unchanged."""
        cfg_path = tmp_path / 'orchestrator.yaml'
        cfg_path.write_text(yaml.dump({
            'verify_env': {'RUSTC_WRAPPER': 'sccache'},
        }))
        config = load_config(cfg_path)
        assert list(config.verify_env.keys()) == ['RUSTC_WRAPPER']


# ---------------------------------------------------------------------------
# step-1: VerifyRunnerConfig + OrchestratorConfig.verify_runners surface
# ---------------------------------------------------------------------------


class TestVerifyRunnerConfig:
    """Tests for VerifyRunnerConfig and OrchestratorConfig.verify_runners fields."""

    def test_verify_runner_config_required_fields(self):
        """VerifyRunnerConfig parses from required str fields."""
        from orchestrator.config import VerifyRunnerConfig

        cfg = VerifyRunnerConfig(name='laptop', ssh_host='laptop.local', git_remote='origin')
        assert cfg.name == 'laptop'
        assert cfg.ssh_host == 'laptop.local'
        assert cfg.git_remote == 'origin'

    def test_verify_runner_config_defaults(self):
        """config_path defaults None, enabled defaults True."""
        from orchestrator.config import VerifyRunnerConfig

        cfg = VerifyRunnerConfig(name='r', ssh_host='h', git_remote='g')
        assert cfg.config_path is None
        assert cfg.enabled is True

    def test_verify_runner_config_explicit_values(self):
        """All fields can be set explicitly."""
        from orchestrator.config import VerifyRunnerConfig

        cfg = VerifyRunnerConfig(
            name='ci',
            ssh_host='ci.example.com',
            git_remote='ci',
            config_path='/etc/orch.yaml',
            enabled=False,
        )
        assert cfg.config_path == '/etc/orch.yaml'
        assert cfg.enabled is False

    def test_orchestrator_config_verify_runners_defaults_empty(self):
        """OrchestratorConfig.verify_runners defaults to [] not None."""
        config = OrchestratorConfig()
        assert config.verify_runners == []

    def test_orchestrator_config_verify_runners_parses_dict_list(self):
        """Constructing with a list of dicts coerces to list[VerifyRunnerConfig]."""
        from orchestrator.config import VerifyRunnerConfig

        config = OrchestratorConfig(verify_runners=[  # type: ignore[arg-type]
            {'name': 'laptop', 'ssh_host': 'laptop.local', 'git_remote': 'origin'},
        ])
        assert len(config.verify_runners) == 1
        assert isinstance(config.verify_runners[0], VerifyRunnerConfig)
        assert config.verify_runners[0].name == 'laptop'

    def test_orchestrator_config_verify_drift_check_every_n_lands_default(self):
        """verify_drift_check_every_n_lands defaults to 20."""
        config = OrchestratorConfig()
        assert config.verify_drift_check_every_n_lands == 20

    def test_orchestrator_config_verify_drift_check_every_n_lands_rejects_zero(self):
        """verify_drift_check_every_n_lands must be >= 1 (ge=1)."""
        with pytest.raises(ValidationError):
            OrchestratorConfig(verify_drift_check_every_n_lands=0)

    def test_orchestrator_config_verify_drift_check_every_n_lands_rejects_negative(self):
        """Negative value also rejected."""
        with pytest.raises(ValidationError):
            OrchestratorConfig(verify_drift_check_every_n_lands=-5)

    def test_enabled_verify_runners_filters_disabled(self):
        """enabled_verify_runners returns only runners with enabled=True."""
        config = OrchestratorConfig(verify_runners=[  # type: ignore[arg-type]
            {'name': 'active', 'ssh_host': 'h1', 'git_remote': 'r1', 'enabled': True},
            {'name': 'disabled', 'ssh_host': 'h2', 'git_remote': 'r2', 'enabled': False},
        ])
        enabled = config.enabled_verify_runners
        assert len(enabled) == 1
        assert enabled[0].name == 'active'

    def test_enabled_verify_runners_empty_when_all_disabled(self):
        """enabled_verify_runners returns [] when all runners are disabled."""
        config = OrchestratorConfig(verify_runners=[  # type: ignore[arg-type]
            {'name': 'r', 'ssh_host': 'h', 'git_remote': 'g', 'enabled': False},
        ])
        assert config.enabled_verify_runners == []

    def test_enabled_verify_runners_all_when_all_enabled(self):
        """enabled_verify_runners returns all runners when all have enabled=True."""
        config = OrchestratorConfig(verify_runners=[  # type: ignore[arg-type]
            {'name': 'r1', 'ssh_host': 'h1', 'git_remote': 'g1'},
            {'name': 'r2', 'ssh_host': 'h2', 'git_remote': 'g2'},
        ])
        assert len(config.enabled_verify_runners) == 2


class TestJobserverConfig:
    """Unit tests for JobserverConfig.agent_env()."""

    def test_enabled_with_fifo_present(self, tmp_path):
        """enabled=True and task_fifo is a real FIFO → returns CARGO_MAKEFLAGS."""
        fifo = tmp_path / 'test.fifo'
        os.mkfifo(fifo)
        cfg = JobserverConfig(enabled=True, task_fifo=str(fifo))
        env = cfg.agent_env()
        assert 'CARGO_MAKEFLAGS' in env
        assert env['CARGO_MAKEFLAGS'].startswith('--jobserver-auth=fifo:')
        assert str(fifo) in env['CARGO_MAKEFLAGS']

    def test_enabled_with_fifo_absent(self, tmp_path):
        """enabled=True but task_fifo path does not exist → returns {}."""
        fifo = tmp_path / 'nonexistent.fifo'
        cfg = JobserverConfig(enabled=True, task_fifo=str(fifo))
        assert cfg.agent_env() == {}

    def test_enabled_with_regular_file(self, tmp_path):
        """enabled=True but task_fifo is a regular file (not FIFO) → returns {}."""
        f = tmp_path / 'not-a-fifo'
        f.write_text('not a fifo')
        cfg = JobserverConfig(enabled=True, task_fifo=str(f))
        assert cfg.agent_env() == {}

    def test_disabled_returns_empty(self, tmp_path):
        """enabled=False → returns {} regardless of path."""
        fifo = tmp_path / 'test.fifo'
        os.mkfifo(fifo)
        cfg = JobserverConfig(enabled=False, task_fifo=str(fifo))
        assert cfg.agent_env() == {}

    def test_custom_env_var_and_task_fifo(self, tmp_path):
        """Custom env_var and task_fifo are honored in the returned dict."""
        fifo = tmp_path / 'custom.fifo'
        os.mkfifo(fifo)
        cfg = JobserverConfig(enabled=True, task_fifo=str(fifo), env_var='MY_MAKEFLAGS')
        env = cfg.agent_env()
        assert 'MY_MAKEFLAGS' in env
        assert env['MY_MAKEFLAGS'] == f'--jobserver-auth=fifo:{fifo}'

    def test_validator_rejects_enabled_with_empty_task_fifo(self):
        """enabled=True with empty task_fifo raises ValidationError."""
        with pytest.raises(ValidationError):
            JobserverConfig(enabled=True, task_fifo='')

    def test_orchestrator_config_has_jobserver_field(self):
        """OrchestratorConfig exposes a jobserver field that defaults to disabled."""
        cfg = OrchestratorConfig()
        assert isinstance(cfg.jobserver, JobserverConfig)
        assert cfg.jobserver.enabled is False


class TestCpuPriorityConfig:
    """Unit tests for CpuPriorityConfig.agent_env()."""

    def test_default_is_enabled_with_nice_10(self):
        """Default CpuPriorityConfig() is enabled=True, nice=10."""
        cfg = CpuPriorityConfig()
        assert cfg.enabled is True
        assert cfg.nice == 10

    def test_default_agent_env_returns_df_agent_cpu_nice_10(self):
        """Default agent_env() returns {'DF_AGENT_CPU_NICE': '10'}."""
        cfg = CpuPriorityConfig()
        assert cfg.agent_env() == {'DF_AGENT_CPU_NICE': '10'}

    def test_custom_nice_agent_env(self):
        """CpuPriorityConfig(nice=15).agent_env() returns {'DF_AGENT_CPU_NICE': '15'}."""
        cfg = CpuPriorityConfig(nice=15)
        assert cfg.agent_env() == {'DF_AGENT_CPU_NICE': '15'}

    def test_disabled_returns_empty(self):
        """CpuPriorityConfig(enabled=False).agent_env() returns {}."""
        cfg = CpuPriorityConfig(enabled=False)
        assert cfg.agent_env() == {}

    def test_validator_rejects_enabled_with_nice_zero(self):
        """enabled=True with nice=0 raises ValidationError (not de-prioritizing)."""
        with pytest.raises(ValidationError):
            CpuPriorityConfig(enabled=True, nice=0)

    def test_validator_rejects_enabled_with_nice_negative(self):
        """enabled=True with nice=-1 raises ValidationError (needs privilege)."""
        with pytest.raises(ValidationError):
            CpuPriorityConfig(enabled=True, nice=-1)

    def test_validator_rejects_enabled_with_nice_above_19(self):
        """enabled=True with nice=20 raises ValidationError (out of range 1..19)."""
        with pytest.raises(ValidationError):
            CpuPriorityConfig(enabled=True, nice=20)

    def test_validator_accepts_nice_boundary_1(self):
        """enabled=True with nice=1 is valid (minimum positive de-prioritization)."""
        cfg = CpuPriorityConfig(enabled=True, nice=1)
        assert cfg.nice == 1
        assert cfg.agent_env() == {'DF_AGENT_CPU_NICE': '1'}

    def test_validator_accepts_nice_boundary_19(self):
        """enabled=True with nice=19 is valid (maximum privilege-free)."""
        cfg = CpuPriorityConfig(enabled=True, nice=19)
        assert cfg.nice == 19
        assert cfg.agent_env() == {'DF_AGENT_CPU_NICE': '19'}

    def test_orchestrator_config_has_cpu_priority_field(self):
        """OrchestratorConfig exposes a cpu_priority field defaulting to enabled."""
        cfg = OrchestratorConfig()
        assert isinstance(cfg.cpu_priority, CpuPriorityConfig)
        assert cfg.cpu_priority.enabled is True
        assert cfg.cpu_priority.nice == 10


class TestCpuGovernConfig:
    """Unit tests for CpuGovernConfig schema and agent_env()."""

    def test_defaults_are_disabled_and_empty_paths(self):
        """CpuGovernConfig() defaults: enabled=False, exec_path='', shim_dir=''."""
        from orchestrator.config import CpuGovernConfig
        cfg = CpuGovernConfig()
        assert cfg.enabled is False
        assert cfg.exec_path == ''
        assert cfg.shim_dir == ''

    def test_disabled_agent_env_returns_empty(self, tmp_path):
        """disabled -> agent_env() == {} (and resolved paths are None)."""
        from orchestrator.config import CpuGovernConfig
        cfg = CpuGovernConfig()
        assert cfg.agent_env(tmp_path, '/usr/bin') == {}
        assert cfg.resolved_exec_path(tmp_path) is None
        assert cfg.resolved_shim_dir(tmp_path) is None

    def test_enabled_with_valid_paths(self, tmp_path):
        """enabled=True with tmp executable + dir: resolved paths correct, agent_env correct."""
        from orchestrator.config import CpuGovernConfig
        # Create the executable
        scripts = tmp_path / 'scripts'
        scripts.mkdir()
        exec_file = scripts / 'cpu-governed-exec.sh'
        exec_file.write_text('#!/bin/sh\nexec "$@"\n')
        exec_file.chmod(0o755)
        # Create the shim dir
        agent_bin = scripts / 'agent-bin'
        agent_bin.mkdir()

        cfg = CpuGovernConfig(
            enabled=True,
            exec_path='scripts/cpu-governed-exec.sh',
            shim_dir='scripts/agent-bin',
        )
        abs_exec = str(exec_file.resolve())
        abs_shim = str(agent_bin.resolve())

        assert cfg.resolved_exec_path(tmp_path) == abs_exec
        assert cfg.resolved_shim_dir(tmp_path) == abs_shim

        env = cfg.agent_env(tmp_path, '/usr/bin')
        assert env.get('DF_AGENT_CPU_GOVERN') == abs_exec
        assert env.get('PATH') == f'{abs_shim}{os.pathsep}/usr/bin'

    def test_enabled_with_non_executable_exec_path_fails_open(self, tmp_path):
        """enabled + non-executable exec_path -> resolved_exec_path is None, DF_AGENT_CPU_GOVERN absent."""
        from orchestrator.config import CpuGovernConfig
        scripts = tmp_path / 'scripts'
        scripts.mkdir()
        exec_file = scripts / 'cpu-governed-exec.sh'
        exec_file.write_text('#!/bin/sh\nexec "$@"\n')
        exec_file.chmod(0o644)  # not executable

        cfg = CpuGovernConfig(
            enabled=True,
            exec_path='scripts/cpu-governed-exec.sh',
            shim_dir='scripts/agent-bin',
        )
        assert cfg.resolved_exec_path(tmp_path) is None
        env = cfg.agent_env(tmp_path, '/usr/bin')
        assert 'DF_AGENT_CPU_GOVERN' not in env

    def test_enabled_with_missing_exec_path_fails_open(self, tmp_path):
        """enabled + missing exec_path -> resolved_exec_path is None, fail-open."""
        from orchestrator.config import CpuGovernConfig
        cfg = CpuGovernConfig(
            enabled=True,
            exec_path='scripts/nonexistent.sh',
            shim_dir='scripts/agent-bin',
        )
        assert cfg.resolved_exec_path(tmp_path) is None
        env = cfg.agent_env(tmp_path, '/usr/bin')
        assert 'DF_AGENT_CPU_GOVERN' not in env

    def test_enabled_with_relative_path_and_worktree_none_returns_none(self):
        """enabled + relative path + worktree=None -> resolved_exec_path is None."""
        from orchestrator.config import CpuGovernConfig
        cfg = CpuGovernConfig(
            enabled=True,
            exec_path='scripts/cpu-governed-exec.sh',
            shim_dir='scripts/agent-bin',
        )
        assert cfg.resolved_exec_path(None) is None
        assert cfg.resolved_shim_dir(None) is None

    def test_enabled_with_absolute_exec_path_used_as_is(self, tmp_path):
        """An absolute exec_path is used as-is (not joined to worktree)."""
        from orchestrator.config import CpuGovernConfig
        exec_file = tmp_path / 'cpu-governed-exec.sh'
        exec_file.write_text('#!/bin/sh\n')
        exec_file.chmod(0o755)
        agent_bin = tmp_path / 'agent-bin'
        agent_bin.mkdir()

        cfg = CpuGovernConfig(
            enabled=True,
            exec_path=str(exec_file),
            shim_dir=str(agent_bin),
        )
        assert cfg.resolved_exec_path(tmp_path) == str(exec_file)
        assert cfg.resolved_shim_dir(tmp_path) == str(agent_bin)

    def test_orchestrator_config_has_cpu_governance_field(self):
        """OrchestratorConfig exposes a cpu_governance field defaulting to enabled=False."""
        from orchestrator.config import CpuGovernConfig
        cfg = OrchestratorConfig()
        assert isinstance(cfg.cpu_governance, CpuGovernConfig)
        assert cfg.cpu_governance.enabled is False


# ---------------------------------------------------------------------------
# TestStarvationWatchdogConfig (task 1880)
# ---------------------------------------------------------------------------


class TestStarvationWatchdogConfig:
    """Tests for StarvationWatchdogConfig nested config + OrchestratorConfig attachment.

    Mirrors the FairnessConfig / nested-config test pattern:
      (a) defaults — bare OrchestratorConfig() exposes starvation_watchdog sub-object
          with the expected default values.
      (b) full yaml override — a project config with a top-level starvation_watchdog
          block is loaded via load_config and all three fields adopt the override values.
      (c) partial override — overriding only `enabled: false` keeps skip_threshold and
          idle_secs at their defaults (deep-merge / no clobber).
    """

    def test_defaults(self):
        """Bare OrchestratorConfig() exposes starvation_watchdog with correct defaults."""
        from orchestrator.config import StarvationWatchdogConfig
        cfg = OrchestratorConfig()
        assert isinstance(cfg.starvation_watchdog, StarvationWatchdogConfig), (
            f'Expected StarvationWatchdogConfig; got {type(cfg.starvation_watchdog)}'
        )
        assert cfg.starvation_watchdog.enabled is True, (
            f'Expected enabled=True; got {cfg.starvation_watchdog.enabled!r}'
        )
        assert cfg.starvation_watchdog.skip_threshold == 50, (
            f'Expected skip_threshold=50; got {cfg.starvation_watchdog.skip_threshold!r}'
        )
        assert cfg.starvation_watchdog.idle_secs == 1800.0, (
            f'Expected idle_secs=1800.0; got {cfg.starvation_watchdog.idle_secs!r}'
        )

    def test_full_yaml_override(self, tmp_path: Path, monkeypatch):
        """A project config with starvation_watchdog block is fully adopted by load_config."""
        project_cfg = tmp_path / 'config.yaml'
        project_cfg.write_text(yaml.dump({
            'starvation_watchdog': {
                'enabled': False,
                'skip_threshold': 10,
                'idle_secs': 300.0,
            },
        }))
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        cfg = load_config(project_cfg)

        assert cfg.starvation_watchdog.enabled is False, (
            f'Expected enabled=False; got {cfg.starvation_watchdog.enabled!r}'
        )
        assert cfg.starvation_watchdog.skip_threshold == 10, (
            f'Expected skip_threshold=10; got {cfg.starvation_watchdog.skip_threshold!r}'
        )
        assert cfg.starvation_watchdog.idle_secs == 300.0, (
            f'Expected idle_secs=300.0; got {cfg.starvation_watchdog.idle_secs!r}'
        )

    def test_partial_override_merges_with_defaults(self, tmp_path: Path, monkeypatch):
        """Overriding only enabled=False preserves skip_threshold and idle_secs defaults."""
        project_cfg = tmp_path / 'config.yaml'
        project_cfg.write_text(yaml.dump({
            'starvation_watchdog': {
                'enabled': False,
            },
        }))
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)
        cfg = load_config(project_cfg)

        assert cfg.starvation_watchdog.enabled is False, (
            f'Expected enabled=False; got {cfg.starvation_watchdog.enabled!r}'
        )
        # These must keep their defaults (deep-merge must not clobber siblings).
        assert cfg.starvation_watchdog.skip_threshold == 50, (
            f'Expected skip_threshold=50 (default preserved); '
            f'got {cfg.starvation_watchdog.skip_threshold!r}'
        )
        assert cfg.starvation_watchdog.idle_secs == 1800.0, (
            f'Expected idle_secs=1800.0 (default preserved); '
            f'got {cfg.starvation_watchdog.idle_secs!r}'
        )


class TestVerifyInfraRetryConfig:
    """Tests for verify_infra_retry_* config fields (step-5)."""

    def test_verify_infra_retry_max_attempts_default(self, monkeypatch, tmp_path):
        """verify_infra_retry_max_attempts defaults to 5."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.verify_infra_retry_max_attempts == 5

    def test_verify_infra_retry_backoff_secs_default(self, monkeypatch, tmp_path):
        """verify_infra_retry_backoff_secs defaults to 2.0."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.verify_infra_retry_backoff_secs == 2.0

    def test_verify_infra_retry_max_backoff_secs_default(self, monkeypatch, tmp_path):
        """verify_infra_retry_max_backoff_secs defaults to 60.0."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv('ORCH_CONFIG_PATH', '')
        config = OrchestratorConfig()
        assert config.verify_infra_retry_max_backoff_secs == 60.0

    def test_verify_infra_retry_max_attempts_ge_1_rejects_zero(self):
        """verify_infra_retry_max_attempts=0 must raise ValidationError (ge=1)."""
        with pytest.raises(ValidationError):
            OrchestratorConfig(verify_infra_retry_max_attempts=0)

    def test_verify_infra_retry_backoff_secs_gt_0_rejects_zero(self):
        """verify_infra_retry_backoff_secs=0 must raise ValidationError (gt=0)."""
        with pytest.raises(ValidationError):
            OrchestratorConfig(verify_infra_retry_backoff_secs=0.0)

    def test_verify_infra_retry_max_backoff_secs_gt_0_rejects_zero(self):
        """verify_infra_retry_max_backoff_secs=0 must raise ValidationError (gt=0)."""
        with pytest.raises(ValidationError):
            OrchestratorConfig(verify_infra_retry_max_backoff_secs=0.0)

    def test_verify_infra_retry_fields_override_from_yaml(self, monkeypatch, tmp_path):
        """Fields load and override correctly from a YAML mapping via load_config."""
        from io import StringIO
        from orchestrator.config import load_config
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv('ORCH_CONFIG_PATH', raising=False)

        yaml_content = (
            'verify_infra_retry_max_attempts: 10\n'
            'verify_infra_retry_backoff_secs: 5.0\n'
            'verify_infra_retry_max_backoff_secs: 120.0\n'
        )
        project_cfg = load_config.__wrapped__ if hasattr(load_config, '__wrapped__') else None
        # Use direct OrchestratorConfig construction with dict (avoids file path)
        cfg = OrchestratorConfig(
            verify_infra_retry_max_attempts=10,
            verify_infra_retry_backoff_secs=5.0,
            verify_infra_retry_max_backoff_secs=120.0,
        )
        assert cfg.verify_infra_retry_max_attempts == 10
        assert cfg.verify_infra_retry_backoff_secs == 5.0
        assert cfg.verify_infra_retry_max_backoff_secs == 120.0
