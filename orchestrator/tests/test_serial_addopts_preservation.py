"""The serial-recovery re-run keeps the governing addopts it used to blank (task 5079).

``verify.py::_serial_pytest_str`` forces a pytest command serial with
``-p no:xdist -o addopts=<value>``. ``-o addopts=`` REPLACES the ini addopts
wholesale, so a bare one also drops the import mode and the marker filter:
the recovery run then collects and selects a different test set from the run
it recovers. ``cockpit/tests/test_root_config_smoke_deselection.py``'s
``_addopts_without_marker_filter`` records the same hazard from the other
side, and its derive-don't-restate rule governs this module too: every
expected token is read from the real ``pyproject.toml`` files and every
command from the real module configs, never restated.

Two halves. The DRIFT GUARD runs against the real repo: the resolver agrees
with pytest's own config discovery for every module command, scoped and
unscoped, and the rewrite drops nothing but xdist. The WIRING tests drive
each serial-recovery call site against a real tmp tree and check it passes
the directory its command runs in, so none is left on the blanking default.
"""

from __future__ import annotations

import asyncio
import shlex
import tomllib
from functools import cache
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from _pytest.config.findpaths import determine_setup
from _verify_config_corpus import DF_CONFIG_PATH, REPO_ROOT, load_config_scalar
from test_verify_cmd import xdist_offenders
from test_verify_env_transient import _XDIST_VANISHED_OUTPUT
from test_verify_main_tip_sweep import MAIN_SHA, _make_confirm_fake_run, _make_git_ops
from test_verify_merge_flake_suppression import _failing, _make_config, _result

from orchestrator import verify, verify_plan
from orchestrator.config import ModuleConfig, OrchestratorConfig, _discover_module_configs
from orchestrator.verify import (
    VerifyResult,
    _scope_to_keyword,
    _serial_pytest_str,
    _worktree_reader,
)
from orchestrator.verify_cmd import parse_config_command

#: Modules whose governing config carries a marker filter or an import mode,
#: so a resolver silently answering ``()`` cannot pass the checks vacuously.
_MODULES_WITH_NON_XDIST_ADDOPTS = ('shared', 'orchestrator', 'fused-memory', 'cockpit', 'scripts')


@cache
def _module_test_commands() -> dict[str, str]:
    """``{prefix: test_command}`` for every discovered module config with one."""
    return {
        prefix: module_config.test_command
        for prefix, module_config in _discover_module_configs(REPO_ROOT).items()
        if module_config.test_command
    }


def _one_real_test_file(command: str) -> str:
    """A tracked test file under *command*'s first target, worktree-root-relative."""
    parsed = parse_config_command(command)
    target_dir = REPO_ROOT / (parsed.cwd_rel or '') / parsed.targets[0]
    first = sorted(target_dir.rglob('test_*.py'))[0]
    return first.relative_to(REPO_ROOT).as_posix()


def _command_shapes() -> list[tuple[str, str]]:
    """``[(label, command)]``: each module's command and its SCOPED, cwd-stripped variant."""
    shapes = []
    for prefix, command in _module_test_commands().items():
        shapes.append((prefix, command))
        scoped = _scope_to_keyword(command, 'pytest', [_one_real_test_file(command)])
        assert scoped is not None
        shapes.append((f'{prefix} (scoped)', scoped))
    return shapes


def _pytest_own_inipath(command: str) -> str | None:
    """The config pytest ITSELF picks for *command*, worktree-root-relative — the oracle."""
    parsed = parse_config_command(command)
    invocation_dir = REPO_ROOT / (parsed.cwd_rel or '')
    _, inipath, _, _ = determine_setup(
        inifile=None,
        override_ini=None,
        args=[str(invocation_dir / target.split('::')[0]) for target in parsed.targets],
        rootdir_cmd_arg=None,
        invocation_dir=invocation_dir,
    )
    return None if inipath is None else inipath.relative_to(REPO_ROOT).as_posix()


def _real_addopts(pyproject: str) -> list[str]:
    """The REAL ``addopts`` of *pyproject*, read with tomllib + shlex rather than our reader."""
    data = tomllib.loads((REPO_ROOT / pyproject).read_text(encoding='utf-8'))
    addopts = data['tool']['pytest']['ini_options'].get('addopts', '')
    return shlex.split(addopts) if isinstance(addopts, str) else list(addopts)


def _non_xdist_requirements(addopts: list[str]) -> list[tuple[str, ...]]:
    """The marker filter and import mode of *addopts*, each as the argv run it must survive as."""
    required: list[tuple[str, ...]] = []
    for index, token in enumerate(addopts):
        if token in ('-m', '--import-mode') and index + 1 < len(addopts):
            required.append((token, addopts[index + 1]))
        elif token.startswith('--import-mode=') or (token.startswith('-m') and len(token) > 2):
            required.append((token,))
    return required


def _rendered_addopts_value(rendered: str) -> str:
    """The LAST ``-o addopts=<value>`` value in *rendered* — the one pytest applies."""
    tokens = shlex.split(rendered)
    values = [
        value.removeprefix('addopts=')
        for flag, value in zip(tokens, tokens[1:], strict=False)
        if flag == '-o' and value.startswith('addopts=')
    ]
    assert values, f'no -o addopts= pair in {rendered!r}'
    return values[-1]


def _contains_run(tokens: list[str], run: tuple[str, ...]) -> bool:
    return any(tuple(tokens[i : i + len(run)]) == run for i in range(len(tokens)))


def _recover(command: str) -> str:
    rendered = _serial_pytest_str(command, invocation_dir=REPO_ROOT)
    assert rendered is not None
    return rendered


class TestSerialRecoveryPreservesTheGoverningAddopts:
    def test_the_resolver_picks_the_config_pytest_itself_picks(self):
        reader = _worktree_reader(REPO_ROOT)
        mismatches = {
            label: (verify_plan.pytest_config_path_for_command(command, reader), oracle)
            for label, command in _command_shapes()
            if verify_plan.pytest_config_path_for_command(command, reader)
            != (oracle := _pytest_own_inipath(command))
        }
        assert not mismatches, f'(ours, pytest determine_setup) disagree: {mismatches}'

    def test_no_marker_filter_or_import_mode_is_dropped(self):
        dropped = {}
        for label, command in _command_shapes():
            inipath = _pytest_own_inipath(command)
            assert inipath is not None, label
            value_tokens = shlex.split(_rendered_addopts_value(_recover(command)))
            missing = [
                run for run in _non_xdist_requirements(_real_addopts(inipath))
                if not _contains_run(value_tokens, run)
            ]
            if missing:
                dropped[label] = missing
        assert not dropped, f'serial recovery dropped governing addopts: {dropped}'

    def test_no_xdist_option_survives_on_argv_or_in_the_addopts(self):
        survivors = {}
        for label, command in _command_shapes():
            rendered = _recover(command)
            found = xdist_offenders(shlex.split(rendered)) + xdist_offenders(
                shlex.split(_rendered_addopts_value(rendered)),
            )
            if found:
                survivors[label] = found
        assert not survivors, (
            f'`-p no:xdist` unregisters every xdist option, so these are a hard '
            f'`unrecognized arguments`: {survivors}'
        )

    def test_the_scripts_leg_carries_the_root_import_mode_and_marker_filter(self):
        """esc-4377-3: the blank collected the scripts union with import-file-mismatch errors."""
        root_addopts = _real_addopts('pyproject.toml')
        import_mode = next(token for token in root_addopts if token.startswith('--import-mode'))
        marker_filter = root_addopts[root_addopts.index('-m') + 1]
        value_tokens = shlex.split(
            _rendered_addopts_value(_recover(_module_test_commands()['scripts'])),
        )
        assert import_mode in value_tokens
        assert _contains_run(value_tokens, ('-m', marker_filter))

    def test_modules_with_non_xdist_addopts_re_supply_a_non_empty_value(self):
        commands = _module_test_commands()
        empty = [
            prefix for prefix in _MODULES_WITH_NON_XDIST_ADDOPTS
            if not _rendered_addopts_value(_recover(commands[prefix]))
        ]
        assert not empty, f'these resolved to the historical blank: {empty}'

    def test_the_fleet_raw_chain_stays_the_documented_blank(self):
        """Each ``cd X &&`` clause reads its own config, but one suffix serves them all."""
        fleet = load_config_scalar(DF_CONFIG_PATH, 'test_command')
        assert parse_config_command(fleet).raw is not None
        assert verify_plan.governing_addopts(fleet, _worktree_reader(REPO_ROOT)) == ()
        tokens = shlex.split(_recover(fleet))
        assert {
            value for flag, value in zip(tokens, tokens[1:], strict=False) if flag == '-o'
        } == {'addopts='}


# ---------------------------------------------------------------------------
# WIRING: each serial-recovery call site passes the directory its command runs in.
# ---------------------------------------------------------------------------

_ROOT_SHAPED_PYPROJECT = (
    '[tool.pytest.ini_options]\n'
    'addopts = "--import-mode=importlib -m \'not smoke and not integration and not warm_lane_bash\'"\n'
)
_SUB_SHAPED_PYPROJECT = (
    '[tool.pytest.ini_options]\n'
    'addopts = "-n auto --dist loadgroup --max-worker-restart=0 -m \'not warm_lane_bash\'"\n'
)
_SUB_ADDOPTS = "-m 'not warm_lane_bash'"
_ROOT_ADDOPTS = "--import-mode=importlib -m 'not smoke and not integration and not warm_lane_bash'"
_SUB_NODE_ID = 'sub/tests/test_x.py::test_t'

#: A worktree with a root-shaped root config and an orchestrator-shaped ``sub``
#: module (discoverable through its own ``orchestrator.yaml``).
_TREE_LAYOUT = {
    'pyproject.toml': _ROOT_SHAPED_PYPROJECT,
    'sub/pyproject.toml': _SUB_SHAPED_PYPROJECT,
    'sub/orchestrator.yaml': 'test_command: "uv run --directory sub pytest tests/ --tb=short -q"\n',
    'sub/tests/test_x.py': 'def test_t():\n    pass\n',
}


def _write_tree(root: Path) -> Path:
    for relpath, content in _TREE_LAYOUT.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
    return root


def _sub_module_config(test_command: str = 'uv run --directory sub pytest tests/ --tb=short -q'):
    return ModuleConfig(prefix='sub', test_command=test_command)


def _assert_carries_sub_config(command: str) -> None:
    assert _rendered_addopts_value(command) == _SUB_ADDOPTS, command
    tokens = shlex.split(command)
    assert not xdist_offenders(tokens), command
    assert sum(token.startswith('addopts=') for token in tokens) == 1, command


def _failing_on_the_sub_node() -> VerifyResult:
    return _failing(f'FAILED {_SUB_NODE_ID}\n1 failed, 10 passed in 1.00s\n')


@pytest.mark.asyncio
class TestEnvTransientRecoveryWiring:
    """``run_verification``'s ENV_TRANSIENT retry resolves the config of the command it re-runs."""

    @staticmethod
    async def _recovery_command(worktree: Path, config: OrchestratorConfig, module_config) -> str:
        invoked: list[str] = []

        async def fake_cmd(cmd, cwd, timeout, env=None, log_path=None, **kwargs):
            invoked.append(cmd)
            if 'pytest' in cmd and 'no:xdist' not in cmd:
                return 4, _XDIST_VANISHED_OUTPUT, False
            return 0, '', False

        with patch('orchestrator.verify._run_cmd', side_effect=fake_cmd):
            result = await verify.run_verification(worktree, config, module_config)

        assert result.passed is True, result
        recoveries = [c for c in invoked if 'pytest' in c and 'no:xdist' in c]
        assert len(recoveries) == 1, invoked
        return recoveries[0]

    @staticmethod
    def _config(worktree: Path) -> OrchestratorConfig:
        return OrchestratorConfig(
            project_root=worktree,
            test_command='uv run pytest tests/',
            lint_command='echo lint',
            type_check_command='echo type',
        )

    async def test_a_scoped_cwd_stripped_module_run_recovers_with_its_own_config(self, tmp_path):
        """The FILE_SCOPED shape; the root's three-clause ``-m`` would be strictly worse than blank."""
        worktree = _write_tree(tmp_path)
        recovery = await self._recovery_command(
            worktree, self._config(worktree),
            _sub_module_config('uv run --project sub pytest sub/tests/test_x.py --tb=short -q'),
        )
        _assert_carries_sub_config(recovery)

    async def test_a_run_without_a_module_config_recovers_with_the_root_config(self, tmp_path):
        worktree = _write_tree(tmp_path)
        recovery = await self._recovery_command(worktree, self._config(worktree), None)
        assert _rendered_addopts_value(recovery) == _ROOT_ADDOPTS


class TestScopedConfirmSitesWiring:
    """The sweep prefilter, main-tip confirm and flake engine each resolve the scoped command's config."""

    def test_the_sweep_prefilter(self, tmp_path):
        worktree = _write_tree(tmp_path)
        rv = AsyncMock(return_value=_result(True))
        with patch.object(verify, 'run_verification', rv):
            asyncio.run(verify._sweep_failure_reproduces_in_isolation(
                worktree, _make_config(worktree), _failing_on_the_sub_node(),
            ))
        rv.assert_awaited_once()
        _assert_carries_sub_config(rv.call_args.args[2].test_command)

    def test_the_main_tip_confirm(self, tmp_path):
        config = _make_config(tmp_path)
        rv = AsyncMock(return_value=_result(True))
        with (
            patch('orchestrator.git_ops._run', side_effect=_make_confirm_fake_run([], _TREE_LAYOUT)),
            patch.object(verify, 'run_verification', rv),
        ):
            asyncio.run(verify.confirm_main_tip_failure_is_real(
                config, _make_git_ops(tmp_path), _failing_on_the_sub_node(), main_sha=MAIN_SHA,
            ))
        rv.assert_awaited_once()
        _assert_carries_sub_config(rv.call_args.args[2].test_command)

    def test_the_merge_flake_engine(self, tmp_path):
        worktree = _write_tree(tmp_path)
        rv = AsyncMock(return_value=_result(True))
        with patch.object(verify, 'run_verification', rv):
            asyncio.run(verify.confirm_merge_verify_flake_suppressible(
                _make_config(worktree), _failing_on_the_sub_node(),
                worktree=worktree, module_configs=[_sub_module_config()],
            ))
        rv.assert_awaited_once()
        command = rv.call_args.args[2].test_command
        _assert_carries_sub_config(command)
        tokens = shlex.split(command)
        addopts_at = next(i for i, token in enumerate(tokens) if token.startswith('addopts='))
        assert tokens[addopts_at + 1] == '--timeout', command
