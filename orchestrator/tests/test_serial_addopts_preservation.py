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

This module is the DRIFT GUARD against the real repo — the resolver agrees
with pytest's own config discovery for every module command, scoped and
unscoped, and the rewrite drops nothing but xdist.
"""

from __future__ import annotations

import shlex
import tomllib
from functools import cache

from _pytest.config.findpaths import determine_setup
from _verify_config_corpus import DF_CONFIG_PATH, REPO_ROOT, load_config_scalar
from test_verify_cmd import xdist_offenders

from orchestrator import verify_plan
from orchestrator.config import _discover_module_configs
from orchestrator.verify import _scope_to_keyword, _serial_pytest_str, _worktree_reader
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
