"""The confirm re-run form, ``verify_cmd.py::serial_confirm_pytest`` (task 6650).

A confirm re-run must run serially without unregistering xdist, because a
test that requests ``worker_id`` or ``testrun_uid`` ERRORs at setup under
``-p no:xdist`` and so can never be confirmed as a flake. Where xdist is in
play the confirm form appends ``-n 0``; where it is not, it is the recovery
form ``serial_pytest``, which the admission ``-n`` cap leaves alone.
"""

from __future__ import annotations

import shlex

import pytest
from _serial_recovery_helpers import (
    ORCH_SHAPED_ADDOPTS,
    ROOT_SHAPED_ADDOPTS,
    addopts_values,
    effective_worker_count,
    xdist_offenders,
)

from orchestrator.verify_cmd import (
    apply_pytest_numprocesses,
    parse_config_command,
    render,
    serial_confirm_pytest,
    serial_pytest,
)

STRUCTURED = 'uv run pytest tests/test_x.py::test_t'
RAW_CHAIN = 'cd a && uv run pytest -n 4 tests/ && cd ../b && uv run pytest tests/'


def _confirmed_structured():
    return serial_confirm_pytest(parse_config_command(STRUCTURED), ORCH_SHAPED_ADDOPTS)


def _confirmed_raw_chain():
    return serial_confirm_pytest(parse_config_command(RAW_CHAIN), ())


class TestWhereXdistIsInPlay:
    def test_a_configured_xdist_session_is_pinned_to_zero_workers(self):
        tokens = shlex.split(render(_confirmed_structured()))

        assert effective_worker_count(tokens) == '0'
        assert 'no:xdist' not in tokens
        assert addopts_values(tokens) == []
        assert tokens[:3] == ['uv', 'run', 'pytest']
        assert tokens[-1] == 'tests/test_x.py::test_t'

    def test_an_argv_worker_count_is_outvoted_rather_than_stripped(self):
        confirmed = serial_confirm_pytest(parse_config_command('pytest -n 4 --dist loadgroup tests/'), ())
        tokens = shlex.split(render(confirmed))

        assert effective_worker_count(tokens) == '0'
        assert tokens[tokens.index('-n') + 1] == '4'
        assert tokens[tokens.index('--dist') + 1] == 'loadgroup'
        assert 'no:xdist' not in tokens

    def test_every_pytest_invocation_of_a_raw_chain_is_pinned(self):
        rendered = render(_confirmed_raw_chain())

        assert rendered.count('-n 0') == 2
        assert rendered.count(' && ') == 3
        assert 'no:xdist' not in rendered


class TestWhereXdistIsNotInPlay:
    def test_the_recovery_form_is_used(self):
        cmd = parse_config_command(STRUCTURED)

        assert serial_confirm_pytest(cmd, ROOT_SHAPED_ADDOPTS) == serial_pytest(cmd, ROOT_SHAPED_ADDOPTS)

    @pytest.mark.parametrize('command', ['uv run ruff check src/', 'mypy src/'])
    def test_a_command_that_is_not_pytest_comes_back_as_itself(self, command):
        cmd = parse_config_command(command)

        assert serial_confirm_pytest(cmd, ORCH_SHAPED_ADDOPTS) is cmd


class TestTheAdmissionCapLeavesAConfirmAlone:
    @pytest.mark.parametrize('build', [_confirmed_structured, _confirmed_raw_chain], ids=['structured', 'raw'])
    def test_a_confirm_form_command_is_not_capped(self, build):
        confirmed = build()

        assert apply_pytest_numprocesses(confirmed, '16') is confirmed

    @pytest.mark.parametrize('command', ['pytest -n 0 tests/', 'pytest -n0 tests/', 'pytest --numprocesses=0 tests/'])
    def test_an_operator_zero_worker_count_is_not_capped(self, command):
        cmd = parse_config_command(command)

        assert apply_pytest_numprocesses(cmd, '16') is cmd

    def test_a_zero_count_outvoted_by_a_later_one_is_capped(self):
        cmd = parse_config_command('pytest -n 0 -n 4 tests/')

        assert effective_worker_count(shlex.split(render(apply_pytest_numprocesses(cmd, '16')))) == '16'


def test_an_env_transient_recovery_of_a_confirm_sheds_the_zero_count():
    rendered = render(_confirmed_structured())
    recovered = render(serial_pytest(parse_config_command(rendered), ORCH_SHAPED_ADDOPTS))

    assert xdist_offenders(shlex.split(recovered)) == []
