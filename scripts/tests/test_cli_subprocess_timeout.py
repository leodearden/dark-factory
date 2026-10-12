"""Resolution tests for scripts/tests/cli_subprocess_timeout.py.

Every case passes an explicit ``environ`` mapping except the one that pins
the ``os.environ`` default, so no parser test mutates process state. The
variable name is unique to this file so an operator's real override of a
consumer's budget can never leak in.
"""
from __future__ import annotations

import pytest
from cli_subprocess_timeout import cli_timeout_from_env

VAR = "EXAMPLE_CLI_TEST_TIMEOUT"


@pytest.mark.parametrize("environ", [{}, {VAR: ""}, {VAR: "   \n\t"}])
def test_unset_or_blank_returns_the_60s_default(environ):
    assert cli_timeout_from_env(VAR, environ) == 60.0


@pytest.mark.parametrize(
    ("raw", "expected"), [("5", 5.0), ("2.5", 2.5), (" 7 ", 7.0)]
)
def test_valid_override_parses(raw, expected):
    assert cli_timeout_from_env(VAR, {VAR: raw}) == expected


@pytest.mark.parametrize("raw", ["abc", "0", "-1", "nan", "inf", "-inf"])
def test_malformed_override_raises_naming_the_var_and_value(raw):
    with pytest.raises(ValueError) as excinfo:
        cli_timeout_from_env(VAR, {VAR: raw})
    assert VAR in str(excinfo.value)
    # repr(), not a bare substring: "0" or "-1" could otherwise match
    # incidental text such as a mention of the 60.0 default.
    assert repr(raw) in str(excinfo.value)


def test_only_the_named_variable_is_read():
    environ = {"B_TIMEOUT": "5"}
    assert cli_timeout_from_env("A_TIMEOUT", environ) == 60.0
    assert cli_timeout_from_env("B_TIMEOUT", environ) == 5.0


def test_process_environment_is_read_by_default(monkeypatch):
    monkeypatch.setenv(VAR, "3")
    assert cli_timeout_from_env(VAR) == 3.0
