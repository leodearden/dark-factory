"""scripts/_predicate_exit.py: a predicate's exit statuses and its usage-error parser."""
from __future__ import annotations

import pytest
from _predicate_exit import EXIT_ERROR, EXIT_NOT_YET, PredicateParser


def test_a_predicate_waits_on_75_and_fails_on_1():
    assert (EXIT_NOT_YET, EXIT_ERROR) == (75, 1)


@pytest.mark.parametrize('argv', [['--unknown'], [], ['--count', 'x']])
def test_a_usage_error_exits_1_naming_the_problem(capsys, argv):
    parser = PredicateParser(prog='probe')
    parser.add_argument('target')
    parser.add_argument('--count', type=int)

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(argv)

    assert excinfo.value.code == EXIT_ERROR
    assert 'probe: error:' in capsys.readouterr().err
