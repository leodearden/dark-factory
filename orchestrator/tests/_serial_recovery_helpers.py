"""Shared helpers for the serial re-run tests (task 5079, task 6650).

``verify_cmd.py::serial_pytest`` forces a pytest command serial for a
recovery re-run with ``-p no:xdist -o addopts=<value>``;
``verify_cmd.py::serial_confirm_pytest`` does it for a confirm re-run with
``-n 0`` where xdist is in play. Their tests, and those of the resolver that
supplies ``<value>``, share four things kept here once: xdist's option
surface read from xdist itself, the ``-o addopts=`` value extractor, the
effective worker-count reader, and the repo-shaped governing configs they
build trees from.
"""

from __future__ import annotations

import json
import shlex
from collections.abc import Sequence
from typing import cast

import pytest


class _RecordingOptionGroup:
    def __init__(self) -> None:
        self.options: list[tuple[tuple[str, ...], dict]] = []

    def addoption(self, *names: str, **attrs) -> None:
        self.options.append((names, attrs))

    _addoption = addoption


class _RecordingParser:
    def __init__(self) -> None:
        self.group = _RecordingOptionGroup()

    def getgroup(self, *_args, **_kwargs) -> _RecordingOptionGroup:
        return self.group

    def addini(self, *_args, **_kwargs) -> None:
        pass


#: argparse actions that consume no value token.
_ZERO_ARG_ACTIONS = frozenset({'store_true', 'store_false', 'store_const', 'append_const', 'count'})


def xdist_option_surface() -> dict[str, bool]:
    """Every option name pytest-xdist registers, mapped to "takes a value token".

    Read from ``xdist.plugin.pytest_addoption`` itself through a recording
    parser stub rather than restated, so an xdist upgrade that adds an option
    turns the shed tests red instead of surfacing as an ``unrecognized
    arguments`` failure on a serial-recovery verify leg.
    """
    from xdist.plugin import pytest_addoption

    parser = _RecordingParser()
    pytest_addoption(cast(pytest.Parser, parser))
    return {
        name: attrs.get('action', 'store') not in _ZERO_ARG_ACTIONS
        for names, attrs in parser.group.options
        for name in names
    }


def xdist_offenders(tokens: Sequence[str]) -> list[str]:
    """The tokens of *tokens* that pytest would read as an xdist option."""
    surface = xdist_option_surface()

    def is_xdist(token: str) -> bool:
        if token in surface:
            return True
        for name, takes_value in surface.items():
            if not takes_value:
                continue
            if name.startswith('--') and token.startswith(f'{name}='):
                return True
            if not name.startswith('--') and token.startswith(name) and len(token) > len(name):
                return True
        return False

    return [token for token in tokens if is_xdist(token)]


def addopts_values(tokens: Sequence[str]) -> list[str]:
    """Every ``-o addopts=<value>`` value in argv *tokens*, in order; pytest applies the last."""
    return [
        value.removeprefix('addopts=')
        for flag, value in zip(tokens, tokens[1:], strict=False)
        if flag == '-o' and value.startswith('addopts=')
    ]


_WORKER_COUNT_OPTIONS = ('-n', '--numprocesses')


def effective_worker_count(tokens: Sequence[str]) -> str | None:
    """The value of the LAST xdist worker-count option in argv *tokens*, or None.

    argparse keeps the last value, so this is the count pytest applies. Reads
    the separate-token (``-n 0``, ``--numprocesses 0``) and attached
    (``-n0``, ``-n=0``, ``--numprocesses=0``) spellings.
    """
    count = None
    for index, token in enumerate(tokens):
        if token in _WORKER_COUNT_OPTIONS and index + 1 < len(tokens):
            count = tokens[index + 1]
        elif token.startswith(('--numprocesses=', '-n=')):
            count = token.partition('=')[2]
        elif token.startswith('-n') and not token.startswith('--') and len(token) > 2:
            count = token[2:]
    return count


#: The repo root's addopts shape: an import mode and a three-clause marker filter.
ROOT_SHAPED_ADDOPTS = (
    '--import-mode=importlib', '-m', 'not smoke and not integration and not warm_lane_bash',
)
#: The orchestrator subproject's addopts shape: xdist options and a marker filter.
ORCH_SHAPED_ADDOPTS = (
    '-n', 'auto', '--dist', 'loadgroup', '--max-worker-restart=0', '-m', 'not warm_lane_bash',
)


def pyproject_with_addopts(addopts: Sequence[str]) -> str:
    """A ``pyproject.toml`` whose ``[tool.pytest.ini_options]`` carries *addopts* as one string."""
    return f'[tool.pytest.ini_options]\naddopts = {json.dumps(shlex.join(addopts))}\n'


ROOT_SHAPED_PYPROJECT = pyproject_with_addopts(ROOT_SHAPED_ADDOPTS)
ORCH_SHAPED_PYPROJECT = pyproject_with_addopts(ORCH_SHAPED_ADDOPTS)
