"""Importing prompt_opt, or its CLI module, loads only what the caller asked for.

The package body is its docstring alone, so ``__main__``'s function-local
imports really do defer ``smoke`` and ``curator_corpus``.

Each probe runs in a FRESH interpreter: this pytest process has already
imported the submodules through sibling test modules, so an in-process
``sys.modules`` check would pass vacuously.
"""

import subprocess
import sys

_CLI_DEFERRED_MODULES = (
    'orchestrator.evals.prompt_opt.smoke',
    'orchestrator.evals.prompt_opt.curator_corpus',
    'orchestrator.agents.roles',
)


def _run_probe(probe: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, '-c', probe], capture_output=True, text=True, timeout=120
    )


def test_importing_the_package_loads_no_submodule():
    result = _run_probe(
        'import sys\n'
        'import orchestrator.evals.prompt_opt as pkg\n'
        'loaded = sorted(\n'
        "    m for m in sys.modules if m.startswith('orchestrator.evals.prompt_opt.')\n"
        ')\n'
        'assert not loaded, loaded\n'
        'assert pkg.__doc__\n'
    )
    assert result.returncode == 0, result.stderr


def test_importing_the_cli_module_does_not_load_what_it_defers():
    result = _run_probe(
        'import sys\n'
        'import orchestrator.evals.prompt_opt.__main__ as m\n'
        f'present = [n for n in {_CLI_DEFERRED_MODULES!r} if n in sys.modules]\n'
        'assert not present, present\n'
        'assert m.cli is not None\n'
    )
    assert result.returncode == 0, result.stderr
