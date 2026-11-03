"""Fixture helpers shared by the inline-suppression scanner's test modules.

Plain functions and constants rather than pytest fixtures, resolved by bare
name through ``scripts/tests/conftest.py``'s ``_THIS_DIR`` entry, as
``cli_subprocess_timeout`` is.  Nothing here imports the CLI, so nothing here
seeds a baseline: the verbs' own test module wraps :func:`track_fixture_tree`
for that.

A REAL GIT REPOSITORY, not a bare directory.  The scanner enumerates its corpus
from ``git ls-files``, so trackedness is a property the fixtures must exercise
rather than one the tests assert by reading the code — the same argument
``scripts/tests/test_design_invariants_consistency.py::_write_scan_tree`` gives.

XDIST-SAFE.  The merge gate runs ``pytest … -n auto --dist loadgroup``, so every
test built on these helpers keeps its mutable state inside ``tmp_path`` and
changes no process-wide state outside ``monkeypatch``.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

from inline_suppression_kinds import Site
from inline_suppression_scan import scan_source

#: Generous, and not a performance assertion: these suites count work and never
#: assert a wall clock.  A subprocess that blows through this has hung, which is
#: worth a loud ``TimeoutExpired`` rather than a silent wait.
SUBPROCESS_TIMEOUT_SECS = 120

#: The ``select``/``ignore`` pair this repository's ``pyproject.toml`` files
#: declare, verbatim.
RUFF_CONFIG = '[tool.ruff.lint]\nselect = ["E", "F", "UP", "B", "SIM", "I"]\nignore = ["E501"]\n'


def scrubbed_git_env() -> dict[str, str]:
    """``os.environ`` with every ``GIT_*`` override removed.

    ``GIT_DIR``, ``GIT_WORK_TREE``, ``GIT_INDEX_FILE`` and
    ``GIT_CEILING_DIRECTORIES`` are inherited by default, and any one of them
    silently retargets a git invocation at a different repository than its
    ``cwd`` implies — ``git -C <path>`` READS as "act on <path>" but only
    changes directory, while ``GIT_DIR`` skips repository discovery outright.
    A fixture that lost that race would ``git add`` into the live checkout.
    Spelled as in ``test_design_invariants_consistency.py::_scrubbed_git_env``;
    ``df_pytest_isolation._df_git_env_hermetic`` is the suite-wide second line
    of the same defence.
    """
    return {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}


def run_git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run ``git`` *args* against *cwd* under :func:`scrubbed_git_env`."""
    return subprocess.run(
        ['git', *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
        timeout=SUBPROCESS_TIMEOUT_SECS,
        env=scrubbed_git_env(),
    )


def write_files(root: Path, files: Mapping[str, str]) -> None:
    """Write every *files* entry under *root*, creating parents as needed.

    Separate from :func:`track_fixture_tree` because the consumer-model tests
    need a tree of ``pyproject.toml`` files and no git at all: the nearest-config
    walk reads the filesystem, so making those tests pay for a repository would
    be ceremony that tests nothing.
    """
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')


def track_fixture_tree(root: Path, files: Mapping[str, str]) -> None:
    """Build a git repo at *root* holding *files*, every one of them TRACKED.

    *files* maps a repo-relative path to its whole content — source modules and
    ``pyproject.toml`` alike, since the scanner reads the nearest pyproject as
    ordinary tracked content and a second parameter for it would only be a
    second way to write a file.

    ``git init -q`` then ``git add -A -f``.  No commit, and no ``user.name`` /
    ``user.email`` — ``git ls-files`` reads the INDEX, not history, so a commit
    would be ceremony the scanner never looks at.  The ``-f`` defeats any
    ambient global gitignore.  A test that needs an UNTRACKED file writes it
    itself after this call.
    """
    write_files(root, files)
    for args in (['init', '-q'], ['add', '-A', '-f']):
        run_git(args, cwd=root)


def sites_in(source: str, *, path: str = 'm.py') -> list[Site]:
    """Every Site in *source*, flattened out of the comments that hold them."""
    return [site for comment in scan_source(source, path=path) for site in comment.sites]
