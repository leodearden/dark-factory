"""The one throwaway-pytest-session harness for the df_pytest_isolation guard family.

A session-scoped guard in ``df_pytest_isolation`` exists to make the RUN exit
non-zero even when every test passed. A fixture cannot fail its own session, so
the only way to observe that contract is to spawn a nested pytest session wired
to the guard and read its exit code and output. Every guard module in this
directory does that through :func:`run_nested_pytest`, so the copy, rootdir
isolation, spawn and timeout cannot drift between them.

It lives here, a sibling helper module, rather than in ``df_pytest_isolation``
for two reasons. Every caller is in ``tests/scripts/``. And that plugin is
imported by every subproject's conftest, which would then load a test-only
harness into every session.

The conftest template and the runner share this file because they are coupled.
:func:`binding_conftest`'s ``sys.path.insert(0, parent)`` resolves the guard
only because :func:`run_nested_pytest` puts the module's copy at the nested
root.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# APPEND, never insert(0, ...): the repo root must stay LAST on sys.path or the
# subproject directories (orchestrator/, shared/, ...) resolve as namespace
# packages shadowing their own src/<pkg>/ — the failure the root conftest.py
# docstring exists to prevent.
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import df_pytest_isolation  # noqa: E402

# Must sit strictly below pytest-timeout's per-test axe (the modules'
# `--timeout=300`), so a wedged nested run reports as the calling test's
# TimeoutExpired, carrying its captured output, rather than the outer test being
# killed mid-assertion. A nested session takes ~2-4s, so this leaves wide
# headroom. Checked by test_nested_pytest_session.py::
# test_the_nested_cap_is_below_this_runs_per_test_timeout.
NESTED_SESSION_TIMEOUT_SECS = 120

# Minimal ini so the nested run's rootdir is the tmp tree and NOT this repo:
# without it pytest walks up looking for an inifile and would inherit this
# repo's addopts (`--import-mode=importlib -m 'not smoke ...'`).
_NESTED_INI = '[pytest]\n'

_HARNESS_FILES = frozenset({'pytest.ini', 'df_pytest_isolation.py'})


def binding_conftest(fixture_name: str, *, setup: str = '') -> str:
    """Source for a nested root conftest that binds *fixture_name* from the copy.

    The binding IS the wiring: importing the fixture into a conftest is what
    registers it for the nested session. *setup* runs after the copy is
    importable and before the binding.
    """
    return (
        'import sys\n'
        'from pathlib import Path\n'
        '\n'
        'sys.path.insert(0, str(Path(__file__).resolve().parent))\n'
        '\n'
        f'{setup}'
        f'from df_pytest_isolation import {fixture_name}  # noqa: F401\n'
    )


def run_nested_pytest(
    root: Path, files: Mapping[str, str], *, targets: Sequence[str] = (),
) -> subprocess.CompletedProcess[str]:
    """Write *files* into a fresh nested tree at *root* and run pytest there.

    *root* is named by the CALLER and created here, so a caller can derive paths
    inside it (e.g. a pidfile) before the run starts and clean up even if this
    raises. *files* maps root-relative paths to source text. Relative *targets*
    resolve against *root* (the cwd); with none, the whole tree is collected.
    """
    clobbered = sorted(_HARNESS_FILES.intersection(files))
    if clobbered:
        raise ValueError(f"files may not replace the harness's own files: {clobbered}")
    root.mkdir(parents=True)
    shutil.copy2(Path(df_pytest_isolation.__file__), root / 'df_pytest_isolation.py')
    (root / 'pytest.ini').write_text(_NESTED_INI)
    for relpath, source in files.items():
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return subprocess.run(
        [
            sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
            *(targets or (str(root),)),
        ],
        cwd=root, capture_output=True, text=True, timeout=NESTED_SESSION_TIMEOUT_SECS,
    )
