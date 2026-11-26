"""Every repo file setup-host.sh references by a literal path must exist and be tracked.

setup-host.sh runs under `set -euo pipefail`, so a repo file it reads that is no
longer there (a retired unit template the installer still seds or copies) aborts
the installer at that line, and every LATER section silently never runs.  A path
it only prints, such as a manual-steps reminder, must exist too, or the reminder
sends the operator to a missing file.  So every literal
`$REPO_ROOT/scripts/<path>` the installer names must be a file git tracks.  A
reference whose path continues with a `$` variable is a loop or a derived name,
which this literal sweep cannot resolve, so it is skipped.

`git_listing` is importable by name only because tests/scripts/conftest.py puts
this directory on sys.path.
"""
import pathlib
import re

import pytest
from git_listing import git
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

REPO_ROOT = pathlib.Path(__file__).parents[2]
SETUP_HOST_SH = REPO_ROOT / "scripts" / "setup-host.sh"

_REPO_SCRIPT_REF_RE = re.compile(r"\$\{?REPO_ROOT\}?/scripts/([^\s\"'`;|&<>()]+)")


def _literal_repo_script_references() -> set[str]:
    """Repo-relative paths of every literal scripts/ file setup-host.sh names."""
    refs = _REPO_SCRIPT_REF_RE.findall(SETUP_HOST_SH.read_text(encoding="utf-8"))
    return {f"scripts/{ref.rstrip('.,:')}" for ref in refs if "$" not in ref}


def test_every_repo_script_setup_host_references_is_tracked() -> None:
    references = _literal_repo_script_references()
    assert references, (
        f"No literal $REPO_ROOT/scripts/<path> reference found in "
        f"{SETUP_HOST_SH.name}; the pattern no longer matches how the installer "
        "names its repo files, so this guard would pass having checked nothing."
    )

    tracked = set(git(REPO_ROOT, "ls-files", "-z", "--", "scripts").split("\0"))
    missing = sorted(
        path for path in references
        if path not in tracked or not (REPO_ROOT / path).is_file()
    )
    assert not missing, (
        f"{SETUP_HOST_SH.name} references {', '.join(missing)}, which "
        f"{'are not git-tracked files' if len(missing) > 1 else 'is not a git-tracked file'}. "
        "The installer runs under `set -euo pipefail`, so a deleted file it still "
        "reads aborts it at that line and every later section never runs. Fix: "
        "remove the installer's reference along with the file, or restore the file."
    )
