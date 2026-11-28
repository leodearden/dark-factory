"""The lock-charter predicate against THIS checkout's tracked index.

``shared.locking`` decides which ``metadata.files`` entries are files, so every
extension-less file tracked here must classify as one, and no allowlisted name
may also be a directory here.  Split out of ``shared/tests/test_locking.py``
(task 5654), which keeps the pure ``shared.locking`` unit tests: these sweeps
run ``git ls-files`` over the whole tree, which shared's verify cache key does
not cover, so they live in this never-cached module.

The reify half of the same properties, and the comparison with reify's
``scripts/lock-charter-guard.sh`` emitters, run nightly OFF the merge gate in
``scripts/check_lock_charter_drift.py`` (task 11623; ``OPERATIONS.md`` section
9a): a sibling repo's working state must not decide a dark-factory merge.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from shared.locking import EXTENSIONLESS_FILENAMES, is_file_path
from shared.whole_tree_guard_family import WHOLE_TREE_SCAN_TEST_TIMEOUT

pytestmark = pytest.mark.timeout(WHOLE_TREE_SCAN_TEST_TIMEOUT)

_DF_REPO_ROOT = Path(__file__).parents[2]


# ---------------------------------------------------------------------------
# GITLINK EXCLUSION (load-bearing, not a measurement artefact): entries with
# mode 160000 are submodule mount points — `graphiti` and `mem0` in
# dark-factory — and are deliberately NOT admitted.  They are extension-less
# and would otherwise qualify, but admitting them would let a task declare an
# entire vendored submodule as its lock charter, which is strictly worse than
# the bug being fixed.  They stay DIRECTORIES by design.  The sweep below
# asserts this filter is doing real work rather than passing vacuously.
# ---------------------------------------------------------------------------

def _tracked_entries(repo_root: Path) -> tuple[list[str], list[str]] | None:
    """``(non_gitlink_paths, gitlink_paths)`` for *repo_root*, or None if not a checkout.

    ``git ls-files -s`` (not plain ``ls-files``) is what makes the mode column
    available.  Mode ``160000`` means a submodule gitlink; those entries are
    returned SEPARATELY rather than discarded, because the sweeps need them to
    prove the exclusion is doing real work.  See the GITLINK EXCLUSION note
    above for why the filter is load-bearing rather than cosmetic.
    """
    result = subprocess.run(
        ['git', '-C', str(repo_root), 'ls-files', '-s', '-z'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    paths: list[str] = []
    gitlinks: list[str] = []
    for entry in result.stdout.split('\0'):
        if not entry:
            continue
        # Format: "<mode> <sha> <stage>\t<path>"
        meta, _, path = entry.partition('\t')
        if not path:
            continue
        if meta.split(' ', 1)[0] == '160000':
            gitlinks.append(path)
        else:
            paths.append(path)
    return paths, gitlinks


def _tracked_file_paths(repo_root: Path) -> list[str] | None:
    """Tracked NON-GITLINK paths under *repo_root*, or None if not a checkout."""
    entries = _tracked_entries(repo_root)
    return None if entries is None else entries[0]


class TestExtensionlessFilenames:
    """Extension-less tracked FILES classify as files (dark_factory #3248).

    The hazard these pin, stated concretely: two tasks editing
    ``hooks/project-checks`` concurrently could both hold NO lock on it, because
    a charter that strips to empty degrades to a per-task ``task-<id>`` synthetic
    lock that conflicts with nothing.
    """

    def test_no_canonical_name_is_also_a_directory(self):
        """No member may appear as a NON-FINAL component of any tracked path.

        This is what bounds the risk of matching on a bare basename.  A member
        that were ALSO a directory name somewhere in the tree would make that
        directory declarable as a lock charter — the exact over-wide-charter
        failure the guard exists to prevent.  Measured zero collisions across
        both repos on 2026-08-09; this pins that property permanently rather
        than leaving it as a one-time observation.

        SCOPE, stated plainly so the guard is not over-read: it bounds the risk
        for THIS checkout only (reify's tree is swept off the gate, by
        ``scripts/check_lock_charter_drift.py::check_extensionless``), while
        the predicate governs lock charters for every project the orchestrator
        targets.  Several members are generic
        basenames (`cargo`, `Dockerfile`, `LICENSE`, `pre-commit`), so a third
        project with a real `tools/cargo/` DIRECTORY would have it classified as
        a file and retained as a subtree-wide prefix lock, unswept by this test.
        That is why admitting a generic basename is a cross-project commitment —
        see the CROSS-PROJECT SCOPE note on the EXTENSIONLESS_FILENAMES vector.
        """
        paths = _tracked_file_paths(_DF_REPO_ROOT)
        if paths is None:
            pytest.skip(f'{_DF_REPO_ROOT} is not a git checkout (git ls-files failed)')
        assert paths, f'git ls-files returned no tracked paths for {_DF_REPO_ROOT}'

        collisions: dict[str, str] = {}
        for path in paths:
            for component in path.split('/')[:-1]:
                if component in EXTENSIONLESS_FILENAMES:
                    collisions.setdefault(component, path)

        assert not collisions, (
            f'{len(collisions)} EXTENSIONLESS_FILENAMES member(s) are also '
            f'DIRECTORY names in dark-factory, so declaring that directory would now '
            f'be accepted as a file-level lock charter:\n'
            + '\n'.join(f'  {name} — e.g. {path}' for name, path in sorted(collisions.items()))
            + '\nRemove the name from the allowlist, or rename the directory.'
        )

    def test_every_tracked_extensionless_file_is_allowlisted(self):
        """Corpus->allowlist self-audit: a newly tracked extensionless file must go RED.

        Turns "add 8 names" into a self-auditing invariant.  When someone adds a
        new extension-less tracked file to this repo, this test names it here
        rather than letting it surface later as a LockCharterViolation at
        submit time — or, worse, as a silent under-lock.  Reify's half runs
        off the gate, in ``scripts/check_lock_charter_drift.py``.

        This sweep IS duplicated in the γ suite, unlike the extension-corpus
        sweep which is deliberately α-free (see
        test_accept_corpus_covers_every_canonical_extension for that rationale).
        The difference is the failure mode: the extension vector is fully pinned
        by the two drift guards, whereas this one gates a REPO-level property —
        a newly-added tracked file changes the correct answer without any test
        vector changing, so whichever suite runs must be able to see it.

        Gitlink mount points are excluded by _tracked_file_paths and must NOT be
        admitted; that exclusion is asserted below.  The gitlink set is DERIVED
        from git's own mode column rather than hardcoded, which matters in both
        directions: hardcoding ``('graphiti', 'mem0')`` made the assertion
        vacuously true for a repo that vendors no submodules, and would have
        gone red for no real defect if dark-factory ever de-vendored them.  A
        corpus with no submodules skips the sub-assertion explicitly instead of
        passing it silently.
        """
        entries = _tracked_entries(_DF_REPO_ROOT)
        if entries is None:
            pytest.skip(f'{_DF_REPO_ROOT} is not a git checkout (git ls-files failed)')
        paths, gitlinks = entries
        assert paths, f'git ls-files returned no tracked paths for {_DF_REPO_ROOT}'

        # The gitlink filter must be doing real work, not passing vacuously.
        if gitlinks:
            overlap = sorted(set(gitlinks) & set(paths))
            assert not overlap, (
                f'{overlap!r} appear as BOTH mode-160000 gitlinks and ordinary '
                f'tracked paths in dark-factory; _tracked_entries must partition the '
                f'corpus, and admitting a submodule mount point would let a task '
                f'declare an entire vendored submodule as its lock charter'
            )
            for gitlink in gitlinks:
                assert not is_file_path(gitlink), (
                    f'{gitlink!r} is a submodule mount point (mode 160000) and '
                    f'must stay a DIRECTORY — see the GITLINK EXCLUSION note on '
                    f'the corpus.  Admitting it would make an entire vendored '
                    f'submodule declarable as a lock charter, strictly worse '
                    f'than the bug #3248 fixed.'
                )

        unknown: dict[str, str] = {}
        for path in paths:
            seg = path.rsplit('/', 1)[-1]
            if '.' in seg:
                continue
            if not is_file_path(path):
                unknown.setdefault(seg, path)

        assert not unknown, (
            f'{len(unknown)} extension-less tracked dark-factory file(s) classify as '
            f'DIRECTORIES, so any task declaring one in metadata.files is either '
            f'rejected with a LockCharterViolation or — silently — stripped to an '
            f'empty charter and given a task-<id> synthetic lock that conflicts '
            f'with nothing:\n'
            + '\n'.join(f'  {name} — e.g. {path}' for name, path in sorted(unknown.items()))
            + '\nAdd each name to EXTENSIONLESS_FILENAMES in '
            'shared/src/shared/locking.py AND '
            'fused-memory/src/fused_memory/middleware/lock_charter_guard.py, '
            'and to reify scripts/lock-charter-guard.sh _EXTLESS (all three must '
            'agree; the nightly scripts/check_lock_charter_drift.py reports '
            'reify), '
            'plus _CANONICAL_EXTENSIONLESS and _EXTENSIONLESS_ACCEPT_PATHS in '
            'both test copies.'
        )
