"""I3 factory grep-guard for SpeculativeItem construction sites (task 1990).

step-5  RED   — no production site is marked yet
step-6  GREEN — the 15 legit factory sites carry the marker; the 2 former
                CAS-loop hand-rebuilds are now dataclasses.replace(...) calls
                and no longer match the grep pattern at all.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_MARKER = '# spec-factory'
_SCOPED_PATH = 'orchestrator/src/orchestrator/'


def test_every_speculative_item_construction_in_src_is_marked(repo_root: Path | None) -> None:
    """Every `SpeculativeItem(` construction under orchestrator/src/ must carry
    the `# spec-factory` marker on its construction line.

    Guards I3: a future hand-rebuild that bypasses `dataclasses.replace(...)`
    and reconstructs a SpeculativeItem field-by-field (the task-1928 bug
    class) shows up here as an un-marked hit.
    """
    if repo_root is None:
        pytest.skip('not running inside a git checkout')

    result = subprocess.run(
        ['git', 'grep', '-n', 'SpeculativeItem(', '--', _SCOPED_PATH],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        pytest.fail(f'git grep failed (exit {result.returncode}): {result.stderr}')

    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    unmarked = [ln for ln in lines if _MARKER not in ln]
    assert not unmarked, (
        f'Found {len(unmarked)} SpeculativeItem( construction(s) in {_SCOPED_PATH} '
        f'without the {_MARKER!r} factory marker (I3 — every src construction must '
        f'be a whitelisted factory site or dataclasses.replace):\n' + '\n'.join(unmarked)
    )
