"""Two-parent merge commits whose resolution a test chooses, in the shape
``GitOps.merge_to_main`` records: first parent = the base/main side, second
parent = the merged branch tip.
"""
from __future__ import annotations

from pathlib import Path

from orchestrator.git_ops import _run


async def resolution_merge(
    repo: Path, *, main_sha: str, branch_tip: str, kept_tree_of: str,
) -> str:
    rc, out, err = await _run(
        [
            'git', 'commit-tree', f'{kept_tree_of}^{{tree}}',
            '-p', main_sha, '-p', branch_tip,
            '-m', 'Resolution merge (test)',
        ],
        cwd=repo,
    )
    assert rc == 0, f'git commit-tree failed: {err}'
    return out.strip()
