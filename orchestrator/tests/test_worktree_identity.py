"""Tests for worktree_identity.read_worktree_title's new-then-old
`.task-meta` resolution (W11 epsilon1 regression).

Before this relocation, `read_worktree_title` read only
`<worktree>/.task/{metadata,plan}.json`.  Once `TaskArtifacts` writes to the
sibling `<worktree_base>/.task-meta/<name>/` root instead (W11 epsilon1),
those legacy paths are permanently empty for every relocated worktree, so
`read_worktree_title` silently returns `None` for everything — and
`identities_match` fails OPEN on `None`, silently disabling the recycled-id
guards.  These tests pin the fix: resolve new-root-then-legacy-root, and
within each root, metadata.json-then-plan.json.
"""
from __future__ import annotations

import json
from pathlib import Path

from orchestrator.artifacts import TaskArtifacts
from orchestrator.worktree_identity import read_worktree_title


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


class TestReadWorktreeTitleRelocatedTaskMeta:
    """`read_worktree_title(worktree)` keyed on
    `TaskArtifacts.meta_root_for(worktree.parent, worktree.name)`."""

    def test_reads_relocated_metadata_json(self, tmp_path: Path):
        """(i) ONLY the relocated .task-meta/<name>/metadata.json carries a
        title — this is the core regression: today this returns None."""
        worktree = tmp_path / 'base' / 'name'
        meta_root = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
        _write_json(meta_root / 'metadata.json', {'title': 'Relocated'})

        assert read_worktree_title(worktree) == 'Relocated'

    def test_new_root_wins_over_legacy(self, tmp_path: Path):
        """(ii) Both the new .task-meta root and the legacy <worktree>/.task
        carry metadata.json titles — the new root must win."""
        worktree = tmp_path / 'base' / 'name'
        meta_root = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
        _write_json(meta_root / 'metadata.json', {'title': 'New'})
        _write_json(worktree / '.task' / 'metadata.json', {'title': 'Old'})

        assert read_worktree_title(worktree) == 'New'

    def test_legacy_plan_json_fallback_preserved(self, tmp_path: Path):
        """(iii) ONLY the legacy <worktree>/.task/plan.json carries a title —
        the pre-existing compat fallback must still work."""
        worktree = tmp_path / 'base' / 'name'
        _write_json(worktree / '.task' / 'plan.json', {'title': 'Legacy'})

        assert read_worktree_title(worktree) == 'Legacy'

    def test_new_root_metadata_then_plan_precedence(self, tmp_path: Path):
        """(iv) Within the NEW root, plan.json carries a title and there is
        no metadata.json there at all — plan.json must still be checked
        (metadata-then-plan precedence applies per-root, not globally)."""
        worktree = tmp_path / 'base' / 'name'
        meta_root = TaskArtifacts.meta_root_for(worktree.parent, worktree.name)
        _write_json(meta_root / 'plan.json', {'title': 'FromPlan'})

        assert read_worktree_title(worktree) == 'FromPlan'

    def test_neither_seeded_returns_none(self, tmp_path: Path):
        """(v) Neither root carries anything — fail-open contract preserved:
        callers (identities_match) treat None as unprovable."""
        worktree = tmp_path / 'base' / 'name'

        assert read_worktree_title(worktree) is None
