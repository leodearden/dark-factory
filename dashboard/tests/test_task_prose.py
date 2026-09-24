"""The per-task prose read behind the Tasks tab's Task Detail pane (task 5815).

The ACTIVE_TASKS list no longer carries a task's description/details; the pane
fetches them for the ONE selected task. That read is addressed by the row's own
uid, so the label resolution and the uid format below are the seam the list
rows and the prose route share.
"""

from __future__ import annotations

from dashboard.config import DashboardConfig
from dashboard.data.active_tasks import project_roots_for_label, task_uid


class TestProjectRootsForLabel:
    """A row's project label resolves back to the configured root(s) it names."""

    def test_a_label_matching_one_root_returns_that_root(self, tmp_path):
        primary = tmp_path / 'alpha'
        other = tmp_path / 'beta'
        for root in (primary, other):
            root.mkdir()
        config = DashboardConfig(project_root=primary, known_project_roots=[other])

        assert project_roots_for_label(config, 'beta') == [other.resolve()]

    def test_an_unknown_label_returns_no_root(self, tmp_path):
        primary = tmp_path / 'alpha'
        primary.mkdir()
        config = DashboardConfig(project_root=primary)

        assert project_roots_for_label(config, 'gamma') == []

    def test_roots_sharing_a_basename_all_come_back_primary_first(self, tmp_path):
        """The label is a BASENAME, so it can name two roots; neither is dropped."""
        primary = tmp_path / 'a' / 'proj'
        other = tmp_path / 'b' / 'proj'
        for root in (primary, other):
            root.mkdir(parents=True)
        config = DashboardConfig(project_root=primary, known_project_roots=[other])

        assert project_roots_for_label(config, 'proj') == [
            primary.resolve(), other.resolve(),
        ]

    def test_task_uid_is_the_rows_identity(self):
        assert task_uid('proj', 19) == 'proj/T-19'
