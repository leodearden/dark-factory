"""Regression sentinel: scripts/flip_task_1153_to_pending.sh must not exist.

This file guards against accidental re-introduction of the transient timer
scaffolding that was created for task 1153. The script was already deleted
once (in a housekeeping pass) and then restored at commit beff8eb6df because
the flip-task-1153.timer ExecStart pointed at its path. Task 1155 performed
the authoritative teardown:
  - stopped the transient systemd-run units (flip-task-1153.timer / .service)
  - git-rm'd scripts/flip_task_1153_to_pending.sh

If this test fails, do NOT restore the script. Instead:
  1. Remove the script: git rm scripts/flip_task_1153_to_pending.sh
  2. Ensure the paired systemd units are stopped:
     systemctl --user stop flip-task-1153.timer flip-task-1153.service
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def test_flip_task_1153_script_does_not_exist() -> None:
    """scripts/flip_task_1153_to_pending.sh must not be present in the repo.

    The script was committed at beff8eb6df and torn down by task 1155.
    Its presence means the dead timer scaffolding has been accidentally
    re-introduced — delete the file again (do not restore the timer).
    """
    script_path = REPO_ROOT / "scripts" / "flip_task_1153_to_pending.sh"
    assert not script_path.exists(), (
        f"Dead script still present at {script_path}\n"
        "This scaffolding (originally committed at beff8eb6df) was removed by task 1155.\n"
        "To fix: git rm scripts/flip_task_1153_to_pending.sh\n"
        "Do NOT restore the file or re-schedule the flip-task-1153 timer."
    )
