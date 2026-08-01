"""Role-allowlist tests for the done_provenance gate.

The 2026-04-27 incident had an `implementer` subprocess call
``set_task_status(done, ...)`` directly. The fused-memory schema-side
validator is the load-bearing control; this test pins the role allowlist as
the additional defense so a regression in the role config can be caught at
unit-test time.
"""

import subprocess
from pathlib import Path

from orchestrator.agents.roles import (
    ARCHITECT,
    DEBUGGER,
    DEEP_REVIEWER,
    IMPLEMENTER,
    JUDGE,
    MERGER,
    REVIEWER_COMPREHENSIVE,
    STEWARD,
)

SET_TASK_STATUS = 'mcp__fused-memory__set_task_status'


class TestStewardCanWriteTaskStatus:
    def test_steward_allows_set_task_status(self):
        assert SET_TASK_STATUS in STEWARD.allowed_tools

    def test_steward_does_not_disallow_set_task_status(self):
        assert SET_TASK_STATUS not in STEWARD.disallowed_tools


class TestWorkflowRolesCannotWriteTaskStatus:
    def test_implementer_disallows_set_task_status(self):
        assert SET_TASK_STATUS in IMPLEMENTER.disallowed_tools

    def test_architect_disallows_set_task_status(self):
        assert SET_TASK_STATUS in ARCHITECT.disallowed_tools

    def test_debugger_disallows_set_task_status(self):
        assert SET_TASK_STATUS in DEBUGGER.disallowed_tools

    def test_merger_disallows_set_task_status(self):
        assert SET_TASK_STATUS in MERGER.disallowed_tools

    def test_judge_disallows_set_task_status(self):
        assert SET_TASK_STATUS in JUDGE.disallowed_tools

    def test_reviewer_comprehensive_disallows_set_task_status(self):
        assert SET_TASK_STATUS in REVIEWER_COMPREHENSIVE.disallowed_tools

    def test_deep_reviewer_disallows_set_task_status(self):
        assert SET_TASK_STATUS in DEEP_REVIEWER.disallowed_tools


class TestStewardPromptDocumentsTwoKinds:
    """The schema is enforced server-side, but the steward needs procedural
    guidance to get the call right the first time."""

    def test_prompt_mentions_kind_merged(self):
        assert 'kind="merged"' in STEWARD.system_prompt

    def test_prompt_mentions_kind_found_on_main(self):
        assert 'kind="found_on_main"' in STEWARD.system_prompt

    def test_prompt_documents_ancestor_check(self):
        assert 'merge-base' in STEWARD.system_prompt
        assert '--is-ancestor' in STEWARD.system_prompt

    def test_prompt_forbids_update_task_for_provenance(self):
        # Layer 1 of the gate is the server-side block on
        # update_task(metadata={done_provenance: ...}); the prompt should
        # match that policy so the steward never tries it.
        assert 'update_task' in STEWARD.system_prompt


class TestAncestryCheckIsThreeOutcome:
    """Corroborates the rc table documented in roles.py's
    ANCESTRY_CHECK_INSTRUCTIONS against the installed git.

    Deliberately holds no assertion about where that block is spliced into
    STEWARD.system_prompt: pinning the splice count is tautological with the
    concatenation that produces the prompt, and would fire spuriously the
    moment a third call site is added.
    """

    def test_real_git_distinguishes_unresolvable_from_off_main(self, tmp_path: Path) -> None:
        """Ties the documented rc table to the installed git, not to prose.

        Verified on git 2.43.0: a commit that only exists on a side branch
        exits 1, an unresolvable object exits 128 with 'Not a valid' in
        stderr, and a commit on main exits 0.
        """
        repo = tmp_path / 'repo'
        repo.mkdir()
        subprocess.run(['git', 'init', '-b', 'main'], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ['git', 'config', 'user.email', 'test@test.com'], cwd=repo, check=True, capture_output=True,
        )
        subprocess.run(['git', 'config', 'user.name', 'Test'], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ['git', '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-m', 'root'],
            cwd=repo, check=True, capture_output=True,
        )
        main_tip = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, check=True, capture_output=True, text=True,
        ).stdout.strip()

        subprocess.run(['git', 'checkout', '-b', 'side'], cwd=repo, check=True, capture_output=True)
        subprocess.run(
            ['git', '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-m', 'side'],
            cwd=repo, check=True, capture_output=True,
        )
        side_tip = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=repo, check=True, capture_output=True, text=True,
        ).stdout.strip()

        off_main = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', side_tip, 'main'],
            cwd=repo, capture_output=True, text=True,
        )
        assert off_main.returncode == 1, (
            f'expected a side-branch-only commit to exit 1; got returncode='
            f'{off_main.returncode!r} stdout={off_main.stdout!r} stderr={off_main.stderr!r}'
        )

        unresolvable = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', '0' * 40, 'main'],
            cwd=repo, capture_output=True, text=True,
        )
        assert unresolvable.returncode == 128, (
            f'expected a bogus SHA to exit 128; got returncode={unresolvable.returncode!r} '
            f'stdout={unresolvable.stdout!r} stderr={unresolvable.stderr!r}'
        )
        assert 'Not a valid' in unresolvable.stderr, unresolvable.stderr

        on_main = subprocess.run(
            ['git', 'merge-base', '--is-ancestor', main_tip, 'main'],
            cwd=repo, capture_output=True, text=True,
        )
        assert on_main.returncode == 0, (
            f'expected the main-tip commit to exit 0; got returncode={on_main.returncode!r} '
            f'stdout={on_main.stdout!r} stderr={on_main.stderr!r}'
        )
