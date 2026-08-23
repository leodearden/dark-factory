"""Ruff config/cache resolution must terminate at the worktree boundary.

Task 3922.  ruff resolves BOTH its settings and its cache by walking parent
directories up from each linted file, and a git VCS root does NOT stop that
walk.  A task worktree at ``<parent>/.worktrees/<id>`` whose own root declares
no ``[tool.ruff]`` therefore escapes: the walk reaches the PARENT checkout's
pyproject.toml, so the merge gate reads the parent's rule set (from its
UNCOMMITTED working tree) and shares the parent's ``.ruff_cache`` with every
concurrently-verifying sibling worktree.

The adopted fix has two halves, and this module guards both:

* the CACHE half is FIXED — ``_run_cmd`` threads its ``cwd`` (the worktree
  root, verbatim, for the lint leg) into ``_target_subprocess_env`` so every
  verify spawn gets a worktree-local ``RUFF_CACHE_DIR``.  Measured rule-neutral
  on ruff 0.15.9, hence applied unconditionally.
* the CONFIG half is made LOUD, not fixed — see ``_ruff_config_escapes``.  It
  cannot be fixed safely at the gate (a ``--config`` pin aimed at a rule-less
  pyproject silently falls back to ruff's built-in defaults) and it must not be
  fatal (hard-failing would red every stale-based worktree at once).

Sibling guard: ``tests/scripts/test_worktree_ruff_config_boundary.py`` pins the
two ruff measurements the decision rests on.
"""

import asyncio

from orchestrator import verify


class TestRunCmdThreadsWorktreeIntoRuffCache:
    """``_run_cmd`` must thread its own ``cwd`` into the env builder.

    Step-2's ``worktree=`` parameter is inert until something passes it.  This
    asserts against a REAL spawned subprocess rather than a monkeypatched
    builder on purpose: the defect being fixed is precisely that a value was
    never threaded through, and a mock-based assertion would pass just as
    happily against the un-wired call.
    """

    def test_ruff_cache_dir_visible_to_the_spawned_command(self, tmp_path):
        rc, out, timed_out = asyncio.run(
            verify._run_cmd('echo "$RUFF_CACHE_DIR"', tmp_path, timeout=30)
        )

        assert rc == 0, out
        assert timed_out is False
        assert out.strip() == str(tmp_path / '.ruff_cache')

    def test_caller_env_overlay_still_reaches_the_spawned_command(self, tmp_path):
        # The operator-override contract holds end-to-end, not just in the
        # builder: verify_env wins over the worktree-derived default.
        chosen = str(tmp_path / 'operator-chosen-cache')
        rc, out, _ = asyncio.run(
            verify._run_cmd(
                'echo "$RUFF_CACHE_DIR"',
                tmp_path,
                timeout=30,
                env={'RUFF_CACHE_DIR': chosen},
            )
        )

        assert rc == 0, out
        assert out.strip() == chosen
