"""Audited archival dispositions for every ``TaskConfigDir`` construction site.

Pure data + rationale, no scanning logic — mirroring
``silent_fallthrough_allowlist.py``'s baseline-record role.
``test_config_dir_archival_gate.py`` holds the assertions.

Why this file exists
--------------------
A ``TaskConfigDir`` IS an agent's ``CLAUDE_CONFIG_DIR``, so whatever destroys it
destroys that session's transcripts with it. Task 3271 found
``archive_task_transcripts`` had exactly TWO non-test call sites against SEVEN
construction sites, and nothing anywhere asserted that ratio — so a config dir
added to ``dry_run_unblock`` (commit 7a07c40820) was ``rmtree``'d with its
investigation transcript intact and stayed invisible, because the teardown
backstop in ``GitOps.cleanup_worktree`` composes ``f'claude-config-{branch}'``
literally and silently never matched the ``-unblock`` suffix.

Each entry below records what destroys the dir and whether its transcripts are
archived first. The point is not that every site MUST archive — two
deliberately should not — but that the answer is a decision on the record
rather than an accident nobody noticed.

Entry schema
------------
``path``         repo-relative posix path of the constructing file
``qualname``     enclosing function/class qualname (``<module>`` at module scope)
``destroyed_by`` what removes the dir
``disposition``  one of :data:`DISPOSITIONS`
``rationale``    why that disposition is correct
``archives_in``  required for ``ARCHIVED``, forbidden otherwise: the
                 ``(path, qualname)`` locations that actually reference
                 ``archive_task_transcripts`` for this dir. The gate re-derives
                 those references from the AST and cross-checks them, so an
                 ``ARCHIVED`` claim whose hook was deleted FAILS rather than
                 sitting here as a comfortable lie. Symmetrically, an
                 ``UNARCHIVED_*`` entry whose own function starts archiving
                 fails too — the record and the code cannot drift apart in
                 either direction.
``follow_up``    conventional for ``UNARCHIVED_GAP``; the task/scope carrying the
                 fix. Prose for a reader, NOT gate-enforced — a non-empty check
                 on free text passes on any string, so the gate asserts nothing
                 about it.

Keys are ``(path, qualname)`` — deliberately NOT line numbers, which drift on
every unrelated edit. The gate compares as a MULTISET: ``UsageGate.__init__``
legitimately constructs two, and each copy in the tree must be matched by a
copy here, so a third added to an already-listed function still trips the gate.

The same drift argument applies to the PROSE below, which is why it names
functions (``GitOps.cleanup_worktree``, ``TaskWorkflow._invoke``,
``gc_run_config_dir``) rather than ``file.py:1234``. Line citations in an
earlier revision of this file had already rotted into unrelated code by the
first review pass — a stale pointer in a record whose entire value is being
accurate is worse than no pointer, because a reader who chases it and finds
nothing concludes the rationale was fabricated. Where a site's approximate
position genuinely helps a reader find it, it is given as ``~:NNN`` and read as
a hint, never as the identity.
"""

from __future__ import annotations

from typing import Any

# Archived before destruction.
ARCHIVED = 'ARCHIVED'
# Not archived, and that is the right call — the transcripts carry no signal
# worth the storage.
UNARCHIVED_BY_DESIGN = 'UNARCHIVED_BY_DESIGN'
# Not archived, and that is a real loss — deferred, not accepted.
UNARCHIVED_GAP = 'UNARCHIVED_GAP'

DISPOSITIONS = frozenset({ARCHIVED, UNARCHIVED_BY_DESIGN, UNARCHIVED_GAP})


AUDITED_SITES: tuple[dict[str, Any], ...] = (
    # ---------------------------------------------------------------- 1 & 2
    {
        'path': 'shared/src/shared/usage_gate.py',
        'qualname': 'UsageGate.__init__',
        'destroyed_by': (
            'cleanup_at_exit=True atexit rmtree (TaskConfigDir.__init__); '
            'UsageGate.shutdown(); UsageGate.sweep_stale_pid_dirs'
        ),
        'disposition': UNARCHIVED_BY_DESIGN,
        'rationale': (
            'Per-account cap-probe dir (~:633). UsageGate._run_probe does issue '
            'a real `claude --print`, so a transcript IS written — but it is a '
            'haiku "Say ok" at --max-turns 1, which has no investigative '
            'content to mine and would be pure volume in the corpus. These '
            'probe dirs are also the only sites that opt into '
            'cleanup_at_exit=True, which TaskConfigDir.__init__ documents as '
            'the deliberate exception to the preserve-for-archival default — so '
            'the non-archival here is an existing, explicit decision, not an '
            'oversight.'
        ),
    },
    {
        'path': 'shared/src/shared/usage_gate.py',
        'qualname': 'UsageGate.__init__',
        'destroyed_by': (
            'cleanup_at_exit=True atexit rmtree (TaskConfigDir.__init__); '
            'UsageGate.shutdown(); UsageGate.sweep_stale_pid_dirs'
        ),
        'disposition': UNARCHIVED_BY_DESIGN,
        'rationale': (
            'Fallback pid-keyed probe dir (~:646), reached when no per-account '
            'dir applies. Same "Say ok" --max-turns 1 probe, same disposition '
            'as its sibling above. Listed as its own entry because the gate '
            'counts sites, not functions.'
        ),
    },
    # -------------------------------------------------------------------- 3
    {
        'path': 'fused-memory/src/fused_memory/reconciliation/harness.py',
        'qualname': 'ReconciliationHarness._resume_guard_reason',
        'destroyed_by': (
            'cli_stage_runner.gc_run_config_dir -> shutil.rmtree'
        ),
        'disposition': UNARCHIVED_GAP,
        'rationale': (
            'Per-run reconciliation config dir (~:2067). No archival exists '
            'anywhere in fused-memory: archive_task_transcripts is imported '
            'only by orchestrator/workflow.py, orchestrator/git_ops.py and '
            'orchestrator/dry_run_unblock.py, so every completed recon run\'s '
            'CLI transcript is destroyed. These are substantive multi-turn '
            'stage transcripts, unlike the usage_gate probes — a genuine loss, '
            'not a design choice. Deferred rather than swept into task 3271: it '
            'is a different package with its own data-dir conventions, no '
            'shared.transcript_archive dependency edge today, and it needs an '
            'archive-root config decision of its own.'
        ),
        'follow_up': (
            'Filed as a task-3271 follow-up: archive fused-memory reconciliation '
            'run config dirs before gc_run_config_dir rmtrees them.'
        ),
    },
    # -------------------------------------------------------------------- 4
    {
        'path': 'fused-memory/src/fused_memory/reconciliation/stages/base.py',
        'qualname': 'BaseStage.run',
        'destroyed_by': (
            'cli_stage_runner.gc_run_config_dir -> shutil.rmtree'
        ),
        'disposition': UNARCHIVED_GAP,
        'rationale': (
            'Per-stage reconciliation config dir (~:242). Same destroyer and '
            'same gap as the harness site above — fused-memory has no archival '
            'path at all. Shares one follow-up with entry 3, since a single '
            'archive-root decision covers both.'
        ),
        'follow_up': (
            'Filed as a task-3271 follow-up: same fused-memory recon archival '
            'work as entry 3.'
        ),
    },
    # -------------------------------------------------------------------- 5
    {
        'path': 'orchestrator/src/orchestrator/dry_run_unblock.py',
        'qualname': 'run_dry_run_unblock',
        'destroyed_by': (
            'config_dir.cleanup() in the finally; on the preserve_config_dir '
            'branch, `git worktree remove --force` at worktree teardown'
        ),
        'disposition': ARCHIVED,
        'archives_in': (
            ('orchestrator/src/orchestrator/dry_run_unblock.py', 'run_dry_run_unblock'),
        ),
        'rationale': (
            'Per-investigation dir named `{task_id}-unblock` (~:303). ARCHIVED '
            'as of task 3271: archive_task_transcripts(config_dir.path, task_id, '
            'None) runs in the finally ABOVE the preserve/cleanup branch, so '
            'both exits are covered by one call. The producer-side call is '
            'MANDATORY, not redundant with the teardown backstop: '
            'GitOps.cleanup_worktree composes its archival target by literal '
            "f-string, `worktree / '.task' / f'claude-config-{branch}'`, with "
            'no glob and no CONFIG_DIR_PREFIX import — so it can never match '
            'this dir\'s `-unblock` suffix. That is why the forensic preserve '
            'branch was ALSO lossy before the fix: it did not save the '
            'transcript, it only deferred the loss from cleanup() to worktree '
            'removal. Behaviour pinned by '
            'orchestrator/tests/test_dry_run_unblock.py::'
            'TestDryRunTranscriptArchival.'
        ),
    },
    # -------------------------------------------------------------------- 6
    {
        'path': 'orchestrator/src/orchestrator/workflow.py',
        'qualname': 'TaskWorkflow._setup_worktree_and_artifacts',
        'destroyed_by': (
            'TaskWorkflow._cleanup_config_dir(); `git worktree remove --force` '
            'at worktree teardown'
        ),
        'disposition': ARCHIVED,
        'archives_in': (
            ('orchestrator/src/orchestrator/workflow.py', 'TaskWorkflow._invoke'),
            ('orchestrator/src/orchestrator/git_ops.py', 'GitOps.cleanup_worktree'),
        ),
        'rationale': (
            'The main per-task config dir (~:2377). Covered twice: the '
            'TaskWorkflow._invoke producer hook archives each finished session, '
            'and GitOps.cleanup_worktree\'s teardown backstop sweeps the whole '
            'dir for anything the producer missed (e.g. a CancelledError-killed '
            'invocation). _shutdown_steps orders cleanup_done_worktree BEFORE '
            'cleanup_config_dir, so the archiving path runs first. This is the '
            'reference shape the dry_run_unblock hook (entry 5) was modelled '
            'on.'
        ),
    },
    # -------------------------------------------------------------------- 7
    {
        'path': 'orchestrator/src/orchestrator/workflow.py',
        'qualname': 'TaskWorkflow._recycle_config_dir',
        'destroyed_by': 'self._config_dir.cleanup() one line earlier',
        'disposition': UNARCHIVED_GAP,
        'rationale': (
            'Wedged-session recycle (~:8220): cleanup() rmtrees the WHOLE dir '
            'and a fresh TaskConfigDir is reconstructed at the identical path '
            'on the next line. The intent is to discard the wedged session, but '
            'the blast radius is the whole dir — any not-yet-flushed sibling '
            '(e.g. `<sid>/subagents/agent-*.jsonl` from a prior healthy '
            'invocation) goes with it, and the wedged session\'s own transcript '
            'is exactly what a legibility miner would want in order to explain '
            'the wedge. Note this is the same MODULE as entry 6, which does '
            'archive — the gate checks the enclosing FUNCTION, so a '
            'module-level import of archive_task_transcripts cannot launder '
            'this gap into a false ARCHIVED. Not fixed here: workflow.py is a '
            'hot, heavily-locked file well outside task 3271\'s scope, and the '
            'right fix is an archival call before the cleanup rather than a '
            'change to the recycle semantics.'
        ),
        'follow_up': (
            'Filed as a task-3271 follow-up: archive before '
            'TaskWorkflow._recycle_config_dir cleans up the wedged session dir.'
        ),
    },
)


# ---------------------------------------------------------------------------
# Module-level notes — NOT site entries (neither constructs a TaskConfigDir),
# recorded here so the audit is not re-derived from scratch next time.
# ---------------------------------------------------------------------------

NOTES = (
    """Adjacent unarchived destruction paths (no TaskConfigDir construction, so
    not gate-scanned — but they destroy config dirs someone else built):

      * GitOps._acquire_warm_lane_impl — the reattach path rmtrees the previous
        occupant's ENTIRE `.task/` dir (`shutil.rmtree(lane / '.task',
        ignore_errors=True)`, guarded by the "belongs to the lane's PREVIOUS
        occupant" comment), taking BOTH its `claude-config-<prev>` and its
        `claude-config-<prev>-unblock` with it, with no archival first.
      * GitOps.cleanup_worktree — its warm-lane and spec-lane early returns
        (`release_warm_lane` / `release_spec_lane`, at the very top of the
        method) fire BEFORE the teardown-archival block, so warm-lane tasks
        never reach the backstop at all. The natural fix is an archival call at
        lane RELEASE rather than at removal, which is a different design from
        the other gaps here.

    Filed as a task-3271 follow-up.""",

    """Merge-speculation lanes — open question RESOLVED, and NOT a gap. Recorded
    so it is not re-opened.

    The pool lanes are named `_spec-<k>` (git_ops.py's PROTECTED_PREFIXES band
    and warm_lane_pool.py), NOT `_merge-<sha>`; the `_merge-*` names belong to
    the ephemeral merge worktree (`_merge-<uuid8>`) and the fixed
    `_merge-verify` lane (PERSISTENT_MERGE_WORKTREE_NAME). GitOps.acquire_spec_lane
    does only `git worktree add --detach`, `git reset --hard`, `git clean`, and
    a CoW seed-warm-lane.sh reflink; the lane is handed to
    merge_queue._run_post_merge_verify, which spawns plain verify subprocesses.
    invoke_agent / invoke_with_cap_retry / CLAUDE_CONFIG_DIR / orchestrator.agents
    appear NOWHERE in git_ops.py, merge_liveness.py, verify.py, verify_runner.py,
    merge_queue.py or the merge_* modules. The one agent reachable from
    merge-verify — merge_queue._spawn_merge_verify_dry_run — passes
    `worktree=str(req.worktree)`, the task's OWN retained worktree, never the
    lane.

    Conclusion: merge-speculation lanes produce no agent transcripts, so the
    absence of `-warm-lanes-worktrees--merge-*` dirs from reify's archive is
    EXPECTED, not missing archival.""",
)
