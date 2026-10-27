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
        'path': 'fused-memory/src/fused_memory/reconciliation/stages/base.py',
        'qualname': 'BaseStage.run',
        'destroyed_by': (
            'cli_stage_runner.gc_run_config_dir, called from '
            'ReconciliationHarness._persist_stage_reports_then_gc_config_dir '
            '(the shared finally tail of run_full_cycle and '
            '_run_remediation_pass) and ReconciliationHarness._recover_one_run'
        ),
        'disposition': ARCHIVED,
        'archives_in': (
            (
                'fused-memory/src/fused_memory/reconciliation/stages/base.py',
                'BaseStage._archive_stage_transcripts',
            ),
            (
                'fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py',
                'gc_run_config_dir',
            ),
        ),
        'rationale': (
            'Per-run reconciliation config dir (~:281), shared by every stage '
            'of the run. ARCHIVED as of task 3972, into '
            '`<explore_codebase_root>/data/orchestrator/agent-transcripts` under '
            'the key `recon-<run_id>` (reconciliation.transcript_archive_root, '
            'resolved by cli_stage_runner.py::recon_archive_root). That is the '
            'root the orchestrator and the legibility miner already use. Two '
            'hooks, because neither is enough alone:'
            '\n\n'
            'The PRODUCER hook, BaseStage._archive_stage_transcripts, runs in '
            'the finally of BaseStage.run and COPIES each stage transcript as '
            'the stage exits, cancellation included. It copies rather than '
            'moves because the σ --resume seam must still read the source. It '
            'is what covers interrupted runs, which skip the terminal GC by '
            'design.'
            '\n\n'
            'The TEARDOWN hook, gc_run_config_dir, calls archive_before_delete, '
            'so deletion is conditional on archival: each transcript is moved '
            'and unlinked only once its durable copy exists. Both harness call '
            'sites pass the resolved root; archive_root=None is the disabled '
            'branch, and only it keeps the plain rmtree.'
            '\n\n'
            'Not covered: a transcript that archive_before_delete HELD after a '
            'failed move, and a stage killed by SIGKILL before its finally ran. '
            'Both stay on disk with nothing to collect them, because '
            'fused-memory has no boot-time orphan sweeper (task 4906, filed '
            'from task 3972).'
        ),
        'follow_up': (
            'CLOSED by task 3972; recorded UNARCHIVED_GAP by task 3271. The '
            'remaining SIGKILL / held-transcript tail is task 4906.'
        ),
    },
    # -------------------------------------------------------------------- 4
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
    # -------------------------------------------------------------------- 5
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
            (
                'orchestrator/src/orchestrator/workflow.py',
                'TaskWorkflow._archive_then_cleanup_config_dir',
            ),
        ),
        'rationale': (
            'The main per-task config dir (~:2390). Covered three ways as of '
            'task 3619. The TaskWorkflow._invoke producer hook archives each '
            'finished session; GitOps.cleanup_worktree\'s teardown backstop '
            'sweeps the whole dir at cold worktree removal; and '
            'TaskWorkflow._archive_then_cleanup_config_dir archives at the '
            'config-dir teardown itself. _shutdown_steps orders '
            'cleanup_done_worktree BEFORE cleanup_config_dir, so the archiving '
            'path runs first. This is the reference shape the '
            'run_dry_run_unblock hook was modelled on.'
            '\n\n'
            'The third entry is the one that makes this disposition hold '
            'unconditionally, and it is a different KIND of coverage from the '
            'other two. Both older hooks are skippable — the producer hook can '
            'be lost to a CancelledError at SIGTERM or to an invocation that '
            'never reached its finally, and the cleanup_worktree backstop is '
            'reached only on COLD removals (the warm- and spec-lane early '
            'returns fire above it; see the NOTES below). The rmtree in '
            '_archive_then_cleanup_config_dir is not skippable, because task '
            '3619 made archival a PRECONDITION of that delete rather than a '
            'step scheduled before it. The observed failure it closes: a run '
            'that preserved the session sidecar and destroyed the transcript '
            'the sidecar pointed at, leaving --resume with a dangling id.'
        ),
    },
    # -------------------------------------------------------------------- 6
    {
        'path': 'orchestrator/src/orchestrator/workflow.py',
        'qualname': 'TaskWorkflow._recycle_config_dir',
        'destroyed_by': 'self._config_dir.cleanup() one line earlier',
        'disposition': ARCHIVED,
        'archives_in': (
            (
                'orchestrator/src/orchestrator/workflow.py',
                'TaskWorkflow._archive_then_cleanup_config_dir',
            ),
        ),
        'rationale': (
            'Wedged-session recycle (~:8311). CLOSED by task 3619; recorded '
            'UNARCHIVED_GAP by task 3271 for the reasons kept below.'
            '\n\n'
            'The gap was: cleanup() rmtree\'d the WHOLE dir and a fresh '
            'TaskConfigDir was reconstructed at the identical path on the next '
            'line. Discarding the wedged session is the intent, but the blast '
            'radius was the whole dir — any not-yet-flushed sibling (e.g. '
            '`<sid>/subagents/agent-*.jsonl` from a prior healthy invocation) '
            'went with it, and the wedged session\'s own transcript is exactly '
            'what a legibility miner would want in order to explain the wedge. '
            'It was left unfixed because workflow.py is a hot, heavily-locked '
            'file well outside task 3271\'s scope, and the right fix was an '
            'archival call before the cleanup rather than a change to the '
            'recycle semantics.'
            '\n\n'
            'That is what landed: the bare cleanup() here became a call to '
            'TaskWorkflow._archive_then_cleanup_config_dir, the single '
            'teardown primitive this function and _cleanup_config_dir '
            'now share — so neither can acquire the archival guard while the '
            'other quietly keeps deleting. The recycle semantics are unchanged; '
            'only what is provably durable is unlinked.'
            '\n\n'
            'Why archives_in names the HELPER and not this function: the gate '
            'scans for NAME references, not a call graph, so archival reached '
            'through a `self._archive_then_cleanup_config_dir()` hop is '
            'credited where the archiver is actually named. The caution that '
            'used to sit here, that this site shares a module with '
            'TaskWorkflow._setup_worktree_and_artifacts, still holds '
            'and still matters — the gate checks the enclosing FUNCTION, so a '
            'module-level import cannot launder a future regression here into a '
            'false ARCHIVED. What it means in the other direction is that this '
            'entry\'s green rests on the helper existing and on this function '
            'routing through it. The gate proves the first; '
            'orchestrator/tests/test_transcript_archive_producer_hook.py::'
            'TestCleanupConfigDirArchivesFirst::'
            'test_recycle_also_archives_before_it_destroys proves the second by '
            'planting a transcript, calling _recycle_config_dir(), and reading '
            'it back out of the archive. Neither half is sufficient alone.'
        ),
        'follow_up': (
            'CLOSED by task 3619 (leaf 2 of '
            'plans/transcript-preservation-seam-prd.md), which routed this '
            'site through TaskWorkflow._archive_then_cleanup_config_dir. No '
            'open follow-up remains; retained as the audit trail for a gap '
            'that task 3271 filed and 3619 discharged.'
        ),
    },
    {
        'path': 'fused-memory/src/fused_memory/middleware/task_curator.py',
        'qualname': 'TaskCurator._transcript_config_dir',
        'destroyed_by': (
            'TaskCurator.close at clean shutdown; in a later process, '
            'task_curator.py::_sweep_stale_curator_config_dirs_once -> '
            'config_dir.py::sweep_stale_pid_dirs with '
            'reclaim=_reclaim_by_archiving (owner pid dead, dir >=300s old); '
            'cleanup_at_exit=False'
        ),
        'disposition': ARCHIVED,
        'archives_in': (
            (
                'fused-memory/src/fused_memory/middleware/task_curator.py',
                '_archive_curator_config_dir',
            ),
        ),
        'rationale': (
            'Per-pid curator dir, built only on the gated path (usage_cap '
            'enabled, i.e. production). ARCHIVED as of task 6008, into '
            "recon_archive_root's root (task 3972's decision, reused through "
            'TaskCurator._transcript_archive_root) under the key '
            'fm-curator-<pid>. Both hooks, TaskCurator.close and the dead-pid '
            "sweep's reclaim step _reclaim_by_archiving, route through "
            '_archive_curator_config_dir -> archive_before_delete, so deletion '
            'is licensed by archival rather than scheduled after it. A held '
            'transcript leaves its dead-pid dir in place, so the next '
            "process's sweep retries it. The kill switch "
            'reconciliation.transcript_archive_enabled restores the plain '
            'rmtree: the sweep removes the dir unarchived and close leaves it '
            'for the sweep.'
            '\n\n'
            'Not covered: a process SIGKILLed whose dir is wiped by a boot-time '
            '/tmp clean before any later sweep runs.'
        ),
        'follow_up': (
            'CLOSED by task 6008; recorded UNARCHIVED_GAP by task 3995 '
            '(esc-3995-8).'
        ),
    },
    {
        'path': 'scripts/legibility/session_runner.py',
        'qualname': 'SessionRunner.__init__',
        'destroyed_by': (
            'session_runner.py::SessionRunner.close -> TaskConfigDir.cleanup on '
            'every exit path (nightly run_nightly try/finally, census.main and '
            'coder.main `with`); cleanup_at_exit=True atexit rmtree; '
            'session_runner.py::_sweep_stale_session_config_dirs_once -> '
            'config_dir.py::sweep_stale_pid_dirs in a later process (owner pid '
            'dead)'
        ),
        'disposition': UNARCHIVED_BY_DESIGN,
        'rationale': (
            'Per-process legibility trickle / census / coder-CLI dir (task '
            '6042). Trickle digests are one-shot classifier calls with no '
            'tools, and census mining and synthesis are classifiers; their '
            'product (the coding record, the census verdict) is persisted by '
            'the caller. Census verify calls are bounded read-only '
            'explorations whose only product is the per-cluster verdict that '
            'the census records. Archiving would also be harmful, not just '
            'redundant: scripts/legibility/inventory.py walks the archive '
            'roots as well as ~/.claude/projects, so archived sessions, which '
            'quote codebook clusters by construction, would be fed back into '
            'the corpus the trickle codes. Retention changed knowingly: before '
            'task 6042 the text-mode `claude -p` left these sessions under the '
            'operator\'s ambient ~/.claude/projects indefinitely; they are now '
            'deleted at close.'
        ),
    },
)


# ---------------------------------------------------------------------------
# Module-level notes — NOT site entries (none constructs a TaskConfigDir),
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

    """ReconciliationHarness._resume_guard_reason: a REMOVED site, recorded
    here so its entry is not re-added.

    Task 3271 audited it as a per-run UNARCHIVED_GAP site. It was really a
    read-only resume-eligibility probe that built a TaskConfigDir only to
    derive a path. TaskConfigDir.__init__ mkdirs and symlinks, so the probe
    re-created the very dir it was probing for, resurrecting runs that had
    already been GC'd, with nothing left to GC them again. Task 3972 replaced
    the construction with the pure derivation
    cli_stage_runner.py::run_config_dir_path, so the function no longer builds
    a config dir and has no entry above.""",
)
