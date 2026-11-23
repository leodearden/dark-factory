"""Audited disposition for every loop-blocking call site in ``fused-memory/src``.

Pure data + rationale, no scanning logic -- mirroring
``silent_fallthrough_allowlist.py`` and ``config_dir_archival_allowlist.py``.
``test_loop_blocking_gate.py`` holds the assertions; ``loop_blocking_scan.py``
holds the detector.

THIS FILE IS THE COMPLETE PER-SITE RECORD of task 4484's caller-side INV-8
census.  ``plans/inv8-caller-side-census-2026-09-03.md`` carries the analysis
and the cluster-level triage and POINTS here; it deliberately does not restate
these rows (INV-9, one-fact-one-home).  There is no ``.json`` twin: the ratchet
already requires this list to exist as code, and a second copy of the same rows
would be an independent copy of a machine-read fact with nothing reconciling
them -- it would have gone stale the moment task 4201 landed and removed two rows,
while this file's staleness is caught by
``TestRatchet::test_no_stale_blessings``.

Why this file exists
--------------------
Task 3778 censused INV-8 by enumerating where the blocking PRIMITIVE is
written, then made a per-MODULE offload claim: ``recon_claim_verification_guard.py``
was recorded as "already offloaded at its call sites".  That was true of the
callers it looked at and false of the others, so a module with one offloaded
caller read as clean and the rest were invisible.  Tasks 4091 and 4201 each
later found live sites in that blind spot, and task 4484 found two more in
``task_curator.py`` that nobody had filed at all.

INV-8's own *Census seam* section names the remedy for a slug missed across
repeated census batches: file a guard task.  This is the guard's ledger.  A
site that reaches a blocking primitive from a coroutine either gets fixed or
gets a row here saying WHY it is tolerable -- there is no third option, and a
row whose reason is "existing" is a silent waiver that defeats the point.

Entry schema
------------
Each entry is a 6-tuple::

    (relpath, qualname, content_hash, disposition, owner, justification)

``relpath``       repo-relative posix path of the CALLING module
``qualname``      dotted qualname of the enclosing coroutine
``content_hash``  ``sha256(ast.unparse(call_node))[:12]`` -- recompute with the
                  scanner when the call changes
``disposition``   one of :data:`DISPOSITIONS`
``owner``         who will remove the row, of the type
                  :data:`OWNER_TYPE_BY_DISPOSITION` fixes for its disposition:
                  ``None`` for ``accepted``, ``TaskRef(<task>)`` for ``filed``,
                  ``TicketRef('tkt_...')`` for ``to_file``.
                  ``test_every_entry_owner_matches_its_disposition`` enforces
                  the type.  A ``to_file`` row's ticket is the one task 4484
                  step-9 filed for it: ``submit_task`` returns a ticket, not a
                  task id, because the curator decides create / combine / drop
                  asynchronously, so no task id exists at filing time.  A
                  ticket-owned row therefore stays ``to_file`` with its
                  ``TicketRef`` until the ticket resolves, then flips to
                  ``filed`` with the resolved task's ``TaskRef`` -- a ``filed``
                  row naming a ticket would point at a decision that has not
                  been made yet.
``justification`` non-empty prose saying WHY.  It may narrate how the row
                  came to its owner (the ticket a task grew from), but never
                  says who owns it now: the ``owner`` column is the only home
                  of that, and the only one the gates check.
                  ``accepted`` must say what makes the cost acceptable (cached
                  / startup-only / measured cheap), not merely that the call
                  exists.

Keys are ``(relpath, qualname, content_hash)`` -- deliberately NOT line
numbers, which drift on every unrelated edit above the site and would make
this a nuisance ratchet reviewers learn to re-bless unread.  The hash is
invariant under reindentation and edits above the site, and changes only when
the call itself changes.

DELETING A ROW IS PART OF THE FIX THAT REMOVES ITS SITE.  ``tests/repo_guards``
is never cached (``verify_cache: never``), so every merge gate runs
``test_loop_blocking_gate.py::TestRatchet::test_no_stale_blessings``, which fails on a
blessing whose site is gone -- so a landed fix that leaves its row behind reds
verify for every subsequent task in the repo until someone edits THIS file.
A ``filed`` row's owning task need not otherwise be editing ``tests/repo_guards``;
that coupling is written out in
``plans/inv8-caller-side-census-2026-09-03.md`` section 7.  The fix is always a
deletion, never a re-bless.

EVERY OWNED ROW HAS AN INV-12 OVERRIDE ELSEWHERE.  Each ``filed`` or
``to_file`` row has a literal override in
``tests/repo_guards/test_repo_guards_governed_lists.py::LOOP_BLOCKING_LEDGER``,
keyed ``relpath::qualname::content_hash`` (``#<n>`` on the n-th repeat of an
identical row) and equal to ``Debt(<owner>)``.  ``accepted`` rows have no
override: they borrow its default, counted by ``default_covers``.  Re-hashing
or deleting an owned row means re-keying or deleting its override IN THE SAME
CHANGE, and adding or deleting an ``accepted`` row means adjusting
``default_covers``; otherwise collecting ``tests/repo_guards`` fails inside
``governed_exceptions`` (a ``MalformedDeclaration`` for a stray override key).

MULTISET, not set.  ``reconcile_against_allowlist`` compares with
``Counter`` subtraction, so a function with two byte-identical blocking calls
needs two rows.  This is load-bearing for THIS gate specifically: set
membership would let a second site inside an already-blessed function pass
silently, which is a per-function restatement of exactly the per-module
blindness that produced the misses above.

ROW PER SITE, TRIAGE PER CAUSE.  Where a cluster shares one root cause -- the
22 MCP handlers reaching the cached ``resolve_main_checkout``, every async
caller of ``ReconciliationHarness._escalate`` -- every row carries the SAME
justification naming that shared cause and its single follow-up -- held in
one named constant where a cluster's text has had to change after filing
(``_ESCALATE_ARCHIVE_SCAN_WHY``), so the next edit lands once.  Every
multi-row cluster added from task 5099 on, and any cluster whose text
changes, holds its justification in one such named constant; an untouched
cluster keeps its inline copies until the task owning its rows deletes them.
Such a constant states the cause and its cost, never a row count or a site
list: the rows referencing it are the membership, and a copy in prose would
go stale unchecked the first time a row joins or leaves.
The ledger stays row-per-site so a 23rd handler cannot be added silently under
a blessed 22; the triage stays cluster-per-defect so the follow-ups are one
task per defect rather than one per line.

Regenerating
------------
There is no generation timestamp here, deliberately: re-running the scan and
diffing this file is then a meaningful reproducibility check rather than a
guaranteed diff.  To re-derive the rows, run from the repo root::

    uv run --project shared pytest tests/repo_guards/test_loop_blocking_gate.py -q

A failure names every site whose disposition is missing or stale.

Baseline measured at HEAD 6696f1ce0c: 167 files scanned, 60 findings.
Task 4484's amendment pass re-measured 62 over the same 167 files: one
row withdrawn as a scanner false positive (create_mcp_server.
_completion_claim_gate -> verify_claims; its sibling row,
create_mcp_server._claim_commit_presence, was then fixed and deleted by
task 4853) and three added by widening the vocabulary to shutil.rmtree.  Task 5099 re-measured at HEAD b4e1349e1c
after widening it to the directory-walk/metadata calls: 190 files scanned,
91 findings (55 before; census section 4c).  Task 6089 re-measured at HEAD
d8b04e7aaa after widening it to Path.resolve / os.path.realpath: 290 files
scanned, 90 findings (80 before; census section 4d).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType, NoneType

from shared.governed_exceptions import TaskRef, TicketRef

#: The fixed disposition vocabulary, each with the type its rows' owner must
#: have; an accepted row has no owner and borrows the INV-12 default.  No
#: inline ``# inv8-ok:`` source-comment marker exists as an alternative: this
#: ledger has to exist anyway for the ratchet, so a marker at the site would be
#: a second home for the same fact with nothing reconciling them (INV-9) -- and
#: writing one would mean editing the very runtime files tasks 4201 and 3778
#: were holding in the merge lane.  A reader at a site follows
#: test_loop_blocking_gate.py to the row that says why.
OWNER_TYPE_BY_DISPOSITION: Mapping[str, type[TaskRef] | type[TicketRef] | type[None]] = (
    MappingProxyType({
        'accepted': NoneType,  # cheap, cached or startup-only (say WHICH); no owner
        'filed': TaskRef,      # the task that owns it
        'to_file': TicketRef,  # the confirmed defect's task 4484 step-9 ticket
    })
)
DISPOSITIONS = frozenset(OWNER_TYPE_BY_DISPOSITION)

#: The shared justification of every ``ReconciliationHarness._escalate`` row.
_ESCALATE_ARCHIVE_SCAN_WHY = (
    'ROOT CAUSE (one defect, 10 rows): the sync '
    'ReconciliationHarness._escalate reaches '
    '_finding_recently_resolved, which read_texts EVERY escalation '
    'record under the queue root AND its archive -- a fan-out on '
    'the loop thread whose cost grows with queue history, so this '
    'trips INV-8\'s fan-out limb as well as its blocking limb. '
    '10 async callers, one fix (offload _escalate, or pre-fetch '
    'resolved_fps once per cycle -- the resolved_fps kwarg already '
    'exists for exactly that). PARTIALLY ADDRESSED BY TASK 5550: '
    '_run_remediation_pass builds its per-pass resolved_fps in a '
    'worker thread, and the fallback scan moved into '
    'reconciliation/escalation_archive.py behind a one-slot memo '
    'keyed on run_id, so consecutive _escalate calls within one run '
    'share one walk. That bounds REPETITION only while no other '
    "run's _escalate interleaves (another key evicts the slot), and "
    'every miss still walks the whole archive inline: _escalate is '
    'still sync -- which is why these ten rows stay `filed` with '
    'their original hashes rather than being deleted as fixed. '
    'The helper deliberately '
    'lives under fused-memory/src/ so the scanner can still follow '
    'that reach; see TestScanHelperStaysInGateScope. The owner '
    'owns the full offload -- do not file '
    'again: task 4484 step-9\'s ticket '
    'tkt_0RT7QXR5AXGWVADW9T3S4DPC2M became task 5072, coalesced '
    'into 5270. 5072\'s text counts 9 callers; the 10th, '
    '_maybe_remediate\'s phantom-citation storm alarm, landed later '
    'with task 4781.'
)

#: Task 5099: an idempotent mkdir of a store's own data dir as it opens.
_MKDIR_ON_STORE_OPEN_WHY = (
    'ACCEPTED, measured cheap (one cause): one idempotent '
    'mkdir(parents=True, exist_ok=True) of the store\'s own data dir as it '
    'opens its SQLite connection -- measured 6-11us on an existing dir '
    '(task 5099: 20000-call timeit, CPython 3.13.9). A store opener pays it '
    'once per store lifetime: at startup, or once per project behind a '
    'cached connection. A per-MCP-call DB opener pays it immediately before '
    'an awaited connect_daemon(...) that already hops threads and costs far '
    'more. Only the directory is created inline; every read and write of '
    'the store itself is awaited.'
)

#: Task 5099: a fixed number of existence / type guards per invocation.
_FIXED_STAT_GUARD_WHY = (
    'ACCEPTED, measured cheap (one cause): a fixed, data-independent '
    'number of stat calls per invocation -- an existence or type guard in '
    'front of the real work, never a scan -- measured 3-4us each against a '
    'warm dentry cache (task 5099: 20000-call timeit of exists/stat/is_dir, '
    'CPython 3.13.9). The work each guard fronts is awaited and costs '
    'orders of magnitude more (a SQLite connect, a subprocess, an LLM '
    'call). The count is set by the code or an operator-authored config '
    'list, never by what is on disk.'
)

#: Task 5099: one exists() per metadata.files entry a task author declared.
_DECLARED_FILES_STAT_WHY = (
    'ACCEPTED, bounded by an authored list (one cause): '
    'middleware/task_interceptor.py::_missing_files does one exists() per '
    'metadata.files entry the task author declared -- 3-4us each (task '
    '5099: 20000-call timeit, CPython 3.13.9), so a 20-file task costs '
    '~0.1 ms. Its callers reach it only on a status transition, never per '
    'read.'
)

#: Task 5099: the curator COMBINE audit line; task 8415 added its item fields.
_COMBINE_AUDIT_APPEND_WHY = (
    'ACCEPTED, bounded by an LLM decision (one cause): '
    'TaskInterceptor._execute_combine reaches _append_combine_audit, an '
    'idempotent mkdir (6-11us, task 5099 timeit) plus one open(\'a\') '
    'append of a single JSON line, with no fsync. The line is about 2 KB '
    '(descriptions and justification truncated to 500 chars) plus, since '
    'task 8415, the items_before/items_after lists: one 16-hex item key '
    'per follow-up item on the target, so roughly 40 bytes per item. It '
    'runs once per curator COMBINE decision, '
    'each of which follows an LLM call of seconds, so the append is never '
    'on a hot or storm-coupled path. Since task 8416 two batch-edge sites '
    'reach it too: TaskInterceptor._resolve_batch_edge (one refusal row per '
    'within-batch combine whose sibling was not created) and '
    'TaskInterceptor._audit_drop_onto_sibling (one row per within-batch drop '
    'onto a sibling task), so at most one append per batch edge, each after '
    'the same batch LLM call. The one storm-coupled defect of this '
    'shape, the EventQueue dead-letter append that fires during a drop '
    'storm, was moved onto a dedicated writer thread by task 6085.'
)

#: Task 6089: what the owner of the measured-cheap ``resolve`` rows its
#: widening added must do with them.
_CHEAP_RESOLVE_OWNER_WHY = (
    ' Task 6089\'s follow-up ticket tkt_0RW94QBGG9RSRP47XQXGGV1KFK became '
    'task 7572. Under the INV-12 day-one ruling '
    '(esc-5603-1) a measured-cheap site is no longer a resting state in this '
    'ledger, so the owner offloads it (asyncio.to_thread, or anyio.Path.resolve '
    'for a Path) and deletes the row, or obtains a per-row Policy ruling.'
)

#: Task 6089: one canonicalising resolve() in front of an awaited open.
_PATH_RESOLVE_GUARD_WHY = (
    'ROOT CAUSE (one debt, measured cheap): one Path.resolve() per '
    'invocation of a single root or database path -- never a walk over what '
    'a directory holds -- canonicalising it immediately before awaited work '
    'that costs far more (an aiosqlite or connect_daemon open). Its cost '
    '(census section 4d) scales with the path\'s component count, which is '
    'set by the path the code, configuration or caller supplies, not by '
    'what is on disk.'
    + _CHEAP_RESOLVE_OWNER_WHY
)

#: The shared justification of every server/tools.py ``resolve_main_checkout`` row.
_MAIN_CHECKOUT_COLD_MISS_WHY = (
    'ROOT CAUSE (one defect): an MCP handler taking a project_root calls '
    'the nested sync create_mcp_server._normalize_project_root, which '
    'reaches models/scope.py::resolve_main_checkout -> '
    'subprocess.run(git). The result is memoised in '
    '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
    'loop thread -- exactly the caveat task 3778\'s own census recorded '
    '("cached but a cold miss runs on the loop thread") and then did '
    'not act on. One fix closes every row, so no handler is offloaded '
    'alone. The owner (S15 α0) offloads every inline '
    '_normalize_project_root call with asyncio.to_thread and deletes these '
    'rows as fixed. Provenance only: task 4484 step-9\'s ticket '
    'tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ created task 5071, which was '
    'coalesced into 5270; 5270 handed this cluster to 9822 on 2026-11-14.'
)

#: Task 6089: run_server canonicalising configured paths before it serves.
_STARTUP_PATH_RESOLVE_WHY = (
    'ROOT CAUSE (one debt, startup only): run_server canonicalises a '
    'configured path with Path.resolve() once per process, while it builds '
    'the server and before server.serve() binds, so no request is in flight '
    'to be delayed. A background loop run_server has already started (the '
    'reconciliation harness, the event queue) can be stalled for that one '
    'resolve (cost: census section 4d), once per process lifetime. If it '
    'ever moves onto a request or reload path, '
    'its content_hash changes and the gate re-asks.'
    + _CHEAP_RESOLVE_OWNER_WHY
)

#: The shared justification of both ``gc_run_config_dir`` rows in harness.py.
_RUN_CONFIG_DIR_GC_WHY = (
    'ROOT CAUSE (one defect, 2 rows): two harness coroutines call the '
    'sync cli_stage_runner.py::gc_run_config_dir inline. The scanner '
    'credits only its shutil.rmtree of the per-run agent config dir -- one '
    'unlink syscall per file, all on the loop thread. Found once task '
    '4484\'s amendment pass added shutil.rmtree to the vocabulary. '
    'UNDERSTATED BY THESE ROWS since task 3972: both calls now pass '
    'archive_root=recon_archive_root(self.config), so with archival enabled '
    'gc_run_config_dir first runs shared.transcript_archive.'
    'archive_before_delete, which copies each stage transcript to a staging '
    'file, os.replaces it into the archive and unlinks the source. That '
    'I/O lives under shared/src, outside the scanner\'s fused-memory/src '
    'scope, so it does not show up as a primitive here. Task 3972 re-keyed '
    'these rows for the new call text. CAUTION FOR THE OFFLOAD: '
    'gc_run_config_dir is documented as synchronous by contract, because an '
    'await is a cancellation point and SIGTERM-driven teardown is exactly '
    'what would then skip the archive-then-delete. An offload must keep the '
    'teardown unskippable. Do not file again: task 4484\'s ticket '
    'tkt_0RT88VW7RRECTHNCVTJXD6M5RJ was combined into task 5072, which '
    'was coalesced into 5270. '
    'NOT A ROW, NOT PART OF THESE ROWS\' FIX: the producer half of the same archival runs '
    'on the loop thread too, in stages/base.py::'
    'BaseStage._archive_stage_transcripts from BaseStage.run\'s finally on '
    'every stage exit -- shared.transcript_archive.archive_task_transcripts '
    'stats each per-run transcript and staged-copies each changed one. The '
    'scanner cannot see it for the same shared/src reason, and stages/base.py '
    'is outside 5270\'s file scope. ACCEPTED, synchronous for the same '
    'cancellation-safety reason, on task 3972\'s measurement over a copy of '
    'the largest live recon run dir (5.0 MB, 2026-10-13, load average '
    '56-170): a full first copy took 10 ms median / 13.5 ms max, a repeat '
    '(mtime-equality skip) 0.5 ms.'
)

#: ``(relpath, qualname, content_hash, disposition, owner, justification)``.
AUDITED_SITES: list[tuple[str, str, str, str, TaskRef | TicketRef | None, str]] = [

    # ---- backends/sqlite_task_backend.py ----
    (
        'fused-memory/src/fused_memory/backends/sqlite_task_backend.py',
        'SqliteTaskBackend._get_write_access',
        '14c6bfa75f63',
        'accepted',
        None,
        _MKDIR_ON_STORE_OPEN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/backends/sqlite_task_backend.py',
        'SqliteTaskBackend.get_statuses_fresh',
        '09007f79712d',
        'accepted',
        None,
        _FIXED_STAT_GUARD_WHY,
    ),

    # ---- maintenance/seed_autopilot_video_triage_guardrails.py ----
    (
        'fused-memory/src/fused_memory/maintenance/seed_autopilot_video_triage_guardrails.py',
        'SeedManager.seed',
        '43bbc3406b55',
        'accepted',
        None,
        'ACCEPTED, no server loop to stall: SeedManager.seed reaches '
        'load_guardrail_payloads\'s exists() inside a one-shot maintenance '
        'CLI driven by asyncio.run in its __main__ block. Nothing under '
        'fused-memory/src imports the module, so the only event loop it '
        'ever runs on is its own, with no concurrent work to block.',
    ),

    # ---- mcp_tools/scheduler_state.py ----
    (
        'fused-memory/src/fused_memory/mcp_tools/scheduler_state.py',
        'read_scheduler_events',
        '09007f79712d',
        'accepted',
        None,
        _FIXED_STAT_GUARD_WHY,
    ),
    (
        'fused-memory/src/fused_memory/mcp_tools/scheduler_state.py',
        'read_scheduler_events',
        '14947dd93734',
        'filed',
        TaskRef(7572),
        _PATH_RESOLVE_GUARD_WHY,
    ),

    # ---- middleware/task_interceptor.py ----
    (
        'fused-memory/src/fused_memory/middleware/task_interceptor.py',
        'TaskInterceptor._apply_status_transition',
        'c387b27712b9',
        'accepted',
        None,
        _DECLARED_FILES_STAT_WHY,
    ),
    (
        'fused-memory/src/fused_memory/middleware/task_interceptor.py',
        'TaskInterceptor._execute_combine',
        '0a7e982628a1',
        'accepted',
        None,
        _COMBINE_AUDIT_APPEND_WHY,
    ),
    (
        'fused-memory/src/fused_memory/middleware/task_interceptor.py',
        'TaskInterceptor._execute_combine',
        'ab64655040c9',
        'accepted',
        None,
        _COMBINE_AUDIT_APPEND_WHY,
    ),
    (
        'fused-memory/src/fused_memory/middleware/task_interceptor.py',
        'TaskInterceptor._resolve_batch_edge',
        '36be88f72819',
        'accepted',
        None,
        _COMBINE_AUDIT_APPEND_WHY,
    ),
    (
        'fused-memory/src/fused_memory/middleware/task_interceptor.py',
        'TaskInterceptor._audit_drop_onto_sibling',
        '1c4cefeb98fb',
        'accepted',
        None,
        _COMBINE_AUDIT_APPEND_WHY,
    ),

    # ---- middleware/task_curator.py ----
    (
        'fused-memory/src/fused_memory/middleware/task_curator.py',
        'TaskCurator._invoke_curator',
        '2be781b317a0',
        'filed',
        TaskRef(7572),
        'ROOT CAUSE (one debt, bounded by an LLM call): _invoke_curator reaches '
        '_resolve_curator_prompt -> self._prompt_store.resolve(...). The '
        'scanner matches the method name resolve, but the receiver is '
        'shared.prompt_artifact.PromptArtifactStore.resolve, not '
        'Path.resolve. That code lives under shared/src, outside the '
        'scanner\'s fused-memory/src scope, and it does block: unpinned it is '
        'one exists() probe of the key\'s heuristics.txt (cost: census '
        'section 4d); pinned it adds a JSON provenance load and a '
        'heuristics.txt read. '
        'PromptArtifactStore.resolve\'s own docstring states this I/O and '
        'the once-per-invocation granularity it expects. It runs once per '
        'curator LLM round-trip, immediately before the awaited self._invoke '
        'of seconds -- the same bound the COMBINE audit append rests on.'
        + _CHEAP_RESOLVE_OWNER_WHY,
    ),

    # ---- middleware/ticket_store.py ----
    (
        'fused-memory/src/fused_memory/middleware/ticket_store.py',
        'TicketStore.initialize',
        'f485b909e349',
        'accepted',
        None,
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- reconciliation/cli_stage_runner.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py',
        'run_stage_via_cli',
        'aa09bf369631',
        'accepted',
        None,
        _FIXED_STAT_GUARD_WHY,
    ),

    # ---- reconciliation/event_queue.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/event_queue.py',
        'EventQueue.start',
        'd325858ad5fe',
        'accepted',
        None,
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- reconciliation/harness.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._recover_one_run',
        'f05da1451b63',
        'filed',
        TaskRef(5270),
        _RUN_CONFIG_DIR_GC_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._persist_stage_reports_then_gc_config_dir',
        'f05da1451b63',
        'filed',
        TaskRef(5270),
        _RUN_CONFIG_DIR_GC_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._recover_stale_runs',
        '0787c60051a4',
        'filed',
        TaskRef(5270),
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._recover_stale_runs',
        'cc7999e2d4b6',
        'filed',
        TaskRef(5270),
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._record_failed_run',
        '652fc88d57ad',
        'filed',
        TaskRef(5270),
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_remediate',
        '1fbc50761be3',
        'filed',
        TaskRef(5270),
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_remediate',
        '51ec3ffac666',
        'filed',
        TaskRef(5270),
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_remediate',
        '8c9728223345',
        'filed',
        TaskRef(5270),
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        'c5a47e52ad4b',
        'to_file',
        TicketRef('tkt_0RT7RJKQ0WXB0T87F8TJ7RTGQH'),
        'ROOT CAUSE (one defect, 2 rows): _run_remediation_pass reads the '
        'orchestrator state files inline on the loop thread '
        '(read_scheduler_state -> read_bytes and orchestrator_started_at '
        '-> read_text). Filesystem, the limb task 3778\'s subprocess-only '
        'vocabulary omitted; distinct from the _escalate cluster in the '
        'same file. Follow-up filed by task 4484 step-9. A third row '
        '(ae4cd95f45ad) was counted here until task 5550 and did not '
        'belong: the scanner placed it at the escalation-archive read_text '
        'in the resolved_fps build, not at an orchestrator state file. '
        '5550 offloaded that scan via asyncio.to_thread, the finding went '
        'stale, and the row was deleted with the fix -- these two survive '
        'unfixed.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        '9e5ae6eb2503',
        'to_file',
        TicketRef('tkt_0RT7RJKQ0WXB0T87F8TJ7RTGQH'),
        'ROOT CAUSE (one defect, 2 rows): _run_remediation_pass reads the '
        'orchestrator state files inline on the loop thread '
        '(read_scheduler_state -> read_bytes and orchestrator_started_at '
        '-> read_text). Filesystem, the limb task 3778\'s subprocess-only '
        'vocabulary omitted; distinct from the _escalate cluster in the '
        'same file. Follow-up filed by task 4484 step-9. A third row '
        '(ae4cd95f45ad) was counted here until task 5550 and did not '
        'belong: the scanner placed it at the escalation-archive read_text '
        'in the resolved_fps build, not at an orchestrator state file. '
        '5550 offloaded that scan via asyncio.to_thread, the finding went '
        'stale, and the row was deleted with the fix -- these two survive '
        'unfixed.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        'edd9a8720cf9',
        'filed',
        TaskRef(5270),
        'ROOT CAUSE: _file_finding_task_escalation, the orchestrator-queue '
        'counterpart of _escalate that task 4821 added after this census '
        'ran, is sync end to end on the loop thread -- the '
        'is_orchestrator_lock_held flock probe (an open plus a '
        'non-blocking LOCK_SH; task 8015), EscalationQueue\'s '
        'mkdir, a read_text of every pending record in the queue root '
        'for the category fold (so it grows with the pending queue), '
        'then make_id\'s fcntl.flock over an fsync\'d counter write and '
        'submit\'s fsync\'d record write. The flock wait is bounded only '
        'by another process\'s hold. Do not file again: added to '
        'task 5270\'s scope 2026-09-16 '
        'beside the _escalate cluster, whose fix shape (offload the '
        'filer) it shares.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        'a4c939b65449',
        'filed',
        TaskRef(5270),
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        'ec953752e2f1',
        'filed',
        TaskRef(5270),
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),

    # ---- reconciliation/journal.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/journal.py',
        'ReconciliationJournal.initialize',
        '863ff109dbaf',
        'accepted',
        None,
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- reconciliation/recon_ledger.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/recon_ledger.py',
        'ReconLedgerStore.initialize',
        'f485b909e349',
        'accepted',
        None,
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- reconciliation/stages/task_knowledge_sync.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py',
        '_run_briefing_known_gaps_script',
        'a23d01e12b5a',
        'accepted',
        None,
        _FIXED_STAT_GUARD_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py',
        '_run_briefing_known_gaps_script',
        '39a6fb8ac63e',
        'accepted',
        None,
        _FIXED_STAT_GUARD_WHY,
    ),

    # ---- reconciliation/stale_priority_override_edge_sweep.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/stale_priority_override_edge_sweep.py',
        'read_live_override_state',
        '7ebd882c849b',
        'to_file',
        TicketRef('tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ'),
        'SAME ROOT CAUSE as the server/tools.py::_normalize_project_root '
        'cluster: a cold miss in models/scope.py::resolve_main_checkout '
        'runs subprocess.run(git) on the loop thread; _MAIN_CHECKOUT_CACHE '
        'makes it cold-miss-only. Filed together with that cluster by task '
        '4484 step-9, since one fix closes both.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/stale_priority_override_edge_sweep.py',
        'read_live_override_state',
        '09007f79712d',
        'accepted',
        None,
        _FIXED_STAT_GUARD_WHY,
    ),

    # ---- reconciliation/targeted.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/targeted.py',
        'TargetedReconciler._sweep_cancelled_descendants',
        'b9609c7cf5b4',
        'to_file',
        TicketRef('tkt_0RT7RKWJG17W03JRC947R8FZZN'),
        '_sweep_cancelled_descendants reaches is_orchestrator_live_for, '
        'which read_texts the orchestrator state file on the loop thread. '
        'Task 3778 mentions is_orchestrator_live_for in its Part 3(a) '
        'hoisting discussion but reconciliation/targeted.py is not in its '
        'metadata.files, so this caller is nobody\'s today. Follow-up filed '
        'by task 4484 step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/targeted.py',
        'TargetedReconciler._sweep_cancelled_descendants',
        'f06e6adb6bbd',
        'accepted',
        None,
        _DECLARED_FILES_STAT_WHY
        + ' This coroutine\'s is_orchestrator_live_for row (b9609c7cf5b4) '
        'has its own owner: if that owner offloads the whole sweep, '
        'delete this row with it.',
    ),

    # ---- reconciliation/verify.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/verify.py',
        'CodebaseVerifier.verify',
        'ab04e5fd6d64',
        'accepted',
        None,
        _FIXED_STAT_GUARD_WHY,
    ),

    # ---- services/durable_queue.py ----
    (
        'fused-memory/src/fused_memory/services/durable_queue.py',
        'DurableWriteQueue.initialize',
        'c57d271d306d',
        'accepted',
        None,
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- services/live_workflow_detector.py ----
    (
        'fused-memory/src/fused_memory/services/live_workflow_detector.py',
        'detect_live_workflow',
        'b9609c7cf5b4',
        'accepted',
        None,
        'ACCEPTED, measured cheap: is_orchestrator_live_for is one '
        'read_text of the small data/orchestrator/orchestrator.lock file '
        'plus an os.kill(pid, 0) probe -- no subprocess -- measured at '
        '~41us/call (2000 calls against the live dark-factory lock, task '
        '3778 merge resolution). It is reached only as the fallback when '
        'the caller did not hoist the project-wide signal: the Live-'
        'Workflow Signals renderer threads it once per render, so this '
        'runs at most once per per-call detector use (harness gate, '
        'recon_write_policy Gate 2), not once per task in a fan-out. The '
        'row surfaced only because task 3778 made detect_live_workflow a '
        'coroutine; its git probes, the real cost, are awaited.',
    ),

    # ---- services/overreach_shadow.py ----
    (
        'fused-memory/src/fused_memory/services/overreach_shadow.py',
        'OverreachShadowStore.open_read_only',
        'fa698adde643',
        'filed',
        TaskRef(7572),
        _PATH_RESOLVE_GUARD_WHY,
    ),

    # ---- services/planned_episode_registry.py ----
    (
        'fused-memory/src/fused_memory/services/planned_episode_registry.py',
        'PlannedEpisodeRegistry.initialize',
        'c57d271d306d',
        'accepted',
        None,
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- services/write_journal.py ----
    (
        'fused-memory/src/fused_memory/services/write_journal.py',
        'WriteJournal.initialize',
        '863ff109dbaf',
        'accepted',
        None,
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- server/main.py ----
    (
        'fused-memory/src/fused_memory/server/main.py',
        'run_server',
        '06814d9a4e68',
        'accepted',
        None,
        'ACCEPTED: run_server calls configured_known_projects_map '
        '(models/scope.py -- Path.resolve() of the primary root, then '
        'open() and yaml.safe_load of each project manifest; the sweep '
        'reports whichever the walk reaches first) during process STARTUP, before the server binds '
        'and begins serving traffic, so there is no concurrent work for it '
        'to stall and no request whose latency it can affect. This is what '
        'an accepted row must say -- what makes the cost acceptable, not '
        'merely that the call exists. If this ever moves onto a request or '
        'reload path, the content_hash changes and the gate re-asks.',
    ),
    (
        'fused-memory/src/fused_memory/server/main.py',
        'run_server',
        '91313b9744f3',
        'accepted',
        None,
        'ACCEPTED, startup only: run_server reaches '
        'build_topic_cluster_store -> TopicClusterStore.open(), a sync '
        'SQLite open (mkdir, connect, schema, hydrate) matched by the method '
        'name open. It runs once per process, before server.serve() binds, '
        'so no request is in flight to be delayed; background tasks '
        'run_server has already started can be stalled for that one open. '
        'It cannot simply hop threads: the store keeps sqlite3 '
        'check_same_thread at its default because its add_memory-path '
        'reader is sync, so the connection must be opened on the loop '
        'thread that uses it. Owned with the other accepted rows by the '
        'LOOP_BLOCKING_LEDGER default (docs/legibility/inv12-day-one-rulings.md). '
        'If it ever moves onto a request or reload path, the content_hash '
        'changes and the gate re-asks.',
    ),
    (
        'fused-memory/src/fused_memory/server/main.py',
        'run_server',
        '8a884f86a26c',
        'filed',
        TaskRef(7572),
        _STARTUP_PATH_RESOLVE_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/main.py',
        'run_server',
        'f6a7ff40db1f',
        'filed',
        TaskRef(7572),
        _STARTUP_PATH_RESOLVE_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/main.py',
        'run_server',
        '43c97372205c',
        'filed',
        TaskRef(7572),
        _STARTUP_PATH_RESOLVE_WHY,
    ),

    # ---- server/tools.py ----
    (
        'fused-memory/src/fused_memory/server/tools.py',
        '_open_overrides_db',
        '14c6bfa75f63',
        'accepted',
        None,
        _MKDIR_ON_STORE_OPEN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        '_connect_overrides_db',
        '14c6bfa75f63',
        'accepted',
        None,
        _MKDIR_ON_STORE_OPEN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        '_open_park_eviction_db',
        '14c6bfa75f63',
        'accepted',
        None,
        _MKDIR_ON_STORE_OPEN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        '_checkpoint_overrides_db_if_exists',
        '051cd59d92be',
        'accepted',
        None,
        _FIXED_STAT_GUARD_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.submit_task',
        '32c0a08cd635',
        'accepted',
        None,
        _FIXED_STAT_GUARD_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_tasks',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_statuses',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_external_statuses',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_task',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.set_task_status',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.set_task_claimant',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.submit_task',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.resolve_ticket',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.list_tickets',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.search_tasks',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.commit_planning',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.update_task',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.remove_task',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.add_dependency',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.remove_dependency',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.set_task_priority_override',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.clear_task_priority_override',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.reorder_pin_queue',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_pin_queue',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_scheduler_state',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_scheduler_events',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.request_park_eviction',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.request_park_eviction',
        'a0b1b5c60eed',
        'filed',
        TaskRef(7572),
        _PATH_RESOLVE_GUARD_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.find_tasks_by_metadata',
        '81d0d2709341',
        'filed',
        TaskRef(9822),
        _MAIN_CHECKOUT_COLD_MISS_WHY,
    ),
]

#: Derived, never hand-maintained beside the rows -- a second list would drift.
ALLOWLIST_KEYS: list[tuple[str, str, str]] = [
    (relpath, qualname, content_hash)
    for relpath, qualname, content_hash, _disposition, _owner, _justification in AUDITED_SITES
]
