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
Each entry is a 5-tuple::

    (relpath, qualname, content_hash, disposition, justification)

``relpath``       repo-relative posix path of the CALLING module
``qualname``      dotted qualname of the enclosing coroutine
``content_hash``  ``sha256(ast.unparse(call_node))[:12]`` -- recompute with the
                  scanner when the call changes
``disposition``   one of :data:`DISPOSITIONS`
``justification`` non-empty prose.  ``accepted`` must say what makes the cost
                  acceptable (cached / startup-only / measured cheap), not
                  merely that the call exists.  ``filed`` must name the task id.
                  ``to_file`` names the TICKET task 4484 step-9 filed for it.
                  ``submit_task`` returns a ticket, not a task id: the curator
                  decides create / combine / drop asynchronously, so no task id
                  exists at filing time.  These rows therefore stay ``to_file``
                  rather than ``filed``, which must name a real task id --
                  ``test_filed_entries_name_their_task`` enforces that, and a
                  ``filed`` row naming a ticket would point at a decision that
                  has not been made yet.  Flip a row to ``filed`` and swap in
                  the task id once its ticket resolves.

Keys are ``(relpath, qualname, content_hash)`` -- deliberately NOT line
numbers, which drift on every unrelated edit above the site and would make
this a nuisance ratchet reviewers learn to re-bless unread.  The hash is
invariant under reindentation and edits above the site, and changes only when
the call itself changes.

DELETING A ROW IS PART OF THE FIX THAT REMOVES ITS SITE.  ``shared/tests`` is
the FIRST segment of this repo's ``test_command``
(``cd shared && uv run pytest tests/``), and
``test_loop_blocking_gate.py::TestRatchet::test_no_stale_blessings`` fails on a
blessing whose site is gone -- so a landed fix that leaves its row behind reds
verify for every subsequent task in the repo until someone edits THIS file.
A ``filed`` row's owning task need not otherwise be editing ``shared/tests``;
that coupling is written out in
``plans/inv8-caller-side-census-2026-09-03.md`` section 7.  The fix is always a
deletion, never a re-bless.

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
guaranteed diff.  To re-derive the rows::

    cd shared && uv run pytest tests/test_loop_blocking_gate.py -q

A failure names every site whose disposition is missing or stale.

Baseline measured at HEAD 6696f1ce0c: 167 files scanned, 60 findings.
Task 4484's amendment pass re-measured 62 over the same 167 files: one
row withdrawn as a scanner false positive (create_mcp_server.
_completion_claim_gate -> verify_claims; its sibling row,
create_mcp_server._claim_commit_presence, was then fixed and deleted by
task 4853) and three added by widening the vocabulary to shutil.rmtree.  Task 5099 re-measured at HEAD b4e1349e1c
after widening it to the directory-walk/metadata calls: 190 files scanned,
91 findings (55 before; census section 4c).
"""

from __future__ import annotations

#: The fixed disposition vocabulary.  No inline ``# inv8-ok:`` source-comment
#: marker exists as an alternative: this ledger has to exist anyway for the
#: ratchet, so a marker at the site would be a second home for the same fact
#: with nothing reconciling them (INV-9) -- and writing one would mean editing
#: the very runtime files tasks 4201 and 3778 were holding in the merge lane.
#: A reader at a site follows test_loop_blocking_gate.py to the row that says why.
DISPOSITIONS = frozenset({
    'accepted',  # measured cheap, cached, or startup-only -- say WHICH
    'filed',     # an existing task owns it -- justification carries the id
    'to_file',   # confirmed defect; a task 4484 step-9 ticket is named
})

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
    'that reach; see TestScanHelperStaysInGateScope. STILL OWNED '
    'BY TASK 5270, which owns the full offload -- do not file '
    'again: task 4484 step-9\'s ticket '
    'tkt_0RT7QXR5AXGWVADW9T3S4DPC2M became task 5072, coalesced '
    'into 5270. 5072\'s text counts 9 callers; the 10th, '
    '_maybe_remediate\'s phantom-citation storm alarm, landed later '
    'with task 4781.'
)

#: The shared justification of every ``reconciliation/backlog_policy.py`` row.
_BACKLOG_POLICY_RECORD_IO_WHY = (
    'ROOT CAUSE (one defect): BacklogPolicy scans, reads and '
    'writes its escalation records on the loop thread. on_judge_unhalt '
    'is_dirs data/escalations, then globs every judge_halt*.json record '
    'in the queue root -- an O(N) directory read that grows with the '
    'pending queue -- read_texts each match and reaches '
    '_restore_policy_keys; _maybe_write_escalation reaches '
    '_merge_onto_persisted; both helpers read_text then write_text the '
    'located record under escalation_id_lock. UNDERSTATED BY THESE '
    'ROWS, and recorded here because no row can carry it: the dominant '
    'blocking work on the write path is the '
    'escalation.dedupe.submit_or_dedupe call one line ABOVE the '
    '_merge_onto_persisted site -- find_dedupe_parent globs and '
    'JSON-parses every pending record in the project queue '
    '(queue.get_pending, O(N); N=41 measured on the live dark_factory '
    'queue 2026-09-18), then queue.submit writes with a durable fsync. '
    'The scanner reports only _merge_onto_persisted because those '
    'primitives live in the escalation package, across a boundary its '
    'fused-memory/src scope cannot follow -- so another row would be a '
    'blessing test_no_stale_blessings rejects, and this paragraph is '
    'the only honest place to state it. Filesystem, the limb task '
    '3778\'s subprocess-only vocabulary omitted. The record '
    'read/write rows are OWNED BY TASK 5270 -- do not file again: task '
    '4484 step-9\'s ticket tkt_0RT7RHRS9ZTJSQK328919XXEJW became task '
    '5075, coalesced into 5270. The is_dir and glob rows task 5099 '
    'added when it widened the vocabulary are TASK 6086, filed to fold '
    'into the same offload.'
)

#: The shared justification of every ``server/manifest_stamping.py`` row.
_MANIFEST_STAMPING_INLINE_IO_WHY = (
    'ROOT CAUSE (one defect): _stamp_capability_manifests_impl '
    'does its sidecar I/O INLINE in one coroutine, with no helper '
    'anywhere for a definition-side census to point at -- the shape '
    'task 3778\'s methodology is structurally blind to: an is_file() '
    'existence probe per distinct manifest path, read_text + '
    'yaml.safe_load of the sidecar, then an atomic write-back '
    '(write_text of yaml.safe_dump to a temp sibling, os.replace onto '
    'the sidecar, and the finally-block unlink of the temp). Task 4201 '
    'measured yaml.safe_load at 8.15 ms for an 11 KB document, so the '
    'parse alone is the same order as a subprocess spawn and this '
    'coroutine pays it twice plus every filesystem round trip above. '
    'One asyncio.to_thread around the whole probe-read-parse-write '
    'closes every row. The read/parse/write rows are OWNED BY '
    'TASK 5276 -- do not file again: task 4484 step-9\'s ticket '
    'tkt_0RT7QYENVS6J9WVWCY3FJAVNFR became task 5073, coalesced into '
    '5276. The is_file, os.replace and unlink rows task 5099 added '
    'when it widened the vocabulary are TASK 6087, filed to fold into '
    'the same offload.'
)

#: The shared justification of every ``CodebaseVerifier.verify`` LLM-tool row.
_VERIFIER_LLM_TOOL_IO_WHY = (
    'ROOT CAUSE (one defect): the async tools '
    'CodebaseVerifier.verify hands the codebase-verification LLM do '
    'their filesystem work inline on the loop thread, once per tool '
    'call the model chooses to make -- an LLM-driven, unbounded call '
    'count. read_file does full_path.read_text(); glob_search runs '
    'codebase_root.glob(pattern) with an LLM-CHOSEN pattern and sorts '
    'every match before keeping 50, so a "**/*" walks the whole '
    'codebase root. The read_file row is OWNED BY TASK 5270 -- do not '
    'file again: task 4484 step-9\'s ticket '
    'tkt_0RT7RM7C7NS1ECYHFBDYP02KDJ became task 5078, coalesced into '
    '5270. The glob_search row task 5099 added when it widened the '
    'vocabulary is TASK 6088, filed to fold into the same offload.'
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

#: Task 5099: the curator COMBINE audit line.
_COMBINE_AUDIT_APPEND_WHY = (
    'ACCEPTED, bounded by an LLM decision (one cause): '
    'TaskInterceptor._execute_combine reaches _append_combine_audit, an '
    'idempotent mkdir (6-11us, task 5099 timeit) plus one open(\'a\') '
    'append of a single JSON line under 2 KB (descriptions truncated to '
    '500 chars), with no fsync. It runs once per curator COMBINE decision, '
    'each of which follows an LLM call of seconds, so the append is never '
    'on a hot or storm-coupled path. The EventQueue dead-letter append is '
    'the one defect of this shape, because it fires during a drop storm.'
)

#: Task 5099: EventQueue._write_dead_letter's inline append.
_DEAD_LETTER_APPEND_WHY = (
    'ROOT CAUSE (one defect): EventQueue._write_dead_letter runs '
    'mkdir + exists + stat + the cascade rotation + open(\'a\') + write '
    'inline on the loop thread, and its async callers reach it without a '
    'hop. The enqueue overflow_drop branch fires EXACTLY when the loop is '
    'saturated, so the blocking append is coupled to the storm it '
    'records, and a rotation renames files on the same thread. '
    'UNDERSTATED BY THESE ROWS: the scanner resolves only self.enqueue, '
    'not other receivers\' event_queue.enqueue(...), so the real caller '
    'population is larger. Filed by task 5099 as TASK 6085.'
)

#: ``(relpath, qualname, content_hash, disposition, justification)``.
AUDITED_SITES: list[tuple[str, str, str, str, str]] = [

    # ---- backends/sqlite_task_backend.py ----
    (
        'fused-memory/src/fused_memory/backends/sqlite_task_backend.py',
        'SqliteTaskBackend._get_write_access',
        '14c6bfa75f63',
        'accepted',
        _MKDIR_ON_STORE_OPEN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/backends/sqlite_task_backend.py',
        'SqliteTaskBackend.get_statuses_fresh',
        '09007f79712d',
        'accepted',
        _FIXED_STAT_GUARD_WHY,
    ),

    # ---- maintenance/seed_autopilot_video_triage_guardrails.py ----
    (
        'fused-memory/src/fused_memory/maintenance/seed_autopilot_video_triage_guardrails.py',
        'SeedManager.seed',
        '43bbc3406b55',
        'accepted',
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
        _FIXED_STAT_GUARD_WHY,
    ),

    # ---- middleware/task_interceptor.py ----
    (
        'fused-memory/src/fused_memory/middleware/task_interceptor.py',
        'TaskInterceptor._apply_status_transition',
        'c387b27712b9',
        'accepted',
        _DECLARED_FILES_STAT_WHY,
    ),
    (
        'fused-memory/src/fused_memory/middleware/task_interceptor.py',
        'TaskInterceptor._execute_combine',
        '15bda1dd6b34',
        'accepted',
        _COMBINE_AUDIT_APPEND_WHY,
    ),
    (
        'fused-memory/src/fused_memory/middleware/task_interceptor.py',
        'TaskInterceptor._execute_combine',
        'd7b1b643f3fe',
        'accepted',
        _COMBINE_AUDIT_APPEND_WHY,
    ),

    # ---- middleware/ticket_store.py ----
    (
        'fused-memory/src/fused_memory/middleware/ticket_store.py',
        'TicketStore.initialize',
        'f485b909e349',
        'accepted',
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- reconciliation/backlog_policy.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py',
        'BacklogPolicy.on_judge_unhalt',
        '8f53c74a5d76',
        'filed',
        _BACKLOG_POLICY_RECORD_IO_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py',
        'BacklogPolicy.on_judge_unhalt',
        '08a103635fd7',
        'filed',
        _BACKLOG_POLICY_RECORD_IO_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py',
        'BacklogPolicy._maybe_write_escalation',
        '0d761c10e563',
        'filed',
        _BACKLOG_POLICY_RECORD_IO_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py',
        'BacklogPolicy.on_judge_unhalt',
        'cb226477b1bf',
        'filed',
        _BACKLOG_POLICY_RECORD_IO_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py',
        'BacklogPolicy.on_judge_unhalt',
        '48269c8edbbd',
        'filed',
        _BACKLOG_POLICY_RECORD_IO_WHY,
    ),

    # ---- reconciliation/cli_stage_runner.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/cli_stage_runner.py',
        'run_stage_via_cli',
        'aa09bf369631',
        'accepted',
        _FIXED_STAT_GUARD_WHY,
    ),

    # ---- reconciliation/event_queue.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/event_queue.py',
        'EventQueue.start',
        'd325858ad5fe',
        'accepted',
        _MKDIR_ON_STORE_OPEN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/event_queue.py',
        'EventQueue.recover',
        '2f9ae36b2bc7',
        'filed',
        _DEAD_LETTER_APPEND_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/event_queue.py',
        'EventQueue.close',
        'c8191bbaf10a',
        'filed',
        _DEAD_LETTER_APPEND_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/event_queue.py',
        'EventQueue._commit_with_retry',
        '93d589ee10dd',
        'filed',
        _DEAD_LETTER_APPEND_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/event_queue.py',
        'EventQueue._enqueue_on_loop',
        'f46e02910e17',
        'filed',
        _DEAD_LETTER_APPEND_WHY,
    ),

    # ---- reconciliation/harness.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._recover_one_run',
        '361d4c634750',
        'to_file',
        'ROOT CAUSE (one defect, 2 rows): two harness coroutines call the '
        'sync cli_stage_runner.py::gc_run_config_dir inline, which '
        'shutil.rmtree()s a run\'s per-run agent config dir -- one unlink '
        'syscall per file, all of them on the loop thread, and a config dir '
        'is not one file. Found only once task 4484\'s amendment pass added '
        'shutil.rmtree to the vocabulary, which is the same gap-B shape the '
        'guard exists to close: the primitive was never enumerated, so the '
        'sites were invisible however the census was run. Follow-up filed by '
        'task 4484 amendment pass.'
        ' Ticket: tkt_0RT88VW7RRECTHNCVTJXD6M5RJ.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._persist_stage_reports_then_gc_config_dir',
        '361d4c634750',
        'to_file',
        'ROOT CAUSE (one defect, 2 rows): two harness coroutines call the '
        'sync cli_stage_runner.py::gc_run_config_dir inline, which '
        'shutil.rmtree()s a run\'s per-run agent config dir -- one unlink '
        'syscall per file, all of them on the loop thread, and a config dir '
        'is not one file. Found only once task 4484\'s amendment pass added '
        'shutil.rmtree to the vocabulary, which is the same gap-B shape the '
        'guard exists to close: the primitive was never enumerated, so the '
        'sites were invisible however the census was run. Follow-up filed by '
        'task 4484 amendment pass.'
        ' Ticket: tkt_0RT88VW7RRECTHNCVTJXD6M5RJ.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._recover_stale_runs',
        '0787c60051a4',
        'filed',
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._recover_stale_runs',
        'cc7999e2d4b6',
        'filed',
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._resume_interrupted_runs',
        'bc3672ec502c',
        'filed',
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness.run_full_cycle',
        '6770f1ceabc5',
        'filed',
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_escalate_stale_task_count_snapshot',
        '560696057da1',
        'filed',
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_remediate',
        '1fbc50761be3',
        'filed',
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_remediate',
        '51ec3ffac666',
        'filed',
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_remediate',
        '8c9728223345',
        'filed',
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        'c5a47e52ad4b',
        'to_file',
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
        'unfixed.'
        ' Ticket: tkt_0RT7RJKQ0WXB0T87F8TJ7RTGQH.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        '9e5ae6eb2503',
        'to_file',
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
        'unfixed.'
        ' Ticket: tkt_0RT7RJKQ0WXB0T87F8TJ7RTGQH.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        '58434c3060da',
        'filed',
        'ROOT CAUSE: _file_finding_task_escalation, the orchestrator-queue '
        'counterpart of _escalate that task 4821 added after this census '
        'ran, is sync end to end on the loop thread -- the '
        'is_orchestrator_live_for lock-file read, EscalationQueue\'s '
        'mkdir, a read_text of every pending record in the queue root '
        'for the category fold (so it grows with the pending queue), '
        'then make_id\'s fcntl.flock over an fsync\'d counter write and '
        'submit\'s fsync\'d record write. The flock wait is bounded only '
        'by another process\'s hold. OWNED BY '
        'TASK 5270 -- do not file again: added to its scope 2026-09-16 '
        'beside the _escalate cluster, whose fix shape (offload the '
        'filer) it shares.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        '920dc9ba5dbb',
        'filed',
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        'ec953752e2f1',
        'filed',
        _ESCALATE_ARCHIVE_SCAN_WHY,
    ),

    # ---- reconciliation/journal.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/journal.py',
        'ReconciliationJournal.initialize',
        '863ff109dbaf',
        'accepted',
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- reconciliation/recon_ledger.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/recon_ledger.py',
        'ReconLedgerStore.initialize',
        'f485b909e349',
        'accepted',
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- reconciliation/stages/task_knowledge_sync.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py',
        '_write_task_count_snapshot',
        'e08ddda37175',
        'to_file',
        'SAME ROOT CAUSE as the server/tools.py::_normalize_project_root '
        'cluster: a cold miss in models/scope.py::resolve_main_checkout '
        'runs subprocess.run(git) on the loop thread; _MAIN_CHECKOUT_CACHE '
        'makes it cold-miss-only. Filed together with that cluster by task '
        '4484 step-9, since one fix closes both.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py',
        '_run_briefing_known_gaps_script',
        'a23d01e12b5a',
        'accepted',
        _FIXED_STAT_GUARD_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py',
        '_run_briefing_known_gaps_script',
        '39a6fb8ac63e',
        'accepted',
        _FIXED_STAT_GUARD_WHY,
    ),

    # ---- reconciliation/stale_priority_override_edge_sweep.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/stale_priority_override_edge_sweep.py',
        'read_live_override_state',
        '7ebd882c849b',
        'to_file',
        'SAME ROOT CAUSE as the server/tools.py::_normalize_project_root '
        'cluster: a cold miss in models/scope.py::resolve_main_checkout '
        'runs subprocess.run(git) on the loop thread; _MAIN_CHECKOUT_CACHE '
        'makes it cold-miss-only. Filed together with that cluster by task '
        '4484 step-9, since one fix closes both.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/stale_priority_override_edge_sweep.py',
        'read_live_override_state',
        '09007f79712d',
        'accepted',
        _FIXED_STAT_GUARD_WHY,
    ),

    # ---- reconciliation/targeted.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/targeted.py',
        'TargetedReconciler._sweep_cancelled_descendants',
        'b9609c7cf5b4',
        'to_file',
        '_sweep_cancelled_descendants reaches is_orchestrator_live_for, '
        'which read_texts the orchestrator state file on the loop thread. '
        'Task 3778 mentions is_orchestrator_live_for in its Part 3(a) '
        'hoisting discussion but reconciliation/targeted.py is not in its '
        'metadata.files, so this caller is nobody\'s today. Follow-up filed '
        'by task 4484 step-9.'
        ' Ticket: tkt_0RT7RKWJG17W03JRC947R8FZZN.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/targeted.py',
        'TargetedReconciler._sweep_cancelled_descendants',
        'f06e6adb6bbd',
        'accepted',
        _DECLARED_FILES_STAT_WHY
        + ' Task 5270 already owns this coroutine\'s is_orchestrator_live_for '
        'row (ticket-derived task 5077): if 5270 offloads the whole sweep, '
        'delete this row with it.',
    ),

    # ---- reconciliation/verify.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/verify.py',
        'CodebaseVerifier.verify.read_file',
        'c04bd0d302eb',
        'filed',
        _VERIFIER_LLM_TOOL_IO_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/verify.py',
        'CodebaseVerifier.verify.glob_search',
        '29916e02709d',
        'filed',
        _VERIFIER_LLM_TOOL_IO_WHY,
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/verify.py',
        'CodebaseVerifier.verify',
        'ab04e5fd6d64',
        'accepted',
        _FIXED_STAT_GUARD_WHY,
    ),

    # ---- services/durable_queue.py ----
    (
        'fused-memory/src/fused_memory/services/durable_queue.py',
        'DurableWriteQueue.initialize',
        'c57d271d306d',
        'accepted',
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- services/live_workflow_detector.py ----
    (
        'fused-memory/src/fused_memory/services/live_workflow_detector.py',
        'detect_live_workflow',
        'b9609c7cf5b4',
        'accepted',
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

    # ---- services/planned_episode_registry.py ----
    (
        'fused-memory/src/fused_memory/services/planned_episode_registry.py',
        'PlannedEpisodeRegistry.initialize',
        'c57d271d306d',
        'accepted',
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- services/write_journal.py ----
    (
        'fused-memory/src/fused_memory/services/write_journal.py',
        'WriteJournal.initialize',
        '863ff109dbaf',
        'accepted',
        _MKDIR_ON_STORE_OPEN_WHY,
    ),

    # ---- server/main.py ----
    (
        'fused-memory/src/fused_memory/server/main.py',
        'run_server',
        'f38dc5fc1e4c',
        'accepted',
        'ACCEPTED: run_server calls build_known_projects_map '
        '(models/scope.py -- open() of each project manifest, then '
        'yaml.safe_load; the sweep reports whichever the walk reaches '
        'first) during process STARTUP, before the server binds '
        'and begins serving traffic, so there is no concurrent work for it '
        'to stall and no request whose latency it can affect. This is what '
        'an accepted row must say -- what makes the cost acceptable, not '
        'merely that the call exists. If this ever moves onto a request or '
        'reload path, the content_hash changes and the gate re-asks.',
    ),
    (
        'fused-memory/src/fused_memory/server/main.py',
        'run_server',
        'e3f8e93f2610',
        'accepted',
        'ACCEPTED, startup only: run_server reaches '
        'build_topic_cluster_store -> TopicClusterStore.open(), a sync '
        'SQLite open matched by the method name open. It genuinely blocks, '
        'but like the build_known_projects_map row beside it, it runs once '
        'during process STARTUP, before the server binds and begins '
        'serving traffic, so there is no concurrent work to stall. If it '
        'ever moves onto a request or reload path, the content_hash '
        'changes and the gate re-asks.',
    ),

    # ---- server/manifest_stamping.py ----
    (
        'fused-memory/src/fused_memory/server/manifest_stamping.py',
        '_stamp_capability_manifests_impl',
        '3817640cc33d',
        'filed',
        _MANIFEST_STAMPING_INLINE_IO_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/manifest_stamping.py',
        '_stamp_capability_manifests_impl',
        'e19569fdcfa0',
        'filed',
        _MANIFEST_STAMPING_INLINE_IO_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/manifest_stamping.py',
        '_stamp_capability_manifests_impl',
        '78912ffb516a',
        'filed',
        _MANIFEST_STAMPING_INLINE_IO_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/manifest_stamping.py',
        '_stamp_capability_manifests_impl',
        '93a769609c9b',
        'filed',
        _MANIFEST_STAMPING_INLINE_IO_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/manifest_stamping.py',
        '_stamp_capability_manifests_impl',
        'dfdb79e84d56',
        'filed',
        _MANIFEST_STAMPING_INLINE_IO_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/manifest_stamping.py',
        '_stamp_capability_manifests_impl',
        '364eac0531d3',
        'filed',
        _MANIFEST_STAMPING_INLINE_IO_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/manifest_stamping.py',
        '_stamp_capability_manifests_impl',
        'edbfd36fd257',
        'filed',
        _MANIFEST_STAMPING_INLINE_IO_WHY,
    ),

    # ---- server/tools.py ----
    (
        'fused-memory/src/fused_memory/server/tools.py',
        '_open_overrides_db',
        '14c6bfa75f63',
        'accepted',
        _MKDIR_ON_STORE_OPEN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        '_connect_overrides_db',
        '14c6bfa75f63',
        'accepted',
        _MKDIR_ON_STORE_OPEN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        '_open_park_eviction_db',
        '14c6bfa75f63',
        'accepted',
        _MKDIR_ON_STORE_OPEN_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        '_checkpoint_overrides_db_if_exists',
        '051cd59d92be',
        'accepted',
        _FIXED_STAT_GUARD_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.submit_task',
        '32c0a08cd635',
        'accepted',
        _FIXED_STAT_GUARD_WHY,
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_tasks',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_statuses',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_external_statuses',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_task',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.set_task_status',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.set_task_claimant',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.submit_task',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.resolve_ticket',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.list_tickets',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.search_tasks',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.commit_planning',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.update_task',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.remove_task',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.add_dependency',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.remove_dependency',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.set_task_priority_override',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.clear_task_priority_override',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.reorder_pin_queue',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_pin_queue',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_scheduler_state',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.get_scheduler_events',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server.request_park_eviction',
        '81d0d2709341',
        'to_file',
        'ROOT CAUSE (one defect, 22 rows): every MCP handler taking a '
        'project_root calls the nested sync _normalize_project_root, which '
        'reaches models/scope.py::resolve_main_checkout -> '
        'subprocess.run(git). The result is memoised in '
        '_MAIN_CHECKOUT_CACHE, so this is a COLD-MISS-ONLY fork+exec on the '
        'loop thread -- exactly the caveat task 3778\'s own census recorded '
        '("cached but a cold miss runs on the loop thread") and then did '
        'not act on. 22 rows, one fix: offload or pre-warm '
        'resolve_main_checkout once. Rows are kept per-site so a 23rd '
        'handler cannot be added silently under a blessed 22. Follow-up '
        'filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ.',
    ),
]

#: Derived, never hand-maintained beside the rows -- a second list would drift.
ALLOWLIST_KEYS: list[tuple[str, str, str]] = [
    (relpath, qualname, content_hash)
    for relpath, qualname, content_hash, _disposition, _justification in AUDITED_SITES
]
