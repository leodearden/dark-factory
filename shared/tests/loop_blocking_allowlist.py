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
justification naming that shared cause and its single follow-up.  The ledger
stays row-per-site so a 23rd handler cannot be added silently under a blessed
22; the triage stays cluster-per-defect so the follow-ups are one task per
defect rather than one per line.

Regenerating
------------
There is no generation timestamp here, deliberately: re-running the scan and
diffing this file is then a meaningful reproducibility check rather than a
guaranteed diff.  To re-derive the rows::

    cd shared && uv run pytest tests/test_loop_blocking_gate.py -q

A failure names every site whose disposition is missing or stale.

Baseline measured at HEAD 6696f1ce0c: 167 files scanned, 60 findings.
Task 4484's amendment pass re-measured 62 over the same 167 files: one
row withdrawn as a scanner false positive (see the
create_mcp_server._claim_commit_presence row) and three added by widening
the vocabulary to shutil.rmtree.
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

#: ``(relpath, qualname, content_hash, disposition, justification)``.
AUDITED_SITES: list[tuple[str, str, str, str, str]] = [

    # ---- middleware/curator_escalator.py ----
    (
        'fused-memory/src/fused_memory/middleware/curator_escalator.py',
        'CuratorEscalator.report_failure',
        '3ba9760c42a6',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): the orchestrator-liveness probe '
        '(_orchestrator_running in middleware/ticket_janitor.py and '
        'middleware/curator_escalator.py, plus '
        'ticket_janitor._surface_probe_defect which reaches it) takes an '
        'fcntl.flock on the loop thread. A lock wait is bounded only by '
        'ANOTHER process\'s hold time, which this one does not control -- '
        'the lock limb of INV-8 that task 3778\'s subprocess-only vocabulary '
        'never enumerated. Follow-up filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QZ4R9MQHJP4MKS78DXQ2Z9.',
    ),

    # ---- middleware/task_curator.py ----
    (
        'fused-memory/src/fused_memory/middleware/task_curator.py',
        'TaskCurator._maybe_blocklist_drop',
        '8e71fbf93a3e',
        'to_file',
        'ROOT CAUSE (one defect, 2 rows): the two SIBLING async def '
        '_maybe_* guards in this file lazily load a YAML registry off disk '
        'on the loop thread with no asyncio.to_thread hop -- '
        '_maybe_blocklist_drop -> load_blocklist '
        '(cancelled_premise_blocklist.read_text + yaml.safe_load) and '
        '_maybe_route_deterministic -> load_operational_registry '
        '(operational_ask_registry.read_text + yaml.safe_load). Identical '
        'shape to the two task 4201 owns, three lines apart, under the same '
        'curator write lock -- and NEITHER had a task filed before task '
        '4484 found them, which is the concrete cost of task 3778\'s '
        'definition-side census. Follow-up filed by task 4484 step-9 (one '
        'task: one shape, one file, and 4201 already owns the sibling).'
        ' Ticket: tkt_0RT7QW5E7RQ3FHF2HQ0BRC6MH0.',
    ),
    (
        'fused-memory/src/fused_memory/middleware/task_curator.py',
        'TaskCurator._maybe_route_deterministic',
        '0f0498be8607',
        'to_file',
        'ROOT CAUSE (one defect, 2 rows): the two SIBLING async def '
        '_maybe_* guards in this file lazily load a YAML registry off disk '
        'on the loop thread with no asyncio.to_thread hop -- '
        '_maybe_blocklist_drop -> load_blocklist '
        '(cancelled_premise_blocklist.read_text + yaml.safe_load) and '
        '_maybe_route_deterministic -> load_operational_registry '
        '(operational_ask_registry.read_text + yaml.safe_load). Identical '
        'shape to the two task 4201 owns, three lines apart, under the same '
        'curator write lock -- and NEITHER had a task filed before task '
        '4484 found them, which is the concrete cost of task 3778\'s '
        'definition-side census. Follow-up filed by task 4484 step-9 (one '
        'task: one shape, one file, and 4201 already owns the sibling).'
        ' Ticket: tkt_0RT7QW5E7RQ3FHF2HQ0BRC6MH0.',
    ),

    # ---- middleware/ticket_janitor.py ----
    (
        'fused-memory/src/fused_memory/middleware/ticket_janitor.py',
        'TicketJanitor.tick',
        '90eccb740171',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): the orchestrator-liveness probe '
        '(_orchestrator_running in middleware/ticket_janitor.py and '
        'middleware/curator_escalator.py, plus '
        'ticket_janitor._surface_probe_defect which reaches it) takes an '
        'fcntl.flock on the loop thread. A lock wait is bounded only by '
        'ANOTHER process\'s hold time, which this one does not control -- '
        'the lock limb of INV-8 that task 3778\'s subprocess-only vocabulary '
        'never enumerated. Follow-up filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QZ4R9MQHJP4MKS78DXQ2Z9.',
    ),
    (
        'fused-memory/src/fused_memory/middleware/ticket_janitor.py',
        'TicketJanitor.tick',
        'f0c22189ed58',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): the orchestrator-liveness probe '
        '(_orchestrator_running in middleware/ticket_janitor.py and '
        'middleware/curator_escalator.py, plus '
        'ticket_janitor._surface_probe_defect which reaches it) takes an '
        'fcntl.flock on the loop thread. A lock wait is bounded only by '
        'ANOTHER process\'s hold time, which this one does not control -- '
        'the lock limb of INV-8 that task 3778\'s subprocess-only vocabulary '
        'never enumerated. Follow-up filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QZ4R9MQHJP4MKS78DXQ2Z9.',
    ),

    # ---- reconciliation/backlog_policy.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py',
        'BacklogPolicy.on_judge_unhalt',
        '8f53c74a5d76',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): BacklogPolicy reads and writes '
        'its escalation records on the loop thread -- on_judge_unhalt '
        'read_texts the record and reaches _restore_policy_keys, and '
        '_maybe_write_escalation reaches _merge_onto_persisted; both '
        'helpers read_text then write_text the located record under '
        'escalation_id_lock. UNDERSTATED BY THESE THREE ROWS, and recorded '
        'here because no row can carry it: the dominant blocking work on '
        'the write path is the escalation.dedupe.submit_or_dedupe call one '
        'line ABOVE the _merge_onto_persisted site -- find_dedupe_parent '
        'globs and JSON-parses every pending record in the project queue '
        '(queue.get_pending, O(N); N=41 measured on the live dark_factory '
        'queue 2026-09-18), then queue.submit writes with a durable fsync. '
        'The scanner reports only _merge_onto_persisted because those '
        'primitives live in the escalation package, across a boundary its '
        'fused-memory/src scope cannot follow -- so a fourth row would be '
        'a blessing test_no_stale_blessings rejects, and this paragraph is '
        'the only honest place to state it. Filesystem, the limb task '
        '3778\'s subprocess-only vocabulary omitted. Follow-up filed by '
        'task 4484 step-9. Ticket: tkt_0RT7RHRS9ZTJSQK328919XXEJW.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py',
        'BacklogPolicy.on_judge_unhalt',
        '08a103635fd7',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): BacklogPolicy reads and writes '
        'its escalation records on the loop thread -- on_judge_unhalt '
        'read_texts the record and reaches _restore_policy_keys, and '
        '_maybe_write_escalation reaches _merge_onto_persisted; both '
        'helpers read_text then write_text the located record under '
        'escalation_id_lock. UNDERSTATED BY THESE THREE ROWS, and recorded '
        'here because no row can carry it: the dominant blocking work on '
        'the write path is the escalation.dedupe.submit_or_dedupe call one '
        'line ABOVE the _merge_onto_persisted site -- find_dedupe_parent '
        'globs and JSON-parses every pending record in the project queue '
        '(queue.get_pending, O(N); N=41 measured on the live dark_factory '
        'queue 2026-09-18), then queue.submit writes with a durable fsync. '
        'The scanner reports only _merge_onto_persisted because those '
        'primitives live in the escalation package, across a boundary its '
        'fused-memory/src scope cannot follow -- so a fourth row would be '
        'a blessing test_no_stale_blessings rejects, and this paragraph is '
        'the only honest place to state it. Filesystem, the limb task '
        '3778\'s subprocess-only vocabulary omitted. Follow-up filed by '
        'task 4484 step-9. Ticket: tkt_0RT7RHRS9ZTJSQK328919XXEJW.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py',
        'BacklogPolicy._maybe_write_escalation',
        '1939296ee9cb',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): BacklogPolicy reads and writes '
        'its escalation records on the loop thread -- on_judge_unhalt '
        'read_texts the record and reaches _restore_policy_keys, and '
        '_maybe_write_escalation reaches _merge_onto_persisted; both '
        'helpers read_text then write_text the located record under '
        'escalation_id_lock. UNDERSTATED BY THESE THREE ROWS, and recorded '
        'here because no row can carry it: the dominant blocking work on '
        'the write path is the escalation.dedupe.submit_or_dedupe call one '
        'line ABOVE the _merge_onto_persisted site -- find_dedupe_parent '
        'globs and JSON-parses every pending record in the project queue '
        '(queue.get_pending, O(N); N=41 measured on the live dark_factory '
        'queue 2026-09-18), then queue.submit writes with a durable fsync. '
        'The scanner reports only _merge_onto_persisted because those '
        'primitives live in the escalation package, across a boundary its '
        'fused-memory/src scope cannot follow -- so a fourth row would be '
        'a blessing test_no_stale_blessings rejects, and this paragraph is '
        'the only honest place to state it. Filesystem, the limb task '
        '3778\'s subprocess-only vocabulary omitted. Follow-up filed by '
        'task 4484 step-9. Ticket: tkt_0RT7RHRS9ZTJSQK328919XXEJW.',
    ),

    # ---- reconciliation/harness.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._recover_one_run',
        '361d4c634750',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): three harness coroutines call the '
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
        'ReconciliationHarness.run_full_cycle',
        '21ec0716d946',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): three harness coroutines call the '
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
        'ReconciliationHarness._run_remediation_pass',
        '21ec0716d946',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): three harness coroutines call the '
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
        'ROOT CAUSE (one defect, 10 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on '
        'the loop thread, so this trips INV-8\'s fan-out limb as well '
        'as its blocking limb, and the cost grows with queue history '
        'rather than staying constant. 10 async callers, one fix '
        '(offload _escalate, or pre-fetch resolved_fps once per cycle '
        '-- the resolved_fps kwarg already exists for exactly that). '
        'OWNED BY TASK 5270 -- do not file again: task 4484 step-9\'s '
        'ticket tkt_0RT7QXR5AXGWVADW9T3S4DPC2M became task 5072, '
        'coalesced into 5270. 5072\'s text counts 9 callers; the 10th, '
        '_maybe_remediate\'s phantom-citation storm alarm, landed later '
        'with task 4781.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._recover_stale_runs',
        'cc7999e2d4b6',
        'filed',
        'ROOT CAUSE (one defect, 10 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on '
        'the loop thread, so this trips INV-8\'s fan-out limb as well '
        'as its blocking limb, and the cost grows with queue history '
        'rather than staying constant. 10 async callers, one fix '
        '(offload _escalate, or pre-fetch resolved_fps once per cycle '
        '-- the resolved_fps kwarg already exists for exactly that). '
        'OWNED BY TASK 5270 -- do not file again: task 4484 step-9\'s '
        'ticket tkt_0RT7QXR5AXGWVADW9T3S4DPC2M became task 5072, '
        'coalesced into 5270. 5072\'s text counts 9 callers; the 10th, '
        '_maybe_remediate\'s phantom-citation storm alarm, landed later '
        'with task 4781.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._resume_interrupted_runs',
        'bc3672ec502c',
        'filed',
        'ROOT CAUSE (one defect, 10 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on '
        'the loop thread, so this trips INV-8\'s fan-out limb as well '
        'as its blocking limb, and the cost grows with queue history '
        'rather than staying constant. 10 async callers, one fix '
        '(offload _escalate, or pre-fetch resolved_fps once per cycle '
        '-- the resolved_fps kwarg already exists for exactly that). '
        'OWNED BY TASK 5270 -- do not file again: task 4484 step-9\'s '
        'ticket tkt_0RT7QXR5AXGWVADW9T3S4DPC2M became task 5072, '
        'coalesced into 5270. 5072\'s text counts 9 callers; the 10th, '
        '_maybe_remediate\'s phantom-citation storm alarm, landed later '
        'with task 4781.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness.run_full_cycle',
        '6770f1ceabc5',
        'filed',
        'ROOT CAUSE (one defect, 10 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on '
        'the loop thread, so this trips INV-8\'s fan-out limb as well '
        'as its blocking limb, and the cost grows with queue history '
        'rather than staying constant. 10 async callers, one fix '
        '(offload _escalate, or pre-fetch resolved_fps once per cycle '
        '-- the resolved_fps kwarg already exists for exactly that). '
        'OWNED BY TASK 5270 -- do not file again: task 4484 step-9\'s '
        'ticket tkt_0RT7QXR5AXGWVADW9T3S4DPC2M became task 5072, '
        'coalesced into 5270. 5072\'s text counts 9 callers; the 10th, '
        '_maybe_remediate\'s phantom-citation storm alarm, landed later '
        'with task 4781.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_escalate_stale_task_count_snapshot',
        '560696057da1',
        'filed',
        'ROOT CAUSE (one defect, 10 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on '
        'the loop thread, so this trips INV-8\'s fan-out limb as well '
        'as its blocking limb, and the cost grows with queue history '
        'rather than staying constant. 10 async callers, one fix '
        '(offload _escalate, or pre-fetch resolved_fps once per cycle '
        '-- the resolved_fps kwarg already exists for exactly that). '
        'OWNED BY TASK 5270 -- do not file again: task 4484 step-9\'s '
        'ticket tkt_0RT7QXR5AXGWVADW9T3S4DPC2M became task 5072, '
        'coalesced into 5270. 5072\'s text counts 9 callers; the 10th, '
        '_maybe_remediate\'s phantom-citation storm alarm, landed later '
        'with task 4781.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_remediate',
        '1fbc50761be3',
        'filed',
        'ROOT CAUSE (one defect, 10 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on '
        'the loop thread, so this trips INV-8\'s fan-out limb as well '
        'as its blocking limb, and the cost grows with queue history '
        'rather than staying constant. 10 async callers, one fix '
        '(offload _escalate, or pre-fetch resolved_fps once per cycle '
        '-- the resolved_fps kwarg already exists for exactly that). '
        'OWNED BY TASK 5270 -- do not file again: task 4484 step-9\'s '
        'ticket tkt_0RT7QXR5AXGWVADW9T3S4DPC2M became task 5072, '
        'coalesced into 5270. 5072\'s text counts 9 callers; the 10th, '
        '_maybe_remediate\'s phantom-citation storm alarm, landed later '
        'with task 4781.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_remediate',
        '51ec3ffac666',
        'filed',
        'ROOT CAUSE (one defect, 10 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on '
        'the loop thread, so this trips INV-8\'s fan-out limb as well '
        'as its blocking limb, and the cost grows with queue history '
        'rather than staying constant. 10 async callers, one fix '
        '(offload _escalate, or pre-fetch resolved_fps once per cycle '
        '-- the resolved_fps kwarg already exists for exactly that). '
        'OWNED BY TASK 5270 -- do not file again: task 4484 step-9\'s '
        'ticket tkt_0RT7QXR5AXGWVADW9T3S4DPC2M became task 5072, '
        'coalesced into 5270. 5072\'s text counts 9 callers; the 10th, '
        '_maybe_remediate\'s phantom-citation storm alarm, landed later '
        'with task 4781.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_remediate',
        '82d9ae32fa2a',
        'filed',
        'ROOT CAUSE (one defect, 10 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on '
        'the loop thread, so this trips INV-8\'s fan-out limb as well '
        'as its blocking limb, and the cost grows with queue history '
        'rather than staying constant. 10 async callers, one fix '
        '(offload _escalate, or pre-fetch resolved_fps once per cycle '
        '-- the resolved_fps kwarg already exists for exactly that). '
        'OWNED BY TASK 5270 -- do not file again: task 4484 step-9\'s '
        'ticket tkt_0RT7QXR5AXGWVADW9T3S4DPC2M became task 5072, '
        'coalesced into 5270. 5072\'s text counts 9 callers; the 10th, '
        '_maybe_remediate\'s phantom-citation storm alarm, landed later '
        'with task 4781.',
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
        'dbf8ae2eb1dc',
        'filed',
        'ROOT CAUSE (one defect, 10 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on '
        'the loop thread, so this trips INV-8\'s fan-out limb as well '
        'as its blocking limb, and the cost grows with queue history '
        'rather than staying constant. 10 async callers, one fix '
        '(offload _escalate, or pre-fetch resolved_fps once per cycle '
        '-- the resolved_fps kwarg already exists for exactly that). '
        'OWNED BY TASK 5270 -- do not file again: task 4484 step-9\'s '
        'ticket tkt_0RT7QXR5AXGWVADW9T3S4DPC2M became task 5072, '
        'coalesced into 5270. 5072\'s text counts 9 callers; the 10th, '
        '_maybe_remediate\'s phantom-citation storm alarm, landed later '
        'with task 4781.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        '1812aa52aaca',
        'filed',
        'ROOT CAUSE (one defect, 10 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on '
        'the loop thread, so this trips INV-8\'s fan-out limb as well '
        'as its blocking limb, and the cost grows with queue history '
        'rather than staying constant. 10 async callers, one fix '
        '(offload _escalate, or pre-fetch resolved_fps once per cycle '
        '-- the resolved_fps kwarg already exists for exactly that). '
        'OWNED BY TASK 5270 -- do not file again: task 4484 step-9\'s '
        'ticket tkt_0RT7QXR5AXGWVADW9T3S4DPC2M became task 5072, '
        'coalesced into 5270. 5072\'s text counts 9 callers; the 10th, '
        '_maybe_remediate\'s phantom-citation storm alarm, landed later '
        'with task 4781.',
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
        'TaskKnowledgeSync.assemble_payload',
        '999e4ab43b34',
        'filed',
        'ROOT CAUSE (one defect, 1 row): the live-workflow git probes '
        'in services/live_workflow_detector.py run sync '
        'subprocess.run(git) per task, reached from this coroutine '
        'with no hop. OWNED BY TASK 3778 -- do not file again; 3778 '
        'measured 29.2s (dark_factory) / 43.4s (reify) per render, and '
        'its Part 2 makes is_workflow_live_for_task async and '
        'propagates that to every consumer, naming '
        '_render_live_workflow_section, '
        'memory_consolidator._build_live_workflow_section and '
        'harness\'s integrity-escalation suppression loop. The cluster '
        'was 3 rows until task 5550: the two harness rows '
        '(c65b42126493, a6c8a890e63e) were the cited-task and '
        'routed-target gates in _run_remediation_pass, and 5550 '
        'offloaded BOTH with asyncio.to_thread (memoised per distinct '
        'task id per pass), leaving the sync closure _task_is_live '
        'itself unchanged as the thread body. Those two findings went '
        'stale and the rows were deleted with that fix -- NOT because '
        '3778 landed, which it has not. assemble_payload is therefore '
        '3778\'s only remaining filed site. memory_consolidator\'s '
        'consumers still block but carry no row: task 4708 routed them '
        'through getattr(self, section.renderer)() in '
        '_render_required_sections, which the scanner cannot resolve. '
        'When 3778 lands, this row goes stale too.',
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

    # ---- reconciliation/verify.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/verify.py',
        'CodebaseVerifier.verify.read_file',
        'c04bd0d302eb',
        'to_file',
        'The async read_file tool handed to the codebase-verification LLM '
        'does full_path.read_text() inline on the loop thread, once per '
        'tool call the model chooses to make -- an LLM-driven, unbounded '
        'call count. Follow-up filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7RM7C7NS1ECYHFBDYP02KDJ.',
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

    # ---- server/manifest_stamping.py ----
    (
        'fused-memory/src/fused_memory/server/manifest_stamping.py',
        '_stamp_capability_manifests_impl',
        '3817640cc33d',
        'to_file',
        'ROOT CAUSE (one defect, 4 rows): _stamp_capability_manifests_impl '
        'does read_text + yaml.safe_load + write_text + yaml.safe_dump '
        'INLINE in one coroutine, with no helper anywhere for a '
        'definition-side census to point at -- the shape task 3778\'s '
        'methodology is structurally blind to. Task 4201 measured '
        'yaml.safe_load at 8.15 ms for an 11 KB document, so the parse '
        'alone is the same order as a subprocess spawn and this coroutine '
        'pays it twice plus two filesystem round trips. One '
        'asyncio.to_thread around the whole read-parse-write closes all '
        'four. Follow-up filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QYENVS6J9WVWCY3FJAVNFR.',
    ),
    (
        'fused-memory/src/fused_memory/server/manifest_stamping.py',
        '_stamp_capability_manifests_impl',
        'e19569fdcfa0',
        'to_file',
        'ROOT CAUSE (one defect, 4 rows): _stamp_capability_manifests_impl '
        'does read_text + yaml.safe_load + write_text + yaml.safe_dump '
        'INLINE in one coroutine, with no helper anywhere for a '
        'definition-side census to point at -- the shape task 3778\'s '
        'methodology is structurally blind to. Task 4201 measured '
        'yaml.safe_load at 8.15 ms for an 11 KB document, so the parse '
        'alone is the same order as a subprocess spawn and this coroutine '
        'pays it twice plus two filesystem round trips. One '
        'asyncio.to_thread around the whole read-parse-write closes all '
        'four. Follow-up filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QYENVS6J9WVWCY3FJAVNFR.',
    ),
    (
        'fused-memory/src/fused_memory/server/manifest_stamping.py',
        '_stamp_capability_manifests_impl',
        '78912ffb516a',
        'to_file',
        'ROOT CAUSE (one defect, 4 rows): _stamp_capability_manifests_impl '
        'does read_text + yaml.safe_load + write_text + yaml.safe_dump '
        'INLINE in one coroutine, with no helper anywhere for a '
        'definition-side census to point at -- the shape task 3778\'s '
        'methodology is structurally blind to. Task 4201 measured '
        'yaml.safe_load at 8.15 ms for an 11 KB document, so the parse '
        'alone is the same order as a subprocess spawn and this coroutine '
        'pays it twice plus two filesystem round trips. One '
        'asyncio.to_thread around the whole read-parse-write closes all '
        'four. Follow-up filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QYENVS6J9WVWCY3FJAVNFR.',
    ),
    (
        'fused-memory/src/fused_memory/server/manifest_stamping.py',
        '_stamp_capability_manifests_impl',
        '93a769609c9b',
        'to_file',
        'ROOT CAUSE (one defect, 4 rows): _stamp_capability_manifests_impl '
        'does read_text + yaml.safe_load + write_text + yaml.safe_dump '
        'INLINE in one coroutine, with no helper anywhere for a '
        'definition-side census to point at -- the shape task 3778\'s '
        'methodology is structurally blind to. Task 4201 measured '
        'yaml.safe_load at 8.15 ms for an 11 KB document, so the parse '
        'alone is the same order as a subprocess spawn and this coroutine '
        'pays it twice plus two filesystem round trips. One '
        'asyncio.to_thread around the whole read-parse-write closes all '
        'four. Follow-up filed by task 4484 step-9.'
        ' Ticket: tkt_0RT7QYENVS6J9WVWCY3FJAVNFR.',
    ),

    # ---- server/tools.py ----
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server._claim_commit_presence',
        '4d52656d51f0',
        'to_file',
        'ROOT CAUSE (one defect, 1 row): the claim-verification MCP handler '
        '_claim_commit_presence calls completion_claim_gate\'s '
        'make_commit_probe INLINE on the loop thread, reaching '
        'subprocess.run(git cat-file). Task 3778\'s census asserted this '
        'module was "already offloaded at its call sites" -- true of the '
        'callers it looked at, false of this one, which is the '
        'counter-example that made task 4484 necessary. Follow-up filed by '
        'task 4484 step-9.'
        ' Ticket: tkt_0RT7RHBAS4A3VH976CE1CJMGK8. WITHDRAWN SIBLING: this '
        'cluster was originally 2 rows. The second '
        '(create_mcp_server._completion_claim_gate -> verify_claims, hash '
        '01e23bf817cf) was a SCANNER false positive, deleted in task 4484\'s '
        'amendment pass: the pre-amendment resolver indexed defs at any '
        'nesting depth, so _verify_task\'s "probe" PARAMETER resolved to the '
        'unrelated nested make_commit_probe.probe. verify_claims is sync by '
        'design and its probes are pre-resolved dict lookups; it reaches no '
        'primitive. The ticket above names both handlers and overstates by '
        'one site.',
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
