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
them -- it would go stale the moment task 4201 lands and removes two rows,
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
                  ``to_file`` names the follow-up task 4484 step-9 files.

Keys are ``(relpath, qualname, content_hash)`` -- deliberately NOT line
numbers, which drift on every unrelated edit above the site and would make
this a nuisance ratchet reviewers learn to re-bless unread.  The hash is
invariant under reindentation and edits above the site, and changes only when
the call itself changes.

MULTISET, not set.  ``reconcile_against_allowlist`` compares with
``Counter`` subtraction, so a function with two byte-identical blocking calls
needs two rows.  This is load-bearing for THIS gate specifically: set
membership would let a second site inside an already-blessed function pass
silently, which is a per-function restatement of exactly the per-module
blindness that produced the misses above.

ROW PER SITE, TRIAGE PER CAUSE.  Where a cluster shares one root cause -- the
22 MCP handlers reaching the cached ``resolve_main_checkout``, the 9 async
callers of ``ReconciliationHarness._escalate`` -- every row carries the SAME
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
"""

from __future__ import annotations

#: The fixed disposition vocabulary.  No inline ``# inv8-ok:`` source-comment
#: marker exists as an alternative: this ledger has to exist anyway for the
#: ratchet, so a marker at the site would be a second home for the same fact
#: with nothing reconciling them (INV-9) -- and writing one would mean editing
#: the very runtime files tasks 4201 and 3778 are holding in the merge lane.
#: A reader at a site follows test_loop_blocking_gate.py to the row that says why.
DISPOSITIONS = frozenset({
    'accepted',  # measured cheap, cached, or startup-only -- say WHICH
    'filed',     # an existing task owns it -- justification carries the id
    'to_file',   # confirmed defect; task 4484 step-9 files the follow-up
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
        'never enumerated. Follow-up filed by task 4484 step-9.',
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
        'task: one shape, one file, and 4201 already owns the sibling).',
    ),
    (
        'fused-memory/src/fused_memory/middleware/task_curator.py',
        'TaskCurator._maybe_premise_refuted_drop',
        '29c51cfad34e',
        'filed',
        'ROOT CAUSE (one defect, 2 rows): async def '
        '_maybe_premise_refuted_drop loads the recon-code-fix premise '
        'registry off disk on the loop thread '
        '(recon_code_fix_premise_guard.read_text + yaml.safe_load), under '
        'the per-project curator write lock, on every task submission. '
        'OWNED BY TASK 4201 -- do not file again. 4201 measured '
        'yaml.safe_load at 8.15 ms for an 11 KB document and asks for '
        'either an asyncio.to_thread wrap or a recorded decision that the '
        'cost is acceptable. When 4201 lands these rows go stale and '
        'test_no_stale_blessings names them for deletion.',
    ),
    (
        'fused-memory/src/fused_memory/middleware/task_curator.py',
        'TaskCurator._maybe_premise_refuted_drop',
        '31040e51c20b',
        'filed',
        'ROOT CAUSE (one defect, 2 rows): async def '
        '_maybe_premise_refuted_drop loads the recon-code-fix premise '
        'registry off disk on the loop thread '
        '(recon_code_fix_premise_guard.read_text + yaml.safe_load), under '
        'the per-project curator write lock, on every task submission. '
        'OWNED BY TASK 4201 -- do not file again. 4201 measured '
        'yaml.safe_load at 8.15 ms for an 11 KB document and asks for '
        'either an asyncio.to_thread wrap or a recorded decision that the '
        'cost is acceptable. When 4201 lands these rows go stale and '
        'test_no_stale_blessings names them for deletion.',
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
        'task: one shape, one file, and 4201 already owns the sibling).',
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
        'never enumerated. Follow-up filed by task 4484 step-9.',
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
        'never enumerated. Follow-up filed by task 4484 step-9.',
    ),

    # ---- reconciliation/backlog_policy.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py',
        'BacklogPolicy.on_judge_unhalt',
        '8f53c74a5d76',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): BacklogPolicy reads and writes '
        'the judge-halt record on the loop thread -- on_judge_unhalt '
        'read_texts the record and reaches _restore_policy_keys (another '
        'read), and _maybe_write_escalation write_texts the escalation. '
        'Filesystem, the limb task 3778\'s subprocess-only vocabulary '
        'omitted. Follow-up filed by task 4484 step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py',
        'BacklogPolicy.on_judge_unhalt',
        '08a103635fd7',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): BacklogPolicy reads and writes '
        'the judge-halt record on the loop thread -- on_judge_unhalt '
        'read_texts the record and reaches _restore_policy_keys (another '
        'read), and _maybe_write_escalation write_texts the escalation. '
        'Filesystem, the limb task 3778\'s subprocess-only vocabulary '
        'omitted. Follow-up filed by task 4484 step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/backlog_policy.py',
        'BacklogPolicy._maybe_write_escalation',
        'ce9dabf347d0',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): BacklogPolicy reads and writes '
        'the judge-halt record on the loop thread -- on_judge_unhalt '
        'read_texts the record and reaches _restore_policy_keys (another '
        'read), and _maybe_write_escalation write_texts the escalation. '
        'Filesystem, the limb task 3778\'s subprocess-only vocabulary '
        'omitted. Follow-up filed by task 4484 step-9.',
    ),

    # ---- reconciliation/harness.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._recover_stale_runs',
        '0787c60051a4',
        'to_file',
        'ROOT CAUSE (one defect, 9 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on the '
        'loop thread, so this trips INV-8\'s fan-out limb as well as its '
        'blocking limb, and the cost grows with queue history rather than '
        'staying constant. 9 async callers, one fix (offload _escalate, or '
        'pre-fetch resolved_fps once per cycle -- the resolved_fps kwarg '
        'already exists for exactly that). Follow-up filed by task 4484 '
        'step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._recover_stale_runs',
        'cc7999e2d4b6',
        'to_file',
        'ROOT CAUSE (one defect, 9 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on the '
        'loop thread, so this trips INV-8\'s fan-out limb as well as its '
        'blocking limb, and the cost grows with queue history rather than '
        'staying constant. 9 async callers, one fix (offload _escalate, or '
        'pre-fetch resolved_fps once per cycle -- the resolved_fps kwarg '
        'already exists for exactly that). Follow-up filed by task 4484 '
        'step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._resume_interrupted_runs',
        'bc3672ec502c',
        'to_file',
        'ROOT CAUSE (one defect, 9 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on the '
        'loop thread, so this trips INV-8\'s fan-out limb as well as its '
        'blocking limb, and the cost grows with queue history rather than '
        'staying constant. 9 async callers, one fix (offload _escalate, or '
        'pre-fetch resolved_fps once per cycle -- the resolved_fps kwarg '
        'already exists for exactly that). Follow-up filed by task 4484 '
        'step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness.run_full_cycle',
        '6770f1ceabc5',
        'to_file',
        'ROOT CAUSE (one defect, 9 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on the '
        'loop thread, so this trips INV-8\'s fan-out limb as well as its '
        'blocking limb, and the cost grows with queue history rather than '
        'staying constant. 9 async callers, one fix (offload _escalate, or '
        'pre-fetch resolved_fps once per cycle -- the resolved_fps kwarg '
        'already exists for exactly that). Follow-up filed by task 4484 '
        'step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_escalate_stale_task_count_snapshot',
        '560696057da1',
        'to_file',
        'ROOT CAUSE (one defect, 9 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on the '
        'loop thread, so this trips INV-8\'s fan-out limb as well as its '
        'blocking limb, and the cost grows with queue history rather than '
        'staying constant. 9 async callers, one fix (offload _escalate, or '
        'pre-fetch resolved_fps once per cycle -- the resolved_fps kwarg '
        'already exists for exactly that). Follow-up filed by task 4484 '
        'step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_remediate',
        '51ec3ffac666',
        'to_file',
        'ROOT CAUSE (one defect, 9 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on the '
        'loop thread, so this trips INV-8\'s fan-out limb as well as its '
        'blocking limb, and the cost grows with queue history rather than '
        'staying constant. 9 async callers, one fix (offload _escalate, or '
        'pre-fetch resolved_fps once per cycle -- the resolved_fps kwarg '
        'already exists for exactly that). Follow-up filed by task 4484 '
        'step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._maybe_remediate',
        '82d9ae32fa2a',
        'to_file',
        'ROOT CAUSE (one defect, 9 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on the '
        'loop thread, so this trips INV-8\'s fan-out limb as well as its '
        'blocking limb, and the cost grows with queue history rather than '
        'staying constant. 9 async callers, one fix (offload _escalate, or '
        'pre-fetch resolved_fps once per cycle -- the resolved_fps kwarg '
        'already exists for exactly that). Follow-up filed by task 4484 '
        'step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        'ae4cd95f45ad',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): _run_remediation_pass read_texts '
        'the orchestrator state files inline on the loop thread (a direct '
        'read_text, plus read_scheduler_state -> read_bytes and '
        'orchestrator_started_at -> read_text). Filesystem, the limb task '
        '3778\'s subprocess-only vocabulary omitted; distinct from the '
        '_escalate cluster in the same file. Follow-up filed by task 4484 '
        'step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        'c5a47e52ad4b',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): _run_remediation_pass read_texts '
        'the orchestrator state files inline on the loop thread (a direct '
        'read_text, plus read_scheduler_state -> read_bytes and '
        'orchestrator_started_at -> read_text). Filesystem, the limb task '
        '3778\'s subprocess-only vocabulary omitted; distinct from the '
        '_escalate cluster in the same file. Follow-up filed by task 4484 '
        'step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        '9e5ae6eb2503',
        'to_file',
        'ROOT CAUSE (one defect, 3 rows): _run_remediation_pass read_texts '
        'the orchestrator state files inline on the loop thread (a direct '
        'read_text, plus read_scheduler_state -> read_bytes and '
        'orchestrator_started_at -> read_text). Filesystem, the limb task '
        '3778\'s subprocess-only vocabulary omitted; distinct from the '
        '_escalate cluster in the same file. Follow-up filed by task 4484 '
        'step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        '8d420bcc982d',
        'filed',
        'ROOT CAUSE (one defect, 4 rows): the live-workflow git probes in '
        'services/live_workflow_detector.py run sync subprocess.run(git) '
        'per task, reached from these coroutines with no hop. OWNED BY TASK '
        '3778 -- do not file again; 3778 measured 29.2s (dark_factory) / '
        '43.4s (reify) per render and its Part 2 names these exact '
        'consumers (_render_live_workflow_section, '
        'memory_consolidator._build_live_workflow_section at both call '
        'sites, and harness\'s is_workflow_live_for_task use) as the '
        'propagation set. 3778 is unmerged (deps 2964, 3751 open); when it '
        'lands these rows go stale.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        'dbf8ae2eb1dc',
        'to_file',
        'ROOT CAUSE (one defect, 9 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on the '
        'loop thread, so this trips INV-8\'s fan-out limb as well as its '
        'blocking limb, and the cost grows with queue history rather than '
        'staying constant. 9 async callers, one fix (offload _escalate, or '
        'pre-fetch resolved_fps once per cycle -- the resolved_fps kwarg '
        'already exists for exactly that). Follow-up filed by task 4484 '
        'step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/harness.py',
        'ReconciliationHarness._run_remediation_pass',
        '1812aa52aaca',
        'to_file',
        'ROOT CAUSE (one defect, 9 rows): the sync '
        'ReconciliationHarness._escalate reaches '
        '_finding_recently_resolved, which read_texts EVERY escalation '
        'record under the queue root AND its archive via '
        'iter_all_escalation_paths -- an UNBOUNDED per-call fan-out on the '
        'loop thread, so this trips INV-8\'s fan-out limb as well as its '
        'blocking limb, and the cost grows with queue history rather than '
        'staying constant. 9 async callers, one fix (offload _escalate, or '
        'pre-fetch resolved_fps once per cycle -- the resolved_fps kwarg '
        'already exists for exactly that). Follow-up filed by task 4484 '
        'step-9.',
    ),

    # ---- reconciliation/stages/memory_consolidator.py ----
    (
        'fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py',
        'MemoryConsolidator.assemble_payload',
        '4022ce2fecc3',
        'to_file',
        'assemble_payload reaches _assemble_remediation_payload, which '
        'read_texts remediation inputs on the loop thread. Adjacent to -- '
        'but not covered by -- task 3778\'s live-workflow propagation set, '
        'which names only _build_live_workflow_section in this file. '
        'Follow-up filed by task 4484 step-9.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py',
        'MemoryConsolidator.assemble_payload',
        '2fabd3e9b396',
        'filed',
        'ROOT CAUSE (one defect, 4 rows): the live-workflow git probes in '
        'services/live_workflow_detector.py run sync subprocess.run(git) '
        'per task, reached from these coroutines with no hop. OWNED BY TASK '
        '3778 -- do not file again; 3778 measured 29.2s (dark_factory) / '
        '43.4s (reify) per render and its Part 2 names these exact '
        'consumers (_render_live_workflow_section, '
        'memory_consolidator._build_live_workflow_section at both call '
        'sites, and harness\'s is_workflow_live_for_task use) as the '
        'propagation set. 3778 is unmerged (deps 2964, 3751 open); when it '
        'lands these rows go stale.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/stages/memory_consolidator.py',
        'MemoryConsolidator._format_assembled_payload',
        '2fabd3e9b396',
        'filed',
        'ROOT CAUSE (one defect, 4 rows): the live-workflow git probes in '
        'services/live_workflow_detector.py run sync subprocess.run(git) '
        'per task, reached from these coroutines with no hop. OWNED BY TASK '
        '3778 -- do not file again; 3778 measured 29.2s (dark_factory) / '
        '43.4s (reify) per render and its Part 2 names these exact '
        'consumers (_render_live_workflow_section, '
        'memory_consolidator._build_live_workflow_section at both call '
        'sites, and harness\'s is_workflow_live_for_task use) as the '
        'propagation set. 3778 is unmerged (deps 2964, 3751 open); when it '
        'lands these rows go stale.',
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
        '4484 step-9, since one fix closes both.',
    ),
    (
        'fused-memory/src/fused_memory/reconciliation/stages/task_knowledge_sync.py',
        'TaskKnowledgeSync.assemble_payload',
        '999e4ab43b34',
        'filed',
        'ROOT CAUSE (one defect, 4 rows): the live-workflow git probes in '
        'services/live_workflow_detector.py run sync subprocess.run(git) '
        'per task, reached from these coroutines with no hop. OWNED BY TASK '
        '3778 -- do not file again; 3778 measured 29.2s (dark_factory) / '
        '43.4s (reify) per render and its Part 2 names these exact '
        'consumers (_render_live_workflow_section, '
        'memory_consolidator._build_live_workflow_section at both call '
        'sites, and harness\'s is_workflow_live_for_task use) as the '
        'propagation set. 3778 is unmerged (deps 2964, 3751 open); when it '
        'lands these rows go stale.',
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
        '4484 step-9, since one fix closes both.',
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
        'by task 4484 step-9.',
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
        'call count. Follow-up filed by task 4484 step-9.',
    ),

    # ---- server/main.py ----
    (
        'fused-memory/src/fused_memory/server/main.py',
        'run_server',
        'f38dc5fc1e4c',
        'accepted',
        'ACCEPTED: run_server calls build_known_projects_map '
        '(yaml.safe_load) during process STARTUP, before the server binds '
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
        'four. Follow-up filed by task 4484 step-9.',
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
        'four. Follow-up filed by task 4484 step-9.',
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
        'four. Follow-up filed by task 4484 step-9.',
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
        'four. Follow-up filed by task 4484 step-9.',
    ),

    # ---- server/tools.py ----
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server._claim_commit_presence',
        '4d52656d51f0',
        'to_file',
        'ROOT CAUSE (one defect, 2 rows): the claim-verification MCP '
        'handlers _claim_commit_presence and _completion_claim_gate call '
        'recon_claim_verification_guard\'s make_commit_probe / verify_claims '
        'INLINE on the loop thread, each reaching subprocess.run(git). Task '
        '3778\'s census asserted this module was "already offloaded at its '
        'call sites" -- true of the callers it looked at, false of these. '
        'This pair IS the counter-example that made task 4484 necessary. '
        'Follow-up filed by task 4484 step-9.',
    ),
    (
        'fused-memory/src/fused_memory/server/tools.py',
        'create_mcp_server._completion_claim_gate',
        '01e23bf817cf',
        'to_file',
        'ROOT CAUSE (one defect, 2 rows): the claim-verification MCP '
        'handlers _claim_commit_presence and _completion_claim_gate call '
        'recon_claim_verification_guard\'s make_commit_probe / verify_claims '
        'INLINE on the loop thread, each reaching subprocess.run(git). Task '
        '3778\'s census asserted this module was "already offloaded at its '
        'call sites" -- true of the callers it looked at, false of these. '
        'This pair IS the counter-example that made task 4484 necessary. '
        'Follow-up filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
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
        'filed by task 4484 step-9.',
    ),
]

#: Derived, never hand-maintained beside the rows -- a second list would drift.
ALLOWLIST_KEYS: list[tuple[str, str, str]] = [
    (relpath, qualname, content_hash)
    for relpath, qualname, content_hash, _disposition, _justification in AUDITED_SITES
]
