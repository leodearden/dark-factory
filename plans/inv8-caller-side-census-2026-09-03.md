# INV-8 caller-side census — audit of task 3778's methodology

**Task 4484.** Audits why task 3778's INV-8 census missed sites that tasks
4091, 4201 and this audit each later found, and lands a mechanical
caller-side gate so the next miss is a red test rather than a fourth
discovery.

**The complete per-site record is `shared/tests/loop_blocking_allowlist.py`.**
This report carries the analysis and the cluster-level triage only; it does
not restate the rows (INV-9, one-fact-one-home). There is deliberately no
`.json` twin: the ratchet already requires that list to exist as code, and a
second copy of the same machine-read rows would go stale the moment task 4201
lands and removes two of them, with nothing reconciling the two homes.

**No generation timestamp appears anywhere in this report or in the ledger**,
so re-running the scan and diffing is a meaningful reproducibility check
rather than a guaranteed diff.

**To regenerate:**

```
cd shared && uv run pytest tests/test_loop_blocking_gate.py -q
```

A failure names every site whose disposition is missing (`unblessed`) or whose
blessing outlived the site (`stale_keys`).

Measured at HEAD `6696f1ce0c`: **167 files scanned, 60 findings.**

---

## 1. What task 3778's census was, and where it lives

There is no `plans/` artifact for it. The enumeration lives verbatim in task
3778's own `description`, as its final paragraph, headed *SCOPE OF THE DEFECT
CLASS*:

> only 7 sync `subprocess.run` sites exist in the whole fused-memory server,
> across 3 modules — live_workflow_detector.py (3, the culprit),
> recon_claim_verification_guard.py (3, **already offloaded at its call
> sites**), models/scope.py (1, `resolve_main_checkout`, cached but a cold
> miss runs on the loop thread). shared/ and escalation/ have zero.

Task 3778's *fix* is not in question and this audit changes none of it. Its
root-cause diagnosis was measured to the kernel (`kernel_clone` samples,
1904 child processes in 18 minutes). Only the census paragraph is wrong, and
it was already wrong when written.

## 2. The two gaps, and which miss each explains

**Gap A — definition-side enumeration with a per-module offload claim.**
The census counted the sites where the *primitive* is written ("7 sync
`subprocess.run` sites ... across 3 modules") and then made a per-MODULE
claim about one of them: `recon_claim_verification_guard.py` is "already
offloaded at its call sites". That is true of the callers it looked at and
false of the others. One caller wrapping a helper in `asyncio.to_thread`
makes the whole module read as clean; every other caller is invisible. The
shape predicts the observed misses exactly, and this audit found the direct
counter-example: `server/tools.py::create_mcp_server._claim_commit_presence`
and `::_completion_claim_gate` call that very module's `make_commit_probe` /
`verify_claims` inline on the loop thread.

The right census question is *which CALLERS reach this primitive without a
hop*, not *is this module offloaded*.

**Gap B — a one-primitive vocabulary.** The census enumerated
`subprocess.run` and nothing else. INV-8's own Rule text names "subprocess,
network, **filesystem, lock**, sleep". `read_text` / `write_text`,
`yaml.safe_load` / `safe_dump` and `fcntl.flock` were never in the
enumeration at all. Both of task 4201's misses and one of task 4091's are
filesystem, not subprocess — so Gap B alone accounts for them even if the
census had been caller-side. This gap was not previously identified.

The two are independent. Fixing either one alone would still have missed
sites.

## 3. Closed-out observation: task 3778 is unmerged, and that is NOT the cause

Re-measured in this worktree at HEAD `6696f1ce0c`: `git merge-base
--is-ancestor` returns non-zero against `origin/main` for `9ae7cc8cad`,
`a8064949e3`, `90484c4d9f` and `b89e13189f`; `git branch -a --contains
9ae7cc8cad` lists only `task/3778`. Task 3778's status is `pending` with
dependencies 2964 and 3751 still open.

This is expected in-flight state. It is recorded here so the observation is
closed rather than re-investigated: the census paragraph quoted in §1 was
already wrong when it was written, independently of whether the branch has
landed. Do not re-open this line of enquiry.

## 4. The caller-side result

60 findings across 167 files in `fused-memory/src`, collapsing to ~14 root
causes. Severity triage, by kind:

**Unbounded fan-out (worst).** `reconciliation/harness.py::ReconciliationHarness._escalate`
— 9 async callers. The sync `_escalate` reaches `_finding_recently_resolved`,
which `read_text`s *every* escalation record under the queue root **and its
archive** via `escalation/queue.py::iter_all_escalation_paths`. The per-call
cost grows with queue history rather than staying constant, so this trips
INV-8's fan-out limb as well as its blocking limb. A `resolved_fps` kwarg
already exists for pre-fetching once per cycle.

**Hot-path, lock-held.** `middleware/task_curator.py`'s four
`async def _maybe_*` registry guards (§5) run under the per-project curator
write lock on every task submission.

**Inline multi-primitive.** `server/manifest_stamping.py::_stamp_capability_manifests_impl`
does `read_text` + `yaml.safe_load` + `write_text` + `yaml.safe_dump` in one
coroutine body, with no helper anywhere for a definition-side census to point
at. Task 4201 measured `yaml.safe_load` at **8.15 ms for an 11 KB document** —
the same order as a subprocess spawn — and this coroutine pays it twice plus
two filesystem round trips.

**Lock waits.** The orchestrator-liveness probe (`middleware/ticket_janitor.py::_orchestrator_running`,
`middleware/curator_escalator.py::CuratorEscalator._orchestrator_running`, and
`ticket_janitor.py::_surface_probe_defect` which reaches it) takes an
`fcntl.flock` on the loop thread — bounded only by another process's hold
time. The lock limb Gap B omitted.

**Cold-miss only, cached.** `server/tools.py::create_mcp_server._normalize_project_root`
— 22 async MCP handlers reach `models/scope.py::resolve_main_checkout`, whose
`subprocess.run(git)` is memoised in `_MAIN_CHECKOUT_CACHE`. This is the one
site 3778's census *did* flag as a cold-miss risk; it was simply never acted
on. Two more callers reach the same helper from
`reconciliation/stages/task_knowledge_sync.py::_write_task_count_snapshot` and
`reconciliation/stale_priority_override_edge_sweep.py::read_live_override_state`,
so one fix closes 24 rows.

**Startup-only — legitimately accepted, not a defect.**
`server/main.py::run_server` calls `build_known_projects_map`
(`yaml.safe_load`) before the server binds, so there is no concurrent work to
stall and no request latency to affect. This is the sole `accepted` row, and
it states what makes the cost acceptable rather than merely that the call
exists. Severity triage is part of the deliverable precisely so the ledger is
not 60 undifferentiated defects.

**Already owned.** 6 rows are `filed`: 2 to task 4201 and 4 to task 3778's
named live-workflow propagation set. This audit files nothing for those.

## 5. Headline finding: four same-shape sites in one file, two of them unfiled

`fused-memory/src/fused_memory/middleware/task_curator.py` has **four**
adjacent `async def _maybe_*` guards that lazily load a YAML registry off
disk on the loop thread with no `asyncio.to_thread` hop:

| coroutine | callee | owner |
|---|---|---|
| `TaskCurator._maybe_blocklist_drop` | `load_blocklist` | **none before task 4484** |
| `TaskCurator._maybe_premise_refuted_drop` | `load_premise_registry` | task 4201 |
| `TaskCurator._maybe_premise_refuted_drop` | `premise_refuted_entry` | task 4201 |
| `TaskCurator._maybe_route_deterministic` | `load_operational_registry` | **none before task 4484** |

Task 4091 fixed a fifth neighbour in the same file (now visible as
`asyncio.to_thread`). Task 4201's own description already suspected the
pattern — "worth checking whether 3778's census methodology systematically
missed the task_curator.py callers" — and this audit confirms it did, and by
more than 4201 knew: two of the four had no task at all.

The point is not the four sites. It is that a definition-side census could
not see this shape *in principle*, while a caller-side one names all four in
one pass. That is why the deliverable is a scanner and a ratchet rather than
another prose census: a prose census is exactly what task 3778 produced, and
it stayed uncorrected across three subsequent discoveries.

## 6. Filed follow-ups

One task per ROOT CAUSE, not per call site.

_Filed in task 4484 step-9.  The ticket id for each root cause is also
carried in the justification of every one of its rows in
`shared/tests/loop_blocking_allowlist.py`, so a reader at a site reaches its
follow-up without coming through this report._

`submit_task` returns a TICKET, not a task id: the curator decides
create / combine / drop asynchronously.  Rows therefore stay `to_file` rather
than `filed` in the ledger until a real task id exists — a `filed` row whose
justification names no task id would point nowhere, and the gate's
`test_filed_entries_name_their_task` enforces that.  Flipping a row to `filed`
with its task id is the follow-up work when the tickets resolve.

Eleven root causes, all 53 `to_file` rows.  The `scope.resolve_main_checkout`
cold miss is one ticket covering two clusters (the 22 MCP handlers and the 2
reconciliation callers) because one fix closes both.

| root cause | rows | ticket |
|---|---|---|
| `task_curator.py` — the two previously-unfiled `_maybe_*` registry loaders | 2 | `tkt_0RT7QW5E7RQ3FHF2HQ0BRC6MH0` |
| `scope.resolve_main_checkout` cold miss (22 MCP handlers + 2 recon callers) | 24 | `tkt_0RT7QWVY61QYCHFCBE6KDTX7TQ` |
| `ReconciliationHarness._escalate` — unbounded queue+archive scan | 9 | `tkt_0RT7QXR5AXGWVADW9T3S4DPC2M` |
| `manifest_stamping._stamp_capability_manifests_impl` — 4 inline primitives | 4 | `tkt_0RT7QYENVS6J9WVWCY3FJAVNFR` |
| `_orchestrator_running` — `fcntl.flock` on the loop (the LOCK limb) | 3 | `tkt_0RT7QZ4R9MQHJP4MKS78DXQ2Z9` |
| `backlog_policy.py` — judge-halt record read/write | 3 | `tkt_0RT7RHRS9ZTJSQK328919XXEJW` |
| `harness.py::_run_remediation_pass` — orchestrator-state reads | 3 | `tkt_0RT7RJKQ0WXB0T87F8TJ7RTGQH` |
| `server/tools.py` claim-verification pair — `recon_claim_verification_guard` inline | 2 | `tkt_0RT7RHBAS4A3VH976CE1CJMGK8` |
| `memory_consolidator.py::_assemble_remediation_payload` | 1 | `tkt_0RT7RK22JXVDXBHRXR2PVHKQ21` |
| `targeted.py::_sweep_cancelled_descendants` → `is_orchestrator_live_for` | 1 | `tkt_0RT7RKWJG17W03JRC947R8FZZN` |
| `verify.py::CodebaseVerifier.verify.read_file` — LLM-driven call count | 1 | `tkt_0RT7RM7C7NS1ECYHFBDYP02KDJ` |

Two were filed at `medium` rather than `low`.  The `server/tools.py`
claim-verification pair, because it is the direct counter-example to task
3778's "already offloaded at its call sites" claim and reaches a `git`
subprocess on the ordinary agent-completion path — the most expensive
primitive on a hot path.  `verify.py`'s `read_file` tool, because the call
count is chosen by the verifier LLM rather than by the code, so it trips the
fan-out limb as well as the blocking limb.  Everything else is `low`.

Nothing was filed for the one `accepted` row, and nothing was filed for the 6
`filed` rows tasks 4201 and 3778 already own — those cite the owning task id
instead.

## 7. What now stands behind INV-8

`shared/tests/test_loop_blocking_gate.py` — the ratchet. A new coroutine call
site reaching a blocking primitive must be fixed or blessed with a stated
reason in the same change; a landed fix must delete its blessing, so the
ledger self-corrects. `shared/tests/loop_blocking_scan.py` — the caller-side
detector. `shared/tests/loop_blocking_allowlist.py` — the dispositioned
ledger, and the complete record this report points at.

The triad mirrors `silent_fallthrough_{scan,allowlist}` +
`test_silent_fallthrough_gate` in the same directory. INV-8's *Census seam*
section is explicit that "a slug violated repeatedly across census batches is
an enforcement gap: file a guard task"; INV-8 had been missed across three
batches (3778 → 4091 → 4201) before task 4484 was filed.
