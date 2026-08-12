# Retrospective sweep: unverified completion claims in `temporal_facts` / `decisions_and_rationale`

**Task 3381 · esc-3085-1 second ask · swept 2026-08-12T13:08:06Z · dark-factory `a138751e80`**

Artifacts in this directory:

| file | what it is |
|---|---|
| `report.json` | byte-for-byte the sweep script's stdout, unedited |
| `provenance.json` | run metadata the report itself could not know |
| `investigation.md` | this file — the human adjudication |

---

## Headline

| measure | value |
|---|---|
| Episodes scanned | **7577** (dark_factory 3030 + reify 4547) |
| Episodes carrying ≥1 completion claim | **1647** |
| Raw `mismatch` verdicts | **166** |
| Raw `unverifiable` verdicts | **30** |
| **Confirmed new fabrications of the esc-3085-1 shape** | **0** |

`mismatch` and `unverifiable` are different facts and are **never summed**, here or in
`report.json`.

**The "two in 17 hours is a rate" hypothesis was NOT borne out.** The sweep found no
second-generation instance of the incident that motivated it. That is a real result, not a
failed run — but read the *Denominator* and *Caveats* sections before treating it as "the
corpus is clean", because this sweep is biased toward under-counting by construction.

The 166 raw mismatches are the **script's verdicts, not adjudicated fabrications**. Every
one was adjudicated by hand below, and all 166 fall into three systematic false-positive
classes. `report.json` is committed unedited so this adjudication can be re-checked
against the raw output.

### Breakdown as reported

By category (all 196 findings): `decisions_and_rationale` 112 · `temporal_facts` 84
By claimed project: `dark_factory` 126 · `reify` 55 · `know_live` 10 · `knowlive` 4 · `solar_challenge_platform` 1
By subject: `task` 161 · `commit` 35 · **`ticket` 0**

**Zero ticket findings is a real negative.** The ticket store was available and initialized
for this run (`provenance.json` → `authorities_resolved`), so the absence is not an
unavailability artifact. esc-3085-1 instance (2)'s exact shape — a completion claim naming
a `tkt_` id the registry does not hold — **did not recur anywhere in the swept corpus.**

---

## Adjudication of all 166 mismatches

### Class A — `filing_dispatch` about a task that EXISTS but is not terminal — **125 findings**

Observed states: 64 `pending`, 52 `deferred`, 7 `in-progress`, 1 `merge-deferred`, 1 `blocked`.

`_verify_task` (completion_claim_gate.py:550) verifies **only** a TERMINAL status, for every
claim kind. But a `filing_dispatch` claim asserts the work *"was filed/queued/cancelled
somewhere"* (:292) — its truth condition is that the artifact **exists**, not that it is
finished. A task reported `pending` is positive evidence the filing claim was TRUE.

The gate's own ticket path already adjudicates filing this way: `_verify_ticket` (:578)
returns `verified` when the row merely EXISTS, whatever its status. The task path does not
make that distinction, so every truthful "filed as task N" about still-open work is flagged.

Examples (full listing in `report.json`):

| claim | project | observed | episode | derived edges |
|---|---|---|---|---|
| task 4136 | reify | deferred | `1879d86f-6ec2-43d6-a5a8-88e6e445a068` | 7 |
| task 4263 | reify | pending | `60f904c8-4354-453a-9313-9555bc77b33b` | 4 |
| task 4746 | reify | pending | `994dec4c-5024-4afd-8afd-a8f66330e3ef` | 4 |

> "The user-observable signal was NOT dropped — it was **refiled as task 4263**"
> "To close the hex/wedge Phase A wiring gap, **filed task 4746** (…)"

Both statements are true, and both tasks exist. **Verdict: false positives.**

### Class B — `applied_work` about a task, prose false positives — **11 findings**

These required reading each episode. None asserts that the named task is complete; the task
id merely co-occurs with completion-ish phrasing.

| task | project | observed | episode | what the text actually says |
|---|---|---|---|---|
| 5623 | reify | in-progress | `1554dafd` | *"task 5623 has landed nothing (still pending)"* — asserts the **opposite** |
| 3267 | dark_factory | pending | `181595f7` | *"**filed** dark-factory task 3267 … as the durable fix"* — a filing statement |
| 3846 | dark_factory | pending | `448d8400` | *"task 3846 **was filed** to give recon's stranded-in-progress finding a landed-on-main check"* |
| 5208 | reify | pending | `2df7dda7` | *"…once 5208 **lands**"* — explicitly future |
| 5317 | reify | pending | `1eefbd1e` | *"task 5317 … **was EXPANDED** to also own …"* — a scope change |
| 4687 | reify | deferred | `b6153503` | the capability *manifest* landed; 4687 is the decompose task |
| 6164 | reify | pending | `3fde06f9` | the real claim is the commit `9ca5c6ad9f`, not the task |
| 3541, 3584, 3578, 3539 | dark_factory | pending | `56052308`, `e64dbe6f`, `baaebac9`, `448d8400` | prose about what tasks *own* or *assert*, not completion |

The `5623` case is the sharpest: the sentence states the task landed nothing and is still
pending, and the extractor flagged it as a completion claim. The negation stripper does not
reach the construction *"has landed nothing"*. **Verdict: false positives.**

### Class C — commit claims — **30 findings**

Checked out of band against both known checkouts with `git cat-file -e <sha>^{commit}`:

| outcome | count |
|---|---|
| Commit **PRESENT** in `dark_factory`, but the sweep probed `reify` | **12** |
| Absent from both known checkouts | 18 |

**The 12 are confirmed false positives, and they are the most instructive finding here.**
The claims are true; the sweep searched the wrong repository. The prose names its repo
explicitly and the probe ignored it:

| sha | probed in | actually present in | claimed text |
|---|---|---|---|
| `5353d209e4` | reify | **dark_factory** | "LANDED on **dark-factory** main (commit 5353d209e4…)" |
| `8face929e4b6…` | reify | **dark_factory** | "already landed via **dark_factory** task 2053 (done, merged commit …)" |
| `ba3626d5` | reify | **dark_factory** | "**dark_factory** task 2879 (…done via commit ba3626d5)" |
| `d9ea04876f` | reify | **dark_factory** | "**dark-factory** commit d9ea04876f merged as be8b778aa9" |
| `72ddf7d1e268…` (×2) | reify | **dark_factory** | cross-project rerouting note naming `dark_factory` |
| `6f29517823`, `4287de86d23bf` (×2), `a05549ab8597…`, `ed251c9b586b…`, `a51f8e4d5f` | reify | **dark_factory** | — |

A reify agent recording a fact about a dark-factory commit has its claim attributed to
`reify` (the writer's project), and the probe then searches reify's object store. The
commit is absent there, so the gate reports `no commit X in project 'reify'` — a
**fabrication accusation against a truthful writer**.

The remaining **18** are absent from both checkouts and are **indeterminate, not confirmed
fabrications**. The benign explanations dominate: dark-factory's merge lane rebases, so a
pre-merge task-branch sha cited in an episode is garbage-collected after the merge (several
of these texts cite exactly that — *"merged as …"*, *"Branch task/3135 tip commit …"*), and
some name a third repository this sweep never opened (`dark-factory/taskmaster-ai` for
`82dccbfa`). Two more (`42c08a8e60`, `f12ea587c5`) come from an episode *narrating* a
mistaken landing belief — meta-discussion about a fabrication, not one. Adjudicating the 18
would require per-sha reflog/third-repo archaeology; none is actionable as written.

---

## How far the denominator can be trusted

- **The corpus is live and the denominator moved during the investigation.** The
  `dark_factory` graph grew 2995 → 3030 Episodic nodes over ~2.5h between two measurements
  (`provenance.json`). 7577 is a snapshot at run time, not a fixed population.
- **The sweep systematically under-counts** (Caveat 1). Authorities are read *today*, not at
  write time: a claim written while its task was in-progress verifies now that the task has
  gone terminal. Most of the corpus predates this run by weeks, so this bias is large and
  one-directional. **A low mismatch count is not evidence of a clean corpus.**
- **Detection needs a named ref** (Caveat 6). A fabricated completion phrased without a task
  id / sha / `tkt_` id is invisible here by construction.
- **Graphiti episodes only** (Caveats 4-5). Mem0-resident records, orphaned edges whose
  source episode was deleted, and `add_episode` writes carrying a caller-supplied
  description (hence no category) are all outside scope.
- **The originating incident's own record is in the corpus and this sweep does not flag it.**
  See below — it is the sharpest available demonstration that the under-count is real
  rather than theoretical.

### The esc-3085-1 instance-(2) record was swept and came back clean

Observed (all measured 2026-08-12, dark-factory `a138751e80`):

- Episode `02090224-7bc9-4485-9291-6748e1042ac9` is **still present**, in the **reify**
  graph, with `source_description = add_memory:temporal_facts` — in scope for this sweep,
  and carrying **no** `[unverified_claim]` prefix (it predates the task-3142 gate).
- It appears in **none** of this sweep's 196 findings.
- Ticket `tkt_0RRRC5AASJ9Z630VP4PCN9H376` — the ref that record claims — **does exist** in
  `data/reconciliation/tickets.db`: `status='combined'`, `task_id='5638'`,
  `created_at=2026-07-27T08:30:56Z`, `resolved_at=2026-07-27T08:31:29Z`, and
  `reason` beginning *"drop: Candidate is fully covered by pool Task 5638 …"*.

So the sweep behaved **correctly**: `_verify_ticket` accepts a row that exists, the row
exists, and the claim verifies. The ticket was not invented.

*Hypothesis (not established here):* the residual falsehood in that record is not the
ticket's existence but its **disposition**. The candidate was *dropped* as a duplicate of
reify task 5638, and `task_id` on the row is `5638` — a **reify** task — so the record's
assertion that the work was *"re-filed into dark_factory's task tree"* did not
materialize as a dark_factory task. An existence-only ticket check cannot see that
distinction, so **the current write-time gate would not catch esc-3085-1 instance (2)
either.** Confirming this would require reading esc-3085-1 itself, which is outside this
task's scope; it is carried into the follow-up rather than asserted here.

## Caveats (reproduced verbatim from `report.json`)

1. **RETROSPECTIVE BIAS, FALSE NEGATIVES:** the write-time gate reads each authority AT WRITE TIME; this sweep reads it TODAY. A claim written while its task was still in-progress VERIFIES today if that task has since gone terminal, so the sweep systematically under-counts. A low mismatch count is therefore NOT evidence the corpus is clean.
2. **RETROSPECTIVE BIAS, FALSE POSITIVES:** 'cancelled' is in shared.task_statuses.TERMINAL, so a claim about work that was later CANCELLED reads as verified here even though the work never landed.
3. **UNVERIFIABLE INDICTS NOBODY:** a task deleted since, a garbage-collected commit, or an expired ticket all read as unverifiable through no fault of the writer. 'unverifiable' and 'mismatch' are different facts and are never summed in this report.
4. **COVERAGE, STORE:** this sweep covers GRAPHITI EPISODES only. A MEM0-resident record is NOT covered, because the two categories in scope are Graphiti-primary and Mem0 is not read at all.
5. **COVERAGE, ORPHANED EDGES:** a derived RELATES_TO edge whose source episode has been deleted is NOT covered, and neither is an episode ingested through add_episode with a caller-supplied description, which carries no category and so falls outside the category scope.
6. **DETECTION BOUND:** claims are detected by the deterministic lexical vocabulary in fused_memory.services.completion_claim_gate, which requires completion PHRASING and a concrete NAMED REF (task id / commit sha / tkt_ id) to co-occur in one clause. A fabricated completion phrased without a named ref is invisible to this sweep by construction.

---

## What this found instead

The sweep's real yield is not a list of fabrications — it is **three reproducible
false-positive classes in the write-time gate itself** (task 3142), each of which causes a
truthful episode to be stamped `[unverified_claim]` in production today:

1. **Filing claims are held to a completion standard.** `_verify_task` requires TERMINAL for
   every claim kind, so a truthful *"filed as task N"* about still-open work is a mismatch.
   `_verify_ticket` already treats existence as sufficient for the same kind of claim.
   Largest class here: **125 of 166**.
2. **Cross-repo commit claims are probed against the wrong repository.** A claim is
   attributed to the writer's project even when the prose names another repo, so the sha is
   sought in the wrong object store. **12 confirmed** — each one a false accusation against
   a writer who was telling the truth.
3. **The negation stripper misses at least one construction.** *"task 5623 has landed
   nothing (still pending)"* was extracted as a completion claim.
4. **A ticket check that tests only existence is necessary but not sufficient.** The
   originating incident's own ticket exists, was `combined`/dropped, and therefore verifies
   — so the gate that exists today would very likely not have caught the record that
   motivated building it (evidence and hypothesis above).

This matters more than the null result: on the write path these misfire *now*, on every new
episode, and the tag they stamp is the very signal a future sweep would trust. Classes 1 and
2 would each have produced far more noise than signal had this sweep been run as a gate.

Filed as follow-up work rather than fixed here — this task's scope is a read-only report,
and the fixes belong in the gate module.

## Invariant

**This sweep invalidated nothing and deleted nothing.** Every episode and edge named above
is left exactly as it was, for human adjudication. The script has no `--apply`,
`--invalidate` or `--delete` path — an absence pinned by `TestReadOnlyByConstruction` — and
reads exclusively over `GRAPH.RO_QUERY`, so read-only is server-enforced rather than
client-promised.

Given the adjudication above, **no edge or episode in this corpus is recommended for
invalidation.** Acting on the raw 166 would have destroyed truthful records.

## Reproducing

```bash
cd fused-memory
PROJECT_ROOT=/home/leo/src/dark-factory \
DASHBOARD_KNOWN_PROJECT_ROOTS=/home/leo/src/dark-factory,/home/leo/src/reify \
RECONCILIATION_DATA_DIR=/home/leo/src/dark-factory/data/reconciliation \
uv run python scripts/audit_unverified_completion_claims.py \
    --project dark_factory --project reify --include-unverifiable
```

The three env vars are **required**, not optional. Without them the project registry
resolves empty and the ticket store is not found, and every task and ticket claim falls to
`unverifiable` — see `provenance.json` → `notes` for the discarded first run that did
exactly that. The script warns loudly on stderr in that case rather than failing silently,
which is how the mis-scoped run was caught.
