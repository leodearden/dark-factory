# Semantically cross-paired `design_decisions` in task plans

**Task 3967.** Every live figure below was measured on **2026-08-16** with
`scripts/scan_plan_decision_pairing.py` and the greps named inline, on this
worktree. Re-run the scanner rather than trusting these numbers — they are
dated observations, not constants, and the corpus moves daily.

> **Authoring hazard — read before editing this file.** Never write a raw
> envelope sentinel here. Every angle bracket in a quoted literal below is the
> HTML entity `&#60;`, matching `docs/mcp-toolcall-xml-leak.md`; source and test
> files use the `\x3c` escape instead. Writing one verbatim would force any
> agent editing this file to emit that literal inside its own tool-call
> envelope, reproducing a defect adjacent to the one documented here.

---

## 1. The shape

A `design_decisions` entry whose `decision` and `rationale` are each perfectly
well-formed prose, but whose **association** is wrong: the rationale recorded
under one decision-line actually argues for a different one.

Both texts are intact. Nothing is malformed, no sentinel is present, no parser
is upset, nothing is truncated. The document round-trips through `json.loads`
and back without complaint. **The only thing that was lost is which text
belonged with which** — and that is not a property of any single string, so no
literal-keyed predicate can see it.

This is a **different damage class** from the envelope leakage that
`plans/toolcall-markup-containment-prd.md` addresses, and the two are disjoint
at the detector. `shared.toolcall_markup.detect` returns `None` on the majority
of the strings collected here (measured below); a mis-pairing is invisible to
it by construction, not by oversight.

The only machine-visible trace a mis-pairing leaves is the **correction entry
an author appends after noticing** — a later entry that opens on a correction
header and says, in words, that a preceding entry was mis-paired. That is what
`shared.decision_pairing` keys on, and it is the reason for the lower-bound
caveat that runs through everything below.

---

## 2. Prevalence — a strict lower bound

Measured 2026-08-16 over `/home/leo/src/dark-factory/.worktrees/.task-meta`:

```
23 mis-paired entries across 20 plans (scanned 1301 plan files, skipped 0).
```

The 20 victim plans:

> 3042, 3098, 3201, 3209, 3210, 3216, 3298, 3337, 3363, 3382, 3415, 3473,
> 3567, 3664, 3668, 3727, 3757, 3918, 4030, 4096

Three plans carry two matched entries each (3209, 3567, 4096); the rest carry
one.

**This is a STRICT LOWER BOUND, and the gap is not estimable.** The predicate
can only find a mis-pairing that a human or an agent *noticed* and then
*documented*. A mis-pairing nobody noticed leaves no trace at all — both texts
are well-formed prose, so there is nothing whatsoever to key on. Read 23 as a
floor. Never report it as "23 mis-pairings exist", and never derive a rate
from it.

### What the matches look like

| Correction header | Entries | | Pairing marker | Entries |
|---|---|---|---|---|
| `CORRECTION` | 19 | | `mis-paired` | 18 |
| `SUPERSEDES` | 2 | | `recorded against the wrong` | 2 |
| `READ THIS INSTEAD` | 2 | | `swapped` | 1 |
| | | | `cross-paired` | 1 |
| | | | `mis-titled` | 1 |

The predicate is a **conjunction** of a start-anchored header *and* explicit
pairing language. Both conjuncts are load-bearing and each is pinned by its own
test — see `shared/src/shared/decision_pairing.py`'s docstring for the
specimens that force each one. Dropping the start-anchor sweeps in task 3692's
plan (prose *about* another plan's mis-pairing) and three entries of 3209 that
merely use the word `supersedes` mid-sentence; dropping the pairing conjunct
sweeps in 3382's decision #5, a genuine design reversal.

### Nine plans carry BOTH damage classes

3201, 3209, 3216, 3337, 3363, 3382, 3473, 3727, 3757 have a mis-paired entry
that *also* carries envelope residue. That is why `envelope_leak` is a column
on every scanner record rather than a second sweep: an operator sees both
classes, and the plans suffering each at once, in one report.

### Occurrences are still landing

- The corpus was **1196** plans when this task was filed, **1299** at the start
  of this session's step 12, and **1301** at step 14 — the last two
  measurements **minutes apart**.
- The newest victim plans (3668, 4030, 4096) have mtimes of **2026-08-15**,
  the day before this measurement.
- **4030 and 4096 carry task ids above 3967's own**, i.e. they were authored
  *after* this task was filed.

This growth is exactly why **no test pins a live count**. A pinned count would
be flaky by construction, and worse it would invert the signal: a predicate
improvement that legitimately detected *more* entries would read as a
regression. Tests replay only the committed corpus
(`shared/tests/fixtures/decision_pairing_corpus.jsonl`) and synthetic
`tmp_path` trees.

---

## 3. Containment — the answer

Two **independent** findings. They are kept separate deliberately, because they
fail for different reasons and would be fixed by different work.

### (a) The write-time tripwire is installed on no production server

Measured 2026-08-16. `grep -rn add_middleware --include=*.py` over the repo
returns exactly **two** in-repo hits, both inside `shared/tests/`
(`test_mcp_markup_middleware.py:346` and
`test_mcp_markup_middleware_corpus.py:297`) — test harnesses. Every other hit
is third-party code under `.venv/`. `MarkupGuardMiddleware` is referenced
nowhere outside its own definition in `shared/src/shared/mcp_markup_middleware.py`
and those tests.

- Task **3689** (build `MarkupGuardMiddleware`) — `done`.
- Task **3690** (register it on all four servers) — **`pending`**.

**So containment at plan-tools is currently ZERO, for BOTH damage classes.**
This is a genuine and damning observation about today, and it is recorded here
rather than as a test: an assertion that the middleware is registered nowhere
would go RED the moment 3690 lands, punishing the work that closes the gap.

### (b) Even once 3690 lands, the tripwire cannot see this shape

The middleware decides by running `shared.toolcall_markup.detect` over each
incoming string argument. Measured over the committed corpus: **34 of the 46
strings** across the 23 positive entries carry **no envelope literal at all**,
so `detect` returns `None` and the write is admitted.

This is not reasoning from the detector's contract — it is a pinned
measurement, taken today without waiting for 3690, by driving the **real**
`MarkupGuardMiddleware` through the shared in-process harness with a
cross-paired `add_design_decision` call. See
`shared/tests/test_decision_pairing_containment.py`:

| Test | What it establishes |
|---|---|
| `TestDetectorBlindness::test_the_envelope_detector_is_blind_to_every_envelope_clean_positive` | The two damage classes are disjoint at the detector. |
| `TestTripwireAdmitsCrossPairedCalls::test_the_call_is_admitted` | The middleware admits a cross-paired call. |
| `…::test_both_arguments_reach_the_tool_verbatim` | Both arguments arrive unmodified. |
| `…::test_no_fact_and_no_escalation_are_emitted` | Nothing is recorded anywhere. |
| `…::test_envelope_damage_on_the_same_tool_does_fire` | The control: same tool, same harness, same policy — only the damage class differs, and *that* one is caught. |

Every case runs under **both** `REJECT_WITH_REPAIR` (plan-tools' declared
policy in PRD section 4, C2) and `FORWARD_REPAIR`, so the finding is a property
of the **detector**, not of one policy tier.

Cite the tests, not this paragraph — they are re-runnable and this is not.

---

## 4. Why repair is impossible

`shared.toolcall_markup.repair` can exist because envelope damage leaves
**residue**: the original argument text is still present with parseable markup
wrapped around it, so a repair is a *slice of its own input*.

Neither precondition holds here.

- There is no residue to strip. Both fields are clean prose.
- There is nothing to reconstruct. A mis-paired document carries both texts but
  **no record of which belonged where**. The association itself is what was
  lost, and it is not recoverable from the document.

A "repair" would therefore have to *guess* which rationale belongs to which
decision — and a guess written back into a plan is strictly worse than the
visible damage, because it launders an unknown into a confident-looking record
that no later reader can tell from a genuine one.

Task **3692** reached the same conclusion from the other direction: it
deliberately asserted no test that task 3567's plan is repaired, despite 3567
being its named damage specimen (its own decision [8]). That restraint was
correct, and this task inherits it rather than re-litigating it.

---

## 5. Why deterministic write-time containment is impossible

Symmetrically, no **local deterministic** predicate can contain this at the
write boundary. A correct `(decision, rationale)` pair and a swapped one both
arrive as clean, well-formed prose. Nothing *in the arguments* distinguishes
them — the difference is a semantic relation between two texts, and at the
moment of the call there is no ground truth to compare against.

An LLM coherence check is the one thing that could. The toolcall-markup PRD's
decision **D2 already rejected LLM mediation on this exact surface**: it would
put the same model class that emits the defect into the write path.

That is not a gap awaiting a cleverer regex. It is why the **correction entry
an author appends after noticing** is not merely the easiest signal to key on —
it is the only machine-visible one that exists.

---

## 6. Mechanism — what the wire evidence refutes, and what it does not

The originating escalation hypothesised a **harness argument-boundary
over-consumption** defect: that the tool-call parser mis-attributed argument
text, producing the swap. That hypothesis is **refuted on the one specimen
where it can be tested cleanly.**

Archived transcripts under `data/orchestrator/agent-transcripts/<task>/` retain
the raw `tool_use` records, including `add_design_decision` `input` dicts as
they arrived on the wire. For **task 3727, all 7 on-disk `(decision, rationale)`
pairs are byte-identical to a wire `tool_use` input** — including the
mis-paired entry [0] that entry [1] corrects, which is the discriminating case.

plan-tools wrote **exactly what it received**. The cross-pairing was already
present in the arguments the model composed. A parser over-consuming an
argument boundary cannot produce a well-formed swap spanning two separate
`tool_use` blocks.

Pinned by `shared/tests/test_decision_pairing_containment.py::TestWireEvidence`
against `shared/tests/fixtures/decision_pairing_wire_evidence.json`, which is
committed because `data/` is gitignored and retention-bounded.

### Scope limit — state it whenever the finding is cited

**This is one clean specimen.** Tasks **3567** and **4096** do *not* reproduce
byte-identity on every field — consistent with task 3692's read-time envelope
repair having rewritten them — and are therefore **INCONCLUSIVE**. They must
not be cited as evidence in either direction, and no wire record for them is
committed (`TestWireEvidence::test_no_inconclusive_specimen_is_committed_as_evidence`
enforces that).

Nothing here licenses a general claim about every victim. The supported finding
is narrow and worth stating precisely: **on at least one clean specimen the
defect was model-authored, so for that specimen the fix does not belong at the
harness boundary.**

---

## 7. Policy for the known victims: detect and report, never repair

**The 20 victim plans are not edited by this task, and should not be edited by
a script.** Two independent reasons:

1. They live under gitignored `.worktrees/`. An edit is untracked mutation of
   runtime state that a **live task may be reading right now** — not a
   reviewable code change. This is also why the scanner's read-only contract is
   *asserted* (bytes and `st_mtime_ns` snapshotted before and after every scan,
   including on the unparseable, undecodable and unreadable error paths)
   rather than merely documented. Those error paths are also *total*: every
   per-file failure is reported as a skip and the sweep continues, so one bad
   plan can never hide the rest of the corpus and silently understate a number
   that is already only ever a lower bound.
2. The go-forward fix is **task 3865's declared territory** — its plan already
   names `supersede_design_decision`, a `status='superseded'` field, and
   `active_design_decisions(plan)` across three read paths. 3865 is `pending`;
   building either half here would duplicate a pending task.

No reader-side filter or writer-side supersede tool is added to `plan_tools.py`
here. A read-time warning there would also be near-worthless: a task's
plan-tools reads only its **own** plan, so it would fire at the one reader who
already knows — the author who just wrote the correction.

### What this task shipped instead

| Artifact | Purpose |
|---|---|
| `shared/src/shared/decision_pairing.py` | The single owner of both marker sets and of the predicate (INV-5). Detection only; no repair counterpart. |
| `scripts/scan_plan_decision_pairing.py` | Re-runnable READ-ONLY CLI. `--root`, `--json`, `--fail-on-hit`, `--require-scanned`. No `--apply`, ever. |
| `shared/tests/fixtures/decision_pairing_corpus.jsonl` + README | 23 positives and 6 negative controls, so prevalence survives `.worktrees/` churn. |
| `shared/tests/fixtures/decision_pairing_wire_evidence.json` | 3727's on-disk pairs beside its wire inputs, so the mechanism finding survives the `data/` retention window. |
| `shared/tests/test_decision_pairing_containment.py` | The containment measurement and the wire-evidence replay. |

**Re-running the sweep:**

```bash
python scripts/scan_plan_decision_pairing.py            # human report, exit 0
python scripts/scan_plan_decision_pairing.py --json     # machine-readable
python scripts/scan_plan_decision_pairing.py --fail-on-hit   # exit 1 on any hit

# The invocation that is SAFE unattended (CI job, timer):
python scripts/scan_plan_decision_pairing.py --fail-on-hit --require-scanned 1
```

`--root` defaults to this checkout's `.worktrees/.task-meta` and resolves
correctly from a task worktree as well as from the main checkout.

**Exit 0 is not by itself evidence of a clean corpus, and `--fail-on-hit` alone
is not a safe gate.** That flag keys only on hits, and a `--root` that is
mistyped, not created yet, or unlistable yields none — so a sweep that read
*nothing* exits 0 exactly as a clean one does. Interactively that is harmless,
because every run prints a `scanned N plan files` summary and a human reads it;
unattended it is not, because only `$?` is consumed. `--require-scanned N`
states the coverage floor in the exit code: **exit 3** when fewer than N plan
files were actually read, which is never a clean run. A coverage failure
outranks a hit in the code (3 beats 1) — both fail, and a sweep short of its
floor found an unknown fraction of what is there. Exit 2 is a usage error;
a negative `N` is rejected rather than silently disabling the gate.

---

## 8. Cross-references

- `plans/toolcall-markup-containment-prd.md` — the envelope-leakage PRD. This
  is a damage class it **does not cover**, and section 3(b) above measures why
  its C2 tripwire cannot be extended to cover it.
- Task **3689** (`done`) — `MarkupGuardMiddleware` itself.
- Task **3690** (`pending`) — registers that middleware on all four servers.
  Until it lands, containment at plan-tools is zero for both damage classes.
- Task **3692** (`done`) — read-time envelope repair; declined to assert repair
  of 3567, and its rewrites are why 3567/4096 are inconclusive as wire evidence.
- Task **3865** (`pending`) — the supersede mechanism and reader-side filter.
  The go-forward fix for everything catalogued here.
- `docs/mcp-toolcall-xml-leak.md` — the sibling damage class, and the source of
  the `&#60;` spelling convention used in this file.
