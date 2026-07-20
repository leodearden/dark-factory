# Routing κ (triage) — input-replayability spike report

**Task:** 2816 (κ-triage, deferred) · **PRD:** `plans/adaptive-model-routing-prd.md` §κ
**Spike run:** 2026-07-20 · **Author:** claude-interactive (investigate-df-2816)
**Verdict:** ❌ **NOT replayable — recommend DROP triage from κ.** The `models.triage`
role is **dormant in production** (0 invocations in 3.4 months); there is **no input
corpus** to replay. Undeferring 2816 would trial a model swap on a role that never runs.

> **Decision owner:** Leo. This report is the STEP-0 spike verdict + a recommendation.
> It changes **nothing** about task 2816's status or scope.

---

## TL;DR

| Question | Answer |
|---|---|
| Is κ-triage offline-replayable? | **No.** |
| Why not? | The `models.triage` call site never fires in production, and its would-be input corpus (`review_suggestions` escalations) is empty (1 record, from 2026-04-07, pre-dating the current architecture). |
| Undefer 2816? | **No.** |
| Then what? | **Drop triage from κ.** Flipping `models.triage` to haiku is **inert** — it re-models a role with **0/15,645** production invocations, delivering **zero** of the stated benefit ("account-pool cap-pressure relief"). Bootstrapping input-logging is futile (logging a site that never fires). |
| Was the original architect right? | **Yes** — "triage not offline-replayable at the named sources" is correct, and understated. The brief's reopening lead rests on a **truncated** esc-2540-1 detail. |

---

## Q1 — What is "triage" here? One notion or two? → **One mechanism, one call site.**

`models.triage` / `role='triage'` governs **exactly one** invocation:
`Steward._pre_triage_suggestions` (`orchestrator/src/orchestrator/steward.py:706-860`;
dispatched at `steward.py:768-814`, `role_name='triage'`). Default model **sonnet**
(`config.py:154`, `defaults.yaml:268`).

It fires **only** when (`steward.py:427-437`):
```python
if escalation.category == 'review_suggestions' and escalation.detail:
    suggestions = json.loads(_strip_hash_prefix(escalation.detail))
    if len(suggestions) >= self.config.suggestion_triage_threshold:   # =10
        escalation = await self._pre_triage_suggestions(escalation)
```
Its **input** is `escalation.detail` — a JSON array of review suggestions — plus the task
record (`build_triage_prompt(suggestions, self.task)`, `steward.py:725`).

**The re-scope's "inner-triage vs review-suggestions triage" is a false dichotomy.**
They are the *same* mechanism: the steward's pre-triage **is** the review-suggestions
triage. "Inner" (it runs inside the steward) and "review_suggestions" (it fires on that
escalation category) are two true descriptions of one call site — not two call sites.
esc-2540-2's parenthetical *"data/escalations/ is escalation-watcher output, NOT steward
inner-triage inputs"* conflated the queue's contents with its provenance; the triage input
**is** an escalation record's `detail`, and those records live in `data/escalations/` —
when they are produced at all (they no longer are; see Q2/Q3).

**Other "triage" in the tree is NOT this role** (ruled out):
- escalation-watcher-auto's L1 triage uses `watcher_model` (default sonnet, `config.py:3167`) — a **separate** config, not `models.triage`.
- `review_checkpoint.py`'s `f.get('triage') == 'create_task|escalate|dismiss'` (`:293-296`, `:535`) is the reviewer classifying **its own** findings inline — no `models.triage` invocation.

---

## Q2 — Are the inputs reconstructable? → **No — production bypasses them entirely.**

The inputs are structurally simple and *would* be reconstructable **if** `review_suggestions`
escalations were persisted. They are not, because the **production path bypasses the
escalation queue**:

- Primary path (`workflow.py:5349-5350`, and `:6953`): review suggestions route to the
  **curator** via `_route_review_suggestions_to_curator` → `submit_task` CandidateTask
  tickets. No `review_suggestions` escalation is created.
- The escalation path — `_escalate_suggestions` (`workflow.py:11493`, the *only* creator of
  `category='review_suggestions'`, `:11600`) — is a **fallback reachable only when
  `self.mcp is None`** (CLI / dry-run / test contexts). It has **no live caller** in a
  running orchestrator.
- The code says so directly (`workflow.py:5347-5348`):
  > `_escalate_suggestions` is retained as the steward fallback **but is no longer called
  > from this path.**

So in a live orchestrator, review-suggestion triage **inputs are never persisted** — not
because they're the wrong kind of record, but because the records are never generated.

(The triage **output** — which suggestions were accepted/skipped — is also not needed for a
D-6 replay-agreement trial: that protocol re-runs haiku *and* sonnet on the same input and
scores their mutual agreement, frontier-adjudicating disagreements. The incumbent re-run is
the reference, not the historical output. This is moot here — there are no inputs.)

---

## Q3 — Corpus size & shape → **Empty. 1 stale record; role has never fired.**

Two independent empirical sweeps (2026-07-20), both decisive:

**(a) Escalation store** — `data/escalations/` live + full `archive/` tree, **2468**
escalation records scanned:

| category | count |
|---|---|
| infra_issue | 1098 |
| task_failure | 547 |
| risk_identified | 319 |
| … | … |
| **review_suggestions** | **1** |

The single `review_suggestions` record is `esc-508-96.json`, timestamp **2026-04-07**
("Steward timeout: 11 review suggestion(s) for triage") — from **before** the curator-route
architecture. Zero produced since.

**(b) Invocation history** — `data/orchestrator/runs.db`, window **2026-04-09 → 2026-07-20**
(~3.4 months, the full production record), **15,645 invocations / 233,373 events**:

| role | invocations |
|---|---|
| implementer | 5,343 |
| reviewer_comprehensive | 4,592 |
| architect | 3,531 |
| … | … |
| module_tagger | 9 |
| **triage** | **0** |
| **judge** | **0** |

`role='triage'` = **0** invocations and **0** events. The role has **never fired** in
recorded production history. (This reproduces and corroborates the original architect's
runs.db finding — esc-2540-1 item 2 — on *current* data.)

**A D-6 replay-agreement trial is impossible here.** It requires N inputs, N repeats, a
pre-measured variance band, and frontier adjudication on disagreements. The available corpus
is 1 stale input — not a trial, a coin flip.

> **Contrast — why module_tagger (task 2540) proceeds but triage does not:** module_tagger
> replays against **ground-truth files** on disk (self-contained; it fired 9× and, more
> importantly, its inputs/labels are reconstructable from the repo tree), so its
> replay-agreement trial is well-founded. Triage has neither firings nor a persisted input
> corpus. Same κ family, opposite verdict.

---

## Q4 — Contract (dep 2485) → **Works, but moot.**

`_pre_triage_suggestions` already runs on the post-2485 verdict-tool transport: it injects
verdict-tools MCP (`_inject_verdict_tools_mcp`, `steward.py:760-762`) and reads the verdict
via `extract_triage_verdict` (`steward.py:834-835`). A replay could go through that transport
or score persisted outputs contract-agnostically (cf. eval-framework task 2478). Either way
is moot — there is nothing to replay.

---

## Q5 — The decision → **Do NOT undefer 2816. DROP triage from κ.**

**Recommendation: drop triage from κ.** Evidence:

1. **Inert flip.** `models.triage` is dormant (0/15,645). Changing its model haiku↔sonnet
   changes the behaviour of a role that **never runs** — zero risk, zero reward. The task's
   stated consumer benefit ("fleet triage dispatches; account-pool cap-pressure relief") is
   unattainable: triage consumes ≈0 of the pool.
2. **No corpus.** 1 stale `review_suggestions` record (2026-04-07); the production path
   (curator) has generated none since. A replay-agreement trial cannot be run.
3. **Bootstrapping is futile.** The PRD's fallback (§κ line 400-401: *"if not [reconstructable],
   bootstrap input logging or drop triage from κ"*) resolves to **drop** here — adding input
   logging at a call site that never fires would still yield an empty corpus.
4. **The reopening lead was based on a truncated record.** esc-2540-1's detail is cut off at
   *"…category='review_suggestions' with >= suggestion_triage_threshold ("* — the parent brief
   inferred "therefore persisted/replayable." The data shows the opposite. The architect's
   original judgment stands and was, if anything, understated.

**To enact the drop (Leo's call — not done here):** cancel task 2816 (per the
change-kind-in-place / cancel-don't-remove norm) with a one-line pointer to this report;
amend PRD §κ's triage bullet to record the drop + rationale. No new dependency; no code
change.

### Adjacent idea for Leo (NOT an undefer of 2816 — a separate, live target)

The suggestion-classification work that `models.triage` *used* to do now lives in the
**curator** (`_route_review_suggestions_to_curator` → `TaskCurator`, a **fused-memory** LLM
component with its **own** model config, independent of orchestrator `models.triage`). If the
real goal is to cheapen suggestion-classification, the curator — not the dead `models.triage`
role — is the live optimization surface. That is a **new** proposal, out of κ's
orchestrator-routing scope, and entirely Leo's to open.

---

## Appendix — reproduction

```bash
# (a) review_suggestions corpus (live + full archive)
python3 - <<'PY'
import json, glob
files = (glob.glob('data/escalations/*.json')
         + glob.glob('data/escalations/archive/*.json')
         + glob.glob('data/escalations/archive/*/*.json'))
rs = [f for f in files
      if (d:=json.load(open(f))) and isinstance(d,dict)
      and d.get('category')=='review_suggestions']
print('review_suggestions records:', len(rs))   # -> 1 (esc-508-96, 2026-04-07)
PY

# (b) triage invocation history
sqlite3 data/orchestrator/runs.db \
  "select role,count(*) from invocations group by role order by 2 desc;"   # triage: absent (0)
```

**Key source refs:** `steward.py:427-437` (fire gate), `steward.py:706-860`
(`_pre_triage_suggestions`, the sole `role='triage'` call site), `workflow.py:5347-5350`
(curator is primary; `_escalate_suggestions` "no longer called"), `workflow.py:11444-11509`
(`_route_review_suggestions_to_curator`, `_escalate_suggestions` fallback gated on
`self.mcp is None`), `config.py:154` / `defaults.yaml:268` (`models.triage` default sonnet).
