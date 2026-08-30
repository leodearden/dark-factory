# Deriving the Landed Sha — Canonical Reference

This document is the single source of truth for **how an agent derives the commit sha under
which a task's work actually landed on `main`** — and for deciding whether it landed at all.
Four skills reference it:

- [`skills/unblock-low-risk/SKILL.md`](../unblock-low-risk/SKILL.md) — autonomous low-risk merges; stamps `done_provenance`, and aborts on a genuine not-landed verdict
- [`skills/unblock/SKILL.md`](../unblock/SKILL.md) — the `branch_on_main()` confirmation its `merge_status` poll loop accepts terminal states against
- [`skills/merge-queue/SKILL.md`](../merge-queue/SKILL.md) — the canonical ancestry check, and the coalesce-train "Follow the superseded successor" rules built on it
- [`skills/orchestrate/SKILL.md`](../orchestrate/SKILL.md) — the manual-resolution workflow's `set_task_status` step, after a hand-performed merge

Each of those call sites keeps its own **dispositions** inline in its own file — what it *does*
with the verdict this ladder returns (abort, keep polling, resubmit, follow a train, clean up).
This document owns only the derivation and the verdicts themselves.

---

<a id="entry-points"></a>
## Two entry points, one ladder

Callers arrive holding one of two things, and the ladder below serves both.

**Marker-first** callers arrive holding a merge-marker search result, and establish branch-ref
existence explicitly:

```bash
git rev-parse --verify --quiet task/<TASK_ID>; echo "ref rc=$?"
```

**Ancestry-first** callers arrive holding an ancestry rc:

```bash
git merge-base --is-ancestor task/<TASK_ID> main; rc=$?; echo "ancestry rc=$rc"
```

**These are the same determination.** `git merge-base --is-ancestor task/<TASK_ID> main` exits
**128 exactly when the branch ref fails to resolve**, so:

| ancestry rc | branch ref | equivalent to |
|---|---|---|
| `128` | does **not** resolve | `ref rc≠0` — [step 2](#step-2)'s *branch GONE* arm |
| `0` or `1` | resolves | `ref rc=0` — [step 2](#step-2)'s *branch still EXISTS* arm |

An ancestry-first caller therefore never needs the `rev-parse` probe: it already holds that
answer. A marker-first caller runs the probe because it arrives holding a *marker* rather than
an rc. Establish ref existence **once**, by whichever route your call site gives you; the two
are not separate gates.

Map your entry point onto the ladder:

- **`ref rc≠0` / `ancestry rc=128`** → [step 2](#step-2)'s *branch GONE* arm decides on the marker alone. An empty marker there is [step 4](#step-4)'s genuine not-landed outcome.
- **`ref rc=0` / `ancestry rc=0`** → [step 2](#step-2)'s *branch EXISTS* arm; containment is required before stamping. A marker that is empty or disqualified falls through to [step 4](#step-4)'s **rc=0** sub-ladder — and since you already hold rc=0, do not re-run the ancestry check to get there.
- **`ancestry rc=1`** → [step 4](#step-4)'s **rc=1** arm directly.

---

<a id="the-ladder"></a>
## The ladder

Every sha recorded in `done_provenance` MUST come from a **task-scoped** search — never from
main's current HEAD, and never from an eyeballed listing. See [Never derive the sha from main's
HEAD](#never-from-head) at the end for why, and the [`DoneProvenance`
contract](#doneprovenance-contract) for what a stamp must carry.

<a id="step-1"></a>
### Step 1 — exact-subject merge-marker search

```bash
git log main --fixed-strings --grep="Merge task/<TASK_ID> into main" --max-count=1 --format=%H
```

This mirrors the in-repo authority, `orchestrator/src/orchestrator/git_ops.py::GitOps.find_merge_marker`
— the same function `merge_status`'s git-authority tier calls on the deleted-branch path.
`--fixed-strings` against the exact subject from
`orchestrator/src/orchestrator/git_ops.py::_merge_subject` (canonical form
`Merge <full-branch> into <main-branch>`) is what makes it **substring-safe**: `Merge task/1
into main` cannot match inside `Merge task/10 into main`, because the `0` falls where the
pattern has a space. Do **not** substitute a bare `--grep="task/<TASK_ID>"` — that is BRE, not
restricted to merge commits, matches any commit merely *mentioning* the task, and re-opens the
`task/1`/`task/10` collision. If a project overrides `git.branch_prefix` (default `task/`) or
`git.main_branch`, build the subject from `_merge_subject` rather than hardcoding it.

- **Returned a sha** → go to [step 2](#step-2); whether it is authoritative depends on the branch ref.
- **Returned nothing** → go to [step 3](#step-3). An empty search is **not** a not-landed verdict.

<a id="step-2"></a>
### Step 2 — is the marker authoritative? Ref existence, then containment

Establish ref existence per [Two entry points](#entry-points) — the `rev-parse --verify
--quiet` probe, or the ancestry rc you already hold.

- **ref rc≠0 / ancestry rc=128 (branch GONE)** → the marker **is** authoritative on its own.
  This is the ordinary post-merge state, not an anomaly:
  `orchestrator/src/orchestrator/git_ops.py::GitOps._delete_branch_if_on_main` deletes any
  branch carrying no commits beyond main, which is exactly what a successful merge leaves
  behind — it is "the single most common post-merge state". A deleted ref is also precisely the
  regime `GitOps.find_merge_marker` itself searches in, so the stale-marker concern below cannot
  apply (no ref exists to have been recreated). Stamp
  `{"kind": "found_on_main", "commit": "<marker sha>", "note": "merge commit located by
  exact-subject marker search"}` and stop here. The `note` is **mandatory**, not decoration —
  see the [`DoneProvenance` contract](#doneprovenance-contract).
- **ref rc=0 / ancestry rc=0 or rc=1 (branch still EXISTS)** → the marker is **NOT authoritative
  on its own**, and `GitOps.find_merge_marker`'s own **branch-existence gate** returns None in
  exactly that situation — it "prevents finding a stale merge marker from a *previous* run of a
  re-opened task that shared the same branch name". Running the search anyway (as we do, because
  the marker is still the best first candidate) means re-supplying that guard ourselves. Require
  containment before stamping:

```bash
git merge-base --is-ancestor task/<TASK_ID> "<marker sha>"; echo "containment rc=$?"
```

The merge that actually brought this branch in must contain the branch's current tip. A stale
marker from a previous incarnation does not: it predates the recreated ref, so the tip is a
*descendant* of it, not an ancestor.

- **containment rc=0** → the marker is this incarnation's true merge commit. Stamp
  `{"kind": "found_on_main", "commit": "<that sha>", "note": "merge commit located by
  exact-subject marker search; containment-verified against branch tip"}` and stop here. The
  `note` is **mandatory** — see the [`DoneProvenance` contract](#doneprovenance-contract).
- **containment rc=1** → a stale marker from a previous incarnation of a re-opened task. Do
  **not** stamp it. Fall through to [step 3](#step-3)/[step 4](#step-4) and let them decide.
- **containment rc=128** → the **marker sha** would not resolve — the branch ref already
  resolved, or you would not be on this arm. The check never rendered a verdict: do not stamp,
  and do not read it as either outcome — re-derive.

(The escalation server layers a second guard on the same risk — the marker must not predate the
recorded `branch_base_sha`; see `escalation/server.py::_found_on_main_response` and the
`merge_status` Tier-3.5 docstring. The containment check above is the shell-side equivalent
available to an agent.)

<a id="step-3"></a>
### Step 3 — an empty marker search is NOT a not-landed verdict

Two kinds of genuinely-landed work carry no `Merge task/<TASK_ID> into main` marker at all: a
**fast-forward / already-contained** landing (no merge commit is ever created), and a
**coalesce-absorbed non-tip train member**, which is merged under the *tip* branch's subject
(see [`merge-queue/SKILL.md`](../merge-queue/SKILL.md)'s "Follow the superseded successor",
rule 3). A hand-performed `git merge --no-ff` is a third: it writes the subject `Merge branch
'task/<TASK_ID>'`, which the orchestrator-shaped marker deliberately does not match (see
[`orchestrate/SKILL.md`](../orchestrate/SKILL.md)'s hand-merge carve-out).

Aborting on an empty search alone would abandon merged work un-stamped — and at several call
sites you are already holding a server-issued `done`/`already_merged` verdict, which you must
not override. So resolve it with the ancestry check before concluding anything.

<a id="step-4"></a>
### Step 4 — the ancestry check: three outcomes, not two

**Never use the two-way idiom `git merge-base --is-ancestor ... && echo "on main" || echo "not
on main"`**: a deleted branch ref exits **128**, which that idiom silently reports as "not on
main" — inverting the truth for the most common post-merge state, since the merge lane deletes
task branches on cleanup (`GitOps._delete_branch_if_on_main`), and on a branch-keyed poll a
*foreign* merger's cleanup can delete it out from under you.

```bash
git merge-base --is-ancestor task/<TASK_ID> main; rc=$?; echo "ancestry rc=$rc"
```

The trailing `echo` is **REQUIRED**, not decoration. `--is-ancestor` prints nothing on rc=0 *or*
rc=1, and the `rc=$?` assignment itself exits 0, so without it the tool reports exit 0 and
identical empty output for "on main" and "NOT on main" — silence you would have to guess at.
Echoing the numeric rc is **not** the two-outcome `&&` idiom banned above: it prints on every
path and keeps all three outcomes distinguishable. Do not "tidy" it away.

#### rc=0 — landed; ancestry has proved it, so not-landed is ruled out

Look for a group/train merge, and **verify it before stamping** — a non-empty result is not
authoritative on its own:

```bash
c=$(git rev-list --ancestry-path --merges task/<TASK_ID>..main | tail -1)
if [ -n "$c" ]; then
    git merge-base --is-ancestor task/<TASK_ID> "$c^1"
    echo "contained-before rc=$?"
fi
```

`--ancestry-path task/<TASK_ID>..main` lists every merge that *descends from* this branch, so
once the branch is on main it also lists every unrelated merge landed afterwards, and `tail -1`
returns the **oldest** of those — the first unrelated task's merge. The containment check on
`$c`'s first parent (main just before that merge) decides:

- **contained-before rc=1** → the branch was not in main before `$c`, so `$c` **is** the merge
  that brought it in. Stamp `{"kind": "found_on_main", "commit": "$c", "note": "absorbed into
  group merge; sha verified by ancestry containment"}`. The note is mandatory here too.
- **contained-before rc=0** (or `$c` empty) → the branch was already in main before `$c`, so
  `$c` is an unrelated later merge. No merge commit exists for this branch. **Do not stamp the
  tip yet** — continue to the citation gate below.

**The citation gate (phantom-branch check).** rc=0 does not by itself prove this branch carries
any work: a branch that never advanced past its creation point has main's own old base commit as
its tip, so it passes ancestry trivially, searches marker-empty, and yields no rev-list
candidate — exactly this arm — while carrying none of the task's work. Stamping it would
fabricate landing evidence for a phantom branch, and the server's only backstop
(`git merge-base --is-ancestor <sha> main`) passes for it. Require a **positive task citation on
main** first:

```bash
git log main --extended-regexp --format='%H %s' \
    --grep='^(merge|impl|amend|fix|test|feat|chore|docs|refactor|style|build)(\(\b<TASK_ID>\b[):]|.*\btask/<TASK_ID>\b)|^Merge task/<TASK_ID> into |\(#?<TASK_ID>\)|\(task <TASK_ID>\)'
```

This is the shell form of
`orchestrator/src/orchestrator/git_ops.py::GitOps.find_task_citation_commit`, which exists for
**exactly this degenerate case** — its docstring: `is_ancestor` "returns True trivially for
zero-commit branches whose tip equals the main HEAD at branch-create time... Requiring a
positive citation on main rejects that degenerate case." The pattern is
`orchestrator/src/orchestrator/git_ops.py::DEFAULT_COMMIT_CITATION_PATTERN`.

**Read the `%s` subject, not just the count.** `--grep` matches the *whole message* and git
applies `^`/`$` per line, so a body line can match spuriously; the function uses `--grep` only
as a coarse pre-filter and re-tests each candidate's **subject** alone. Do the same: walk the
output most-recent-first and take the first row whose **subject** cites this task.

- **A subject-matching row exists** → a genuine fast-forward / already-contained landing: real
  work citing this task is on main. Stamp the branch tip per
  `orchestrator/src/orchestrator/agents/briefing.py`'s fast-forward rule,
  `{"kind": "found_on_main", "commit": "<git rev-parse task/<TASK_ID>>", "note": "fast-forward
  merge, no separate merge commit; landing confirmed by task citation <citing sha> on main"}`
  (rc=0 guarantees the ref still resolves). Recording the citing sha in the note is what makes
  the stamp auditable.
- **No subject-matching row** → nothing on main cites this task. This is the **phantom-branch**
  case, NOT a landing. Do **not** stamp. Stop and report it as not-landed/phantom-branch. This
  is the same signal [`unblock/SKILL.md`](../unblock/SKILL.md)'s `already_merged` guidance
  treats as **not done** — the two rules agree, and neither may be overridden by the other.
- **The project sets `git.commit_citation_pattern: ""`** (citation checking opted out) → the
  check returns nothing *by configuration*, so it proves neither landing nor phantom. No content
  proof is available on this arm: do **not** stamp, and report that the gate could not be
  evaluated rather than reading the empty result as either verdict. **Un-evaluable is not a
  not-landed verdict** — call sites that take a not-landed action (abort, `merge_cancel`,
  resubmit) must not take it here.

Do **not** substitute `git cherry main task/<TASK_ID>` on this arm. That test reports only
commits reachable from the branch but **not** from main; on this arm ancestry rc=0 has just
proved every branch commit *is* reachable from main, so it prints nothing for a genuine
fast-forward and a phantom branch alike and cannot separate them. (`git cherry` is correct where
[`merge-queue/SKILL.md`](../merge-queue/SKILL.md) rule 2b uses it — the rc=1 arm, where the
branch is *not* an ancestor — and that usage stands.)

#### rc=1 — branch exists, not an ancestor of main: **not a not-landed verdict on its own**

Carve out the coalesce case before concluding anything. If this task was ever `superseded` by a
`coalesce-*` train (or absorption is otherwise plausible), rc=1 is the **normal and permanent**
post-landing state for a non-tip train member: the tip is rebased onto main before the merge,
rewriting every stacked commit's sha, and this member's own ref is never advanced to them, so it
can never become an ancestor of main (`GitOps._delete_branch_if_on_main` also *retains* the ref,
making rc=128 the *rare* outcome here, not the common one). Reading rc=1 as not-landed there
would abandon landed work un-credited — the very failure this ladder exists to prevent, and
especially damaging at call sites already holding a server-issued `done`/`already_merged`
verdict.

Instead follow [`merge-queue/SKILL.md`](../merge-queue/SKILL.md)'s "Follow the superseded
successor" rules 2–3: check the **TIP's** merge marker and this task's own scheduler status,
honour rule 2b's **veto** (any non-`done` status means never self-stamp), confirm by content
with `git cherry main task/<TASK_ID>`, and take rule 2b's **landed-but-not-credited** exit
rather than reporting not-landed. **rc=1 is NOT not-landed on the `coalesce-*` arm.**

Only outside that arm — no train absorption anywhere in this task's history — is rc=1 a genuine
not-landed outcome. What to do with it there is the call site's own disposition (`unblock` and
`merge-queue` keep polling / resubmit; `unblock-low-risk` aborts).

#### rc=128 with an empty marker search — a genuine not-landed outcome

The branch ref is gone and nothing on main cites the task, so no ref remains that could prove
otherwise. Treat as NOT landed and stop rather than stamping provenance.

The rc=128 marker search must be the **exact-subject** one from [step 1](#step-1) — **never an
unfiltered `git log main --merges | head -5`.** An unfiltered listing takes no task argument, so
on any repo with merge history it always prints something: "a hit" would be unconditionally true
and "no hit" unreachable, and every rc=128 — *including a typo'd branch name, the wrong
worktree, or a branch that was never pushed*, which all exit 128 too — would be recorded as
landed with some unrelated task's merge sha. The server's `done_provenance` backstop is only
`git merge-base --is-ancestor <sha> main`, which any recent merge on main passes, so nothing
downstream would catch it.

---

<a id="doneprovenance-contract"></a>
## The `DoneProvenance` contract

`kind` is **required on every payload.** `fused-memory/src/fused_memory/middleware/task_interceptor.py::_validate_done_provenance`
rejects a kind-less blob with `done_provenance.kind is required`, and
`shared/src/shared/task_metadata.py::DoneProvenance` has no default for it.

`kind='found_on_main'` **requires BOTH a `commit` AND a `note`.** `DoneProvenance` raises
"commit is required when kind='found_on_main'" and "note is required when
kind='found_on_main'" **independently**, so a commit-less blob and a note-less blob are *both*
rejected: since the post-3092 hardening there is no note-only fallback and no commit-only one.
Every stamp in the ladder above must carry both.

`kind='merged'` requires only a `commit` — the note rule is `found_on_main`'s. Use `merged` when
this branch supplied the merge you are recording; use `found_on_main` when the work was already
on main when you found it.

Where the ladder finds no honest commit, the answer is to write **nothing at all** — never to
substitute a convenient sha. **Declining to stamp *is* an available option**, and on the failing
arms it is the required one. "No single commit applies" is never an escape into a note-only
payload; it is an instruction to derive one or to stop.

<a id="never-from-head"></a>
## Never derive the sha from main's HEAD or an eyeballed listing

Do **not** use `git log --format=%H -1 main`, and do **not** eyeball `git log main --oneline
-20`. Neither is scoped to this task. `-1 main` is main's *current HEAD*, which is this task's
merge commit only when this merge happens to be the newest commit on main — on a live merge
queue it usually is not, so you would record an unrelated task's merge as this one's provenance.
An eyeballed listing has the same defect with an extra step. `git merge-base` is wrong for a
third reason: it yields the common ancestor, **not** the merge commit.

The server's only backstop is `git merge-base --is-ancestor <sha> main`, which passes for every
recent commit on main and would not catch any of these.
