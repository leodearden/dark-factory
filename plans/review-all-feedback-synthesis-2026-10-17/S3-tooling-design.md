# S3 — Phase 6 program-state tooling: comparison and design

Seat S3, 2026-10-17. Read-only. Inputs are named in the brief. The run labels follow the brief: **A** is solar_challenge_platform (R2), **B** is solar_challenge (R3), and **C** is reify (R1).

Path abbreviations used below:
- `A/` = `plans/review-all-feedback-solar_challenge_platform-20261013/` in dark-factory.
- `B/` = `plans/review-all-feedback/review-all-solar_challenge-20261013/data/` in dark-factory.
- `B-live/` = the B lead's still-live scratch, `/tmp/claude-1000/-home-leo-src-solar-challenge/a6995651-…/scratchpad/review-all/review-all-solar_challenge-20261013/`. Run B is still in progress, and this copy is newer than the package copy.
- `C/` = `~/.claude/projects/-home-leo-src-reify/review-all-reify-20261010/scratch/`.

Two facts I measured, beyond what the retrospectives state:
- **B's package copy never calls its FILED renderer.** `B/scratch/gen_program.py` (copied 18:59) defines `filed_rows_from_store` and never calls it. `B-live/gen_program.py:223` adds `*filed_rows_from_store(),`, and solar-challenge commit `dfcbde8` ("FILED table now rendered from the task store") lands that change at 19:15 on 2026-10-17.
- **Some program tasks lack `prd_path`.** Among tasks carrying `metadata.x_finding_run`, measured read-only from each `tasks.db`:

  | Run | Tasks with `x_finding_run` | With `prd_path` | Without `prd_path` |
  |---|---|---|---|
  | A | 331 | 233 | 98 |
  | C (reify) | 528 | 488 | 40 |

  The ones without `prd_path` are Phase 5 mechanical filings plus pre-existing tasks the streams adopted or stamped. A renderer keyed on `prd_path` alone misses the adopted ones.

---

## A. Comparison table

| Capability | Run A (scp) | Run B (sc) | Run C (reify) |
|---|---|---|---|
| **Stream table (waves, deps)** | Hard-coded in `A/program-state/gen_program_doc.py`: `ITEM`, `DEPS` (lines 9–13). The wave is *derived* by `wave()` = 1 + max of the upstream waves, which is good. | Hard-coded in `B/scratch/gen_program.py`, with `WAVES` and `DEPS` as two hand-written sources that can disagree. `DEPS` is edited twice: the literal `'S17': ['S1', 'S18']` is later overwritten by `DEPS['S17'] = ['S1', 'S4']`. Extra keys are added in code: `LOOSE = {'S25': [...], ...}`. The script writes `streams.json` as an output, not an input. | Rendered once by `C/gen_program_doc.py`. Every later change was a hand edit or regex edit of the doc. |
| **FILED rows: source** | **From returns.** `gen_program_doc.py` lines 97–103 render `filed/S*.json`, which the lead writes from seat returns. The resumed lead hand-wrote S1, S4, S10 and S13. **Defect:** the seat returned a structured held leaf (`{"label":"ε","task_id":"1205","x_seam_wait":{"stream":"S1","contract":…}}`), but `filed/S10.json` stores a truncated string: `'1205 (epsilon) x_seam_wait S1 BillingConfig.from_governed_finance(store, at, *, base) as-o'`. | **From the store.** `B-live/gen_program.py::filed_rows_from_store` uses a read-only sqlite query: `where json_extract(metadata,'$.x_finding_run')=? and json_extract(metadata,'$.human_decomposed') is not null order by id`. **Defects:** (i) the `human_decomposed` filter drops adopted or stamped pre-existing tasks; 51 of the 188 run tasks lack it, including pending #284, which R3 §3.7 says a stream adopted. (ii) The stream comes from a hand-kept `PRD_STREAM` dict (LEAD-HANDOFF §1: "add new PRD paths to that dict"). An unmapped path renders as the raw path. (iii) The gate is hard-coded as `where task_id=772`. (iv) `order by id` interleaves the streams, and live status (`773 (done)`) is rendered, so every render diffs the doc. | **From returns, by regex.** `C/wave2/apply_filed.py` line 9 matches `rf'^\|\s*{re.escape(S)}\s*\|'`, and wave 1 uses the first line of `filed/S*.md` (`C/finalize_wave1.py` line 10). **Defect:** the row shape drifted across returns: `FILED ROW:` (S2, S6, S25, S26), `FILED row:` (S24), and a bare row with no header (S1, S5, S8, S10, S12, S16, S18). |
| **Rulings (human)** | `rulings.json` entries are `{item:int, stream, ruling, options_considered[], consequence, ruled_by, date}`, rendered at `gen_program_doc.py` lines 61–62. R15 was re-worded in place on 10-16 (PROVENANCE.md), with no supersede record. | `rulings.json` entries are `{id:"D1", item, ruling, options[], consequence, keys[], ruled_by}`. There is no date. Rendered at `gen_program.py` line 172. | `rulings.json` entries are `{item:"S1 …", ruling, options_considered, consequence, keys, ruled_by, date, note?}`. Only the 22 rulings from the deliberation sitting are in the file. Rulings Leo made inside sessions (S6 R1–R8, S24 R1–R5) live only in each PRD. Leo's reversal of S22 R1 lives in `C/lead-followups.md`. |
| **Lead rulings / amendments** | `prd-results/S*.json` holds string lists: `lead_rulings_on_open_questions`, `revisions_to_earlier_rulings`, `lead_actions_now`, `lead_actions_at_decompose`, `cross_reference_lines`. Rendered at `gen_program_doc.py` lines 67–78. Row amendments are prose, e.g. "program row S4: add fk-ce84f840bb9a". | `lead_rulings.json` entries are `{id:"L1", stream, q, ruling}`, with no `ruled_by` and no date (rendered at line 176). Row amendments are code patches: `B-live/gen_program.py:142` has `_s['deletable'] = _s['deletable'].replace("dead output:/period: keys", …)`. | Prose in `C/prd-results/S*.md` ("OPEN QUESTIONS and LEAD RULINGS: Q1 …") plus a hand-edited `## Lead amendments` section. Key reassignment is a to-do line in `lead-followups.md`: "reassign fk-030602c2fd62 from S11 to S10". |
| **Cross-reference ledger** | `cross_reference_lines` strings with no target field. A decompose return's lines even carry the FILED row itself: in `seat-DECOMPOSE-S10-a35f42d8.txt`, `cross_reference_lines[0]` is `"FILED row S10 \| docs/prds/…"`. | A `XREF` list literal in code (`gen_program.py` lines 37–111, plus 8 more lines in B-live). The target is a string prefix convention, e.g. `'S5 (from S7, S4, S17): …'`. | Appended to doc sections by `apply_filed.py` lines 29–32. Lines are found again by `C/wave2/xref_for.py`, which scans a hard-coded `SECTIONS` list (lines 6–8). **Defect (R1 F10):** the first version missed the waves 2–4 section. Also, `apply_filed.py` appends at `doc.rstrip()` (the doc end), wherever the heading actually sits. |
| **Held-leaf ledger** | `filed/S*.json` has `held_leaves[str]` and `release_duties[str]`. Releases are recorded only in the trailer string at line 104: "Held leaves released so far: 1133 -> 1126 …". | `x_seam_wait` is read from the store into each row (`held: x_seam_wait={...}`). The release plan for about 20 held leaves on S29 is only prose in LEAD-HANDOFF.md §4 item 3. There is no owner field. | A prose cell in the FILED row ("8821 ξ (x_seam_wait S4, S1, S10, S25 …; released by the lead)"). Releases are tracked as checkboxes in `lead-followups.md`. In the store, #8741's `x_seam_wait` carries an ad-hoc field `"released":"2026-10-13 by S6 (dep #8830)"`. Contract re-wording (#8741, #8851) is prose only (R1 F8). |
| **Dated action ledger** | A hard-coded prose string in code (`gen_program_doc.py` line 104), appended five times (R2 §4: "a 400-word paragraph"). It sits beside `A/lead-actions/lead-actions-batch-{3..19}.md`. | None. There is a one-line string literal "Coordinator interventions: …". | `C/lead-followups.md`: timestamped checkbox prose. |
| **Clerk action files** | `lead-actions-batch-N.md`: markdown sections A–D with written rules ("get_task first; skip if in-progress/done/cancelled", "Never touch 1065") and a JSON return shape (`{"cancelled":[…],"skipped":[…],…}`). These worked with zero errors (R2 §3). **Defect:** the lead re-typed each seat's prose `lead_actions` into these files. | None. The decompose sessions applied their own actions. | None. The sessions applied them, and leftovers became lead to-dos. |
| **Brief generation / wave-awareness** | `A/program-state/gen_prd_briefs.py` runs once, for all streams. Briefs carry full finding statements, measurements and proposals (lines 30–41) and the filings rows verbatim (lines 42–47). Seam rows come from a regex over the doc (line 12, `'per the stream' in l`). The return shape is prose (lines 49–51). **Not wave-aware:** no upstream filed ids, so decompose prompts were hand-written (`A/briefs/resume-session/brief-*.md`, `filed-ids.md`). Briefs pointed at fleet result files that were later reaped (R2 §6). | The generator is not in the package. The briefs are static and written at program time. `B/spawn-briefs/S4.md` says "upstream deps: S1" with no ids, and truncates each finding statement to about 200 characters plus "…". Decompose prompts were hand-written per LEAD-HANDOFF §2. | The wave-1 `C/gen_prd_briefs.py` was then wrapped by `C/wave2/gen_session_brief.py`. That wrapper is wave-aware: it embeds the `xref_for.py` output, appends `/reflect` (line 28), and saves `args/<S>.json` for `regen.sh`. **Defect:** it countermands the stale brief in prose ("its preamble … does NOT apply to you", line 14) instead of regenerating it. Upstream ids arrive as one free-text positional argument. `SCR` and `DOC` are hard-coded. |
| **Result-block parsing** | Seats returned bare JSON. Some wrapped it in a fence (`seat-DECOMPOSE-S22-a7489257.txt` begins with a ```` ```json ```` fence, so `json.loads` fails). Two shapes appear: author `{stream, prd_path, manifest_path, open_questions, seam_dependencies_on_other_streams, cross_reference_lines, leaves_estimate}` and decompose `{…, filed, held_leaves, gate_dependencies_added, commit_planning_result, lead_actions[str], notes}`. | `result.md` with a "structured block" in free form; the lead read and applied it by hand. | Regex: `apply_filed.py` line 13 is `r'^CROSS-REFERENCE LINES:?(.*?)(?=^\s*(?:UNCERTAIN\|```\|---)\b\|\Z)'`. `C/wave2/flatten_result.py` exists only to fold nested bullets. |
| **Append-only apply** | Full regeneration from the inputs, with no hand edits to the doc, so no partition bug is possible. But the ledger lived in code. | Full regeneration. **But** LEAD-HANDOFF §1 says: "If /tmp was cleared: … you may append to it directly". That fallback breaks single-point-of-truth (SPOT). | `apply_filed.py` refuses a duplicate row (line 12) and asserts growth (line 33, `assert len(doc) > before`). **Defect in `C/finalize_wave1.py`:** line 15 is `head, _, tail = doc.partition('## FILED — program status')` and line 21 is `open(P, 'w').write(head + '\n'.join(table) + '\n')`. `tail` is never written, which dropped three binding sections (R1 F3). |
| **Session watching** | None. The lead died while four returns were in flight, and they were recovered from subagent transcripts (R2 §2.2). | Manual `find ~/.claude/fleet/sessions -maxdepth 2 -name result.md -newermt …` (LEAD-HANDOFF §1). | `C/wave2/watch_sessions.py` keys on title from `watch.txt` and reports a terminal session or a result.md mtime change, deduplicated via `reported.txt`. **Defects:** `LIVE = {'running','idle','awaiting-input',…}` (line 7), so a stalled session never reports (R1 F6: 13 h). It does not archive result.md (reaped in 1–2 days). `except Exception: continue` (lines 13–14) hides unreadable records. It ran under about 100 Monitor re-arms. |
| **Gate integrity check** | A ten-line script inline in the resumed transcript, not in the package (R2 §3: 67 = 67). The gate id 1065 is hard-coded in the prose at line 96. | Renders the gate's deps (`select depends_on from dependencies where task_id=772`) but never compares them with the high leaves. | A sonnet seat did the check (`lead-followups.md` 10-15 21:00: "57 program high non-terminal tasks … after adding #8545 (59 deps)"). |
| **Handoff brief** | None was written. The resume took about an hour of forensics (R2 §2.2). | `B/spawn-briefs/LEAD-HANDOFF.md` (15 KB): where things live, conventions, state, remaining work, hazards, addendum. The state is hand-typed and points at `/tmp` scratch. | A handoff brief plus `lead-followups.md` plus the persistent scratch under `~/.claude/projects/…`. It worked, but the brief asserted sections that had already been dropped (R1 F12). |

---

## B. Which single design to adopt

**The rule: one source of truth per fact, and the doc is only a render.** The three runs split along this line:
- **A and B regenerate the whole doc from inputs.** That is correct: it removes the partition bug class entirely. But each kept part of its state as code: `XREF`, `PRD_STREAM`, `DEPS`, the S5 `replace(...)` patch, and A's trailer string.
- **C edits the doc in place.** That needed regex parsers (heuristic 12) and produced the dropped-tail incident.

So adopt A/B's render-everything model. Move every hand-kept fact into data files, and pick the owner of each fact as follows:

| Fact | Single source | Why |
|---|---|---|
| Which tasks exist for the program, their status, priority, keys, labels and deps | **Task store**: tasks with `metadata.x_finding_run == run_id`, unioned with tasks whose `prd_path` is a program PRD | This is B's design, and R3 §2 found the doc "never disagreed with the store". Returned rows (A, C) are claims about the store. They drifted (A's truncated held leaf) and needed parsing (C). |
| Which stream a task belongs to | Derived. Match the task's `metadata.prd_path` against a stream's `prd_path` (recorded when its author return is applied). Failing that, use the task's `x_finding_key`, because each key belongs to exactly one stream (§Artefact 3 invariant). | This removes B's hand-kept `PRD_STREAM`. It also picks up adopted tasks, which have keys but no `prd_path`, and which B's `human_decomposed` filter dropped. |
| Held state and contract | **Task store**. `x_seam_wait` gains `release_owner`. A release stamps `x_seam_released`. | The store already carries `x_seam_wait`; C's ad-hoc `released` field shows the need. |
| The gate and its deps | **Task store**: the task carrying `metadata.trigger_chain.run_id == run_id`, plus its dependencies | No hard-coded 1065 or 772. |
| Rulings (human and lead), stream amendments, cross-references, dated lead actions | **Program state**: append-only files written by the lead's scripts | The store cannot hold these. Today they live in code, prose or doc sections. |
| What a seat claimed | **The return file, kept immutable** | Used as audit and as the input to apply-time checks against the store, never as a render source. |

**The trade-off accepted.** A render from the store shows only what the store says.
- **Costs:** the lead's script needs read access to the target `tasks.db`. B's read-only sqlite URI is endorsed for reads (`scripts/tasks_db_schema.py` module docstring). The MCP lookup `find_tasks_by_metadata` (landed 2026-10-17, `237c38194f`) returns only `task_id`, `title`, `status`, `priority` and `updated_at`, which is not enough for rows.
- **Ordering:** a stream's FILED row appears only once its tasks exist.
- **Gain:** transcription errors disappear as a class. Every apply step can check a seat's claim against the store. This replaces the "lead verifies each filed batch with `get_task`" step in deliberation.md §Launching with a script.

**To stop the doc churning,** the FILED render carries no live task status. It shows only program-level states: filed, held, released, or cancelled-as-accepted. Re-rendering after the orchestrator moves tasks then produces no diff (R1 F13).

**The two existing rules survive.**
- *One editor per document:* only the lead runs the scripts that write the program state and render the doc. Seats and spawned sessions write only their own return file (and their own PRD).
- *Seats never commit:* nothing in the tooling commits. It prints the `git commit --only` path list for the lead.

---

## C. Proposed program-state schema

**Location: split by mutability.**
- **Working home:** `~/.claude/review-all/<run_id>/`. It holds the mutable, high-frequency files: returns as received, archived hand-backs, briefs, clerk batches, the session watch list, and the cache.
  - *Not `/tmp`:* A's lead died with all its state in `/tmp`; B's handoff had to say "if /tmp was cleared".
  - *Not the target's main checkout:* it is machine-operated, and a dirty file there can stall the merge queue. B left PRDs uncommitted for 15 h during a drain (R2 §3).
  - *Not C's `~/.claude/projects/<encoded-cwd>/…`:* the encoding rule then has to be re-implemented. `run_id` already names the project, so a flat path needs no encoding.
- **Published snapshot:** `<plans_dir>/review-all-program-<project_id>-<date>.state/`, holding `program.json`, `rulings.jsonl` and `ledger.jsonl`.
  - `program_doc.py` writes it beside the rendered doc, and the lead commits both in the same `git commit --only`, so it adds no commits.
  - It is the durable copy a successor on another host, a gap-fill instrument, or the next `/review-all` can read. The next run needs the rulings to detect contradictions (R1 F1 found 8).
  - The home is authoritative. The snapshot is its last publication. Resume prefers the home and falls back to the snapshot plus the store.

```
<plans_dir>/review-all-program-<project_id>-<date>.md            rendered; never hand-edited
<plans_dir>/review-all-program-<project_id>-<date>.state/
    program.json        written once at the end of deliberation; never edited after
    rulings.jsonl       append-only
    ledger.jsonl        append-only
~/.claude/review-all/<run_id>/
    inbox/<S>.<pass>.json      a seat or session writes its return here (its own file)
    returns/<S>.<pass>[.<n>].json   apply_return.py moves the inbox file here; never overwritten
    handbacks/<S>.<pass>.md    result.md / hand-back prose copied on first sight (survives reaping)
    briefs/<S>.<pass>.md       gen_brief.py output, generated at spawn time
    actions/batch-<n>.json  actions/batch-<n>.result.json
    sessions.json              {title: {stream, pass, record_dir}} written by gen_brief.py --spawn
    HANDOFF.md                 rendered by program_doc.py --handoff
```

**`program.json`.** These are the deliberation outputs. Every field below was hand-coded by at least one run.

```json
{"schema": "review-all-program/1",
 "run_id": "...", "project_id": "...", "project_root": "...", "as_of_sha": "...",
 "record_path": "plans/<run_id>.json", "doc_path": "plans/review-all-program-<pid>-<date>.md",
 "streams": [{"id": "S4", "name": "...", "slug": "...", "remedy": "...", "deletable": "...",
              "keys": ["fk-..."], "mode": "agent|spawn", "size": "S|M|L", "risk": "low|medium|high",
              "deps": ["S1"], "seams": ["..."]}],
 "seams_extra":  [{"seam": "...", "owner": "S9", "consumers": "..."}],
 "invariants":   [{"n": 1, "statement": "...", "owner": "S4", "inv_id": null}],
 "routed":       [{"key": "fk-...", "route": "alerting PRD (tasks 440-491)"}],
 "conventions_extra": ["..."], "triage": ["..."]}
```

- `wave` is not stored. It is derived as 1 + max of the upstream waves (A's `wave()`). This removes B's second source, the hand-kept `WAVES`.
- `prd_path` is not stored here. It enters through the author return (ledger `stream_status` event).
- `routed` replaces A's keyword router: `ROUTE` plus `next(r for k, r in ROUTE if k in text)` is a meaningful-string parser.
- `invariants`, `triage` and `conventions_extra` replace B's `INV_OWNER`, `INV_NOTES` and its triage literals, and A's `OWN`.

**`rulings.jsonl`.** One line per ruling, human or lead.

```json
{"id": "D12", "ruled_by": "human|lead|lead (--yes)", "date": "2026-10-14", "stream": "S4|null",
 "item": "...", "question": "...", "ruling": "...", "options": ["..."], "consequence": "...",
 "keys": ["fk-..."], "source": "deliberation|returns/S4.author.json|session <title>", "supersedes": "D9|null"}
```

- Human (`D`) and lead (`L`) rulings share one shape. That folds together B's two files and A's `prd-results` string lists.
- `options` and `consequence` are required when `ruled_by == human`, per report-format §Artefact 3.
- A re-worded or reversed ruling is a **new line with `supersedes`**. That covers A's R15 and C's S22 R1. The renderer shows the newest ruling and notes what it superseded.
- Rulings made inside a spawned session are appended from that session's return. That fills C's gap for S6 R1–R8 and S24 R1–R5.

**`ledger.jsonl`.** One dated event per line. This is the only place lead actions, amendments and cross-references live.

```json
{"seq": 41, "at": "2026-10-15T14:12Z", "by": "lead|<session title>", "op": "...", "...": "op fields"}
```

| `op` | Fields | Replaces |
|---|---|---|
| `stream_status` | `stream`, `status` (`briefed`, `spawned`, `authored`, `committed`, `decomposed` or `recorded`), `prd_path?`, `return?` | R2 §2.2's per-stream status; B's `PRD_STREAM` |
| `amend_stream` | `stream`, `field` (`keys`, `deps`, `deletable`, `mode` or `remedy`), `change` (`add`, `remove` or `set`), `value`, `ruling` | B's `LOOSE` dict and its `replace()` patch; C's "reassign fk-030602c2fd62"; A's "program row S4: add …" |
| `amend_program` | `section` (`seams_extra`, `invariants`, `routed`, `conventions_extra` or `triage`), `value`, `ruling?` | Late triage lines (B S3 Q6) and conventions edits (A S9 migrations bullet) |
| `xref` | `from`, `to` (a stream id, `"program"`, or a task id), `text`, `return` | B's `XREF` literal; C's xref sections and `xref_for.py`; A's `cross_reference_lines` |
| `action` | `id`, `op` (`cancel`, `edge`, `note`, `stamp`, `repath`, `release` or `amend`), `task`, `target`, `text`, `needs_human`, `return` | Seats' prose `lead_actions`; A's re-typed batch files |
| `action_result` | `id`, `state` (`applied`, `skipped`, `rejected` or `kept-for-human`), `batch`, `detail` | A's clerk JSON returns; C's checkboxes |
| `note` | `text` | A's trailer paragraphs ("Leo 2026-10-15 evening: …"); B's "Coordinator interventions" |
| `commit` | `sha`, `paths` | `commit-msg-*.txt` and the logs in C |

- **Filed per stream: no file.** It is derived from the store at render time, and the immutable return in `returns/` is its audit.
- **Held leaves: no file.** The store's `x_seam_wait` is extended to `{stream, contract, release_owner}`, where `contract` is the contract as specified. On release, the leaf gains `x_seam_released = {by, at, onto: [ids], contract_as_delivered}`.
  - This is R1 F8's `contract_as_specified` / `contract_as_delivered` pair, plus R2's "stamp `x_seam_released_by`".
  - Both keys are in the `x_` namespace.
  - `release_owner` defaults to the waited-on stream. It is the lead when that stream's session is already running at hold time (R1 F8: S8 → #8851).
- **Gate: no file.** It is the store's `trigger_chain` task.

---

## D. The `/prd` seat return block

Agent seats and spawned sessions both write one JSON object, schema `review-all-prd-return/1`, to `~/.claude/review-all/<run_id>/inbox/<S>.<pass>.json`. A spawned session also ends its `result.md` with the same object in a ```` ```json ```` fence, because the registry hand-back is the human-readable channel. An agent seat's final message is one line: `<S> <pass>: <n> filed, <h> held, <q> questions, <a> actions → inbox/<S>.<pass>.json`. This is R2 §2.1's rule that the lead reads digests, never return JSON.

**Fields**, where `*` marks required fields:

- `schema*`, `run_id*`, `stream*`, `pass*` (`author` or `decompose`), `session*` (`{kind: agent|spawn, title}`).
- `prd_path*`, `manifest_path`, `sidecar_path`.
- `commits` (shas). Empty for agent seats; see question G1.
- `open_questions` (`[{id, question, options[], recommendation, needs_human}]`). `needs_human` is true for questions that reverse a ruling, cancel a live branch, or move a board figure (LEAD-HANDOFF §2's list).
- `rulings_in_session` (`[{id, question, ruling, options[], consequence}]`): rulings Leo made inside a spawned session. They are appended to `rulings.jsonl` with `ruled_by: human`.
- `seam_dependencies` (`[{stream, contract}]`).
- `xrefs` (`[{to, text}]`), where `to` is a stream id, `"program"`, or a task id. Never the FILED row; A's S10 put one there.
- `lead_actions` (`[{op, task, target, text, needs_human}]`), where `op` is one of `cancel`, `edge`, `note`, `stamp`, `repath`, `release` or `amend`:

  | `op` | Effect |
  |---|---|
  | `cancel` | `target` is the subsuming leaf id(s). `text` becomes `x_acceptance_reason`. |
  | `edge` | `add_dependency(id=task, depends_on=target)`. `text` is an optional coordination note. |
  | `note` | Appends `text` to the task's details. |
  | `stamp` | `target` holds the keys to merge into `x_finding_key`. |
  | `repath` | `target` is the new path for a `delivered_checks` entry; `text` names the check. |
  | `release` | Releases a held leaf. `target` is the upstream id(s); `text` is `contract_as_delivered`. |
  | `amend` | Rewrites a pre-existing task in place. `text` is the new scope. |

  `amend` is not in R2's list, but A amended 11 tasks in place and C amended #5023 and #4958.

**Decompose pass only:**
- `filed*` (`[{label, task_id, priority, keys[], gate_dep, adopted}]`).
- `held` (`[{label, task_id, waits_on, contract, release_owner}]`).
- `released` (`[{task_id, onto[], contract_as_delivered}]`): other streams' held leaves this session released. C's S6 released #8741 and #8718.
- `commit_planning` (`{committed[], excluded[], stamping_errors[]}`).
- `other_filed` (`[{task_id, why}]`): follow-up defects outside the PRD. C's S6 filed #8879–#8885; A filed 1298 and 1304.
- `notes`: at most 600 characters.

Dependencies are deliberately absent from the return. The store holds them; A's `depends_on` duplicated them.

**Example (decompose):**

```json
{"schema": "review-all-prd-return/1", "run_id": "review-all-solar_challenge_platform-20261013",
 "stream": "S10", "pass": "decompose", "session": {"kind": "agent", "title": "S10-decompose"},
 "prd_path": "docs/prds/review-all-planner-boundary-contracts.md",
 "sidecar_path": "docs/prds/review-all-planner-boundary-contracts.capability-manifest.yaml",
 "commits": [],
 "filed": [{"label": "β", "task_id": 1203, "priority": "high", "keys": ["fk-c0ef97acd037"], "gate_dep": true, "adopted": false},
           {"label": "δ", "task_id": 516, "priority": "high", "keys": ["fk-6de37ba38543"], "gate_dep": true, "adopted": true}],
 "held": [{"label": "ε", "task_id": 1205, "waits_on": "S1",
           "contract": "BillingConfig.from_governed_finance(store, at, *, base) as-of with a logged default (C2)",
           "release_owner": "S1"}],
 "released": [],
 "commit_planning": {"committed": [1202, 1203, 1204, 516, 1206, 1207, 1020, 1208], "excluded": [1205], "stamping_errors": []},
 "open_questions": [],
 "xrefs": [{"to": "S1", "text": "S1's governed-finance builder leaf releases S10 ε #1205 (add_dependency 1205 -> <leaf>, commit_planning 1205)."}],
 "lead_actions": [
   {"op": "edge", "task": 601, "target": [1205], "text": "COORDINATION NOTE (review-all S10): ε delivers 601's per-window finance read.", "needs_human": false},
   {"op": "cancel", "task": 1210, "target": [1208], "text": "accepted into review-all S10 ι (task 1208)", "needs_human": false}],
 "other_filed": [],
 "notes": "§8 step 1 found only 1020 carrying fk-160bca1a028e; 899 at 0.61 is in flight, wired as a dependency of α rather than cancelled."}
```

**Validation** is done by `apply_return.py`. It rejects the return, and writes nothing, on any of the following:
- unknown `schema` or `op`;
- `stream` not in `program.json`;
- a decompose return before the author return was applied;
- a `keys` entry outside the stream's keys after amendments;
- a `held.waits_on` that is not a stream id;
- a filed id absent from the store;
- a store row whose `x_finding_run` differs, or whose `prd_path` differs while `adopted` is false;
- a held id not `deferred`, or lacking `x_seam_wait`;
- a `gate_dep` that is not a dependency of the gate;
- `notes` over the cap.

---

## E. Script set

All scripts go under `skills/review-all/scripts/program/`, use only the Python stdlib, and are run by the lead with `python3`.

| Script | Inputs → outputs | Donor | When |
|---|---|---|---|
| `state.py` (library; no CLI) | It is the single point of truth (SPOT) for: the schema constants, the return validator, the home and snapshot paths, append-only JSONL writes (it refuses to rewrite a line), and the one store reader (read-only sqlite URI, B's pattern). The store reader returns the program's tasks (by `x_finding_run`, unioned with program `prd_path`s), the gate (by `trigger_chain.run_id`) and the gate's deps. The other scripts import it and never query on their own. | B `gen_program.py::filed_rows_from_store`, generalised and with its filter fixed | **Must** |
| `apply_return.py <inbox file \| result.md>` | Extracts the last ```` ```json ```` fence whose object has `schema`, or reads a bare JSON file. Validates it (§D). Moves it to `returns/` and never overwrites. Appends `stream_status`, `xref`, `action`, `amend_stream` (for key reassignments the lead rules) and `rulings_in_session` entries. Prints one digest line plus the open questions that are `needs_human`. | C `apply_filed.py` (duplicate guard, append-only); A `filed/` and `prd-results/` shapes. Makes `flatten_result.py` obsolete. | **Must** |
| `program_doc.py [--roster] [--handoff]` | Renders the doc in report-format §Artefact 3 order from `program.json`, the latest rulings, the ledger and the store. Writes the `.state/` snapshot. Prints the `git commit --only` path list. **It refuses to write when a `## ` heading of the existing doc is missing from the new render** (the R1 F3/F12 self-check), unless `--allow-drop <heading>` is given. `--roster` prints one line per stream: status, filed, held, gate deps, open actions, open questions. `--handoff` writes `HANDOFF.md` with: sources of truth, the roster, unreleased held leaves with their owners, actions not yet applied, and remaining streams by wave. | A `gen_program_doc.py` (section order, amendments rendering, derived waves); B `gen_program.py` (store FILED, `ruled_by`, keys per ruling, no key in two streams: `dup = …` at its tail); B `LEAD-HANDOFF.md` section list | **Must** |
| `gen_brief.py <S> <author\|decompose> [--spawn <title>]` | Writes `briefs/<S>.<pass>.md` **at spawn time**. It contains: the stream row after amendments; the rulings binding the stream (by `stream` or by key); the finding statements and measurements in full; the Phase 5 filings rows verbatim; every `xref` with `to == S`; upstream filed ids from the store; held leaves elsewhere whose `release_owner == S`, as release duties; the gate id; the §D schema and example; and the inbox path. For a spawn it also adds checkpointing result.md after each pass, `HANDOFF.md` on a context or usage limit, and `/reflect` at the end. `--spawn` records the title in `sessions.json` and prints the spawn command with the brief as a prompt *file* (R3 §3.8). Re-running it regenerates. There are no saved args, because the state is the args. | A `gen_prd_briefs.py` (full statements, verbatim filings); C `gen_session_brief.py` (wave-awareness, `/reflect`, regeneration); B `spawn-briefs/S4.md` layout | **Must** |
| `clerk_batch.py write \| ingest <result>` | `write` turns accepted `action` entries that have no result into `actions/batch-<n>.json` plus a clerk prompt carrying A's standing rules: get_task first; skip in-progress, done or cancelled tasks; skip a branch with commits; metadata merge, never append; never touch the gate. A `release` op does all of: `add_dependency`, rewrite the description to `contract_as_delivered`, stamp `x_seam_released`, and run `commit_planning` on that leaf, which re-stamps the sidecar. `ingest` appends `action_result` entries. | A `lead-actions-batch-*.md` rules and return shape | **Must**: it removes the re-typing (R2 §3) |
| `gate_check.py` | Store: the high and critical tasks with `x_finding_run == run` that are not cancelled, minus the gate's deps, gives *missing*. The gate's deps minus program tasks gives *foreign*. It also reports tasks carrying a program `prd_path` without `x_finding_run`, which would escape the completion gate's measure. It exits nonzero on any finding. It runs before the lead asks the human to `commit_planning` the gate. | R2 §3's inline script; C's sonnet coverage seat (found #8545) | **Must** (small) |
| `watch_sessions.py [--stall-min N]` | Reads `sessions.json` and the fleet `record.json` files. On first sight of a result.md it **copies it to `handbacks/` and runs `apply_return.py`** if the result.md carries a fence. It reports terminal sessions. It reports a stall when status is `idle` or `awaiting-input`, `record.question` is null, the transcript tail's last message is from the assistant, and the transcript mtime is older than N minutes. It never swallows an unreadable record; it prints it. | C `watch_sessions.py` (dedup via `reported.txt`) | **Must** for archiving, because reaping destroyed hand-backs in A. Stall detection can wait. |
| `release_held.py` | Lists the store's held leaves, each with its owner, its specified contract, and the owning stream's filed leaves as candidate targets. It emits `release` actions for the lead to accept. | None (A `release_duties` strings, C checkboxes) | **Can wait**: `program_doc.py --roster` and the clerk `release` op cover the next run |
| `key_index.py` | Not built. Contract §8 step 1 is now one `find_tasks_by_metadata(key="x_finding_key", value=<key>)` call (membership match on lists; landed 2026-10-17 `237c38194f`, documented in `57fdc61eb2`). R1 F9's cache answered a problem that tool removes. | — | **Drop**, unless the running fused-memory lacks the tool when the next run starts |

---

## F. Skill-text changes the tooling implies

- **`SKILL.md` §Lead context budget:** extend it to Phase 6. The lead reads `apply_return.py` digests and `program_doc.py --roster`, never return JSON or full hand-backs (R2 §2.1).
- **`SKILL.md` §Phase 6:**
  - The program doc is rendered by `scripts/program/program_doc.py` after every applied step and is never hand-edited.
  - The state lives in `~/.claude/review-all/<run_id>/`, with the snapshot in `<doc>.state/`. Never `/tmp`.
- **`SKILL.md` new §Lead handoff:**
  - Hand off before the limit (R3 §3.6), with `program_doc.py --handoff`.
  - The successor starts from `--roster`, and the heading self-check guards against dropped sections.
- **`SKILL.md` `--resume <run_id>`:** add a Phase 6 meaning. Read the state home, or the snapshot, and the store, and continue from the roster's gaps (R2 §2.2).
- **`references/deliberation.md` §Recording:** each ruling is appended to `rulings.jsonl` with `ruled_by`. A reversal is a new ruling with `supersedes`, never an edit.
- **`references/deliberation.md` §The program doc:**
  - Replace "Written once" with: deliberation writes `program.json` and the doc is rendered from it.
  - Amendments are `amend_stream` and `amend_program` ledger entries.
  - Lead scripts never partition-and-rewrite (R1 F3).
- **`references/deliberation.md` §Launching:**
  - Briefs are generated at spawn time by `gen_brief.py`.
  - Seats and sessions write the §D block to `inbox/`, and `apply_return.py` replaces "applies the returned rows" and the manual `get_task` sweep.
  - `lead_actions` go through `clerk_batch.py`.
  - Held leaves carry `release_owner`; a release stamps `x_seam_released`.
- **`references/deliberation.md` §Closing:** `gate_check.py` must exit 0 before the human is asked to `commit_planning` the gate.
- **`references/report-format.md` §Artefact 3:**
  - Name the `.state/` snapshot and its three files.
  - `## FILED` is rendered from the store, one row per stream, without live status. State this as an explicit delta from hotspot-survey's "one row per task, Severity".
  - Add `## Held leaves`, with columns: leaf, waits on, contract as specified, release owner, released onto, contract as delivered.
  - Add `## Ledger`, which replaces the coordinator-interventions prose.
  - Mark synthesis counts as `measured at <as_of>` (R1 F11).
- **`references/report-format.md` held-leaf bullet:** add `release_owner` to `x_seam_wait`, and `x_seam_released` on release.
- **`skills/spawn/SKILL.md` and `spawn-claude.sh`:** accept a prompt file and append `/reflect` (R3 §3.8). `gen_brief.py --spawn` depends on this.

---

## G. Risks and open questions for Leo

- **G1 — Who commits a spawn stream's PRD?** deliberation.md says seats never commit, and runs A and B had the lead commit. C's spawned sessions committed their own PRD and sidecar (`gen_session_brief.py` step 2), and that kept the doc's one-editor rule intact. The tooling records either (`commits`). Pick one rule.
- **G2 — Commit cadence.** R2 §3 says commit per return, because uncommitted PRDs sat through a drain. R1 F13 says batch per wave, because each docs commit cost a merge re-verify (≥ 5 h on #6504). The snapshot rides in the same commit either way. The skill needs one rule.
- **G3 — Store access is a host-local sqlite read of the target's main checkout.** It is endorsed for reads, but a lead on another host cannot render FILED. The MCP alternative (`get_task` per id) costs the lead's context.
- **G4 — Two new metadata conventions.** `x_seam_wait.release_owner` and `x_seam_released` would be stamped across target projects. They are in the `x_` namespace, so nothing validates them. Should they become blessed keys?
- **G5 — Delivery path.** These are about seven scripts with tests. Should they go through `/prd` and the orchestrator in dark-factory, or be hand-carried before the next `/review-all`? Run B is still mid-Phase-6 on its hand-rolled generator, and migrating it is not proposed.
- **G6 — Return discipline under a usage cap.** A session killed mid-pass (A's five limit-killed seats, C's S26) leaves no inbox file. The design treats a missing or degenerate return (under 100 characters, R2 §2.4) as "relaunch", and nothing is applied partially. Confirm that a partial decompose is never applied from transcripts by hand.
