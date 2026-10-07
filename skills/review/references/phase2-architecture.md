# Phase 2: Architectural Coherence — Detailed Guide

This phase answers: **is the codebase internally consistent, complete, and cheap to change?**

Per-task verification cannot see what only shows across module boundaries — broken wiring, plausible stubs, type mismatches at integration points, tests that mock so much they test nothing — nor the cost a change imposes on the next agent. That is this phase's job. Seat routing is SKILL.md "Routing: who does what"; everything below runs in the pinned tree.

## The judging lens

Applies to every step that judges code (Steps 2–7) and to carried-finding re-verification.

1. **Take in `$QUALITY_DOC` in full** before judging — `Read docs/code-quality.md`, or the embedded `guidance()` block when the project has no copy (SKILL.md) — and open every judging seat's prompt the same way (contract §10). Cite it; do not paraphrase its readings into prompts or findings.
2. **Tag** every finding with the heuristics it rests on, lens first (`references/run-record.md`), and name the heuristic in the statement the way the doc asks.
3. **Measurements the doc lists under "Do not steer by"** may appear in a statement as context, never as the reason for a finding or an input to its severity (§10).
4. **Severity** is contract §4. **Class** (mechanical or structural) is §6 and is set in Phase 3, but say in the proposal whether it chooses between designs.

### Look-fors by heuristic

Steps 3 and 5 put their own questions to heuristics 1, 3, 7 and 9–13. The rest are judged where this table says, from candidates found with the instruments in the doc's "What to measure" table. The doc's reading of each heuristic is the test; a measurement only finds candidates.

| Lens | Judged in | Candidates from |
|---|---|---|
| `h6` | Steps 4, 5 | modules and classes with fan-in from unrelated callers, or whose `git log` shows unrelated changes landing together |
| `h2` | Step 4 | per-function cognitive complexity (`complexipy`) and nesting depth for files in the Phase 2 set; read as the doc's max/total pair |
| `h4` | Step 4 | per-function length (AST counts or `radon raw`); functions that do more than one thing |
| `h5` | Steps 4, 5 | instance attributes assigned outside construction, transient values promoted to instance or module lifetime, module-level mutable state |
| `h8` | Step 5 | mutable structures crossing a module boundary; shared structures mutated in place after hand-off |
| `comments` | Step 4 | prose-to-code ratio (`radon raw`); comments defending a decision or answering a reviewer; rationale pointing into memory or an escalation rather than a tracked file |
| `tests` | Step 7 | tests patching private names by dotted path, reading private attributes, or needing a reach-back or function-local import to reach their subject |
| `h14` | Step 4 | only through the doc's measurement protocol for heuristic 14, recorded in the statement; never a line count alone. Always `structural` (§6) |

Install `radon` and `complexipy` into the pinned tree's venv if absent (`uv pip install radon complexipy`). A step that cannot run its instrument says so in `evidence`, not silently.

## Step 0: Scope and carry set

### Changed scope (`mode: since`)

A sonnet seat (low effort) computes, in the pinned tree:

1. **Changed files**: `git diff --name-only <since> <as_of_sha> -- <scope paths>`, minus the briefing's `exclude`/`global_exclude` and deleted paths.
2. **Direct importers** of each changed module, one hop. Python in the `<pkg>/src/<pkg>/` layout: derive the dotted name from the path after `src/` (drop `.py` and `__init__`), then `git grep -l -E '^\s*(from|import)\s+<dotted>(\s|\.|$)' <as_of_sha> -- '*.py'`, plus relative imports inside the same package (`from .<leaf> import`, `from . import <leaf>`). Other languages: the same question in their import form (`use crate::<mod>`, `from './<mod>'`).
3. **Phase 2 file set** = changed ∪ importers. Record both counts in `evidence`.

`mode: full` uses every file in scope.

### Carry set re-verification (both modes)

The carry set is every finding in the `since` report with disposition `open`, `filed:<id>` or `accepted:<id>` (SKILL.md "Pin the run"). Split it across sonnet seats (medium effort, ≤10 findings each). Each seat's prompt carries `$QUALITY_DOC`; the seat then, for each finding, checks at the pinned tree whether the anchor exists and whether the statement still holds, and returns one of:

- `present` — with `confirmed` or `weakened` and, if the severity or wording should change, what changed;
- `gone` — the cost the statement describes is no longer in the code;
- `moved` — the symbol was renamed or the file split; give the new anchor.

The coordinator applies them: `present` → copy the finding, append `run_id` to `last_seen`, fill `change_note` if verdict or severity changed; `gone` → disposition `fixed:<as_of_sha>`; `moved` → new key per §2 with the old key appended to `supersedes`, disposition carried. An `accepted:<id>` whose accepting task is now `done`, or whose anchor moved, reopens as `open` (§7).

A step below that mints a key already in the carry set merges into that entry rather than adding a second.

## Step 1: Run the project's `/audit` skill (if available)

Some projects ship an `/audit` slash command — an automated detector suite (reify's `reify-audit` covers P1 producer-orphan, P2 consumer-stub, P5 phantom-done). When present it runs first and takes the mechanical scanning off the later steps.

1. **Detect.** `.claude/skills/audit/SKILL.md` at the project root. Absent → `f_infra.audit_skill_present: false`, skip.
2. **Window.** `mode: since` → `--since` is the committer date of `since` (`git show -s --format=%cI <since>`). `mode: full` → now minus the briefing's `audit.window_days` (default 14).
3. **Invoke.** `Skill(audit, args="--pattern P1,P2,P5 --since <iso>")`. It writes `data/audit-runs/<ts>.json`, escalates high, files medium, logs low, and keeps its own dedupe index.
4. **Fold in.** Keep the raw counts in `f_infra` and convert each audit finding into a `findings` entry tagged `kind:audit` (lens first), disposition `filed:<filed_task_id>` for medium, `open` with the escalation id in the statement for high. Phase 3 never re-files or re-escalates them.

## Step 1.5: Read the other instruments' reports (contract §9)

Inputs prioritise; they are not findings to copy. Record every input in `inputs_consumed`.

1. **Hotspot survey.** The newest `bug-hotspot-survey-*-full-findings.json` in the project's plans directory (dark-factory `plans/`, reify `docs/notes/`); its `method.run_id` is `hotspot-survey-<project_id>-…` and `method.as_of_sha` its tree. A report from before the contract has no `method.run_id` — record it as `legacy:<filename>` and treat its anchors as dated. Use its ranked areas as Step 4 candidates and its churn exonerations (when present) to down-weight healthy churn. Re-verify any anchor before acting on it; check its `as_of_sha` against ours.
2. **`/review-all`.** The newest `review-all-<project_id>-<YYYYMMDD>[-<n>].json` in the plans directory, never a `review-all-program-*` file (`method.run_id`, `method.as_of_sha`). Its open findings in scope go to the top of Step 4's list.
3. **Confusion codebook** (`docs/legibility/confusion-codebook.yaml`). Never Read it raw — it runs to tens of thousands of lines. A sonnet seat runs this digest from the pinned tree and returns its output sorted by sightings:

   ```python
   import re, yaml
   book = yaml.load(open("docs/legibility/confusion-codebook.yaml"), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
   for e in book.get("entries", []):
       if e.get("status") in ("retired", "fixed"):
           continue
       anchors = e.get("anchor") or e.get("fix_where") or []
       anchors = sorted({re.sub(r":[\d,\-]+$", "", a) for a in ([anchors] if isinstance(anchors, str) else anchors)})
       print(len(e.get("sightings") or []), e["id"], e.get("finding_key", "-"), e.get("area", "-"),
             ", ".join(anchors), e.get("title", "")[:80], sep=" | ")
   ```

   Anchors with three or more sightings inside the Phase 2 file set are confusion-dense: add them to Step 4. A /review finding that rests on a codebook entry lists `agent-transcripts` in `evidence_source`. Record the codebook as `codebook@<as_of_sha>`.
4. **Metrics snapshot.** The latest committed `plans/quality-metrics/*.json`, newest by commit (`git log -1 --diff-filter=A --name-only --format= -- plans/quality-metrics/`), rendered with `scripts/quality_metrics_snapshot.py --summary <path>`; its file records and `import_graph` section are read with a small script, never by eye. Record its `run_id` in `inputs_consumed`. It is context for choosing Step 4's modules, never a ranking (contract §10): nothing is ordered by a measure.

A /review finding whose key equals one in a consumed report is the same finding (§9): keep that report's `first_seen`, append this `run_id` to `last_seen`, and say in `change_note` if our evidence changes its verdict or severity.

Never launch `/hotspot-survey` or `/review-all` from here: both are deliberate user decisions. If a mature repo has no hotspot report, suggest one in the summary.

## Step 2: Stub and placeholder audit

### Mechanical scan (sonnet seat, low effort)

Over the Phase 2 file set:

```
TODO, FIXME, HACK, XXX
raise NotImplementedError
pass  (sole function body — not except blocks or abstract methods)
...   (Ellipsis as implementation — not type stubs or overloads)
return None  (in functions annotated with a non-None return type)
return {}  or return []  (hardcoded empty returns in non-trivial functions)
"not implemented", "placeholder", "stub"
```

For each hit: file, enclosing `path::symbol`, five lines of context either side.

### Cross-reference (coordinator)

1. **The task that claimed it.** `git log -S'<symbol>' --format='%h %s' -- <file>` finds the commits that introduced the symbol; read the task id from the commit message or branch, then `get_task`. A `done` task whose brief covered the symbol and a placeholder body is an **unintended stub** (`kind:stub`, high). Never scan the whole task tree with `get_tasks`.
2. **Briefing `known_gaps`.** A matching gap counts only through its `accepted_by` task: record the finding anyway, naming that task in the statement; Phase 3 applies §8 step 1 to the task to set the disposition.
3. **Memory** — `search(query="decision to defer {symbol}", project_id="<project_id>")`.
4. **Code context** — ABC methods, Protocols, type stubs, fixtures: not findings.

A marker with no owning task breaks the briefing's TODO-tracking convention when the project has one (`kind:convention`).

## Step 3: Critical-path tracing (requires briefing)

No briefing: skip. `mode: since`: trace only the paths whose trace touches the Phase 2 file set.

For each key scenario, follow the code end to end. At each step:

1. **Does the function exist** where the path expects it?
2. **Does it call the next step**, or return early, call something else, or branch around it?
3. **Are types compatible at the boundary?** `Optional[X]` where `X` is expected, `dict` where a dataclass is, `str` where an enum is (`h12`).
4. **Are runtime dependencies satisfied?** Conditional imports, config keys read but undefined, undocumented env vars, services assumed running.
5. **Is error handling coherent** along the path, or does each module do its own thing (`h10`)?

Common failures: wiring gaps (A and B built by separate tasks, never connected), mock-masked failures, conditional short-circuits, stale imports that still resolve.

Record the path and step, what the code does against what the scenario needs, the evidence, and a proposal. Anchor at the step's function.

## Step 4: Deep read of high-risk modules

Steps 2–3 find structural gaps. This step finds behavioural bugs and the comprehension cost of the code, by reading it.

### Choose the modules (5–10 per area)

From the Phase 2 file set, in this order:

- `/review-all`'s open findings (Step 1.5);
- hotspot-ranked areas, minus exonerated churn (Step 1.5);
- confusion-dense codebook anchors (Step 1.5);
- modules the metrics snapshot shows in `cycles`, with reach-back or deferred imports, or reached by tests' `private_patch_targets` (Step 1.5);
- server startup, config loading, pipeline stages and orchestration, infrastructure files (Dockerfiles, units, CI), shared utilities;
- the briefing's `stability_concerns`.

### What to look for

Read bodies, not signatures, and think about what runs.

- **Wrong variable passed** — a logical id where a path is needed; both strings, so only reading both sides catches it.
- **Hardcoded assumptions** — a provider client constructed regardless of config; a port or path that only works on one machine.
- **Missing initialization or cleanup** — constructed but never started, opened but never closed.
- **Port/path/ID drift** between code, config, units and docs.
- **The look-fors** for `h2`, `h4`, `h5`, `h6`, `comments` and `h14` from the table above.

### Process

1. List the modules; state why each was chosen.
2. Read each thoroughly.
3. Verify every candidate: trace the call chain, check the config, confirm the mismatch.
4. Record the anchor, the claim with its measured facts, and the proposal.

## Step 5: Cross-module consistency

Over boundaries with at least one side in the Phase 2 file set.

- **API surface**: naming of like operations (`h1`), error types for like failures, return types for like data, parameter conventions.
- **Data across boundaries**: serialisation at each crossing; optional fields one side sets `None` and the other expects absent; enums and constants duplicated rather than imported (`h11`); mutable values handed across and then mutated (`h8`); strings carrying structure (`h12`).
- **Configuration**: keys read but undefined, defaults that disagree with config files, keys nothing reads, ports and paths that disagree across config, code, units and docs.
- **Coupling**: reaching into another module's attributes or relying on call order (`h7`); reach-back imports, function-local imports placed to break cycles, re-export shims (`h13`); flat peer meshes where layering belongs (`h9`); one axis of variation implemented as flag checks scattered across modules (`h3`).

## Step 5.5: Design-invariants audit

1. **Detect** `docs/legibility/design-invariants.md`; absent → skip.
2. **Read** it and audit the Phase 2 file set against each invariant's checkable question. The doc is normative: cite invariant ids, never restate the list.
3. **Record** each violation as a finding whose lens is `inv-<n>` (INV-5 → `inv-5`), followed by the heuristic the invariant encodes when `$QUALITY_DOC` names one.

## Step 6: Dead code and orphans

### Enumerate (sonnet seat, low effort, whole tree in both modes)

- Orphan modules never imported
- Exports in `__init__.py` or `__all__` never imported
- Functions defined but never called (excluding entry points, CLI and MCP handlers, fixtures)
- Config keys nothing reads
- Test files whose subjects no longer exist

### Validate (coordinator)

Framework-called handlers, external entry points, pytest discovery and `importlib` imports are alive. Judge only candidates whose key is not already in the carry set. When unsure, say "possibly dead — verify before removing" in the statement and set `verdict: weakened`.

## Step 7: Test coverage and the Tests stance

Not line coverage: whether the tests that matter would catch breakage. Over the Phase 2 file set and the traced paths.

- Is there an integration test that runs the path with real implementations? What exactly is mocked, and could the mock diverge from the real thing unnoticed?
- Would the test fail if A stopped calling B?
- The `tests` look-fors from the table above: each test reaching a module's internals is an interface-design finding anchored at the **module it reaches into** (the seam is usually the defect), lens `tests`, with the patch targets as evidence.

## Step 8: Write the findings

Every finding from Steps 0–7 goes into the report's one `findings` list with the schema in `references/run-record.md`; mint each key per contract §2 from the plain `area`, the normalised `anchor` and `tags[0]` with the one key implementation — `shared/src/shared/finding_key.py` once it lands, until then `python skills/hotspot-survey/scripts/findings_artefact.py key <area> <anchor> <primary_tag>` from a dark-factory checkout; never a hand-rolled hash. Set `verdict` from your own verification: `confirmed`, `weakened` (true but smaller than first claimed), `refuted` (kept with the refutation), `unverified` only when verification did not run, with the reason in `method.extra`. Fill `method.verification`.

Display:

```markdown
### Phase 2: Architectural Coherence
- Scope: 41 changed + 22 importers (since 3f2a…); 12 carried: 9 present, 2 fixed, 1 moved
- Inputs: hotspot-survey-<project_id>-20260930, review-all-<project_id>-20261003, codebook (3 dense anchors in scope)
- Findings: 2 high · 5 medium · 3 low — h7 ×2, tests ×3, inv-9 ×1, h13 ×1, kind:defect ×1 …
- Audit: 1 escalated, 2 filed by /audit
```
