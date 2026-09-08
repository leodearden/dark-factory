# Code quality

This is the single normative definition of code quality for Dark Factory and
for every project the factory operates. Ratified by Leo on 2026-09-03 during
the merge-lane quality discussion. Other places (`CLAUDE.md`, `CONTRIBUTING.md`
§4, `review/briefing.yaml`, the `/prd` overlay, `~/.claude/CLAUDE.md`, the
fused-memory `preferences_and_norms` record) point here and do not restate it —
a restated list went stale once already (task 3802, INV-9 `one-fact-one-home`).

## Definition

**Code quality is the expected cost and risk of the next change.**

In a factory-operated codebase the next change is made by an agent working
from a partial view of the code, reviewed by an agent, and verified by machine.
So quality means two things: how cheaply and safely an agent can make a
correct change, and how likely a wrong change is to be caught before it lands.

Two corollaries:

- The reader has a bounded context window. Everything a file demands of its
  reader — length, nesting, cross-file lookups, prose to wade through — is a
  cost, not a neutral fact.
- The mechanical gates in `CONTRIBUTING.md` §4 (pytest, ruff, pyright) are the
  **floor**. This document is the **bar**. A change can pass every gate and be
  correctly rejected against it.

## The fourteen heuristics

The headline of each is Leo's wording. The reading beneath it is the
interpretation confirmed on 2026-09-03; apply both by name.

1. **Informative names.** A name says what the thing is and does. A name that
   lies — says one thing while the code does another — is the worst class of
   defect here. Review judgement; no useful metric.
2. **Simple control flows.** Few decision points per unit and shallow nesting.
   Length is heuristic 4; this one is about branching and depth.
3. **Carefully factored orthogonal dimensions of variability.** Parnas
   information hiding applied to axes of change: identify the independent ways
   behaviour varies and give each axis one mechanism, so N axes cost N
   decisions rather than 2^N inline flag checks.
4. **Small function scopes.** A function does one thing and fits in view.
5. **Minimum data access scopes and lifetimes.** Declare closest to use, live
   shortest. Instance state only for what must outlive a call; a transient
   value promoted to instance lifetime "for convenience" is a violation.
6. **Well-defined purpose for each entity.** The purpose of a module, class or
   function is statable in one sentence without "and".
7. **Prefer stateless interactions between modules.** Calls carry what they
   need and return results. No reaching into another module's attributes, no
   temporal coupling on a sequence of prior calls. Explicitly owned durable
   state (a store with one owner) is fine; it is the *interaction* that must be
   stateless.
8. **Prefer immutable data.** Frozen values crossing boundaries; transitions
   produce new values rather than mutating shared structures in place.
9. **Deep modules with appropriate nesting and coherent narrow interfaces.**
   Ousterhout's sense: depth is functionality divided by interface size.
   "Nesting" means layering — modules compose in strata, each hiding the one
   below — not a flat bag of peers importing each other.
10. **Clear invariants, informatively, redundantly, uniformly enforced.** Three
    requirements. *Informatively*: a violation names the invariant and the
    offending values (INV-2). *Redundantly*: checked at more than one point —
    at construction and at use — so no single missed check lets a breach
    through. *Uniformly*: one mechanism everywhere, not ad-hoc ifs. The
    invariant's definition has one home and every enforcement point calls it.
    Redundant enforcement is not redundant reconstruction: a census that
    re-derives state from several containers in order to check it is the
    smell, not the cure.
11. **SPOT — single point of truth.** Each fact, definition and policy lives in
    exactly one place; everything else derives from or points at it (INV-5,
    INV-9).
12. **Structured data instead of meaningful strings.** Parse at the boundary
    into typed values. No routing on string prefixes, no status strings that
    are really enums, no ad-hoc parsers of internal values.
13. **Files make internal sense in isolation.** The general behaviour of any
    file can be understood by reading that file alone. Failing this signals
    porous module boundaries: reach-back imports into a parent module,
    function-local imports placed to break cycles, and re-export shims are the
    measurable symptoms.
14. **No file too large.** The composition hierarchy contains and partitions
    complexity, so no file has to be big. Big is context-dependent; in this
    repo treat ~1,500 lines as a soft ceiling and 2,000 (one default `Read`
    call) as an alarm. **No cheating**: a split is legitimate only when every
    resulting file passes heuristic 13. Size is necessary, not sufficient —
    small satellites that are function-bags over a parent's private state
    fail 13 while passing 14.

## Two stances

- **Comments.** Aim for code that is clear with no or low comments. Needing
  abundant and escalating amounts of commenting is a symptom of poor clarity,
  and comments drift. Rationale that must persist belongs in memory or the
  incident record with a pointer from the code, not inline (INV-9). This does
  not license deleting existing rationale during unrelated work — see
  `CONTRIBUTING.md` §2 on tolerated drift.
- **Tests.** Test access to a module's internals is an interface design smell.
  Tests drive public seams and, where the behaviour is git or the filesystem,
  real fixtures. A test that patches a module's private names by dotted path,
  or reads private attributes, is pinning implementation rather than
  behaviour; reworking such seams is in scope for quality work, not a
  distraction from it.

## What to measure, and what not to steer by

Track quality as five separable aspects. Each has an instrument; several also
need review judgement.

| Aspect | What to measure | Instrument |
|---|---|---|
| Comprehensibility | per-function length, nesting depth, cognitive complexity (Sonar threshold 15), prose-to-code ratio, lines a reader must load to change one behaviour | `radon raw`, `complexipy`, AST counts |
| Changeability | fan-in and fan-out, reach-back and deferred imports, re-exports, distinct patch targets tests reach into, share of commits landing in one file | import graph, `grep`, `git log` |
| Verifiability | mutation score on hot functions, tests exercising the real substrate, coverage only when the fixtures are real | `coverage` with real fixtures, mutation testing |
| Correctness in the field | fix and amend rate per file, regression mentions, incident-linked commits | `git log`; lagging, watch but do not steer |
| Operational honesty | `except Exception` sites, fail-soft paths without an emitted event, violation messages that carry values | `grep`, INV-2, INV-11 |

Read complexity as a pair: the **maximum per function** should fall while the
**module total** stays flat or falls. A total that rises during a refactor
means complexity was added, not moved.

Do not steer by any of these:

- **Raw line count.** Prose can be most of a file (the merge lane's main
  module was 55% comments and docstrings when measured).
- **Average complexity.** A file can average B while eight functions score F.
- **Line coverage under autouse stubs.** A suite that stubs the thing under
  test to "passed" reports coverage of paths it cannot fail.
- **Test count or test-to-code ratio.** Tests that pin implementation are a
  liability with a green tick.

Tooling: `radon` (cyclomatic complexity, maintainability index, raw counts)
and `complexipy` (cognitive complexity, with snapshot and per-function ceiling
modes for ratchets). Install into the workspace venv with
`uv pip install radon complexipy` if absent.

## Relationship to the design invariants

`docs/legibility/design-invariants.md` is the list of *checkable* gate
questions that `/prd` G7 and `/review` phase 2 run. Several already encode a
heuristic in checkable form: INV-2 `structured-facts-at-failure` (heuristic
10, *informatively*), INV-5 `no-lockstep-duplication` and INV-9
`one-fact-one-home` (heuristic 11), INV-11 `no-silent-fail-soft`. This
document is the definition those invariants serve; it is not itself a gate.
Promote a heuristic to an invariant only in a checkable form with a fixture,
via the four-site lockstep edit `CONTRIBUTING.md` §6 describes.

## Reach

Interactive sessions get this document through `CLAUDE.md` and
`~/.claude/CLAUDE.md`. `/review`'s integration reviewer gets it through
`review/briefing.yaml`; `/prd` through its project overlay. Orchestrator-
dispatched task and reviewer agents receive only their role system prompt,
which cannot follow a cross-reference, so for them the substance must be
carried inline in the reviewer prompts — that wiring is tracked as a task
rather than assumed.
