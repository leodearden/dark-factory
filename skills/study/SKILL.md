---
name: study
description: >-
  Load a deep, discussion-ready understanding of a specific piece of code into context BEFORE
  reasoning about it. Use whenever the goal is comprehension and readiness rather than an immediate
  edit — triggers like "read the code for X in detail and get ready to discuss it", "study how X
  works", "get a deep understanding of X before we talk", or any time the user is about to explore a
  subtle aspect, do an authoritative analysis of some behaviour/logic, or design a tricky
  feature/change/refactor. Reach for this even when the user doesn't say "study": any request to get
  primed on a specific function, class, module, subsystem, or feature ahead of a hard conversation
  should pull this in.
---

# Study

Build a faithful, discussion-ready mental model of a specific piece of code and load it into context,
so the conversation that follows — exploring a subtlety, an authoritative behavioural analysis, or
designing a tricky change — starts from real understanding instead of pattern-matching on names.

The deliverable is **readiness**, demonstrated by a synthesis. By the end you should be able to
answer hard questions about this code, explain *why* it's shaped the way it is, predict its behaviour
on edge cases, and point at the parts that are subtle or load-bearing. Coverage for its own sake
isn't the goal — a faithful model is.

## How to study

The steps below are a default order, not a checklist to grind through. Match effort to the target:
a gnarly 40-line function deserves line-by-line tracing; a subsystem wants architecture-first, then
drilling into the hot spots. Stop reading outward when more reading stops changing your model.

1. **Pin down the target.** Work out exactly what "X" refers to — a function, class, module,
   subsystem, or feature — and find its real entry point(s). If it's ambiguous, locate the candidates
   and confirm with the user rather than studying the wrong thing.

2. **Read the actual code, fully.** Open the source and read it top to bottom. Don't infer behaviour
   from names or guess at bodies you haven't opened — that's how confident-but-wrong models get
   built. Note the structure as you go.

3. **Follow the references that matter.** Pull in what the code can't be understood without:
   - **Callers** — who invokes it and with what assumptions; this reveals the real contract.
   - **Callees & dependencies** — what it leans on, and the data structures / config / constants it
     touches.
   - **Tests** — often the clearest executable spec of intended behaviour and the edge cases the
     authors cared about. Read them as documentation.
   Expand outward until the model is solid; don't chase references that won't change your conclusions.

4. **Trace the important paths.** Walk the main control and data flow concretely, and push a real
   input through the tricky parts. Track state, side effects, ordering, concurrency, and how errors
   and edge cases are handled. The behaviour you can trace, you understand; the behaviour you assume,
   you don't.

5. **Recover intent.** Read comments, docstrings, and any design docs; use `git log` / `git blame` on
   the load-bearing lines when *why it's like this* is unclear. Keep "what the code does" separate
   from "what it's meant to do" — the gap between them is often the whole point of the study.

6. **Verify empirically when it's cheaper than arguing.** When a behaviour is subtle enough that
   reading leaves real doubt, confirm it instead of speculating: run the existing tests, or write a
   *throwaway* repro / experiment. **Never modify tracked source** to do this — scratch files and
   temporary scripts only, cleaned up after. An hour of debate about what a line does is usually
   beaten by ten seconds of running it.

7. **Interrogate it.** Actively hunt for the things worth discussing: invariants and assumptions,
   edge cases, failure modes, surprising or non-obvious behaviour, tight coupling, and anything that
   contradicts the apparent intent. These are what make the upcoming conversation hard, so surface
   them deliberately rather than waiting to trip over them.

## Synthesis

Close with concise study notes — enough to prove the model is real and to anchor the discussion, not
a transcript of everything read. Adapt the shape to the target; a good default:

- **What it is** — one paragraph: purpose and responsibility, in plain terms.
- **Shape** — the key components and how they fit (entry points, main pieces, core data structures).
- **How it works** — the important flow(s), at the right altitude for what comes next.
- **Subtle / load-bearing** — the non-obvious bits: invariants, assumptions, edge cases, the lines
  you'd break something by "cleaning up".
- **Watch-outs / open questions** — risks, smells, ambiguities, and anything you'd want to confirm
  before changing it.
- **Reading map** — the handful of `file:line` anchors worth jumping to (they're clickable here).

Then state plainly that you're ready to go deep, and offer the two or three threads you found most
interesting or most likely to matter — that gives the user a fast way to steer.

## Principles

- **Read, don't guess.** Cite `file:line` for claims. Where you're inferring rather than quoting,
  say so; where you're genuinely unsure, say *that* instead of confabulating a clean answer. A
  flagged unknown is more useful than a confident wrong one.
- **Fact vs. inference.** Distinguish "the code does X" (you read it) from "I think this implies Y"
  (you reasoned it). The user is about to make decisions on your model; the seams need to show.
- **Depth tracks the target.** Don't flatten everything to the same resolution — spend the detail
  where the difficulty actually is.
- **Stay read-only on tracked code.** This skill is for understanding, not changing. Experiments and
  notes live in throwaway files; the repo ends as it started.
