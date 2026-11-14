# A7 — where the design discussion actually happened, and what protocol keeps it (2026-10-19)

Three opus/high readers classified every spawned /prd session of the three programs by its human turns (definitions in `A7-BRIEF-COMMON.md`; per-program detail in `A7-reify.md`, `A7-scp.md`, `A7-sc.md`). Counting rule: a session is `design-discussion` when at least one human turn introduced a consideration not among the presented options and the PRD changed as a result; `elaborate` when ≥ 3 turns on one point or a ruling/recommendation was overturned.

## 1. The numbers

| program | spawned author sessions | design-discussion | elaborate | ruling-only | no human turn | where else discussion happened |
|---|---|---|---|---|---|---|
| reify | 12 (13 transcripts; S10 continued) | 4 (S10, S1, S8, S18) | 3 (S6, S12, S16) | 5 | 0 | — |
| solar_challenge_platform | 13 | 0 | 0 | 2 | 11 | the four `/team` ruling sessions (2 elaborate: S15-Q3, 382) |
| solar_challenge | 22 | 1 (S19) | 0 | 10 | 11 | the capacity-at-events `/team` investigation (D41); the lead's deliberation sitting (D3, D12) |
| **all** | **47** | **5** | **3** | **17** | **22** | |

So 8 of 47 spawned sessions (17 %) carried design discussion, 7 of them in reify. In the two solar programs the discussion Leo remembers happened in the lead's batched sitting, in spawned `/team` ruling sessions, or in a `/team` investigation Leo himself launched — all three are the shape of the proposed protocol, not of a spawned /prd terminal.

## 2. How the reify discussions actually ran (the mechanism matters more than the count)

- Leo answered 78 questions across the 12 sessions: 64 by taking the `(Recommended)` option, 3 by taking another listed option, **11 with free text**. Every design discussion started from one of those 11.
- **All 11 free-text answers were to questions that already carried a recommended option.** A "needs the human only when no recommendation converges" filter would have routed every one of them to the lead.
- Every discussion needed **a follow-up round with the author still holding its context**: a probe, a delegated investigation or a re-proposal (1–30 min), then a re-ask. The value was the live second round, not Leo's real-time presence.
- The sessions already batched their questions (3–4 per AskUserQuestion) and Leo answered hours later; S6's questions waited 16.6 h. Spawn-to-pass-1 median was 21.7 h.
- Two points in one session (S6: enum-only `bind`, the CoreXY test) emerged from Leo reading the draft rather than from a listed question. That is 2 emergent points in 47 sessions.
- Lead rulings in agent mode were refuted by critics twice (S4 OQ-1, OQ-3). The lead is not a safe substitute for Leo on design questions.

## 3. The protocol test as proposed

"Spawn only when the design choice is itself the human's (open_choices unresolved, or size L / risk high)": no program doc has an `open_choices` column and no deliberation ruling reserved a fork, so the test reduces to size/risk.
- reify: 11 of 12 would still spawn (15 of 26 streams are size L, so the filter barely filters); **S12 (M/medium, elaborate) would not** — it would have lost the E3 authorisation, Error-on-shadowing and one-home-per-invariant rulings.
- scp: nothing lost (no author session had discussion; the two elaborate ruling sessions came from would-not-spawn streams but reached Leo through the sitting).
- sc: nothing lost (S19 would spawn; CAP came from a sitting answer).

Conclusion: the size/risk test is the wrong lever in both directions — it spawns ruling-only streams and drops S12 — and the "no converging recommendation" test is wrong too. The lever is **who answers the author's open questions**, not **which streams get a terminal**.

## 4. Proposed A7 (revised)

1. **Author in agent mode**, every stream. Nobody attends authoring; the seat returns `open_questions` each with options and a recommendation.
2. **Critic → fold before the sitting** (A3), so critic blockers that need a ruling reach the same sitting as the author's questions.
3. **Every design-level open question goes to Leo**, in one batched **author-pass sitting** per wave, screens of ≤ 4 in the `deliberation.md` shape, recommendation shown. The lead rules only questions the author marks `lead_rulable` (naming, file placement, seam ids), and even those are shown to Leo as one uncontested block for a yes/no. The draft PRD path is on every screen so Leo can read the draft if he wants (the 2-in-47 emergent case).
4. **A free-text answer opens a follow-up round in the sitting**: an investigating seat (sonnet or opus, in the lead's session) probes or re-proposes and the item returns to a later screen. The sitting ends when no item is pending a re-ask. This is what the reify sessions did across hours and what scp's ruling sessions and sc's CAP investigation did; it just happens in one place.
5. **Leo can escalate any item to a spawned `/team` investigation** (the CAP shape) when it deserves an open-ended conversation; that session is self-terminating (A8) and returns its ruling into the sitting.
6. **Fold-verifier → commit → decompose as agent seats**, non-interactive; a decompose-time `needs_human` question (scp's 582-vs-617 came from decompose) batches to the next sitting or a short closing screen.

What this costs and buys, against Leo's priorities:
- **PRDs right**: all 11 reify overrides survive (they were answers to listed questions); S12 is no longer lost; critic blockers reach Leo alongside author questions. Residual: emergent draft-reading points, mitigated by the draft link.
- **Leo's attention**: the same ~80 answers per program, in one or two sittings instead of twelve terminals over three days with 17–27 h idle waits; no `/reflect` typing into 12 terminals (scp: Leo visited all 13 and none produced a PRD comment).
- **CPU**: nothing idle.
- **Tokens**: more (critic per stream, investigating seats); last priority.

Open for Leo: (a) accept this shape; (b) whether the lead may rule anything at all without showing it (I recommend showing even the `lead_rulable` block); (c) whether the sitting's follow-up round should run inline in the lead session (my recommendation) or as a spawned `/team` per item (scp's shape, 0.5M tokens and a terminal each).
