# A7: reify program `review-all-reify-20261010`, design discussion in the spawned /prd sessions

Sources: the transcripts in `/home/leo/.claude/projects/-home-leo-src-reify/` (top-level `*.jsonl` only); the program doc `/home/leo/src/reify/plans/review-all-program-reify-20261012.md` (§Streams, §Resolved decisions); the session results in `.../review-all-reify-20261010/scratch/filed/<S>-result.md`; the synthesis JSON `/home/leo/src/reify/plans/review-all-reify-20261010.json`, used for one finding's proposal; and the retrospective's `report.md` §F5. All timestamps are UTC from the transcripts. Every script was read-only and run with `python3 -I`; the scripts are in `/tmp/claude-1000/-home-leo-src-dark-factory/313f60b8-8658-4b7a-b926-9c70da52c4ce/scratchpad/a7r/`.

**What counts as a human turn.** A human turn is any one of these:
- an AskUserQuestion (AQ) answer;
- a typed user message, other than the spawn prompt and the `/prd` skill expansion chained onto it in the same millisecond;
- a `queued_command` attachment with `origin.kind=human`. These are messages Leo typed while the agent was busy. They are not `type:user` entries, and the first pass missed one.

Operational lines count as human turns but are tagged `ops` (`/reflect`, "Cap reset…", "now logged in"). Of a session's AQ turns, the ones where every answer is a presented option label are counted as `aq-answer`. AQ turns that carry free text are counted separately.

## A. Discovery

The discovery rule was: the first user message references `spawn-briefs/review-all-reify-20261010-S<n>-session.md`. One more session was added, the S10 continuation, found by grepping `review-all-reify-20261010-S10`. That gives 13 sessions covering 12 streams. Every `-session` brief in the brief dir has a transcript.

| session id | stream | size (bytes) | human turns: total / aq-answer (option-only) / AQ with free text / typed or queued / ops | AQ questions answered | human chars (answers + typed, excl. ops slash) |
|---|---|---|---|---|---|
| 01ac8509-8c43-4430-9662-d937924fca27 | S6 | 6,742,755 | 7 / 3 / 2 / 2 (1 queued, 1 typed incl. `/reflect`) / 0 | 10 (3 free text) | 1,429 |
| f987748b-e2ba-4de2-b200-ba6bb0006d81 | S24 | 5,263,326 | 3 / 2 / 0 / 0 / 1 (`/reflect`) | 5 | 153 |
| 5a2937d3-5044-4036-b7d5-6f253035bfcc | S5 | 5,639,388 | 4 / 3 / 0 / 0 / 1 | 9 | 346 |
| 9bc5a3b6-9452-4513-b20c-993a54fe4629 | S25 | 5,260,275 | 4 / 3 / 0 / 1 (keyboard-trouble note) / 0 | 6 (one asked twice) | 213 |
| d46c13c9-b699-4175-92b4-43d7037c3a43 | S8 | 4,862,865 | 3 / 1 / 1 / 0 / 1 | 4 (1 free text) | 222 |
| 966c75df-306e-4e61-9ae7-ccc608ed53c2 | S10 | 6,658,149 | 6 / 3 / 1 / 0 / 2 (context-limit handoff directives) | 7 (1 free text) | 621 |
| 0a849b08-bd21-430e-a131-04a2022f6f47 | S10 (continuation after context limit) | 1,171,314 | 1 / 0 / 0 / 0 / 1 (`/reflect`) | 0 | 0 |
| 5d2be912-452b-41ae-880e-18e59e941e1d | S12 | 5,477,132 | 4 / 1 / 2 / 0 / 1 | 9 (3 free text) | 992 |
| c2b1f753-d7dd-4a69-92b7-908fc14819f5 | S1 | 5,801,387 | 5 / 1 / 1 / 0 / 3 (login, cap reset, `/reflect`) | 5 (1 free text) | 329 |
| e3295574-2c25-4036-adfd-9d5f16c5d2bb | S26 | 5,032,265 | 4 / 2 / 0 / 0 / 2 (queued "limit reset (again)", `/reflect`) | 7 | 270 |
| 2f934d0e-b936-4080-a5cc-6bcc3e4ad496 | S18 | 5,472,646 | 5 / 2 / 1 / 0 / 2 ("sorry, do continue", `/reflect`) | 6 (1 free text) | 250 |
| 56d13c5c-fac8-4d22-a924-c36889084aaf | S16 | 4,716,345 | 4 / 2 / 1 / 0 / 1 (cap reset) | 4 (1 free text) | 425 |
| a2845907-1208-427c-9f02-29a2b0cb3af8 | S2 | 6,744,035 | 3 / 3 / 0 / 0 / 0 | 6 | 205 |

**Totals.** Across the 12 streams Leo answered 78 AQ questions:
- 64 by picking the "(Recommended)" option;
- 3 by picking a non-recommended presented option (S6 "Enum only", S24 "Hold ξ until they file", S5 "Declare the .ri surface");
- 11 with free text.

Apart from one queued message in S6 (12:41:22), there was no free-form design discussion outside the AQ channel. Every design discussion happened inside AQ "Other" free text, followed by a re-ask.

**Out of scope.** These matched the program name but are not spawned /prd stream sessions:
- `04168ab8-8f3e-433f-bf8c-8c585cf229fe` and `57aa9e1a-e5c7-418a-bc4a-b05a5fdf139b`: the successor lead's `/team --fable` handoff session. These are two files with an identical start timestamp, probably a resume or fork copy.
- `a52a7d3e-a3b7-4d5a-93ba-66d008c72160`: the gap-fill deliberation.
- `b0ff05e0-926c-4d8c-9cfd-a62716462615` and `b9839e5f-e47b-4da8-9305-5730a135cf76`: the gap-fill /hotspot-survey and /review runs.

**Agent-mode streams, not in scope (never spawned).** S4 and S23, plus the 12 wave-1 streams: S3, S7, S9, S11, S13, S14, S15, S19, S20, S21, S22, S27.

## B. Classification (one row per stream; S10 is both of its sessions)

Two inputs apply to every stream row:
- **Program-doc row.** All 12 streams have `mode=spawn`, and the mode was chosen by Leo in the 10-12 deliberation. The doc has no `open_choices` column. Each stream's "Options considered" line under §Resolved decisions was closed by that human ruling, and none of those rulings reserves a design fork for Leo.
- **Would-spawn verdict.** Judged on "size L **or** risk high". Section C also gives the AND reading.

| stream | label | evidence (≤ 250 chars each) | what changed in the PRD | stream row (mode / size / risk / open_choices) | would-spawn | caught-by-sitting / emergent |
|---|---|---|---|---|---|---|
| **S6** | **elaborate-design-discussion** | 01ac8509 @2026-10-13T11:56:50: "Talk about this in more detail. What sort of cycle are we talking about here? […] Can a cycle legitimately come about? What does it mean if it does? How do we handle it at the moment? How might we handle it? discuss some options with their ramifications." · @12:41:22 (queued, quoting the session's own discussion text): "\"...bind then accepts it as well as today's raw shapes...\" Does anything in the work of removing the acceptance of today's raw shapes?" | R1 is a new option A, "normalize loop closures at construction" (tie, brace and ring become one form; a self-loop is rejected). It replaced the recommended "Reject at build time". The session said "the typed record needs no class field" (12:03:59), which drops the program ruling's "typed LoopClosure record **classified** at construction". Leo added a CoreXY test case (12:16:56). R6 makes free_values a per-joint `Map<Joint, JointPosition>` enum with a Snapshot warm start, out of the free_values design discussion Leo asked for. R7 makes bind enum-only, a non-recommended pick that followed his queued question. R8 gives symmetric loop unknowns. The probes added 5 new defects and 7 follow-up defects (#8879–#8885). This is ≥ 3 turns each on loop closure (11:56, 12:16, 14:50) and on free_values/bind (12:16, 12:40, 12:41, 12:54). | spawn / L / medium / none ("Options considered: reify-ir home; reify-stdlib loop_closure_value.rs home; accept as standing", ruled) | would-spawn | **Mixed.** The seeds (cycle class, kinematic.ri) were listed questions, so caught-by-sitting, but they needed 3 to 4 rounds with fresh probes. R7 bind and the CoreXY test were **emergent**, from Leo reading the session's discussion. |
| **S10** | **design-discussion** (2 turns; overturned the session's recommendation, not a program ruling, so not elaborate under a strict reading) | 966c75df @2026-10-13T18:11:49, on "Is the OCCT-free build a supported configuration?" (Rec: "Supported, ungated"): "Why do we want to support this? if there's a good reason it should be tested properly. If there isn't, it should be removed cleanly." · @18:17:48: "Remove, 2 S10 leaves (Recommended)" | The OCCT/gmsh/OpenVDB-free build was **removed**. build.rs now fails closed, stubs.rs (1,013 lines) and the gmsh/openvdb stubs are deleted, and about 1,100 lines of `cfg(has_*)` gates and 652 `*_AVAILABLE` refs are swept, in 2 new leaves. S14 κ #8632 was cancelled, and INV-GEO-4 was corrected in docs/invariants.md. Both arms of the synthesis proposal for fk-107e28de9029 ("add one cheap gate step … or cut the stub to the consumed surface") kept the configuration. | spawn / L / high / none ("choice 1 … choice 2" ruled) | would-spawn | caught-by-sitting (a listed question), then one investigation round (answer at 18:11, follow-up question at 18:14, final answer at 18:17) |
| **S12** | **elaborate-design-discussion** (overturned a program-doc ruling) | 5d2be912 @2026-10-14T10:02:41: "Editing E3 tasks is explicitly authorised (the task would have been edited if it were in the control group, so should also be edited in the combined group)." · @12:18:28: "Why do we have two places that invariants live? Is there a benefit that justifies the duplication? If not, how hard would they be to unify? If so, what is the benefit? Delegate the investigation…" | R1: S12 owns the measurement seam (MeasurementContext, CheckEnginePlan), and E3 tasks #6898, #6904 and #5973 were amended to consume it. The recommendation had been "Split: ε free, ζ after", which avoided touching E3. R6: trait shadowing becomes an **Error** now (π #9030), against the recommended "Module shadows prelude". This came from Leo's shadowing question at 12:18:28 ("…I'm inclined to say reject shadowing under the principle of least surprise. Is there a positive use case…"). R7: each invariant id gets a single home via S24 ξ #8821, with no registry row. That changes the program-doc ruling "promoted … through the four-site lockstep edit" for INV-SF-6, and the result was relayed to S24 and S1. | spawn (named "— spawn" in the Uncontested block) / **M / medium** / none ("All ten approved as specified") | **would-not-spawn** | caught-by-sitting (all three were listed questions), but two needed delegated investigations and a second round (12:18 → 12:48 → 13:09) |
| **S16** | **elaborate-design-discussion** (≥ 3 turns on the dims-overload support policy) | 56d13c5c @2026-10-15T11:08:12, on "Ratify the refinement (Recommended)", where an unsupported support was only a Warning: "Good catch. I'm inclined to say that requesting an unsupported support type should be an error rather than a warning (If the system can't model the supports that have been requested, then it's going to produce a wildly wrong answer). What do you think?" | R3 makes an unrealizable dims support a coded **Error**, `FeaDimsSupportNotRealized` (exit 1), and makes the in-plane tip force re-route to tet. μ #9119 deletes S15 γ #8683's support-drop Warning. The three turns on this point were 10-14 16:16 (R1 capability routing), 10-15 11:08 and 10-15 13:09. | spawn / L / medium / none ("proceed; accept as standing") | would-spawn | caught-by-sitting (a ratify question), then one round; Leo's final answer came about 2 h later |
| **S1** | **design-discussion** | c2b1f753 @2026-10-14T14:16:41, on "Test ctors" (Rec "Rename + gate"): "Which option gives the best test architecture and best (most reliably accurate and easily written & maintained) tests in the long run?" · @14:18:31: "Yes, one ctor + hermetic" | R1: one `Engine::new(EngineInputs)` for front ends **and** tests, with an `EngineInputs::hermetic()` preset and about 1,100 test sites migrated by one scripted leaf; no gated test constructor survives. This is a new design that replaced the recommended "Rename + gate". | spawn / L / high / none (ruled) | would-spawn | caught-by-sitting (a listed question), with a re-ask about 1 min later |
| **S8** | **design-discussion** | d46c13c9 @2026-10-13T18:16:55, on "Inline geo" (Rec "Coded error, bind with let"): "Eventually: Hoist inline geometry args. If it doesn't fit gracefully here then  Coded error, bind with let + a prd milestone for the hoist" | Adds the coded error `E_GeometryCallInValuePosition` **plus** a new `[MILESTONE]` leaf φ #8913, a hoisting PRD decision gate. That leaf is a new element. | spawn / L / medium / none (ruled) | would-spawn | caught-by-sitting (one round) |
| **S18** | **design-discussion** (borderline: the new element is scheduling and policy, not product design) | 2f934d0e @2026-10-15T10:18:27, on "Port window" (Rec "Hold verify-only ones"): "Hold everything behind ζ, pin the chain up to and including ζ" · @13:16:58: "Behaviour changes only (Recommended)" | D13 hold policy: behaviour-changing verify.sh tasks are held behind ζ #9112 (result.md: "28 tasks held behind ζ"), with priority pins on the chain up to ζ (16 pinned tasks, cleared by ω #9117). The pins were Leo's addition. The follow-up excluded the 106 fixture-registration tasks. | spawn / L / high / none (ruled; note: "Leo's framing: … bash brings nothing durable") | would-spawn | caught-by-sitting (a listed question), plus one follow-up round |
| S24 | ruling-only | f987748b @2026-10-13T11:47:02: "PLINECITE, code prose (Recommended)" · @13:48:36: "Hold ξ until they file" (a non-recommended presented option) | R1–R5 as presented | spawn / L / medium / none | would-spawn | n/a |
| S5 | ruling-only | 5a2937d3 @2026-10-13T18:51:12: "Declare the .ri surface" (a non-recommended presented option) · @2026-10-14T09:52:31: "Transitional reify-eval → reify-analysis edge (Recommended)" | as presented | spawn / L / medium / none | would-spawn | n/a |
| S25 | ruling-only | 9bc5a3b6 @2026-10-13T18:27:08: "Unit literal (Recommended)" … · @2026-10-14T09:51:33: "that was keyboard trouble (not an answer). Ask again" | as presented | spawn / L / high / none (MCP clause "STRENGTHENED" by the deliberation) | would-spawn | n/a |
| S26 | ruling-only | e3295574 @2026-10-14T15:23:43: OQ1–OQ4 all "(Recommended)" · @2026-10-15T10:30:01: OQ5–OQ7 all "(Recommended)" | as presented | spawn / L / medium / none | would-spawn | n/a |
| S2 | ruling-only | a2845907 @2026-10-15T10:16:40: "Adopt as first leaf (Recommended)" … · @17:02:31: "Narrow θ, file follow-up (Recommended)" | as presented | spawn / L / high / none | would-spawn | n/a |

## C. Counts

**Counting rules.**
- There are 13 sessions and 12 streams. Labels are per stream, and S10's continuation session folds into S10. On its own that continuation would be `no-human` apart from one `/reflect`.
- The `elaborate` label uses the brief's rule strictly: ≥ 3 human turns on one design point, or the outcome overturned the review-all **synthesis**'s recommended option or a **program-doc ruling**. Overturning only the session's own "(Recommended)" option does not count.
- Would-spawn uses "size L **or** risk high", with no open_choices column and no reserved fork anywhere in the doc.

**Sessions and labels.**
- Sessions found: 13, covering streams 12 of 12, with no missing transcripts.
- By label: 3 elaborate (S6, S12, S16); 4 design-discussion (S10, S1, S8, S18); 5 ruling-only (S24, S5, S25, S26, S2); no stream was `no-human`.
- Under a looser reading in which overturning the session's recommendation counts, S10 and S1 also become elaborate.

**Would-spawn verdicts.**
- Under OR: 11 would-spawn and 1 would-not-spawn (S12).
- Under the AND reading (L and high): would-spawn S1, S2, S10, S18, S25; would-not-spawn S5, S6, S8, S12, S16, S24, S26.

**Cross-tab of label × verdict:**

| label | OR: would-spawn | OR: would-not | AND: would-spawn | AND: would-not |
|---|---|---|---|---|
| elaborate | S6, S16 | S12 | — | S6, S12, S16 |
| design-discussion | S10, S1, S8, S18 | — | S10, S1, S18 | S8 |
| ruling-only | S24, S5, S25, S26, S2 | — | S25, S2 | S24, S5, S26 |

**Streams with design discussion or elaborate that would not spawn.**

Under OR, one stream:
- **S12.** Lost: R1 (only Leo could say that editing E3 control-group tasks is authorised; the seat's recommendation avoided E3, so S12 would not have owned the measurement seam); R6 (trait shadowing → Error, against the recommended module-shadows); R7 (single home per invariant id, which changes the program's four-site promotion and binds S24 and S1).

Under AND, four streams:
- **S6.** Normalized loop closures instead of the recommended build-time rejection, the CoreXY test, per-joint free_values, and enum-only bind. These are the largest design deltas of the program.
- **S12.** As above.
- **S16.** An unsupported support becomes an Error, not a silent-ish Warning (a "wildly wrong answer" class).
- **S8.** The hoisting milestone; without it the inline-geometry hoist would have been a dead end with only a coded error.

**Decision-relevant observations.**
1. Every one of the 11 free-text overrides was given against a question that carried a "(Recommended)" option. A sitting filter that sends the human only questions with no converging recommendation would have routed all of them to the lead.
2. Every design discussion needed at least one more round with the live author context: a fresh probe (S6), a delegated investigation (S10, S12), or a re-proposal (S1, S16, S18). The rounds took between 1 and 30 minutes of session time. A one-shot batched sitting catches the seed question but must allow a second round.
3. The spawned sessions already batched their questions, 3 to 4 per AQ call, and Leo answered them asynchronously. For S6 the questions were posted at 10-12 19:22 and answered at 10-13 11:56. The value of spawning was therefore the live follow-up, not real-time attendance.
4. Under OR, the criterion does not discriminate. 15 of the 26 streams are size L, so it would also spawn the agent-mode streams S4, S7, S19 and S23, and miss only S12.
5. The retrospective's §F5 names S1, S6 and S18 as the valuable forks. The transcripts add S12, S16, S10 and S8, and S12 is the one stream the OR protocol would not spawn.

## D. Data-quality notes

- **Missing transcripts.** None. All 12 `-session` briefs have one. The non-session briefs `...-S6.md` and `...-S24.md` (the original lead's briefs) have no separate session. S10's `HANDOFF.md` was reaped (retrospective §F5 log), but the continuation transcript `0a849b08` exists. It contains only the continuation prompt and `/reflect`.
- **Queued messages.** Leo's mid-turn typing appears as `attachment.type=queued_command` with `origin.kind=human`, not as `type:user`. One design-relevant turn (S6 12:41:22) is only visible that way. S26's "limit reset (again)" is the other.
- **Ambiguous `/reflect` turns.** The `/reflect` command turns carry `origin.kind=human`, but nothing shows whether Leo typed them or a wrapper injected them. They are counted as ops human turns and never affect a label.
- **Ambiguous `/prd` turns.** The chained `/prd` turn at spawn+~40 ms has `origin=human`, but it is the spawn prompt's own chaining, so it is excluded.
- **S25 duplicate question.** S25 asked one question twice after a "keyboard trouble" interrupt (09:50:59, then 09:53:12). Both answers were "This session (Recommended)", and the duplicate is counted in Qs answered.
- **S6's overturn of the program ruling is inferred.** The basis is the session's 12:03:59 option text ("The typed record needs no class field") combined with result.md R1. I did not open the PRD to confirm that no class field survives.
- **S10's label is strict.** It rests on reading "synthesis's recommended option" as the review-all synthesis. fk-107e28de9029's proposal offered two arms, and both kept the stub configuration, so the removal exceeds the synthesis rather than overturning a single recommended option.
- **S18's label.** S18 is design-discussion only because of the "pin the chain" addition, a scheduling and policy element. Without it, S18 would be ruling-only with one non-recommended pick.
- **No `open_choices` field.** The program doc has none. Every spawned stream's mode was itself a human deliberation ruling ("Proceed as a spawn stream"). S12 sits in the Uncontested block but is still marked "— spawn", even though it is M/medium.
