# Critic, delta pass on the revised `/tmp/team-weights-1112/REPORT.md` (2026-11-12, written about 17:35Z)

Read-only, same rules as the first pass. Report read: the file as written at 17:04:58Z. Code: Dark Factory `main`
at `0f21f5e457`. New scripts and outputs are in `/tmp/team-weights-1112/critic/`: `proxylog.py` to `proxylog4.py`
(the local proxy's log; they print no key, token or address), `prereset.py`, `taskverify.py`, `delta1.py`, each with
its `*_out.txt` where one was kept. The proxy config and account files were read for their non-secret fields only
(strategy, priority, flags); no secret value was printed or copied.

**Verdict: answer-changing items remain. A 3, M 6, m 9.** All three A items sit in text that was new in the rewrite
or arose where two of my own first-pass corrections were combined. Nothing from the first pass was dropped.

---

## 1. First-pass items: how each was folded

| Item | Folded | Note |
|---|---|---|
| A1 laptop unweighted | Correctly | But see DA3: with merge verifies now left out of the project weight, the laptop edit does nothing for the three paused projects |
| A2 trial without baseline; owners 7112 and 8827 | Correctly | See DA3 for the size of the effect the experiment must detect |
| A3 floor per project | Correctly, for merge verifies | The same check was not carried to the runs the revised recommendation does weight: DA2 |
| A4 merge verifies left out | Adopted as the recommendation | Its consequences were not re-derived: DA2, DA3 |
| A5 option 2 trigger | Correctly | Two residues: DM2 (the ruling row over-extends), DM5 (my own "37%" was loose) |
| A6 T3 weight | Correctly | |
| A7 pause question | Correctly | DM3 (a rule switch inside the section 2 note, which came from my proposed footnote) |
| M1, M3 to M7, M9 to M12, M14 to M19 | Correctly | M19: "blocked on file locks" for task 3394 now verified (m1) |
| M2 table note | Folded as I proposed, and my proposal mixed two rules | DM3 |
| M8 projection at 3 to 1 | Correctly as worded | Now describes a design the report no longer recommends: DA3 |
| M13 reversibility | Correctly | The edit is a commit on each project's main branch; not said (m2) |

---

## 2. A findings

### DA1. Section 6, "Always, with a release by calendar", and the recommendation built on it: not sound as worded

- Where: section 6, third bullet under "The open design point"; "Recommendation for discussion. Option 2 with the
  calendar release"; section 7 rulings row 5 ("Option 2 with a calendar uses none"); decision 5 ("All week with a
  release before each reset").
- What is wrong:
  1. **The account about to reset is almost never the one with allowance left.** Over the 19 weekly resets from
     10-25 to 11-12, the resetting account had already been capped for the whole 6 hours before its reset in 16 of
     them. It was open in 6% of those window minutes (7% for a 1-hour window, 5% for 12 hours). The report says the
     same thing elsewhere: "every account reached its weekly limit in every week measured". (Measured with no budget
     in force; under a budget more allowance may survive to the reset. Rule: the allowance seat's cap intervals, all
     ten event databases.)
  2. **The scheduler does not choose the account.** The account gate takes the first open account in list order
     (`shared/src/shared/usage_gate.py::UsageGate.try_lease`). Lifting a project's budget in the window before
     account X's reset releases its routine work onto whichever account is first open, not onto X. Some account was
     open in 65% of the window minutes, and in nearly all of them it was a different, recently reset account (for
     example before max-f's Wednesday 13:00 reset the only open account was max-c, which had reset three hours
     earlier). The resetting account was also the first open one in 3% of window minutes. So the release would
     mostly spend fresh allowance, the opposite of its purpose.
  3. **The windows cover much of the week and include the worst hours.** Seven 6-hour windows are a quarter of the
     week; 12-hour windows nearly half. The window before max-c's Wednesday 10:00 reset is the end of the Monday
     21:00 to Wednesday 10:00 gap, where 37 of the 52 stall hours fell. No account was open at its midpoint in any of
     the three weeks measured.
  4. **"Needs no cap state" holds only for the unsound form.** To do what it says, a release has to know the
     resetting account is still open (the program's own in-memory view, which the gate already keeps) and has to
     send the released work to that account (a change to account selection: soonest reset first among open
     accounts, which changes the list order your 10-06 ruling refers to and is the same family as the "choosing
     accounts evenly" adjunct the report has just set aside).
  5. **The calendar would be a second copy of a fact.** The reset hours are known today only from cap messages each
     program parses when it hits a cap. `allowance.md` (unknown U4) could not tell whether they are fixed in UTC or
     move with daylight saving, and max-b's and max-c's reset dates skip weeks. A calendar in config needs a check
     against the reset times the gate observes, or it drifts silently (design rule INV-9, one fact one home,
     `docs/legibility/design-invariants.md`).
- What a budget does when a project under-spends (asked): with fixed dollar shares and no working release, the
  other projects hold their routine work at their own shares while accounts stay open, and that allowance is lost
  at the reset. This is the present situation (scoping-intake and stream-touch nearly out of work, three projects
  halted). The risk is real; the calendar release does not address it, for reasons 1 and 2.
- In its favour, and not a conflict: a fixed lift does bound the hold (design rule INV-7), and option 2 itself
  conflicts with no ruling I could find. The budget test compares live trailing spend with a fixed share, the same
  shape as the daily cost ceiling; held work lowers spend, so the loop damps, unlike the load-derived count the
  05-28 ruling is about.
- Evidence: `critic/prereset.py` (prereset_out.txt).
- Proposed text for the bullet: "*Always, with a release*: the budget applies all week and is lifted when allowance
  would otherwise be lost. A release by calendar alone does not work: in 16 of the 19 weekly resets measured the
  account about to reset had been capped for at least the 6 hours before it, and the scheduler cannot direct work
  to a particular account, so a lift before a reset would mostly spend other, fresh accounts. A release that works
  needs the program's own view that accounts are open (for example 'at least three accounts open', true 37% of the
  time before the pause; a busy program's view agreed with the fleet's in 81 to 100% of minutes, and reads 'all
  open' for a few minutes after a restart) and, to save allowance at a reset, account selection by soonest reset.
  The first is a second yes/no lever of the allowed shape; the second changes the list order."
- Proposed text for the recommendation: "Option 2, as one deterministic rule with fixed shares and no state shared
  between projects. Start it without a release, either all week or only inside the Monday 21:00 to Wednesday 10:00
  gap (where 72% of the stall hours measured fell), and report allowance unspent at each reset from the first week.
  Every account reached its limit in every week measured, so that loss has not yet been seen; with projects idle it
  will be, and the release to add then is the one based on the program's own view of open accounts, not a calendar."
- Decision 5 then reads: "**An allowance budget per project (option 2): yes, and applied when?** All week, or only
  inside the Monday-to-Wednesday gap. And if under-spending projects leave allowance unused: may a program lift its
  budget when its own view shows several accounts open (a second yes/no lever; it reads no file and remembers
  nothing across a restart)?"

### DA2. Section 5 recommendation: leaving merge verifies alone moves the floor onto task verifies and background runs, where it is worse, and nobody measured it

- Where: section 5 "Recommendation for discussion. Layout A, project factor on task verifies, background runs and
  services"; "removes the merge-verify half of the floor problem"; section 8 first failure mode; decision 2;
  migration step 4. (The phrase is from my first-pass A4; I did not check the other half then.)
- What is wrong: section 4's floor table covers merge verifies only. The runs the recommendation would weight down
  have their own durations and the same per-command limits. Before the pause, at equal weights, test runs by kind
  (time the run held its slot; `verify_admission_plan` events, 10-25 to 11-08 16:00Z):

  | Project | Task verify: median / 90th pct (min), runs | Background run: median / 90th pct, runs | Limit per command | Timed out already |
  |---|---|---|---|---|
  | reify | 35 / 58, 488 | 74 / 120, 37 | 120 | task 1; **background 6 of 37 (16%)** |
  | solar-challenge-platform | 25 / 40, 682 | 30 / 50, 54 | 90 | none |
  | solar-challenge | 17 / 23, 517 | 21 / 30, 187 | 60 | none |
  | dark-factory (for scale) | 3 / 20, 1,783 | 2 / 17, 781 | 120 | task 30 (1.7%), background 6 |

  On the report's own assumption (run time scales inversely with CPU share), a third of the share would put 36%
  (reify), 31% (solar-challenge-platform) and 22% (solar-challenge) of task verifies past the limit, and 76%, 50%
  and 55% of background runs. Reify's background run is at its limit today. (Projection; the same rough comparison
  as section 4.)
- Why it is worse than the merge-verify half: the one measurement on record of starvation killing test workers is
  on task verifies, not merge verifies: "0 crashes in 76 merge-role legs vs 14 in about 138 task-role legs" (task
  5529's description, quoted in `rulings.md` R6). And section 4 already says that in today's flat layout the low
  project's task verifies and background runs "are squeezed far harder than its merge verify".
- Evidence: `critic/taskverify.py` (taskverify_out.txt); limits from each project's loaded config.
- Proposed text, section 5 recommendation, second sentence: "The same floor applies to the runs this does weight.
  Before the pause the 90th-percentile task verify took 58 of reify's 120-minute limit, 40 of
  solar-challenge-platform's 90 and 23 of solar-challenge's 60; a third of the share would put 22 to 36% of them
  past the limit, and reify's background run already times out 16% of the time. (Projection.) So: no more than
  about 2 to 1 on task verifies, reify's background run left at today's weight, and the service weight wherever you
  set it (agents have no time limit to breach)."
- Section 4: add the task-verify and background columns to the floor table, and replace "Either merge verifies are
  left out ... or the ratio is set per project" with "Leaving merge verifies out does not remove the floor; the
  ratio must be set per project and per kind of run."
- Decision 2: append "Either way the task-verify factor is limited to about 2 to 1 by the same arithmetic."

### DA3. Sections 1.4, 5, 8 and decision 3: the projection quoted is for a design the report no longer recommends; the recommended design's gain is about one landing a day at most, none of it on the laptop, and a week-then-week comparison cannot detect it

- Where: section 1 item 4 ("the workstation merge verify at 37 to 40 minutes instead of 44 (a third to two thirds
  of what the pause gave), the others losing 0 to 4 landings a day ... in two places per project"); section 5
  Layout A second bullet and the recommendation; migration step 4; decision 3.
- What is wrong:
  1. **Two rules side by side.** "37 to 40 minutes" is 3 to 1 applied to everything, merge verifies included
     (`pause/projection_out.txt`). The recommendation leaves merge verifies out. On the same model the recommended
     design reaches 8.1 of the 10.8 minutes (other projects' agents 4.8, task verifies 3.0, background runs 0.3):

     | Design (projection, same model and assumptions) | Workstation merge verify, from 43.9 min | Dark Factory landings gained, bound |
     |---|---|---|
     | 3 to 1 on everything (what item 4 quotes) | 36.7 to 40.3 | 0.8 to 1.9 a day |
     | 3 to 1 on task verifies, background runs and services (the recommendation) | 38.5 to 41.2 | 0.6 to 1.3 a day |
     | 2 to 1 on verifies (DA2), 3 to 1 on services | 39.0 to 41.5 | 0.5 to 1.2 a day |

     Landing bound by the pause seat's rule (two-week landing rate 23.1 a day, 41% of landings verified on the
     workstation, each minute saved converted one for one). "The others losing 0 to 4 landings a day" was likewise
     computed for a 3-to-1 share of everything.
  2. **Most of that rests on one term, about which two seats disagree.** 4.8 of the 8.1 minutes is the agents term,
     reached only by the weight on each service. It is the model's one well-determined non-merge term (0.66 ± 0.15
     minutes per agent); the task-verify and background terms are not distinguishable from zero. The mechanisms
     seat's reading of the same lever is that a weight on a service "does almost nothing", because agents are mostly
     not CPU-bound (`mechanisms.md` M2(a)). Only the experiment can settle it, and it should switch the service
     weight separately from the verify weights so the two can be told apart.
  3. **Nothing on the laptop.** Only Dark Factory runs task verifies on the laptop (`verify_task_host_policy` is set
     in its config alone; the other three projects' task verifies carry no runner). The three paused projects send
     the laptop merge verifies only, and those stay at equal weight under the recommendation. So "set in two places
     per project" changes nothing on the laptop for them, and the laptop contention (4.8 minutes per other
     project's merge verify, on the host that carried 54 to 58% of Dark Factory's landings) is untouched by design.
     The bullet justifies the laptop edit with the merge-verify shares (73 to 87%), which no longer applies.
  4. **The comparison as specified cannot detect an effect this size.** In landings: about 1 a day against a
     day-to-day spread of 2.5 to 6.9 (`pause.md` F6). In verify minutes: 2.4 to 5.4 minutes, while the two weeks
     before the pause carry a trend of 0.56 ± 0.13 minutes a day in the same quantity (`pause.md` F4, the "day"
     term), which is 3.9 minutes over a week. A baseline week followed by a weights week confounds the two. The
     weights reload without a restart and apply from the next run, so alternating blocks of a day or two would
     separate them; reify task 8827 asks for at least a week per setting, so the block length is yours.
- Evidence: `pause/projection_out.txt`, `pause/weighttest_out.txt`; grep of `verify_task_` in the four project
  configs and the runner field of `workflow_verify` events since 10-25 (only Dark Factory: 84 laptop, 235 local).
- Proposed text, section 1 item 4: "**Weights on CPU would recover a small part of the verify speed-up at a small
  cost.** Projection for the recommended design (project factor on task verifies, background runs and services;
  merge verifies left alone): Dark Factory's workstation merge verify at about 39 to 41 minutes instead of 44, worth
  about one landing a day at most. Nothing changes on the laptop, where projects collide only through merge
  verifies, and nothing on the allowance. The weights need no new code. Whether they work at all is an open
  experiment that two pending tasks already own, and you answered 'not now' to it on 10-13."
- Proposed text, section 5 Layout A second bullet: replace the "in two places per project" sentence with "They are
  set in the project's own config. The laptop has a separate file per project, but under the recommendation it
  needs no change: the paused projects run only merge verifies there."
- Proposed text, decision 3: "**Go on the weight experiment?** It is what reify task 8827 specifies and what you
  answered 'not now' to on 10-13. The projected prize is small (2 to 5 minutes on the workstation merge verify,
  about one landing a day at most), and the verify time was already drifting by about half a minute a day before
  the pause, so a baseline week followed by a weights week cannot separate the two. Alternate the setting in blocks
  of a day or two instead, and switch the service weight separately from the verify weights."
- Migration step 4: drop "and in its laptop file".

---

## 3. M findings

**DM1. Section 6, "Interactive sessions also draw on the pool": verified, with four corrections.**
- Verified: the unit `cliproxyapi.service` is active (since 11-04 08:32Z) with routing strategy `fill-first` and
  session affinity on. The account files carry priorities A 100, max-c 16, max-g 15, max-f 14, max-e 13, max-b 12,
  max-d 12, max-h 11, which is the order the report gives. The factory's list starts with max-d. Per day the log
  shows A serving alone while it lasts, then the highest-priority account not in cooldown.
- Span: the log runs from 2026-11-01 10:50Z to now, the proxy's whole life (10 MB against a 200 MB cap; nothing
  has been rotated out). "The retained log" is all of it.
- "Mentions" are requests. 99% of the lines naming an account are the proxy's account-selection line, one per
  request, each with its own request id (30,549 selection lines against 30,572 request lines). So the figure can be
  stated as a count: of 29,749 requests served, 10,410 (35%) went to factory accounts; by time spent streaming,
  35.9 of 97.1 hours (37%). The counts move while sessions run: at 17:10Z the mentions were 11,858 for the factory
  accounts and 7,474 for max-c, against the report's 11,321 and 6,937. A count of requests is a fair measure of how
  interactive traffic was shared between accounts. It is not spend.
- **Most of the max-c figure is today.** max-c served 7,116 proxied requests in all, 4,877 of them on 11-12
  (17.1 of its 25.1 streaming hours), concentrated between 11:00Z and 17:00Z. Account A last served at 11-11
  16:31Z. Today's two studies ran through this proxy: this critic session's own address for the Claude service is
  the proxy's. So the 6,937 says little about the week of 11-04.
- **The 11-04 week.** Between max-c's reset on 11-04 10:00Z and the factory's detection of its cap on 11-06 17:30Z
  the proxy sent it 2,046 requests (7.2 streaming hours). For scale, account A's one clean weekly lump (11-09 13:07Z
  to its cap on 11-11 16:31Z) was 11,603 requests and 35.4 streaming hours. If A's weekly limit is the same size as
  a factory account's (not known), the overflow in that window was about a fifth of an account-week, roughly $400
  of a shortfall of about $1,230 (the factory recorded $743 against a usual $1.97k). And max-c capped at the same
  low level the week before ($728, reset 10-28, capped 11-02 22:22Z) with no proxy traffic at all (0 requests; the
  proxy first used max-c on 11-04). So the overflow is at most a partial cause of that week. The proxy also sent
  max-b 1,271 requests inside its 11-06 week ($1,504), which the report does not mention. Four of the six low weeks
  (max-c 10-07 and 10-28, max-b 10-09 and 10-30) had no proxy traffic.
- "No record says how much interactive work took" is too strong. The proxy log gives requests and streaming time
  per account, and interactive session transcripts carry token counts per message (checked on one current
  transcript: 78 of 306 lines carry them). Nobody has joined the two.
- Two facts for decision 12. The proxy's first overflow account is max-c, the account whose Wednesday 10:00 reset
  ends the factory's 37-hour gap. And max-c's current week: 5,080 proxied requests (17.9 streaming hours) since its
  reset on 11-11 10:00Z, about half of A's weekly lump on the same rough scale, so expect it to cap early this week
  and show as another low week in the factory's records.
- Rough size overall (indication only, same caveats): 35.9 streaming hours on factory accounts in 11.3 days is
  about 0.6 of an account-week per week, of the order of a tenth of the seven-account pool.
- Evidence: `critic/proxylog.py`, `proxylog2.py`, `proxylog3.py`, `proxylog4.py`.
- Proposed text for the second and third bullets: "Its log, which covers the proxy's whole life from 11-01, records
  one line per request with the account chosen: 10,410 of 29,749 requests (35%) went to factory accounts, 7,116 of
  them to max-c. It records no tokens or cost. Nearly half of the factory-account requests (4,877) are from today,
  11-12, when account A had run out and this study and this morning's ran through the proxy onto max-c. So the
  allowance left for the factory is not a fixed quantity. As a rough indication only, interactive overflow has
  taken of the order of a tenth of the pool since 11-01. In max-c's low week of 11-04 the proxy sent it 2,046
  requests, on a rough scale a fifth of a weekly lump: a partial cause at most, since max-c capped just as low the
  week before with no overflow. The low weeks before that are unexplained."

**DM2. Section 7 rulings row 5 and decision 5: "Any variant that reads or remembers caps ... need your ruling";
"triggered by capped accounts (which touches your 11-12 ruling ...)".** This over-extends the ruling. Its words are
"cap persistence, a cap authority" and "No process ... reads or writes a cap file". A program consulting its own
in-memory view of the accounts is what every gate does today on every call. What the ruling covers is remembering
caps across a restart or sharing them between programs (my first-pass A5 point 4 said only that). The difference
matters now, because the own-view test is what a working release needs (DA1). Proposed row text: "Option 2 needs
none. A test on the program's own in-memory view reads no file and is outside the ruling's words; it is a second
yes/no lever (row 2). Remembering caps across a restart, sharing them, or option 3's shared view of spend need your
ruling."

**DM3. Section 2 table note and the "Rules" line.** The column is finished tasks; the note's 30, 36 and 44 are
landings. By the column's rule the three figures are 27, 40 and 40. "Agrees with landings to within 2 a day" holds
for the two whole windows (22.0 against 23.7; 39.9 against 40.2) and not for the sub-windows, where they differ by
3.0 to 3.7 (my first-pass M2 said "every window"; that was wrong). The two measures also disagree on whether
anything stepped at the 11-10 deploy: finished tasks 40.0 then 39.8; landings 36.4 then 43.5. There is no double
counting (51 and 71 landing rows are 51 and 71 distinct requests and tasks; `critic/delta1.py`); the sets differ at
the edges and by tasks finished without a merge. Proposed note: "Dark Factory's 'before' is the two-week mean. By
landings (a different count, used in section 3) it was 30 a day in the 72 hours before the pause, 36 during the
pause before the 11-10 deploy and 44 after; by finished tasks 27, 40 and 40. The two counts agree over the whole
windows and differ by up to 4 a day over these short ones, so the size of any step at the deploy is not firm."

**DM4. Section 8, the resume paragraph and migration steps 1 and 2.**
- The pair that resumes together is the light pair. Before the pause the two solar projects together used about
  2.4 of the 11.8 cores the three paused projects took (verifies 1.05, services 1.38); reify used about 9.4
  (`/tmp/team-gates-1112/capacity.md` F9). Reify resumes later, on its own prerequisites. The 11-07 precedent
  quoted (pressure 34 to 61) included reify. Say so: "The two that resume together are the light ones (about 2.4
  cores between them before the pause, against reify's 9.4), with up to 38 agents; reify, the heavy one, resumes
  later and separately."
- "A reduced task limit raised in steps": the task limit takes effect only at an orchestrator restart
  (`OPERATIONS.md`, "Red tier (restart-only)": `max_concurrent_tasks`). Each step is a restart of that project's
  orchestrator. The cheap moment to set limits, step 2 included, is now, while the projects are halted with nothing
  running; an operator halt is re-asserted after a restart by design (halt-retirement plan, class table).
- "One project at a time" needs a mechanism. The resume tasks finish as soon as their script returns, so a
  dependency between them staggers by seconds. A `delayed` milestone on the second task does it with no code: it
  fires a set number of seconds after the task's prerequisites are met and is allowed on this kind of task
  (`docs/task-authoring.md` section 6).
- Proposed step 1: "Before the prerequisites land: set reduced task limits in the two solar projects' configs and
  restart their orchestrators while they are still halted (the limit applies only at a restart; raising it later is
  another restart). Put a delay on the second resume task (a `delayed` milestone) so they do not resume together."

**DM5. Section 6, "*Only when one account is left*: true 37% of the time".** 37% is one account or none left (at
least six capped, which includes the 10.2% with all seven). Exactly one left: 27%. The loose wording was mine.
Proposed: "true 27% of the time (37% counting the time none is left)".

**DM6. Section 6 option list, "*By calendar only* ... Simplest; protects only that gap."** Now that this is one of
the two forms left standing (DA1), give its coverage: 7 of the 10 stalls and 37.0 of their 51.7 hours (72%) fell
inside the Monday 21:00 to Wednesday 10:00 gap; the other three (11-07, 11-08, 11-09; 14.7 hours) ended at the
Saturday, Sunday and Monday resets. (`allowance/caps_out.txt`, stall table.)

---

## 4. m findings

- m1. Task 3394 "blocked on file locks": verified at 16:31Z (64 lock refusals; holders are tasks 5611 on
  `OPERATIONS.md` and 7993 on `config.py`; waiting 11.4 days). It was pinned at 16:30Z today, after the report's
  status read.
- m2. Section 5, "Undoing it is one reload per project": each project's config is tracked in that project's
  repository, so the change and its undo are commits on that project's main branch plus the reload.
- m3. Section 1 item 5 and section 6, "nothing records how much": see DM1.
- m4. Section 4 table, Laptop CPU row, "set in the laptop's own config files": true, and a no-op under the
  recommendation (DA3).
- m5. Section 5, "keeps about three quarters of what weights can reach on the workstation": three quarters of the
  modelled contention (8.1 of 10.8 minutes), of which a weight removes a third to two thirds. Say "of the
  contention a weight could act on".
- m6. Migration step 4, "a weight on its service": for the five projects that run verifies inside their service
  (scoping-intake and stream-touch among them) a service weight also covers their merge verifies. They are not in
  the resumed three, but decision 6 asks whether to bring them in.
- m7. Section 12, critic row, "all folded in": add this pass.
- m8. Statuses re-read 17:18Z, all as the report states: 8526, 8527, 8528 pending; 7720 blocked; 3394, 5529, 7112,
  8262, 5313, 6628, 8791, 8792, 8793 pending; reify 8827 pending; 112 held-back tasks ready behind 8528.
- m9. Section 6, option 2's "Does not give" cell still says "unless a release rule is added (below)"; after DA1
  point it at the own-view release.

## 5. Sentences a reader with no context could not act on (remaining)

1. Section 2, "T3 Code": never introduced. "T3 Code (the desktop app most of your interactive sessions run in)".
2. Section 2 table, "also held by its alpha gate": "by a gate task that waits for reify's alpha release".
3. Section 6 option 2, "effective priority": "its own tier, or the tier of a task that depends on it if higher".
4. Section 6 option 1 and option 2, "when max-b and max-h are switched": "switched on or off at Anthropic's side".
5. Section 5, "a pre-registered comparison": "a comparison whose measures are fixed before it starts".
6. Section 7 owners, "the three resume tasks are its expiry in practice": "the three resume tasks are what ends it".

## 6. Claims in the new text that I tried to refute and could not

- The proxy's order and strategy (A first; max-c, g, f, e; b and d; h; fill-first), and that the factory's order
  starts with max-d.
- That the proxy records no tokens or cost.
- That the overflow cannot explain the low weeks before 11-04.
- Option 2 without a release: no conflict with a recorded ruling; it shares nothing and remembers nothing.
- "By calendar only" needs no cap state.
- The baseline-first comparison is the right shape in principle (DA3 is about its size and schedule).
- The resume facts: identical prerequisites for the two solar resume tasks, limits 24 and 14, 61 and 223 ready, 112
  held-back tasks, hold threshold of one running task.
- The decisions list has twelve entries and section 10 names the ones a PRD needs.
- No new figure pairs under different rules beyond DA3 point 1 and DM3.
