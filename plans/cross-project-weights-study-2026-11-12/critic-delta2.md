# Critic, third pass on `/tmp/team-weights-1112/REPORT.md` (file as written 17:29:23Z; this pass about 17:50Z)

Read-only, same rules. Scope: the folds of DA1 to DA3 and DM1 to DM6, leftovers, the two new tables, and the two
new sentences on what a null experiment or a declined one would mean.

**Verdict: nothing answer-changing remains. A 0, M 3, m 7.** One of the M items (DM-c) is a conclusion sentence
that overreaches and should be narrowed before the report goes out.

## 1. Folds

| Item | Folded | Note |
|---|---|---|
| DA1 calendar release | Correctly | Bullets, recommendation, rulings row 5, decision 5 all agree |
| DA2 floor on task verifies and background runs | Correctly | Figures match `critic/taskverify_out.txt`. My own "about 2 to 1" was loose: M-b |
| DA3 projection for the recommended design | Correctly | Table rows 1 to 3 match my delta file. The "for scale" row is new and mixes rules: M-a |
| DM1 proxy | Correctly | Two small residues: m3, m4 |
| DM2 ruling row | Correctly | |
| DM3 section 2 note | Correctly | |
| DM4 resume paragraph, steps 1 and 2 | Correctly | m5 on the "delayed" setting |
| DM5, DM6 | Correctly | |

Leftovers searched for and not found: "calendar release" as a recommendation, "in two places", "37 to 40 minutes",
"removes the merge-verify half", "nothing records", "reads or remembers", "within 2 a day", "0 to 4 landings".
One leftover contradiction found (m3).

## 2. M findings

**M-a. Section 5, table "What the recommended design is projected to give", last row: "The pause (for scale) |
about 36 observed | 2.5 to 5.6 a day".** The caption says "same model and assumptions throughout", and this row is
neither. The three design rows are model means from 43.9 minutes and count landings gained through the workstation
verify only. The pause row gives an observed median (against a median of 43.4) and a landing bound that also counts
the laptop. By the rule of the other rows the pause is 33.1 minutes (37.0 with Dark Factory's own growth during the
pause; about 36 observed) and 1.8 to 3.1 landings a day (`pause/projection_out.txt`, last line of the first block).
As printed, the recommended design looks like a fifth of the pause; like for like it is a third to a half.
Proposed row: "The pause, same model and same channel (for scale) | 33.1 to 37.0 (about 36 observed) | 1.8 to 3.1
a day (2.5 to 5.6 when the laptop is counted as well)".

**M-b. Section 5 recommendation and table row 3, section 4, decision 2: "about 2 to 1 on task verifies"; "2 to 1 on
task verifies and background runs".** The 2-to-1 figure was mine, from 90th-percentile arithmetic. Run through the
same projection as the rest of section 4 (run time inversely proportional to share; `verify_admission_plan` test
runs, 10-25 to 11-08), the share of runs put past the per-command limit is:

| Project | Task verifies at 1.5 to 1 / 2 to 1 / 3 to 1 | Background runs at 1.5 to 1 / 2 to 1 / 3 to 1 |
|---|---|---|
| reify | 2% / 9% / 36% | 41% / 62% / 76% |
| solar-challenge-platform | 0% / 5% / 31% | 0% / 15% / 50% |
| solar-challenge | 0% / 3% / 22% | 2% / 10% / 55% |

So 2 to 1 is not clean on task verifies (3 to 9%), and the recommendation covers only reify's background run while
table row 3 applies 2 to 1 to all background runs (10 and 15% on the solar pair). The background term is 0.3 of the
10.8 modelled minutes, so leaving every background run alone gives up nothing measurable. Proposed text: "Keep the
factor on task verifies between 1.5 and 2 to 1 (the projection puts 0 to 2% of them past their limit at 1.5 to 1
and 3 to 9% at 2 to 1), leave background runs at today's weight in all three projects, and set the service weight
freely." Table row 3: "1.5 to 2 to 1 on task verifies, background runs left alone, 3 to 1 on services | about 39 to
42 | 0.5 to 1.2 a day". (At 1.5 to 1 the model gives 39.7 to 41.8 minutes.) Decision 2's last sentence: "about 1.5
to 2 to 1".

**M-c. Section 5, "If the experiment shows no effect, the honest conclusion is that CPU weights between projects
are not worth a PRD and the allowance is the whole of the problem"; decision 3, "Declining is a reasonable answer,
and it would mean CPU weights are left out of the PRD."** Declining the experiment is a reasonable answer on this
evidence: the projected prize is about one landing a day. The two consequences drawn from it are overreach.
1. *The experiment tests one design.* It weights only task verifies and services, at no more than 2 to 1, on the
   workstation. A null result says nothing about the lever the design leaves out: other projects' merge verifies,
   the model's largest per-run term (6.8 ± 1.0 minutes) and the only way projects collide on the laptop (4.8
   minutes each, on the host that verified 54 to 58% of Dark Factory's landings). By the pause seat's own rule a
   3-to-1 factor on merge verifies on the laptop alone projects to 0.8 to 1.6 landings a day, as much as the whole
   recommended design. That lever needs your reading of "merge verify is never throttled", and its floor on the
   laptop was not examined: the per-project ratios in decision 2 (reify about 1 to 1) come from workstation run
   times, while the same projects' laptop merge verifies ran 28 / 38 minutes (reify), 11 / 16 (solar-challenge-platform) and 6 / 9
   (solar-challenge), median / 90th percentile, and which time limit applies there was not established. A null also says nothing about one
   group per project (task 3394), which is wanted for measurement whatever weights do.
2. *"No effect" is undefined.* With about 20 workstation merge verifies a day and a spread of about 10 minutes, two
   weeks of alternating blocks resolve a difference of about 2.5 minutes at two standard errors. The projected
   effect is 2.4 to 4.9 minutes. A result that fails to reach significance in two weeks cannot exclude the low end.
   The threshold has to be stated before the run (reify task 8827 asks for exactly that).
3. *"The allowance is the whole of the problem" contradicts the report's own items 6 and 7 and decision 12.* Memory
   pressure caused 57 of the 69 dispatch holds and is unweighted and unattributed; the resume is unguarded; the
   laptop has no arbitration. None of those is the allowance, and none is settled by this experiment.
- Proposed closing sentence: "If the experiment shows less than about 2 minutes (a threshold to fix before it
  starts), the conclusion is that weights on task verifies and services are not worth building. It would say
  nothing about merge verifies, which are the largest term and the only contention on the laptop and are held out
  by your directive, nor about memory, the resume guard or project groups for measurement."
- Proposed decision 3 ending: "Declining is a reasonable answer. It would leave this form of CPU weight out of the
  PRD; the merge-verify question (decision 2), the resume guard, memory and project groups stand on their own."

## 3. m findings

- m1. Section 4 floor table: the task-verify and background figures are the test run alone (the longest command,
  the right thing to set beside a per-command limit); the merge-verify figures are the whole verify. Say so in the
  line above the table.
- m2. Section 4, "14 crashes in about 138 task verifies against none in 76 merge verifies": add that it is Dark
  Factory's own, 09-25 to 09-28.
- m3. Section 10, "(interactive overflow is now a candidate for one of them)" contradicts section 6 ("It explains
  little"). Proposed: "(interactive overflow explains at most part of two of the six)".
- m4. Section 6, "Four of the six low weeks ... had no proxy traffic": one of the four had 8 requests. And max-b's
  11-06 week ($1,504) took 1,271 proxied requests, which is the second of the two partly explained weeks.
- m5. Migration step 2, the "delayed" setting: `docs/task-authoring.md` section 6 describes it as set when a task
  is filed. Whether it can be added to the existing resume task was not checked; it may mean re-filing that task.
- m6. Section 5, "agents mostly do not wait for CPU": the mechanisms seat's point is that agents use little CPU
  (about 60 agents for about one core). "Use little CPU".
- m7. Section 6, "37 of their 52 hours (72%)": 37.0 of 51.7.

## 4. Checked and found right

- Section 5 table rows 1 to 3 and section 1 item 4 against `pause/projection_out.txt` and my delta arithmetic.
- Widened floor table against `critic/others_verify_out.txt` and `critic/taskverify_out.txt`; limits against each
  project's loaded config.
- "22 to 36%" and "half to three quarters" in section 4.
- Proxy figures (29,749; 10,410; 7,116; 4,877; 2,046; A's last request 11-11 16:31Z).
- Step 1's premise that a halt survives a restart: the three paused orchestrators restarted on 11-10 and stayed
  halted.
- Decision 5, rulings rows 2 and 5, and the option 2 "Does not give" cell agree with the new release text.
