# Write triage on the solar projects: calibration, misfile origins, recommendation (task 7147)

Sources, all committed:

- `calibration/write_triage_calibration_report_solar_challenge.{json,md}` and
  `calibration/write_triage_calibration_report_solar_challenge_platform.{json,md}`:
  per-project calibration over the live-store fixtures described in
  `tests/fixtures/README.md`.
- `calibration/write_triage_solar_soak_origins.json`: the origin tables, census
  fidelity, guard residual and band replay. It is produced by
  `scripts/attribute_write_triage_soak_misfiles.py` from the 2026-10-19 soak
  evidence (read-only; input sha256s in its `provenance`). It runs under the
  shipped write path: t_high 0.8868 / t_low 0.5230, judge_candidate_count 5,
  guard floor 0.60 / gap 0.05.

Every rate below is k/n with its Wilson 95% interval. Unless stated otherwise,
a "misfile" is a contested attach rated `contested_misfile`, counted over
rated contested attaches. That is the census headline's measure. `solar` is
solar_challenge + solar_challenge_platform + scoping_intake;
`dark_factory_reify` is dark_factory + reify.

## 1. Calibration outcome

Both projects refuse in every category, for the same measured reason. Each
category reports `empty_class` with **n_duplicate = 0**.

| project | category | true_dup | unrelated | hard_negative | outcome |
|---|---|---|---|---|---|
| solar_challenge | observations_and_summaries | 0 | 860 | 175 | refused, `empty_class` (n_negative 1035) |
| solar_challenge | procedural_knowledge | 0 | 17 | 4 | refused, `empty_class` (n_negative 21) |
| solar_challenge | preferences_and_norms | 0 | 5 | 1 | refused, `empty_class` (n_negative 6) |
| solar_challenge_platform | observations_and_summaries | 0 | 465 | 96 | refused, `empty_class` (n_negative 561) |
| solar_challenge_platform | procedural_knowledge | 0 | 722 | 98 | refused, `empty_class` (n_negative 820) |
| solar_challenge_platform | preferences_and_norms | 0 | 0 | 0 | refused, `empty_class` (one record, no pair) |

The zero comes from the corpus, not from the extraction. No solar canonical
calls any peer a restatement or duplicate. Most are INDEX records that name
each peer for the one claim or occurrence it holds, so under R2 every peer is
`distinct`. The judgement calls are listed in `tests/fixtures/README.md`.
Neither project gets a `write_triage_project_calibrations` entry.

**The pooled write-path bands against these fixtures.** With no duplicate
pairs there is no duplicate lower tail to place either edge against. Only the
negatives can be read:

- The pooled t_high 0.8868 sits inside the in-corpus hard-negative tail.
  solar_challenge's hard negatives (distinct peers of one topic) reach max
  0.9978 with p95 0.9747; solar_challenge_platform's reach max 0.9604 with
  p95 0.9377. So at least 5% of each corpus's same-topic distinct pairs score
  in the deterministic restate band, all of them in
  observations_and_summaries (p75 0.8608 and 0.9057; no other category's
  negatives reach t_high). solar_challenge_platform's
  observations_and_summaries also has an *unrelated* pair at 0.9002.
- The pooled t_low 0.5230 sits below nearly every hard negative: the minima
  are 0.4938 and 0.4542, and p05 is 0.5798 and 0.6137. It also sits below
  about half of solar_challenge's unrelated pairs (median 0.5420). So almost
  every same-topic pair, and many cross-topic ones, reach the judge.

These are negatives only. They say the pooled edges do not *separate* solar
distinct pairs. They cannot say where a solar duplicate would fall.

## 2. Census fidelity

| group | measured | census headline |
|---|---|---|
| solar | 97/230 = 0.4217 [0.3597, 0.4863] | 98/232 = 0.422 [0.36, 0.49] |
| dark_factory_reify | 54/239 = 0.2259 [0.1775, 0.2831] | 55/240 = 0.229 [0.18, 0.29] |

Disclosed mismatch: three rated contested attaches are missing from the
measured n. All three are children deleted since the soak, so their records
carry no triage cosine and the loader excludes them (`no_triage_cosine`, 6
rows in all): solar loses one misfile and one correct, dark_factory one
misfile. The loader also excludes 47 soak children that triage never
routed: the caller set `parent_id`/`kind` itself, and every one lacks the
triage metadata signature (`caller_declared_child`). The excluded ids are in
the JSON's `provenance.population.excluded`.

Per project, the gap holds in both solar projects: solar_challenge
59/131 = 0.45 [0.37, 0.54] and solar_challenge_platform 38/97 = 0.39
[0.30, 0.49], against dark_factory 33/124 = 0.27 [0.20, 0.35] and reify
21/115 = 0.18 [0.12, 0.26]. Of the restate-band writes, 117/119 were recorded
`restated`, matching team A's figure.

## 3. Where the misfiles come from (contested attaches)

| axis | cell | solar | dark_factory_reify |
|---|---|---|---|
| band | judge | 97/229 = 0.42 [0.36, 0.49] | 54/239 = 0.23 [0.18, 0.28] |
| band | restated | 0/1 | 0/0 |
| hoist | direct | 67/200 = 0.34 [0.27, 0.40] | 44/224 = 0.20 [0.15, 0.25] |
| hoist | via_child | 30/30 = 1.00 [0.89, 1.00] | 10/15 = 0.67 [0.42, 0.85] |
| retarget | kept | 48/138 = 0.35 [0.27, 0.43] | 24/120 = 0.20 [0.14, 0.28] |
| retarget | moved | 49/92 = 0.53 [0.43, 0.63] | 30/119 = 0.25 [0.18, 0.34] |
| record class | completion_ledger | 9/14 = 0.64 [0.39, 0.84] | 13/24 = 0.54 [0.35, 0.72] |
| record class | task_stamped | 43/123 = 0.35 [0.27, 0.44] | 21/133 = 0.16 [0.11, 0.23] |
| record class | untasked | 45/93 = 0.48 [0.39, 0.58] | 20/82 = 0.24 [0.16, 0.35] |
| cos_parent | below t_low | 5/5 = 1.00 [0.57, 1.00] | 2/2 = 1.00 [0.34, 1.00] |
| cos_parent | t_low – 0.65 | 35/46 = 0.76 [0.62, 0.86] | 15/26 = 0.58 [0.39, 0.74] |
| cos_parent | 0.65 – 0.75 | 39/90 = 0.43 [0.34, 0.54] | 23/99 = 0.23 [0.16, 0.32] |
| cos_parent | 0.75 – t_high | 18/88 = 0.20 [0.13, 0.30] | 14/112 = 0.13 [0.08, 0.20] |

What separates solar from dark_factory_reify:

- **Not the band.** 229 of solar's 230 rated contested attaches are judged,
  so the excess is all in the judge band.
- **Hoisting accounts for part of it.** Every solar via-child contested
  attach is a misfile (30/30). Solar also hoists twice as often: 30/230 of
  its contested attaches against 15/240.
- **Most of it is not hoisting.** Direct attaches still misfile at 0.34
  against 0.20, and those intervals do not overlap.
- **The judge's retarget.** When the judge moves a solar contested attach
  off the band winner, it misfiles at 0.53 [0.43, 0.63]. For
  dark_factory_reify that figure is 0.25, and dark_factory_reify's moved and
  kept rates do not differ.
- **Not the completion ledger.** It misfiles heavily in both groups (0.64
  and 0.54). Solar's excess sits in task-stamped and untasked writes.
- **Lower cosine to the filed parent.** 51/230 of solar's contested attaches
  sit below 0.65, against 29/240, and solar misfiles at about twice the rate
  in the 0.65–0.75 bucket too.

Amended and restated attaches were adjudicated only on samples, so most of
their cells are too small to separate the groups. Amended misfile is 6/19 =
0.32 [0.15, 0.54] against 7/21 = 0.33 [0.17, 0.55]. Restated misfile, under
the misfile/contested_misfile definition, is 6/26 = 0.23 [0.11, 0.42]
against 3/13 = 0.23 [0.08, 0.50]. Every cell is in `origin_tables`, broken
down per project as well.

## 4. Residual under the live 7158 attach guards

`guard_residual` replays `AttachGuard.refusals` over each soak attach.

| group | contested removed | rated misfiles removed | rated non-misfiles removed | misfile before | misfile after |
|---|---|---|---|---|---|
| solar | 63 / 230 | 48 | 15 | 97/230 = 0.42 [0.36, 0.49] | 49/167 = 0.29 [0.23, 0.37] |
| dark_factory_reify | 27 / 240 | 15 | 12 | 54/239 = 0.23 [0.18, 0.28] | 39/212 = 0.18 [0.14, 0.24] |

The guards remove about half of solar's contested misfiles. The gap rule
fires 54 times and the floor 24 times; some attaches trip both. For each
rated non-misfile removed, the guards remove 3.2 misfiles on solar but only
1.25 on dark_factory_reify. The solar excess narrows from about 0.20 to about
0.11, but the point estimates still differ: 0.29 against 0.18, with intervals
overlapping only at 0.23–0.24.

What survives the guards is, by construction, a set of attaches with cosine
to the filed parent of at least 0.60 that are not judge-band hoists across a
gap. Inference, not measured per cell: solar's surviving excess is the part
of section 3 the guards cannot reach. That is direct judged attaches in the
0.60–0.75 range, with the judge's retargets among them.

On amended attaches the guard removes 172 of 590 on solar and 106 of 633 on
dark_factory_reify. That costs 3 rated solar attaches, 2 of them misfiles.

## 5. Pooled vs per-project band replay

Both arms replay to the diagonal:

- `project`: the reports' chosen_t_high and chosen_t_low.
- `project_category`: per_category, keyed on the write's own category.

| project | arm | restated→restated | judge→judge | stored→stored | movers |
|---|---|---|---|---|---|
| solar_challenge | project / project_category | 65 | 848 | 1 | 0 |
| solar_challenge_platform | project / project_category | 16 | 643 | 2 | 0 |

No edge was derived, so every arm edge is the pooled fallback (`edges` in the
JSON records `pooled_fallback` on each). No attach is removed, none is lost,
and there are no unknown counterfactuals. This null is a measurement, not an
omission. Per-project calibration produced no band that would route a single
solar write differently.

## 6. Caveats

- **In-sample.** The origin tables, the guard values (chosen on this soak by
  task 7158) and the residual all come from one population. The residual is
  an upper bound on what the guards would achieve on new writes.
- **10-19 state.** The slates (`slate20`) and each slate entry's canonical
  and kind were re-read after the soak. The hoist axis therefore trusts the
  soak's own filed parent over the slate's `canon`; one restate-band write
  shows the two disagreeing. The cosines are text-embedding-3-small, and
  nothing here transfers across the task-6807 embedder cutover.
- **Sample sizes.** Contested attaches are fully rated (230 and 239).
  Amended (19 and 21 rated) and restated (26 and 13) attaches are sampled, so
  their cells are wide. scoping_intake contributes 2 rated contested attaches.
- **Attach-source approximation.** For a judged attach the judge did not
  retarget, the guard replay and the restate-band hoist take the band winner
  as the attach source. The judge may in fact have named a sibling that
  hoists to the same canonical.
- **Exclusions.** 6 writes have no triage cosine, and 47 are
  caller-declared children (section 2).

## 7. Recommendation: a judge-side change

**Per-project write-path bands: no.** There is nothing to configure. Both
solar calibrations refuse every category for want of a single
duplicate-labelled pair, and the replay moves zero writes. Reopening 5547 R3
for solar would buy no measured routing change.

**Leaving it alone: also no.** Under the shipped guards, solar's contested
misfile rate is still 0.29 [0.23, 0.37] against 0.18 [0.14, 0.24], and the
excess sits where a band edge cannot reach:

- inside the judge band (229/230 judged), in the band any per-project t_low
  or t_high would leave the write in;
- on direct attaches (0.34 against 0.20);
- at parent cosines of 0.65–0.75 (0.43 against 0.23), where correct attaches
  sit too, so neither guard rule removes them.

**The measured lever is the judge's contested verdict when it retargets off
the band winner.** On solar that path misfiles at 0.53 [0.43, 0.63], against
0.35 [0.27, 0.43] when the judge keeps the winner. On dark_factory_reify the
two are indistinguishable (0.25 and 0.20). A judge-side change should be
measured against that cell first, ahead of any further band or guard tuning:
for example, a stricter bar for a retargeted `contested`, or showing the
judge the retarget target's cosine. The guard residual by axis is not in
this document. A follow-up should compute it before choosing the change,
because the gap rule already removes the hoisted share of those retargets.
