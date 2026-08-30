# confusion census 2026-08-26

Project: dark_factory

## Saturation

- batches: 2
- stop reason: saturated
  - batch 0: dup_rate=1.00 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 1: dup_rate=0.90 (total=20, succeeded=20, failed=0, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | ops |
| --- | --- |
| ops | 1 |

## Synthesis

All verification and grounding are done — the single finding is grounded against `scripts/legibility/digest.py`, the PRD's §7.2.1 designed-outcomes contract, and the codebook's three adjacent items. Here is the synthesis document for the runner:

---

# Confusion census — 2026-08-26

**Date:** 2026-08-26
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests → per-finding verification against current main (Sonnet) → this synthesis (Fable). The one finding restated below survived the verification stage; this synthesis adds context-reading against the current tree and codebook only — no diagnosis appears here that was not itself verified.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml` (dispositions in §5 are inputs to the merger, which promotes/rejects in place).
**Run notes:** seventh completed periodic census, at the PRD's 5-day hard floor after the previous one. Previous: 2026-08-21 (`plans/confusion-census-2026-08-21.md`, 2 verified findings), 2026-08-16 (3 findings, all accrual), 2026-08-10 (1 finding), 2026-08-05 (zero novel verified clusters), 2026-07-31 (15 findings / 4 clusters + 1 one-off). Saturation statistics and filed-task ids are appended by the census runner outside this synthesis.

## Corpus

- **1 verified finding, 1 session** (631e7374), one sighting.
- Composition: an **instrument-defect** finding — the confusion is produced by the legibility digest pipeline itself, not by a fleet agent. It lands beside two catalogued items (pending candidate `cand-20260819-1`, open entry `entry-cand-20260729-2`) without being a sighting of either as written; the cause family — designed watcher behavior read as failure — is already in the codebook, on an adjacent surface.
- Phase-stamp coverage: **0 of 2 stamps are `unknown`** — second consecutive cycle with full coverage. The sighting originates and manifests in `ops`.
- Session 631e7374 already appears in the codebook **twice**, both coded from its 2026-08-24 digest: under `entry-cand-20260722-3` (rotation launch context omits `DARK_FACTORY_ROOT`) and under `entry-cand-20260729-2` (where its sighting note records exit-124 *correctly* categorized as `designed_outcome`, `tool_error: 0` — i.e. the task-3610 tally fix working on this very session). This census adds a third, distinct defect from the same session.
- A one-finding corpus supports verification and placement, not trend claims. Continuity notes below are labeled as observations over a minimal base.

## Executive summary (observations)

1. **The digest's "Retry Loops" section flags a healthy escalation-watcher rotation's designed once-per-cycle calls as retry loops.** Session 631e7374's digest renders `Bash x6: date -u ...` and `Bash x6: ... scripts/watcher-rearm.sh --queue-dir ... --level 1 --timeout 3600` as Retry Loops entries. The verified finding: the retry-loop heuristic counts repeated identical tool calls across the whole 4-hour rotation without distinguishing designed periodic re-invocation — one wall-clock check plus one bounded re-arm per ~3600s cycle, hitting the intended `CEILING exit=124` three times — from failure-driven retries. The same digest's `signal_counts` show `tool_error=0` and `self_correct=0`: nothing in the session failed, and the section surfaces a "looping" signal for entirely by-design behavior.

## Origin × manifestation matrix

Rows = `origin_phase`, columns = `manifested_phase`. Counts are verified sightings (1 total). `merge` and `verify` kept explicitly to show their zeros; no `unknown` stamps this cycle.

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| ops | · | · | · | · | · | · | · | 1 | · | **1** |
| **total** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **1** | **0** | **1** |

Readings (observational): the single sighting sits on the diagonal. No merge- or verify-manifested sighting for a sixth consecutive cycle; the last five corpora total 8 findings, so this absence still carries little evidence on the PRD's motivating architect/implement→merge hypothesis.

## 1. Verified clusters

### 1.1 "Retry Loops" bucket renders designed per-cycle watcher re-invocation as a retry signal (1 sighting, session 631e7374)

An escalation-watcher rotation's digest carries a Retry Loops section listing the rotation's two per-cycle calls — the wall-clock check (`date -u +"%Y-%m-%dT%H:%M:%S%z"`, ×6) and the canonical bounded re-arm (`scripts/watcher-rearm.sh --queue-dir /home/leo/src/dark-factory/data/escalations --level 1 --timeout 3600`, Bash `timeout: 3660000`, ×6) — as near-identical retry loops. The rotation was healthy: three of the re-arms ended in the wrapper's intended `WATCHER_REARM_OUTCOME: CEILING exit=124`, and the digest's own `signal_counts` report `tool_error=0`, `self_correct=0`. The re-arm command is the exact shape the watcher skill prescribes on current main (`skills/escalation-watcher/SKILL.md` mandates `timeout: 3660000` for `--timeout 3600` and documents "re-arming the identical command" as the healthy continuation; the byte-for-byte match for this command string was verified by the 08-21 census, finding 1.2, on the same command).

**Verified mechanism (read from current main):** `scripts/legibility/digest.py::find_retry_loops` groups every `tool_use` block in the session by `(tool name, canonical sorted-JSON input)` and flags any group recurring at least `RETRY_MIN = 3` times. It is deliberately deterministic and dependency-free — its docstring: no fuzzy similarity, just "same tool, same canonical-JSON input, again" (a design sibling of the decoy-FAIL decision, PRD §13.2). The detector has no time or adjacency dimension and performs no join against tool results: six identical calls spaced one per hour across a healthy rotation are, to this detector, indistinguishable from six back-to-back failing retries. Task 3610's designed/genuine split (`digest.py::classify_error_content` → `iter_genuine_errors` / `iter_designed_outcomes`) operates on the disjoint `tool_result` error-neighborhood scan and shares no code with `find_retry_loops`; the 3610 fix could not have touched this surface and did not claim to. The two layers told different stories about the same session: the tally layer correctly reported zero errors and counted the ceilings under `designed_outcome`, while the Retry Loops section surfaced the same churn as a confusion signal.

**Blast surface (verified structural facts):** `retry_loops` has no `signal_counts` key and no `digest.py::SIGNAL_WEIGHTS` entry — the section's docstring calls it "a structural section, not one of the 5 scored signal classes" — so this false signal contributes zero to the confusion score and cannot lift a session's sampling rank. Its entire cost falls on readers of the rendered digest: the nightly Haiku trickle coder, census miners, and humans. One aggravating structural fact, stated from `digest.py::SECTION_PRIORITY` and not claimed to have manifested in this sighting: under the 15KB soft cap, `designed_outcomes` is trimmed *first* and `retry_loops` second — a byte-pressured digest sheds the section that explains the churn is designed before it sheds the section presenting the churn as loops (the frontmatter `signal_counts.designed_outcome` count survives either way). Finally, arithmetic on verified constants: at one check + one re-arm per ~3600s cycle, any healthy rotation of three or more cycles necessarily crosses `RETRY_MIN` — every long watcher rotation's digest will carry this false signal until the section learns the designed/genuine distinction.

**Relation to the codebook (observation, not a merge):** this is **not** a sighting of `watcher-loop-harness-mismatch` — that entry catalogues the watcher fighting the harness, and none of its frictions (env-var exit 2, foreground kills, rejection loops) appear here; the watcher behaved exactly as specified and the confusion lives in the instrument reading it. One of that entry's existing sightings (session f41dd7df, ×25 re-arm) already noted in passing that the bounded-wait pattern "still reads as a churn of identical failing-looking calls" — this finding isolates that reading hazard as its own instrument defect. The nearest catalogued items are: pending candidate `cand-20260819-1` ("Watcher-rearm designed_outcome capture incomplete vs retry loop invocation count", session c07ee777, 08-19), which juxtaposes the same two sections but with the inverse complaint — it took the Retry Loops ×6 as ground truth and asked why Designed Outcomes listed only 4; and open entry `entry-cand-20260729-2` (digest scorer counting exit-124 as `tool_error`), whose own 08-24 sighting on this same session records the tally half fixed. The two candidates and this finding are facets of one missing join between the retry-loop grouping and the designed-outcome classification.

## 2. One-off sightings

None beyond the cluster above (itself single-sighting this cycle).

## 3. Cross-cutting observations

1. **The residue of a landed fix moved one surface over, again.** The 08-21 census observed this for the watcher skills (fix-shape present in the sighting, friction relocated to the unmodeled rejection path); this cycle shows the same shape inside the instrument: task 3610 partitioned the error *tally* into genuine/designed, and the next sighting accrues on the *rendered section* that partition never covered. An entry marked fixed at one layer can keep generating sightings at the adjacent layer, and count-based readings will misattribute them.
2. **One session, three catalogued defects, three layers.** 631e7374 now documents an environment defect (rotation launch omits `DARK_FACTORY_ROOT`), a tally-layer fix *working* (exit-124 → `designed_outcome`, under `entry-cand-20260729-2`), and a rendering-layer gap (this finding). The same session is simultaneously the positive control for one instrument fix and the specimen for the next instrument defect.
3. **The instrument is this cycle's only subject.** The sole verified finding is a false positive produced by the legibility pipeline about a healthy session. For a census whose corpus *is* digests, instrument false positives are self-referential risk: `cand-20260819-1` was itself mined from this same Retry Loops/Designed Outcomes juxtaposition, and the codebook already carries a family of digest self-ingestion candidates. A false signal that costs nothing in scoring can still spend miner and coder attention every night.
4. **Zero `unknown` phase stamps for a second consecutive cycle** (2/2 after 08-21's 4/4), on a minimal base.

## 4. Remediation candidates

To be filed via the curator path (plain `submit_task`; curator dedup is the protection — PRD §6.9).

| # | Candidate | Cluster | Size |
|---|---|---|---|
| R1 | Give the Retry Loops section designed-outcome awareness by a deterministic join, annotating rather than suppressing: for each `digest.py::find_retry_loops` group, count member calls whose paired `tool_result` was classified a designed outcome by the existing `classify_error_content` partition, and render the count on the group's line (e.g. `Bash x6 — 3 designed-outcome results [CEILING exit=124]`). Annotation preserves the detector's dependency-free contract (PRD §13.2 sibling decision) and §7.2.1's fail-toward-genuine principle — a genuine retry storm interleaved with ceilings stays fully visible — while making the section self-disambiguating for the trickle coder and census miners. It also makes `cand-20260819-1`'s 4-vs-6 count mismatch legible in place. Implementer should decide the `instrument_version` question explicitly: an annotation that changes neither `signal_counts` nor the section partition arguably needs no bump per §7.2.2, but the same-transcript-different-rendering test should be applied deliberately, not assumed | 1.1 | S |

## 5. Codebook dispositions (input to the merger; promote/reject in place, never delete)

| Cluster / finding | Suggested disposition |
|---|---|
| 1.1 Retry Loops renders designed per-cycle re-invocation as retries | Attach as a sighting to `cand-20260819-1`, retitled around the shared mechanism (Retry Loops section blind to the designed-outcome classification) so both facets — designed churn rendered as loops (this finding) and the count mismatch between the two sections (its founding sighting) — live under one candidate; or mint a sibling candidate if the merger keeps the two complaints separate. Explicitly **not** a sighting of `watcher-loop-harness-mismatch` (watcher behavior healthy; record that discriminator) and **not** of `entry-cand-20260729-2` (tally layer, fixed and confirmed on this same session — but cross-reference it as where the residue moved from). 631e7374's two existing sightings concern different defects and stay where they are |

## 6. Method notes for the next census

- If R1 lands, the check is direct: a later watcher-rotation digest should show the annotated Retry Loops line, and the nightly trickle should stop minting rearm-churn candidates from healthy rotations. Persisting new candidates of this shape *after* the annotation renders would point at the coder prompt, not the digest.
- The 08-21 census's R1 (rejected-call clause in the watcher skills) shows `_none filed._` in that report's Filed Tasks section — the next census should establish whether it was filed through another path or dropped, before treating its cluster as remediated.
- Carried forward from 08-21: whether the missing-required-parameter class (its finding 1.1) recurs on first-party tools, and whether the merger consolidated the stray-comma Read candidates (`cand-20260806-12`/`cand-20260812-19`). Neither appears in this cycle's one-finding corpus, which bounds what the absence says.

---

*Synthesis note to the runner: written from the 1 verified finding supplied; mechanism and blast-surface claims verified by reading `scripts/legibility/digest.py` (`find_retry_loops`, `RETRY_MIN`, `signal_counts`, `SIGNAL_WEIGHTS`, `SECTION_PRIORITY`, the `classify_error_content`/`iter_designed_outcomes` partition) and `plans/confusion-reduction-prd.md` §7.2.1–7.2.2 on current main at synthesis time; codebook relations verified by reading `docs/legibility/confusion-codebook.yaml` (entries `watcher-loop-harness-mismatch`, `entry-cand-20260729-2`, `entry-cand-20260722-3`, candidate `cand-20260819-1`, and session 631e7374's two existing sightings); the re-arm `timeout: 3660000` contract confirmed present in `skills/escalation-watcher/SKILL.md`. No tasks filed and no codebook edits made from this synthesis — filing and merger application are the runner's steps.*


## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=40, sonnet verify=2, fable synthesis=1, haiku headroom-probe=2
