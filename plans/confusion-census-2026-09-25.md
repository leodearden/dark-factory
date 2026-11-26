# confusion census 2026-09-25

Project: dark_factory

## Saturation

- batches: 3
- stop reason: saturated
  - batch 0: dup_rate=0.89 (total=20, succeeded=19, failed=1, saturated=False)
  - batch 1: dup_rate=1.00 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 2: dup_rate=1.00 (total=20, succeeded=20, failed=0, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | recon |
| --- | --- |
| recon | 1 |

## Synthesis

**Date:** 2026-09-25
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests, per-finding verification against current main (Sonnet), then this synthesis (Fable). One finding reached synthesis. This document adds a read of the source session's transcript (`~/.claude/projects/-home-leo-src-dark-factory/88b152e9-….jsonl`, 829 records), the trickle-coder transcript that produced the second codebook sighting, the recon watcher skill on main, `recon-watch/run.sh`, and the codebook. Every mechanism claim below names the evidence it rests on.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml`. Dispositions in §3 are inputs to the merger.
**Run notes:** thirteenth completed periodic census, at the PRD's 5-day floor after 2026-09-20 (`census-state.json` reads 2026-09-20). Previous corpora: 09-20 (9 verified findings), 09-15 (1), 09-10 (2), 09-05 (1), 08-31 (1), 08-26 (1), 08-21 (2), 08-16 (3), 08-10 (1), 08-05 (0), 07-31 (15 / 4 clusters), 07-24 (52). Saturation statistics and filed-task ids are appended by the runner outside this synthesis. The synthesis sandbox refused direct reads of `~/.claude/recon-watch/loop.sh` and of the lease directory, so the loop script is read here only as the source session's own `cat` of it (transcript record 473), which is verbatim and sufficient for every claim made about it.

### Corpus

- **1 verified finding, 1 session, 1 sighting, 1 cluster.** The session is an interactive `recon-escalation-watcher` run (Opus, `effort=xhigh`, CLI 2.1.278), not an orchestrated task session. It opened 2026-09-21 12:38Z, claimed the `recon-watcher-dark_factory` lease at 12:41:54Z (`decision=acquired`, `holder_liveness=none`), and continued into session `b3d32dd4` at 10:53Z on 09-22.
- **The finding is already in the codebook, twice, and the verifier presented it as novel.** `cand-20260921-9` records this session under this exact quote, stamped `ops × ops`, disposition pending. `cand-20260922-8` records the same quote against session `65b07799`, stamped `unknown × ops`, disposition pending. Session `65b07799` is a Haiku trickle-coder run of 44 seconds and two assistant turns (cost-state record: $0.07, zero tool time) whose transcript mentions `trickle-coder`, `digest` and `codebook` throughout; it produced no probe of its own. Its "sighting" is the coder reading the 09-21 digest, which is the re-ingested-content shape task 5685's classifier exists to filter (`scripts/legibility/digest.py::is_reingested_content`). `cand-20260922-7` (the over-broad `pgrep`) is the same artifact for `cand-20260921-10`.
- **Role stamps.** The verifier stamped `recon × recon`. The PRD's enum reserves `recon` for reconciliation stages; the actor here is the human-facing watcher that closes the reconciliation queue, and the existing candidate already carries `ops × ops`. This synthesis keeps `ops × ops`; the runner's matrix renders the verifier's stamp.

### Executive summary (observations)

1. **The misidentification lasted under three minutes and was corrected by reading the script.** At 13:47:44Z the agent noticed two `watcher-rearm.sh` pairs on the queue started 57 seconds apart; at 13:48:35Z, after tracing grandparents, it wrote that `loop.sh` was "an independent recon-watch loop driving its own watcher on the same queue" and went to check for a Claude child that "could also be closing records"; at 13:49:01Z it said "Now it's clear, and it matters. Let me confirm by reading the loop script"; at 13:49:42Z the `cat` returned the header; at 13:49:47Z it wrote "it's a deliberate component, not a stray"; at 13:51:19Z it reported the correction to the user under the heading "A correction and a finding". No record was closed, reaped, or killed on the wrong belief.
2. **The loop's contract is stated in exactly one place, and that place is outside the repo.** `~/.claude/recon-watch/loop.sh` is untracked: `git ls-files recon-watch` lists only `mcp.json` and `run.sh`, and a repo-wide grep for `loop.sh` outside the codebook hits only an unrelated `l2-watch-loop.sh` mention in the 09-09 capacity study. Its header (transcript record 473) reads: "Detached recon-queue watcher loop… One long-lived shell + one python inotify child at a time -- the shape that survives the harness background-task reaper. Deliberately does NOT heartbeat the lease: the lease must go stale if the owning Claude session dies, or a dead holder would block the next watcher." The process had been running since 2026-09-13 15:31:46 (`etime` 7d23h at the probe) and its `history.log` held 1,146 lines from 09-12.
3. **The skill the watcher runs under describes a different lease model and does not mention the loop.** `skills/recon-escalation-watcher/SKILL.md` instructs the Claude session to `lease-claim` at startup and `lease-heartbeat` "each time you (re)start this watcher subprocess" (lines 302–309), says the claim "replaces any pgrep/ps-tree archaeology for a duplicate recon watcher" (line 79), and names `recon-watch/run.sh` as the launcher (line 90). Neither the skill, `run.sh`, nor `scripts/watcher-rearm.sh` contains the string `loop.sh` or describes a non-heartbeating persistent half. The agent therefore had a documented reason to expect that a second `watcher-rearm.sh` on the same queue meant a second closer, and nothing in its briefing saying otherwise.
4. **The verifier's "treated its stale lease as a fault" is not evidenced in the transcript.** At startup the lease directory already held `recon-watcher-dark_factory.lease`; the `lease-reap` printed nothing; the claim succeeded with no holder. The agent's later phrase "the stale lease I reaped at startup, working as intended rather than a fault" is a retrospective reading. No turn before 13:48Z labels that lease a fault. The verifier's "under time pressure" likewise has no support in the records read.
5. **The same session then found something the loop's own log confirms.** The agent used `history.log` to show `esc-7692-1` had re-fired 26 times at 300-second back-off until its 12:58:41Z resolve, and that `esc-7692-2` followed at 13:03:39Z. That reading is what the 09-21 feedback memory on in-flight recon re-files cites. The misidentification and the finding came from the same probe sequence.

### Origin × manifestation matrix

The runner's matrix renders the verifier's stamps. The table below carries the refined stamp this synthesis establishes (§Corpus).

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| ops | · | · | · | · | · | · | · | 1 | · | **1** |
| **total** | **0** | **0** | **0** | **0** | **0** | **0** | **0** | **1** | **0** | **1** |

Readings, observational. One sighting, in an interactive operator session, manifesting within three minutes of its origin. The `merge` and `verify` columns are zero for a twelfth consecutive cycle; the PRD's motivating architect/implement→merge hypothesis remains untested by the thirteen post-07-24 corpora, which now total 23 findings.

### 1. Verified cluster

#### 1.1 The recon-watch persistent loop is a deliberate no-heartbeat component documented only in its own untracked header, and the watcher skill describes a lease model that does not include it (1 sighting, session `88b152e9`, 2026-09-21 13:48–13:51Z)

**The trace.** Records 456–485 of the transcript. The trigger was a process probe that showed two `scripts/watcher-rearm.sh --queue-dir …/data/reconciliation/escalations --timeout 3600` processes: pid 1591323 (parent 63540, `/bin/bash /home/leo/.claude/recon-watch/loop.sh`) and pid 1633220 (the session's own Bash snapshot shell). The agent's first reading, at 13:48:35Z, was "an independent recon-watch loop driving its own watcher on the same queue". Its next probe (13:48:35Z) asked "any claude under it?" and found `loop.sh(63540)---bash(1591323)---uv(1591350)-+-python3(1591411)`, no `claude`. It then read the script header (13:49:01Z to 13:49:42Z) and reversed itself in the following turn.

**What the agent could have read beforehand.** The skill on main gives the lease as the single duplicate-detection mechanism and assigns heartbeating to the Claude session. The loop's design point, that it must not heartbeat so a dead owner's lease expires, is stated in the loop's header and nowhere in the repo. The session's own report frames the residual as "one redundancy to note, not a problem": two read-only inotify watchers on one queue, fires journaled twice, one closer.

**What the sighting cost.** Four tool calls and about 2.7 minutes between the wrong sentence and the correction; the user-facing report carried the correction explicitly. Nothing was acted on under the wrong belief.

**Codebook state.** `cand-20260921-9` (this session, `ops × ops`) is the record for this sighting. `cand-20260922-8` (session `65b07799`, `unknown × ops`) carries the same quote from a Haiku trickle-coder run that ingested the 09-21 digest and made no probe of its own. No promoted entry covers the two-part watcher design; the nearest entries are the run_in_background reaping notes on lines 1715–1745, which describe the failure the loop was built to survive, not the loop.

### 2. Dispositions (inputs to the merger)

- **`cand-20260921-9`**: the sighting is verified against the transcript; keep as the canonical record for this cluster. The cause text ("observational ambiguity") is consistent with what was read; the specific observation that the contract lives only in an untracked file outside the repo is not yet in its cause text.
- **`cand-20260922-8`** and **`cand-20260922-7`**: same quotes as `cand-20260921-9` and `cand-20260921-10`, sourced from a 44-second Haiku trickle-coder transcript with no probes. These are re-ingested-content sightings of the 09-21 session, not independent observations.
- **Verifier stamp `recon × recon`**: the existing candidate's `ops × ops` matches the actor; this synthesis does not adopt the verifier's stamp.

### 3. Not verified, stated for the record

- Whether the lease that pre-existed at 12:40Z was stale, and what `lease-reap` did with it, is not visible: the reap printed nothing and the claim reported no holder.
- Whether any later recon watcher session has repeated the misidentification is not checked; the 09-22 continuation session `b3d32dd4` was not read.


## Filed Tasks

_1 ticket(s) filed -- the curator's create/combine/drop decision is still pending, so no task id exists yet; resolve_ticket returns the task id once it does._

- tkt_0RV1TMT1HA5312FK0PEM517GT3

## Cost

invoke calls: sonnet miner=60, sonnet verify=3, fable synthesis=1, haiku headroom-probe=2
