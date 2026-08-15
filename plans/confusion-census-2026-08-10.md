# confusion census 2026-08-10

Project: dark_factory

## Saturation

- batches: 2
- stop reason: saturated
  - batch 0: dup_rate=0.95 (total=20, succeeded=19, failed=1, saturated=True)
  - batch 1: dup_rate=1.00 (total=20, succeeded=18, failed=2, saturated=True)

## Origin x Manifestation Matrix

| origin \ manifested | implement |
| --- | --- |
| implement | 1 |

## Synthesis

# Confusion census — 2026-08-10

**Date:** 2026-08-10
**Method:** periodic census per `plans/confusion-reduction-prd.md` §5 (η): stratified-random saturation mining (Sonnet) over session digests → per-finding verification against current main (Sonnet) → this synthesis (Fable). The single finding restated below survived the verification stage; this synthesis adds context-reading against the current tree only — no diagnosis appears here that was not itself verified.
**Companion artifact:** `docs/legibility/confusion-codebook.yaml` (dispositions in §5 are inputs to the merger, which promotes/rejects in place).
**Run notes:** fourth completed periodic census. Previous: 2026-08-05 (`plans/confusion-census-2026-08-05.md`, zero novel verified clusters), 2026-07-31 (15 findings / 4 clusters + 1 one-off). Saturation statistics and filed-task ids are appended by the census runner outside this synthesis.

## Corpus

- **1 verified finding, 1 session** (00997bbb, orchestrated-task, worktree `.worktrees/3871`).
- Composition: for the first cycle since the census began, **zero findings concern the legibility pipeline's own digest instrument** (80% of the 07-31 corpus). The sole finding is an environment/tooling confusion in an ordinary implement-phase task session.
- Phase-stamp coverage: full — the one sighting is stamped `implement`/`implement`, no unknowns.
- A one-finding corpus supports counting and verification, not trend claims. Where this synthesis notes continuity with prior cycles, it is labeled as observation over a near-empty base.

## Executive summary (observations)

1. **An implementer needed a fact from an installed third-party package's source and, lacking any surfaced convention for where installed packages live, reached for a whole-filesystem scan.** The agent for task 3871 ran `grep -rn 'timeout_keep_alive' $(find / -path '*/uvicorn/config.py' -not -path '/proc/*' ...)`; the `find /` traversal exceeded the 2-minute Bash timeout and the call was killed (exit 143) having returned nothing.
2. **The lookup itself was task-required; only the method was confused.** Task 3871's spec (item 1) explicitly requires setting the dashboard client's `keepalive_expiry` *below the MCP servers' own keep-alive timeout* — which is uvicorn's default `timeout_keep_alive`. The agent was doing exactly what the task demanded.
3. **The answer was locally present the whole time, verified on the live tree:** `.worktrees/3871/.venv/lib/python3.13/site-packages/uvicorn/config.py:216` (`timeout_keep_alive: int = 5`) — inside the very worktree the agent had just `cd`'d into. The main checkout's root `.venv` holds the same file. A scoped lookup (`uv run python -c 'import uvicorn; print(uvicorn.__file__)'`, consistent with the project's documented `uv run` idiom) resolves in under a second.
4. **The sighting's cause statement is confirmed against current docs:** CLAUDE.md, CONTRIBUTING.md, and SETUP.md all surface `uv sync`/`uv run` but none states where the venv tree physically lives (no agent-facing mention of `.venv` or `site-packages`), and none gives a recipe for locating an installed package's source.

## Origin × manifestation matrix

Rows = `origin_phase`, columns = `manifested_phase`. Counts are verified sightings (1 total). `merge` and `verify` kept explicitly to show their zeros.

| origin \ manifested | prd | architect | implement | verify | review | merge | recon | ops | unknown | **total** |
|---|---|---|---|---|---|---|---|---|---|---|
| implement | · | · | 1 | · | · | · | · | · | · | **1** |
| **total** | **0** | **0** | **1** | **0** | **0** | **0** | **0** | **0** | **0** | **1** |

Readings (observational): the single sighting is on-diagonal. No merge- or verify-manifested sighting for a third consecutive cycle — but the last two corpora total 1 finding, so this absence carries almost no evidence either way on the PRD's motivating architect/implement→merge hypothesis.

## 1. Verified clusters

### 1.1 Unscoped `find /` for an installed package's source — the venv location is unsurfaced, and the scan cost consumes the entire tool call (1 sighting, 1 session: 00997bbb)

Session 00997bbb is the implementer for task 3871 (dashboard: bound the shared httpx pool — split from 3857), working in `.worktrees/3871`. To satisfy the spec's requirement that `keepalive_expiry` sit below the MCP servers' keep-alive timeout, the agent needed uvicorn's default `timeout_keep_alive`. With no surfaced convention for where the venv/site-packages tree lives, it composed:

```
grep -rn 'timeout_keep_alive' $(find / -path '*/uvicorn/config.py' -not -path '/proc/*' ...)
```

The whole-filesystem traversal ran until the Bash tool's 2-minute timeout killed it (exit 143, "Command timed out after 2m 0s"), returning no result.

Verified against the live tree: the target file exists at `.worktrees/3871/.venv/lib/python3.13/site-packages/uvicorn/config.py`, with the sought fact at line 216 (`timeout_keep_alive: int = 5`); the main checkout's root `.venv` carries the identical file. Scoped alternatives that were available: `uv run python -c 'import uvicorn; print(uvicorn.__file__)'` (matches the `uv run` idiom CONTRIBUTING.md and SETUP.md already document for tests/lint/typecheck), or a `find` rooted at the worktree. Verified doc gap: no agent-facing document states the venv location (`<root>/.venv`, per-worktree `.venv` in task worktrees) or an installed-package source-lookup recipe.

**Relation to the codebook (observation, not a merge):** adjacent to `oneoff-2026-07-07` ("wrong-first-path probes from undocumented repo layout", 7 accrued sightings through 08-07) — both are path-discovery failures rooted in unsurfaced layout conventions. Two distinctions argue against filing it as another sighting of that entry: (a) the target is *outside the repo tree* — installed third-party source, which the repo-map entry's scope doesn't cover; (b) the cost shape differs — a wrong-first-path probe fails fast and cheap (a `not found`, then a retry), whereas an unscoped scan fails slow, consuming the full tool-call budget before returning nothing. The one observed consequence is a burned 2-minute call; whether the agent subsequently recovered is not established by the sighting.

## 2. One-off sightings

None beyond the cluster above.

## 3. Cross-cutting observations

With a one-finding corpus, these are continuity notes, not patterns:

1. **The confusion is in method selection, not goal formation.** The task text itself induced the lookup (the keep-alive comparison is spec'd); the environment then offered no cheap path to the answer and no warning about the expensive one. This is the same "undiscoverable convention rediscovered by failure" family the codebook already tracks for in-repo layout (`oneoff-2026-07-07`) and environment prerequisites (the `DARK_FACTORY_ROOT` candidates), extended here to out-of-tree package sources.
2. **The digest-instrument clusters that dominated 07-24 and 07-31 produced zero verified sightings for a second consecutive cycle.** Over corpora of 0 and 1 findings this is weak evidence of the fixes holding; noted, not concluded.

## 4. Remediation candidates

To be filed via the curator path (plain `submit_task`; curator dedup is the protection — PRD §6.9).

| # | Candidate | Cluster | Size |
|---|---|---|---|
| R1 | Surface the venv convention in agent-facing docs (CLAUDE.md repo map is the natural home, alongside the `<pkg>/src/<pkg>` lines that discharged the in-repo variant of this class): venvs live at `<root>/.venv` and per-worktree `.venv` in task worktrees; to locate an installed package's source, use `uv run python -c 'import <pkg>; print(<pkg>.__file__)'`; note that an unscoped `find /` is a tool-timeout hazard, not a fallback | 1.1 | S |

## 5. Codebook dispositions (input to the merger; promote/reject in place, never delete)

| Cluster / finding | Suggested disposition |
|---|---|
| 1.1 unscoped `find /` for installed-package source | New candidate — do **not** file as a sighting of `oneoff-2026-07-07` (out-of-tree target, slow-fail cost shape; see §1.1); cross-reference that entry as the adjacent in-repo member of the same unsurfaced-layout family |

## 6. Method notes for the next census

- If R1 lands, the discriminating signal next cycle is zero new sightings of this shape in post-fix sessions; a recurrence should prompt checking whether dispatched-agent briefings actually carry the repo-map content interactive sessions see.
- `oneoff-2026-07-07` accrued its seventh sighting on 08-07 and remains `mined-unverified` with no filed remediation recorded on the entry; the two entries' fates should be read together, since one doc change plausibly addresses both.
- Two consecutive near-empty verified corpora (0, then 1) alongside early saturation suggests the nightly trickle is absorbing the head of the distribution, which is the PRD's intended steady state; the census's marginal value now rests on the novelty-spike trigger, worth keeping an eye on rather than tuning yet.


## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=40, sonnet verify=1, fable synthesis=1, haiku headroom-probe=2
