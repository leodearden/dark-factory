# confusion census 2026-10-10

Project: dark_factory

## Method

```yaml
run_id: census-dark_factory-20261010
as_of_sha: 12cf3ebd163c228423cfe5018013497595d559d6
since: none
evidence:
  window:
  - '2026-10-03T00:00:00+00:00'
  - '2026-10-10T00:00:00+00:00'
  sessions_enumerated: 2258
  skipped_coded: 55
  skipped_zero_signal: 861
  mined: 40
  ledger_rows: 55
verification:
  confirmed: 0
  weakened: 0
  refuted: 1
  unverified: 0
cost:
  miner_calls: 40
  verify_calls: 1
  synthesis_calls: 1
  probe_calls: 2
  embedding_calls: 0
  wall_clock_secs: 2081.64
inputs_consumed: []
extra:
  inputs_consumed_note: this census reads no other instrument's report yet; matching
    sightings against /review, /hotspot-survey and /review-all findings arrives with
    the pre-screen (plans/census-incremental-prd.md leaf L6)
  ledger_created_this_run: false
  ledger_state: ok
  ledger_pruned: 0
  ledger_error: null
  verify_normalisations:
    anchor_rewritten: 0
    anchor_from_remediation: 0
    anchor_from_title: 0
    tag_dropped: 0
    severity_missing: 0
    severity_substituted: 0
    severity_reason_missing: 0
    route_defaulted: 0
    remediation_rejected: 0
    reason_missing: 0
```

## Saturation

- batches: 2
- stop reason: saturated
- operator batch cap: 50 batch(es) (not reached -- mining stopped by: saturated)
  - batch 0: dup_rate=1.00 (total=20, succeeded=20, failed=0, saturated=True)
  - batch 1: dup_rate=0.95 (total=20, succeeded=20, failed=0, saturated=True)

## Verification

- **ALL 1 verified-candidate cluster(s) were REJECTED and none survived.** Suspect a SYSTEMIC verifier failure (model unreachable, tool access denied, or unparseable verdicts) rather than genuinely unfounded claims: this is the observable signature of the 2026-08-03 sandbox incident, in which the verify subprocess was rooted outside the censused tree and every read was permission-denied. Check the run's per-cluster 'verify failed' warnings before reading this census as unremarkable -- a run with real findings is being reported as an empty one if this is systemic.
- handed all 1 novel cluster(s) to the verifier; operator verify cap: 150 (not reached).

## Origin x Manifestation Matrix

_No sightings recorded._

## Synthesis

No novel, verified confusion clusters this census.

## Filed Tasks

_none filed._

## Cost

invoke calls: sonnet miner=40, sonnet verify=1, fable synthesis=1, haiku headroom-probe=2
