# Base-distance report — pre-redrive vs. post-redrive

**Generated artifact. Do not edit by hand — regenerate (see below).**

This is the measured record of what task 4759's two provenance fixes moved.
`--base-distance-report` compares a BEFORE manifest against what a redrive
produces right now, so once the redrive has landed the committed state can no
longer produce this evidence on its own: run against the committed manifest it
compares the post-redrive rows with themselves and correctly reports no
movement. The BEFORE side is therefore read from the pre-redrive manifest,
extracted from git history.

Reproduce with:

```bash
git show 6da60348fd:orchestrator/src/orchestrator/evals/tasks_hard_v2/_meta/curation.json \
  > /tmp/curation-pre-redrive.json
python3 scripts/mint_hard_v2_fixtures.py --base-distance-report \
  --before-manifest /tmp/curation-pre-redrive.json
```

`6da60348fd` is the last commit to touch the manifest before the redrive
(`dc73a9acc5`). The AFTER side is re-derived live against the three source
checkouts, so re-running this on a machine where they are absent, or after
their `main` has moved, will not reproduce the table byte-for-byte — the
distances are measurements of live history, not of the repo.

## What moved

Three fixtures were carrying a false `reference_unavailable` because the
matcher only knew the `Merge task/<id> into main` spelling. Their landing
merges are colon-spelled, and resolving them replaced an approximated base
with the task's true branch point (`M^1`):

| fixture | before | distance from the true branch point (all/1st-parent) | after |
|---|---|---|---|
| `reify_task_4026` | `timestamp_walk` | **245 / 78** | `merge_first_parent` |
| `reify_task_2573` | `status_autocommit` | **977 / 794** | `merge_first_parent` |
| `reify_task_2379` | `status_autocommit` | **189 / 178** | `merge_first_parent` |

A fourth, `reify_task_4086`, kept its `timestamp_walk` rung (it has no landing
merge under either spelling) but its base moved `a18a306574` -> `743a0e2e9a`:
the rung-3 walk gained `--first-parent`, so where it previously returned a
commit that only ever lived on a merged-in side branch — a tree state that was
never a state of `main` — it now returns a real state of `main`. That change is
NOT visible as a distance below, because without a landing merge the true
branch point is not derivable from git at all and both distances read `n/a`.
The 19 `n/a` rows are the honest report of that limit, not an absence of
movement.

## Full table

| fixture | before rung | before base | dist (all/1st-parent) | after rung | after base | dist (all/1st-parent) |
|---|---|---|---|---|---|---|
| `reify_task_2320` | merge_first_parent | `78c83758b4` | 0/0 | merge_first_parent | `78c83758b4` | 0/0 |
| `reify_task_2324` | status_autocommit | `66b9b752bd` | n/a | status_autocommit | `66b9b752bd` | n/a |
| `reify_task_2325` | status_autocommit | `0cfbb2d73c` | n/a | status_autocommit | `0cfbb2d73c` | n/a |
| `reify_task_2330` | status_autocommit | `8a15412b9d` | n/a | status_autocommit | `8a15412b9d` | n/a |
| `reify_task_2336` | status_autocommit | `37b1440170` | n/a | status_autocommit | `37b1440170` | n/a |
| `reify_task_2379` | status_autocommit | `7d2ed24d28` | 189/178 | merge_first_parent | `3d05445513` | 0/0 |
| `reify_task_2384` | status_autocommit | `7070df2397` | n/a | status_autocommit | `7070df2397` | n/a |
| `reify_task_2531` | status_autocommit | `c14f1f7595` | n/a | status_autocommit | `c14f1f7595` | n/a |
| `reify_task_2573` | status_autocommit | `4e12e7a06d` | 977/794 | merge_first_parent | `1be3560556` | 0/0 |
| `reify_task_2654` | merge_first_parent | `441d5af3ea` | 0/0 | merge_first_parent | `441d5af3ea` | 0/0 |
| `reify_task_2655` | status_autocommit | `5e80284e37` | n/a | status_autocommit | `5e80284e37` | n/a |
| `reify_task_2656` | merge_first_parent | `8d8a0e1740` | 0/0 | merge_first_parent | `8d8a0e1740` | 0/0 |
| `reify_task_2696` | status_autocommit | `ce759ec04d` | n/a | status_autocommit | `ce759ec04d` | n/a |
| `reify_task_2699` | timestamp_walk | `fb265292f2` | n/a | timestamp_walk | `fb265292f2` | n/a |
| `reify_task_2778` | status_autocommit | `7c7f5c75a9` | n/a | status_autocommit | `7c7f5c75a9` | n/a |
| `reify_task_2908` | timestamp_walk | `d5da039b1d` | n/a | timestamp_walk | `d5da039b1d` | n/a |
| `reify_task_2911` | merge_first_parent | `61c069353f` | 0/0 | merge_first_parent | `61c069353f` | 0/0 |
| `reify_task_2958` | merge_first_parent | `4b0bd05a34` | 0/0 | merge_first_parent | `4b0bd05a34` | 0/0 |
| `reify_task_3004` | timestamp_walk | `6645b90dc5` | n/a | timestamp_walk | `6645b90dc5` | n/a |
| `reify_task_3024` | merge_first_parent | `62492a2456` | 0/0 | merge_first_parent | `62492a2456` | 0/0 |
| `reify_task_3092` | timestamp_walk | `3f6cbf9f45` | n/a | timestamp_walk | `3f6cbf9f45` | n/a |
| `reify_task_3095` | merge_first_parent | `fe3b8ab103` | 0/0 | merge_first_parent | `fe3b8ab103` | 0/0 |
| `reify_task_3228` | merge_first_parent | `5150d11bcb` | 0/0 | merge_first_parent | `5150d11bcb` | 0/0 |
| `reify_task_3443` | merge_first_parent | `630c616b55` | 0/0 | merge_first_parent | `630c616b55` | 0/0 |
| `reify_task_3586` | timestamp_walk | `6485f8f53b` | n/a | timestamp_walk | `6485f8f53b` | n/a |
| `reify_task_3779` | merge_first_parent | `e80dab6b99` | 0/0 | merge_first_parent | `e80dab6b99` | 0/0 |
| `reify_task_3822` | timestamp_walk | `2ceaf9ec17` | n/a | timestamp_walk | `2ceaf9ec17` | n/a |
| `reify_task_3834` | merge_first_parent | `7c79728630` | 0/0 | merge_first_parent | `7c79728630` | 0/0 |
| `reify_task_3845` | timestamp_walk | `2ceaf9ec17` | n/a | timestamp_walk | `2ceaf9ec17` | n/a |
| `reify_task_3883` | timestamp_walk | `2ceaf9ec17` | n/a | timestamp_walk | `2ceaf9ec17` | n/a |
| `reify_task_4026` | timestamp_walk | `e21d047026` | 245/78 | merge_first_parent | `794d321596` | 0/0 |
| `reify_task_4086` | timestamp_walk | `a18a306574` | n/a | timestamp_walk | `743a0e2e9a` | n/a |
| `reify_task_4370` | merge_first_parent | `264ee8cd20` | 0/0 | merge_first_parent | `264ee8cd20` | 0/0 |
| `reify_task_4832` | merge_first_parent | `817636f656` | 0/0 | merge_first_parent | `817636f656` | 0/0 |
| `df_task_1229` | merge_first_parent | `9adda2df34` | 0/0 | merge_first_parent | `9adda2df34` | 0/0 |
| `df_task_2169` | merge_first_parent | `4c40b7fa23` | 0/0 | merge_first_parent | `4c40b7fa23` | 0/0 |
| `df_task_2260` | merge_first_parent | `20c934ca59` | 0/0 | merge_first_parent | `20c934ca59` | 0/0 |
| `df_task_882` | timestamp_walk | `fd4758fcff` | n/a | timestamp_walk | `fd4758fcff` | n/a |
| `kl_task_543` | merge_first_parent | `dd2d3ca026` | 0/0 | merge_first_parent | `dd2d3ca026` | 0/0 |

39 fixture(s); 19 still have an APPROXIMATED base after the redrive. Approximated is exactly "did not resolve to `M^1` of a landing merge", so for every one of them the true branch point is not derivable from git and BOTH distances are UNMEASURABLE — reported as `n/a`, never as 0. Distances are REPORTED, not asserted against a threshold.
- `reify_task_2324`: No landing merge for task 2324 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_2325`: No landing merge for task 2325 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_2330`: No landing merge for task 2330 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_2336`: No landing merge for task 2336 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_2384`: No landing merge for task 2384 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_2531`: No landing merge for task 2531 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_2655`: No landing merge for task 2655 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_2696`: No landing merge for task 2696 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_2699`: No landing merge for task 2699 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_2778`: No landing merge for task 2778 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_2908`: No landing merge for task 2908 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_3004`: No landing merge for task 3004 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_3092`: No landing merge for task 3092 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_3586`: No landing merge for task 3586 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_3822`: No landing merge for task 3822 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_3845`: No landing merge for task 3845 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_3883`: No landing merge for task 3883 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `reify_task_4086`: No landing merge for task 4086 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.
- `df_task_882`: No landing merge for task 882 under either accepted subject spelling, so the true branch point is not derivable from git and the distance is UNMEASURABLE — not zero. The base stays approximated; a readout that depends on a true branch point should exclude this fixture.

