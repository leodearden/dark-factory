# Merge-leg pytest `-n` A/B (16 vs 8) — cut 2026-09-16T14:09+00:00

Source: `/home/leo/src/dark-factory/data/orchestrator/runs.db` (`merge_verify` events, runner=local, since 2026-09-08T10:18+00:00).
Arms from `config_reload` events changing `verify_env.PYTEST_XDIST_AUTO_NUM_WORKERS`:

- 2026-09-08T10:18+00:00 → `16`
- 2026-09-10T19:06+00:00 → `8`

| arm | window | n (all) | failed | spec | n (passed, non-spec) | median s | p90 s | max s | ≥3500 s |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 2026-09-08T10:38+00:00 → 2026-09-10T18:16+00:00 | 52 | 27 | 12 | 14 | 3212 | 4270 | 4289 | 5 |
| 8 | 2026-09-10T20:13+00:00 → 2026-09-16T14:06+00:00 | 86 | 27 | 28 | 39 | 4162 | 5329 | 6857 | 35 |

## Verdict

- rule: median(8) vs median(16), passed non-speculative, n>=10/arm, same<=+5%, worse>=+15%
- median ratio 8/16: **1.296**
- **recommendation: keep merge leg at 16 (8 costs the serial bottleneck material wall-clock); amend task 3589 to inject an explicit -n 16 on the merge role**

Caveats: `merge_verify.duration_ms` is the whole merge verify (all modules, test+lint+type), not the
orchestrator test leg alone; concurrent load is not controlled for beyond the 2026-09-08 bounds
(task slots 1, offline lane/agent shells capped at 8 after the first redeploy). A restart re-reads the
same file, so arms survive restarts. Duration under the merge cold ceiling only; a timed-out verify
is a failed row here.
