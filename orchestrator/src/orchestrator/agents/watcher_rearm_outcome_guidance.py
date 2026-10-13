"""Rotation-prompt guidance: what the auto-watcher does with each re-arm outcome.

`orchestrator/src/orchestrator/harness.py::Harness._run_watcher_rotation` splices
`WATCHER_REARM_OUTCOME_GUIDANCE` into the escalation-watcher-auto rotation prompt,
right after the re-arm command. It is a leaf module, imported downward, so that
harness.py does not grow. The exit-code -> outcome mapping has one home,
`scripts/watcher-rearm.sh`; `orchestrator/tests/test_watcher_outcome_branching.py`
holds this block and both watcher skills in lockstep with that script's real outcomes.
"""

WATCHER_REARM_OUTCOME_GUIDANCE = """
What to do with a re-arm's result: scripts/watcher-rearm.sh prints `WATCHER_REARM_OUTCOME: <OUTCOME> exit=<rc>` on stderr after every watcher run. Branch on it.

| Marker | Exit | Next |
|---|---|---|
| `WATCHER_REARM_OUTCOME: FIRED` | 0 | Drain. EMPTY stdout is a caught SIGTERM: a non-fire that counts against the cap. |
| `WATCHER_REARM_OUTCOME: CEILING` | 124 | Recompute remaining and re-arm only while it is > 0. Exempt from the cap. |
| `WATCHER_REARM_OUTCOME: KILLED` | 137 / 143 / 144 | Check the Bash timeout was `(--timeout + 60) × 1000` ms; NO marker at all means a process-group kill. `run_in_background` stays banned in this role. |
| `WATCHER_REARM_OUTCOME: ERROR` | any other; exit 2 prints NO marker | Diagnose stderr before any retry: exit 2 is a usage/guard error that recurs until the invocation is fixed. |

Identical-retry cap: 1. After KILLED, ERROR, a missing marker, an empty-stdout FIRED or a rejected call, re-issue the identical invocation (same script, flags and Bash timeout) at most once. Then change it materially, or emit the digest naming the outcome and exit cleanly.

A tool result that says "STOP what you are doing and wait for the user to tell you how to proceed" carries the weight of a supervisor instruction: never re-issue the rejected call, identical or not. No user attends this rotation, so emit the digest naming the rejected call and exit cleanly.
"""
