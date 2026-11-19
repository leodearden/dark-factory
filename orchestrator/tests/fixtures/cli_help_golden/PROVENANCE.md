# Provenance — `cli_help_golden` fixtures

Captured from base `51e3715697`, before task 6748 (eval CLI moved into
`orchestrator/evals/cli.py`, config preamble folded into one decorator)
touched `orchestrator/src/orchestrator/cli.py`. It records every command's
`--help` as it was, so the claim that the move leaves command names and
options byte-identical is pinned against a record and not against the code
itself.

| File | Producing code site | Checked by |
|---|---|---|
| `help.json` | `orchestrator/tests/test_cli_help_golden.py::render_cli_help` applied to `orchestrator.cli.main`, dumped with `json.dumps(..., indent=2, sort_keys=True)` plus a trailing newline | `orchestrator/tests/test_cli_help_golden.py` |

Keys are space-joined command paths (`''` is the root group, then e.g.
`eval`, `eval-shadow report`, `verify-cache status`); values are the
`--help` stdout rendered by click's `CliRunner` at `terminal_width=100`, so
wrapping does not depend on the terminal the test runs in.

Never regenerate this file to turn a red test green. A diff means an
operator-visible command, option or help text changed, and that change must
be deliberate.
