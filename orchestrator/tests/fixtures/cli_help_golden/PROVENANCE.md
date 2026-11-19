# Provenance — `cli_help_golden` fixtures

Captured from base `28787ba3bc`, rendered from a `git archive` of that
commit's `orchestrator/src` and `shared/src`, before task 6748 (eval CLI moved
into `orchestrator/evals/cli.py`, config preamble folded into one decorator)
edited them. It records every command's usage and options as they were, so
the claim that the move leaves command names and options unchanged is pinned
against a record and not against the code itself.

| File | Producing code site | Checked by |
|---|---|---|
| `help.json` | `orchestrator/tests/test_cli_help_golden.py::render_cli_options` applied to `orchestrator.cli.main`, dumped with `json.dumps(..., indent=2, sort_keys=True)` plus a trailing newline | `orchestrator/tests/test_cli_help_golden.py` |

Keys are space-joined command paths (`''` is the root group, then e.g.
`eval`, `eval-shadow report`, `verify-cache status`). Each value is
`{"usage": ..., "options": [[declaration, help], ...]}`, taken from click's
`Command.collect_usage_pieces` and `Parameter.get_help_record`, unwrapped.
Those are the calls click's `format_usage` and `format_options` render, so
the record is the Usage line after the command path plus the Options
section.

What it pins: the command set; each command's usage pieces; and, in
declaration order, each option's declaration, metavar, help text and the
`[default: ...]` / `[required]` extras click renders. What it deliberately
does not pin: command descriptions (docstrings) and a group's Commands
listing.

## Regenerating

From the checkout root:

    .venv/bin/python orchestrator/tests/test_cli_help_golden.py --write

(equivalently `uv run python orchestrator/tests/test_cli_help_golden.py --write`).
It prints the golden path and the `orchestrator.cli` file it rendered, so you
can see which tree it came from. To re-capture from an older base instead of
the working tree:

    BASE=<sha>
    D=/tmp/cli-help-golden-base; rm -rf "$D"; mkdir -p "$D"
    git archive "$BASE" orchestrator/src shared/src | tar -x -C "$D"
    PYTHONPATH="$D/orchestrator/src:$D/shared/src" .venv/bin/python orchestrator/tests/test_cli_help_golden.py --write

`PYTHONPATH` outranks the venv's editable install, so check that the printed
`orchestrator.cli` file is under `$D`.

## Policy

If you change a command's usage or options on purpose, regenerate this file
in the same commit and say so in the commit message. A diff nobody can
explain is a regression: fix the code, not the golden.
