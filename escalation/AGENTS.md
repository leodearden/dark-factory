# escalation — agent notes

## `datetime.UTC` vs `datetime.timezone.utc`

This package requires Python `>=3.11` (see `requires-python` in
`pyproject.toml`) and its ruff config selects the `UP` (pyupgrade) rule set,
which includes `UP017` — mandating `datetime.UTC` over `datetime.timezone.utc`
on 3.11+. Before touching either spelling, run `ruff check` and trust its
verdict; do not assert a "`<3.11` compat" rationale in a commit message
without first grepping `requires-python` in `pyproject.toml` to confirm it.
This flip-flopped 6 times on an unchecked premise before this note was added
(PRD `plans/task-status-authority-prd.md`, contract C10 / finding 10.5).
