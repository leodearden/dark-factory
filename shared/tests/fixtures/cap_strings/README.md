# Cap-string golden corpus

`corpus.json` is the checked-in golden corpus for
`shared/tests/test_invocation_outcome.py`'s B3 test, which parametrizes over
every record and asserts `classify_invocation` produces the recorded
`expected` `InvocationOutcome` variant.

## Record schema

```
{
  "id": "<unique slug>",
  "backend": "claude" | "codex" | "gemini",
  "stderr": "<optional, default ''>",
  "output": "<optional, default ''>",
  "api_error_status": <optional int, default null>,
  "timed_out": <optional bool, default false>,
  "transcript_turns": <optional int|null, default null>,
  "turns": <optional int, default 0>,
  "cost_usd": <optional float, default 0.0>,
  "success": <optional bool, default false>,
  "strict_confirm": <bool, required>,
  "expected": "OK" | "CapHit" | "NearCap" | "AuthFailed" | "CliLocalError" | "ZeroOutputWedge" | "Failure",
  "resets_at": "set" | "none" (optional; only meaningful when expected == "CapHit"),
  "provenance": "<fix-commit sha(s) and/or source test, free text>"
}
```

Only `strict_confirm`, `expected`, `provenance`, and `id` are required. All
`AgentResult`-relevant inputs are optional and default the way
`shared.cli_invoke.AgentResult` itself defaults (`success=False`,
`output=''`, `stderr=''`, `turns=0`, `cost_usd=0.0`, `timed_out=False`,
`transcript_turns=None`, `api_error_status=None`).

## Provenance

Rows are transcribed from (or documented as motivated by) these fix commits
and pre-existing exhaustive fixtures:

- `ba38ce4ee1` — reclassify "You're now using extra" as CAP
- `b88b4625d5` — "extra usage" prefixes
- `e3df395c9f`, `77d1d18c49`, `66daedbc76` — narrow bare `upgrade` to
  `upgrade your plan` / `upgrade your subscription`
- `b5f6b04ac1` — auth_failed → capped demote on cap-prefix re-probe
- `1e8a9b2dd0` — "extra usage" + `is_error` handling
- `7d1fa90075` — reify-3604: don't treat local CLI errors (e.g. `--session-id`
  collision) as usage caps
- `shared/tests/test_usage_gate_exhaustive.py` (`TestCapDetectionPatterns`) —
  verbatim realistic Claude CLI cap/near-cap strings and the bare-upgrade
  negative case
- `shared/src/shared/cli_invoke.py:357` (`is_zero_output_timeout`) and
  `:799-805` (narrow `{401, 403}` AuthFailed routing, 429 exclusion) — the
  field-driven (non-string) rows

## Appending new rows

When Claude, Codex, or Gemini change their CLI/API wording, append a new
record here rather than editing `test_invocation_outcome.py` — the B3 test
loads every record in this file automatically. Give the new row a
descriptive `id`, set `provenance` to the commit or incident that surfaced
the new wording, and pick the narrowest correct `expected` variant. Do not
delete superseded rows unless the old wording is confirmed gone from every
supported CLI version — keeping them guards against regressions.
