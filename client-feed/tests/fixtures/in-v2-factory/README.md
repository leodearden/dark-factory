# in-v2-factory: the factory's own `in/inbox.jsonl`

`inbox.jsonl` is hand-written by the factory to `plans/client-feed-prd.md`
§14.3: a restated and a quarantined `defect_reported` line between two
`reveal_requested` lines, in the compact JSON the portal writes.

The portal's vendored writer output now sits beside it, in `../in-v2`, and the
same tests run over both.

`test_in_v2_fixtures.py` validates each line and derives B23's rejection cases
by mutating copies of them. No test edits this file in place.
