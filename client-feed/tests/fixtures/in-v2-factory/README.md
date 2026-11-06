# in-v2-factory: the factory's own `in/inbox.jsonl`

`inbox.jsonl` is hand-written by the factory to `plans/client-feed-prd.md`
§14.3: a restated and a quarantined `defect_reported` line between two
`reveal_requested` lines, in the compact JSON the portal writes.

It stands in until σ3 vendors the portal's `in-v2` fixture, byte for byte,
beside it.

`test_in_v2_factory_fixture.py` validates each line and derives B23's
rejection cases by mutating copies of them. No test edits this file in place.
