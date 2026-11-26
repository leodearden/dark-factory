# in-v2: the portal's `in/inbox.jsonl` with defect reports

`inbox.jsonl` is the client portal's `in/inbox.jsonl` for project
`demo-client` with defect reports: a restated and a quarantined
`defect_reported` line, seq 2 and 3, between the `reveal_requested` lines
seq 1 and 4. It is vendored verbatim from scoping-intake `tests/fixtures/in-v2`
at commit `53e69ba3a4063ccc0cec7222f702954f01fd28f0`.

The portal's writers make this file. It is never edited by hand here: a change
on the portal's side is re-vendored, byte for byte, into this directory.

`client-feed/tests/test_in_v2_fixtures.py` validates it (B23), and
`fused-memory/tests/test_client_defect_bridge_e2e.py` files it against the real
fused-memory tool layer (B36).

No test reads `~/src/scoping-intake`.
