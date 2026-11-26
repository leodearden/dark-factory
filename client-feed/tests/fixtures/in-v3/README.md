# in-v3: the portal's `in/inbox.jsonl` with defect rulings

`inbox.jsonl` is the client portal's `in/inbox.jsonl` for project
`demo-client` with defect rulings, nine lines in all: three `reveal_requested`
lines (seq 1, 6 and 9), and three block-2 `defect_reported` lines (seq 2, 4 and
5), each ruled later by a `defect_ruled` line (seq 3, 7 and 8) with
`not_a_defect`, `not_covered` and `change_request`. It is vendored verbatim
from scoping-intake `tests/fixtures/in-v3` at commit
`9fc89da7b7b87022027279768735328c50a87390`.

The portal's writers make this file. It is never edited by hand here: a change
on the portal's side is re-vendored, byte for byte, into this directory.

`client-feed/tests/test_inbox_defect_ruled.py` validates it (B37), and
`fused-memory/tests/test_client_defect_bridge_e2e.py` runs the filer over it
against the real fused-memory tool layer (B38).

No test reads `~/src/scoping-intake`.
