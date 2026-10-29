# in-v1: the portal's `in/` files

`engagement.json` and `inbox.jsonl` are the client portal's `in/` files for
project `demo-client`, vendored verbatim from scoping-intake
`tests/fixtures/in-v1` at commit `f5cea40d8354190c9fa0cce260b2e69adc79ea3a`.

The portal's writers make these files. They are never edited by hand here: a
change on the portal's side is re-vendored, byte for byte, into this directory.

Their `delivered_commit` (`9b1e4c7a…`) is a fixture sha that no test repository
holds. A test that needs a usable engagement replaces it with `null` after
reading the vendored file.

No test reads `~/src/scoping-intake`.
