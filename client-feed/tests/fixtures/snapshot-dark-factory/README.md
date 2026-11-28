# snapshot-dark-factory

`feed.jsonl` and `state.json` here are one `client-feed snapshot` of this repository.

How they were made:
- The snapshot ran on a shared, unchecked-out clone whose `main` is the generating commit (the heartbeat's `head`).
- Its stores are the read-only live slice of `plans/client-feed-prd.md`'s master tasks: their dependencies,
  events, task results, verify archives and open escalations
  (`client-feed/tests/_live_slice.py::copy_live_slice`).
- The policy is project `dark-factory` with `visible_globs` `*/src/**` and `*/tests/**`, and `max_cases_per_run` 20.
- The inbox asked for `client-feed/src/client_feed/envelope.py` and `../x`.

Consumer: the client portal's κ (scoping-intake task 56, which waits on `dark_factory:6973`) vendors these files as
its real-data fixture.

The file set is fixed: `feed.jsonl` and `state.json`, beside this README.

Regenerate them with the one command below, run from the repository root; it needs the main checkout's live task store:

```
cd client-feed && CLIENT_FEED_REGENERATE_SNAPSHOT_FIXTURE=1 uv run pytest tests/test_isolation.py -k b22
```

The files are written only after every B22 assertion has passed. Never hand-edit them.
`client-feed/tests/test_isolation.py::test_b22_a_snapshot_of_this_checkout_over_its_live_slice_is_valid_and_leaks_nothing`
checks each fresh snapshot, and
`client-feed/tests/test_isolation.py::test_the_committed_dark_factory_snapshot_is_a_valid_bounded_leak_free_feed`
checks the committed files.
