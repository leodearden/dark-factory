# FalkorDB recovery dumps

A small set of FalkorDB `.rdb` dumps, and the manifests and reports beside
them, are kept on purpose. They are archives, not backups of the live graph.

## The ruling

Leo's Q2 ruling, 2026-10-19, at the solar_challenge L2 sitting (task 7153):
the graph losses from the pre-6616 cross-graph moves and the pre-4986 merges
were accepted without a backfill, on condition that the pre-move dumps stay
available. They hold the only copies of the lost `expired_at` and `MENTIONS`
values, so a backfill stays possible if a reader of either is ever adopted.

## Retention

Never delete, prune or "clean up" these files or their directories as stale
backups, whatever the disk pressure. Retiring one is Leo's decision, and it is
recorded by removing it from `scripts/recovery_archive.py::RECOVERY_ARCHIVES`.

That constant is the list. Each entry names an archive directory, and for each
dump in it the pinned size, sha256 and what it is the pre-state of. Read it
there; this page does not repeat it.

## How they travel

Both data trees are gitignored, so the dumps never reach a fresh clone. The
migration scripts carry them instead:

- `scripts/export-data.sh` section 1b copies every archive directory, siblings
  included, to `<export>/recovery-archive/<repo path>`. It runs before
  fused-memory is stopped, and a missing or altered dump fails the export.
- `scripts/import-data.sh` section 2b copies them back to the same repo paths
  before any live data is restored, and fails the import on the same terms.
- `scripts/verify-migration.sh` section 4b checks every pinned dump's size and
  sha256, and fails verification if one is missing or altered.

They are never copied into `fused-memory/data/falkordb`: they are archives, not
a graph to load, and `ArchiveDir` refuses any path at or under that directory.

## Checking them by hand

```bash
python3 scripts/recovery_archive.py check <repo root>
```

It prints one `ok` line per intact dump, naming what it is the pre-state of,
and one `FAILED` line per problem. It exits 1 if any dump is missing, the
wrong size or the wrong sha256. To restore them from an export, run
`python3 scripts/recovery_archive.py copy <export>/recovery-archive <repo root>`.

## Using one for a backfill

Load the dump into the scratch FalkorDB
([OPERATIONS.md §"The scratch FalkorDB"](../OPERATIONS.md#the-scratch-falkordb)),
never into production.
