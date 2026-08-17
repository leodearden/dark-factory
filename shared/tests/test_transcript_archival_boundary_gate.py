"""ε B+H boundary gate — shared-primitive half (task 2732).

The end-to-end transcript-archival boundary gate over the ALREADY-INTEGRATED
code paths landed by α (producer hook + archiver primitive), β (teardown
backstop), γ (legibility archive mining) and δ (retention GC). See
``plans/agent-transcript-archival-prd.md`` Appendix B for the matrix::

    E1  a completed session's transcript is archived at completion
    E2  the archive survives worktree teardown
    E3  the teardown backstop is idempotent w.r.t. the producer
    E4  the archive is credential-safe (only projects/**.jsonl is ever copied)
    E5  legibility mining enumerates the archived transcript
    E6  a resumed session re-archives its grown transcript (last-write-wins)
    E7  an archive failure is SOFT (task still succeeds) and LOUD (counted+logged)
    E8  the retention GC prunes by cap, loudly; default caps are a no-op

**This file owns E4.** Its two siblings own the rest:

* ``orchestrator/tests/test_transcript_archival_boundary_gate.py`` — E1, E6,
  E2, E3, E7 (the producer-hook and teardown-backstop rows).
* ``scripts/tests/test_transcript_archival_boundary_gate.py`` — E5, E8 (the
  legibility-mining and retention-GC rows).

The gate is three files rather than one because ``verify`` is directory-scoped:
each package's ``orchestrator.yaml`` declares its own ``test_command``, so a
single cross-package module would run in exactly one lane and a shared-only or
scripts-only diff would never exercise its rows. This file additionally imports
ONLY :mod:`shared.transcript_archive`, so the package dependency direction
holds and ``test_pure_stdlib_leaves.py``'s pure-stdlib-leaf guarantee for that
module is not disturbed.

ARCHIVE FORMAT: plain ``.jsonl``, byte-verbatim, NO added suffix. Task 3618
(leaf α of ``plans/transcript-preservation-seam-prd.md``) dropped gzip from the
archive AFTER the PRD was written, so Appendix B's "gz round-trips" wording for
E1/E5 is stale — do not read it as a gap. The residual-``.jsonl.gz`` contract
that survived 3618 (not enumerated, but counted + warned) is pinned by E5 in
the scripts file.

No row mocks the component under test: every row here drives the REAL
:func:`shared.transcript_archive.archive_task_transcripts`.

CREDENTIAL-EXCLUSION ORACLE: :func:`_relative_file_walk` is the single
recursive walk E4's "absent from the archive" claims are made through. It is
deliberately used TWICE per run — once over the archive root (where the
credential must be absent) and once over the SOURCE config dir (where it must
be PRESENT) — so the absence assertions are demonstrably checking a live walk
that finds credentials when they exist, rather than an empty one that would
vacuously satisfy every ``not in``.


Fixtures are kept module-local (no conftest.py additions), matching
``orchestrator/tests/test_transcript_archive_backstop.py``'s documented choice.
"""

from __future__ import annotations

from pathlib import Path

from shared.transcript_archive import archive_task_transcripts

# The encoded-project dir the fake transcripts are laid down under.
ENC = '-home-leo-projX'

TASK_ID = '2732'
SID = 'sess-A'

# An OAuth-token-shaped payload, distinctive enough that a substring scan of
# the whole archive tree is a meaningful "no credential leaked" oracle.
CREDENTIAL_PAYLOAD = (
    b'{"claudeAiOauth":{"accessToken":'
    b'"sk-ant-oat01-E4-BOUNDARY-GATE-CANARY-DO-NOT-LEAK","refreshToken":'
    b'"sk-ant-ort01-E4-BOUNDARY-GATE-CANARY-DO-NOT-LEAK"}}'
)
CREDENTIAL_CANARY = b'E4-BOUNDARY-GATE-CANARY-DO-NOT-LEAK'


def _config_dir(root: Path, task_id: str) -> Path:
    """The per-task Claude config dir shape the archiver is pointed at."""
    return root / '.task' / f'claude-config-{task_id}'


def _write_transcript(config_dir: Path, rel: str, data: bytes) -> Path:
    """Lay down a transcript at ``<config_dir>/projects/<rel>`` and return it."""
    p = config_dir / 'projects' / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def _relative_file_walk(root: Path) -> list[str]:
    """Every FILE under *root*, recursively, as sorted posix relative paths.

    The credential-exclusion oracle (see the module docstring): the one walk
    E4's presence AND absence claims are both made through, so the same code
    that reports "no ``.credentials.json`` under the archive root" is proven,
    in the same test, to report one under the source config dir.
    """
    return sorted(
        p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
    )


def _build_realistic_config_dir(root: Path) -> tuple[Path, dict[str, Path]]:
    """Build a per-task config dir with the real mix of files a live one holds.

    Returns ``(config_dir, sources)`` where ``sources`` maps the archive-relative
    path each transcript SHOULD land at to its source file. Everything else laid
    down here is a decoy that must never reach the archive: the per-task OAuth
    ``.credentials.json`` and a ``settings.json`` at the config-dir ROOT (both
    structurally outside ``projects/``), a ``shell-snapshots/`` dir beside them,
    and a non-``.jsonl`` ``notes.txt`` INSIDE ``projects/<enc>/`` — the one decoy
    the ``projects/``-rooted glob alone would not exclude, so it is the file that
    makes the ``*.jsonl`` half of the filter load-bearing.
    """
    config_dir = _config_dir(root, TASK_ID)
    (config_dir).mkdir(parents=True, exist_ok=True)

    (config_dir / '.credentials.json').write_bytes(CREDENTIAL_PAYLOAD)
    (config_dir / 'settings.json').write_bytes(b'{"decoy": true}\n')
    snapshots = config_dir / 'shell-snapshots'
    snapshots.mkdir()
    (snapshots / 'snapshot-bash-1.sh').write_bytes(b'# decoy\n')

    main = _write_transcript(
        config_dir, f'{ENC}/{SID}.jsonl', b'{"type":"user","cwd":"/tmp/x"}\n'
    )
    subagent = _write_transcript(
        config_dir,
        f'{ENC}/{SID}/subagents/agent-deadbeef.jsonl',
        b'{"type":"assistant","subagent":"deadbeef"}\n',
    )
    _write_transcript(config_dir, f'{ENC}/notes.txt', b'not a transcript\n')

    return config_dir, {
        f'{TASK_ID}/{ENC}/{SID}.jsonl': main,
        f'{TASK_ID}/{ENC}/{SID}/subagents/agent-deadbeef.jsonl': subagent,
    }


class TestE4CredentialSafe:
    """E4 — the archive is credential-safe by CONSTRUCTION.

    ``archive_task_transcripts`` only ever globs ``*.jsonl`` UNDER
    ``projects/``, so the per-task OAuth ``.credentials.json`` (which lives at
    the config-dir root) is structurally unreachable, as is any non-``.jsonl``
    file. This row drives the REAL helper in BOTH of its production call
    shapes against one realistic config dir and asserts the property by
    walking the WHOLE archive tree, not by checking two expected paths exist.
    """

    def test_only_projects_jsonl_reaches_the_archive(self, tmp_path: Path):
        config_dir, sources = _build_realistic_config_dir(tmp_path / 'wt')
        archive_root = tmp_path / 'archive'

        # Both production call shapes, into the SAME archive root: α's producer
        # hook passes the just-finished session id; β's teardown backstop passes
        # None to sweep whatever is still un-archived.
        archive_task_transcripts(
            config_dir, TASK_ID, SID, archive_root=archive_root
        )
        archive_task_transcripts(
            config_dir, TASK_ID, None, archive_root=archive_root
        )

        archived = _relative_file_walk(archive_root)

        # NEGATIVE CONTROL — the oracle is a LIVE walk, not an empty one. Run
        # the identical walk over the SOURCE config dir: it must find the
        # credential (and the other decoys) it reports absent from the archive.
        # Without this, every `not in` below would pass vacuously if the walk
        # were broken, mis-rooted, or the archive simply never written.
        source_walk = _relative_file_walk(config_dir)
        source_names = {Path(name).name for name in source_walk}
        assert '.credentials.json' in source_names
        assert 'settings.json' in source_names
        assert 'notes.txt' in source_names
        assert any(
            CREDENTIAL_CANARY in (config_dir / name).read_bytes()
            for name in source_walk
        )
        assert archived, 'archive root is empty — the absence assertions below would be vacuous'

        # The whole tree is exactly the two transcripts — nothing else, at any
        # depth, under any name.
        assert archived == sorted(sources)

        # Suffix invariant, stated independently of the set equality so a future
        # layout change cannot quietly admit a non-.jsonl artefact.
        assert all(name.endswith('.jsonl') for name in archived)

        # None of the decoys is present under ANY name anywhere in the tree.
        basenames = {Path(name).name for name in archived}
        assert '.credentials.json' not in basenames
        assert 'settings.json' not in basenames
        assert 'notes.txt' not in basenames
        assert 'snapshot-bash-1.sh' not in basenames

        # ...and the credential BYTES do not appear in any archived file either
        # (an inlined/renamed copy would pass the name checks above).
        for name in archived:
            assert CREDENTIAL_CANARY not in (archive_root / name).read_bytes()

        # Byte-verbatim plain .jsonl: the archived copy IS the source, with no
        # container to decompress (task 3618 dropped gzip).
        for rel, src in sources.items():
            assert (archive_root / rel).read_bytes() == src.read_bytes()
