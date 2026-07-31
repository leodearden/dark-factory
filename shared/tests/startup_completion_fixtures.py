"""Startup-completion fixture loader — task 3324.

The committed, empirically-derived corpus describing what a Claude CLI
invocation's ``CLAUDE_CONFIG_DIR`` looks like at each stage of startup, for a
healthy run and for each PRD-named wedge shape.  Consumed by
`test_startup_completion_fixtures.py` today and by task 3326's watchdog
two-regime startup grace (contract C5) next.

WHY THIS MODULE EXISTS
----------------------
C5 needs a predicate answering *"has the CLI finished starting up, even though
turn 1 has not landed?"* so a genuinely-started invocation stuck in a server
retry cycle can be given a longer grace than a wedge that never reached session
init.  :func:`evaluate_startup_completion_predicate` is the REFERENCE
implementation of the chosen predicate, built exclusively on substrate that is
already public on main (``shared.cli_invoke.read_transcript_records`` /
``transcript_exists``).  3326 ports it into production and can diff against this.

Import it as a top-level module — `shared/tests/conftest.py` already puts this
directory on ``sys.path``::

    import startup_completion_fixtures as scf

    for row in scf.load_startup_completion_corpus():
        config_dir, session_id = scf.materialize_config_dir(row, tmp_path)
        assert my_production_predicate(config_dir, session_id) == (
            row['expected_startup_complete']
        )

See `docs/startup-completion-artifact-matrix.md` for the artifact matrix these
rows summarise, the named predicate, and the failure-mode table.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from collections.abc import Mapping
from typing import Any, NotRequired, TypedDict

from shared.cli_invoke import read_transcript_records

_TESTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TESTS_DIR.parent / 'src'
for _p in (str(_TESTS_DIR), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_FIXTURES_DIR = _TESTS_DIR / 'fixtures' / 'startup_completion'

HEALTHY_CORPUS_PATH = _FIXTURES_DIR / 'startup_completion_healthy.json'
WEDGE_CORPUS_PATH = _FIXTURES_DIR / 'startup_completion_wedge.json'
RAW_CAPTURE_PATH = _FIXTURES_DIR / 'startup_completion_probe_raw.jsonl'

#: Both curated corpus files, in load order.  Mirrors the ``_CORPUS_PATH``
#: pattern at `test_invocation_outcome.py`'s B3 cap-string corpus.
CORPUS_PATHS: tuple[Path, ...] = (HEALTHY_CORPUS_PATH, WEDGE_CORPUS_PATH)

REGIMES = frozenset({'healthy', 'wedge'})

#: The three PRD-named wedge shapes plus the ``transcript_unreadable`` degrade
#: case C5 must handle explicitly.
WEDGE_SHAPES = frozenset(
    {
        'from_source_build',
        'uv_resolving',
        'mcp_init_hang',
        'transcript_unreadable',
    }
)

_TREE_KINDS = frozenset({'file', 'dir', 'symlink', 'vanished'})

#: Filenames whose CONTENT must never appear in a committed row — presence and
#: size metadata only.  ``TaskConfigDir.write_credentials`` puts a live OAuth
#: access token in ``.credentials.json``, and the healthy observation is taken
#: from a config dir that really holds one.
CREDENTIAL_FILENAMES = frozenset({'.credentials.json'})


class StartupCompletionRow(TypedDict):
    """One curated observation of a config dir at a point in startup.

    Totality mirrors :data:`_REQUIRED_KEYS`, which :func:`validate_row` enforces
    at load time: every key below is present on a validated row, so a consumer
    (3326's tests included) can subscript it without a ``.get()`` dance.  The two
    genuinely-optional key is ``NotRequired``:

    - ``transcript_raw_lines`` — present only on the truncated/unparseable
      degrade rows, which express their transcript as literal lines rather than
      as parsed records.

    ``source_path`` is required because it describes a LOADED row:
    :func:`load_startup_completion_corpus` stamps it on every row it returns.
    """

    id: str
    regime: str
    wedge_shape: str | None
    sample_offset_secs: float
    session_id: str
    config_dir_tree: list[dict]
    transcript_relpath: str | None
    transcript_records: list[dict] | None
    proc: dict
    expected_startup_complete: bool | None
    substrate_returns: dict
    provenance: dict
    source_path: str
    transcript_raw_lines: NotRequired[list[str]]


# ---------------------------------------------------------------------------
# Secret hygiene
# ---------------------------------------------------------------------------

#: ``(name, regex)`` pairs matching credential-shaped material.  Kept in sync
#: with ``startup_completion_probe._CREDENTIAL_PATTERNS`` — the probe applies
#: them as a CAPTURE-time gate (so unredacted material never reaches disk) and
#: this module applies them as a COMMIT-time assertion (so a later hand-edit
#: cannot reintroduce what the probe would have refused to write).  Both halves
#: are needed: capture-time alone is not safe under maintenance, and
#: commit-time alone is not safe during a fresh probe run.
_CREDENTIAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ('sk-ant-token', r'sk-ant-'),
    ('oauth-blob', r'claudeAiOauth'),
    ('access-token', r'accessToken'),
    ('refresh-token', r'refreshToken'),
    ('bearer-jwt', r'Bearer\s+eyJ'),
    # Generic long base64url run — catches a raw token pasted without any of the
    # named markers above.  64 chars is comfortably longer than the base64-ish
    # substrings that appear in ordinary content (session UUIDs are 36 with
    # hyphens; the longest incidental run in the committed corpus is far below
    # this), so it adds coverage without firing on legitimate data.
    ('long-base64url-run', r'[A-Za-z0-9_-]{64,}'),
)


def assert_no_credential_material(text: str, *, source: str) -> None:
    """Raise ``AssertionError`` if *text* carries credential-shaped material.

    *source* names what is being scanned (a path, or ``synthetic:<label>``) and
    is echoed in the failure message together with the pattern name and the
    match offset, so a failure says WHERE to look rather than just "assertion
    failed".  The matched text itself is never echoed — a guard that printed the
    secret it caught would defeat its own purpose.
    """
    for name, pattern in _CREDENTIAL_PATTERNS:
        match = re.search(pattern, text)
        if match is not None:
            raise AssertionError(
                f'credential material in {source}: pattern {name!r} matched at '
                f'offset {match.start()} (match text withheld). Redact it — '
                f'record credential-bearing paths by presence/size only.'
            )


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

_REQUIRED_KEYS: tuple[str, ...] = (
    'id',
    'regime',
    'wedge_shape',
    'sample_offset_secs',
    'session_id',
    'config_dir_tree',
    'transcript_relpath',
    'transcript_records',
    'proc',
    'expected_startup_complete',
    'substrate_returns',
    'provenance',
)


def validate_row(row: Mapping[str, Any]) -> None:
    """Raise ``AssertionError`` if *row* violates the documented schema.

    The single schema gate — `fixtures/startup_completion/README.md` documents
    exactly these rules in prose.
    """
    row_id = row.get('id', '<no id>')

    for key in _REQUIRED_KEYS:
        assert key in row, f'{row_id}: missing required key {key!r}'

    assert isinstance(row['id'], str) and row['id'], f'{row_id}: id must be a non-empty str'
    assert row['regime'] in REGIMES, f'{row_id}: regime {row["regime"]!r} not in {sorted(REGIMES)}'

    # wedge_shape is None IFF the regime is healthy.
    if row['regime'] == 'healthy':
        assert row['wedge_shape'] is None, f'{row_id}: healthy rows must not carry a wedge_shape'
    else:
        assert row['wedge_shape'] in WEDGE_SHAPES, (
            f'{row_id}: wedge_shape {row["wedge_shape"]!r} not in {sorted(WEDGE_SHAPES)}'
        )

    assert isinstance(row['sample_offset_secs'], (int, float)), (
        f'{row_id}: sample_offset_secs must be numeric'
    )
    assert isinstance(row['session_id'], str) and row['session_id'], (
        f'{row_id}: session_id must be a non-empty str'
    )

    tree = row['config_dir_tree']
    assert isinstance(tree, list), f'{row_id}: config_dir_tree must be a list'
    for entry in tree:
        assert isinstance(entry, dict), f'{row_id}: tree entries must be dicts'
        assert isinstance(entry.get('relpath'), str), f'{row_id}: tree entry needs a str relpath'
        assert entry.get('kind') in _TREE_KINDS, (
            f'{row_id}: tree entry {entry.get("relpath")!r} has kind {entry.get("kind")!r}'
        )
        # Credential-bearing paths are recorded by presence/size ONLY.  A
        # committed row that inlined content here would put a live OAuth token
        # in git; see assert_no_credential_material's docstring.
        assert 'content' not in entry or Path(entry['relpath']).name not in CREDENTIAL_FILENAMES, (
            f'{row_id}: {entry["relpath"]!r} must not inline content'
        )

    relpath = row['transcript_relpath']
    assert relpath is None or isinstance(relpath, str), (
        f'{row_id}: transcript_relpath must be str|None'
    )

    records = row['transcript_records']
    assert records is None or isinstance(records, list), (
        f'{row_id}: transcript_records must be list|None'
    )
    if relpath is None:
        assert records is None, (
            f'{row_id}: an absent transcript cannot carry records — '
            f'read_transcript_records returns None when the file cannot be located'
        )

    if 'transcript_raw_lines' in row:
        assert isinstance(row['transcript_raw_lines'], list), (
            f'{row_id}: transcript_raw_lines must be a list of str'
        )
        assert relpath is not None, (
            f'{row_id}: transcript_raw_lines needs a transcript_relpath to write to'
        )

    assert isinstance(row['proc'], dict), f'{row_id}: proc must be a dict'

    verdict = row['expected_startup_complete']
    assert verdict is None or isinstance(verdict, bool), (
        f'{row_id}: expected_startup_complete must be bool|None, got {verdict!r}'
    )
    if relpath is not None:
        assert isinstance(verdict, bool), (
            f'{row_id}: a locatable transcript must yield a bool verdict, not the '
            f'unreadable sentinel None'
        )

    substrate = row['substrate_returns']
    assert isinstance(substrate, dict), f'{row_id}: substrate_returns must be a dict'
    for key in (
        'transcript_exists',
        'read_transcript_records_is_none',
        'record_count',
        'count_transcript_turns',
    ):
        assert key in substrate, f'{row_id}: substrate_returns missing {key!r}'

    provenance = row['provenance']
    assert isinstance(provenance, dict), f'{row_id}: provenance must be a dict'
    for key in ('probe_run_id', 'mode', 'cli_version', 'capture_method'):
        assert provenance.get(key), f'{row_id}: provenance missing {key!r}'


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_startup_completion_corpus() -> list[StartupCompletionRow]:
    """Load and return every curated row from BOTH corpus files, in file order.

    Each row is stamped with ``source_path`` (the corpus file's basename) so a
    caller can tell which file a row came from without re-reading either.
    """
    rows: list[StartupCompletionRow] = []
    for path in CORPUS_PATHS:
        payload = json.loads(path.read_text(encoding='utf-8'))
        for row in payload['rows']:
            row['source_path'] = path.name
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def snapshot_config_dir(
    config_dir: Path,
    *,
    epoch: float | None = None,
    prune_prefixes: tuple[str, ...] = (),
) -> list[dict]:
    """Return a sorted, content-free description of every entry under *config_dir*.

    THE single sampler: `startup_completion_probe.py` imports this to describe a
    live config dir, and :func:`materialize_config_dir`'s round trip is checked
    against it.  One function means probe output and materialized trees can never
    drift into describing different things.

    Each entry is ``{relpath, kind, size, mtime_delta_secs}``; ``kind`` is
    ``file`` / ``dir`` / ``symlink`` (or ``vanished`` for an entry that
    disappeared mid-walk, which is a real observation of a live dir, not an
    error).  Contents are NEVER inlined — ``.credentials.json`` is recorded by
    presence and size only.

    ``prune_prefixes`` collapses a subtree to a ``pruned_descendants`` count on
    the prefix entry.  It defaults to EMPTY here (a materialized tree is already
    small and must round-trip exactly); the probe passes its own default for
    live dirs, where the plugin marketplace git clone would otherwise dominate.
    """
    entries: list[dict] = []
    if not config_dir.exists():
        return entries
    pruned_counts: dict[str, int] = {}
    for path in sorted(config_dir.rglob('*')):
        try:
            relpath = str(path.relative_to(config_dir))
            prefix = next(
                (p for p in prune_prefixes if relpath == p or relpath.startswith(p + os.sep)),
                None,
            )
            if prefix is not None and relpath != prefix:
                pruned_counts[prefix] = pruned_counts.get(prefix, 0) + 1
                continue
            if path.is_symlink():
                kind, size = 'symlink', None
            elif path.is_dir():
                kind, size = 'dir', None
            else:
                kind, size = 'file', path.lstat().st_size
            mtime_delta = None
            if epoch is not None:
                mtime_delta = round(path.lstat().st_mtime - epoch, 3)
            entries.append(
                {
                    'relpath': relpath,
                    'kind': kind,
                    'size': size,
                    'mtime_delta_secs': mtime_delta,
                }
            )
        except OSError:
            entries.append(
                {
                    'relpath': str(path),
                    'kind': 'vanished',
                    'size': None,
                    'mtime_delta_secs': None,
                }
            )
    for entry in entries:
        if entry['relpath'] in pruned_counts:
            entry['pruned_descendants'] = pruned_counts[entry['relpath']]
    return entries


def materialize_config_dir(row: Mapping[str, Any], dest: Path) -> tuple[Path, str]:
    """Rebuild *row*'s observed config dir under *dest*; return ``(dir, session_id)``.

    *row* is any mapping carrying the observation keys — a curated
    :class:`StartupCompletionRow` from the corpus, or a raw observation object
    straight out of ``startup_completion_probe.py``'s JSONL.  Accepting both is
    what lets the live drift guard materialize a FRESH probe sample through the
    exact same path 3326's tests use for committed rows.

    The entry point 3326's tests call.  The rebuilt tree is a real filesystem, so
    a production predicate — and the real ``_run_subprocess`` watchdog — can be
    pointed at it and will behave as they would against the observed dir:

    - directories are created; plain files are written as a zero-or-placeholder
      payload of the recorded ``size`` (contents were never captured, and the
      predicate reads none of them);
    - a recorded ``symlink`` is created as a DANGLING symlink to a
      deliberately-absent target.  The production dir's ``settings.json`` points
      into the invoking user's ``~/.claude/``, which is neither reproducible nor
      relevant; what matters is that the entry is a symlink, which is what the
      recorded ``kind`` — and the round trip — assert;
    - the transcript is written at ``transcript_relpath`` from
      ``transcript_records`` (one JSON object per line), so
      ``_resolve_transcript_path``'s ``projects/*/<session_id>.jsonl`` glob
      resolves exactly as it does for a real config dir.  A row carrying
      ``transcript_raw_lines`` writes those literal lines instead — that is how
      the truncated/unparseable degrade row is expressed.
    """
    config_dir = Path(dest)
    config_dir.mkdir(parents=True, exist_ok=True)

    # Shortest-first so a parent dir always exists before its children.
    for entry in sorted(row['config_dir_tree'], key=lambda e: e['relpath'].count(os.sep)):
        if entry['kind'] == 'vanished':
            continue
        path = config_dir / entry['relpath']
        path.parent.mkdir(parents=True, exist_ok=True)
        if entry['kind'] == 'dir':
            path.mkdir(parents=True, exist_ok=True)
        elif entry['kind'] == 'symlink':
            if not path.is_symlink():
                path.symlink_to(config_dir / '__absent_symlink_target__')
        else:
            size = entry.get('size') or 0
            path.write_bytes(b'\0' * size)

    relpath = row['transcript_relpath']
    if relpath is not None:
        transcript = config_dir / relpath
        transcript.parent.mkdir(parents=True, exist_ok=True)
        if 'transcript_raw_lines' in row:
            body = '\n'.join(row['transcript_raw_lines'])
        else:
            body = '\n'.join(json.dumps(record) for record in (row['transcript_records'] or []))
        transcript.write_text(body + ('\n' if body else ''), encoding='utf-8')

    return (config_dir, row['session_id'])


# ---------------------------------------------------------------------------
# The chosen predicate — SESSION-TRANSCRIPT-MATERIALIZED
# ---------------------------------------------------------------------------

def evaluate_startup_completion_predicate(config_dir: Path, session_id: str) -> bool | None:
    """Has the CLI finished starting up, even though turn 1 has not landed?

    The REFERENCE implementation of the predicate named
    **SESSION-TRANSCRIPT-MATERIALIZED** in
    `docs/startup-completion-artifact-matrix.md`, which task 3326 ports into
    production for contract C5.  Built exclusively on substrate that is already
    public on main — ``shared.cli_invoke.read_transcript_records`` — so the
    discrimination is proven against today's code, and the port inherits that
    function's tolerant parsing and ``None``-on-unreadable semantics for free.

    Definition::

        records = read_transcript_records(config_dir, session_id)
        None  if records is None      # cannot locate/read — cannot prove
        True  if len(records) >= 1    # session init reached; prompt enqueued
        False otherwise               # file exists but carries no record yet

    Tri-state, mirroring the house convention that ``None`` means "unreadable,
    cannot prove either way".  ``count_transcript_turns`` already returns ``None``
    on an unlocatable transcript, and the existing startup kill deliberately
    fires only on an explicit ``live_turns == 0``, never on ``None``
    (cli_invoke.py:2111-2119).  C5's "predicate unreadable -> degrade to today's
    120s behaviour" needs that third state: a two-valued predicate would have to
    fold unreadable into True (extending the bound for a possible wedge) or
    False (killing a possibly-healthy server-retry cycle).

    WHY >= 1 RECORD, and not something narrower: the probe observed the leading
    record types to be ``queue-operation`` (prompt enqueue), ``queue-operation``,
    ``attachment`` (the SessionStart hook) and ``user`` — every one of them
    written BEFORE any ``assistant`` record.  Their presence proves the CLI
    reached session init and accepted the prompt.  A narrower rule keyed on a
    specific record type would pin this predicate to one CLI version's record
    vocabulary for no gain in discrimination; see the report's rejected
    alternatives.
    """
    records = read_transcript_records(config_dir, session_id)
    if records is None:
        return None
    return len(records) >= 1
