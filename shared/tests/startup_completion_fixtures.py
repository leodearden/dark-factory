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
import sys
from pathlib import Path
from typing import TypedDict

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


class StartupCompletionRow(TypedDict, total=False):
    """One curated observation of a config dir at a point in startup."""

    id: str
    regime: str
    wedge_shape: str | None
    sample_offset_secs: float
    session_id: str
    config_dir_tree: list[dict]
    transcript_relpath: str | None
    transcript_records: list[dict] | None
    transcript_raw_lines: list[str]
    proc: dict
    expected_startup_complete: bool | None
    substrate_returns: dict
    provenance: dict
    source_path: str


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


def validate_row(row: dict) -> None:
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
