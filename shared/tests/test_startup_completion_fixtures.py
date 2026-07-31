"""Tests for the startup-completion fixture corpus (task 3324).

The corpus is the committed, empirically-derived evidence for the two-regime
watchdog startup grace (PRD `plans/server-side-api-error-handling-prd.md`,
consumer task 3326 / contract C5).  Every assertion here is a runtime or
data-contract assertion — schema validity, secret hygiene, materialization
round-trip, predicate verdicts, wedge-shape coverage.  Nothing asserts on
documentation prose, and nothing asserts a wall-clock threshold: the observed
sample offsets are provenance, not a bound anyone can guarantee.

See `docs/startup-completion-artifact-matrix.md` for the matrix these rows
summarise and for the named predicate they pin.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import startup_completion_fixtures as scf

# The closed sets the schema validates against, restated here so a silent
# widening of the module's own constants cannot pass unnoticed.
_REGIMES = {'healthy', 'wedge'}
_WEDGE_SHAPES = {
    'from_source_build',
    'uv_resolving',
    'mcp_init_hang',
    'transcript_unreadable',
}

_REQUIRED_KEYS = (
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
    'provenance',
)


class TestCorpusLoads:
    """The loader resolves both corpus files and every row validates."""

    def test_corpus_is_non_empty(self):
        rows = scf.load_startup_completion_corpus()
        assert isinstance(rows, list)
        assert rows, 'startup-completion corpus is empty'

    def test_corpus_paths_exist_and_are_both_drawn_from(self):
        assert scf.HEALTHY_CORPUS_PATH.exists(), scf.HEALTHY_CORPUS_PATH
        assert scf.WEDGE_CORPUS_PATH.exists(), scf.WEDGE_CORPUS_PATH
        assert set(scf.CORPUS_PATHS) == {scf.HEALTHY_CORPUS_PATH, scf.WEDGE_CORPUS_PATH}

        rows = scf.load_startup_completion_corpus()
        sources = {row['source_path'] for row in rows}
        assert sources == {
            str(scf.HEALTHY_CORPUS_PATH.name),
            str(scf.WEDGE_CORPUS_PATH.name),
        }, f'corpus rows must be drawn from BOTH files, got {sources}'

    def test_both_regimes_are_represented(self):
        rows = scf.load_startup_completion_corpus()
        assert {row['regime'] for row in rows} == _REGIMES

    def test_every_row_has_the_required_keys(self):
        for row in scf.load_startup_completion_corpus():
            missing = [key for key in _REQUIRED_KEYS if key not in row]
            assert not missing, f'row {row.get("id")!r} missing keys {missing}'

    def test_every_row_validates(self):
        # validate_row is the single documented schema gate; a row that the
        # loader accepts but validate_row rejects would let a malformed row
        # reach 3326's tests.
        for row in scf.load_startup_completion_corpus():
            scf.validate_row(row)

    def test_regime_and_wedge_shape_are_a_closed_set(self):
        for row in scf.load_startup_completion_corpus():
            assert row['regime'] in _REGIMES, row['id']
            if row['regime'] == 'healthy':
                assert row['wedge_shape'] is None, (
                    f'{row["id"]}: healthy rows carry no wedge_shape'
                )
            else:
                assert row['wedge_shape'] in _WEDGE_SHAPES, (
                    f'{row["id"]}: unknown wedge_shape {row["wedge_shape"]!r}'
                )

    def test_expected_startup_complete_is_tristate(self):
        # bool | None, mirroring the predicate's own return type: None is the
        # "artifacts unreadable — cannot prove either way" sentinel that drives
        # C5's conservative degrade, and is legal ONLY when the row's transcript
        # genuinely cannot be read.
        for row in scf.load_startup_completion_corpus():
            value = row['expected_startup_complete']
            assert value is None or isinstance(value, bool), (
                f'{row["id"]}: expected_startup_complete must be bool|None, got {value!r}'
            )
            if row['transcript_relpath'] is not None:
                assert isinstance(value, bool), (
                    f'{row["id"]}: a readable transcript must yield a bool verdict, '
                    f'not the unreadable sentinel'
                )

    def test_row_ids_are_unique_across_both_files(self):
        ids = [row['id'] for row in scf.load_startup_completion_corpus()]
        assert len(ids) == len(set(ids)), f'duplicate row ids: {sorted(ids)}'


class TestCorpusSecretHygiene:
    """The committed artifacts must not carry credential material.

    Load-bearing, not theatre: the healthy observation is captured from a real
    ``CLAUDE_CONFIG_DIR`` whose ``.credentials.json`` holds a live OAuth access
    token (``TaskConfigDir.write_credentials``).  The probe redacts at capture
    time; this asserts over what is actually committed, so a later hand-edit
    cannot reintroduce a token.
    """

    # Synthetic, obviously-fake stand-ins for each real credential shape.
    _CREDENTIAL_SHAPED = (
        ('anthropic-key', 'sk-ant-oat01-FAKEFAKEFAKE'),
        (
            'oauth-blob',
            '{"claudeAiOauth": {"accessToken": "FAKE-not-a-real-token"}}',
        ),
        ('bearer-jwt', 'Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.FAKE'),
        ('refresh-token', '{"refreshToken": "FAKE-refresh"}'),
    )

    @pytest.mark.parametrize(
        ('label', 'payload'),
        _CREDENTIAL_SHAPED,
        ids=[label for label, _ in _CREDENTIAL_SHAPED],
    )
    def test_guard_raises_on_credential_shaped_input(self, label, payload):
        with pytest.raises(AssertionError) as excinfo:
            scf.assert_no_credential_material(payload, source=f'synthetic:{label}')
        # The failure must be legible — naming the source and where it matched,
        # not just "assertion failed".
        assert f'synthetic:{label}' in str(excinfo.value)

    def test_guard_passes_on_clean_text(self):
        scf.assert_no_credential_material(
            'projects/-home-leo-src-dark-factory/<uuid>.jsonl', source='synthetic:clean'
        )

    @pytest.mark.parametrize(
        'path_attr',
        ['HEALTHY_CORPUS_PATH', 'WEDGE_CORPUS_PATH', 'RAW_CAPTURE_PATH'],
    )
    def test_committed_artifacts_are_clean(self, path_attr):
        path: Path = getattr(scf, path_attr)
        scf.assert_no_credential_material(
            path.read_text(encoding='utf-8'), source=str(path)
        )

    def test_credential_files_are_recorded_by_metadata_only(self):
        for row in scf.load_startup_completion_corpus():
            for entry in row['config_dir_tree']:
                if Path(entry['relpath']).name in scf.CREDENTIAL_FILENAMES:
                    assert 'content' not in entry, (
                        f'{row["id"]}: {entry["relpath"]} must be recorded by '
                        f'presence/size only, never inlined content'
                    )
                    # Presence/size metadata is exactly what the matrix needs.
                    assert entry['kind'] == 'file'
                    assert isinstance(entry['size'], int)

    def test_at_least_one_row_actually_observed_a_credentials_file(self):
        # Guards the guard: if no committed row carried .credentials.json at all,
        # the metadata-only assertion above would pass vacuously.
        observed = [
            row['id']
            for row in scf.load_startup_completion_corpus()
            for entry in row['config_dir_tree']
            if Path(entry['relpath']).name in scf.CREDENTIAL_FILENAMES
        ]
        assert observed, 'no committed row observed a .credentials.json entry'
