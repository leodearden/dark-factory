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

import copy
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import startup_completion_fixtures as scf

from shared import cli_invoke

# Live re-probe gate — mirrors test_cli_invoke_integration.py's _AVAILABLE_TOKENS
# discovery, and startup_completion_probe._oauth_token's wider A..G sweep, so a
# machine with no accounts records a legible skip instead of a spurious failure.
_OAUTH_TOKEN_PRESENT = any(
    os.environ.get(f'CLAUDE_OAUTH_TOKEN_{c}') for c in 'ABCDEFG'
)

# ONE pinned copy per concept, asserted equal to the module's own constant by
# TestPinnedConstants.  Restating a closed set here is deliberate — a silent
# widening of scf.REGIMES / scf.WEDGE_SHAPES / scf._REQUIRED_KEYS would
# otherwise pass unnoticed — but three near-copies is worse than none, so every
# other use in this file reads from these names and drift is a test failure
# rather than a silent divergence.  (The copy this replaces had already lost
# 'substrate_returns', a key several tests here subscript directly.)
_REGIMES = frozenset({'healthy', 'wedge'})

_WEDGE_SHAPES = frozenset(
    {
        'from_source_build',
        'uv_resolving',
        'mcp_init_hang',
        'transcript_unreadable',
    }
)

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
    'substrate_returns',
    'provenance',
)

#: Sub-keys validate_row requires inside the two nested dicts.
_REQUIRED_SUBSTRATE_KEYS = (
    'transcript_exists',
    'read_transcript_records_is_none',
    'record_count',
    'count_transcript_turns',
)
_REQUIRED_PROVENANCE_KEYS = ('probe_run_id', 'mode', 'cli_version', 'capture_method')


class TestPinnedConstants:
    """The restated constants above must equal the module's own.

    This is the whole anti-drift property, in one assertion per concept, instead
    of hand-maintained near-copies scattered through the file.
    """

    def test_regimes_match(self):
        assert scf.REGIMES == _REGIMES

    def test_wedge_shapes_match(self):
        assert scf.WEDGE_SHAPES == _WEDGE_SHAPES

    def test_required_keys_match(self):
        assert scf._REQUIRED_KEYS == _REQUIRED_KEYS


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


def _base_row() -> dict[str, Any]:
    """A known-good, fully-populated row to mutate: healthy, transcript present.

    Deep-copied, so a mutation cannot leak into another test's view of the
    corpus.  Widened from :class:`StartupCompletionRow` to a plain dict on the
    way out — the point of every case below is to build a row the TypedDict
    forbids.
    """
    rows = [
        row
        for row in scf.load_startup_completion_corpus()
        if row['regime'] == 'healthy' and row['transcript_relpath'] is not None
    ]
    assert rows, 'no healthy row with a transcript to build negative cases from'
    return copy.deepcopy(dict(rows[0]))


def _with(row: dict[str, Any], **overrides) -> dict[str, Any]:
    row.update(overrides)
    return row


def _inline_credential_content(row: dict[str, Any]) -> dict[str, Any]:
    """Violate the "credential-bearing paths by presence/size only" rule."""
    for entry in row['config_dir_tree']:
        if Path(entry['relpath']).name in scf.CREDENTIAL_FILENAMES:
            entry['content'] = 'not-a-real-token'
            return row
    row['config_dir_tree'].append(
        {
            'relpath': '.credentials.json',
            'kind': 'file',
            'size': 16,
            'content': 'not-a-real-token',
        }
    )
    return row


#: One negative case per rule ``validate_row`` enforces.  Each ``mutate`` takes a
#: known-good (already deep-copied) row and returns it violating exactly that
#: rule.  Missing-key rules are parametrized separately, off the module's own
#: key tuples, so a new required key is covered the moment it is added.
_REJECTION_CASES: tuple[tuple[str, Callable[[dict], dict]], ...] = (
    ('id-not-a-str', lambda r: _with(r, id=123)),
    ('id-empty', lambda r: _with(r, id='')),
    ('regime-unknown', lambda r: _with(r, regime='bogus')),
    ('healthy-row-carries-a-wedge-shape', lambda r: _with(r, wedge_shape='mcp_init_hang')),
    ('wedge-shape-unknown', lambda r: _with(r, regime='wedge', wedge_shape='bogus')),
    ('wedge-row-without-a-shape', lambda r: _with(r, regime='wedge', wedge_shape=None)),
    ('sample-offset-not-numeric', lambda r: _with(r, sample_offset_secs='5.0')),
    ('session-id-empty', lambda r: _with(r, session_id='')),
    ('session-id-not-a-str', lambda r: _with(r, session_id=None)),
    ('tree-not-a-list', lambda r: _with(r, config_dir_tree={'relpath': 'projects'})),
    ('tree-entry-not-a-dict', lambda r: _with(r, config_dir_tree=['projects'])),
    (
        'tree-entry-relpath-not-a-str',
        lambda r: _with(r, config_dir_tree=[{'relpath': 7, 'kind': 'file'}]),
    ),
    (
        'tree-entry-kind-unknown',
        lambda r: _with(r, config_dir_tree=[{'relpath': 'projects', 'kind': 'socket'}]),
    ),
    ('credential-file-inlines-content', _inline_credential_content),
    ('transcript-relpath-not-a-str', lambda r: _with(r, transcript_relpath=7)),
    ('transcript-records-not-a-list', lambda r: _with(r, transcript_records={})),
    ('absent-transcript-carries-records', lambda r: _with(r, transcript_relpath=None)),
    ('raw-lines-not-a-list', lambda r: _with(r, transcript_raw_lines='one line')),
    (
        'raw-lines-element-not-a-str',
        lambda r: _with(r, transcript_raw_lines=[{'type': 'user'}]),
    ),
    (
        'raw-lines-without-a-relpath',
        lambda r: _with(
            r,
            transcript_relpath=None,
            transcript_records=None,
            transcript_raw_lines=['{"type": "user"}'],
        ),
    ),
    ('proc-not-a-dict', lambda r: _with(r, proc=[])),
    ('verdict-not-tristate', lambda r: _with(r, expected_startup_complete='true')),
    (
        'locatable-transcript-with-the-none-sentinel',
        lambda r: _with(r, expected_startup_complete=None),
    ),
    ('substrate-returns-not-a-dict', lambda r: _with(r, substrate_returns=[])),
    ('provenance-not-a-dict', lambda r: _with(r, provenance=[])),
    (
        'provenance-key-empty',
        lambda r: _with(r, provenance={**r['provenance'], 'cli_version': ''}),
    ),
)


class TestValidateRowRejects:
    """Every schema rule must actually FIRE.

    ``validate_row`` is exported to task 3326 as a reusable gate and is run by
    the loader over every row — but every positive test feeds it rows that
    already pass.  A dead, inverted or typo'd assertion (the compound
    ``'content' not in entry or ... not in CREDENTIAL_FILENAMES``, say, or the
    ``relpath is None -> records is None`` rule) would silently never fire, and
    3326 would append a malformed row that the "single gate" waved through.
    ``assert_no_credential_material`` already carries parametrized negative
    tests below; this applies the same pattern to the larger validator.
    """

    def test_the_unmutated_base_row_validates(self):
        # Guards the guard: if the base row did not pass, every mutation below
        # would "fail validation" for the wrong reason and prove nothing.
        scf.validate_row(_base_row())

    @pytest.mark.parametrize(
        ('rule', 'mutate'),
        _REJECTION_CASES,
        ids=[rule for rule, _ in _REJECTION_CASES],
    )
    def test_rule_rejects(self, rule, mutate):
        row = mutate(_base_row())
        with pytest.raises(AssertionError) as excinfo:
            scf.validate_row(row)
        # Every rule carries a message: a bare `assert x` would raise with an
        # empty string, which is the failure mode this file's own docstrings
        # argue against.
        assert str(excinfo.value), f'{rule}: validate_row rejected without a message'

    @pytest.mark.parametrize('key', _REQUIRED_KEYS)
    def test_missing_required_key_is_rejected(self, key):
        row = _base_row()
        del row[key]
        with pytest.raises(AssertionError) as excinfo:
            scf.validate_row(row)
        assert repr(key) in str(excinfo.value), (
            f'the failure for a missing {key!r} must name the key'
        )

    @pytest.mark.parametrize('key', _REQUIRED_SUBSTRATE_KEYS)
    def test_missing_substrate_key_is_rejected(self, key):
        row = _base_row()
        del row['substrate_returns'][key]
        with pytest.raises(AssertionError) as excinfo:
            scf.validate_row(row)
        assert repr(key) in str(excinfo.value)

    @pytest.mark.parametrize('key', _REQUIRED_PROVENANCE_KEYS)
    def test_missing_provenance_key_is_rejected(self, key):
        row = _base_row()
        del row['provenance'][key]
        with pytest.raises(AssertionError) as excinfo:
            scf.validate_row(row)
        assert repr(key) in str(excinfo.value)

    def test_the_loader_applies_the_gate(self, monkeypatch, tmp_path):
        # The point of the gate living IN the loader: 3326 calls
        # load_startup_completion_corpus(), not validate_row(), so a schema check
        # that only ran in this repo's suite would not cover the consumer's
        # branch at all.
        bad = _with(_base_row(), regime='bogus')
        path = tmp_path / 'startup_completion_bogus.json'
        path.write_text(json.dumps({'rows': [bad]}), encoding='utf-8')
        monkeypatch.setattr(scf, 'CORPUS_PATHS', (path,))

        with pytest.raises(AssertionError) as excinfo:
            scf.load_startup_completion_corpus()
        assert str(bad['id']) in str(excinfo.value), (
            'a load-time rejection must name the offending row'
        )

        # ...and the documented escape hatch really does skip it.
        assert len(scf.load_startup_completion_corpus(validate=False)) == 1


class TestSnapshotSampler:
    """Direct coverage for `snapshot_config_dir` branches the round trip skips."""

    def test_vanished_entry_records_a_relative_relpath(self, tmp_path, monkeypatch):
        # `vanished` is documented as a real observation of a live dir, so it IS
        # expected on a future probe run — but materialize_config_dir and the
        # round-trip test both skip it, so nothing else can catch a wrong
        # relpath here.  An absolute one would break the tree-entry contract and
        # leak the capturing host's username/worktree layout into a committed row.
        config_dir = tmp_path / 'cfg'
        (config_dir / 'projects').mkdir(parents=True)
        (config_dir / 'projects' / 'gone.jsonl').write_text('{}\n', encoding='utf-8')

        real_is_symlink = Path.is_symlink

        def flaky_is_symlink(self):
            if self.name == 'gone.jsonl':
                raise OSError(2, 'vanished mid-walk')
            return real_is_symlink(self)

        monkeypatch.setattr(Path, 'is_symlink', flaky_is_symlink)
        entries = scf.snapshot_config_dir(config_dir)

        vanished = [entry for entry in entries if entry['kind'] == 'vanished']
        assert vanished, 'the forced OSError did not produce a vanished entry'
        assert vanished[0]['relpath'] == os.path.join('projects', 'gone.jsonl')
        assert not Path(vanished[0]['relpath']).is_absolute()
        assert str(tmp_path) not in json.dumps(entries), (
            'a vanished entry must not carry the capturing host absolute path'
        )


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


def _corpus_rows():
    return scf.load_startup_completion_corpus()


def _row_ids():
    return [row['id'] for row in _corpus_rows()]


class TestMaterialization:
    """Rows rebuild into a real filesystem 3326's predicate can be pointed at.

    This is the exact entry point 3326's tests call, so it is asserted HERE
    rather than left to the consumer.  A corpus of recorded verdicts alone would
    let a downstream test pass against a predicate that never touches the
    filesystem — defeating the point of validating substrate.
    """

    @pytest.mark.parametrize('row', _corpus_rows(), ids=_row_ids())
    def test_every_tree_entry_is_materialized_with_its_kind(self, row, tmp_path):
        config_dir, session_id = scf.materialize_config_dir(row, tmp_path)

        assert isinstance(config_dir, Path)
        assert session_id == row['session_id']
        assert config_dir.exists()

        for entry in row['config_dir_tree']:
            if entry['kind'] == 'vanished':
                continue
            path = config_dir / entry['relpath']
            if entry['kind'] == 'symlink':
                assert path.is_symlink(), f'{row["id"]}: {entry["relpath"]} not a symlink'
            elif entry['kind'] == 'dir':
                assert path.is_dir(), f'{row["id"]}: {entry["relpath"]} not a dir'
            else:
                assert path.is_file(), f'{row["id"]}: {entry["relpath"]} not a file'

    @pytest.mark.parametrize('row', _corpus_rows(), ids=_row_ids())
    def test_transcript_presence_matches_the_row(self, row, tmp_path):
        config_dir, session_id = scf.materialize_config_dir(row, tmp_path)
        resolved = sorted(config_dir.glob(f'projects/*/{session_id}.jsonl'))

        if row['transcript_relpath'] is None:
            # The build/uv wedges never reached session init, so nothing may
            # resolve — this is what makes read_transcript_records return None.
            assert not resolved, (
                f'{row["id"]}: no transcript should resolve, found {resolved}'
            )
            return

        assert (config_dir / row['transcript_relpath']).is_file()
        assert resolved, f'{row["id"]}: the projects/*/<sid>.jsonl glob must resolve'

    @pytest.mark.parametrize('row', _corpus_rows(), ids=_row_ids())
    def test_materialized_record_types_match_in_order(self, row, tmp_path):
        if row['transcript_relpath'] is None or row.get('transcript_raw_lines'):
            pytest.skip('no parseable transcript for this row')
        config_dir, _session_id = scf.materialize_config_dir(row, tmp_path)

        written = []
        for line in (config_dir / row['transcript_relpath']).read_text().splitlines():
            if line.strip():
                written.append(json.loads(line))

        assert [r.get('type') for r in written] == [
            r.get('type') for r in row['transcript_records']
        ], f'{row["id"]}: materialized record type sequence differs from the row'

    @pytest.mark.parametrize('row', _corpus_rows(), ids=_row_ids())
    def test_snapshot_round_trips_the_tree(self, row, tmp_path):
        # snapshot_config_dir is the SAME sampler the probe used, so a round trip
        # proves probe output and materialized trees are describable by one
        # function — if they diverged, 3326 would be testing against a shape the
        # watchdog never actually sees.
        config_dir, _session_id = scf.materialize_config_dir(row, tmp_path)
        observed = scf.snapshot_config_dir(config_dir)

        expected_paths = [
            e['relpath'] for e in row['config_dir_tree'] if e['kind'] != 'vanished'
        ]
        observed_paths = [e['relpath'] for e in observed]
        assert observed_paths == sorted(expected_paths), (
            f'{row["id"]}: round-tripped tree differs from the recorded tree'
        )
        observed_kinds = {e['relpath']: e['kind'] for e in observed}
        for entry in row['config_dir_tree']:
            if entry['kind'] == 'vanished':
                continue
            assert observed_kinds[entry['relpath']] == entry['kind'], (
                f'{row["id"]}: {entry["relpath"]} round-tripped as '
                f'{observed_kinds[entry["relpath"]]!r}, recorded {entry["kind"]!r}'
            )

    def test_materialize_is_isolated_per_destination(self, tmp_path):
        # 3326 materializes many rows into one tmp_path-derived tree; two rows
        # must not collide.
        rows = _corpus_rows()
        first, _ = scf.materialize_config_dir(rows[0], tmp_path / 'a')
        second, _ = scf.materialize_config_dir(rows[-1], tmp_path / 'b')
        assert first != second
        assert first.exists() and second.exists()


class TestWedgeShapeCoverage:
    """The corpus covers every shape C5 has to survive, and stays probe-backed.

    Coverage is the difference between "the predicate works on the case we
    happened to capture" and "the predicate was measured against every wedge the
    PRD names".  Provenance-linkage is what stops a curated row from drifting
    free of the empirical capture it claims to summarise — a hand-written row
    that no probe ever observed is exactly the fabrication this task must not
    produce.
    """

    # The three PRD-named wedge shapes, plus the reader-side degrade case C5
    # must handle explicitly (artifacts unreadable -> conservative fallback) —
    # read from the single pinned copy, sorted for a deterministic id order.
    @pytest.mark.parametrize('shape', sorted(_WEDGE_SHAPES))
    def test_every_required_wedge_shape_has_a_row(self, shape):
        # `wedge_shape is not None` is guaranteed for wedge rows by validate_row;
        # restating it here keeps the failure message's sort well-typed.
        shapes = [
            row['wedge_shape']
            for row in scf.load_startup_completion_corpus()
            if row['regime'] == 'wedge' and row['wedge_shape'] is not None
        ]
        assert shape in shapes, (
            f'wedge corpus has no row for {shape!r}; observed {sorted(set(shapes))}'
        )

    def test_transcript_unreadable_covers_both_degrade_variants(self):
        # The two genuinely different ways the artifacts stop being readable:
        # (a) nothing resolves for the watched session -> read_transcript_records
        #     returns None -> predicate None -> C5 degrades to today's bound;
        # (b) the file resolves but every line is truncated/unparseable ->
        #     read_transcript_records returns [] (tolerant parsing) -> predicate
        #     False.  Both are conservative, but they are NOT the same return,
        #     and 3326 must handle each.
        rows = [
            row
            for row in scf.load_startup_completion_corpus()
            if row['wedge_shape'] == 'transcript_unreadable'
        ]
        assert rows, 'no transcript_unreadable degrade rows'

        absent = [row for row in rows if row['transcript_relpath'] is None]
        present_but_unparseable = [
            row
            for row in rows
            if row['transcript_relpath'] is not None and row.get('transcript_raw_lines')
        ]
        assert absent, 'no transcript_unreadable row where the transcript never resolves'
        assert present_but_unparseable, (
            'no transcript_unreadable row where the file exists but every line is '
            'truncated/unparseable'
        )

    def test_healthy_corpus_has_a_pre_first_token_row(self):
        # The incident shape the whole two-regime grace exists for: the CLI has
        # finished starting up but no assistant turn has landed yet, so today's
        # single 120s startup bound is the only thing holding it.  A corpus
        # without this row would not describe the case C5 changes.
        pre_first_token = [
            row
            for row in scf.load_startup_completion_corpus()
            if row['regime'] == 'healthy'
            and row['substrate_returns']['count_transcript_turns'] == 0
        ]
        assert pre_first_token, (
            'healthy corpus carries no pre-first-token row (observed assistant '
            'turns == 0) — the exact state the two-regime grace discriminates'
        )
        # And such a row must be the one the predicate calls started, or the
        # corpus would not demonstrate the discrimination at all.
        assert any(row['expected_startup_complete'] is True for row in pre_first_token)

    def test_every_row_is_linked_to_a_raw_probe_run(self):
        raw_run_ids = {
            json.loads(line)['probe_run_id']
            for line in scf.RAW_CAPTURE_PATH.read_text(encoding='utf-8').splitlines()
            if line.strip()
        }
        assert raw_run_ids, 'raw capture carries no probe_run_id'

        for row in scf.load_startup_completion_corpus():
            run_id = row['provenance']['probe_run_id']
            assert run_id in raw_run_ids, (
                f'{row["id"]}: provenance.probe_run_id {run_id!r} is not present in '
                f'{scf.RAW_CAPTURE_PATH.name} — a curated row must be traceable to '
                f'the observation it summarises, never hand-written'
            )

    def test_derived_rows_declare_their_derivation(self):
        # A row that could not be produced by spawning a CLI (the reader-side
        # degrade cases) is still probe-backed, but it is a TRANSFORM of a raw
        # observation rather than a raw observation.  Saying so in provenance is
        # what keeps "empirically observed" an honest claim.
        for row in scf.load_startup_completion_corpus():
            method = row['provenance']['capture_method']
            if method == 'live_spawn':
                continue
            assert row['provenance'].get('derived_from'), (
                f'{row["id"]}: capture_method {method!r} must name the raw sample '
                f'it was derived from'
            )
            assert row['provenance'].get('derivation'), (
                f'{row["id"]}: capture_method {method!r} must describe the transform '
                f'applied to that sample'
            )


class TestPredicateDiscrimination:
    """The core deliverable: the chosen predicate's verdict on every row."""

    @pytest.mark.parametrize('row', _corpus_rows(), ids=_row_ids())
    def test_predicate_matches_the_recorded_verdict(self, row, tmp_path):
        config_dir, session_id = scf.materialize_config_dir(row, tmp_path)
        verdict = scf.evaluate_startup_completion_predicate(config_dir, session_id)
        assert verdict is row['expected_startup_complete'], (
            f'{row["id"]}: predicate returned {verdict!r}, row records '
            f'{row["expected_startup_complete"]!r}'
        )

    @pytest.mark.parametrize('row', _corpus_rows(), ids=_row_ids())
    def test_committed_substrate_returns_match_the_row(self, row, tmp_path):
        # Proves the predicate is derived from artifacts already readable on main
        # today — not from new production code 3326 has yet to write.
        config_dir, session_id = scf.materialize_config_dir(row, tmp_path)
        recorded = row['substrate_returns']

        assert cli_invoke.transcript_exists(config_dir, session_id) is (
            recorded['transcript_exists']
        ), f'{row["id"]}: transcript_exists differs from the recorded observation'

        records = cli_invoke.read_transcript_records(config_dir, session_id)
        assert (records is None) is recorded['read_transcript_records_is_none'], (
            f'{row["id"]}: read_transcript_records None-ness differs'
        )
        assert (None if records is None else len(records)) == recorded['record_count'], (
            f'{row["id"]}: record_count differs'
        )
        assert cli_invoke.count_transcript_turns(config_dir, session_id) == (
            recorded['count_transcript_turns']
        ), f'{row["id"]}: count_transcript_turns differs'

    def test_predicate_emits_all_three_states_and_tracks_reachability(self, tmp_path):
        """The corpus separates *startup reached* from *startup not reached*.

        NOT healthy from wedge — it does not separate those, and the report says
        so (§6, finding F2).  One genuine wedge row
        (`wedge_mcp_init_hang_pre_first_token`) evaluates True, identical to the
        healthy rows, because transcript creation turned out not to be gated on
        MCP `initialize` completing; and the only `False` anywhere comes from the
        derived truncated-transcript degrade row.  A "healthy True / wedge False"
        assertion would therefore claim a discrimination the artifacts do not
        support, which is exactly what this task must not do.

        What the corpus DOES demonstrate, and what C5 rests on, is asserted here:
        the predicate emits all three states, and its verdict tracks transcript
        REACHABILITY exactly — reachable yields a bool, unreachable yields the
        `None` sentinel.
        """
        verdicts = []
        for index, row in enumerate(_corpus_rows()):
            config_dir, session_id = scf.materialize_config_dir(row, tmp_path / str(index))
            verdict = scf.evaluate_startup_completion_predicate(config_dir, session_id)
            verdicts.append(verdict)

            if row['substrate_returns']['transcript_exists']:
                assert isinstance(verdict, bool), (
                    f'{row["id"]}: the transcript is reachable, so the verdict must '
                    f'be a bool, got {verdict!r}'
                )
            else:
                assert verdict is None, (
                    f'{row["id"]}: the transcript is unreachable, so the verdict '
                    f'must be the None sentinel, got {verdict!r}'
                )

        # A predicate that returned one constant would satisfy every per-row
        # assertion above and below; all three states must actually occur.
        assert {True, False, None} <= set(verdicts), (
            f'the predicate never emits all three states over the corpus — '
            f'observed {set(verdicts)}, so at least one branch is unexercised'
        )

    @pytest.mark.parametrize('row', _corpus_rows(), ids=_row_ids())
    def test_unreadable_transcripts_yield_the_none_sentinel(self, row, tmp_path):
        # The tri-state contract C5's conservative degrade rests on: "cannot
        # locate/read the transcript" must never be folded into True or False.
        config_dir, session_id = scf.materialize_config_dir(row, tmp_path)
        unreadable = cli_invoke.read_transcript_records(config_dir, session_id) is None
        verdict = scf.evaluate_startup_completion_predicate(config_dir, session_id)
        if unreadable:
            assert verdict is None, (
                f'{row["id"]}: an unreadable transcript must yield None, got {verdict!r}'
            )
        else:
            assert verdict is not None, (
                f'{row["id"]}: a readable transcript must yield a bool, got None'
            )


@pytest.mark.integration
@pytest.mark.skipif(
    not _OAUTH_TOKEN_PRESENT,
    reason='Requires at least 1 OAuth account in env (CLAUDE_OAUTH_TOKEN_[A-G])',
)
class TestLiveReprobe:
    """Drift guard: re-probe a live CLI and diff it against the committed corpus.

    The corpus is a snapshot of CLI 2.1.220's startup artifacts.  The predicate
    is deliberately keyed on transcript existence and non-emptiness rather than
    on record types, so a vocabulary change does not break it — but a change to
    WHERE the transcript is written, or to whether it is created before the
    first token, would silently invalidate the whole two-regime grace.  This
    test makes that failure loud instead of letting the corpus rot.

    ``@pytest.mark.integration`` keeps it out of the default run
    (``addopts = "-m 'not integration'"`` in `shared/pyproject.toml`), so it
    never participates in the RED/GREEN loop.  Run it deliberately after a CLI
    bump::

        cd shared && uv run pytest tests/test_startup_completion_fixtures.py -m integration
    """

    @staticmethod
    def _committed_pre_first_token_row():
        """The committed healthy row this re-probe is diffed against."""
        rows = [
            row
            for row in scf.load_startup_completion_corpus()
            if row['regime'] == 'healthy'
            and row['substrate_returns']['count_transcript_turns'] == 0
            and row['transcript_relpath'] is not None
        ]
        assert rows, 'no committed healthy pre-first-token row to diff against'
        # Lowest record count — the earliest committed observation of the
        # boundary, i.e. the tightest bar a fresh probe has to clear.
        return min(rows, key=lambda r: r['substrate_returns']['record_count'])

    @pytest.fixture(scope='class')
    def fresh_observations(self, tmp_path_factory):
        """Run the probe live once and return its observations."""
        import startup_completion_probe as probe

        out = tmp_path_factory.mktemp('reprobe') / 'fresh.jsonl'
        rc = probe.main(['--mode', 'healthy', '--out', str(out)])
        assert rc == 0, f'probe exited {rc}'
        observations = [
            json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
            if line.strip()
        ]
        assert observations, 'probe emitted no observations'
        return observations

    @pytest.fixture(scope='class')
    def fresh_pre_first_token(self, fresh_observations):
        """The last pre-turn-1 sample of the fresh run, guarded ONCE.

        Selected in one place because pytest does not make test_b/test_c depend
        on test_a: an empty list would otherwise surface as a bare
        ``IndexError`` from two tests instead of the DRIFT message.  And an empty
        list is precisely the drift this class exists to detect — a CLI that
        never exposes a pre-turn-1 window leaves the predicate nothing to act in.
        The probe run is real money and non-repeatable, so the failure has to be
        legible the first time.
        """
        pre = [o for o in fresh_observations if o['sample_kind'] == 'pre_first_token']
        assert pre, (
            'DRIFT: the probe captured no pre_first_token sample. Either the CLI '
            'no longer exposes a window between session init and the first '
            'assistant record, or the sampler raced past it. Re-run the probe; if '
            'it reproduces, the SESSION-TRANSCRIPT-MATERIALIZED predicate has no '
            'window to act in and C5 needs re-deriving. '
            'See docs/startup-completion-artifact-matrix.md §4.'
        )
        return pre[-1]

    def test_a_transcript_still_materializes_before_the_first_token(
        self, fresh_pre_first_token
    ):
        # The load-bearing claim: the CLI writes projects/*/<sid>.jsonl BEFORE
        # any assistant record.  If a future CLI stops doing that, the predicate
        # can no longer distinguish "started" from "never started" and C5's
        # extended bound would be handed out on no evidence.
        sample = fresh_pre_first_token

        assert sample['substrate_returns']['transcript_exists'] is True, (
            'DRIFT: no transcript resolved at the pre-first-token boundary. '
            f'CLI {sample["cli_version"]} no longer materializes '
            'projects/*/<session-id>.jsonl before the first assistant record — '
            'the SESSION-TRANSCRIPT-MATERIALIZED predicate is invalidated. '
            'See docs/startup-completion-artifact-matrix.md §4.'
        )
        assert sample['substrate_returns']['count_transcript_turns'] == 0, (
            'DRIFT: the pre-first-token sample already carries an assistant '
            f'turn ({sample["substrate_returns"]["count_transcript_turns"]}); '
            'the probe never observed the pre-turn-1 window this corpus is about'
        )

    def test_b_predicate_still_returns_true_at_that_boundary(
        self, fresh_pre_first_token, tmp_path
    ):
        # Materialize the FRESH observation through the same loader path 3326
        # uses, and run the committed predicate against it.
        sample = fresh_pre_first_token
        config_dir, session_id = scf.materialize_config_dir(sample, tmp_path)
        verdict = scf.evaluate_startup_completion_predicate(config_dir, session_id)

        assert verdict is True, (
            f'DRIFT: the committed predicate returns {verdict!r} on a fresh '
            f'{sample["cli_version"]} pre-first-token observation, not True. '
            f'Observed transcript_relpath={sample["transcript_relpath"]!r}, '
            f'substrate_returns={sample["substrate_returns"]!r}.'
        )

    def test_c_record_type_prefix_still_matches_the_committed_row(
        self, fresh_pre_first_token
    ):
        # Not load-bearing for the predicate (see the report's rejected
        # alternatives — a type-keyed rule was deliberately NOT chosen), but it
        # is the earliest signal that the CLI's record vocabulary moved, which
        # is worth knowing before something that IS load-bearing follows.
        committed = self._committed_pre_first_token_row()
        committed_records = committed['transcript_records']
        assert committed_records is not None, (
            f'{committed["id"]}: a healthy pre-first-token row must carry parsed '
            'transcript_records to diff a fresh probe against'
        )
        expected = [r.get('type') for r in committed_records]

        sample = fresh_pre_first_token
        observed = [r.get('type') for r in (sample['transcript_records'] or [])]

        # Prefix comparison, not equality: the record COUNT at the boundary is
        # a race with the sampler (4 and 5 records were both observed across the
        # two committed runs), so only the shared-length prefix is stable.
        n = min(len(expected), len(observed))
        assert n > 0, f'DRIFT: fresh pre-first-token sample carries no records ({observed})'
        assert observed[:n] == expected[:n], (
            'DRIFT: transcript record-type prefix changed.\n'
            f'  committed ({committed["id"]}, CLI '
            f'{committed["provenance"]["cli_version"]}): {expected}\n'
            f'  observed  (CLI {sample["cli_version"]}): {observed}\n'
            'The predicate does not key on record types, so this is a warning '
            'sign rather than a break — but re-run the probe and append a fresh '
            'corpus row per fixtures/startup_completion/README.md.'
        )
