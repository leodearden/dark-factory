"""Tests for the startup-completion probe HARNESS (task 4097).

Distinct from ``test_startup_completion_fixtures.py``, which is about the
committed CORPUS.  This module pins the two hygiene contracts of the probe
itself — the capture-time redaction gate (:func:`startup_completion_probe._gate`
/ ``_scrub_value``) and the lifecycle of the OAuth-bearing ``TaskConfigDir`` that
``run_live_probe`` writes a live access token into.

Every test here is fast and OFFLINE: the ``run_live_probe`` tests inject their
failure before ``subprocess.Popen``, monkeypatch ``_cli_version``/``_oauth_token``
and pin ``TaskConfigDir`` at ``tmp_path``, so nothing spawns a CLI, nothing spends
money and no real token is ever written.  They are deliberately unmarked (no
``@pytest.mark.integration``) so these fixes stay covered on every ordinary run.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import IO, Any

import pytest
import startup_completion_fixtures as scf
import startup_completion_probe as probe

from shared.config_dir import _PID_SUFFIX_RE, CONFIG_DIR_PREFIX

#: A long base64url run: 70 chars, no ``/`` neighbours, so it trips
#: ``GENERIC_CREDENTIAL_PATTERNS`` exactly as a raw pasted token would.
_LONG_RUN = 'A' * 70


def _minimal_observation(**overrides: Any) -> dict[str, Any]:
    """The smallest observation shaped like what :func:`probe.observe` assembles.

    Carries the identity fields and ``substrate_returns`` because
    ``run_live_probe`` subscripts ``candidate['substrate_returns']
    ['count_transcript_turns']`` on the pre-first-token path.
    """
    observation: dict[str, Any] = {
        'probe_run_id': 'test',
        'mode': 'healthy',
        'sample_index': 0,
        'sample_kind': 'scheduled',
        'sample_offset_secs': 0.25,
        'substrate_returns': {
            'transcript_exists': False,
            'read_transcript_records_is_none': True,
            'record_count': None,
            'count_transcript_turns': None,
        },
    }
    observation.update(overrides)
    return observation


def _encoded_is_clean(value: Any) -> bool:
    """True when *value*'s JSON encoding carries no generic credential run."""
    return (
        probe.scan_for_credential_material(
            json.dumps(value), probe._GENERIC_CREDENTIAL_PATTERNS
        )
        is None
    )


class TestScrubValueScrubsDictKeys:
    """``_scrub_value`` must scrub string dict KEYS, not just string leaves.

    The invariant under defence is ``_gate``'s documented blast-radius contract
    (probe lines 185-192): the GENERIC long-run branch is a heuristic that may
    fire on a long non-secret identifier, so it SUBSTITUTES and warns — it must
    never raise.  A raise there propagates out of the sampling loop through
    ``run_live_probe``'s ``finally`` and destroys the entire real-money capture,
    for a harness whose whole purpose is to be re-run after a CLI bump.

    ``_scrub_value`` recursed into dict VALUES only, so a credential-shaped KEY
    survived unscrubbed into ``_gate``'s post-scrub
    ``assert_no_credential_material`` — turning the never-raise branch into a
    raise.  Latent today (``redact_record`` projects records onto a fixed
    allow-list, so every key is probe-authored) and armed the moment that
    allow-list widens, which its own docstring anticipates.
    """

    def test_credential_shaped_key_is_scrubbed(self):
        result = probe._scrub_value({_LONG_RUN: 'ok'}, probe._GENERIC_CREDENTIAL_PATTERNS)
        assert _encoded_is_clean(result), (
            f'a credential-shaped dict key survived scrubbing: {json.dumps(result)[:120]}'
        )
        assert list(result.values()) == ['ok'], 'scrubbing a key must not disturb its value'

    def test_credential_shaped_key_nested_under_a_list_is_scrubbed(self):
        # The real observation shape: transcript_records is a list of dicts.
        value = {'transcript_records': [{_LONG_RUN: 1}]}
        result = probe._scrub_value(value, probe._GENERIC_CREDENTIAL_PATTERNS)
        assert _encoded_is_clean(result), (
            'a credential-shaped key nested under a list survived scrubbing'
        )
        assert list(result['transcript_records'][0].values()) == [1]

    def test_gate_does_not_raise_on_a_key_side_generic_hit(self):
        observation = _minimal_observation(transcript_records=[{_LONG_RUN: 1}])
        # No pytest.raises: returning is the whole contract.
        result = probe._gate(observation)
        scf.assert_no_credential_material(
            json.dumps(result), source='synthetic:key-side-hit'
        )

    def test_non_string_keys_round_trip_untouched(self):
        result = probe._scrub_value({1: 'x'}, probe._GENERIC_CREDENTIAL_PATTERNS)
        assert result == {1: 'x'}, 'a non-string key must not be coerced or rewritten'


class TestScrubbedKeysDoNotCollide:
    """Scrubbing keys must not silently drop an entry.

    Two distinct keys can both rewrite to ``<redacted>`` and collapse into one
    dict entry — silent data loss, which is exactly the fail-soft this repo's
    design invariants reject (loud-over-silent-degradation / no-silent-fail-soft).
    A gate whose job is to make a capture legible must not quietly delete half of
    what it was asked to redact.
    """

    def test_two_credential_shaped_keys_both_survive(self):
        value = {_LONG_RUN: 1, 'B' * 70: 2}
        result = probe._scrub_value(value, probe._GENERIC_CREDENTIAL_PATTERNS)
        assert len(result) == 2, f'a scrubbed key collision dropped an entry: {result!r}'
        assert list(result.values()) == [1, 2], 'values must survive in insertion order'
        assert len(set(result)) == 2, 'the two scrubbed keys must stay distinct'

    def test_scrubbed_key_colliding_with_a_literal_key_survives(self):
        # '<redacted>' can already be present as a literal key — a previous scrub
        # pass, or a value the CLI itself emitted.
        value = {'<redacted>': 0, _LONG_RUN: 1}
        result = probe._scrub_value(value, probe._GENERIC_CREDENTIAL_PATTERNS)
        assert len(result) == 2, f'a scrubbed key collided with a literal key: {result!r}'
        assert sorted(result.values()) == [0, 1]

    def test_collision_disambiguation_is_deterministic(self):
        value = {_LONG_RUN: 1, 'B' * 70: 2, 'C' * 70: 3}
        first = probe._scrub_value(value, probe._GENERIC_CREDENTIAL_PATTERNS)
        second = probe._scrub_value(dict(value), probe._GENERIC_CREDENTIAL_PATTERNS)
        # A re-run of the probe over equal input must not produce a differently
        # keyed row, or two captures of the same shape would not compare equal.
        assert first == second
        assert list(first) == list(second)

    def test_disambiguated_keys_stay_credential_clean(self):
        value = {_LONG_RUN: 1, 'B' * 70: 2}
        result = probe._scrub_value(value, probe._GENERIC_CREDENTIAL_PATTERNS)
        assert _encoded_is_clean(result), (
            'the collision disambiguator must not re-form a credential-shaped run'
        )


#: A leaf whose RAW form carries only a 63-char run (below the 64 threshold) but
#: whose JSON ENCODING carries 64: ``json.dumps`` renders the tab as ``\t``, and
#: the escape's literal ``t`` is in ``[A-Za-z0-9_-]``, extending the run by one.
_ENCODING_EXTENDED_LEAF = '\t' + 'A' * 63


class TestEncodedDomainParity:
    """``_scrub_value`` must produce a value clean in its JSON-ENCODED form.

    The live instance of the same defect the key route makes latent: ``_gate``
    SCANS ``json.dumps(observation)`` while ``_scrub_value`` SUBSTITUTES on raw
    string leaves.  The two domains differ, so JSON escaping can manufacture a
    64-char base64url run that raw scrubbing cannot see — and the post-scrub
    ``assert_no_credential_material``, which scans the encoded form again, then
    raises out of the never-raise branch.

    Reachable today through ``run_exit.stderr_tail``: arbitrary CLI stderr, where
    tabs, control characters and non-ASCII are routine, gated by
    ``_gate(_drain_exit(...))`` after every sample has already been paid for.
    """

    def test_premise_json_escaping_can_manufacture_a_run(self):
        # Pin the premise itself, so this class cannot pass vacuously if the
        # pattern or its lookarounds ever change.
        raw_hit = probe.scan_for_credential_material(
            _ENCODING_EXTENDED_LEAF, probe._GENERIC_CREDENTIAL_PATTERNS
        )
        encoded_hit = probe.scan_for_credential_material(
            json.dumps(_ENCODING_EXTENDED_LEAF), probe._GENERIC_CREDENTIAL_PATTERNS
        )
        assert raw_hit is None, 'the raw leaf must be BELOW the run threshold'
        assert encoded_hit is not None, 'the encoded leaf must be AT the run threshold'

    def test_gate_does_not_raise_on_an_encoding_extended_stderr_tail(self):
        observation = _minimal_observation(
            run_exit={'stderr_tail': _ENCODING_EXTENDED_LEAF}
        )
        result = probe._gate(observation)
        scf.assert_no_credential_material(
            json.dumps(result), source='synthetic:encoded-stderr-tail'
        )

    def test_gate_does_not_raise_on_an_encoding_extended_key(self):
        observation = _minimal_observation(
            transcript_records=[{_ENCODING_EXTENDED_LEAF: 1}]
        )
        result = probe._gate(observation)
        scf.assert_no_credential_material(
            json.dumps(result), source='synthetic:encoded-key'
        )

    def test_gate_does_not_raise_on_a_non_string_scalar_whose_encoding_trips(self):
        # A 70-digit int is not a str, so no substitution branch sees it at all —
        # but json.dumps renders it as 70 base64url-class characters.
        observation = _minimal_observation(config_dir_tree=[{'size': int('9' * 70)}])
        result = probe._gate(observation)
        scf.assert_no_credential_material(
            json.dumps(result), source='synthetic:encoded-scalar'
        )


#: The keys a poisoned row is allowed to carry.  Everything else of the input
#: observation must be dropped: the row's whole claim is that it is clean BY
#: CONSTRUCTION, which only holds if every field is a probe-owned literal or a
#: non-string scalar.
_POISONED_KEYS = frozenset(
    {
        'redaction_failed',
        'redaction_failure_pattern',
        'mode',
        'sample_kind',
        'sample_index',
        'sample_offset_secs',
        'substrate_returns',
    }
)


class TestGateNeverRaisesOnGenericHit:
    """The never-raise contract must hold STRUCTURALLY, not by proof.

    Steps 2/4/6 make the scrub correct, but the guarantee would then rest on a
    composition argument — and a composition argument is exactly what failed the
    first time.  So the residual case is handled by construction: a still-dirty
    observation degrades to a minimal ``redaction_failed`` row plus a stderr
    WARNING, losing ONE sample, rather than raising and losing every sample of an
    already-paid-for live run.

    ``_scrub_value`` is monkeypatched to the identity here, standing in for any
    FUTURE scrub gap of the same class as the two just closed.
    """

    @staticmethod
    def _dirty_observation(**overrides: Any) -> dict[str, Any]:
        return _minimal_observation(transcript_records=[{'text': _LONG_RUN}], **overrides)

    def test_returns_a_poisoned_row_instead_of_raising(self, monkeypatch, capsys):
        monkeypatch.setattr(probe, '_scrub_value', lambda value, patterns: value)
        result = probe._gate(self._dirty_observation())

        assert result['redaction_failed'] is True
        assert result['redaction_failure_pattern'] == 'long-base64url-run'
        scf.assert_no_credential_material(
            json.dumps(result), source='synthetic:poisoned-row'
        )
        stderr = capsys.readouterr().err
        assert 'redaction' in stderr.lower(), (
            f'the degradation must be LOUD; stderr was: {stderr!r}'
        )

    def test_in_range_identity_fields_survive(self, monkeypatch):
        monkeypatch.setattr(probe, '_scrub_value', lambda value, patterns: value)
        result = probe._gate(
            self._dirty_observation(
                mode='healthy', sample_kind='scheduled', sample_index=3,
                sample_offset_secs=1.5,
            )
        )
        assert result['mode'] == 'healthy'
        assert result['sample_kind'] == 'scheduled'
        assert result['sample_index'] == 3
        assert result['sample_offset_secs'] == 1.5

    def test_out_of_range_identity_fields_degrade_to_none(self, monkeypatch):
        monkeypatch.setattr(probe, '_scrub_value', lambda value, patterns: value)
        # An arbitrary string in a closed-set field is the very thing that could
        # smuggle credential material back into the "clean by construction" row.
        result = probe._gate(
            self._dirty_observation(
                mode='not-a-mode', sample_kind='not-a-kind',
                sample_index='not-an-int', sample_offset_secs='not-a-float',
            )
        )
        assert result['mode'] is None
        assert result['sample_kind'] is None
        assert result['sample_index'] is None
        assert result['sample_offset_secs'] is None

    def test_every_declared_sample_kind_survives(self, monkeypatch):
        monkeypatch.setattr(probe, '_scrub_value', lambda value, patterns: value)
        for kind in probe.SAMPLE_KINDS:
            result = probe._gate(self._dirty_observation(sample_kind=kind))
            assert result['sample_kind'] == kind, (
                f'{kind!r} is emitted by the samplers but not in SAMPLE_KINDS'
            )

    def test_substrate_returns_is_carried_and_scalar_filtered(self, monkeypatch):
        monkeypatch.setattr(probe, '_scrub_value', lambda value, patterns: value)
        # run_live_probe subscripts candidate['substrate_returns']
        # ['count_transcript_turns'] on the pre-first-token path — a placeholder
        # without it would trade the raise for a KeyError at the same blast radius.
        result = probe._gate(
            self._dirty_observation(
                substrate_returns={
                    'transcript_exists': True,
                    'read_transcript_records_is_none': False,
                    'record_count': 7,
                    'count_transcript_turns': _LONG_RUN,  # non-scalar-shaped intruder
                }
            )
        )
        substrate = result['substrate_returns']
        assert set(substrate) == {
            'transcript_exists',
            'read_transcript_records_is_none',
            'record_count',
            'count_transcript_turns',
        }
        assert substrate['transcript_exists'] is True
        assert substrate['record_count'] == 7
        assert substrate['count_transcript_turns'] is None, (
            'a non-bool/int value must be filtered out, not carried through'
        )

    def test_substrate_returns_defaults_when_absent(self, monkeypatch):
        monkeypatch.setattr(probe, '_scrub_value', lambda value, patterns: value)
        observation = self._dirty_observation()
        del observation['substrate_returns']
        result = probe._gate(observation)
        assert result['substrate_returns']['count_transcript_turns'] is None

    def test_no_other_observation_key_is_carried_over(self, monkeypatch):
        monkeypatch.setattr(probe, '_scrub_value', lambda value, patterns: value)
        result = probe._gate(
            self._dirty_observation(
                config_dir_tree=[{'relpath': 'projects/x.jsonl'}],
                run_exit={'stderr_tail': 'boom'},
                session_id='abc',
                spawn_argv=['claude', '--print'],
            )
        )
        assert set(result) <= _POISONED_KEYS, (
            f'the poisoned row carried unvetted input fields: {set(result) - _POISONED_KEYS}'
        )
        for leaked in ('transcript_records', 'config_dir_tree', 'run_exit', 'spawn_argv'):
            assert leaked not in result


# ---------------------------------------------------------------------------
# run_live_probe: lifecycle of the OAuth-bearing config dir
# ---------------------------------------------------------------------------

#: Obviously fake, mirroring ``TestCorpusSecretHygiene``'s synthetic payload
#: convention: a grep of the tree for credential-shaped strings must stay clean,
#: and no test may ever put a real token on disk.
_FAKE_OAUTH_ENV_VAR = 'CLAUDE_CODE_OAUTH_TOKEN_A'
_FAKE_OAUTH_TOKEN = 'FAKE-not-a-real-token'


class _ProbeRecorder:
    """What a monkeypatched :func:`run_live_probe` run allocated.

    Holds the resources whose reclamation is under test, so an assertion can
    name the exact object the probe created rather than re-deriving its path.
    """

    def __init__(self) -> None:
        self.configs: list[Any] = []
        self.config_task_ids: list[str] = []
        self.config_kwargs: list[dict[str, Any]] = []
        self.stub_dirs: list[Path] = []
        self.temp_paths: list[Path] = []
        self.handles: list[IO[bytes]] = []


@pytest.fixture
def probe_recorder(monkeypatch, tmp_path) -> _ProbeRecorder:
    """Make ``run_live_probe`` offline and record what it allocates.

    ``_cli_version`` and ``_oauth_token`` are stubbed so no ``claude --version``
    subprocess runs and no REAL token is ever written; ``TaskConfigDir`` is the
    genuine class, only re-based under ``tmp_path`` so the credential write and
    the cleanup being tested are the real ones.
    """
    recorder = _ProbeRecorder()
    monkeypatch.setattr(probe, '_cli_version', lambda: 'test-cli')
    monkeypatch.setattr(
        probe, '_oauth_token', lambda: (_FAKE_OAUTH_ENV_VAR, _FAKE_OAUTH_TOKEN)
    )

    real_config_cls = probe.TaskConfigDir

    def _recording_config_dir(task_id: str, base_dir: Path | None = None, **kwargs: Any):
        config = real_config_cls(task_id, base_dir=tmp_path, **kwargs)
        recorder.configs.append(config)
        recorder.config_task_ids.append(task_id)
        recorder.config_kwargs.append(dict(kwargs))
        return config

    monkeypatch.setattr(probe, 'TaskConfigDir', _recording_config_dir)

    real_mkdtemp = tempfile.mkdtemp

    def _recording_mkdtemp(*args: Any, **kwargs: Any) -> str:
        path = real_mkdtemp(*args, **kwargs)
        recorder.stub_dirs.append(Path(path))
        return path

    monkeypatch.setattr(probe.tempfile, 'mkdtemp', _recording_mkdtemp)
    return recorder


def _run_probe(tmp_path: Path, **overrides: Any) -> list[dict]:
    """Call ``run_live_probe`` with harmless defaults; never reaches ``Popen``."""
    kwargs: dict[str, Any] = {
        'mode': 'healthy',
        'probe_run_id': 'test-run',
        'cwd': tmp_path,
        'prompt': 'probe',
        'model': 'test-model',
        'permission_mode': 'default',
        'offsets': (0.0,),
        'max_secs': 0.0,
        'hold_secs': 0,
        'keep_config_dir': False,
    }
    kwargs.update(overrides)
    return probe.run_live_probe(**kwargs)


def _record_argv_temp_paths(monkeypatch, recorder: _ProbeRecorder) -> None:
    """Wrap ``_build_argv`` so the temp files it creates are recorded, not replaced.

    The REAL ``_build_argv`` still runs, so its sysprompt (and, for ``mcp_wedge``,
    its MCP-config) tempfiles genuinely exist and genuinely have to be reclaimed.
    """
    real_build_argv = probe._build_argv

    def _recording_build_argv(*args: Any, **kwargs: Any):
        argv, temp_paths = real_build_argv(*args, **kwargs)
        recorder.temp_paths.extend(temp_paths)
        return (argv, temp_paths)

    monkeypatch.setattr(probe, '_build_argv', _recording_build_argv)


def _inject(kind: str, monkeypatch, recorder: _ProbeRecorder) -> type[Exception]:
    """Install a failure at *kind*, all BEFORE ``subprocess.Popen``.

    Nothing spawns, so these tests are free and instant.  Returns the exception
    class the injected failure raises.
    """
    if kind == 'build_argv':
        def _boom_build_argv(*args: Any, **kwargs: Any):
            # Assert the credential-bearing window is REAL before failing in it,
            # so this test cannot pass vacuously against a probe that (say)
            # stopped writing credentials at all.
            config = recorder.configs[-1]
            assert (config.path / '.credentials.json').exists(), (
                'the OAuth token must already be on disk at this injection point, '
                'or the leak this test defends against would not exist'
            )
            raise RuntimeError('injected: _build_argv')

        monkeypatch.setattr(probe, '_build_argv', _boom_build_argv)
        return RuntimeError

    if kind == 'spawn_env':
        # The REAL _build_argv runs first, so by now its sysprompt tempfile
        # exists and must also be reclaimed.
        _record_argv_temp_paths(monkeypatch, recorder)

        def _boom_spawn_env(*args: Any, **kwargs: Any):
            raise RuntimeError('injected: _spawn_env')

        monkeypatch.setattr(probe, '_spawn_env', _boom_spawn_env)
        return RuntimeError

    if kind == 'capture_open':
        # The SECOND 'wb' open fails, so the first handle is already open and
        # must be closed by the teardown rather than left to the GC.
        real_path_open = Path.open
        state = {'wb_opens': 0}

        def _recording_open(self: Path, *args: Any, **kwargs: Any):
            mode = args[0] if args else kwargs.get('mode', 'r')
            if mode != 'wb':
                return real_path_open(self, *args, **kwargs)
            state['wb_opens'] += 1
            if state['wb_opens'] == 2:
                raise OSError('injected: second capture file')
            handle = real_path_open(self, *args, **kwargs)
            recorder.handles.append(handle)  # pyright: ignore[reportArgumentType]
            return handle

        monkeypatch.setattr(Path, 'open', _recording_open)
        return OSError

    raise AssertionError(f'unknown injection kind {kind!r}')


@pytest.fixture(params=['build_argv', 'spawn_env', 'capture_open'])
def injected_failure(request, monkeypatch, probe_recorder) -> type[Exception]:
    return _inject(request.param, monkeypatch, probe_recorder)


class TestRunLiveProbeCleanup:
    """An exception before the spawn must not leak the OAuth-bearing config dir.

    ``run_live_probe`` writes a LIVE OAuth access token to
    ``<tmp>/claude-config-startup-probe-<mode>-<pid>/.credentials.json`` (mode
    0600) and only ``config.cleanup()`` — reached solely through the ``finally``
    of a ``try`` that opens several statements LATER — ever removes it.  Any
    raise in between (a missing ``claude`` binary, a full /tmp, an
    interpreter-level error in argv/env construction) strands a real token on
    disk indefinitely: ``TaskConfigDir`` is constructed with the default
    ``cleanup_at_exit=False``, and no sweep covers this prefix.

    Every injection point here is BEFORE ``subprocess.Popen``, so these tests
    spawn nothing, spend nothing and write only an obviously-fake token.
    """

    def test_config_dir_does_not_survive_the_exception(
        self, probe_recorder, injected_failure, tmp_path
    ):
        with pytest.raises(injected_failure, match='injected'):
            _run_probe(tmp_path)
        assert probe_recorder.configs, 'the probe never built a config dir — bad injection'
        config_path = probe_recorder.configs[-1].path
        assert not config_path.exists(), (
            f'the OAuth-bearing config dir survived the exception: {config_path}'
        )

    def test_stub_dir_does_not_survive_the_exception(
        self, probe_recorder, injected_failure, tmp_path
    ):
        with pytest.raises(injected_failure, match='injected'):
            _run_probe(tmp_path)
        assert probe_recorder.stub_dirs, 'the probe never built a stub dir — bad injection'
        stub_dir = probe_recorder.stub_dirs[-1]
        assert not stub_dir.exists(), f'the stub dir survived the exception: {stub_dir}'

    def test_argv_temp_files_do_not_survive_the_exception(
        self, probe_recorder, monkeypatch, tmp_path
    ):
        exc_type = _inject('spawn_env', monkeypatch, probe_recorder)
        with pytest.raises(exc_type, match='injected'):
            _run_probe(tmp_path)
        assert probe_recorder.temp_paths, '_build_argv produced no temp file to reclaim'
        for path in probe_recorder.temp_paths:
            assert not path.exists(), f'an argv temp file survived the exception: {path}'

    def test_open_capture_handles_are_closed(self, probe_recorder, monkeypatch, tmp_path):
        exc_type = _inject('capture_open', monkeypatch, probe_recorder)
        with pytest.raises(exc_type, match='injected'):
            _run_probe(tmp_path)
        assert probe_recorder.handles, 'no capture handle was opened — bad injection'
        for handle in probe_recorder.handles:
            assert handle.closed, (
                'a capture file handle was left to the GC rather than closed by teardown'
            )


class TestConfigDirLifetime:
    """``cleanup_at_exit`` must MIRROR ``--keep-config-dir``, not be hardcoded.

    The ``finally`` fixed above covers a raise INSIDE ``run_live_probe``.  It
    cannot cover one that escapes it — a ``SystemExit`` from ``main``, a
    ``KeyboardInterrupt`` on a 30-second live run, a raise in the caller between
    probe and write — and ``TaskConfigDir`` defaults to ``cleanup_at_exit=False``,
    so today nothing reclaims the token dir in those cases either.

    The negation is load-bearing rather than cosmetic: ``TaskConfigDir.__init__``
    registers ``atexit.register(shutil.rmtree, ...)``, so an unconditional
    ``True`` would silently destroy at interpreter exit exactly the dir
    ``--keep-config-dir`` exists to preserve for post-run inspection.
    """

    def test_default_run_registers_the_atexit_teardown(
        self, probe_recorder, monkeypatch, tmp_path
    ):
        exc_type = _inject('build_argv', monkeypatch, probe_recorder)
        with pytest.raises(exc_type, match='injected'):
            _run_probe(tmp_path, keep_config_dir=False)
        assert probe_recorder.config_kwargs[-1].get('cleanup_at_exit') is True, (
            'without cleanup_at_exit the token dir survives any exit that escapes '
            f'run_live_probe; kwargs were {probe_recorder.config_kwargs[-1]!r}'
        )

    def test_keep_config_dir_suppresses_the_atexit_teardown(
        self, probe_recorder, monkeypatch, tmp_path
    ):
        exc_type = _inject('build_argv', monkeypatch, probe_recorder)
        with pytest.raises(exc_type, match='injected'):
            _run_probe(tmp_path, keep_config_dir=True)
        assert probe_recorder.config_kwargs[-1].get('cleanup_at_exit') is False, (
            'an unconditional atexit rmtree would silently defeat --keep-config-dir'
        )

    def test_keep_config_dir_survives_a_pre_spawn_failure(
        self, probe_recorder, monkeypatch, tmp_path
    ):
        # The step-10 finally must not override the operator's inspect flag: a
        # failed run is precisely when the dir is worth looking at.
        exc_type = _inject('build_argv', monkeypatch, probe_recorder)
        with pytest.raises(exc_type, match='injected'):
            _run_probe(tmp_path, keep_config_dir=True)
        config_path = probe_recorder.configs[-1].path
        assert config_path.exists(), (
            f'--keep-config-dir was ignored on the failure path: {config_path} is gone'
        )


@pytest.fixture
def sweep_calls(monkeypatch, probe_recorder) -> list[str]:
    """Record ``sweep_stale_pid_dirs`` calls and reset the once-per-process flag.

    The reset matters: without it the flag's value would depend on whether an
    earlier test in this process already consumed it, and these tests would pass
    or fail by ordering rather than by behaviour.
    """
    calls: list[str] = []

    def _recording_sweep(prefix: str) -> int:
        calls.append(prefix)
        return 0

    monkeypatch.setattr(probe, '_probe_dir_sweep_done', False, raising=False)
    monkeypatch.setattr(probe, 'sweep_stale_pid_dirs', _recording_sweep, raising=False)
    return calls


class TestStaleProbeDirSweep:
    """Nothing reclaims a ``startup-probe-`` dir left behind by a SIGKILLed run.

    ``atexit`` covers a clean exit and the ``finally`` covers a raise, but no
    hook survives SIGKILL — and a killed probe leaves a 0600 ``.credentials.json``
    holding a live OAuth token in /tmp forever.  ``sweep_stale_pid_dirs`` exists
    for exactly this and is called in production, but it is PREFIX-SCOPED by
    contract and its only caller (``usage_gate``) sweeps ``usage-gate-probe-``,
    so it never touches this probe's dirs.

    Mirrors ``usage_gate._sweep_stale_probe_dirs_once``'s shape rather than
    inventing one: same constant pair, same once-per-process flag set BEFORE the
    call, same never-raise contract.
    """

    def test_swept_prefix_and_created_names_are_the_same_string(self):
        # Not a tautology: it pins that the sweep key is DERIVED from the task-id
        # stem rather than retyped, which is how the two silently drift apart.
        assert probe._PROBE_DIR_PREFIX == CONFIG_DIR_PREFIX + probe._PROBE_TASK_ID_PREFIX

    def test_created_dir_name_is_attributable_to_this_pid(
        self, probe_recorder, monkeypatch, tmp_path
    ):
        exc_type = _inject('build_argv', monkeypatch, probe_recorder)
        with pytest.raises(exc_type, match='injected'):
            _run_probe(tmp_path)
        name = probe_recorder.configs[-1].path.name
        assert name.startswith(probe._PROBE_DIR_PREFIX), (
            f'{name!r} is outside the swept prefix, so the sweep can never reclaim it'
        )
        match = _PID_SUFFIX_RE.search(name)
        assert match is not None, (
            f'{name!r} has no -<pid> suffix, so sweep_stale_pid_dirs treats it as '
            'unattributable and deliberately leaves it alone'
        )
        assert match.group(1) == str(os.getpid())

    def test_run_live_probe_sweeps_once_with_the_probe_prefix(
        self, probe_recorder, sweep_calls, monkeypatch, tmp_path
    ):
        exc_type = _inject('build_argv', monkeypatch, probe_recorder)
        with pytest.raises(exc_type, match='injected'):
            _run_probe(tmp_path)
        assert sweep_calls == [probe._PROBE_DIR_PREFIX]

    def test_the_sweep_is_once_per_process(
        self, probe_recorder, sweep_calls, monkeypatch, tmp_path
    ):
        exc_type = _inject('build_argv', monkeypatch, probe_recorder)
        for _ in range(2):
            with pytest.raises(exc_type, match='injected'):
                _run_probe(tmp_path)
        # The sweep reclaims OTHER (dead) processes' leftovers, so re-running it
        # per probe re-scans /tmp for no benefit.
        assert sweep_calls == [probe._PROBE_DIR_PREFIX]

    def test_a_raising_sweep_does_not_fail_the_probe(
        self, probe_recorder, sweep_calls, monkeypatch, tmp_path
    ):
        def _exploding_sweep(prefix: str) -> int:
            sweep_calls.append(prefix)
            raise RuntimeError('sweep exploded')

        monkeypatch.setattr(probe, 'sweep_stale_pid_dirs', _exploding_sweep, raising=False)
        _inject('build_argv', monkeypatch, probe_recorder)
        # The INJECTED failure must surface, not the sweep's: tmp hygiene must
        # never be able to fail a capture that has already been paid for.
        with pytest.raises(RuntimeError, match='injected: _build_argv'):
            _run_probe(tmp_path)
        assert sweep_calls == [probe._PROBE_DIR_PREFIX]
        assert probe._probe_dir_sweep_done is True, (
            'the flag must be set BEFORE the call, or a raising sweep re-runs on '
            'every subsequent probe'
        )


# ---------------------------------------------------------------------------
# Teardown that itself fails
# ---------------------------------------------------------------------------


class _ExplodingCloseHandle:
    """A capture-file handle whose FIRST ``close()`` raises, then behaves.

    Models the shape the capture files are most likely to hit in the field: a
    buffered writer flushes on close, so a full ``/tmp`` surfaces as ENOSPC
    *there* rather than at write time — the very scenario ``run_live_probe``'s
    own comment names as motivating the capture-file design.  Everything else
    proxies to the real handle, so the probe's use of it is unchanged.
    """

    def __init__(self, handle: IO[bytes]) -> None:
        self._handle = handle
        self.close_attempts = 0

    def close(self) -> None:
        self.close_attempts += 1
        if self.close_attempts == 1:
            raise OSError('injected: close() failed (ENOSPC on flush)')
        self._handle.close()

    @property
    def closed(self) -> bool:
        return self._handle.closed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


class _FakePopen:
    """A spawned child that never exits, recording whether teardown reaped it.

    ``pid`` is ``-1`` so ``sample_proc`` finds no ``/proc`` entry and stays cheap;
    ``poll()`` never reports an exit, so the sampling loop leaves through its
    ``deadline`` branch with the child still "running" — the state in which the
    ``finally``'s ``kill()`` is the only thing standing between a raise and an
    orphaned process.
    """

    def __init__(self) -> None:
        self.pid = -1
        self.returncode: int | None = None
        self.kill_calls = 0
        self.wait_calls = 0

    def poll(self) -> int | None:
        return None

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int | None:
        self.wait_calls += 1
        return self.returncode


def _explode_first_capture_close(monkeypatch, sink: list[Any]) -> None:
    """Make the FIRST capture handle's ``close()`` raise; collect both handles.

    The first close is the FIRST statement of the teardown, so this is the
    injection that proves the whole block is reachable past a broken step rather
    than just its tail.
    """
    real_path_open = Path.open

    def _recording_open(self: Path, *args: Any, **kwargs: Any):
        mode = args[0] if args else kwargs.get('mode', 'r')
        if mode != 'wb':
            return real_path_open(self, *args, **kwargs)
        handle = real_path_open(self, *args, **kwargs)
        wrapped = _ExplodingCloseHandle(handle) if not sink else handle
        sink.append(wrapped)
        return wrapped

    monkeypatch.setattr(Path, 'open', _recording_open)


class _TeardownCase:
    """One (in-flight failure, teardown-step failure) pair under test.

    ``in_flight`` is the message of the exception ``run_live_probe`` is already
    unwinding when the teardown breaks — the one that must reach the caller
    unmasked, because it names the actual cause and the teardown OSError does not.
    """

    def __init__(
        self,
        *,
        kind: str,
        in_flight: str,
        probe_kwargs: dict[str, Any] | None = None,
        handles: list[Any] | None = None,
    ) -> None:
        self.kind = kind
        self.in_flight = in_flight
        self.probe_kwargs = probe_kwargs or {}
        self.handles = handles if handles is not None else []


def _inject_teardown_failure(
    kind: str, monkeypatch, recorder: _ProbeRecorder
) -> _TeardownCase:
    """Install an in-flight failure AND a failure inside the teardown itself.

    Layers on top of the ``probe_recorder`` fixture: the ``mkdtemp`` /
    ``Path.open`` / ``Path.unlink`` wrappers here wrap whatever that fixture has
    already installed, so the recorder keeps seeing everything the probe allocates.
    """
    if kind == 'stub_subdir':
        # The case the suppress comment in the `finally` claims to cover and does
        # not: the suppress sits around `rmdir`, one statement AFTER the glob loop
        # whose `unlink` is what actually raises on a subdirectory
        # (IsADirectoryError, an OSError).
        recording_mkdtemp = tempfile.mkdtemp

        def _planting_mkdtemp(*args: Any, **kwargs: Any) -> str:
            path = recording_mkdtemp(*args, **kwargs)
            # Sorted first, so the glob loop trips on it BEFORE the ordinary
            # entry — otherwise a naive loop would look isolated by luck.
            (Path(path) / '0-planted-subdir').mkdir()
            (Path(path) / '1-planted-file').write_text('a file the child recreated')
            return path

        monkeypatch.setattr(probe.tempfile, 'mkdtemp', _planting_mkdtemp)
        _inject('spawn_env', monkeypatch, recorder)
        return _TeardownCase(kind=kind, in_flight='injected: _spawn_env')

    if kind == 'handle_close':
        # The in-flight failure has to land AFTER the capture files are open, so
        # `Popen` itself is replaced by a raise — nothing is ever spawned, which
        # is the point.  RuntimeError (not the FileNotFoundError a missing binary
        # would really give) keeps the not-masked assertion crisp against the
        # teardown's OSError.
        def _boom_popen(*args: Any, **kwargs: Any):
            raise RuntimeError('injected: subprocess.Popen')

        monkeypatch.setattr(probe.subprocess, 'Popen', _boom_popen)
        _record_argv_temp_paths(monkeypatch, recorder)
        handles: list[Any] = []
        _explode_first_capture_close(monkeypatch, handles)
        return _TeardownCase(
            kind=kind, in_flight='injected: subprocess.Popen', handles=handles
        )

    if kind == 'temp_path_unlink':
        real_unlink = Path.unlink

        def _failing_unlink(self: Path, *args: Any, **kwargs: Any):
            if recorder.temp_paths and self == recorder.temp_paths[0]:
                raise OSError('injected: unlink() failed')
            return real_unlink(self, *args, **kwargs)

        _inject('spawn_env', monkeypatch, recorder)
        monkeypatch.setattr(Path, 'unlink', _failing_unlink)
        # mcp_wedge is the mode whose _build_argv produces TWO temp files, so a
        # failure on the first has a real second to skip — no fake path needed.
        return _TeardownCase(
            kind=kind, in_flight='injected: _spawn_env', probe_kwargs={'mode': 'mcp_wedge'}
        )

    raise AssertionError(f'unknown teardown-failure kind {kind!r}')


@pytest.fixture(params=['stub_subdir', 'handle_close', 'temp_path_unlink'])
def teardown_failure(request, monkeypatch, probe_recorder) -> _TeardownCase:
    return _inject_teardown_failure(request.param, monkeypatch, probe_recorder)


class TestTeardownIsFailureIsolated:
    """A raise INSIDE the teardown must not cancel the rest of the teardown.

    Every injection above fires while the ``try`` body is still running, so
    nothing yet exercises the ``finally`` when the ``finally`` is what breaks.
    And it can break: only ``stub_dir.rmdir()`` sits inside a
    ``contextlib.suppress(OSError)`` — the handle closes, the ``proc.kill()``,
    the ``temp_paths`` loop and the stub-dir glob ahead of it are unguarded.  A
    subdirectory in the stub dir is enough (``Path.unlink`` on a directory raises
    IsADirectoryError), and the consequences are exactly the ones the block's own
    comment says are impossible: the original exception is REPLACED by a
    misleading OSError from tidy-up, the child is never reaped, and
    ``config.cleanup()`` is skipped — leaving a 0600 ``.credentials.json``
    holding a live OAuth token on disk.
    """

    def test_the_original_exception_is_not_masked(
        self, probe_recorder, teardown_failure, tmp_path
    ):
        with pytest.raises(RuntimeError) as excinfo:
            _run_probe(tmp_path, **teardown_failure.probe_kwargs)
        assert str(excinfo.value) == teardown_failure.in_flight, (
            'the teardown failure replaced the exception it was unwinding, so the '
            'caller is told about tidy-up instead of the actual cause'
        )

    def test_the_credential_dir_is_still_reclaimed(
        self, probe_recorder, teardown_failure, tmp_path
    ):
        # Whatever surfaced, the one piece of teardown with security consequences
        # must have run.
        with pytest.raises((RuntimeError, OSError)):
            _run_probe(tmp_path, **teardown_failure.probe_kwargs)
        config_path = probe_recorder.configs[-1].path
        assert not config_path.exists(), (
            f'a failure while tidying up skipped config.cleanup(): {config_path} '
            'still holds a 0600 .credentials.json'
        )

    def test_the_failing_step_is_reported_to_stderr(
        self, probe_recorder, monkeypatch, tmp_path, capsys
    ):
        # Loud, not silent: a probe that quietly failed to tidy up leaves no
        # trace of what it left behind (no-silent-fail-soft).
        case = _inject_teardown_failure('stub_subdir', monkeypatch, probe_recorder)
        with pytest.raises(RuntimeError, match=case.in_flight):
            _run_probe(tmp_path, **case.probe_kwargs)
        shutil.rmtree(probe_recorder.stub_dirs[-1], ignore_errors=True)
        err = capsys.readouterr().err
        assert 'WARNING' in err and 'teardown' in err, (
            f'the swallowed teardown failure was never reported; stderr was {err!r}'
        )

    def test_stub_entries_after_the_failing_one_are_still_unlinked(
        self, probe_recorder, monkeypatch, tmp_path
    ):
        case = _inject_teardown_failure('stub_subdir', monkeypatch, probe_recorder)
        with pytest.raises(RuntimeError, match=case.in_flight):
            _run_probe(tmp_path, **case.probe_kwargs)
        stub_dir = probe_recorder.stub_dirs[-1]
        leftover = stub_dir / '1-planted-file'
        survived = leftover.exists()
        # The planted SUBDIR legitimately survives — nothing in the teardown
        # removes a subtree — so reclaim it here rather than littering /tmp.
        shutil.rmtree(stub_dir, ignore_errors=True)
        assert not survived, (
            'one unlinkable stub entry cancelled the rest of the glob loop'
        )
        assert not probe_recorder.configs[-1].path.exists()

    def test_the_other_capture_handle_is_still_closed(
        self, probe_recorder, monkeypatch, tmp_path
    ):
        case = _inject_teardown_failure('handle_close', monkeypatch, probe_recorder)
        with pytest.raises(RuntimeError, match=case.in_flight):
            _run_probe(tmp_path, **case.probe_kwargs)
        assert len(case.handles) == 2, 'both capture files should have been opened'
        assert case.handles[0].close_attempts == 1, 'the injected close failure never fired'
        assert case.handles[1].closed, (
            'a failing stdout close skipped the stderr close — the second handle '
            'was left to the GC'
        )
        for path in probe_recorder.temp_paths:
            assert not path.exists(), f'an argv temp file survived: {path}'
        assert not probe_recorder.stub_dirs[-1].exists()
        assert not probe_recorder.configs[-1].path.exists()

    def test_temp_paths_after_the_failing_one_are_still_unlinked(
        self, probe_recorder, monkeypatch, tmp_path
    ):
        case = _inject_teardown_failure('temp_path_unlink', monkeypatch, probe_recorder)
        with pytest.raises(RuntimeError, match=case.in_flight):
            _run_probe(tmp_path, **case.probe_kwargs)
        assert len(probe_recorder.temp_paths) == 2, (
            'mcp_wedge should build two temp files; without a second one this '
            'test could not tell a skipped loop from a finished one'
        )
        blocked, following = probe_recorder.temp_paths
        blocked_survived = blocked.exists()
        following_survived = following.exists()
        # os.unlink, not Path.unlink: the failing wrapper is still installed.
        for path in (blocked, following):
            with contextlib.suppress(OSError):
                os.unlink(path)
        assert blocked_survived, 'the injected unlink failure never fired'
        assert not following_survived, (
            'one unlinkable temp path cancelled the rest of the loop'
        )
        assert not probe_recorder.stub_dirs[-1].exists()
        assert not probe_recorder.configs[-1].path.exists()

    def test_a_failing_first_step_still_reaps_the_child(
        self, probe_recorder, monkeypatch, tmp_path
    ):
        """The orphaned-child half, which no config-dir assertion reaches.

        ``_drain_exit`` normally reaps the child — but it is exactly what does
        NOT run when the ``try`` body raises, which is when the ``finally``'s
        ``kill()`` is the only reaper left.  Put a raise in the FIRST teardown
        step and that reaper is skipped, leaking a live CLI process (with the
        OAuth token in its environment) for the life of the machine.
        """
        spawned: list[_FakePopen] = []

        def _fake_popen(*args: Any, **kwargs: Any) -> _FakePopen:
            proc = _FakePopen()
            spawned.append(proc)
            return proc

        monkeypatch.setattr(probe.subprocess, 'Popen', _fake_popen)

        def _boom_drain_exit(*args: Any, **kwargs: Any) -> dict:
            raise RuntimeError('injected: _drain_exit')

        monkeypatch.setattr(probe, '_drain_exit', _boom_drain_exit)
        handles: list[Any] = []
        _explode_first_capture_close(monkeypatch, handles)

        with pytest.raises(RuntimeError, match='injected: _drain_exit'):
            _run_probe(tmp_path)

        assert spawned, 'the run never reached the spawn — bad injection'
        assert handles and handles[0].close_attempts == 1, 'the close failure never fired'
        assert spawned[-1].kill_calls == 1, (
            'a raise in the first teardown step orphaned the child: nothing else '
            'in this process will ever reap it'
        )
        assert spawned[-1].wait_calls == 1
        assert not probe_recorder.configs[-1].path.exists()
