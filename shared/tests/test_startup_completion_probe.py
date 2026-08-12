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

import json
from typing import Any

import startup_completion_fixtures as scf
import startup_completion_probe as probe

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
