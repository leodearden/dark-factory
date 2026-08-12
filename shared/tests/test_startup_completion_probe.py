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
