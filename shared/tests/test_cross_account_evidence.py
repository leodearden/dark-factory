"""Contract tests for ``_cross_account_evidence`` — the cross-account measurement kit.

An ORDINARY (unmarked) test module: it runs in CI and in the verify lane, makes
no real CLI calls, and costs nothing.  The live measurement it supports lives in
``test_cli_invoke_integration.py::TestCrossAccountResume`` behind
``-m integration``.

What these tests are for (task 3484).  The cross-account resume question has now
gone two rounds with **0 valid runs** (task 3454, then again on 2026-08-05)
because the integration module hard-coded its account pair to the first two
tokens in env, and those were capped.  ``select_token_pair`` makes the pair
steerable so a measurement can be aimed at whichever accounts are actually
healthy when a window opens; ``format_run_evidence`` / ``emit_run_evidence``
make each run leave a durable record.  Both are pure/injected, so both are
pinnable here rather than only observable during live spend.
"""

from __future__ import annotations

import io
import json

import pytest
from _capacity_skip import REAL_CLI_CAP_MESSAGES
from _cross_account_evidence import (
    emit_run_evidence,
    format_run_evidence,
    select_token_pair,
)

from shared.cli_invoke import AgentResult


def _env(*letters: str, **extra: str) -> dict[str, str]:
    """Build a fake environ holding ``CLAUDE_OAUTH_TOKEN_<L>`` for each letter."""
    environ = {f'CLAUDE_OAUTH_TOKEN_{ch}': f'tok-{ch.lower()}' for ch in letters}
    environ.update(extra)
    return environ


def _result(output: str = '', *, success: bool = True, **kwargs) -> AgentResult:
    return AgentResult(success=success, output=output, **kwargs)


def _record(**overrides) -> dict:
    """A ``format_run_evidence`` call with sane defaults, overridable per-test."""
    kwargs: dict = {
        'account_a': 'CLAUDE_OAUTH_TOKEN_F',
        'account_b': 'CLAUDE_OAUTH_TOKEN_C',
        'r1': _result('OK', session_id='sid-1'),
        'r2': _result('ZEPPELIN'),
        'r1_transcript_present': True,
        'r1_transcript_records': 12,
        'control_passed': True,
    }
    kwargs.update(overrides)
    return format_run_evidence(**kwargs)


class TestSelectTokenPairDefault:
    """No override set — today's ``_AVAILABLE_TOKENS[0]/[1]`` behaviour, exactly."""

    def test_returns_first_two_in_scan_order(self):
        (name_a, tok_a), (name_b, tok_b) = select_token_pair(_env('B', 'C', 'D'))
        assert (name_a, tok_a) == ('CLAUDE_OAUTH_TOKEN_B', 'tok-b')
        assert (name_b, tok_b) == ('CLAUDE_OAUTH_TOKEN_C', 'tok-c')

    def test_scan_order_is_bcdefg_not_environ_order(self):
        """Order comes from the fixed B,C,D,E,F,G scan, not dict insertion order."""
        environ = {
            'CLAUDE_OAUTH_TOKEN_F': 'tok-f',
            'CLAUDE_OAUTH_TOKEN_C': 'tok-c',
            'CLAUDE_OAUTH_TOKEN_E': 'tok-e',
        }
        pair = select_token_pair(environ)
        assert [name for name, _ in pair] == [
            'CLAUDE_OAUTH_TOKEN_C',
            'CLAUDE_OAUTH_TOKEN_E',
        ]

    def test_g_is_reachable(self):
        """G is a real account in .env that the BCDEF scan could not reach at all."""
        pair = select_token_pair(_env('F', 'G'))
        assert [name for name, _ in pair] == [
            'CLAUDE_OAUTH_TOKEN_F',
            'CLAUDE_OAUTH_TOKEN_G',
        ]

    def test_empty_string_token_does_not_count_as_available(self):
        environ = _env('C', 'D')
        environ['CLAUDE_OAUTH_TOKEN_B'] = ''
        pair = select_token_pair(environ)
        assert [name for name, _ in pair] == [
            'CLAUDE_OAUTH_TOKEN_C',
            'CLAUDE_OAUTH_TOKEN_D',
        ]

    def test_fewer_than_two_available_raises(self):
        """Callers turn this into a skip — it must not silently return one account."""
        with pytest.raises(ValueError) as exc:
            select_token_pair(_env('C'))
        assert 'CLAUDE_OAUTH_TOKEN_C' in str(exc.value) or 'two' in str(exc.value).lower()

    def test_no_tokens_at_all_raises(self):
        with pytest.raises(ValueError):
            select_token_pair({})


class TestSelectTokenPairOverride:
    """``CROSS_ACCOUNT_RESUME_TOKENS`` aims the measurement at a healthy pair."""

    def test_override_selects_named_pair_in_order(self):
        """Order is load-bearing: A starts the session, B resumes it."""
        environ = _env('B', 'C', 'D', 'E', 'F', 'G')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = 'F,C'
        (name_a, tok_a), (name_b, tok_b) = select_token_pair(environ)
        assert (name_a, tok_a) == ('CLAUDE_OAUTH_TOKEN_F', 'tok-f')
        assert (name_b, tok_b) == ('CLAUDE_OAUTH_TOKEN_C', 'tok-c')

    def test_override_order_is_not_normalised_to_scan_order(self):
        """'F,C' must NOT come back as (C, F) — that would swap which account resumes."""
        environ = _env('C', 'F')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = 'F,C'
        assert [name for name, _ in select_token_pair(environ)] == [
            'CLAUDE_OAUTH_TOKEN_F',
            'CLAUDE_OAUTH_TOKEN_C',
        ]

    def test_override_accepts_full_var_names(self):
        environ = _env('C', 'F')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = (
            'CLAUDE_OAUTH_TOKEN_F,CLAUDE_OAUTH_TOKEN_C'
        )
        assert [name for name, _ in select_token_pair(environ)] == [
            'CLAUDE_OAUTH_TOKEN_F',
            'CLAUDE_OAUTH_TOKEN_C',
        ]

    def test_override_tolerates_whitespace_and_case(self):
        environ = _env('C', 'F')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = '  f , C  '
        assert [name for name, _ in select_token_pair(environ)] == [
            'CLAUDE_OAUTH_TOKEN_F',
            'CLAUDE_OAUTH_TOKEN_C',
        ]

    def test_override_naming_unset_var_raises_and_names_the_entry(self):
        """A silent fall-back to the default pair is how a run gets aimed at a
        capped account without anyone noticing.  Fail loudly instead."""
        environ = _env('C', 'F')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = 'F,D'
        with pytest.raises(ValueError) as exc:
            select_token_pair(environ)
        message = str(exc.value)
        assert 'D' in message
        assert 'CROSS_ACCOUNT_RESUME_TOKENS' in message

    def test_override_naming_unknown_letter_raises_and_names_the_entry(self):
        environ = _env('C', 'F')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = 'F,ZZZ'
        with pytest.raises(ValueError) as exc:
            select_token_pair(environ)
        assert 'ZZZ' in str(exc.value)

    def test_override_naming_same_account_twice_raises(self):
        """A same-account 'cross-account' run is not a measurement."""
        environ = _env('C', 'F')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = 'C,C'
        with pytest.raises(ValueError) as exc:
            select_token_pair(environ)
        assert 'CLAUDE_OAUTH_TOKEN_C' in str(exc.value)

    def test_override_with_wrong_arity_raises(self):
        environ = _env('C', 'F', 'G')
        for value in ('C', 'C,F,G', ''):
            environ['CROSS_ACCOUNT_RESUME_TOKENS'] = value
            with pytest.raises(ValueError):
                select_token_pair(environ)

    def test_override_wins_even_when_default_pair_is_available(self):
        """The whole point: B,C are present and would be the default, but the
        operator has measured them capped and aimed at F,G instead."""
        environ = _env('B', 'C', 'F', 'G')
        environ['CROSS_ACCOUNT_RESUME_TOKENS'] = 'G,F'
        assert [name for name, _ in select_token_pair(environ)] == [
            'CLAUDE_OAUTH_TOKEN_G',
            'CLAUDE_OAUTH_TOKEN_F',
        ]


class TestFormatRunEvidenceFields:
    """The record must carry exactly the fields task 3484 has to record."""

    def test_carries_every_required_field(self):
        record = _record()
        assert set(record) == {
            'account_a',
            'account_b',
            'r1_session_id',
            'r1_transcript_present',
            'r1_transcript_records',
            'r2_output',
            'r2_stderr',
            'r2_success',
            'codeword_recalled',
            'control_passed',
            'verdict',
        }

    def test_records_which_accounts_were_actually_used(self):
        record = _record(
            account_a='CLAUDE_OAUTH_TOKEN_G', account_b='CLAUDE_OAUTH_TOKEN_E'
        )
        assert record['account_a'] == 'CLAUDE_OAUTH_TOKEN_G'
        assert record['account_b'] == 'CLAUDE_OAUTH_TOKEN_E'

    def test_records_r1_session_id_and_r2_streams(self):
        record = _record(
            r1=_result('OK', session_id='eeec059e-be5d-413d-bae7-15274dd758c3'),
            r2=_result('ZEPPELIN', stderr='some warning', success=True),
        )
        assert record['r1_session_id'] == 'eeec059e-be5d-413d-bae7-15274dd758c3'
        assert record['r2_stderr'] == 'some warning'
        assert record['r2_success'] is True

    def test_round_trips_through_json(self):
        record = _record()
        assert json.loads(json.dumps(record)) == record


class TestFormatRunEvidenceVerbatim:
    """Outputs are stored VERBATIM — truncation is how a cap message hides."""

    def test_long_output_is_not_truncated(self):
        long_output = 'ZEPPELIN ' + ('x' * 20_000)
        record = _record(r2=_result(long_output))
        assert record['r2_output'] == long_output

    def test_surrounding_whitespace_is_not_stripped(self):
        record = _record(r2=_result('\n  ZEPPELIN  \n'))
        assert record['r2_output'] == '\n  ZEPPELIN  \n'

    def test_stderr_is_not_truncated_or_redacted(self):
        stderr = 'boom ' + ('y' * 20_000)
        record = _record(r2=_result('ZEPPELIN', stderr=stderr))
        assert record['r2_stderr'] == stderr


class TestFormatRunEvidenceTranscript:
    """Absent and empty are different findings, so they get different values."""

    def test_absent_transcript_yields_none_not_zero(self):
        record = _record(r1_transcript_present=False, r1_transcript_records=0)
        assert record['r1_transcript_present'] is False
        assert record['r1_transcript_records'] is None

    def test_absent_transcript_with_none_count_stays_none(self):
        record = _record(r1_transcript_present=False, r1_transcript_records=None)
        assert record['r1_transcript_records'] is None

    def test_present_but_empty_transcript_keeps_zero(self):
        record = _record(r1_transcript_present=True, r1_transcript_records=0)
        assert record['r1_transcript_present'] is True
        assert record['r1_transcript_records'] == 0

    def test_present_transcript_keeps_its_count(self):
        record = _record(r1_transcript_present=True, r1_transcript_records=12)
        assert record['r1_transcript_records'] == 12


class TestFormatRunEvidenceVerdict:
    """The verdict encodes how a run must be READ, including 'do not read it'."""

    def test_codeword_recalled_is_case_insensitive(self):
        assert _record(r2=_result('the codeword was zeppelin'))['codeword_recalled'] is True

    def test_recalled_codeword_yields_preserved(self):
        record = _record(r2=_result('ZEPPELIN'))
        assert record['codeword_recalled'] is True
        assert record['verdict'] == 'preserved'

    def test_missing_codeword_yields_not_preserved(self):
        record = _record(r2=_result("I don't have any prior context."))
        assert record['codeword_recalled'] is False
        assert record['verdict'] == 'not_preserved'

    @pytest.mark.parametrize('cap_message', REAL_CLI_CAP_MESSAGES)
    def test_real_cap_messages_yield_void_capped(self, cap_message):
        """A capped account is NOT evidence of context loss — it is a void run.

        Parametrised over the single-homed corpus in ``_capacity_skip`` (task
        3483) rather than a local copy of the strings: a second hand-maintained
        copy of a verbatim transcript is the drift that voided the 3454 attempt.
        """
        record = _record(r2=_result(cap_message, success=False))
        assert record['verdict'] == 'void_capped'

    def test_cap_message_on_stderr_also_yields_void_capped(self):
        record = _record(r2=_result('', stderr=REAL_CLI_CAP_MESSAGES[0], success=False))
        assert record['verdict'] == 'void_capped'

    def test_void_capped_wins_over_not_preserved(self):
        """This precedence is the whole point: the 2026-08-01 run looked like
        lost context and was really a weekly cap."""
        record = _record(
            r2=_result(
                "You've hit your weekly limit · resets Aug 5, 11am", success=False
            )
        )
        assert record['codeword_recalled'] is False
        assert record['verdict'] == 'void_capped'

    def test_control_passed_tracks_none_false_true_distinctly(self):
        assert _record(control_passed=None)['control_passed'] is None
        assert _record(control_passed=False)['control_passed'] is False
        assert _record(control_passed=True)['control_passed'] is True


class TestEmitRunEvidence:
    """One JSON line to the stream; also to the evidence file when asked."""

    def test_writes_one_json_line_to_stream(self):
        stream = io.StringIO()
        record = _record()
        emit_run_evidence(record, {}, stream)
        payloads = [
            line for line in stream.getvalue().splitlines() if line.strip()
        ]
        assert len(payloads) == 1
        assert json.loads(payloads[0].split(' ', 1)[1]) == record

    def test_returns_the_json_line(self):
        record = _record()
        line = emit_run_evidence(record, {}, io.StringIO())
        assert json.loads(line) == record
        assert '\n' not in line

    def test_appends_to_evidence_path_when_set(self, tmp_path):
        path = tmp_path / 'nested' / 'evidence.jsonl'
        environ = {'CROSS_ACCOUNT_EVIDENCE_PATH': str(path)}
        first = _record(r1=_result('OK', session_id='sid-1'))
        second = _record(r1=_result('OK', session_id='sid-2'))
        emit_run_evidence(first, environ, io.StringIO())
        emit_run_evidence(second, environ, io.StringIO())
        lines = path.read_text(encoding='utf-8').strip().splitlines()
        assert [json.loads(line)['r1_session_id'] for line in lines] == [
            'sid-1',
            'sid-2',
        ]

    def test_does_not_touch_filesystem_when_path_unset(self, tmp_path):
        emit_run_evidence(_record(), {}, io.StringIO())
        assert list(tmp_path.iterdir()) == []

    def test_blank_evidence_path_is_treated_as_unset(self, tmp_path):
        emit_run_evidence(
            _record(), {'CROSS_ACCOUNT_EVIDENCE_PATH': ''}, io.StringIO()
        )
        assert list(tmp_path.iterdir()) == []

    def test_verbatim_multiline_output_stays_on_one_line_in_the_file(self, tmp_path):
        """A record whose r2_output contains newlines must not split the JSONL."""
        path = tmp_path / 'evidence.jsonl'
        record = _record(r2=_result('line one\nline two\nZEPPELIN'))
        emit_run_evidence(
            record, {'CROSS_ACCOUNT_EVIDENCE_PATH': str(path)}, io.StringIO()
        )
        lines = path.read_text(encoding='utf-8').strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])['r2_output'] == 'line one\nline two\nZEPPELIN'


@pytest.fixture
def reload_integration_module(monkeypatch):
    """Reload ``test_cli_invoke_integration`` under a controlled fake environ.

    Yields ``reload(**tokens)`` -> the reloaded module.  Every
    ``CLAUDE_OAUTH_TOKEN_*`` letter and the override var are cleared first, so
    the machine's real ``.env`` (which has B..G all set) cannot leak in and make
    these assertions environment-dependent.

    Reloading mutates the module object other collected items may hold, so on
    teardown the real environment is restored (``monkeypatch.undo()`` first,
    since this fixture finalises BEFORE monkeypatch's own teardown) and the
    module is reloaded once more under it — no reload side effect escapes.
    """
    import importlib

    module = importlib.import_module('test_cli_invoke_integration')

    def _reload(**tokens: str):
        for ch in 'ABCDEFG':
            monkeypatch.delenv(f'CLAUDE_OAUTH_TOKEN_{ch}', raising=False)
        monkeypatch.delenv('CROSS_ACCOUNT_RESUME_TOKENS', raising=False)
        for name, value in tokens.items():
            monkeypatch.setenv(name, value)
        return importlib.reload(module)

    try:
        yield _reload
    finally:
        monkeypatch.undo()
        importlib.reload(module)


class TestIntegrationModuleUsesSelectTokenPair:
    """The integration module must RESOLVE its pair through ``select_token_pair``.

    Not a structure or docstring test — a runtime-behaviour test of module-level
    selection.  It is the only thing that would catch the helper sitting unused
    beside a still-hard-coded ``_AVAILABLE_TOKENS[0]/[1]``, i.e. a live
    measurement silently running on the capped default pair.  That is the precise
    failure this task exists to end: two rounds, 0 valid runs, both because the
    pair could not be aimed.
    """

    def test_override_steers_the_module_level_pair(self, reload_integration_module):
        module = reload_integration_module(
            CLAUDE_OAUTH_TOKEN_B='tok-b',
            CLAUDE_OAUTH_TOKEN_C='tok-c',
            CLAUDE_OAUTH_TOKEN_F='tok-f',
            CROSS_ACCOUNT_RESUME_TOKENS='F,C',
        )
        assert module._ACCOUNT_PAIR is not None
        (name_a, tok_a), (name_b, tok_b) = module._ACCOUNT_PAIR
        assert (name_a, tok_a) == ('CLAUDE_OAUTH_TOKEN_F', 'tok-f')
        assert (name_b, tok_b) == ('CLAUDE_OAUTH_TOKEN_C', 'tok-c')

    def test_falls_back_to_first_two_available_without_override(
        self, reload_integration_module
    ):
        module = reload_integration_module(
            CLAUDE_OAUTH_TOKEN_C='tok-c',
            CLAUDE_OAUTH_TOKEN_E='tok-e',
            CLAUDE_OAUTH_TOKEN_F='tok-f',
        )
        assert module._ACCOUNT_PAIR is not None
        assert [name for name, _ in module._ACCOUNT_PAIR] == [
            'CLAUDE_OAUTH_TOKEN_C',
            'CLAUDE_OAUTH_TOKEN_E',
        ]

    def test_account_a_is_the_session_starting_account(
        self, reload_integration_module
    ):
        """The control/baseline tests must run on account A, not on token index 0."""
        module = reload_integration_module(
            CLAUDE_OAUTH_TOKEN_B='tok-b',
            CLAUDE_OAUTH_TOKEN_C='tok-c',
            CLAUDE_OAUTH_TOKEN_G='tok-g',
            CROSS_ACCOUNT_RESUME_TOKENS='G,B',
        )
        assert module._ACCOUNT_A == ('CLAUDE_OAUTH_TOKEN_G', 'tok-g')

    def test_bad_override_does_not_crash_collection(self, reload_integration_module):
        """A mis-set override must skip the cross-account test, not break the
        whole module's collection — and it must NOT silently run the default pair."""
        module = reload_integration_module(
            CLAUDE_OAUTH_TOKEN_B='tok-b',
            CLAUDE_OAUTH_TOKEN_C='tok-c',
            CROSS_ACCOUNT_RESUME_TOKENS='F,C',  # F is unset here
        )
        assert module._ACCOUNT_PAIR is None
        assert module._PAIR_ERROR is not None
        assert 'F' in module._PAIR_ERROR
        # Single-account tests still have somebody to run on.
        assert module._ACCOUNT_A == ('CLAUDE_OAUTH_TOKEN_B', 'tok-b')

    def test_single_token_leaves_pair_unset(self, reload_integration_module):
        module = reload_integration_module(CLAUDE_OAUTH_TOKEN_C='tok-c')
        assert module._ACCOUNT_PAIR is None
        assert module._ACCOUNT_A == ('CLAUDE_OAUTH_TOKEN_C', 'tok-c')

    def test_no_tokens_leaves_both_unset(self, reload_integration_module):
        module = reload_integration_module()
        assert module._ACCOUNT_PAIR is None
        assert module._ACCOUNT_A is None
