"""Shared test kit: the cross-account resume MEASUREMENT harness (task 3484).

Two jobs, both pure enough to unit-test without spending live CLI budget (their
contract tests are in ``test_cross_account_evidence.py``, an ordinary unmarked
module):

1. ``select_token_pair`` — pick WHICH two OAuth accounts a cross-account resume
   measurement runs on.  ``test_cli_invoke_integration.py`` used to hard-code the
   first two tokens in env, which is why the question has twice been answered
   with 0 valid runs: on 2026-08-01 (task 3454) and again on 2026-08-05, the
   default pair happened to be capped.  A ``CROSS_ACCOUNT_RESUME_TOKENS='F,C'``
   override lets the operator aim the run at whichever pair a pre-flight probe
   found healthy, in the moment the window opens.

2. ``format_run_evidence`` / ``emit_run_evidence`` — leave a durable per-run
   record.  The live tests assert but emit nothing, so a green run printed
   nothing to paste into a verdict and a red run printed only an assertion
   message.  Both prior rounds were only diagnosable because a human read r2's
   transcript turn verbatim and recognised a cap message.

Imported by bare module name (``from _cross_account_evidence import ...``);
``conftest.py`` prepends ``shared/tests`` to ``sys.path``, same convention as
``_capacity_skip.py``.  The leading underscore keeps pytest from collecting it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import IO

from _capacity_skip import result_looks_like_capacity_failure

from shared.cli_invoke import AgentResult

#: The account letters scanned, in order, when no override is set.
#:
#: WIDER than the ``BCDEF`` scan ``test_cli_invoke_integration`` shipped with:
#: ``CLAUDE_OAUTH_TOKEN_G`` is a real account in ``.env`` that the old scan could
#: not reach at all, which needlessly shrank the pool this measurement draws
#: from.  ``A`` stays out deliberately — it is the interactive/primary account,
#: not a fleet worker, and the old scan excluded it too.
TOKEN_LETTERS: tuple[str, ...] = ('B', 'C', 'D', 'E', 'F', 'G')

#: Env var naming the pair to measure, e.g. ``'F,C'`` or
#: ``'CLAUDE_OAUTH_TOKEN_F,CLAUDE_OAUTH_TOKEN_C'``.  First entry is account A
#: (starts the session), second is account B (resumes it).
PAIR_OVERRIDE_VAR = 'CROSS_ACCOUNT_RESUME_TOKENS'

#: Optional env var; when set, ``emit_run_evidence`` appends each record as one
#: JSON line to this path in addition to writing it to the stream.
EVIDENCE_PATH_VAR = 'CROSS_ACCOUNT_EVIDENCE_PATH'

TokenPair = tuple[tuple[str, str], tuple[str, str]]


def available_tokens(environ: Mapping[str, str]) -> list[tuple[str, str]]:
    """``[(var_name, token), ...]`` for every set token var, in scan order.

    Pure: reads only the injected *environ*, never ``os.environ``.
    """
    return [
        (var, environ[var])
        for var in (f'CLAUDE_OAUTH_TOKEN_{ch}' for ch in TOKEN_LETTERS)
        if environ.get(var)
    ]


def _normalise_entry(entry: str) -> str:
    """``' f '`` / ``'F'`` / ``'CLAUDE_OAUTH_TOKEN_F'`` -> ``'CLAUDE_OAUTH_TOKEN_F'``."""
    stripped = entry.strip()
    if stripped.upper().startswith('CLAUDE_OAUTH_TOKEN_'):
        return stripped.upper()
    return f'CLAUDE_OAUTH_TOKEN_{stripped.upper()}'


def select_token_pair(environ: Mapping[str, str]) -> TokenPair:
    """Resolve the ``(account_a, account_b)`` pair for a cross-account measurement.

    Returns ``((name_a, token_a), (name_b, token_b))`` where each ``name`` is the
    env var NAME (so an evidence record can say which real accounts were used)
    and each ``token`` is its value.

    Order is load-bearing and never normalised: account A starts the session,
    account B issues the ``--resume``.  Reversing them measures a different thing.

    Without *PAIR_OVERRIDE_VAR*, returns the first two available tokens in
    ``TOKEN_LETTERS`` order — byte-identical to the ``_AVAILABLE_TOKENS[0]/[1]``
    behaviour the integration module shipped with, so opting out changes nothing.

    Raises:
        ValueError: if the override is malformed, names an unset/unknown var, or
            names the same account twice; or if no override is set and fewer than
            two tokens are available.  Callers turn this into a ``pytest.skip``.
            Failing loudly is the point — a silent fall-back to the default pair
            is precisely how a measurement gets aimed at a capped account without
            anyone noticing, which is what produced two rounds of 0 valid runs.
    """
    override = environ.get(PAIR_OVERRIDE_VAR)
    if override is not None and override.strip():
        entries = [part for part in override.split(',') if part.strip()]
        if len(entries) != 2:
            raise ValueError(
                f'{PAIR_OVERRIDE_VAR}={override!r} must name exactly TWO accounts '
                f"separated by a comma (e.g. 'F,C'); got {len(entries)}. "
                'The first starts the session, the second resumes it.'
            )
        names = [_normalise_entry(entry) for entry in entries]
        for entry, name in zip(entries, names, strict=True):
            if not environ.get(name):
                raise ValueError(
                    f'{PAIR_OVERRIDE_VAR} names {entry.strip()!r} -> {name}, '
                    'which is unset or empty in the environment. Available: '
                    f'{[var for var, _ in available_tokens(environ)]}. '
                    'Refusing to fall back to the default pair: a silent '
                    'fall-back is how a measurement gets aimed at a capped '
                    'account without anyone noticing.'
                )
        if names[0] == names[1]:
            raise ValueError(
                f'{PAIR_OVERRIDE_VAR}={override!r} names the same account twice '
                f'({names[0]}). A same-account "cross-account" resume is not a '
                'measurement — use the same-account control test for that.'
            )
        return ((names[0], environ[names[0]]), (names[1], environ[names[1]]))

    if override is not None:
        # Set but blank/whitespace: an operator meant to aim the run and the
        # value got lost (unexpanded shell var, empty CI secret).  Silently
        # running the default pair is the failure this function exists to stop.
        raise ValueError(
            f'{PAIR_OVERRIDE_VAR} is set but empty. Either unset it to use the '
            "default pair, or name two accounts (e.g. 'F,C')."
        )

    tokens = available_tokens(environ)
    if len(tokens) < 2:
        raise ValueError(
            'A cross-account resume measurement needs TWO accounts; found '
            f'{len(tokens)}: {[var for var, _ in tokens]}. Scanned '
            f'{[f"CLAUDE_OAUTH_TOKEN_{ch}" for ch in TOKEN_LETTERS]}.'
        )
    return (tokens[0], tokens[1])


def format_run_evidence(
    *,
    account_a: str,
    account_b: str,
    r1: AgentResult,
    r2: AgentResult,
    r1_transcript_present: bool,
    r1_transcript_records: int | None,
    control_passed: bool | None,
) -> dict:
    """Build the JSON-serialisable per-run record task 3484 must record verbatim.

    Every field the task requires, and nothing derived destructively:

    - ``account_a`` / ``account_b`` — the env var NAMES actually used, so the
      record is self-describing about which accounts produced it.
    - ``r1_session_id``, ``r1_transcript_present``, ``r1_transcript_records``.
    - ``r2_output`` / ``r2_stderr`` — **verbatim**: no truncation, no strip, no
      redaction.  A truncated cap message is what made the 3454 round ambiguous,
      and a cap phrasing the guard has never seen is the one remaining way a
      capped account can still masquerade as context loss.
    - ``r2_success``, ``codeword_recalled``.
    - ``control_passed`` — ``None`` when the same-account control did not run in
      this process (unknown is not the same as failed).
    - ``verdict`` — ``'void_capped'`` when the cap guard matches r2, else
      ``'preserved'`` / ``'not_preserved'`` from ``codeword_recalled``.

    ``r1_transcript_records`` is ``None`` (not ``0``) when the transcript is
    absent: "no transcript on disk" and "a transcript with no records" are
    different findings, and only the first would rule the run out on
    transcript-reachability grounds.

    ``void_capped`` takes precedence over ``not_preserved`` deliberately — the
    2026-08-01 cross-account run looked exactly like lost context and was really
    a weekly cap with no model turn.  The classification delegates to
    ``_capacity_skip.result_looks_like_capacity_failure``; do NOT grow a second
    cap-text matcher here.  Task 3483 single-homed that corpus and added a
    bidirectional drift guard against production's ``classify_invocation``
    precisely because a divergent marker list is what voided that attempt.
    """
    codeword_recalled = 'ZEPPELIN' in (r2.output or '').upper()
    if result_looks_like_capacity_failure(r2):
        verdict = 'void_capped'
    else:
        verdict = 'preserved' if codeword_recalled else 'not_preserved'
    return {
        'account_a': account_a,
        'account_b': account_b,
        'r1_session_id': r1.session_id,
        'r1_transcript_present': r1_transcript_present,
        'r1_transcript_records': (
            r1_transcript_records if r1_transcript_present else None
        ),
        'r2_output': r2.output,
        'r2_stderr': r2.stderr,
        'r2_success': r2.success,
        'codeword_recalled': codeword_recalled,
        'control_passed': control_passed,
        'verdict': verdict,
    }


def emit_run_evidence(
    record: Mapping[str, object],
    environ: Mapping[str, str],
    stream: IO[str],
) -> str:
    """Write *record* as ONE JSON line to *stream*; append it to the evidence file.

    The line goes to *stream* (pytest shows it under ``-s``) prefixed with
    ``CROSS_ACCOUNT_RESUME_EVIDENCE`` so it is greppable out of a noisy live run,
    and — when ``CROSS_ACCOUNT_EVIDENCE_PATH`` is set in *environ* — is appended
    verbatim to that path, parents created.  Absent (or blank) that var, the
    filesystem is not touched at all.

    ``json.dumps`` escapes embedded newlines, so a multi-line ``r2_output``
    cannot split the JSONL record across lines.

    Returns the JSON line (no trailing newline) so a caller can also attach it to
    an assertion message or to an escalation's structured evidence.
    """
    line = json.dumps(record, sort_keys=True)
    stream.write(f'\nCROSS_ACCOUNT_RESUME_EVIDENCE {line}\n')
    stream.flush()
    path_value = environ.get(EVIDENCE_PATH_VAR)
    if path_value:
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as fh:
            fh.write(f'{line}\n')
    return line


