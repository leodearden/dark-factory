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

2. ``format_run_evidence`` / ``format_r1_failure_evidence`` /
   ``emit_run_evidence`` — leave a durable per-run record.  The live tests
   assert but emit nothing, so a green run printed nothing to paste into a
   verdict and a red run printed only an assertion message.  Both prior rounds
   were only diagnosable because a human read r2's transcript turn verbatim and
   recognised a cap message.  A round that dies at r1 (account A capped) gets a
   record too — those are the rounds this question keeps losing, and a bare
   pytest skip line is not a record.

WHICH accounts exist at all is no longer decided here: the
``CLAUDE_OAUTH_TOKEN_*`` scan moved to ``_oauth_accounts`` (task 3700), which is
now its single home across the four places that had each hand-rolled it.  This
module keeps only the measurement-specific question — which PAIR of the
available accounts to run on — and imports the fleet letter set rather than
owning it.

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
from _oauth_accounts import FLEET_TOKEN_LETTERS, available_tokens, token_var_names

from shared.cli_invoke import AgentResult

#: Env var naming the pair to measure, e.g. ``'F,C'`` or
#: ``'CLAUDE_OAUTH_TOKEN_F,CLAUDE_OAUTH_TOKEN_C'``.  First entry is account A
#: (starts the session), second is account B (resumes it).
PAIR_OVERRIDE_VAR = 'CROSS_ACCOUNT_RESUME_TOKENS'

#: Optional env var; when set, ``emit_run_evidence`` appends each record as one
#: JSON line to this path in addition to writing it to the stream.
EVIDENCE_PATH_VAR = 'CROSS_ACCOUNT_EVIDENCE_PATH'

TokenPair = tuple[tuple[str, str], tuple[str, str]]


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
    ``FLEET_TOKEN_LETTERS`` order — byte-identical to the ``_AVAILABLE_TOKENS[0]/[1]``
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
            f'{list(token_var_names(FLEET_TOKEN_LETTERS))}.'
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
    - ``r2_success``, ``r2_subtype``, ``codeword_recalled``.
    - ``control_passed`` — ``None`` when the same-account control did not run in
      this process (unknown is not the same as failed).
    - ``verdict`` — one of ``'void_capped'`` / ``'void_error'`` / ``'preserved'``
      / ``'not_preserved'``; see the precedence note below.

    ``r1_transcript_records`` is ``None`` (not ``0``) when the transcript is
    absent: "no transcript on disk" and "a transcript with no records" are
    different findings, and only the first would rule the run out on
    transcript-reachability grounds.

    VERDICT PRECEDENCE, and why each step exists::

        preserved     the codeword is present
        void_capped   r2 matches the cap guard
        void_error    r2 FAILED for some non-cap reason
        not_preserved r2 SUCCEEDED and the codeword is absent

    ``preserved`` outranks BOTH void classes, uniformly: a recalled codeword
    cannot be produced without the prior context, so it is positive evidence
    even on an r2 that is otherwise marked failed or carries a limit message.
    That principle used to be applied to ``void_error`` but not to
    ``void_capped``, and the asymmetry made this helper disagree with the live
    test's own skip guard (``not r2.success and
    result_looks_like_capacity_failure(r2)``): an r2 that SUCCEEDED, answered
    ``ZEPPELIN``, and merely carried a NEAR-cap annotation on stderr
    ("You're close to reaching your usage limit." — which production classifies
    as annotation-only, not a cap hit, and which can accompany a perfectly
    healthy invocation) passed the test green while the emitted record called
    the run void.  The record IS the deliverable of this measurement, so a green
    run silently recorded as void is the same class of mis-scoring the
    ``void_error`` class was added to close.

    ``void_capped`` is deliberately NOT gated on ``not r2.success``, even though
    that would also close the disagreement above.  It outranks ``not_preserved``
    because the 2026-08-01 cross-account run looked exactly like lost context
    and was really a weekly cap with no model turn, and adding a success gate
    would make a cap that arrived on a run the CLI marked successful score
    ``not_preserved`` — reopening that exact trap on the strength of an
    assumption about CLI exit semantics that this suite has never measured.
    Ordering ``preserved`` first fixes the disagreement without betting on it:
    the residual difference (a cap-looking r2 that did NOT recall the codeword
    is recorded ``void_capped`` while the assertion goes red) is the fail-loud
    direction — a human reads the run.

    The cap classification delegates to
    ``_capacity_skip.result_looks_like_capacity_failure``; do NOT grow a second
    cap-text matcher here.  Task 3483 single-homed that corpus and added a
    bidirectional drift guard against production's ``classify_invocation``
    precisely because a divergent marker list is what voided that attempt.

    ``void_error`` closes the SAME trap one layer deeper (task 3484, steward
    decision at iteration 1).  An ``error_max_budget_usd`` abort is not a
    capacity failure, so the cap guard does not match it, and its output is
    EMPTY, so no codeword is found either — which scored it ``not_preserved``,
    i.e. a budget abort recorded as evidence of a production defect.  Measured
    2026-08-05: at the old ``max_budget_usd`` of $0.01 one run aborted and
    another spent $0.0083674 (84% of the ceiling), so this fired intermittently.
    Any ``r2.success is False`` that is not a cap now voids the run, which
    covers API errors, timeouts and empty output as well as budget.

    The consequence of the whole ladder is that ``not_preserved`` requires r2 to
    have SUCCEEDED, carried no limit message, and still not recalled — which is
    exactly, and only, the shape of the real production defect.
    """
    codeword_recalled = 'ZEPPELIN' in (r2.output or '').upper()
    if codeword_recalled:
        verdict = 'preserved'
    elif result_looks_like_capacity_failure(r2):
        verdict = 'void_capped'
    elif not r2.success:
        verdict = 'void_error'
    else:
        verdict = 'not_preserved'
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
        # The discriminator for a void_error run ('error_max_budget_usd',
        # 'error_max_turns', ...).  Recorded because it was previously visible
        # only in the live pytest output, which is not what gets pasted into a
        # verdict — leaving a reader unable to tell WHY a run was voided.
        'r2_subtype': r2.subtype,
        'codeword_recalled': codeword_recalled,
        'control_passed': control_passed,
        'verdict': verdict,
    }


def format_r1_failure_evidence(
    *,
    account_a: str,
    account_b: str,
    r1: AgentResult,
    control_passed: bool | None,
) -> dict:
    """Record for a round that never reached the resume — r1 itself failed.

    A round where account A is capped (or errors) at r1 produces no cross-account
    observation at all, so it cannot go through ``format_run_evidence``: there is
    no r2 to score.  It still has to leave a durable record.  These are precisely
    the rounds this measurement keeps losing — two of the three attempts at this
    question ended with account A capped — and without a record the operator sees
    only a one-line pytest skip, which is why the 2026-08-01 round was
    diagnosable only by a human reading a transcript by hand.

    Superset of ``format_run_evidence``'s key set, so one JSONL file can be read
    uniformly: every r2 field is ``None`` ("never issued", distinct from an r2
    that ran and returned nothing), ``codeword_recalled`` is False, and the
    transcript fields are ``False``/``None`` because a failed r1 is not probed.
    The r1 diagnostics the normal record has no need for (r1 succeeded there, by
    construction) are added here, because ``verdict`` alone cannot say WHY the
    round died and the raw text is the only thing that distinguishes a capped
    account from a broken harness.

    Verdict is ``'void_r1_capped'`` when the cap guard matches r1, else
    ``'void_r1_error'``.  Both are VOID: they are not evidence about context
    preservation in either direction, and they do not count toward the round's
    valid runs.
    """
    return {
        'account_a': account_a,
        'account_b': account_b,
        'r1_session_id': r1.session_id,
        'r1_success': r1.success,
        'r1_output': r1.output,
        'r1_stderr': r1.stderr,
        'r1_subtype': r1.subtype,
        'r1_transcript_present': False,
        'r1_transcript_records': None,
        'r2_output': None,
        'r2_stderr': None,
        'r2_success': None,
        'r2_subtype': None,
        'codeword_recalled': False,
        'control_passed': control_passed,
        'verdict': (
            'void_r1_capped'
            if result_looks_like_capacity_failure(r1)
            else 'void_r1_error'
        ),
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

    **The file write can never void the run it exists to record.** An
    unwritable ``CROSS_ACCOUNT_EVIDENCE_PATH`` (read-only mount, a plain file at
    a parent component, a permission the runner lacks) raises ``OSError``, and
    raising it from here would turn a completed live measurement — two real CLI
    calls, already paid for, inside a window that may not reopen for days — into
    a test ERROR before the assertions even run.  The evidence file is a
    side-channel; the stream copy above is already written and flushed, so the
    record survives.  The failure is reported to *stream* with the path and the
    exception rather than swallowed, so it cannot look like the operator simply
    forgot to set the var.

    Returns the JSON line (no trailing newline) so a caller can also attach it to
    an assertion message or to an escalation's structured evidence.
    """
    line = json.dumps(record, sort_keys=True)
    stream.write(f'\nCROSS_ACCOUNT_RESUME_EVIDENCE {line}\n')
    stream.flush()
    path_value = environ.get(EVIDENCE_PATH_VAR)
    if path_value:
        path = Path(path_value)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('a', encoding='utf-8') as fh:
                fh.write(f'{line}\n')
        except OSError as exc:
            stream.write(
                f'CROSS_ACCOUNT_RESUME_EVIDENCE_WRITE_FAILED {EVIDENCE_PATH_VAR}='
                f'{path_value!r}: {exc!r}. The record above is the surviving '
                'copy; the run itself is unaffected.\n'
            )
            stream.flush()
    return line


