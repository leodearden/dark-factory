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

from collections.abc import Mapping

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


