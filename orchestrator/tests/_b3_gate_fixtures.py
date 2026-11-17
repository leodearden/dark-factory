"""Scaffolding shared by the orchestrator/tests/test_b3_gate*.py modules; import it, do not copy it."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

LOW_RISK_ENTRY = {
    'proposal_text': 'Fix it',
    'risk_label': 'low',
    'files_referenced': ['foo.py'],
    'block_reason': 'test',
    'investigated_at': '2026-06-04T09:00:00+00:00',
    'head_sha': 'aaabbbccc',
    'main_sha': 'dddeeefff',
}


def fake_git_fresh(args, cwd):
    """Fake git runner: HEAD == recorded sha, diff is empty (fresh)."""
    if 'rev-parse' in args:
        return (0, 'aaabbbccc')
    return (0, '')  # diff empty


def fake_git_never_called(args, cwd):
    raise AssertionError(f'git should not have been called; args={args}')


def fake_git_diff_fails(args, cwd):
    """Fake git runner: rev-parse HEAD succeeds (P1 passes), but diff returns non-zero rc."""
    if 'rev-parse' in args:
        return (0, 'aaabbbccc')
    # diff call fails (e.g. bad revision, transient error)
    return (128, 'fatal: bad revision dddeeefff..main')


def make_agent_result(*, success=True, cost_usd=0.50, structured_output=None,
                      subtype='', output='', duration_ms=1000,
                      timed_out=False, transcript_turns=None, session_id='',
                      stderr='', turns=0, api_error_status=None,
                      schema_salvaged=False, output_tokens=None,
                      ended_awaiting_background=False):
    """Create a minimal MagicMock AgentResult stand-in for _build_entry tests.

    NOTE: duplicated verbatim in test_dry_run_unblock.py (task 2020 review:
    code_duplication). Both copies must be extended together whenever
    shared.cli_invoke.AgentResult gains/renames a field consumed by
    classify_agent_failure(). Out of this task's locked scope: hoisting both
    into the existing orchestrator/tests/_orch_helpers.py shared-helper
    module (the established convention for cross-suite test helpers in this
    package) is a follow-up.
    """
    r = MagicMock()
    r.success = success
    r.cost_usd = cost_usd
    r.structured_output = structured_output
    r.subtype = subtype
    r.output = output
    r.duration_ms = duration_ms
    # Faithful diagnostic-field defaults (mirrors shared.cli_invoke.AgentResult).
    # A bare MagicMock auto-vivifies every unset attribute as a truthy
    # MagicMock, which would silently corrupt classify_agent_failure()'s
    # decision rules: .timed_out truthy -> misclassifies every failure as
    # TIMED_OUT -> infra; .schema_salvaged truthy -> misclassifies a generic
    # failure as STRUCTURAL instead of UNKNOWN. Setting explicit defaults for
    # every field classify_agent_failure() consults (including
    # .output_tokens, read when formatting the MAX_TURNS summary) keeps
    # these mocks faithful stand-ins for its decision rules.
    r.timed_out = timed_out
    r.transcript_turns = transcript_turns
    r.session_id = session_id
    r.stderr = stderr
    r.turns = turns
    r.api_error_status = api_error_status
    r.schema_salvaged = schema_salvaged
    r.output_tokens = output_tokens
    # task 2761: classify_agent_failure now reads .ended_awaiting_background
    # (rule 2, above the timed_out rule). A bare MagicMock auto-vivifies it as
    # a truthy stand-in, which would misclassify every failure as
    # ENDED_AWAITING_BACKGROUND -> investigation_failed. Default it False.
    r.ended_awaiting_background = ended_awaiting_background
    return r


def diverge_feature_branch(path: Path) -> str:
    """Create a feature branch with one extra commit so HEAD != main.

    Returns the new HEAD sha (feature commit).
    """
    p = str(path)
    subprocess.run(['git', '-C', p, 'checkout', '-b', 'feature'],
                   check=True, capture_output=True)
    (path / 'extra.txt').write_text('feature work')
    subprocess.run(['git', '-C', p, 'add', '.'], check=True, capture_output=True)
    subprocess.run(['git', '-C', p, 'commit', '-m', 'feature commit'],
                   check=True, capture_output=True)
    result = subprocess.run(['git', '-C', p, 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True)
    return result.stdout.strip()
