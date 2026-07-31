"""Drift gate: the shared verify-config corpus vs. the live committed YAML.

The verify-scoper suites (``test_verify_cmd.py``, ``test_verify_plan.py``,
``test_verify_scope_kappa.py``) assert byte-identical outcomes over the repo's
*real* orchestrator config commands, hoisted into the single definition site
``_verify_config_corpus.py``. Their whole value rests on those constants still
being the live values — a corpus that has silently drifted is a suite that
proves the scoper handles commands nobody runs any more.

Until task 3220 that claim was a code COMMENT ("verified byte-identical to the
live YAML", ``test_verify_plan.py:43``) — true when written, unenforced after.
This module replaces the comment with an executable check.

Two halves, both needed:

  * FORWARD (``TestRootScalarsMatchLiveYaml``,
    ``TestModuleLintCommandsMatchLiveYaml``) — each corpus constant equals the
    value its YAML key holds today. Catches a config edit that leaves the
    corpus behind.
  * COMPLETENESS (``TestCorpusCoversEveryLiveLintCommand``) — the set of
    modules whose ``orchestrator.yaml`` defines a ``lint_command`` is exactly
    the set the corpus covers. Catches a NEW subproject the corpus never grew
    to cover, which no forward check can structurally see.

Every comparison is ``==`` on the WHOLE string — never ``in``, a substring, or
normalised whitespace. A loose comparison here would pass while the scoper
goldens exercise a command shape that no longer exists, which is precisely the
failure this gate is built to catch.
"""

from __future__ import annotations

import pytest
from _verify_config_corpus import (
    DF_CONFIG_PATH,
    FM_CONFIG_PATH,
    FM_LINT_COMMAND,
    REPO_ROOT,
    ROOT_LINT_COMMAND,
    ROOT_TEST_COMMAND,
    ROOT_TYPE_CHECK_COMMAND,
    load_config_scalar,
)

# (corpus constant name, its value, the config file, the YAML key).
# Parametrised as data rather than four near-identical test bodies so a fifth
# scalar is one tuple, not another copy of the assertion.
_ROOT_SCALAR_CASES = [
    ('FM_LINT_COMMAND', FM_LINT_COMMAND, FM_CONFIG_PATH, 'lint_command'),
    ('ROOT_LINT_COMMAND', ROOT_LINT_COMMAND, DF_CONFIG_PATH, 'lint_command'),
    ('ROOT_TYPE_CHECK_COMMAND', ROOT_TYPE_CHECK_COMMAND, DF_CONFIG_PATH, 'type_check_command'),
    ('ROOT_TEST_COMMAND', ROOT_TEST_COMMAND, DF_CONFIG_PATH, 'test_command'),
]


class TestRootScalarsMatchLiveYaml:
    """The four scalar corpus constants are byte-identical to their live YAML values."""

    @pytest.mark.parametrize(
        ('const_name', 'const_value', 'config_path', 'yaml_key'),
        _ROOT_SCALAR_CASES,
        ids=[case[0] for case in _ROOT_SCALAR_CASES],
    )
    def test_constant_equals_live_value(self, const_name, const_value, config_path, yaml_key):
        live = load_config_scalar(config_path, yaml_key)
        assert const_value == live, (
            f'_verify_config_corpus.{const_name} has drifted from the live config.\n'
            f'  source: {config_path.relative_to(REPO_ROOT)}::{yaml_key}\n'
            f'  corpus: {const_value!r}\n'
            f'  live:   {live!r}\n'
            f'FIX: update {const_name} in orchestrator/tests/_verify_config_corpus.py to the '
            f'live value above, then re-run the verify-scoper suites — their goldens encode '
            f'this command\'s shape and may need updating too. Do NOT loosen this comparison '
            f'to a substring or normalised match: the whole point is that the scoper suites '
            f'exercise the exact command the fleet actually runs.'
        )
