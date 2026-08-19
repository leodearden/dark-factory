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
    MODULE_LINT_COMMANDS,
    REPO_ROOT,
    ROOT_LINT_COMMAND,
    ROOT_TEST_COMMAND,
    ROOT_TYPE_CHECK_COMMAND,
    SCRIPTS_CONFIG_PATH,
    SCRIPTS_LINT_COMMAND,
    discover_lint_command_modules,
    load_config_scalar,
)

# (corpus constant name, its value, the config file, the YAML key).
# Parametrised as data rather than four near-identical test bodies so a fifth
# scalar is one tuple, not another copy of the assertion.
_ROOT_SCALAR_CASES = [
    ('FM_LINT_COMMAND', FM_LINT_COMMAND, FM_CONFIG_PATH, 'lint_command'),
    ('SCRIPTS_LINT_COMMAND', SCRIPTS_LINT_COMMAND, SCRIPTS_CONFIG_PATH, 'lint_command'),
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


class TestModuleLintCommandsMatchLiveYaml:
    """Each per-module ``lint_command`` is byte-identical to its live YAML value.

    These six share one 2-segment shape — ``uv run --project M --directory M
    ruff check src/ tests/ && python3
    fused-memory/scripts/check_bare_magicmock_config.py M/tests`` — which is
    why the corpus can generate them from the module name. ``fused-memory`` is
    the lone 3-segment chain (it runs a second sibling checker), so it does not
    fit the comprehension and lives in ``FM_LINT_COMMAND`` instead, covered by
    ``TestRootScalarsMatchLiveYaml`` above.
    """

    @pytest.mark.parametrize('module', sorted(MODULE_LINT_COMMANDS))
    def test_module_lint_command_equals_live_value(self, module):
        config_path = REPO_ROOT / module / 'orchestrator.yaml'
        live = load_config_scalar(config_path, 'lint_command')
        assert MODULE_LINT_COMMANDS[module] == live, (
            f'_verify_config_corpus.MODULE_LINT_COMMANDS[{module!r}] has drifted from the '
            f'live config.\n'
            f'  source: {config_path.relative_to(REPO_ROOT)}::lint_command\n'
            f'  corpus: {MODULE_LINT_COMMANDS[module]!r}\n'
            f'  live:   {live!r}\n'
            f'FIX: if {module} still shares the common 2-segment shape, the comprehension in '
            f'_verify_config_corpus.py needs updating; if its chain shape has diverged, give '
            f'it a dedicated constant the way fused-memory has FM_LINT_COMMAND, and drop it '
            f'from MODULE_LINT_COMMANDS. Either way also check test_verify_plan.py\'s goldens, '
            f'which iterate this dict. Do NOT loosen this comparison to a substring match.'
        )


class TestCorpusCoversEveryLiveLintCommand:
    """Every module that defines a ``lint_command`` today is covered by the corpus.

    The completeness half of the drift gate. The forward checks above are
    structurally blind to one failure mode: a NEW subproject lands its own
    ``orchestrator.yaml`` with a ``lint_command``, the corpus never grows to
    cover it, and the scoper suites go on claiming to "exercise the real
    configs" while silently missing one. Only a check that DISCOVERS the live
    set can see that.
    """

    def test_discovered_modules_are_exactly_the_covered_modules(self):
        discovered = discover_lint_command_modules()
        covered = {'fused-memory', 'scripts'} | set(MODULE_LINT_COMMANDS)

        uncovered = discovered - covered
        assert not uncovered, (
            f'These modules define a lint_command that the verify-config corpus does not '
            f'cover: {sorted(uncovered)}.\n'
            f'FIX: if the chain matches the common 2-segment shape, add the module name to '
            f'MODULE_LINT_COMMANDS in orchestrator/tests/_verify_config_corpus.py; if the '
            f'shape differs, give it a dedicated constant the way fused-memory has '
            f'FM_LINT_COMMAND. Then extend the verify-scoper goldens to exercise it — an '
            f'uncovered module is a real command shape the scoper has never been tested '
            f'against.'
        )

        stale = covered - discovered
        assert not stale, (
            f'The verify-config corpus covers these modules, but no live lint_command was '
            f'discovered for them: {sorted(stale)}.\n'
            f'FIX: the module\'s orchestrator.yaml has lost its lint_command, or the module '
            f'was deleted or renamed. Drop the corpus entry (MODULE_LINT_COMMANDS, or the '
            f'dedicated constant) together with the verify-scoper goldens that consume it.'
        )
