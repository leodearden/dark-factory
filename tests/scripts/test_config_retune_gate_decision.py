"""Decision gate: pre-merge gating for a retune of an operational config knob.

Task 3886, spun out of task 3866. The QUESTION: commit ``094d634465`` retuned
``lock_depth`` 4 -> 12 in ``dark-factory-orchestrator.yaml`` and turned main
red, because ``orchestrator/tests/conftest.py``'s autouse
``_isolate_orch_config`` binds every orchestrator test to that operational
file, and three tests carried fixture literals derived from the OLD depth. The
task framed three options: (1) add a pre-merge gate for yaml retunes, (2)
narrow the ``_isolate_orch_config`` contract so unit tests read package
defaults, (3) accept the exposure.

The DECISION — reject (2), reject (1) as framed, adopt (3) qualified, and name
the fourth option the task did not list (the knob-agnostic assertion form) —
lives in the ``DECIDED — pre-merge gate for operational config retunes`` block
in ``dark-factory-orchestrator.yaml``, sited next to ``lock_depth`` where a
future retuner will read it.

This file is the EXECUTABLE half of that record. A documented-but-ungated
decision is the same defect class these config tasks exist to close, so every
assertion here is on RUNTIME state — the config as the production loader
resolves it, the production predicates applied to it, the production
module-config discovery walk — never on comment or docstring prose.

NO PINNED INTEGER CONSTANTS. Every assertion is written in the knob-agnostic
form: it reads the live value rather than naming ``12`` or ``4``. Two house
rules converge on this. ``test_tests_scripts_module_config.py`` carries two
CORRECTED-IN-PLACE repairs of exactly this task's defect class (made by task
3866) and names the remedy — read ``cfg.lock_depth`` live so the next retune
of that knob falsifies nothing. And a guard written to prevent
constant-pinning breakage must not itself pin a constant, or the next retune
red-walls it and the fix becomes the disease.

Production code is cited BY SYMBOL throughout this file and never by
file:line — task 3445's explicit correction of the convention task 3350
established, after every line pin copied forward had already rotted at HEAD.
This task re-confirmed that the hard way: every line anchor its own plan
carried had drifted between two passes over the same unchanged code.

MUST-NOT-SKIP CONTRACT. No ``pytest.importorskip``, no try/except-and-skip.
An unimportable ``orchestrator.config``, ``orchestrator.verify`` or
``orchestrator.verify_plan`` must FAIL this guard rather than silently pass it
(``test_skills_module_config_decision.py``'s precedent).

PLACEMENT IS LOAD-BEARING, NOT STYLISTIC. This file lives in ``tests/scripts/``
because that directory carries its own module config, so the guard actually
runs under FULL_SUITE, on every review checkpoint, ``run_main_tip_sweep`` and
merge-role ``merge_verify_breadth: full``. It is also the only lever available:
``_discover_module_configs`` skips prefix ``.``, so a repo-root artifact like
``dark-factory-orchestrator.yaml`` cannot be routed to a module config of its
own.
"""
from __future__ import annotations

import pathlib

import pytest
from orchestrator.config import OrchestratorConfig

from orchestrator import verify, verify_plan

REPO_ROOT = pathlib.Path(__file__).parents[2]

# The repo-root config carrying the DECIDED block this file gates, and the
# subject of the decision itself: the operational file a retune edits.
# ``dark-factory-orchestrator.yaml`` is the canonical, REQUIRED filename for a
# project's top-level orchestrator config (it is what the dashboard's
# escalation-URL discovery keys on); the legacy spellings are a discovery
# fallback for unmigrated projects, not a choice this repo has.
DF_CONFIG_NAME = 'dark-factory-orchestrator.yaml'
ROOT_CONFIG_PATH = REPO_ROOT / DF_CONFIG_NAME


def _root_config(monkeypatch: pytest.MonkeyPatch) -> OrchestratorConfig:
    """Load the repo-root config through the PRODUCTION loader, anchored at ROOT_CONFIG_PATH.

    COPIED (task 3886) from ``test_tests_scripts_module_config.py``, which is
    itself a copy of the same helper in ``test_scripts_module_config.py`` and
    ``test_module_verify_budgets.py``. A test file importing a sibling test
    file couples two guards that must be able to fail independently, and this
    anchor is load-bearing enough that it must be visibly present in the file
    that depends on it.

    THE COST OF THAT, RECORDED RATHER THAN LEFT IMPLICIT, per the house
    record-rather-than-absorb idiom (task 3460). This makes a FOURTH verbatim
    copy of a helper that is pure setup, and the no-cross-import argument —
    sound for ASSERTIONS — does not reach ``tests/scripts/conftest.py``, which
    already exists and is pytest's idiomatic home for exactly this. That
    de-triplication is ALREADY FILED as a follow-up by task 3703 (see the
    corresponding docstring in ``test_tests_scripts_module_config.py``); this
    file adds a fourth copy to a known, tracked debt rather than a silent one.
    It is not reached for here for the same reason 3703 declined it:
    ``conftest.py`` is outside this task's locked file list, and editing a file
    five sibling guards depend on would widen the blast radius of a decision
    task.

    ANCHORING ``ORCH_CONFIG_PATH`` IS LOAD-BEARING, not hygiene.
    ``project_root`` is only a model FIELD and selects nothing:
    ``OrchestratorConfig.settings_customise_sources`` builds its
    ``YamlSettingsSource`` from ``os.environ['ORCH_CONFIG_PATH']`` alone,
    falling back to a CWD-relative ``config.yaml``. Both ambient states are
    wrong here, in OPPOSITE directions:

      * UNSET — the state INSIDE VERIFY, because
        ``verify._target_subprocess_env`` deliberately scrubs the whole
        ``ORCH_`` prefix (task 2957) — finds no file, so every value collapses
        to the pydantic DEFAULTS, a config this repo does not declare. That
        failure mode is this task's own subject matter one level up: the
        decision below turns on unit tests reading OPERATIONAL rather than
        DECLARED values, and a guard that silently read declared values would
        report green on the very substitution it exists to forbid.
      * SET, as an operator's shell has it, points at whichever checkout that
        orchestrator serves — typically the MAIN one, not this worktree. Every
        assertion would then be about a different checkout's yaml and report
        GREEN on a worktree that had actually regressed.

    Setting the env var IS the production load path (``config.load_config``
    stamps ``os.environ['ORCH_CONFIG_PATH']`` before constructing), so this
    stays a read through the real loader, pinned to THIS worktree's committed
    yaml rather than left to the ambient environment.

    Fails LOUDLY on a missing file rather than silently: ``YamlSettingsSource``
    SKIPS a non-existent ``config_path`` instead of raising, so a bad path
    would yield the pydantic DEFAULTS with no error at all.
    """
    assert ROOT_CONFIG_PATH.is_file(), (
        f'{ROOT_CONFIG_PATH} does not exist, so anchoring ORCH_CONFIG_PATH at '
        'it would silently load the pydantic DEFAULTS instead (YamlSettingsSource '
        'skips a non-existent path rather than raising), and every value read '
        'from the returned config would be about a config this repo does not '
        f'declare. {DF_CONFIG_NAME} is the canonical, required filename for a '
        "project's top-level orchestrator config"
    )
    monkeypatch.setenv('ORCH_CONFIG_PATH', str(ROOT_CONFIG_PATH))
    return OrchestratorConfig(project_root=REPO_ROOT)


def test_merge_lane_already_gates_a_config_only_retune(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The merge-lane arm: option (1) as framed buys nothing not already held.

    This is the executable form of the DECIDED block's CORRECTION 1 — the
    task's framed premise "there is NO pre-merge gate" is HALF-FALSE. Both
    halves of what makes it false are asserted here on runtime state, because
    the decision to decline a NEW merge-lane gate rests entirely on the two
    that already exist. If either lapses, this decision needs re-taking and
    this guard is what says so.

    (1) ``merge_verify_breadth: "full"`` makes merge-role verify run every
    REGISTERED module's full suite regardless of what the diff touches —
    ``verify_plan.derive_verify_plan`` gates on ``_merge_breadth_is_full`` and
    fans out through ``_derive_full_suite_runs``. Asserted through that
    production predicate rather than only against the raw field, so the guard
    cannot drift from the function that consumes it.

    (2) The deterministic manifest-drift backstop
    ``verify._merge_config_only_diff_forces_full_gate`` matches this repo's own
    operational config, so a merge-role config-only diff that touches it is
    forced onto the full per-subproject gate. Asserted through the production
    predicate rather than by re-implementing ``fnmatch``, so the guard cannot
    drift from the matcher it guards.

    WHY (2) NEEDS TO BE WIRED AT ALL, given (1) — the fact that carries the one
    production change this task makes, and the reason this is not INV-5
    duplication of a gate already held. Both call sites of that predicate sit
    in the ``else:`` of ``if role == 'merge' and is_merge_verify:``. So when
    ``is_merge_verify`` is True, INV-1's nothing-to-run escalation already
    covers a no-.py/.rs diff and the glob is irrelevant. It is the
    ``is_merge_verify`` FALSE merge-role probe path where ``should_override``
    reduces to this predicate OR the async reify verify-pipeline-guard consult
    — and that consult falls open for dark-factory, so with empty globs (the
    shipped default, short-circuiting to False in O(1)) NOTHING can force the
    full gate there. Wiring the glob closes that path and converts "a
    config-only diff forces the full gate" from an incidental side effect of
    breadth into a DECLARED intent.
    """
    cfg = _root_config(monkeypatch)

    # (1) The broad merge gate is declared AND the production predicate agrees.
    assert cfg.merge_verify_breadth == 'full', (
        f'{DF_CONFIG_NAME} declares merge_verify_breadth='
        f'{cfg.merge_verify_breadth!r}, not "full". Task 3886 declined to add a '
        'new pre-merge gate for yaml retunes SPECIFICALLY because this one '
        'already runs every registered module\'s full suite on a merge '
        'regardless of what the diff touches. Narrowing it re-opens the '
        'exposure that decision accepted, so re-take the decision in the '
        f'DECIDED block in {DF_CONFIG_NAME} rather than editing this assertion'
    )
    assert verify_plan._merge_breadth_is_full(cfg) is True, (
        'verify_plan._merge_breadth_is_full rejects the operational config even '
        f'though it declares merge_verify_breadth={cfg.merge_verify_breadth!r}. '
        'That predicate is what verify_plan.derive_verify_plan actually gates '
        'on before fanning out via _derive_full_suite_runs, so the declared '
        'value alone does not establish the gate — this pair is asserted '
        'together precisely so the guard cannot drift from its consumer'
    )

    # (2) The deterministic config-only backstop covers this repo's own config,
    #     evaluated through the production matcher rather than a local fnmatch.
    assert verify._merge_config_only_diff_forces_full_gate(cfg, [DF_CONFIG_NAME]) is True, (
        f'a merge-role config-only diff touching {DF_CONFIG_NAME} does NOT force '
        'the full per-subproject gate: '
        'verify._merge_config_only_diff_forces_full_gate returns False for it '
        f'against git.merge_config_only_full_gate_globs='
        f'{cfg.git.merge_config_only_full_gate_globs!r}. Empty globs (the '
        'shipped default) short-circuit to False in O(1). This matters on the '
        'is_merge_verify=False merge-role probe path, where both call sites sit '
        "in the else: of `if role == 'merge' and is_merge_verify:` — INV-1's "
        'nothing-to-run escalation does NOT reach there, and the reify '
        'verify-pipeline-guard consult falls open for dark-factory, so this '
        'predicate is the only thing that can force the full gate. Task 3886 '
        f'wired it in the git: block of {DF_CONFIG_NAME}; restore that entry '
        'rather than deleting this assertion'
    )
