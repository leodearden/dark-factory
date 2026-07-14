"""Integration-gate suite ι: B+H boundary tests over the verify decision layer.

This file is the ι B+H integration-gate LEAF for the verify-plan PRD
(plans/verify-plan-prd.md; Boundary-test sketch rows 1-12; capability
manifest block ι).  It drives the REAL, already-landed verify decision layer
(α-θ) end-to-end across every seam, facing BOTH producer and runner sides of
each contract.

α-θ are all merged on main and are OUT OF SCOPE for this task — this module
contains NO production code changes:
  α verify_categories.py     — FailureCategory + CATEGORY_POLICY exhaustiveness
  β verify_cmd.py            — VerifyCmd / parse_config_command / render / mutators
  γ verify_plan.py           — derive_verify_plan (plan goldens)
  δ verify_classify.py       — classify_failure (tool-isolation)
  ε verify.py                — CheckRun / VerifyAttempt (timeout consistency)
  ζ unblock_types.py         — BlockRecord / BlockClass
  η merge_queue.py           — block-path spawn -> dry-run proposal
  θ git_ops.py               — ephemeral_worktree (no-prune probes)

REAL vs FAKED
-------------
REAL (composed as the genuine article, never hand-seeded): GitOps over a
real git repo; ``derive_verify_plan``, ``classify_failure``, ``check_proposal``,
``_run_post_merge_verify``, ``VerifyCmd`` + its mutators, ``VerifyAttempt``,
and ``ephemeral_worktree`` itself.

FAKED (boundary only — the ssh/build/agent edges, never the decision layer):
``run_scoped_verification``, ``run_full_verification``,
``orchestrator.dry_run_unblock.invoke_agent``, and ``git_ops._run`` where a
subprocess argv-spy is needed (scenario 12 only — everywhere else git
subprocesses run for real against the fixture repo).

Each scenario class below is RED until its paired wiring step imports the
exercised symbols and ports the needed test-local helpers (see each class's
own docstring); "GREEN" means the real-object driver correctly exercises the
already-landed code, not that new production logic was written anywhere. A
scenario that stays RED for a genuine composition reason is a design_concern
escalation, not a patch to α-θ (see plan.json design_decisions).

§ Scenario index (Boundary-test sketch rows 1-12, capability manifest §ι)
--------------------------------------------------------------------------
  1.  VerifyCmd render round-trip + producer<->runner scoped-pytest drive (P2).
  2.  OPAQUE never scoped (P1).
  3.  Plan golden — root conftest -> FULL_SUITE (D1, task-1077).
  4.  Plan golden — lone data module -> SKIPPED-with-reason (task-1852).
  5.  Plan golden — structural file -> unscoped pyright, module + fallback (D2).
  6.  Classifier tool-isolation (C1).
  7.  Category exhaustiveness (F1).
  8.  CheckRun/VerifyAttempt timeout consistency (the verify.py:2735-2744 drift).
  9.  Merge-verify block -> gateable proposal (the coverage gap + B4).
  10. POST_MERGE_RED_MAIN preserved (B2, task-1680).
  11. Legacy proposal bridge (B3) + BlockRecord round-trip (B1).
  12. ephemeral_worktree no-prune across both probes (E1/E2).
"""

from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
from enum import StrEnum
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator import (
    b3_gate,
    merge_queue,
    unblock_types,
    verify,
    verify_categories,
    verify_classify,
    verify_cmd,
    verify_plan,
)
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.git_ops import GitOps, _run
from orchestrator.merge_queue import MergeRequest
from orchestrator.verify import CheckRun, VerifyAttempt
from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory, _validate_exhaustive
from orchestrator.verify_classify import classify_failure
from orchestrator.verify_cmd import (
    ToolKind,
    VerifyCmd,
    parse_config_command,
    render,
    scope_to,
    strip_cwd,
)
from orchestrator.verify_plan import (
    PlannedRun,
    ScopeKind,
    VerifyPlan,
    derive_verify_plan,
)

# ── Repo seeding (ported from test_merge_queue_two_layer_integration.py) ──────


async def _setup_repo(repo: Path) -> None:
    """Initialise a minimal git repo with a README committed on main."""
    await _run(['git', 'init', '-b', 'main'], cwd=repo)
    await _run(['git', 'config', 'user.email', 'test@test.com'], cwd=repo)
    await _run(['git', 'config', 'user.name', 'Test'], cwd=repo)
    (repo / 'README.md').write_text('# Test\n')
    await _run(['git', 'add', '-A'], cwd=repo)
    await _run(['git', 'commit', '-m', 'Initial commit'], cwd=repo)


# ── Fixtures (ported from test_merge_queue_two_layer_integration.py) ─────────


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / 'repo'
    repo.mkdir()
    asyncio.run(_setup_repo(repo))
    return repo


@pytest.fixture
def git_config() -> GitConfig:
    return GitConfig(
        main_branch='main',
        branch_prefix='task/',
        remote='origin',
        worktree_dir='.worktrees',
        push_after_advance=False,
    )


@pytest.fixture
def git_ops(git_config: GitConfig, git_repo: Path) -> GitOps:
    return GitOps(git_config, git_repo)


@pytest.fixture
def config(git_repo: Path, git_config: GitConfig) -> OrchestratorConfig:
    return OrchestratorConfig(project_root=git_repo, git=git_config)


# ── MergeRequest builder (ported from test_merge_queue_two_layer_integration.py) ──


def _make_req(
    task_id: str,
    branch: str,
    config: OrchestratorConfig,
    git_repo: Path,
    *,
    module_configs: list[ModuleConfig] | None = None,
    task_files: list[str] | None = None,
    merge_first_enqueued_at: float | None = 1000.0,
    request_id: str | None = None,
) -> MergeRequest:
    """Build a minimal MergeRequest with a fresh event-loop future.

    Must be called from within an async context (asyncio.get_running_loop()).
    The optional *request_id* kwarg lets a test pin a stable identity; when
    omitted a fresh UUID is auto-generated (MergeRequest's own default).
    """
    kwargs: dict = {}
    if request_id is not None:
        kwargs['request_id'] = request_id
    return MergeRequest(
        task_id=task_id,
        branch=branch,
        worktree=git_repo,
        pre_rebased=False,
        task_files=task_files,
        module_configs=module_configs or [],
        config=config,
        result=asyncio.get_running_loop().create_future(),
        merge_first_enqueued_at=merge_first_enqueued_at,
        **kwargs,
    )


# ── step-1/2: Scenarios 1+2 — VerifyCmd render round-trip (P2) + OPAQUE never
#              scoped (P1); Boundary-test sketch rows 1-2 ────────────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]

# The real orchestrator/orchestrator.yaml pytest test_command (uv `--directory`
# form) — drives the scenario-1b producer<->runner scoped drive. NOT used for
# the strict round-trip assertion in scenario 1a: render() normalises a
# `--directory` flag into a leading `cd <dir> &&`, which is argv-equivalent in
# *effect* but not shlex-list-equal to the original (one extra `&&` token) —
# the round-trip corpus below is ported from test_verify_cmd.py's own fixtures,
# which are already --directory-free/leading-cd-form and provably round-trip.
_PYTEST_UV_DIRECTORY_RAW = (
    'uv run --project orchestrator --directory orchestrator pytest tests/ --tb=short -q'
)

# The historical broken lint/type_check &&-chain (orchestrator/config.yaml:50)
# — recognised-but-unstructurable (ruff + a follow-up script, not pytest/cargo
# chain-aware), classifies OPAQUE, and must never be scoped (P1).
_LINT_CHAIN_RAW = (
    'uv run ruff check shared escalation fused-memory orchestrator dashboard && '
    'python3 fused-memory/scripts/check_bare_magicmock_config.py shared/tests '
    'escalation/tests fused-memory/tests orchestrator/tests dashboard/tests'
)


class TestVerifyCmdRoundTrip:
    """Scenarios 1+2 — VerifyCmd render round-trip (P2) + OPAQUE never scoped (P1).

    Drives the REAL parse_config_command/render/scope_to/strip_cwd (β) against
    representative non-OPAQUE config commands (ported from test_verify_cmd.py's
    proven-safe round-trip corpus) and closes the producer<->runner loop for
    the pytest case: scope_to's structured output (producer side) is rendered
    and ACTUALLY EXECUTED (runner side) against a throwaway probe file,
    proving the two sides agree on what "scoped" means. Row 2/P1 pins the
    real historical broken lint/type_check &&-chain (orchestrator/config.yaml:50)
    as OPAQUE and never scoped.

    RED until step-2 GREEN imports parse_config_command/render/ToolKind/
    scope_to/strip_cwd/VerifyCmd from orchestrator.verify_cmd and adds the
    subprocess-exec wiring.
    """

    # ── scenario 1a / Row 1 / P2: render round-trip is argv-equivalent per
    #    non-OPAQUE ToolKind. Corpus ported verbatim from test_verify_cmd.py's
    #    TestRenderRoundTrip.test_round_trip_argv_equivalent (β) — the exact
    #    fixtures already proven to dodge both round-trip hazards (a
    #    `--directory` flag normalising into an extra leading `cd &&` token,
    #    and a flags/targets interleaving render always re-emits flags-first).
    @pytest.mark.parametrize(
        'raw',
        [
            'pytest tests/test_x.py',
            'ruff check src/foo.py',
            'pyright src/foo.py',
            'cargo test --workspace',
            'cargo clippy --workspace',
            'npx --version',
            'uv run --project shared pytest tests/x.py',
            'cd fused-memory && npx pyright',
        ],
        ids=[
            'pytest', 'ruff', 'pyright', 'cargo_test', 'cargo_clippy', 'npx',
            'uv_run_project', 'leading_cd_npx',
        ],
    )
    def test_render_round_trip_is_argv_equivalent_per_tool_kind(self, raw):
        """shlex.split(render(parse(x))) == shlex.split(x) for every non-OPAQUE tool."""
        cmd = parse_config_command(raw)
        assert cmd.tool is not ToolKind.OPAQUE
        assert shlex.split(render(cmd)) == shlex.split(raw)

    # ── scenario 1b / Row 1: producer<->runner scoped-pytest drive ──────────
    def test_scoped_pytest_producer_and_runner_agree_on_scope(self, tmp_path):
        """scope_to's structured output (producer) IS what render+exec (runner) runs.

        Producer side: parse the real orchestrator.yaml test_command and
        scope_to() it down to one throwaway probe file. Runner side: render()
        that scoped VerifyCmd (strip_cwd'd so it runs unchanged from the repo
        root with an absolute probe path) and ACTUALLY EXECUTE it via
        `bash -c` — if the two sides disagreed on what "scoped" means, this
        would either fail to launch or collect more than the one scoped test.
        """
        probe = tmp_path / 'test_verify_cmd_scope_probe.py'
        probe.write_text('def test_probe_passes():\n    assert True\n')

        parsed = parse_config_command(_PYTEST_UV_DIRECTORY_RAW)
        scoped = scope_to(parsed, [str(probe)])
        runnable = strip_cwd(scoped)

        argv = shlex.split(render(runnable))
        assert argv[-1] == str(probe)
        assert 'tests/' not in argv  # the original unscoped target is gone, not appended
        assert '--tb=short' in argv and '-q' in argv  # other flags survive scoping

        proc = subprocess.run(
            ['bash', '-c', render(runnable)],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert '1 passed' in proc.stdout

    # ── scenario 2 / Row 2 / P1: OPAQUE is never scoped ──────────────────────
    def test_opaque_lint_chain_is_never_scoped(self):
        """The historical broken &&-chain classifies OPAQUE and scope_to no-ops on it.

        render(parse(x)) == x verbatim (raw retained unchanged) — this is the
        exact lint_command value at orchestrator/config.yaml:50 that regressed
        under the old string-surgery scoper.
        """
        cmd = parse_config_command(_LINT_CHAIN_RAW)
        assert cmd.tool is ToolKind.OPAQUE
        assert scope_to(cmd, ['orchestrator/src/orchestrator/foo.py']) is cmd
        assert render(parse_config_command(_LINT_CHAIN_RAW)) == _LINT_CHAIN_RAW


# ── step-3/4: Scenario 3 — plan golden: root conftest -> FULL_SUITE (D1,
#              task-1077); Boundary-test sketch row 3 ───────────────────────

# task-1077 (git-verified fix commits d7504d432d + cb7277926d): conftest.py
# must trigger the full unscoped suite, never be passed directly to pytest as
# a target (pytest >= 9 exits 1 "no tests ran" on a bare conftest target).
# Reconstructed verbatim from test_verify_plan.py's own golden fixture.
ROOT_CONFTEST_DIFF: list[str] = ['orchestrator/tests/conftest.py']

# Canned file contents for the dict-backed fake worktree_reader below (ported
# from test_verify_plan.py). Seeded with STRUCTURAL_DIFF's Protocol-bearing
# content once scenario 5 introduces it; every other path — including this
# scenario's ROOT_CONFTEST_DIFF file — reads back as None, which
# classify_file must treat as "not detected", never an error.
_FAKE_FILE_CONTENTS: dict[str, str] = {}


def fake_worktree_reader(path: str) -> str | None:
    """Dict-backed stand-in for real file I/O (``Callable[[str], str | None]``).

    Keeps derive_verify_plan pure and unit-testable without touching a real
    filesystem: returns canned content for paths seeded into
    _FAKE_FILE_CONTENTS, else None.
    """
    return _FAKE_FILE_CONTENTS.get(path)


def _run_for(plan: VerifyPlan, prefix: str, tool_word: str) -> PlannedRun | None:
    """Find *prefix*'s PlannedRun whose reason names *tool_word* (e.g. ``'pytest:'``).

    Tool identity is recoverable from ``cmd.tool`` for a non-SKIPPED run, but a
    SKIPPED slot carries ``cmd=None`` (D3's "explicit reasoned skip, never a
    dropped command") — so ``derive_verify_plan`` always prefixes each
    per-tool ``PlannedRun.reason`` with its tool name, keeping the reason the
    tool-identity signal of last resort.
    """
    return next(
        (r for r in plan.runs if r.module_prefix == prefix and r.reason.startswith(tool_word)),
        None,
    )


class TestPlanGoldenConftest:
    """Scenario 3 — GOLDEN task-1077: a touched conftest.py widens pytest to

    FULL_SUITE with the verbatim unscoped test_command, and the reason names
    conftest (D1). Also pins D3: the resulting VerifyPlan is JSON-serialisable.

    RED until step-4 GREEN imports derive_verify_plan/ScopeKind/PlannedRun/
    VerifyPlan from orchestrator.verify_plan and ports fake_worktree_reader/
    _run_for from test_verify_plan.py.
    """

    def test_root_conftest_full_suites_pytest_with_json_serialisable_plan(self):
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command=(
                'uv run --project orchestrator --directory orchestrator '
                'pytest tests/ --tb=short -q'
            ),
            lint_command='uv run --directory orchestrator ruff check src/',
        )
        plan = derive_verify_plan(ROOT_CONFTEST_DIFF, [mc], None, fake_worktree_reader)
        run = _run_for(plan, 'orchestrator', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.FULL_SUITE
        # Structural equality against the same parse_config_command transform
        # sidesteps render()'s documented cwd_rel-as-leading-`cd`
        # normalisation (not always byte-identical to a --directory-form
        # input — see verify_cmd.render's docstring / scenario 1a above).
        assert mc.test_command is not None
        assert run.cmd == parse_config_command(mc.test_command)
        assert 'conftest' in run.reason.lower()

        # D3: VerifyPlan.to_dict() is a plain JSON-native structure.
        json.dumps(plan.to_dict())


# ── step-5/6: Scenario 4 — plan golden: lone data module -> SKIPPED-with-
#              reason (task-1852); Boundary-test sketch row 4 ───────────────

# task-1852 (git-verified fix commits 4fbed6c4fb + 7c9b316260): a non-test
# data module under tests/ is test-tree but NOT pytest-collectable (passing
# it to pytest produces rc=5 "no tests ran"). Reconstructed verbatim from
# test_verify_plan.py's own golden fixture.
DATA_MODULE_DIFF: list[str] = ['shared/tests/silent_fallthrough_allowlist.py']


class TestPlanGoldenDataModule:
    """Scenario 4 — GOLDEN task-1852: a lone data module, driven through the

    FALLBACK path (module_configs=[]) against the bare 'pytest' default,
    degrades to an explicit reasoned SKIPPED — never a silent None and never
    a fabricated run that would rc=5.
    """

    def test_data_module_bare_pytest_default_skips_with_reason(self):
        config = OrchestratorConfig(project_root=Path('/fake'), test_command='pytest')
        plan = derive_verify_plan(DATA_MODULE_DIFF, [], config, fake_worktree_reader)
        run = _run_for(plan, '__fallback__', 'pytest:')
        assert run is not None
        assert run.scope_kind is ScopeKind.SKIPPED
        assert run.reason
        assert DATA_MODULE_DIFF[0] in run.reason
        assert '1852' in run.reason


# ── step-7/8: Scenario 5 — plan golden: structural file -> unscoped pyright,
#              module + fallback paths (D2); Boundary-test sketch row 5 ─────

# A Protocol-defining source file (D2): file-scoped pyright cannot verify
# cross-file Protocol conformance, so a STRUCTURAL file must widen pyright to
# the unscoped package-wide command in BOTH the module and fallback paths.
# Reconstructed verbatim from test_verify_plan.py's own golden fixture.
STRUCTURAL_DIFF: list[str] = ['orchestrator/src/orchestrator/interfaces.py']

# STRUCTURAL is only detected when a type_check_command is configured (so
# content is actually read); seed the canned Protocol-bearing content this
# scenario needs into the shared fake-reader backing dict.
_FAKE_FILE_CONTENTS[STRUCTURAL_DIFF[0]] = 'class Foo(Protocol):\n    def m(self) -> None: ...\n'


class TestPlanGoldenStructural:
    """Scenario 5 — GOLDEN D2: a Protocol-bearing source file widens pyright

    to the unscoped FULL_SUITE command (never file-scoped — cross-file
    Protocol conformance can't be checked from one file) and skips pytest,
    in BOTH the module-config path and the fallback path (the latent gap
    _build_fallback_config never closed).

    RED until step-8 seeds _FAKE_FILE_CONTENTS with STRUCTURAL_DIFF[0]'s
    Protocol-bearing content — STRUCTURAL is only detected when content is
    read, and fake_worktree_reader returns None for any unseeded path.
    """

    def test_structural_file_full_suites_pyright_module_path(self):
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --project orchestrator --directory orchestrator pytest tests/',
            type_check_command=(
                'uv run --project orchestrator --directory orchestrator pyright src/ tests/'
            ),
        )
        plan = derive_verify_plan(STRUCTURAL_DIFF, [mc], None, fake_worktree_reader)

        pyright_run = _run_for(plan, 'orchestrator', 'pyright:')
        assert pyright_run is not None
        assert pyright_run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.type_check_command is not None
        assert pyright_run.cmd == parse_config_command(mc.type_check_command)
        assert STRUCTURAL_DIFF[0] in pyright_run.reason

        pytest_run = _run_for(plan, 'orchestrator', 'pytest:')
        assert pytest_run is not None
        assert pytest_run.scope_kind is ScopeKind.SKIPPED

    def test_structural_file_full_suites_pyright_fallback_path(self):
        config = OrchestratorConfig(project_root=Path('/fake'), test_command='pytest')
        plan = derive_verify_plan(STRUCTURAL_DIFF, [], config, fake_worktree_reader)

        pyright_run = _run_for(plan, '__fallback__', 'pyright:')
        assert pyright_run is not None
        assert pyright_run.scope_kind is ScopeKind.FULL_SUITE
        assert pyright_run.cmd == parse_config_command(config.type_check_command)
        assert STRUCTURAL_DIFF[0] in pyright_run.reason

        pytest_run = _run_for(plan, '__fallback__', 'pytest:')
        assert pytest_run is not None
        assert pytest_run.scope_kind is ScopeKind.SKIPPED


# ── step-9/10: Scenario 6 — classifier tool-isolation (C1); Boundary-test
#              sketch row 6 ──────────────────────────────────────────────────

_CARGO_TOOL_KINDS = [ToolKind.CARGO_TEST, ToolKind.CARGO_CLIPPY]


class TestClassifierToolIsolation:
    """Scenario 6 — GOLDEN C1: each tool's pattern table is its own narrow

    list, never a continuation of another tool's table — proven both
    directions. A cargo run never sees pytest's INTERNALERROR token (a); a
    pytest run never lets a cargo-CLI token swallow its own FAILED line, the
    PRD's headline C1 example (b). Cargo goldens (c) are re-grounded in the
    historical re-grounding commits 1703f86f95/18f57fe922/1aed67cd56/
    264d5b5e8a (tasks 1103/1109/1116); b40a3e0a7f is a cargo-scoping change
    (not classify_failure) and is deliberately excluded (G6).

    RED until step-10 imports classify_failure/FailureCategory from
    orchestrator.verify_classify/orchestrator.verify_categories.
    """

    # ── (a) cargo dispatch never reaches the pytest table ───────────────────
    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_pytest_internalerror_not_reachable_via_cargo(self, tool):
        output = 'INTERNALERROR> pytest crashed unexpectedly\n'
        result = classify_failure(tool, 1, output, False)
        assert result == FailureCategory.UNKNOWN_TEST_FAILURE, (
            f'pytest INTERNALERROR must not leak into the cargo table, got {result!r}'
        )

    # ── (b) reverse signal: a cargo token embedded in pytest output ─────────
    def test_cargo_token_in_pytest_output_still_classifies_test_failure(self):
        output = 'error: no such subcommand: `tset`\nFAILED tests/test_x.py::test_y\n'
        result = classify_failure(ToolKind.PYTEST, 1, output, False)
        assert result == FailureCategory.TEST_FAILURE, (
            f'a cargo CLI token in pytest output must not swallow the FAILED line '
            f'into cargo_cli_error, got {result!r}'
        )

    # ── (c) cargo goldens (tasks 1103/1109/1116) ─────────────────────────────
    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_cargo_cli_error_exclude_pattern(self, tool):
        output = (
            'Compiling my-crate v0.1.0\n'
            'error: --exclude can only be used together with --workspace\n'
        )
        assert classify_failure(tool, 1, output, False) == FailureCategory.CARGO_CLI_ERROR

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_cargo_cli_error_no_such_subcommand(self, tool):
        output = 'error: no such subcommand: `tset`\n'
        assert classify_failure(tool, 1, output, False) == FailureCategory.CARGO_CLI_ERROR

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_compile_error_rustc_code(self, tool):
        output = 'error[E0308]: mismatched types\n  --> src/lib.rs:10:5\n'
        assert classify_failure(tool, 1, output, False) == FailureCategory.COMPILE_ERROR

    @pytest.mark.parametrize('tool', _CARGO_TOOL_KINDS)
    def test_rustc_top_level_diagnostics_are_unknown_test_failure(self, tool):
        output = (
            'Compiling my-crate v0.1.0\n'
            'error: aborting due to previous errors\n'
            'error: could not compile `my-crate` (lib) due to previous error\n'
        )
        assert classify_failure(tool, 1, output, False) == FailureCategory.UNKNOWN_TEST_FAILURE


# ── step-11/12: Scenario 7 — category exhaustiveness (F1); Boundary-test
#               sketch row 7 ─────────────────────────────────────────────────


class TestCategoryExhaustiveness:
    """Scenario 7 — GOLDEN F1: CATEGORY_POLICY must carry exactly one row per

    FailureCategory member, driven via the EXTRACTED _validate_exhaustive
    guard (mirrors test_verify_categories.py's TestValidateExhaustive) rather
    than an importlib.reload+monkeypatch harness — no such harness exists in
    the repo, and the guard IS the real import-time assert's own logic
    factored out into reusable, unit-testable form.

    RED until step-12 imports _validate_exhaustive/CATEGORY_POLICY from
    orchestrator.verify_categories (FailureCategory is already imported).
    """

    def _make_synth(self):
        class _Synth(StrEnum):
            A = 'a'
            B = 'b'

        any_row = next(iter(CATEGORY_POLICY.values()))
        return _Synth, any_row

    def test_missing_member_raises_and_names_it(self):
        """A synthetic member with no policy row fires the F1 guard, naming it."""
        _Synth, any_row = self._make_synth()
        with pytest.raises(AssertionError, match='B'):
            _validate_exhaustive(_Synth, {_Synth.A: any_row})

    def test_real_shipped_table_satisfies_its_own_guard(self):
        """Complement: the landed table already satisfies F1 at import time."""
        _validate_exhaustive(FailureCategory, CATEGORY_POLICY)


# ── step-13/14: Scenario 8 — CheckRun/VerifyAttempt timeout consistency (the
#               verify.py:2735-2744 drift); Boundary-test sketch row 8 ───────


def _check_run(
    label, rc=0, timed_out=False, cmd='cmd', output='', started_at='ts', duration_secs=1.0,
):
    """Build a CheckRun with sane defaults, overriding only what a case cares about.

    Ported from test_verify_attempt.py's ``_run`` builder, renamed
    ``_check_run`` here — this module already imports
    ``orchestrator.git_ops._run`` (the real subprocess runner
    ``_setup_repo`` uses for the real-git-repo fixtures), and a same-named
    module-level def would silently shadow it for every later caller.
    """
    return CheckRun(
        label=label, cmd=cmd, rc=rc, output=output, timed_out=timed_out,
        started_at=started_at, duration_secs=duration_secs,
    )


class TestCheckRunTimeoutConsistency:
    """Scenario 8 — GOLDEN: the result's `timed_out` flag (any_timed_out +

    pure_timeout_failure) and classify_failure's own INFRA_TIMEOUT category
    are single-sourced from the same CheckRun.timed_out via VerifyAttempt, so
    the two can never drift apart (the verify.py:2735-2744 hazard this task's
    ε makes structurally impossible). Mirrors the env-recovery rebuild at
    verify.py:3207-3208 (``attempt = VerifyAttempt([new_test, attempt.lint,
    attempt.type]); timed_out = (not attempt.passed) and
    attempt.pure_timeout_failure``).

    RED until step-14 imports CheckRun/VerifyAttempt from orchestrator.verify
    and ports a CheckRun-builder helper from test_verify_attempt.py (named
    ``_check_run`` here, NOT that module's bare ``_run`` — this file already
    imports ``orchestrator.git_ops._run`` for the real-git-repo fixtures'
    ``_setup_repo``, and a same-named module-level def would silently shadow
    it for every later caller).
    """

    def test_timed_out_test_leg_drives_both_channels_into_agreement(self):
        attempt = VerifyAttempt([
            _check_run('test', rc=1, timed_out=True),
            _check_run('lint', rc=0),
            _check_run('type', rc=0),
        ])
        assert attempt.any_timed_out is True
        assert attempt.pure_timeout_failure is True

        result_timed_out = (not attempt.passed) and attempt.pure_timeout_failure
        assert result_timed_out is True

        category = classify_failure(
            ToolKind.PYTEST, attempt.test.rc, attempt.test.output, attempt.test.timed_out,
        )
        assert category == FailureCategory.INFRA_TIMEOUT

        # The invariant: category==INFRA_TIMEOUT requires attempt.test.timed_out
        # (classify_failure's timed_out guard wins before any output pattern
        # is even consulted), which forces any_timed_out, which forces
        # result_timed_out — both channels read the SAME CheckRun.timed_out,
        # so the category can never flip to infra_timeout while the result
        # stays timed_out=False (the 2735-2744 drift is structurally
        # impossible here).
        assert attempt.test.timed_out is True

    def test_poison_case_real_failure_alongside_timeout_is_not_pure(self):
        attempt = VerifyAttempt([
            _check_run('test', rc=1, timed_out=False),
            _check_run('type', rc=1, timed_out=True),
        ])
        assert attempt.pure_timeout_failure is False
