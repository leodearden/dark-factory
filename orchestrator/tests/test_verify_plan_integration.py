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
import shutil
import subprocess
from enum import StrEnum
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Cross-file test-module imports: an established convention in this suite
# (test_merge_queue_dry_run_unblock.py imports these same test_dry_run_unblock
# helpers), but one that couples this module to the internal helper shapes of
# sibling test files — a rename or signature change in either breaks this
# module with no compile-time guard. ROOT_CONFTEST_DIFF/DATA_MODULE_DIFF/
# STRUCTURAL_DIFF below are deliberately NOT imported this same way (this
# module keeps its own copies, see the comment above ROOT_CONFTEST_DIFF) to
# avoid adding a second, tighter such dependency for its core golden
# fixtures; test_golden_diffs_match_test_verify_plan_source() near the end of
# the plan-golden section turns that copy's silent-drift risk into a loud
# test failure instead.
from test_dry_run_unblock import _make_agent_result, _RecordingScheduler
from test_verify_plan import (
    DATA_MODULE_DIFF as _SRC_DATA_MODULE_DIFF,
)
from test_verify_plan import (
    ROOT_CONFTEST_DIFF as _SRC_ROOT_CONFTEST_DIFF,
)
from test_verify_plan import (
    STRUCTURAL_DIFF as _SRC_STRUCTURAL_DIFF,
)

from orchestrator import verify
from orchestrator.b3_gate import ABORT, FRESH, POST_MERGE_RED_MAIN_REASON_PREFIX, check_proposal
from orchestrator.config import GitConfig, ModuleConfig, OrchestratorConfig
from orchestrator.git_ops import PROTECTED_PREFIXES, GitOps, WorktreeKind, _run
from orchestrator.merge_queue import (
    MergeRequest,
    QueuedBranch,
    _DryRunInvestigationHandles,
    _run_post_merge_verify,
)
from orchestrator.unblock_types import BlockClass, BlockRecord
from orchestrator.verify import (
    CheckRun,
    VerifyAttempt,
    VerifyResult,
    run_main_tip_sweep,
    verify_failure_is_preexisting_on_main,
)
from orchestrator.verify_categories import CATEGORY_POLICY, FailureCategory, _validate_exhaustive
from orchestrator.verify_classify import classify_failure
from orchestrator.verify_cmd import (
    ToolKind,
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
        branch=QueuedBranch.parse(branch, config.git.branch_prefix),
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
    @pytest.mark.skipif(
        shutil.which('uv') is None,
        reason='requires uv on PATH (and a synced orchestrator venv) to exec the real scoped pytest command',
    )
    @pytest.mark.slow  # heavyweight: real `uv run ... pytest` subprocess; deselect with -m "not slow and not warm_lane_bash"
    def test_scoped_pytest_producer_and_runner_agree_on_scope(self, tmp_path):
        """scope_to's structured output (producer) IS what render+exec (runner) runs.

        Producer side: parse the real orchestrator.yaml test_command and
        scope_to() it down to one throwaway probe file. Runner side: render()
        that scoped VerifyCmd (strip_cwd'd so it runs unchanged from the repo
        root with an absolute probe path) and ACTUALLY EXECUTE it via
        `bash -c` — if the two sides disagreed on what "scoped" means, this
        would either fail to launch or collect more than the one scoped test.

        This is the only scenario in the module that shells a real external
        toolchain (uv + a synced orchestrator venv) instead of driving the
        decision layer in-process, so it is environment-coupled: marked
        ``@pytest.mark.slow`` for deselection (``-m "not slow and not warm_lane_bash"``) and skipped
        outright when `uv` isn't on PATH, rather than flaking a fast/offline
        run. The in-process argv assertions above (scenario 1a) already prove
        producer<->runner scope agreement and remain the always-run portion.
        The `slow` marker is registered in orchestrator/pyproject.toml's
        `markers` list as of task 3506 — see that entry for the deselect-
        composition rationale.
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
# Reconstructed verbatim from test_verify_plan.py's own golden fixture (kept
# as a local copy rather than a direct import, to avoid a second tight
# cross-file dependency alongside the test_dry_run_unblock import above) —
# test_golden_diffs_match_test_verify_plan_source() near the end of the
# plan-golden section below guards this copy (and DATA_MODULE_DIFF/
# STRUCTURAL_DIFF) against silently drifting from that source.
ROOT_CONFTEST_DIFF: list[str] = ['orchestrator/tests/conftest.py']


def _make_fake_worktree_reader(contents: dict[str, str] | None = None):
    """Build a ``Callable[[str], str | None]`` stand-in for real file I/O.

    Keeps derive_verify_plan pure and unit-testable without touching a real
    filesystem: returns canned content for paths seeded into *contents*,
    else None. *contents* is copied into a closure-local dict rather than
    read from a shared module-level mutable global, so each scenario's
    canned file contents are isolated by construction — a future scenario
    that seeds a different value for an overlapping path can never observe,
    or be observed by, another scenario's reader.
    """
    seeded = dict(contents) if contents else {}

    def _reader(path: str) -> str | None:
        return seeded.get(path)

    return _reader


# Scenarios 3+4 need no canned content: every path they touch (this
# scenario's ROOT_CONFTEST_DIFF file, scenario 4's DATA_MODULE_DIFF file
# below) must read back as None, which classify_file treats as "not
# detected", never an error.
fake_worktree_reader = _make_fake_worktree_reader()


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
        # Assert on the semantically meaningful content the reason contract
        # actually promises — the offending path is named and the run is an
        # explicit reasoned skip, never a silent None. Deliberately NOT
        # asserting a bare '1852' task-number substring here: a harmless
        # rewording of the SKIPPED reason that still explains the skip would
        # break that check without any real behaviour regression (the
        # task-1852 provenance is already documented on DATA_MODULE_DIFF
        # above, per G6).
        assert DATA_MODULE_DIFF[0] in run.reason


# ── step-7/8: Scenario 5 — plan golden: structural file -> unscoped pyright,
#              module + fallback paths (D2); Boundary-test sketch row 5 ─────

# A Protocol-defining source file (D2): file-scoped pyright cannot verify
# cross-file Protocol conformance, so a STRUCTURAL file must widen pyright to
# the unscoped package-wide command in BOTH the module and fallback paths.
# Reconstructed verbatim from test_verify_plan.py's own golden fixture.
STRUCTURAL_DIFF: list[str] = ['orchestrator/src/orchestrator/interfaces.py']

# STRUCTURAL is only detected when a type_check_command is configured (so
# content is actually read); this scenario gets its OWN reader, seeded only
# with STRUCTURAL_DIFF[0]'s Protocol-bearing content — isolated by
# construction from the scenario-3/4 reader above (and from any future
# scenario), not merely by convention.
_structural_worktree_reader = _make_fake_worktree_reader(
    {STRUCTURAL_DIFF[0]: 'class Foo(Protocol):\n    def m(self) -> None: ...\n'}
)


class TestPlanGoldenStructural:
    """Scenario 5 — GOLDEN D2: a Protocol-bearing source file widens pyright

    to the unscoped FULL_SUITE command (never file-scoped — cross-file
    Protocol conformance can't be checked from one file) in BOTH the
    module-config path and the fallback path (the latent gap
    _build_fallback_config never closed).

    pytest: the module-config path now full-suites pytest too, via the
    task-role pytest floor (λ, task 2589 R3) — a structural-only diff counts
    as source, non-test .py, so the default role='task' floors it to
    FULL_SUITE instead of the pre-λ SKIPPED. The fallback path is NOT
    covered by the floor (module-config-branch only — no owning-module
    suite to floor to) and keeps the legacy SKIPPED shape.

    RED until step-8 builds this class's own reader (_structural_worktree_reader)
    seeded with STRUCTURAL_DIFF[0]'s Protocol-bearing content — STRUCTURAL is
    only detected when content is read, and an unseeded path always reads
    back None.
    """

    def test_structural_file_full_suites_pyright_module_path(self):
        mc = ModuleConfig(
            prefix='orchestrator',
            test_command='uv run --project orchestrator --directory orchestrator pytest tests/',
            type_check_command=(
                'uv run --project orchestrator --directory orchestrator pyright src/ tests/'
            ),
        )
        plan = derive_verify_plan(STRUCTURAL_DIFF, [mc], None, _structural_worktree_reader)

        pyright_run = _run_for(plan, 'orchestrator', 'pyright:')
        assert pyright_run is not None
        assert pyright_run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.type_check_command is not None
        assert pyright_run.cmd == parse_config_command(mc.type_check_command)
        assert STRUCTURAL_DIFF[0] in pyright_run.reason

        pytest_run = _run_for(plan, 'orchestrator', 'pytest:')
        assert pytest_run is not None
        assert pytest_run.scope_kind is ScopeKind.FULL_SUITE
        assert mc.test_command is not None
        assert pytest_run.cmd == parse_config_command(mc.test_command)

    def test_structural_file_full_suites_pyright_fallback_path(self):
        config = OrchestratorConfig(project_root=Path('/fake'), test_command='pytest')
        plan = derive_verify_plan(STRUCTURAL_DIFF, [], config, _structural_worktree_reader)

        pyright_run = _run_for(plan, '__fallback__', 'pyright:')
        assert pyright_run is not None
        assert pyright_run.scope_kind is ScopeKind.FULL_SUITE
        assert pyright_run.cmd == parse_config_command(config.type_check_command)
        assert STRUCTURAL_DIFF[0] in pyright_run.reason

        pytest_run = _run_for(plan, '__fallback__', 'pytest:')
        assert pytest_run is not None
        assert pytest_run.scope_kind is ScopeKind.SKIPPED


def test_golden_diffs_match_test_verify_plan_source():
    """Guard the three verbatim-copied golden diffs (scenarios 3-5) against
    silent drift from their source of truth in test_verify_plan.py.

    ROOT_CONFTEST_DIFF/DATA_MODULE_DIFF/STRUCTURAL_DIFF above are
    deliberately this module's OWN copies rather than a direct import — this
    integration-gate module already carries one cross-file dependency (the
    test_dry_run_unblock helpers imported at module top) and reconstructing
    the diffs keeps it from taking on a second, tighter one on
    test_verify_plan.py's internals for its core golden fixtures. But a copy
    can silently drift from its source. This test converts that risk into a
    loud, isolated test failure: if test_verify_plan.py's own goldens are
    ever renamed or their values updated without a matching update here,
    this fails instead of the two modules quietly asserting different things
    against the same historical incidents (task-1077 / task-1852).
    """
    assert ROOT_CONFTEST_DIFF == _SRC_ROOT_CONFTEST_DIFF
    assert DATA_MODULE_DIFF == _SRC_DATA_MODULE_DIFF
    assert STRUCTURAL_DIFF == _SRC_STRUCTURAL_DIFF


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


# ── step-15/16: Scenario 9 — merge-verify block -> gateable proposal (the
#               coverage gap + B4); Boundary-test sketch row 9 ──────────────


class TestMergeVerifyBlockProducesGateableProposal:
    """Scenario 9 — GOLDEN: a merge-verify RED (generic task-fault, non-timeout)

    must, via the REAL (unpatched) _run_post_merge_verify + run_dry_run_unblock,
    write a dry_run_proposals[] entry with block_class=MERGE_VERIFY_RED that
    b3_gate.check_proposal accepts as non-ABORT (B4) — closing the coverage
    gap where merge_queue's post-merge-verify block path produced a
    MergeOutcome('blocked') but never spawned an investigation, leaving
    check_proposal permanently ABORT ('no proposal to gate') for the entire
    merge-verify-RED class.

    FAKED (boundary only): run_scoped_verification (-> a failing non-timeout
    VerifyResult), verify_failure_is_preexisting_on_main (-> not preexisting,
    so the generic task-fault branch is reached rather than main-health-red),
    and orchestrator.dry_run_unblock.invoke_agent (-> a low-risk structured
    proposal). Everything else — GitOps, the real git_repo, the real
    _run_post_merge_verify block path, the fire-and-forget spawn,
    run_dry_run_unblock's own git-anchor capture, and check_proposal — runs
    for real.

    RED until step-16 imports _run_post_merge_verify/_DryRunInvestigationHandles
    from orchestrator.merge_queue, check_proposal/ABORT from
    orchestrator.b3_gate, BlockClass from orchestrator.unblock_types,
    VerifyResult from orchestrator.verify, and _RecordingScheduler/
    _make_agent_result from test_dry_run_unblock (cross-file convention
    already used by test_merge_queue_dry_run_unblock.py).
    """

    def test_merge_verify_red_produces_gateable_proposal(
        self, tmp_path, config, git_ops, git_repo,
    ):
        merge_wt = tmp_path / 'merge-wt'
        merge_wt.mkdir()

        scheduler = _RecordingScheduler({'dry_run_proposals': []})
        handles = _DryRunInvestigationHandles(scheduler=scheduler)

        compile_error_result = VerifyResult(
            passed=False,
            test_output='',
            lint_output='',
            type_output='error TS2322: StatusBar.tsx:12',
            summary='tsc failed',
            cause_hint='error TS2322: StatusBar.tsx',
            category='compile_error',
        )
        structured = {
            'proposal_text': 'Fix the scoped lint failure',
            'risk_label': 'low',
            'files_referenced': ['orchestrator/src/orchestrator/foo.py'],
        }
        agent_result = _make_agent_result(structured_output=structured)

        reqs: list = []

        async def _drive():
            # _make_req must run inside the loop it builds a Future against.
            req = _make_req('99', 'task/99', config, git_repo)
            reqs.append(req)
            with (
                patch(
                    'orchestrator.merge_queue.run_scoped_verification',
                    new=AsyncMock(return_value=compile_error_result),
                ),
                patch(
                    'orchestrator.merge_queue.verify_failure_is_preexisting_on_main',
                    new=AsyncMock(return_value=(False, '')),
                ),
                patch(
                    'orchestrator.dry_run_unblock.invoke_agent',
                    new=AsyncMock(return_value=agent_result),
                ),
                # task 2633: run_dry_run_unblock clamps a low-risk
                # MERGE_VERIFY_RED proposal to 'human-review-required' unless
                # the run's event history proves merge-completion eligibility
                # ((b) a passing workflow_verify AND (c) a phase_enter(merge)).
                # This GOLDEN scenario models the eligible happy path (verify+
                # review passed, landing jammed), so stub the predicate True and
                # the proposal stays a gateable low-risk one.
                patch(
                    'orchestrator.dry_run_unblock.merge_completion_eligible',
                    return_value=True,
                ),
            ):
                outcome = await _run_post_merge_verify(
                    git_ops, req, merge_wt,
                    timeouts={},
                    enospc_retries={},
                    max_timeouts=3,
                    max_enospc=1,
                    dry_run_handles=handles,
                    event_store=MagicMock(),
                )
                # Drain the fire-and-forget investigation (real
                # run_dry_run_unblock, real git subprocess calls against
                # req.worktree, mocked invoke_agent) before the loop tears
                # down.
                await asyncio.sleep(0)
                if handles.background_tasks:
                    await asyncio.gather(
                        *handles.background_tasks, return_exceptions=True,
                    )
                return outcome

        outcome = asyncio.run(_drive())
        req = reqs[0]
        assert outcome is not None
        assert outcome.status == 'blocked'

        proposals = scheduler._meta.get('dry_run_proposals', [])
        assert proposals, 'Expected a dry_run_proposals entry to be written'
        entry = proposals[-1]
        assert entry['block_class'] == BlockClass.MERGE_VERIFY_RED, (
            f"Expected block_class=MERGE_VERIFY_RED; got {entry.get('block_class')!r}"
        )
        assert entry['risk_label'] == 'low', (
            f"Expected risk_label='low'; got {entry.get('risk_label')!r}"
        )

        def _fake_run_git(args: list[str], cwd: str) -> tuple[int, str]:
            """HEAD always matches the recorded sha; footprint diff is empty."""
            if 'rev-parse' in args:
                return (0, entry['head_sha'])
            return (0, '')

        verdict = check_proposal(
            entry, worktree=str(req.worktree), category='task_failure',
            run_git=_fake_run_git,
        )
        assert verdict['verdict'] != ABORT, (
            f'Expected a non-ABORT (gateable) verdict; got {verdict!r}'
        )


# ── step-17/18: Scenarios 10+11 — b3_gate proposal routing (B2/B3) + BlockRecord
#               round-trip (B1); Boundary-test sketch rows 10-11 ─────────────


class TestB3GateProposalRouting:
    """Scenarios 10+11 — b3_gate.check_proposal's dual-read routing (B2/B3)

    plus the BlockRecord wire round-trip (B1).

    Row 10/B2 (task-1680): a typed block_class=POST_MERGE_RED_MAIN entry
    hard-aborts BEFORE risk_label or any git check — regardless of
    risk_label. Row 11/B3: the legacy dual-read bridge — a pre-block_class
    proposal still aborts via the prose-prefix sniff (block_reason startswith
    POST_MERGE_RED_MAIN_REASON_PREFIX) or the status-key-presence sniff, but
    a TYPED entry with a stray 'status' key must NOT be caught by that
    legacy status sniff (it is gated on block_class is None). B1: BlockRecord
    is a wire-round-trip-safe frozen dataclass.

    RED until step-18 imports FRESH/POST_MERGE_RED_MAIN_REASON_PREFIX from
    orchestrator.b3_gate and BlockRecord from orchestrator.unblock_types.
    """

    _INVESTIGATED_AT = '2026-06-04T09:00:00+00:00'

    # ── scenario 10 / Row 10 / B2 ────────────────────────────────────────
    def test_post_merge_red_main_block_class_hard_aborts_before_git(self):
        """block_class=post_merge_red_main aborts before risk/git, either risk_label."""
        for risk_label in ('low', 'human-review-required'):
            calls: list = []

            def _spy(args, cwd, _calls=calls):
                _calls.append(args)
                return (0, 'deadbeef')

            entry = {
                'block_class': BlockClass.POST_MERGE_RED_MAIN,
                'risk_label': risk_label,
                'head_sha': 'aaabbbccc',
                'main_sha': 'dddeeefff',
                'files_referenced': ['foo.py'],
                'investigated_at': self._INVESTIGATED_AT,
            }
            verdict = check_proposal(
                entry, worktree='/tmp', category=None, run_git=_spy,
            )
            assert verdict['verdict'] == ABORT, (
                f'risk_label={risk_label!r}: expected ABORT, got {verdict!r}'
            )
            assert calls == [], (
                f'run_git must never be called for POST_MERGE_RED_MAIN '
                f'(risk_label={risk_label!r}); got calls={calls}'
            )

    # ── scenario 11 / Row 11 / B3 legacy bridge ──────────────────────────
    def test_legacy_prose_prefix_without_block_class_aborts(self):
        """No block_class, but block_reason carries the post-merge prefix -> ABORT."""

        def _never_called(args, cwd):
            raise AssertionError(f'git should not have been called; args={args}')

        entry = {
            'risk_label': 'low',
            'block_reason': POST_MERGE_RED_MAIN_REASON_PREFIX + ': type-check failed on main',
            'head_sha': 'aaabbbccc',
            'main_sha': 'dddeeefff',
            'files_referenced': [],
            'investigated_at': self._INVESTIGATED_AT,
        }
        verdict = check_proposal(
            entry, worktree='/tmp', category=None, run_git=_never_called,
        )
        assert verdict['verdict'] == ABORT, f'expected ABORT via legacy prefix, got {verdict!r}'

    def test_legacy_status_key_without_block_class_aborts(self):
        """No block_class, but a 'status' key (failure entry) -> ABORT via status-sniff."""

        def _never_called(args, cwd):
            raise AssertionError(f'git should not have been called; args={args}')

        entry = {
            'risk_label': 'low',
            'status': 'investigation_failed',
            'head_sha': 'aaabbbccc',
            'main_sha': 'dddeeefff',
            'files_referenced': [],
            'investigated_at': self._INVESTIGATED_AT,
        }
        verdict = check_proposal(
            entry, worktree='/tmp', category=None, run_git=_never_called,
        )
        assert verdict['verdict'] == ABORT, f'expected ABORT via status sniff, got {verdict!r}'

    def test_typed_entry_with_stray_status_key_does_not_abort_via_status_sniff(self):
        """A TYPED (block_class-bearing) entry with a stray 'status' key must
        reach FRESH — the legacy status sniff is gated on block_class is None
        and must not misfire on the typed path (bridge routes identically to
        pre-change behaviour ONLY on the legacy path)."""

        def _fake_git_fresh(args, cwd):
            if 'rev-parse' in args:
                return (0, 'aaabbbccc')
            return (0, '')

        entry = {
            'block_class': BlockClass.MERGE_VERIFY_RED,
            'risk_label': 'low',
            'status': 'x',
            'head_sha': 'aaabbbccc',
            'main_sha': 'dddeeefff',
            'files_referenced': [],
            'investigated_at': self._INVESTIGATED_AT,
        }
        verdict = check_proposal(
            entry, worktree='/tmp', category=None, run_git=_fake_git_fresh,
        )
        assert verdict['verdict'] == FRESH, (
            f'status-key presence must not abort the typed path, got {verdict!r}'
        )

    # ── B1: BlockRecord wire round-trip ──────────────────────────────────
    def test_block_record_round_trip_full(self):
        record = BlockRecord(
            block_class=BlockClass.MERGE_VERIFY_RED,
            risk_label='low',
            head_sha='h',
            main_sha='m',
            files_referenced=['f'],
            investigated_at='ts',
        )
        assert BlockRecord.from_dict(record.to_dict()) == record

    def test_block_record_round_trip_none_shas_and_empty_files(self):
        record = BlockRecord(
            block_class=BlockClass.AGENT_FAILURE,
            risk_label='human-review-required',
            head_sha=None,
            main_sha=None,
            files_referenced=[],
            investigated_at='',
        )
        assert BlockRecord.from_dict(record.to_dict()) == record


# ── step-19/20: Scenario 12 — ephemeral_worktree no-prune across both probes
#               (E1/E2); Boundary-test sketch row 12 ────────────────────────

_MAIN_TIP_SHA = 'c' * 40


def _make_fake_git_ops_run(add_rcs: list[int], calls: list[list[str]]):
    """Fake ``orchestrator.git_ops._run`` recording every argv into *calls*.

    Ported from test_ephemeral_worktree.py's ``_make_fake_run``. ``git
    worktree add`` return codes are consumed in order from *add_rcs* (the
    last entry repeats once exhausted). A successful add mkdirs the
    ``--detach`` target, mirroring what real ``git worktree add`` does, so
    the CM's unconditional ``shutil.rmtree`` has something real on disk to
    remove. Every other command (e.g. ``git worktree remove``) always
    succeeds.
    """
    state = {'add_calls': 0}

    async def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if 'worktree' in cmd and 'add' in cmd:
            idx = state['add_calls']
            rc = add_rcs[idx] if idx < len(add_rcs) else add_rcs[-1]
            state['add_calls'] += 1
            if rc == 0:
                detach_idx = cmd.index('--detach')
                Path(cmd[detach_idx + 1]).mkdir(parents=True, exist_ok=True)
            return (rc, '', '' if rc == 0 else 'lock contention')
        return (0, '', '')

    return _fake_run


_MAIN_TIP_PASSING_RESULT = VerifyResult(
    passed=True, test_output='', lint_output='', type_output='',
    summary='all checks passed',
)


class TestEphemeralWorktreeNoPrune:
    """Scenario 12 — GOLDEN: both main-tip probes

    (verify_failure_is_preexisting_on_main, run_main_tip_sweep) build their
    throwaway worktree via the REAL GitOps.ephemeral_worktree CM and NEVER
    issue a broad ``git worktree prune`` (DD5/E1 — the incident-prevention
    invariant this task exists to add), and both WorktreeKind prefixes are
    E2-registered in PROTECTED_PREFIXES.

    REAL: GitOps over the module's real git_repo fixture; the CM's own
    add/remove/cleanup control flow.
    FAKED (boundary only): orchestrator.git_ops._run (an argv-recording fake
    standing in for the ssh/subprocess boundary) and the inner verifies
    (run_scoped_verification / run_full_verification -> a passing
    VerifyResult; the preexisting-on-main probe's failing_result carries a
    category/cause_hint UNIQUE to this test so its key can never collide
    with another test's entry in the process-wide, TTL-cached
    verify._PROBE_CACHE).

    RED until step-20 imports verify_failure_is_preexisting_on_main +
    run_main_tip_sweep from orchestrator.verify and WorktreeKind +
    PROTECTED_PREFIXES from orchestrator.git_ops.
    """

    def test_verify_failure_is_preexisting_on_main_routes_through_cm_no_prune(
        self, tmp_path, config, git_ops,
    ):
        # Clear the process-wide TTL probe cache first. verify_failure_is_
        # preexisting_on_main short-circuits ephemeral_worktree entirely on a
        # cache hit keyed by (main_sha, category, normalised cause_hint); the
        # category/cause_hint below are unique to THIS test to dodge
        # collisions with other tests, but a same-process rerun of this exact
        # test (pytest-repeat, a rerun-on-failure plugin, etc.) would replay
        # the identical key within the 300s TTL and fail
        # `spied_cm.call_count >= 1` spuriously without this reset.
        verify._PROBE_CACHE.clear()

        git_ops.get_main_sha = AsyncMock(return_value=_MAIN_TIP_SHA)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)
        worktree = tmp_path / 'task-wt'
        worktree.mkdir()

        failing_result = VerifyResult(
            passed=False, test_output='', lint_output='', type_output='',
            summary='iota_scenario12_mainprobe_signal',
            cause_hint='task iota scenario 12 ephemeral_worktree routing signal (mainprobe)',
            category='task_iota_scenario12_mainprobe',
        )

        calls: list = []
        with (
            patch.object(
                verify, 'run_scoped_verification',
                new=AsyncMock(return_value=_MAIN_TIP_PASSING_RESULT),
            ),
            patch('orchestrator.git_ops._run', side_effect=_make_fake_git_ops_run([0], calls)),
            patch.object(
                git_ops, 'ephemeral_worktree', wraps=git_ops.ephemeral_worktree,
            ) as spied_cm,
        ):
            asyncio.run(
                verify_failure_is_preexisting_on_main(
                    worktree, config, [], ['src/foo.tsx'], failing_result, git_ops,
                )
            )

        assert spied_cm.call_count >= 1, (
            'expected verify_failure_is_preexisting_on_main to build its probe '
            'worktree via git_ops.ephemeral_worktree(); got 0 calls'
        )
        assert not any(
            'worktree' in c and 'prune' in c for c in calls
        ), f'probe must NEVER issue a prune argv (DD5/E1); got calls={calls}'
        assert any(
            'worktree' in c and 'remove' in c and '--force' in c for c in calls
        ), f'expected a scoped "git worktree remove --force" argv; got calls={calls}'

    def test_run_main_tip_sweep_routes_through_cm_no_prune(self, config, git_ops):
        git_ops.get_main_sha = AsyncMock(return_value=_MAIN_TIP_SHA)
        git_ops.worktree_base.mkdir(parents=True, exist_ok=True)

        calls: list = []

        async def _fake_full_verify(project_root, cfg, **kwargs):
            return _MAIN_TIP_PASSING_RESULT

        with (
            patch.object(verify, 'run_full_verification', side_effect=_fake_full_verify),
            patch('orchestrator.git_ops._run', side_effect=_make_fake_git_ops_run([0], calls)),
            patch.object(
                git_ops, 'ephemeral_worktree', wraps=git_ops.ephemeral_worktree,
            ) as spied_cm,
        ):
            result = asyncio.run(run_main_tip_sweep(config, git_ops))

        assert result is not None, f'expected a (sha, VerifyResult) tuple, got {result!r}'
        assert spied_cm.call_count >= 1, (
            'expected run_main_tip_sweep to build its sweep worktree via '
            'git_ops.ephemeral_worktree(); got 0 calls'
        )
        assert not any(
            'worktree' in c and 'prune' in c for c in calls
        ), f'probe must NEVER issue a prune argv (DD5/E1); got calls={calls}'
        assert any(
            'worktree' in c and 'remove' in c and '--force' in c for c in calls
        ), f'expected a scoped "git worktree remove --force" argv; got calls={calls}'

    def test_both_worktree_kind_prefixes_are_e2_registered(self):
        assert WorktreeKind.MAIN_PROBE.value in PROTECTED_PREFIXES, (
            f'expected MAIN_PROBE prefix in PROTECTED_PREFIXES; got keys='
            f'{list(PROTECTED_PREFIXES)!r}'
        )
        assert WorktreeKind.MAIN_SWEEP.value in PROTECTED_PREFIXES, (
            f'expected MAIN_SWEEP prefix in PROTECTED_PREFIXES; got keys='
            f'{list(PROTECTED_PREFIXES)!r}'
        )
