"""Phase-1 eval↔production integration gate — boundary suite B1–B9 (+ capstone).

PRD eval-framework-revival §ι, Contract H, Boundary-test sketch B1–B9.

This is the "C-as-integration-gate" for the eval-framework-revival PRD: the
two-way eval↔production seam test proving that the mechanisms delivered by the
Phase-1 tasks α–ε *compose* correctly, not merely that each unit works in
isolation. Where α–ε each carry a focused unit test, this suite drives the
REAL composed eval code paths against one refreshed fixture:

  - ``build_eval_orch_config(...)`` from a live ``load_config()`` base  (α/β/D5)
  - ``apply_eval_profile`` / ``EVAL_PROFILE`` parity                (β/D3/D4/D8)
  - ``snapshots.get_diff(worktree, base_commit)``                        (γ/D1)
  - ``RecordingMemorySink`` real-loopback-HTTP write capture             (ε/D8)
  - ``evals.runner._build_eval_scheduler`` → ``_StubMcpSession``         (δ/D2)
  - ``workflow._spawn_dry_run_unblock`` disabled early-return               (D4)
  - ``workflow.build_workflow`` single construction point               (α/P2)
  - ``metrics.compute_composite`` contract-agnostic scoring            (P4/B8)

Each B# faces BOTH sides of its seam: the production default AND the eval-side
divergence, so the test proves the profile actually flips the behaviour rather
than the fixture merely happening to match production.

B# → mechanism map
------------------
  B1  parity              ``eval_profile_divergence(load_config()) == EVAL_PROFILE``
  B2  factory single-pt   a new mandatory ``TaskWorkflow`` arg breaks ``build_workflow()``
  B3  dispatch tripwire   every ``scheduler.py`` ``dispatch_tool`` literal has a stub branch
  B4  non-empty diff      ``get_diff`` yields the full committed diff; the judge grades it
  B5  memory isolation    default url is the null sentinel; the sink captures the write
  B6  no mid-eval rebase  eval config disables ``rebase_before_verify`` + ``inter_iteration_rebase``
  B7  no unblock spawn    ``_spawn_dry_run_unblock`` spawns 0 tasks with ``unblock_auto`` off
  B8  contract-agnostic   ``quality_from_review_artifact`` scores legacy == envelope shape
  B9  no claimant spam    a real eval ``Scheduler`` heartbeat logs zero warnings

Why this suite is deterministic and CI-runnable (design decision)
-----------------------------------------------------------------
A literal ``run_eval`` over an April fixture (df_task_12/13/18) invokes real
Claude agents AND runs the fixture's OWN ``verify_commands`` (``uv sync`` + the
full ``cd orchestrator && uv run pytest tests/ -x`` suite) — slow, costly,
non-deterministic, and OAuth-bound; it cannot run on every commit. The
correctness ι actually gates is the framework SEAMS, all of which ARE
exercised deterministically here against a self-contained committed-diff git
fixture (``committed_diff_fixture`` below). The "one fixture through a real
eval run" signal is honoured by the capstone (``test_capstone_one_fixture_...``)
driving that one fixture through the composed eval code paths.

Operator runbook — the on-demand PAID full-agent run (NOT a CI test)
--------------------------------------------------------------------
To exercise a genuine end-to-end paid eval over a real April fixture (real
architect/implementer/reviewer agents AND the fixture's own ``uv sync`` +
full-pytest verify), an operator runs, from a machine with a live OAuth
session and the eval fixtures + their ``evals/<id>`` branches / pre-task
commits checked out::

    # From the repo root, one contender over one fixture. The eval runner's
    # __main__ routes to orchestrator.cli:eval_cmd; select the task + config
    # and (optionally) point memory writes at a RecordingMemorySink so the
    # intended writes are captured instead of dropped at the null sentinel:
    cd orchestrator
    uv run python -m orchestrator.evals.runner \
        eval --task evals/tasks/df_task_12.json \
             --config claude-sonnet-max \
             --task-timeout-min 180 --orch-timeout-min 150

This spends real budget (~$5+ per contender) and runs the fixture's own
``uv sync`` + full pytest verify inside the eval worktree, so it is
deliberately kept OUT of the automated suite — it is the manual acceptance
step BEHIND the deterministic gate in this module, not part of it. Memory
isolation is automatic (``EVAL_PROFILE`` pins ``fused_memory.url`` to the
non-routable null sentinel); pass ``memory_endpoint=<RecordingMemorySink.url>``
via ``run_eval`` only when you want to capture the writes. See
``plans/eval-framework-revival-prd.md`` §ι for the full Contract-H contract.
"""

from __future__ import annotations

import inspect
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Coroutine

import pytest

import orchestrator
from orchestrator import config as _config
from orchestrator.config import OrchestratorConfig, load_config
from orchestrator.evals import profile as _profile
from orchestrator.evals.profile import EVAL_PROFILE
from orchestrator.mcp import verdict_tools as _verdict_tools

# ---------------------------------------------------------------------------
# Shared committed-diff git fixture (test_snapshots.py _git / tmp_repo
# convention) — a self-contained stand-in for a paid ``evals/<id>`` fixture.
# ---------------------------------------------------------------------------

# The added line landed as a COMMIT on the eval branch. Kept free of any token
# ``judge._strip_metadata`` would rewrite (model names / session ids /
# timestamps) so it survives verbatim into B4's graded prompt.
_ADDED_FILE = 'LANDED.py'
_ADDED_MARKER = 'BOUNDARY_FIXTURE_SENTINEL = "committed_diff_present"'
# An ``evals/<id>``-style branch name for the landed change (D1 shape).
_EVAL_BRANCH = 'evals/df_boundary'


def _git(args: list[str], cwd: Path) -> str:
    """Run a git command in *cwd* and return stripped stdout.

    Mirrors ``test_snapshots.py::_git`` — ``check=True`` so any git failure
    surfaces as a loud ``CalledProcessError`` rather than a silent empty diff.
    """
    return subprocess.run(
        ['git', *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def committed_diff_fixture(tmp_path: Path) -> tuple[Path, str]:
    """Build a tiny git repo with a base commit + a COMMITTED landed change.

    Reproduces the D1 shape the eval framework must grade: the landed change
    is a real *commit* on an ``evals/<id>``-style branch (NOT a working-tree
    edit), so only ``git diff base..HEAD`` (``snapshots.get_diff``) surfaces
    it — the removed metadata-read + uncommitted-only fallback would return
    ''.  Returns ``(worktree_path, base_commit)`` where ``worktree_path`` is a
    git working dir whose ``HEAD`` is the landed commit and ``base_commit`` is
    its main-ancestor parent.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()

    _git(['init', '-q', '-b', 'main'], cwd=repo)
    _git(['config', 'user.email', 'test@example.com'], cwd=repo)
    _git(['config', 'user.name', 'Test User'], cwd=repo)
    _git(['config', 'commit.gpgsign', 'false'], cwd=repo)

    # Base commit (the main-ancestor pre_task_commit).
    (repo / 'README.md').write_text('base\n')
    _git(['add', 'README.md'], cwd=repo)
    _git(['commit', '-q', '-m', 'base commit'], cwd=repo)
    base_commit = _git(['rev-parse', 'HEAD'], cwd=repo)

    # The landed change, COMMITTED on an evals/<id>-style branch.
    _git(['checkout', '-q', '-b', _EVAL_BRANCH], cwd=repo)
    (repo / _ADDED_FILE).write_text(_ADDED_MARKER + '\n')
    _git(['add', _ADDED_FILE], cwd=repo)
    _git(['commit', '-q', '-m', 'landed change'], cwd=repo)

    return repo, base_commit


# ---------------------------------------------------------------------------
# Fake invoke_agent for B4 — records the prompt it receives and returns a
# canned structured verdict, so the judge/compare grading path runs without a
# real (paid, non-deterministic) LLM call.
# ---------------------------------------------------------------------------


def _fake_invoke_agent(
    captured_prompts: list[str],
) -> Callable[..., Coroutine[Any, Any, SimpleNamespace]]:
    """Return an ``invoke_agent`` replacement that records prompts.

    Monkeypatch the returned coroutine over ``judge.invoke_agent`` (or
    ``compare.invoke_agent``); every call appends its ``prompt`` kwarg to
    *captured_prompts* and returns a canned ``JUDGE_SCHEMA``-shaped verdict via
    both ``structured_output`` and ``output`` (the exact cascade ``run_judge``
    reads).  No real agent runs, so B4 grades the real committed diff for free.
    """

    async def _fake(**kwargs: Any) -> SimpleNamespace:
        captured_prompts.append(str(kwargs.get('prompt', '')))
        return SimpleNamespace(
            structured_output={
                'winner': 'A',
                'confidence': 0.9,
                'reasoning': 'canned boundary-suite verdict',
            },
            output='{"winner": "A", "confidence": 0.9, "reasoning": "r"}',
        )

    return _fake


def _load_default_config(tmp_path: Path) -> OrchestratorConfig:
    """Load a deterministic pure-code-default config via the REAL ``load_config()``.

    B1 (and the capstone) must face the production config-LOAD entry point,
    ``load_config`` — not a hand-built ``OrchestratorConfig()`` — yet stay
    deterministic regardless of the host's ``dark-factory-orchestrator.yaml``
    (which could pre-set a profile leaf and make the divergence
    machine-dependent, the exact failure the parity gate must not have).

    The ``code_default_config`` fixture can't serve here: it points
    ``ORCH_CONFIG_PATH`` at an ABSENT file, and ``load_config`` raises
    ``ConfigRequiredError`` on a missing config file (see ``config.load_config``)
    rather than falling through to defaults. Instead we write a minimal config
    setting only ``project_root``; ``load_config`` layers it over the
    package-bundled ``defaults.yaml``, so every profile leaf resolves to its
    pure code default — through the real production entry point.
    """
    cfg_path = tmp_path / 'orchestrator.yaml'
    cfg_path.write_text(f'project_root: {tmp_path}\n')
    return load_config(cfg_path)


# ── B1: config-profile parity (P1 / D5) ────────────────────────────────────


def test_b1_eval_profile_divergence_parity_and_tripwire(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B1 — ``eval_profile_divergence(load_config())`` is EXACTLY ``EVAL_PROFILE``.

    The integration-level parity tripwire, facing the REAL ``load_config()``
    production config-LOAD entry point (distinct from β's unit test, which
    constructs ``OrchestratorConfig`` directly). Two halves:

    (1) Parity — the divergence report's key set equals ``set(EVAL_PROFILE)``
        and every eval-side value equals the documented profile value (incl.
        the D8 ``fused_memory.url`` null sentinel); each base-side value
        genuinely differs, so no leaf is a vacuous no-op.
    (2) Tripwire — a leaked non-profile field (``max_amendment_rounds``, which
        ``EVAL_PROFILE`` never touches) makes the report NAME the offender and
        its key set no longer equal ``set(EVAL_PROFILE)`` — the RED signal B1
        guards.

    Exactness holds by construction: ``apply_eval_profile`` is a
    ``model_copy(update=)`` that changes exactly the profile leaves, and β
    proves all 6 differ from their code defaults, so the divergence is
    precisely the 6 documented keys — never more (a leak) or fewer (a no-op).
    """
    base = _load_default_config(tmp_path)

    divergence = _profile.eval_profile_divergence(base)

    # (1) Parity: exact key-set match against the documented profile.
    assert set(divergence) == set(EVAL_PROFILE)
    for leaf, expected_eval in EVAL_PROFILE.items():
        base_value, eval_value = divergence[leaf]
        assert eval_value == expected_eval, (
            f'{leaf}: eval-side {eval_value!r} != documented {expected_eval!r}'
        )
        assert base_value != expected_eval, (
            f'{leaf}: base already equals its profile value — vacuous divergence'
        )

    # (2) Tripwire: an undocumented leak trips the parity check and is named.
    # eval_profile_divergence computes apply_eval_profile(base) vs base, so we
    # simulate a leaked production field by having apply_eval_profile ALSO
    # change a non-profile leaf (max_amendment_rounds). Patched on the module
    # so eval_profile_divergence's own module-global call resolves the leak.
    real_apply = _profile.apply_eval_profile

    def _leaky_apply(cfg: OrchestratorConfig) -> OrchestratorConfig:
        return real_apply(cfg).model_copy(
            update={'max_amendment_rounds': cfg.max_amendment_rounds + 1},
        )

    monkeypatch.setattr(_profile, 'apply_eval_profile', _leaky_apply)

    leaked = _profile.eval_profile_divergence(base)

    assert set(leaked) != set(EVAL_PROFILE)
    assert 'max_amendment_rounds' in leaked


# ── B8: contract-agnostic scoring (P4) ─────────────────────────────────────

# The single reviewer payload shape both contract eras reduce to: 1 blocking +
# 2 suggestion issues. ``write_review`` persists this dict verbatim to
# ``reviews/<name>.json`` (legacy ``--json-schema`` era); the MCP verdict-tool
# era wraps the SAME payload in ``_envelope(...)['verdict']`` under
# ``verdicts/<role>.json``. Both reduce to identical blocking/suggestion counts.
_B8_REVIEWER = 'reviewer_comprehensive'
_B8_PAYLOAD = {
    'reviewer': _B8_REVIEWER,
    'verdict': 'ISSUES_FOUND',
    'issues': [
        {'severity': 'blocking', 'description': 'b1', 'location': 'm/x.py:1'},
        {'severity': 'suggestion', 'description': 's1', 'location': 'm/x.py:2'},
        {'severity': 'suggestion', 'description': 's2', 'location': 'm/x.py:3'},
    ],
    'summary': 'one blocking, two suggestions',
}


def test_b8_quality_from_review_artifact_contract_agnostic() -> None:
    """B8 — one score from BOTH the legacy payload and the verdict-tool envelope.

    Scoring flows reviews/<name>.json → aggregate_reviews (blocking vs
    suggestion split by ``severity``) → compute_composite. The MCP verdict-tool
    path writes a schema-versioned envelope ``{role, schema_version, session_id,
    emitted_at, verdict: <payload>}``; ``_run_reviewer`` unwraps
    ``envelope['verdict']`` to the SAME payload the legacy ``--json-schema``
    reviewer persists. So both contract eras reduce to one payload → one score.

    Asserts:
    (1) identical float score from the legacy payload and the
        ``_envelope``-wrapped payload (exact by construction — same
        blocking/suggestion counts → same compute_composite);
    (2) content-sensitivity — a PASS / zero-issue artifact scores strictly
        higher than the 1-blocking artifact (the score is not a constant);
    (3) scoring reads ONLY the artifact dict + scalar knobs (no transcript /
        file-path / worktree argument) — the P4 substrate μ's driver consumes.

    RED: ``orchestrator.evals.scoring`` / ``quality_from_review_artifact`` does
    not exist yet.
    """
    from orchestrator.evals import scoring

    legacy_artifact = dict(_B8_PAYLOAD)
    envelope_artifact = _verdict_tools._envelope(
        _B8_REVIEWER, 'sid-b8', dict(_B8_PAYLOAD),
    )

    plan_steps = 5
    legacy_score = scoring.quality_from_review_artifact(
        legacy_artifact, plan_steps=plan_steps,
    )
    envelope_score = scoring.quality_from_review_artifact(
        envelope_artifact, plan_steps=plan_steps,
    )

    # (1) Contract-agnostic: identical score from both persisted shapes.
    assert isinstance(legacy_score, float)
    assert legacy_score == envelope_score

    # (2) Content-sensitive: a clean PASS artifact scores strictly higher.
    clean_artifact = {
        'reviewer': _B8_REVIEWER,
        'verdict': 'PASS',
        'issues': [],
        'summary': 'clean',
    }
    clean_score = scoring.quality_from_review_artifact(
        clean_artifact, plan_steps=plan_steps,
    )
    assert clean_score > legacy_score

    # (3) Reads only the artifact dict + scalar knobs — never a transcript,
    # file path, or worktree. Pins the signature so no such argument creeps in.
    params = set(inspect.signature(scoring.quality_from_review_artifact).parameters)
    assert params == {'artifact', 'plan_steps', 'debug_cycles'}


# ── B2: factory single construction point (P2) ─────────────────────────────

# ``TaskWorkflow(`` not preceded by a word char or dot — a construction call,
# not ``build_workflow`` / ``class TaskWorkflow:`` / an attribute access. Reused
# verbatim from test_workflow_factory.py (composed into the boundary suite).
_CONSTRUCT_RE = re.compile(r"(?<![\w.])TaskWorkflow\(")
_FACTORY_CALL_RE = re.compile(r"(?<![\w.])build_workflow\(")


def _orchestrator_src_root() -> Path:
    """Resolve ``orchestrator/src/orchestrator/`` from the imported package."""
    return Path(orchestrator.__file__).resolve().parent


def test_b2_factory_single_construction_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """B2 — a new mandatory ``TaskWorkflow`` arg breaks the single build site.

    Two composed halves that together encode Invariant P2 (single-point
    ownership); neither alone is sufficient:

    (1) Runtime: replace ``orchestrator.workflow.TaskWorkflow`` with a subclass
        whose ``__init__`` demands a NEW mandatory kwarg the factory does not
        pass, then assert ``build_workflow(...)`` raises ``TypeError`` — proving
        a new required ``TaskWorkflow`` param breaks the single construction
        point, hence BOTH dispatch sites (harness.py + evals/runner.py, which
        both route through ``build_workflow``) at once.
    (2) Static: ``workflow.py`` is the ONLY production-source file constructing
        ``TaskWorkflow(``, and both dispatch sites contain a ``build_workflow(``
        call — proving there is no third bypass site.
    """
    from orchestrator import workflow as workflow_mod
    from orchestrator.workflow import build_workflow

    # (1) Runtime: new mandatory kwarg the factory never threads → TypeError.
    class _NeedsExtraParam:
        def __init__(self, *, new_required_param: object, **kwargs: object) -> None:
            self.new_required_param = new_required_param

    monkeypatch.setattr(workflow_mod, 'TaskWorkflow', _NeedsExtraParam)
    with pytest.raises(TypeError):
        build_workflow(
            assignment=object(),  # type: ignore[arg-type]
            config=object(),  # type: ignore[arg-type]
            git_ops=object(),  # type: ignore[arg-type]
            scheduler=object(),  # type: ignore[arg-type]
            briefing=object(),  # type: ignore[arg-type]
            mcp=object(),  # type: ignore[arg-type]
        )

    # (2) Static single-point (compose the existing grep-guard invariant).
    src_root = _orchestrator_src_root()
    offenders: set[str] = set()
    for path in src_root.rglob('*.py'):
        text = path.read_text(encoding='utf-8', errors='ignore')
        for line in text.splitlines():
            code = line.split('#', 1)[0]  # drop trailing comment
            if _CONSTRUCT_RE.search(code):
                offenders.add(path.relative_to(src_root).as_posix())
                break
    assert offenders == {'workflow.py'}, (
        f'Direct TaskWorkflow(...) construction is only permitted in workflow.py; '
        f'offending files: {sorted(offenders)}'
    )

    for rel in ('harness.py', 'evals/runner.py'):
        source = (src_root / rel).read_text(encoding='utf-8')
        assert _FACTORY_CALL_RE.search(source), (
            f'{rel} must construct its TaskWorkflow via build_workflow('
        )
