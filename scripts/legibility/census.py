#!/usr/bin/env python3
"""The legibility census CLI: ``census.py --project-root <root> [--force] ...``.

:func:`main` gates on ``census_trigger.decide_for_project`` unless
``--force``, wires the real seams -- one pooled session runner whose
per-stage invokes (:func:`census_stage_specs`) back mining, the headroom
probe, ``census_verify.build_verify_fn`` and :func:`build_synthesize_fn`;
the fused-memory ``submit_task`` poster and task store
(``alarm.post_tool_call``); ``alarm.raise_alarm`` for escalations; the
landing gate, the scoped commit and its rollback -- and runs
``census_run.run_census``. A deferral, a NO-FIRE or a landed run exits 0; a
run that did not land, or any fail-loud error, exits 1.

Layers (measured in task 6753's plan): census_headroom, census_mining < census_report, census_verify < census_filing < census_run < census.
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
import os
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import assert_never

# Self-bootstrap for standalone `python scripts/legibility/census.py` runs
# -- must run BEFORE the `legibility.*` imports below, since a direct
# script invocation puts only scripts/legibility/ (not scripts/) on
# sys.path. Skipped under pytest/normal package import: __name__ is
# 'legibility.census' or 'census', never '__main__' (mirrors
# sampling.py:37-38).
if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Bind `shared` and `orchestrator` to the SAME checkout as this script via a
# __file__-relative path, never the MAIN checkout an editable install puts on
# sys.path (tasks 2881/2882/3329). Unconditional, not inside the `__main__`
# guard above, so it holds under pytest and package import too;
# `orchestrator.agents.code_quality` renders the synthesis prompt's quality
# definition (docs/quality-findings-contract.md §10).
_SHARED_SRC = Path(__file__).resolve().parents[2] / "shared" / "src"
if str(_SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(_SHARED_SRC))
_ORCH_SRC = Path(__file__).resolve().parents[2] / "orchestrator" / "src"
if str(_ORCH_SRC) not in sys.path:
    sys.path.insert(0, str(_ORCH_SRC))

from legibility import (  # noqa: E402
    alarm,
    census_headroom,
    census_identity,
    census_run,
    census_state,
    census_trigger,
    census_verify,
    census_window,
    codebook,
    config,
    embeddings,
    filing_policy,
    landing_gate,
    layout,
    session_ledger,
    session_runner,
    synthesis,
    task_store,
    unlanded,
)
from orchestrator.agents import code_quality  # noqa: E402

logger = logging.getLogger("legibility.census")


class CensusPreconditionFailed(RuntimeError):
    """A check before any spend refused this run."""


DEFAULT_PROJECTS_ROOT = Path.home() / ".claude" / "projects"
"""Root of the encoded ~/.claude/projects session-transcript tree that
inventory.enumerate_sessions scans -- distinct from a censused project's
OWN project_root (which only ever holds that project's docs/legibility/
state, not any transcripts)."""

DEFAULT_HARNESS_CONFIG_PATH = layout.config_path(Path(__file__).resolve().parents[2])
"""legibility.yaml of the checkout that ships this census: the harness whose
own components can be a confusion's fix surface. Its project_root names the
MAIN checkout even when the census runs from a worktree."""


def _synthesis_prompt(brief: synthesis.Brief, *, quality_block: str) -> str:
    """Prompt for the real Fable synthesize_fn: observations on this run's new
    and re-observed findings (*brief*), as the C6 document ``synthesis.parse``
    validates, judged against *quality_block* (``code_quality.guidance``)."""
    return (
        "You are the periodic-census synthesis writer for the dark-factory "
        "agent-confusion codebook (plans/census-incremental-prd.md C6). You have "
        "no tools: everything you need is in this message.\n\n"
        "The brief below holds only this run's findings: the new ones this run "
        "promoted, and the re_observed ones that gained a session this run. Write "
        "observations only, never a diagnosis that was not itself verified, and "
        "never narrate history or earlier reports: the runner renders the standing "
        "findings and every count itself.\n\n"
        "new: an observation on a new finding. re_observed: what this run's "
        "evidence changes about a finding, if anything. phase_refinements: a "
        "judged finding's origin and manifested phase, where the brief's evidence "
        "places them better than its stamp; they are applied to the matrix and to "
        "the entries this run created. corrections: a standing re-observed entry "
        "whose title or cause this run's evidence refutes; they are applied to the "
        "codebook. notes: anything else a reader should weigh, never verified.\n\n"
        "A new finding's tags name the heuristics of the quality definition below "
        "that it rests on; cite those heuristics by name.\n\n"
        + census_verify.QUALITY_DEFINITION_MARKER + quality_block
        + "\n=== RESPONSE ===\n" + synthesis.reply_instructions()
        + "\n=== BRIEF ===\n" + json.dumps(brief.to_record())
    )


def build_synthesize_fn(invoke):
    """Build the real ``synthesize_fn(brief, *, model) -> str`` seam: one
    Fable call via *invoke*, its raw reply left for ``synthesis.parse``. It
    always calls: census_run.run_census makes no call for an empty brief."""
    quality_block = code_quality.guidance()

    def _synthesize_fn(brief: synthesis.Brief, *, model: str) -> str:
        return invoke(_synthesis_prompt(brief, quality_block=quality_block), model)

    return _synthesize_fn


_FUSED_MEMORY_URL_ENV_VAR = "FUSED_MEMORY_MCP_URL"
_DEFAULT_FUSED_MEMORY_URL = "http://localhost:8002"
"""Mirrors census_trigger.default_status_fetcher's identical env
var/default -- submit_task lives on the same fused-memory MCP server this
project's get_statuses call already targets."""


def _fused_memory_base_url() -> str:
    return os.environ.get(_FUSED_MEMORY_URL_ENV_VAR, _DEFAULT_FUSED_MEMORY_URL)


def default_submit_fn(**kwargs) -> dict:
    """Real curator-path ``submit_task`` poster (fused-memory MCP server) --
    build_task_payloads' payloads are forwarded verbatim as this call's
    arguments."""
    return alarm.post_tool_call(
        _fused_memory_base_url(), "submit_task", kwargs, timeout=30.0, client_name="legibility-census",
    )


def _parse_cli_date(value: str) -> date:
    return date.fromisoformat(value)


def _positive_int(value: str) -> int:
    """``argparse`` ``type=`` for the operator cost caps: an int >= 1.

    A cap of 0 or a negative value is rejected at the CLI boundary (exit
    2, with a message) rather than half-applied downstream: ``--max-batches
    0`` would still code one full batch (the cap is checked after a batch
    is coded) and then render a self-contradictory coverage line, and
    ``--max-verify-clusters -1`` would slice ``novel_clusters[:-1]``,
    verifying all but the LAST cluster while reporting ``cap=-1`` as if
    honored. Both are exactly the silent, mis-honored cap these flags exist
    to make impossible -- so a nonsense value fails loud. Omitting the flag,
    not passing 0, is how you ask for no cap."""
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected an integer, got {value!r}"
        ) from None
    if parsed < 1:
        raise argparse.ArgumentTypeError(
            f"must be 1 or greater, got {parsed} -- omit the flag entirely for no cap"
        )
    return parsed


_MINING_MAX_TURNS, _MINING_MAX_BUDGET_USD = 2, 2.0
_VERIFY_MAX_TURNS, _VERIFY_MAX_BUDGET_USD = 100, 10.0
_SYNTHESIS_MAX_TURNS, _SYNTHESIS_MAX_BUDGET_USD = 2, 50.0
"""Per-call ceilings the CLI requires, set well above any measured call so
``cfg.timeouts`` stays the binding bound. Verify alone takes turns: it
explores the tree with Read/Grep/Glob."""


def census_stage_specs(cfg, *, project_root):
    """How each census stage calls the model:
    ``(mining, verify, synthesis)``.

    Each carries its OWN timeout from ``cfg.timeouts`` (see
    ``config.Timeouts`` for why each stage needs its own budget). Mining —
    which also backs the headroom probe — and synthesis are pure classifiers
    with no tools; verify may Read/Grep/Glob the censused tree and nothing
    else, so it can never write into what it is censusing.

    *project_root* is the cwd of ALL THREE, not of verify alone. VERIFY is
    where its absence was proven fatal (fleet session census-reify-3386101,
    2026-08-03): it is the only stage whose prompt directs the model to read
    the target tree, and the CLI scopes tool access to its cwd tree, so
    censusing a project from some other directory permission-denied every
    verifier read. Scoping only verify would leave mining and synthesis
    rooted in whatever directory the operator launched from — an asymmetry a
    later reader would file as a bug. The cost: the CLI loads the censused
    project's CLAUDE.md into every mining call, the highest-volume stage. It
    is the right trade, and if that context ever measures material the lever
    is a settings flag on those two stages, not un-scoping their cwd.
    """
    timeouts = cfg.timeouts
    return (
        session_runner.StageSpec(
            name="census-mining", cwd=project_root,
            timeout_secs=timeouts.census_mining_secs,
            max_turns=_MINING_MAX_TURNS, max_budget_usd=_MINING_MAX_BUDGET_USD,
            tools=session_runner.CLASSIFIER,
        ),
        session_runner.StageSpec(
            name="census-verify", cwd=project_root,
            timeout_secs=timeouts.census_verify_secs,
            max_turns=_VERIFY_MAX_TURNS, max_budget_usd=_VERIFY_MAX_BUDGET_USD,
            tools=session_runner.READ_ONLY_EXPLORER,
        ),
        session_runner.StageSpec(
            name="census-synthesis", cwd=project_root,
            timeout_secs=timeouts.census_synthesis_secs,
            max_turns=_SYNTHESIS_MAX_TURNS, max_budget_usd=_SYNTHESIS_MAX_BUDGET_USD,
            tools=session_runner.CLASSIFIER,
        ),
    )


def build_stage_invokes(cfg, *, project_root, runner):
    """The three per-stage ``invoke(prompt, model)`` seams on *runner*:
    ``(mining_invoke, verify_invoke, synthesis_invoke)``, one per
    :func:`census_stage_specs` entry. ``mining_invoke`` also backs the
    headroom probe (``census_run.run_census`` routes both through its single ``invoke``
    param), so the probe rotates over the pool exactly as mining does."""
    return tuple(
        runner.invoker(spec)
        for spec in census_stage_specs(cfg, project_root=project_root)
    )


def main(argv: list[str] | None = None, *, poster: alarm.Poster | None = None) -> int:
    """CLI entrypoint.

    Unless ``--force``, gates on ``census_trigger.decide_for_project``
    (belt-and-braces for a manual/operator run -- the production caller,
    task epsilon's nightly trickle, only launches this entrypoint when
    zeta already fired) and no-ops -- prints the NO-FIRE reasons, exit 0 --
    when it would not fire. ``--force`` bypasses the gate entirely and is
    recorded in the report header as operator-initiated (``census_run.run_census``'s
    own ``force`` parameter).

    Otherwise builds the real default seams (headless-CLI ``invoke`` for
    mining/verify/synthesis, the fused-memory poster for ``submit_fn``,
    :func:`alarm.raise_alarm` (through *poster*) for ``escalate_fn``,
    ``census_trigger.default_status_fetcher`` for the outcome feed's get_statuses snapshot,
    a :class:`census_window.WindowBatchSource` over the mining window,
    ``unlanded.commit_only`` for the scoped commit, and the ``unlanded.roll_back`` that puts
    back a commit that did not land) and runs the full pipeline via
    ``census_run.run_census``. ``--harness-config`` (default ``DEFAULT_HARNESS_CONFIG_PATH``)
    names the project whose task tree receives harness-owned fix surfaces,
    and is loaded before the trigger gate so an unloadable one fails loud
    rather than quietly filing them into the censused project.

    Three OPERATOR COST-CONTROL flags bound what a single run may spend,
    each defaulting to today's unbounded behavior so a flagless
    invocation is unchanged. The nightly trickle's launch passes
    ``--max-batches`` / ``--max-verify-clusters`` from the project's
    legibility.yaml ``census.trickle_caps`` (``config.TrickleCensusCaps``;
    either one omitted when set to null). census.py itself never reads that
    block, so these flags remain the single cap mechanism.
    ``--max-batches N`` bounds mining (the
    capped-away sessions are not ledgered, so the next census resumes them),
    ``--max-verify-clusters N`` bounds per-cluster verification (a deferred
    cluster is re-adjudicated only if that confusion RECURS), and
    ``--dry-run-filing`` writes every would-be task payload to
    ``plans/<report basename>-payloads.json`` beside the report for human
    review instead of filing it -- those payloads must then be filed by
    hand, since this run still advances the codebook and ledgers its
    sessions, and a later census will not re-file them. They are composable
    and reusable, not first-census-only, though an attended FIRST census
    against an empty codebook is where all three matter most: saturation
    cannot bound that run, since every batch's dup_rate then only
    measures "the miner found nothing to match". Both numeric caps take
    ``type=_positive_int``, so a nonsense value (0 or negative) exits 2 at
    the CLI boundary instead of being half-applied. None of the three lets
    a bounded run masquerade as a complete one -- see
    ``census_report.census_report_sections``.

    Each run gets a fresh ``census_identity.RunIdentity``
    (``census_identity.allocate_run_identity``), so a second census on one
    day writes ``...-2`` files and never overwrites the first, and reads the
    project at ``as_of_sha``, its HEAD (``census_identity.resolve_as_of_sha``),
    resolved after the gate and before any spend.

    Returns non-zero only on a genuine fail-loud error. Every such exit
    after the config has loaded has raised exactly one ERROR alarm:
    :func:`alarm.fail_loud_boundary`'s for any raise, a refused precondition
    included, or ``census_run.run_census``'s own for an ``Unlanded`` census whose
    commit did not land. Before the config loads there is no escalation target, so a bad
    ``--project-root`` or an unloadable config is reported on stderr only.
    A deferred (headroom-preflight) outcome still exits 0, mirroring
    ``census_trigger``'s own CLI contract of reserving a non-zero exit for
    an operator-facing failure, not an expected defer/no-fire outcome.

    Configures logging FIRST, before arg parsing and so before the trigger
    gate, which is what gets this module's INFO lines (the empty-codebook
    line, the per-stage progress lines) into the journal on EVERY
    invocation -- including one that no-ops at the gate. Without it root
    sits at its WARNING default and they are all silently dropped. Shares
    ``config.configure_logging`` with ``nightly.main()`` rather than
    inlining a second ``basicConfig``, so the env var
    (``LEGIBILITY_LOG_LEVEL``), the default level and the format cannot
    drift between the two entrypoints (INV-5 no-lockstep-duplication).
    """
    config.configure_logging()

    parser = argparse.ArgumentParser(
        prog="census",
        description="Legibility periodic-census runner "
        "(plans/confusion-reduction-prd.md section 5.7, task eta).",
    )
    # Required, deliberately no default: an implicit-cwd default on a
    # money-spending entrypoint was the root enabler of task 3269. A
    # resolved-project-id default was rejected as a second guessing mechanism.
    parser.add_argument(
        "--project-root", required=True,
        help="ABSOLUTE root of the project being censused. Required -- there "
        "is deliberately no default, so the census can never silently target "
        "whatever directory it happens to be launched from.",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to the project's legibility.yaml "
        "(default: <project-root>/docs/legibility/legibility.yaml).",
    )
    parser.add_argument(
        "--harness-config", default=None,
        help="Path to the legibility.yaml of the project that owns tooling fix "
        "surfaces (dark-factory): a verified cluster whose fix surface is that "
        "project's files into its task tree (default: this census checkout's "
        "own docs/legibility/legibility.yaml).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Bypass the census_trigger gate (operator-initiated run).",
    )
    parser.add_argument(
        "--date", default=None, type=_parse_cli_date,
        help="Census date YYYY-MM-DD (default: today UTC).",
    )
    parser.add_argument(
        "--max-batches", type=_positive_int, default=None,
        help="Operator cost control: stop mining after N batches (N >= 1; omit "
        "for no cap). Omit for today's behavior -- mine until novelty saturates "
        "or the source runs out. A capped run is deliberately PARTIAL coverage "
        "and says so in the report; its stop_reason is 'capped', not "
        "'exhausted'. The capped-away sessions are not ledgered as coded, so the "
        "next census resumes them.",
    )
    parser.add_argument(
        "--max-verify-clusters", type=_positive_int, default=None,
        help="Operator cost control: verify at most N novel clusters (one "
        "Sonnet call each), taken in mining order (N >= 1; omit for no cap). "
        "Omit to verify every novel cluster. The deferred remainder still "
        "merges into the codebook as pending candidates -- deferred, never "
        "dropped -- but is re-adjudicated only if the same confusion RECURS; "
        "its sessions are ledgered as coded and not re-mined.",
    )
    parser.add_argument(
        "--dry-run-filing", action="store_true",
        help="Operator cost control: write every would-be task payload to "
        "plans/<report basename>-payloads.json for human review and file "
        "NOTHING. Everything else (codebook update, promotions, report, "
        "census-state advance, session ledger) proceeds normally -- so the "
        "payloads must be filed by hand; a later census will not re-file them. "
        "An existing payload file is never overwritten (the payloads go to a "
        "numbered sibling instead).",
    )
    args = parser.parse_args(argv)

    # Resolved, not used raw: --project-root is required but may still be
    # passed relative by an operator. An unresolved relative root would make
    # the stage cwd binding below vacuous (cwd="." IS the launcher's cwd —
    # exactly the bug that binding exists to close) and would silently falsify
    # census_verify.py::_verify_prompt's "ABSOLUTE paths only" contract, since it
    # interpolates str(project_root) straight into the prompt. One resolve()
    # at the CLI boundary makes the prompt text, the subprocess cwd and all
    # four output paths name the same absolute tree.
    project_root = Path(args.project_root).resolve()

    # Rejected LOUDLY here, at the CLI boundary, rather than left to fail
    # somewhere downstream. Now that project_root is also the stage
    # subprocess cwd, a typo'd root surfaces on the FIRST invoke -- the
    # headroom probe -- as a session_runner.InvocationFailed, and preflight_headroom
    # deliberately folds ANY probe exception into HeadroomResult(ok=False).
    # Without this check `census --project-root /typo --config <real one>`
    # would exit 0 with "census deferred: headroom probe invocation
    # failed: ..." plus an INFO escalation: an operator typo wearing the
    # exact costume of a usage-limit defer, on every subsequent run. A
    # non-existent root is never a deferral -- it is a bad argument.
    if not project_root.is_dir():
        print(
            f"census: --project-root {args.project_root!r} resolves to "
            f"{project_root}, which is not an existing directory",
            file=sys.stderr,
        )
        return 1

    config_path = Path(args.config) if args.config else layout.config_path(project_root)

    try:
        cfg = config.load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - a broken/missing config fails loud at CLI startup
        print(f"census: failed to load config at {config_path}: {exc}", file=sys.stderr)
        return 1

    project = alarm.AlarmProject.of(cfg)
    escalate_fn = functools.partial(
        alarm.raise_alarm, project, alarm.AlarmSource.CENSUS, poster=poster,
    )
    try:
        with alarm.fail_loud_boundary(alarm.AlarmSource.CENSUS, project, poster=poster):
            return _run_configured_census(
                args, project_root=project_root, cfg=cfg, config_path=config_path,
                escalate_fn=escalate_fn,
            )
    except Exception as exc:
        print(f"census: FAILED -- {exc}", file=sys.stderr)
        return 1


def _run_configured_census(
    args: argparse.Namespace, *, project_root: Path, cfg: config.LegibilityConfig,
    config_path: Path, escalate_fn,
) -> int:
    """:func:`main`'s run once *cfg* has loaded. Raises on every failure but an
    ``Unlanded`` outcome, which ``census_run.run_census`` has already escalated."""
    mismatch = config.project_root_mismatch(project_root, cfg, config_path)
    if mismatch is not None:
        raise CensusPreconditionFailed(mismatch)

    harness_config_path = (
        Path(args.harness_config) if args.harness_config else DEFAULT_HARNESS_CONFIG_PATH
    )
    try:
        harness_cfg = config.load_config(harness_config_path)
    except Exception as exc:
        raise CensusPreconditionFailed(
            f"failed to load harness config at {harness_config_path}: {exc}"
        ) from exc
    harness_project = filing_policy.ProjectRef(
        project_root=harness_cfg.project_root, project_id=harness_cfg.project_id,
    )

    now = datetime.now(UTC)
    census_day = args.date if args.date is not None else now.date()
    date_str = census_day.isoformat()
    status_fetcher = census_trigger.default_status_fetcher(project_root)

    if not args.force:
        decision = census_trigger.decide_for_project(cfg, now=now)
        if not decision.fire:
            print(f"census: NO-FIRE for {cfg.project_id} -- pass --force to run anyway")
            for reason in decision.reasons:
                print(f"  {reason}")
            return 0

    try:
        as_of_sha = census_identity.resolve_as_of_sha(project_root)
    except RuntimeError as exc:
        raise CensusPreconditionFailed(f"cannot resolve as_of_sha -- {exc}") from exc

    codebook_path = layout.codebook_path(project_root)
    census_state_path = layout.census_state_path(project_root)
    prior = census_identity.prior_census(census_state.load(census_state_path))
    plans_dir = project_root / "plans"
    identity = census_identity.allocate_run_identity(
        project_id=cfg.project_id,
        day=census_day,
        plans_dir=plans_dir,
        taken_run_ids=frozenset(run_id for run_id in (prior.run_id,) if run_id is not None),
    )
    report_path = plans_dir / f"{identity.basename}.md"
    dry_run_payloads_path = (
        plans_dir / f"{identity.basename}-payloads.json" if args.dry_run_filing else None
    )

    try:
        codebook_dict = codebook.load(codebook_path)
    except FileNotFoundError:
        logger.info(
            "census: no codebook at %s yet, starting from an empty v2 document", codebook_path,
        )
        codebook_dict = {"version": 2, "entries": [], "candidates": []}

    batch_source = census_window.WindowBatchSource(
        cfg,
        projects_root=DEFAULT_PROJECTS_ROOT,
        now=now,
        ledger_path=session_ledger.ledger_path(cfg.project_id),
        last_census_at=prior.last_census_at,
    )

    # ONE pooled session runner for the whole census: every stage and every
    # headroom probe rotates over the shared account pool per invocation
    # (task 6042), and cap state learned by one stage carries to the next.
    with session_runner.open_pooled_runner(
        label=f"legibility-census[{cfg.project_id}]",
    ) as runner:
        mining_invoke, verify_invoke, synthesis_invoke = build_stage_invokes(
            cfg, project_root=project_root, runner=runner,
        )
        effects = census_run.CensusEffects(
            batch_source=batch_source,
            selection_of=lambda: batch_source.selection,
            invoke=mining_invoke,
            # The in-verify re-probe rides the SAME cwd-scoped invoke as the
            # verifier it guards (so it fails exactly when the verifier would)
            # but on the cheap trickle tier, matching every other headroom
            # gate -- one probe implementation, one model tier, four sites.
            verify_fn=census_verify.build_verify_fn(
                str(project_root),
                verify_invoke,
                headroom_probe=functools.partial(
                    census_headroom.preflight_headroom, verify_invoke, model=cfg.models.trickle,
                ),
            ),
            synthesize_fn=build_synthesize_fn(synthesis_invoke),
            embedder=embeddings.Embedder(
                embed_fn=embeddings.openai_embed, cache_path=embeddings.cache_path(cfg.project_id),
            ),
            submit_fn=default_submit_fn,
            task_store=task_store.McpTaskStore(
                url=_fused_memory_base_url(),
                post=functools.partial(alarm.post_tool_call, client_name="legibility-census"),
            ),
            status_fetcher=status_fetcher,
            escalate_fn=escalate_fn,
            gate=functools.partial(landing_gate.run_gate, project_root, cfg.census.landing_gate),
            commit=functools.partial(unlanded.commit_only, project_root),
            roll_back=functools.partial(
                unlanded.roll_back, project_root,
                project_id=cfg.project_id, label=f"census-{date_str}",
            ),
        )
        outcome = census_run.run_census(
            effects=effects,
            codebook_dict=codebook_dict,
            config=cfg,
            project_root=str(project_root),
            project_id=cfg.project_id,
            codebook_path=codebook_path,
            census_state_path=census_state_path,
            report_path=report_path,
            date=date_str,
            run_id=identity.run_id,
            as_of_sha=as_of_sha,
            since=prior.as_of_sha,
            force=args.force,
            max_batches=args.max_batches,
            max_verify_clusters=args.max_verify_clusters,
            dry_run_payloads_path=dry_run_payloads_path,
            harness_project=harness_project,
        )

    return _exit_code(outcome)


def _exit_code(outcome: census_run.CensusOutcome) -> int:
    """Print *outcome*'s one summary line and return main's exit code: 0 for a
    deferral or a landed run, 1 for a run that did not land, which census_run.run_census
    has already escalated."""
    match outcome:
        case census_run.Deferred(stage=stage, reason=reason, counts=counts):
            print(f"census: deferred -- stage={stage} unverified_clusters={counts.unverified} -- {reason}")
            return 0
        case census_run.Unlanded(rollback=rollback, gate=gate) if not gate.passed:
            print(
                f"census: landing gate refused the census -- {gate.describe()} -- {rollback.describe()}",
                file=sys.stderr,
            )
            return 1
        case census_run.Unlanded(rollback=rollback, filed_ticket_ids=filed_ticket_ids):
            print(
                f"census: commit did not land -- {rollback.describe()} filed_tickets={len(filed_ticket_ids)}",
                file=sys.stderr,
            )
            return 1
        case census_run.Done():
            print(_done_line(outcome))
            return 0
        case _:
            assert_never(outcome)


def _done_line(done: census_run.Done) -> str:
    """The landed run's summary line; each optional clause appears only when
    it carries information, so a normal run's line stays byte-identical."""
    ledger_failed = f" ledger_write_failed: {done.ledger.error}" if isinstance(done.ledger, census_run.LedgerFailed) else ""
    unreachable = done.filing.disposal.unreachable
    dedup_unreachable = f" dedup_unreachable: {unreachable}" if unreachable else ""
    if done.dry_run is not None:
        return (
            f"census: done -- report={done.report_path} "
            f"dry-run-filing: {done.dry_run.payload_count} payload(s) -> "
            f"{done.dry_run.path} (nothing filed) "
            f"stop_reason={done.stop_reason}{ledger_failed}{dedup_unreachable}"
        )
    unresolved = f" unresolved_verdicts={done.unresolved_verdicts}" if done.unresolved_verdicts else ""
    skipped = f" skipped_deletions={len(done.skipped_deletions)}" if done.skipped_deletions else ""
    return (
        f"census: done -- report={done.report_path} "
        f"filed_tickets={len(done.filed_ticket_ids)} "
        f"stop_reason={done.stop_reason}{unresolved}{skipped}{ledger_failed}{dedup_unreachable}"
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
