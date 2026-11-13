"""scripts/legibility/census_trigger.py — periodic legibility census trigger
evaluator.

The trigger contract is plans/census-incremental-prd.md C7. The census fires
on (a) the previous run's weighted completion -- scripts/check_run_completion.py
over the tasks that run filed under ``x_finding_run``, against
``census.completion_threshold_pct`` -- or (b) a novelty spike relative to a
trailing baseline (§4.7 (b)). Neither fires inside the floor, measured from
the session watermark (§4.7 Floor). Elapsed time alone fires nothing.
plans/confusion-reduction-prd.md §6, §7.4 and §7.5 are the original trigger,
config-block and census-state contracts this refines.

Evaluated at the end of each nightly trickle run, by census.py's own gate,
and via the standalone ``evaluate`` CLI subcommand below.

The completion check runs as a subprocess through an injected
``CompletionRunner`` (default ``run_completion_check``), so the decision core
stays pure and every failure of the check is testable with a fake. Its
verdict is parsed once, at that boundary, into a ``Completion``.

The trigger reads the census block of the ``LegibilityConfig`` its caller
loaded (``legibility.config.load_config`` validates it strictly), and the
census state through ``census_state.load``.

The get_statuses helpers (``extract_statuses``, ``default_status_fetcher``)
serve census.py's outcome feed, not the trigger.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, assert_never, overload

from legibility import census_identity, census_state, codebook, layout, state_docs
from legibility.config import Census, LegibilityConfig, load_config, project_root_mismatch

if (_SHARED_SRC := str(Path(__file__).resolve().parents[2] / "shared" / "src")) not in sys.path:
    sys.path.insert(0, _SHARED_SRC)

from shared.mcp_session import McpCallError, SyncMcpSession

logger = logging.getLogger("legibility.census_trigger")


@overload
def _as_utc(value: datetime) -> datetime: ...


@overload
def _as_utc(value: None) -> None: ...


def _as_utc(value: datetime | None) -> datetime | None:
    """Normalize a datetime to timezone-aware UTC. A naive datetime (e.g.
    parsed from a bare `YYYY-MM-DD` codebook date) is assumed to already be
    UTC. `None` passes through unchanged.

    Overloaded because the pass-through is exactly correlated with the input:
    the plain `datetime | None -> datetime | None` signature discarded that,
    so `evaluate`'s `now` (declared non-Optional) came back Optional and made
    every subsequent arithmetic on it a type error. The overloads state what
    the body already does; the runtime implementation is unchanged."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


# ---------------------------------------------------------------------------
# completion — condition (a): the previous run's weighted completion
# (plans/census-incremental-prd.md C7 (a))
# ---------------------------------------------------------------------------

class CompletionStatus(StrEnum):
    FIRE = "fire"
    NOT_YET = "not_yet"
    NO_PREVIOUS_RUN = "no_previous_run"
    FILED_NOTHING = "filed_nothing"
    ERROR = "error"


_MEASURED = frozenset({CompletionStatus.FIRE, CompletionStatus.NOT_YET})


@dataclass(frozen=True)
class Completion:
    """What scripts/check_run_completion.py said about the previous census
    run's filed tasks, parsed once at the subprocess boundary.

    Each field belongs to the statuses that use it: ``run_id`` is None
    exactly for NO_PREVIOUS_RUN, ``share`` and ``vacuous`` are set only on a
    measured FIRE or NOT_YET, and ``detail`` is set exactly for ERROR."""

    status: CompletionStatus
    run_id: str | None
    threshold_pct: int
    share: float | None = None
    vacuous: bool = False
    detail: str = ""

    def __post_init__(self) -> None:
        if (self.run_id is None) != (self.status is CompletionStatus.NO_PREVIOUS_RUN):
            raise ValueError(f"Completion {self.status}: run_id {self.run_id!r} does not fit")
        if self.status not in _MEASURED and (self.share is not None or self.vacuous):
            raise ValueError(f"Completion {self.status}: carries a share, but measured nothing")
        if bool(self.detail) != (self.status is CompletionStatus.ERROR):
            raise ValueError(f"Completion {self.status}: detail {self.detail!r} does not fit")

    @property
    def fires(self) -> bool:
        return self.status is CompletionStatus.FIRE

    @property
    def reason(self) -> str:
        threshold = f"(threshold {self.threshold_pct / 100:.2f})"
        match self.status:
            case CompletionStatus.FIRE | CompletionStatus.NOT_YET:
                line = f"completion: {self._shown_share()} of {self.run_id} {threshold}"
                return line + (" -> FIRE" if self.fires else "")
            case CompletionStatus.NO_PREVIOUS_RUN:
                return "completion: no previous run id (census-state has no last_census_run_id) -> N/A"
            case CompletionStatus.FILED_NOTHING:
                return f"completion: run {self.run_id} filed no tasks -> N/A"
            case CompletionStatus.ERROR:
                return f"completion: check of {self.run_id} failed ({self.detail}) -> N/A"

    def _shown_share(self) -> str:
        if self.share is not None:
            return f"{self.share:g}"
        return "vacuous" if self.vacuous else "unreadable"


COMPLETION_SCRIPT = Path(__file__).resolve().parents[1] / "check_run_completion.py"
COMPLETION_CHECK_TIMEOUT_SECS = 120
CompletionRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

_EMPTY_RUN_ERROR = "empty_run"


def run_completion_check(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run check_run_completion.py under this interpreter, capturing stdout;
    its stderr diagnostics go straight to the caller's journal."""
    return subprocess.run(
        [sys.executable, str(COMPLETION_SCRIPT), *args],
        stdout=subprocess.PIPE,
        text=True,
        timeout=COMPLETION_CHECK_TIMEOUT_SECS,
        check=False,
    )


def _verdict_line(stdout: str | None) -> Mapping[str, Any] | None:
    lines = [line for line in (stdout or "").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        verdict = json.loads(lines[-1])
    except json.JSONDecodeError:
        return None
    return verdict if isinstance(verdict, dict) else None


def _completion_from_exit(
    run_id: str, threshold_pct: int, returncode: int, verdict: Mapping[str, Any] | None
) -> Completion:
    verdict = verdict or {}
    share = verdict.get("share")
    concluded = {os.EX_OK: CompletionStatus.FIRE, os.EX_TEMPFAIL: CompletionStatus.NOT_YET}
    if returncode in concluded:
        return Completion(
            concluded[returncode],
            run_id,
            threshold_pct,
            share=share if isinstance(share, float) else None,
            vacuous=verdict.get("vacuous") is True,
        )
    error_code = verdict.get("error")
    if returncode == 2 and error_code == _EMPTY_RUN_ERROR:
        return Completion(CompletionStatus.FILED_NOTHING, run_id, threshold_pct)
    named = (
        "no verdict line" if not verdict
        else error_code if isinstance(error_code, str)
        else "no error code"
    )
    return Completion(
        CompletionStatus.ERROR, run_id, threshold_pct, detail=f"exit {returncode}, {named}"
    )


def _warn_not_applicable(completion: Completion, check_output: str = "") -> None:
    logger.warning(
        "%s -- condition (a) fails safe (no fire on completion); the novelty spike "
        "still evaluates (plans/census-incremental-prd.md C7 (a))%s",
        completion.reason,
        f"; check output: {bounded_repr(check_output)}" if check_output else "",
    )


def check_completion(
    run_id: str | None,
    *,
    project_root: str | Path,
    threshold_pct: int,
    runner: CompletionRunner,
) -> Completion:
    """Condition (a): whether the previous census run *run_id* has landed
    *threshold_pct* percent of its filed tasks' weight in *project_root*,
    and only there (D9). Never raises; every outcome that is not FIRE or
    NOT_YET logs exactly one WARNING."""
    if run_id is None:
        completion = Completion(CompletionStatus.NO_PREVIOUS_RUN, None, threshold_pct)
        _warn_not_applicable(completion)
        return completion

    args = [
        "--run", run_id,
        "--threshold", f"{threshold_pct / 100:g}",
        "--project-root", str(Path(project_root).resolve()),
    ]
    try:
        done = runner(args)
    except Exception as exc:
        completion = Completion(
            CompletionStatus.ERROR,
            run_id,
            threshold_pct,
            detail=f"{type(exc).__name__}: {bounded_repr(str(exc))}",
        )
        _warn_not_applicable(completion)
        return completion

    completion = _completion_from_exit(
        run_id, threshold_pct, done.returncode, _verdict_line(done.stdout)
    )
    if completion.status is CompletionStatus.ERROR:
        _warn_not_applicable(completion, (done.stdout or "").strip())
    elif completion.status is CompletionStatus.FILED_NOTHING:
        _warn_not_applicable(completion)
    return completion


# ---------------------------------------------------------------------------
# Decision + evaluate() — pure §8.5 decision core (no I/O)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """The census-trigger verdict: whether to fire, one human-readable reason
    line per condition and the floor, and the completion arm's verdict (None
    when the evaluation never reached it)."""

    fire: bool
    reasons: list[str]
    completion: Completion | None = None


class FloorAnchorKind(Enum):
    SESSION_WATERMARK = "session watermark"
    LAST_CENSUS_AT = "last_census_at (no watermark yet)"
    EARLIEST_CODEBOOK_DATE = "earliest codebook date (never censused)"


@dataclass(frozen=True)
class FloorAnchor:
    """The instant the census floor is measured from, and where it came from."""

    kind: FloorAnchorKind
    at: datetime


def _floor_condition(
    now_utc: datetime, anchor: FloorAnchor | None, config: Census
) -> tuple[bool, str]:
    if anchor is None:
        return True, (
            "floor: no anchor (no session watermark, last_census_at or codebook date)"
            " -> BLOCKS all conditions"
        )
    days = (now_utc - _as_utc(anchor.at)).total_seconds() / 86400.0
    blocks = days < config.floor_days
    line = f"floor: {days:.1f}d since {anchor.kind.value} (floor {config.floor_days}d)"
    return blocks, line + (" -> BLOCKS all conditions" if blocks else "")


def _count_within(first_seens_utc: list[datetime], *, end: datetime, hours: int) -> int:
    start = end - timedelta(hours=hours)
    return sum(1 for fs in first_seens_utc if start <= fs <= end)


def _novelty_condition(
    now_utc: datetime, candidate_first_seens: list[datetime], config: Census
) -> tuple[bool, str]:
    seen = [_as_utc(fs) for fs in candidate_first_seens]
    novelty = config.novelty_spike
    hours = novelty.window_hours
    baseline_days = novelty.baseline_days
    count = _count_within(seen, end=now_utc, hours=hours)
    head = f"novelty-spike: {count} within {hours}h"

    history_days = (now_utc - min(seen)).total_seconds() / 86400.0 if seen else 0.0
    if history_days < baseline_days:
        return False, (
            f"{head} (baseline needs {baseline_days}d of candidate history, "
            f"have {history_days:.1f}d) -> N/A"
        )

    baseline = (
        statistics.median(
            _count_within(seen, end=now_utc - timedelta(days=k), hours=hours)
            for k in range(1, baseline_days + 1)
        )
        if baseline_days
        else 0.0
    )
    spike = count >= novelty.count and count >= novelty.multiple * baseline
    line = f"{head} (baseline median {baseline:g}, x{novelty.multiple})"
    return spike, line + (" -> FIRE" if spike else "")


def evaluate(
    *,
    now: datetime,
    floor_anchor: FloorAnchor | None,
    completion: Completion,
    candidate_first_seens: list[datetime],
    config: Census,
) -> Decision:
    """Pure decision core (plans/census-incremental-prd.md C7): fire when
    (a) *completion* fires or (b) the novelty spike does, unless the floor
    measured from *floor_anchor* blocks. The floor never fires by itself, and
    its reason line always names its anchor. The novelty condition is
    relative to the median of the daily window counts over the trailing
    baseline_days, and N/A until that much candidate history exists. The
    reasons are the completion, novelty and floor lines, in that order.
    """
    now_utc = _as_utc(now)
    spike, novelty_line = _novelty_condition(now_utc, candidate_first_seens, config)
    floor_blocks, floor_line = _floor_condition(now_utc, floor_anchor, config)
    return Decision(
        fire=(completion.fires or spike) and not floor_blocks,
        reasons=[completion.reason, novelty_line, floor_line],
        completion=completion,
    )


# ---------------------------------------------------------------------------
# codebook_signal — extract the never-censused anchor + novelty-spike dates
# from a task γ / codebook.load() dict (§7.3 candidates schema)
# ---------------------------------------------------------------------------

def _parse_date(value: object) -> datetime | None:
    """Parse an ISO-ish date/datetime string; return None (never raise) for
    anything unparseable or non-string, so one bad codebook record can't
    take down the whole evaluation."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def codebook_signal(codebook: dict) -> tuple[datetime | None, list[datetime]]:
    """Extract the census-trigger signal from a `codebook.load()`-shaped
    dict (task γ / 2575's schema): the earliest structured sighting or
    candidate `first_seen`/sighting date across the WHOLE codebook (used to
    anchor the floor when never censused), and every candidate's
    `first_seen` date on its own (used for the novelty-spike window count). Unparseable dates are skipped rather than raising.

    Returns `(earliest_or_None, sorted_candidate_first_seens)`. An empty
    codebook (`entries: []`, `candidates: []`) returns `(None, [])`.
    """
    all_dates: list[datetime] = []
    candidate_first_seens: list[datetime] = []

    for entry in codebook.get("entries") or []:
        for sighting in entry.get("sightings") or []:
            parsed = _parse_date(sighting.get("date"))
            if parsed is not None:
                all_dates.append(parsed)

    for candidate in codebook.get("candidates") or []:
        first_seen = _parse_date(candidate.get("first_seen"))
        if first_seen is not None:
            candidate_first_seens.append(first_seen)
            all_dates.append(first_seen)
        for sighting in candidate.get("sightings") or []:
            parsed = _parse_date(sighting.get("date"))
            if parsed is not None:
                all_dates.append(parsed)

    earliest = min(all_dates) if all_dates else None
    return earliest, sorted(candidate_first_seens)


# ---------------------------------------------------------------------------
# extract_statuses + default_status_fetcher — the census run's get_statuses
# outcome feed (census_run.py::_read_statuses)
# ---------------------------------------------------------------------------

class StatusFetchUnavailable(Exception):
    """Raised by a status_fetcher (in particular `default_status_fetcher`'s
    returned callable) or by `extract_statuses` when a usable get_statuses
    snapshot cannot be obtained -- no network, a non-2xx response, a
    malformed payload. A dedicated exception lets the census run's outcome feed
    (`census_run.py::_read_statuses`) catch fetch failures deterministically."""


# A `StatusFetchUnavailable` message is re-emitted verbatim into the census
# report and the nightly journal, and `bounded_repr` also bounds this
# module's own WARNINGs. An unbounded `{payload!r}` would dump a whole
# malformed-but-large body (get_statuses over a big project is thousands of
# entries) into a single line. Diagnosis needs only the shape, so reprs are
# truncated and the type name / key list, which identifies the fault, is
# always reported outside the truncated part.
MAX_REPR_CHARS = 200
MAX_REPORTED_KEYS = 20


def bounded_repr(value, limit: int = MAX_REPR_CHARS) -> str:
    """`repr(value)` truncated to *limit* chars with an explicit elision
    marker, so a log line stays one readable line regardless of payload size."""
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (repr truncated, {len(text)} chars total)"


def bounded_keys(mapping) -> str:
    """A bounded, order-stable summary of a mapping's keys. Sorts by `str` so
    a payload with mixed-type keys cannot raise `TypeError` from inside the
    error path itself -- a failure while reporting a failure is the worst
    possible place to be clever."""
    keys = sorted(mapping, key=str)
    shown = keys[:MAX_REPORTED_KEYS]
    suffix = f", ... ({len(keys)} total)" if len(keys) > MAX_REPORTED_KEYS else ""
    return f"{shown}{suffix}"


def extract_statuses(payload) -> Mapping[str, Any]:
    """The `{task id string: status}` mapping of a get_statuses payload, or
    raise `StatusFetchUnavailable` if *payload* is not a usable get_statuses
    envelope. Its one consumer is the census run's outcome feed
    (`census_run.py::_read_statuses`).

    `{"statuses": {}}` is a valid, empty answer; only a non-dict payload, a
    missing or non-mapping `statuses`, or the tool-error envelope is
    unusable. fused-memory's `@mcp_tool_errors` returns its
    `{"error", "error_type"}` dict on an `isError: false` response, so it
    reaches the caller looking like a genuine reply, and reading it as an
    empty snapshot once fabricated a census baseline (task 3291).
    """
    if not isinstance(payload, dict):
        raise StatusFetchUnavailable(
            f"get_statuses payload is not a dict (got {type(payload).__name__}): "
            f"{bounded_repr(payload)}"
        )

    # The fused-memory tool-error envelope, checked before the shape
    # complaint so the operator sees the actual server-side reason (e.g.
    # "project_root must be a non-empty absolute path") instead of merely
    # being told a `statuses` key was missing.
    if "statuses" not in payload and "error" in payload:
        raise StatusFetchUnavailable(
            f"get_statuses returned a tool-error envelope "
            f"({payload.get('error_type') or 'error'}): "
            f"{bounded_repr(payload.get('error'))}"
        )

    if "statuses" not in payload:
        raise StatusFetchUnavailable(
            f"get_statuses payload has no 'statuses' key (keys: {bounded_keys(payload)})"
        )

    statuses = payload["statuses"]
    if not isinstance(statuses, Mapping):
        sized = f", len={len(statuses)}" if hasattr(statuses, "__len__") else ""
        raise StatusFetchUnavailable(
            f"get_statuses 'statuses' is not a mapping "
            f"(got {type(statuses).__name__}{sized}): {bounded_repr(statuses)}"
        )

    return statuses


_FUSED_MEMORY_URL_ENV_VAR = "FUSED_MEMORY_MCP_URL"
_DEFAULT_FUSED_MEMORY_URL = "http://localhost:8002"  # dashboard.config.DEFAULT_FUSED_MEMORY_URLS[0]


def default_status_fetcher(project_root: str | Path):
    """Return a zero-arg get_statuses caller for *project_root*.

    The production fetcher for census.py's outcome feed (`census.main`). It
    calls fused-memory at `$FUSED_MEMORY_MCP_URL` (default
    `http://localhost:8002`) and returns the tool's reply, the
    `{"statuses": {...}}` mapping `extract_statuses` reads. Any failure
    raises `StatusFetchUnavailable` naming the URL.

    *project_root* is resolved to an ABSOLUTE path here, at the wire, for
    every consumer at once: the server resolves the argument against its own
    cwd, and fused-memory's `_normalize_project_root` rejects a relative
    path. An absolute root is sent unchanged.
    """
    project_root_str = str(Path(project_root).resolve())
    base_url = os.environ.get(_FUSED_MEMORY_URL_ENV_VAR, _DEFAULT_FUSED_MEMORY_URL)

    def _fetch() -> dict:
        try:
            with SyncMcpSession(
                base_url, client_name="legibility-census-outcome-feed", timeout=10.0,
            ) as session:
                return session.call_tool("get_statuses", {"project_root": project_root_str})
        except McpCallError as exc:
            raise StatusFetchUnavailable(
                f"get_statuses unreachable at {base_url}: {exc}"
            ) from exc

    return _fetch


# ---------------------------------------------------------------------------
# decide_for_project — high-level assembly (config + state + codebook signal)
# ---------------------------------------------------------------------------

def _floor_anchor(
    read: state_docs.Missing | state_docs.Ok[census_state.CensusState],
    *,
    earliest_codebook_date: datetime | None,
) -> FloorAnchor | None:
    """The instant the floor is measured from: the session watermark, else
    midnight UTC of last_census_at, else, for a never-censused project, the
    earliest codebook date."""
    match read:
        case state_docs.Missing():
            if earliest_codebook_date is None:
                return None
            return FloorAnchor(FloorAnchorKind.EARLIEST_CODEBOOK_DATE, earliest_codebook_date)
        case state_docs.Ok(value=state):
            if state.session_watermark is not None:
                return FloorAnchor(FloorAnchorKind.SESSION_WATERMARK, state.session_watermark)
            if state.last_census_at is None:
                return None
            return FloorAnchor(
                FloorAnchorKind.LAST_CENSUS_AT,
                datetime.combine(state.last_census_at, time.min, tzinfo=UTC),
            )
        case _:
            assert_never(read)


def decide_for_project(
    cfg: LegibilityConfig,
    *,
    now: datetime | None = None,
    completion_runner: CompletionRunner = run_completion_check,
) -> Decision:
    """Assemble the census-trigger Decision for *cfg*'s project: the census
    state, the codebook novelty signal and the previous run's completion
    (its run id read from the census state), fed with *cfg*'s census
    thresholds to the pure `evaluate()` core. The nightly trickle, census.py's
    gate and the `evaluate` CLI all decide through here.

    The thresholds are the validated config's, so there is no config fallback
    left. *now* defaults to the current UTC time. A malformed census state
    short-circuits to a NO-FIRE Decision with no completion, before the
    codebook or the completion check is touched -- `census_state.load` has
    already logged its one WARNING.

    This function NEVER raises. The census state, the codebook and the
    completion check are each an operator-editable file or an external check,
    so each is validated and degrades to a neutral value plus one WARNING,
    taking out only the condition it feeds. `nightly.evaluate_census_step`
    documents this guarantee; it is held here.
    """
    project_root = Path(cfg.project_root)
    now = now if now is not None else datetime.now(UTC)

    read = census_state.load(layout.census_state_path(project_root))
    if isinstance(read, state_docs.Malformed):
        return Decision(
            fire=False,
            reasons=["census state is malformed -- failing safe (no fire)"],
        )

    codebook_path = layout.codebook_path(project_root)
    try:
        codebook_data = codebook.load(codebook_path)
        if not isinstance(codebook_data, dict):
            raise ValueError(f"expected a YAML mapping, got {type(codebook_data).__name__}")
        earliest_sighting, candidate_first_seens = codebook_signal(codebook_data)
    except Exception as exc:  # noqa: BLE001 - a bad codebook must fail safe, not crash
        logger.warning("codebook at %s is unreadable: %s", codebook_path, exc)
        earliest_sighting, candidate_first_seens = None, []

    completion = check_completion(
        census_identity.prior_census(read).run_id,
        project_root=project_root,
        threshold_pct=cfg.census.completion_threshold_pct,
        runner=completion_runner,
    )
    return evaluate(
        now=now,
        floor_anchor=_floor_anchor(read, earliest_codebook_date=earliest_sighting),
        completion=completion,
        candidate_first_seens=candidate_first_seens,
        config=cfg.census,
    )


# ---------------------------------------------------------------------------
# CLI — `evaluate --project-root <path> [--config <path>]`
# ---------------------------------------------------------------------------

def _config_line(config_path: Path, cfg: LegibilityConfig) -> str:
    census = cfg.census
    novelty = census.novelty_spike
    return (
        f"config: {config_path} (project_root {cfg.project_root}) -- "
        f"completion_threshold_pct={census.completion_threshold_pct} "
        f"floor_days={census.floor_days} "
        f"novelty_spike.count={novelty.count} "
        f"novelty_spike.window_hours={novelty.window_hours} "
        f"novelty_spike.multiple={novelty.multiple} "
        f"novelty_spike.baseline_days={novelty.baseline_days}"
    )


def _cmd_evaluate(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config) if args.config else layout.config_path(project_root)
    try:
        cfg = load_config(config_path)
    except Exception as exc:
        print(f"census_trigger: failed to load config at {config_path}: {exc}", file=sys.stderr)
        return 1
    mismatch = project_root_mismatch(project_root, cfg, config_path)
    if mismatch is not None:
        print(f"census_trigger: {mismatch}", file=sys.stderr)
        return 1

    print(_config_line(config_path, cfg))
    decision = decide_for_project(cfg)
    print(f"DECISION: {'FIRE' if decision.fire else 'NO-FIRE'}")
    for reason in decision.reasons:
        print(reason)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="census_trigger",
        description="Legibility periodic-census trigger evaluator "
        "(plans/census-incremental-prd.md C7). Exits 1 when the legibility.yaml "
        "does not load or names another project_root than --project-root, else "
        "0; an evaluation error prints NO-FIRE.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate whether the periodic census should fire"
    )
    evaluate_parser.add_argument("--project-root", default=".")
    evaluate_parser.add_argument(
        "--config", default=None,
        help="Path to the project's legibility.yaml "
        "(default: <project-root>/docs/legibility/legibility.yaml).",
    )
    evaluate_parser.set_defaults(func=_cmd_evaluate)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        logger.warning("census_trigger evaluate failed unexpectedly: %s", exc)
        print(f"DECISION: NO-FIRE (evaluator error: {exc})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
