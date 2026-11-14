#!/usr/bin/env bash
# memory-metadata-coverage-census.sh -- the committed nightly MEASURE-then-
# REHEARSE action for canonical/topic stamping coverage (task 4006). Invoked by
# scripts/memory-metadata-coverage-census.service's ExecStart (systemd
# .timer-driven, nightly 05:00).
#
# ONE CADENCE, TWO HALVES, IN SEQUENCE.
#
#   1. fused-memory/scripts/census_memory_metadata.py -- the GAUGE. A
#      deterministic paginated Qdrant count+scroll (never semantic search) that
#      regenerates plans/memory-metadata-census-report.{json,md} and APPENDS one
#      compact scalar row to plans/memory-metadata-coverage-history.json, which
#      is what turns a snapshot into a trend.
#
#   2. fused-memory/scripts/retro_stamp_topics.py -- the MECHANISM that closes
#      the gap the gauge just measured (PRD leaf θ, task 3201), run as a full
#      rehearsal.
#
# Pairing them on one timer is the point of the job: an operator reads the gap
# and the stamping that would close it in the SAME journal entry, instead of a
# number nobody is accountable for. The census runs FIRST so the rehearsal is
# read against a report generated moments before rather than last night's.
#
# Then a third, non-measuring step LANDS what the census regenerated on main
# through the target's own merge queue, waiting for that landing to settle.
# The census measures a detached generation worktree of main's tip, so the
# machine-operated project_root checkout is never written; why, and how to
# read each outcome, is OPERATIONS.md §12.
#
# THE REHEARSAL IS NEVER `--apply`. 3201 writes `canonical: true` in bulk. Under
# the shipped `memory_metadata.enforce: false` default those writes are not even
# rejected by 3198's write-time per-(project,topic) uniqueness check -- it
# censuses the violation and lets the write through -- so an unattended
# `--apply` could manufacture exactly the violations this census exists to
# count. Committing the stamps stays an operator decision:
#
#   uv run --frozen --project fused-memory python \
#       fused-memory/scripts/retro_stamp_topics.py --apply
#
# NO STEP CAN ABORT ANOTHER, AND THE WRAPPER ALWAYS EXITS 0. A recurring
# `oneshot` that can fail enters systemd `failed` state and STAYS there,
# silently stopping the whole nightly job (the lesson already written into
# scripts/fused-memory-flag-marker-sweep.sh). That matters more here than for
# most jobs: the census exits 1 BY DESIGN whenever `coverage.complete` is false,
# which on a live corpus that orchestrators write to during the scroll is a
# routine outcome, not a fault. Propagating it would wedge the timer on the
# first churny night and quietly end the trend -- while the artifacts, which
# carry the evidence of the shortfall, were written anyway. Both exit codes are
# narrated instead, so exiting 0 never means the failure is invisible. A stop
# signal (a unit stop, TimeoutStartSec) ends the run with 143 or 130 once the
# landing in flight has cancelled its request.
#
# The ONE exception is a refusal to START: when the shared uv resolver cannot
# be loaded or uv cannot be resolved, the wrapper exits 127 (the code a bare
# missing uv gives) before either half runs. That is a configuration fault
# that would fail every night identically with nothing measured, so a failed
# unit is the signal it needs. Once the halves run, the wrapper exits 0.
#
# RUNS UNDER THE SERVICE ENV (sourced .env + CONFIG_PATH / PROJECT_ROOT /
# FALKORDB_URI roots) and under `uv run` so `fused_memory.*` imports resolve --
# mirrors scripts/fused-memory-flag-marker-sweep.sh and the
# fused-memory/scripts/cgl_eta_auto_apply.sh runbook lesson: a fused-memory
# maintenance action must run under the SERVICE env, not a bare shell, or the
# census silently narrows to a different config and censuses a different
# collection than the one the artifacts claim.
#
# COVERAGE_CENSUS_CMD / RETRO_STAMP_CMD override the two interpreter prefixes
# (tests inject fake recorders here to assert argument construction and
# ordering without touching uv, live Qdrant or the live corpus) -- mirrors the
# FLAG_MARKER_SWEEP_CMD / REIFY_CLOSURE_SWEEP_CMD seam convention. REPO is
# similarly overridable so tests can point it at a tmp dir with no `.env` (a
# no-op source). CENSUS_COMMIT=0 measures and rehearses but lands nothing;
# CENSUS_LANDING_TIMEOUT_SECS bounds the wait for the landing.
#
# For a half whose *_CMD seam is unset, uv is resolved to an absolute path by
# scripts/lib/resolve_uv.sh::require_uv_bin (that lib holds the ladder and the
# boot-catch-up cause), after the .env source so a UV_BIN or PATH set there is
# honored. A set seam bypasses resolution.
#
# `set -e` is deliberately NOT set: both halves are allowed to fail and are
# graded explicitly below.
set -uo pipefail

# ── AMBIENT GIT REDIRECTION IS NEUTRALISED BEFORE ANY GIT RUNS ──────────────
#
# Every git call below is spelled `git -C "$REPO" <verb>`, which READS as "act
# on $REPO" but is not: `-C` only changes directory, while GIT_DIR and its
# siblings SKIP repository discovery outright. An ambient GIT_DIR therefore
# redirects every call site -- the generation worktree's and the landing's
# included -- into a repository this script was never told about, with both
# $REPO and -C inert. Measured 2026-08-31: that is how a test run of
# scripts/tests/test_install_memory_metadata_coverage_census_timer.py committed
# placeholder content onto main in the live project_root checkout and rewrote
# its .git/config identity, despite every path in that harness being pinned to
# a tmp dir.
#
# GIT_CEILING_DIRECTORIES does NOT cover this (the suite-wide first defence in
# df_pytest_isolation.py): a ceiling bounds the upward WALK, and an explicit
# GIT_DIR never walks.
#
# Unsetting is correct for production too, not just under test: this wrapper
# always means $REPO and has no legitimate use for an inherited git context.
# The systemd unit hands it a clean env; a git hook, an interactive `git
# commit` shell, or a test harness does not.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_COMMON_DIR \
      GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_NAMESPACE \
      GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM GIT_CONFIG_COUNT GIT_CONFIG_PARAMETERS

case "${BASH_SOURCE[0]}" in */*) _self_dir="${BASH_SOURCE[0]%/*}" ;; *) _self_dir=. ;; esac
_uv_lib="$_self_dir/lib/resolve_uv.sh"
# shellcheck source=lib/resolve_uv.sh
source "$_uv_lib" || { echo "${0##*/}: ERROR: cannot load the shared uv resolver $_uv_lib -- refusing to guess which uv to run." >&2; exit 127; }

REPO="${REPO:-/home/leo/src/dark-factory}"
FM="$REPO/fused-memory"

set -a
# shellcheck disable=SC1091
[ -f "$REPO/.env" ] && source "$REPO/.env"
set +a
export CONFIG_PATH="${CONFIG_PATH:-$FM/config/config.yaml}"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export FALKORDB_URI="${FALKORDB_URI:-redis://localhost:6379}"

# --top-n 400 matches the committed artifact's recorded params, so a nightly
# regeneration diffs as CONTENT drift and never as a rendering-width change.
CENSUS_TOP_N="${CENSUS_TOP_N:-400}"

# shellcheck disable=SC2206
CENSUS_CMD=(${COVERAGE_CENSUS_CMD:-})
# shellcheck disable=SC2206
STAMP_CMD=(${RETRO_STAMP_CMD:-})
if [ "${#CENSUS_CMD[@]}" -eq 0 ] || [ "${#STAMP_CMD[@]}" -eq 0 ]; then
    UV_RESOLVED="$(require_uv_bin)" || exit $?
    UV_PYTHON=("$UV_RESOLVED" run --frozen --project "$FM" python)
    [ "${#CENSUS_CMD[@]}" -eq 0 ] && CENSUS_CMD=("${UV_PYTHON[@]}")
    [ "${#STAMP_CMD[@]}" -eq 0 ] && STAMP_CMD=("${UV_PYTHON[@]}")
fi

ARTIFACTS=(
    plans/memory-metadata-census-report.json
    plans/memory-metadata-census-report.md
    plans/memory-metadata-coverage-history.json
)

# ── WHERE THE CENSUS MEASURES ───────────────────────────────────────────────
#
# A git checkout root is measured in a detached GENERATION WORKTREE of main's
# tip, running that worktree's own copy of the census, so no checkout's working
# tree is written but the generation worktree's (OPERATIONS.md §12). A $REPO
# inside no checkout runs in place. A $REPO inside a checkout that is not its
# root, or whose generation worktree cannot be prepared, is not measured.
#
# The EXIT trap removes the generation worktree. TERM and INT exit through it
# only once the foreground step returns (scripts/lib/deploy_main_copy.sh
# records why that order is load-bearing).
gen_scratch=""
remove_generation_worktree() {
    [ -n "$gen_scratch" ] || return 0
    [ -d "$gen_scratch/tree" ] && git -C "$REPO" worktree remove --force "$gen_scratch/tree" >/dev/null 2>&1
    git -C "$REPO" worktree prune >/dev/null 2>&1
    rm -rf "$gen_scratch"
}
trap remove_generation_worktree EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

census_root=""
land_refusal=""
if ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
    census_root="$REPO"
    land_refusal="skipped:not-a-git-repo"
elif ! repo_toplevel=$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null) || \
     [ "$(cd "$REPO" 2>/dev/null && pwd -P)" != "$(cd "$repo_toplevel" 2>/dev/null && pwd -P)" ]; then
    # FAIL CLOSED. The repository git resolves for $REPO is not $REPO itself:
    # a git env var a future git release adds, a symlinked or nested checkout,
    # or $REPO pointed at a subdirectory. Measuring or landing there would act
    # on something this run was not told about; declining costs one nightly
    # row, which the next run's append restores.
    echo "memory-metadata-coverage-census: REFUSING to measure or land — \$REPO ($REPO)" \
         "is not the root of the repository git resolves for it" \
         "(${repo_toplevel:-<none>})" >&2
    land_refusal="refused:repo-not-toplevel"
elif ! BASE=$(git -C "$REPO" rev-parse --verify --quiet 'main^{commit}') || \
     ! gen_scratch=$(mktemp -d) || \
     ! git -C "$REPO" worktree add --detach --quiet "$gen_scratch/tree" "$BASE" >&2; then
    echo "memory-metadata-coverage-census: could not prepare a generation worktree of" \
         "$REPO's main (${BASE:-no main branch}); not measuring in place instead" >&2
    land_refusal="failed:generation-worktree"
else
    census_root="$gen_scratch/tree"
fi

if [ -n "$census_root" ]; then
    echo "memory-metadata-coverage-census: censusing $census_root (top-n $CENSUS_TOP_N)"
    "${CENSUS_CMD[@]}" "$census_root/fused-memory/scripts/census_memory_metadata.py" --top-n "$CENSUS_TOP_N"
    census_rc=$?
    if [ "$census_rc" -ne 0 ]; then
        echo "memory-metadata-coverage-census: census exited $census_rc — the" \
             "artifacts still carry the evidence; rehearsing the retro stamp anyway" >&2
    fi
else
    census_rc=skipped
    echo "memory-metadata-coverage-census: census skipped ($land_refusal)" >&2
fi

echo "memory-metadata-coverage-census: rehearsing the retro topic/canonical stamp (DRY RUN)"
"${STAMP_CMD[@]}" "$FM/scripts/retro_stamp_topics.py"
stamp_rc=$?
if [ "$stamp_rc" -ne 0 ]; then
    echo "memory-metadata-coverage-census: retro-stamp rehearsal exited $stamp_rc" >&2
fi

# ── LAND WHAT THE CENSUS REGENERATED ────────────────────────────────────────
#
# Through scripts/_merge_queue_landing.py::land_files: one commit of the
# changed artifacts, read from the generation worktree, on a unique deploy
# branch, through one normal-lane request on $REPO's own merge queue, waited
# for and confirmed by git. `commit_step` names the outcome in ONE stable
# token on the done(...) line (the vocabulary is OPERATIONS.md §12's); the
# heredoc prints it as its last stdout line. `commit_rc` says whether the
# landing step failed.
SCRIPT_DIR=$(cd "$_self_dir" && pwd)
PY="$SCRIPT_DIR/../.venv/bin/python3"
[ -x "$PY" ] || PY=python3

land_artifacts() {
    "$PY" - "$@" <<'LAND_PY'
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

script_dir, repo, base, census_root, label, message, *changed = sys.argv[1:]
sys.path.insert(0, script_dir)


def verdict(token, narration, *, rc):
    print(f"memory-metadata-coverage-census: {narration}", file=sys.stderr, flush=True)
    print(token, flush=True)
    sys.exit(rc)


try:
    from _merge_queue_landing import LandingNotConfirmed, land_files, reclaim_retained_deploys
    from dashboard.config import discover_root_escalation_url
except ImportError as exc:
    from _config_reload_gate import transport_not_importable_remedy
    verdict("failed:transport_not_importable", transport_not_importable_remedy(exc), rc=1)

url = discover_root_escalation_url(Path(repo))
try:
    port = urlsplit(url).port if url else None
except ValueError:
    port = None
if port is None:
    verdict("refused:no-escalation-port",
            f"{repo}'s orchestrator config names no escalation port ({url}), so there is no "
            f"merge queue to land through", rc=1)

files = {rel: (Path(census_root) / rel).read_bytes() for rel in changed}
try:
    in_flight = reclaim_retained_deploys(repo, label, port=str(port))
    if in_flight:
        verdict("skipped:landing-in-flight",
                f"an earlier night's landing may still land ({', '.join(in_flight)}); skipping "
                f"tonight's so the two cannot conflict on the append-only history", rc=0)
    landing = land_files(
        repo, files, base, port=str(port), label=label, message=message,
        timeout_secs=os.environ.get("CENSUS_LANDING_TIMEOUT_SECS") or None,
    )
except LandingNotConfirmed as exc:
    verdict(f"failed:{exc.failure.value}", exc.detail, rc=1)
verdict(f"landed:{landing.request_id or landing.branch}",
        f"landed {landing.branch} request_id={landing.request_id} as {landing.landed_as}", rc=0)
LAND_PY
}

commit_rc=0
if [ "${CENSUS_COMMIT:-1}" != "1" ]; then
    commit_step="skipped:disabled"
    echo "memory-metadata-coverage-census: CENSUS_COMMIT=0 — measured and" \
         "rehearsed; landing nothing"
elif [ -n "$land_refusal" ]; then
    commit_step="$land_refusal"
    case "$land_refusal" in skipped:*) ;; *) commit_rc=1 ;; esac
    echo "memory-metadata-coverage-census: not landing the artifacts ($land_refusal)" >&2
else
    changed=()
    for rel in "${ARTIFACTS[@]}"; do
        [ -n "$(git -C "$census_root" status --porcelain --untracked-files=all -- "$rel")" ] &&
            changed+=("$rel")
    done
    if [ "${#changed[@]}" -eq 0 ]; then
        commit_step="skipped:no-drift"
        echo "memory-metadata-coverage-census: artifacts unchanged — nothing to land"
    else
        verdict_file="$gen_scratch/landing-verdict"
        land_artifacts "$SCRIPT_DIR" "$REPO" "$BASE" "$census_root" memory-metadata-census \
            "chore(census): nightly canonical/topic coverage census $(date -u +%Y-%m-%d)" \
            "${changed[@]}" > "$verdict_file"
        commit_rc=$?
        mapfile -t verdict_lines < "$verdict_file"
        commit_step="failed:no-verdict"
        [ "${#verdict_lines[@]}" -gt 0 ] && [ -n "${verdict_lines[-1]}" ] &&
            commit_step="${verdict_lines[-1]}"
        [ "$commit_step" = "failed:no-verdict" ] && commit_rc=1
    fi
fi

# `commit-step=` is appended AFTER `commit=$commit_rc` so every existing
# `'commit=0' in ...` / `'commit=1' in ...` substring check keeps matching.
# `commit_rc` keeps its own meaning ("did the landing step fail?"); the token
# is orthogonal ("which branch ran?"): several tokens narrate commit=0, which
# is exactly what made "declined" and "landed" ambiguous in the journal before
# this field existed.
echo "memory-metadata-coverage-census: done (census=$census_rc stamp=$stamp_rc commit=$commit_rc commit-step=$commit_step)"
# Always 0 once the halves have run (the pre-flight 127 is the header's one
# exception) — a recurring systemd oneshot that exits non-zero enters `failed`
# state and STAYS there, silently ending the append-only trend this job
# exists to build (see the header, and scripts/fused-memory-flag-marker-sweep.sh).
# That is a strictly worse outcome than the one missed nightly row a refusal
# costs -- the next run's append restores it. The loud-over-silent-
# degradation norm this invokes is satisfied by LEGIBILITY, not by a wedged
# timer: commit-step= above names exactly which branch declined and why, so
# `commit=0` can no longer read as "landed" when it was not.
exit 0
