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
# Then a third, non-measuring step COMMITS what the census regenerated. It is
# not part of the measure/rehearse pairing -- it is what keeps the trend from
# evaporating; see the block above the commit itself for why that is a
# data-loss question rather than a tidiness one.
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
# scripts/reify-closure-staleness-sweep.sh). That matters more here than for
# most jobs: the census exits 1 BY DESIGN whenever `coverage.complete` is false,
# which on a live corpus that orchestrators write to during the scroll is a
# routine outcome, not a fault. Propagating it would wedge the timer on the
# first churny night and quietly end the trend -- while the artifacts, which
# carry the evidence of the shortfall, were written anyway. Both exit codes are
# narrated instead, so exiting 0 never means the failure is invisible.
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
# no-op source). CENSUS_COMMIT=0 skips the artifact commit, for an ad-hoc
# operator run that should leave the checkout alone.
#
# `set -e` is deliberately NOT set: both halves are allowed to fail and are
# graded explicitly below.
set -uo pipefail

# ── AMBIENT GIT REDIRECTION IS NEUTRALISED BEFORE ANY GIT RUNS ──────────────
#
# Every git call below is spelled `git -C "$REPO" <verb>`, which READS as "act
# on $REPO" but is not: `-C` only changes directory, while GIT_DIR and its
# siblings SKIP repository discovery outright. An ambient GIT_DIR therefore
# redirects all five call sites -- `commit --only` and the scoped `add --`
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
      GIT_CONFIG_GLOBAL GIT_CONFIG_SYSTEM GIT_CONFIG_COUNT

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
CENSUS_CMD=(${COVERAGE_CENSUS_CMD:-uv run --frozen --project "$FM" python})
# shellcheck disable=SC2206
STAMP_CMD=(${RETRO_STAMP_CMD:-uv run --frozen --project "$FM" python})

echo "memory-metadata-coverage-census: censusing $REPO (top-n $CENSUS_TOP_N)"
"${CENSUS_CMD[@]}" "$FM/scripts/census_memory_metadata.py" --top-n "$CENSUS_TOP_N"
census_rc=$?
if [ "$census_rc" -ne 0 ]; then
    echo "memory-metadata-coverage-census: census exited $census_rc — the" \
         "artifacts still carry the evidence; rehearsing the retro stamp anyway" >&2
fi

echo "memory-metadata-coverage-census: rehearsing the retro topic/canonical stamp (DRY RUN)"
"${STAMP_CMD[@]}" "$FM/scripts/retro_stamp_topics.py"
stamp_rc=$?
if [ "$stamp_rc" -ne 0 ]; then
    echo "memory-metadata-coverage-census: retro-stamp rehearsal exited $stamp_rc" >&2
fi

# ── COMMIT WHAT THE CENSUS REGENERATED ──────────────────────────────────────
#
# The census rewrites three GIT-TRACKED files under $REPO/plans/ (it resolves
# them from its own __file__, so they land in $REPO no matter the cwd), and
# $REPO defaults to the project_root checkout that CLAUDE.md flags as
# MACHINE-OPERATED: the merge worker, the startup reconciler and git hooks all
# act on it directly. Leaving the artifacts dirty every morning is therefore
# not untidiness but a silent data-loss path --
# plans/memory-metadata-coverage-history.json is APPEND-ONLY and IS the trend
# this job exists to produce, so an uncommitted append that the merge worker's
# advance path resets away takes that night's row with it, permanently and
# with no error anywhere. Committing is what makes the trend durable.
#
# SCOPED, per CLAUDE.md's rule for this checkout: `git commit --only <paths>`
# and never a bare `git commit` (which would sweep a concurrent process's
# staged work into the census's nightly commit), never `git add -A`, and never
# `git stash` (refs/stash is ONE ref shared by every worktree and the merge
# worker's advance path also consumes it -- incident 13674d3c68).
#
# Same seam and same fallback as scripts/legibility/census.py's
# _build_default_commit, which already commits docs/legibility/census-state.json
# from the 03:00 job, and scripts/legibility/nightly.py::_git_commit_docs_only:
# a path git has never tracked makes `--only` fail with "did not match any
# file", which is retried once after a scoped `git add -- <paths>`.
#
# BEST-EFFORT, LIKE BOTH OTHER HALVES: the outcome is narrated and never
# propagated (see the always-exit-0 oneshot rationale in the header), and it
# runs even when the census exited non-zero -- a `coverage.complete: false`
# night still WROTE artifacts carrying the evidence of the shortfall, and that
# evidence is exactly what needs to reach the repo.
#
# CENSUS_COMMIT=0 skips the step entirely, for an operator running the gauge
# ad-hoc who does not want a commit (pairs with the census's own --no-history).
ARTIFACTS=(
    plans/memory-metadata-census-report.json
    plans/memory-metadata-census-report.md
    plans/memory-metadata-coverage-history.json
)

commit_rc=0
if [ "${CENSUS_COMMIT:-1}" != "1" ]; then
    echo "memory-metadata-coverage-census: CENSUS_COMMIT=0 — leaving the" \
         "regenerated artifacts uncommitted"
elif ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
    echo "memory-metadata-coverage-census: $REPO is not a git repository —" \
         "skipping the artifact commit" >&2
elif ! repo_toplevel=$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null) || \
     [ "$(cd "$REPO" 2>/dev/null && pwd -P)" != "$(cd "$repo_toplevel" 2>/dev/null && pwd -P)" ]; then
    # FAIL CLOSED. The repository git resolves for $REPO is not $REPO itself,
    # so this run would commit into something it was not told about. The unset
    # above removes the known cause; this refuses on ANY residual one (a git
    # env var added by a future git release, a symlinked or nested checkout,
    # $REPO pointed at a subdirectory). Committing into the wrong repository is
    # the defect; declining to commit is merely a missed nightly row, which the
    # next run's append restores.
    echo "memory-metadata-coverage-census: REFUSING to commit — \$REPO ($REPO)" \
         "is not the root of the repository git resolves for it" \
         "(${repo_toplevel:-<none>}); leaving the artifacts uncommitted" >&2
    commit_rc=1
elif [ -z "$(git -C "$REPO" status --porcelain -- "${ARTIFACTS[@]}" 2>/dev/null)" ]; then
    # The ordinary quiet-night outcome. Checked BEFORE committing so that
    # "nothing to commit" -- which git reports with a NON-ZERO code -- is never
    # narrated as a failure.
    echo "memory-metadata-coverage-census: artifacts unchanged — nothing to commit"
else
    commit_msg="chore(census): nightly canonical/topic coverage census $(date -u +%Y-%m-%d)"
    commit_out=$(git -C "$REPO" commit --only "${ARTIFACTS[@]}" \
        -m "$commit_msg" 2>&1)
    commit_rc=$?
    # NEVER `printf '%s' "$commit_out" | grep -qi ...` here. `grep -q` exits
    # the instant it matches, and the match is on line 1 of git's own error
    # message; the `printf` writer can then die of SIGPIPE, `set -o pipefail`
    # (line 68) promotes the pipeline's status to 141, and a TRUE predicate
    # reads FALSE — silently skipping the retry below on the very first night
    # this path is ever committed. Measured: 0.6% (25/4000) at git's natural
    # ~270-byte message under fleet load, 100% once the message passes the
    # ~64KB pipe buffer. `${commit_out,,}` is a pure parameter expansion: it
    # forks nothing, so the predicate is a function of the message, not of how
    # much of it a subprocess got to read before another one hung up on it.
    # scripts/setup-host.sh::_parity_verdict already reached this same
    # conclusion at a sibling site; this adopts its settled remedy.
    if [ "$commit_rc" -ne 0 ] && \
       [[ "${commit_out,,}" == *'did not match any file'* ]]; then
        # First-ever run for a path git has never tracked — stage just those
        # paths and retry once.
        git -C "$REPO" add -- "${ARTIFACTS[@]}" >/dev/null 2>&1
        commit_out=$(git -C "$REPO" commit --only "${ARTIFACTS[@]}" \
            -m "$commit_msg" 2>&1)
        commit_rc=$?
    fi
    if [ "$commit_rc" -ne 0 ]; then
        echo "memory-metadata-coverage-census: artifact commit exited" \
             "$commit_rc — the regenerated files are still on disk, but" \
             "UNCOMMITTED in a machine-operated checkout: $commit_out" >&2
    else
        echo "memory-metadata-coverage-census: committed the regenerated artifacts"
    fi
fi

echo "memory-metadata-coverage-census: done (census=$census_rc stamp=$stamp_rc commit=$commit_rc)"
# Always 0 — see the oneshot rationale in the header.
exit 0
