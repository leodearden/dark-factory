#!/usr/bin/env bash
# scripts/thin-warm-lane.sh — Free-first target-reclaim primitive for the
# warm-lane CoW pool.
#
# PRD docs/prds/warm-lane-pool-sizing-lifecycle.md §9.3 (Pillar C — lifecycle).
#
# FREE-FIRST: removes <lane_dir>/target outright, freeing the divergent
# extents directly — no reseed staged by default (acquire_lane always
# re-seeds from base at acquire time — D10, cow-seeding). Progresses at true
# ENOSPC (a straight rm never needs the transient 2x space a reflink clone
# briefly does). Optional --reseed re-seeds a thin base clone via
# seed-warm-lane.sh --fresh-checkout AFTER the free (T2: free-before-stage
# ordering — never inherits the ENOSPC-deadlock class task 5168 fixed for
# warm-lane-gc.sh).
#
# Usage:
#   scripts/thin-warm-lane.sh <lane_dir> [--reseed] [--base <base_target_dir>] \
#       [--seed-script <path>]
#
# Options:
#   --reseed              After freeing target/, re-seed a thin base clone:
#                         <seed-script> <base_target_dir> <lane_dir> --fresh-checkout
#                         --assume-lane-lock-held. thin already holds
#                         ${LANE_DIR}.lock on FD 9 (T3) across the seed call, so it
#                         passes --assume-lane-lock-held so seed-warm-lane.sh skips
#                         its own (now default-on) lane-lock acquire rather than
#                         self-refusing against thin's held lock (esc-5214/task 5354).
#                         Best-effort: a failed re-seed is logged but does not
#                         change the exit code (see Exit codes below).
#                         Requires --base. Default OFF (no clone staged).
#   --base DIR             base_target_dir to seed from. Required with --reseed.
#                         Per seed-warm-lane.sh's D8 resolve convention, this MUST
#                         be the CONCRETE .gen.N path, NOT the <base>/target
#                         symlink -- thin-warm-lane.sh forwards it verbatim,
#                         unresolved, to seed-warm-lane.sh.
#   --seed-script PATH     Seed primitive invoked by --reseed (default: sibling
#                         scripts/seed-warm-lane.sh). Hermetic test seam.
#   -h, --help              Print this message and exit (0).
#
# Preconditions (caller-asserted FREE is independently re-verified via the
# lane flock — T3; the other two are checked here before any work):
#   lane_dir must exist and be a directory.
#   lane_dir must be under $REIFY_WARM_LANE_MOUNT, when that env var is set.
#   lane_dir must not resolve to (or be literally named) the base dir, and
#   must not be the mount root itself.
#
# Stdout: resolved <lane_dir> on success. Stderr: all diagnostics.
#
# Exit codes:
#   0   — target/ freed. With --reseed, re-seeding is attempted best-effort:
#         a failed re-seed is logged (stderr "Re-seed FAILED") but does NOT
#         change the exit code, since the free — the operation this script
#         guarantees — already succeeded (D10: acquire always re-seeds from
#         base anyway, so a lane a caller re-acquires is never left cold).
#   1   — Precondition guard refusal (nonexistent lane, ==base dir, not under
#         $REIFY_WARM_LANE_MOUNT).
#   2   — Usage error (unknown flag, missing positional, --reseed without --base).
#   75  — EX_TEMPFAIL: the lane's own flock is held by a live consumer
#         (ASSIGNED lane; inv.2 one-consumer-per-lane-at-a-time).
#
# Invariants (PRD §9.3):
#   T1 — only target/ is ever removed; source tree / .git / uncommitted WIP
#        are never touched, so unlanded WIP always survives on the branch.
#   T2 — free-first: bytes are freed BEFORE any reseed clone is staged.
#   T3 — the lane's own flock (<lane_dir>.lock) is acquired non-blocking;
#        refuses (75) rather than blocking or stealing an ASSIGNED lane.
#        Assumes <lane_dir>.lock already exists (pool acquire/release
#        convention, inv.2): if it does not, opening it below creates it on
#        the fly, which can itself fail under a truly exhausted filesystem
#        (ENOSPC) before any bytes are freed. Only bites a lane thinned
#        outside the pool's acquire/release lifecycle.

set -euo pipefail

# ── log helpers (all write to stderr) ────────────────────────────────────────
info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*" >&2; }
ok()    { printf '\033[1;32m[ok]\033[0m    %s\n' "$*" >&2; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*" >&2; }
err()   { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

# ── usage ─────────────────────────────────────────────────────────────────────
_usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") <lane_dir> [--reseed] [--base <base_target_dir>] [--seed-script <path>]

  Free-first target-reclaim primitive for the warm-lane CoW pool (PRD
  docs/prds/warm-lane-pool-sizing-lifecycle.md §9.3). Removes <lane_dir>/target
  outright, freeing bytes BEFORE any reseed clone is staged (T2). No reseed is
  staged by default (acquire_lane always re-seeds from base — D10).

  <lane_dir>              Warm-lane pool lane to thin. Must exist, be under
                          \$REIFY_WARM_LANE_MOUNT (when set), and not be the
                          base dir.

  --reseed                After freeing target/, re-seed a thin base clone
                          (--seed-script <base_target_dir> <lane_dir> --fresh-checkout
                          --assume-lane-lock-held; thin holds \${LANE_DIR}.lock on FD 9
                          across the call, so seed skips its own default acquire).
                          Best-effort: a failed re-seed is logged but does not
                          change the exit code. Requires --base. Default OFF.
  --base DIR              base_target_dir to seed from. Required with --reseed.
                          Per seed-warm-lane.sh's D8 resolve convention, this MUST
                          be the CONCRETE .gen.N path, NOT the <base>/target
                          symlink -- forwarded verbatim, unresolved.
  --seed-script PATH      Seed primitive to invoke for --reseed (default:
                          sibling scripts/seed-warm-lane.sh). Hermetic test seam.
  -h, --help              Print this message and exit (0).

  Stdout: resolved <lane_dir> on success.
  Stderr: all diagnostics.

  Exit codes:
    0   — target/ freed. With --reseed, re-seeding is best-effort: a failed
          re-seed is logged (stderr) but does not change the exit code.
    1   — Precondition guard refusal (nonexistent lane, ==base dir, not under
          \$REIFY_WARM_LANE_MOUNT).
    2   — Usage error (unknown flag, missing positional, --reseed without --base).
    75  — EX_TEMPFAIL: the lane's flock is held by a live consumer (ASSIGNED
          lane; inv.2 one-consumer-per-lane).
EOF
}

# ── arg parsing ───────────────────────────────────────────────────────────────
RESEED=""
BASE_TARGET_DIR=""
SEED_SCRIPT=""
_POSITIONALS=()

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            _usage
            exit 0
            ;;
        --reseed)
            RESEED=1
            shift
            ;;
        --base)
            [ $# -ge 2 ] || { err "--base requires a value"; exit 2; }
            BASE_TARGET_DIR="$2"
            shift 2
            ;;
        --seed-script)
            [ $# -ge 2 ] || { err "--seed-script requires a value"; exit 2; }
            SEED_SCRIPT="$2"
            shift 2
            ;;
        -*)
            err "Unknown flag: $1"
            err "Run '$(basename "$0") --help' for usage."
            exit 2
            ;;
        *)
            _POSITIONALS+=("$1")
            shift
            ;;
    esac
done

if [ "${#_POSITIONALS[@]}" -lt 1 ]; then
    err "Missing required positional argument: <lane_dir>"
    err "Run '$(basename "$0") --help' for usage."
    exit 2
fi
if [ "${#_POSITIONALS[@]}" -gt 1 ]; then
    err "Unexpected extra positional argument(s): ${_POSITIONALS[*]:1}"
    exit 2
fi
# Strip a trailing slash so sibling-path constructions (e.g. "$LANE_DIR.lock")
# never land inside the lane instead of beside it.
LANE_DIR="${_POSITIONALS[0]%/}"

# --reseed requires --base (usage-shape check; fail fast, before any work).
if [ -n "$RESEED" ] && [ -z "$BASE_TARGET_DIR" ]; then
    err "--reseed requires --base <base_target_dir>"
    err "Run '$(basename "$0") --help' for usage."
    exit 2
fi

# Default --seed-script: sibling scripts/seed-warm-lane.sh, resolved via
# BASH_SOURCE so this script works when invoked via a relative/symlinked path.
if [ -z "$SEED_SCRIPT" ]; then
    _script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    SEED_SCRIPT="$_script_dir/seed-warm-lane.sh"
fi

# ── lane_dir existence guard ───────────────────────────────────────────────────
if [ ! -d "$LANE_DIR" ]; then
    err "lane_dir does not exist or is not a directory: $LANE_DIR"
    exit 1
fi

_rp_lane_dir="$(realpath -m "$LANE_DIR")"

# ── under-mount guard: gated on REIFY_WARM_LANE_MOUNT being set, trailing-
# slash prefix compare (mirrors seed-warm-lane.sh L428-440) so a sibling path
# like /mnt/warm-lanes-evil never falsely matches /mnt/warm-lanes ──────────────
if [ -n "${REIFY_WARM_LANE_MOUNT:-}" ]; then
    _rp_mount="$(realpath -m "${REIFY_WARM_LANE_MOUNT}")"
    case "$_rp_lane_dir/" in
        "$_rp_mount"/*) ;;
        *)
            err "Precondition guard: lane_dir is not under REIFY_WARM_LANE_MOUNT"
            err "  lane_dir: $_rp_lane_dir"
            err "  REIFY_WARM_LANE_MOUNT (canonicalized): $_rp_mount"
            exit 1
            ;;
    esac
fi

# ── self-clobber (≠base, ≠mount-root) guard: refuse if lane_dir resolves to
# <mount>/base (when REIFY_WARM_LANE_MOUNT is set), is literally named "base"
# regardless of the mount, or IS the mount root itself (mirrors seed-warm-lane.sh's
# self-clobber guard). The mount-root case is NOT caught by the under-mount
# guard above: its trailing-slash case-match treats "$mount/" as matching
# "$mount"/* (glob * also matches the empty tail), so the mount root alone
# would otherwise slip through as "under" the mount ───────────────────────────
_self_clobber=0
if [ -n "${REIFY_WARM_LANE_MOUNT:-}" ]; then
    _rp_mount_base="$(realpath -m "${REIFY_WARM_LANE_MOUNT}/base")"
    [ "$_rp_lane_dir" = "$_rp_mount_base" ] && _self_clobber=1
    [ "$_rp_lane_dir" = "$_rp_mount" ] && _self_clobber=1
fi
[ "$(basename "$_rp_lane_dir")" = "base" ] && _self_clobber=1
if [ "$_self_clobber" = "1" ]; then
    err "Precondition guard: lane_dir resolves to (or is named) the base dir, or is the mount root — refusing to thin"
    err "  lane_dir: $_rp_lane_dir"
    exit 1
fi

# ── T3: acquire the lane's own flock (non-blocking); refuse on contention ─────
# Mirrors warm-lane-gc.sh's live-consumer probe (L488-500): the same lane-lock
# convention (<lane_dir>.lock) other pool tooling (acquire/release, GC) uses.
LANE_LOCK="${LANE_DIR}.lock"
# The pool's acquire/release convention (inv.2) guarantees <lane_dir>.lock
# already exists for any lane that went through acquire_lane. If it is
# missing here, the exec below creates it on the fly -- on a genuinely
# exhausted filesystem that create can itself fail with ENOSPC before any
# bytes are freed. Log a breadcrumb so that failure mode reads as a lock-file
# creation problem rather than a bare, unexplained set -e abort.
[ -e "$LANE_LOCK" ] || info "Lane lock does not exist yet, creating: $LANE_LOCK (lane may never have been acquired through the pool)"
exec 9>"$LANE_LOCK"
if ! flock -n 9; then
    exec 9>&-
    err "Lane lock held by a live consumer (flock -n failed): $LANE_LOCK"
    err "Refusing to thin an ASSIGNED lane (inv.2: one consumer per lane at a time)."
    exit 75
fi
# FD 9 stays open (lock held) for the rest of the run; bash releases it on exit.

# ── FREE-FIRST reclaim (T1/T2): only target/ is ever removed ──────────────────
# Mirrors warm-lane-gc.sh's disk-pressure fast-path (L521-539): rm-stderr
# capture avoided set -e (the if-condition context) so a failure is reported
# with the actual error rather than a bare non-zero abort.
info "Thinning lane: $_rp_lane_dir"
_rm_err=""
if _rm_err="$(rm -rf "$LANE_DIR/target" 2>&1)"; then
    ok "Freed $LANE_DIR/target"
else
    err "rm -rf $LANE_DIR/target failed: ${_rm_err:-<rm produced no output>}"
    exit 1
fi

# ── optional --reseed (AFTER the free; T2 free-before-stage ordering) ─────────
# Redirect the seed-script's own stdout to stderr: thin-warm-lane's stdout
# contract is exactly one line (the resolved lane_dir, echoed below) and
# seed-warm-lane.sh has its own "resolved target path" stdout contract that
# would otherwise corrupt it.
if [ -n "$RESEED" ]; then
    info "Re-seeding thin base clone via $SEED_SCRIPT ..."
    # --assume-lane-lock-held: thin ALREADY holds ${LANE_DIR}.lock on FD 9 (T3,
    # line 236), held across this seed call. seed-warm-lane.sh acquires the lane
    # lock BY DEFAULT under --fresh-checkout (esc-5214/task 5354 fail-safe), so
    # without this opt-out it would re-open+flock the same file on its own FD 9
    # and self-refuse against thin's held lock (flock is not re-entrant across a
    # process tree). --assume-lane-lock-held tells seed to skip its own acquire;
    # thin's FD-9 lock already provides the inv.2 one-consumer exclusivity.
    if "$SEED_SCRIPT" "$BASE_TARGET_DIR" "$LANE_DIR" --fresh-checkout --assume-lane-lock-held >&2; then
        ok "Re-seeded: $LANE_DIR/target"
    else
        warn "Re-seed FAILED ($SEED_SCRIPT); lane target/ was already freed (success not masked)"
    fi
fi

ok "Thinned lane: $_rp_lane_dir"
echo "$_rp_lane_dir"
exit 0
