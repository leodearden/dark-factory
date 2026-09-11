#!/usr/bin/env bash
# scripts/warm-lane-disk-guard.sh — Fail-closed disk-pressure admission guard
# for the warm-lane CoW pool. Checks BOTH free bytes AND free inodes on the
# warm-lane mount; reflink shares extents but NOT inodes, so a bytes-only
# guard is insufficient (PRD docs/prds/warm-lane-pool-space-safety.md §2, D4).
#
# `check --soft` (task 5175, contract §9.4 of
# docs/prds/warm-lane-pool-sizing-lifecycle.md) adds a PROACTIVE soft floor
# ABOVE the hard floor: dark-factory's dispatch throttle polls this to back
# off admission before the hard floor (exit 75) is ever reached. Modelled on
# scripts/fleet-load-detector.sh's exit-3 "oversubscribed" convention — exit 3
# is a distinct throttle-not-requeue signal, never confused with the
# EX_TEMPFAIL 75 requeue-this-command semantics.
#
# Usage:
#   scripts/warm-lane-disk-guard.sh check [--mount DIR] [--min-free-gib N] [--min-free-inodes N]
#                                          [--soft] [--soft-free-gib N] [--soft-free-inodes N]
#
# Subcommands:
#   check   Measure free bytes and inodes; exit 0 if both healthy, 75 if below floor.
#           With --soft: additionally exit 3 (+ stdout sentinel) if healthy but
#           below the soft floor.
#
# Options (env defaults shown):
#   --mount DIR            Warm-lane mount point  (env: REIFY_WARM_LANE_MOUNT)
#   --min-free-gib N       Minimum free GiB required (env: REIFY_WARM_LANE_DISK_GUARD_MIN_FREE_GIB;
#                          default: 50; maps to dark-factory-orchestrator.yaml warm_lane_pool.min_free_gib)
#   --min-free-inodes N    Minimum free inodes required (env: REIFY_WARM_LANE_DISK_GUARD_MIN_FREE_INODES;
#                          default: 500000; maps to dark-factory-orchestrator.yaml warm_lane_pool.min_free_inodes)
#   --soft                 Enable the soft-floor throttle check (see Exit codes: 3).
#   --soft-free-gib N      Soft free-GiB floor; must exceed --min-free-gib (env:
#                          REIFY_WARM_LANE_DISK_GUARD_SOFT_FREE_GIB; default: 500;
#                          maps to dark-factory-orchestrator.yaml warm_lane_pool.soft_free_gib)
#   --soft-free-inodes N   Soft free-inodes floor; must exceed --min-free-inodes (env:
#                          REIFY_WARM_LANE_DISK_GUARD_SOFT_FREE_INODES; default: 5000000;
#                          maps to dark-factory-orchestrator.yaml warm_lane_pool.soft_free_inodes)
#   -h, --help             Print this message and exit.
#
# Exit codes:
#   0   — Healthy: free_bytes >= min_free_gib AND free_inodes >= min_free_inodes,
#          AND (if --soft) at/above the soft floor too.
#   75  — Backpressure (EX_TEMPFAIL): either HARD axis below floor, OR fail-closed
#          measurement failure (df exits non-zero / emits unparseable output).
#          Orchestrator requeues 75. Takes precedence over the soft check (E3).
#   3   — Soft pressure (--soft only): above the hard floor (admission still
#          allowed) but below the soft floor on either axis. Stdout carries
#          `@@REIFY_WARM_LANE_SOFT_PRESSURE@@ free_gib=<G> budget_gib=<B>` —
#          a throttle-not-requeue signal for dark-factory's dispatch admission
#          loop (distinct from 75, mirrors fleet-load-detector.sh's exit 3).
#   2   — Usage error: unknown flag, missing/unknown subcommand, missing mount,
#          non-integer threshold, or (--soft only) soft floor <= hard floor.
#   0   — --help.
#
# Env knobs:
#   REIFY_WARM_LANE_DISK_GUARD_DF           df command override (default: df)
#   REIFY_WARM_LANE_DISK_GUARD_MIN_FREE_GIB minimum free bytes in GiB (default: 50)
#   REIFY_WARM_LANE_DISK_GUARD_MIN_FREE_INODES minimum free inodes (default: 500000)
#   REIFY_WARM_LANE_DISK_GUARD_SOFT_FREE_GIB soft free bytes in GiB (default: 500;
#                                             only consulted under --soft)
#   REIFY_WARM_LANE_DISK_GUARD_SOFT_FREE_INODES soft free inodes (default: 5000000;
#                                             only consulted under --soft)
#   REIFY_WARM_LANE_MOUNT                   warm-lane mount point (shared with preflight)
#
# The df invocation: "$DF" -B1 --output=avail,iavail -- "$MOUNT"
# (GNU coreutils df; Linux/XFS-only infra — same constraint as test_warm_lane_pool.sh)
# Reads both axes from one consistent snapshot.
#
# stdout contract: hard `check` (no --soft) NEVER writes to stdout in any case
# (diagnostics are all on stderr). The ONLY stdout output the guard ever
# produces is the soft-pressure sentinel line under `check --soft`.

set -euo pipefail

# ── log helpers (all write to stderr) ─────────────────────────────────────────
info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*" >&2; }
ok()    { printf '\033[1;32m[ok]\033[0m    %s\n' "$*" >&2; }
err()   { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }
hint()  { err "Hint:  $*"; }

# ── usage ──────────────────────────────────────────────────────────────────────
_usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") check [--mount DIR] [--min-free-gib N] [--min-free-inodes N]
                               [--soft] [--soft-free-gib N] [--soft-free-inodes N]

  Fail-closed disk-pressure admission guard for the warm-lane CoW pool.
  Checks BOTH free bytes AND free inodes (reflink shares extents, not inodes).
  Exits 0 when both axes are healthy; exits 75 (EX_TEMPFAIL) to signal
  backpressure — the orchestrator requeues exit-75 as transient infra pressure.

  --soft additionally checks a PROACTIVE soft floor (above the hard floor):
  when healthy-but-below-soft, exits 3 with a stdout sentinel — a
  throttle-not-requeue signal for dark-factory's dispatch admission loop.

  Subcommands:
    check   Measure free bytes and inodes against floor thresholds.

  Options:
    --mount DIR            Warm-lane mount point (default: \$REIFY_WARM_LANE_MOUNT)
    --min-free-gib N       Minimum free GiB required
                           (default: \$REIFY_WARM_LANE_DISK_GUARD_MIN_FREE_GIB or 50)
                           Maps to: dark-factory-orchestrator.yaml warm_lane_pool.min_free_gib
    --min-free-inodes N    Minimum free inodes required
                           (default: \$REIFY_WARM_LANE_DISK_GUARD_MIN_FREE_INODES or 500000)
                           Maps to: dark-factory-orchestrator.yaml warm_lane_pool.min_free_inodes
    --soft                 Enable the soft-floor throttle check (exit 3).
    --soft-free-gib N      Soft free-GiB floor; must exceed --min-free-gib
                           (default: \$REIFY_WARM_LANE_DISK_GUARD_SOFT_FREE_GIB or 500)
                           Maps to: dark-factory-orchestrator.yaml warm_lane_pool.soft_free_gib
    --soft-free-inodes N   Soft free-inodes floor; must exceed --min-free-inodes
                           (default: \$REIFY_WARM_LANE_DISK_GUARD_SOFT_FREE_INODES or 5000000)
                           Maps to: dark-factory-orchestrator.yaml warm_lane_pool.soft_free_inodes
    -h, --help             Print this message and exit.

  Exit codes:
    0   — All checked axes healthy (hard, and soft if --soft).
    75  — Below-threshold on either HARD axis, OR fail-closed measurement
          failure. Takes precedence over the soft check.
    3   — (--soft only) Above the hard floor but below the soft floor;
          stdout carries @@REIFY_WARM_LANE_SOFT_PRESSURE@@ free_gib=<G> budget_gib=<B>.
    2   — Usage error (wiring bug, not transient pressure); includes a
          --soft floor configured <= its hard-floor counterpart.
EOF
}

# ── defaults ───────────────────────────────────────────────────────────────────
DF="${REIFY_WARM_LANE_DISK_GUARD_DF:-df}"
MOUNT="${REIFY_WARM_LANE_MOUNT:-}"
MIN_FREE_GIB="${REIFY_WARM_LANE_DISK_GUARD_MIN_FREE_GIB:-50}"
MIN_FREE_INODES="${REIFY_WARM_LANE_DISK_GUARD_MIN_FREE_INODES:-500000}"
SOFT_MODE=0
SOFT_FREE_GIB="${REIFY_WARM_LANE_DISK_GUARD_SOFT_FREE_GIB:-500}"
SOFT_FREE_INODES="${REIFY_WARM_LANE_DISK_GUARD_SOFT_FREE_INODES:-5000000}"

# ── arg parsing ────────────────────────────────────────────────────────────────
SUBCOMMAND=""

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            _usage; exit 0 ;;
        --mount)
            [ $# -ge 2 ] || { err "--mount requires a value"; exit 2; }
            MOUNT="$2"; shift 2 ;;
        --min-free-gib)
            [ $# -ge 2 ] || { err "--min-free-gib requires a value"; exit 2; }
            MIN_FREE_GIB="$2"; shift 2 ;;
        --min-free-inodes)
            [ $# -ge 2 ] || { err "--min-free-inodes requires a value"; exit 2; }
            MIN_FREE_INODES="$2"; shift 2 ;;
        --soft)
            SOFT_MODE=1; shift ;;
        --soft-free-gib)
            [ $# -ge 2 ] || { err "--soft-free-gib requires a value"; exit 2; }
            SOFT_FREE_GIB="$2"; shift 2 ;;
        --soft-free-inodes)
            [ $# -ge 2 ] || { err "--soft-free-inodes requires a value"; exit 2; }
            SOFT_FREE_INODES="$2"; shift 2 ;;
        check)
            SUBCOMMAND="check"; shift ;;
        -*)
            err "Unknown flag: $1"
            err "Run '$(basename "$0") --help' for usage."
            exit 2 ;;
        *)
            err "Unknown subcommand: $1"
            err "Run '$(basename "$0") --help' for usage."
            exit 2 ;;
    esac
done

# Validate subcommand
if [ -z "$SUBCOMMAND" ]; then
    err "Missing subcommand. Expected: check"
    err "Run '$(basename "$0") --help' for usage."
    exit 2
fi

# Validate mount
if [ -z "$MOUNT" ]; then
    err "Warm-lane mount not specified. Set REIFY_WARM_LANE_MOUNT or pass --mount DIR."
    err "Run '$(basename "$0") --help' for usage."
    exit 2
fi

# Validate threshold integers (misconfiguration is a wiring bug, not transient pressure;
# exit 2 = usage error so it is loud rather than fail-open or crash).
# A non-integer MIN_FREE_GIB would make $(( MIN_FREE_GIB * ... )) evaluate to 0 and pass
# everything; a non-integer MIN_FREE_INODES would make [ "$avail_inodes" -lt ... ] throw
# 'integer expression expected', aborting with a non-75 code the orchestrator won't requeue.
if ! printf '%s\n' "$MIN_FREE_GIB" | grep -qE '^[0-9]+$'; then
    err "REIFY_WARM_LANE_DISK_GUARD_MIN_FREE_GIB (or --min-free-gib) is not a valid integer: '$MIN_FREE_GIB'."
    err "Run '$(basename "$0") --help' for usage."
    exit 2
fi
if ! printf '%s\n' "$MIN_FREE_INODES" | grep -qE '^[0-9]+$'; then
    err "REIFY_WARM_LANE_DISK_GUARD_MIN_FREE_INODES (or --min-free-inodes) is not a valid integer: '$MIN_FREE_INODES'."
    err "Run '$(basename "$0") --help' for usage."
    exit 2
fi

# Soft-floor config validation (--soft only; keeps the hard `check` path
# byte-behaviorally identical when --soft is absent — E2). Runs BEFORE the
# df call so a wiring bug (soft <= hard) is loud (exit 2) regardless of
# measurement outcome — a VALID soft config still fail-closes to 75 on a
# df failure (E3; see the fail-closed df read below).
if [ "$SOFT_MODE" -eq 1 ]; then
    if ! printf '%s\n' "$SOFT_FREE_GIB" | grep -qE '^[0-9]+$'; then
        err "REIFY_WARM_LANE_DISK_GUARD_SOFT_FREE_GIB (or --soft-free-gib) is not a valid integer: '$SOFT_FREE_GIB'."
        err "Run '$(basename "$0") --help' for usage."
        exit 2
    fi
    if ! printf '%s\n' "$SOFT_FREE_INODES" | grep -qE '^[0-9]+$'; then
        err "REIFY_WARM_LANE_DISK_GUARD_SOFT_FREE_INODES (or --soft-free-inodes) is not a valid integer: '$SOFT_FREE_INODES'."
        err "Run '$(basename "$0") --help' for usage."
        exit 2
    fi

    # E1: soft floor must strictly exceed the hard floor on each axis, else
    # this is a wiring bug (not transient pressure) — exit 2 loudly.
    if [ "$SOFT_FREE_GIB" -le "$MIN_FREE_GIB" ]; then
        err "Soft floor misconfigured: --soft-free-gib (${SOFT_FREE_GIB} GiB) must exceed --min-free-gib (${MIN_FREE_GIB} GiB) (hard floor)."
        hint "Raise --soft-free-gib above the hard floor, or lower --min-free-gib."
        exit 2
    fi
    if [ "$SOFT_FREE_INODES" -le "$MIN_FREE_INODES" ]; then
        err "Soft floor misconfigured: --soft-free-inodes (${SOFT_FREE_INODES}) must exceed --min-free-inodes (${MIN_FREE_INODES}) (hard floor)."
        hint "Raise --soft-free-inodes above the hard floor, or lower --min-free-inodes."
        exit 2
    fi
fi

# ── check subcommand ───────────────────────────────────────────────────────────
info "warm-lane-disk-guard.sh check: mount=$MOUNT  min_free_gib=$MIN_FREE_GIB  min_free_inodes=$MIN_FREE_INODES"

# Single df call: avail bytes (via -B1) + avail inodes, consistent snapshot.
# Captured without tripping set -e (fail-closed contract: df failure → deny).
df_out="$("$DF" -B1 --output=avail,iavail -- "$MOUNT")" || {
    err "df command failed (DF='$DF', MOUNT='$MOUNT'). Cannot confirm disk health; denying admission (fail-closed)."
    hint "Check that '$DF' is available and '$MOUNT' is accessible."
    exit 75
}
data_line="$(printf '%s\n' "$df_out" | tail -n +2 | head -n 1)"
avail_bytes="$(printf '%s\n' "$data_line" | awk '{print $1}')"
avail_inodes="$(printf '%s\n' "$data_line" | awk '{print $2}')"

# Validate integer fields before comparison (fail-closed on non-integer output).
if ! printf '%s\n' "$avail_bytes" | grep -qE '^[0-9]+$'; then
    err "df reported non-integer avail bytes: '$avail_bytes'. Cannot confirm disk health; denying admission (fail-closed)."
    hint "Verify '$DF -B1 --output=avail,iavail -- $MOUNT' produces parseable output."
    exit 75
fi
if ! printf '%s\n' "$avail_inodes" | grep -qE '^[0-9]+$'; then
    err "df reported non-integer avail inodes: '$avail_inodes'. Cannot confirm disk health; denying admission (fail-closed)."
    hint "Verify '$DF -B1 --output=avail,iavail -- $MOUNT' produces parseable output."
    exit 75
fi

info "  avail_bytes=$avail_bytes  avail_inodes=$avail_inodes"

# ── Bytes gate ─────────────────────────────────────────────────────────────────
min_bytes=$(( MIN_FREE_GIB * 1024 * 1024 * 1024 ))
if [ "$avail_bytes" -lt "$min_bytes" ]; then
    err "Free bytes below floor: ${avail_bytes} bytes available < ${min_bytes} bytes required (${MIN_FREE_GIB} GiB)."
    err "Warm-lane pool is at risk of ENOSPC; denying admission until disk space is reclaimed."
    hint "Reclaim space or lower REIFY_WARM_LANE_DISK_GUARD_MIN_FREE_GIB (currently ${MIN_FREE_GIB} GiB)."
    exit 75
fi

# ── Inodes gate ────────────────────────────────────────────────────────────────
if [ "$avail_inodes" -lt "$MIN_FREE_INODES" ]; then
    err "Free inodes below floor: ${avail_inodes} available < ${MIN_FREE_INODES} required."
    err "Warm-lane reflinks consume inodes (not just extents); denying admission until inodes are reclaimed."
    hint "Reclaim inodes or lower REIFY_WARM_LANE_DISK_GUARD_MIN_FREE_INODES (currently ${MIN_FREE_INODES})."
    exit 75
fi

# ── Soft gate (proactive throttle; --soft only, and only once the hard gate
# above has already passed) ─────────────────────────────────────────────────────
# dark-factory's dispatch-admission loop polls this so it can back off BEFORE
# the hard floor (exit 75) is ever reached. budget_gib is soft_free_gib (the
# free-space throttle floor this guard measures) — this guard reads only
# `df` and cannot compute the §9.2 resident_divergent_budget (a du-based
# measure that is α/β's job, not this guard's).
if [ "$SOFT_MODE" -eq 1 ]; then
    soft_bytes=$(( SOFT_FREE_GIB * 1024 * 1024 * 1024 ))
    if [ "$avail_bytes" -lt "$soft_bytes" ] || [ "$avail_inodes" -lt "$SOFT_FREE_INODES" ]; then
        free_gib=$(( avail_bytes / 1024 / 1024 / 1024 ))
        err "Soft pressure: free_gib=${free_gib} (soft floor ${SOFT_FREE_GIB} GiB) / avail_inodes=${avail_inodes} (soft floor ${SOFT_FREE_INODES})."
        hint "Above the hard floor (admission still allowed) but below the soft floor; dark-factory's dispatch throttle should back off."
        printf '@@REIFY_WARM_LANE_SOFT_PRESSURE@@ free_gib=%s budget_gib=%s\n' "$free_gib" "$SOFT_FREE_GIB"
        exit 3
    fi
fi

ok "check: disk space healthy (bytes=${avail_bytes} >= ${min_bytes}, inodes=${avail_inodes} >= ${MIN_FREE_INODES})."
exit 0
