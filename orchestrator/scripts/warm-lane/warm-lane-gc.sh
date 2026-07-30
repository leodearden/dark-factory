#!/usr/bin/env bash
# scripts/warm-lane-gc.sh — Task-side GC for the warm-lane CoW pool.
# Reclaims divergent FREE lanes (reset via α seed primitive) and removes
# orphan cold worktrees; respects inv.preserve (dirty/unlanded/live-consumer).
#
# Part of PRD docs/prds/warm-lane-pool-space-safety.md §8.4, §10 δ.
# Consumer: dark-factory ε (invokes reclaim on the disk-pressure path).
#
# Usage:
#   scripts/warm-lane-gc.sh reclaim \
#       --mount WORKTREE_BASE \
#       [--worktrees-dir DIR] \
#       [--base-target SYMLINK] \
#       [--main-ref REF] \
#       [--lane-glob GLOB] \
#       [--protect-glob GLOB] \
#       [--seed-script PATH] \
#       [--disk-pressure]
#
#   OR (legacy / explicit form):
#   scripts/warm-lane-gc.sh reclaim \
#       --worktrees-dir DIR \
#       --base-target SYMLINK \
#       [...]
#
# Subcommands:
#   reclaim   Scan worktrees-dir; reset reclaimable lanes via α seed primitive;
#             remove reclaimable orphan worktrees; preserve dirty/unlanded/live lanes.
#
# Options:
#   --mount WORKTREE_BASE  Warm-lane WORKTREES base dir — the same value dark-factory
#                          passes as str(self.worktree_base) to BOTH warm-lane scripts
#                          (git_ops.py _run_warm_lane_gc_reclaim line 1401 and
#                          _run_warm_lane_disk_guard line 1364).  This is the WORKTREES
#                          DIR (/home/leo/src/warm-lanes/worktrees), NOT the XFS mount
#                          point (/home/leo/src/warm-lanes).
#                          Derives: WORKTREES_DIR=$MOUNT (lanes live directly under it),
#                                   BASE_TARGET=$(dirname "$MOUNT")/base/target
#                                   (the XFS-sibling base dir, one level up from
#                                   the worktrees-dir).
#                          The two scripts consume the shared value differently:
#                            warm-lane-disk-guard.sh: df's a path on the XFS volume
#                              (any path on the volume suffices for disk-space checks).
#                            warm-lane-gc.sh (this script): treats it as the
#                              worktrees-dir and derives BASE_TARGET from its PARENT.
#                          The worktrees-dir satisfies both because it is simultaneously
#                          on the XFS volume AND the directory that holds the lanes.
#                          Explicit --worktrees-dir / --base-target override the derived
#                          defaults, so all existing callers that pass both explicit
#                          flags are unchanged.
#   --worktrees-dir DIR    Directory holding lane/worktree entries.
#                          Required unless --mount is given (or env override set).
#   --base-target SYMLINK  Symlink at <base>/target → <base>/target.gen.N.
#                          Required unless --mount is given (or env override set).
#                          Resolved to its concrete .gen.N dir before invoking α.
#   --main-ref REF         Git ref for "main" branch (default: main).
#   --lane-glob GLOB       Glob matching pool-lane entries (default: _lane-*,_spec-*).
#                          Matched entries are reset via α, not removed.
#   --protect-glob GLOB    Glob matching entries to never touch (default:
#                          _merge-*,_mainprobe-*,_mainsweep-*,_solo-*,
#                          _substrate-gate-*,_offline-deep,_iact-*). Matched
#                          entries are skipped entirely. This is the full set
#                          of orchestrator-managed non-pool worktree kinds
#                          dark-factory mints directly under the warm-lane
#                          mount (git_ops.py ephemeral_worktree /
#                          PROTECTED_PREFIXES); none of them may ever be
#                          orphan-removed by Pass 2 — e.g. _mainprobe-*/
#                          _mainsweep-* must survive while a background
#                          integrity sweep is live (task 5221).
#   --extra-protect-glob GLOB
#                          Additive protect glob (opt-in): ADDS to (never
#                          replaces) the --protect-glob set above, comma-joined,
#                          so a caller can protect extra entries WITHOUT
#                          restating the default list (avoids a G7
#                          lockstep-duplication hit). Matched entries are
#                          skipped entirely (Pass 1 reset AND Pass 2 remove) and
#                          counted as preserved, exactly like --protect-glob.
#                          An EXPLICIT-PIN escape hatch: the caller names the
#                          entries. It is NOT the liveness guard — a live
#                          process reference is detected by the script itself,
#                          per entry, in both passes (task 5572), so no caller
#                          needs to compute or pass one. Default:
#                          REIFY_WARM_LANE_GC_EXTRA_PROTECT_GLOB (any non-empty
#                          value = on). Off by default.
#   --seed-script PATH     Path to the α seed primitive (default: sibling seed-warm-lane.sh).
#                          Overridable for hermetic testing.
#   --disk-pressure        Fast-path: reclaim a lane by `rm -rf <lane>/target`
#                          instead of invoking the α reflink-reseed clone — no
#                          transient 2×-space requirement. Valid because
#                          acquire_lane always re-seeds from base (D10 §9.5).
#                          Applies to every flock-free lane in Pass 1
#                          (always-reclaim); counted as `reset` in the
#                          summary. Default: REIFY_WARM_LANE_GC_DISK_PRESSURE
#                          (any non-empty value = on). Off by default.
#   -h, --help             Print this message and exit.
#
# Exit codes:
#   0  — Completed sweep (best-effort; per-candidate failures warn + continue).
#   1  — Runtime error: could not resolve required argument (e.g. base-target symlink).
#   2  — Usage/WIRING error: unknown flag, missing subcommand, missing required
#        option, or a missing sibling library (scripts/lib_live_refs.sh). The
#        library case is a wiring error, not a runtime one: nothing about the
#        invocation could have avoided it and no retry will fix it — the
#        deployment is incomplete. Same class as an unknown flag, hence 2.
#
# Env knobs (all overridable by flags):
#   REIFY_WARM_LANE_GC_MOUNT            — default --mount (worktree_base)
#   REIFY_WARM_LANE_GC_WORKTREES_DIR    — default --worktrees-dir
#   REIFY_WARM_LANE_GC_BASE_TARGET      — default --base-target
#   REIFY_WARM_LANE_GC_MAIN_REF         — default --main-ref (default: main)
#   REIFY_WARM_LANE_GC_LANE_GLOB        — default --lane-glob
#   REIFY_WARM_LANE_GC_PROTECT_GLOB     — default --protect-glob
#   REIFY_WARM_LANE_GC_EXTRA_PROTECT_GLOB — default --extra-protect-glob (any
#                                         non-empty value = on); off by default
#   REIFY_WARM_LANE_GC_SEED_SCRIPT      — default --seed-script
#   REIFY_WARM_LANE_GC_DISK_PRESSURE    — default --disk-pressure (any non-empty
#                                         value = on); off by default
#
# Design notes:
#   - --mount is the dark-factory consumer interface.  Dark-factory passes the same
#     value (str(self.worktree_base)) to both warm-lane scripts; this script and
#     warm-lane-disk-guard.sh consume it differently:
#       THIS script (gc.sh): --mount = the WORKTREES DIR.  The value is assigned to
#         WORKTREES_DIR directly, and BASE_TARGET is derived from its PARENT:
#           WORKTREES_DIR  = <--mount value>
#           BASE_TARGET    = $(dirname <--mount value>)/base/target
#         On the real host: --mount = /home/leo/src/warm-lanes/worktrees, so
#           WORKTREES_DIR = /home/leo/src/warm-lanes/worktrees
#           BASE_TARGET   = /home/leo/src/warm-lanes/base/target
#       warm-lane-disk-guard.sh: --mount = any path on the XFS volume; it is passed
#         to df for disk-space checks, so the worktrees-dir works equally well.
#     The worktrees-dir satisfies both contracts because it is simultaneously ON the
#     XFS volume (so disk-guard's df finds the right filesystem) AND the directory
#     that holds the lanes (so gc.sh's WORKTREES_DIR assignment is correct).
#     Explicit --worktrees-dir / --base-target override the derived values, so the
#     hermetic test harness continues to use arbitrary temp paths without change.
#   - Reclaimability is computed purely from filesystem + git + flock; dark-factory
#     FREE/ASSIGNED state is NOT consulted. "FREE/idle" ≈ no live consumer holding
#     the lane flock (mirroring refresh-warm-base.sh reader-refcount GC).
#   - inv.preserve shared predicate (_is_reclaimable): skip on dirty tracked changes
#     (git status --porcelain), unlanded ahead-of-main (merge-base --is-ancestor),
#     or live consumer (flock -n -x <dir>.lock fails).
#     A fourth preserve reason sits OUTSIDE that predicate, in BOTH pass loops:
#     a live PROCESS REFERENCE at or under the entry (cwd / open fd / mmap;
#     live_ref_present, task 5572). It is checked per entry immediately before
#     the reset (Pass 1) or the remove (Pass 2), under the flock the loop
#     already holds, and AFTER _is_reclaimable in Pass 2 (cheapest decisive
#     predicate first — see the live-reference gate note below). Its share of
#     the preserved count is reported separately as preserved_live_ref=.
#   - Always-reclaim (Pass 1 only, task 5326): a FREE pool lane whose
#     live-consumer flock is free is reclaimed REGARDLESS of dirty tracked
#     changes, ahead-of-main tip, or backing-task status. acquire_lane ALWAYS
#     re-seeds from base (§9.5), so a FREE lane's divergent target/ is never
#     reused; committed work lives on refs/heads/task/NNNN and reset touches
#     only target/, never the source tree or branch (sizing-lifecycle T1).
#     Preserving a flock-free lane's target/ thus yields zero warm-cache value
#     and only accretes disk. Pass 1 has exactly TWO preserve gates, checked in
#     this order: (1) the live-consumer flock (inv.2), and (2) a live PROCESS
#     REFERENCE at or under the lane dir (cwd / open fd / mmap — task 5572).
#     This subsumes the former Tier-3 terminal-task reclaim
#     (task 5167): the rebase-orphan ahead-of-main lane it targeted is now
#     reclaimed by the general rule. Pass 2 keeps the clean+landed
#     _is_reclaimable rule.
#   - Live-reference gate (task 5572, both passes): the inv.2 flock is held only
#     across the ACQUIRE reseed, NOT across the consumer's long cargo verify
#     build, so during that build the flock is FREE and always-reclaim would wipe
#     <lane>/target out from under a live build (esc-5375-1). The check lives
#     HERE, in the primitive, rather than in a wrapper, because there are TWO
#     entry points and only one of them ever passed --extra-protect-glob:
#     warm-lane-gc-sweep.sh (the δ systemd backstop) did; dark-factory's ε path
#     (git_ops.py _run_warm_lane_gc_reclaim) invokes `warm-lane-gc.sh reclaim
#     --mount <base>` and never has. Every caller now inherits the guard with no
#     dark-factory change. It runs PER LANE, immediately before that lane's
#     reset and under the flock this loop already holds — not as an up-front
#     snapshot, which would leave a lane that goes live mid-traversal
#     unprotected across a 25-40 minute pass. Over-preserving is safe and
#     TEMPORARY: a lingering reference costs one extra sweep.
#   - COST REVERSAL, stated explicitly (task 5572 review). Moving from ONE batch
#     /proc scan to a PER-ENTRY scan is O(entries × /proc), and the sweep comment
#     this replaced called a per-entry scan "too costly for the hot path". That
#     claim is hereby SUPERSEDED, not silently contradicted — with measurements:
#       * one live_ref_present call ......... ~1.9s  (find+awk over /proc ~1.5s
#                                                     + grep over /proc/*/maps
#                                                     ~0.38s), on 1159 live
#                                                     processes / 11527 fd+cwd
#                                                     symlinks
#       * a 93-entry pool, Pass 1 + Pass 2 ... ~2-3 min of added wall clock
#     measured 2026-07-28 on the live host. This lands on dark-factory's ε
#     admission path (git_ops.py _run_warm_lane_gc_reclaim), which imposes NO
#     timeout, against a reclaim pass that already runs 25-40 minutes.
#     Accepted, because the alternatives are worse:
#       * a batch scan is what created the TOCTOU this gate exists to close;
#       * "only flock-FREE entries pay the walk" is TRUE but nearly vacuous — in
#         a pool at rest almost every lane is flock-free, so budget for the full
#         entries × 1.9s, not for a fraction of it;
#       * awk's early exit only fires once EVERY candidate is found, so the
#         common case (free entry, about to be reclaimed) always pays the full
#         walk — and that is exactly the verdict that must be fresh.
#     What IS bought back, cheaply: live_ref_present short-circuits after the
#     cwd/fd pass (~1.5s instead of ~1.9s for a live entry), and Pass 2 runs the
#     ~18ms _is_reclaimable git pair BEFORE the walk so only an orphan that would
#     actually be removed pays it. An up-front batch classification used only to
#     pre-mark live entries was CONSIDERED and rejected: it costs one extra full
#     scan and only breaks even once ≥2 entries are live at pass start, which a
#     pool at rest never is.
#   - α reuse: resolve base symlink → concrete gen, hold flock -s during α call
#     (D8 reader-refcount seam; same contract as the acquire path).
#   - Safety-ranked order: reset lanes first (cheap), then remove orphans (destructive).
#   - Stdout: machine-readable summary line only —
#     `reclaim: reset=N removed=N preserved=N preserved_live_ref=N
#     preserved_assigned=N`. New fields are APPENDED so prefix-matching
#     consumers keep working.
#     Stderr: all diagnostics (info/ok/warn/err).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shared live-reference /proc scanner (live_ref_present). Fail LOUDLY if it is
# missing: a silently-absent liveness guard is precisely the failure mode task
# 5572 exists to remove. Exit 2 (usage/WIRING), not 1 (runtime) — an incomplete
# deployment, in the same class as an unknown flag; see the exit-code table.
if [ ! -f "$SCRIPT_DIR/lib_live_refs.sh" ]; then
    echo "warm-lane-gc.sh: ERROR — scripts/lib_live_refs.sh not found next to warm-lane-gc.sh" >&2
    exit 2
fi
# shellcheck source=scripts/lib_live_refs.sh
source "$SCRIPT_DIR/lib_live_refs.sh"

# Shared durable-lane-record reader (lane_state_read / lane_state_class), task
# 3074. Pass 1's reclaimability gate consults dark-factory's own lane record
# through it — see the Pass-1 record gate below.
# shellcheck source=scripts/lib_lane_state.sh
source "$SCRIPT_DIR/lib_lane_state.sh"

# ── log helpers (all write to stderr) ─────────────────────────────────────────
info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*" >&2; }
ok()    { printf '\033[1;32m[ok]\033[0m    %s\n' "$*" >&2; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*" >&2; }
err()   { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

# ── usage ──────────────────────────────────────────────────────────────────────
_usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") reclaim --mount WORKTREE_BASE [OPTIONS]
   or: $(basename "$0") reclaim --worktrees-dir DIR --base-target SYMLINK [OPTIONS]

  Task-side GC for the warm-lane CoW pool.
  Reclaims divergent FREE lanes (reset via α seed primitive) and removes
  orphan cold worktrees; respects inv.preserve (dirty/unlanded/live-consumer).

  Subcommands:
    reclaim   Scan worktrees-dir; reset reclaimable lanes; remove orphan worktrees.

  Required options (one of):
    --mount WORKTREE_BASE  Worktrees base dir (DF consumer interface); derives
                           --worktrees-dir and --base-target automatically.
    --worktrees-dir DIR    Directory holding lane/worktree entries (explicit).
    --base-target SYMLINK  Symlink <base>/target → <base>/target.gen.N (explicit).
    Validation: --mount OR (both --worktrees-dir AND --base-target).

  Optional options:
    --main-ref REF        Git ref for 'main' (default: main).
    --lane-glob GLOB      Glob for pool-lane entries (default: _lane-*,_spec-*).
    --protect-glob GLOB   Glob for protected entries (default:
                          _merge-*,_mainprobe-*,_mainsweep-*,_solo-*,
                          _substrate-gate-*,_offline-deep,_iact-*) — the full
                          set of orchestrator-managed non-pool worktree kinds,
                          which must never be orphan-removed (e.g. ephemeral
                          verify/sweep worktrees while a background integrity
                          sweep is live).
    --extra-protect-glob GLOB
                          Additive protect glob: ADDS to (never replaces) the
                          --protect-glob set, so a caller protects extra entries
                          without restating the default list (default:
                          REIFY_WARM_LANE_GC_EXTRA_PROTECT_GLOB). Off by default.
    --seed-script PATH    Path to α seed primitive (default: sibling seed-warm-lane.sh).
    --disk-pressure       Fast-path: reclaim via `rm -rf <lane>/target` instead
                          of the α reflink-reseed clone (default:
                          REIFY_WARM_LANE_GC_DISK_PRESSURE). Off by default.
    -h, --help            Print this message and exit.

  Exit codes:
    0  — Completed sweep (per-candidate failures warn + continue).
    1  — Runtime error (e.g. base-target symlink unresolvable).
    2  — Usage/wiring error (incl. a missing sibling scripts/lib_live_refs.sh).

  Output:
    stdout: machine-readable summary:
            reclaim: reset=N removed=M preserved=K preserved_live_ref=L preserved_assigned=A
            (L is the share of K held back by a live process reference — the
             only preserve reason that can shield an entry indefinitely)
            (A is the share of K held back by dark-factory's own durable lane
             record reading assigned/in_use — the only preserve reason that is
             AUTHORITATIVE rather than a probe, and one that self-clears when
             the lane is released)
    stderr: all diagnostics.
EOF
}

# ── defaults ───────────────────────────────────────────────────────────────────
MOUNT="${REIFY_WARM_LANE_GC_MOUNT:-}"
WORKTREES_DIR="${REIFY_WARM_LANE_GC_WORKTREES_DIR:-}"
BASE_TARGET="${REIFY_WARM_LANE_GC_BASE_TARGET:-}"
MAIN_REF="${REIFY_WARM_LANE_GC_MAIN_REF:-main}"
LANE_GLOB="${REIFY_WARM_LANE_GC_LANE_GLOB:-}"
PROTECT_GLOB="${REIFY_WARM_LANE_GC_PROTECT_GLOB:-}"
# Extra additive protect glob (task 5378): any non-empty value adds to (never
# replaces) the effective protect set; empty/unset = off. Off by default.
EXTRA_PROTECT_GLOB="${REIFY_WARM_LANE_GC_EXTRA_PROTECT_GLOB:-}"
SEED_SCRIPT="${REIFY_WARM_LANE_GC_SEED_SCRIPT:-}"
# Disk-pressure fast-path (task 5167): any non-empty value = on; off by default.
DISK_PRESSURE="${REIFY_WARM_LANE_GC_DISK_PRESSURE:-}"

# ── arg parsing ────────────────────────────────────────────────────────────────
SUBCOMMAND=""

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            _usage; exit 0 ;;
        --mount)
            [ $# -ge 2 ] || { err "--mount requires a value"; exit 2; }
            MOUNT="$2"; shift 2 ;;
        --worktrees-dir)
            [ $# -ge 2 ] || { err "--worktrees-dir requires a value"; exit 2; }
            WORKTREES_DIR="$2"; shift 2 ;;
        --base-target)
            [ $# -ge 2 ] || { err "--base-target requires a value"; exit 2; }
            BASE_TARGET="$2"; shift 2 ;;
        --main-ref)
            [ $# -ge 2 ] || { err "--main-ref requires a value"; exit 2; }
            MAIN_REF="$2"; shift 2 ;;
        --lane-glob)
            [ $# -ge 2 ] || { err "--lane-glob requires a value"; exit 2; }
            LANE_GLOB="$2"; shift 2 ;;
        --protect-glob)
            [ $# -ge 2 ] || { err "--protect-glob requires a value"; exit 2; }
            PROTECT_GLOB="$2"; shift 2 ;;
        --extra-protect-glob)
            [ $# -ge 2 ] || { err "--extra-protect-glob requires a value"; exit 2; }
            EXTRA_PROTECT_GLOB="$2"; shift 2 ;;
        --seed-script)
            [ $# -ge 2 ] || { err "--seed-script requires a value"; exit 2; }
            SEED_SCRIPT="$2"; shift 2 ;;
        --disk-pressure)
            DISK_PRESSURE="1"; shift ;;
        reclaim)
            SUBCOMMAND="reclaim"; shift ;;
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

# ── validate subcommand ────────────────────────────────────────────────────────
if [ -z "$SUBCOMMAND" ]; then
    err "Missing subcommand. Expected: reclaim"
    err "Run '$(basename "$0") --help' for usage."
    exit 2
fi

# ── apply --mount derivation (before required-options validation) ──────────────
# --mount sets WORKTREES_DIR and BASE_TARGET when not already set by explicit flags.
# Explicit --worktrees-dir / --base-target always override the derived values.
if [ -n "$MOUNT" ]; then
    [ -n "$WORKTREES_DIR" ] || WORKTREES_DIR="$MOUNT"
    [ -n "$BASE_TARGET"   ] || BASE_TARGET="$(dirname "$MOUNT")/base/target"
fi

# ── validate required options ──────────────────────────────────────────────────
# Valid: --mount (derives both) OR both --worktrees-dir AND --base-target explicit.
# Invalid: exactly one of --worktrees-dir / --base-target without --mount.
if [ -z "$WORKTREES_DIR" ]; then
    err "Missing required option: --mount WORKTREE_BASE or --worktrees-dir DIR"
    err "Run '$(basename "$0") --help' for usage."
    exit 2
fi
if [ -z "$BASE_TARGET" ]; then
    err "Missing required option: --mount WORKTREE_BASE or --base-target SYMLINK"
    err "Run '$(basename "$0") --help' for usage."
    exit 2
fi

# ── apply defaults for optional globs and seed-script ─────────────────────────
[ -n "$LANE_GLOB" ]    || LANE_GLOB="_lane-*,_spec-*"
# This list mirrors dark-factory's PROTECTED_PREFIXES worktree-kind inventory
# (git_ops.py) — every orchestrator-managed non-pool worktree kind minted
# directly under the warm-lane mount. Keeping the full set here means Pass 2
# (destructive orphan removal) can only ever target genuine pool lanes
# (_lane-*/_spec-*, handled by Pass 1's reset instead) and genuine cold
# non-underscore orphans (e.g. legacy task-*) — never a live managed worktree
# such as an ephemeral _mainsweep-*/_mainprobe-* verify sweep (task 5221).
# _merge-* MUST stay first/present: it is _merge-verify's ONLY gc protection
# (dark-factory's .merge_verify.lock is a different path gc never inspects),
# so this list only ever grows, never narrows an existing prefix.
[ -n "$PROTECT_GLOB" ] || PROTECT_GLOB="_merge-*,_mainprobe-*,_mainsweep-*,_solo-*,_substrate-gate-*,_offline-deep,_iact-*"
# Extra additive protect glob (task 5378): --extra-protect-glob /
# REIFY_WARM_LANE_GC_EXTRA_PROTECT_GLOB ADDS to (never replaces) the effective
# protect set — comma-joined onto whatever PROTECT_GLOB resolved to above
# (its default OR an explicit --protect-glob). This lets a caller EXPLICITLY PIN
# entries it names itself, WITHOUT restating the default list (avoiding a G7
# lockstep-duplication hit). It is not the liveness guard: a live process
# reference is detected per entry by this script, in both passes (task 5572).
# It flows through the SAME _matches_glob path in the enumeration loop, so a
# matched entry is skipped ENTIRELY in Pass 1 (reset) and Pass 2 (remove) and
# counted as preserved. PROTECT_GLOB is always non-empty here (defaulted just
# above), so the comma-join never yields a leading comma / empty pattern.
if [ -n "$EXTRA_PROTECT_GLOB" ]; then
    PROTECT_GLOB="${PROTECT_GLOB},${EXTRA_PROTECT_GLOB}"
fi
if [ -z "$SEED_SCRIPT" ]; then
    SEED_SCRIPT="$SCRIPT_DIR/seed-warm-lane.sh"
fi

# ── helper: name matches a glob pattern ───────────────────────────────────────
# _matches_glob <name> <comma-separated-globs>
_matches_glob() {
    local name="$1"
    local globs="$2"
    local g
    # Split comma-separated globs
    local IFS=","
    for g in $globs; do
        # shellcheck disable=SC2254
        case "$name" in
            $g) return 0 ;;
        esac
    done
    return 1
}

# ── helper: is this dir a git worktree? ───────────────────────────────────────
_is_git_worktree() {
    local dir="$1"
    [ -d "$dir" ] || return 1
    git -C "$dir" rev-parse --git-dir >/dev/null 2>&1 || return 1
}

# ── shared reclaimability predicate ───────────────────────────────────────────
# _is_reclaimable <dir>
# Returns 0 (reclaimable) or 1 (preserve) with diagnostic to stderr.
# Note: does NOT acquire the flock — that is the caller's responsibility.
_is_reclaimable() {
    local dir="$1"
    local name
    name="$(basename "$dir")"

    # (a) dirty tracked changes
    local dirty
    # --untracked-files=no excludes '??' lines (untracked artifacts like target/)
    # so only uncommitted changes to TRACKED files are flagged as dirty.
    dirty="$(git -C "$dir" status --porcelain --untracked-files=no 2>/dev/null)" || {
        warn "preserving $name: git status failed — treating as dirty"
        return 1
    }
    if [ -n "$dirty" ]; then
        warn "preserving $name: dirty WIP (uncommitted tracked changes)"
        return 1
    fi

    # (b) unlanded ahead-of-main commits
    if ! git -C "$dir" merge-base --is-ancestor HEAD "$MAIN_REF" 2>/dev/null; then
        warn "preserving $name: unlanded ahead-of-main commits"
        return 1
    fi

    return 0
}

# ── reclaim subcommand ─────────────────────────────────────────────────────────
_do_reclaim() {
    local reset_count=0
    local removed_count=0
    local preserved_count=0
    # Sub-count of preserved_count attributable to the live-reference gate
    # (BOTH passes). Reported separately because preserved_count lumps together
    # protect-glob skips, flock holds, dirty/unlanded orphans and live
    # references, and only the live-reference share can shield an entry
    # INDEFINITELY: a single abandoned shell whose cwd sits in a lane keeps that
    # lane out of reclaim on every subsequent pass, including under
    # --disk-pressure where reclaim is the ENOSPC response. Surfacing the count
    # makes a permanently-shielded pool visible in dark-factory's logs; the
    # per-entry `preserving <name>: live consumer (process reference)` warn on
    # stderr names WHICH entries, so an operator can find and clear the holder.
    local preserved_live_ref_count=0
    # Sub-count of preserved_count attributable to the Pass-1 lane-record gate
    # (task 3075). Reported SEPARATELY from preserved_live_ref_count, not folded
    # into it, because the two shares mean opposite things about pool health: a
    # rising preserved_assigned is a BUSY pool and self-clears the moment
    # dark-factory releases the lane, whereas a rising preserved_live_ref can
    # shield a lane INDEFINITELY. One combined number would hide a permanently
    # shielded pool behind ordinary business.
    local preserved_assigned_count=0

    info "warm-lane-gc.sh reclaim: worktrees_dir=$WORKTREES_DIR  base_target=$BASE_TARGET  main_ref=$MAIN_REF"

    # Resolve the base-target symlink to its concrete gen dir (D8 seam).
    # α requires the concrete path — cp -a copies the symlink otherwise.
    local resolved_gen
    if ! resolved_gen="$(readlink -f "$BASE_TARGET" 2>/dev/null)"; then
        err "Cannot resolve base-target symlink: $BASE_TARGET"
        return 1  # runtime error; exit 2 is reserved for usage/wiring errors
    fi
    local gen_lock="${resolved_gen}.lock"
    touch "$gen_lock" 2>/dev/null || true
    info "  resolved_gen=$resolved_gen  gen_lock=$gen_lock"

    # Enumerate all immediate subdirs in the worktrees-dir.
    # We collect entries first so we can do safety-ranked two-pass order:
    #   pass 1: reset reclaimable lanes (cheap, source-tree-preserving)
    #   pass 2: remove reclaimable orphans (destructive)

    local -a lane_candidates=()
    local -a orphan_candidates=()

    local entry name
    for entry in "$WORKTREES_DIR"/*/; do
        # Strip trailing slash
        entry="${entry%/}"
        [ -d "$entry" ] || continue
        name="$(basename "$entry")"

        # Skip protected entries entirely — count them as preserved in the summary
        # (they are not reclaimed, which is the user-visible meaning of "preserved").
        if _matches_glob "$name" "$PROTECT_GLOB"; then
            info "  skipping protected: $name"
            preserved_count=$((preserved_count + 1))
            continue
        fi

        # Only process git worktrees
        if ! _is_git_worktree "$entry"; then
            info "  skipping non-git-worktree: $name"
            continue
        fi

        if _matches_glob "$name" "$LANE_GLOB"; then
            lane_candidates+=("$entry")
        else
            orphan_candidates+=("$entry")
        fi
    done

    # ── Pass 1: reset reclaimable lanes ───────────────────────────────────────
    local lane
    for lane in "${lane_candidates[@]+${lane_candidates[@]}}"; do
        name="$(basename "$lane")"
        local lane_lock="${WORKTREES_DIR}/${name}.lock"

        # Acquire the lane lock NON-BLOCKING in the PARENT shell so the same
        # file description (and advisory lock) spans the reclaimability check
        # AND the seed-script call — no check→act race window.
        # Mirror: refresh-warm-base.sh §GC (flock held across the rm).
        exec 8>"$lane_lock"
        if ! flock -n 8; then
            exec 8>&-
            warn "preserving $name: live consumer (flock held)"
            preserved_count=$((preserved_count + 1))
            continue
        fi

        # ── Pass-1 record gate (task 3075, reify esc-5334-6) ─────────────────
        # Pass 1 has THREE preserve gates, checked in this order:
        #   (1) the live-consumer flock (inv.2) — acquired above;
        #   (2) THIS gate: dark-factory's own durable lane record at
        #       <worktrees-dir>/.lane-state/<lane>.json;
        #   (3) a live PROCESS REFERENCE at or under the lane dir (task 5572).
        #
        # The record is read BEFORE the /proc walk because it is BOTH the
        # cheaper and the more decisive predicate — cheapest-decisive-first, the
        # same rationale already documented for Pass 2's ordering. One file read
        # + a couple of seds is ~1ms against the walk's measured ~1.9s, and its
        # verdict is AUTHORITATIVE (dark-factory wrote it) rather than a probe.
        # So an assigned lane short-circuits before the walk and a BUSY pool
        # gets cheaper, not more expensive.
        #
        # Single-argument form deliberately: lane_state_read derives
        # `<dirname of lane-dir>/.lane-state`, which for a lane at
        # $WORKTREES_DIR/<name> is exactly $WORKTREES_DIR/.lane-state — the same
        # path warm-lane-audit.sh computes as $MOUNT/.lane-state. No flag, no
        # precomputed map, nothing marshalled in from a caller.
        #
        # PLACEMENT is load-bearing on four axes — the same shape as the
        # live-reference gate note below:
        #   - INSIDE the loop, immediately before the reset, resolved from THIS
        #     lane's own path. Hoisting it above the loop — or precomputing a
        #     state map, or accepting an `--assigned-lanes CSV` — reintroduces
        #     exactly the up-front-snapshot TOCTOU this gate removes: one verdict
        #     computed once, then consumed across a 25-40 minute lane-by-lane
        #     traversal. That is not theoretical. warm-lane-gc-sweep.sh computed
        #     its protect CSV once and consumed it across a sweep that ran
        #     06:32-06:59 BST while _lane-5 dropped OUT of the protect set at
        #     06:32; and that same sweep reset _lane-5 at 20:55Z while a task-5334
        #     agent session started 20:36Z was live (PRD §2). Block S-toctou goes
        #     RED against any such hoist.
        #   - UNDER the flock this loop already holds, so the read and the reset
        #     share one critical section and nothing can reassign the lane
        #     between verdict and action.
        #   - BEFORE both reset branches below (the --disk-pressure rm and the α
        #     reseed), so neither path can bypass it — the same "before BOTH
        #     branches" property Block K5 pins for the live-reference gate.
        #   - AFTER the flock acquire, so a flock-held lane keeps its own
        #     distinct diagnostic and all three preserve reasons stay
        #     distinguishable in dark-factory's logs.
        #
        # This is a COST REVERSAL of the direction the header's existing COST
        # REVERSAL note describes, and is stated rather than left to be inferred:
        # an ASSIGNED lane now short-circuits BEFORE the ~1.9s walk, so the
        # busier the pool the cheaper the pass — the opposite of what task 5572
        # had to accept.
        #
        # FAIL-OPEN is load-bearing. lane_state_read reports `unknown` for a
        # missing/unreadable/corrupt record, and UNKNOWN and QUARANTINED both
        # fall through to the gates below and then to reclaim. Preserving on
        # UNKNOWN would freeze reclaim the moment .lane-state/ is absent — the
        # 2026-07-10 ENOSPC accretion outage, re-created.
        lane_state_read "$lane" >/dev/null
        local lane_class
        lane_class="$(lane_state_class "$LANE_STATE_RAW")"
        if [ "$lane_class" = "ASSIGNED" ]; then
            exec 8>&-
            # The `(state=<raw>)` suffix keeps the PRD-pinned prefix
            # `preserving _lane-5: assigned to task 5334` matching as a
            # substring while surfacing `in_use` vs `assigned` — the distinction
            # leaf δ will begin writing and leaf ε depends on. A record with no
            # task_id gets its OWN wording: `assigned to task ` with an empty id
            # would be a dangling fact.
            if [ -n "$LANE_STATE_TASK_ID" ]; then
                warn "preserving $name: assigned to task $LANE_STATE_TASK_ID (state=$LANE_STATE_RAW)"
            else
                warn "preserving $name: assigned, no task id in record (state=$LANE_STATE_RAW)"
            fi
            preserved_count=$((preserved_count + 1))
            preserved_assigned_count=$((preserved_assigned_count + 1))
            continue
        fi

        # Deliberately ASYMMETRIC, and the asymmetry is the point.
        # `no-readable-record` is the ORDINARY reading for every recordless
        # `_iact-*` and manual operator worktree, so warning on it would emit a
        # line per such entry per pass and train operators to ignore the
        # channel. `unparseable-record` means the record IS there and something
        # wrote it wrong — an assigned lane has silently degraded back to the
        # pre-γ `FREE ≈ flock-free` regime, which is precisely the defect this
        # gate removes. That must be visible and attributable (INV-2,
        # structured-facts-at-failure) — and it must still NOT preserve: failing
        # closed on a corrupt record would let one bad write freeze a lane out
        # of reclaim forever.
        #
        # There is deliberately NO third branch for an unrecognized state:
        # lib_lane_state.sh does not set that cause, deriving it instead from
        # lane_state_class returning UNKNOWN for a non-empty raw, so there is no
        # second copy of the recognized-state table here to drift.
        if [ "${LANE_STATE_CAUSE:-}" = "unparseable-record" ]; then
            warn "$name: lane-state record is present but unparseable-record — falling back to the /proc live-consumer scan for this lane"
        fi

        # Live-reference gate (task 5572). The flock above proves no consumer is
        # mid-ACQUIRE; it does NOT prove no consumer is mid-BUILD, because the
        # inv.2 flock is released once the acquire reseed completes while the
        # verify build runs on for minutes (esc-5375-1). So before resetting,
        # ask /proc directly: does any live process reference this lane dir or
        # anything under it (cwd / open fd / mmap)?
        #
        # Placement is load-bearing on three axes:
        #   - AFTER the flock acquire, so a flock-held lane keeps its own
        #     distinct diagnostic — the two preserve reasons must stay
        #     distinguishable in dark-factory's logs. This also spares
        #     flock-held lanes the ~1.9s O(processes × fds) walk, but do NOT
        #     read that as the cost being small: in a pool at rest nearly every
        #     lane is flock-free, so the pass pays lanes × ~1.9s. See the COST
        #     REVERSAL note in the header for the measured budget.
        #   - INSIDE the loop, immediately before the reset. Hoisting it above
        #     the loop would reintroduce the up-front-snapshot TOCTOU this gate
        #     removes: one verdict computed once, then consumed across a 25-40
        #     minute lane-by-lane traversal, leaving any lane that goes live
        #     mid-pass unprotected.
        #   - Before BOTH reset branches below (--disk-pressure rm and the α
        #     reseed), so neither path can bypass it.
        #
        # Bias toward OVER-preserving: a genuinely FREE lane has no referencing
        # process, and a lingering reference is reclaimed on the next pass once
        # it clears. Under-preserving corrupts a live build.
        if live_ref_present "$lane"; then
            exec 8>&-
            warn "preserving $name: live consumer (process reference)"
            preserved_count=$((preserved_count + 1))
            preserved_live_ref_count=$((preserved_live_ref_count + 1))
            continue
        fi

        # Always-reclaim (task 5326): once the live-consumer flock is held
        # (acquired above ⇒ no live consumer) and no live process references the
        # lane, a FREE pool lane is reclaimed UNCONDITIONALLY — dirty tracked
        # changes, an ahead-of-main tip, and backing-task status are NOT
        # consulted. acquire_lane ALWAYS re-seeds a
        # lane from base (cow-seeding §9.5), so a FREE lane's divergent target/
        # is never reused; committed work lives on the durable
        # refs/heads/task/NNNN branch ref, and reset touches only target/, never
        # the source tree or branch (sizing-lifecycle Invariant T1). Preserving
        # a flock-free lane's target/ therefore yields zero warm-cache value and
        # only accretes disk. The flock (inv.2) and the live process reference
        # above are the two Pass-1 preserve gates; Pass 2 (destructive orphan
        # removal) keeps the conservative clean+landed _is_reclaimable rule.
        if [ -n "$DISK_PRESSURE" ]; then
            # Disk-pressure fast-path (task 5167): delete target/ outright
            # instead of invoking the α reflink-reseed clone — no transient
            # 2×-space requirement (the clone briefly needs old+new to
            # coexist; a straight rm never does). Valid because acquire_lane
            # ALWAYS re-seeds from base (D10 §9.5), so an empty/missing
            # target/ is a legal lane state. No gen-lock needed here: unlike
            # α, this path never reads the base/gen tree. Still under the
            # lane flock acquired above, mirroring the manual 2026-07-10
            # remediation.
            info "  resetting lane (disk-pressure): $name"
            local rm_err
            if rm_err="$(rm -rf "$lane/target" 2>&1)"; then
                ok "  reset lane (disk-pressure): $name"
                reset_count=$((reset_count + 1))
            else
                warn "  disk-pressure reset failed for $name: ${rm_err:-<rm produced no output>}; continuing"
                preserved_count=$((preserved_count + 1))
            fi
        else
            # Invoke α while the lane lock is held in the parent shell (FD 8;
            # the action subshell inherits it, the parent still owns the lock)
            # AND a shared gen lock is held on FD 9 (flock -s; D8 reader-refcount
            # seam). Pass --assume-lane-lock-held: seed-warm-lane.sh acquires the
            # lane lock BY DEFAULT under --fresh-checkout (esc-5214/task 5354
            # fail-safe), and its own FD-9 acquire would BOTH self-refuse against
            # gc's FD-8 lane lock AND clobber gc's FD-9 gen lock. The opt-out
            # makes seed skip its own acquire; gc's held locks already provide the
            # inv.2 one-consumer exclusivity + the gen reader-refcount.
            info "  resetting lane: $name"
            if (
                exec 9>"$gen_lock"
                flock -s 9
                "$SEED_SCRIPT" "$resolved_gen" "$lane" --fresh-checkout --assume-lane-lock-held
            ) 2>&1 | while IFS= read -r line; do warn "  [seed] $line"; done; then
                ok "  reset lane: $name"
                reset_count=$((reset_count + 1))
            else
                warn "  reset failed for $name (seed-script error); continuing"
                preserved_count=$((preserved_count + 1))
            fi
        fi
        exec 8>&-  # release lane lock; NOT removed — persists as per-lane mutex
    done

    # ── Pass 2: remove reclaimable orphans ────────────────────────────────────
    local orphan
    for orphan in "${orphan_candidates[@]+${orphan_candidates[@]}}"; do
        name="$(basename "$orphan")"
        local orphan_lock="${WORKTREES_DIR}/${name}.lock"

        # Same single-acquisition pattern as Pass 1: non-blocking exclusive
        # acquire in the parent shell held across reclaimability check + remove.
        exec 8>"$orphan_lock"
        if ! flock -n 8; then
            exec 8>&-
            warn "preserving $name: live consumer (flock held)"
            preserved_count=$((preserved_count + 1))
            continue
        fi

        # Reclaimability check (under the lock). Ordered BEFORE the live-reference
        # gate, cheapest-decisive-predicate first: measured on this host,
        # `git status --porcelain --untracked-files=no` is ~16ms and
        # `merge-base --is-ancestor` ~2ms, against ~1.9s for one live_ref_present
        # /proc walk — two orders of magnitude apart. A dirty or ahead-of-main
        # orphan is the COMMON Pass-2 preserve case, and it is already preserved
        # by this check, so paying the walk first would burn ~1.9s per orphan to
        # reach the identical decision. (An earlier revision ordered liveness
        # first for diagnostic attribution; the attribution is preserved anyway —
        # a live orphan that is ALSO clean+landed still reaches the gate below and
        # is still reported as a live-consumer preserve. The only thing given up
        # is naming liveness on an orphan that git would have preserved regardless,
        # which changes no outcome.)
        if ! _is_reclaimable "$orphan"; then
            exec 8>&-
            preserved_count=$((preserved_count + 1))
            continue
        fi

        # Live-reference gate (task 5572) — the SAME check Pass 1 runs, on the
        # destructive path, and the LAST gate before `git worktree remove
        # --force`. Kept AFTER the flock acquire for the same reasons as Pass 1
        # (see that comment), and immediately before the remove so no snapshot
        # can go stale between verdict and action.
        if live_ref_present "$orphan"; then
            exec 8>&-
            warn "preserving $name: live consumer (process reference)"
            preserved_count=$((preserved_count + 1))
            preserved_live_ref_count=$((preserved_live_ref_count + 1))
            continue
        fi

        # Determine the primary worktree to run git worktree remove from.
        # Use awk (not grep|head|cut) to avoid SIGPIPE under set -o pipefail:
        # head -n1 closes the pipe early, which can deliver SIGPIPE (141) to
        # git/grep and propagate a spurious non-zero status via pipefail.
        local primary
        primary="$(git -C "$orphan" worktree list --porcelain 2>/dev/null \
            | awk '/^worktree /{print substr($0,10); exit}')" || {
            exec 8>&-
            warn "  cannot determine primary worktree for $name; skipping"
            preserved_count=$((preserved_count + 1))
            continue
        }

        info "  removing orphan worktree: $name (primary=$primary)"
        if (
            git -C "$primary" worktree remove --force "$orphan"
        ); then
            ok "  removed orphan: $name"
            removed_count=$((removed_count + 1))
            # Orphan lock file is cleaned up on success: once the worktree slot
            # no longer exists, the lock file has no future coordination role.
            # Lane lock files (Pass 1) intentionally persist across sweeps as
            # permanent per-lane mutexes for consumer coordination (see inv.2).
            rm -f "$orphan_lock" 2>/dev/null || true
        else
            warn "  remove failed for $name; continuing"
            preserved_count=$((preserved_count + 1))
        fi
        exec 8>&-  # release orphan lock
    done

    # ── Summary ───────────────────────────────────────────────────────────────
    # preserved_live_ref and preserved_assigned are APPENDED, never interposed:
    # existing consumers match the reset=/removed=/preserved= prefix (and the
    # `preserved=N preserved_live_ref=M` adjacency), so a trailing field extends
    # the line without breaking them.
    printf 'reclaim: reset=%d removed=%d preserved=%d preserved_live_ref=%d preserved_assigned=%d\n' \
        "$reset_count" "$removed_count" "$preserved_count" "$preserved_live_ref_count" \
        "$preserved_assigned_count"
    ok "reclaim complete: reset=$reset_count removed=$removed_count preserved=$preserved_count preserved_live_ref=$preserved_live_ref_count preserved_assigned=$preserved_assigned_count"
}

# ── dispatch ───────────────────────────────────────────────────────────────────
case "$SUBCOMMAND" in
    reclaim)
        _do_reclaim
        ;;
esac
