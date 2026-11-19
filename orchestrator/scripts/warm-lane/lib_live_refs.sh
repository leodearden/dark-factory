#!/usr/bin/env bash
# scripts/lib_live_refs.sh — Shared live-reference /proc scanner for the warm-lane pool.
#
# Sourced by (three consumers, one definition — G7 no-lockstep-duplication):
#   scripts/warm-lane-gc.sh        — the ε reclaim PRIMITIVE. Calls
#       live_ref_present per lane (Pass 1) and per orphan (Pass 2), immediately
#       before the reset/remove, while that entry's flock is already held. Every
#       caller of gc.sh therefore inherits the guard — including dark-factory's
#       ε path (git_ops.py _run_warm_lane_gc_reclaim), which invokes
#       `warm-lane-gc.sh reclaim --mount <base>` and never passes
#       --extra-protect-glob (task 5572).
#   scripts/warm-lane-gc-sweep.sh  — the δ periodic backstop. Calls
#       live_referenced_paths ONCE per sweep for the whole batch of dead-PID
#       .reseed-trash candidates (_reap_stale_trash, task 5326).
#   scripts/thin-warm-lane.sh      — release-thin. Calls live_ref_present
#       ONCE per invocation (its T4 gate), under the held T3 <lane>.lock flock
#       and immediately before its `rm -rf <lane>/target` (task 5307,
#       README Delta 17).
#
# FUNCTIONS (defined when sourced):
#   live_referenced_paths RP...
#       Batch predicate: print the subset of the given resolved realpaths that
#       ANY live process references (cwd / open fd / mmap). ONE /proc walk for
#       the WHOLE set.
#   live_ref_present DIR
#       Single-candidate predicate: return 0 iff DIR (or a descendant) is
#       referenced by a live process; 1 otherwise. Short-circuits: the cheaper
#       verdict wins, see below.
#
# MEASURED COST (this host, 2026-11-01: ~1.5-1.7k live processes, ~18.9k fd/cwd
# symlinks, loadavg ~90-97 on 32 cores; ranges over back-to-back samples):
#   cwd/fd pass (find+awk over /proc/<pid>/{cwd,fd}) .. 0.42-0.88s — unavoidable
#                                                       for a candidate that turns
#                                                       out NOT to be referenced
#   mmap pass  (grep over /proc/*/maps) ............... 0.04-0.33s
#   one negative live_ref_present (both passes) ....... 0.57-1.37s
# A cwd-or-fd hit skips the mmap pass. The depth-3 `find /proc` walk the cwd/fd
# pass replaced (task 8089) measured 0.90-1.38s in the same samples.
# The awk early-exit only fires once EVERY candidate is accounted for, so a
# free (about-to-be-reclaimed) candidate always pays the full walk — and that
# is precisely the verdict that must be fresh. See the "cost reversal" note in
# scripts/warm-lane-gc.sh's header for what this means for a whole reclaim pass
# and why the per-entry placement is still the right trade.
#
# Neither function can distinguish "scanned, found nothing" from "scan failed":
# every stage is `|| true`-guarded because find/grep over /proc legitimately
# return non-zero on other users' permission-denied entries and on SIGPIPE from
# awk's early exit, so under `set -o pipefail` the exit status is unreliable
# (see the per-function note below). Callers get the fail-OPEN behaviour the
# sweep has always shipped: a failed scan reads as "no reference". That property
# is heavier now that the same verdict gates `rm -rf <lane>/target` and
# `git worktree remove --force`, so _live_refs_env_probe warns ONCE per process
# when the scan environment looks degraded — see its own note for what it does
# and does not catch, and for why it warns rather than failing CLOSED.

# Source guard — prevent double-sourcing.
if [ "${_REIFY_LIB_LIVE_REFS_SH_SOURCED:-}" = "1" ]; then
    return 0 2>/dev/null || true
fi
_REIFY_LIB_LIVE_REFS_SH_SOURCED=1

# _live_refs_env_probe — one-time, O(1) sanity check that the scan environment
# can actually SEE other processes; warns once on stderr when it cannot.
#
# Why it exists: both scan passes are `|| true`-guarded and the verdict is read
# from OUTPUT, so a wholesale scan failure (procfs not mounted in a container,
# a hidepid= mount option that hides other uids' entries) is indistinguishable
# from "nothing is live" — and that silently-empty verdict now sits in front of
# `rm -rf <lane>/target` and `git worktree remove --force`. The probe cannot fix
# that, but it makes a fail-open scan ATTRIBUTABLE in dark-factory's logs
# instead of invisible.
#
# Warn-only, deliberately NOT fail-closed: treating a degraded scan as "every
# entry is live" would freeze reclaim entirely, and reclaim is the response to
# the ENOSPC the pool is trying to avoid. A visible warning plus an unchanged
# fail-open verdict is the lesser failure.
#
# O(1) by construction — ONE readlink plus one readdir of /proc. The stronger
# probe (run the fd/cwd pass and confirm it saw this process's own cwd) would
# cost a full fd/cwd pass per process (see MEASURED COST), i.e. it would pay the
# very cost the short-circuit in live_ref_present exists to cut. What this catches: /proc
# absent/unmounted, and a hidepid= mount that leaves us seeing only ourselves.
# What it does NOT catch: a partially-unreadable /proc (some entries denied),
# which still degrades the verdict silently.
_REIFY_LIVE_REFS_ENV_PROBED=0
_live_refs_env_probe() {
    [ "${_REIFY_LIVE_REFS_ENV_PROBED:-0}" = "1" ] && return 0
    _REIFY_LIVE_REFS_ENV_PROBED=1

    local self_cwd=""
    self_cwd="$(readlink /proc/self/cwd 2>/dev/null)" || self_cwd=""

    # Glob-only headcount: no per-entry stat, no symlink resolution. Under a
    # hidepid=2 mount this collapses to this process's own entry; on any host
    # healthy enough to be running a reclaim it is in the hundreds.
    local -a proc_dirs=(/proc/[0-9]*)
    local nproc=${#proc_dirs[@]}
    [ -d "${proc_dirs[0]}" ] || nproc=0   # glob-nomatch guard (literal pattern)

    if [ -n "$self_cwd" ] && [ "$nproc" -ge 5 ]; then
        return 0
    fi
    printf '[warn]  live-ref scan DEGRADED: /proc exposes %d process entries and self-cwd %s — the liveness scan fails OPEN, so every candidate reads as "not live" and reclaim will proceed unguarded. Check that /proc is mounted and not hidepid-restricted for this uid.\n' \
        "$nproc" "${self_cwd:+resolves}${self_cwd:-does NOT resolve}" >&2
    return 0
}

# _live_refs_scan_fdcwd RP... — the cwd + open-fd HALF of the scan. Split out of
# live_referenced_paths (task 5572 review) purely so live_ref_present can take a
# verdict from this pass alone and skip the mmap pass on a hit; its producer was
# since narrowed by task 8089 (README Delta 15). Not a public entry point:
# callers want live_referenced_paths or live_ref_present, which run both halves.
_live_refs_scan_fdcwd() {
    [ $# -gt 0 ] || return 0

    # cwd + open-fd references: find prints every process's cwd and open-fd
    # symlink TARGETS (%l = the kernel-canonical target — no per-fd fork). It is
    # handed exactly /proc/<pid>/cwd (reported as itself, depth 0) and
    # /proc/<pid>/fd (its entries, depth 1) as STARTING POINTS, so nothing else
    # under /proc is read (task 8089). xargs keeps the pid list out of a single
    # argv (no E2BIG fail-open on a huge host); `sh -c 'exec find "$@" ...'`
    # puts the paths BEFORE find's expression, where a bare `xargs find` would
    # append them after it and find would print nothing. awk loads the
    # candidate realpaths as input records (the NR==FNR two-file trick — avoids
    # `-v`'s backslash-escape processing on data), then for each target prints
    # any candidate it equals or is a "$cand/" descendant of (string compare, so
    # arbitrary path characters are safe), each candidate at most once, exiting
    # early once every candidate is accounted for.
    { printf '%s\0' /proc/[0-9]*/cwd /proc/[0-9]*/fd \
          | xargs -0 sh -c 'exec find "$@" -maxdepth 1 -type l -printf "%l\n"' sh; } 2>/dev/null \
        | awk '
            NR==FNR { if ($0 != "") cand[++n] = $0; next }
            {
                for (i = 1; i <= n; i++)
                    if (!(i in seen) && ($0 == cand[i] || index($0, cand[i] "/") == 1)) {
                        print cand[i]; seen[i] = 1; nfound++
                    }
                if (n > 0 && nfound >= n) exit
            }
        ' <(printf '%s\n' "$@") - 2>/dev/null || true
}

# _live_refs_scan_maps RP... — the mmap HALF of the scan. Same split rationale
# as _live_refs_scan_fdcwd; logic byte-identical to the inline version it
# replaces. Run SECOND, whichever pass is cheaper on the host (see MEASURED
# COST), because the reference shape this gate exists to catch — a live
# cargo/rustc build inside <lane>/target — always holds a cwd and open fds
# there, so the fd/cwd pass is the one far likelier to short-circuit first.
_live_refs_scan_maps() {
    [ $# -gt 0 ] || return 0

    # mmap'd regions: a single grep over every process's maps for a path mapped
    # from INSIDE any candidate dir. Each pattern carries a trailing "/" so the
    # fixed-string match is path-BOUNDARY-aware: ".../_lane-1/" matches a maps
    # line ".../_lane-1/target/x" but NOT the longer sibling ".../_lane-10/..."
    # whose basename merely has "_lane-1" as a name-prefix. A bare-substring
    # match (the original) spuriously mapped an "_lane-10" build's maps line back
    # to a free "_lane-1" whenever the longer path was not itself a live
    # candidate — e.g. a lane torn down mid-build, whose "(deleted)" mapping
    # lingers — perpetually shielding low-numbered lanes' divergent target/ from
    # reclaim (esc-5378 review). This mirrors the cwd/fd pass's cand"/"
    # descendant test above; a mmap'd file always lives UNDER the dir, so
    # requiring "/" drops no real reference. The trailing "/" is then stripped so
    # each emitted line equals the input candidate realpath EXACTLY — callers
    # key on it (the reaper's `referenced` set). (The reaper's trash inputs
    # already carried a ".<pid>" delimiter, so this only tightens — never
    # loosens — its matching.)
    grep -hoFf <(printf '%s/\n' "$@") /proc/[0-9]*/maps 2>/dev/null \
        | sed 's:/$::' || true
}

# live_referenced_paths RP... — given one or more resolved realpaths (trash
# dirs for the reaper, OR pool-lane / orphan-worktree dirs for the reclaim
# live-consumer gate — task 5378, task 5572), print (one per line,
# deduplicated) the subset that ANY live process references via its cwd
# symlink, an open fd symlink, or a mmap'd region in /proc/<pid>/maps.
# Performs exactly ONE /proc symlink walk and ONE /proc/*/maps grep for the
# WHOLE candidate set — O(/proc) once, NOT once per entry: with N stranded
# entries the older per-entry scan was O(N × /proc) (a single /proc walk
# already cost ~84s on a ~1300-process host, and the ENOSPC strand that
# motivates the reaper can leave many entries at once).
#
# The verdict is read from captured OUTPUT, never a pipeline exit status:
# find/grep over /proc return non-zero on the unavoidable permission-denied
# entries of other users' processes (and SIGPIPE on an early awk exit), so
# under `set -o pipefail` the exit status is an unreliable signal. A vanishing
# /proc entry only produces suppressed stderr; every scan is `|| true`-guarded
# so it never aborts under set -e.
live_referenced_paths() {
    [ $# -gt 0 ] || return 0
    _live_refs_env_probe || true

    # Both halves, unconditionally: a BATCH caller needs the full matched subset,
    # so there is nothing to short-circuit on (unlike live_ref_present, which
    # needs only a yes/no). Duplicates across the two passes are possible and
    # always were — every caller keys the output into a set.
    _live_refs_scan_fdcwd "$@"
    _live_refs_scan_maps "$@"
}

# live_ref_present DIR — single-candidate predicate over the two scan halves.
# Returns 0 iff DIR itself, or anything under it, is referenced by a live
# process (cwd / open fd / mmap); 1 otherwise.
#
# SHORT-CIRCUITS on the cwd/fd pass (task 5572 review): a single yes/no verdict
# does not need the mmap pass once the likelier-to-hit one has already answered
# "yes", so a live candidate skips the mmap pass's cost. A NEGATIVE
# verdict still pays both passes — unavoidably, since "not referenced" is only
# established by exhausting every reference kind, and that negative is exactly
# the verdict that authorises the reset.
#
# DIR is resolved with `readlink -f` FIRST, and that is mandatory rather than
# defensive: /proc cwd/fd symlink targets and /proc/*/maps paths are
# kernel-canonical, so matching a literal "$WORKTREES_DIR/<name>" would
# silently never match under a symlinked mount — a guard that quietly never
# fires, i.e. the same class of failure the per-lane gate exists to remove.
#
# An unresolvable DIR returns 1 (not live): the path cannot be matched against
# kernel-canonical /proc targets at all, so there is nothing to compare. This
# mirrors the batch callers' per-entry `[ -n "$rp" ] || continue` skip.
live_ref_present() {
    local dir="${1:-}"
    [ -n "$dir" ] || return 1

    # Probed BEFORE the realpath resolution below: a /proc that cannot be read
    # also breaks `readlink -f`, and that early `return 1` is itself a fail-open
    # verdict, so the warning has to precede it or the worst case stays silent.
    _live_refs_env_probe || true

    local rp
    rp="$(readlink -f "$dir" 2>/dev/null)" || rp=""
    [ -n "$rp" ] || return 1

    local hits
    hits="$(_live_refs_scan_fdcwd "$rp")" || hits=""
    [ -n "$hits" ] && return 0

    hits="$(_live_refs_scan_maps "$rp")" || hits=""
    [ -n "$hits" ]
}
