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
#       ANY live process references (cwd / open fd / mmap, directly or through
#       a bind-mount point in another mount namespace). ONE /proc walk for the
#       WHOLE set.
#   live_ref_present DIR
#       Single-candidate predicate: return 0 iff DIR (or a descendant) is
#       referenced by a live process; 1 otherwise. Short-circuits: the cheaper
#       verdict wins, see below.
#
# Three halves: cwd/fd and mmap read /proc as the host sees it; the namespace
# half (task 10784, README Delta 20) translates references made through a
# bind-mount point in another mount namespace, per
# plans/canonical-build-path-shared-cache-prd.md decision 16.
#
# MEASURED COST (this host, 2026-11-01: ~1.5-1.7k live processes, ~18.9k fd/cwd
# symlinks, loadavg ~90-97 on 32 cores; ranges over back-to-back samples):
#   cwd/fd pass (find+awk over /proc/<pid>/{cwd,fd}) .. 0.42-0.88s — unavoidable
#                                                       for a candidate that turns
#                                                       out NOT to be referenced
#   mmap pass  (grep over /proc/*/maps) ............... 0.04-0.33s
#   one negative live_ref_present (both passes) ....... 0.57-1.37s
# Namespace half (task 10784; 2026-11-18, a sandboxed session that could read
# no other namespace's root, ~1.6k processes, loadavg ~80-110): 0.14s alone
# with no readable foreign namespace (one sample). One negative
# live_ref_present, interleaved samples: 1.00-1.69s before, 1.40-1.81s after.
# A host with readable namespaces binding a candidate pays a find and a maps
# read per such namespace on top; that case was not measured.
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

# _live_refs_link_targets — read NUL-separated /proc/<pid>/cwd and
# /proc/<pid>/fd STARTING POINTS on stdin and print every symlink TARGET they
# hold (%l = the kernel-canonical target — no per-fd fork). A cwd is reported as
# itself (depth 0) and an fd dir by its entries (depth 1), so nothing else under
# /proc is read (task 8089). xargs keeps the pid list out of a single argv (no
# E2BIG fail-open on a huge host), and -r keeps an empty list from running a
# bare find over the current directory; `sh -c 'exec find "$@" ...'` puts the
# paths BEFORE find's expression, where a bare `xargs find` would append them
# after it and find would print nothing.
_live_refs_link_targets() {
    xargs -0 -r sh -c 'exec find "$@" -maxdepth 1 -type l -printf "%l\n"' sh 2>/dev/null || true
}

# _live_refs_match_candidates RP... — read path TARGETS on stdin and print each
# candidate RP that some target equals or is a "RP/" descendant of, each at most
# once, exiting early once every candidate is accounted for. The one definition
# of the boundary-safe match every half feeds. awk loads the candidates as input
# records (the NR==FNR two-file trick — avoids `-v`'s backslash-escape
# processing on data); the string compare keeps arbitrary path characters safe.
_live_refs_match_candidates() {
    awk '
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

# _live_refs_scan_fdcwd RP... — the cwd + open-fd HALF of the scan. Split out of
# live_referenced_paths (task 5572 review) purely so live_ref_present can take a
# verdict from this pass alone and skip the mmap pass on a hit; its producer was
# since narrowed by task 8089 (README Delta 15). Not a public entry point:
# callers want live_referenced_paths or live_ref_present, which run every half.
_live_refs_scan_fdcwd() {
    [ $# -gt 0 ] || return 0
    printf '%s\0' /proc/[0-9]*/cwd /proc/[0-9]*/fd \
        | _live_refs_link_targets | _live_refs_match_candidates "$@" || true
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

# The mount-namespace HALF (task 10784, README Delta 20): rationale in
# plans/canonical-build-path-shared-cache-prd.md decision 16. A process in
# another mount namespace that sees a candidate at a bind-mount point names it
# by that mount point, which the two host halves cannot match. The helpers below
# pass whitespace-free records — pids, "dev:ino" namespace ids, candidate
# INDEXES, and mount points still in mountinfo's \ooo-escaped form — so only
# the final rewrite handles real path text. Every awk here is POSIX (no gawk
# functions, no {n} intervals): the laptop may run mawk.

# The \ooo unescape proc(5) defines for mountinfo, as an awk function.
_LIVE_REFS_AWK_UNESCAPE='
function unescape(s,    out, i, d) {
    out = ""
    while ((i = index(s, "\\")) > 0) {
        d = substr(s, i + 1, 3)
        if (d ~ /^[0-3][0-7][0-7]$/) {
            out = out substr(s, 1, i - 1) sprintf("%c", substr(d, 1, 1) * 64 + substr(d, 2, 1) * 8 + substr(d, 3, 1))
            s = substr(s, i + 4)
        } else {
            out = out substr(s, 1, i)
            s = substr(s, i + 1)
        }
    }
    return out s
}
'

# _live_refs_foreign_pids OWN — "nsid pid" for every pid whose mount namespace
# (stat dev:ino of /proc/<pid>/ns/mnt, the identity namespaces(7) compares) is
# not OWN. One batched stat; the glob lists only /proc.
_live_refs_foreign_pids() {
    printf '%s\0' /proc/[0-9]*/ns/mnt \
        | xargs -0 -r stat -L -c '%d:%i %n' 2>/dev/null \
        | awk -v own="$1" '$1 != own { split($2, p, "/"); print $1, p[3] }' || true
}

# _live_refs_readable_representatives CENSUS — "nsid pid" naming, per foreign
# namespace, the first of its pids whose /proc/<pid>/root can be stat'ed. A
# namespace with none is absent: skipped, never read as a reference.
_live_refs_readable_representatives() {
    local readable
    readable="$(awk '{ print "/proc/" $2 "/root" }' <<<"$1" \
        | xargs -r stat -L -c '%n' 2>/dev/null || true)"
    [ -n "$readable" ] || return 0
    awk 'NR==FNR { split($0, p, "/"); ok[p[3]] = 1; next }
         ($2 in ok) && !($1 in done) { done[$1] = 1; print }' \
        <(printf '%s\n' "$readable") <(printf '%s\n' "$1") 2>/dev/null || true
}

# _live_refs_aliases IDS REPRESENTATIVES — "pid index mount-point" for every
# mount point at which a representative's namespace shows candidate <index>.
# IDS holds "index maj:min dev ino" per candidate. mountinfo field 3 must equal
# the candidate's maj:min before anything is stat'ed, so an unrelated (possibly
# hung) filesystem is never touched; the stat of /proc/<pid>/root<mount point>
# must then equal the candidate's dev and ino. getline, not file operands, so a
# pid that vanished mid-scan costs only its own lines.
_live_refs_aliases() {
    local binds statted
    binds="$(awk 'NR==FNR { dev[$2] = dev[$2] " " $1; next }
        {
            f = "/proc/" $2 "/mountinfo"
            while ((getline line < f) > 0) {
                split(line, m, " ")
                if (m[3] in dev) {
                    k = split(dev[m[3]], idx, " ")
                    for (j = 1; j <= k; j++) print $2, idx[j], m[5]
                }
            }
            close(f)
        }' <(printf '%s\n' "$1") <(printf '%s\n' "$2") 2>/dev/null || true)"
    [ -n "$binds" ] || return 0
    statted="$(LC_ALL=C awk "$_LIVE_REFS_AWK_UNESCAPE"'{ print "/proc/" $1 "/root" unescape($3) }' <<<"$binds" \
        | tr '\n' '\0' | xargs -0 -r stat -L -c '%d %i %n' -- 2>/dev/null || true)"
    [ -n "$statted" ] || return 0
    LC_ALL=C awk "$_LIVE_REFS_AWK_UNESCAPE"'
        FILENAME == ARGV[1] { id[$1] = $3 " " $4; next }
        FILENAME == ARGV[2] { name = $0; sub(/^[^ ]+ [^ ]+ /, "", name); seen[name] = $1 " " $2; next }
        { key = "/proc/" $1 "/root" unescape($3); if ((key in seen) && seen[key] == id[$2]) print }
    ' <(printf '%s\n' "$1") <(printf '%s\n' "$statted") <(printf '%s\n' "$binds") 2>/dev/null || true
}

# _live_refs_namespace_targets — read pids on stdin; print their cwd and open-fd
# targets and their mapped pathnames, one per line, as the host sees them.
_live_refs_namespace_targets() {
    local -a pids
    local pid
    mapfile -t pids
    [ "${#pids[@]}" -gt 0 ] || return 0
    for pid in "${pids[@]}"; do printf '/proc/%s/cwd\0/proc/%s/fd\0' "$pid" "$pid"; done \
        | _live_refs_link_targets
    printf '%s\n' "${pids[@]}" | awk '{
        f = "/proc/" $1 "/maps"
        while ((getline line < f) > 0)
            if (sub(/^[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +[^ ]+ +/, "", line) && substr(line, 1, 1) == "/")
                print line
        close(f)
    }' 2>/dev/null || true
}

# _live_refs_rewrite PATHS ALIASES — read targets on stdin; for each one at or
# under an alias's mount point, print candidate <index> (line <index> of PATHS)
# followed by the rest of the target. ALIASES holds "index mount-point". The
# mount point is compared as ${mp%/}/ so that '/', the lane bound as the
# namespace's root, covers every absolute path. Other targets are dropped: the
# host halves already judged them as they are.
_live_refs_rewrite() {
    LC_ALL=C awk "$_LIVE_REFS_AWK_UNESCAPE"'
        FILENAME == ARGV[1] { cand[FNR] = $0; next }
        FILENAME == ARGV[2] { n++; idx[n] = $1; base[n] = unescape($2); sub(/\/$/, "", base[n]); next }
        $0 != "" {
            for (j = 1; j <= n; j++)
                if ((base[j] != "" && $0 == base[j]) || index($0, base[j] "/") == 1)
                    print cand[idx[j]] substr($0, length(base[j]) + 1)
        }
    ' <(printf '%s\n' "$1") <(printf '%s\n' "$2") - 2>/dev/null || true
}

# _live_refs_scan_namespaces RP... — the mount-namespace HALF. Translates the
# targets of every process in a foreign namespace that binds a candidate back
# to that candidate, then runs them through the shared matcher. Fails open like
# the host halves: an unidentifiable own namespace, or a namespace none of whose
# roots can be stat'ed, adds nothing.
_live_refs_scan_namespaces() {
    [ $# -gt 0 ] || return 0
    local own census representatives aliases line dev rest ino ids="" paths="" n=0
    own="$(stat -L -c '%d:%i' /proc/self/ns/mnt 2>/dev/null || true)"
    [ -n "$own" ] || return 0
    census="$(_live_refs_foreign_pids "$own")"
    [ -n "$census" ] || return 0
    representatives="$(_live_refs_readable_representatives "$census")"
    [ -n "$representatives" ] || return 0

    # maj:min from st_dev with glibc's gnu_dev_major/minor masks: stat's
    # %Hd:%Ld would do it, but only on coreutils new enough to have them.
    while IFS= read -r line; do
        dev="${line%% *}"; rest="${line#* }"; ino="${rest%% *}"
        case "$dev$ino" in ''|*[!0-9]*) continue ;; esac
        n=$((n + 1))
        ids+="$n $(( ((dev & 0xfff00) >> 8) | ((dev >> 32) & 0xfffff000) )):$(( (dev & 0xff) | ((dev >> 12) & 0xffffff00) )) $dev $ino"$'\n'
        paths+="${rest#* }"$'\n'
    done < <(stat -L -c '%d %i %n' -- "$@" 2>/dev/null || true)
    [ "$n" -gt 0 ] || return 0

    aliases="$(_live_refs_aliases "${ids%$'\n'}" "$representatives")"
    [ -n "$aliases" ] || return 0

    local ns rep
    while read -r ns rep; do
        grep -q "^$rep " <<<"$aliases" || continue
        awk -v ns="$ns" '$1 == ns { print $2 }' <<<"$census" \
            | _live_refs_namespace_targets \
            | _live_refs_rewrite "${paths%$'\n'}" "$(awk -v p="$rep" '$1 == p { print $2, $3 }' <<<"$aliases")"
    done <<<"$representatives" | _live_refs_match_candidates "$@" || true
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

    # Every half, unconditionally: a BATCH caller needs the full matched subset,
    # so there is nothing to short-circuit on (unlike live_ref_present, which
    # needs only a yes/no). Duplicates across the passes are possible and
    # always were — every caller keys the output into a set.
    _live_refs_scan_fdcwd "$@"
    _live_refs_scan_maps "$@"
    _live_refs_scan_namespaces "$@"
}

# live_ref_present DIR — single-candidate predicate over the three scan halves.
# Returns 0 iff DIR itself, or anything under it, is referenced by a live
# process (cwd / open fd / mmap, directly or through a bind-mount point in
# another mount namespace); 1 otherwise.
#
# SHORT-CIRCUITS on the cwd/fd pass (task 5572 review): a single yes/no verdict
# does not need the mmap pass once the likelier-to-hit one has already answered
# "yes", so a live candidate skips the mmap pass's cost; the namespace half runs
# only after both host halves miss. A NEGATIVE verdict still pays every half —
# unavoidably, since "not referenced" is only established by exhausting every
# reference kind, and that negative is exactly the verdict that authorises the
# reset.
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
    [ -n "$hits" ] && return 0

    hits="$(_live_refs_scan_namespaces "$rp")" || hits=""
    [ -n "$hits" ]
}
