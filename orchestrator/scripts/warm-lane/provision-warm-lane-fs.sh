#!/usr/bin/env bash
# scripts/provision-warm-lane-fs.sh — Provision the XFS-reflink loopback volume
# used as the warm-lane CoW pool substrate.
#
# Usage:
#   mount=$(scripts/provision-warm-lane-fs.sh [--size-gib N] [--img PATH] [--mount DIR])
#   export REIFY_WARM_LANE_MOUNT=$mount
#
# Stdout:  ONLY the resolved mount directory (bare path, no trailing newline
#          beyond the echo).  Mirrors setup-worktree-debug-port.sh: the stdout
#          value is safe to capture with $(...).
# Stderr:  All diagnostics, progress messages, and errors.
#
# Invariants:
#   P1 (never reformat a populated image): mkfs.xfs fires only when the image
#      is ABSENT, or PRESENT-AND-BYTE-EMPTY with explicit opt-in
#      (REIFY_WARM_LANE_ALLOW_MKFS=1).  This guard keys on file emptiness, not
#      on the blkid probe result, so a failed/indeterminate probe can never
#      defeat it.  An image with XFS magic (blkid TYPE==xfs) is re-attached
#      and remounted, never reformatted.  A currently-mounted image is an
#      idempotent no-op (B1).
#   P2 (probe mandatory, fail-closed): after every mount or mount-verify step a
#      `cp --reflink=always` probe is run inside the mount.  Any failure prints
#      an actionable error to stderr and exits non-zero with NOTHING on stdout.
#      There is no silent cold-copy fallback.
#
# Defaults:
#   --size-gib  4096          (overridable; PRD §9.1 / §13 Q1)
#   --img       /media/leo/data_lv_1/leo/reify-warm-lanes.img
#   --mount     ${REIFY_WARM_LANE_MOUNT:-<worktree_base>/warm-lanes}
#               (<worktree_base> is derived from REPO_ROOT's parent, ascending
#                past a `worktrees/` directory so that the default mount lives
#                next to the worktrees tree, not inside one worktree.)
#
# XFS inode geometry (-i maxpct=50 -i size=512):
#   Reflink shares EXTENTS, not inodes — base + N CoW lanes are each a full
#   independent inode set.  Peak inode demand ≈ 25× per-target (~1.2–1.5M for
#   N=24).  The XFS default imaxpct=25 starved at ~1.09M inodes (97%) on a
#   1000 GiB image before bytes ran low.  imaxpct=50 raises the dynamic inode
#   ceiling from 25% of image blocks to 50%.  isize=512 pins the inode size to
#   the XFS default for deterministic per-GiB headroom calculations.
#
# Privileged operations (fallocate into /var/lib, mkfs, losetup, mount, chown)
# are routed through $SUDO:
#   sudo -n    when EUID != 0 (non-interactive: fails fast instead of
#              TTY-blocking on a password prompt under systemd --user)
#   ''         when root
#   $REIFY_WARM_LANE_SUDO  override (set '' in tests to bypass sudo entirely)

set -euo pipefail

# ── log helpers (all write to stderr) ─────────────────────────────────────────
info()  { printf '\033[1;34m[info]\033[0m  %s\n' "$*" >&2; }
ok()    { printf '\033[1;32m[ok]\033[0m    %s\n' "$*" >&2; }
warn()  { printf '\033[1;33m[warn]\033[0m  %s\n' "$*" >&2; }
err()   { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; }

# ── locate repo root ───────────────────────────────────────────────────────────
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_SCRIPT_DIR/.." && pwd)"

# ── default mount dir: ascend past worktrees/ if present ──────────────────────
_default_mount() {
    local parent
    parent="$(dirname "$REPO_ROOT")"
    # If the repo root is inside a worktrees/ directory, surface one level higher
    # so the warm-lanes dir lives beside the worktrees tree, not inside a worktree.
    if [ "$(basename "$parent")" = "worktrees" ]; then
        parent="$(dirname "$parent")"
    fi
    echo "$parent/warm-lanes"
}

# ── usage ──────────────────────────────────────────────────────────────────────
_usage() {
    cat >&2 <<EOF
Usage: $(basename "$0") [--size-gib N] [--img PATH] [--mount DIR]
       $(basename "$0") --grow --size-gib N [--img PATH] [--mount DIR]

  Provision (or idempotently verify) the XFS-reflink loopback volume used as
  the warm-lane CoW pool substrate.

  Options:
    --size-gib N    Image size in GiB (default: 4096)
    --img PATH      Image file path (default: /media/leo/data_lv_1/leo/reify-warm-lanes.img)
    --mount DIR     Mount point (default: \${REIFY_WARM_LANE_MOUNT:-$(_default_mount)})
    --grow          ONLINE grow the ALREADY-MOUNTED volume to --size-gib N GiB
                     (requires --size-gib explicitly; monotone — idempotent
                     no-op if already exactly N GiB, refuses to shrink; see P3)
    -h, --help      Print this message and exit

  Stdout:  the resolved mount directory (bare path)
  Stderr:  all diagnostics

  Invariants:
    P1 — never reformat a populated image (guard keys on file emptiness, not
         the blkid probe result: mkfs only when absent, or byte-empty with
         REIFY_WARM_LANE_ALLOW_MKFS=1)
    P2 — reflink probe mandatory and fail-closed (cp --reflink=always)
    P3 — --grow is monotone: never shrinks a populated image. Already at N
         GiB is a silent idempotent no-op; a smaller N is refused LOUDLY
         (non-zero exit, no mutation) rather than absorbed.

  \$SUDO override: set REIFY_WARM_LANE_SUDO='' to bypass sudo (for tests).
  REIFY_WARM_LANE_ALLOW_MKFS=1: opt-in to (re)format a PRESENT but
    byte-empty image (default: refuse).
EOF
}

# ── arg parsing ────────────────────────────────────────────────────────────────
SIZE_GIB=4096
IMG="/media/leo/data_lv_1/leo/reify-warm-lanes.img"
MOUNT="${REIFY_WARM_LANE_MOUNT:-$(_default_mount)}"
GROW=0
SIZE_GIB_SET=0   # tracks whether --size-gib was EXPLICITLY passed (--grow requires it)

while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            _usage
            exit 0
            ;;
        --grow)
            GROW=1
            shift
            ;;
        --size-gib)
            [ $# -ge 2 ] || { err "--size-gib requires a value"; exit 2; }
            SIZE_GIB="$2"
            SIZE_GIB_SET=1
            shift 2
            ;;
        --img)
            [ $# -ge 2 ] || { err "--img requires a value"; exit 2; }
            IMG="$2"
            shift 2
            ;;
        --mount)
            [ $# -ge 2 ] || { err "--mount requires a value"; exit 2; }
            MOUNT="$2"
            shift 2
            ;;
        *)
            err "Unknown flag: $1"
            err "Run '$(basename "$0") --help' for usage."
            exit 2
            ;;
    esac
done

# Validate SIZE_GIB is a positive integer (avoid silent interpolation errors in fallocate)
if ! [[ "$SIZE_GIB" =~ ^[0-9]+$ ]]; then
    err "--size-gib requires a positive integer, got: '$SIZE_GIB'"
    exit 2
fi

# Canonicalize IMG to an absolute, symlink-resolved path ONCE, up front, so
# every `losetup -j "$IMG"` lookup below (fresh-provision's _attach_loop AND
# --grow's loop-resolve/backing-file guard) matches the SAME path losetup
# recorded at attach time. A relative or symlinked --img would otherwise
# resolve to an empty LOOP and --grow would wrongly refuse to grow a
# filesystem that is genuinely mounted from that image. Falls back to the
# given value if it can't be resolved yet (e.g. parent dir doesn't exist)
# rather than aborting — the existing fresh-provision path surfaces that
# failure with a clearer error at fallocate/mkfs time.
IMG="$(readlink -f "$IMG" 2>/dev/null || echo "$IMG")"

# ── $SUDO indirection ──────────────────────────────────────────────────────────
# Override: REIFY_WARM_LANE_SUDO (set to '' in tests to bypass sudo entirely)
if [ -n "${REIFY_WARM_LANE_SUDO+x}" ]; then
    SUDO="${REIFY_WARM_LANE_SUDO}"
elif [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo -n"
else
    SUDO=""
fi

# ── Cleanup tracking for partial-provision failure ────────────────────────────
# Track resources allocated in THIS run; the EXIT trap undoes them on non-zero exit.
# Idempotent path (B1) never sets these — only resources we create are cleaned up.
_LOOP_PROVISIONED=""   # loop device attached in this run
_MOUNT_PROVISIONED=""  # mount point mounted in this run

_cleanup_on_exit() {
    local exit_code=$?
    [ $exit_code -eq 0 ] && return
    # Undo partial provisioning: unmount first, then detach loop device
    if [ -n "$_MOUNT_PROVISIONED" ]; then
        warn "Cleaning up failed provision: unmounting $_MOUNT_PROVISIONED"
        $SUDO umount "$_MOUNT_PROVISIONED" 2>/dev/null || true
    fi
    if [ -n "$_LOOP_PROVISIONED" ]; then
        warn "Cleaning up failed provision: detaching loop device $_LOOP_PROVISIONED"
        $SUDO losetup -d "$_LOOP_PROVISIONED" 2>/dev/null || true
    fi
}
trap _cleanup_on_exit EXIT

# ── --grow: online grow of the mounted loopback XFS (task #5173, β) ───────────
# Its own mode, dispatched BEFORE the B1/re-attach/fresh-provision logic below:
# fallocate -l <N>GiB <img>  →  losetup -c <loopdev>  →  xfs_growfs <mount>.
# The cleanup-tracking vars above stay empty on this path, so the EXIT trap
# above never fires a false unmount/detach for a grow run.
if [ "$GROW" -eq 1 ]; then
    # Guard order: requires-size -> mountpoint -> loop-resolve -> backing ->
    # size-measure -> three-way decision.

    # Requires-size: no default grow target is ever baked in (§13 Q2) — the
    # operator must supply the live-measured value explicitly.
    [ "$SIZE_GIB_SET" -eq 1 ] || {
        err "--grow requires an explicit --size-gib <N> (no default grow target)"
        exit 2
    }

    mountpoint -q "$MOUNT" || {
        err "--grow requires an already-mounted filesystem at $MOUNT"
        exit 1
    }

    # Resolve the loop device already backing $IMG (mirrors _attach_loop's
    # `losetup -j` pattern below) — grow never allocates a NEW loop device.
    LOOP="$($SUDO losetup -j "$IMG" -O NAME --noheadings 2>/dev/null | head -1 || true)"

    # Backing-file guard: refuse unless $IMG is confirmed to be the CURRENT
    # backing file for $MOUNT — never grow the wrong image.
    SRC="$(findmnt -n -o SOURCE "$MOUNT" 2>/dev/null || true)"
    if [ -z "$LOOP" ] || [ "$SRC" != "$LOOP" ]; then
        err "$IMG is not the mounted backing file at $MOUNT (resolved loop='$LOOP', mount source='$SRC') — refusing to grow"
        exit 1
    fi

    # Live current size — measured from the IMG file itself, the exact
    # quantity `fallocate -l <N>GiB "$IMG"` mutates (never a frozen constant
    # — P4). Deliberately NOT the mounted filesystem's `df`-reported usable
    # size: XFS log/AG-header/metadata overhead makes df's usable size a few
    # GiB below the raw N-GiB image size, which would leave the exact-equal
    # idempotent branch below practically unreachable after a real grow (a
    # re-run at the same N would keep seeing cur_gib < N via df and re-fire
    # fallocate/losetup -c/xfs_growfs every time).
    cur_bytes="$(stat -c '%s' "$IMG")"
    cur_gib=$((cur_bytes / 1073741824))

    # Three-way monotone decision (P3): smaller refuses LOUD; equal is a
    # silent idempotent no-op; larger grows. Never absorb a shrink request.
    if [ "$SIZE_GIB" -lt "$cur_gib" ]; then
        err "--size-gib $SIZE_GIB is smaller than the current ${cur_gib}GiB at $MOUNT — refusing to shrink (P3: grow is monotone)"
        exit 1
    elif [ "$SIZE_GIB" -eq "$cur_gib" ]; then
        info "$MOUNT is already ${cur_gib}GiB (>= requested ${SIZE_GIB}GiB) — no-op"
        echo "$MOUNT"
        exit 0
    fi

    info "Growing $IMG from ${cur_gib}GiB to ${SIZE_GIB}GiB (online, mounted at $MOUNT) ..."
    $SUDO fallocate -l "${SIZE_GIB}GiB" "$IMG"
    $SUDO losetup -c "$LOOP"
    $SUDO xfs_growfs "$MOUNT"
    ok "Grew warm-lane volume to ${SIZE_GIB}GiB at $MOUNT"

    echo "$MOUNT"
    exit 0
fi

# ── reflink probe (P2) ─────────────────────────────────────────────────────────
# Mandatory on every success path.  Uses cp --reflink=always (NOT auto) so a
# non-reflink mount fails loudly rather than silently falling back to a full copy.
_probe_reflink() {
    local mount_dir="$1"
    local probe_dir="$mount_dir/.reflink-probe-$$"
    mkdir -p "$probe_dir"
    echo "probe" > "$probe_dir/src"
    if ! cp --reflink=always "$probe_dir/src" "$probe_dir/dst"; then
        rm -rf "$probe_dir" 2>/dev/null || true
        err "Reflink probe FAILED at $mount_dir — provisioning aborted."
        err "The filesystem at $mount_dir does not support reflinks."
        err "Refusing to fall back to cold copies (P2 invariant)."
        exit 1
    fi
    rm -rf "$probe_dir" 2>/dev/null || true
}

# ── loop-device attach (reuse existing to avoid double-backing) ───────────────
# Checks `losetup -j` for an existing association before allocating a new device.
# Prevents two loop devices backing the same image when a prior run left one
# attached but unmounted (e.g. manual umount without a matching losetup -d).
_attach_loop() {
    local img="$1"
    local existing
    existing="$($SUDO losetup -j "$img" -O NAME --noheadings 2>/dev/null | head -1 || true)"
    if [ -n "$existing" ]; then
        info "Image $img already attached to $existing — reusing existing loop device"
        echo "$existing"
    else
        $SUDO losetup --find --show "$img"
    fi
}

# ── image type classification (de-privileged probe) ───────────────────────────
# Runs blkid UNPRIVILEGED (no $SUDO) — the image file is expected to be
# world-readable, so an unprivileged read suffices and avoids the footgun
# where a broken/TTY-less $SUDO silently swallows the probe result and gets
# misread as "not xfs" (which would defeat the P1 never-reformat guard).
# No `2>/dev/null || true` swallow: rc and stderr are captured and classified
# rather than discarded, so a probe that could not run is never coerced into
# "unformatted".
#   rc==0            -> _IMG_CLASS=xfs iff stdout=='xfs', else 'other'
#   rc==2             -> _IMG_CLASS=unformatted (blkid: no recognizable signature)
#   any other nonzero -> _IMG_CLASS=indeterminate (probe could not run/complete)
# Permission-denied fallback: blkid(8) returns rc 2 for BOTH "unformatted" and
# "impossible to gather info about the device" (which includes some
# unreadable/permission cases), so rc alone cannot disambiguate them. If the
# unprivileged attempt's stderr indicates a permission failure AND `$SUDO
# true` succeeds (passwordless sudo is actually available), re-probe via
# `$SUDO blkid` and reclassify from ITS rc/stdout instead — an under-sudo
# failure reclassifies as indeterminate (never "unformatted"). Gating on
# `$SUDO true` avoids attempting (and blocking behind) a privileged re-probe
# when sudo is known-broken; when it IS known-broken, _IMG_PERM_DENIED_NO_SUDO
# is set so the caller can tell an operator "fix sudo/readability" apart from
# "this really isn't XFS" instead of silently leaving the earlier guess in place.
# Sets globals: _IMG_CLASS, _IMG_TYPE (raw stdout), _IMG_STDERR, _IMG_RC,
# _IMG_PERM_DENIED_NO_SUDO (1 iff permission-denied AND no working sudo fallback).

# rc==0 classification rule, shared by both the unprivileged and the
# privileged (sudo fallback) probes below so it lives in exactly one place.
_class_from_rc0() {
    if [ "$1" = "xfs" ]; then
        echo "xfs"
    else
        echo "other"
    fi
}

_classify_img() {
    local img="$1"
    local rc=0
    local errfile
    errfile="$(mktemp)"
    local out
    out="$(blkid -o value -s TYPE "$img" 2>"$errfile")" || rc=$?
    _IMG_STDERR="$(cat "$errfile")"
    rm -f "$errfile"
    _IMG_TYPE="$out"
    _IMG_RC="$rc"
    _IMG_PERM_DENIED_NO_SUDO=""
    if [ "$rc" -eq 0 ]; then
        _IMG_CLASS="$(_class_from_rc0 "$out")"
        return
    fi
    if [ "$rc" -eq 2 ]; then
        _IMG_CLASS="unformatted"
    else
        _IMG_CLASS="indeterminate"
    fi

    if printf '%s' "$_IMG_STDERR" | grep -qiE 'permission denied|not permitted'; then
        if $SUDO true 2>/dev/null; then
            local sudo_rc=0
            local sudo_errfile
            sudo_errfile="$(mktemp)"
            local sudo_out
            sudo_out="$($SUDO blkid -o value -s TYPE "$img" 2>"$sudo_errfile")" || sudo_rc=$?
            _IMG_STDERR="$(cat "$sudo_errfile")"
            rm -f "$sudo_errfile"
            _IMG_TYPE="$sudo_out"
            _IMG_RC="$sudo_rc"
            if [ "$sudo_rc" -eq 0 ]; then
                _IMG_CLASS="$(_class_from_rc0 "$sudo_out")"
            else
                _IMG_CLASS="indeterminate"
            fi
        else
            # Passwordless sudo is unavailable, so the permission-denied read
            # could never be disambiguated from a true non-XFS payload — the
            # classification above (unformatted/indeterminate) is a guess,
            # not a confirmed read.
            _IMG_PERM_DENIED_NO_SUDO=1
        fi
    fi
}

# ── main ──────────────────────────────────────────────────────────────────────

info "provision-warm-lane-fs.sh: img=$IMG  mount=$MOUNT  size=${SIZE_GIB}GiB"

# ── B1 / P1: idempotent no-op — already mounted and probe passes ───────────────
if [ -f "$IMG" ] && mountpoint -q "$MOUNT"; then
    info "Image $IMG is already mounted at $MOUNT; verifying reflink probe (P2)..."
    _probe_reflink "$MOUNT"
    ok "Warm-lane volume already provisioned and healthy at $MOUNT"
    echo "$MOUNT"
    exit 0
fi

# ── P1: existing image with XFS magic — re-attach and mount, never reformat ───
if [ -f "$IMG" ]; then
    _classify_img "$IMG"
    if [ "$_IMG_CLASS" = "xfs" ]; then
        info "Image $IMG has XFS magic — re-attaching (P1: never reformat a populated image)..."
        LOOP="$(_attach_loop "$IMG")"
        _LOOP_PROVISIONED="$LOOP"
        info "Attached $IMG to $LOOP"
        mkdir -p "$MOUNT"
        $SUDO mount "$LOOP" "$MOUNT"
        _MOUNT_PROVISIONED="$MOUNT"
        $SUDO chown "$(id -u):$(id -g)" "$MOUNT"
        info "Mounted $LOOP at $MOUNT"
        _probe_reflink "$MOUNT"
        ok "Warm-lane volume re-mounted at $MOUNT"
        echo "$MOUNT"
        exit 0
    fi
    # Image exists but is NOT positively XFS. Permission to reformat is keyed
    # on FILE-EMPTINESS + explicit opt-in — NEVER on the probe result, which
    # is exactly what let a swallowed/misread probe cause the outage: fall
    # through to fresh provision only when the file is byte-empty AND
    # REIFY_WARM_LANE_ALLOW_MKFS=1 is set; a populated image otherwise always
    # refuses, even if the probe is wrong.
    if [ ! -s "$IMG" ] && [ "${REIFY_WARM_LANE_ALLOW_MKFS:-}" = "1" ]; then
        warn "Image $IMG is present but byte-empty and REIFY_WARM_LANE_ALLOW_MKFS=1 — (re)formatting under explicit opt-in"
    else
        if [ "$_IMG_CLASS" = "indeterminate" ]; then
            err "blkid probe could not run (rc=$_IMG_RC); type is INDETERMINATE — NOT assuming unformatted"
        fi
        if [ "${_IMG_PERM_DENIED_NO_SUDO:-}" = "1" ]; then
            err "could not read image to classify (permission denied; passwordless sudo unavailable) — this may NOT be a non-XFS payload; fix sudo/readability for a definitive read"
        fi
        err "Refusing to reformat a populated image (P1); this is NOT a reflink-capability fault; set REIFY_WARM_LANE_ALLOW_MKFS=1 to force"
        exit 1
    fi
fi

# ── Fresh provision ────────────────────────────────────────────────────────────
info "Allocating ${SIZE_GIB} GiB image at $IMG ..."
$SUDO fallocate -l "${SIZE_GIB}GiB" "$IMG"

info "Formatting $IMG as XFS with reflink=1,bigtime=1,imaxpct=50,isize=512 ..."
# -i maxpct=50: raise XFS dynamic inode ceiling from the 25% default to 50%.
#   Reflink shares EXTENTS not inodes — each CoW lane is a full inode set, so
#   peak demand grows with N (≈25× per-target for N=24); the default starved at
#   ~1.09M inodes (97%) on a 1000 GiB image before bytes ran low (#4718).
# -i size=512: pin inode size to the XFS default for deterministic headroom.
$SUDO mkfs.xfs -f -m reflink=1,bigtime=1 -i maxpct=50 -i size=512 "$IMG"

info "Attaching $IMG to loop device ..."
LOOP="$(_attach_loop "$IMG")"
_LOOP_PROVISIONED="$LOOP"
info "Attached to $LOOP"

mkdir -p "$MOUNT"
$SUDO mount "$LOOP" "$MOUNT"
_MOUNT_PROVISIONED="$MOUNT"
$SUDO chown "$(id -u):$(id -g)" "$MOUNT"
info "Mounted $LOOP at $MOUNT"

_probe_reflink "$MOUNT"

ok "Warm-lane volume provisioned at $MOUNT"

# STDOUT contract: ONLY this line goes to stdout.
echo "$MOUNT"
