#!/usr/bin/env bash
# orchestrator/tests/warm-lane/test_provision_warm_lane_fs.sh
# Hermetic tests for orchestrator/scripts/warm-lane/provision-warm-lane-fs.sh.
#
# PATH-stubs fallocate/mkfs.xfs/losetup/mount/umount/mountpoint/blkid/cp/sudo/chown/
# stat/xfs_growfs/findmnt record their argv to a CALLS_FILE; env-driven stub behaviour:
#   REIFY_TEST_REFLINK_OK  — cp stub: "1" -> exit 0; else print error + exit 1
#   REIFY_TEST_MOUNTED     — mountpoint stub: "1" -> exit 0 (mounted); else exit 1
#   REIFY_TEST_IMG_XFS     — blkid stub default-type fallback: "1" -> type "xfs"; else empty
#   REIFY_TEST_BLKID_RC     — blkid stub (unprivileged) exit code (default 0)
#   REIFY_TEST_BLKID_TYPE   — blkid stub (unprivileged) stdout type (default: derived
#                             from REIFY_TEST_IMG_XFS, see above)
#   REIFY_TEST_BLKID_STDERR — blkid stub (unprivileged) stderr text (default: none)
#   REIFY_TEST_BLKID_SUDO_RC / _TYPE / _STDERR — same three knobs for a blkid
#                             invocation running under the sudo stub
#                             (REIFY_TEST_UNDER_SUDO=1), so an under-sudo re-probe
#                             can return a different result than the unprivileged one
#   REIFY_TEST_UNDER_SUDO  — exported "1" by the sudo stub (after stripping a
#                            leading -n) before it exec's, so downstream stubs
#                            (blkid) know they're running under sudo
#   REIFY_TEST_FS_SIZE_GIB — stat stub (--grow live-size measurement): current IMG
#                            file size in GiB, emitted as bytes (`stat -c '%s'`
#                            lookalike; matches what `fallocate -l <N>GiB "$IMG"`
#                            mutates, NOT the mounted filesystem's smaller
#                            df-reported usable size) (default 0)
#   REIFY_TEST_LOOP_FOR_IMG — losetup stub: loop device echoed for `-j` (existing-
#                            attachment lookup); GATED on being set — unset preserves
#                            the default-empty "no existing attachment" behaviour
#   REIFY_TEST_MNT_SOURCE  — findmnt stub: mount SOURCE device echoed (default
#                            /dev/loop99), used by --grow's backing-file guard
#   REIFY_WARM_LANE_SUDO   — set "" in all run_helper calls to bypass sudo
#
# run_helper captures STDOUT and STDERR SEPARATELY:
#   OUT     — captured stdout from the script
#   ERR_OUT — captured stderr from the script
#   RC      — exit code
#
# Blocks:
#   A — CLI guard: --help, unknown flag
#   B — Fresh-provision happy path + size default/override + STDOUT contract
#   C — Probe-fail-loud (boundary B2 / invariant P2): non-reflink mount
#   D — Idempotent no-op (boundary B1 / invariant P1): second-run mounted
#   E — P1 deep: existing populated image (XFS magic), unmounted
#   P — Online grow (boundary B3 / invariants P3/P4): happy path + guards
#
# Driven by orchestrator/tests/test_warm_lane_bash_suite.py (PORTED_TESTS).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=orchestrator/tests/warm-lane/lib_warm_lane_paths.sh
source "$SCRIPT_DIR/lib_warm_lane_paths.sh"
SCRIPT="$WARM_LANE_SCRIPTS_DIR/provision-warm-lane-fs.sh"

[ -f "$SCRIPT_DIR/test_helpers.sh" ] || {
    echo "ERROR: test_helpers.sh not found at $SCRIPT_DIR/test_helpers.sh"
    exit 1
}
# shellcheck source=tests/infra/test_helpers.sh
source "$SCRIPT_DIR/test_helpers.sh"

echo "=== scripts/provision-warm-lane-fs.sh hermetic tests (task 4659) ==="

# ──────────────────────────────────────────────────────────────────────────────
# Shared temp state
# ──────────────────────────────────────────────────────────────────────────────
_TMPDIRS=()
cleanup() {
    for d in "${_TMPDIRS[@]+${_TMPDIRS[@]}}"; do rm -rf "$d"; done
}
trap cleanup EXIT

STUB_DIR="$(mktemp -d /tmp/test-warm-lane-stub-XXXXXX)"
_TMPDIRS+=("$STUB_DIR")

CALLS_FILE="$(mktemp /tmp/test-warm-lane-calls-XXXXXX)"
_TMPDIRS+=("$CALLS_FILE")

ERR_FILE="$(mktemp /tmp/test-warm-lane-err-XXXXXX)"
_TMPDIRS+=("$ERR_FILE")

# ── PATH stubs ─────────────────────────────────────────────────────────────────

# fallocate stub: record argv, exit 0
cat > "$STUB_DIR/fallocate" << 'STUB_EOF'
#!/usr/bin/env bash
echo "fallocate $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
exit 0
STUB_EOF
chmod +x "$STUB_DIR/fallocate"

# mkfs.xfs stub: record argv, exit 0
cat > "$STUB_DIR/mkfs.xfs" << 'STUB_EOF'
#!/usr/bin/env bash
echo "mkfs.xfs $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
exit 0
STUB_EOF
chmod +x "$STUB_DIR/mkfs.xfs"

# losetup stub: record argv; print fake loop device when --show is present.
# -j (existing-attachment lookup, used by _attach_loop AND the --grow branch's
# loop-resolve step): emits $REIFY_TEST_LOOP_FOR_IMG when that env is SET,
# GATED on it being set so the default-empty behaviour blocks E/K/N depend on
# (no existing attachment -> fall through to `--find --show`) is unchanged.
cat > "$STUB_DIR/losetup" << 'STUB_EOF'
#!/usr/bin/env bash
echo "losetup $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
for arg in "$@"; do
    if [ "$arg" = "--show" ]; then
        echo "/dev/loop99"
        exit 0
    fi
done
if [ -n "${REIFY_TEST_LOOP_FOR_IMG:-}" ]; then
    for arg in "$@"; do
        if [ "$arg" = "-j" ]; then
            echo "$REIFY_TEST_LOOP_FOR_IMG"
            exit 0
        fi
    done
fi
exit 0
STUB_EOF
chmod +x "$STUB_DIR/losetup"

# mount stub: record argv, exit 0
cat > "$STUB_DIR/mount" << 'STUB_EOF'
#!/usr/bin/env bash
echo "mount $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
exit 0
STUB_EOF
chmod +x "$STUB_DIR/mount"

# umount stub: record argv, exit 0
cat > "$STUB_DIR/umount" << 'STUB_EOF'
#!/usr/bin/env bash
echo "umount $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
exit 0
STUB_EOF
chmod +x "$STUB_DIR/umount"

# mountpoint stub: exit 0 when REIFY_TEST_MOUNTED=1, else exit 1
cat > "$STUB_DIR/mountpoint" << 'STUB_EOF'
#!/usr/bin/env bash
echo "mountpoint $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
[ "${REIFY_TEST_MOUNTED:-}" = "1" ] && exit 0
exit 1
STUB_EOF
chmod +x "$STUB_DIR/mountpoint"

# blkid stub: two independent knob sets — unprivileged (default) and
# under-sudo (selected via REIFY_TEST_UNDER_SUDO=1, set by the sudo stub
# below) — so a test can express "unpriv blkid says X, sudo blkid says Y".
# Unprivileged type defaults to the legacy REIFY_TEST_IMG_XFS-derived value
# when REIFY_TEST_BLKID_TYPE is unset, preserving byte-behavior for blocks A-J.
cat > "$STUB_DIR/blkid" << 'STUB_EOF'
#!/usr/bin/env bash
echo "blkid $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
if [ "${REIFY_TEST_UNDER_SUDO:-}" = "1" ]; then
    _rc="${REIFY_TEST_BLKID_SUDO_RC:-0}"
    _type="${REIFY_TEST_BLKID_SUDO_TYPE:-}"
    _stderr="${REIFY_TEST_BLKID_SUDO_STDERR:-}"
else
    _rc="${REIFY_TEST_BLKID_RC:-0}"
    if [ -n "${REIFY_TEST_BLKID_TYPE+x}" ]; then
        _type="${REIFY_TEST_BLKID_TYPE}"
    elif [ "${REIFY_TEST_IMG_XFS:-}" = "1" ]; then
        _type="xfs"
    else
        _type=""
    fi
    _stderr="${REIFY_TEST_BLKID_STDERR:-}"
fi
[ -n "$_stderr" ] && echo "$_stderr" >&2
[ -n "$_type" ] && echo "$_type"
exit "$_rc"
STUB_EOF
chmod +x "$STUB_DIR/blkid"

# cp stub: if REIFY_TEST_REFLINK_OK=1 -> exit 0; else print error + exit 1
cat > "$STUB_DIR/cp" << 'STUB_EOF'
#!/usr/bin/env bash
echo "cp $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
if [ "${REIFY_TEST_REFLINK_OK:-}" = "1" ]; then
    exit 0
fi
echo "cp: failed to clone: Operation not supported" >&2
exit 1
STUB_EOF
chmod +x "$STUB_DIR/cp"

# sudo stub: record argv (BEFORE stripping, so "-n" is preserved in
# CALLS_FILE), strip a leading "-n" (bash `exec` rejects it as an unknown
# flag), export REIFY_TEST_UNDER_SUDO=1 so downstream stubs (blkid) can
# select their under-sudo knob set, then passthrough-exec the remaining argv.
cat > "$STUB_DIR/sudo" << 'STUB_EOF'
#!/usr/bin/env bash
echo "sudo $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
export REIFY_TEST_UNDER_SUDO=1
if [ "${1:-}" = "-n" ]; then
    shift
fi
exec "$@"
STUB_EOF
chmod +x "$STUB_DIR/sudo"

# chown stub: record argv, exit 0 (no real ownership change needed in tests)
cat > "$STUB_DIR/chown" << 'STUB_EOF'
#!/usr/bin/env bash
echo "chown $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
exit 0
STUB_EOF
chmod +x "$STUB_DIR/chown"

# stat stub (--grow live-size measurement): record argv, emit the IMG file's
# "current size" in bytes per REIFY_TEST_FS_SIZE_GIB — measures the image
# file itself (the exact quantity `fallocate -l <N>GiB "$IMG"` mutates), NOT
# the mounted filesystem's smaller df-reported usable size (see the script's
# comment at the --grow cur_bytes measurement for why that mismatch matters).
cat > "$STUB_DIR/stat" << 'STUB_EOF'
#!/usr/bin/env bash
echo "stat $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
echo "$(( ${REIFY_TEST_FS_SIZE_GIB:-0} * 1073741824 ))"
exit 0
STUB_EOF
chmod +x "$STUB_DIR/stat"

# xfs_growfs stub (--grow online-grow step): record argv, exit 0
cat > "$STUB_DIR/xfs_growfs" << 'STUB_EOF'
#!/usr/bin/env bash
echo "xfs_growfs $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
exit 0
STUB_EOF
chmod +x "$STUB_DIR/xfs_growfs"

# findmnt stub (--grow backing-file guard): record argv, echo the mount's
# SOURCE device (default /dev/loop99, overridable via REIFY_TEST_MNT_SOURCE)
cat > "$STUB_DIR/findmnt" << 'STUB_EOF'
#!/usr/bin/env bash
echo "findmnt $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
echo "${REIFY_TEST_MNT_SOURCE:-/dev/loop99}"
exit 0
STUB_EOF
chmod +x "$STUB_DIR/findmnt"

# ── run_helper ─────────────────────────────────────────────────────────────────
# Invokes the script under the stub PATH with REIFY_WARM_LANE_SUDO="" by
# default. A caller may override by setting REIFY_WARM_LANE_SUDO before
# invoking run_helper (e.g. `REIFY_WARM_LANE_SUDO='false' run_helper ...`) to
# exercise a forced-fail or real sudo-stub path; existing callers never set
# it, so `${REIFY_WARM_LANE_SUDO-}` defaults to "" and behavior is unchanged.
# Sets OUT (stdout), ERR_OUT (stderr), RC (exit code) as globals.
run_helper() {
    local rc=0
    > "$ERR_FILE"
    OUT="$(
        REIFY_TEST_CALLS_FILE="$CALLS_FILE" \
        REIFY_WARM_LANE_SUDO="${REIFY_WARM_LANE_SUDO-}" \
        PATH="$STUB_DIR:$PATH" \
            bash "$SCRIPT" "$@" 2>"$ERR_FILE"
    )" || rc=$?
    ERR_OUT="$(cat "$ERR_FILE")"
    RC=$rc
}

reset_calls() {
    > "$CALLS_FILE"
}

# ──────────────────────────────────────────────────────────────────────────────
# Block A — CLI guard: --help and unknown flag
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block A: CLI guard ---"

# A1: --help exits 0
reset_calls
run_helper --help
assert "A1: --help exits 0" test "$RC" -eq 0
assert "A1: --help prints 'usage' or 'Usage' on stderr" \
    bash -c 'printf "%s\n" "$1" | grep -qi "usage"' _ "$ERR_OUT"

# A2: unknown flag exits non-zero (2)
reset_calls
run_helper --unknown-flag-xyz
assert "A2: unknown flag exits non-zero" test "$RC" -ne 0


# ──────────────────────────────────────────────────────────────────────────────
# Block B — Fresh-provision happy path + size default/override + STDOUT contract
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block B: fresh provision happy path ---"

B_TMP="$(mktemp -d /tmp/test-warm-lane-b-XXXXXX)"
_TMPDIRS+=("$B_TMP")
B_IMG="$B_TMP/img"
B_MNT="$B_TMP/mnt"
mkdir -p "$B_MNT"

# B1: fresh provision (img absent, not mounted, reflink probe passes) exits 0
reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS="" REIFY_TEST_REFLINK_OK=1 \
    run_helper --img "$B_IMG" --mount "$B_MNT"
assert "B1: fresh provision exits 0" test "$RC" -eq 0

# B2: STDOUT is EXACTLY the mount path (single bare line, nothing else)
assert "B2: STDOUT is exactly the mount path" \
    bash -c '[ "$1" = "$2" ]' _ "$OUT" "$B_MNT"

# B3: fallocate invoked with 4096GiB default size
assert "B3: fallocate invoked with 4096GiB (default size)" \
    bash -c 'grep "^fallocate" "$1" | grep -q "4096GiB"' _ "$CALLS_FILE"

# B4: mkfs.xfs invoked with reflink=1
assert "B4: mkfs.xfs invoked with reflink=1" \
    bash -c 'grep "^mkfs.xfs" "$1" | grep -q "reflink=1"' _ "$CALLS_FILE"

# B5: mkfs.xfs invoked with bigtime=1
assert "B5: mkfs.xfs invoked with bigtime=1" \
    bash -c 'grep "^mkfs.xfs" "$1" | grep -q "bigtime=1"' _ "$CALLS_FILE"

# B6: losetup invoked targeting the img
assert "B6: losetup invoked with --find --show" \
    bash -c 'grep "^losetup" "$1" | grep -q -- "--find"' _ "$CALLS_FILE"

# B7: mount invoked targeting the mount dir
assert "B7: mount invoked targeting mount dir" \
    bash -c 'grep "^mount" "$1" | grep -qF "'"$B_MNT"'"' _ "$CALLS_FILE"

# B8: cp probe invoked with --reflink=always
assert "B8: cp probe invoked with --reflink=always" \
    bash -c 'grep "^cp" "$1" | grep -q -- "--reflink=always"' _ "$CALLS_FILE"

# B9: stderr is non-empty (diagnostics on stderr, not stdout)
assert "B9: stderr is non-empty (diagnostics on stderr)" \
    bash -c '[ -n "$1" ]' _ "$ERR_OUT"

# B10: --size-gib override: re-run with 123, fallocate gets 123GiB
reset_calls
B2_TMP="$(mktemp -d /tmp/test-warm-lane-b2-XXXXXX)"
_TMPDIRS+=("$B2_TMP")
mkdir -p "$B2_TMP/mnt"
REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS="" REIFY_TEST_REFLINK_OK=1 \
    run_helper --img "$B2_TMP/img" --mount "$B2_TMP/mnt" --size-gib 123
assert "B10: --size-gib 123 passes 123GiB to fallocate" \
    bash -c 'grep "^fallocate" "$1" | grep -q "123GiB"' _ "$CALLS_FILE"


# ──────────────────────────────────────────────────────────────────────────────
# Block C — Probe-fail-loud (boundary B2 / invariant P2): non-reflink mount
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block C: probe-fail-loud (P2 invariant) ---"

C_TMP="$(mktemp -d /tmp/test-warm-lane-c-XXXXXX)"
_TMPDIRS+=("$C_TMP")
C_IMG="$C_TMP/img"
C_MNT="$C_TMP/mnt"
mkdir -p "$C_MNT"

# C1: script exits non-zero when cp probe fails
reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS="" REIFY_TEST_REFLINK_OK=0 \
    run_helper --img "$C_IMG" --mount "$C_MNT"
assert "C1: probe failure exits non-zero" test "$RC" -ne 0

# C2: stderr names the reflink failure (actionable message)
assert "C2: stderr names reflink failure (matches /reflink|Operation not supported/i)" \
    bash -c 'printf "%s\n" "$1" | grep -qiE "reflink|Operation not supported"' _ "$ERR_OUT"

# C3: STDOUT is EMPTY (no mount path printed — P2 fail-closed, no silent fallback)
assert "C3: STDOUT is EMPTY on probe failure (P2 invariant)" \
    bash -c '[ -z "$1" ]' _ "$OUT"

# C4: cp --reflink=always probe was recorded (failure came from the probe, not a pre-guard)
assert "C4: cp --reflink=always probe was invoked before failure" \
    bash -c 'grep "^cp" "$1" | grep -q -- "--reflink=always"' _ "$CALLS_FILE"


# ──────────────────────────────────────────────────────────────────────────────
# Block D — Idempotent no-op (boundary B1 / invariant P1): already mounted
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block D: idempotent no-op (B1/P1) ---"

D_TMP="$(mktemp -d /tmp/test-warm-lane-d-XXXXXX)"
_TMPDIRS+=("$D_TMP")
D_IMG="$D_TMP/img"
D_MNT="$D_TMP/mnt"
mkdir -p "$D_MNT"
# Simulate: img exists (second run) and is mounted
touch "$D_IMG"

# D1: idempotent no-op exits 0
reset_calls
REIFY_TEST_MOUNTED=1 REIFY_TEST_IMG_XFS="" REIFY_TEST_REFLINK_OK=1 \
    run_helper --img "$D_IMG" --mount "$D_MNT"
assert "D1: idempotent no-op exits 0" test "$RC" -eq 0

# D2: STDOUT is exactly the mount path
assert "D2: idempotent STDOUT is exactly the mount path" \
    bash -c '[ "$1" = "$2" ]' _ "$OUT" "$D_MNT"

# D3: NO mkfs.xfs (never reformat)
assert "D3: idempotent no-op: NO mkfs.xfs called" \
    bash -c '! grep -q "^mkfs.xfs" "$1"' _ "$CALLS_FILE"

# D4: NO fallocate (no re-allocation)
assert "D4: idempotent no-op: NO fallocate called" \
    bash -c '! grep -q "^fallocate" "$1"' _ "$CALLS_FILE"

# D5: cp --reflink=always probe STILL ran (re-verify even on idempotent path)
assert "D5: idempotent no-op: cp --reflink=always probe still ran" \
    bash -c 'grep "^cp" "$1" | grep -q -- "--reflink=always"' _ "$CALLS_FILE"


# ──────────────────────────────────────────────────────────────────────────────
# Block E — P1 deep: existing populated image (XFS magic), unmounted
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block E: P1 deep — existing XFS image, unmounted ---"

E_TMP="$(mktemp -d /tmp/test-warm-lane-e-XXXXXX)"
_TMPDIRS+=("$E_TMP")
E_IMG="$E_TMP/img"
E_MNT="$E_TMP/mnt"
mkdir -p "$E_MNT"
# Simulate: img exists with XFS magic but is NOT mounted
touch "$E_IMG"

# E1: re-attach+mount existing XFS image exits 0
reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS=1 REIFY_TEST_REFLINK_OK=1 \
    run_helper --img "$E_IMG" --mount "$E_MNT"
assert "E1: re-attach existing XFS image exits 0" test "$RC" -eq 0

# E2: STDOUT is exactly the mount path
assert "E2: STDOUT is exactly the mount path" \
    bash -c '[ "$1" = "$2" ]' _ "$OUT" "$E_MNT"

# E3: NO mkfs.xfs (P1: never reformat a populated image)
assert "E3: P1 — NO mkfs.xfs for existing XFS image" \
    bash -c '! grep -q "^mkfs.xfs" "$1"' _ "$CALLS_FILE"

# E4: NO fallocate (P1: no re-allocation)
assert "E4: P1 — NO fallocate for existing XFS image" \
    bash -c '! grep -q "^fallocate" "$1"' _ "$CALLS_FILE"

# E5: losetup WAS invoked (re-attach the loop device)
assert "E5: losetup was invoked (re-attach existing image)" \
    bash -c 'grep -q "^losetup" "$1"' _ "$CALLS_FILE"

# E6: mount WAS invoked (re-mount the loop device)
assert "E6: mount was invoked (re-mount existing image)" \
    bash -c 'grep "^mount" "$1" | grep -qF "'"$E_MNT"'"' _ "$CALLS_FILE"

# E7: cp --reflink=always probe ran
assert "E7: cp --reflink=always probe ran after re-mount" \
    bash -c 'grep "^cp" "$1" | grep -q -- "--reflink=always"' _ "$CALLS_FILE"


# ──────────────────────────────────────────────────────────────────────────────
# Block G — Non-XFS existing POPULATED image: fail-closed refusal (P1 negative
# case, task #4987). Supersedes the old reprovision-on-non-xfs expectation —
# silently reprovisioning any non-positively-XFS image was the outage-enabling
# behavior (a swallowed/misread probe defeated the old guard). The strengthened
# P1 gate now keys on file-emptiness + explicit opt-in, not on the probe alone.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block G: non-XFS existing POPULATED image → fail-closed refuse ---"

G_TMP="$(mktemp -d /tmp/test-warm-lane-g-XXXXXX)"
_TMPDIRS+=("$G_TMP")
G_IMG="$G_TMP/img"
G_MNT="$G_TMP/mnt"
mkdir -p "$G_MNT"
# Simulate: img file is POPULATED with real bytes but blkid reports no XFS
# magic (REIFY_TEST_IMG_XFS="") — e.g. a live data volume the probe misread.
printf 'populated-not-xfs' > "$G_IMG"

# G1: refuses — exits non-zero (no opt-in given)
reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS="" REIFY_TEST_REFLINK_OK=1 \
    run_helper --img "$G_IMG" --mount "$G_MNT"
assert "G1: non-XFS POPULATED image with no opt-in exits non-zero (fail-closed)" \
    test "$RC" -ne 0

# G2: stderr contains the refusal message
assert "G2: stderr contains 'Refusing to reformat'" \
    bash -c 'printf "%s\n" "$1" | grep -q "Refusing to reformat"' _ "$ERR_OUT"

# G3: stderr names the escape-hatch env var (actionable)
assert "G3: stderr names REIFY_WARM_LANE_ALLOW_MKFS (actionable escape hatch)" \
    bash -c 'printf "%s\n" "$1" | grep -q "REIFY_WARM_LANE_ALLOW_MKFS"' _ "$ERR_OUT"

# G4: NO mkfs.xfs (P1 strengthened: never reformat a populated image)
assert "G4: NO mkfs.xfs invoked (P1 strengthened)" \
    bash -c '! grep -q "^mkfs.xfs" "$1"' _ "$CALLS_FILE"

# G5: NO fallocate (no destructive re-allocation of a populated image)
assert "G5: NO fallocate invoked" \
    bash -c '! grep -q "^fallocate" "$1"' _ "$CALLS_FILE"

# G6: STDOUT is EMPTY (fail-closed, nothing printed on refusal)
assert "G6: STDOUT is EMPTY on refusal" \
    bash -c '[ -z "$1" ]' _ "$OUT"


# ──────────────────────────────────────────────────────────────────────────────
# Block L — INDETERMINATE classification: probe could not run → fail-closed
# (task #4987). A blkid rc that is neither 0 (found) nor 2 (unformatted) means
# the probe could not complete — this must NEVER be coerced into "unformatted"
# or "not xfs"; it must refuse exactly like a confirmed non-xfs image, with
# stderr wording that names the ambiguity distinctly from "unformatted".
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block L: INDETERMINATE probe (rc=4) → fail-closed refuse ---"

L_TMP="$(mktemp -d /tmp/test-warm-lane-l-XXXXXX)"
_TMPDIRS+=("$L_TMP")
L_IMG="$L_TMP/img"
L_MNT="$L_TMP/mnt"
mkdir -p "$L_MNT"
# Simulate: img file is POPULATED; blkid probe returns rc=4 (could not run/
# complete — distinct from rc=2 "unformatted") with empty stdout.
printf 'populated-indeterminate' > "$L_IMG"

reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_BLKID_RC=4 REIFY_TEST_REFLINK_OK=1 \
    run_helper --img "$L_IMG" --mount "$L_MNT"

# L1: refuses — exits non-zero
assert "L1: INDETERMINATE probe (rc=4) exits non-zero (fail-closed)" \
    test "$RC" -ne 0

# L2: stderr contains the refusal message
assert "L2: stderr contains 'Refusing to reformat'" \
    bash -c 'printf "%s\n" "$1" | grep -q "Refusing to reformat"' _ "$ERR_OUT"

# L3: stderr distinctly notes the probe is INDETERMINATE — not silently
# classified as plain "unformatted". The message may still reference the word
# "unformatted" to explain the distinction (e.g. "NOT assuming unformatted"),
# so this checks for the INDETERMINATE classification keyword rather than
# excluding that word outright.
assert "L3: stderr notes probe is INDETERMINATE / could-not-run" \
    bash -c 'printf "%s\n" "$1" | grep -qiE "indeterminate|could not run|could not complete"' _ "$ERR_OUT"

# L4: NO mkfs.xfs (P1 strengthened, indeterminate never treated as "safe to format")
assert "L4: NO mkfs.xfs invoked" \
    bash -c '! grep -q "^mkfs.xfs" "$1"' _ "$CALLS_FILE"

# L5: STDOUT is EMPTY on refusal
assert "L5: STDOUT is EMPTY on refusal" \
    bash -c '[ -z "$1" ]' _ "$OUT"


# ──────────────────────────────────────────────────────────────────────────────
# Block M — byte-empty opt-in escape hatch (task #4987)
# Permission to reformat a present, non-xfs image is `! -s AND explicit
# opt-in`, NOT `! -s` alone: a byte-empty image still refuses without
# REIFY_WARM_LANE_ALLOW_MKFS=1, and reformats only when both conditions hold.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block M: byte-empty + opt-in escape hatch ---"

M_TMP="$(mktemp -d /tmp/test-warm-lane-m-XXXXXX)"
_TMPDIRS+=("$M_TMP")
M_IMG="$M_TMP/img"
M_MNT="$M_TMP/mnt"
mkdir -p "$M_MNT"
# Simulate: img file is present but BYTE-EMPTY (0 bytes) — `! -s` is true.
touch "$M_IMG"

# M-permit: byte-empty + explicit opt-in -> reformat proceeds
reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS="" REIFY_TEST_REFLINK_OK=1 REIFY_WARM_LANE_ALLOW_MKFS=1 \
    run_helper --img "$M_IMG" --mount "$M_MNT"

assert "M1: byte-empty + opt-in exits 0" test "$RC" -eq 0

assert "M2: byte-empty + opt-in invokes mkfs.xfs" \
    bash -c 'grep -q "^mkfs.xfs" "$1"' _ "$CALLS_FILE"

assert "M3: byte-empty + opt-in invokes fallocate" \
    bash -c 'grep -q "^fallocate" "$1"' _ "$CALLS_FILE"

assert "M4: byte-empty + opt-in STDOUT is exactly the mount path" \
    bash -c '[ "$1" = "$2" ]' _ "$OUT" "$M_MNT"

assert "M5: stderr notes byte-empty + explicit opt-in reformat" \
    bash -c '
        printf "%s\n" "$1" | grep -qi "byte-empty" || exit 1
        printf "%s\n" "$1" | grep -q "REIFY_WARM_LANE_ALLOW_MKFS" || exit 1
    ' _ "$ERR_OUT"

# M-refuse: SAME byte-empty image, NO opt-in -> still refuses (permission is
# `!-s AND opt-in`, not `!-s` alone). RED after step-4 alone would actually
# pass this half already (step-4 refuses unconditionally); it is step-5's
# M-permit half that is RED until step-6 adds the escape.
reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS="" REIFY_TEST_REFLINK_OK=1 \
    run_helper --img "$M_IMG" --mount "$M_MNT"

assert "M6: byte-empty WITHOUT opt-in still exits non-zero (fail-closed)" \
    test "$RC" -ne 0

assert "M7: byte-empty WITHOUT opt-in stderr contains 'Refusing to reformat'" \
    bash -c 'printf "%s\n" "$1" | grep -q "Refusing to reformat"' _ "$ERR_OUT"

assert "M8: byte-empty WITHOUT opt-in: NO mkfs.xfs" \
    bash -c '! grep -q "^mkfs.xfs" "$1"' _ "$CALLS_FILE"

assert "M9: byte-empty WITHOUT opt-in: STDOUT is EMPTY" \
    bash -c '[ -z "$1" ]' _ "$OUT"


# ──────────────────────────────────────────────────────────────────────────────
# Block H — mkfs inode-arg contract (task #4718)
# Asserts that mkfs.xfs is called with -i maxpct=50 and -i size=512,
# and that the --size-gib knob is independent of the new inode args.
# Reuses run_helper + CALLS_FILE harness (same form as B4/B5).
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block H: mkfs inode-arg contract (maxpct=50 / size=512) ---"

H_TMP="$(mktemp -d /tmp/test-warm-lane-h-XXXXXX)"
_TMPDIRS+=("$H_TMP")
H_IMG="$H_TMP/img"
H_MNT="$H_TMP/mnt"
mkdir -p "$H_MNT"

# Drive fresh provision (same env as B1: not mounted, no XFS magic, reflink passes)
reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS="" REIFY_TEST_REFLINK_OK=1 \
    run_helper --img "$H_IMG" --mount "$H_MNT"

# H1: mkfs.xfs carries -i maxpct=50 (raised inode cap) [RED on base branch]
assert "H1: mkfs.xfs carries maxpct=50 (raised inode cap)" \
    bash -c 'grep "^mkfs.xfs" "$1" | grep -q "maxpct=50"' _ "$CALLS_FILE"

# H2: mkfs.xfs carries -i size=512 (pinned inode size) [RED on base branch]
assert "H2: mkfs.xfs carries size=512 (pinned inode size)" \
    bash -c 'grep "^mkfs.xfs" "$1" | grep -q "size=512"' _ "$CALLS_FILE"

# H3a: regression — mkfs.xfs still carries reflink=1 (GREEN, no-regression lock)
assert "H3a: mkfs.xfs still carries reflink=1 (no regression)" \
    bash -c 'grep "^mkfs.xfs" "$1" | grep -q "reflink=1"' _ "$CALLS_FILE"

# H3b: regression — mkfs.xfs still carries bigtime=1 (GREEN, no-regression lock)
assert "H3b: mkfs.xfs still carries bigtime=1 (no regression)" \
    bash -c 'grep "^mkfs.xfs" "$1" | grep -q "bigtime=1"' _ "$CALLS_FILE"

# H4: size-knob coexistence — --size-gib 5 flows to fallocate independently of inode args
reset_calls
H4_TMP="$(mktemp -d /tmp/test-warm-lane-h4-XXXXXX)"
_TMPDIRS+=("$H4_TMP")
mkdir -p "$H4_TMP/mnt"
REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS="" REIFY_TEST_REFLINK_OK=1 \
    run_helper --img "$H4_TMP/img" --mount "$H4_TMP/mnt" --size-gib 5

assert "H4a: --size-gib 5 passes 5GiB to fallocate (knob independence)" \
    bash -c 'grep "^fallocate" "$1" | grep -q "5GiB"' _ "$CALLS_FILE"

assert "H4b: mkfs.xfs still carries maxpct=50 when size-gib overridden" \
    bash -c 'grep "^mkfs.xfs" "$1" | grep -q "maxpct=50"' _ "$CALLS_FILE"


# ──────────────────────────────────────────────────────────────────────────────
# Block I — real-geometry proof, root-free, skip-guarded (task #4718)
# Drives a REAL mkfs.xfs through the provisioning script to produce a genuine
# XFS image, then asserts via xfs_info/xfs_db that imaxpct=50 and isize=512.
# Sparse backing file (truncate -s 1G) so mkfs runs in sub-second time.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block I: real-geometry proof (xfs_info imaxpct=50 / isize=512) ---"

if command -v mkfs.xfs >/dev/null 2>&1 && command -v xfs_info >/dev/null 2>&1 && command -v xfs_db >/dev/null 2>&1; then

    # run_helper_realfs: like run_helper but with REAL mkfs.xfs.
    # fallocate stub creates a 1G sparse backing file via truncate (no loop device).
    # All other privileged ops (losetup/mount/chown/cp/mountpoint/blkid/sudo)
    # are stubbed (copied from STUB_DIR). mkfs.xfs is intentionally omitted from
    # the stub dir so the real binary runs and produces genuine XFS geometry.
    run_helper_realfs() {
        local rc=0
        > "$ERR_FILE"
        local realfs_stub_dir
        realfs_stub_dir="$(mktemp -d /tmp/test-warm-lane-realfs-stub-XXXXXX)"
        # fallocate stub: record argv AND create a sparse 1G backing file
        cat > "$realfs_stub_dir/fallocate" << 'REALFS_STUB_EOF'
#!/usr/bin/env bash
echo "fallocate $*" >> "${REIFY_TEST_CALLS_FILE:-/dev/null}"
# Create a sparse backing file at the image path (last positional arg)
img="${*: -1}"
truncate -s 1G "$img"
exit 0
REALFS_STUB_EOF
        chmod +x "$realfs_stub_dir/fallocate"
        # Copy privileged-op stubs; mkfs.xfs is intentionally NOT copied so
        # the real /usr/sbin/mkfs.xfs (or equivalent) runs end-to-end.
        cp "$STUB_DIR/losetup"    "$realfs_stub_dir/losetup"
        cp "$STUB_DIR/mount"      "$realfs_stub_dir/mount"
        cp "$STUB_DIR/chown"      "$realfs_stub_dir/chown"
        cp "$STUB_DIR/cp"         "$realfs_stub_dir/cp"
        cp "$STUB_DIR/mountpoint" "$realfs_stub_dir/mountpoint"
        cp "$STUB_DIR/blkid"      "$realfs_stub_dir/blkid"
        cp "$STUB_DIR/sudo"       "$realfs_stub_dir/sudo"
        OUT="$(
            REIFY_TEST_CALLS_FILE="$CALLS_FILE" \
            REIFY_WARM_LANE_SUDO="" \
            PATH="$realfs_stub_dir:$PATH" \
                bash "$SCRIPT" "$@" 2>"$ERR_FILE"
        )" || rc=$?
        ERR_OUT="$(cat "$ERR_FILE")"
        RC=$rc
        rm -rf "$realfs_stub_dir"
    }

    I_TMP="$(mktemp -d /tmp/test-warm-lane-i-XXXXXX)"
    _TMPDIRS+=("$I_TMP")
    I_IMG="$I_TMP/img"   # must NOT pre-exist (triggers the fresh-provision path)
    I_MNT="$I_TMP/mnt"
    mkdir -p "$I_MNT"

    # Drive fresh provision with real mkfs.xfs on a 1 GiB sparse image
    reset_calls
    REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS="" REIFY_TEST_REFLINK_OK=1 \
        run_helper_realfs --img "$I_IMG" --mount "$I_MNT" --size-gib 1

    # I0: script exits 0 (fresh provision succeeds end-to-end with real mkfs)
    assert "I0: real fresh provision via script exits 0" test "$RC" -eq 0

    # I1: xfs_info reports imaxpct=50 [RED at old default imaxpct=25]
    assert "I1: xfs_info reports imaxpct=50 (raised inode cap)" \
        bash -c 'xfs_info "$1" 2>/dev/null | grep -q "imaxpct=50"' _ "$I_IMG"

    # I2: xfs_info reports isize=512 (pinned to XFS default; self-documenting)
    assert "I2: xfs_info reports isize=512 (pinned inode size)" \
        bash -c 'xfs_info "$1" 2>/dev/null | grep -q "isize=512"' _ "$I_IMG"

    # I3: xfs_db cross-check: imax_pct = 50 [RED at old default imax_pct=25]
    assert "I3: xfs_db cross-check imax_pct = 50" \
        bash -c 'xfs_db -r -c "sb 0" -c "p imax_pct" "$1" 2>/dev/null | grep -q "imax_pct = 50"' _ "$I_IMG"

    # I4: inodes_per_gib = 1073741824 * imaxpct/100 / isize > 600000
    #     Threshold strictly between old imaxpct=25 (→524288) and new imaxpct=50 (→1048576)
    #     so this assertion is RED at the old default and GREEN only at imaxpct=50.
    assert "I4: inodes_per_gib > 600000 (imaxpct=50 headroom, vs 524288 at default 25%)" \
        bash -c '
            xfs_out=$(xfs_info "$1" 2>/dev/null)
            imaxpct=$(printf "%s\n" "$xfs_out" | grep -o "imaxpct=[0-9]*" | head -1 | cut -d= -f2)
            isize=$(printf "%s\n" "$xfs_out" | grep -o "isize=[0-9]*" | head -1 | cut -d= -f2)
            [ -n "$imaxpct" ] && [ -n "$isize" ] || exit 1
            inodes_per_gib=$(( 1073741824 * imaxpct / 100 / isize ))
            [ "$inodes_per_gib" -gt 600000 ]
        ' _ "$I_IMG"

else
    echo "  SKIP: Block I — mkfs.xfs, xfs_info, or xfs_db unavailable"
fi


# ──────────────────────────────────────────────────────────────────────────────
# Block J — default img/size in --help (task #4720)
# Asserts that --help output (on stderr) reflects the new canonical defaults:
#   --img  /media/leo/data_lv_1/leo/reify-warm-lanes.img
#   --size-gib  4096
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block J: default img/size in --help ---"

reset_calls
run_helper --help
assert "J1: --help ERR_OUT contains new default img path" \
    bash -c 'printf "%s\n" "$1" | grep -qF "/media/leo/data_lv_1/leo/reify-warm-lanes.img"' _ "$ERR_OUT"
assert "J2: --help ERR_OUT contains 4096 (new default size)" \
    bash -c 'printf "%s\n" "$1" | grep -q "4096"' _ "$ERR_OUT"


# ──────────────────────────────────────────────────────────────────────────────
# Block K — probe de-privileged / sudo-independent (task #4987)
# The type-probe must run blkid UNPRIVILEGED and must never coerce a blocked
# privileged probe into "not xfs". Forces every $SUDO op to fail outright
# (REIFY_WARM_LANE_SUDO='false') so a lingering `$SUDO blkid` would silently
# vanish under the old `2>/dev/null || true` swallow — proving the probe no
# longer depends on sudo succeeding.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block K: probe de-privileged / sudo-independent ---"

K_TMP="$(mktemp -d /tmp/test-warm-lane-k-XXXXXX)"
_TMPDIRS+=("$K_TMP")
K_IMG="$K_TMP/img"
K_MNT="$K_TMP/mnt"
mkdir -p "$K_MNT"
# Simulate: img exists with XFS magic, NOT mounted, every sudo call fails outright.
touch "$K_IMG"

reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS=1 REIFY_TEST_REFLINK_OK=1 REIFY_WARM_LANE_SUDO="false" \
    run_helper --img "$K_IMG" --mount "$K_MNT"

# K1: a BARE unprivileged 'blkid' call was recorded (never routed through $SUDO)
# RED on base: `$SUDO blkid`="false blkid" never execs the real blkid stub.
assert "K1: blkid probe runs unprivileged (bare call recorded, sudo-independent)" \
    bash -c 'grep -q "^blkid " "$1"' _ "$CALLS_FILE"

# K2: NO mkfs.xfs (P1 — never reformat, holds even mid-failure)
assert "K2: NO mkfs.xfs called (P1 holds even when sudo is broken)" \
    bash -c '! grep -q "^mkfs.xfs" "$1"' _ "$CALLS_FILE"

# K3: stderr shows the re-attach branch (positive xfs classification), NOT the
# reprovision/no-XFS-magic branch (which fires when the probe result is swallowed).
# RED on base: base misreads the swallowed-empty type and prints "reprovisioning".
assert "K3: stderr shows re-attach/never-reformat, NOT reprovision/no-XFS-magic" \
    bash -c '
        printf "%s\n" "$1" | grep -qiE "re-attach|never reformat" || exit 1
        ! printf "%s\n" "$1" | grep -qiE "reprovision|no XFS magic" || exit 1
    ' _ "$ERR_OUT"

# K4: exits non-zero with EMPTY stdout (forced-fail sudo fails the subsequent
# losetup/mount) — the asserted signal is the branch taken + probe
# independence, not a clean exit 0.
assert "K4: exits non-zero with empty STDOUT (forced-fail sudo fails downstream ops)" \
    bash -c 'test "$1" -ne 0 && [ -z "$2" ]' _ "$RC" "$OUT"

# K5 (companion, GREEN throughout): production fstab-recovery path — same XFS
# image but ALREADY MOUNTED — the sudo-free B1 branch succeeds regardless of
# how broken sudo is, documenting that a normally-booted, fstab-mounted image
# provisions successfully even when sudo cannot run at all.
reset_calls
REIFY_TEST_MOUNTED=1 REIFY_TEST_IMG_XFS=1 REIFY_TEST_REFLINK_OK=1 REIFY_WARM_LANE_SUDO="false" \
    run_helper --img "$K_IMG" --mount "$K_MNT"
assert "K5a: already-mounted B1 branch exits 0 even with broken sudo" \
    test "$RC" -eq 0
assert "K5b: already-mounted B1 STDOUT is exactly the mount path" \
    bash -c '[ "$1" = "$2" ]' _ "$OUT" "$K_MNT"
assert "K5c: already-mounted B1: NO mkfs.xfs (sudo-free production path)" \
    bash -c '! grep -q "^mkfs.xfs" "$1"' _ "$CALLS_FILE"


# ──────────────────────────────────────────────────────────────────────────────
# Block N — `sudo -n` threading + unprivileged-first + permission-denied
# fallback (task #4987). Driven with REIFY_WARM_LANE_SUDO='sudo -n' so the
# real (enhanced) sudo stub is exercised end-to-end, rather than bypassed.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block N: sudo -n threading / unpriv-first / perm-denied fallback ---"

N_TMP="$(mktemp -d /tmp/test-warm-lane-n-XXXXXX)"
_TMPDIRS+=("$N_TMP")
N_IMG="$N_TMP/img"
N_MNT="$N_TMP/mnt"
mkdir -p "$N_MNT"
# Simulate: img exists, not mounted — drives the re-attach branch either via
# an immediately-successful unprivileged probe (N1) or via the permission-
# denied -> sudo fallback (N2).
touch "$N_IMG"

# N1: unpriv-first — the unprivileged blkid call succeeds outright, so NO
# privileged re-probe should ever be attempted (no redundant sudo blkid call).
reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS=1 REIFY_TEST_REFLINK_OK=1 REIFY_WARM_LANE_SUDO='sudo -n' \
    run_helper --img "$N_IMG" --mount "$N_MNT"

assert "N1a: unpriv-first re-attach exits 0" test "$RC" -eq 0

assert "N1b: first blkid-matching CALLS_FILE line is a bare unprivileged call" \
    bash -c 'grep -m1 "blkid" "$1" | grep -qE "^blkid "' _ "$CALLS_FILE"

assert "N1c: NO privileged 'sudo ... blkid' fallback call (unpriv already succeeded)" \
    bash -c '! grep -qE "^sudo.*blkid" "$1"' _ "$CALLS_FILE"

assert "N1d: NO mkfs.xfs (re-attach, not reprovision)" \
    bash -c '! grep -q "^mkfs.xfs" "$1"' _ "$CALLS_FILE"

assert "N1e: STDOUT is exactly the mount path" \
    bash -c '[ "$1" = "$2" ]' _ "$OUT" "$N_MNT"

# N3 (part): -n is threaded to the genuinely-privileged ops in this re-attach flow.
assert "N3a: losetup invoked with a 'sudo -n' prefix" \
    bash -c 'grep -qE "^sudo -n losetup" "$1"' _ "$CALLS_FILE"

assert "N3b: mount invoked with a 'sudo -n' prefix" \
    bash -c 'grep -qE "^sudo -n mount" "$1"' _ "$CALLS_FILE"

# N2: permission-denied fallback — the unprivileged probe fails with a
# permission-denied stderr; the privileged re-probe (via $SUDO) succeeds and
# reports xfs, so classification must still land on re-attach, never on
# reprovision.
reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_IMG_XFS="" \
REIFY_TEST_BLKID_RC=2 REIFY_TEST_BLKID_STDERR='blkid: Permission denied' \
REIFY_TEST_BLKID_SUDO_RC=0 REIFY_TEST_BLKID_SUDO_TYPE='xfs' \
REIFY_TEST_REFLINK_OK=1 REIFY_WARM_LANE_SUDO='sudo -n' \
    run_helper --img "$N_IMG" --mount "$N_MNT"

assert "N2a: perm-denied unpriv probe + successful sudo fallback -> exits 0" \
    test "$RC" -eq 0

assert "N2b: STDOUT is exactly the mount path" \
    bash -c '[ "$1" = "$2" ]' _ "$OUT" "$N_MNT"

assert "N2c: bare unprivileged 'blkid' call recorded" \
    bash -c 'grep -qE "^blkid " "$1"' _ "$CALLS_FILE"

assert "N2d: privileged 'sudo ... blkid' fallback call recorded" \
    bash -c 'grep -qE "^sudo.*blkid" "$1"' _ "$CALLS_FILE"

assert "N2e: bare blkid call precedes the sudo blkid fallback call" \
    bash -c '
        bare_line=$(grep -n "^blkid " "$1" | head -1 | cut -d: -f1)
        sudo_line=$(grep -n "^sudo.*blkid" "$1" | head -1 | cut -d: -f1)
        [ -n "$bare_line" ] && [ -n "$sudo_line" ] && [ "$bare_line" -lt "$sudo_line" ]
    ' _ "$CALLS_FILE"

assert "N2f: NO mkfs.xfs (re-attach via fallback classification, not reprovision)" \
    bash -c '! grep -q "^mkfs.xfs" "$1"' _ "$CALLS_FILE"


# ──────────────────────────────────────────────────────────────────────────────
# Block O — permission-denied classification edge cases (task #4987 review):
# pins the crux fail-closed guarantee that Block K only covers at the
# "already has XFS magic" happy path and Block N2 only covers when the sudo
# fallback SUCCEEDS. Here the unprivileged probe hits permission-denied on a
# POPULATED image and either (a) sudo itself is broken, so no privileged
# re-probe is even possible, or (b) sudo works but the privileged re-probe
# itself fails — both must refuse, never reformat, and (b) must reclassify to
# INDETERMINATE rather than "unformatted".
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block O: permission-denied edge cases (broken sudo / failed re-probe) ---"

O_TMP="$(mktemp -d /tmp/test-warm-lane-o-XXXXXX)"
_TMPDIRS+=("$O_TMP")
O_IMG="$O_TMP/img"
O_MNT="$O_TMP/mnt"
mkdir -p "$O_MNT"
# Simulate: img file is POPULATED (real bytes); unpriv blkid hits permission-denied.
printf 'populated-permission-denied' > "$O_IMG"

# O-broken-sudo: unpriv probe permission-denied AND sudo itself is broken
# (sudo true fails) -> no privileged re-probe is even attempted -> must still
# refuse (never coerce an unreadable populated image into "safe to format"
# just because sudo happens to be down). This is the crux outage scenario:
# root-owned/unreadable populated image + broken sudo must never reach mkfs.
reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_BLKID_RC=2 REIFY_TEST_BLKID_STDERR='blkid: Permission denied' \
REIFY_TEST_REFLINK_OK=1 REIFY_WARM_LANE_SUDO='false' \
    run_helper --img "$O_IMG" --mount "$O_MNT"

assert "O1: perm-denied probe + broken sudo (sudo true fails) exits non-zero (fail-closed)" \
    test "$RC" -ne 0

assert "O2: perm-denied probe + broken sudo: stderr contains 'Refusing to reformat'" \
    bash -c 'printf "%s\n" "$1" | grep -q "Refusing to reformat"' _ "$ERR_OUT"

assert "O3: perm-denied probe + broken sudo: NO mkfs.xfs invoked" \
    bash -c '! grep -q "^mkfs.xfs" "$1"' _ "$CALLS_FILE"

assert "O4: perm-denied probe + broken sudo: STDOUT is EMPTY" \
    bash -c '[ -z "$1" ]' _ "$OUT"

# O-failed-reprobe: unpriv probe permission-denied, sudo itself IS available
# (sudo true succeeds via the real sudo stub) but the PRIVILEGED re-probe
# fails outright (rc=4) -> reclassify to indeterminate, NEVER "unformatted".
reset_calls
REIFY_TEST_MOUNTED="" REIFY_TEST_BLKID_RC=2 REIFY_TEST_BLKID_STDERR='blkid: Permission denied' \
REIFY_TEST_BLKID_SUDO_RC=4 \
REIFY_TEST_REFLINK_OK=1 REIFY_WARM_LANE_SUDO='sudo -n' \
    run_helper --img "$O_IMG" --mount "$O_MNT"

assert "O5: perm-denied probe + failed privileged re-probe exits non-zero (fail-closed)" \
    test "$RC" -ne 0

assert "O6: perm-denied probe + failed privileged re-probe: stderr contains 'Refusing to reformat'" \
    bash -c 'printf "%s\n" "$1" | grep -q "Refusing to reformat"' _ "$ERR_OUT"

assert "O7: perm-denied probe + failed privileged re-probe: stderr notes INDETERMINATE (never unformatted)" \
    bash -c 'printf "%s\n" "$1" | grep -qiE "indeterminate|could not run|could not complete"' _ "$ERR_OUT"

assert "O8: perm-denied probe + failed privileged re-probe: NO mkfs.xfs invoked" \
    bash -c '! grep -q "^mkfs.xfs" "$1"' _ "$CALLS_FILE"

assert "O9: perm-denied probe + failed privileged re-probe: STDOUT is EMPTY" \
    bash -c '[ -z "$1" ]' _ "$OUT"


# ──────────────────────────────────────────────────────────────────────────────
# Block P — Online grow (boundary B3): happy path (task #5173, β).
# `--grow --size-gib <N>` performs an ONLINE grow of the mounted loopback XFS:
#   fallocate -l <N>GiB <img>  →  losetup -c <loopdev>  →  xfs_growfs <mount>
# Guard cases (idempotent no-op, no-shrink P3, backing-file mismatch,
# requires-size) are added on top of this happy path in a later iteration.
# ──────────────────────────────────────────────────────────────────────────────
echo ""
echo "--- Block P: online grow (B3) ---"

P_TMP="$(mktemp -d /tmp/test-warm-lane-p-XXXXXX)"
_TMPDIRS+=("$P_TMP")
P_IMG="$P_TMP/img"
P_MNT="$P_TMP/mnt"
mkdir -p "$P_MNT"
# Simulate: img exists and is currently mounted (online grow requires this).
touch "$P_IMG"

# P-happy: mounted 6144 GiB FS; `--grow --size-gib 8192` grows it to 8192 GiB.
reset_calls
REIFY_TEST_MOUNTED=1 REIFY_TEST_LOOP_FOR_IMG=/dev/loop99 REIFY_TEST_MNT_SOURCE=/dev/loop99 \
REIFY_TEST_FS_SIZE_GIB=6144 REIFY_WARM_LANE_SUDO="" \
    run_helper --img "$P_IMG" --mount "$P_MNT" --grow --size-gib 8192

assert "P-happy1: --grow (N > current size) exits 0" test "$RC" -eq 0

assert "P-happy2: STDOUT is exactly the mount path" \
    bash -c '[ "$1" = "$2" ]' _ "$OUT" "$P_MNT"

assert "P-happy3: fallocate invoked with 8192GiB" \
    bash -c 'grep "^fallocate" "$1" | grep -q "8192GiB"' _ "$CALLS_FILE"

assert "P-happy4: losetup -c /dev/loop99 invoked (grow the loop device to match)" \
    bash -c 'grep "^losetup" "$1" | grep -q -- "-c /dev/loop99"' _ "$CALLS_FILE"

assert "P-happy5: xfs_growfs invoked targeting the mount dir" \
    bash -c 'grep "^xfs_growfs" "$1" | grep -qF "'"$P_MNT"'"' _ "$CALLS_FILE"

assert "P-happy6: fallocate precedes losetup -c precedes xfs_growfs" \
    bash -c '
        falloc_line=$(grep -n "^fallocate" "$1" | head -1 | cut -d: -f1)
        losetup_c_line=$(grep -n -- "^losetup.*-c /dev/loop99" "$1" | head -1 | cut -d: -f1)
        growfs_line=$(grep -n "^xfs_growfs" "$1" | head -1 | cut -d: -f1)
        [ -n "$falloc_line" ] && [ -n "$losetup_c_line" ] && [ -n "$growfs_line" ] || exit 1
        [ "$falloc_line" -lt "$losetup_c_line" ] && [ "$losetup_c_line" -lt "$growfs_line" ]
    ' _ "$CALLS_FILE"

# P-idempotent: FS already AT the target size -> silent no-op (exit 0, echo
# mount, but NO mutation) — the exact-equal half of the three-way decision.
reset_calls
REIFY_TEST_MOUNTED=1 REIFY_TEST_LOOP_FOR_IMG=/dev/loop99 REIFY_TEST_MNT_SOURCE=/dev/loop99 \
REIFY_TEST_FS_SIZE_GIB=8192 REIFY_WARM_LANE_SUDO="" \
    run_helper --img "$P_IMG" --mount "$P_MNT" --grow --size-gib 8192

assert "P-idempotent1: --grow (N == current size) exits 0 (pure no-op)" \
    test "$RC" -eq 0

assert "P-idempotent2: STDOUT is exactly the mount path" \
    bash -c '[ "$1" = "$2" ]' _ "$OUT" "$P_MNT"

assert "P-idempotent3: NO fallocate recorded (pure no-op)" \
    bash -c '! grep -q "^fallocate" "$1"' _ "$CALLS_FILE"

assert "P-idempotent4: NO xfs_growfs recorded (pure no-op)" \
    bash -c '! grep -q "^xfs_growfs" "$1"' _ "$CALLS_FILE"

# P-noshrink: N < current size -> refuse LOUDLY (P3 no-shrink/monotone), never
# absorbed as a silent no-op — surfaces an operator's mistyped smaller target.
reset_calls
REIFY_TEST_MOUNTED=1 REIFY_TEST_LOOP_FOR_IMG=/dev/loop99 \
REIFY_TEST_FS_SIZE_GIB=8192 REIFY_WARM_LANE_SUDO="" \
    run_helper --img "$P_IMG" --mount "$P_MNT" --grow --size-gib 4096

assert "P-noshrink1: --grow (N < current size) exits non-zero (P3 no-shrink)" \
    test "$RC" -ne 0

assert "P-noshrink2: STDOUT is EMPTY" \
    bash -c '[ -z "$1" ]' _ "$OUT"

assert "P-noshrink3: stderr names shrink/no-shrink/monotone" \
    bash -c 'printf "%s\n" "$1" | grep -qiE "shrink|monotone"' _ "$ERR_OUT"

assert "P-noshrink4: NO fallocate recorded" \
    bash -c '! grep -q "^fallocate" "$1"' _ "$CALLS_FILE"

# P-backing: img is NOT the mounted backing file (findmnt SOURCE mismatches
# the resolved loop device) -> refuse, never grow the wrong image.
reset_calls
REIFY_TEST_MOUNTED=1 REIFY_TEST_LOOP_FOR_IMG=/dev/loop99 REIFY_TEST_MNT_SOURCE=/dev/loopOTHER \
REIFY_TEST_FS_SIZE_GIB=6144 REIFY_WARM_LANE_SUDO="" \
    run_helper --img "$P_IMG" --mount "$P_MNT" --grow --size-gib 8192

assert "P-backing1: --grow (img not the mounted backing file) exits non-zero" \
    test "$RC" -ne 0

assert "P-backing2: STDOUT is EMPTY" \
    bash -c '[ -z "$1" ]' _ "$OUT"

assert "P-backing3: stderr names the backing-file mismatch" \
    bash -c 'printf "%s\n" "$1" | grep -qiE "backing|not the mounted"' _ "$ERR_OUT"

assert "P-backing4: NO fallocate recorded" \
    bash -c '! grep -q "^fallocate" "$1"' _ "$CALLS_FILE"

# P-requires-size: --grow with no --size-gib -> exit 2, actionable stderr (no
# default grow target is ever baked in — the operator must supply the
# live-measured value, §13 Q2).
reset_calls
REIFY_TEST_MOUNTED=1 REIFY_TEST_LOOP_FOR_IMG=/dev/loop99 REIFY_WARM_LANE_SUDO="" \
    run_helper --img "$P_IMG" --mount "$P_MNT" --grow

assert "P-requires-size1: --grow without --size-gib exits 2" \
    test "$RC" -eq 2

assert "P-requires-size2: stderr names the --size-gib requirement" \
    bash -c 'printf "%s\n" "$1" | grep -q -- "--size-gib"' _ "$ERR_OUT"


test_summary
