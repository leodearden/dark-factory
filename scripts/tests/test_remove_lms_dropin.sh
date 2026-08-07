#!/usr/bin/env bash
# End-to-end self-test for scripts/remove-lms-arm-worktree-dropin.sh.
#
# Task 3750.  That script runs UNSANDBOXED and UNATTENDED as a deterministic
# `before_done` deploy action, so its success path must be exercised somewhere
# other than production.  This installs a THROWAWAY systemd user template
# (lms-dropin-selftest@) plus a drop-in, drives the real script against it via
# the LMS_* test seams, and asserts both the refusal and the removal paths
# against the real systemd user manager.
#
# It never touches lms-arm@ or any real unit, and cleans up on every exit path.
#
# Usage: scripts/tests/test_remove_lms_dropin.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/remove-lms-arm-worktree-dropin.sh"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

TEMPLATE='lms-dropin-selftest@'
UNIT="$UNIT_DIR/${TEMPLATE}.service"
DROPIN_DIR="$UNIT_DIR/${TEMPLATE}.service.d"
DROPIN="$DROPIN_DIR/10-worktree-3713.conf"

# A directory that genuinely contains scripts/local-model-serving/lms_serve.py,
# so the script's P1/P3 preconditions can pass without task 3713 being merged.
FAKE_REPO="$(mktemp -d)"
mkdir -p "$FAKE_REPO/scripts/local-model-serving"
printf 'import sys\n\nif __name__ == "__main__":\n    sys.exit(0)\n' \
    > "$FAKE_REPO/scripts/local-model-serving/lms_serve.py"

fails=0
check() {  # check <label> <expected> <actual>
    if [ "$2" = "$3" ]; then
        echo "  PASS: $1"
    else
        echo "  FAIL: $1 -- expected '$2', got '$3'" >&2
        fails=$((fails + 1))
    fi
}

cleanup() {
    rm -f "$DROPIN" "$UNIT" 2>/dev/null || true
    rmdir "$DROPIN_DIR" 2>/dev/null || true
    rm -rf "$FAKE_REPO" 2>/dev/null || true
    systemctl --user daemon-reload 2>/dev/null || true
}
trap cleanup EXIT

install_fixture() {
    mkdir -p "$DROPIN_DIR"
    cat > "$UNIT" <<EOF
[Unit]
Description=Throwaway self-test template for task 3750 (never started)
[Service]
Type=oneshot
WorkingDirectory=$FAKE_REPO
ExecStart=/bin/true
EOF
    cat > "$DROPIN" <<EOF
[Service]
WorkingDirectory=/nonexistent/selftest-worktree
EOF
    systemctl --user daemon-reload
}

echo "== fixture: throwaway template + drop-in =="
install_fixture
wd="$(systemctl --user show -p WorkingDirectory --value "${TEMPLATE}probe.service")"
check "drop-in is in effect before the run" "/nonexistent/selftest-worktree" "$wd"

echo "== case 1: REFUSES when the launcher is absent from the repo root =="
set +e
LMS_UNIT_TEMPLATE="$TEMPLATE" LMS_REPO_ROOT="/nonexistent/not-a-repo" \
    "$SCRIPT" >/dev/null 2>&1
rc=$?
set -e
check "exit code is 1" "1" "$rc"
check "drop-in still present after refusal" "yes" "$([ -e "$DROPIN" ] && echo yes || echo no)"

echo "== case 2: REMOVES and verifies when the launcher is present =="
set +e
LMS_UNIT_TEMPLATE="$TEMPLATE" LMS_REPO_ROOT="$FAKE_REPO" "$SCRIPT" >/dev/null 2>&1
rc=$?
set -e
check "exit code is 0" "0" "$rc"
check "drop-in removed" "no" "$([ -e "$DROPIN" ] && echo yes || echo no)"
check "drop-in dir removed" "no" "$([ -d "$DROPIN_DIR" ] && echo yes || echo no)"
check "template still installed" "yes" "$([ -f "$UNIT" ] && echo yes || echo no)"
wd="$(systemctl --user show -p WorkingDirectory --value "${TEMPLATE}probe.service")"
check "WorkingDirectory now resolves to the repo root" "$FAKE_REPO" "$wd"

echo "== case 3: idempotent re-run is a clean no-op =="
set +e
LMS_UNIT_TEMPLATE="$TEMPLATE" LMS_REPO_ROOT="$FAKE_REPO" "$SCRIPT" >/dev/null 2>&1
rc=$?
set -e
check "exit code is 0 on re-run" "0" "$rc"
check "template survives the re-run" "yes" "$([ -f "$UNIT" ] && echo yes || echo no)"

echo "== case 4: REFUSES when the template is missing =="
rm -f "$UNIT"
systemctl --user daemon-reload
set +e
LMS_UNIT_TEMPLATE="$TEMPLATE" LMS_REPO_ROOT="$FAKE_REPO" "$SCRIPT" >/dev/null 2>&1
rc=$?
set -e
check "exit code is 1 when template absent" "1" "$rc"

echo
if [ "$fails" -eq 0 ]; then
    echo "test_remove_lms_dropin: ALL CHECKS PASSED"
else
    echo "test_remove_lms_dropin: $fails CHECK(S) FAILED" >&2
    exit 1
fi
