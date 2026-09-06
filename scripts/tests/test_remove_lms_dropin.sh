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
# WHO RUNS THIS (task 4200).  Until 4200 nothing did: this was the only .sh
# among 61 pytest modules in scripts/tests/, and pytest collects only
# `test_*.py`, so the file sat in the tree checked by nobody.  It is now driven
# by scripts/tests/test_remove_lms_dropin_wrapper.py, which IS collected by the
# default suite (`pytest ... scripts/tests/`) and so runs on every verify.
# It remains directly runnable by hand: `scripts/tests/test_remove_lms_dropin.sh`.
#
# LMS_SELFTEST_TEMPLATE is a TEST SEAM, not a tuning knob, and it is
# LOAD-BEARING rather than cosmetic: a run against the DEFAULT template is NOT
# SAFE TO EXECUTE CONCURRENTLY.  Every instance would write and `rm` the same
# absolute unit path under the one shared ~/.config/systemd/user, and case 4's
# `rm -f "$UNIT"` would tear down a sibling run's fixture mid-test.  Measured
# under task 4200: two concurrent default-template runs BOTH fail
# deterministically ("drop-in still present after refusal", "template survives
# the re-run", and a WorkingDirectory resolving to the OTHER run's mktemp dir).
# The fleet runs max_concurrent_tasks: 48 against one $HOME, so the wrapper
# passes a unique per-invocation name through this seam.  Three concurrent runs
# with distinct templates all pass 12/12.
#
# COST, and why the wrapper ALSO serializes.  Distinct names fix the collision
# above; they do nothing about CONTENTION on the one shared `systemd --user`
# manager.  MEASURED on the operator host: `systemctl --user daemon-reload` is
# globally serialized at 0.85-0.94s against 66 unit files, this file performs 3
# of them and the script under test 2 more, and a solo run costs 5.20s -- so a
# run is essentially all daemon-reload and its wall clock scales LINEARLY in
# the number of concurrent runs, unique names or not.  The wrapper therefore
# holds an flock over ~/.config/systemd/user/.lms-dropin-selftest.lock for the
# whole cycle, so at most one process on the host drives this file at a time.
# Running it BY HAND does not take that lock; expect to slow (not break) a
# concurrent verify if you do.
#
# Usage: scripts/tests/test_remove_lms_dropin.sh
#        LMS_SELFTEST_TEMPLATE='lms-dropin-selftest-<unique>@' scripts/tests/test_remove_lms_dropin.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/scripts/remove-lms-arm-worktree-dropin.sh"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

TEMPLATE="${LMS_SELFTEST_TEMPLATE:-lms-dropin-selftest@}"
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
