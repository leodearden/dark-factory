#!/usr/bin/env bash
# Install an AppArmor profile that allows bwrap to create user namespaces.
#
# On Ubuntu 24.10+ / kernels with apparmor_restrict_unprivileged_userns=1,
# bwrap fails with "setting up uid map: Permission denied" unless it has an
# AppArmor profile granting the 'userns' permission.
#
# Usage:  sudo ./setup-bwrap.sh

set -euo pipefail

PROFILE_PATH="/etc/apparmor.d/bwrap"

if [ "$(id -u)" -ne 0 ]; then
    echo "error: must be run as root (sudo $0)" >&2
    exit 1
fi

# Check if the restriction is even active
restrict=$(cat /proc/sys/kernel/apparmor_restrict_unprivileged_userns 2>/dev/null || echo 0)
if [ "$restrict" = "0" ]; then
    echo "apparmor_restrict_unprivileged_userns is not active — bwrap should work without a profile."
    exit 0
fi

# Check bwrap exists
if ! command -v bwrap &>/dev/null; then
    echo "bwrap not found. Install it: sudo apt install bubblewrap"
    exit 1
fi

cat > "$PROFILE_PATH" <<'PROFILE'
abi <abi/4.0>,
include <tunables/global>

profile bwrap /usr/bin/bwrap flags=(unconfined) {
  userns,
}
PROFILE

apparmor_parser -r "$PROFILE_PATH"
echo "AppArmor profile installed and loaded: $PROFILE_PATH"

# Verify
if bwrap --ro-bind / / --dev /dev --proc /proc -- /bin/true 2>/dev/null; then
    echo "Verification: bwrap sandbox works."
else
    echo "Warning: bwrap still failing after profile install." >&2
    exit 1
fi
