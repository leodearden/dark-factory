#!/usr/bin/env bash
set -euo pipefail

# Restart the dark-factory-dashboard.service (leaf service — no drain needed).
#
# Usage:
#   restart-dashboard.sh          # restart and poll for readiness
#
# Any positional arguments (e.g. a stray --drain passed by a generic caller)
# are silently ignored — the dashboard has no recon layer to drain.
#
# Fire-and-forget detachment is handled by the coordinator's default executor
# (start_new_session=True), so this script must not background itself.

HEALTH_URL="http://127.0.0.1:8080/healthz"
SERVICE="dark-factory-dashboard.service"
HEALTH_TIMEOUT=30

# 1. Restart
echo "Restarting ${SERVICE}..."
systemctl --user restart "${SERVICE}"

# 2. Best-effort readiness poll — non-fatal on timeout (leaf service)
echo -n "Waiting for readiness..."
deadline=$((SECONDS + HEALTH_TIMEOUT))
while [[ $SECONDS -lt $deadline ]]; do
    if curl -sf "${HEALTH_URL}" > /dev/null 2>&1; then
        echo " OK"
        echo "dark-factory-dashboard restarted successfully."
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo " timed out"
echo "WARNING: dashboard did not respond within ${HEALTH_TIMEOUT}s (non-fatal — leaf service)."
exit 0
