#!/usr/bin/env bash
set -euo pipefail

# Graceful restart of fused-memory with optional drain.
#
# Usage:
#   restart-fused-memory.sh            # simple restart (SIGTERM + health wait)
#   restart-fused-memory.sh --drain    # drain recon first, then restart

HEALTH_URL="http://localhost:8002/health"
SERVICE="fused-memory"
DRAIN_TIMEOUT=120
HEALTH_TIMEOUT=30

drain=false
if [[ "${1:-}" == "--drain" ]]; then
    drain=true
fi

# 1. Optional drain: signal harness to stop new cycles, wait for current ones
if $drain; then
    pid=$(systemctl --user show -p MainPID "$SERVICE" | cut -d= -f2)
    if [[ "$pid" -gt 0 ]]; then
        echo "Sending SIGUSR1 to fused-memory (PID $pid) to drain recon..."
        kill -USR1 "$pid"

        # Poll journal for drain completion
        deadline=$((SECONDS + DRAIN_TIMEOUT))
        while [[ $SECONDS -lt $deadline ]]; do
            if journalctl --user -u "$SERVICE" --since "10 seconds ago" --no-pager -q \
                | grep -q "Harness fully drained"; then
                echo "Harness fully drained."
                break
            fi
            sleep 5
        done

        if [[ $SECONDS -ge $deadline ]]; then
            echo "WARNING: Drain timed out after ${DRAIN_TIMEOUT}s, proceeding with restart anyway."
        fi
    else
        echo "Service not running, skipping drain."
    fi
fi

# 2. Restart
echo "Restarting fused-memory..."
systemctl --user restart "$SERVICE"

# 3. Wait for health
echo -n "Waiting for health..."
deadline=$((SECONDS + HEALTH_TIMEOUT))
while [[ $SECONDS -lt $deadline ]]; do
    if curl -sf "$HEALTH_URL" > /dev/null 2>&1; then
        echo " OK"
        echo "fused-memory restarted successfully."
        exit 0
    fi
    echo -n "."
    sleep 1
done

echo " FAILED"
echo "ERROR: fused-memory did not become healthy within ${HEALTH_TIMEOUT}s"
exit 1
