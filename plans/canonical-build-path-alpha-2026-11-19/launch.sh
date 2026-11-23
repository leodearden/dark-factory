#!/usr/bin/env bash
# launch.sh <ID> <mode>: one run as a low-priority systemd user unit with a clean environment,
# plus its governor.  The unit is canonalpha-<ID>.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ID="$1"; MODE="$2"; UNIT="canonalpha-$ID"
case "$(hostname)" in leo-MS-7C35) RES=/home/leo/canon-alpha-20261119/results ;; *) RES=/media/leo/Internal-2nd/canon-alpha-20261119/results ;; esac
mkdir -p "$RES"
# Run from a snapshot so that editing the harness cannot disturb a running unit (bash reads lazily).
SNAP="$RES/bin-$ID"; rm -rf "$SNAP"; cp -a "$HERE" "$SNAP"; HERE="$SNAP"
systemd-run --user --unit "$UNIT" -p CPUWeight=30 -p IOWeight=30 -p Nice=10 --working-directory=/home/leo \
    /usr/bin/env -i HOME=/home/leo USER=leo LOGNAME=leo XDG_RUNTIME_DIR="/run/user/$(id -u)" \
    DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus" \
    /bin/bash "$HERE/alpha.sh" run "$ID" "$MODE"
setsid nohup bash "$HERE/gov.sh" "$UNIT.service" "$RES/gov-$ID.log" >/dev/null 2>&1 < /dev/null &
echo "launched $UNIT; governor pid $!"
