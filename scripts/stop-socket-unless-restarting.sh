#!/bin/sh
# ExecStopPost hook for a socket-activated user service, invoked with %n.
#
# The service's .socket keeps its port bound across restarts. A deliberate
# stop (systemctl stop, disable --now) must close that port too, or the next
# connection would start the service again. Only a deliberate stop leaves a
# `stop` job on the service; a restart shows a `restart` job and a crash
# awaiting auto-restart shows none, and both keep the socket.
set -eu

service="$1"
socket="${service%.service}.socket"

if systemctl --user list-jobs --no-legend --no-pager "$service" \
    | awk -v unit="$service" '$2 == unit && $3 == "stop" { found = 1 } END { exit !found }'; then
  systemctl --user stop --no-block "$socket"
fi
