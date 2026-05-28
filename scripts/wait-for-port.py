#!/usr/bin/env python3
"""Block until all given TCP endpoints accept a connection, or time out.

Usage: wait-for-port.py [--timeout SECONDS] HOST:PORT [HOST:PORT ...]
Exit 0 once every endpoint is reachable; exit 1 on timeout.

Used as a systemd ExecStartPre to order a `systemctl --user` service behind a
dependency the user manager cannot express with After=/Requires=:
  * docker containers (falkordb/qdrant) live under the SYSTEM manager, so a
    user unit cannot order against them -- poll their published ports instead;
  * a sibling user unit (fused-memory) whose FIRST start can lose a boot race
    and fail -- a hard Requires= would permanently cancel us, so we use Wants=
    + this probe, which simply waits for the port to come back.

curl is broken on this host (libcurl.so.4 load failure), so this uses raw
sockets rather than an HTTP client.
"""
import argparse
import socket
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--timeout", type=float, default=280.0,
                   help="overall budget in seconds (keep < TimeoutStartSec)")
    p.add_argument("endpoints", nargs="+", metavar="HOST:PORT")
    args = p.parse_args()

    targets = []
    for e in args.endpoints:
        host, _, port = e.rpartition(":")
        targets.append((host or "127.0.0.1", int(port)))

    deadline = time.monotonic() + args.timeout
    pending = list(targets)
    while pending:
        still = []
        for host, port in pending:
            try:
                with socket.create_connection((host, port), timeout=2):
                    pass
            except OSError:
                still.append((host, port))
        pending = still
        if not pending:
            return 0
        if time.monotonic() >= deadline:
            sys.stderr.write(f"wait-for-port: timed out waiting for {pending}\n")
            return 1
        time.sleep(2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
