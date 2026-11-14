#!/usr/bin/env bash
# orchestrator/scripts/warm-lane/lib_lane_seed.sh — one question for
# warm-lane-gc.sh's Pass 1: is this lane's cache unchanged since its last seed
# on the current base generation? Designed to be sourced, not executed.
#
# The single home of the one exception to PRD warm-lane-infra-repatriation
# invariant C-1: the only place dark-factory reads a project's cache dir
# (<lane>/target) and the project-owned seed stamp beside the lane. Every
# absent, unreadable or doubtful reading answers "changed", so the caller
# resets exactly as it did before this lib existed. Rationale, measurements and
# the probe's blind spot: README.md "Delta 14".

if [ "${_DF_LIB_LANE_SEED_SH_SOURCED:-}" = "1" ]; then
    return 0 2>/dev/null || true
fi
_DF_LIB_LANE_SEED_SH_SOURCED=1

# Why the last lane_seed_unchanged call answered as it did; diagnostics only.
LANE_SEED_REASON=''

# _lane_seed_stamp_path LANE — where the project writes LANE's stamp: a SIBLING
# of the lane's realpath (reify scripts/verify-ledger.py::seed_stamp_path).
_lane_seed_stamp_path() {
    local real
    real="$(readlink -f -- "$1")" || return 1
    printf '%s.seed-stamp.json' "$real"
}

# argv: STAMP TARGET CURRENT_GEN. Exit 0 and print why only when the stamp is
# trustworthy, names CURRENT_GEN exactly, and no probed dir under TARGET was
# written at or after the seed's exit; otherwise exit 1 and print why not.
# Single-quoted bash literal, so the program itself uses double quotes only.
_LANE_SEED_PROBE_PY='
import glob, json, math, os, sys, time

PROBES = ("", "*/", "*/deps", "*/.fingerprint", "*/build", "*/incremental",
          "*/*/deps", "*/*/.fingerprint")

def changed(reason):
    print(reason)
    sys.exit(1)

def probed_paths(target):
    for pattern in PROBES:
        if not pattern:
            yield target
        else:
            yield from sorted(glob.glob(os.path.join(target, pattern)))

stamp_path, target, current_gen = sys.argv[1:4]
try:
    with open(stamp_path, encoding="utf-8") as fh:
        stamp = json.load(fh)
    if not isinstance(stamp, dict):
        changed("seed stamp is not a JSON object")
    schema = stamp.get("schema")
    if schema != "reify.lane-seed/1":
        changed(f"seed stamp schema {schema!r} is not reify.lane-seed/1")
    base_gen = stamp.get("base_gen")
    if not isinstance(base_gen, str) or not base_gen:
        changed("seed stamp names no base_gen")
    if not current_gen:
        changed("the current base generation is unresolvable")
    if base_gen != current_gen:
        changed(f"seeded from {base_gen!r}, the base is now {current_gen!r}")
    exited_at = stamp.get("exited_at")
    if (isinstance(exited_at, bool) or not isinstance(exited_at, (int, float))
            or not math.isfinite(exited_at)):
        changed(f"seed stamp exited_at {exited_at!r} is not a finite number")
    if exited_at > time.time():
        changed(f"seed stamp exited_at {exited_at!r} is in the future")
    if not os.path.isdir(target):
        changed("no target/ dir")
    for path in probed_paths(target):
        if os.stat(path).st_mtime >= exited_at:
            written = os.path.relpath(path, os.path.dirname(target))
            changed(f"{written} written since the seed exited")
except Exception as exc:
    changed("seed stamp probe failed: " + " ".join(f"{type(exc).__name__}: {exc}".split()))
print(f"unchanged since its seed from {base_gen} exited at {exited_at:.0f}")
'

# lane_seed_unchanged LANE CURRENT_GEN — rc 0 iff LANE's cache is unchanged
# since its last seed from CURRENT_GEN (the caller's exact `readlink -f` of the
# base symlink). Any other rc means reset. Sets LANE_SEED_REASON, which stays
# EMPTY when there is no stamp: the ordinary reading for every project that
# writes none, so it starts no python3 and asks for no log line.
lane_seed_unchanged() {
    local lane="$1" current_gen="$2" stamp rc=0
    LANE_SEED_REASON=''
    stamp="$(_lane_seed_stamp_path "$lane")" || return 1
    [ -e "$stamp" ] || return 1
    LANE_SEED_REASON="$(python3 -I -c "$_LANE_SEED_PROBE_PY" \
        "$stamp" "$lane/target" "$current_gen" 2>&1)" || rc=$?
    if [ "$rc" -ne 0 ] && [ -z "$LANE_SEED_REASON" ]; then
        LANE_SEED_REASON="seed stamp probe exited $rc"
    fi
    [ "$rc" -eq 0 ]
}
