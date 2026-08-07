#!/usr/bin/env bash
set -euo pipefail

# Restart EVERY running orchestrator-*.service user unit and verify a fresh
# ActiveEnterTimestampMonotonic for each.
#
# All orchestrator units run the same orchestrator package from this repo, so
# a change to orchestrator/src must be deployed by restarting all of them —
# not only orchestrator-dark-factory (see restart-orchestrator.sh for the
# single-unit variant this generalizes).
#
# INTENDED CALLER: an operator shell, or a task_kind='deterministic' deploy
# task's `before_done.script` with target_unit='orchestrator-dark-factory.service'.
# Because the unit list includes the dark-factory orchestrator itself, the
# deploy task MUST set target_unit to the dark-factory unit so the
# DeterministicRunner routes execution through its cgroup-escaping detached
# `systemd-run --user` path (deterministic_runner.py,
# _default_schedule_detached_restart; done = 'scheduled'). A BLOCKING
# invocation from inside any orchestrator's own cgroup would be killed
# mid-script under KillMode=control-group — same constraint documented in
# restart-orchestrator.sh, extended here to every unit in the list.
#
# Defensive ordering: the invoking orchestrator's own unit (default
# orchestrator-dark-factory.service, override via SELF_UNIT) is restarted
# LAST, so if this script is ever mistakenly run blocking from inside that
# unit, every other orchestrator still gets restarted before the caller dies.
#
# Usage:
#   restart-all-orchestrators.sh [--drain]
#
# --drain enables the per-unit merge-drain gate (task 2397, γ of the
# orchestrator fleet-redeploy PRD): before restarting each unit, its
# α-produced (task 2395) merge-idle heartbeat
# (orchestrator/src/orchestrator/fleet_heartbeat.py) is read via
# scripts/drain_check.py, classified into idle/busy/stale/absent. Two
# deliberately-opposite fail directions:
#   - busy (fresh, mid-merge): restart is DEFERRED — rechecked every
#     ORCH_DRAIN_POLL_INTERVAL_SECS — until the unit drains (goes idle) or
#     ORCH_RESTART_FORCE_FIRE_AFTER_SECS elapses, at which point the restart
#     is FORCED anyway (one re-verified merge accepted; recover_pending_merges
#     makes this crash-safe). Fails toward PROTECTING the merge. If a
#     deferred unit's heartbeat itself goes stale/absent mid-wait (e.g. it
#     crashed while mid-merge and stopped heartbeating), it is NOT held for
#     the rest of the busy grace — a dead unit isn't merging, so it drops
#     into the shorter stale/absent handling below instead.
#   - stale/absent (heartbeat missing, unreadable, or too old): given a
#     shorter ORCH_DRAIN_UNKNOWN_GRACE_SECS grace to start reporting,
#     rechecked every ORCH_DRAIN_POLL_INTERVAL_SECS; a fresh idle/busy
#     reading that appears during the grace re-classifies into the idle/busy
#     handling above, and if the heartbeat is still stale/absent once the
#     grace elapses, the restart proceeds anyway. Fails toward CONVERGENCE
#     (the opposite direction from busy — a not-reporting unit must not block
#     the fleet restart forever).
# Without --drain, the script behaves exactly as before: an immediate,
# uncapped restart-all with zero heartbeat reads.
#
# Env knobs (all optional, ${VAR:-default} style):
#   ORCH_FLEET_DIR                       fleet-common heartbeat dir
#                                         (default: /home/leo/src/dark-factory/data/fleet)
#   ORCH_RESTART_FORCE_FIRE_AFTER_SECS    busy-grace before a forced restart
#                                         (default: 4500 = 75m)
#   ORCH_DRAIN_FRESH_WINDOW_SECS          heartbeat freshness window
#                                         (default: 120 = 2x run-loop tick)
#   ORCH_DRAIN_POLL_INTERVAL_SECS         re-check interval while deferring
#                                         (busy) or awaiting a stale/absent
#                                         heartbeat (default: 30)
#   ORCH_DRAIN_UNKNOWN_GRACE_SECS         grace for a stale/absent heartbeat
#                                         before proceeding anyway (default: 120)

FIELDS="MainPID,ActiveState,ActiveEnterTimestamp,ActiveEnterTimestampMonotonic"
VERIFY_TIMEOUT="${RESTART_VERIFY_TIMEOUT:-30}"
# Grace re-probe window (task 2961): the fleet's slowest-draining unit
# (long merge-verifies, warm-lane drop-in -- reify in practice) can have its
# restart job superseded/canceled and re-run by systemd's own supervision
# AFTER VERIFY_TIMEOUT already expired, which previously made
# restart_and_verify() declare the unit FAILED (and the caller escalate
# critical) seconds before the unit actually came up fresh. Give a unit that
# is still not fresh at the VERIFY_TIMEOUT deadline this many additional
# seconds to re-probe into before genuinely declaring it failed.
#
# Serial-latency tradeoff (amend, task 2961): restart_and_verify() runs once
# per unit in series, so a genuinely-dead unit now costs VERIFY_TIMEOUT +
# VERIFY_GRACE (default 30s + 120s) to surface as a failure, and a whole-fleet
# outage (e.g. a host reboot with every unit down) adds ~VERIFY_GRACE per dead
# unit before the caller can escalate. Confirmed acceptable against the
# fleet-deploy cadence, so the 120s default is deliberately kept:
#   1. The escalating callers (deploy task before_done.script, merge-landed
#      coordinator, watchdog backstop) fire this script DETACHED via
#      `systemd-run --user ... --collect` (service_restart.py) -- a Type=simple
#      transient unit with no runtime timeout -- so the extra grace only DELAYS
#      a genuine FAILED verdict; it never manufactures a new kill/timeout.
#   2. This same chokepoint's --drain gate already tolerates
#      ORCH_RESTART_FORCE_FIRE_AFTER_SECS (default 4500s = 75m) of deferral PER
#      busy unit, so the caller's runtime budget already dwarfs an added
#      ~120s/unit for the (rarer) dead-unit case.
#   3. The fleet redeploy re-fires at most once per
#      orchestrator_restart_min_interval_secs (default 28800s = 8h), so even a
#      ~7-unit outage's ~14m of added serial latency is <4% of one interval and
#      only postpones an escalation that is not time-critical (a dead fleet
#      stays dead until repaired regardless).
# A cumulative per-run grace cap was considered and deliberately NOT added: it
# could starve a legitimately slow-draining unit restarted late in the run of
# its full grace, reintroducing the exact false-FAILED escalation this fixes.
VERIFY_GRACE="${RESTART_VERIFY_GRACE_SECS:-120}"
SELF_UNIT="${SELF_UNIT:-orchestrator-dark-factory.service}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
FLEET_DIR="${ORCH_FLEET_DIR:-/home/leo/src/dark-factory/data/fleet}"
# Shared fleet-deploy clock (task 2396, fleet-redeploy β): this script is the
# SOLE on-disk writer, stamped only once every unit has verified fresh below.
# Mirrors the coordinator's {ts, iso} state_path schema (service_restart.py
# FLEET_DEPLOY_CLOCK_RELPATH / _persist_last_fire_wall) so both the
# coordinator and scripts/orchestrator-watchdog.py read what this writes.
CLOCK_FILE="${ORCH_FLEET_DEPLOY_CLOCK:-$REPO_DIR/data/orchestrator/last_redeploy_orchestrator.json}"
FORCE_FIRE_AFTER_SECS="${ORCH_RESTART_FORCE_FIRE_AFTER_SECS:-4500}"
DRAIN_FRESH_WINDOW_SECS="${ORCH_DRAIN_FRESH_WINDOW_SECS:-120}"
DRAIN_POLL_INTERVAL_SECS="${ORCH_DRAIN_POLL_INTERVAL_SECS:-30}"
DRAIN_UNKNOWN_GRACE_SECS="${ORCH_DRAIN_UNKNOWN_GRACE_SECS:-120}"
# Deliberate NON-stdout return channel for drain_await_fresh (task 3852):
# that function's poll loop must run in the caller's own shell, not inside
# a forked command-substitution subshell, or a SIGKILL of the top-level
# script pid leaves the subshell running and orphaned (reparented to
# systemd --user, PPID 2036) for up to DRAIN_UNKNOWN_GRACE_SECS more.
# drain_await_fresh writes its verdict here instead of printing it to
# stdout; callers MUST invoke it as a plain command (never via "$(...)")
# and then read _DRAIN_VERDICT. NEVER declare this `local` anywhere --
# doing so would silently break the channel for whichever caller shadowed
# it.
_DRAIN_VERDICT=""

DRAIN_ENABLED=0
for arg in "$@"; do
    case "$arg" in
        --drain)
            DRAIN_ENABLED=1
            ;;
        *)
            echo "ERROR: unexpected argument: $arg" >&2
            exit 1
            ;;
    esac
done

read_field() {
    # $1 = `systemctl show` output blob, $2 = field name
    printf '%s\n' "$1" | grep "^$2=" | cut -d= -f2-
}

get_state() {
    # $1 = unit name
    systemctl --user show -p "$FIELDS" "$1"
}

restart_and_verify() {
    # $1 = unit name.  Returns 0 on verified-fresh restart, 1 otherwise.
    local unit="$1"
    local baseline baseline_pid baseline_mono state pid active mono deadline
    baseline="$(get_state "$unit")"
    baseline_pid="$(read_field "$baseline" MainPID)"
    baseline_mono="$(read_field "$baseline" ActiveEnterTimestampMonotonic)"

    echo "Restarting ${unit} (baseline MainPID=${baseline_pid})..."
    systemctl --user restart "$unit"

    echo -n "Verifying fresh start for ${unit}..."
    deadline=$((SECONDS + VERIFY_TIMEOUT))
    while [[ $SECONDS -lt $deadline ]]; do
        state="$(get_state "$unit")"
        pid="$(read_field "$state" MainPID)"
        active="$(read_field "$state" ActiveState)"
        mono="$(read_field "$state" ActiveEnterTimestampMonotonic)"
        # ActiveEnterTimestampMonotonic (+ ActiveState=active + a live pid) is
        # the authoritative freshness signal.  PID inequality is NOT required:
        # the kernel can reuse the old MainPID for the new process, which
        # would make a clean restart look spuriously stale.
        if [[ "$pid" -gt 0 && "$active" == "active" && "$mono" -gt "$baseline_mono" ]]; then
            echo " OK (new MainPID=${pid})"
            return 0
        fi
        sleep 1
    done

    # Grace re-probe (task 2961): VERIFY_TIMEOUT expired without seeing a
    # fresh ActiveEnterTimestampMonotonic, but a slow-draining unit's actual
    # stop/start can still be in flight underneath a superseded/canceled
    # restart job -- declaring FAILED here, before that in-flight start has
    # had a chance to land, produces a false failure/escalation for a unit
    # that verifies fresh moments later. Re-probe on the same cadence for up
    # to VERIFY_GRACE more seconds before giving up for real.
    echo -n " not yet fresh after ${VERIFY_TIMEOUT}s, re-probing for up to ${VERIFY_GRACE}s more..."
    deadline=$((SECONDS + VERIFY_GRACE))
    while [[ $SECONDS -lt $deadline ]]; do
        state="$(get_state "$unit")"
        pid="$(read_field "$state" MainPID)"
        active="$(read_field "$state" ActiveState)"
        mono="$(read_field "$state" ActiveEnterTimestampMonotonic)"
        if [[ "$pid" -gt 0 && "$active" == "active" && "$mono" -gt "$baseline_mono" ]]; then
            echo " OK (new MainPID=${pid}, verified fresh during grace re-probe)"
            return 0
        fi
        sleep 1
    done
    echo " FAILED"
    return 1
}

drain_check_verdict() {
    # $1 = unit name.  Prints exactly one of idle/busy/stale/absent to
    # stdout -- never more, never less -- regardless of what drain_check.py
    # itself produced.  If the python3 invocation fails to exit 0 (e.g.
    # python3 missing from PATH, or a future drain_check.py change that
    # raises before -- or after partially printing -- a verdict), any
    # output it did produce is discarded and the result is "absent" --
    # fail-toward-convergence, same as an unreadable heartbeat file --
    # rather than aborting the entire restart-all run under `set -e`.  If
    # it exits 0 but the captured output is not exactly one recognized
    # token (e.g. a future change emits a partial token plus a trailing
    # line), that is ALSO coerced to "absent" rather than trusted verbatim,
    # so a malformed or multi-line reading can't silently misclassify the
    # downstream idle/busy/stale/absent string comparisons.  The drain gate
    # must not become a hard dependency on drain_check.py always behaving.
    # stderr is left unsuppressed so a real failure is still visible in the
    # script's own output.
    local raw
    raw="$(python3 "$SCRIPT_DIR/drain_check.py" --unit "$1" --fleet-dir "$FLEET_DIR" \
        --fresh-window "$DRAIN_FRESH_WINDOW_SECS")" || raw="absent"
    case "$raw" in
        idle|busy|stale|absent)
            printf '%s\n' "$raw"
            ;;
        *)
            printf '%s\n' "absent"
            ;;
    esac
}

drain_await_fresh() {
    # $1 = unit name.  Waits up to DRAIN_UNKNOWN_GRACE_SECS for a stale or
    # absent heartbeat to become fresh (idle or busy), re-polling every
    # DRAIN_POLL_INTERVAL_SECS.  Contract: sets the module-global
    # _DRAIN_VERDICT to the resulting verdict -- idle or busy if a fresh
    # reading appeared before the grace elapsed, or the original
    # stale/absent verdict if the grace elapsed with no fresh reading
    # (fail-toward-convergence: the caller proceeds with the restart) --
    # always returns 0, and prints nothing to stdout.
    #
    # MUST be invoked as a plain command, e.g. `drain_await_fresh "$unit"`
    # then read `$_DRAIN_VERDICT` -- and MUST NEVER be called via command
    # substitution (`verdict="$(drain_await_fresh "$unit")"`). Command
    # substitution runs this function's poll loop inside a forked
    # subshell; a SIGKILL of the top-level script pid does not reach that
    # subshell, so it survives, reparents to systemd --user (PPID 2036),
    # and keeps forking one `python3 drain_check.py` per
    # DRAIN_POLL_INTERVAL_SECS for up to DRAIN_UNKNOWN_GRACE_SECS more --
    # an orphan observed running for ~27.8h in the wild before being
    # manually reaped (task 3852).
    #
    # Guard below enforces that contract: a caller who reintroduces
    # command substitution gets a loud, immediate failure instead of the
    # silent-empty-verdict regression described above.
    if [[ ${BASH_SUBSHELL:-0} -ne 0 ]]; then
        echo "BUG(task 3852): drain_await_fresh must run in the main shell, not a subshell (BASH_SUBSHELL=$BASH_SUBSHELL); its poll loop would survive a kill of the top-level script pid" >&2
        return 1
    fi
    local unit="$1"
    local grace_start elapsed_grace
    _DRAIN_VERDICT="$(drain_check_verdict "$unit")"
    grace_start=$SECONDS
    while [[ "$_DRAIN_VERDICT" == "stale" || "$_DRAIN_VERDICT" == "absent" ]]; do
        elapsed_grace=$((SECONDS - grace_start))
        if [[ $elapsed_grace -ge $DRAIN_UNKNOWN_GRACE_SECS ]]; then
            break
        fi
        sleep "$DRAIN_POLL_INTERVAL_SECS"
        _DRAIN_VERDICT="$(drain_check_verdict "$unit")"
    done
    return 0
}

drain_gate() {
    # $1 = unit name.  Only called when --drain was passed.  Blocks
    # (poll-and-recheck) until it is safe to restart $1.
    #
    # - idle (fresh, merge_idle) returns immediately (transparent).
    # - stale/absent (heartbeat missing or too old) is given a bounded
    #   ORCH_DRAIN_UNKNOWN_GRACE_SECS grace via drain_await_fresh: a fresh
    #   idle/busy reading that appears during the grace re-classifies into
    #   the branches below; if the grace elapses with the heartbeat still
    #   stale/absent, the restart proceeds anyway (fail-toward-convergence
    #   -- the opposite fail direction from a confirmed-busy unit, which
    #   fails toward protecting the merge).
    # - busy (fresh, mid-merge) defers with a journal line and polls every
    #   DRAIN_POLL_INTERVAL_SECS until the unit drains (idle) or
    #   FORCE_FIRE_AFTER_SECS elapses, at which point it force-proceeds
    #   anyway.  A unit that goes stale/absent WHILE deferred (it stopped
    #   heartbeating -- e.g. crashed mid-merge) is NOT held for the rest of
    #   that busy grace: it drops into the same bounded drain_await_fresh
    #   handling as the top-level stale/absent case, since a dead unit
    #   isn't actually merging.  If it resumes busy afterward, deferral
    #   keeps counting from when the unit FIRST went busy, not from that
    #   resumption -- the force-fire deadline is anchored once, so a
    #   busy<->stale/absent oscillation can't defer the forced restart
    #   indefinitely.
    local unit="$1"
    local verdict start_secs elapsed
    drain_await_fresh "$unit"
    verdict="$_DRAIN_VERDICT"

    if [[ "$verdict" == "stale" || "$verdict" == "absent" ]]; then
        echo "proceeding with restart of ${unit}: heartbeat ${verdict} after ${DRAIN_UNKNOWN_GRACE_SECS}s grace"
        return 0
    fi

    if [[ "$verdict" == "idle" ]]; then
        return 0
    fi

    # verdict == "busy" here: idle and stale/absent are both handled above.
    echo "deferring restart of ${unit}: mid-merge (grace $((FORCE_FIRE_AFTER_SECS / 60))m)"
    start_secs=$SECONDS
    while true; do
        elapsed=$((SECONDS - start_secs))
        if [[ $elapsed -ge $FORCE_FIRE_AFTER_SECS ]]; then
            echo "force-restarting ${unit}: mid-merge grace of ${FORCE_FIRE_AFTER_SECS}s exceeded"
            return 0
        fi
        sleep "$DRAIN_POLL_INTERVAL_SECS"
        verdict="$(drain_check_verdict "$unit")"
        if [[ "$verdict" == "idle" ]]; then
            echo "resuming restart of ${unit}: drained"
            return 0
        fi
        if [[ "$verdict" == "stale" || "$verdict" == "absent" ]]; then
            # Stopped heartbeating mid-defer -- apply the shorter bounded
            # grace instead of continuing to wait out the full busy grace.
            drain_await_fresh "$unit"
            verdict="$_DRAIN_VERDICT"
            if [[ "$verdict" == "idle" ]]; then
                echo "resuming restart of ${unit}: drained"
                return 0
            elif [[ "$verdict" == "busy" ]]; then
                # Alive and merging again -- resume deferring.  start_secs
                # is intentionally NOT reset here: the force-fire deadline
                # is anchored once, from when this unit first went busy, so
                # a busy<->stale/absent oscillation can't keep deferring
                # the forced restart indefinitely (the elapsed check at the
                # top of the loop measures total wall-clock since draining
                # began, not since the most recent busy transition).
                echo "deferring restart of ${unit}: mid-merge (grace $((FORCE_FIRE_AFTER_SECS / 60))m)"
            else
                echo "proceeding with restart of ${unit}: heartbeat ${verdict} after ${DRAIN_UNKNOWN_GRACE_SECS}s grace"
                return 0
            fi
        fi
    done
}

stamp_fleet_deploy_clock() {
    # Atomically stamps CLOCK_FILE with the current epoch/UTC-ISO time:
    # mktemp a sibling file IN the same directory, write it, then `mv -f`
    # onto CLOCK_FILE (same-filesystem rename -- atomic). Schema is {ts,
    # iso}, matching the coordinator's _persist_last_fire_wall, so
    # float(raw['ts']) reads it identically from either writer. Called ONLY
    # from the all-units-verified-fresh exit-0 path below -- never on a
    # failed/partial verify or the early no-running-units exit -- so a
    # failed fleet restart can never silence the watchdog backstop (I2).
    local clock_dir tmp_file
    clock_dir="$(dirname "$CLOCK_FILE")"
    mkdir -p "$clock_dir"
    tmp_file="$(mktemp "$clock_dir/.last_redeploy_orchestrator.XXXXXX")"
    printf '{"ts": %s, "iso": "%s"}\n' \
        "$(date +%s)" "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" > "$tmp_file"
    mv -f "$tmp_file" "$CLOCK_FILE"
}

# Enumerate running orchestrator units at run time (robust to which projects
# are enabled on this host), deferring SELF_UNIT to the end.
mapfile -t running_units < <(
    systemctl --user list-units 'orchestrator-*.service' \
        --state=running --no-legend --plain 2>/dev/null | awk '{print $1}'
)

if [[ ${#running_units[@]} -eq 0 ]]; then
    echo "No running orchestrator-*.service units found; nothing to restart."
    exit 0
fi

ordered_units=()
self_present=0
for unit in "${running_units[@]}"; do
    if [[ "$unit" == "$SELF_UNIT" ]]; then
        self_present=1
    else
        ordered_units+=("$unit")
    fi
done
if [[ $self_present -eq 1 ]]; then
    ordered_units+=("$SELF_UNIT")
fi

echo "Restarting ${#ordered_units[@]} orchestrator unit(s): ${ordered_units[*]}"

failures=()
for unit in "${ordered_units[@]}"; do
    if [[ $DRAIN_ENABLED -eq 1 ]]; then
        drain_gate "$unit"
    fi
    if ! restart_and_verify "$unit"; then
        failures+=("$unit")
    fi
done

if [[ ${#failures[@]} -gt 0 ]]; then
    echo "ERROR: restart did not verify fresh for: ${failures[*]}" >&2
    exit 1
fi

stamp_fleet_deploy_clock

echo "All ${#ordered_units[@]} orchestrator unit(s) restarted and verified fresh."
exit 0
