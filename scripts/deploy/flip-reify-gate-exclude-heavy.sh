#!/usr/bin/env bash
set -euo pipefail

# flip-reify-gate-exclude-heavy.sh
#
# CONSOLIDATED offline-lane reify deploy: flips BOTH REIFY_GATE_EXCLUDE_HEAVY
# and REIFY_RUN_ALL_EXCLUDE_HOST_INFRA in one sanctioned commit + one reload.
# (IE4's separate flip-reify-run-all-exclude-host-infra.sh was consolidated in
# here; task 1982 retired.) NOTE: filename kept for the deterministic
# before_done reference; it now covers both knobs.
#
# Idempotently set both knobs to "1" in the reify-tracked orchestrator.yaml's
# verify_env: block, commit the change in the reify repo, and signal an
# orchestrator config reload.
#
# INTENDED CALLER: a task_kind='deterministic' deploy task's
# `before_done.script`. This script only performs the config-repo-side flip +
# commit + reload SIGNAL; the reload MECHANIC (in-place reload vs target_unit
# self-kill restart) is selected by the deterministic runner around
# signal_config_reload().
#
# Usage:
#   flip-reify-gate-exclude-heavy.sh [--check|--dry-run]
#
# --check / --dry-run: print the one-line diff(s) that WOULD be applied and
# exit 0 without touching the repo. If both knobs are already flipped, --check
# reports the same "already flipped (no-op)" state as apply instead of the
# intended-diff line, since applying would change nothing.
#
# Env overrides (mirrors restart-orchestrator.sh's RESTART_VERIFY_TIMEOUT):
#   REIFY_REPO - path to the reify checkout (default: /home/leo/src/reify)

REIFY_REPO="${REIFY_REPO:-/home/leo/src/reify}"
CONFIG_FILE="orchestrator.yaml"
CONFIG_PATH="$REIFY_REPO/$CONFIG_FILE"
KNOBS=("REIFY_GATE_EXCLUDE_HEAVY" "REIFY_RUN_ALL_EXCLUDE_HOST_INFRA")

MODE="apply"

for arg in "$@"; do
    case "$arg" in
        --check|--dry-run)
            MODE="check"
            ;;
        *)
            echo "ERROR: unexpected argument: $arg" >&2
            exit 1
            ;;
    esac
done

is_flipped_in() {
    local file="$1" knob="$2"
    grep -qE "^[[:space:]]*${knob}:[[:space:]]*\"?1\"?[[:space:]]*\$" "$file"
}

# True only when EVERY knob is already present — a single missing knob still
# triggers the (single) consolidated commit.
is_flipped() {
    local knob
    for knob in "${KNOBS[@]}"; do
        is_flipped_in "$CONFIG_PATH" "$knob" || return 1
    done
    return 0
}

print_intended_diff() {
    local knob
    for knob in "${KNOBS[@]}"; do
        if ! is_flipped_in "$CONFIG_PATH" "$knob"; then
            echo "+  ${knob}: \"1\"   (under verify_env: in ${CONFIG_PATH})"
        fi
    done
}

# Deterministic-runner seam: this only EMITS the observable reload signal. The
# actual reload MECHANIC (in-place config reload vs a target_unit self-kill
# restart via the deterministic runner) is selected by the runner -- it
# wraps/replaces this function's body, not the call site.
signal_config_reload() {
    echo "flip-reify-gate: config-reload signal emitted"
}

apply_knob() {
    # Comment-preserving targeted edit (NOT a YAML round-trip): strip any
    # pre-existing knob line (any indentation/value), then insert the canonical
    # line as the first child immediately after the unique top-level
    # `verify_env:` anchor. Single awk pass, atomic mktemp+mv.
    local knob="$1"
    local tmp
    tmp="$(mktemp "${CONFIG_PATH}.XXXXXX")"
    awk -v knob="$knob" '
        $0 ~ "^[[:space:]]*" knob ":" { next }
        { print }
        /^verify_env:/ && !inserted {
            print "  " knob ": \"1\""
            inserted = 1
        }
    ' "$CONFIG_PATH" > "$tmp"

    if ! is_flipped_in "$tmp" "$knob"; then
        rm -f "$tmp"
        echo "ERROR: failed to insert ${knob} into ${CONFIG_PATH}; aborting without committing" >&2
        exit 1
    fi

    # Preserve the original file's mode -- mktemp defaults to 0600, which would
    # otherwise silently reset orchestrator.yaml's permissions on every flip.
    chmod --reference="$CONFIG_PATH" "$tmp"
    mv "$tmp" "$CONFIG_PATH"
}

apply_flip() {
    if ! grep -qE "^verify_env:" "$CONFIG_PATH"; then
        echo "ERROR: ${CONFIG_PATH} has no top-level verify_env: anchor; refusing to apply a blind edit" >&2
        exit 1
    fi

    local knob
    for knob in "${KNOBS[@]}"; do
        apply_knob "$knob"
    done
}

main() {
    if is_flipped; then
        echo "flip-reify-gate: already flipped (no-op)"
        exit 0
    fi

    if [[ "$MODE" == "check" ]]; then
        print_intended_diff
        exit 0
    fi

    apply_flip

    git -C "$REIFY_REPO" add -- "$CONFIG_FILE"
    if ! git -C "$REIFY_REPO" diff --cached --quiet -- "$CONFIG_FILE"; then
        # Sanction handshake for reify's ENFORCE main-gate: --no-verify skips
        # pre-commit/commit-msg hooks but NOT the reference-transaction hook, so
        # an unsanctioned main move is aborted (rc=128). Create reify's one-shot
        # sentinel right before the commit so the hook consumes it and allows
        # the move (mirrors reify's own git.main_gate_mark_command). The
        # ABSOLUTE common-dir is load-bearing: this script runs with
        # cwd=/home/leo/src/dark-factory, so a relative .git would write the
        # sentinel into the WRONG repo. The trap clears a stray sentinel if the
        # commit aborts.
        _reify_gcd="$(git -C "$REIFY_REPO" rev-parse --path-format=absolute --git-common-dir)"
        trap 'rm -f "$_reify_gcd/reify-main-gate-ok"' EXIT
        touch "$_reify_gcd/reify-main-gate-ok"
        git -C "$REIFY_REPO" commit --no-verify -m "deploy: flip REIFY_GATE_EXCLUDE_HEAVY=1 + REIFY_RUN_ALL_EXCLUDE_HOST_INFRA=1 (consolidated offline-lane deploy)"
        signal_config_reload
    fi
}

main
