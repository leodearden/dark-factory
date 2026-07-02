#!/usr/bin/env bash
set -euo pipefail

# flip-reify-gate-exclude-heavy.sh — idempotently flip
# REIFY_GATE_EXCLUDE_HEAVY=1 in the reify-tracked orchestrator.yaml's
# verify_env: block, commit the change in the reify repo, and signal an
# orchestrator config reload.
#
# INTENDED CALLER: a task_kind='deterministic' deploy task's
# `before_done.script` (dark_factory task ε2). This script only performs
# the config-repo-side flip + commit + reload SIGNAL; the reload MECHANIC
# (in-place reload vs target_unit self-kill restart) is selected by ε2's
# deterministic runner around signal_config_reload() (PRD §11.6 open-Q).
#
# Usage:
#   flip-reify-gate-exclude-heavy.sh [--check|--dry-run]
#
# --check / --dry-run: print the one-line diff that WOULD be applied and
# exit 0 without touching the repo.
#
# Env overrides (mirrors restart-orchestrator.sh's RESTART_VERIFY_TIMEOUT):
#   REIFY_REPO - path to the reify checkout (default: /home/leo/src/reify)

REIFY_REPO="${REIFY_REPO:-/home/leo/src/reify}"
CONFIG_FILE="orchestrator.yaml"
CONFIG_PATH="$REIFY_REPO/$CONFIG_FILE"
KNOB="REIFY_GATE_EXCLUDE_HEAVY"

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

is_flipped() {
    grep -qE "^[[:space:]]*${KNOB}:[[:space:]]*\"?1\"?[[:space:]]*\$" "$CONFIG_PATH"
}

print_intended_diff() {
    echo "+  ${KNOB}: \"1\"   (under verify_env: in ${CONFIG_PATH})"
}

# ε2 seam (PRD §11.6 open-Q): this only EMITS the observable reload signal.
# The actual reload MECHANIC (in-place config reload vs a target_unit
# self-kill restart via the deterministic runner) is selected by ε2 --
# it wraps/replaces this function's body, not ε1's call site.
signal_config_reload() {
    echo "flip-reify-gate: config-reload signal emitted"
}

apply_flip() {
    # Comment-preserving targeted edit (NOT a YAML round-trip): strip any
    # pre-existing knob line (any indentation/value), then insert the
    # canonical line as the first child immediately after the unique
    # top-level `verify_env:` anchor. Single awk pass, atomic mktemp+mv.
    local tmp
    tmp="$(mktemp "${CONFIG_PATH}.XXXXXX")"
    awk -v knob="$KNOB" '
        $0 ~ "^[[:space:]]*" knob ":" { next }
        { print }
        /^verify_env:/ && !inserted {
            print "  " knob ": \"1\""
            inserted = 1
        }
    ' "$CONFIG_PATH" > "$tmp"
    mv "$tmp" "$CONFIG_PATH"
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
        git -C "$REIFY_REPO" commit --no-verify -m "deploy: flip ${KNOB}=1 (exclude heavy from reify verify gate)"
        signal_config_reload
    fi
}

main
