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
#   flip-reify-gate-exclude-heavy.sh
#
# Env overrides (mirrors restart-orchestrator.sh's RESTART_VERIFY_TIMEOUT):
#   REIFY_REPO - path to the reify checkout (default: /home/leo/src/reify)

REIFY_REPO="${REIFY_REPO:-/home/leo/src/reify}"
CONFIG_FILE="orchestrator.yaml"
CONFIG_PATH="$REIFY_REPO/$CONFIG_FILE"
KNOB="REIFY_GATE_EXCLUDE_HEAVY"

MODE="apply"

is_flipped() {
    grep -qE "^[[:space:]]*${KNOB}:[[:space:]]*\"?1\"?[[:space:]]*\$" "$CONFIG_PATH"
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
    apply_flip

    git -C "$REIFY_REPO" add -- "$CONFIG_FILE"
    git -C "$REIFY_REPO" commit -m "deploy: flip ${KNOB}=1 (exclude heavy from reify verify gate)"
}

main
