#!/usr/bin/env bash
set -euo pipefail

# enable_laptop_persistent_worktree.sh
#
# SCOPE: workstation-side idempotent flip of `git.persistent_merge_worktree`
# to `true` in the laptop's reify-laptop.yaml orchestrator config, applied
# remotely over ssh. Authored + committed by task 2310 (PRD
# plans/laptop-warm-verify-flock-orphan-prd.md, task delta1); consumed by
# delta2 (a deferred deterministic-deploy task) as before_done.script. This
# script does not flip the production flag as part of authoring it -- it is
# exercised in tests only, against fake ssh targets.
#
# Usage:
#   enable_laptop_persistent_worktree.sh [--check|--dry-run]
#
# --check / --dry-run: print the intended change and exit 0 without
# mutating the remote config. If the flag is already enabled, --check
# reports the same "already enabled (no-op)" state as apply, since applying
# would change nothing.
#
# Env overrides:
#   SSH                - ssh invocation prefix
#                         (default: "ssh -o BatchMode=yes -o ConnectTimeout=10")
#   LAPTOP_HOST         - target host (default: leo-laptop)
#   LAPTOP_CONFIG_PATH  - path to the laptop's orchestrator config
#                         (default: /home/leo/.config/orchestrator/reify-laptop.yaml)
#   REMOTE_PYTHON       - python interpreter on the remote host, used for
#                         YAML parse/readback validation (default: python3)
#   BACKUP_LABEL        - suffix for the pre-edit backup file
#                         <config>.bak-<label> (default: a timestamp)

SSH="${SSH:-ssh -o BatchMode=yes -o ConnectTimeout=10}"
LAPTOP_HOST="${LAPTOP_HOST:-leo-laptop}"
LAPTOP_CONFIG_PATH="${LAPTOP_CONFIG_PATH:-/home/leo/.config/orchestrator/reify-laptop.yaml}"
REMOTE_PYTHON="${REMOTE_PYTHON:-python3}"
BACKUP_LABEL="${BACKUP_LABEL:-$(date +%Y%m%d%H%M%S)}"

MODE="apply"

for arg in "$@"; do
    case "$arg" in
        --check|--dry-run)
            MODE="check"
            ;;
    esac
done

# Single-quoted heredoc: everything inside is expanded on the REMOTE host by
# `bash -s -- "$CONFIG" "$MODE" "$LABEL" "$PY"`, not locally. Positional args:
#   $1 = CONFIG (path to the laptop's orchestrator config)
#   $2 = MODE   (apply | check)
#   $3 = LABEL  (backup-file label; unused until the apply edit lands)
#   $4 = PY     (remote python interpreter)
if ! $SSH "$LAPTOP_HOST" bash -s -- "$LAPTOP_CONFIG_PATH" "$MODE" "$BACKUP_LABEL" "$REMOTE_PYTHON" <<'REMOTE_PAYLOAD_EOF'
set -euo pipefail

CONFIG="$1"
MODE="$2"
LABEL="$3"
PY="$4"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config file not found: $CONFIG" >&2
    exit 1
fi

if ! "$PY" -c "import sys, yaml; yaml.safe_load(open(sys.argv[1]))" "$CONFIG" 2>/dev/null; then
    echo "ERROR: ${CONFIG} failed to parse as YAML" >&2
    exit 1
fi

key_present() {
    grep -qE '^[[:space:]]*persistent_merge_worktree[[:space:]]*:' "$CONFIG"
}

git_anchor_present() {
    grep -qE '^git:' "$CONFIG"
}

if ! key_present && ! git_anchor_present; then
    echo "ERROR: refusing blind edit: ${CONFIG} has neither a persistent_merge_worktree key nor a top-level git: anchor" >&2
    exit 1
fi

already_enabled() {
    "$PY" -c "
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
sys.exit(0 if (d.get('git') or {}).get('persistent_merge_worktree') is True else 1)
" "$CONFIG"
}

print_intended_diff() {
    echo "+  persistent_merge_worktree: true   (under git: in ${CONFIG})"
}

if already_enabled; then
    echo "enable-laptop-persistent-worktree: already enabled (no-op)"
    exit 0
fi

if [[ "$MODE" == "check" ]]; then
    print_intended_diff
    exit 0
fi

cp -p "$CONFIG" "${CONFIG}.bak-${LABEL}"

tmp="$(mktemp "${CONFIG}.XXXXXX")"
awk '
    $0 ~ "^[[:space:]]*persistent_merge_worktree[[:space:]]*:" { next }
    { print }
    /^git:/ && !inserted {
        print "  persistent_merge_worktree: true"
        inserted = 1
    }
' "$CONFIG" > "$tmp"

chmod --reference="$CONFIG" "$tmp"
mv "$tmp" "$CONFIG"

already_enabled || {
    echo "ERROR: ${CONFIG} did not read back git.persistent_merge_worktree: true after edit" >&2
    exit 1
}

echo "enable-laptop-persistent-worktree: persistent_merge_worktree set to true"
REMOTE_PAYLOAD_EOF
then
    echo "ERROR: remote payload failed on ${LAPTOP_HOST}" >&2
    exit 1
fi
