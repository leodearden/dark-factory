#!/usr/bin/env bash
# watcher-rearm.sh -- canonical bounded-wait + re-arm wrapper for
# escalation.watcher, shared by skills/escalation-watcher/SKILL.md and
# skills/escalation-watcher-auto/SKILL.md.
#
# Solves the watcher-loop-harness-mismatch cluster (agent-legibility survey
# S2): callers no longer hand-maintain a growing --exclude-id list across
# re-arms -- this wrapper owns a persisted --exclude-file (default
# "<queue-dir>/.watcher-rearm-exclude-l<level>") that a loop caller appends
# a deliberately-pending id to instead of rebuilding the whole list.
#
# Usage:
#   watcher-rearm.sh --level N [--queue-dir DIR] [--timeout SECS]
#                     [--exclude-file PATH] [--baseline]
#                     [--ntfy-url URL] [--task-id ID] [--check]
#
# Env:
#   DARK_FACTORY_ROOT     required; repo root used to locate the escalation
#                         project for `uv run --project`.
#   PROJECT_ROOT          optional; used to derive a default --queue-dir
#                         ($PROJECT_ROOT/data/escalations) when --queue-dir
#                         is not passed.
#   WATCHER_REARM_PYTHON  optional interpreter override (default:
#                         `uv run --project $DARK_FACTORY_ROOT/escalation
#                         python`); tests inject `sys.executable` here to
#                         drive the REAL escalation.watcher module directly
#                         instead of spawning `uv run`.
#
# Exit codes (preserved verbatim from escalation.watcher, except usage=2):
#   0    FIRED    -- one matching escalation printed to stdout.
#   124  CEILING  -- --timeout expired with nothing pending.
#   137|143|144  KILLED -- external kill (SIGKILL/SIGTERM/etc).
#   2    usage error / missing required input (silent-no-op guard) --
#        NEVER a silent no-op: a diagnostic is always printed to stderr.
#   *    ERROR -- any other non-zero exit from the watcher itself.
#
# A machine-readable `WATCHER_REARM_OUTCOME: <FIRED|CEILING|KILLED|ERROR>
# exit=<rc>` line is always emitted to STDERR (never stdout), so callers can
# grep the outcome without disturbing the watcher's JSON on stdout -- do
# NOT pipe `2>&1` if you parse stdout as JSON.
#
# Bash-tool contract: a `--timeout 540` call blocks for up to 540s. The
# CALLER's Bash tool timeout must be >= 600000ms (10 min), or the bounded
# wait itself gets killed by the harness's 2-minute default before it can
# return -- this was the 07-09 exit-143 failure mode this wrapper exists to
# prevent.
set -uo pipefail

QUEUE_DIR=""
LEVEL=""
TIMEOUT=540
EXCLUDE_FILE_ARG=""
BASELINE=0
NTFY_URL=""
TASK_ID=""
CHECK=0

while [ $# -gt 0 ]; do
    case "$1" in
        --queue-dir)
            [ $# -ge 2 ] || { echo "watcher-rearm.sh: missing value for --queue-dir" >&2; exit 2; }
            QUEUE_DIR="$2"
            shift 2
            ;;
        --level)
            [ $# -ge 2 ] || { echo "watcher-rearm.sh: missing value for --level" >&2; exit 2; }
            LEVEL="$2"
            shift 2
            ;;
        --timeout)
            [ $# -ge 2 ] || { echo "watcher-rearm.sh: missing value for --timeout" >&2; exit 2; }
            TIMEOUT="$2"
            shift 2
            ;;
        --exclude-file)
            [ $# -ge 2 ] || { echo "watcher-rearm.sh: missing value for --exclude-file" >&2; exit 2; }
            EXCLUDE_FILE_ARG="$2"
            shift 2
            ;;
        --baseline)
            BASELINE=1
            shift
            ;;
        --ntfy-url)
            [ $# -ge 2 ] || { echo "watcher-rearm.sh: missing value for --ntfy-url" >&2; exit 2; }
            NTFY_URL="$2"
            shift 2
            ;;
        --task-id)
            [ $# -ge 2 ] || { echo "watcher-rearm.sh: missing value for --task-id" >&2; exit 2; }
            TASK_ID="$2"
            shift 2
            ;;
        --check)
            CHECK=1
            shift
            ;;
        *)
            echo "watcher-rearm.sh: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

# Resolve QUEUE_DIR: explicit --queue-dir wins, else derive from PROJECT_ROOT.
if [ -z "$QUEUE_DIR" ] && [ -n "${PROJECT_ROOT:-}" ]; then
    QUEUE_DIR="$PROJECT_ROOT/data/escalations"
fi

# Silent-no-op guard: every required input is loudly diagnosed on stderr
# and exits 2 -- this wrapper must never silently do nothing.
if [ -z "${DARK_FACTORY_ROOT:-}" ] || [ ! -d "${DARK_FACTORY_ROOT:-/nonexistent}" ]; then
    echo "watcher-rearm.sh: DARK_FACTORY_ROOT must be set to a valid directory" >&2
    exit 2
fi
if [ -z "$QUEUE_DIR" ]; then
    echo "watcher-rearm.sh: no queue dir resolved -- pass --queue-dir or set PROJECT_ROOT" >&2
    exit 2
fi
if [ -z "$LEVEL" ]; then
    echo "watcher-rearm.sh: --level is required" >&2
    exit 2
fi

# Wrapper-owned exclude-file: an explicit --exclude-file wins, else default
# to a per-level path inside the queue dir (mkdir/touch + wiring into the
# watcher invocation land alongside the invocation itself).
EXCLUDE_FILE="${EXCLUDE_FILE_ARG:-$QUEUE_DIR/.watcher-rearm-exclude-l$LEVEL}"

if [ "$CHECK" -eq 1 ]; then
    echo "watcher-rearm.sh: would watch queue-dir=$QUEUE_DIR level=$LEVEL timeout=$TIMEOUT exclude-file=$EXCLUDE_FILE"
    exit 0
fi

# Interpreter prefix: WATCHER_REARM_PYTHON overrides the default `uv run`
# invocation (tests inject sys.executable to drive the real module without
# uv). Deliberately unquoted below so a multi-word override word-splits
# into separate argv entries, same as the default.
WATCHER_CMD=${WATCHER_REARM_PYTHON:-uv run --project "$DARK_FACTORY_ROOT/escalation" python}

WATCHER_ARGS=(-m escalation.watcher --queue-dir "$QUEUE_DIR" --level "$LEVEL" --timeout "$TIMEOUT")
[ -n "$NTFY_URL" ] && WATCHER_ARGS+=(--ntfy-url "$NTFY_URL")
[ -n "$TASK_ID" ] && WATCHER_ARGS+=(--task-id "$TASK_ID")
[ "$BASELINE" -eq 1 ] && WATCHER_ARGS+=(--baseline)

# shellcheck disable=SC2086
$WATCHER_CMD "${WATCHER_ARGS[@]}"
rc=$?

case "$rc" in
    0)
        echo "WATCHER_REARM_OUTCOME: FIRED exit=0" >&2
        ;;
    124)
        echo "WATCHER_REARM_OUTCOME: CEILING exit=124" >&2
        ;;
    137|143|144)
        echo "WATCHER_REARM_OUTCOME: KILLED exit=$rc" >&2
        ;;
    *)
        echo "WATCHER_REARM_OUTCOME: ERROR exit=$rc" >&2
        ;;
esac

exit "$rc"
