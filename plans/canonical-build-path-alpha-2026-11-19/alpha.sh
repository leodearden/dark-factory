#!/usr/bin/env bash
# Gate 10782 (task α of plans/canonical-build-path-shared-cache-prd.md), 2026-11-19.
#
# The study's trial harness (plans/sccache-path-key-study-2026-11-18/trial/exp2.sh) reshaped to the
# PRD: the whole verify leg runs with its worktree bound at CANON (the same string on both hosts);
# a FOREGROUND sccache server, a child of the leg, addressed by a unix socket inside the build
# directory; its store a private `rclone serve webdav` on the workstation, which leo-laptop reaches
# over tailscale; PWD set to the working directory; no PID namespace; one pinned CARGO_HOME.
#
#   alpha.sh store-start | store-stop       (workstation) the private store
#   alpha.sh setup                          clone reify into $ROOT/repo at the trial's commit X
#   alpha.sh smoke                          prove namespace + foreground UDS server + store on a toy
#   alpha.sh run <ID> <all|all+tests>       one never-built worktree, the verify's four build steps
#                                           (launch under systemd-run --user, see launch.sh)
#   alpha.sh burst <n> <label> [start_epoch] <n> foreground servers started at once against the store
#   alpha.sh agent <worktree> <crate>       (workstation) a sandboxed Claude session at CANON
#   alpha.sh clean                          remove worktrees and clone; keep results/
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CANON=/home/leo/.reify-canon-alpha/ws     # stand-in for /opt/reify-canon/ws: same string on both hosts
SRC=/home/leo/src/reify
X=27c69f616f70d4a77bf7164b7bcf062404a9aaad     # the 2026-11-18 trial's commit, so N1/N2 are comparators
MAIN=84d560850933bdd25dfb6e92a7b73ddb24d14b8a  # the trial's main, so --scope branch is the same 24 packages
case "$(hostname)" in
    leo-MS-7C35) HOST=workstation; ROOT=/home/leo/canon-alpha-20261119 ;;
    leo-Laptop-16-*) HOST=laptop; ROOT=/media/leo/Internal-2nd/canon-alpha-20261119 ;;
    *) echo "unknown host $(hostname)" >&2; exit 64 ;;
esac
RES="$ROOT/results"
STORE_DIR=/home/leo/canon-alpha-20261119/store   # workstation only
STORE_PORT=4231                                  # never 4226 (production sccache)
STORE_PREFIX=alpha
FALLBACK="$ROOT/fallback"                        # never ~/.cache/sccache

export PATH="$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
export LC_ALL=C.UTF-8
export CARGO_HOME=/tmp/reify-agent-cargo-home    # reify verify_env's pin (and the agents')
export REIFY_RUN_ALL_EXCLUDE_HOST_INFRA=1 REIFY_GATE_EXCLUDE_HEAVY=1 REIFY_BUILD_CACHE_ARM=sccache
unset DF_VERIFY_ROLE SCCACHE_SERVER_PORT SCCACHE_WEBDAV_ENDPOINT SCCACHE_WEBDAV_USERNAME SCCACHE_WEBDAV_PASSWORD

log() { mkdir -p "$RES"; printf '%s %s %s\n' "$(date -u +%H:%M:%SZ)" "$HOST" "$*" | tee -a "$RES/alpha.log" >&2; }
die() { log "FATAL: $*"; echo "$*" >"$RES/FAILED"; exit 1; }
free_gib() { df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9'; }

store_env() {  # the store as this host reaches it
    [ -r "$ROOT/store.env" ] || die "no $ROOT/store.env"
    # shellcheck disable=SC1091
    . "$ROOT/store.env"
}

# ---- private sccache server: foreground child, unix socket, store backend (PRD C4/C5) -----------
SPID='' SOCK=''
server_start() {  # server_start <socket> <logfile>
    SOCK="$1"
    store_env
    mkdir -p "$(dirname "$SOCK")"
    rm -f "$SOCK"
    ( exec 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&-
      exec env -u SCCACHE_DIR SCCACHE_START_SERVER=1 SCCACHE_NO_DAEMON=1 SCCACHE_IDLE_TIMEOUT=0 \
          SCCACHE_SERVER_UDS="$SOCK" SCCACHE_LOG=debug \
          SCCACHE_WEBDAV_ENDPOINT="$ALPHA_STORE_URL" SCCACHE_WEBDAV_KEY_PREFIX="$STORE_PREFIX" \
          SCCACHE_WEBDAV_USERNAME="$ALPHA_STORE_USER" SCCACHE_WEBDAV_PASSWORD="$ALPHA_STORE_PASS" \
          nice -n 15 sccache ) </dev/null >"$2.stdout" 2>"$2" &
    SPID=$!
    local i
    for i in $(seq 1 600); do   # never query before the socket exists: a client finding no listener auto-starts one
        [ -S "$SOCK" ] && break
        kill -0 "$SPID" 2>/dev/null || return 1
        sleep 0.1
    done
    [ -S "$SOCK" ] || return 1
    SCCACHE_SERVER_UDS="$SOCK" sccache --show-stats --stats-format json >/dev/null 2>&1
}
server_stop() {  # server_stop <stats-json-out>
    [ -n "$SPID" ] || return 0
    if kill -0 "$SPID" 2>/dev/null; then
        SCCACHE_SERVER_UDS="$SOCK" sccache --show-stats --stats-format json >"$1" 2>/dev/null
        SCCACHE_SERVER_UDS="$SOCK" sccache --stop-server >/dev/null 2>&1
    else
        echo '{"error":"server died before stop"}' >"$1"
    fi
    wait "$SPID" 2>/dev/null
    rm -f "$SOCK"
    SPID=''
}
client_env() {  # what the command sees: the socket, a local fallback dir, a short idle timeout; no store
    export SCCACHE_SERVER_UDS="$SOCK" SCCACHE_DIR="$FALLBACK" SCCACHE_IDLE_TIMEOUT=1800
    mkdir -p "$FALLBACK"
}

# ns <worktree> <cmd...>: the worktree bound at CANON, cwd and PWD = CANON (PRD C3).
ns() { local wt="$1"; shift
    /usr/bin/bwrap --dev-bind / / --bind "$wt" "$CANON" --chdir "$CANON" --setenv PWD "$CANON" --die-with-parent -- "$@"; }

mk_wt() {
    local wt="$ROOT/wt/$1"
    [ -e "$wt" ] && die "worktree $wt already exists"
    git -C "$ROOT/repo" worktree add --detach "$wt" "$X" >>"$RES/alpha.log" 2>&1 || die "worktree add $1 failed"
}

# ---- the leg -------------------------------------------------------------------------------------
extract_plan() {  # unchanged from exp2.sh
    local out="$1" cfg
    mkdir -p "$out"
    ./scripts/verify.sh test --scope branch --include-infra --print-plan >"$out/plan.test.txt" 2>"$out/plan.test.err" || return 2
    ./scripts/verify.sh lint --scope branch --include-infra --print-plan >"$out/plan.lint.txt" 2>"$out/plan.lint.err" || return 3
    cfg="$(REIFY_NEXTEST_CLI_TEST_THREADS= ./scripts/gen-nextest-config.sh 2>"$out/gen-nextest.err")" || return 4
    [ -f "$cfg" ] || return 5
    grep -E '^# (export |\. )' "$out/plan.test.txt" | sed 's/^# //' | grep -v CARGO_MAKEFLAGS >"$out/env.sh"
    echo 'unset CARGO_MAKEFLAGS MAKEFLAGS' >>"$out/env.sh"
    echo 'export CARGO_BUILD_JOBS=8' >>"$out/env.sh"
    : >"$out/steps.tsv"
    grep -v '^#' "$out/plan.test.txt" | grep -F 'cargo nextest run' | grep -v -F 'reify-gui --features gui' \
        | sed -e 's/cargo nextest run /cargo nextest run --no-run /' -e "s#[^ ;]*reify-nextest-occt\.[^ ;]*#$cfg#g" \
        | while IFS= read -r c; do printf 'debug_nextest_build\t%s\n' "$c"; done >>"$out/steps.tsv"
    grep -v '^#' "$out/plan.test.txt" | grep -F 'cargo nextest run' | grep -F 'reify-gui --features gui' \
        | sed -e 's/cargo nextest run /cargo nextest run --no-run /' -e "s#[^ ;]*reify-nextest-occt\.[^ ;]*#$cfg#g" \
        | while IFS= read -r c; do printf 'gui_nextest_build\t%s\n' "$c"; done >>"$out/steps.tsv"
    grep -v '^#' "$out/plan.lint.txt" | grep -F 'cargo clippy' \
        | while IFS= read -r c; do printf 'clippy\t%s\n' "$c"; done >>"$out/steps.tsv"
    grep -v '^#' "$out/plan.lint.txt" | grep -F 'cargo check -p reify-gui' \
        | while IFS= read -r c; do printf 'gui_check\t%s\n' "$c"; done >>"$out/steps.tsv"
    grep -v '^#' "$out/plan.test.txt" | grep -E 'tree-sitter-generate\.sh|tree-sitter-freshness\.sh ensure' >"$out/preflight.sh"
    [ -s "$out/steps.tsv" ] || return 6
}

one_step() {  # one_step <id> <name> <cmd> <fatal>
    local id="$1" name="$2" cmd="$3" fatal="$4" out="$RES/$1" rc t0
    python3 "$HERE/snap.py" >"$out/$name.before.json"
    t0=$(date +%s.%N)
    ( . "$out/env.sh" && export CARGO_HOME=/tmp/reify-agent-cargo-home && bash -c "$cmd" ) >"$out/$name.log" 2>&1 </dev/null
    rc=$?
    python3 "$HERE/snap.py" >"$out/$name.after.json"
    printf '%s\t%s\t%s\t%s\t%.1f\n' "$HOST" "$id" "$name" "$rc" "$(echo "$(date +%s.%N) - $t0" | bc)" >>"$RES/steps.tsv"
    log "$id/$name rc=$rc"
    [ "$rc" -eq 0 ] || [ "$fatal" -eq 0 ] || die "$id/$name failed rc=$rc (see $out/$name.log)"
}

cmd_inner() {  # INSIDE the namespace: cmd_inner <id> <mode>
    local id="$1" mode="$2" out="$RES/$1" name cmd
    [ "$(pwd -P)" = "$CANON" ] || die "$id: not at $CANON (pwd -P = $(pwd -P))"
    [ "$PWD" = "$CANON" ] || die "$id: PWD=$PWD, not $CANON"
    [ -f Cargo.toml ] || die "$id: no worktree bound at $CANON"
    [ "$(free_gib)" -ge 60 ] || die "less than 60 GiB free on $ROOT before $id"
    mkdir -p "$out" "$CANON/target"
    RUN_OUT="$out"
    trap 'server_stop "$RUN_OUT/sccache-end.json"' EXIT
    server_start "$CANON/target/.reify-sccache.$$.sock" "$out/server.log" || die "$id: private server failed to start ($(tail -3 "$out/server.log" | tr '\n' ' '))"
    client_env
    grep -q 'storage write check failed' "$out/server.log" && log "$id: WARNING server started read-only"
    extract_plan "$out" || die "$id: extract_plan failed rc=$?"
    log "$id: head $(git rev-parse --short=10 HEAD) toplevel $(git rev-parse --show-toplevel) PWD=$PWD sock=$SOCK"
    printf '{"run":"%s","host":"%s","mode":"%s","head":"%s","target_existed_fingerprints":%s}\n' "$id" "$HOST" "$mode" \
        "$(git rev-parse HEAD)" "$(find target -maxdepth 3 -name .fingerprint -type d 2>/dev/null | grep -q . && echo true || echo false)" >"$out/meta.json"
    env | grep -E '^(CARGO_|RUSTC|SCCACHE_|PWD=|PATH=)' | sort >"$out/env.observed.txt"
    ( . "$out/env.sh" && bash -e "$out/preflight.sh" ) >"$out/preflight.log" 2>&1 || die "$id preflight failed"
    while IFS=$'\t' read -r name cmd; do
        cmd="$(printf '%s' "$cmd" | sed -E 's/timeout --kill-after=60 [0-9]+m //')"
        one_step "$id" "$name" "$cmd" 1
    done <"$out/steps.tsv"
    if [ "$mode" = "all+tests" ]; then
        cmd="$(awk -F'\t' '$1=="debug_nextest_build"{print $2}' "$out/steps.tsv" \
            | sed -E -e 's/timeout --kill-after=60 [0-9]+m //' -e 's/ --no-run / --no-fail-fast /' \
                     -e 's/-E "[^"]*"/-E "package(reify-syntax) | package(reify-config) | package(reify-audit)"/')"
        one_step "$id" tests_in_namespace "timeout --kill-after=60 40m $cmd" 0
    fi
    kill -0 "$SPID" 2>/dev/null && echo alive >"$out/server_alive_at_end" || echo dead >"$out/server_alive_at_end"
    du -sb target 2>/dev/null | cut -f1 >"$out/target_bytes"
}

cmd_run() {  # cmd_run <ID> <mode>: one never-built worktree, the whole leg in one namespace
    local id="$1" mode="$2"
    [ -d "$CANON" ] && [ -z "$(ls -A "$CANON")" ] || die "$CANON missing or not empty"
    mkdir -p "$RES"
    rm -f "$RES/$id.DONE" "$RES/FAILED"
    export WARMEXP_CG="/sys/fs/cgroup$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)"
    [ -r "$WARMEXP_CG/cpu.stat" ] || die "cannot read $WARMEXP_CG/cpu.stat"
    mk_wt "$id"
    [ -e "$ROOT/wt/$id/target" ] && die "$id: target exists in a fresh worktree"
    ns "$ROOT/wt/$id" bash "$HERE/alpha.sh" inner "$id" "$mode" || exit 1
    ss -lxp | grep reify-sccache >"$RES/$id/sccache_listeners_after" || true
    log "$id complete"
    touch "$RES/$id.DONE"
}

# ---- store (workstation) -----------------------------------------------------------------------
cmd_store_start() {
    [ "$HOST" = workstation ] || die "the store lives on the workstation"
    mkdir -p "$STORE_DIR/$STORE_PREFIX" "$RES"
    python3 - "$STORE_DIR/$STORE_PREFIX" <<'PY'
import itertools, os, sys
h = '0123456789abcdef'
for a, b, c in itertools.product(h, h, h):
    os.makedirs(os.path.join(sys.argv[1], a, b, c), exist_ok=True)
PY
    if [ ! -s "$ROOT/store.secret" ]; then umask 077; head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' >"$ROOT/store.secret"; fi
    local pw; pw="$(cat "$ROOT/store.secret")"
    umask 077
    printf 'ALPHA_STORE_URL=http://127.0.0.1:%s\nALPHA_STORE_USER=alpha\nALPHA_STORE_PASS=%s\n' "$STORE_PORT" "$pw" >"$ROOT/store.env"
    printf 'ALPHA_STORE_URL=http://100.107.28.56:%s\nALPHA_STORE_USER=alpha\nALPHA_STORE_PASS=%s\n' "$STORE_PORT" "$pw" >"$ROOT/store.laptop.env"
    systemd-run --user --unit canonalpha-store -p Restart=always -p RestartSec=2 --setenv=RCLONE_CONFIG=/dev/null \
        /home/leo/bin/rclone serve webdav "$STORE_DIR" --addr "127.0.0.1:$STORE_PORT" --addr "100.107.28.56:$STORE_PORT" \
        --user alpha --pass "$pw" --dir-cache-time 5s --log-file "$RES/rclone.log" --log-level NOTICE
    sleep 2
    curl -s -o /dev/null -w 'store probe %{http_code}\n' -u "alpha:$pw" "http://127.0.0.1:$STORE_PORT/$STORE_PREFIX/"
}
cmd_store_stop() { systemctl --user stop canonalpha-store; }

# ---- smoke ---------------------------------------------------------------------------------------
cmd_smoke() {
    local t="$ROOT/smoke"; rm -rf "$t"; mkdir -p "$t/ws/src" "$RES/smoke"
    cat >"$t/ws/Cargo.toml" <<'EOF'
[package]
name = "alpha_smoke"
version = "0.1.0"
edition = "2021"
EOF
    printf 'pub fn here() -> &%sstatic str { env!("CARGO_MANIFEST_DIR") }\n#[test] fn t() { assert!(here().starts_with("%s")); }\n' "'" "$CANON" >"$t/ws/src/lib.rs"
    ns "$t/ws" bash -c '
        . "'"$HERE"'/alpha.sh" lib
        echo "pwd -P=$(pwd -P) PWD=$PWD"
        server_start "'"$CANON"'/target/.reify-sccache.$$.sock" "'"$RES"'/smoke/server.log" || { echo START FAILED; tail -5 "'"$RES"'/smoke/server.log"; exit 3; }
        client_env
        RUSTC_WRAPPER=sccache CARGO_INCREMENTAL=0 cargo test --offline -q 2>&1 | tail -3
        server_stop "'"$RES"'/smoke/stats.json"
        python3 -c "import json;s=json.load(open(\"'"$RES"'/smoke/stats.json\"));st=s[\"stats\"];print(\"location\",s.get(\"cache_location\"),\"hits\",st[\"cache_hits\"][\"counts\"],\"misses\",st[\"cache_misses\"][\"counts\"],\"writes\",st[\"cache_writes\"],\"write_errors\",st[\"cache_write_errors\"])"
        echo "sockets left: $(ls -A "'"$CANON"'/target/" | grep -c sock)"'
    echo "listeners left: $(ss -lxp | grep -c reify-sccache)"
}

# ---- burst: n servers at once against the store (PRD boundary row 16, C5) -----------------------
cmd_burst() {  # cmd_burst <n> <label> [start_epoch]
    local n="$1" label="$2" at="${3:-}" d="$RES/burst-$2" i
    mkdir -p "$d"; store_env
    if [ -n "$at" ]; then while [ "$(date +%s)" -lt "$at" ]; do sleep 0.05; done; fi
    for i in $(seq 1 "$n"); do
        ( exec env -u SCCACHE_DIR SCCACHE_START_SERVER=1 SCCACHE_NO_DAEMON=1 SCCACHE_IDLE_TIMEOUT=0 \
            SCCACHE_SERVER_UDS="$d/s$i.sock" SCCACHE_WEBDAV_ENDPOINT="$ALPHA_STORE_URL" SCCACHE_WEBDAV_KEY_PREFIX="${BURST_PREFIX:-$STORE_PREFIX}" \
            SCCACHE_WEBDAV_USERNAME="$ALPHA_STORE_USER" SCCACHE_WEBDAV_PASSWORD="$ALPHA_STORE_PASS" sccache ) </dev/null >/dev/null 2>"$d/s$i.err" &
        echo $! >"$d/s$i.pid"
    done
    date +%s.%N >"$d/started"
    for i in $(seq 1 "$n"); do
        local k; for k in $(seq 1 300); do [ -S "$d/s$i.sock" ] && break; sleep 0.1; done
    done
    # one unique C compile through each server: a read-write server writes it, a read-only one cannot
    for i in $(seq 1 "$n"); do
        printf 'int alpha_burst_%s_%s_%s(void){return %s;}\n' "$HOST" "${label//[^A-Za-z0-9]/_}" "$i" "$i" >"$d/u$i.c"
        if [ -S "$d/s$i.sock" ]; then
            SCCACHE_SERVER_UDS="$d/s$i.sock" sccache cc -c "$d/u$i.c" -o "$d/u$i.o" 2>>"$d/s$i.client.err"
            sleep 1   # the entry is written after the compile returns
            SCCACHE_SERVER_UDS="$d/s$i.sock" sccache --show-stats --stats-format json >"$d/s$i.stats.json" 2>/dev/null
        fi
    done
    for i in $(seq 1 "$n"); do
        [ -S "$d/s$i.sock" ] && SCCACHE_SERVER_UDS="$d/s$i.sock" sccache --stop-server >/dev/null 2>&1
        wait "$(cat "$d/s$i.pid")" 2>/dev/null; rm -f "$d/s$i.sock"
    done
    python3 -I - "$d" "$n" "$HOST" "$label" <<'PY'
import json, os, sys
d, n, host, label = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
up = ro_log = rw_writes = ro_by_write = 0
rows = []
for i in range(1, n + 1):
    err = open(os.path.join(d, f's{i}.err')).read()
    ro = 'storage write check failed' in err
    try:
        st = json.load(open(os.path.join(d, f's{i}.stats.json')))['stats']
        up += 1
    except (OSError, ValueError):
        st = None
    w = st['cache_writes'] if st else None
    we = st['cache_write_errors'] if st else None
    ro_log += ro
    rw_writes += bool(w)
    ro_by_write += (st is not None and not w)
    rows.append({'i': i, 'readonly_log': ro, 'writes': w, 'write_errors': we,
                 'err_tail': err.strip().splitlines()[-1][:200] if err.strip() else ''})
summary = {'host': host, 'label': label, 'started': n, 'answered': up, 'readonly_by_log': ro_log,
           'wrote_entry': rw_writes, 'no_write': ro_by_write, 'rows': rows}
json.dump(summary, open(os.path.join(d, 'summary.json'), 'w'), indent=1)
print(json.dumps({k: v for k, v in summary.items() if k != 'rows'}))
PY
}

# ---- agent: one real sandboxed Claude session at CANON (PRD decision 7, C8) ----------------------
cmd_agent() {  # cmd_agent <worktree> <crate>
    local lane="$1" crate="$2" d="$RES/agent" cfg
    [ "$HOST" = workstation ] || die "agent runs on the workstation"
    mkdir -p "$d" "$lane/.task"
    cfg="$lane/.task/claude-config-alpha"
    mkdir -p "$cfg"
    ln -sf /home/leo/.claude/settings.json "$cfg/settings.json"
    python3 -I - "$cfg/.credentials.json" <<'PY'
import json, os, sys
# As TaskConfigDir.write_credentials does for an agent: the pool account's token as accessToken.
tok = next(l.split('=', 1)[1].strip().strip('"\'') for l in open('/home/leo/src/dark-factory/.env')
           if l.strip().startswith('CLAUDE_OAUTH_TOKEN_F='))
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
os.write(fd, json.dumps({'claudeAiOauth': {'accessToken': tok}}).encode()); os.close(fd)
PY
    local prompt="You are in a reify git worktree. Its build directory is used at the canonical path $CANON (already bound in this session; do not create or bind anything). Run exactly this shell command and nothing else, in one Bash tool call with its timeout parameter set to 1800000 (it compiles for several minutes): cd $CANON/crates/$crate && cargo test --offline -p $crate 2>&1 | tail -40 ; echo CARGO_EXIT=\${PIPESTATUS[0]} . Then reply with one line: PASS or FAIL, the number of tests passed and failed from cargo's summary lines, and the CARGO_EXIT value."
    log "agent: lane=$lane crate=$crate"
    # The agent's environment: the stand-in trampoline first on PATH, then ~/.cargo/bin (PRD C4, C8).
    ( cd "$lane" && env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" LANG=C.UTF-8 TERM=dumb \
        PATH="$HERE/agent-bin:$HOME/.cargo/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin" \
        CARGO_HOME=/tmp/reify-agent-cargo-home CLAUDE_CONFIG_DIR="$cfg" \
        CLAUDE_CODE_OAUTH_TOKEN="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["claudeAiOauth"]["accessToken"])' "$cfg/.credentials.json")" \
        ALPHA_CANON="$CANON" ALPHA_STORE_ENV="$ROOT/store.env" ALPHA_JOURNAL="$d/journal.jsonl" ALPHA_FALLBACK="/tmp/canon-alpha-fallback" \
        /usr/bin/bwrap --dev-bind / / --bind "$lane" "$CANON" --die-with-parent -- \
        python3 /home/leo/src/dark-factory/orchestrator/src/orchestrator/agents/landlock_exec.py \
            --writable "$lane/.task" --writable "$lane" --writable "$d" -- \
        claude -p --model claude-sonnet-5-5 --dangerously-skip-permissions --output-format json --max-turns 8 "$prompt"
    ) >"$d/session.json" 2>"$d/session.err"
    echo "agent rc=$?" | tee -a "$d/rc.txt"
    python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("result","")[:2000]); print("is_error", d.get("is_error"), "turns", d.get("num_turns"), "cost", d.get("total_cost_usd"))' "$d/session.json" 2>&1 | tee "$d/result.txt"
    cat "$d/journal.jsonl" 2>/dev/null
}

cmd_setup() {
    mkdir -p "$RES" "$ROOT/wt" "$CANON"
    [ -d "$ROOT/repo" ] && die "$ROOT/repo already exists"
    git -C "$SRC" cat-file -e "$X^{commit}" || die "X not in $SRC"
    git clone --shared --no-checkout "$SRC" "$ROOT/repo" >>"$RES/alpha.log" 2>&1 || die "clone failed"
    git -C "$ROOT/repo" update-ref refs/heads/main "$MAIN" || die "update-ref main failed"
    log "setup done: X=$X MAIN=$MAIN CANON=$CANON"
}

cmd_clean() {
    [ -d "$ROOT" ] || exit 0
    rm -rf "$ROOT/wt" "$ROOT/repo" "$ROOT/smoke"
    echo "kept: $RES"
}

case "${1:-}" in
    lib) ;;   # sourced for its functions
    setup) cmd_setup ;;
    smoke) cmd_smoke ;;
    store-start) cmd_store_start ;;
    store-stop) cmd_store_stop ;;
    run) shift; cmd_run "$@" ;;
    inner) shift; cmd_inner "$@" ;;
    burst) shift; cmd_burst "$@" ;;
    agent) shift; cmd_agent "$@" ;;
    clean) cmd_clean ;;
    *) echo "usage: alpha.sh setup|smoke|store-start|store-stop|run|burst|agent|clean" >&2; exit 64 ;;
esac
