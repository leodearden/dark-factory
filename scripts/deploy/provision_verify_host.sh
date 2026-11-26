#!/usr/bin/env bash
set -euo pipefail

# provision_verify_host.sh
#
# Converge a remote merge-verify host onto the verify-host contract
# (plans/verify-host-roster-and-rented-trial-prd.md §6.1; SETUP.md §12).
#
# The steps ARE the contract table: their ids come from
# `verify_host_contract.py requirements`, one step_<id> function per id, and
# the script refuses to run when the two sets differ. Each step checks, then
# converges what it can; what no script can change (Landlock, disk, an existing
# credential file) is left to the closing check_verify_host.py run, whose exit
# code a full run returns. Everything host-specific (names, users, paths) comes
# from the private verify-host roster; nothing here names a machine.
#
# Every remote step runs as `ssh <host> bash -s -- ARGS` with its script on
# stdin (expanded on the host). Privilege goes through `sudo -n` only: a
# refusal stops the step and prints the command to run by hand.
#
# Usage:
#   provision_verify_host.sh HOST [--roster FILE] [--dry-run | --sync-config]
#   provision_verify_host.sh --list-steps
#     (default)      every step in table order, then check_verify_host.py
#     --dry-run      print each step and the resolved host parameters; no ssh
#     --sync-config  install only the per-host YAML files (after a roster edit)
#     --list-steps   print the step ids, which are the contract's requirement ids
#
# Env overrides:
#   VERIFY_HOST_PYTHON        interpreter for the contract (default: this checkout's .venv)
#   VERIFY_HOST_SSH           ssh binary (default: ssh)
#   VERIFY_HOST_WRAPPER_PATH  the wrapper on the host (default: /usr/local/bin/orchestrator)
#   ORCH_VERIFY_HOSTS_FILE    the roster (default: $XDG_CONFIG_HOME/orchestrator/verify-hosts.yaml)

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="$(cd "$HERE/../.." && pwd)"
PY="${VERIFY_HOST_PYTHON:-$CHECKOUT/.venv/bin/python}"
CONTRACT="$HERE/verify_host_contract.py"
CHECKER="$HERE/check_verify_host.py"
SWAPFILE=/swapfile
SWAPFILE_SIZE_GIB=8
DF_MAIN_BRANCH=main
CURRENT_STEP=setup

die() { echo "[provision_verify_host] ERROR: $*" >&2; exit 1; }

usage() {
    [ $# -eq 0 ] || echo "[provision_verify_host] $*" >&2
    echo "Usage: $(basename "$0") HOST [--roster FILE] [--dry-run | --sync-config]" >&2
    echo "       $(basename "$0") --list-steps" >&2
    exit 2
}

# Functions every remote step's script can call; sent ahead of it on stdin.
read -r -d '' REMOTE_HELPERS <<'HELPERS' || true
say() { echo "[$STEP@$(hostname)] $*"; }
fail() { echo "[$STEP@$(hostname)] ERROR: $*" >&2; exit 1; }
sudo_n() {
    sudo -n "$@" && return 0
    local by_hand; printf -v by_hand '%q ' "$@"
    fail "sudo -n refused; run by hand on the host: sudo $by_hand"
}
apt_install() {
    local wanted=() package
    for package in "$@"; do
        [ "$(dpkg-query -W -f '${Status}' "$package" 2>/dev/null)" = 'install ok installed' ] || wanted+=("$package")
    done
    [ ${#wanted[@]} -eq 0 ] && return 0
    sudo_n env DEBIAN_FRONTEND=noninteractive apt-get update -q
    sudo_n env DEBIAN_FRONTEND=noninteractive apt-get install -y -q "${wanted[@]}"
}
HELPERS

remote() {
    {
        printf 'set -euo pipefail\nexport %s\nSTEP=%q\n' "$TOOL_PATH_PRELUDE" "$CURRENT_STEP"
        printf '%s\n' "$REMOTE_HELPERS"
        cat
    } | "$SSH_BIN" "${SSH_OPTS[@]}" "$SSH_HOST" bash -s -- "${@@Q}"
}

step_network() {
    "$SSH_BIN" "${SSH_OPTS[@]}" "$SSH_HOST" true \
        || die "ssh $SSH_HOST failed with the dispatcher's ssh options: add the ssh alias and pin the host key, then re-run"
}

step_os_and_user() {
    remote "$HOST_NAME" "$REQUIRED_OS_ID" "$REQUIRED_OS_VERSION_ID" <<'REMOTE'
host_name="$1"; os_id="$2"; os_version_id="$3"
[ "$(id -u)" != 0 ] || fail "refusing to provision as root; ssh in as the verify user"
. /etc/os-release
[ "$ID" = "$os_id" ] && [ "$VERSION_ID" = "$os_version_id" ] || fail "needs $os_id $os_version_id, found $ID $VERSION_ID"
apt_install git bash util-linux curl ca-certificates xz-utils python3 locales
git config --global user.name >/dev/null || git config --global user.name "$host_name verify host"
git config --global user.email >/dev/null || git config --global user.email "$(id -un)@$host_name"
say "$ID $VERSION_ID; user $(id -un); git identity set"
REMOTE
}

step_time_zone_and_locale() {
    [ -n "$REF_TIME_ZONE" ] || die "this workstation's time zone is unreadable (timedatectl), so there is nothing to match"
    remote "$REF_TIME_ZONE" "$REF_LOCALE" <<'REMOTE'
time_zone="$1"; locale="$2"
apt_install tzdata-legacy locales
[ "$(timedatectl show -p Timezone --value)" = "$time_zone" ] || sudo_n timedatectl set-timezone "$time_zone"
case "$locale" in
    ''|C|POSIX) ;;
    *)
        normalise() { tr 'A-Z' 'a-z' | tr -d -- -; }
        available="$(locale -a | normalise)"
        grep -qxF -- "$(printf '%s' "$locale" | normalise)" <<<"$available" || sudo_n locale-gen "$locale"
        [ "$( (. /etc/default/locale 2>/dev/null; echo "${LANG:-}") )" = "$locale" ] || sudo_n update-locale "LANG=$locale"
        ;;
esac
say "time zone $time_zone; locale ${locale:-C}; tzdata-legacy installed"
REMOTE
}

step_toolchain() {
    [ -n "$REF_UV_VERSION" ] && [ -n "$REF_NODE_VERSION" ] \
        || die "this workstation's uv or node version is unreadable, so there is nothing to match"
    remote "$REF_UV_VERSION" "$REF_NODE_VERSION" <<'REMOTE'
uv_version="$1"; node_version="$2"
mkdir -p "$HOME/.local/bin"
if [ "$(uv --version 2>/dev/null | cut -d' ' -f2)" != "$uv_version" ]; then
    curl -LsSf "https://astral.sh/uv/$uv_version/install.sh" | env UV_NO_MODIFY_PATH=1 sh
fi
uv python install 3.13
if [ "$("$HOME/.local/node/bin/node" --version 2>/dev/null)" != "$node_version" ]; then
    case "$(uname -m)" in
        x86_64) node_arch=x64 ;;
        aarch64) node_arch=arm64 ;;
        *) fail "nodejs.org ships no linux build this step installs for architecture $(uname -m) (x86_64 and aarch64 only)" ;;
    esac
    base="https://nodejs.org/dist/$node_version"
    tarball="node-$node_version-linux-$node_arch.tar.xz"
    scratch="$(mktemp -d)"
    trap 'rm -rf "$scratch"' EXIT
    curl -fsSL "$base/$tarball" -o "$scratch/$tarball"
    sums="$(curl -fsSL "$base/SHASUMS256.txt")"
    grep -- " $tarball\$" <<<"$sums" | (cd "$scratch" && sha256sum -c --quiet -)
    rm -rf "$HOME/.local/node"
    mkdir -p "$HOME/.local/node"
    tar -xJf "$scratch/$tarball" -C "$HOME/.local/node" --strip-components=1
fi
for tool in node npm npx; do
    ln -sfn "$HOME/.local/node/bin/$tool" "$HOME/.local/bin/$tool"
done
say "uv $(uv --version | cut -d' ' -f2); node $(node --version); CPython 3.13 at $(uv python find 3.13)"
REMOTE
}

step_systemd() {
    remote <<'REMOTE'
user="$(id -un)"
if [ "$(loginctl show-user "$user" -p Linger --value 2>/dev/null)" != yes ]; then
    loginctl enable-linger "$user" 2>/dev/null || sudo_n loginctl enable-linger "$user"
fi
say "linger on for $user"
REMOTE
}

step_kernel() {
    remote <<'REMOTE'
[ -w /var/tmp ] || sudo_n chmod 1777 /var/tmp
say "/var/tmp writable; Landlock cannot be provisioned, the closing check reports it"
REMOTE
}

step_memory() {
    remote "$SWAPFILE" "$SWAPFILE_SIZE_GIB" <<'REMOTE'
swapfile="$1"; size_gib="$2"
swap_kib="$(awk '$1 == "SwapTotal:" { print $2 }' /proc/meminfo)"
if [ "${swap_kib:-0}" = 0 ]; then
    [ -e "$swapfile" ] || { sudo_n fallocate -l "${size_gib}G" "$swapfile"; sudo_n chmod 600 "$swapfile"; sudo_n mkswap "$swapfile"; }
    sudo_n swapon "$swapfile"
    fstab="$(cat /etc/fstab)"
    grep -qF -- "$swapfile none swap" <<<"$fstab" || echo "$swapfile none swap sw 0 0" | sudo_n tee -a /etc/fstab >/dev/null
fi
say "swap $(awk '$1 == "SwapTotal:" { print $2 }' /proc/meminfo) KiB"
REMOTE
}

step_disk() {
    echo "[disk] nothing to converge; the closing check reports free space and synchronous-write latency"
}

step_checkouts() {
    local checkout_branch_pairs=("$DF_CHECKOUT_PATH" "$DF_MAIN_BRANCH") i
    for i in "${!PROJECT_ROOTS[@]}"; do
        checkout_branch_pairs+=("${PROJECT_ROOTS[$i]}" "${PROJECT_MAIN_BRANCHES[$i]}")
    done
    remote "${checkout_branch_pairs[@]}" <<'REMOTE'
checkouts=0
while [ $# -gt 0 ]; do
    checkout="$1"; branch="$2"; shift 2
    mkdir -p "$checkout"
    [ -e "$checkout/.git" ] || git init -q -b "$branch" "$checkout"
    git -C "$checkout" config receive.denyCurrentBranch updateInstead
    checkouts=$((checkouts + 1))
done
say "$checkouts checkouts accept pushes to their branch"
REMOTE
    GIT_SSH_COMMAND="$SSH_BIN ${SSH_OPTS[*]}" \
        git -C "$CHECKOUT" push --quiet "$SSH_HOST:$DF_CHECKOUT_PATH" "$DF_MAIN_BRANCH:refs/heads/$DF_MAIN_BRANCH" \
        || die "pushing this workstation's $DF_MAIN_BRANCH to $SSH_HOST:$DF_CHECKOUT_PATH failed"
    remote "$DF_CHECKOUT_PATH" <<'REMOTE'
cd "$1"
uv sync --all-packages
npm ci
say "$1 synced; its venv carries the orchestrator entry point"
REMOTE
}

step_wrapper() {
    remote "$WRAPPER_PATH" "$(base64 -w0 <"$WORK/wrapper.sh")" <<'REMOTE'
target="$1"
staged="$(mktemp)"
trap 'rm -f "$staged"' EXIT
printf '%s' "$2" | base64 -d >"$staged"
if cmp -s "$staged" "$target"; then
    say "$target unchanged"
    exit 0
fi
bash -n "$staged" || fail "the rendered wrapper does not parse; nothing was written"
dir="$(dirname "$target")"
mkdir -p "$dir" 2>/dev/null || sudo_n mkdir -p "$dir"
if [ -w "$dir" ]; then priv=(); else sudo_n true; priv=(sudo -n); fi
new="$dir/.$(basename "$target").new.$$"
{ "${priv[@]}" install -m 755 "$staged" "$new" && "${priv[@]}" mv -f -- "$new" "$target"; } \
    || { "${priv[@]}" rm -f -- "$new" 2>/dev/null || true; fail "installing $target failed; it is unchanged"; }
probe="$(ORCH_GATE_BIN=/bin/echo "$target" probe 2>/dev/null)" || fail "$target failed with ORCH_GATE_BIN=/bin/echo"
[ "$probe" = probe ] || fail "$target ignores ORCH_GATE_BIN (printed '$probe')"
say "installed $target"
REMOTE
}

step_per_host_yaml() {
    local files=() i
    for i in "${!PROJECT_IDS[@]}"; do
        files+=("${CONFIG_PATHS[$i]}" "$(base64 -w0 <"$WORK/${PROJECT_IDS[$i]}.yaml")")
    done
    remote "${files[@]}" <<'REMOTE'
while [ $# -gt 0 ]; do
    target="$1"; content="$2"; shift 2
    mkdir -p "$(dirname "$target")"
    staged="$(mktemp "$(dirname "$target")/.$(basename "$target").XXXXXX")"
    printf '%s' "$content" | base64 -d >"$staged"
    if cmp -s "$staged" "$target"; then
        rm -f "$staged"
        say "$target unchanged"
    else
        mv -f "$staged" "$target"
        say "wrote $target"
    fi
done
REMOTE
}

step_no_credentials() {
    echo "[no_credentials] nothing to converge (a credential file is never deleted); the closing check reports one"
}

self_check() {
    local defined expected orphans missing
    defined="$(declare -F | awk '$3 ~ /^step_/ { print substr($3, 6) }' | sort)"
    expected="$(printf '%s\n' "${STEPS[@]}" | sort)"
    orphans="$(comm -23 <(printf '%s\n' "$defined") <(printf '%s\n' "$expected") | xargs)"
    missing="$(comm -13 <(printf '%s\n' "$defined") <(printf '%s\n' "$expected") | xargs)"
    if [ -n "$orphans" ] || [ -n "$missing" ]; then
        echo "[provision_verify_host] ERROR: the steps differ from the contract table:" \
            "step functions with no requirement: [${orphans}]; requirements with no step: [${missing}]" >&2
        exit 2
    fi
}

dry_run() {
    local titles id title i
    titles="$("$PY" "$CONTRACT" requirements --titles)"
    echo "host $HOST_NAME via ssh $SSH_HOST (wrapper $WRAPPER_PATH)"
    echo "dark factory checkout $DF_CHECKOUT_PATH (gate checkouts $GATE_CHECKOUTS_PATH)"
    for i in "${!PROJECT_IDS[@]}"; do
        echo "project ${PROJECT_IDS[$i]} checkout ${PROJECT_ROOTS[$i]} on ${PROJECT_MAIN_BRANCHES[$i]} (config ${CONFIG_PATHS[$i]})"
    done
    while IFS=$'\t' read -r id title; do
        echo "$id: $title"
    done <<<"$titles"
}

main() {
    local mode=full host='' roster_args=() status=0
    while [ $# -gt 0 ]; do
        case "$1" in
            --list-steps|--dry-run|--sync-config)
                [ "$mode" = full ] || usage "choose one of --list-steps, --dry-run and --sync-config"
                mode="${1#--}" ;;
            --roster)
                [ $# -ge 2 ] || usage "--roster needs a file"
                roster_args=(--roster "$2"); shift ;;
            -*) usage "unknown option: $1" ;;
            *)
                [ -z "$host" ] || usage "one host at a time"
                host="$1" ;;
        esac
        shift
    done

    local steps_text
    steps_text="$("$PY" "$CONTRACT" requirements)" || die "could not read the contract's requirement ids"
    mapfile -t STEPS <<<"$steps_text"
    self_check
    if [ "$mode" = list-steps ]; then
        printf '%s\n' "${STEPS[@]}"
        return 0
    fi
    [ -n "$host" ] || usage "name the host to provision"

    WORK="$(mktemp -d)"
    trap 'rm -rf "$WORK"' EXIT
    "$PY" "$CONTRACT" render "$host" "${roster_args[@]}" --out "$WORK" || exit 1
    source "$WORK/host.env"

    case "$mode" in
        dry-run) dry_run ;;
        sync-config) CURRENT_STEP=per_host_yaml; step_per_host_yaml ;;
        full)
            for CURRENT_STEP in "${STEPS[@]}"; do
                echo "== $CURRENT_STEP"
                "step_$CURRENT_STEP"
            done
            "$PY" "$CHECKER" "$host" "${roster_args[@]}" || status=$?
            return "$status"
            ;;
    esac
}

main "$@"
