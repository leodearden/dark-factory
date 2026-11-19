#!/usr/bin/env bash
# build_cache_drift_probe.sh [<reify checkout>]
#
# Prints one "name<TAB>digest-or-value" line per input that reaches an sccache
# key for reify's cached compiles (PRD C13,
# plans/canonical-build-path-shared-cache-prd.md). Run it on both hosts
# against their reify checkouts and diff the two outputs: zero differences
# outside the #info lines are required before gates iota and kappa share one
# store between the hosts. It is read-only and builds nothing.
#
# cc1, cc1plus and as are printed although sccache hashes only the compiler
# driver (PRD decision 20): a host whose driver matches but whose cc1 differs
# would serve the other host's objects. dpkg.dev-packages and
# pkg-config.modules catch build-script outcomes such as x11-dl's baked
# pkg-config result, which a -dev package installed on one host only changes
# (libxss-dev on the laptop, esc-10782-1).
#
# The checkout defaults to $REIFY_ROOT, else $HOME/src/reify; exit 2 if it is
# not a directory. REIFY_PROBE_DEEP=1 adds the header-closure digest of the
# two hand-written C++ wrappers (cc -M, preprocess only).
set -u
export LC_ALL=C
R="${1:-${REIFY_ROOT:-$HOME/src/reify}}"
if [ ! -d "$R" ]; then
  echo "build_cache_drift_probe: $R is not a directory (the reify checkout)" >&2
  exit 2
fi
cd "$R" || exit 2   # rust-toolchain.toml pins 1.96.0 only under the repo; elsewhere rustup serves the default toolchain
dig() { sha256sum | cut -c1-16; }
emit() { printf '%s\t%s\n' "$1" "${2//$'\n'/ }"; }
filehash() { [ -e "$1" ] && sha256sum "$(readlink -f "$1")" | cut -c1-16 || echo ABSENT; }

# ---- Rust side (rustc binary + sysroot are key inputs; std rlibs ship with the sysroot)
RUSTC=${RUSTC:-$(rustup which rustc 2>/dev/null || command -v rustc)}
emit rustc.vV "$($RUSTC -vV | grep -E '^(release|commit-hash|host|LLVM)' | tr '\n' ' ')"
SYS=$($RUSTC --print sysroot)
emit rustc.bin "$(filehash "$RUSTC")"
emit sysroot.lib "$(find "$SYS/lib" -maxdepth 1 -type f \( -name '*.so' -o -name '*.rlib' \) -print0 | sort -z | xargs -0 sha256sum | cut -d' ' -f1 | dig)"
emit sysroot.rustlib "$(find "$SYS/lib/rustlib/x86_64-unknown-linux-gnu/lib" -type f -print0 | sort -z | xargs -0 sha256sum | cut -d' ' -f1 | dig)"
emit toolchain.pin "$(grep -E '^channel' "$R/rust-toolchain.toml")"

# ---- C/C++ driver chain (sccache hashes the driver; cc1/cc1plus/as are NOT in the key)
for t in cc c++; do
  emit "$t.driver" "$(filehash "$(command -v $t)")"
  emit "$t.version" "$($t -dumpfullversion -dumpmachine | tr '\n' ' ')"
done
emit cc1 "$(filehash "$(cc -print-prog-name=cc1)")"
emit cc1plus "$(filehash "$(c++ -print-prog-name=cc1plus)")"
emit as "$(filehash "$(command -v as)")"
emit dpkg.toolchain "$(dpkg-query -W -f '${Package}=${Version} ' libc6-dev libstdc++-13-dev libgcc-13-dev linux-libc-dev g++-13 gcc-13 cpp-13 binutils 2>&1)"
dpkg_status="$(dpkg-query -W -f '${db:Status-Abbrev} ${Package}=${Version}\n' 2>/dev/null)"
emit dpkg.dev-packages "$(printf '%s\n' "$dpkg_status" | awk '$1 == "ii" && $2 ~ /^[^=]*-dev=/ { print $2 }' | sort | dig)"

# ---- headers that reach preprocessed output
emit dpkg.occt "$(dpkg-query -W -f '${Package}=${Version}\n' 'libocct*' 2>/dev/null | sort | dig)"
emit occt.include.tree "$(find /usr/include/opencascade -type f -print0 2>/dev/null | sort -z | xargs -0 sha256sum | cut -d' ' -f1 | dig)"
emit reify-deps.conda-meta "$(ls /opt/reify-deps/conda-meta 2>/dev/null | grep -v '^history$' | dig)"
emit reify-deps.static-archives "$(find /opt/reify-deps/lib /opt/reify-deps/manifold/lib -maxdepth 1 -name '*.a' -print0 2>/dev/null | sort -z | xargs -0 sha256sum | cut -d' ' -f1 | dig)"
if [ "${REIFY_PROBE_DEEP:-0}" = 1 ]; then
  for w in occt:/usr/include/opencascade openvdb:/opt/reify-deps/include; do
    k=${w%%:*}; inc=${w#*:}
    emit "deps.$k.header-closure" "$(c++ -std=c++17 -M -MG -I"$inc" -isystem /opt/reify-deps/include -I"$R/crates/reify-kernel-$k/cpp" \
       "$R/crates/reify-kernel-$k/cpp/${k}_wrapper.cpp" 2>/dev/null | tr -d '\\ ' | tr '\n' ' ' | tr ' ' '\n' | grep '^/' | sort -u | xargs sha256sum 2>/dev/null | cut -d' ' -f1 | dig)"
  done
fi

# ---- build-script outcomes that become rustc-cfg / rustc-env (key inputs)
emit native-dep.resolution "$(python3 -I - "$R/crates/reify-build-utils/native-deps.toml" <<'EOF'
import sys, os, tomllib
t = tomllib.load(open(sys.argv[1], 'rb'))
out = []
for dep, row in sorted(t.items()):
    for kind in ('include', 'lib'):
        ov = os.environ.get(row[f'{kind}_env'], '')
        first = next((d for d in row[f'{kind}_candidates'] if os.path.exists(os.path.join(d, row[f'{kind}_sentinel']))), None)
        out.append(f'{dep}.{kind}={ov or first}')
print(' '.join(out))
EOF
)"
slvs_version="$(pkg-config --modversion slvs 2>&1)"
emit pkg-config.slvs "${slvs_version%%$'\n'*}; header=$(test -e /usr/include/slvs.h && echo y || echo n)"
pc_modules="$(pkg-config --list-all 2>/dev/null)"
emit pkg-config.modules "$(printf '%s\n' "$pc_modules" | while read -r m _; do
  [ -z "$m" ] || printf '%s=%s\n' "$m" "$(pkg-config --modversion "$m" 2>&1)"
done | sort | dig)"
emit tree-sitter.cli "$(command -v tree-sitter >/dev/null && { tree-sitter --version; filehash "$(command -v tree-sitter)"; } | tr '\n' ' ' || echo ABSENT)"

# ---- ambient knobs that must be absent or equal
emit env.cflags "$(env | grep -E '^(CC|CXX|CFLAGS|CXXFLAGS|CPPFLAGS|LDFLAGS|RUSTFLAGS|CARGO_ENCODED_RUSTFLAGS|CARGO_BUILD_RUSTFLAGS|CARGO_TARGET_[A-Z0-9_]*_RUSTFLAGS|RUSTC_WRAPPER|RUSTC_WORKSPACE_WRAPPER|RUSTDOCFLAGS|HOST_C[A-Z]*|TARGET_C[A-Z]*|CRATE_CC_[A-Z_]*|CC_[A-Za-z0-9_]*|CXX_[A-Za-z0-9_]*)=' | sort | dig)"
emit env.cargo-vars "$(env | grep -E '^CARGO_' | grep -v -E '^CARGO_(MAKEFLAGS|BUILD_JOBS|REGISTRIES_)' | sort | tr '\n' ' ')"
emit env.read-by-env-macro "$(env | grep -E '^(ICU4X_DATA_DIR|TAURI_[A-Z_]*|WRY_[A-Z_]*|MATMUL_[A-Z_]*|RUST_LIBC_UNSTABLE_[A-Z_]*|RUSTC_BOOTSTRAP|SOURCE_DATE_EPOCH|ZSTD_SYS_[A-Z_]*|LIBSQLITE3_[A-Z_]*|PKG_CONFIG[A-Z_]*|SLVS_[A-Z_]*|OCCT_[A-Z_]*|GMSH_[A-Z_]*|OPENVDB_[A-Z_]*)=' | sort | dig)"
emit user.home "$(id -un) $HOME"
emit cargo.home "${CARGO_HOME:-$HOME/.cargo}"
emit cargo.config "$(for f in "${CARGO_HOME:-$HOME/.cargo}"/config.toml "${CARGO_HOME:-$HOME/.cargo}"/config "$HOME/.cargo/config.toml" "$HOME/.cargo/config" "$R/.cargo/config.toml"; do [ -e "$f" ] && echo "$f=$(filehash "$f")"; done | tr '\n' ' ')"
emit sccache.version "$(sccache --version)"
# informational only, never compared:
emit "#info.cpu" "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)"
