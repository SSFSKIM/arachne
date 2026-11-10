#!/usr/bin/env bash
# The truth oracle (core/oracle, its own Cargo package): formatting, clippy (the core's clippy.toml
# applies: no fused multiply-add, no platform transcendental functions), its tests, and its
# independence (ci/oracle_check.py: it names no core module but the XF schema, and its tree holds no
# GPU crate). `oracle.sh test` runs the tests alone.
set -euo pipefail
source "$(dirname "$0")/toolchain.sh"
cd oracle
if [ "${1:-}" = test ]; then
  cargo test --release --locked
  exit
fi
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --release --locked
python3 ../ci/oracle_check.py
