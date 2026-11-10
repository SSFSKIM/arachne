#!/usr/bin/env bash
# The truth oracle (core/oracle, its own Cargo package): formatting, clippy (the core's clippy.toml
# applies: no fused multiply-add, no platform transcendental functions), the lexical independence
# check (ci/oracle_check.py, a fast first pass: it names no core module but the XF schema, and its
# tree holds no GPU crate), then the proof: ci/oracle_isolated.py builds and tests the oracle in a
# tree whose only core source is the XF schema, and shows two escaped reaches into the core fail to
# compile there. Test files the contract lets reach the core run in the normal tree.
# `oracle.sh test` runs the tests alone, in the normal tree.
set -euo pipefail
source "$(dirname "$0")/toolchain.sh"
cd oracle
if [ "${1:-}" = test ]; then
  cargo test --release --locked
  exit
fi
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
python3 ../ci/oracle_check.py
python3 ../ci/oracle_isolated.py
python3 ../ci/oracle_isolated.py --plant path
python3 ../ci/oracle_isolated.py --plant include
shopt -s nullglob
for t in tests/contract*.rs; do
  cargo test --release --locked --test "$(basename "$t" .rs)"
done
