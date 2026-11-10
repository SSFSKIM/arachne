#!/usr/bin/env bash
# The truth oracle (core/oracle, its own Cargo package): formatting, clippy (the core's clippy.toml
# applies: no fused multiply-add, no platform transcendental functions), the lexical independence
# check (ci/oracle_check.py, a courtesy pass: it names no core module but the XF schema, and its
# tree holds no GPU crate), then ci/oracle_isolated.py: the oracle built and tested in a directory
# git archive fills with the isolated checkout's list alone, and five plants reaching for core files
# that fail to build because the files are absent. Here it runs beside the full checkout, so it is
# not yet the structural proof (an absolute reach into this checkout would build); that is a job
# whose checkout holds the list alone. Test files the contract lets reach the core run in the
# normal tree. `oracle.sh test` runs the tests alone, in the normal tree.
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
python3 ../ci/oracle_isolated.py --from-git HEAD
shopt -s nullglob
for t in tests/contract*.rs; do
  cargo test --release --locked --test "$(basename "$t" .rs)"
done
