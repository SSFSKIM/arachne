#!/usr/bin/env bash
# The truth oracle (core/oracle, its own Cargo package): formatting, clippy (the core's clippy.toml
# applies: no fused multiply-add, no platform transcendental functions), the lexical independence
# check (ci/oracle_check.py, a courtesy pass: it names no core module but the XF schema, and its
# tree holds no GPU crate), the plant runner's own tests (ci/test_oracle_plants.py: a plant that
# does not apply fails the run), the plants' anchors (oracle_plants.py --anchors: each plant's text
# occurs exactly once in its file, read without a build), then, outside CI, ci/oracle_isolated.py:
# the oracle built and tested in a directory git archive fills with the isolated checkout's list
# alone, and five plants reaching for core files that fail to build because the files are absent.
# Beside a full checkout that is not the structural proof (an absolute reach into this checkout
# would build); the proof is the gate core-oracle-isolated, whose job prunes its own checkout to the
# list. Test files the contract lets reach the core run in the normal tree. `oracle.sh test` runs
# the tests alone, in the normal tree.
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
python3 ../ci/test_oracle_plants.py
python3 ../ci/oracle_plants.py --anchors
if [ -z "${CI:-}" ]; then
  python3 ../ci/oracle_isolated.py --from-git HEAD
fi
shopt -s nullglob
for t in tests/contract*.rs tests/probes.rs; do
  cargo test --release --locked --test "$(basename "$t" .rs)"
done
