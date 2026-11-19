#!/usr/bin/env bash
# The simulation crate's static checks: formatting; clippy (whose configuration forbids fused
# multiply-add and the platform's transcendental functions in this crate's own code); the checks'
# own tests; the dependency tree (no GPU crate, no arachne-core, simba with libm_force); the census
# of run-time CPU detection; and the import list of the probe, built with the shipped features.
set -euo pipefail
source "$(dirname "$0")/toolchain.sh"
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
python3 ci/selftest.py
python3 ci/tree_check.py deps
python3 ci/census.py
cargo build --release --locked --example probe
if ! python3 ci/imports.py target/release/examples/probe; then
  # Where each call comes from: the same build with line tables.
  cargo build --profile audit --locked --example probe
  python3 ci/imports.py --where target/audit/examples/probe
  exit 1
fi
