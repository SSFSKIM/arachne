#!/usr/bin/env bash
# The GPU crate's static checks: formatting, clippy, one wgpu at the pinned version in the lock, and
# the core's own dependency check (no GPU crate reached the core).
set -euo pipefail
source "$(dirname "$0")/toolchain.sh"
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
python3 ci/lock_check.py
cd ../core
source ci/toolchain.sh
python3 ci/tree_check.py deps
