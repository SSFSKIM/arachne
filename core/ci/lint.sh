#!/usr/bin/env bash
# The core's static checks: formatting, clippy (whose configuration forbids fused multiply-add and
# the platform's transcendental functions), and its dependency tree (no GPU crate; kurbo without
# its std feature).
set -euo pipefail
source "$(dirname "$0")/toolchain.sh"
cargo fmt --check
cargo clippy --locked --all-targets --all-features -- -D warnings
python3 ci/tree_check.py deps
