#!/usr/bin/env bash
# The simulation crate's Rust tests.
set -euo pipefail
source "$(dirname "$0")/toolchain.sh"
cargo test --release --locked
