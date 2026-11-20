#!/usr/bin/env bash
# The hosted macOS lane: the crate's tests, and the probe's import list on macOS.
set -euo pipefail
source "$(dirname "$0")/toolchain.sh"
cargo test --release --locked
cargo build --release --locked --example probe
python3 ci/imports.py target/release/examples/probe
