#!/usr/bin/env bash
# The hosted macOS lane: the crate's tests, the probe's import list on macOS, and the binding
# (its wheel, its extension's import list, its tests).
set -euo pipefail
source "$(dirname "$0")/toolchain.sh"
cargo test --release --locked -- --show-output
cargo build --release --locked --example probe
python3 ci/imports.py target/release/examples/probe
bash ci/binding.sh 3.14
