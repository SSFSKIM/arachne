#!/usr/bin/env bash
# The simulation crate's Rust tests. Each closed-form case prints its closed form, its tolerance
# and the measured value (tests/closed_forms.rs), shown here for passing tests too.
set -euo pipefail
source "$(dirname "$0")/toolchain.sh"
cargo test --release --locked -- --show-output
