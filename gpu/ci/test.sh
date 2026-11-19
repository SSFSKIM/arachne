#!/usr/bin/env bash
# The GPU crate's Rust tests, in software mode on lavapipe.
set -euo pipefail
bash "$(dirname "$0")/software_vulkan.sh"
source "$(dirname "$0")/toolchain.sh"
export ARACHNE_GPU_TEST_MODE=software
cargo test --release --locked -- --nocapture
