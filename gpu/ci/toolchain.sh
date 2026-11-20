# Sourced by the GPU crate's CI scripts: moves to gpu/ and installs the toolchain
# rust-toolchain.toml pins, whatever rustup's auto-install policy is.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
channel=$(sed -n 's/^channel = "\(.*\)"$/\1/p' rust-toolchain.toml)
rustup toolchain install "$channel" --profile minimal --component rustfmt,clippy --no-self-update
cargo --version
