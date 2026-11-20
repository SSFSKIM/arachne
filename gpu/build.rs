//! Records the wgpu version this build is locked to (Cargo.lock), the pin's `wgpu` field.

use std::{env, fs, path::Path};

fn main() {
    let lock =
        Path::new(&env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR")).join("Cargo.lock");
    println!("cargo::rerun-if-changed={}", lock.display());
    let text =
        fs::read_to_string(&lock).expect("the GPU crate builds from its committed Cargo.lock");
    let versions = locked_versions(&text, "wgpu");
    let [version] = versions.as_slice() else {
        panic!("Cargo.lock must hold exactly one wgpu, found {versions:?}");
    };
    println!("cargo::rustc-env=ARACHNE_GPU_WGPU_VERSION={version}");
}

/// The versions of every package named `name` in a Cargo.lock.
fn locked_versions(lock: &str, name: &str) -> Vec<String> {
    let wanted = format!("name = \"{name}\"");
    let mut lines = lock.lines();
    let mut found = Vec::new();
    while let Some(line) = lines.next() {
        if line == wanted
            && let Some(version) = lines
                .next()
                .and_then(|v| v.strip_prefix("version = \""))
                .and_then(|v| v.strip_suffix('"'))
        {
            found.push(version.to_string());
        }
    }
    found
}
