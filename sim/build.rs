//! Names the build in a bake's identity: the compiler's version, the target and the target
//! features the crate is compiled with.

use std::env;
use std::process::Command;

fn main() {
    let rustc = env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|text| text.trim().to_string())
        .unwrap_or_else(|| "rustc unknown".into());
    let target = env::var("TARGET").unwrap_or_default();
    let mut features: Vec<String> = env::var("CARGO_CFG_TARGET_FEATURE")
        .unwrap_or_default()
        .split(',')
        .filter(|f| !f.is_empty())
        .map(str::to_string)
        .collect();
    features.sort();
    println!(
        "cargo:rustc-env=ARACHNE_SIM_BUILD={version}; {target}; {}",
        features.join(",")
    );
    println!("cargo:rerun-if-env-changed=RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-changed=build.rs");
}
