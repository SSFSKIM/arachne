//! `ci/oracle_check.py` passes the oracle as it stands and fails it once a core module other than
//! the XF schema is planted in it.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn oracle_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn check(args: &[&Path]) -> Output {
    let script = oracle_dir().join("../ci/oracle_check.py");
    let mut cmd = Command::new("python3");
    cmd.arg("-I").arg(script).arg("--no-tree");
    let mut flag = ["--oracle", "--xf"].into_iter();
    for a in args {
        cmd.arg(flag.next().unwrap()).arg(a);
    }
    cmd.output().expect("python3 runs")
}

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let (src, dst) = (e.path(), to.join(e.file_name()));
        if e.file_type().unwrap().is_dir() {
            copy_tree(&src, &dst);
        } else {
            std::fs::copy(&src, &dst).unwrap();
        }
    }
}

/// A copy of the oracle's sources and manifest under a fresh directory.
fn copy_of_oracle(tag: &str) -> PathBuf {
    let root =
        std::env::temp_dir().join(format!("arachne-oracle-check-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let dir = root.join("oracle");
    copy_tree(&oracle_dir().join("src"), &dir.join("src"));
    copy_tree(&oracle_dir().join("tests"), &dir.join("tests"));
    std::fs::copy(oracle_dir().join("Cargo.toml"), dir.join("Cargo.toml")).unwrap();
    dir
}

fn plant(dir: &Path, file: &str, line: &str) {
    let path = dir.join(file);
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, format!("{text}\n{line}\n")).unwrap();
}

#[test]
fn the_oracle_names_only_the_xf_schema() {
    let out = check(&[]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{stdout}{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("oracle: allowed arachne_core::xf ("),
        "{stdout}"
    );
    assert!(stdout.contains("src/stream.rs"), "{stdout}");
}

#[test]
fn a_planted_core_module_fails_the_check() {
    for (tag, file, line) in [
        ("composite", "src/lib.rs", "use arachne_core::composite;"),
        ("group", "src/lib.rs", "use arachne_core::{xf, order};"),
        (
            "renamed",
            "tests/camera.rs",
            "use arachne_core as core_crate;",
        ),
        (
            "path",
            "src/lib.rs",
            "#[path = \"../../src/composite.rs\"]\nmod composite;",
        ),
        (
            "include",
            "src/draw.rs",
            "const _: &str = include_str!(\"../../src/path.rs\");",
        ),
    ] {
        let dir = copy_of_oracle(tag);
        plant(&dir, file, line);
        let out = check(&[&dir]);
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert_eq!(out.status.code(), Some(1), "{tag}: {stdout}");
        assert!(
            stdout.contains(&format!("oracle: FAIL {file}:")),
            "{tag}: {stdout}"
        );
    }
}

#[test]
fn a_comment_naming_a_core_module_is_not_a_dependency() {
    let dir = copy_of_oracle("comment");
    plant(
        &dir,
        "src/lib.rs",
        "// arachne_core::composite is the core's, never the oracle's.",
    );
    assert!(check(&[&dir]).status.success());
}

#[test]
fn the_xf_module_naming_another_core_module_fails_the_check() {
    let root = std::env::temp_dir().join(format!("arachne-oracle-xf-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let xf = root.join("xf");
    copy_tree(&oracle_dir().join("../src/xf"), &xf);
    let out = check(&[&oracle_dir(), &xf]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stdout)
    );
    plant(&xf, "wire.rs", "use crate::composite::Frame;");
    let out = check(&[&oracle_dir(), &xf]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(1), "{stdout}");
    assert!(stdout.contains("src/xf/wire.rs:"), "{stdout}");
}
