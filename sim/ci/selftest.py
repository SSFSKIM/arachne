"""Tests of the checks under ``sim/ci/``: the import list's names and the census's list.

Run from ``sim/``: ``python3 ci/selftest.py``. Standard library only, so it needs no
environment; each ``test_`` function is run in turn.
"""

from __future__ import annotations

import atexit
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import census  # noqa: E402
import imports  # noqa: E402
import tree_check  # noqa: E402


def test_each_listed_function_and_its_single_precision_form() -> None:
    for name in ("sin", "sinf", "atan2f", "expm1", "log1pf", "powf", "hypot", "tanhf"):
        assert imports.platform_math(name) == name


def test_fused_multiply_add() -> None:
    assert imports.platform_math("fma") == "fma"
    assert imports.platform_math("fmaf") == "fmaf"


def test_exact_operations_are_not_listed() -> None:
    exact = ("fmax", "fmaxf", "fminf", "sqrtf", "floorf", "ldexp", "frexpf", "memcpy")
    for name in (*exact, "__powisf2"):
        assert imports.platform_math(name) is None, name


def test_names_that_only_begin_like_a_listed_one() -> None:
    for name in ("sinkhorn", "expand", "logger", "cosmos", "sinh_cosh_table", "atanh2"):
        assert imports.platform_math(name) is None, name


def test_glibc_version_suffix() -> None:
    assert imports.platform_math("sinf@GLIBC_2.2.5") == "sinf"
    assert imports.platform_math("exp@GLIBC_2.29") == "exp"


def test_macos_underscore_and_sincos_stret() -> None:
    assert imports.platform_math("_sinf") == "sinf"
    assert imports.platform_math("___sincosf_stret") == "sincosf"
    assert imports.platform_math("___sincos_stret") == "sincos"


def test_glibc_finite_aliases() -> None:
    assert imports.platform_math("__exp_finite") == "exp"
    assert imports.platform_math("__log2f_finite") == "log2f"


def test_nm_lines() -> None:
    listing = (
        "                 U sinf@GLIBC_2.2.5\n"
        "                 U memcpy@GLIBC_2.14\n"
        "                 w __gmon_start__\n"
    )
    assert imports.undefined_from_nm(listing) == [
        "sinf@GLIBC_2.2.5",
        "memcpy@GLIBC_2.14",
        "__gmon_start__",
    ]


def test_pe_import_table() -> None:
    listing = (
        "The Import Tables:\n"
        "  lookup 0001a2b0 time 00000000 fwd 00000000 name 0001a9f2 dll 0001a000\n\n"
        "    DLL Name: api-ms-win-crt-math-l1-1-0.dll\n"
        "    Hint/Ord  Name\n"
        "          0  sinf\n"
        "         12  cosf\n\n"
        "    DLL Name: KERNEL32.dll\n"
        "    Hint/Ord  Name\n"
        "        551  GetLastError\n"
    )
    assert imports.undefined_from_pe(listing) == ["sinf", "cosf", "GetLastError"]


def test_plt_and_got_calls() -> None:
    relocations = (
        "OFFSET           TYPE              VALUE\n"
        "0000000000043fe8 R_X86_64_GLOB_DAT  sinf@GLIBC_2.2.5\n"
        "0000000000043ff0 R_X86_64_JUMP_SLOT  cosf@GLIBC_2.2.5\n"
    )
    disassembly = (
        "   1a2b3:\tcall   *0x2fe2(%rip)        # 43fe8 <sinf@GLIBC_2.2.5>\n"
        "   1a2c0:\tcall   *0x2fe2(%rip)        # 43ff0 <_GLOBAL_OFFSET_TABLE_+0x30>\n"
        "   1a2d0:\tcall   1a000 <cosf@plt>\n"
        "   1a2e0:\tjmp    *0x2fe2(%rip)        # 43fe8 <x>\n"
        "   1a2f0:\tcall   1a000 <memcpy@plt>\n"
        "   1a300:\tmov    0x2fe2(%rip),%rbx        # 43ff0 <_GLOBAL_OFFSET_TABLE_+0x30>\n"
        "   1a307:\tcall   *%rbx\n"
    )
    slots = imports.got_slots(relocations)
    assert slots == {0x43FE8: "sinf@GLIBC_2.2.5", 0x43FF0: "cosf@GLIBC_2.2.5"}
    assert imports.platform_math_references(disassembly, slots) == [
        (0x1A2B3, "sinf"),
        (0x1A2C0, "cosf"),
        (0x1A2D0, "cosf"),
        (0x1A2E0, "sinf"),
        (0x1A300, "cosf"),
    ]


def test_source_path_is_shortened_to_the_crate() -> None:
    registry = "/r/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/rapier2d-0.36.1/src/a.rs"
    assert imports.short_path(registry) == "rapier2d-0.36.1/src/a.rs"
    git = "/r/.cargo/git/checkouts/rapier-6a2b1c/b303bc1/src/a.rs"
    assert imports.short_path(git) == "git:rapier-6a2b1c/b303bc1/src/a.rs"


def test_addr2line_frames_with_rust_paths_in_names() -> None:
    lines = [
        "0x000000000001a2b3",
        "rapier2d::dynamics::AngularLimitParams::new",
        "/r/registry/src/index.crates.io-x/rapier2d-0.36.1/src/h.rs:62",
        "probe::two::run",
        "/w/sim/examples/probe/two.rs:10",
        "0x000000000001a2d0",
        "glamx::eigen3::eigenvalues",
        "/r/registry/src/index.crates.io-x/glamx-0.3.1/src/eigen3.rs:105",
    ]
    assert imports.addr2line_frames(lines) == {
        0x1A2B3: [
            "rapier2d-0.36.1/src/h.rs:62 (rapier2d::dynamics::AngularLimitParams::new)",
            "/w/sim/examples/probe/two.rs:10 (probe::two::run)",
        ],
        0x1A2D0: ["glamx-0.3.1/src/eigen3.rs:105 (glamx::eigen3::eigenvalues)"],
    }


CPU = "cpufeatures 0.2.17 sha256:aa"
MM = "matrixmultiply 0.3.10 sha256:bb"
LIBM = "libm 0.2.16 sha256:b6d2"


def test_census_hits_compare_with_the_reviewed_list() -> None:
    hits = [
        census.Hit(CPU, "src/x86.rs", 40, "is_x86_feature_detected!"),
        census.Hit(MM, "src/x86/macros.rs", 12, "detected"),
    ]
    reviewed = census.parse_reviewed(
        "# comment\n"
        f"{CPU} src/x86.rs:40  the detector itself\n"
        "wide 1.0.3 sha256:cc src/lib.rs:7  gone from the tree\n"
    )
    unlisted, stale = census.compare(hits, reviewed)
    assert [h.key for h in unlisted] == [f"{MM} src/x86/macros.rs:12"]
    assert stale == ["wide 1.0.3 sha256:cc src/lib.rs:7"]


def test_census_entry_for_every_hit_of_a_file_of_one_locked_source() -> None:
    hits = [
        census.Hit(LIBM, "src/math/arch/x86/detect.rs", 63, "__cpuid(0)"),
        census.Hit(LIBM, "src/math/arch/x86/detect.rs", 77, "__cpuid(1)"),
        census.Hit("libm 0.2.16 sha256:ffff", "src/math/arch/x86/detect.rs", 63, "x"),
    ]
    reviewed = census.parse_reviewed(
        f"{LIBM} src/math/arch/x86/detect.rs:*  selects the fused multiply-add\n"
    )
    unlisted, stale = census.compare(hits, reviewed)
    assert [h.key for h in unlisted] == [
        "libm 0.2.16 sha256:ffff src/math/arch/x86/detect.rs:63"
    ]
    assert stale == []


def test_census_names_the_locked_source() -> None:
    lock = (
        "version = 4\n\n"
        "[[package]]\n"
        'name = "libm"\nversion = "0.2.16"\n'
        'source = "registry+https://github.com/rust-lang/crates.io-index"\n'
        'checksum = "b6d2"\n\n'
        "[[package]]\n"
        'name = "rapier2d"\nversion = "0.36.1"\n'
        'source = "git+https://github.com/dimforge/rapier?rev=b303#b303bc13"\n\n'
        "[[package]]\n"
        'name = "arachne-sim"\nversion = "0.1.0"\n'
    )
    index = census.lock_index(lock)
    registry = "registry+https://github.com/rust-lang/crates.io-index"
    assert census.source_id(index, "libm", "0.2.16", registry) == "sha256:b6d2"
    git = "git+https://github.com/dimforge/rapier?rev=b303#b303bc13"
    assert census.source_id(index, "rapier2d", "0.36.1", git) == "git:b303bc13"
    assert census.source_id(index, "arachne-sim", "0.1.0", None) == "local"


def test_census_reads_the_shipped_tree_only() -> None:
    metadata = {
        "resolve": {
            "root": "sim",
            "nodes": [
                {
                    "id": "sim",
                    "deps": [
                        {"pkg": "rapier", "dep_kinds": [{"kind": None}]},
                        {"pkg": "sha2", "dep_kinds": [{"kind": "dev"}]},
                        {
                            "pkg": "both",
                            "dep_kinds": [{"kind": "dev"}, {"kind": "build"}],
                        },
                    ],
                },
                {
                    "id": "rapier",
                    "deps": [{"pkg": "libm", "dep_kinds": [{"kind": None}]}],
                },
                {"id": "libm", "deps": []},
                {
                    "id": "sha2",
                    "deps": [{"pkg": "cpufeatures", "dep_kinds": [{"kind": None}]}],
                },
                {"id": "cpufeatures", "deps": []},
                {"id": "both", "deps": []},
            ],
        }
    }
    assert census.shipped(metadata) == {"sim", "rapier", "libm", "both"}


def test_census_entry_needs_a_reason() -> None:
    try:
        census.parse_reviewed(f"{CPU} src/x86.rs:40\n")
    except ValueError:
        return
    raise AssertionError("an entry without its reason was accepted")


def test_census_walks_a_library_outside_its_manifest_directory() -> None:
    package = {
        "manifest_path": "/co/crates/rapier2d/Cargo.toml",
        "targets": [
            {"kind": ["lib"], "src_path": "/co/crates/rapier2d/../../src/lib.rs"},
            {"kind": ["custom-build"], "src_path": "/co/crates/rapier2d/build.rs"},
            {"kind": ["test"], "src_path": "/co/crates/rapier2d/tests/a.rs"},
            {"kind": ["example"], "src_path": "/co/examples/b.rs"},
        ],
    }
    base, roots = census.compiled_roots(package)
    assert base == Path("/co")
    assert roots == [Path("/co/src"), Path("/co/crates/rapier2d/build.rs")]


def test_census_walks_a_build_script_directory_of_its_own() -> None:
    package = {
        "manifest_path": "/r/libm-0.2.16/Cargo.toml",
        "targets": [
            {"kind": ["lib"], "src_path": "/r/libm-0.2.16/src/lib.rs"},
            {"kind": ["custom-build"], "src_path": "/r/libm-0.2.16/build/main.rs"},
        ],
    }
    base, roots = census.compiled_roots(package)
    assert base == Path("/r/libm-0.2.16")
    assert roots == [Path("/r/libm-0.2.16/src"), Path("/r/libm-0.2.16/build")]


def crate(files: dict[str, str]) -> Path:
    """A package's sources written into a fresh directory; returns the manifest's directory."""
    scratch = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, scratch)
    root = Path(scratch) / "pkg"
    for name, text in {"Cargo.toml": "", **files}.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def package_at(root: Path, *targets: tuple[str, str]) -> dict[str, Any]:
    return {
        "manifest_path": str(root / "Cargo.toml"),
        "targets": [{"kind": [k], "src_path": str(root / p)} for k, p in targets],
    }


def test_census_follows_the_modules_a_build_script_declares() -> None:
    root = crate(
        {
            "build.rs": "use std::env;\n\nmod configure;\n\nfn main() {}\n",
            "configure.rs": 'pub fn f() {\n    if is_x86_feature_detected!("avx") {}\n}\n',
            "tests/t.rs": 'if is_x86_feature_detected!("avx") {}\n',
            "src/lib.rs": "",
        }
    )
    package = package_at(root, ("lib", "src/lib.rs"), ("custom-build", "build.rs"))
    base, files = census.compiled_sources(package)
    assert root / "configure.rs" in files
    assert root / "tests/t.rs" not in files
    hits = census.search(CPU, base, files)
    assert [h.key for h in hits] == [f"{CPU} configure.rs:2"]


def test_census_follows_a_module_path_outside_the_library_directory() -> None:
    root = crate(
        {
            "src/lib.rs": '#[cfg(unix)]\n#[path = "../shared/x.rs"]\npub(crate) mod x;\nmod inline {}\n',
            "shared/x.rs": "mod inner;\n",
            "shared/inner.rs": "let r = __cpuid(1);\n",
            "shared/x/inner.rs": "let r = __cpuid(1);\n",
        }
    )
    # A file loaded through #[path] owns its directory, as mod.rs does: its `mod inner;` is
    # shared/inner.rs, and shared/x/inner.rs is not compiled.
    base, files = census.compiled_sources(package_at(root, ("lib", "src/lib.rs")))
    assert root / "shared/inner.rs" in files
    assert root / "shared/x/inner.rs" not in files
    assert [h.key for h in census.search(CPU, base, files)] == [
        f"{CPU} shared/inner.rs:1"
    ]


def test_census_stops_on_a_module_it_cannot_find() -> None:
    root = crate({"build.rs": "mod missing;\nfn main() {}\n"})
    try:
        census.compiled_sources(package_at(root, ("custom-build", "build.rs")))
    except SystemExit:
        return
    raise AssertionError("a declared module with no file was passed over")


def test_census_patterns() -> None:
    assert census.detects('    if is_x86_feature_detected!("avx2") {')
    assert census.detects('cpufeatures::new!(cpuid_avx2, "avx2");')
    assert census.detects("let r = __cpuid(1);")
    assert census.detects("unsafe { getauxval(AT_HWCAP) }")
    assert not census.detects('#[cfg(target_feature = "avx2")]')


def test_tree_features_by_package() -> None:
    listing = (
        "arachne-sim v0.1.0 (/w/sim)|\n"
        "simba v0.10.2|libm,libm_force,wide\n"
        "glam v0.33.12|libm,nostd-libm,scalar-math\n"
        "glam v0.30.10|std (*)\n"
        "simba v0.10.2|libm,libm_force,wide (*)\n"
    )
    assert tree_check.features_by_package(listing) == {
        "arachne-sim v0.1.0": set(),
        "simba v0.10.2": {"libm", "libm_force", "wide"},
        "glam v0.33.12": {"libm", "nostd-libm", "scalar-math"},
        "glam v0.30.10": {"std"},
    }


def vendored_metadata(
    vendor: Path, **override: tuple[str | None, str]
) -> dict[str, Any]:
    packages = []
    for name, directory in tree_check.VENDORED.items():
        source, path = override.get(
            name, (None, str(vendor / directory / "Cargo.toml"))
        )
        packages.append({"name": name, "source": source, "manifest_path": path})
    return {"packages": packages}


def test_tree_vendored_crates_from_vendor_pass() -> None:
    vendor = Path("/w/sim/vendor")
    lock = '[[package]]\nname = "rapier2d"\nversion = "0.36.1"\n'
    assert tree_check.check_vendored(vendored_metadata(vendor), lock, vendor) == []


def test_tree_a_vendored_crate_from_elsewhere_fails() -> None:
    vendor = Path("/w/sim/vendor")
    git = "git+https://github.com/dimforge/parry?rev=dbcce40#dbcce40"
    elsewhere = vendored_metadata(
        vendor,
        parry2d=(git, "/home/.cargo/git/checkouts/parry/crates/parry2d/Cargo.toml"),
        glamx=(None, "/w/other/glamx/Cargo.toml"),
    )
    problems = tree_check.check_vendored(elsewhere, "", vendor)
    assert len(problems) == 2, problems
    assert problems[0].startswith("parry2d resolves from git+")
    assert problems[1].startswith("glamx is built from /w/other/glamx/Cargo.toml")


def test_tree_a_git_source_in_the_lock_fails() -> None:
    vendor = Path("/w/sim/vendor")
    lock = (
        '[[package]]\nname = "glamx"\nversion = "0.3.1"\n'
        'source = "git+https://github.com/dimforge/glamx?rev=78c97db#78c97db"\n'
    )
    problems = tree_check.check_vendored(vendored_metadata(vendor), lock, vendor)
    assert problems == [
        "Cargo.lock holds a git source: glamx "
        "git+https://github.com/dimforge/glamx?rev=78c97db#78c97db"
    ]


def test_tree_a_vendored_crate_missing_or_twice_fails() -> None:
    vendor = Path("/w/sim/vendor")
    metadata = vendored_metadata(vendor)
    metadata["packages"] = [
        p for p in metadata["packages"] if p["name"] != "rapier3d"
    ] + [{"name": "glamx", "source": None, "manifest_path": "/x/Cargo.toml"}]
    problems = tree_check.check_vendored(metadata, "", vendor)
    assert problems == [
        "rapier3d: 0 packages in the tree, not 1",
        "glamx: 2 packages in the tree, not 1",
    ]


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"selftest: {len(tests)} passed")
