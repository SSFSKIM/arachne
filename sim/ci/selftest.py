"""Tests of the checks under ``sim/ci/``: the import list's names and the census's list.

Run from ``sim/``: ``python3 ci/selftest.py``. Standard library only, so it needs no
environment; each ``test_`` function is run in turn.
"""

from __future__ import annotations

import sys
from pathlib import Path

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


def test_census_hits_compare_with_the_reviewed_list() -> None:
    hits = [
        census.Hit("cpufeatures 0.2.17", "src/x86.rs", 40, "is_x86_feature_detected!"),
        census.Hit("matrixmultiply 0.3.10", "src/x86/macros.rs", 12, "detected"),
    ]
    reviewed = census.parse_reviewed(
        "# comment\n"
        "cpufeatures 0.2.17 src/x86.rs:40  the detector itself\n"
        "wide 1.0.3 src/lib.rs:7  gone from the tree\n"
    )
    unlisted, stale = census.compare(hits, reviewed)
    assert [h.key for h in unlisted] == ["matrixmultiply 0.3.10 src/x86/macros.rs:12"]
    assert stale == ["wide 1.0.3 src/lib.rs:7"]


def test_census_entry_for_every_hit_of_a_file() -> None:
    hits = [
        census.Hit("libm 0.2.16", "src/math/arch/x86/detect.rs", 63, "__cpuid(0)"),
        census.Hit("libm 0.2.16", "src/math/arch/x86/detect.rs", 77, "__cpuid(1)"),
        census.Hit("libm 0.2.17", "src/math/arch/x86/detect.rs", 63, "__cpuid(0)"),
    ]
    reviewed = census.parse_reviewed(
        "libm 0.2.16 src/math/arch/x86/detect.rs:*  selects the fused multiply-add\n"
    )
    unlisted, stale = census.compare(hits, reviewed)
    assert [h.key for h in unlisted] == ["libm 0.2.17 src/math/arch/x86/detect.rs:63"]
    assert stale == []


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
        census.parse_reviewed("cpufeatures 0.2.17 src/x86.rs:40\n")
    except ValueError:
        return
    raise AssertionError("an entry without its reason was accepted")


def test_census_reads_only_what_a_dependency_builds() -> None:
    assert census.built("src/x86.rs")
    assert census.built("build.rs")
    assert not census.built("tests/x86.rs")
    assert not census.built("benches/a.rs")
    assert not census.built("examples/a/main.rs")


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


if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"selftest: {len(tests)} passed")
