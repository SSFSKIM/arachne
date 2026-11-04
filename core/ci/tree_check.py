"""The core is CPU only, and its arithmetic is the same on every machine.

``tree_check.py deps``
    Fails if the core's dependency tree (every kind, every target, every feature) holds a GPU crate,
    if kurbo (the path rasterizer's stroker) is missing from it, or if kurbo or polycool is built
    with its ``std`` feature, which routes its float functions to the platform's libm instead of
    the pure-Rust one.

``tree_check.py links <extension>``
    Fails if the built extension links a GPU library. Prints what it links.

Run from ``core/``.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys

GPU_CRATES = re.compile(
    r"^(wgpu|wgpu[-_].*|ash|ash-.*|metal|metal-.*|objc2-metal.*|glow|glutin.*|gl|gl_generator|"
    r"glium|khronos.*|egl|vulkan.*|vulkano.*|vk-.*|naga|gpu-.*|d3d12|cuda.*|cudarc|opencl.*|ocl|"
    r"lavapipe)$"
)
GPU_LIBRARIES = re.compile(
    r"(libgl[xs]?\.|libgles|libegl|vulkan|libdrm|libgbm|cuda|opencl|metal|opengl|moltenvk|lavapipe)",
    re.IGNORECASE,
)


def cargo(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["cargo", *args], capture_output=True, text=True, check=False)


def check_deps() -> list[str]:
    tree = cargo(
        "tree",
        "--locked",
        "--all-features",
        "--target",
        "all",
        "-e",
        "all",
        "--prefix",
        "none",
        "--format",
        "{p}",
    )
    if tree.returncode != 0:
        sys.exit(f"cargo tree failed:\n{tree.stderr}")
    crates = sorted({line.split()[0] for line in tree.stdout.splitlines() if line})
    print(f"deps: {len(crates)} crates: {' '.join(crates)}")
    problems = [f"a GPU crate in the tree: {c}" for c in crates if GPU_CRATES.match(c)]

    # kurbo, and polycool (its cubic root finder), take their float functions from the platform's
    # libm when built with std.
    for crate, required in (("kurbo", True), ("polycool", False)):
        features = cargo(
            "tree",
            "--locked",
            "--all-features",
            "--target",
            "all",
            "-e",
            "features",
            "-i",
            crate,
        )
        if features.returncode != 0:
            if "did not match any packages" not in features.stderr:
                sys.exit(f"cargo tree -i {crate} failed:\n{features.stderr}")
            if required:
                problems.append(f"{crate} is not in the tree")
        elif not features.stdout.startswith(f"{crate} v"):
            sys.exit(
                f"cargo tree -i {crate} printed what this check does not read:\n{features.stdout}"
            )
        elif f'{crate} feature "std"' in features.stdout:
            problems.append(
                f"{crate} is built with its std feature (the platform's libm)"
            )
        else:
            print(f"deps: {crate} is built without std")
    return problems


def check_links(path: str) -> list[str]:
    if platform.system() == "Darwin":
        out = subprocess.run(
            ["otool", "-L", path], capture_output=True, text=True, check=True
        ).stdout
        libraries = [line.split()[0] for line in out.splitlines()[1:] if line.strip()]
    else:
        out = subprocess.run(
            ["readelf", "-d", path], capture_output=True, text=True, check=True
        ).stdout
        libraries = re.findall(r"\(NEEDED\)\s+Shared library: \[(.+?)\]", out)
    print(f"links: {path} links {', '.join(libraries) or 'nothing'}")
    return [
        f"a GPU library is linked: {lib}"
        for lib in libraries
        if GPU_LIBRARIES.search(lib)
    ]


def main() -> None:
    if len(sys.argv) >= 2 and sys.argv[1] == "deps":
        problems = check_deps()
    elif len(sys.argv) == 3 and sys.argv[1] == "links":
        problems = check_links(sys.argv[2])
    else:
        sys.exit(__doc__)
    for problem in problems:
        print(f"FAIL: {problem}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
