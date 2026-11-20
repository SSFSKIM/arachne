"""What a built artifact asks of the platform's maths library.

``imports.py <binary>``
    Lists the binary's undefined symbols and fails if one is a transcendental function of the
    platform's maths library or a fused multiply-add: those are computed differently from one
    system to the next, and in glibc from one CPU to the next. Prints ``imports: no platform
    maths`` when the list holds none.

``imports.py --list <binary>``
    Prints every undefined symbol (or, for a Windows executable, every imported function) and
    the platform maths among them, and never fails: for the record of a system that binds
    nothing.

``imports.py --where <binary>``
    For each call to platform maths in a Linux binary built with line tables (the ``audit``
    profile), prints the source lines it comes from, innermost first.

Run from ``sim/``. Linux and macOS read ``nm``; Windows reads ``llvm-objdump -p`` from the
toolchain's ``llvm-tools`` component.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# The transcendental functions of C's maths library, and fused multiply-add. Each also has its
# single-precision form (the name with ``f``). sqrt, fabs, floor, fmin, fmax and the like are
# exact IEEE operations and are not listed.
TRANSCENDENTAL = [
    "sin",
    "cos",
    "sincos",
    "tan",
    "asin",
    "acos",
    "atan",
    "atan2",
    "exp",
    "exp2",
    "exp10",
    "expm1",
    "log",
    "log2",
    "log10",
    "log1p",
    "pow",
    "cbrt",
    "hypot",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "fma",
]
LISTED = frozenset(TRANSCENDENTAL) | frozenset(name + "f" for name in TRANSCENDENTAL)


def platform_math(symbol: str) -> str | None:
    """The listed function a symbol names, or None.

    Undoes the decorations a symbol carries: a glibc version (``sinf@GLIBC_2.2.5``), the
    leading underscore of Mach-O (``_sinf``), Apple's ``__sincosf_stret`` and glibc's
    ``__expf_finite``.
    """
    name = symbol.split("@", 1)[0]
    candidates = {name, name.lstrip("_")}
    for suffix in ("_stret", "_finite"):
        candidates |= {c.removesuffix(suffix) for c in candidates}
    candidates |= {c.lstrip("_") for c in candidates}
    found = sorted(c for c in candidates if c in LISTED)
    return found[0] if found else None


def undefined_from_nm(listing: str) -> list[str]:
    """The symbol names of ``nm -u`` output."""
    return [line.split()[-1] for line in listing.splitlines() if line.strip()]


def undefined_from_pe(listing: str) -> list[str]:
    """The imported function names of ``llvm-objdump -p`` output for a PE executable."""
    names = []
    in_table = False
    for line in listing.splitlines():
        if "DLL Name:" in line:
            in_table = True
            continue
        if not line.strip():
            in_table = False
            continue
        entry = re.match(r"^\s+(\d+)\s+(\S+)\s*$", line)
        if in_table and entry:
            names.append(entry.group(2))
    return names


def undefined(binary: str) -> list[str]:
    """The binary's undefined symbols (its imported functions, for a Windows executable)."""
    if binary.endswith(".exe"):
        out = run([llvm_objdump(), "-p", binary])
        return undefined_from_pe(out)
    return undefined_from_nm(run(["nm", "-u", binary]))


def llvm_objdump() -> str:
    found = shutil.which("llvm-objdump")
    if found:
        return found
    sysroot = Path(run(["rustc", "--print", "sysroot"]).strip())
    for candidate in sorted(sysroot.glob("lib/rustlib/*/bin/llvm-objdump*")):
        return str(candidate)
    sys.exit("no llvm-objdump: rustup component add llvm-tools")


def got_slots(relocations: str) -> dict[int, str]:
    """GOT and PLT slot addresses and their symbols, from ``objdump -R`` output."""
    slots = {}
    for line in relocations.splitlines():
        fields = line.split()
        if len(fields) == 3 and re.fullmatch(r"[0-9a-f]+", fields[0]):
            slots[int(fields[0], 16)] = fields[2]
    return slots


def platform_math_references(
    disassembly: str, slots: dict[int, str]
) -> list[tuple[int, str]]:
    """Each instruction of ``objdump -d`` output that reaches platform maths: (address, function).

    A call goes through the PLT (``call 1a000 <sinf@plt>``) or straight through the GOT
    (``call *0x2fe2(%rip)  # 43fe8 <...>``); a call through a register loads the GOT slot
    first (``mov 0x2fe2(%rip),%rbx  # 43ff0 <...>``). A slot is resolved through the
    relocations.
    """
    references = []
    for line in disassembly.splitlines():
        head = re.match(r"^\s*([0-9a-f]+):\s+\S+\s+(.*)$", line)
        if not head:
            continue
        address, operand = int(head.group(1), 16), head.group(2)
        plt = re.search(r"<([^>]+)@plt>", operand)
        got = re.search(r"\(%rip\).*#\s*([0-9a-f]+)\b", operand)
        symbol = (
            plt.group(1) if plt else slots.get(int(got.group(1), 16)) if got else None
        )
        function = platform_math(symbol) if symbol else None
        if function:
            references.append((address, function))
    return references


def short_path(path: str) -> str:
    """A source path from the registry or a git checkout, from its crate's directory on."""
    registry = re.search(r"/registry/src/[^/]+/(.*)$", path)
    if registry:
        return registry.group(1)
    git = re.search(r"/git/checkouts/(.*)$", path)
    if git:
        return "git:" + git.group(1)
    return path


def addr2line_frames(lines: list[str]) -> dict[int, list[str]]:
    """``addr2line -a -f -i`` output by address: its frames as ``path:line (function)``."""
    frames: dict[int, list[str]] = {}
    current = None
    pending = None
    for line in lines:
        if re.fullmatch(r"0x[0-9a-f]+", line):
            current, pending = int(line, 16), None
            frames[current] = []
        elif current is not None and pending is None:
            pending = line
        elif current is not None:
            frames[current].append(f"{short_path(line)} ({pending})")
            pending = None
    return frames


def where(binary: str) -> None:
    slots = got_slots(run(["objdump", "-R", binary]))
    calls = platform_math_references(
        run(["objdump", "-d", "--no-show-raw-insn", binary]), slots
    )
    if not calls:
        print(f"where: no call to platform maths in {binary}")
        return
    lines = run(
        ["addr2line", "-e", binary, "-a", "-f", "-C", "-i", *[hex(a) for a, _ in calls]]
    ).splitlines()
    frames = addr2line_frames(lines)
    for address, function in calls:
        chain = frames.get(address, [])
        print(f"where: {function} at {address:#x}")
        for frame in chain[:4]:
            print(f"    {frame}")


def run(command: list[str]) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout


def main() -> None:
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--where":
        where(args[1])
        return
    listing = len(args) == 2 and args[0] == "--list"
    if not (listing or len(args) == 1):
        sys.exit(__doc__)
    binary = args[-1]
    symbols = undefined(binary)
    found = sorted({f for f in map(platform_math, symbols) if f})
    system = f"{platform.system().lower()}-{platform.machine().lower()}"
    if listing:
        print(
            f"imports ({system}): {len(symbols)} undefined: {' '.join(sorted(symbols))}"
        )
        print(f"imports ({system}): platform maths: {' '.join(found) or 'none'}")
        return
    if found:
        print(f"FAIL: {binary} calls the platform's maths library: {' '.join(found)}")
        sys.exit(1)
    print(f"imports: no platform maths ({len(symbols)} undefined symbols in {binary})")


if __name__ == "__main__":
    main()
