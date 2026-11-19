"""sim-probe.yml's last job: the probe's output on three systems, side by side.

``compare_probes.py <dir>``
    Reads ``probe-<system>.txt`` for linux-x86_64, windows-x86_64 and macos-arm64, prints each
    world's digest and each case's bits for the three, and fails unless the two x86-64 files are
    byte-identical. Whether macOS on arm64 agrees is printed and binds nothing: identity across
    architectures is not required.
"""

from __future__ import annotations

import sys
from pathlib import Path

SYSTEMS = ("linux-x86_64", "windows-x86_64", "macos-arm64")


def read(text: str) -> dict[str, str]:
    """``world <dim> <digest>`` and ``case <name> <bits>`` lines, by their first two words."""
    entries: dict[str, str] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0] in ("world", "case"):
            name = f"{fields[0]} {fields[1]}"
            if name in entries:
                sys.exit(f"compare: {name} appears twice")
            entries[name] = fields[2]
    return entries


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    texts = {s: (Path(sys.argv[1]) / f"probe-{s}.txt").read_bytes() for s in SYSTEMS}
    outputs = {s: read(t.decode()) for s, t in texts.items()}
    names = list(dict.fromkeys(n for s in SYSTEMS for n in outputs[s]))
    width = max(len(n) for n in names)
    print(f"{'':{width}}  " + "  ".join(SYSTEMS))
    for name in names:
        values = [outputs[s].get(name, "-") for s in SYSTEMS]
        marks = "" if values[0] == values[1] else "  x86-64 DISAGREE"
        arm = "" if values[2] == values[0] else "  (arm64 differs)"
        print(f"{name:{width}}  " + "  ".join(v[:16] for v in values) + marks + arm)
    linux, windows = texts["linux-x86_64"], texts["windows-x86_64"]
    agree = sum(
        outputs["linux-x86_64"].get(n) == outputs["windows-x86_64"].get(n)
        for n in names
    )
    print(f"probe: linux-x86_64 and windows-x86_64 agree on {agree}/{len(names)} cases")
    if linux == windows:
        print("probe: the linux-x86_64 and windows-x86_64 outputs are byte-identical")
    else:
        print("probe: the linux-x86_64 and windows-x86_64 outputs differ")
        sys.exit(1)


if __name__ == "__main__":
    main()
