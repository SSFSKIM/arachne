"""``python -m arachne.assets admit <directory> [--write]``: admit an asset
by its geometry, or a media asset by its frames (:mod:`.admission`).

It prints, one per line: ``asset: <name> <version> (<kind>) <digest>``;
``PASS <check>`` or ``FAIL <check>: <what is wrong>`` for each check, by
name; ``admission: <passed>/<checks> pass``; and with ``--write``, the
manifest written. It exits 0 when every check passes, 1 when one fails, and
2, saying why on standard error, when the asset cannot be read.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from ..lint.raised import RaisedFinding
from .admission import admit, write_checks
from .reader import read_asset


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command ``argv`` (the process's own by default) and give its
    exit status.
    """
    parser = argparse.ArgumentParser(
        prog="python -m arachne.assets",
        description="Admit an asset by its geometry, or a media asset by its frames.",
        epilog=(
            "Admission draws the asset at its default pose, at the lowest, middle"
            " and highest value of each parameter, and with each drawing of each"
            " slot, and fails a hole it opens inside the figure, a part that"
            " leaves its parent, and an anchor off its part. It judges shape"
            " alone. A still, a clip or a sprite is admitted by its frames: each"
            " must decode exactly, twice, to the digest its manifest lists, and"
            " have the manifest's size, and no pixel less than opaque where the"
            " manifest says the asset has no alpha. The module's documentation"
            " (python -m pydoc arachne.assets.admission) lists what the asset"
            " reader refuses and the two forms it passes that are drawn otherwise"
            " than SVG draws them."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("admit", help="admit the asset in a directory")
    command.add_argument("directory", help="the asset's directory")
    command.add_argument(
        "--write",
        action="store_true",
        help="record the checks' results with the asset's digest in its manifest",
    )
    arguments = parser.parse_args(argv)
    try:
        asset = read_asset(arguments.directory)
    except RaisedFinding as finding:
        print(str(finding), file=sys.stderr)
        return 2
    report = admit(asset)
    for line in report.lines():
        print(line)
    if arguments.write:
        print(f"written: {write_checks(asset.directory, report)}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
