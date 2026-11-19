"""Fails unless a pytest JUnit report collected at least one test and skipped none.

``python3 junit_check.py <report.xml>``
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def main() -> None:
    root = ET.parse(sys.argv[1]).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = sum(int(s.get("tests", 0)) for s in suites)
    skipped = sum(int(s.get("skipped", 0)) for s in suites)
    print(f"junit: {tests} tests, {skipped} skipped")
    if tests == 0:
        sys.exit("FAIL: no test was collected")
    if skipped:
        names = [
            f"{case.get('classname')}::{case.get('name')}"
            for case in root.iter("testcase")
            if case.find("skipped") is not None
        ]
        sys.exit(f"FAIL: a test was skipped: {', '.join(names)}")


if __name__ == "__main__":
    main()
