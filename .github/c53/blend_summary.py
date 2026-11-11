"""Summarise blends-*.jsonl: cases and records carrying paint_b, and one example in full."""

import json
import sys
from pathlib import Path

lines = []
for p in sorted(Path(sys.argv[1]).glob("blends-*.jsonl")):
    lines += [json.loads(x) for x in p.read_text().splitlines() if x]
lines.sort(key=lambda x: x["case"])
print("cases carrying paint_b:", len(lines))
print("distinct blend records (painter, slot):", sum(x["blend_records"] for x in lines))
print("blend record sets over all deltas:", sum(x["blend_sets"] for x in lines))
print("band-capable blend records:", sum(x["band_capable"] for x in lines))
print("label cases among them:", sum(1 for x in lines if x["facing"]))
for x in lines:
    print(f"  {x['blend_records']:5d} {x['blend_sets']:6d} {x['band_capable']:3d}  {x['case']}")
tabled = [x for x in lines if str(x["example"].get("fill_a_form", "")).startswith("table")
          or str(x["example"].get("fill_b_form", "")).startswith("table")]
pick = (tabled or lines or [None])[0]
if pick:
    print("example:", json.dumps({"case": pick["case"], **pick["example"]}, indent=1))

for x in lines:
    if not x.get("band_log"):
        continue
    print("band-capable blend sets:", x["case"])
    for g in x["band_log"]:
        print("  ", json.dumps(g))
