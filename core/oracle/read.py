"""The reader of the oracle's decisions files (``ORACLE_FORMAT`` 1, ``FORMAT.md`` beside this file).

Dependency-free: the standard library only. ``read(path)`` returns a dictionary::

    {
        "header": {...},
        "fragments": [...],
        "covers": [...],
        "pixels": [...],
        "bands": [...],
        "pairs": [...],
        "colours": [...],
        "pixel_colours": [...],
        "labels": [...],
        "refused": [...],
    }

``DTYPES`` gives each fixed-size section's layout as a numpy dtype description, so a caller with
numpy reads a section whole: ``numpy.fromfile(path, numpy.dtype(DTYPES["COVR"]), count, offset=...)``
with the offset and count that ``sections(path)`` returns.

``python3 read.py --canonical <file>`` prints the decisions as canonical text (floats as the hex of
their bits), the same text the oracle's own tests produce from the decisions they wrote.
"""

from __future__ import annotations

import struct
import sys

ORACLE_FORMAT = 1
MAGIC = b"AROR"
HEADER_SIZE = 88
NONE_U32 = 0xFFFFFFFF

#: A sample's coverage verdict.
COVER = {0: "absent", 1: "yes", 2: "either"}
#: A fragment's layer.
LAYER = {0: "background_stroke", 1: "fill", 2: "stroke", 3: "mesh", 4: "points"}
#: Why a frame is ``truth_unable``.
REASON = {
    1: "unknown_kind",
    2: "reserved_kind",
    3: "unknown_feature",
    4: "unknown_light_model",
    5: "xf_newer",
    6: "invalid_delta",
    7: "no_view",
    8: "core_refuses",
    9: "stream_refused",
}
#: Why a stream is refused from a delta on (``stream_refused``): the delta's source-row refusal code.
CAUSE = {
    1: "class_named_twice",
    2: "slot_named_twice",
    3: "slot_removed",
    4: "row_empty",
    5: "path_empty",
    6: "class_out_of_range",
    7: "slot_not_held",
}
#: A band row's status at a sample.
BAND_STATUS = {0: "absent", 1: "bounded", 2: "band_unbounded", 3: "curve_recut"}
#: A pair's label at a sample.
LABEL = {
    0: "truth_unable",
    1: "depth_a",
    2: "depth_b",
    3: "tie_band",
    4: "exact_tie",
    5: "own_layers@1",
    6: "one_plane@1",
    7: "lines@1",
    8: "near_parallel@1",
}
#: A fragment's colour kind (``COLR``): a gradient, not computed; one solid; per pixel (``COLP``).
COLOUR_KIND = {0: "gradient", 1: "solid", 2: "per_pixel"}
#: A pair's edge bits.
EDGE_WEAK, EDGE_A_FRONT, EDGE_B_FRONT = 1, 2, 4
#: Fragment flags: unbounded, ``truth_unable`` at every sample of the window (no cover rows);
#: a mesh triangle whose alpha byte is 0, not present under the contract (no rows, no pairs).
FLAG_UNBOUNDED = 1
FLAG_ZERO_ALPHA = 2
#: Written where a field holds nothing.
NONE_U32 = 0xFFFFFFFF
#: A label's role toward another fragment at a sample (``LABL``).
LABEL_ROLE = {
    0: "front",
    1: "anchor",
    2: "open",
    3: "kind_unknown",
    4: "alpha_straddle",
}
#: An open label pair's cause (``LABL``), and an open class's cause.
PAIR_CAUSE = {
    0: None,
    1: "interval",
    2: "folded",
    3: "disconnected",
    4: "mesh",
    5: "label",
}
CLASS_CAUSE = {
    0: None,
    1: "unmapped",
    2: "generic_only",
    3: "paths_disagree",
    4: "no_source",
    5: "cylinder_grouped",
    6: "point_cloud",
}

DTYPES = {
    "FRAG": [
        ("id", "<u4"),
        ("entry", "<u4"),
        ("slot", "<u4"),
        ("object", "<u4"),
        ("part", "<u4"),
        ("sub", "<u4"),
        ("layer", "u1"),
        ("kind", "u1"),
        ("flags", "<u2"),
        ("group", "<u4"),
        ("epsilon", "<f8"),
        ("bbox", "<i4", (4,)),
        ("plane", "<u4"),
        ("label", "<u4"),
    ],
    "COVR": [
        ("x", "<u4"),
        ("y", "<u4"),
        ("fragment", "<u4"),
        ("cover", "u1", (16,)),
        ("cell_lo", "u1", (16,)),
        ("cell_hi", "u1", (16,)),
        ("pixel_lo", "u1"),
        ("pixel_hi", "u1"),
        ("pad", "u1", (2,)),
        ("pixel_area", "<f8"),
        ("cell_area", "<f8", (16,)),
        ("depth", "<f8", (16,)),
        ("band", "<f8", (16,)),
    ],
    "PIXL": [("first", "<u8"), ("count", "<u4")],
    "BAND": [
        ("x", "<u4"),
        ("y", "<u4"),
        ("fragment", "<u4"),
        ("which", "u1"),
        ("pad", "u1", (3,)),
        ("o", "<f8", (16,)),
        ("r", "<f8", (16,)),
        ("s_minus", "<f8", (16,)),
        ("s_plus", "<f8", (16,)),
        ("status", "u1", (16,)),
    ],
    "COLR": [
        ("fragment", "<u4"),
        ("kind", "u1"),
        ("rgba", "u1", (4,)),
        ("pad", "u1", (3,)),
    ],
    "COLP": [
        ("x", "<u4"),
        ("y", "<u4"),
        ("fragment", "<u4"),
        ("lo", "u1", (4,)),
        ("hi", "u1", (4,)),
        ("reserved", "<u4"),
    ],
    "PAIR": [
        ("x", "<u4"),
        ("y", "<u4"),
        ("a", "<u4"),
        ("b", "<u4"),
        ("sample", "u1"),
        ("edges", "u1"),
        ("label", "u1"),
        ("provenance", "u1"),
        ("offered", "u1"),
        ("plate", "u1"),
        ("pad", "u1", (2,)),
    ],
    "LABL": [
        ("x", "<u4"),
        ("y", "<u4"),
        ("label", "<u4"),
        ("fragment", "<u4"),
        ("sample", "u1"),
        ("role", "u1"),
        ("pair_cause", "u1"),
        ("class_cause", "u1"),
    ],
    "REFU": [
        ("reason", "u1"),
        ("kind", "u1"),
        ("cause", "u1"),
        ("pad", "u1"),
        ("entry", "<u4"),
        ("slot", "<u4"),
        ("object", "<u4"),
        ("bits", "<u8"),
        ("at_seq", "<u8"),
        ("text_start", "<u4"),
        ("text_len", "<u4"),
    ],
}

_FRAG = struct.Struct("<6I2BHId4i2I")
_COVR_HEAD = struct.Struct("<3I16s16s16s2B2x")
_F16 = struct.Struct("<16d")
_PIXL = struct.Struct("<QI")
_REFU = struct.Struct("<3Bx3I2Q2I")
_BAND_HEAD = struct.Struct("<3IB3x")
_PAIR = struct.Struct("<4I6B2x")
_LABL = struct.Struct("<4I4B")
_COLR = struct.Struct("<IB4s3x")
_COLP = struct.Struct("<3I4s4s4x")
_SIZES = {
    "FRAG": 64,
    "COVR": 456,
    "PIXL": 12,
    "BAND": 544,
    "PAIR": 24,
    "COLR": 12,
    "COLP": 24,
    "LABL": 20,
    "REFU": 40,
}


class FormatError(ValueError):
    """The file is not an oracle decisions file this reader knows."""


def _header(b: bytes) -> tuple[dict, list[tuple[str, int, int, int]]]:
    if b[:4] != MAGIC:
        raise FormatError("not an oracle decisions file")
    fmt, header_size, xf_major, xf_minor, width, height = struct.unpack_from(
        "<4H2I", b, 4
    )
    if fmt != ORACLE_FORMAT:
        raise FormatError(f"ORACLE_FORMAT {fmt} is not one this reader knows")
    window = list(struct.unpack_from("<4I", b, 20))
    status = b[36]
    (seq,) = struct.unpack_from("<Q", b, 40)
    raw = struct.unpack_from("<32b", b, 48)
    offsets = [[raw[2 * i], raw[2 * i + 1]] for i in range(16)]
    (count,) = struct.unpack_from("<I", b, 80)
    directory = []
    for i in range(count):
        tag, size, off, n = struct.unpack_from("<4sIQQ", b, header_size + 24 * i)
        directory.append((tag.decode("ascii"), size, off, n))
    header = {
        "format": fmt,
        "xf": [xf_major, xf_minor],
        "width": width,
        "height": height,
        "window": window,
        "status": "decided" if status == 0 else "truth_unable",
        "seq": seq,
        "offsets": offsets,
    }
    return header, directory


def sections(path: str) -> dict[str, dict]:
    """Each section's tag mapped to its ``offset``, ``count`` and ``record_size``."""
    with open(path, "rb") as f:
        b = f.read()
    _, directory = _header(b)
    return {
        tag: {"offset": off, "count": n, "record_size": size}
        for tag, size, off, n in directory
    }


def read_bytes(b: bytes) -> dict:
    header, directory = _header(b)
    found = {tag: (size, off, n) for tag, size, off, n in directory}
    for tag, size in _SIZES.items():
        if tag in found and found[tag][0] != size:
            raise FormatError(
                f"section {tag} has records of {found[tag][0]} bytes, not {size}"
            )
    out = {
        "header": header,
        "fragments": [],
        "covers": [],
        "pixels": [],
        "bands": [],
        "pairs": [],
        "colours": [],
        "pixel_colours": [],
        "labels": [],
        "refused": [],
    }
    if "FRAG" in found:
        size, off, n = found["FRAG"]
        for i in range(n):
            v = _FRAG.unpack_from(b, off + i * size)
            out["fragments"].append(
                {
                    "id": v[0],
                    "entry": v[1],
                    "slot": v[2],
                    "object": v[3],
                    "part": v[4],
                    "sub": v[5],
                    "layer": v[6],
                    "kind": v[7],
                    "flags": v[8],
                    "group": v[9],
                    "epsilon": v[10],
                    "bbox": list(v[11:15]),
                    "plane": v[15],
                    "label": v[16],
                }
            )
    if "COVR" in found:
        size, off, n = found["COVR"]
        for i in range(n):
            at = off + i * size
            x, y, fragment, cover, lo, hi, plo, phi = _COVR_HEAD.unpack_from(b, at)
            (area,) = struct.unpack_from("<d", b, at + 64)
            out["covers"].append(
                {
                    "x": x,
                    "y": y,
                    "fragment": fragment,
                    "cover": list(cover),
                    "cell_lo": list(lo),
                    "cell_hi": list(hi),
                    "pixel_lo": plo,
                    "pixel_hi": phi,
                    "pixel_area": area,
                    "cell_area": list(_F16.unpack_from(b, at + 72)),
                    "depth": list(_F16.unpack_from(b, at + 200)),
                    "band": list(_F16.unpack_from(b, at + 328)),
                }
            )
    if "PIXL" in found:
        size, off, n = found["PIXL"]
        out["pixels"] = [list(_PIXL.unpack_from(b, off + i * size)) for i in range(n)]
    if "BAND" in found:
        size, off, n = found["BAND"]
        for i in range(n):
            at = off + i * size
            x, y, fragment, which = _BAND_HEAD.unpack_from(b, at)
            out["bands"].append(
                {
                    "x": x,
                    "y": y,
                    "fragment": fragment,
                    "which": which,
                    "o": list(_F16.unpack_from(b, at + 16)),
                    "r": list(_F16.unpack_from(b, at + 144)),
                    "s_minus": list(_F16.unpack_from(b, at + 272)),
                    "s_plus": list(_F16.unpack_from(b, at + 400)),
                    "status": list(b[at + 528 : at + 544]),
                }
            )
    if "PAIR" in found:
        size, off, n = found["PAIR"]
        for i in range(n):
            x, y, a, b_, sample, edges, label, provenance, offered, plate = (
                _PAIR.unpack_from(b, off + i * size)
            )
            out["pairs"].append(
                {
                    "x": x,
                    "y": y,
                    "a": a,
                    "b": b_,
                    "sample": sample,
                    "edges": edges,
                    "label": label,
                    "label_name": LABEL.get(label, "unknown"),
                    "provenance": provenance,
                    "offered": offered,
                    "plate": plate,
                }
            )
    if "LABL" in found:
        size, off, n = found["LABL"]
        for i in range(n):
            x, y, label, fragment, sample, role, pc, cc = _LABL.unpack_from(
                b, off + i * size
            )
            out["labels"].append(
                {
                    "x": x,
                    "y": y,
                    "label": label,
                    "fragment": fragment,
                    "sample": sample,
                    "role": role,
                    "role_name": LABEL_ROLE.get(role, "unknown"),
                    "pair_cause": pc,
                    "class_cause": cc,
                }
            )
    if "COLR" in found:
        size, off, n = found["COLR"]
        for i in range(n):
            fragment, kind, rgba = _COLR.unpack_from(b, off + i * size)
            out["colours"].append(
                {
                    "fragment": fragment,
                    "kind": kind,
                    "kind_name": COLOUR_KIND.get(kind, "unknown"),
                    "rgba": list(rgba),
                }
            )
    if "COLP" in found:
        size, off, n = found["COLP"]
        for i in range(n):
            x, y, fragment, lo, hi = _COLP.unpack_from(b, off + i * size)
            out["pixel_colours"].append(
                {"x": x, "y": y, "fragment": fragment, "lo": list(lo), "hi": list(hi)}
            )
    text = b""
    if "TEXT" in found:
        _, off, n = found["TEXT"]
        text = b[off : off + n]
    if "REFU" in found:
        size, off, n = found["REFU"]
        for i in range(n):
            reason, kind, cause, entry, slot, obj, bits, at_seq, start, length = (
                _REFU.unpack_from(b, off + i * size)
            )
            out["refused"].append(
                {
                    "reason": reason,
                    "reason_name": REASON.get(reason, "unknown"),
                    "kind": kind,
                    "cause": cause,
                    "cause_name": CAUSE.get(cause),
                    "entry": entry,
                    "slot": slot,
                    "object": obj,
                    "bits": bits,
                    "at_seq": at_seq,
                    "detail": text[start : start + length].decode("utf-8", "replace"),
                }
            )
    return out


def read(path: str) -> dict:
    """The decisions file at ``path``."""
    with open(path, "rb") as f:
        return read_bytes(f.read())


def _hex(v: float) -> str:
    return struct.pack("<d", v)[::-1].hex()


def _list(vs) -> str:
    return ",".join(str(v) for v in vs)


def canonical(d: dict) -> str:
    """The decisions as canonical text, one line per item."""
    h = d["header"]
    lines = [
        f"header format={h['format']} xf={h['xf'][0]}.{h['xf'][1]} size={h['width']}x{h['height']} "
        f"window={_list(h['window'])} status={h['status']} seq={h['seq']}",
        "offsets " + " ".join(f"{o[0]},{o[1]}" for o in h["offsets"]),
    ]
    lines.extend(
        f"fragment id={f['id']} entry={f['entry']} slot={f['slot']} object={f['object']} "
        f"part={f['part']} sub={f['sub']} layer={f['layer']} kind={f['kind']} flags={f['flags']} "
        f"group={f['group']} plane={f['plane']} label={f['label']} epsilon={_hex(f['epsilon'])} "
        f"bbox={_list(f['bbox'])}"
        for f in d["fragments"]
    )
    lines.extend(
        f"cover x={c['x']} y={c['y']} fragment={c['fragment']} cover={_list(c['cover'])} "
        f"cell={','.join(f'{lo}-{hi}' for lo, hi in zip(c['cell_lo'], c['cell_hi'], strict=True))} "
        f"pixel={c['pixel_lo']}-{c['pixel_hi']} area={_hex(c['pixel_area'])} "
        f"cells={_list(_hex(v) for v in c['cell_area'])} depth={_list(_hex(v) for v in c['depth'])} "
        f"band={_list(_hex(v) for v in c['band'])}"
        for c in d["covers"]
    )
    lines.extend(
        f"band x={r['x']} y={r['y']} fragment={r['fragment']} which={r['which']} "
        f"o={_list(_hex(v) for v in r['o'])} r={_list(_hex(v) for v in r['r'])} "
        f"s_minus={_list(_hex(v) for v in r['s_minus'])} "
        f"s_plus={_list(_hex(v) for v in r['s_plus'])} status={_list(r['status'])}"
        for r in d["bands"]
    )
    lines.extend(
        f"pair x={r['x']} y={r['y']} sample={r['sample']} a={r['a']} b={r['b']} "
        f"edges={r['edges']} label={r['label']} provenance={r['provenance']} "
        f"offered={r['offered']} plate={r['plate']}"
        for r in d["pairs"]
    )
    lines.extend(
        f"label x={r['x']} y={r['y']} sample={r['sample']} label={r['label']} "
        f"fragment={r['fragment']} role={r['role']} pair_cause={r['pair_cause']} "
        f"class_cause={r['class_cause']}"
        for r in d["labels"]
    )
    lines.extend(
        f"colour fragment={r['fragment']} kind={r['kind']} rgba={_list(r['rgba'])}"
        for r in d["colours"]
    )
    lines.extend(
        f"pixel_colour x={r['x']} y={r['y']} fragment={r['fragment']} lo={_list(r['lo'])} "
        f"hi={_list(r['hi'])}"
        for r in d["pixel_colours"]
    )
    lines.extend(
        f"refused reason={r['reason']} kind={r['kind']} cause={r['cause']} entry={r['entry']} "
        f"slot={r['slot']} object={r['object']} bits={r['bits']} at_seq={r['at_seq']} detail={r['detail']}"
        for r in d["refused"]
    )
    return "".join(line + "\n" for line in lines)


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[0] != "--canonical":
        print("usage: read.py --canonical <file>", file=sys.stderr)
        return 2
    sys.stdout.write(canonical(read(argv[1])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
