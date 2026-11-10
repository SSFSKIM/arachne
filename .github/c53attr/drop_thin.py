"""Scratch: with ATTR_DROP_THIN set, the core skips every stroke layer under 0.8 px that is not
drawn as a dot (the strokes stroke_tolerance changes), so a frame without them shows their pixels."""

from pathlib import Path

path = Path("core/src/path.rs")
text = path.read_text()
anchor = "            let Some((spec, width)) = layer else { continue };\n            let k = k as u32;\n"
assert text.count(anchor) == 1, "anchor"
insert = (
    "            if k != 1\n"
    '                && std::env::var_os("ATTR_DROP_THIN").is_some()\n'
    "                && !strokes_as_dot(draw, &shown)\n"
    "                && width * self.view.cairo_line_width_multiple * scale < 0.8\n"
    "            {\n"
    "                continue;\n"
    "            }\n"
)
path.write_text(text.replace(anchor, anchor + insert))
