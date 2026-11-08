"""The fonts a Text is shaped with, by content: what the typesetting cache
(:mod:`.typeset_cache`) keys a Text's SVG by, beside its arguments.

Pango itself is asked, through the library ManimPango loaded, so the answer
is what renders on this platform (fontconfig's choice on Linux, CoreText's
on macOS): each span is laid out as ManimPango lays it out, and every run of
the layout names the font it is shaped with, the fonts Pango falls back to
for characters the span's font lacks included. A font is named by Pango's
description of it, the digest of its data (its font file, or its tables, as
HarfBuzz reads them) and its variation coordinates.

Where Pango cannot be asked so (a ManimPango built without these symbols, or
a layout whose runs are not as this module reads them), the answer names the
host instead: a file kept then is never shared with another host, rather
than shared on a guess.
"""

from __future__ import annotations

__all__ = ["text_fonts"]

import ctypes
import functools
import hashlib
import socket
from collections.abc import Sequence
from typing import Any

P = ctypes.c_void_p
U = ctypes.c_uint

#: The HarfBuzz tag of a font's ``head`` table, whose checksum covers the
#: whole font file.
_HEAD = int.from_bytes(b"head", "big")


class _Analysis(ctypes.Structure):
    # The start of the public ``PangoAnalysis``: two engine pointers, kept
    # for the ABI, then the font the run is shaped with.
    _fields_ = [("shape_engine", P), ("lang_engine", P), ("font", P)]


class _Item(ctypes.Structure):
    # The public ``PangoItem``, up to its analysis.
    _fields_ = [
        ("offset", ctypes.c_int),
        ("length", ctypes.c_int),
        ("num_chars", ctypes.c_int),
        ("analysis", _Analysis),
    ]


class _GlyphItem(ctypes.Structure):
    # The public ``PangoGlyphItem``, a layout's run, up to its glyphs.
    _fields_ = [("item", ctypes.POINTER(_Item)), ("glyphs", P)]


#: Each face's data digest, by what names the face: its description, its
#: ``head`` table and its data's size.
_DIGESTS: dict[tuple[str, bytes, int], str] = {}


class _Pango:
    """The Pango, GLib and HarfBuzz functions this module calls, from the
    library ManimPango loaded; ``OSError`` where one is missing or a layout
    is not as this module reads it.
    """

    def __init__(self) -> None:
        from manimpango import cmanimpango

        library = ctypes.CDLL(cmanimpango.__file__)

        def bind(name: str, result: Any, *args: Any) -> Any:
            try:
                function = getattr(library, name)
            except AttributeError as missing:
                raise OSError(f"ManimPango exports no {name}.") from missing
            function.restype = result
            function.argtypes = list(args)
            return function

        self.font_map = bind("pango_cairo_font_map_get_default", P)
        self.context = bind("pango_font_map_create_context", P, P)
        self.layout = bind("pango_layout_new", P, P)
        self.unref = bind("g_object_unref", None, P)
        self.free = bind("g_free", None, P)
        self.new = bind("pango_font_description_new", P)
        self.free_description = bind("pango_font_description_free", None, P)
        self.family = bind(
            "pango_font_description_set_family", None, P, ctypes.c_char_p
        )
        self.style = bind("pango_font_description_set_style", None, P, ctypes.c_int)
        self.weight = bind("pango_font_description_set_weight", None, P, ctypes.c_int)
        self.size = bind("pango_font_description_set_size", None, P, ctypes.c_int)
        self.set_font = bind("pango_layout_set_font_description", None, P, P)
        self.text = bind(
            "pango_layout_set_text", None, P, ctypes.c_char_p, ctypes.c_int
        )
        self.markup = bind(
            "pango_layout_set_markup", None, P, ctypes.c_char_p, ctypes.c_int
        )
        self.iterate = bind("pango_layout_get_iter", P, P)
        self.free_iterator = bind("pango_layout_iter_free", None, P)
        self.run = bind(
            "pango_layout_iter_get_run_readonly", ctypes.POINTER(_GlyphItem), P
        )
        self.next_run = bind("pango_layout_iter_next_run", ctypes.c_int, P)
        self.is_a = bind("g_type_check_instance_is_a", ctypes.c_int, P, ctypes.c_size_t)
        self.font_type = bind("pango_font_get_type", ctypes.c_size_t)()
        self.describe = bind("pango_font_describe", P, P)
        self.to_string = bind("pango_font_description_to_string", P, P)
        self.hb_font = bind("pango_font_get_hb_font", P, P)
        self.face = bind("hb_font_get_face", P, P)
        self.blob = bind("hb_face_reference_blob", P, P)
        self.table = bind("hb_face_reference_table", P, P, ctypes.c_uint32)
        self.tags = bind(
            "hb_face_get_table_tags",
            U,
            P,
            U,
            ctypes.POINTER(U),
            ctypes.POINTER(ctypes.c_uint32),
        )
        self.data = bind("hb_blob_get_data", P, P, ctypes.POINTER(U))
        self.length = bind("hb_blob_get_length", U, P)
        self.destroy = bind("hb_blob_destroy", None, P)
        self.coords = bind(
            "hb_font_get_var_coords_normalized",
            ctypes.POINTER(ctypes.c_int),
            P,
            ctypes.POINTER(U),
        )
        # One character in the default font: one run of one item, whose
        # font is a Pango font; else the runs are not read as they are laid.
        self.fonts([("a", "", 0, 400)], 10 * 1024, markup=False, check=True)

    def fonts(
        self,
        spans: Sequence[tuple[str, str, int, int]],
        size: int,
        *,
        markup: bool,
        check: bool = False,
    ) -> set[tuple[str, str]]:
        """Each font a run is shaped with, laying out each of ``spans`` (its
        text or markup, family, Pango style and weight) at ``size`` Pango
        units, by :meth:`named`. Read while the layout holds them.
        """
        found: dict[int, tuple[str, str]] = {}
        items = []
        context = self.context(self.font_map())
        layout = self.layout(context)
        try:
            for text, family, style, weight in spans:
                description = self.new()
                if family:
                    self.family(description, family.encode())
                self.style(description, style)
                self.weight(description, weight)
                self.size(description, size)
                self.set_font(layout, description)
                self.free_description(description)
                (self.markup if markup else self.text)(layout, text.encode(), -1)
                iterator = self.iterate(layout)
                try:
                    while True:
                        run = self.run(iterator)
                        if run:
                            item = run.contents.item.contents
                            items.append((item.offset, item.length, item.num_chars))
                            if check and items != [(0, 1, 1)]:
                                raise OSError(
                                    "A Pango layout's runs are not as this module reads them."
                                )
                            font = item.analysis.font
                            if not font or not self.is_a(font, self.font_type):
                                raise OSError("A layout's run has no Pango font.")
                            if font not in found:
                                found[font] = self.named(font)
                        if not self.next_run(iterator):
                            break
                finally:
                    self.free_iterator(iterator)
        finally:
            self.unref(layout)
            self.unref(context)
        if check and len(items) != 1:
            raise OSError("A Pango layout's runs are not as this module reads them.")
        return set(found.values())

    def named(self, font: int) -> tuple[str, str]:
        """``font``'s description (with its variation coordinates) and the
        digest of its data.
        """
        description = self.describe(font)
        text = self.to_string(description)
        try:
            name = ctypes.string_at(text).decode()
        finally:
            self.free(text)
            self.free_description(description)
        hb_font = self.hb_font(font)
        face = self.face(hb_font)
        count = U(0)
        values = self.coords(hb_font, ctypes.byref(count))
        if count.value:
            name += " @" + ",".join(str(values[i]) for i in range(count.value))
        blob = self.blob(face)
        size = self.length(blob)
        self.destroy(blob)
        key = (name, self._bytes(self.table(face, _HEAD)), size)
        if key not in _DIGESTS:
            _DIGESTS[key] = _digest(self._data(face))
        return name, _DIGESTS[key]

    def _bytes(self, blob: int) -> bytes:
        try:
            length = U(0)
            data = self.data(blob, ctypes.byref(length))
            return ctypes.string_at(data, length.value) if length.value else b""
        finally:
            self.destroy(blob)

    def _data(self, face: int) -> bytes:
        """The face's data: its font file's, where HarfBuzz holds it whole,
        else every table it names, by tag.
        """
        whole = self._bytes(self.blob(face))
        if whole:
            return whole
        total = self.tags(face, 0, None, None)
        tags = (ctypes.c_uint32 * total)()
        count = U(total)
        self.tags(face, 0, ctypes.byref(count), tags)
        parts = [
            tag.to_bytes(4, "big") + self._bytes(self.table(face, tag))
            for tag in sorted(tags[: count.value])
        ]
        if not parts:
            raise OSError("HarfBuzz reads no data of a font.")
        return b"".join(parts)


@functools.cache
def _loaded() -> _Pango:
    return _Pango()


def _pango() -> _Pango:
    """The functions this module calls (``OSError`` where they cannot be)."""
    try:
        return _loaded()
    except ImportError as missing:
        raise OSError(str(missing)) from missing


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_fonts(
    spans: Sequence[tuple[str, str, str, str]], size: float, *, markup: bool = False
) -> list[list[str]]:
    """The fonts ``spans`` (each its text, or markup, and its family, slant
    and weight, by ManimPango's names) are shaped with at ``size`` points:
    each ``["font", description, digest]``, sorted, once each; or, where
    Pango cannot be asked, ``[["host", name]]``.
    """
    from manimpango import PangoUtils

    laid = [
        (
            text,
            family,
            PangoUtils.str2style(slant).value,
            PangoUtils.str2weight(weight).value,
        )
        for text, family, slant, weight in spans
        if text
    ]
    try:
        fonts = _pango().fonts(laid, int(size * 1024 + 0.5), markup=markup)
    except OSError:
        return [["host", socket.gethostname()]]
    return sorted(["font", name, digest] for name, digest in fonts)
