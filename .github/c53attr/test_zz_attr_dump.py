"""Scratch: the core's and CE's frames of test_face_ends' pinned ellipsoid."""

import hashlib
import os

import numpy as np

from arachne import Create

from . import test_face_ends as t


def test_dump():
    out = {}
    for alpha in (0.0, 1.0):
        _, frame = t._painted(t._at(t._ellipsoid(), Create, alpha))
        out[f"core_{alpha}"] = np.asarray(frame)
        out[f"ce_{alpha}"] = t._ce(t._at(t._ellipsoid(), Create, alpha))
        print("HASH", alpha, hashlib.sha256(frame.tobytes()).hexdigest())
    np.savez_compressed(os.environ["ATTR_DUMP"], **out)
