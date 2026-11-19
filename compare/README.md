# arachne-compare

Compares two sequences of frames and blocks only on differences a viewer could see. It needs
`numpy` and `scipy`, and nothing of Arachne, so a renderer's own test kit can call it.

- **Presence**: an object drawn on one side and not on the other, at any size.
- **Cluster**: an 8-connected cluster of changed pixels covering more than 0.02% of the frame,
  in which some pixel changed by more than 16 of 255 in any channel.
- **Flicker**: such a cluster between consecutive frames, at pixels of no object that changed.

Everything smaller is counted and never blocks. Two modes say what "changed" means. `same`
compares one renderer with itself: any difference counts. `cross` compares two renderers or
two drivers: a pixel counts only above a floor (8 of 255), and the pixels on the boundary of
each object's footprint in the reference's object map are counted per object, against an
optional budget, instead of joining clusters. Below 1080 lines of height an edge pixel is a
footprint pixel with one of its four neighbours outside it; from 1080, every pixel within `height // 1080`
of one too.

The cheap pass (`cheap`, `cheap_self`) reads every frame. The structural pass (`structural`,
`flicker`) reads the frames `sample` picks, with each side's `ObjectMap` and the `ChangedSet`
of objects that changed between two frames. `compare` runs both and returns a `Verdict`
whose `to_json()` is canonical. A backend hands maps in (`HandedMaps`), or a `MapSource`
lists boxes and draws maps for the objects `choose_objects` picks.

```python
from fractions import Fraction
from arachne_compare import ArrayFrames, ChangedSet, HandedMaps, compare

verdict = compare(
    ArrayFrames(reference_frames, Fraction(30)),
    ArrayFrames(candidate_frames, Fraction(30)),
    mode="cross",
    maps=HandedMaps(reference_maps, candidate_maps),
    changed=lambda a, b: ChangedSet(frozenset()),
    plays=[(0, len(reference_frames))],
    fps=Fraction(30),
    seed=0,
    key="case-1",
)
print(verdict.verdict, verdict.to_json())
```

`python -m arachne_compare.calibration <base 1920x1080 png> <base 480x270 png> <out dir>`
writes pairs of frames just under and just over each threshold, with the verdict in each mode.
