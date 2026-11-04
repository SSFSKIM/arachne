# The parity kit

What a backend other than the core's CPU path must draw before it is used: the class probes'
frames, within a bound per resolve class.

- `probes.rs`: the fixtures. The class probes (two surfaces crossing, a curve behind a surface,
  transparency over an opaque surface, a label fixed in frame over a surface, the depth rule both
  ways, a stroke crossing a surface, a surface edge-on, an orbiting camera) and the seam probe,
  built in code as XF streams, with the pixels each one asserts where CE's painter order is wrong
  (`tests/probes.rs`). Compiled into the core as `arachne_core::probes`; the binding's
  `probe_fixtures()` hands them to Python.
- `references/<fixture>.npz`: the core's frames of each fixture (`frame_data`, uint8, frames x
  height x width x 4); `references/<fixture>.resolves.npz`, each reference pixel's resolve class
  (`frame_data`, uint8, frames x height x width x 3, the index into `RESOLVES` — background, fast,
  coverage, sample — in every channel); `references/streams.json`, the sha256 of the stream each
  fixture's references were drawn from. One uint8 `frame_data` per file is the only `.npz` the
  repository's leak check admits.
- `tolerances.json`: per resolve class, the largest channel difference a backend's pixel may show
  against the reference pixel of that class.

Streams are never stored, in any encoding: a backend regenerates them at the pinned revision.

    python -m arachne_core.parity --backend cpu                  # PASS <fixture> ..., parity: n/n pass
    python -m arachne_core.parity --backend cpu --regenerate     # rewrite references that changed
    python -m arachne_core.parity --backend cpu --write-streams DIR   # the fixtures as XF files

Run from `core/` (or pass `--kit`). A change to a probe changes its stream's sha256, and the runner
fails that fixture until its reference is regenerated.
