"""The lint's codes: stable strings, append-only.

Each code has its severity (``error``, ``warning`` or ``note``), its
``class`` (the name a report gives its kind, as the exception names of the
checks the clearance family comes from), what it means, how to fix it, and
the fields its findings' ``detail`` may carry. A code is never renamed,
removed or given another meaning; new ones are appended. Thresholds are in
scene units of the home frame, with pixel forms stated at a picture 1080
pixels tall (:data:`THRESHOLD_HEIGHT`).

``render.unsupported`` and ``render.fallback`` are the drawing's codes,
here with the strings, severities and meanings the drawing gave them; the
Frame IR's own table re-exports them from here once both live in one tree.
The ``image.`` codes are reserved for the image checks.

A clearance finding of the temporal rules (:mod:`.watch`, :mod:`.departures`)
may carry ``clock_margin``: seconds past the duration that decided it (held
long enough, settled long enough), when that is less than a frame of the
render's grid, where an engine whose plays last whole frames can decide
otherwise.
"""

from __future__ import annotations

__all__ = [
    "CODES",
    "MOTION_REFERENCE_FPS",
    "REFERENCE_RESOLUTION",
    "RESERVED",
    "SEVERITIES",
    "THRESHOLD_HEIGHT",
    "Code",
    "code",
]

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

#: The severities, gravest first. Only an ``error`` fails a report.
SEVERITIES = ("error", "warning", "note")

#: Code prefixes kept for checks to come.
RESERVED = ("image.",)

#: The picture height, in pixels, the table's pixel thresholds are stated
#: at; a rule scales them to the home frame.
THRESHOLD_HEIGHT = 1080
#: The picture every pixel threshold is measured on: production's delivery
#: size, 1080 pixels on its shorter side, so one reference pixel is the
#: home frame's shorter side over :data:`THRESHOLD_HEIGHT`. Deliberately not
#: the session's render size: a threshold must not move with the quality a
#: scene is rendered or linted at.
REFERENCE_RESOLUTION = (1920, THRESHOLD_HEIGHT)
#: The clock motion is measured on, in frames a second: a body's steps, its
#: travel and what a step across a cut may be within, read at the frames a
#: render shows at each of its instants. A 15 fps render shows every one, a
#: faster one is read at the frames it shows at them. Deliberately not the
#: render's rate: a verdict must not move with the rate a scene is rendered
#: or linted at.
MOTION_REFERENCE_FPS = 15


@dataclass(frozen=True)
class Code:
    """One lint code: its string, severity, class, meaning and fix, the
    names of the ``detail`` fields its findings may carry (besides
    ``folded``, which a report adds to any finding look-alikes fold into),
    and whether look-alikes of it fold: a code whose every finding is an
    event of its own, at a time of its own, never does.
    """

    code: str
    severity: str
    cls: str
    meaning: str
    fix: str
    detail: tuple[str, ...] = ()
    folds: bool = True


_CLEARANCE = "Clearance"

_TABLE = (
    Code(
        "clip.frame_edge",
        "error",
        _CLEARANCE,
        "a text's glyphs, or a shape that was wholly inside, run past the frame's edge",
        "Move or shrink it so it sits inside the frame with a margin, or give the"
        " layout room before it is placed.",
        ("edge", "past", "clock_margin"),
    ),
    Code(
        "overlap.text",
        "error",
        _CLEARANCE,
        "a text overlaps another text or a shape",
        "Move one of the two apart, or place one against the other with a gap.",
        ("overlap", "clock_margin"),
    ),
    Code(
        "gap.text",
        "error",
        _CLEARANCE,
        "two texts sit closer than the gap the scene declared for texts",
        "Widen the space between the two texts to the declared gap.",
        ("gap", "required", "clock_margin"),
    ),
    Code(
        "contact.held",
        "error",
        _CLEARANCE,
        "two things that are not text touch or overlap",
        "Move them apart, or declare the contact meant by exempting the pair.",
        ("gap", "required", "clock_margin"),
    ),
    Code(
        "occlusion.glyph",
        "error",
        _CLEARANCE,
        "something drawn after a text paints over its glyphs",
        "Draw the text after what covers it, raise it in the paint order, or move"
        " the covering shape off it.",
        ("covered", "clock_margin"),
    ),
    Code(
        "label.left_behind",
        "error",
        _CLEARANCE,
        "a label stays where its referent was after the referent moved away",
        "Move the label with its referent: carry it in the same animation, or"
        " place it from the referent's position each frame.",
        ("referent", "distance", "clock_margin"),
    ),
    Code(
        "axes.outside",
        "error",
        _CLEARANCE,
        "a point or mobject asserted to lie on the axes' plotted region lies outside it",
        "Keep the point within the axes' ranges: clamp it, widen the ranges, or"
        " stop drawing where it leaves them.",
        ("axes", "point"),
    ),
    Code(
        "replay.diverged",
        "error",
        "ReplayDiverged",
        "two executions of one source differ at an instant both evaluated (an"
        " error: what is shown may not be what was checked), or a play restored"
        " at a recorded instant differs from what was recorded there and was"
        " ticked instead (a note: the engine corrected itself)",
        "Make the scene deterministic: seed its random generators and keep the"
        " files it reads unchanged between runs.",
        ("kind", "play", "reason", "changed_inputs"),
    ),
    Code(
        "source.changed",
        "note",
        "SourceChanged",
        "the scene's source changed since its record was made, so the record"
        " is of a program the file no longer is",
        "Open the scene again to lint the source as it is now.",
    ),
    Code(
        "motion.jump",
        "warning",
        "MotionJump",
        "a body visible on both sides of the boundary between two plays is in"
        " another place after it, with nothing carrying it, while it moves on"
        " either side",
        "Animate the move, or fade the body out and in again if the cut is meant.",
        ("side", "distance", "from", "to", "boundary"),
        folds=False,
    ),
    Code(
        "render.unsupported",
        "error",
        "Unsupported",
        "a frame's description carries a record kind or feature bit the drawer"
        " does not know; the frame is refused, never drawn partly",
        "Draw the scene with Cairo, or leave out the feature the drawer does not know.",
        ("subject",),
    ),
    Code(
        "render.fallback",
        "warning",
        "RenderFallback",
        "a scene was drawn with Cairo because the Frame IR's path refused"
        " something in it (a camera class of the scene's own; for a 3D scene,"
        " anything the core refuses); the finding names the cause and its"
        " source line",
        "Use the engine's own camera classes and methods unchanged for the"
        " faster path.",
        ("camera", "cause", "methods", "where"),
    ),
    Code(
        "text.small",
        "note",
        "TextSmall",
        "a text is shorter than 20 pixels at 1080 pixels tall, hard to read",
        "Scale the text up, or give it a larger font size.",
        ("height_px",),
    ),
    Code(
        "sim.quantity_unreadable",
        "error",
        "QuantityUnreadable",
        "a declared quantity of an object in the scene has no readable value"
        " at the instant: its name has no registered prefix, its unit is not"
        " registered, its value is not finite or not of its kind, or its source"
        " holds no value there (a simulation's quantity outside the stay it was"
        " baked over); it is answered as null with the reason, never as a number",
        "Declare the quantity under a registered prefix and unit with a finite"
        " value of its kind, and read it only where its source holds a value.",
        ("name", "unit", "reason"),
    ),
    Code(
        "render.gpu_required",
        "error",
        "GpuRequired",
        "a scene holding a GPU-only capability was asked to draw on a host with"
        " no GPU; nothing is drawn or written. Raised by drawing calls only,"
        " never listed by the lint, whose findings are the same on every host",
        "Render on a GPU host; lint and state_at answer here.",
        ("capabilities", "play", "object"),
    ),
    Code(
        "render.gpu_live",
        "error",
        "GpuLive",
        "a scene holding a GPU-only capability has its frames drawn before its"
        " record is complete, which only Cairo does; nothing is written. Listed"
        " by the lint on every host",
        "Remove what makes the scene draw as it runs, which the message names,"
        " or the GPU-only capability.",
        ("capabilities", "play", "object", "cause"),
    ),
    Code(
        "asset.manifest_missing",
        "error",
        "AssetManifestMissing",
        "an asset asked for through the asset API has no manifest: its"
        " directory holds no manifest.json, or no such directory or fixture"
        " exists",
        "Give the asset's directory its manifest.json, or name a directory that"
        " has one: a path as given, or under the assets directory, or"
        " fixture:<name> for one of the engine's fixtures.",
        ("asset", "searched"),
    ),
    Code(
        "asset.manifest_invalid",
        "error",
        "AssetManifestInvalid",
        "an asset's manifest cannot be parsed, is of a major version or kind the"
        " reader does not know, requires a feature the reader lacks, or misses"
        " or contradicts a required member",
        "Correct the member the finding names, as the asset manifest's rules"
        " state it, or have the asset's maker export it again.",
        ("asset", "member", "reason", "name"),
    ),
    Code(
        "asset.file_missing",
        "error",
        "AssetFileMissing",
        "a file an asset's manifest names is not there",
        "Put the file at the path the manifest gives, relative to the asset's"
        " directory, or correct the path in the manifest.",
        ("asset", "file"),
    ),
    Code(
        "asset.unreadable",
        "error",
        "AssetUnreadable",
        "a file of an asset cannot be read or parsed, or holds something the"
        " asset reader does not draw exactly as the file says (text, an embedded"
        " image, a gradient or pattern paint, a dash, a clip path, a mask, a"
        " filter, a percentage length, a reference to another file or to"
        " nothing), or draws nothing",
        "Export the file again as plain vector shapes with flat colours: outline"
        " text, flatten masks, clips and filters, and embed nothing.",
        ("asset", "file", "reason"),
    ),
    Code(
        "asset.remote",
        "error",
        "AssetRemote",
        "an asset is named by an address that would be fetched, or its manifest"
        " names a file by anything but a plain path inside the asset's"
        " directory: an address, an absolute path, a path with .. or a"
        " backslash, or one through a link",
        "Copy the file into the asset's directory and name it by a plain path"
        " relative to it; assets are read from local directories only.",
        ("asset", "file"),
    ),
    Code(
        "rig.name_unknown",
        "error",
        "RigNameUnknown",
        "a rig is asked for a part, parameter, slot or drawing its manifest"
        " does not have; the finding's kind says which",
        "Use one of the names the asset's manifest declares; the finding lists"
        " the nearest.",
        ("asset", "kind", "name", "nearest"),
    ),
    Code(
        "rig.range",
        "error",
        "RigRange",
        "a rig parameter is given a value outside the range its asset was tested"
        " over, from its manifest's min to its max",
        "Give the parameter a value from its min to its max; a pose outside them"
        " was never checked to draw whole.",
        ("asset", "parameter", "value", "min", "max"),
    ),
    Code(
        "annotation.anchor_unknown",
        "error",
        "AnnotationAnchorUnknown",
        "an anchor is asked of an asset whose manifest does not have it: the lint"
        " reports it, and drawing refuses the scene",
        "Use one of the anchors the asset's manifest declares; the finding lists"
        " the nearest.",
        ("asset", "name", "nearest"),
    ),
    Code(
        "asset.unadmitted",
        "error",
        "AssetUnadmitted",
        "an asset's manifest does not hold the complete, passing checks of the"
        " admission tool this engine supports, recorded at the digest of the"
        " asset as it is: it was never admitted, was admitted by another tool"
        " version or has changed since, or a check is missing or failing; the"
        " finding's reason says which",
        "Admit the asset with python -m arachne.assets admit <its directory>"
        " --write; where a check fails, correct the asset at its source and"
        " admit it again.",
        ("asset", "reason", "checks", "expected", "found"),
    ),
    Code(
        "annotation.detached",
        "error",
        "AnnotationDetached",
        "an annotation points more than a reference pixel from its anchor at an"
        " instant the lint reads: it reads the anchor before what moves the"
        " anchor's object, a frame late",
        "Add the annotation to the scene after what it points at (and after"
        " whatever moves that with an updater), so its updater runs after theirs.",
        ("distance", "allowed", "anchor"),
    ),
    Code(
        "timing.exact_content",
        "error",
        "TimingExactContent",
        "stepped timing or line boil is declared on a family that holds exact"
        " content: text, a number, a coordinate system or a plot, an annotation,"
        " a schematic mark, a diagram, or the scene's camera's frame, whose"
        " drawing must change at every frame",
        "Declare the timing on the illustration alone, a group that holds"
        " none of these; keep labels, plots, annotation and the camera beside it,"
        " where they stay smooth.",
        ("object", "camera"),
    ),
    Code(
        "timing.updater",
        "error",
        "TimingUpdater",
        "an object with declared timing is moved by something the engine cannot"
        " hold to its timing: the scene's own updater on it, an animation of the"
        " scene's own or UpdateFromFunc or UpdateFromAlphaFunc on it, an engine"
        " animation that moves it from where it stands (MoveAlongPath, PhaseFlow,"
        " MaintainPositionRelativeTo), a Succession's part that begins in the"
        " middle of a play after a part that does not animate it, its updating"
        " suspended by the scene's code, an animation that takes it out of the"
        " scene without suspending its updating or as a Succession's part, or a"
        " scene updater registered after the timing was declared",
        "Move the illustration with the engine's animations that write where it"
        " goes (Pose, .animate, Transform and the rest) or its rig's parameters;"
        " in a Succession, animate it in the part before one that moves it, or"
        " play that part on its own; leave its updating to the engine, and take"
        " it out of the scene with a play of its own; register scene updaters"
        " before declaring timing.",
        ("kind", "name"),
    ),
    Code(
        "media.inexact",
        "error",
        "MediaInexact",
        "a frame file of a still, clip, sprite or raster member is not in a form"
        " the engine decodes exactly, or a decoded frame does not have the digest"
        " its manifest lists; the form is checked as the asset loads, each frame's"
        " digest each time a process decodes it, and the finding's reason says"
        " which",
        "Store each frame as a PNG of 8 bits per channel, in colour or colour with"
        " alpha, and admit the asset again; a digest reason on an admitted asset"
        " means this host decodes it otherwise and must not draw it.",
        ("asset", "file", "reason", "expected", "found"),
    ),
    Code(
        "media.clock_past_end",
        "error",
        "MediaClockPastEnd",
        "a media object that is shown has a clock that reads before its clip's"
        " first frame or after its last, in a program that neither loops nor"
        " holds; raised at the instant the timeline evaluates it, and when a"
        " frame showing such an object is drawn",
        "End the object's stay before its clip ends, or say what follows: loop(),"
        ' or run(then="hold").',
        ("asset", "object", "t", "media_time", "duration"),
    ),
    Code(
        "sim.force_input",
        "error",
        "ForceInput",
        "a world's callback force can reach a mobject or the scene: between two"
        " frames a displayed mobject holds the last frame's values, not the"
        " tick's, so a force computed from one changes with the frame rate;"
        " raised as the force is declared",
        "Read each body, another world's too, by its handle from the world"
        " state the callback is handed; keep mobjects, worlds and the scene out"
        " of the callback's closure, defaults and globals.",
        ("force", "names"),
    ),
    Code(
        "sim.withheld",
        "error",
        "FeatureWithheld",
        "a scene declares a solver feature that is withheld, because its"
        " arithmetic is not yet the same on every machine; raised as the feature"
        " is declared",
        "Build the motion without the feature the finding names.",
        ("feature",),
    ),
    Code(
        "sim.unstable",
        "error",
        "SimulationUnstable",
        "a simulated body left what the simulation computes: its state went"
        " non-finite, or its speed or angular speed rose above its world's"
        " declared limit, which sits below the solver's own cap, so that motion"
        " the solver scaled down to its cap is always found (its value then a"
        " lower bound: it reached at least the cap); found at the tick it"
        " happened, in the record's bake",
        "Find what drives the body too hard (a force that grows without bound,"
        " two attracting bodies that meet, a joint or spring too stiff or too"
        " soft for the tick) and calm it; or, for motion that is meant to be"
        " that fast, raise the world's speed_limit or angular_speed_limit, to a"
        " value strictly below the solver's cap.",
        ("tick", "signal", "value", "limit", "capped"),
    ),
    Code(
        "sim.tunnelled",
        "error",
        "SimulationTunnelled",
        "two simulated bodies that should collide possibly passed through each"
        " other in one 60 Hz tick: touching at neither end of the step, with no"
        " contact in the solver's step, their shapes swept through the step's"
        " motion overlapped, or came within the allowed linear error of each"
        " other, which could not be told from a crossing (overlapped says"
        " which); found at the tick that ends the step, in the record's bake",
        "The tick is fixed, so let no body cross more than its own thickness in"
        " a step: thicken the thinner body, slow the faster one, or stiffen or"
        " soften the joint that flings it; and keep the world's"
        " continuous_collision on, which stops a fast body at a fixed one.",
        ("tick", "overlapped"),
    ),
)

#: Every code, by its string, in the order they were added.
CODES: Mapping[str, Code] = MappingProxyType({entry.code: entry for entry in _TABLE})


def code(name: str) -> Code:
    """The code ``name``; a :class:`KeyError` naming it for one the table
    does not hold, reserved ones included.
    """
    try:
        return CODES[name]
    except KeyError:
        reserved = name.startswith(RESERVED)
        why = "is reserved" if reserved else "is not a lint code"
        raise KeyError(f"{name!r} {why}") from None
