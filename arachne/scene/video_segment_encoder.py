"""Encoding for one silent cached video segment."""

from __future__ import annotations

from contextlib import suppress
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from arachne._config.video_encoder import VideoEncoderSpec
from arachne.typing import RGBAPixelArray

__all__ = ["VideoSegmentEncoder"]


class VideoSegmentEncoder:
    """Encode top-left-origin C-contiguous RGBA frames into one video segment.

    Frames are placed on the frame grid: a frame's presentation time counts
    the frames of the grid before it, on a time base of one frame. With the
    spec's ``gaps``, a run of equal frames (repeats, or consecutive frames
    with equal bytes) is encoded once, as one frame whose duration covers the
    run; the next frame's time leaves a gap. Every frame carries its duration,
    so the last one's keeps a trailing hold in the segment's duration.
    """

    def __init__(self, *, target: Path, spec: VideoEncoderSpec) -> None:
        self.target = target
        self.spec = spec
        self._next_pts = 0
        self._closed = False
        #: With gaps: the run being written, and how many frames it lasts.
        self._pending: RGBAPixelArray | None = None
        self._pending_frames = 0
        self.target.parent.mkdir(parents=True, exist_ok=True)

        container = None
        try:
            container = av.open(
                self.target,
                mode="w",
                format=self.spec.container_format,
            )
            stream = container.add_stream(
                self.spec.codec,
                rate=self.spec.frame_rate,
                options=dict(self.spec.options),
            )
            stream.pix_fmt = self.spec.pixel_format
            stream.width = self.spec.width
            stream.height = self.spec.height
            if self.spec.gaps:
                # The encoder copies each frame's duration to its packet.
                stream.codec_context.flags |= av.codec.context.Flags.frame_duration
        except BaseException as error:
            if container is not None:
                with suppress(BaseException):
                    container.close()
            with suppress(OSError):
                self.target.unlink(missing_ok=True)
            raise self._operation_error("open", error) from error

        self._container = container
        self._stream = stream

    def _profile_description(self) -> str:
        return (
            f"{self.spec.container_format}/{self.spec.codec}/"
            f"{self.spec.pixel_format}, {self.spec.width}x{self.spec.height} "
            f"at {self.spec.frame_rate} fps"
        )

    def _operation_error(self, operation: str, error: BaseException) -> RuntimeError:
        return RuntimeError(
            f"Failed to {operation} video segment {self.target} "
            f"({self._profile_description()}): {error}",
        )

    def _validate_frame(self, pixels: RGBAPixelArray, repeat: int) -> None:
        if self._closed:
            raise RuntimeError(f"Video segment encoder for {self.target} is closed.")
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
            raise ValueError("Frame repeat must be a positive integer.")
        if not isinstance(pixels, np.ndarray):
            raise TypeError("Video segment frames must be NumPy arrays.")
        expected_shape = (self.spec.height, self.spec.width, 4)
        if pixels.shape != expected_shape:
            raise ValueError(
                f"Video segment frames must have shape {expected_shape}; "
                f"got {pixels.shape}.",
            )
        if pixels.dtype != np.uint8:
            raise TypeError(
                f"Video segment frames must use uint8; got {pixels.dtype}.",
            )
        if not pixels.flags.c_contiguous:
            raise ValueError("Video segment frames must be C-contiguous.")

    @property
    def _time_base(self) -> Fraction:
        return Fraction(
            self.spec.frame_rate.denominator,
            self.spec.frame_rate.numerator,
        )

    def write_frame(self, pixels: RGBAPixelArray, *, repeat: int = 1) -> None:
        """Encode ``pixels`` for the next ``repeat`` frames of the grid."""
        self._validate_frame(pixels, repeat)
        if self.spec.gaps:
            self._extend_run(pixels, repeat)
            return
        try:
            for _ in range(repeat):
                frame = av.VideoFrame.from_ndarray(pixels, format="rgba")
                frame.pts = self._next_pts
                frame.time_base = self._time_base
                self._next_pts += 1
                for packet in self._stream.encode(frame):
                    self._container.mux(packet)
        except BaseException as error:
            raise self._operation_error("encode", error) from error

    def _extend_run(self, pixels: RGBAPixelArray, repeat: int) -> None:
        """Add ``repeat`` frames of ``pixels`` to the run being written, or
        encode that run and start another: a run's duration is known only
        once a different frame, or the end, arrives.
        """
        pending = self._pending
        if pending is not None and (
            pixels is pending or np.array_equal(pixels, pending)
        ):
            self._pending_frames += repeat
            return
        try:
            self._encode_run()
        except BaseException as error:
            raise self._operation_error("encode", error) from error
        self._pending, self._pending_frames = pixels, repeat

    def _encode_run(self) -> None:
        """Encode the run being written, if any, as one frame lasting it."""
        if self._pending is None:
            return
        frame = av.VideoFrame.from_ndarray(self._pending, format="rgba")
        frame.pts = self._next_pts
        frame.time_base = self._time_base
        frame.duration = self._pending_frames
        self._next_pts += self._pending_frames
        self._pending, self._pending_frames = None, 0
        for packet in self._stream.encode(frame):
            self._container.mux(packet)

    def finish(self) -> None:
        """Flush encoded packets and close the segment."""
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        try:
            try:
                self._encode_run()
                for packet in self._stream.encode():
                    self._container.mux(packet)
            except Exception as error:
                first_error = error
        finally:
            try:
                self._container.close()
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise self._operation_error("finish", first_error) from first_error

    def abort(self) -> None:
        """Close resources and remove the incomplete target."""
        first_error: Exception | None = None
        try:
            if not self._closed:
                self._closed = True
                try:
                    self._container.close()
                except Exception as error:
                    first_error = error
        finally:
            try:
                self.target.unlink(missing_ok=True)
            except Exception as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise self._operation_error("abort", first_error) from first_error
