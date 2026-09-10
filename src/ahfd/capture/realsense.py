"""Intel RealSense sources: live D435i and recorded .bag playback.

Both share one code path on purpose. A .bag replays deterministically with
`set_real_time(False)`, so a fall recorded once can be re-run frame-for-frame
as many times as tuning needs, and the live and offline paths cannot drift
apart. That determinism is what makes the golden-clip regression tests possible.

Depth handling follows the review of the original `depth_filter.py`:

* Filters run in **disparity space** (depth2disparity -> spatial/temporal ->
  disparity2depth), which is what librealsense recommends and is measurably
  better than filtering raw depth.
* The threshold is raised to 6 m. At the original 3 m the far side of an 8.4 m
  cubicle is clipped entirely.
* Two depth planes are emitted. `depth` is the full chain including hole
  filling, for visualisation and the floor point cloud. `depth_raw` is
  everything *except* hole filling, for any measurement -- because hole filling
  invents confident, wrong depths exactly at object boundaries, which is
  precisely where limb keypoints sit.

`pyrealsense2` is an optional dependency and has no aarch64 wheel, so it is
imported lazily and only here. Everything else in the project runs without it.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from ahfd.capture.base import Frame, SourceMeta
from ahfd.types import Intrinsics


def _import_rs():
    try:
        import pyrealsense2 as rs
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "pyrealsense2 is not installed. Install the realsense extra "
            "(uv pip install -e '.[realsense]'), or on the Jetson build "
            "librealsense from source -- there is no aarch64 wheel."
        ) from exc
    return rs


class _RealSenseBase:
    """Shared pipeline: filters, alignment, IMU gravity, frame assembly."""

    def __init__(self, color_size=(1920, 1080), depth_size=(848, 480), fps=30, with_depth=False):
        # with_depth defaults OFF: nothing downstream consumes depth (the
        # geometry is homography-based), but capturing + filtering + aligning it
        # every frame is the reason the RealSense ran at <15 fps while the webcam
        # hit 40-60. RGB-only makes the RealSense as fast as any 1080p source.
        # The IMU stays on (it is cheap and calibration needs it).
        self._rs = _import_rs()
        self._color_size = color_size
        self._depth_size = depth_size
        self._fps = fps
        self._with_depth = with_depth
        self._pipeline = self._rs.pipeline()
        self._config = self._rs.config()
        self._align = None
        self._filters = self._build_filters()
        self._intrinsics: Intrinsics | None = None
        self._depth_scale = 0.001
        self._gravity: np.ndarray | None = None

    def _build_filters(self):
        rs = self._rs
        threshold = rs.threshold_filter()
        threshold.set_option(rs.option.min_distance, 0.3)
        threshold.set_option(rs.option.max_distance, 6.0)  # was 3.0: clipped far beds

        to_disparity = rs.disparity_transform(True)
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.filter_magnitude, 2)
        spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial.set_option(rs.option.filter_smooth_delta, 20)
        temporal = rs.temporal_filter()
        temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
        temporal.set_option(rs.option.filter_smooth_delta, 20)
        to_depth = rs.disparity_transform(False)
        hole = rs.hole_filling_filter()

        return {
            "threshold": threshold,
            "to_disparity": to_disparity,
            "spatial": spatial,
            "temporal": temporal,
            "to_depth": to_depth,
            "hole": hole,
        }

    def _start(self):
        rs = self._rs
        profile = self._pipeline.start(self._config)
        if self._with_depth:
            self._align = rs.align(rs.stream.color)
            depth_sensor = profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())

    def _read_gravity(self, frames) -> np.ndarray | None:
        """Gravity vector in the camera frame, from the accelerometer.

        This is what lets tilt be recovered continuously instead of measured by
        hand -- the reason to prefer a D435i over a plain webcam even where
        depth is unusable at range.
        """
        rs = self._rs
        for frame in frames:
            if frame.is_motion_frame() and frame.get_profile().stream_type() == rs.stream.accel:
                data = frame.as_motion_frame().get_motion_data()
                return np.array([data.x, data.y, data.z], dtype=float)
        return None

    def _measure_and_filtered(self, depth_frame):
        """Return (measure_depth, display_depth) as uint16 arrays.

        measure_depth: threshold + spatial + temporal, no hole filling.
        display_depth: the full chain including hole filling.
        """
        f = self._filters
        common = f["threshold"].process(depth_frame)
        common = f["to_disparity"].process(common)
        common = f["spatial"].process(common)
        common = f["temporal"].process(common)
        common = f["to_depth"].process(common)

        measure = np.asanyarray(common.get_data()).copy()
        display = np.asanyarray(f["hole"].process(common).get_data()).copy()
        return measure, display

    def _assemble(self, frames, index: int, t: float) -> Frame | None:
        # RGB-only path (default): no align, no depth filtering -- the expensive
        # per-frame work that made the RealSense slow. Depth path kept for any
        # future depth feature, behind with_depth.
        if self._with_depth:
            aligned = self._align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                return None
        else:
            color_frame = frames.get_color_frame()
            depth_frame = None
            if not color_frame:
                return None

        if self._intrinsics is None:
            intr = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
            self._intrinsics = Intrinsics(
                width=intr.width,
                height=intr.height,
                fx=intr.fx,
                fy=intr.fy,
                cx=intr.ppx,
                cy=intr.ppy,
            )

        bgr = np.asanyarray(color_frame.get_data())
        measure = display = None
        if depth_frame is not None:
            measure, display = self._measure_and_filtered(depth_frame)

        gravity = self._read_gravity(frames)
        if gravity is not None:
            self._gravity = gravity

        return Frame(
            index=index,
            t=t,
            bgr=bgr,
            depth=display,
            depth_raw=measure,
            depth_scale=self._depth_scale,
            intrinsics=self._intrinsics,
            gravity=self._gravity,
        )

    def close(self) -> None:
        try:
            self._pipeline.stop()
        except Exception:  # pragma: no cover - already stopped
            pass


class RealSenseSource(_RealSenseBase):
    """Live D435i."""

    def __init__(self, color_size=(1920, 1080), depth_size=(848, 480), fps=30, with_depth=False):
        super().__init__(color_size, depth_size, fps, with_depth=with_depth)
        rs = self._rs
        self._config.enable_stream(
            rs.stream.color, color_size[0], color_size[1], rs.format.bgr8, fps
        )
        if with_depth:
            self._config.enable_stream(
                rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, fps
            )
        # IMU streams for the gravity vector (cheap; calibration needs accel).
        self._config.enable_stream(rs.stream.accel)
        self._config.enable_stream(rs.stream.gyro)

    @property
    def meta(self) -> SourceMeta:
        return SourceMeta(
            uri="rs://",
            width=self._color_size[0],
            height=self._color_size[1],
            fps=float(self._fps),
            has_depth=self._with_depth,
        )

    def __iter__(self) -> Iterator[Frame]:
        import time

        self._start()
        index = 0
        start = time.monotonic()
        try:
            while True:
                frames = self._pipeline.wait_for_frames()
                frame = self._assemble(frames, index, time.monotonic() - start)
                if frame is not None:
                    yield frame
                    index += 1
        finally:
            self.close()


class BagSource(_RealSenseBase):
    """Recorded .bag playback, deterministic and frame-exact."""

    def __init__(self, path: str):
        super().__init__()
        rs = self._rs
        self._path = path
        self._config.enable_device_from_file(path, repeat_playback=False)

    @property
    def meta(self) -> SourceMeta:
        return SourceMeta(
            uri="bag://" + self._path,
            width=self._color_size[0],
            height=self._color_size[1],
            fps=float(self._fps),
            has_depth=True,
        )

    def __iter__(self) -> Iterator[Frame]:
        rs = self._rs
        profile = self._pipeline.start(self._config)
        self._align = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        # Deterministic: do not drop frames to keep up with wall-clock.
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)

        index = 0
        try:
            while True:
                ok, frames = self._pipeline.try_wait_for_frames()
                if not ok:
                    break
                t = frames.get_timestamp() / 1000.0
                frame = self._assemble(frames, index, t)
                if frame is not None:
                    yield frame
                    index += 1
        finally:
            self.close()
