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

# A marginal USB link, or another program (RealSense Viewer, Teams, a browser)
# momentarily grabbing the camera, makes the first open or first frame fail
# intermittently even though the device is healthy a second later. Retry the
# open a few times -- escalating to a hardware reset -- rather than letting one
# bad moment crash the whole run. See _RealSenseBase._start_with_retry.
_FIRST_FRAME_TIMEOUT_MS = 8000
_START_ATTEMPTS = 4


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


def _device_has_imu(rs) -> bool:
    """True if a connected RealSense exposes accel+gyro.

    The D435i has an IMU; the D435f does NOT (the 'f' is an IR filter, not the
    'i' IMU). Requesting accel/gyro on a device without them makes
    ``pipeline.start`` fail with "Couldn't resolve requests", so callers enable
    the IMU streams only when this returns True.
    """
    try:
        for dev in rs.context().query_devices():
            for sensor in dev.query_sensors():
                for prof in sensor.get_stream_profiles():
                    if prof.stream_type() in (rs.stream.accel, rs.stream.gyro):
                        return True
    except Exception:  # pragma: no cover - hardware dependent
        pass
    return False


class _RealSenseBase:
    """Shared pipeline: filters, alignment, IMU gravity, frame assembly."""

    def __init__(
        self,
        color_size=(1920, 1080),
        depth_size=(848, 480),
        fps=30,
        with_depth=False,
        *,
        infrared=False,
        ir_index=1,
        ir_size=(1280, 720),
        emitter=None,
        max_laser=False,
        max_range_m=6.0,
        spatial_magnitude=2,
    ):
        # with_depth defaults OFF: nothing downstream consumes depth (the
        # geometry is homography-based), but capturing + filtering + aligning it
        # every frame is the reason the RealSense ran at <15 fps while the webcam
        # hit 40-60. RGB-only makes the RealSense as fast as any 1080p source.
        # The IMU stays on (it is cheap and calibration needs it).
        #
        # infrared streams the left IR imager instead of colour -- a grayscale
        # night-vision image for low light. `emitter` controls the dot
        # projector: leave it None to keep the device default, False to switch
        # it off. For IR-as-pose-input you want it OFF (the projected dots
        # otherwise cover the scene) plus an external IR floodlight so the room
        # is lit; the caller sets that policy. The IR imager is a *different*
        # camera from colour -- its own intrinsics and resolution -- so a stream
        # switched to IR needs its own calibration.
        self._rs = _import_rs()
        self._color_size = color_size
        self._depth_size = depth_size
        self._fps = fps
        self._with_depth = with_depth
        self._infrared = infrared
        self._ir_index = ir_index
        self._ir_size = ir_size
        self._emitter = emitter
        self._max_laser = max_laser
        self._max_range_m = max_range_m
        self._spatial_magnitude = spatial_magnitude
        self._pipeline = self._rs.pipeline()
        self._config = self._rs.config()
        self._align = None
        self._filters = self._build_filters()
        self._intrinsics: Intrinsics | None = None
        self._depth_scale = 0.001
        self._gravity: np.ndarray | None = None
        # Set on the first depth frame: None = unknown, True = the stored
        # depth->colour extrinsics are corrupt (rosbag2 .db3 playback mangles
        # them), so rs.align emits all-zero depth and we resample manually.
        self._manual_align: bool | None = None
        self._remap = None  # cached (v_idx, u_idx, valid) for the manual warp

    def _build_filters(self):
        rs = self._rs
        threshold = rs.threshold_filter()
        threshold.set_option(rs.option.min_distance, 0.3)
        # The far cut. 6 m was the default (3 m clipped far beds); raise it to see
        # further, at the cost of noisier depth -- the D435i can report ~10 m.
        threshold.set_option(rs.option.max_distance, float(self._max_range_m))

        to_disparity = rs.disparity_transform(True)
        spatial = rs.spatial_filter()
        # filter_magnitude = number of smoothing passes (1-5). More = less
        # spatial noise, at the cost of rounding off the person's edges.
        spatial.set_option(rs.option.filter_magnitude, float(max(1, min(5, self._spatial_magnitude))))
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
        device = profile.get_device()

        # The dot projector lives on the stereo (depth) sensor, and can be set
        # whether or not depth is streamed -- turning it off gives a clean IR
        # image. Best-effort: not every firmware exposes the option.
        if self._emitter is not None:
            try:
                sensor = device.first_depth_sensor()
                if sensor.supports(rs.option.emitter_enabled):
                    sensor.set_option(
                        rs.option.emitter_enabled, 1.0 if self._emitter else 0.0
                    )
            except Exception:  # pragma: no cover - best-effort hardware option
                pass

        if self._with_depth:
            self._align = rs.align(rs.stream.color)
            depth_sensor = device.first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())

            # Maxing the projector power puts more IR light on far surfaces, so
            # depth stays dense out to 4-6 m instead of dropping to sparse
            # speckle. Best-effort: not every firmware exposes laser_power, and
            # it only matters when the emitter is on (which it is, by default,
            # whenever depth is streamed). Near-range accuracy is unaffected.
            if self._max_laser:
                try:
                    if depth_sensor.supports(rs.option.laser_power):
                        rng = depth_sensor.get_option_range(rs.option.laser_power)
                        depth_sensor.set_option(rs.option.laser_power, rng.max)
                except Exception:  # pragma: no cover - best-effort hardware option
                    pass

    def _reset_device(self) -> None:
        """Hardware-reset the device and wait for it to re-enumerate.

        Last-ditch recovery when a plain retry keeps failing: a reset clears a
        wedged pipeline that a previous program (or an unclean exit) left behind.
        """
        import time

        rs = self._rs
        try:
            devices = rs.context().query_devices()
            if len(devices) > 0:
                devices[0].hardware_reset()
        except Exception:  # pragma: no cover - best-effort recovery
            pass
        time.sleep(4.0)  # re-enumeration takes a few seconds

    def _start_with_retry(self) -> None:
        """Start the pipeline and confirm frames actually flow, retrying through
        the intermittent failures a marginal link or a competing program cause.

        A start can 'succeed' while no frames ever arrive (the device is there
        but something else holds it, or the link stalled), so each attempt also
        waits for one real frame as a health check before handing control to the
        caller. Most bad moments clear within a retry or two; a hardware reset is
        the last resort before giving up with an actionable message.
        """
        import time

        rs = self._rs
        last: Exception | None = None
        for attempt in range(_START_ATTEMPTS):
            try:
                self._start()
                self._pipeline.wait_for_frames(_FIRST_FRAME_TIMEOUT_MS)
                return
            except Exception as exc:  # noqa: BLE001 - retry on any capture failure
                last = exc
                try:
                    self._pipeline.stop()
                except Exception:
                    pass
                if attempt >= _START_ATTEMPTS - 1:
                    break
                if attempt == _START_ATTEMPTS - 2:
                    self._reset_device()  # escalate before the final attempt
                else:
                    time.sleep(1.5)
                self._pipeline = rs.pipeline()  # a fresh pipeline for the retry
        raise RuntimeError(
            "RealSense did not deliver frames after "
            + str(_START_ATTEMPTS)
            + " attempts (last error: "
            + str(last)
            + "). This is a USB/ownership issue, not a config one: close any other "
            "program using the camera (RealSense Viewer, Teams, Zoom, a browser tab), "
            "use the cable that shipped with the D435i in a USB-3 port with no hub, "
            "and unplug/replug to reset."
        ) from last

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

    def _extrinsics_sane(self, frames) -> bool:
        """True if the stored depth->colour extrinsics are a valid transform.

        rosbag2 (.db3) recording can serialise garbage inter-stream extrinsics
        (rotation entries in the hundred-thousands, metre-scale+ translation),
        which makes rs.align project every depth pixel out of frame -> all-zero
        aligned depth. Detect that so we can resample manually instead.
        """
        try:
            dp = frames.get_depth_frame().get_profile()
            cp = frames.get_color_frame().get_profile()
            e = dp.get_extrinsics_to(cp)
        except Exception:  # pragma: no cover - hardware/format dependent
            return False
        rot = np.asarray(e.rotation, dtype=float)
        tr = np.asarray(e.translation, dtype=float)
        if not (np.all(np.isfinite(rot)) and np.all(np.isfinite(tr))):
            return False
        # A rotation matrix has entries in [-1, 1]; the D435i depth<->colour
        # translation is ~1.5 cm. Anything wildly bigger than that is corrupt.
        return bool(np.all(np.abs(rot) <= 1.5) and np.all(np.abs(tr) < 1.0))

    def _resample_native_to_color(self, native, depth_frame):
        """Warp a native-resolution depth array into the colour pixel grid.

        Used only when the stored extrinsics are corrupt (see
        ``_extrinsics_sane``). With the ~1.5 cm baseline ignored -- negligible
        parallax past ~2 m -- the depth->colour map reduces to a pure
        focal-length / principal-point resample, independent of range. A
        backward gather (nearest neighbour, no interpolation across depth
        edges) fills each colour pixel from its source depth pixel.
        """
        intr = self._intrinsics  # colour intrinsics (set from the colour frame)
        di = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        hc, wc = intr.height, intr.width
        if self._remap is None or self._remap[0] != (hc, wc):
            uc = np.arange(wc, dtype=np.float32)
            vc = np.arange(hc, dtype=np.float32)
            ud = np.round((uc - intr.cx) * (di.fx / intr.fx) + di.ppx).astype(np.int64)
            vd = np.round((vc - intr.cy) * (di.fy / intr.fy) + di.ppy).astype(np.int64)
            uok = (ud >= 0) & (ud < di.width)
            vok = (vd >= 0) & (vd < di.height)
            valid = np.outer(vok, uok)
            self._remap = ((hc, wc), np.clip(vd, 0, di.height - 1),
                           np.clip(ud, 0, di.width - 1), valid)
        _, vd, ud, valid = self._remap
        out = native[np.ix_(vd, ud)]
        out[~valid] = 0
        return out

    def _assemble_ir(self, frames, index: int, t: float) -> Frame | None:
        """Assemble a Frame from the left IR imager, as a 3-channel grey image.

        The IR frame is single-channel Y8; replicating it to three channels
        lets the pose stage -- which expects a BGR image -- consume it with no
        change. `np.repeat` returns a fresh array, so (like the colour path's
        .copy()) the Frame owns its pixels and cannot pin the frame pool.
        """
        ir_frame = frames.get_infrared_frame(self._ir_index)
        if not ir_frame:
            return None

        if self._intrinsics is None:
            intr = ir_frame.get_profile().as_video_stream_profile().get_intrinsics()
            self._intrinsics = Intrinsics(
                width=intr.width, height=intr.height,
                fx=intr.fx, fy=intr.fy, cx=intr.ppx, cy=intr.ppy,
            )

        ir = np.asanyarray(ir_frame.get_data())  # (H, W) uint8
        bgr = np.repeat(ir[:, :, None], 3, axis=2)  # grey -> 3-channel, owns pixels

        gravity = self._read_gravity(frames)
        if gravity is not None:
            self._gravity = gravity

        return Frame(
            index=index,
            t=t,
            bgr=bgr,
            depth=None,
            depth_raw=None,
            depth_scale=self._depth_scale,
            intrinsics=self._intrinsics,
            gravity=self._gravity,
        )

    def _assemble(self, frames, index: int, t: float) -> Frame | None:
        if self._infrared:
            return self._assemble_ir(frames, index, t)

        # RGB-only path (default): no align, no depth filtering -- the expensive
        # per-frame work that made the RealSense slow. Depth path kept for any
        # future depth feature, behind with_depth.
        native_depth = None  # set only when we must resample manually
        if self._with_depth:
            if self._manual_align is None:
                self._manual_align = not self._extrinsics_sane(frames)
            if self._manual_align:
                # Corrupt stored extrinsics: skip rs.align (it would zero the
                # depth) and take the native depth to resample ourselves.
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                native_depth = depth_frame
            else:
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

        # .copy() is load-bearing: get_data() returns a view into the frame's
        # buffer from librealsense's fixed pool (~16 frames). Any consumer that
        # HOLDS frames -- `ahfd bench` accumulates them -- would pin the whole
        # pool and stall the stream at frame 16, and a recycled buffer would also
        # corrupt a held image. Copying makes each Frame own its pixels.
        bgr = np.asanyarray(color_frame.get_data()).copy()
        measure = display = None
        if depth_frame is not None:
            measure, display = self._measure_and_filtered(depth_frame)
            if native_depth is not None:
                # depth_frame was native; warp the filtered depth into the
                # colour grid so joint pixels index it the same as when aligned.
                measure = self._resample_native_to_color(measure, native_depth)
                display = self._resample_native_to_color(display, native_depth)

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

    def __init__(
        self,
        color_size=(1920, 1080),
        depth_size=(848, 480),
        fps=30,
        with_depth=False,
        *,
        infrared=False,
        ir_index=1,
        ir_size=(1280, 720),
        emitter=None,
        max_laser=False,
        max_range_m=6.0,
        spatial_magnitude=2,
    ):
        super().__init__(
            color_size, depth_size, fps, with_depth=with_depth,
            infrared=infrared, ir_index=ir_index, ir_size=ir_size, emitter=emitter,
            max_laser=max_laser, max_range_m=max_range_m, spatial_magnitude=spatial_magnitude,
        )
        rs = self._rs
        if infrared:
            # Left IR imager, single-channel Y8. Colour is not enabled -- one
            # stream is all pose needs, and it keeps the frame rate up.
            self._config.enable_stream(
                rs.stream.infrared, ir_index, ir_size[0], ir_size[1], rs.format.y8, fps
            )
        else:
            self._config.enable_stream(
                rs.stream.color, color_size[0], color_size[1], rs.format.bgr8, fps
            )
        if with_depth:
            self._config.enable_stream(
                rs.stream.depth, depth_size[0], depth_size[1], rs.format.z16, fps
            )
        # IMU streams for the gravity vector -- only when the device has an IMU.
        # The D435i exposes accel+gyro; the D435f does NOT, and requesting absent
        # streams makes pipeline.start fail ("Couldn't resolve requests"). Enable
        # them conditionally so both cameras work. Without an IMU there is no live
        # gravity, so the height/ground modes fall back to a calibrated tilt.
        self._has_imu = _device_has_imu(rs)
        if self._has_imu:
            self._config.enable_stream(rs.stream.accel)
            self._config.enable_stream(rs.stream.gyro)

    @property
    def meta(self) -> SourceMeta:
        width, height = self._ir_size if self._infrared else self._color_size
        return SourceMeta(
            uri="rs://ir" if self._infrared else "rs://",
            width=width,
            height=height,
            fps=float(self._fps),
            has_depth=self._with_depth,
        )

    def __iter__(self) -> Iterator[Frame]:
        import time

        self._start_with_retry()
        index = 0
        start = time.monotonic()
        try:
            while True:
                try:
                    frames = self._pipeline.wait_for_frames()
                except RuntimeError:
                    # A transient mid-stream stall on a marginal link. Give it one
                    # more, longer, chance before propagating -- a single missed
                    # frameset should not end a capture.
                    frames = self._pipeline.wait_for_frames(_FIRST_FRAME_TIMEOUT_MS)
                frame = self._assemble(frames, index, time.monotonic() - start)
                if frame is not None:
                    yield frame
                    index += 1
        finally:
            self.close()


class BagSource(_RealSenseBase):
    """Recorded .bag playback, deterministic and frame-exact."""

    def __init__(
        self,
        path: str,
        with_depth: bool = False,
        max_range_m: float = 6.0,
        spatial_magnitude: int = 2,
    ):
        # with_depth defaults off so plain replay stays fast; extraction for the
        # depth features passes True to also emit the aligned depth per frame.
        # The filter chain re-runs on playback (the .bag stores RAW depth), so
        # max_range_m / spatial_magnitude let you tune denoising after the fact.
        super().__init__(
            with_depth=with_depth,
            max_range_m=max_range_m,
            spatial_magnitude=spatial_magnitude,
        )
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
