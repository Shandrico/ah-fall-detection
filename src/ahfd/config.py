"""Configuration loading.

Every tunable lives in YAML, never in source. A ward that needs different
thresholds is a config change, not a code change -- which matters because bed
heights, camera placement and lighting differ per cubicle.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class PoseConfig(BaseModel):
    backend: str = "rtmo"
    model_size: str = "s"  # RTMO: s/m/l
    mode: str = "performance"  # RTMPose top-down: performance/balanced/lightweight
    model_input_size: tuple[int, int] = (640, 640)
    device: str = "cpu"
    runtime: str = "onnxruntime"
    min_score: float = 0.3
    min_keypoint_score: float = 0.3


class SmoothingConfig(BaseModel):
    enabled: bool = True
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0


class ViewConfig(BaseModel):
    mode: str = "skeleton"
    show_bbox: bool = False
    show_ids: bool = True
    show_fps: bool = True
    show_metrics: bool = True  # per-person metric readout + calibration check


class PrivacyConfig(BaseModel):
    allow_raw_capture: bool = False


class DetectConfig(BaseModel):
    """Fall thresholds, in metres, seconds and metres per second.

    These live here rather than in the per-camera calibration on purpose:
    because they are metric, they describe a *ward*, not a camera, so one set
    covers every camera in the room. Keeping them out of the calibration file
    is what stops them being quietly re-tuned per camera.
    """

    enabled: bool = False

    upright_h: float = 0.70
    sitting_h: tuple[float, float] = (0.40, 0.70)
    bed_band: tuple[float, float] = (-0.25, 0.45)

    down_spread: tuple[float, float] = (0.9, 3.0)
    down_h_torso: float = 0.90

    vz_trigger: float = -0.90
    vz_frames: int = 3
    drop_trigger: float = 0.45
    drop_window_s: float = 0.8
    min_track_age_s: float = 1.0
    rest_deadline_s: float = 2.5

    suspect_s: float = 1.5
    confirm_s: float = 8.0
    confirm_motion_max: float = 0.15
    recover_h: float = 0.70

    slow_down_s: float = 20.0
    bed_exit_s: float = 3.0

    min_valid_kp: int = 8
    min_mean_conf: float = 0.40
    cooldown_s: float = 60.0

    def to_thresholds(self):
        """Build the detector's threshold object, dropping `enabled`."""
        from ahfd.detect import FallThresholds

        values = self.model_dump()
        values.pop("enabled", None)
        return FallThresholds(**values)


class AlertConfig(BaseModel):
    console: bool = True
    jsonl_path: str | None = None
    min_severity: int = 0


class CaptureConfig(BaseModel):
    # Requested capture resolution. None keeps the camera/source default.
    # Higher resolution gives a sharper dashboard and better distant-person
    # keypoints -- but if detection is on, a calibration for this exact
    # resolution is required (intrinsics are per-resolution).
    width: int | None = None
    height: int | None = None


class SourceOption(BaseModel):
    """One labelled camera in the dashboard's picker."""

    label: str
    uri: str


class DashboardConfig(BaseModel):
    host: str = "127.0.0.1"  # localhost only by default -- not exposed to the network
    port: int = 8000
    # RGB reverses the skeleton-only ward stance, so it is opt-in. Default is
    # the privacy-safe skeleton view.
    show_rgb: bool = False
    jpeg_quality: int = 90  # 0-100; higher is sharper and larger per frame

    # Cameras offered in the page's picker. A connected RealSense is appended
    # automatically; nothing else is probed, because scanning webcam indices is
    # slow and can grab a device another program is using.
    sources: list[SourceOption] = Field(default_factory=list)
    # Restrict the model picker. Empty means every backend the code supports.
    # A ward will want to hide `yolo`, which is AGPL and benchmark-only.
    backends: list[str] = Field(default_factory=list)
    # The free-text URI box. False on a ward: only the cameras listed above.
    allow_custom_source: bool = True
    # How long to wait for a retiring pipeline to let go of its camera.
    switch_timeout_s: float = 5.0


class Config(BaseModel):
    source: str = "webcam://0"
    calibration: str | None = None  # path to a per-camera calib YAML
    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    pose: PoseConfig = Field(default_factory=PoseConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    detect: DetectConfig = Field(default_factory=DetectConfig)
    alert: AlertConfig = Field(default_factory=AlertConfig)
    view: ViewConfig = Field(default_factory=ViewConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML, falling back to built-in defaults."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)
    if not path.exists():
        return Config()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)
