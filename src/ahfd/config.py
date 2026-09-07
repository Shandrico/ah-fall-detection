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
    model_size: str = "s"
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


class PrivacyConfig(BaseModel):
    allow_raw_capture: bool = False


class Config(BaseModel):
    source: str = "webcam://0"
    pose: PoseConfig = Field(default_factory=PoseConfig)
    smoothing: SmoothingConfig = Field(default_factory=SmoothingConfig)
    view: ViewConfig = Field(default_factory=ViewConfig)
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
