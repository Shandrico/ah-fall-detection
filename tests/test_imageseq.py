"""Image-sequence source tests."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ahfd.capture import open_source
from ahfd.capture.imageseq import ImageSequenceSource, _natural_key


def write_images(directory, names, size=(48, 64)):
    for name in names:
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        cv2.imwrite(str(directory / name), img)


class TestNaturalSort:
    def test_frame_2_sorts_before_frame_10(self):
        from pathlib import Path

        names = ["frame_10.png", "frame_2.png", "frame_1.png"]
        ordered = sorted((Path(n) for n in names), key=_natural_key)
        assert [p.name for p in ordered] == [
            "frame_1.png",
            "frame_2.png",
            "frame_10.png",
        ]


class TestImageSequenceSource:
    def test_reads_frames_in_order(self, tmp_path):
        write_images(tmp_path, ["f_1.png", "f_2.png", "f_3.png"])
        src = ImageSequenceSource(tmp_path, fps=15.0)
        frames = list(src)
        assert len(frames) == 3
        assert [f.index for f in frames] == [0, 1, 2]

    def test_synthesises_timestamps_from_fps(self, tmp_path):
        write_images(tmp_path, ["a.png", "b.png", "c.png"])
        frames = list(ImageSequenceSource(tmp_path, fps=10.0))
        assert frames[0].t == pytest.approx(0.0)
        assert frames[1].t == pytest.approx(0.1)
        assert frames[2].t == pytest.approx(0.2)

    def test_meta_reports_real_frame_size(self, tmp_path):
        write_images(tmp_path, ["a.png"], size=(48, 64))
        src = ImageSequenceSource(tmp_path)
        assert src.meta.width == 64
        assert src.meta.height == 48
        assert src.meta.frame_count == 1
        assert src.meta.has_depth is False

    def test_frames_carry_no_depth(self, tmp_path):
        write_images(tmp_path, ["a.png"])
        frame = next(iter(ImageSequenceSource(tmp_path)))
        assert frame.depth is None
        assert frame.bgr is not None

    def test_ignores_non_image_files(self, tmp_path):
        write_images(tmp_path, ["a.png", "b.png"])
        (tmp_path / "notes.txt").write_text("not an image")
        assert len(list(ImageSequenceSource(tmp_path))) == 2

    def test_empty_directory_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="no images"):
            ImageSequenceSource(tmp_path)

    def test_missing_directory_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="not a directory"):
            ImageSequenceSource(tmp_path / "nope")

    def test_factory_opens_seq_uri(self, tmp_path):
        write_images(tmp_path, ["a.png"])
        src = open_source("seq://" + str(tmp_path))
        assert isinstance(src, ImageSequenceSource)
