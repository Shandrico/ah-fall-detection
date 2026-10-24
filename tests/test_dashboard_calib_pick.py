"""A recording typed in the source box gets its camera's calibration."""

from ahfd.dashboard.controller import DashboardController


def _controller(default="calib/d435i.yaml"):
    c = object.__new__(DashboardController)
    c._calib_by_uri = {
        "rs://?depth=1": "calib/d435i.yaml",
        "bag:///ssd/ah_bags_d435f/clip1_09.db3": "calib/d435f.yaml",
    }
    c._default_calib = default
    return c


def test_listed_source_keeps_its_own_calibration():
    assert _controller()._calib_for("bag:///ssd/ah_bags_d435f/clip1_09.db3") == "calib/d435f.yaml"


def test_typed_recording_matches_the_camera_named_in_its_path():
    c = _controller()
    assert c._calib_for("bag:///ssd/ah_bags_d435f/clip2_test.db3") == "calib/d435f.yaml"
    assert c._calib_for("bag:///ssd/ah_bags_d435i/clip2_test.db3") == "calib/d435i.yaml"


def test_unknown_recording_falls_back_and_live_stays_pose_only():
    c = _controller()
    assert c._calib_for("file:///videos/ward.mp4") == "calib/d435i.yaml"
    assert c._calib_for("rs://ir") is None
    assert _controller(default=None)._calib_for("file:///videos/ward.mp4") is None
