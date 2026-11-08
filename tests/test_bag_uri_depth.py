"""bag:// URIs take ?depth=1 so a dashboard replay can run the depth pipeline."""

import ahfd.capture.realsense as rs_mod
from ahfd.capture.factory import open_source


def test_bag_uri_depth_query(monkeypatch):
    seen = []

    class FakeBag:
        def __init__(self, path, with_depth=False, max_range_m=6.0, spatial_magnitude=2):
            seen.append((path, with_depth, max_range_m, spatial_magnitude))

    monkeypatch.setattr(rs_mod, "BagSource", FakeBag)
    open_source("bag:///ssd/f1_01.db3")
    open_source("bag:///ssd/f1_01.db3?depth=1&max_range=4&spatial=1")
    assert seen == [("/ssd/f1_01.db3", False, 6.0, 2), ("/ssd/f1_01.db3", True, 4.0, 1)]
