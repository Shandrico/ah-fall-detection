"""The dashboard's review page: only listed .mp4 files, and only when a folder is given."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from ahfd.dashboard.review import find_video, list_videos
from ahfd.dashboard.server import DashboardServer
from ahfd.dashboard.state import DashboardState

DATA = bytes(range(256)) * 4


@pytest.fixture
def root(tmp_path):
    (tmp_path / "fall").mkdir()
    (tmp_path / "bed_exit").mkdir()
    (tmp_path / "fall" / "f1_01_d435f.mp4").write_bytes(DATA)
    (tmp_path / "fall" / "f1_02_d435f.mp4.part.mp4").write_bytes(DATA)
    (tmp_path / "fall" / "notes.txt").write_text("secret")
    (tmp_path / "bed_exit" / "s1_01_d435i.mp4").write_bytes(DATA)
    (tmp_path / "outside.mp4").write_bytes(DATA)
    return tmp_path


def serve(review_dir):
    srv = DashboardServer(DashboardState(), host="127.0.0.1", port=0, review_dir=review_dir)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:" + str(srv.server_address[1])


def fetch(url, **headers):
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, b"", dict(e.headers)


def test_lists_finished_videos_in_their_folders_only(root):
    assert [(v["group"], v["name"]) for v in list_videos(root)] == [
        ("bed_exit", "s1_01_d435i.mp4"),
        ("fall", "f1_01_d435f.mp4"),
    ]


@pytest.mark.parametrize("root_value", [None, "", "/no/such/folder"])
def test_an_absent_folder_is_an_empty_list(root_value):
    assert list_videos(root_value) == []
    assert find_video(root_value, "fall", "x.mp4") is None


def test_serves_a_listed_video_with_seeking(root):
    srv, url = serve(root)
    try:
        code, body, _ = fetch(url + "/review/video/fall/f1_01_d435f.mp4")
        assert code == 200 and body == DATA
        code, body, headers = fetch(url + "/review/video/fall/f1_01_d435f.mp4", Range="bytes=10-19")
        assert code == 206 and body == DATA[10:20]
        assert headers["Content-Range"] == "bytes 10-19/" + str(len(DATA))
        listed = json.loads(fetch(url + "/api/review")[1])
        assert len(listed["videos"]) == 2
    finally:
        srv.shutdown()


@pytest.mark.parametrize(
    "path",
    [
        "/review/video/fall/notes.txt",
        "/review/video/fall/f1_02_d435f.mp4.part.mp4",
        "/review/video/fall/..%2Foutside.mp4",
        "/review/video/..%2Ffall/f1_01_d435f.mp4",
        "/review/video/../outside.mp4",
        "/review/video/fall/",
    ],
)
def test_anything_unlisted_is_not_found(root, path):
    srv, url = serve(root)
    try:
        assert fetch(url + path)[0] == 404
    finally:
        srv.shutdown()


def test_no_folder_means_no_page_and_no_videos(root):
    srv, url = serve(None)
    try:
        for path in ("/review", "/api/review", "/review/video/fall/f1_01_d435f.mp4"):
            assert fetch(url + path)[0] == 404
    finally:
        srv.shutdown()
