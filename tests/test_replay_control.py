"""Dashboard replay/player control plane (scrub / play / pause / step)."""

from __future__ import annotations

from ahfd.dashboard.state import DashboardState


class TestReplayControl:
    def test_off_for_a_live_camera(self):
        s = DashboardState()
        assert s.snapshot().get("replay") is None
        # commands are refused until a seekable source announces itself
        assert s.replay_control("play") is False
        assert s.take_replay_command() == (False, None)

    def test_seekable_file_enables_the_player(self):
        s = DashboardState()
        s.announce_replay(300, gen=0)
        assert s.snapshot()["replay"] == {"total": 300, "cur": 0, "paused": False}

    def test_pause_play(self):
        s = DashboardState()
        s.announce_replay(300, gen=0)
        assert s.replay_control("pause") is True
        assert s.take_replay_command() == (True, None)
        assert s.replay_control("play") is True
        assert s.take_replay_command() == (False, None)

    def test_seek_is_a_one_shot_target(self):
        s = DashboardState()
        s.announce_replay(300, gen=0)
        assert s.replay_control("seek", 120) is True
        assert s.take_replay_command()[1] == 120     # target delivered once
        assert s.take_replay_command()[1] is None    # and cleared

    def test_step_moves_from_current_and_pauses(self):
        s = DashboardState()
        s.announce_replay(300, gen=0)
        s.publish_replay_pos(40, gen=0)
        assert s.replay_control("step", 5) is True
        paused, seek = s.take_replay_command()
        assert paused is True and seek == 45

    def test_position_updates_show_in_snapshot(self):
        s = DashboardState()
        s.announce_replay(300, gen=0)
        s.publish_replay_pos(150, gen=0)
        assert s.snapshot()["replay"]["cur"] == 150

    def test_a_switch_clears_the_player(self):
        s = DashboardState()
        s.announce_replay(300, gen=0)
        s.begin_generation(source="webcam://0")  # switch to a live camera
        assert s.snapshot().get("replay") is None
        # and a stale command published against the old gen is ignored
        s.publish_replay_pos(10, gen=0)
        assert s.snapshot().get("replay") is None

    def test_stale_generation_is_fenced(self):
        s = DashboardState()
        gen = s.begin_generation(source="file://clip.mp4")
        s.announce_replay(300, gen=gen)
        assert s.snapshot()["replay"]["total"] == 300
        # a retired pipeline cannot move the slider
        s.publish_replay_pos(99, gen=gen - 1)
        assert s.snapshot()["replay"]["cur"] == 0
