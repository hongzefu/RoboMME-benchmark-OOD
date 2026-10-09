"""C10 video composition: real calls to RobommeRecordWrapper's ``_video_*`` methods and the file naming of close.

- Daily gate: synthetic frames ≥64×64; checks the h5 raw pixels are not modified by the overlay, NO RECORD filtering, no reset frame added,
  frame size normalization, success/FAILED/NO_OBJECT naming, encoding failure does not affect h5; mp4 writing is replaced with a bookkeeping stand-in.
- slow: real libx264 encoding and read back of the frame count (needs ffmpeg; recorded as "Unverified" if missing).
"""
from __future__ import annotations

import copy

import h5py
import numpy as np
import pytest

from recording_fakes import (
    IMG,
    SEG_COLS,
    SEG_ID,
    SEG_ROWS,
    Event,
    FakeTaskEnv,
    front_rgb,
    record_module,
    run_episode,
    segmentation,
    wrist_rgb,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*to get variables from other wrappers")

KINDS = ["official", "hard"]
RED = [255, 0, 0]


@pytest.fixture(params=KINDS)
def mod(request):
    return record_module(request.param)


@pytest.fixture
def wrapper(mod, tmp_path):
    w = mod.RobommeRecordWrapper(FakeTaskEnv([]), dataset=str(tmp_path / "out"), env_id="FakeTask", episode=1, seed=2, save_video=True)
    yield w
    try:
        w.h5_file.close()
    except Exception:  # noqa: BLE001
        pass


@pytest.fixture
def writes(mod, monkeypatch):
    """Bookkeeping stand-in: records the file name and frames of every mp4 write (does not call ffmpeg)."""
    log = []

    def fake_write(self, frames, output_path):
        log.append((output_path.name, [f.copy() for f in frames]))

    monkeypatch.setattr(mod.RobommeRecordWrapper, "_video_write_mp4", fake_write)
    return log


def test_prepare_step_frames_leaves_inputs_untouched(wrapper):
    base = front_rgb(5)
    wrist = wrist_rgb(5)[:32, :32].copy()  # different size → takes the resize branch
    seg = segmentation(True)[..., 0]
    seg_result = seg.copy()
    snap = [copy.deepcopy(x) for x in (base, wrist, seg, seg_result)]
    wrapper.segmentation_points = [[14, 24]]
    out = wrapper._video_prepare_step_frames(base, wrist, seg, seg_result, np.zeros_like(seg))
    for before, after in zip(snap, (base, wrist, seg, seg_result)):
        assert before.tobytes() == after.tobytes()  # raw pixels written into h5 not modified by the overlay
    comb = out["combined"]
    assert comb.shape == (IMG, 5 * IMG, 3) and comb.dtype == np.uint8
    np.testing.assert_array_equal(comb[:, :IMG], base)
    # tile 3: raw segmentation colorized: inside the square the color of that id, outside black
    color = wrapper.color_map[SEG_ID]
    assert comb[SEG_ROWS[0], 2 * IMG + SEG_COLS[0]].tolist() == color
    assert comb[0, 2 * IMG].tolist() == [0, 0, 0]
    # tile 5: base image plus red dot, red dot at the cached center (14, 24)
    assert comb[14, 4 * IMG + 24].tolist() == RED
    assert base[14, 24].tolist() != RED
    # online row: segmentation all 0 → tile 4 all black
    assert not out["combined_online"][:, 3 * IMG:4 * IMG].any()


def test_apply_overlays_border_and_text(wrapper):
    """Asserts properties only: the text area is stacked above the original frame, more lines make it taller, there is a minimum height; the original pixel area is unchanged. Exact line height is not pinned in this block."""
    frame = front_rgb(3)
    snap = frame.copy()
    out3 = wrapper._video_apply_overlays(frame, True, ["a b", None, "c", "d"])  # None filtered out → 3 lines
    assert frame.tobytes() == snap.tobytes()
    h3 = out3.shape[0] - IMG
    body = out3[h3:]
    assert body[0, 0].tolist() == RED and body[-1, -1].tolist() == RED  # demonstration frames get a red border
    np.testing.assert_array_equal(body[IMG // 4:-IMG // 4, IMG // 4:-IMG // 4], frame[IMG // 4:-IMG // 4, IMG // 4:-IMG // 4])
    plain = wrapper._video_apply_overlays(frame, False, [])
    np.testing.assert_array_equal(plain, frame)  # non-demonstration, no goal → unchanged
    one = wrapper._video_apply_overlays(frame, False, "x")
    two = wrapper._video_apply_overlays(frame, False, ["x", "y"])
    h1, h2 = one.shape[0] - IMG, two.shape[0] - IMG
    np.testing.assert_array_equal(one[h1:], frame)  # non-demonstration: original pixel area unchanged pixel by pixel
    assert 0 < h1 < h3  # 3-line text area taller than 1-line
    assert h1 == h2  # minimum height exists: 1 line and 2 lines both take the same minimum
    assert out3.shape[1] == one.shape[1] == IMG  # width unchanged


def test_append_step_frame_normalizes_size(wrapper):
    a = np.zeros((70, 80, 3), dtype=np.uint8)
    b = np.full((90, 80, 3), 7, dtype=np.uint8)
    wrapper._video_append_step_frame(a, False)
    wrapper._video_append_step_frame(b, True)
    assert [f.shape for f in wrapper.video_frames] == [(70, 80, 3), (70, 80, 3)]
    assert len(wrapper.no_object_video_frames) == 1 and wrapper.no_object_video_frames[0] is wrapper.video_frames[1]
    assert int(wrapper.video_frames[1][35, 40, 0]) == 7


@pytest.mark.parametrize("success", [True, False])
def test_flush_names(wrapper, writes, success):
    wrapper.video_frames = [front_rgb(1)]
    wrapper.no_object_video_frames = [front_rgb(2)]
    wrapper._video_flush_episode_files(success, "T_ep1_seed2", "easy_goal")
    names = [n for n, _ in writes]
    if success:
        assert names == ["T_ep1_seed2_easy_goal.mp4", "success_NO_OBJECT_T_ep1_seed2_easy_goal.mp4"]
    else:
        assert names == ["FAILED_T_ep1_seed2_easy_goal.mp4", "FAILED_NO_OBJECT_T_ep1_seed2_easy_goal.mp4"]


def test_flush_skips_when_empty_or_disabled(wrapper, writes, tmp_path):
    wrapper._video_flush_episode_files(True, "p", "s")
    wrapper.save_video = False
    wrapper.video_frames = [front_rgb(1)]
    wrapper._video_flush_episode_files(True, "p", "s")
    assert writes == [] and not (tmp_path / "out" / "videos").exists()


def test_flush_swallows_encoder_failure(wrapper, mod, monkeypatch):
    def boom(self, frames, path):
        raise RuntimeError("encoder broken")

    monkeypatch.setattr(mod.RobommeRecordWrapper, "_video_write_mp4", boom)
    wrapper.video_frames = [front_rgb(1)]
    wrapper._video_flush_episode_files(True, "p", "s")  # does not raise


# ---------------------------------------------------------------- video in the closed loop


EPISODE = [
    Event(name="NO RECORD", demo=True),
    Event(name="watch", demo=True),
    Event(name="watch", demo=True),
    Event(name="NO RECORD", demo=False, task_index=1),
    Event(name="pick", demo=False, task_index=1),
    Event(name="pick", demo=False, task_index=2, terminated=True, success=True),
]
RECORDED = 4  # number of non-NO RECORD steps; reset produces no frame


def _goal_patch(mod, monkeypatch, goals):
    monkeypatch.setattr(mod.task_goal, "get_language_goal", lambda env, env_id: list(goals))


def test_episode_video_frames_and_name(mod, writes, monkeypatch, tmp_path):
    _goal_patch(mod, monkeypatch, ["Goal A, now", "goal/b"])
    _, _, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, EPISODE, env_id="MyTask")
    assert [n for n, _ in writes] == ["MyTask_ep3_seed77_easy_Goal_A_now__ALT__goal_b.mp4"]
    frames = writes[0][1]
    assert len(frames) == RECORDED
    assert len({f.shape for f in frames}) == 1
    # demonstration frames (first 2) have red borders in the corners; online frames' bottom-left corner is base image pixel (B channel 200), not red
    for f in frames[:2]:
        assert f[-1, 0].tolist() == RED
    for f in frames[2:]:
        assert f[-1, 0].tolist() != RED
    with h5py.File(path, "r") as f:
        ep = f["episode_3"]
        n = len([k for k in ep if k.startswith("timestep_")])
        assert n == RECORDED  # video frames correspond one-to-one with h5 records
        np.testing.assert_array_equal(ep["timestep_0/obs/front_rgb"][()], front_rgb(2))
        assert [s.decode() for s in ep["setup/task_goal"][()]] == ["Goal A, now", "goal/b"]


def test_failed_episode_video_prefix_and_no_h5(mod, writes, monkeypatch, tmp_path):
    _goal_patch(mod, monkeypatch, [])
    events = EPISODE[:-1] + [Event(name="pick", task_index=2, terminated=True, success=False)]
    _, env, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, events)
    assert [n for n, _ in writes] == ["FAILED_FakeTask_ep3_seed77_easy_no_goal.mp4"]
    with h5py.File(path, "r") as f:
        assert "episode_3" not in f


@pytest.mark.parametrize("fail,suffix", [("xy", "_FailRecoverXY"), ("z", "_FailRecoverZ"), (None, "_FailRecover")])
def test_fail_recover_suffix(mod, writes, monkeypatch, tmp_path, fail, suffix):
    _goal_patch(mod, monkeypatch, [])
    from recording_fakes import make_wrapper, drive

    w, env = make_wrapper(mod.RobommeRecordWrapper, tmp_path, EPISODE)
    env.use_fail_planner, env.fail = True, fail
    w.reset()
    drive(w, env, EPISODE)
    w.close()
    assert [n for n, _ in writes] == [f"FakeTask_ep3_seed77{suffix}_easy_no_goal.mp4"]


def test_no_object_video_when_target_missing(mod, writes, monkeypatch, tmp_path):
    """When the target cannot be found in the segmentation map at a subgoal switch → that frame also goes into the NO_OBJECT video, and grounded text falls back to the task name."""
    _goal_patch(mod, monkeypatch, [])
    events = [
        Event(name="pick", task_index=0, subgoal="pick <obj>"),
        Event(name="place", task_index=1, subgoal="place at <obj>", seg_visible=False),
        Event(name="place", task_index=1, subgoal="place at <obj>", terminated=True, success=True),
    ]
    _, _, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, events)
    names = [n for n, _ in writes]
    assert names == ["FakeTask_ep3_seed77_easy_no_goal.mp4", "success_NO_OBJECT_FakeTask_ep3_seed77_easy_no_goal.mp4"]
    assert len(writes[0][1]) == 3 and len(writes[1][1]) == 1
    with h5py.File(path, "r") as f:
        assert f["episode_3/timestep_0/info/grounded_subgoal"][()].decode() == "pick <14, 24>"
        assert f["episode_3/timestep_1/info/grounded_subgoal"][()].decode() == "place"


def test_encoder_failure_does_not_block_h5(mod, monkeypatch, tmp_path):
    def boom(self, frames, path):
        raise RuntimeError("encoder broken")

    monkeypatch.setattr(mod.RobommeRecordWrapper, "_video_write_mp4", boom)
    _, _, path, _ = run_episode(mod.RobommeRecordWrapper, tmp_path, EPISODE)
    with h5py.File(path, "r") as f:
        assert len([k for k in f["episode_3"] if k.startswith("timestep_")]) == RECORDED


# ---------------------------------------------------------------- slow: real encoding


def _need_ffmpeg():
    try:
        import imageio_ffmpeg

        imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        pytest.skip("Unverified: ffmpeg missing")


@pytest.mark.slow
def test_real_mp4_roundtrip(mod, monkeypatch, tmp_path):
    _need_ffmpeg()
    import imageio

    _goal_patch(mod, monkeypatch, [])
    run_episode(mod.RobommeRecordWrapper, tmp_path, EPISODE)
    videos = sorted((tmp_path / "out" / "videos").iterdir())
    assert [v.name for v in videos] == ["FakeTask_ep3_seed77_easy_no_goal.mp4"]
    with imageio.get_reader(videos[0].as_posix()) as r:
        n = sum(1 for _ in r)
    assert n == RECORDED
