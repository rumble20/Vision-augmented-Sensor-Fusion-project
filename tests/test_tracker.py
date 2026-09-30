import numpy as np
import pytest

from simulate import run, rmse
from tracker.detector import Detection
from tracker.pipeline import Tracker


def test_no_estimate_before_first_detection():
    assert Tracker().step(None) is None


def test_gate_rejects_far_outlier():
    trk = Tracker(dt=0.1)
    for _ in range(20):
        trk.step(Detection(100.0, 100.0))
    trk.step(Detection(900.0, 900.0))
    assert trk.frames_since_update == 1


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_tracking_survives_frame_drops(seed):
    truth, est, hold, trk, dropped = run(seed=seed)
    assert not np.isnan(est).any()
    assert rmse(truth, est, dropped) < 20.0
    # and clearly better than just holding the last box
    assert rmse(truth, est, dropped) < 0.5 * rmse(truth, hold, dropped)


def test_recovers_after_camera_comes_back():
    truth, est, hold, trk, dropped = run()
    after = range(dropped[-1] + 10, len(truth))
    assert rmse(truth, est, after) < 5.0


def test_predicted_path_heads_the_right_way():
    truth, est, hold, trk, dropped = run(n_frames=150, drop=(60, 70))
    pred = trk.predict_path(3)
    # 3 steps ahead of the last frame is still within a few pixels of the arc
    from simulate import true_path
    future = true_path(153)[-3:]
    assert np.linalg.norm(pred - future, axis=1).max() < 20.0
