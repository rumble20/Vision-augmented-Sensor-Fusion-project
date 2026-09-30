import numpy as np

from tracker.detector import SimulatedDetector
from tracker.pipeline import Tracker

DT = 0.1


def true_path(n_frames, v=40.0, yaw_rate=0.15):
    """target driving through a gentle arc (pixel units)"""
    x = np.array([100.0, 100.0, v, 0.0, yaw_rate])
    from tracker.ekf import EKF
    path = []
    for _ in range(n_frames):
        x = EKF.f(x, DT)
        path.append(x[:2].copy())
    return np.array(path)


def run(n_frames=200, drop=(80, 100), seed=1):
    truth = true_path(n_frames)
    dropped = range(*drop)
    det = SimulatedDetector(noise_pix=3.0, dropped_frames=dropped, seed=seed)
    trk = Tracker(dt=DT)

    est = np.full((n_frames, 2), np.nan)
    hold = np.full((n_frames, 2), np.nan)  # baseline: last detection
    last = None
    for k in range(n_frames):
        d = det.detect(truth[k], k)
        if d is not None:
            last = [d.cx, d.cy]
        s = trk.step(d)
        if s is not None:
            est[k] = s[:2]
        if last is not None:
            hold[k] = last
    return truth, est, hold, trk, dropped


def rmse(a, b, idx):
    idx = list(idx)
    return float(np.sqrt(np.mean(np.sum((a[idx] - b[idx]) ** 2, axis=1))))


if __name__ == "__main__":
    truth, est, hold, trk, dropped = run()
    print(f"rmse over the {len(dropped)} dropped frames")
    print(f"  ekf coasting       : {rmse(truth, est, dropped):6.1f} px")
    print(f"  hold last detection: {rmse(truth, hold, dropped):6.1f} px")
    print(f"rmse over all frames : {rmse(truth, est, range(5, len(truth))):6.1f} px")
    print("predicted next 5 positions:")
    print(np.round(trk.predict_path(5), 1))
