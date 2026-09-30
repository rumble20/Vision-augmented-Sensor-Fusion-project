import numpy as np

from tracker.ekf import EKF


class Tracker:
    """Runs the ekf on top of a detector. When there is no detection
    (frame drop or a rejected one) only the predict step runs."""

    def __init__(self, dt=0.1, gate=9.21, warmup=8, max_rejects=5, **ekf_kwargs):
        self.dt = dt
        self.gate = gate  # chi2, 2 dof, 99%
        self.warmup = warmup  # updates accepted without gating
        self.max_rejects = max_rejects  # in a row, then the track is lost
        self.n_updates = 0
        self.rejects = 0
        self.ekf_kwargs = ekf_kwargs
        self.ekf = None
        self.frames_since_update = 0

    def _init_filter(self, det):
        # nothing is known about speed or heading on the first frame, so the
        # prior has to be wide, a tight one locks the gate onto a wrong track
        x0 = [det.cx, det.cy, 1.0, 0.0, 0.0]
        P0 = np.diag([10.0, 10.0, 50.0 ** 2, np.pi ** 2, 0.5 ** 2])
        self.ekf = EKF(x0, P0, **self.ekf_kwargs)
        self.n_updates = 0
        self.rejects = 0
        self.frames_since_update = 0

    def step(self, det):
        """det is a Detection or None. Returns the state estimate or None
        if the target was never seen."""
        if self.ekf is None:
            if det is None:
                return None
            self._init_filter(det)
            return self.ekf.x.copy()

        self.ekf.predict(self.dt)
        if det is None:
            self.frames_since_update += 1
        elif self.n_updates < self.warmup or self._inside_gate(det):
            self.ekf.update([det.cx, det.cy])
            self.n_updates += 1
            self.rejects = 0
            self.frames_since_update = 0
        else:
            self.rejects += 1
            self.frames_since_update += 1
            if self.rejects >= self.max_rejects:
                self._init_filter(det)
        return self.ekf.x.copy()

    def _inside_gate(self, det):
        y = np.array([det.cx, det.cy]) - self.ekf.H @ self.ekf.x
        S = self.ekf.H @ self.ekf.P @ self.ekf.H.T + self.ekf.R
        return float(y @ np.linalg.solve(S, y)) < self.gate

    def predict_path(self, n_steps):
        return self.ekf.predict_trajectory(self.dt, n_steps)
