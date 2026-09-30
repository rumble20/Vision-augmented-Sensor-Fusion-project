import numpy as np


class EKF:
    """EKF with a constant turn rate and velocity motion model.

    state = [x, y, v, yaw, yaw_rate], measurement = [x, y] (bbox centre in pixels)
    """

    def __init__(self, x0, P0, q_v=1.0, q_yaw_rate=0.05, r_pix=3.0):
        self.x = np.asarray(x0, dtype=float)
        self.P = np.asarray(P0, dtype=float)
        self.q_v = q_v
        self.q_yaw_rate = q_yaw_rate
        self.R = np.eye(2) * r_pix ** 2
        self.H = np.array([[1.0, 0, 0, 0, 0],
                           [0, 1.0, 0, 0, 0]])

    @staticmethod
    def f(x, dt):
        px, py, v, yaw, w = x
        if abs(w) < 1e-4:
            # straight line, avoids dividing by ~0
            return np.array([px + v * np.cos(yaw) * dt,
                             py + v * np.sin(yaw) * dt,
                             v, yaw, w])
        return np.array([px + v / w * (np.sin(yaw + w * dt) - np.sin(yaw)),
                         py + v / w * (np.cos(yaw) - np.cos(yaw + w * dt)),
                         v, yaw + w * dt, w])

    @staticmethod
    def jacobian_f(x, dt, eps=1e-6):
        # numerical jacobian, the analytic one is long and easy to get wrong
        n = len(x)
        F = np.zeros((n, n))
        for i in range(n):
            d = np.zeros(n)
            d[i] = eps
            F[:, i] = (EKF.f(x + d, dt) - EKF.f(x - d, dt)) / (2 * eps)
        return F

    def predict(self, dt):
        F = self.jacobian_f(self.x, dt)
        Q = np.diag([0.0, 0.0, self.q_v * dt, 0.0, self.q_yaw_rate * dt]) ** 2 \
            + np.diag([0.1 * dt, 0.1 * dt, 0.0, 0.01 * dt, 0.0])
        self.x = self.f(self.x, dt)
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        y = np.asarray(z, dtype=float) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        return y, S

    def predict_trajectory(self, dt, n_steps):
        """roll the motion model forward without touching the filter state"""
        x = self.x.copy()
        out = []
        for _ in range(n_steps):
            x = self.f(x, dt)
            out.append(x[:2].copy())
        return np.array(out)
