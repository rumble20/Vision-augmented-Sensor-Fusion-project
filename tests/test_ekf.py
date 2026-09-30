import numpy as np
from tracker.ekf import EKF


def test_straight_line_motion():
    x = EKF.f(np.array([0.0, 0.0, 10.0, 0.0, 0.0]), dt=1.0)
    assert np.allclose(x[:2], [10.0, 0.0])


def test_turn_keeps_speed_and_advances_yaw():
    x = EKF.f(np.array([0.0, 0.0, 5.0, 0.0, 0.2]), dt=0.5)
    assert np.isclose(x[2], 5.0)
    assert np.isclose(x[3], 0.1)


def test_jacobian_matches_finite_difference_on_straight_line():
    x = np.array([1.0, 2.0, 3.0, 0.3, 0.0])
    F = EKF.jacobian_f(x, dt=0.1)
    # d px / d v = cos(yaw) * dt
    assert np.isclose(F[0, 2], np.cos(0.3) * 0.1, atol=1e-5)


def test_update_pulls_estimate_towards_measurement():
    ekf = EKF([0, 0, 0, 0, 0], np.eye(5) * 100)
    ekf.update([10.0, 5.0])
    assert abs(ekf.x[0] - 10.0) < 1.0
    assert abs(ekf.x[1] - 5.0) < 1.0


def test_predict_grows_covariance():
    ekf = EKF([0, 0, 5, 0, 0], np.eye(5))
    before = np.trace(ekf.P)
    ekf.predict(0.1)
    assert np.trace(ekf.P) > before
