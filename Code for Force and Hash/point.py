# 文件名: point.py
import numpy as np
from generic_kalman_filter import GenericKalmanFilter

def setup_kf_for_2d_motion(dt, initial_x, initial_y):
    A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    B = np.array([[0.5*dt**2, 0], [dt, 0], [0, 0.5*dt**2], [0, dt]])
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    Q = np.eye(4) * 1.0
    R = np.eye(2) * 0.5
    P = np.eye(4) * 10
    x0 = np.array([[initial_x], [0], [initial_y], [0]])
    return GenericKalmanFilter(A=A, B=B, H=H, x=x0, P=P, Q=Q, R=R)

class Point:
    def __init__(self, id, x, y, radius=4, time_step=0.05):
        self.id = id
        self.x = x
        self.y = y
        self.radius = radius
        self.vx = 0
        self.vy = 0
        self.kf = setup_kf_for_2d_motion(time_step, x, y)

    def update_position(self, x, y, time_step):
        if time_step > 0:
            self.vx = (x - self.x) / time_step
            self.vy = (y - self.y) / time_step
        self.x, self.y = x, y
        self.kf.update(np.array([[x], [y]]))
