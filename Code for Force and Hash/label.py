# 文件名: label.py
from point import Point
from generic_kalman_filter import GenericKalmanFilter
import numpy as np

# Label类独立文件
class Label:
    def __init__(self, id, feature, text, width, height, mass=1, time_step=0.05):
        self.id = id
        self.feature = feature  # 关联的Point对象
        self.text = text
        self.width = width
        self.height = height
        self.mass = mass
        self.x = feature.x + feature.radius + 10
        self.y = feature.y
        self.vx = 0
        self.vy = 0
        self.ax = 0 # 上一帧加速度
        self.ay = 0
        self.kf = self.setup_kf_for_2d_motion(time_step, self.x, self.y)

    @property
    def center_x(self):
        return self.x + self.width / 2

    @property
    def center_y(self):
        return self.y + self.height / 2

    @staticmethod
    def setup_kf_for_2d_motion(dt, initial_x, initial_y):
        A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
        B = np.array([[0.5*dt**2, 0], [dt, 0], [0, 0.5*dt**2], [0, dt]])
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
        Q = np.eye(4) * 1.0
        R = np.eye(2) * 0.5
        P = np.eye(4) * 10
        x0 = np.array([[initial_x], [0], [initial_y], [0]])
        return GenericKalmanFilter(A=A, B=B, H=H, x=x0, P=P, Q=Q, R=R)
