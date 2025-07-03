# 文件名: models.py
import math
import numpy as np
from generic_kalman_filter import GenericKalmanFilter

def setup_kf_for_2d_motion(dt, initial_x, initial_y):
    """一个辅助函数，用于创建和配置针对我们问题的通用卡尔曼滤波器实例"""
    # 状态转移矩阵 A
    A = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])
    # 控制矩阵 B (假设控制输入是加速度)
    B = np.array([[0.5*dt**2, 0], [dt, 0], [0, 0.5*dt**2], [0, dt]])
    # 测量矩阵 H
    H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    # 过程噪声协方差 Q
    Q = np.eye(4) * 0.1 
    # 测量噪声协方差 R
    R = np.eye(2) * 1.0
    # 初始状态协方差 P
    P = np.eye(4)
    # 初始状态向量 x
    x0 = np.array([[initial_x], [0], [initial_y], [0]])

    return GenericKalmanFilter(A=A, B=B, H=H, x=x0, P=P, Q=Q, R=R)

class Point:
    def __init__(self, id, x, y, radius=1, time_step=0.1):
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

class Label:
    def __init__(self, id, feature, text, width, height, mass=1.0, time_step=0.1):
        self.id = id
        self.feature = feature
        self.text = text
        self.width = width
        self.height = height
        self.mass = mass
        self.x = feature.x + feature.radius + 10
        self.y = feature.y
        self.vx = 0
        self.vy = 0
        self.ax = 0 # 存储上一帧的加速度
        self.ay = 0
        self.kf = setup_kf_for_2d_motion(time_step, self.x, self.y)

    @property
    def center_x(self):
        return self.x + self.width / 2

    @property
    def center_y(self):
        return self.y + self.height / 2