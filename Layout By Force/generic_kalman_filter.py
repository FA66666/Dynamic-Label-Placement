# generic_kalman_filter.py
"""
一个更通用、更完整的卡尔曼滤波器实现。
它在初始化时接收所有必要的矩阵，使其能够适应不同的系统模型。
"""
import numpy as np

class GenericKalmanFilter:
    def __init__(self, A, B, H, x, P, Q, R):
        """
        初始化一个通用的卡尔曼滤波器。
        :param A: 状态转移矩阵 (State Transition Matrix)
        :param B: 控制矩阵 (Control Matrix)
        :param H: 测量矩阵 (Observation Matrix)
        :param x: 初始状态向量 (Initial State Vector)
        :param P: 初始状态协方差矩阵 (Initial State Covariance)
        :param Q: 过程噪声协方差矩阵 (Process Noise Covariance)
        :param R: 测量噪声协方差矩阵 (Measurement Noise Covariance)
        """
        self.A = A
        self.B = B
        self.H = H
        self.x = x
        self.P = P
        self.Q = Q
        self.R = R

    def predict(self, u=0):
        """
        预测下一时刻的状态。
        :param u: 控制向量 (Control Vector)，代表已知的外部影响，如加速度。
        """
        self.x = np.dot(self.A, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        """
        用实际测量值 z 来更新滤波器。
        :param z: 测量向量 (Measurement Vector)
        """
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.A.shape[0])
        self.P = np.dot(I - np.dot(K, self.H), self.P)