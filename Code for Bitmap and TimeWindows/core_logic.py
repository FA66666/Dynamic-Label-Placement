import math
import numpy as np
import config

class Vec2:
    """一个简单的二维向量类，用于处理位置和速度。"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other): return Vec2(self.x + other.x, self.y + other.y)
    def __sub__(self, other): return Vec2(self.x - other.x, self.y - other.y)
    def __mul__(self, scalar): return Vec2(self.x * scalar, self.y * scalar)
    def __repr__(self): return f"Vec2({self.x}, {self.y})"
    @property
    def as_tuple(self): return (self.x, self.y)

class Feature:
    """代表一个数据点（锚点），内置卡尔曼滤波器用于追踪和预测其运动。"""
    def __init__(self, feature_id, text, label_size, initial_pos, dt=1.0):
        self.id = feature_id
        self.text = text
        self.size = label_size
        self.positions = {0: Vec2(initial_pos[0], initial_pos[1])}
        self.dt = dt
        self.kf_x = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=float)
        self.kf_F = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.kf_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.kf_Q = np.eye(4) * config.KALMAN_Q
        self.kf_R = np.eye(2) * config.KALMAN_R
        self.kf_P = np.eye(4) * config.KALMAN_P_INITIAL

    def add_position(self, frame_idx, position_tuple):
        self.positions[frame_idx] = Vec2(position_tuple[0], position_tuple[1])

    def get_position_at(self, frame_idx):
        return self.positions.get(frame_idx)

    def update(self, z):
        S = self.kf_H @ self.kf_P @ self.kf_H.T + self.kf_R
        K = self.kf_P @ self.kf_H.T @ np.linalg.inv(S)
        y = z - self.kf_H @ self.kf_x
        self.kf_x = self.kf_x + K @ y
        self.kf_P = (np.eye(4) - K @ self.kf_H) @ self.kf_P

    def predict_step(self):
        self.kf_x = self.kf_F @ self.kf_x
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q

    def predict_future(self, steps):
        temp_x = self.kf_x.copy()
        for _ in range(steps):
            temp_x = self.kf_F @ temp_x
        return Vec2(temp_x[0], temp_x[1])

class Label:
    """代表标签的视觉实体，管理其位置、动画和躲避逻辑。"""
    def __init__(self, feature):
        self.feature = feature
        self.id = feature.id
        self.history = {}
        self.avoidance_plan = None
        self.current_offset = None
        self.previous_angle = None

    def add_frame_data(self, frame_idx, bbox, pos_model):
        self.history[frame_idx] = {'bbox': bbox, 'pos_model': pos_model}
    
    def set_avoidance_plan(self, start_frame, end_frame, start_offset, target_offset):
        self.avoidance_plan = {
            "start_frame": start_frame, "end_frame": end_frame,
            "start_offset": start_offset, "target_offset": target_offset,
        }

    def cancel_avoidance_plan(self):
        self.avoidance_plan = None

    def _update_angle(self, anchor, bbox):
        center_x = bbox[0] + bbox[2] / 2
        center_y = bbox[1] + bbox[3] / 2
        self.previous_angle = math.degrees(math.atan2(center_y - anchor.y, center_x - anchor.x))

    def calculate_pos_at_frame(self, frame_idx, placer):
        """计算并返回标签在指定帧的最终位置。"""
        current_anchor = self.feature.get_position_at(frame_idx)
        if current_anchor is None:
            current_anchor = self.feature.predict_future(0)
            
        w, h = self.feature.size
        
        if self.avoidance_plan:
            # 如果有躲避计划，则沿着贝塞尔曲线更新 current_offset
            plan = self.avoidance_plan
            start_offset = self.current_offset
            target_offset = plan['target_offset']
            duration = max(1, plan['end_frame'] - plan['start_frame'])
            time_step = 1.0 / duration

            p0 = start_offset; p3 = target_offset
            p1 = p0 + (p3 - p0) * 0.3 + Vec2(-(p3 - p0).y, (p3 - p0).x) * 0.3
            p2 = p0 + (p3 - p0) * 0.7 + Vec2(-(p3 - p0).y, (p3 - p0).x) * 0.3
            t, inv_t = time_step, 1.0 - time_step
            self.current_offset = (p0*(inv_t**3)) + (p1*3*(inv_t**2)*t) + (p2*3*inv_t*(t**2)) + (p3*(t**3))
        
        # 无论是躲避中还是静态，最终位置都由 current_anchor + current_offset 决定
        if self.current_offset is None:
             # 安全保障, 正常应由 main.py 初始化
             self.current_offset = Vec2(w / 2, -h / 2) 
             
        final_pos = current_anchor + self.current_offset
        final_bbox = (final_pos.x, final_pos.y, w, h)
        
        # pos_model 在第1帧后不再有意义，始终记录为None
        final_model = self.history[0]['pos_model'] if frame_idx == 0 and 0 in self.history else None
        
        self.add_frame_data(frame_idx, final_bbox, final_model)
        self._update_angle(current_anchor, final_bbox)
        return final_bbox, final_model