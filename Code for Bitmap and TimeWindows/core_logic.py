import math
import numpy as np
import config

class Vec2:
    """一个简单的二维向量类，支持基本向量运算。"""
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

        # 卡尔曼滤波器状态初始化
        self.kf_x = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=float) # 状态: [x, y, vx, vy]
        self.kf_F = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]) # 状态转移矩阵
        self.kf_H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) # 观测矩阵
        self.kf_Q = np.eye(4) * config.KALMAN_Q # 过程噪声
        self.kf_R = np.eye(2) * config.KALMAN_R # 观测噪声
        self.kf_P = np.eye(4) * config.KALMAN_P_INITIAL # 状态协方差

    def add_position(self, frame_idx, position_tuple):
        self.positions[frame_idx] = Vec2(position_tuple[0], position_tuple[1])

    def get_position_at(self, frame_idx):
        return self.positions.get(frame_idx)

    def update(self, z):
        """卡尔曼滤波器：根据新的观测值z更新状态。"""
        S = self.kf_H @ self.kf_P @ self.kf_H.T + self.kf_R
        K = self.kf_P @ self.kf_H.T @ np.linalg.inv(S)
        y = z - self.kf_H @ self.kf_x
        self.kf_x = self.kf_x + K @ y
        self.kf_P = (np.eye(4) - K @ self.kf_H) @ self.kf_P

    def predict_step(self):
        """卡尔曼滤波器：预测下一帧的状态。"""
        self.kf_x = self.kf_F @ self.kf_x
        self.kf_P = self.kf_F @ self.kf_P @ self.kf_F.T + self.kf_Q

    def predict_future(self, steps):
        """使用滤波器预测未来steps帧后的位置。"""
        temp_x = self.kf_x.copy()
        for _ in range(steps):
            temp_x = self.kf_F @ temp_x
        return Vec2(temp_x[0], temp_x[1])

class Label:
    """代表标签的视觉实体，管理其位置、动画和躲避逻辑。"""
    def __init__(self, feature):
        self.feature = feature
        self.id = feature.id
        self.history = {} # 记录每帧的位置信息
        self.avoidance_plan = None # 存储躲避动画计划
        self.current_offset = None # 标签相对于锚点的当前偏移
        self.previous_angle = None # 上一帧的引导线角度

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
        self.previous_angle = math.atan2(center_y - anchor.y, center_x - anchor.x)

    def calculate_pos_at_frame(self, frame_idx, placer):
        """计算并返回标签在指定帧的最终位置。"""
        current_anchor = self.feature.get_position_at(frame_idx)
        if current_anchor is None:
            current_anchor = self.feature.predict_future(0)
            
        w, h = self.feature.size
        final_bbox, final_model = None, None

        if self.avoidance_plan:
            # 如果有躲避计划，则执行贝塞尔曲线插值动画。
            plan = self.avoidance_plan
            
            # 安全地获取上一帧的偏移作为动画起点
            last_pos = self.feature.get_position_at(frame_idx - 1)
            if frame_idx > 0 and (frame_idx - 1) in self.history and last_pos is not None:
                 start_offset = Vec2(*self.history[frame_idx - 1]['bbox'][:2]) - last_pos
            else:
                 start_offset = self.current_offset if self.current_offset else Vec2(0,0)

            target_offset = plan['target_offset']
            duration = max(1, plan['end_frame'] - plan['start_frame'])
            time_step = 1.0 / duration

            # 计算贝塞尔曲线控制点并插值一步
            p0 = start_offset; p3 = target_offset
            p1 = p0 + (p3 - p0) * 0.3 + Vec2(-(p3 - p0).y, (p3 - p0).x) * 0.3
            p2 = p0 + (p3 - p0) * 0.7 + Vec2(-(p3 - p0).y, (p3 - p0).x) * 0.3
            t, inv_t = time_step, 1.0 - time_step
            self.current_offset = (p0*(inv_t**3)) + (p1*3*(inv_t**2)*t) + (p2*3*inv_t*(t**2)) + (p3*(t**3))
            
            final_pos = current_anchor + self.current_offset
            final_bbox = (final_pos.x, final_pos.y, w, h)
        else:
            # 如果没有躲避计划，则保持当前偏移或使用静态位置模型。
            if self.current_offset is not None:
                final_pos = current_anchor + self.current_offset
                final_bbox = (final_pos.x, final_pos.y, w, h)
            else:
                if frame_idx > 0 and (frame_idx-1) in self.history:
                    last_pos_model = self.history[frame_idx-1]['pos_model']
                    point_index = placer.point_id_map.index(self.feature.id)
                    final_bbox = placer._position_model(point_index, last_pos_model, current_anchor.x, current_anchor.y)
                    final_model = last_pos_model
                else: # 仅用于第0帧或异常情况
                    final_bbox = self.history[0]['bbox'] if 0 in self.history else placer._position_model(placer.point_id_map.index(self.feature.id), 4, current_anchor.x, current_anchor.y)
                    final_model = self.history[0]['pos_model'] if 0 in self.history else 4

        self.add_frame_data(frame_idx, final_bbox, final_model)
        self._update_angle(current_anchor, final_bbox)
        return final_bbox, final_model