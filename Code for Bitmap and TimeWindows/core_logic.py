import math
import numpy as np
import config # 导入配置文件

# --- 辅助类：用于二维向量操作 ---
class Vec2:
    """一个简单的二维向量类，用于处理位置和速度。"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def __repr__(self):
        return f"Vec2({self.x}, {self.y})"

    @property
    def as_tuple(self):
        return (self.x, self.y)

# ----------------------------------------------------------------------
# Feature 类 (已集成卡尔曼滤波器)
# 职责：代表场景中的“特征点”或“锚点”，只管理自身的数据和时序位置。
# ----------------------------------------------------------------------
class Feature:
    """代表一个场景中的数据特征点，并使用卡尔曼滤波器进行状态估计和预测。"""
    def __init__(self, feature_id, text, label_size, initial_pos, dt=1.0):
        self.id = feature_id
        self.text = text
        self.size = label_size
        self.positions = {0: Vec2(initial_pos[0], initial_pos[1])} # 只记录真实观测值
        self.dt = dt # 时间步长

        # --- 卡尔曼滤波器初始化 (从 config 加载参数) ---
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

# ----------------------------------------------------------------------
# Label 类
# 职责：作为 Feature 的视觉表现，负责决定自身显示的位置、模型和躲避动画。
# ----------------------------------------------------------------------
class Label:
    """代表一个标签的视觉表现，管理其显示位置和躲避行为。"""
    def __init__(self, feature):
        self.feature = feature
        self.id = feature.id
        self.history = {}
        self.avoidance_plan = None
        self.current_offset = None

    def add_frame_data(self, frame_idx, bbox, pos_model):
        self.history[frame_idx] = {'bbox': bbox, 'pos_model': pos_model}
    
    def set_avoidance_plan(self, start_frame, end_frame, start_offset, target_offset):
        self.avoidance_plan = {
            "start_frame": start_frame, "end_frame": end_frame,
            "start_offset": start_offset, "target_offset": target_offset,
        }

    def cancel_avoidance_plan(self):
        self.avoidance_plan = None

    def calculate_pos_at_frame(self, frame_idx, placer):
        current_anchor = self.feature.get_position_at(frame_idx)
        w, h = self.feature.size
        
        if self.avoidance_plan:
            plan = self.avoidance_plan
            
            # 动画逻辑修改：不再严格执行到end_frame，而是将计划视为朝向目标的引导
            # 起点是上一帧的实际位置
            if frame_idx > 0 and (frame_idx - 1) in self.history:
                 start_offset = Vec2(*self.history[frame_idx - 1]['bbox'][:2]) - self.feature.get_position_at(frame_idx-1)
            else: # Fallback for the first frame
                 start_offset = self.current_offset if self.current_offset else Vec2(0,0)

            target_offset = plan['target_offset']
            
            # 持续时间决定了移动速度，持续时间越短，移动越快
            duration = max(1, plan['end_frame'] - plan['start_frame'])
            time_step = 1.0 / duration # 每帧移动路径的1/duration

            # 使用贝塞尔曲线计算“一小步”
            p0 = start_offset
            p3 = target_offset
            p1 = p0 + (p3 - p0) * 0.3 + Vec2(-(p3 - p0).y, (p3 - p0).x) * 0.3
            p2 = p0 + (p3 - p0) * 0.7 + Vec2(-(p3 - p0).y, (p3 - p0).x) * 0.3
            
            t, inv_t = time_step, 1.0 - time_step
            # 只插值一小步，得到新的偏移量
            new_offset = (p0 * (inv_t**3)) + (p1 * 3 * (inv_t**2) * t) + \
                     (p2 * 3 * inv_t * (t**2)) + (p3 * (t**3))

            # 将计算出的新偏移量作为当前帧的持久化偏移量
            self.current_offset = new_offset
            
            final_pos = current_anchor + self.current_offset
            final_bbox = (final_pos.x, final_pos.y, w, h)
            self.add_frame_data(frame_idx, final_bbox, None)
            return final_bbox, None

        # 如果没有躲避计划，使用默认定位逻辑
        if self.current_offset is not None:
            final_pos = current_anchor + self.current_offset
            final_bbox = (final_pos.x, final_pos.y, w, h)
            self.add_frame_data(frame_idx, final_bbox, None)
            return final_bbox, None
        else:
            if frame_idx > 0:
                last_pos_model = self.history[frame_idx-1]['pos_model']
                point_index = placer.point_id_map.index(self.feature.id)
                default_bbox = placer._position_model(point_index, last_pos_model, current_anchor.x, current_anchor.y)
                self.add_frame_data(frame_idx, default_bbox, last_pos_model)
                return default_bbox, last_pos_model
            else:
                 return self.history[0]['bbox'], self.history[0]['pos_model']