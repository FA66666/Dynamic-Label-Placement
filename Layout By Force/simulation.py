from point import Point
from label import Label
import numpy as np
import random

class SimulationEngine:
    def __init__(self, params, force_calculator):
        self.params = params
        self.force_calculator = force_calculator
        self.features = {}
        self.labels = {}

    def initialize_from_data(self, initial_frame_data):
        """从初始帧数据创建特征点和标签对象"""
        time_step = self.params['time_step']
        for pid, data in initial_frame_data['points'].items():
            point_id = int(pid)
            feature = Point(point_id, data['anchor'][0], data['anchor'][1], time_step=time_step)
            label = Label(point_id, feature, data['text'], data['size'][0], data['size'][1], time_step=time_step)
            self.features[point_id] = feature
            self.labels[point_id] = label
    
    def update_feature_positions(self, frame_data, time_step):
        for pid, data in frame_data['points'].items():
            point_id = int(pid)
            if point_id in self.features:
                self.features[point_id].update_position(data['anchor'][0], data['anchor'][1], time_step)

    def _build_grid(self):
        """构建空间网格用于邻域查找优化"""
        grid = {}
        cell_size = self.params['CellSize']
        all_objects = list(self.labels.values()) + list(self.features.values())
        for p in all_objects:
            x = p.center_x if isinstance(p, Label) else p.x  # 标签用中心点，特征点用位置
            y = p.center_y if isinstance(p, Label) else p.y
            gx, gy = int(x // cell_size), int(y // cell_size)  # 计算网格坐标
            cell_key = (gx, gy)
            if cell_key not in grid: grid[cell_key] = []
            grid[cell_key].append(p)
        return grid

    def _check_predicted_collision(self, label, pred_label_pos, neighbor, pred_neighbor_pos):
        """检查预测位置是否会发生碰撞（AABB检测）"""
        lx_min, ly_min = pred_label_pos[0], pred_label_pos[1]
        lx_max = lx_min + label.width
        ly_max = ly_min + label.height

        if isinstance(neighbor, Label):
            nx_min, ny_min = pred_neighbor_pos[0], pred_neighbor_pos[1]
            nx_max = nx_min + neighbor.width
            ny_max = ny_min + neighbor.height
        else:
            nx_min = pred_neighbor_pos[0] - neighbor.radius
            nx_max = pred_neighbor_pos[0] + neighbor.radius
            ny_min = pred_neighbor_pos[1] - neighbor.radius
            ny_max = pred_neighbor_pos[1] + neighbor.radius
            
        return not (lx_max < nx_min or lx_min > nx_max or ly_max < ny_min or ly_min > ny_max)

    def step(self, time_step):
        """执行一个仿真步骤：预测->决策->更新"""
        grid = self._build_grid()
        new_forces = {}

        predicted_positions = {}
        all_objects = list(self.labels.values()) + list(self.features.values())
        for obj in all_objects:
            u = np.array([[obj.ax], [obj.ay]]) if isinstance(obj, Label) else 0  # 标签有加速度控制输入
            predicted_state = obj.kf.predict(u=u)
            predicted_positions[obj.id] = (predicted_state[0,0], predicted_state[2,0])

        for label_id, label in self.labels.items():
            apply_force = False
            lx, ly = int(label.center_x // self.params['CellSize']), int(label.center_y // self.params['CellSize'])
            candidate_neighbors = []
            for dx in [-1, 0, 1]:  # 搜索3x3网格区域
                for dy in [-1, 0, 1]:
                    cell_key = (lx + dx, ly + dy)
                    if cell_key in grid:
                        candidate_neighbors.extend(grid[cell_key])
            
            pred_label_pos = predicted_positions[label_id]
            for neighbor in candidate_neighbors:
                if neighbor.id == label.id: continue
                pred_neighbor_pos = predicted_positions[neighbor.id]
                
                if self._check_predicted_collision(label, pred_label_pos, neighbor, pred_neighbor_pos):
                    apply_force = True  # 预测到碰撞，需要施加力
                    break
            
            if apply_force:
                neighbor_labels = [p for p in candidate_neighbors if isinstance(p, Label)]
                neighbor_features = [p for p in candidate_neighbors if isinstance(p, Point)]
                total_fx, total_fy = self.force_calculator.compute_total_force_for_label(
                    label, neighbor_labels, neighbor_features
                )
                direction = self.params.get('force_direction', 'xy')
                if direction != 'xy':  # 力方向投影约束
                    fx_abs = abs(total_fx)
                    fy_abs = abs(total_fy)
                    
                    if fx_abs > fy_abs:
                        total_fy = 0  # 保留x方向力
                    elif fy_abs > fx_abs:
                        total_fx = 0  # 保留y方向力
                    else:
                        if random.choice([True, False]):  # 随机选择保留方向
                            total_fy = 0
                        else:
                            total_fx = 0
                new_forces[label_id] = (total_fx, total_fy)
            else:
                fx_inherent, fy_inherent = self.force_calculator._compute_inherent_forces(label)
                new_forces[label_id] = (fx_inherent, fy_inherent)

        max_step = 10  # 限制单步最大移动距离
        for label_id, label in self.labels.items():
            fx, fy = new_forces[label_id]
            label.ax, label.ay = fx / label.mass, fy / label.mass
            label.vx += label.ax * time_step
            label.vy += label.ay * time_step
            dx = label.vx * time_step
            dy = label.vy * time_step
            step_norm = (dx**2 + dy**2) ** 0.5
            if step_norm > max_step:  # 防止移动过快导致不稳定
                scale = max_step / (step_norm + 1e-8)
                dx *= scale
                dy *= scale
            label.x += dx
            label.y += dy
            label.x = min(max(label.x, 0), 1000 - label.width)  # 边界约束
            label.y = min(max(label.y, 0), 1000 - label.height)
            label.kf.update(np.array([[label.x], [label.y]]))  # 更新卡尔曼滤波器