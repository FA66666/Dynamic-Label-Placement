# 文件名: simulation.py 
from point import Point
from label import Label
import numpy as np

class SimulationEngine:
    def __init__(self, params, force_calculator):
        self.params = params
        self.force_calculator = force_calculator
        self.features = {}
        self.labels = {}

    def initialize_from_data(self, initial_frame_data):
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
        grid = {}
        cell_size = self.params['CellSize']
        all_objects = list(self.labels.values()) + list(self.features.values())
        for p in all_objects:
            x = p.center_x if isinstance(p, Label) else p.x
            y = p.center_y if isinstance(p, Label) else p.y
            gx, gy = int(x // cell_size), int(y // cell_size)
            cell_key = (gx, gy)
            if cell_key not in grid: grid[cell_key] = []
            grid[cell_key].append(p)
        return grid

    def _check_predicted_collision(self, label, pred_label_pos, neighbor, pred_neighbor_pos):
        """
        它接收的 pred_..._pos 是标签的左上角坐标或锚点的中心点坐标。
        """
        # 计算主标签的预测边界 (左上角 -> 边界)
        lx_min, ly_min = pred_label_pos[0], pred_label_pos[1]
        lx_max = lx_min + label.width
        ly_max = ly_min + label.height

        # 计算邻居的预测边界
        if isinstance(neighbor, Label):
            # 邻居是标签 (左上角 -> 边界)
            nx_min, ny_min = pred_neighbor_pos[0], pred_neighbor_pos[1]
            nx_max = nx_min + neighbor.width
            ny_max = ny_min + neighbor.height
        else: # 邻居是特征点 (中心点 -> 边界)
            nx_min = pred_neighbor_pos[0] - neighbor.radius
            nx_max = pred_neighbor_pos[0] + neighbor.radius
            ny_min = pred_neighbor_pos[1] - neighbor.radius
            ny_max = pred_neighbor_pos[1] + neighbor.radius
            
        # AABB 碰撞检测
        return not (lx_max < nx_min or lx_min > nx_max or ly_max < ny_min or ly_min > ny_max)

    def step(self, time_step):
        grid = self._build_grid()
        new_forces = {}

        # 1. 预测阶段
        predicted_positions = {}
        all_objects = list(self.labels.values()) + list(self.features.values())
        for obj in all_objects:
            u = np.array([[obj.ax], [obj.ay]]) if isinstance(obj, Label) else 0
            predicted_state = obj.kf.predict(u=u)
            predicted_positions[obj.id] = (predicted_state[0,0], predicted_state[2,0])

        # 2. 决策与行动阶段
        for label_id, label in self.labels.items():
            apply_force = False
            lx, ly = int(label.center_x // self.params['CellSize']), int(label.center_y // self.params['CellSize'])
            candidate_neighbors = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    cell_key = (lx + dx, ly + dy)
                    if cell_key in grid:
                        candidate_neighbors.extend(grid[cell_key])
            
            pred_label_pos = predicted_positions[label_id]
            for neighbor in candidate_neighbors:
                if neighbor.id == label.id: continue
                pred_neighbor_pos = predicted_positions[neighbor.id]
                
                # 直接使用修正后的函数，不再需要在这里做坐标转换
                if self._check_predicted_collision(label, pred_label_pos, neighbor, pred_neighbor_pos):
                    apply_force = True
                    break
            
            if apply_force:
                neighbor_labels = [p for p in candidate_neighbors if isinstance(p, Label)]
                neighbor_features = [p for p in candidate_neighbors if isinstance(p, Point)]
                total_fx, total_fy = self.force_calculator.compute_total_force_for_label(
                    label, neighbor_labels, neighbor_features
                )
                new_forces[label_id] = (total_fx, total_fy)
            else:
                fx_inherent, fy_inherent = self.force_calculator._compute_inherent_forces(label)
                new_forces[label_id] = (fx_inherent, fy_inherent)

        # 3. 更新状态
        for label_id, label in self.labels.items():
            fx, fy = new_forces[label_id]
            label.ax, label.ay = fx / label.mass, fy / label.mass
            label.vx += label.ax * time_step
            label.vy += label.ay * time_step
            label.x += label.vx * time_step
            label.y += label.vy * time_step
            # 注意：我们的卡尔曼滤波器状态x就是左上角位置，所以这里直接用label.x, label.y是正确的
            label.kf.update(np.array([[label.x], [label.y]]))