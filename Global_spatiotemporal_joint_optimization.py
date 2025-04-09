import math
import random
import matplotlib.pyplot as plt
import os
import numpy as np

# 参数设置（参考论文附录A1）
paramsA1 = {
    'Wlabel-label': 80,
    'Wlabel-feature': 50,
    'Worient': [4, 3, 2, 1],  # 四个象限的权重
    'Wdistance': 20,
    'Wout-of-axes': 320,
    'Wintersect': 1,  # leader线交叉惩罚权重
    'Wradius': 20,
    'Wangle': 10,
    'Dlabel-collision': 30,
    'delta_t': 5  # 特征未来预测时间间隔
}

class LabelOptimizer:
    def __init__(self, labels, features, params, max_x=1000, max_y=1000):
        self.labels = labels
        self.features = features
        self.params = params
        self.constraints = {}
        self.joint_sets = []
        self.max_x = max_x  # 可视区域最大X坐标
        self.max_y = max_y  # 可视区域最大Y坐标


    def calculate_angle_delta(self, theta):
        """计算标签的角度差异"""
        # 角度范围限制为 0 到 2π
        return min(abs(theta - 0), 2 * math.pi - abs(theta - 0))

    def calculate_rectangle_overlap(self, i, j, label_positions):
        """计算两个标签（矩形与矩形）的重叠面积"""
        x1, y1 = label_positions[i]
        x2, y2 = label_positions[j]
        l1, w1 = self.labels[i].length, self.labels[i].width
        l2, w2 = self.labels[j].length, self.labels[j].width

        # 计算矩形边界
        rect1 = {
            'x_min': x1,
            'x_max': x1 + w1,
            'y_min': y1,
            'y_max': y1 + l1
        }
        rect2 = {
            'x_min': x2,
            'x_max': x2 + w2,
            'y_min': y2,
            'y_max': y2 + l2
        }

        # 计算重叠区域
        overlap_x = max(0, min(rect1['x_max'], rect2['x_max']) - max(rect1['x_min'], rect2['x_min']))
        overlap_y = max(0, min(rect1['y_max'], rect2['y_max']) - max(rect1['y_min'], rect2['y_min']))
        return self.params['Wlabel-label'] * overlap_x * overlap_y

    def calculate_rectangle_circle_overlap(self, i, label_positions):
        """计算矩形与圆的重叠面积（精确计算）"""
        x, y = label_positions
        l, w = self.labels[i].length, self.labels[i].width
        feature_center = self.features[i].position
        feature_radius = self.features[i].radius

        # 矩形的四个角坐标
        corners = [
            (x, y),
            (x + w, y),
            (x, y + l),
            (x + w, y + l)
        ]

        # 计算圆心到矩形的距离
        dx = feature_center[0] - x
        dy = feature_center[1] - y
        distance = math.hypot(dx, dy)

        # 分离轴定理判断是否相交
        if distance > feature_radius + max(w, l) / 2:
            return 0

        # 精确计算重叠面积（使用积分或几何分解）
        return self.params['Wlabel-feature'] * min(math.pi * feature_radius ** 2, l * w)

    def cartesian_to_polar(self, cartesian):
        """笛卡尔坐标转极坐标"""
        x, y = cartesian
        r = math.hypot(x, y)
        theta = math.atan2(y, x)
        return r, theta

    def lines_intersect(self, line1, line2):
        """判断两条线段是否相交"""

        def ccw(A, B, C):
            """计算三点的顺时针或逆时针关系"""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = line1  # line1: (x1, y1) -> (x2, y2)
        C, D = line2  # line2: (x3, y3) -> (x4, y4)

        # 确保 A, B, C, D 都是坐标元组 (x, y)
        if not all(isinstance(p, tuple) and len(p) == 2 for p in [A, B, C, D]):
            raise ValueError("Each point in line1 and line2 must be a tuple (x, y)")

        # 判断线段 AB 和 CD 是否相交
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def calculate_leader_intersections(self, label_positions):
        """计算leader线交叉次数"""
        intersections = 0
        # 确保 label_positions 中不包含 None 值
        valid_positions = [pos for pos in label_positions if pos is not None]
        # print(label_positions)

        # 遍历每一对有效标签位置
        for i in valid_positions:
            for j in valid_positions:
                # 获取标签 i 和 j 的位置
                line1 = (label_positions[i], self.features[i].position)
                line2 = (label_positions[j], self.features[j].position)

                # 判断这两条线段是否相交
                if self.lines_intersect(line1, line2):
                    intersections += 1

        return intersections

    def check_out_of_axes(self, label_positions):
        """计算标签超出可视区域的面积"""
        total_area = 0
        for i, pos in enumerate(label_positions):
            x, y = pos
            label = self.labels[i]
            label_width = label.width
            label_height = label.length

            # 计算各边超出量
            clipped_left = max(0, -x)  # 左侧超出
            clipped_right = max(0, (x + label_width) - self.max_x)  # 右侧超出
            clipped_top = max(0, -y)  # 上侧超出
            clipped_bottom = max(0, (y + label_height) - self.max_y)  # 下侧超出

            # 计算超出区域面积
            total_area += (
                clipped_left * label_height +
                clipped_right * label_height +
                clipped_top * label_width +
                clipped_bottom * label_width
            )

        return total_area

    def simulated_annealing(self, initial_positions, joint_set, max_iter=5000):
        """模拟退火优化"""
        current_features = list(joint_set['set'])
        num_labels = len(current_features)
        current_pos = initial_positions.copy()
        best_pos = current_pos.copy()  # 初始最优位置就是当前的位置
        current_energy = self.calculate_static_energy(current_pos, joint_set)
        best_energy = current_energy
        temp = 1000.0

        for _ in range(max_iter):
            # 生成新解（仅扰动当前联合集中的标签）
            new_pos = {}
            for key, (x, y) in current_pos.items():  # 迭代字典项，key 是标签 ID，(x, y) 是坐标
                new_pos[key] = (
                    x + random.uniform(-temp / 100, temp / 100),
                    y + random.uniform(-temp / 100, temp / 100)
                )

            new_energy = self.calculate_static_energy(new_pos, joint_set)
            delta = new_energy - current_energy

            # 接受准则
            if delta < 0 or math.exp(-delta / temp) > random.random():
                current_pos = new_pos
                current_energy = new_energy
                if new_energy < best_energy:
                    best_pos = new_pos  # 更新最优位置
                    best_energy = new_energy

            temp *= 0.99  # 降温

        return best_pos  # 返回的是字典格式：{label_id: (x, y)}

    def detect_joint_sets(self):
        """检测时空交集区域（joint sets）- 按照论文方法改进"""
        joint_sets = []
        max_frames = len(self.features[0].trajectory) if self.features else 0
        
        for t in range(max_frames):
            current_set = set()
            
            # 计算每对特征在当前时间点及未来时间段内的空间关系
            for i in range(len(self.features)):
                for j in range(i + 1, len(self.features)):
                    # 检查当前帧到未来delta_t帧内的轨迹
                    has_interaction = False
                    for dt in range(min(self.params['delta_t'], max_frames - t)):
                        future_frame = t + dt
                        pos_i = self.features[i].trajectory[future_frame]
                        pos_j = self.features[j].trajectory[future_frame]
                        
                        # 计算特征之间的距离和它们的标签大小
                        dx = pos_i[0] - pos_j[0]
                        dy = pos_i[1] - pos_j[1]
                        distance = math.hypot(dx, dy)
                        
                        # # 考虑标签大小的动态阈值
                        # label_i_size = math.hypot(self.labels[i].width, self.labels[i].length)
                        # label_j_size = math.hypot(self.labels[j].width, self.labels[j].length)
                        # threshold = self.params['Dlabel-collision']
                        
                        if distance <10:
                            has_interaction = True
                            break
                    
                    if has_interaction:
                        current_set.update({i, j})
            
            if current_set:
                # 计算joint set的复杂度
                complexity = self.calculate_joint_set_complexity(current_set, t)
                joint_sets.append({
                    'set': current_set, 
                    'frame': t,
                    'complexity': complexity
                })
        
        # 按复杂度降序排序
        self.joint_sets = sorted(joint_sets, key=lambda x: x['complexity'], reverse=True)

    def calculate_joint_set_complexity(self, feature_set, frame):
        """计算joint set的复杂度"""
        complexity = 0
        features_list = list(feature_set)
        
        # 计算特征密度
        positions = [self.features[i].trajectory[frame] for i in features_list]
        min_x = min(p[0] for p in positions)
        max_x = max(p[0] for p in positions)
        min_y = min(p[1] for p in positions)
        max_y = max(p[1] for p in positions)
        
        area = (max_x - min_x + 1) * (max_y - min_y + 1)
        density = len(feature_set) / area if area > 0 else float('inf')
        
        # 计算轨迹交叉
        intersections = 0
        for i in range(len(features_list)):
            for j in range(i + 1, len(features_list)):
                if self.check_trajectory_intersection(features_list[i], features_list[j]):
                    intersections += 1
        
        # 综合考虑密度和交叉数
        complexity = density * (1 + intersections)
        return complexity

    def check_trajectory_intersection(self, feature_i, feature_j):
        """检查两个特征的轨迹是否相交"""
        traj_i = self.features[feature_i].trajectory
        traj_j = self.features[feature_j].trajectory
        
        for t in range(len(traj_i) - 1):
            line1 = (traj_i[t], traj_i[t + 1])
            line2 = (traj_j[t], traj_j[t + 1])
            if self.lines_intersect(line1, line2):
                return True
        return False

    def calculate_static_energy(self, label_positions, joint_set):
        """改进的静态能量计算"""
        E_static = 0
        E_overlap = 0
        E_position = 0
        E_aesthetics = 0
        E_constraint = 0
        
        current_features = list(joint_set['set'])
        
        # 计算重叠能量 (E_overlap)
        for i in range(len(current_features)):
            for j in range( len(current_features)):
                if i==j:
                    continue
                global_i = current_features[i]
                global_j = current_features[j]
                
                # 标签间重叠
                O_ij = self.calculate_rectangle_overlap(global_i, global_j, label_positions)
                E_overlap += self.params['Wlabel-label'] * O_ij

        for i in range(len(current_features)):
            for j in range(len(current_features)):
                global_i = current_features[i]
                global_j = current_features[j]
                           
                # 标签与特征重叠
                P_ij = self.calculate_label_feature_overlap(global_i, global_j, label_positions)
                E_overlap += self.params['Wlabel-feature'] * P_ij
        
        
        # 计算位置能量 (E_position)
        for i in range(len(current_features)):
            feature_idx = current_features[i]
            label_pos = label_positions[feature_idx]
            feature_pos = self.features[feature_idx].position
            
            # 计算极坐标
            dx = label_pos[0] - feature_pos[0]
            dy = label_pos[1] - feature_pos[1]
            r, theta = self.cartesian_to_polar((dx, dy))
            
            # 计算方向能量
            quadrant = self.get_quadrant(theta)
            E_position += self.params['Worient'][quadrant-1] * self.calculate_angle_delta(theta)
            
            # 计算距离能量
            E_position += self.params['Wdistance'] * r
        
        # 计算美学能量 (E_aesthetics)
        out_of_axes_area = self.check_out_of_axes([label_positions[i] for i in current_features])
        leader_intersections = self.calculate_leader_intersections(label_positions)
        E_aesthetics = (
            self.params['Wout-of-axes'] * out_of_axes_area +
            self.params['Wintersect'] * leader_intersections
        )
        
        # 计算约束能量 (E_constraint)
        if self.constraints:
            for idx in current_features:
                if idx in self.constraints:
                    current_pos = label_positions[idx]
                    dx = current_pos[0] - self.features[idx].position[0]
                    dy = current_pos[1] - self.features[idx].position[1]
                    r_p, theta_p = self.cartesian_to_polar((dx, dy))
                    r_l, theta_l = self.constraints[idx]
                    
                    E_constraint += (
                        self.params['Wradius'] * abs(r_p - r_l) +
                        self.params['Wangle'] * abs(theta_p - theta_l)
                    )
        
        return E_overlap + E_position + E_aesthetics + E_constraint

    def calculate_label_feature_overlap(self, label_idx, feature_idx, label_positions):
        """精确计算标签与特征的重叠面积"""
        label_pos = label_positions[label_idx]
        feature_pos = self.features[feature_idx].position
        feature_radius = self.features[feature_idx].radius
        label_width = self.labels[label_idx].width
        label_height = self.labels[label_idx].length
        
        # 计算矩形的四个顶点
        rect_corners = [
            (label_pos[0], label_pos[1]),
            (label_pos[0] + label_width, label_pos[1]),
            (label_pos[0], label_pos[1] + label_height),
            (label_pos[0] + label_width, label_pos[1] + label_height)
        ]
        
        # 计算圆心到矩形各边的最短距离
        min_dist = float('inf')
        for i in range(4):
            p1 = rect_corners[i]
            p2 = rect_corners[(i+1)%4]
            dist = self.point_to_line_distance(feature_pos, p1, p2)
            min_dist = min(min_dist, dist)
        
        # 如果最短距离大于特征半径，则无重叠
        if min_dist > feature_radius:
            return 0
        
        # 计算重叠面积（使用数值积分方法）
        overlap_area = self.numerical_integration_overlap(
            feature_pos, feature_radius,
            label_pos, label_width, label_height
        )
        
        return overlap_area

    def point_to_line_distance(self, point, line_start, line_end):
        """计算点到线段的最短距离"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        # 线段长度的平方
        l2 = (x2-x1)**2 + (y2-y1)**2
        
        if l2 == 0:
            return math.hypot(x0-x1, y0-y1)
        
        # 参数t表示投影点在线段上的位置
        t = max(0, min(1, ((x0-x1)*(x2-x1) + (y0-y1)*(y2-y1))/l2))
        
        # 投影点坐标
        projection_x = x1 + t*(x2-x1)
        projection_y = y1 + t*(y2-y1)
        
        return math.hypot(x0-projection_x, y0-projection_y)

    def numerical_integration_overlap(self, circle_center, circle_radius, rect_pos, rect_width, rect_height):
        """使用数值积分计算圆与矩形的重叠面积"""
        dx = 0.5  # 积分步长
        dy = 0.5
        
        overlap_area = 0
        x_start = max(rect_pos[0], circle_center[0] - circle_radius)
        x_end = min(rect_pos[0] + rect_width, circle_center[0] + circle_radius)
        y_start = max(rect_pos[1], circle_center[1] - circle_radius)
        y_end = min(rect_pos[1] + rect_height, circle_center[1] + circle_radius)
        
        for x in np.arange(x_start, x_end, dx):
            for y in np.arange(y_start, y_end, dy):
                # 检查点是否在圆内和矩形内
                in_circle = (x - circle_center[0])**2 + (y - circle_center[1])**2 <= circle_radius**2
                in_rect = (x >= rect_pos[0] and x <= rect_pos[0] + rect_width and
                          y >= rect_pos[1] and y <= rect_pos[1] + rect_height)
                
                if in_circle and in_rect:
                    overlap_area += dx * dy
        
        return overlap_area

    def get_quadrant(self, theta):
        """根据角度确定象限"""
        theta = theta % (2 * math.pi)
        if 0 <= theta < math.pi/2:
            return 1
        elif math.pi/2 <= theta < math.pi:
            return 2
        elif math.pi <= theta < 3*math.pi/2:
            return 3
        else:
            return 4

    def optimize(self):
        """全局静态优化流程"""
        self.detect_joint_sets()

        output_dir = "optimized_labels"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        first_frame_positions = {}
        all_joint_set_positions = []

        # 针对各个 joint set 分别进行优化，并更新约束
        for idx, joint_set in enumerate(self.joint_sets):
            current_features = list(joint_set['set'])
            frame_number = joint_set['frame']

            # 初始化每个标签的位置（仅对 joint set 内的标签）
            initial_positions = {}
            for feature_idx in current_features:
                feature = self.features[feature_idx]
                angle = random.uniform(0, 2 * math.pi)
                radius = feature.radius + self.labels[feature_idx].length / 2
                x = feature.position[0] + radius * math.cos(angle)
                y = feature.position[1] + radius * math.sin(angle)
                initial_positions[feature_idx] = (x, y)

            # 使用模拟退火进行优化，返回格式为 {label_id: (x, y)}
            optimized_positions = self.simulated_annealing(initial_positions, joint_set)

            # 更新约束条件（joint set 内标签的约束）
            for i, idx_feat in enumerate(current_features):
                pos = optimized_positions[idx_feat]
                dx = pos[0] - self.features[idx_feat].position[0]
                dy = pos[1] - self.features[idx_feat].position[1]
                r, theta = self.cartesian_to_polar((dx, dy))
                self.constraints[idx_feat] = (r, theta)

            # 更新 joint set 的最终位置
            joint_set['position'] = optimized_positions

            # 更新对应标签的位置
            for i, idx_feat in enumerate(current_features):
                self.labels[idx_feat].position = optimized_positions[idx_feat]

            # 绘制并保存该 joint set 的标签布局图像
            self.plot_label_layout(frame_number, joint_set, optimized_positions, output_dir, idx)

            # 记录 joint set 中的标签位置（注意，这里只记录 joint set 内的标签）
            if frame_number == 0:
                first_frame_positions = {self.labels[idx_feat].id: optimized_positions[idx_feat] for idx_feat in
                                         current_features}

            all_joint_set_positions.append({
                'frame': frame_number,
                'positions': {self.labels[idx_feat].id: optimized_positions[idx_feat] for idx_feat in current_features}
            })

        # 对所有标签（整个轨迹）的第0帧进行统一优化：
        # 对于不在任一 joint set 内的标签，采用它们当前的初始位置
        all_label_ids = [label.id for label in self.labels]
        combined_positions = {}
        for idx in range(len(self.labels)):
            # 如果 joint set 中已经有，则用 joint set 中的；否则，使用当前标签的位置
            if idx in first_frame_positions:
                combined_positions[idx] = first_frame_positions[self.labels[idx].id]
            else:
                combined_positions[idx] = self.labels[idx].position

        # 对所有标签进行一次全局模拟退火优化，传递全体标签的约束
        # 构造全体标签的联合集（set）
        full_set = {'set': list(range(len(self.labels)))}
        final_first_frame_positions = self.simulated_annealing(combined_positions, full_set)

        return final_first_frame_positions, all_joint_set_positions

    def plot_label_layout(self, frame_number, joint_set, optimized_positions, output_dir, joint_set_idx):
        """绘制标签布局并保存图像"""
        plt.figure(figsize=(10, 10))

        # 获取特征点的位置
        feature_positions = [self.features[i].position for i in joint_set['set']]

        # 提取特征点的 x 和 y 坐标
        feature_x = [pos[0] for pos in feature_positions]
        feature_y = [pos[1] for pos in feature_positions]
        plt.scatter(feature_x, feature_y, color='red', label='Features')

        # 提取优化后的标签的 x 和 y 坐标
        label_x = [pos[0] for pos in optimized_positions.values()]  # 取字典中的值并提取 x 坐标
        label_y = [pos[1] for pos in optimized_positions.values()]  # 取字典中的值并提取 y 坐标
        plt.scatter(label_x, label_y, color='blue', label='Labels')

        # 绘制标签与特征之间的连接线（leader线）
        for i, feature_idx in enumerate(joint_set['set']):
            feature_pos = self.features[feature_idx].position
            plt.plot([feature_pos[0], label_x[i]], [feature_pos[1], label_y[i]], 'gray', linestyle='dotted')

        plt.title(f"Label Placement for Frame {frame_number}, Joint Set {joint_set_idx}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")

        plt.legend()

        # 保存图像
        plt.savefig(f"{output_dir}/frame_{frame_number}_joint_set_{joint_set_idx}.png")
        plt.close()
