import math
import random
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as patches

# 参数设置（参考论文附录A1）
paramsA1 = {
    'Wlabel-label': 80,
    'Wlabel-feature': 50,
    'Worient': [1, 2, 3, 4],  # 四个象限的权重
    'Wdistance': 20,
    'Wout-of-axes': 320,
    'Wintersect': 1,  # leader线交叉惩罚权重
    'Wradius': 20,
    'Wangle': 10,
    'delta_t': 1  # 特征未来预测时间间隔
}

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

class LabelOptimizer:
    """标签布局优化器，包含能量计算和模拟退火算法"""
    def __init__(self, labels, features, params, max_x=1000, max_y=1000):
        self.labels = labels
        self.features = features
        self.params = params
        self.constraints = {}
        self.joint_sets = []
        self.max_x = max_x  # 可视区域最大X坐标
        self.max_y = max_y  # 可视区域最大Y坐标

    def calculate_angle_delta(self, theta):
        """确定标签在哪个象限"""
        quadrant = self.get_quadrant(theta)
        return quadrant  # 返回象限而非角度差异

    def calculate_label_label_overlap(self, i, j, label_positions):
        """计算两个标签（矩形与矩形）的重叠面积"""
        x1, y1 = label_positions[i]
        x2, y2 = label_positions[j]
        l1, w1 = self.labels[i].length, self.labels[i].width
        l2, w2 = self.labels[j].length, self.labels[j].width

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

        overlap_x = max(0, min(rect1['x_max'], rect2['x_max']) - max(rect1['x_min'], rect2['x_min']))
        overlap_y = max(0, min(rect1['y_max'], rect2['y_max']) - max(rect1['y_min'], rect2['y_min']))
        return overlap_x * overlap_y

    def calculate_label_feature_overlap(self, label_idx, feature_idx, label_positions):
        """精确计算标签与特征的重叠面积"""
        label_pos = label_positions[label_idx]
        feature_pos = self.features[feature_idx].position
        feature_radius = self.features[feature_idx].radius
        label_width = self.labels[label_idx].width
        label_length = self.labels[label_idx].length

        rect_corners = [
            (label_pos[0] - label_length/2, label_pos[1]-  label_width/2),
            (label_pos[0] + label_length/2, label_pos[1]-  label_width/2),
            (label_pos[0] - label_length/2, label_pos[1] + label_width/2),
            (label_pos[0] + label_length/2, label_pos[1] + label_width/2)
        ]

        min_dist = float('inf')
        for i in range(4):
            p1 = rect_corners[i]
            p2 = rect_corners[(i+1)%4]
            dist = self.point_to_line_distance(feature_pos, p1, p2)
            min_dist = min(min_dist, dist)

        if min_dist > feature_radius:
            return 0

        # 计算重叠面积（使用数值积分方法）
        overlap_area = self.numerical_integration_overlap(
            feature_pos, feature_radius,
            label_pos, label_width, label_length
        )

        return overlap_area

    def point_to_line_distance(self, point, line_start, line_end):
        """计算点到线段的最短距离"""
        x0, y0 = point
        x1, y1 = line_start
        x2, y2 = line_end

        l2 = (x2-x1)**2 + (y2-y1)**2

        if l2 == 0:
            return math.hypot(x0-x1, y0-y1)

        t = max(0, min(1, ((x0-x1)*(x2-x1) + (y0-y1)*(y2-y1))/l2))

        projection_x = x1 + t*(x2-x1)
        projection_y = y1 + t*(y2-y1)

        return math.hypot(x0-projection_x, y0-projection_y)

    def numerical_integration_overlap(self, circle_center, circle_radius, rect_pos, rect_width, rect_height):
        """使用数值积分计算圆与矩形的重叠面积"""
        dx = 0.01
        dy = 0.01

        overlap_area = 0
        x_start = max(rect_pos[0], circle_center[0] - circle_radius)
        x_end = min(rect_pos[0] + rect_width, circle_center[0] + circle_radius)
        y_start = max(rect_pos[1], circle_center[1] - circle_radius)
        y_end = min(rect_pos[1] + rect_height, circle_center[1] + circle_radius)

        for x in np.arange(x_start, x_end, dx):
            for y in np.arange(y_start, y_end, dy):
                in_circle = (x - circle_center[0])**2 + (y - circle_center[1])**2 <= circle_radius**2
                in_rect = (x >= rect_pos[0] - rect_height/2 and x <= rect_pos[0] + rect_height/2 and
                          y >= rect_pos[1] -  rect_width/2 and y <= rect_pos[1] + rect_width/2)

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

        A, B = line1
        C, D = line2

        # 确保点是坐标元组 (x, y)
        if not all(isinstance(p, tuple) and len(p) == 2 for p in [A, B, C, D]):
            raise ValueError("每个点必须是 (x, y) 元组")

        # 判断线段 AB 和 CD 是否相交
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    def calculate_leader_intersections(self, label_positions):
        """计算leader线交叉次数"""
        intersections = 0
        valid_positions = [pos for pos in label_positions if pos is not None]

        for i in valid_positions:
            for j in valid_positions:
                line1 = (label_positions[i], self.features[i].position)
                line2 = (label_positions[j], self.features[j].position)

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

            clipped_left = max(0, -x)
            clipped_right = max(0, (x + label_width) - self.max_x)
            clipped_top = max(0, -y)
            clipped_bottom = max(0, (y + label_height) - self.max_y)

            total_area += (
                clipped_left * label_height +
                clipped_right * label_height +
                clipped_top * label_width +
                clipped_bottom * label_width
            )

        return total_area

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

    def detect_joint_sets(self):
        """检测时空交集区域（joint sets）- 按照论文方法改进"""
        joint_sets = []
        max_frames = len(self.features[0].trajectory) if self.features else 0

        for t in range(max_frames):
            current_set = set()

            # 检查所有特征对，在未来 delta_t 帧内是否存在距离过近的情况
            for i in range(len(self.features)):
                for j in range(i + 1, len(self.features)):
                    has_interaction = False
                    for dt in range(min(self.params['delta_t'], max_frames - t)):
                        future_frame = t + dt
                        pos_i = self.features[i].trajectory[future_frame]
                        pos_j = self.features[j].trajectory[future_frame]

                        dx = pos_i[0] - pos_j[0]
                        dy = pos_i[1] - pos_j[1]
                        distance = math.hypot(dx, dy)

                        if distance < 5: # 距离阈值判断交互
                            has_interaction = True
                            break

                    if has_interaction:
                        current_set.update({i, j})

            if current_set:
                complexity = self.calculate_joint_set_complexity(current_set, t)
                
                # 记录当前帧中 joint set 内所有特征点的坐标
                feature_positions = {}
                for feat_idx in current_set:
                    feature_positions[feat_idx] = self.features[feat_idx].trajectory[t]
                
                # --- 计算并输出 joint set 内特征点距离 ---
                print(f"\n--- Joint Set at Frame {t} (Complexity: {complexity:.2f}) ---")
                feature_list = list(current_set)
                for idx1 in range(len(feature_list)):
                    for idx2 in range(idx1 + 1, len(feature_list)):
                        feat1_idx = feature_list[idx1]
                        feat2_idx = feature_list[idx2]
                        pos1 = feature_positions[feat1_idx]
                        pos2 = feature_positions[feat2_idx]
                        dist = math.hypot(pos1[0] - pos2[0], pos1[1] - pos2[1])
                        print(f"  Distance between F{feat1_idx} and F{feat2_idx}: {dist:.2f}")
                # ---------------------------------------
                
                joint_sets.append({
                    'set': current_set,
                    'frame': t,
                    'complexity': complexity,
                    'feature_positions': feature_positions  # 存储当前帧特征点坐标
                })

        # 按复杂度降序排序
        self.joint_sets = sorted(joint_sets, key=lambda x: x['complexity'], reverse=True)

    def calculate_joint_set_complexity(self, feature_set, frame):
        """计算joint set的复杂度 = 密度 * (1 + 轨迹交叉数)"""
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

        # 综合复杂度
        complexity = density * (1 + intersections)
        return complexity

    def calculate_static_energy(self, label_positions, joint_set):
        """计算给定联合集的静态能量 E_static = E_overlap + E_position + E_aesthetics + E_constraint"""
        E_overlap = 0
        E_position = 0
        E_aesthetics = 0
        E_constraint = 0

        current_features = list(joint_set['set'])
        # 使用 joint_set 中存储的特征点坐标（如果有）
        feature_positions = joint_set.get('feature_positions', {})

        # 1. 重叠能量 (E_overlap)
        # 1a. 标签间重叠
        for i in range(len(current_features)):
            for j in range(i + 1, len(current_features)):
                global_i = current_features[i]
                global_j = current_features[j]
                O_ij = self.calculate_label_label_overlap(global_i, global_j, label_positions)
                E_overlap += self.params['Wlabel-label'] * O_ij

        # 1b. 标签与特征点重叠
        for i in range(len(current_features)):
            label_i = current_features[i]
            for j in range(len(self.features)):
                if j != label_i:  # 不计算标签与自身对应特征的重叠
                    P_ij = self.calculate_label_feature_overlap(label_i, j, label_positions)
                    E_overlap += self.params['Wlabel-feature'] * P_ij

        # 2. 位置能量 (E_position)
        for i in range(len(current_features)):
            feature_idx = current_features[i]
            label_pos = label_positions[feature_idx]
            
            # 优先使用 joint_set 中存储的特征点坐标
            if feature_idx in feature_positions:
                feature_pos = feature_positions[feature_idx]
            else:
                feature_pos = self.features[feature_idx].position

            dx = label_pos[0] - feature_pos[0]
            dy = label_pos[1] - feature_pos[1]
            r, theta = self.cartesian_to_polar((dx, dy))

            # 方向能量 (基于象限)
            quadrant = self.get_quadrant(theta)
            E_position += self.params['Worient'][quadrant-1] * self.calculate_angle_delta(theta)

            # 距离能量
            E_position += self.params['Wdistance'] * r

        # 3. 美学能量 (E_aesthetics)
        out_of_axes_area = self.check_out_of_axes([label_positions[i] for i in current_features])
        leader_intersections = self.calculate_leader_intersections(label_positions)
        E_aesthetics = (
            self.params['Wout-of-axes'] * out_of_axes_area + # 超出边界惩罚
            self.params['Wintersect'] * leader_intersections # Leader线交叉惩罚
        )

        # 4. 约束能量 (E_constraint)
        for idx in current_features:
            # 仅当特征出现在多个联合集中时，才计算约束（保证标签位置一致性）
            if idx in self.constraints and self.check_feature_in_multiple_joint_sets(idx):
                current_pos = label_positions[idx]
                
                # 优先使用 joint_set 中存储的特征点坐标
                if idx in feature_positions:
                    feature_pos = feature_positions[idx]
                else:
                    feature_pos = self.features[idx].position
                
                dx = current_pos[0] - feature_pos[0]
                dy = current_pos[1] - feature_pos[1]
                r_p, theta_p = self.cartesian_to_polar((dx, dy))
                r_l, theta_l = self.constraints[idx] # 来自上一个优化过的joint set的约束

                E_constraint += (
                    self.params['Wradius'] * abs(r_p - r_l) + # 半径约束
                    self.params['Wangle'] * abs(theta_p - theta_l) # 角度约束
                )

        return E_overlap + E_position + E_aesthetics + E_constraint

    def simulated_annealing(self, initial_positions, joint_set, max_iter=1000):
        """模拟退火优化: 仅优化 joint_set 内的标签位置"""
        current_features = list(joint_set['set'])
        current_pos = initial_positions.copy() # 包含所有标签的当前已知位置
        best_pos = current_pos.copy()
        current_energy = self.calculate_static_energy(current_pos, joint_set)
        best_energy = current_energy
        temp = 1000.0

        for _ in range(max_iter):
            # 生成新解: 仅扰动当前 joint_set 中的标签
            new_pos = current_pos.copy() # 先复制当前所有位置
            for feat_idx in current_features: # 只对 joint set 中的点进行扰动
                x, y = current_pos[feat_idx]
                new_pos[feat_idx] = (
                    x + random.uniform(-temp / 100, temp / 100),
                    y + random.uniform(-temp / 100, temp / 100)
                )

            new_energy = self.calculate_static_energy(new_pos, joint_set)
            delta = new_energy - current_energy

            # Metropolis 接受准则
            if delta < 0 or math.exp(-delta / temp) > random.random():
                current_pos = new_pos
                current_energy = new_energy
                if new_energy < best_energy:
                    best_pos = new_pos # 更新最优解
                    best_energy = new_energy

            temp *= 0.99  # 降温

        # 返回的 best_pos 包含所有标签的位置，但只有 joint_set 内的标签经过了优化
        return best_pos

    def optimize(self):
        """全局静态优化流程：检测joint sets -> 依次优化 -> (可选全局优化) -> 返回最终位置"""
        self.detect_joint_sets()
        # print(self.joint_sets)

        output_dir = "optimized_labels_all_steps"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 初始化标签位置（如果尚未设置）
        for label in self.labels:
            if not hasattr(label, 'position') or label.position is None:
                label.position = None # 初始设为 None

        first_frame_positions = {}  # 存储第一帧优化后的位置
        all_joint_set_positions = []

        # 1. 按复杂度顺序，依次优化每个 joint set
        for idx, joint_set in enumerate(self.joint_sets):
            current_features = list(joint_set['set'])
            frame_number = joint_set['frame']

            # 准备当前 joint set 的初始位置用于模拟退火
            initial_positions_for_sa = {}
            for feature_idx in current_features:
                if self.labels[feature_idx].position: # 如果已有位置（来自之前的优化），则使用
                     initial_positions_for_sa[feature_idx] = self.labels[feature_idx].position
                else: # 否则，生成随机初始位置
                     feature = self.features[feature_idx]
                     angle = random.uniform(0, 2 * math.pi)
                     label_dim = max(self.labels[feature_idx].length, self.labels[feature_idx].width)
                     radius = feature.radius + label_dim / 2 + 5 # 保证在特征点外围
                     x = feature.position[0] + radius * math.cos(angle)
                     y = feature.position[1] + radius * math.sin(angle)
                     initial_positions_for_sa[feature_idx] = (x, y)
                     self.labels[feature_idx].position = (x,y) # 同时更新标签对象的初始位置

            # 对当前 joint set 进行模拟退火优化
            optimized_positions = self.simulated_annealing(initial_positions_for_sa, joint_set)

            # 更新优化后的标签位置和约束条件
            for idx_feat in current_features:
                pos = optimized_positions[idx_feat]
                self.labels[idx_feat].position = pos # 更新全局标签状态

                # 计算并存储极坐标约束，供后续 joint set 使用
                dx = pos[0] - self.features[idx_feat].position[0]
                dy = pos[1] - self.features[idx_feat].position[1]
                r, theta = self.cartesian_to_polar((dx, dy))
                self.constraints[idx_feat] = (r, theta)

            # 存储当前 joint set 的优化结果
            joint_set['position'] = {f_idx: optimized_positions[f_idx] for f_idx in current_features}

            # 绘制当前步骤的结果图
            self.plot_label_layout(self.features, self.labels, joint_set, output_dir, idx)

            # 记录第一帧和所有 joint set 的位置信息
            if frame_number == 0:
                for idx_feat in current_features:
                    first_frame_positions[idx_feat] = optimized_positions[idx_feat]

            all_joint_set_positions.append({
                'frame': frame_number,
                'positions': {idx_feat: optimized_positions[idx_feat] for idx_feat in current_features}
            })
        # print(all_joint_set_positions)

        # 2. (可选) 对所有标签进行一次最终的全局优化
        # 这一步可以用于进一步微调，但主要结果来自按 joint set 顺序优化的过程
        combined_positions = {}
        for idx, label in enumerate(self.labels):
            if label.position:
                 combined_positions[idx] = label.position
            else: # 如果有标签从未出现在任何 joint set 中（理论上不应发生）
                 feature = self.features[idx]
                 angle = random.uniform(0, 2 * math.pi)
                 label_dim = max(label.length, label.width)
                 radius = feature.radius + label_dim / 2 + 5
                 x = feature.position[0] + radius * math.cos(angle)
                 y = feature.position[1] + radius * math.sin(angle)
                 combined_positions[idx] = (x,y)
                 self.labels[idx].position = (x,y) # 确保所有标签都有位置

        full_set = {'set': list(range(len(self.labels)))}
        final_positions_indexed = self.simulated_annealing(combined_positions, full_set)

        # 更新最终位置到标签对象
        for idx, pos in final_positions_indexed.items():
             self.labels[idx].position = pos

        # 准备返回给 main.py 的结果 (使用标签 ID 作为 key)
        final_id_positions = {lbl.id: lbl.position for lbl in self.labels if lbl.position}

        return final_id_positions, all_joint_set_positions

    def check_feature_in_multiple_joint_sets(self, feature_idx):
        """检查特征是否出现在多个联合集中"""
        count = 0
        for joint_set in self.joint_sets:
            if feature_idx in joint_set['set']:
                count += 1
                if count > 1:
                    return True
        return False

    def plot_label_layout(self, all_features, all_labels, optimized_joint_set_info, output_dir, joint_set_idx):
        """绘制优化特定关节集后所有标签和特征的布局, 并标注ID, 标签用红色矩形表示"""
        fig, ax = plt.subplots(figsize=(12, 12))
        frame_number = optimized_joint_set_info['frame']
        optimized_indices = list(optimized_joint_set_info['set'])
        text_offset = 1.5

        # --- 绘制所有特征点 ---
        all_feature_pos = [(f.position[0], f.position[1]) for f in all_features]
        ax.scatter([p[0] for p in all_feature_pos], [p[1] for p in all_feature_pos],
                   color='gray', alpha=0.6, label='所有特征点', s=50)
        # 添加特征索引文本
        for i, pos in enumerate(all_feature_pos):
            ax.text(pos[0] + text_offset, pos[1] + text_offset, f'F{i}', fontsize=8, color='dimgray')

        # --- 绘制所有标签 (红色矩形框) ---
        current_label_positions = {}
        for i, label in enumerate(all_labels):
            if hasattr(label, 'position') and label.position:
                current_label_positions[i] = label.position

        if current_label_positions:
            # 移除之前的 scatter 调用
            # other_label_pos = {i: pos for i, pos in current_label_positions.items() if i not in optimized_indices}
            # opt_label_pos_dict = {i: pos for i, pos in current_label_positions.items() if i in optimized_indices}

            # if other_label_pos:
            #      ax.scatter([p[0] for p in other_label_pos.values()], [p[1] for p in other_label_pos.values()],
            #                  color='lightgray', alpha=0.7, label='其他标签 (当前)', s=30)
            
            # 改为绘制矩形
            for i, pos in current_label_positions.items():
                label_obj = all_labels[i]
                x, y = pos # 假设 pos 是左下角坐标
                width = label_obj.width
                height = label_obj.length
                # 所有标签都用红色矩形框
                lw = 1.5 if i in optimized_indices else 1.0 # 优化过的稍粗一点
                rect = patches.Rectangle((x, y), width, height, linewidth=lw, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                # 添加标签ID文本 (位置可能需要微调)
                text_color = 'blue' if i in optimized_indices else 'darkgray'
                font_weight = 'bold' if i in optimized_indices else 'normal'
                ax.text(x + text_offset, y + text_offset, f'L{label_obj.id}', fontsize=9, color=text_color, weight=font_weight)


        # --- 高亮显示当前优化的 Joint Set (只高亮特征点) ---
        opt_feature_pos = [all_features[i].position for i in optimized_indices]
        # valid_opt_label_pos_dict = {i: pos for i, pos in opt_label_pos_dict.items()}

        if opt_feature_pos:
            # 保持高亮特征点
            ax.scatter([p[0] for p in opt_feature_pos], [p[1] for p in opt_feature_pos],
                       color='red', label=f'优化的特征 (Set {joint_set_idx})', s=70, edgecolors='black')
            # 添加优化特征的索引文本 (保持)
            for i in optimized_indices:
                 pos = all_features[i].position
                 ax.text(pos[0] + text_offset, pos[1] + text_offset, f'F{i}', fontsize=9, color='red', weight='bold')

        # 移除高亮标签点的 scatter
        # if valid_opt_label_pos_dict:
        #     ax.scatter([p[0] for p in valid_opt_label_pos_dict.values()], [p[1] for p in valid_opt_label_pos_dict.values()],
        #                 color='blue', label=f'优化的标签 (Set {joint_set_idx})', s=50, edgecolors='black')
        #     # 添加优化标签的ID文本 (已合并到矩形绘制部分)
        #     # for i, pos in valid_opt_label_pos_dict.items():
        #     #     ax.text(pos[0] + text_offset, pos[1] + text_offset, f'L{all_labels[i].id}', fontsize=9, color='blue', weight='bold')

        # --- 绘制 Leader Lines (保持) ---
        for i, label_pos in current_label_positions.items():
            feature_pos = all_features[i].position
            line_color = 'black' if i in optimized_indices else 'grey'
            line_style = '-' if i in optimized_indices else ':'
            line_width = 1.0 if i in optimized_indices else 0.5
            ax.plot([feature_pos[0], label_pos[0]], [feature_pos[1], label_pos[1]],
                    color=line_color, linestyle=line_style, linewidth=line_width)

        # --- 图表设置 (使用 ax 对象) ---
        ax.set_title(f"优化 Joint Set {joint_set_idx} (Frame {frame_number}) 后的布局 - 含ID")
        ax.set_xlabel("X 坐标")
        ax.set_ylabel("Y 坐标")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small') # 图例置于图外
        ax.grid(True, linestyle='--', alpha=0.5)
        # Determine plot limits dynamically or use fixed ones if preferred
        ax.set_xlim(0, self.max_x)
        ax.set_ylim(0, self.max_y)
        ax.set_aspect('equal', adjustable='box') # 替换 plt.axis('equal')
        fig.tight_layout(rect=[0, 0, 0.85, 1]) # 调整布局为图例留出空间

        # --- 保存图像 ---
        fig.savefig(f"{output_dir}/all_layout_after_joint_set_{joint_set_idx}_frame_{frame_number}_with_ids_rects.png", bbox_inches='tight') # 更新文件名
        plt.close(fig) # 关闭 figure 对象
