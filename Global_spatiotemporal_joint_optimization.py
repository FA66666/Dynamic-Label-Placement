import math
import random
import matplotlib.pyplot as plt
import os


# 参数设置（参考论文附录A1）
params = {
    'Wlabel-label': 80,
    'Wlabel-feature': 50,
    'Worient': [1, 2, 3, 4],  # 四个象限的权重
    'Wdistance': 20,
    'Wout-of-axes': 320,
    'Wintersect': 1,  # leader线交叉惩罚权重
    'Wradius': 20,
    'Wangle': 10
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
        x, y = label_positions[i]
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

    def calculate_leader_intersections(self, label_positions):
        """计算leader线交叉次数"""
        intersections = 0
        for i in range(len(self.labels)):
            for j in range(i + 1, len(self.labels)):
                # 计算leader线是否交叉
                line1 = (
                    label_positions[i],
                    self.features[i].position
                )
                line2 = (
                    label_positions[j],
                    self.features[j].position
                )
                if self.lines_intersect(line1, line2):
                    intersections += 1
        return intersections

    def lines_intersect(self, line1, line2):
        """判断两条线段是否相交"""

        def ccw(A, B, C):
            """计算三点的顺时针或逆时针关系"""
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        A, B = line1  # line1: (x1, y1) -> (x2, y2)
        C, D = line2  # line2: (x3, y3) -> (x4, y4)

        # 判断线段 AB 和 CD 是否相交
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

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

    def simulated_annealing(self, initial_positions, joint_set, max_iter=1000):
        """模拟退火优化"""
        current_features = list(joint_set['set'])
        num_labels = len(current_features)

        current_pos = initial_positions.copy()
        best_pos = current_pos.copy()
        current_energy = self.calculate_static_energy(current_pos, joint_set)
        best_energy = current_energy
        temp = 1000.0

        for _ in range(max_iter):
            # 生成新解（仅扰动当前联合集中的标签）
            new_pos = [
                (x + random.uniform(-temp / 100, temp / 100),
                 y + random.uniform(-temp / 100, temp / 100))
                for x, y in current_pos
            ]

            new_energy = self.calculate_static_energy(new_pos, joint_set)
            delta = new_energy - current_energy

            # 接受准则
            if delta < 0 or math.exp(-delta / temp) > random.random():
                current_pos = new_pos
                current_energy = new_energy
                if new_energy < best_energy:
                    best_pos = new_pos
                    best_energy = new_energy

            temp *= 0.99

        return best_pos

    def calculate_static_energy(self, label_positions, joint_set):
        """静态能量计算"""
        E_static = 0
        E_constraint = 0

        # 获取当前联合集中的全局特征索引
        current_features = list(joint_set['set'])

        # 计算静态能量项（E_static）
        for i in range(len(current_features)):
            for j in range(len(current_features)):
                if i == j:
                    continue
                global_i = current_features[i]
                global_j = current_features[j]

                overlap_area = self.calculate_rectangle_overlap(global_i, global_j, label_positions)
                E_static += self.params['Wlabel-label'] * overlap_area

                label_i_pos = label_positions[i]
                label_j_pos = label_positions[j]
                E_static += self.calculate_rectangle_circle_overlap(global_i, label_i_pos) * self.params[
                    'Wlabel-feature']
                E_static += self.calculate_rectangle_circle_overlap(global_j, label_j_pos) * self.params[
                    'Wlabel-feature']

        # 计算位置能量项（E_position）
        for i in range(len(current_features)):
            feature = self.features[current_features[i]]
            label_pos = label_positions[i]

            dx = label_pos[0] - feature.position[0]
            dy = label_pos[1] - feature.position[1]
            r, theta = self.cartesian_to_polar((dx, dy))

            E_position = self.params['Worient'] * self.calculate_angle_delta(theta) + self.params['Wdistance'] * r
            E_static += E_position

        # 计算美学能量项（E_aesthetics）
        for i in range(len(current_features)):
            X = self.check_out_of_axes([label_positions[i]])
            I = self.calculate_leader_intersections(label_positions)
            E_aesthetics = self.params['Wout-of-axes'] * X + self.params['Wintersect'] * I
            E_static += E_aesthetics

        # 计算约束能量项（E_constraint）
        common_features = set(current_features).intersection(self.constraints.keys())
        for idx in common_features:
            current_pos = label_positions[current_features.index(idx)]
            dx = current_pos[0] - self.features[idx].position[0]
            dy = current_pos[1] - self.features[idx].position[1]
            r_p, theta_p = self.cartesian_to_polar((dx, dy))

            r_l, theta_l = self.constraints[idx]

            E_constraint += (
                    self.params['Wradius'] * abs(r_p - r_l) +
                    self.params['Wangle'] * abs(theta_p - theta_l)
            )

        return E_static + E_constraint

    def detect_joint_sets(self):
        """检测空间-时间交集区域并排序"""
        joint_sets = []
        max_frames = len(self.features[0].trajectory) if self.features else 0

        for t in range(max_frames):
            current_positions = [f.trajectory[t] for f in self.features]
            current_set = set()
            for i in range(len(self.features)):
                for j in range(i + 1, len(self.features)):
                    dx = current_positions[i][0] - current_positions[j][0]
                    dy = current_positions[i][1] - current_positions[j][1]
                    distance = math.hypot(dx, dy)
                    if distance < 1.0:
                        current_set.update({i, j})
            if current_set:
                # 将位置保存为联合集的一部分
                joint_sets.append({'set': current_set, 'frame': t, 'position': None})

        # 按集合大小降序排序
        self.joint_sets = sorted(joint_sets, key=lambda x: -len(x['set']))


    def optimize(self):
        """全局静态优化流程"""
        self.detect_joint_sets()

        # 创建存放图像的文件夹
        output_dir = "optimized_labels"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for idx, joint_set in enumerate(self.joint_sets):
            current_features = list(joint_set['set'])
            frame_number = joint_set['frame']

            # 初始标签位置
            initial_positions = []
            for feature_idx in current_features:
                feature = self.features[feature_idx]
                angle = random.uniform(0, 2 * math.pi)
                radius = feature.radius + self.labels[feature_idx].length / 2
                x = feature.position[0] + radius * math.cos(angle)
                y = feature.position[1] + radius * math.sin(angle)
                initial_positions.append((x, y))

            # 使用模拟退火进行优化
            optimized_positions = self.simulated_annealing(initial_positions, joint_set)

            # 存储约束条件（极坐标形式），并传递给下一个联合集
            for i, idx in enumerate(current_features):
                pos = optimized_positions[i]
                dx = pos[0] - self.features[idx].position[0]
                dy = pos[1] - self.features[idx].position[1]
                r, theta = self.cartesian_to_polar((dx, dy))
                self.constraints[idx] = (r, theta)

            # 更新联合集的位置
            joint_set['position'] = optimized_positions

            # 更新标签位置
            for i, idx in enumerate(current_features):
                self.labels[idx].position = optimized_positions[i]

            # 绘制并保存标签布局图像
            self.plot_label_layout(frame_number, joint_set, optimized_positions, output_dir, idx)

        return self.labels

    def plot_label_layout(self, frame_number, joint_set, optimized_positions, output_dir, joint_set_idx):
        """绘制标签布局并保存图像"""
        plt.figure(figsize=(10, 10))

        # 获取当前帧上的特征位置
        feature_positions = [self.features[i].position for i in joint_set['set']]

        # 绘制特征点
        feature_x = [pos[0] for pos in feature_positions]
        feature_y = [pos[1] for pos in feature_positions]
        plt.scatter(feature_x, feature_y, color='red', label='Features')

        # 绘制标签点
        label_x = [pos[0] for pos in optimized_positions]
        label_y = [pos[1] for pos in optimized_positions]
        plt.scatter(label_x, label_y, color='blue', label='Labels')

        # 绘制标签与特征之间的连线（leader线）
        for i, feature_idx in enumerate(joint_set['set']):
            feature_pos = self.features[feature_idx].position
            plt.plot([feature_pos[0], label_x[i]], [feature_pos[1], label_y[i]], 'gray', linestyle='dotted')

        # 设置图形的标题和标签
        plt.title(f"Label Placement for Frame {frame_number}, Joint Set {joint_set_idx}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")

        # 添加图例
        plt.legend()

        # 保存图像
        plt.savefig(f"{output_dir}/frame_{frame_number}_joint_set_{joint_set_idx}.png")
        plt.close()

