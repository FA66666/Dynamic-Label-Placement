import math
import random

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

    def detect_joint_sets(self):
        """检测空间-时间交集区域并排序（修改后）"""
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
                joint_sets.append({'set': current_set, 'frame': t})

        # 按集合大小降序排序
        self.joint_sets = sorted(joint_sets, key=lambda x: -len(x['set']))

    def calculate_static_energy(self, label_positions, joint_set):
        """静态能量计算（包含约束项，修改后）"""
        E_overlap = 0
        E_position = 0
        E_aesthetics = 0
        E_constraint = 0

        # 计算标签间重叠
        for i in range(len(self.labels)):
            for j in range(len(self.labels)):
                if i == j:
                    continue
                E_overlap += self.calculate_rectangle_overlap(i, j, label_positions)

        # 计算标签与特征的重叠
        for i in range(len(self.labels)):
            E_overlap += self.calculate_rectangle_circle_overlap(i, label_positions)

        # 计算位置能量
        for i, pos in enumerate(label_positions):
            dx = pos[0] - self.features[i].position[0]
            dy = pos[1] - self.features[i].position[1]
            r, theta = self.cartesian_to_polar((dx, dy))
            quadrant = int(theta // 90) % 4
            E_position += self.params['Worient'][quadrant] * r
            E_position += self.params['Wdistance'] * r

        # 计算美学能量
        leader_intersections = self.calculate_leader_intersections(label_positions)
        out_of_axes_area = self.check_out_of_axes(label_positions)
        E_aesthetics = self.params['Wout-of-axes'] * out_of_axes_area
        E_aesthetics += self.params['Wintersect'] * leader_intersections

        # 计算约束能量
        common_features = set(joint_set['set']).intersection(self.constraints.keys())  # 使用 joint_set['set']
        for idx in common_features:
            prev_r, prev_theta = self.constraints[idx]
            dx = label_positions[idx][0] - self.features[idx].position[0]
            dy = label_positions[idx][1] - self.features[idx].position[1]
            current_r = math.hypot(dx, dy)
            current_theta = math.atan2(dy, dx)
            E_constraint += self.params['Wradius'] * abs(current_r - prev_r)
            E_constraint += self.params['Wangle'] * abs(current_theta - prev_theta)

        return E_overlap + E_position + E_aesthetics + E_constraint

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
        # 这里简化为返回圆形与矩形的最小包围圆重叠面积
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
                # 假设leader线是直线连接标签中心到特征中心
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
        # 使用分离轴定理实现线段相交检测
        # 这里简化为返回True/False
        # 实际应用中需实现精确计算
        return True

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
                clipped_left * label_height +  # 左侧超出面积
                clipped_right * label_height +  # 右侧超出面积
                clipped_top * label_width +  # 上侧超出面积
                clipped_bottom * label_width  # 下侧超出面积
            )

        return total_area

    def simulated_annealing(self, initial_positions, joint_set, max_iter=1000):
        """模拟退火优化（修改后，使用约束项）"""
        current_pos = initial_positions.copy()
        best_pos = current_pos.copy()
        current_energy = self.calculate_static_energy(current_pos, joint_set)
        best_energy = current_energy
        temp = 1000.0  # 初始温度

        for _ in range(max_iter):
            # 生成新解（扰动幅度与温度相关）
            new_pos = [
                (x + random.uniform(-temp / 100, temp / 100),
                 y + random.uniform(-temp / 100, temp / 100))
                for x, y in current_pos
            ]

            # 计算新解能量
            new_energy = self.calculate_static_energy(new_pos, joint_set)
            delta = new_energy - current_energy

            # 接受准则
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_pos = new_pos
                current_energy = new_energy
                if new_energy < best_energy:
                    best_pos = new_pos
                    best_energy = new_energy

            # 温度衰减（线性退火）
            temp *= 0.99

        return best_pos

    def optimize(self):
        """全局静态优化流程（修改后）"""
        # 1. 检测交集区域
        self.detect_joint_sets()

        # 2. 按复杂度降序优化每个交集区域
        for joint_set in self.joint_sets:
            label_indices = list(joint_set['set'])  # 获取交点集合
            initial_positions = []

            # 初始化候选位置（基于特征位置）
            for idx in label_indices:
                feature = self.features[idx]
                angle = random.uniform(0, 2 * math.pi)
                radius = feature.radius + self.labels[idx].length / 2
                x = feature.position[0] + radius * math.cos(angle)
                y = feature.position[1] + radius * math.sin(angle)
                initial_positions.append((x, y))

            # 执行模拟退火优化
            optimized_positions = self.simulated_annealing(initial_positions, joint_set)

            # 存储约束条件（极坐标形式）
            for i, idx in enumerate(label_indices):
                dx = optimized_positions[i][0] - self.features[idx].position[0]
                dy = optimized_positions[i][1] - self.features[idx].position[1]
                r = math.hypot(dx, dy)
                theta = math.atan2(dy, dx)
                self.constraints[idx] = (r, theta)

            # 更新标签位置
            for i, idx in enumerate(label_indices):
                self.labels[idx].position = optimized_positions[i]

        return self.labels

def cartesian_to_polar(position):
    """
    将笛卡尔坐标转换为极坐标
    :param position: (x, y) 坐标
    :return: (r, theta) 极坐标中的半径和角度
    """
    x, y = position
    r = math.sqrt(x ** 2 + y ** 2)  # 半径
    theta = math.atan2(y, x)  # 角度
    return r, theta
