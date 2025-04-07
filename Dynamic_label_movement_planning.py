import math

# 参数设置（严格遵循论文附录A2）
paramsA2 = {
    'wlabel-collision': 50,  # 标签间碰撞力权重（取40-60中间值）
    'Dlabel-collision': 30,  # 标签碰撞距离阈值
    'wfeature-collision': 75,  # 标签-特征碰撞力权重（取50-100中间值）
    'Dfeature-collision': 17,  # 标签-特征碰撞距离阈值
    'wpull': 25,  # 拉力权重
    'Dpull': 18,  # 拉力作用距离
    'c_friction': 0.7,  # 摩擦系数
    'Wtime': 15,  # 时间约束力权重
    'Wspace': 20,  # 空间约束力权重（新增，假设值）
    'Dspace': 100,  # 空间约束力作用距离
    'delta_t': 5,  # 特征未来预测时间间隔
}

class DynamicLabelOptimizer:
    def __init__(self, labels, features, params, constraints=None, max_x=1000, max_y=1000):
        self.labels = labels
        self.features = features
        self.params = params
        self.constraints = constraints or {}  # 静态优化的约束位置
        self.max_x = max_x
        self.max_y = max_y

    def compute_label_label_repulsion(self, i, j, label_positions):
        """计算标签之间的排斥力（论文公式3.2.1）"""
        if i not in label_positions or j not in label_positions:
            return (0.0, 0.0)  # 如果标签ID没有在字典中，跳过

        x1, y1 = label_positions[i]
        x2, y2 = label_positions[j]
        s_i = self.labels[i].length
        s_j = self.labels[j].length
        distance = math.hypot(x1 - x2, y1 - y2)
        d = max(distance - 0.5 * (s_i + s_j), 0)

        if distance >= self.params['Dlabel-collision']:
            return (0.0, 0.0)

        magnitude = self.params['wlabel-collision'] * min(d / self.params['Dlabel-collision'] - 1, 0)
        nx = (x1 - x2) / distance if distance != 0 else 0
        ny = (y1 - y2) / distance if distance != 0 else 0
        return (magnitude * nx, magnitude * ny)

    def compute_label_feature_repulsion(self, i, label_positions):
        """计算标签与特征的排斥力（论文公式3.2.1）"""
        if i not in label_positions:
            return (0.0, 0.0)  # 如果标签ID没有在字典中，跳过

        label_x, label_y = label_positions[i]
        feature_x, feature_y = self.features[i].position
        s_i = self.labels[i].length
        r_j = self.features[i].radius
        distance = math.hypot(label_x - feature_x, label_y - feature_y)
        d = max(distance - 0.5 * (s_i + r_j), 0)

        if distance >= self.params['Dfeature-collision']:
            return (0.0, 0.0)

        magnitude = self.params['wfeature-collision'] * min(d / self.params['Dfeature-collision'] - 1, 0)
        nx = (label_x - feature_x) / distance if distance != 0 else 0
        ny = (label_y - feature_y) / distance if distance != 0 else 0
        return (magnitude * nx, magnitude * ny)

    def compute_pulling_force(self, i, label_positions):
        """计算标签拉力（论文公式3.2.1）"""
        if i not in label_positions:
            return (0.0, 0.0)  # 如果标签ID没有在字典中，跳过

        label_x, label_y = label_positions[i]
        feature_x, feature_y = self.features[i].position
        s_i = self.labels[i].length
        r_i = self.features[i].radius
        distance = math.hypot(label_x - feature_x, label_y - feature_y)
        effective_distance = distance - 0.5 * (s_i + r_i)

        if effective_distance <= self.params['Dpull']:
            return (0.0, 0.0)

        magnitude = self.params['wpull'] * math.log(effective_distance - self.params['Dpull'] + 1)
        nx = (feature_x - label_x) / distance if distance != 0 else 0
        ny = (feature_y - label_y) / distance if distance != 0 else 0
        return (magnitude * nx, magnitude * ny)

    def compute_friction(self, i, velocities):
        """计算摩擦力（论文公式3.2.1）"""
        v_label_x, v_label_y = velocities[i]
        v_feature_x, v_feature_y = self.features[i].velocity
        delta_vx = v_label_x - v_feature_x
        delta_vy = v_label_y - v_feature_y
        return (
            -self.params['c_friction'] * delta_vx,
            -self.params['c_friction'] * delta_vy
        )

    def compute_time_constraint(self, i, label_positions):
        """
        计算时间约束力（论文公式3.2.2）
        要求：基于未来特征位置计算标签与未来特征的接近程度。
        如果特征对象具有轨迹（trajectory），则使用轨迹中的未来位置。
        """
        total_fx = 0.0
        total_fy = 0.0

        # 当前标签位置
        label_x, label_y = label_positions[i]

        # 获取当前特征的速度，计算未来的位置
        feature = self.features[i]
        trajectory = feature.trajectory  # 特征的轨迹数组

        # 获取未来特征位置（假设 delta_t 是时间步长）
        future_frame = min(len(trajectory) - 1, int(self.params['delta_t']))  # 计算未来的帧数
        future_position = trajectory[future_frame]

        # 计算标签与未来特征位置的距离
        dx_future = label_x - future_position[0]
        dy_future = label_y - future_position[1]
        distance_future = math.hypot(dx_future, dy_future)

        # 计算标签与特征当前时间的距离
        feature_x, feature_y = feature.position
        dx_current = label_x - feature_x
        dy_current = label_y - feature_y
        distance_current = math.hypot(dx_current, dy_current)

        # 目标：如果标签与未来位置的距离较大，则施加拉力将标签向未来位置拉近
        # 定义差值：delta = distance_future - distance_current
        delta = distance_future - distance_current

        # 计算速度比和距离比
        speed_ratio = max(math.hypot(feature.velocity[0], feature.velocity[1]), 1)  # 保证不除零
        distance_ratio = distance_future / (distance_current + 1e-6)  # 防止除零

        # 当未来距离较小时，施加排斥力
        if distance_future < self.params['Dfeature-collision']:  # 可调整阈值
            magnitude = self.params['Wtime'] * math.log(speed_ratio + 1)
            nx = dx_future / distance_future if distance_future != 0 else 0
            ny = dy_future / distance_future if distance_future != 0 else 0

            # 如果 delta > 0，施加正向力，拉近标签；如果 delta < 0，施加反向力，避免过早靠近
            if delta > 0:
                total_fx += magnitude * nx
                total_fy += magnitude * ny
            else:
                total_fx += -magnitude * nx
                total_fy += -magnitude * ny

        return (total_fx, total_fy)

    def compute_space_constraint(self, i, label_positions):
        """计算空间约束力（论文公式3.2.2），拉向约束位置"""
        if i not in label_positions:
            return (0.0, 0.0)  # 如果标签ID没有在字典中，跳过

        if i not in self.constraints:
            return (0.0, 0.0)

        constraint_x, constraint_y = self.constraints[i]
        dx = label_positions[i][0] - constraint_x
        dy = label_positions[i][1] - constraint_y
        distance = math.hypot(dx, dy)

        if distance < self.params['Dspace']:
            magnitude = self.params['Wspace'] * math.log(distance + 1)
            nx = dx / distance if distance != 0 else 0
            ny = dy / distance if distance != 0 else 0
            return (-magnitude * nx, -magnitude * ny)  # 拉向约束位置
        return (0.0, 0.0)

    def compute_resultant_force(self, i, label_positions, velocities):
        """计算所有力的合力（论文公式3.2.3）"""
        total_fx = 0.0
        total_fy = 0.0

        # 标签-标签排斥力
        for j in range(len(self.labels)):
            if i != j:
                fx, fy = self.compute_label_label_repulsion(i, j, label_positions)
                total_fx += fx
                total_fy += fy

        # 标签-特征排斥力
        fx, fy = self.compute_label_feature_repulsion(i, label_positions)
        total_fx += fx
        total_fy += fy

        # 拉力
        fx, fy = self.compute_pulling_force(i, label_positions)
        total_fx += fx
        total_fy += fy

        # 摩擦力
        fx, fy = self.compute_friction(i, velocities)
        total_fx += fx
        total_fy += fy

        # 时间约束力
        fx, fy = self.compute_time_constraint(i, label_positions)
        total_fx += fx
        total_fy += fy

        # 空间约束力
        fx, fy = self.compute_space_constraint(i, label_positions)
        total_fx += fx
        total_fy += fy

        return (total_fx, total_fy)

    def update_positions(self, label_positions, velocities, time_delta):
        """更新标签位置（论文力计算流程）"""
        new_positions = {}
        new_velocities = {}

        for i in range(len(self.labels)):
            # 获取当前标签的位置和速度
            if i not in label_positions or i not in velocities:
                continue  # 跳过不在字典中的标签

            # 计算合力
            force = self.compute_resultant_force(i, label_positions, velocities)
            acceleration_x, acceleration_y = force  # 假设质量为1

            # 欧拉积分更新速度
            new_vx = velocities[i][0] + acceleration_x * time_delta
            new_vy = velocities[i][1] + acceleration_y * time_delta

            # 更新位置
            new_x = label_positions[i][0] + velocities[i][0] * time_delta
            new_y = label_positions[i][1] + velocities[i][1] * time_delta

            # 边界约束
            new_x = max(0, min(self.max_x, new_x))
            new_y = max(0, min(self.max_y, new_y))

            # 更新字典中的位置和速度
            new_positions[i] = (new_x, new_y)
            new_velocities[i] = (new_vx, new_vy)

        return new_positions, new_velocities

    def optimize_labels(self, initial_positions, initial_velocities, time_delta, max_iter=1000):
        """力导向优化过程（论文3.2.3流程）"""
        current_positions = initial_positions.copy()  # 字典格式
        current_velocities = initial_velocities.copy()  # 字典格式

        # print(current_positions)
        for _ in range(max_iter):
            # 使用字典格式的当前位置和速度
            new_positions, new_velocities = self.update_positions(
                current_positions, current_velocities, time_delta
            )
            current_positions = new_positions
            current_velocities = new_velocities

        return current_positions, current_velocities
