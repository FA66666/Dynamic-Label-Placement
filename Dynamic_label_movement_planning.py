import math

# 参数设置（严格遵循论文附录A2）
paramsA2 = {
    'wlabel-collision': 50,  # 标签间碰撞力权重（取40-60中间值）
    'Dlabel-collision': 30,  # 标签碰撞距离阈值
    'wfeature-collision': 70,  # 标签-特征碰撞力权重（取50-100中间值）
    'Dfeature-collision': 17,  # 标签-特征碰撞距离阈值
    'wpull': 25,  # 拉力权重
    'Dpull': 18,  # 拉力作用距离
    'c_friction': 0.7,  # 摩擦系数
    'Wtime': 15,  # 时间约束力权重
    'Wspace': 20,  # 空间约束力权重
    'Dspace': 100,  # 空间约束力作用距离
    'delta_t': 0.1,  # 特征未来预测时间间隔
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
        s_i = math.hypot(self.labels[i].width, self.labels[i].length) / 2
        s_j = math.hypot(self.labels[j].width, self.labels[j].length) / 2
        distance = math.hypot(x1 - x2, y1 - y2)
        d = distance - 0.5 * (s_i + s_j)

        # # if d >= 0 or distance >= self.params['Dlabel-collision']:
        if d >= 0 :
            return (0.0, 0.0)
        magnitude = self.params['wlabel-collision'] * min(d / self.params['Dlabel-collision'] - 1, 0)
        nx = (x1 - x2) /(distance+1e-6)
        ny = (y1 - y2) /(distance+1e-6)
        return (-magnitude * nx, -magnitude * ny)

    def compute_label_feature_repulsion(self, i, label_positions):
        """计算标签与特征的排斥力（论文公式3.2.1）"""
        if i not in label_positions:
            return (0.0, 0.0)  # 如果标签ID没有在字典中，跳过

        label_x, label_y = label_positions[i]
        feature_x, feature_y = self.features[i].position
        s_i = math.hypot(self.labels[i].width, self.labels[i].length) / 2
        r_j = self.features[i].radius
        distance = math.hypot(label_x - feature_x, label_y - feature_y)
        d = distance - 0.5 * (s_i + r_j)

        magnitude = self.params['wfeature-collision'] * min(d / self.params['Dfeature-collision'] - 1, 0)
        nx = (label_x - feature_x) /(distance+1e-4)
        ny = (label_y - feature_y) /(distance+1e-4)
        return (magnitude * nx, magnitude * ny)

    def compute_pulling_force(self, i, label_positions):
        """计算标签拉力（论文公式3.2.1）"""
        if i not in label_positions:
            return (0.0, 0.0)  # 如果标签ID没有在字典中，跳过

        label_x, label_y = label_positions[i]
        feature_x, feature_y = self.features[i].position
        s_i = math.hypot(self.labels[i].width, self.labels[i].length) / 2
        r_i = self.features[i].radius
        distance = math.hypot(label_x - feature_x, label_y - feature_y)
        effective_distance = math.fabs(distance - 0.5 * (s_i + r_i))

        if effective_distance <= self.params['Dpull']:
            # print("effective_distance",effective_distance)
            return (0.0, 0.0)


        magnitude = self.params['wpull'] * math.log(effective_distance - self.params['Dpull'] + 1)
        nx = (feature_x - label_x) /(distance+1e-4)
        ny = (feature_y - label_y) /(distance+1e-4)
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

    def compute_time_constraint(self, i, j, label_positions):
        if i not in label_positions:
            return (0.0, 0.0)
        
        # 标签位置
        label_x, label_y = label_positions[i]
        
        # 当前特征
        feature_i = self.features[i]
        feature_j = self.features[j] if j < len(self.features) else None
        
        if feature_j is None:
            return (0.0, 0.0)
        
        # 计算未来位置
        v_i_mag = math.hypot(feature_i.velocity[0], feature_i.velocity[1])
        v_j_mag = math.hypot(feature_j.velocity[0], feature_j.velocity[1])
        
        # 按论文计算未来位置
        t_j_future_x = feature_j.position[0] + (feature_j.velocity[0] - feature_i.velocity[0]) * self.params['delta_t']
        t_j_future_y = feature_j.position[1] + (feature_j.velocity[1] - feature_i.velocity[1]) * self.params['delta_t']
        
        # 计算到未来位置的距离
        dx = label_x - t_j_future_x
        dy = label_y - t_j_future_y
        distance = math.hypot(dx, dy)
             
        # 按论文公式计算
        # 确保速度比率至少为1，避免对数域错误
        if v_i_mag < 1e-5 and v_j_mag < 1e-5:
            return (0.0, 0.0)  # 如果两个特征速度都接近零，不施加时间约束力
            
        velocity_ratio = max(v_i_mag, v_j_mag) / (min(v_i_mag, v_j_mag) + 1e-6)  # 防止除零
        
        # 确保速度比率至少为1，避免负对数
        velocity_ratio = max(1.0, velocity_ratio)
        
        # 计算力的大小
        magnitude = self.params['Wtime'] * math.log(velocity_ratio) * min(distance / self.params['Dfeature-collision'] - 1, 0)
        
        # 计算方向
        nx = dx / distance
        ny = dy / distance
        
        return (magnitude * nx, magnitude * ny)

    def compute_space_constraint(self, i, label_positions):
        """计算空间约束力（论文公式3.2.2），拉向约束位置"""
        if i not in label_positions:
            return (0.0, 0.0)
        
        # 获取当前标签位置
        current_x, current_y = label_positions[i]
        
        # 如果没有约束位置，返回零力
        if i not in self.constraints:
            return (0.0, 0.0)
        
        # 获取约束位置（即将到来的特征关节集中的位置）
        constraint_x, constraint_y = self.constraints[i]
        
        # 计算距离
        dx = current_x - constraint_x
        dy = current_y - constraint_y
        distance = math.hypot(dx, dy)
        

        
        # 按论文公式直接计算，不加额外条件
        magnitude = self.params['Wspace'] * math.log(distance + 1)
        nx = dx / distance
        ny = dy / distance
        
        return (-magnitude * nx, -magnitude * ny)  # 拉向约束位置

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
        fx, fy = self.compute_time_constraint(i, i, label_positions)
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

            print("new_vx",velocities[i][0] * time_delta)
            print("new_vy",velocities[i][1] * time_delta)
            # 更新位置

            new_x = label_positions[i][0] + velocities[i][0] * time_delta
            new_y = label_positions[i][1] + velocities[i][1] * time_delta

            # 边界约束
            new_x = max(0, min(self.max_x, new_x))
            new_y = max(0, min(self.max_y, new_y))

            # 更新字典中的位置和速度
            new_positions[i] = (new_x, new_y)
            new_velocities[i] = (new_vx, new_vy)
        # print(new_velocities)
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


