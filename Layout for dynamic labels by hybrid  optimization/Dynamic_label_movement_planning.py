import math

 
paramsA2 = {
    'wlabel-collision': 50,   
    'Dlabel-collision': 30,   
    'wfeature-collision': 75,   
    'Dfeature-collision': 17,   
    'wpull': 25,   
    'Dpull': 18,   
    'wfriction': 6,   
    'c_friction': 0.7,   
    'Wtime': 15,   
    'Wspace': 20,   
    'Dspace': 100,   
    'delta_t': 0.03,   
}

class DynamicLabelOptimizer:
    def __init__(self, labels, features, params, constraints=None, max_x=1000, max_y=1000):
        self.labels = labels
        self.features = features
        self.params = params
        self.constraints = constraints or {}   
        self.max_x = max_x
        self.max_y = max_y

    # 标签-标签排斥力
    def compute_label_label_repulsion(self, i, j, label_positions):
        x1, y1 = label_positions[i]
        x2, y2 = label_positions[j]
        
        ei_x = self.labels[i].length / 2   
        ei_y = self.labels[i].width / 2    
        ej_x = self.labels[j].length / 2
        ej_y = self.labels[j].width / 2
        dx = abs(x1 - x2) - 0.5 * (ei_x + ej_x)
        dy = abs(y1 - y2) - 0.5 * (ei_y + ej_y)
        d_label = max(dx, dy)   
               
        v = min(d_label / self.params['Dlabel-collision'] - 1, 0)
        magnitude = self.params['wlabel-collision'] * v

        d_vec_x = x1 - x2
        d_vec_y = y1 - y2
        d_norm = math.hypot(d_vec_x, d_vec_y)+1e-6
        nx = d_vec_x / d_norm
        ny = d_vec_y / d_norm
        return (magnitude * nx, magnitude * ny)

    # 要素-标签排斥力    
    def compute_label_feature_repulsion(self, i, label_positions):
        total_fx = 0.0
        total_fy = 0.0
        label_x, label_y = label_positions[i]
        
        si_x = self.labels[i].length / 2
        si_y = self.labels[i].width / 2
        
        for j in range(len(self.features)):
            if i == j:
                continue
            feature_x, feature_y = self.features[j].position
            rj = self.features[j].radius   
            
 
            dx = abs(label_x - feature_x) - 0.5 * (si_x + rj)
            dy = abs(label_y - feature_y) - 0.5 * (si_y + rj)
            d_label_feature = max(dx, dy)   
            
            v = min(d_label_feature / self.params['Dfeature-collision'] - 1, 0)
            magnitude = self.params['wfeature-collision'] * v           

            d_vec_x = label_x - feature_x
            d_vec_y = label_y - feature_y
            d_norm = math.hypot(d_vec_x, d_vec_y)+1e-6
                       
            nx = d_vec_x / d_norm
            ny = d_vec_y / d_norm
            total_fx += magnitude * nx
            total_fy += magnitude * ny
            
        return (total_fx, total_fy)

    # 拉力
    def compute_pulling_force(self, i, label_positions):
        if i not in label_positions:
            return (0.0, 0.0)
        label_x, label_y = label_positions[i]
        feature_x, feature_y = self.features[i].position
    
        si_x = self.labels[i].length / 2
        si_y = self.labels[i].width / 2
        ri = self.features[i].radius
        
 
        dx = abs(label_x - feature_x) - 0.5 * (si_x + ri)
        dy = abs(label_y - feature_y) - 0.5 * (si_y + ri)
        d_effective = max(dx, dy)   
        
        expr = d_effective - self.params['Dpull']
        if expr > 0:
            magnitude = self.params['wpull'] * math.log(expr + 1)
            
            # Direction from label to feature (pulling toward feature)
            d_vec_x = feature_x - label_x
            d_vec_y = feature_y - label_y
            d_norm = math.hypot(d_vec_x, d_vec_y)
            
            if d_norm > 1e-6:
                # Normalize direction vector
                nx = d_vec_x / d_norm
                ny = d_vec_y / d_norm
                return (magnitude * nx, magnitude * ny)
        return (0.0, 0.0)

    # 摩擦力
    def compute_friction(self, i, velocities):
        v_label_x, v_label_y = velocities[i]
        v_feature_x, v_feature_y = self.features[i].velocity
        delta_vx = v_label_x - v_feature_x
        delta_vy = v_label_y - v_feature_y
        return (
            -self.params['wfriction'] * self.params['c_friction'] * delta_vx,
            -self.params['wfriction'] * self.params['c_friction'] * delta_vy
        )

    # 时间约束力
    def compute_time_constraint(self, i, j, label_positions):
        if i not in label_positions or j not in label_positions or i == j:
            return (0.0, 0.0)

        # 标签i和j的当前位置
        label_i_x, label_i_y = label_positions[i]
        label_j_x, label_j_y = label_positions[j]

        # 获取对应的特征点
        feature_i = self.features[i]
        feature_j = self.features[j] if j < len(self.features) else None    
        if feature_j is None:
            return (0.0, 0.0)
 
        # 计算特征点的速度大小
        v_i_mag = math.hypot(feature_i.velocity[0], feature_i.velocity[1])
        v_j_mag = math.hypot(feature_j.velocity[0], feature_j.velocity[1])
        
        # 避免除零错误
        if v_i_mag < 1e-6 and v_j_mag < 1e-6:
            return (0.0, 0.0)
        
        # 计算速度比（按论文公式）
        max_velocity = max(v_i_mag, v_j_mag)
        min_velocity = min(v_i_mag, v_j_mag) + 1e-6
        velocity_ratio_log = math.log(max_velocity / min_velocity)
        
        # 计算标签j的未来位置（假设标签跟随特征点运动）
        l_j_future_x = label_j_x + feature_j.velocity[0] * self.params['delta_t']
        l_j_future_y = label_j_y + feature_j.velocity[1] * self.params['delta_t']

        # 计算距离比
        current_distance = math.hypot(label_i_x - label_j_x, label_i_y - label_j_y)
        future_distance = math.hypot(label_i_x - l_j_future_x, label_i_y - l_j_future_y)

        if current_distance < 1e-6:
            return (0.0, 0.0)
            
        distance_ratio = current_distance / (future_distance + 1e-6)

        # 计算力的大小（按论文公式）
        magnitude = self.params['Wtime'] * velocity_ratio_log * min(distance_ratio - 1, 0)

        # 计算力的方向：从标签i指向标签j的未来位置
        dx = l_j_future_x - label_i_x
        dy = l_j_future_y - label_i_y
        distance = math.hypot(dx, dy)
        
        if distance < 1e-6:
            return (0.0, 0.0)
            
        nx = dx / distance
        ny = dy / distance
        return (magnitude * nx, magnitude * ny)
        
    # 空间约束力
    def compute_space_constraint(self, i, label_positions):
        if i not in label_positions:
            return (0.0, 0.0)
        current_x, current_y = label_positions[i]
        label_id = self.labels[i].id

        if label_id not in self.constraints:
            return (0.0, 0.0)
        
        constraint_x, constraint_y = self.constraints[label_id]  
        dx = current_x - constraint_x
        dy = current_y - constraint_y

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return (0.0, 0.0) 
            
        distance = math.hypot(dx, dy)
        
        # 按论文公式计算力的大小
        magnitude = self.params['Wspace'] * math.log(distance + 1)
        
        # 计算力的方向：从当前位置指向约束位置（吸引力）
        # 论文公式：(pi - p'i) / ||pi - p'i||，但这是排斥方向
        # 空间约束力应该是吸引力，所以取负号
        nx = -dx / distance   # 负号使力指向约束位置
        ny = -dy / distance
        
        return (magnitude * nx, magnitude * ny)   
    
    # 计算合力
    def compute_resultant_force(self, i, label_positions, velocities):
        total_fx = 0.0
        total_fy = 0.0
        for j in range(len(self.labels)):
            if i != j:
                fx, fy = self.compute_label_label_repulsion(i, j, label_positions)
                total_fx += fx
                total_fy += fy

        fx, fy = self.compute_label_feature_repulsion(i, label_positions)
        total_fx += fx
        total_fy += fy

        fx, fy = self.compute_pulling_force(i, label_positions)
        total_fx += fx
        total_fy += fy

        fx, fy = self.compute_friction(i, velocities)
        total_fx += fx
        total_fy += fy
 
        for j in range(len(self.features)):
            if i != j:
                fx, fy = self.compute_time_constraint(i, j, label_positions)
                total_fx += fx
                total_fy += fy

 
        fx, fy = self.compute_space_constraint(i, label_positions)
        total_fx += fx
        total_fy += fy
        return (total_fx, total_fy)

    def update_positions(self, label_positions, velocities, time_delta):
        
        new_positions = {}
        new_velocities = {}

        for i in range(len(self.labels)):
            if i not in label_positions or i not in velocities:
                continue   
 
            force = self.compute_resultant_force(i, label_positions, velocities)
            acceleration_x, acceleration_y = force   

            new_vx = velocities[i][0] + acceleration_x * time_delta
            new_vy = velocities[i][1] + acceleration_y * time_delta
            
            new_x = label_positions[i][0] + velocities[i][0] * time_delta
            new_y = label_positions[i][1] + velocities[i][1] * time_delta

            new_x = max(0, min(self.max_x, new_x))
            new_y = max(0, min(self.max_y, new_y))

            new_positions[i] = (new_x, new_y)
            new_velocities[i] = (new_vx, new_vy)
        return new_positions, new_velocities

    def optimize_labels(self, initial_positions, initial_velocities, time_delta, max_iter=1000):
        
        current_positions = initial_positions.copy()  
        current_velocities = initial_velocities.copy()  

        for _ in range(max_iter):
 
            new_positions, new_velocities = self.update_positions(
                current_positions, current_velocities, time_delta
            )
            current_positions = new_positions
            current_velocities = new_velocities
        return current_positions, current_velocities


