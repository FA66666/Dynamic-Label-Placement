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
    'delta_t': 0.1,   
}

class DynamicLabelOptimizer:
    def __init__(self, labels, features, params, constraints=None):
        self.labels = labels
        self.features = features
        self.params = params
        self.constraints = constraints or {}   

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
               
        v = abs(min(d_label / self.params['Dlabel-collision'] - 1, 0))
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
            feature_x, feature_y = self.features[j].position
            rj = self.features[j].radius   
            
 
            dx = abs(label_x - feature_x) - 0.5 * (si_x + rj)
            dy = abs(label_y - feature_y) - 0.5 * (si_y + rj)
            d_label_feature = max(dx, dy)   
            
            v = abs(min(d_label_feature / self.params['Dfeature-collision'] - 1, 0))
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

        # 标签i的当前位置
        label_i_x, label_i_y = label_positions[i]

        # 获取对应的特征点 i 和 j
        feature_i = self.features[i]
        feature_j = self.features[j] if j < len(self.features) else None
        if feature_j is None:
            return (0.0, 0.0)
 
        # 计算特征点的速度大小 |vi| 和 |vj|
        v_i_mag = math.hypot(feature_i.velocity[0], feature_i.velocity[1])
        v_j_mag = math.hypot(feature_j.velocity[0], feature_j.velocity[1])
        
        # 避免除零错误
        if min(v_i_mag, v_j_mag) < 1e-6:
            return (0.0, 0.0)
        
        # 计算速度比的对数: ln(max(|vi|,|vj|) / min(|vi|,|vj|))
        velocity_ratio_log = math.log(max(v_i_mag, v_j_mag) / min(v_i_mag, v_j_mag))
        
        # 计算特征j的未来相对位置 l'_j = lj + (Vj - Vi) * dt
        relative_vx = feature_j.velocity[0] - feature_i.velocity[0]
        relative_vy = feature_j.velocity[1] - feature_i.velocity[1]
        l_j_future_x = feature_j.position[0] + relative_vx * self.params['delta_t']
        l_j_future_y = feature_j.position[1] + relative_vy * self.params['delta_t']

        # 计算当前距离 ||pi - lj|| 和未来距离 ||pi - l'_j||
        current_distance = math.hypot(label_i_x - feature_j.position[0], label_i_y - feature_j.position[1])
        future_distance = math.hypot(label_i_x - l_j_future_x, label_i_y - l_j_future_y)

        if future_distance < 1e-6:
            return (0.0, 0.0)
        
        # 计算距离比率项: min(||pi - lj|| / ||pi - l'_j|| - 1, 0)
        distance_ratio_term = min(current_distance / future_distance - 1, 0)

        # 计算力的大小
        magnitude = self.params['Wtime'] * velocity_ratio_log * distance_ratio_term

        # 计算力的方向: (pi - l'_j) / ||pi - l'_j||
        dx = label_i_x - l_j_future_x
        dy = label_i_y - l_j_future_y
        
        # 归一化方向向量
        nx = dx / future_distance
        ny = dy / future_distance
        
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
        
        # 力的方向：从约束点 p' 指向当前标签位置 p (p - p')
        dx = current_x - constraint_x
        dy = current_y - constraint_y

        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return (0.0, 0.0) 
            
        # 力的大小：ln(||p - p'|| + 1)
        magnitude = math.log(distance + 1)

        # 归一化方向向量
        nx = dx / distance
        ny = dy / distance
        
        magnitude = self.params['Wspace'] * math.log(distance + 1)

        nx = dx / distance
        ny = dy / distance
        
        # 此处返回的是吸引力，将标签拉向约束位置
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

            new_positions[i] = (new_x, new_y)
            new_velocities[i] = (new_vx, new_vy)
        return new_positions, new_velocities

    def optimize_labels(self, initial_positions, initial_velocities, time_delta):
        
        current_positions = initial_positions.copy()  
        current_velocities = initial_velocities.copy()  

        new_positions, new_velocities = self.update_positions(
            current_positions, current_velocities, time_delta
        )
        current_positions = new_positions
        current_velocities = new_velocities
        return current_positions, current_velocities