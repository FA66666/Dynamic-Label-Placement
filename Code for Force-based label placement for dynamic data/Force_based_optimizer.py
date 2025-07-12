import math
from config import params_Adj, param_NoAdj, global_params
class ForceBasedOptimizer:
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels if labels is not None else []

        # 动态计算弱力参数
        if self.labels:
            max_label_dimension = max(
                max(label.length, label.width) for label in self.labels
            )
            # 更新弱力参数
            if param_NoAdj['m_weak_collision'] is None:
                param_NoAdj['m_weak_collision'] = max_label_dimension
            if param_NoAdj['m_weak_feature'] is None:
                param_NoAdj['m_weak_feature'] = max_label_dimension

    # 标签-标签排斥力
    def compute_label_label_repulsion(self,i,j,label_positions):
        x1, y1 = label_positions[i]
        x2, y2 = label_positions[j]

        ei_x = self.labels[i].length / 2   
        ei_y = self.labels[i].width / 2    
        ej_x = self.labels[j].length / 2
        ej_y = self.labels[j].width / 2
        dx = abs(x1 - x2) - 0.5 * (ei_x + ej_x)
        dy = abs(y1 - y2) - 0.5 * (ei_y + ej_y)
        d_label = max(dx, dy) 

        v = min(d_label/params_Adj['m_collision']-1, 0)  
        d_vec_x = x1 - x2
        d_vec_y = y1 - y2
        d_norm = math.hypot(d_vec_x, d_vec_y)+1e-6
        nx = d_vec_x / d_norm
        ny = d_vec_y / d_norm
        return (v * nx, v * ny)
    
    # 标签-标签弱排斥力
    def compute_label_label_weak_repulsion(self,i,j,label_positions):
        x1, y1 = label_positions[i]
        x2, y2 = label_positions[j]

        ei_x = self.labels[i].length / 2   
        ei_y = self.labels[i].width / 2    
        ej_x = self.labels[j].length / 2
        ej_y = self.labels[j].width / 2
        dx = abs(x1 - x2) - 0.5 * (ei_x + ej_x)
        dy = abs(y1 - y2) - 0.5 * (ei_y + ej_y)
        d_label = max(dx, dy) 

        v = min(d_label/param_NoAdj['m_weak_collision']-1, 0)  
        d_vec_x = x1 - x2
        d_vec_y = y1 - y2
        d_norm = math.hypot(d_vec_x, d_vec_y)+1e-6
        nx = d_vec_x / d_norm
        ny = d_vec_y / d_norm
        return (v * nx, v * ny)
    
    # 标签-特征排斥力
    def compute_label_feature_repulsion(self, i ,label_positions):
        total_fx = 0
        total_fy = 0
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
            
            v = min(d_label_feature / params_Adj['m_feature'] - 1, 0)
                     
            d_vec_x = label_x - feature_x
            d_vec_y = label_y - feature_y
            d_norm = math.hypot(d_vec_x, d_vec_y)+1e-6
                       
            nx = d_vec_x / d_norm
            ny = d_vec_y / d_norm
            total_fx += v * nx
            total_fy += v * ny
            
        return (total_fx, total_fy)
    
    # 标签-特征弱排斥力
    def compute_label_feature_weak_repulsion(self, i ,label_positions):
        total_fx = 0
        total_fy = 0
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
            
            v = min(d_label_feature / param_NoAdj['m_weak_feature'] - 1, 0)
                     
            d_vec_x = label_x - feature_x
            d_vec_y = label_y - feature_y
            d_norm = math.hypot(d_vec_x, d_vec_y)+1e-6
                       
            nx = d_vec_x / d_norm
            ny = d_vec_y / d_norm
            total_fx += v * nx
            total_fy += v * ny
            
        return (total_fx, total_fy)
    

    # 标签运动预测力
    def compute_movement_prediction_force(self, i, j, label_positions, velocities):
        """
        计算标签i受到标签j运动的预测力
        基于论文中的运动预测算法
        """
        if i == j:
            return (0.0, 0.0)
            
        # 获取标签位置和速度
        xi, yi = label_positions[i]
        xj, yj = label_positions[j]
        vi_x, vi_y = velocities[i]
        vj_x, vj_y = velocities[j]
        
        # 计算相对速度 v_diff = v_j - v_i
        vdiff_x = vj_x - vi_x
        vdiff_y = vj_y - vi_y
        vdiff_norm = math.hypot(vdiff_x, vdiff_y)+1e-6
            
        # 标签尺寸
        ei_x = self.labels[i].length / 2
        ei_y = self.labels[i].width / 2
        ej_x = self.labels[j].length / 2
        ej_y = self.labels[j].width / 2
        
        # 计算垂直于速度方向的距离 d_⊥
        d_perp_numerator = (ei_x + ej_x) * abs(vdiff_y) + (ei_y + ej_y) * abs(vdiff_x)
        d_perp = d_perp_numerator / (2 * ((vdiff_norm)**2+ 1e-6))  # 防止除以0
        
        # 计算平行于速度方向的距离 d_∥
        d_parallel = max((ei_x + ej_x) / 2, (ei_y + ej_y) / 2)
        
               # 缩放矩阵 D
        d_perp_inv = 1.0 / (d_perp+1e-6 )
        d_parallel_inv = 1.0 / (d_parallel+ 1e-6)

        # 修复拼写错误：d_prep_inv 应该是 d_perp_inv
        d_perp_inv /= param_NoAdj['m_label_predict']
        d_parallel_inv /= param_NoAdj['m_label_predict']

        # 旋转矩阵 R - 将坐标系旋转到运动方向
        vdiff_unit_x = vdiff_x / vdiff_norm
        vdiff_unit_y = vdiff_y / vdiff_norm
        
        # 位置差向量
        pos_diff_x = xi - xj
        pos_diff_y = yi - yj
        
        # 应用旋转变换到运动方向坐标系
        rotated_x = vdiff_unit_x * pos_diff_x + vdiff_unit_y * pos_diff_y
        rotated_y = -vdiff_unit_y * pos_diff_x + vdiff_unit_x * pos_diff_y
        
        # 应用缩放矩阵
        scaled_x = d_perp_inv * rotated_x
        scaled_y = d_parallel_inv * rotated_y
        
        # 计算缩放后的距离
        scaled_norm = math.hypot(scaled_x, scaled_y) + 1e-6
        
        # Heaviside函数 H(x) - 当x>0时返回1，否则返回0
        def heaviside(x):
            return 1.0 if x > 0 else 0.0


        y_x = (scaled_x / scaled_norm) * heaviside(scaled_x) * max(0, 1 - scaled_norm) + 1e-6
        y_y = (scaled_y / scaled_norm) * heaviside(scaled_y) * max(0, 1 - scaled_norm) + 1e-6
            
        # 反向旋转回原坐标系
        force_x = vdiff_unit_x * y_x - vdiff_unit_y * y_y
        force_y = vdiff_unit_y * y_x + vdiff_unit_x * y_y
        
        return (vdiff_norm * force_x, vdiff_norm * force_y)

    # 点运动预测力
    def compute_point_movement_prediction_force(self, i, j, label_positions, feature_velocities):
        """
        计算标签i受到特征点j运动的预测力
        基于论文中的运动预测算法，只考虑标签的尺寸
        """
        # 获取标签位置和特征点位置、速度
        xi, yi = label_positions[i]
        xj, yj = self.features[j].position
        vj_x, vj_y = feature_velocities[j]
        
        # 特征点速度（标签速度为0，因为我们只考虑特征点的运动）
        vdiff_x = vj_x
        vdiff_y = vj_y
        vdiff_norm = math.hypot(vdiff_x, vdiff_y) + 1e-6
        
        # 如果特征点没有运动，返回零力
        if vdiff_norm < 1e-6:
            return (0.0, 0.0)
            
        # 标签尺寸（只考虑标签的尺寸）
        ei_x = self.labels[i].length / 2
        ei_y = self.labels[i].width / 2
        
        # 计算垂直于速度方向的距离 d_⊥
        d_perp_numerator = ei_x * abs(vdiff_y) + ei_y * abs(vdiff_x)
        d_perp = d_perp_numerator / (vdiff_norm**2 + 1e-6)
        
        # 计算平行于速度方向的距离 d_∥
        d_parallel = max(ei_x, ei_y)
        
        # 缩放矩阵 D
        d_perp_inv = 1.0 / (d_perp + 1e-6)
        d_parallel_inv = 1.0 / (d_parallel + 1e-6)
        
        d_perp_inv /= param_NoAdj['m_point_predict']
        d_parallel_inv /= param_NoAdj['m_point_predict']
        
        # 旋转矩阵 R - 将坐标系旋转到运动方向
        vdiff_unit_x = vdiff_x / vdiff_norm
        vdiff_unit_y = vdiff_y / vdiff_norm
        
        # 位置差向量（标签相对于特征点的位置）
        pos_diff_x = xi - xj
        pos_diff_y = yi - yj
        
        # 应用旋转变换到运动方向坐标系
        rotated_x = vdiff_unit_x * pos_diff_x + vdiff_unit_y * pos_diff_y
        rotated_y = -vdiff_unit_y * pos_diff_x + vdiff_unit_x * pos_diff_y
        
        # 应用缩放矩阵
        scaled_x = d_perp_inv * rotated_x
        scaled_y = d_parallel_inv * rotated_y
        
        # 计算缩放后的距离
        scaled_norm = math.hypot(scaled_x, scaled_y)
        
        # Heaviside函数 H(x) - 当x>0时返回1，否则返回0
        def heaviside(x):
            return 1.0 if x > 0 else 0.0
        
        
        y_x = (scaled_x / (scaled_norm + 1e-6)) * heaviside(scaled_x) * max(0, 1 - scaled_norm)
        y_y = (scaled_y / (scaled_norm + 1e-6)) * heaviside(scaled_y) * max(0, 1 - scaled_norm)
        
        # 反向旋转回原坐标系
        force_x = vdiff_unit_x * y_x - vdiff_unit_y * y_y
        force_y = vdiff_unit_y * y_x + vdiff_unit_x * y_y
        
        return (vdiff_norm * force_x, vdiff_norm * force_y)

    #拉力
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

        expr = d_effective - param_NoAdj['m_pull']
        if expr > 0:
            magnitude = math.log(expr + 1)

            d_vec_x = feature_x - label_x
            d_vec_y = feature_y - label_y
            d_norm = math.hypot(d_vec_x, d_vec_y) + 1e-6
            nx = d_vec_x / d_norm
            ny = d_vec_y / d_norm
            return (magnitude * nx, magnitude * ny)
        else:
            return (0.0, 0.0)
        
    # 摩擦力
    def compute_friction(self, i ,velocities):
        v_label_x, v_label_y = velocities[i]
        v_feature_x, v_feature_y = self.features[i].velocity
        delta_vx = v_label_x - v_feature_x
        delta_vy = v_label_y - v_feature_y
        return (-delta_vx, -delta_vy)
    
    # 计算合力
    def compute_resultant_force(self,i , label_positions, velocities):
        total_fx = 0.0
        total_fy = 0.0

        # 计算标签-标签排斥力
        for j in range(len(self.labels)):
            if i!= j:
                fx,fy  = self.compute_label_label_repulsion(i, j, label_positions)
                weak_fx, weak_fy = self.compute_label_label_weak_repulsion(i, j, label_positions)
                total_fx += (fx * params_Adj['c_collision'] + weak_fx * param_NoAdj['c_weak_collision'])
                total_fy += (fy * params_Adj['c_collision'] + weak_fy * param_NoAdj['c_weak_collision'])

        # 计算标签-特征排斥力
        fx, fy = self.compute_label_feature_repulsion(i, label_positions)
        weak_fx, weak_fy = self.compute_label_feature_weak_repulsion(i, label_positions)
        total_fx += (fx * param_NoAdj['c_feature'] + weak_fx * param_NoAdj['c_weak_feature'])
        total_fy += (fy * param_NoAdj['c_feature'] + weak_fy * param_NoAdj['c_weak_feature'])

        # 计算运动预测力 (标签-标签预测力)
        for j in range(len(self.labels)):
            if i != j:
                pred_fx, pred_fy = self.compute_movement_prediction_force(i, j, label_positions, velocities)
                total_fx += pred_fx * param_NoAdj['c_label_predict']
                total_fy += pred_fy * param_NoAdj['c_label_predict']

        # 计算特征点运动预测力
        feature_velocities = [feature.velocity for feature in self.features]
        for j in range(len(self.features)):
            pred_fx, pred_fy = self.compute_point_movement_prediction_force(i, j, label_positions, feature_velocities)
            total_fx += pred_fx * param_NoAdj['c_point_predict']
            total_fy += pred_fy * param_NoAdj['c_point_predict']

        # 计算拉力
        fx, fy = self.compute_pulling_force(i, label_positions)
        total_fx += fx * params_Adj['c_pull'] 
        total_fy += fy * params_Adj['c_pull']

        # 计算摩擦力
        fx, fy = self.compute_friction(i, velocities)
        total_fx += fx * param_NoAdj['c_friction']
        total_fy += fy * param_NoAdj['c_friction']

        return (total_fx, total_fy)
    
    # 更新标签位置
        # 更新标签位置
    def update_label_positions(self, label_positions, velocities, dt):
        """
        使用Euler积分方法更新标签位置
        包含静态摩擦阈值处理
        """
        new_positions = {}
        new_velocities = {}
        
        for i in range(len(self.labels)):
            if i not in label_positions:
                continue
                
            # 计算合力
            total_fx, total_fy = self.compute_resultant_force(i, label_positions, velocities)
            
            # 加速度 = 合力 (假设质量为1)
            acceleration_x = total_fx
            acceleration_y = total_fy
            
            # 获取当前速度
            current_vx, current_vy = velocities[i] if i in velocities else (0.0, 0.0)
            
            # 计算新速度 v_i^{t'} = v_i^t + a_i * Δt
            new_vx = current_vx + acceleration_x * dt
            new_vy = current_vy + acceleration_y * dt
            
            # 静态摩擦阈值检查
            # 计算 max(||v_i^{t'}||, ||f_total^i|| / c_friction)
            velocity_magnitude = math.hypot(new_vx, new_vy)
            force_magnitude = math.hypot(total_fx, total_fy)
            friction_coefficient = param_NoAdj['c_friction']
            
           
            force_over_friction = force_magnitude / (friction_coefficient + 1e-6)
           
                
            threshold_condition = max(velocity_magnitude, force_over_friction)
            
            # 如果低于静态阈值，则不应用力（保持当前状态）
            if threshold_condition < params_Adj['c_static']:
                # 速度设为零，位置不变
                new_velocities[i] = (0.0, 0.0)
                new_positions[i] = label_positions[i]
            else:
                # 正常更新
                new_velocities[i] = (new_vx, new_vy)
                
                # 计算新位置 l_i' = l_i + v_i^{t'} * Δt
                current_x, current_y = label_positions[i]
                new_x = current_x + new_vx * dt
                new_y = current_y + new_vy * dt
                
                new_positions[i] = (new_x, new_y)
        
        return new_positions, new_velocities
    
       # 运行多步优化
    def optimize(self, label_positions, velocities, num_steps, dt):
        """
        运行多步优化
        """
        current_positions = label_positions.copy()
        current_velocities = velocities.copy()
        
        for step in range(num_steps):
            current_positions, current_velocities = self.update_label_positions(
                current_positions, current_velocities, dt
            )
            
        return current_positions, current_velocities


    
   