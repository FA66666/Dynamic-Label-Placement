import math
from point import Point
from label import Label

class ForceCalculator:
    def __init__(self, params):
        self.params = params

    def compute_total_force_for_label(self, label, neighbor_labels, neighbor_features):
        """计算标签受到的总力（邻域力 + 内在力）"""
        fx_neighbor, fy_neighbor = self._compute_neighbor_forces(label, neighbor_labels, neighbor_features)
        fx_inherent, fy_inherent = self._compute_inherent_forces(label)
        return fx_neighbor + fx_inherent, fy_neighbor + fy_inherent

    def _compute_neighbor_forces(self, label, neighbor_labels, neighbor_features):
        """计算邻域物体对标签的碰撞排斥力"""
        p = self.params
        fx, fy = 0, 0

        for N in neighbor_labels:
            if N.id == label.id: continue
            x1,y1 = label.center_x, label.center_y
            x2,y2 = N.x, N.y

            ei_x = label.width / 2
            ei_y = label.height / 2
            ej_x = N.width / 2
            ej_y = N.height / 2
            dx = abs(x1 - x2) - 0.5 * (ei_x + ej_x)
            dy = abs(y1 - y2) - 0.5 * (ei_y + ej_y)

            d = max(dx, dy)  
            
            
            magnitude = p['wlabel-collision'] * min(d / p['Dlabel-collision'] - 1, 0)

            # 计算排斥力方向
            d_vec_x = x1 - x2
            d_vec_y = y1 - y2
            d_norm = math.hypot(d_vec_x, d_vec_y)+1e-6
            nx = d_vec_x / d_norm
            ny = d_vec_y / d_norm
            
            fx += magnitude * nx
            fy += magnitude * ny
        
        for N in neighbor_features:
            if N.id == label.id: continue
            label_x, label_y = label.center_x, label.center_y
            feature_x, feature_y = N.x, N.y
            si_x = label.width / 2
            si_y = label.height / 2
            rj = N.radius  # 特征点的半径

            dx = abs(label_x - feature_x) - 0.5 * (si_x + rj)
            dy = abs(label_y - feature_y) - 0.5 * (si_y + rj)
            d_label_feature = max(dx, dy)   

            # 计算力的方向
            d_vec_x = label_x - feature_x
            d_vec_y = label_y - feature_y
            dist = math.hypot(d_vec_x, d_vec_y) + 1e-6

            
            magnitude = p['wfeature-collision'] * min(d_label_feature / p['Dfeature-collision'] - 1, 0)
            nx = (label.center_x - N.x) / (dist + 1e-6)
            ny = (label.center_y - N.y) / (dist + 1e-6)
            fx += magnitude * nx
            fy += magnitude * ny
                
        return fx, fy

    def _compute_inherent_forces(self, label):
        """计算标签的内在力（拉力 + 摩擦力）"""
        p = self.params
        fx, fy = 0, 0
        feature = label.feature

        # 计算标签和特征点之间的有效距离（边界距离）
        label_x, label_y = label.center_x, label.center_y
        feature_x, feature_y = feature.x, feature.y
        
        si_x = label.width / 2
        si_y = label.height / 2
        ri = feature.radius
        
        # 计算边界之间的距离
        dx = abs(label_x - feature_x) - 0.5 * (si_x + ri)
        dy = abs(label_y - feature_y) - 0.5 * (si_y + ri)
        d_effective = max(dx, dy)
        
        # 计算拉力
        expr = d_effective - p['Dpull']
        if expr > 0:
            mag_pull = p['wpull'] * math.log(expr + 1)
            
            # 从标签指向特征点的方向（拉向特征点）
            d_vec_x = feature_x - label_x
            d_vec_y = feature_y - label_y
            d_norm = math.hypot(d_vec_x, d_vec_y)
            
            if d_norm > 1e-6:
                # 归一化方向向量
                nx_pull = d_vec_x / d_norm
                ny_pull = d_vec_y / d_norm
                fx += mag_pull * nx_pull
                fy += mag_pull * ny_pull

        delta_vx = label.vx - feature.vx  # 相对速度
        delta_vy = label.vy - feature.vy
        fx += -p['c_friction'] * delta_vx  # 摩擦力与相对速度反向
        fy += -p['c_friction'] * delta_vy
        
        return fx, fy