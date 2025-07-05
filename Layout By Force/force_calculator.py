# 文件名: force_calculator.py
import math
from point import Point
from label import Label

class ForceCalculator:
    def __init__(self, params):
        self.params = params

    def compute_total_force_for_label(self, label, neighbor_labels, neighbor_features):
        total_fx, total_fy = 0, 0
        fx_neighbor, fy_neighbor = self._compute_neighbor_forces(label, neighbor_labels, neighbor_features)
        fx_inherent, fy_inherent = self._compute_inherent_forces(label)
        total_fx = fx_neighbor + fx_inherent
        total_fy = fy_neighbor + fy_inherent
        return total_fx, total_fy

    def _compute_neighbor_forces(self, label, neighbor_labels, neighbor_features):
        p = self.params
        fx, fy = 0, 0

        # 1. 标签-标签碰撞力 (f_label-collision)
        for N in neighbor_labels:
            if N.id == label.id: continue
            dist = math.hypot(label.center_x - N.center_x, label.center_y - N.center_y)
            # 简化后的距离计算
            s_i = math.hypot(label.width, label.height) / 2
            s_j = math.hypot(N.width, N.height) / 2
            # 论文中的 d_label-label
            d = dist - (s_i + s_j)
            
            if d < p.get('Dlabel-collision', 0): # 仅在重叠或接近时计算
                # 论文中的 f_label-collision 公式
                magnitude = p['wlabel-collision'] * min(d / p['Dlabel-collision'] - 1, 0)
                nx = (label.center_x - N.center_x) / (dist + 1e-6)
                ny = (label.center_y - N.center_y) / (dist + 1e-6)
                fx += magnitude * nx
                fy += magnitude * ny
        
        # 2. 标签-特征点碰撞力
        for N in neighbor_features:
            if N.id == label.id: continue
            dist = math.hypot(label.center_x - N.x, label.center_y - N.y)
            s_i = math.hypot(label.width, label.height) / 2
            # 论文中的 d_label-feature
            d = dist - (s_i + N.radius)
            
            if d < p.get('Dfeature-collision', 0): # 仅在重叠或接近时计算
                magnitude = p['wfeature-collision'] * min(d / p['Dfeature-collision'] - 1, 0)
                nx = (label.center_x - N.x) / (dist + 1e-6)
                ny = (label.center_y - N.y) / (dist + 1e-6)
                fx += magnitude * nx
                fy += magnitude * ny
                
        return fx, fy

    def _compute_inherent_forces(self, label):
        p = self.params
        fx, fy = 0, 0
        feature = label.feature

        # 3. 特征点-标签牵引力 (f_pull)
        dist_pull = math.hypot(label.center_x - feature.x, label.center_y - feature.y)
        if dist_pull > p['Dpull']:
            # 论文中的 f_pull 公式
            mag_pull = p['wpull'] * math.log(dist_pull - p['Dpull'] + 1)
            # 方向是从标签指向特征点
            nx_pull = (feature.x - label.center_x) / (dist_pull + 1e-6)
            ny_pull = (feature.y - label.center_y) / (dist_pull + 1e-6)
            fx += mag_pull * nx_pull
            fy += mag_pull * ny_pull

        # 4. 摩擦力 (f_friction)
        # 论文中的 f_friction 公式
        delta_vx = label.vx - feature.vx
        delta_vy = label.vy - feature.vy
        fx += -p['c_friction'] * delta_vx
        fy += -p['c_friction'] * delta_vy
        
        return fx, fy