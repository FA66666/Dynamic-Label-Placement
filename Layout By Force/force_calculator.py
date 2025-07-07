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
            dist = math.hypot(label.center_x - N.center_x, label.center_y - N.center_y)
            s_i = math.hypot(label.width, label.height) / 2  # 标签等效半径
            s_j = math.hypot(N.width, N.height) / 2
            d = dist - (s_i + s_j)  # 实际间距
            
            if d < p.get('Dlabel-collision', 0):  # 仅在重叠或接近时产生排斥力
                magnitude = p['wlabel-collision'] * min(d / p['Dlabel-collision'] - 1, 0)
                nx = (label.center_x - N.center_x) / (dist + 1e-6)  # 排斥方向单位向量
                ny = (label.center_y - N.center_y) / (dist + 1e-6)
                fx += magnitude * nx
                fy += magnitude * ny
        
        for N in neighbor_features:
            if N.id == label.id: continue
            dist = math.hypot(label.center_x - N.x, label.center_y - N.y)
            s_i = math.hypot(label.width, label.height) / 2
            d = dist - (s_i + N.radius)
            
            if d < p.get('Dfeature-collision', 0):
                magnitude = p['wfeature-collision'] * min(d / p['Dfeature-collision'] - 1, 0)
                nx = (label.center_x - N.x) / (dist + 1e-6)
                ny = (label.center_y - N.y) / (dist + 1e-6)
                fx += magnitude * nx
                fy += magnitude * ny
                
        return fx, fy

    def _compute_inherent_forces(self, label):
        """计算标签的内在力（牵引力 + 摩擦力）"""
        p = self.params
        fx, fy = 0, 0
        feature = label.feature

        dist_pull = math.hypot(label.center_x - feature.x, label.center_y - feature.y)
        if dist_pull > p['Dpull']:  # 距离超过阈值时产生牵引力
            mag_pull = p['wpull'] * math.log(dist_pull - p['Dpull'] + 1)
            nx_pull = (feature.x - label.center_x) / (dist_pull + 1e-6)  # 指向特征点
            ny_pull = (feature.y - label.center_y) / (dist_pull + 1e-6)
            fx += mag_pull * nx_pull
            fy += mag_pull * ny_pull

        delta_vx = label.vx - feature.vx  # 相对速度
        delta_vy = label.vy - feature.vy
        fx += -p['c_friction'] * delta_vx  # 摩擦力与相对速度反向
        fy += -p['c_friction'] * delta_vy
        
        return fx, fy