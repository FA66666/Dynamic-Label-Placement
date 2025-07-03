# 文件名: force_calculator.py
import math
from models import Point, Label

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
        for N in neighbor_labels:
            if N.id == label.id: continue
            dist = math.hypot(label.center_x - N.center_x, label.center_y - N.center_y)
            if dist < p['D_critical'] or dist < p['R_adaptive']:
                s_i = math.hypot(label.width, label.height)
                s_j = math.hypot(N.width, N.height)
                d = dist - 0.5 * (s_i + s_j)
                magnitude = p['wlabel-collision'] * min(d / p['Dlabel-collision'] - 1, 0)
                nx = (label.center_x - N.center_x) / (dist + 1e-6)
                ny = (label.center_y - N.center_y) / (dist + 1e-6)
                fx += magnitude * nx
                fy += magnitude * ny
        for N in neighbor_features:
            if N.id == label.id: continue
            dist = math.hypot(label.center_x - N.x, label.center_y - N.y)
            if dist < p['D_critical'] or dist < p['R_adaptive']:
                s_i = math.hypot(label.width, label.height)
                d = dist - 0.5 * (s_i + N.radius)
                magnitude = p['wfeature-collision'] * min(d / p['Dfeature-collision'] - 1, 0)
                nx = (label.center_x - N.x) / (dist + 1e-6)
                ny = (label.center_y - N.y) / (dist + 1e-6)
                fx += magnitude * nx
                fy += magnitude * ny
                v_label_mag = math.hypot(label.vx, label.vy)
                v_feature_mag = math.hypot(N.vx, N.vy)
                if min(v_label_mag, v_feature_mag) > 1e-6:
                    velocity_ratio = max(v_label_mag, v_feature_mag) / min(v_label_mag, v_feature_mag)
                    magnitude_t = p['Wtime'] * math.log(velocity_ratio + 1) * min(d / p['Dfeature-collision'] - 1, 0)
                    fx += magnitude_t * nx
                    fy += magnitude_t * ny
        return fx, fy

    def _compute_inherent_forces(self, label):
        p = self.params
        fx, fy = 0, 0
        feature = label.feature
        dist_pull = math.hypot(label.center_x - feature.x, label.center_y - feature.y)
        if dist_pull > p['Dpull']:
             mag_pull = p['wpull'] * math.log(dist_pull - p['Dpull'] + 1)
             nx_pull = (feature.x - label.center_x) / (dist_pull + 1e-6)
             ny_pull = (feature.y - label.center_y) / (dist_pull + 1e-6)
             fx += mag_pull * nx_pull
             fy += mag_pull * ny_pull
        delta_vx = label.vx - feature.vx
        delta_vy = label.vy - feature.vy
        fx += -p['c_friction'] * delta_vx
        fy += -p['c_friction'] * delta_vy
        return fx, fy