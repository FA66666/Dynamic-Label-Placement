# 参数（来自 Appendix A1 和 A2）
import random

import numpy as np

W_LABEL_LABEL = 80
W_LABEL_FEATURE = 50
W_ORIENTATION = [1, 2, 3, 4]  # 四个象限的优先级
W_DISTANCE = 20
W_OUT_OF_AXES = 320
W_INTERSECT = 1
W_RADIUS = 10  # 约束传递中的半径权重
W_ANGLE = 5  # 约束传递中的角度权重
MAX_RADIUS = 300

class Label:
    def __init__(self, feature, angle, radius):
        self.feature = feature
        self.angle = angle
        self.radius = radius
        self.position = self.polar_to_cartesian(angle, radius)
        self.velocity = np.array([0.0, 0.0])

    def polar_to_cartesian(self, angle, radius):
        return np.array([radius * np.cos(np.deg2rad(angle)), radius * np.sin(np.deg2rad(angle))])

    def cartesian_to_polar(self):
        x, y = self.position
        r = np.hypot(x, y)
        theta = np.rad2deg(np.arctan2(y, x)) % 360
        return theta, r

    def copy(self):
        new_label = Label(self.feature, self.angle, self.radius)
        new_label.position = self.position.copy()
        new_label.velocity = self.velocity.copy()
        return new_label


class Feature:
    def __init__(self, id, path):
        self.id = id
        self.path = np.array(path)
        self.current_pos = None
        self.velocity = np.array([0.0, 0.0])
        self.path_history = []

    def update_velocity(self, previous_position, current_position, delta_time):
        self.velocity = (current_position - previous_position) / delta_time



def detect_joint_sets(features):
    joint_sets = []
    for i in range(len(features)):
        for j in range(len(features)):
            if i == j:
                continue
            # 使用两个特征轨迹的最小长度进行遍历
            min_length = min(len(features[i].path), len(features[j].path))
            for t in range(min_length):
                pos_i = features[i].path[t]
                pos_j = features[j].path[t]
                if np.linalg.norm(pos_i - pos_j) < 50:  # 距离阈值
                    joint_sets.append((set([i, j]), t))
                    break
    # 合并相同时间的联合集
    merged_sets = {}
    for s, t in joint_sets:
        if t in merged_sets:
            merged_sets[t].update(s)
        else:
            merged_sets[t] = s
    return sorted([(s, t) for t, s in merged_sets.items()], key=lambda x: len(x[0]), reverse=True)

def get_rect(entity):
    pos = entity.position
    return {'x': pos[0] - 20, 'y': pos[1] - 10, 'width': 40, 'height': 20}


def calculate_rect_overlap(rect1, rect2):
    left = max(rect1['x'], rect2['x'])
    right = min(rect1['x'] + rect1['width'], rect2['x'] + rect2['width'])
    bottom = max(rect1['y'], rect2['y'])
    top = min(rect1['y'] + rect1['height'], rect2['y'] + rect2['height'])
    if left >= right or bottom >= top:
        return 0
    return (right - left) * (top - bottom)


def calculate_overlap(labels, features):
    E_overlap = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rect1 = get_rect(labels[i])
            rect2 = get_rect(labels[j])
            overlap_area = calculate_rect_overlap(rect1, rect2)
            E_overlap += W_LABEL_LABEL * overlap_area
        for feat in features:
            label_rect = get_rect(labels[i])
            feat_rect = {'x': feat.current_pos[0] - 8, 'y': feat.current_pos[1] - 8, 'width': 16, 'height': 16}
            overlap_area = calculate_rect_overlap(label_rect, feat_rect)
            E_overlap += W_LABEL_FEATURE * overlap_area
    return E_overlap


def calculate_intersections(labels):
    intersections = 0
    for i, label1 in enumerate(labels[:-1]):
        p1, q1 = label1.position, label1.feature.current_pos
        for label2 in labels[i + 1:]:
            p2, q2 = label2.position, label2.feature.current_pos
            if line_intersects(p1, q1, p2, q2):
                intersections += 1
    return intersections


def line_intersects(p1, q1, p2, q2):
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return ccw(p1, q1, p2) != ccw(p1, q1, q2) and ccw(p2, q2, p1) != ccw(p2, q2, q1)

def energy_function(labels, features, previous_labels=None):
    E_overlap = 0
    E_position = 0
    E_aesthetics = 0
    E_constraint = 0

    # 计算重叠能量（E_overlap）
    for i in range(len(labels)):
        rect_i = get_rect(labels[i])
        for j in range(i + 1, len(labels)):
            rect_j = get_rect(labels[j])
            overlap_area = calculate_rect_overlap(rect_i, rect_j)
            E_overlap += W_LABEL_LABEL * overlap_area
        for feat in features:
            feat_rect = {'x': feat.current_pos[0] - 8, 'y': feat.current_pos[1] - 8, 'width': 16, 'height': 16}
            E_overlap += W_LABEL_FEATURE * calculate_rect_overlap(rect_i, feat_rect)

    # 计算位置能量（E_position）
    for label in labels:
        angle, radius = label.cartesian_to_polar()
        quadrant = (int(angle // 90) % 4)  # 0-3对应四个象限
        E_position += W_ORIENTATION[quadrant] * W_DISTANCE * radius

    # 计算美学能量（E_aesthetics）
    intersections = calculate_intersections(labels)
    E_aesthetics += intersections * W_INTERSECT
    for label in labels:
        rect = get_rect(label)
        if np.hypot(rect['x'], rect['y']) > MAX_RADIUS:
            E_aesthetics += W_OUT_OF_AXES

    # 计算约束能量（E_constraint）
    if previous_labels:
        for label, prev_label in zip(labels, previous_labels):
            theta_curr, r_curr = label.cartesian_to_polar()
            theta_prev, r_prev = prev_label.cartesian_to_polar()
            E_constraint += W_RADIUS * abs(r_curr - r_prev) + W_ANGLE * abs(theta_curr - theta_prev)

    return E_overlap + E_position + E_aesthetics + E_constraint


def simulated_annealing(features, T_initial=1000, max_iterations=1000, previous_labels=None):
    labels = [Label(feat, random.uniform(0, 360), random.uniform(100, 200)) for feat in features]
    current_energy = energy_function(labels, features, previous_labels)
    best_labels = [label.copy() for label in labels]
    best_energy = current_energy
    T = T_initial
    for _ in range(max_iterations):
        new_labels = [label.copy() for label in labels]
        idx = random.randint(0, len(new_labels) - 1)
        new_labels[idx].angle += np.random.normal(0, 5)
        new_labels[idx].radius += np.random.normal(0, 10)
        new_labels[idx].position = new_labels[idx].polar_to_cartesian(new_labels[idx].angle, new_labels[idx].radius)
        new_energy = energy_function(new_labels, features, previous_labels)
        delta_energy = new_energy - current_energy
        if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
            labels = [label.copy() for label in new_labels]
            current_energy = new_energy
            if new_energy < best_energy:
                best_labels = [label.copy() for label in new_labels]
                best_energy = new_energy
        T *= 0.99
        if T < 1e-3:
            break
    return best_labels

