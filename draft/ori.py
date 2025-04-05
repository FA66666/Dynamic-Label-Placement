import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import random

# 参数（来自 Appendix A1 和 A2）
W_LABEL_LABEL = 80
W_LABEL_FEATURE = 50
W_ORIENTATION = [1, 2, 3, 4]  # 四个象限的优先级
W_DISTANCE = 20
W_OUT_OF_AXES = 320
W_INTERSECT = 1
W_RADIUS = 10  # 约束传递中的半径权重
W_ANGLE = 5    # 约束传递中的角度权重
MAX_RADIUS = 300

# 力导向参数
W_LABEL_COLLISION = 50
D_LABEL_COLLISION = 30
W_FEATURE_COLLISION = 60
D_FEATURE_COLLISION = 17
W_PULL = 25
D_PULL = 18
C_FRICTION = 0.7
W_FRICTION = 6
W_SPACE = 20
W_TIME = 15
DT = 0.1
DELTA_T = 5  # 时间力中的未来时间（秒）

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


def detect_joint_sets(features):
    joint_sets = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            for t in range(len(features[0].path)):
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


def exponential_curve_prediction(feature, current_frame, future_frames=5):
    if current_frame < 2:
        velocity = feature.velocity if current_frame > 0 else np.array([0.0, 0.0])
        return [feature.path[current_frame] + velocity * t * DT for t in range(1, future_frames + 1)]
    # 简单指数曲线模拟（可替换为更复杂的模型）
    positions = feature.path[:current_frame + 1]
    velocities = np.diff(positions, axis=0) / DT
    avg_velocity = np.mean(velocities[-2:], axis=0)
    predicted = []
    for t in range(1, future_frames + 1):
        decay = np.exp(-t * DT)
        predicted_pos = feature.path[current_frame] + avg_velocity * t * DT * decay
        predicted.append(predicted_pos)
    return predicted

def calculate_forces(labels, features, joint_sets, current_frame):
    forces = []
    for label in labels:
        f_total = np.array([0.0, 0.0])
        feat = label.feature

        # Label-Label Collision Force
        for other in labels:
            if label == other:
                continue
            d = np.linalg.norm(label.position - other.position)
            if d < D_LABEL_COLLISION:
                direction = (label.position - other.position) / d
                f_total += direction * (D_LABEL_COLLISION - d) * W_LABEL_COLLISION

        # Label-Feature Collision Force
        d_f = np.linalg.norm(label.position - feat.current_pos)
        if d_f < D_FEATURE_COLLISION:
            direction_f = (label.position - feat.current_pos) / d_f
            f_total += direction_f * (D_FEATURE_COLLISION - d_f) * W_FEATURE_COLLISION

        # Pulling Force
        pull_dir = feat.current_pos - label.position
        d_pull = np.linalg.norm(pull_dir)
        if d_pull > D_PULL:
            direction_pull = pull_dir / d_pull
            f_total += direction_pull * W_PULL * np.log(d_pull - D_PULL + 1)

        # Friction Force
        f_total += -C_FRICTION * (label.velocity - feat.velocity) * W_FRICTION

        # Space Constraint Force
        f_space = np.array([0.0, 0.0])
        for jset, t in joint_sets:
            if label.feature.id in jset and t > current_frame:
                future_pos = feat.path[t] if t < len(feat.path) else feat.current_pos
                direction_space = future_pos - label.position
                d_space = np.linalg.norm(direction_space)
                f_space += np.log(d_space + 1) * direction_space / d_space * W_SPACE
                break
        if np.linalg.norm(label.position) > MAX_RADIUS:
            direction_space = -label.position / np.linalg.norm(label.position)
            f_space += direction_space * W_SPACE
        f_total += f_space

        # 时间力：提前预测并调整标签位置
        predicted_path = exponential_curve_prediction(feat, current_frame)
        for future_pos in predicted_path[:1]:  # 只取未来位置的前一个预测
            d_time = np.linalg.norm(label.position - future_pos)
            if d_time < D_PULL:
                direction_time = (label.position - future_pos) / d_time
                v_ratio = max(np.linalg.norm(feat.velocity), 1e-6) / max(np.linalg.norm(feat.velocity), 1e-6)
                f_total += direction_time * W_TIME * np.log(v_ratio) * min(d_time / D_PULL - 1, 0)

        forces.append(f_total)
    return forces


def update_positions(labels, forces):
    for label, force in zip(labels, forces):
        label.velocity += force * DT
        label.position += label.velocity * DT
        r = np.hypot(*label.position)
        if r > MAX_RADIUS:
            angle = np.arctan2(label.position[1], label.position[0])
            label.position = np.array([MAX_RADIUS * np.cos(angle), MAX_RADIUS * np.sin(angle)])
        label.angle, label.radius = label.cartesian_to_polar()

def generate_simulation_data(num_features=3, num_frames=50):
    features = []
    for i in range(num_features):
        path = []
        for t in range(num_frames):
            if i == 0:
                theta = np.pi - (2 * np.pi / (num_frames - 1)) * t
                x = -100 + 100 * np.cos(theta)
                y = 100 * np.sin(theta)
            elif i == 1:
                theta = (2 * np.pi / (num_frames - 1)) * t
                x = 100 + 100 * np.cos(theta)
                y = 100 * np.sin(theta)
            else:
                x = -300 + (600 / (num_frames - 1)) * t
                y = 0
            x += np.random.normal(0, 5)
            y += np.random.normal(0, 5)
            path.append([x, y])
        features.append(Feature(i, path))
    return features

def calculate_metrics(labels, features, previous_labels):
    overlap = calculate_overlap(labels, features)
    position_change = sum(np.linalg.norm(label.position - prev.position)
                         for label, prev in zip(labels, previous_labels) if prev is not None)
    intersections = calculate_intersections(labels)
    return overlap, position_change, intersections

def update(frame, features, labels_list, previous_labels_list):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    colors = ['red', 'green', 'blue']

    for idx, feat in enumerate(features):
        feat.current_pos = feat.path[frame]
        if len(feat.path_history) == 0 or not np.array_equal(feat.path_history[-1], feat.current_pos):
            feat.path_history.append(feat.current_pos.copy())
        if frame > 0:
            feat.update_velocity(feat.path[frame - 1], feat.current_pos, DT)
        ax.plot([p[0] for p in feat.path_history], [p[1] for p in feat.path_history],
                color=colors[idx], alpha=0.5, linewidth=1)
        ax.scatter(feat.current_pos[0], feat.current_pos[1], color=colors[idx], s=50, zorder=3)

    current_labels = labels_list[frame]
    for label in current_labels:
        rect = get_rect(label)
        ax.add_patch(Rectangle((rect['x'], rect['y']), rect['width'], rect['height'],
                               fill=True, color='skyblue', alpha=0.7))
        ax.plot([label.position[0], label.feature.current_pos[0]],
                [label.position[1], label.feature.current_pos[1]],
                color='gray', linestyle='--', linewidth=0.8)

    prev_labels = previous_labels_list[frame] if frame > 0 else [None] * len(features)
    overlap, pos_change, intersections = calculate_metrics(current_labels, features, prev_labels)
    ax.set_title(f"Frame {frame} | Overlap: {overlap:.1f} | Pos Change: {pos_change:.1f} | Intersections: {intersections}")
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')

if __name__ == "__main__":
    features = generate_simulation_data(num_features=3, num_frames=50)
    labels_list = []
    previous_labels_list = [None]

    # 全局检测联合集
    joint_sets = detect_joint_sets(features)
    joint_labels = {}
    prev_labels = None
    for jset, t in joint_sets:
        joint_features = [features[i] for i in jset]
        for feat in joint_features:
            feat.current_pos = feat.path[t]
        optimized_labels = simulated_annealing(joint_features, previous_labels=prev_labels)
        for label in optimized_labels:
            joint_labels[(label.feature.id, t)] = label
        prev_labels = optimized_labels

    # 初始帧全局优化
    for feat in features:
        feat.current_pos = feat.path[0]
    initial_labels = simulated_annealing(features)
    labels_list.append(initial_labels)

    # 后续帧处理
    for frame in range(1, len(features[0].path)):
        for feat in features:
            feat.current_pos = feat.path[frame]
        current_labels = [label.copy() for label in labels_list[-1]]

        # 应用联合集约束
        for i, label in enumerate(current_labels):
            key = (label.feature.id, frame)
            if key in joint_labels:
                current_labels[i] = joint_labels[key].copy()

        # 力导向优化
        for _ in range(10):
            forces = calculate_forces(current_labels, features, joint_sets, frame)
            update_positions(current_labels, forces)

        labels_list.append(current_labels)
        previous_labels_list.append(labels_list[-2])

    fig = plt.figure(figsize=(10, 10))
    anim = animation.FuncAnimation(fig, update, frames=len(features[0].path),
                                   fargs=(features, labels_list, previous_labels_list),
                                   interval=200)
    anim.save("dynamic_labels_optimized.gif", writer='pillow', fps=10, dpi=100)
