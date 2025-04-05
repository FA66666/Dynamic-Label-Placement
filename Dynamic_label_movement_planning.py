import numpy as np
from Global_spatiotemporal_joint_optimization import calculate_overlap, calculate_intersections

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
MAX_RADIUS = 300

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

        for other_feat in features:
            if other_feat == feat:
                continue
            v_i = feat.velocity
            v_j = other_feat.velocity
            delta_v = v_j - v_i
            l_j_prime = other_feat.current_pos + delta_v * DELTA_T
            d_current = np.linalg.norm(label.position - other_feat.current_pos)
            d_future = np.linalg.norm(label.position - l_j_prime)
            if d_current > d_future > 0:
                direction_time = (label.position - l_j_prime) / d_future
                v_max = max(np.linalg.norm(v_i), np.linalg.norm(v_j), 1e-6)
                v_min = min(np.linalg.norm(v_i), np.linalg.norm(v_j), 1e-6)
                ratio = d_future / d_current
                force_magnitude_time = np.log(v_max / v_min) * min(ratio - 1, 0) * W_TIME
                f_total += direction_time * force_magnitude_time

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


def calculate_metrics(labels, features, previous_labels):
    overlap = calculate_overlap(labels, features)
    position_change = sum(np.linalg.norm(label.position - prev.position)
                         for label, prev in zip(labels, previous_labels) if prev is not None)
    intersections = calculate_intersections(labels)
    return overlap, position_change, intersections
