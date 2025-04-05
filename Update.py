import numpy as np
from matplotlib.patches import Rectangle
import imageio
from skimage import measure
from scipy import ndimage
from Dynamic_label_movement_planning import calculate_metrics
from Global_spatiotemporal_joint_optimization import  get_rect


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

def track_points(input_path):
    # 读取输入GIF
    reader = imageio.get_reader(input_path)
    frames = []
    for im in reader:
        frames.append(im)

    # 处理可能缺失的fps
    try:
        fps = reader.get_meta_data()['fps']
    except KeyError:
        # 使用默认帧率（如10帧/秒）
        fps = 10

    reader.close()

    prev_gray = None
    trajectories = []

    for frame_idx in range(len(frames)):
        current_frame = frames[frame_idx]
        # 确保当前帧为uint8类型
        current_frame = current_frame.astype(np.uint8)

        # 转换为灰度图并确保类型为uint8
        current_gray = current_frame[..., :3].mean(axis=2).astype(np.uint8)

        if frame_idx == 0:
            prev_gray = current_gray
            continue

        # 计算帧间差异
        diff = np.abs(current_gray.astype(np.int16) - prev_gray.astype(np.int16)).astype(np.uint8)
        diff[diff < 20] = 0  # 调整阈值以过滤噪声
        diff[diff >= 20] = 255

        # 形态学处理：先腐蚀后膨胀
        kernel = np.ones((3, 3), np.uint8)
        diff = ndimage.binary_erosion(diff, structure=kernel).astype(np.uint8)
        diff = ndimage.binary_dilation(diff, structure=kernel).astype(np.uint8)

        # 寻找连通区域
        labels = measure.label(diff)
        regions = measure.regionprops(labels)

        current_points = []
        for region in regions:
            if region.area < 10:  # 过滤小区域
                continue
            y, x = region.centroid
            current_points.append((int(x), int(y)))

        # 更新轨迹
        new_trajectories = []
        for point in current_points:
            min_dist = float('inf')
            closest_traj = None
            for traj in trajectories:
                last_p = traj[-1]
                dist = (point[0] - last_p[0]) ** 2 + (point[1] - last_p[1]) ** 2
                if dist < min_dist:
                    min_dist = dist
                    closest_traj = traj
            if min_dist < 25:  # 距离阈值（像素）
                closest_traj.append(point)
                new_trajectories.append(closest_traj)
            else:
                new_trajectories.append([point])
        trajectories = new_trajectories

        prev_gray = current_gray
    return trajectories


def update(frame, features, labels_list, previous_labels_list):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)
    colors = ['red', 'green', 'blue']

    for idx, feat in enumerate(features):
        # Ensure frame is within the bounds of the path length
        if frame >= len(feat.path):
            feat.current_pos = feat.path[-1]  # Use the last position if frame exceeds path length
        else:
            feat.current_pos = feat.path[frame]  # Normal behavior if frame is within bounds

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
    ax.set_title(
        f"Frame {frame} | Overlap: {overlap:.1f} | Pos Change: {pos_change:.1f} | Intersections: {intersections}")
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')


