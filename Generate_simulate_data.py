import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


class Feature:
    def __init__(self, id, path):
        self.id = id
        self.path = np.array(path)
        self.current_pos = None
        self.velocity = np.array([0.0, 0.0])
        self.path_history = []  # 路径历史仍保留，但不再使用

    def update_velocity(self, previous_position, current_position, delta_time):
        self.velocity = (current_position - previous_position) / delta_time

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

def update(frame, features):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    colors = ['red', 'green', 'blue']

    for idx, feat in enumerate(features):
        feat.current_pos = feat.path[frame]
        if len(feat.path_history) == 0 or not np.array_equal(feat.path_history[-1], feat.current_pos):
            feat.path_history.append(feat.current_pos.copy())  # 保留路径历史数据（未使用）
        if frame > 0:
            feat.update_velocity(feat.path[frame - 1], feat.current_pos, 0.1)

        # 仅保留当前点的绘制
        ax.scatter(
            feat.current_pos[0],
            feat.current_pos[1],
            color=colors[idx],
            s=50,
            zorder=3
        )

    # ax.set_title(f"Frame {frame}")
    ax.axis('off')  # 隐藏坐标轴和边框
    ax.set_aspect('equal')

if __name__ == "__main__":
    features = generate_simulation_data(num_features=3, num_frames=50)
    fig = plt.figure(figsize=(10, 10))

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(features[0].path),
        fargs=(features,),
        interval=200
    )

    anim.save("input.gif", writer='pillow', fps=15, dpi=100)