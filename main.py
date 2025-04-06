from PIL import Image, ImageSequence, ImageDraw

from Dynamic_label_movement_planning import DynamicLabelOptimizer, params as dynamic_params
from Global_spatiotemporal_joint_optimization import LabelOptimizer, params as static_params
from initialize import initialize_features_from_gif
from label import Label

# 参数设置（合并静态和动态参数）
global_params = {
    **static_params,
    **dynamic_params,
    'max_x': 1000,
    'max_y': 1000,
}

def main():
    # 读取GIF并初始化特征点
    gif_path = 'input.gif'
    features = initialize_features_from_gif(gif_path)

    # 创建标签对象（初始位置在特征右侧）
    labels = []
    for feature in features:
        initial_x = feature.position[0] + feature.radius + 20
        initial_y = feature.position[1]
        label = Label(
            id=feature.id,
            feature=feature,
            position=(initial_x, initial_y),
            length=40,
            width=16,
            velocity=(0, 0)
        )
        labels.append(label)

    # 全局静态优化（仅针对第一帧）
    static_optimizer = LabelOptimizer(labels, features, global_params['max_x'], global_params['max_y'])
    optimized_labels = static_optimizer.optimize()

    # 初始化动态优化所需的变量
    current_positions = [label.position for label in optimized_labels]
    velocities = [(0.0, 0.0) for _ in labels]  # 初始速度设为零

    # 初始化动态优化器
    dynamic_optimizer = DynamicLabelOptimizer(
        labels=optimized_labels,
        features=features,
        params=dynamic_params,
        constraints=static_optimizer.constraints,
        max_x=global_params['max_x'],
        max_y=global_params['max_y']
    )

    # 获取GIF的所有帧
    gif = Image.open(gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

    output_frames = []  # 这里收集每一帧

    # 处理每一帧并添加到GIF帧列表
    for frame_idx in range(100):  # 生成100帧GIF
        current_frame = frames[frame_idx % len(frames)]  # 循环使用现有帧

        # 更新特征位置（假设特征按轨迹移动）
        for feature in features:
            if frame_idx < len(feature.trajectory):
                feature.position = feature.trajectory[frame_idx]

        # 计算当前帧的标签位置
        if frame_idx == 0:
            # 第一帧使用静态优化结果
            current_positions = [label.position for label in optimized_labels]
            velocities = [(0, 0) for _ in labels]
        else:
            # 后续帧使用动态优化
            new_positions, new_velocities = dynamic_optimizer.optimize_labels(
                initial_positions=current_positions,
                initial_velocities=velocities,
                time_delta=0.1,
                max_iter=100
            )
            current_positions = new_positions
            velocities = new_velocities

        # 更新标签位置
        for i, pos in enumerate(current_positions):
            labels[i].position = pos
            labels[i].velocity = velocities[i]

        # 绘制当前帧的标签
        draw_labels_on_frame(current_frame, labels)

        # 将当前帧添加到GIF帧列表
        output_frames.append(current_frame)

        print(f"Processed frame {frame_idx + 1}")

    # 保存为GIF文件，设置每帧之间的时间间隔为100ms
    gif_output_path = 'output.gif'
    output_frames[0].save(
        gif_output_path,
        save_all=True,
        append_images=output_frames[1:],
        duration=100,  # 每帧显示时间为100毫秒
        loop=0  # 动画循环播放
    )
    print(f"Saved GIF to {gif_output_path}")

def draw_labels_on_frame(frame, labels):
    """在图像帧上绘制标签矩形"""
    draw = ImageDraw.Draw(frame)
    for label in labels:
        x, y = map(int, label.position)
        width = label.width
        height = label.length

        # 计算矩形坐标（左上角和右下角）
        left = x - width // 2
        top = y - height // 2
        right = x + width // 2
        bottom = y + height // 2

        # 绘制矩形（红色边框，透明填充）
        draw.rectangle([left, top, right, bottom], outline='red', width=2)

        # 可选：绘制连接线（leader line）到特征中心
        feature_x, feature_y = label.feature.position
        draw.line([(x, y), (feature_x, feature_y)], fill='blue', width=1)

    return frame

if __name__ == '__main__':
    main()
