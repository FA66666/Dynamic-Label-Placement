import random
from PIL import Image, ImageSequence, ImageDraw
from Dynamic_label_movement_planning import DynamicLabelOptimizer, paramsA2 as dynamic_params
from Global_spatiotemporal_joint_optimization import LabelOptimizer, paramsA1 as static_params, paramsA1
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
    gif_path = 'input2.gif'
    features = initialize_features_from_gif(gif_path)

    # 创建标签对象（初始位置在特征右侧）
    labels = []
    for feature in features:
        initial_x = feature.position[0] + 50
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

        # 全局静态优化
    static_optimizer = LabelOptimizer(labels, features, paramsA1, global_params['max_x'], global_params['max_y'])
    first_frame_position, all_joint_set_positions = static_optimizer.optimize()
    
    # 使用第一帧坐标更新标签位置
    for label in labels:
        if label.id in first_frame_position:
            label.position = first_frame_position[label.id]
    
    print("全局静态优化完成")
    current_positions = first_frame_position  # 使用静态优化器返回的第一帧位置
    velocities = {label.id: (0.0, 0.0) for label in labels}  # 使用字典形式初始化速度

    # 初始化动态优化器
    print("labels", current_positions)
    dynamic_optimizer = DynamicLabelOptimizer(
        labels=labels,
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
    for frame_idx in range(len(frames)):  # 生成100帧GIF
        current_frame = frames[frame_idx % len(frames)]  # 循环使用现有帧

        # 更新特征位置（假设特征按轨迹移动）
        for feature in features:
            if frame_idx < len(feature.trajectory):
                feature.position = feature.trajectory[frame_idx]
        # 计算当前帧的标签位置
        if frame_idx == 0:
            # 第一帧使用静态优化结果
            current_positions = first_frame_position
            # 初始化动态优化器
            velocities = {label.id: (0.0, 0.0) for label in labels}  # 初始化速度为零
             # 对不足的标签执行动态优化
            new_positions, new_velocities = dynamic_optimizer.optimize_labels(
                initial_positions=current_positions,
                initial_velocities=velocities,
                time_delta=0.05,
                max_iter=1
            )
        else:
            # 对不足的标签执行动态优化
            new_positions, new_velocities = dynamic_optimizer.optimize_labels(
                initial_positions=current_positions,
                initial_velocities=velocities,
                time_delta=0.1,
                max_iter=1
            )

            current_positions = new_positions
            velocities = new_velocities

        # 更新标签位置
        for i, label_id in enumerate(current_positions):
            labels[i].position = current_positions[label_id]  # 使用 ID 从字典中获取位置
            labels[i].velocity = velocities[label_id]  # 使用 ID 从字典中获取速度
        print(frame_idx,"labels",current_positions)

        # 绘制当前帧的标签
        draw_labels_on_frame(current_frame, labels)

        # 将当前帧添加到GIF帧列表
        output_frames.append(current_frame)

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
        left = x - height // 2
        top = y - width // 2
        right = x + height // 2
        bottom = y + width // 2

        # 绘制矩形（红色边框，透明填充）
        draw.rectangle([left, top, right, bottom], outline='red', width=2)

        # 可选：绘制连接线（leader line）到特征中心
        feature_x, feature_y = label.feature.position
        draw.line([(x, y), (feature_x, feature_y)], fill='blue', width=1)

    return frame

if __name__ == '__main__':
    main()
