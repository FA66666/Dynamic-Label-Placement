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
    gif_path = 'input.gif'
    features = initialize_features_from_gif(gif_path)

    # 创建标签对象（初始位置在特征右侧）
    labels = []
    for feature in features:
        initial_x = feature.position[0] 
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

    # 设置第一帧标签位置
    first_frame_positions = {0: {label.id: label.position for label in labels}}  # 保存第一帧标签位置

        # 全局静态优化
    static_optimizer = LabelOptimizer(labels, features, paramsA1, global_params['max_x'], global_params['max_y'])
    first_frame_positions_dict, all_joint_set_positions = static_optimizer.optimize()
    
    # 使用第一帧坐标更新标签位置
    for label in labels:
        if label.id in first_frame_positions_dict:
            label.position = first_frame_positions_dict[label.id]
    
    # 使用更新后的标签重新初始化静态优化器
    static_optimizer = LabelOptimizer(labels, features, paramsA1, global_params['max_x'], global_params['max_y'])
    _, all_joint_set_positions = static_optimizer.optimize()
    
    print("全局静态优化完成")
    print(_)
    # 初始化动态优化所需的变量
    current_positions = _  # 使用静态优化器返回的第一帧位置
    print(current_positions)
    velocities = {label.id: (0.0, 0.0) for label in labels}  # 使用字典形式初始化速度

    # 初始化动态优化器
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

    temp_positions = _

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
            current_positions = _
            velocities = {label.id: (0.0, 0.0) for label in labels}  # 初始化速度为零
        else:
            # # 检查当前帧是否在 all_joint_set_positions 中的帧内
            # joint_set_positions = None
            # for joint_set in all_joint_set_positions:
            #     if frame_idx == joint_set['frame']:
            #         joint_set_positions = joint_set['positions']
            #         break

            # if joint_set_positions:
            #     # 如果当前帧在 all_joint_set_positions 的帧内，使用 all_joint_set_positions 中的结果
            #     if len(joint_set_positions) < 3:
            #         # 把 joint_set_position 中与 temp_position 的 id 相同的 label 信息复制到 temp_position 中
            #         for label_id in joint_set_positions:
            #             if label_id in temp_positions:
            #                 # 保持已存在标签 ID 的位置信息
            #                 temp_positions[label_id] = joint_set_positions[label_id]
            #     current_positions = joint_set_positions
            # else:
                # 否则，继续使用力导向优化
                if len(current_positions) < 3:
                    current_positions = temp_positions

                # 对不足的标签执行动态优化
                new_positions, new_velocities = dynamic_optimizer.optimize_labels(
                    initial_positions=current_positions,
                    initial_velocities=velocities,
                    time_delta=0.1,
                    max_iter=1000
                )

                # 如果新位置有缺失，暂存并更新
                missing_labels = [label.id for label in labels if label.id not in new_positions]
                if missing_labels:
                    for label_id in missing_labels:
                        new_positions[label_id] = current_positions[label_id]

                current_positions = new_positions
                velocities = new_velocities

        # 更新标签位置
        for i, label_id in enumerate(current_positions):
            labels[i].position = current_positions[label_id]  # 使用 ID 从字典中获取位置
            labels[i].velocity = velocities[label_id]  # 使用 ID 从字典中获取速度

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
