import random
from PIL import Image, ImageSequence, ImageDraw
from Dynamic_label_movement_planning import DynamicLabelOptimizer, paramsA2 as dynamic_params
from Global_spatiotemporal_joint_optimization import LabelOptimizer, paramsA1 as static_params, paramsA1
from generate_data import generate_trajectories
from initialize import initialize_features_from_data
from label import Label

# 参数设置（合并静态和动态参数）
global_params = {
    **static_params,
    **dynamic_params,
    'max_x': 1000,
    'max_y': 1000,
}

def create_new_frame(labels, frame_size=(1000, 1000)):
    """创建新的空白帧并绘制标签"""
    # 创建新的空白图像
    frame = Image.new('RGB', frame_size, 'white')
    draw = ImageDraw.Draw(frame)
    
 # 绘制坐标轴（X轴和Y轴）
    draw.line([(50, 950), (950, 950)], fill='black', width=2)  # X轴：0-1500
    draw.line([(50, 50), (50, 950)], fill='black', width=2)    # Y轴：0-1500

    # 设置刻度间隔为 300 单位（总范围 0-1500，共 5 个主刻度）
    interval = 300

    # 绘制刻度和标签
    for i in range(0, 1501, interval):
        # 计算像素位置（X轴和Y轴的转换比例：1500单位对应 900像素）
        x_pixel = 50 + (i * (900 / 1500))  # X轴：50是起点，900像素是总长度
        y_pixel = 50 + (i * (900 / 1500))  # Y轴：50是起点，900像素是总长度

        # 确保像素位置为整数
        x_pixel = int(round(x_pixel))
        y_pixel = int(round(y_pixel))

        # 绘制X轴刻度线和标签
        draw.line([(x_pixel, 950), (x_pixel, 940)], fill='black', width=1)
        draw.text((x_pixel - 15, 955), str(i), fill='black')

        # 绘制Y轴刻度线和标签（注意图像坐标系的Y轴方向）
        draw.line([(50, y_pixel), (60, y_pixel)], fill='black', width=1)
        draw.text((20, y_pixel - 5), str(i), fill='black')
    
    # 绘制标签
    for label in labels:
        x, y = map(int, label.position)
        width = label.width
        height = label.length

        # 计算矩形坐标（左上角和右下角）
        left = x - height // 2
        top = y - width // 2
        right = x + height // 2
        bottom = y + width // 2

        # 绘制矩形
        draw.rectangle([left, top, right, bottom], outline='red', width=2)

        # 绘制特征点
        feature_x, feature_y = map(int, label.feature.position)
        draw.ellipse([feature_x-5, feature_y-5, feature_x+5, feature_y+5], 
                    fill=label.feature.color, outline='black')

        # 绘制连接线（从矩形中心到特征点中心）
        draw.line([(x, y), (feature_x, feature_y)], fill='blue', width=1)

    return frame

def main():
    # 生成轨迹数据
    red_positions, green_positions, blue_positions = generate_trajectories()
    
    # 使用轨迹数据初始化特征点
    features = initialize_features_from_data(
        red_positions,
        green_positions,
        blue_positions,
        frame_interval=0.05
    )

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
    current_positions = first_frame_position
    velocities = {label.id: (0.0, 0.0) for label in labels}

    # 初始化动态优化器
    dynamic_optimizer = DynamicLabelOptimizer(
        labels=labels,
        features=features,
        params=dynamic_params,
        constraints=static_optimizer.constraints,
        max_x=global_params['max_x'],
        max_y=global_params['max_y']
    )

    # 创建新的帧列表
    output_frames = []
    num_frames = len(features[0].trajectory)  # 使用第一个特征的轨迹长度作为总帧数

    # 处理每一帧
    for frame_idx in range(num_frames):
        # 更新特征位置
        for feature in features:
            if frame_idx < len(feature.trajectory):
                feature.position = feature.trajectory[frame_idx]

        # 计算当前帧的标签位置
        if frame_idx == 0:
            current_positions = first_frame_position
            velocities = {label.id: (0.0, 0.0) for label in labels}
            new_positions, new_velocities = dynamic_optimizer.optimize_labels(
                initial_positions=current_positions,
                initial_velocities=velocities,
                time_delta=0.05,
                max_iter=1
            )
        else:
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
            labels[i].position = current_positions[label_id]
            labels[i].velocity = velocities[label_id]

        # 创建新帧并添加到列表
        new_frame = create_new_frame(labels)
        output_frames.append(new_frame)

    # 保存GIF
    gif_output_path = 'output.gif'
    output_frames[0].save(
        gif_output_path,
        save_all=True,
        append_images=output_frames[1:],
        duration=100,
        loop=0
    )
    print(f"已保存GIF到 {gif_output_path}")

if __name__ == '__main__':
    main()
