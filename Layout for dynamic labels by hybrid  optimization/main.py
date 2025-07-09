import random
from PIL import Image, ImageSequence, ImageDraw
from Dynamic_label_movement_planning import DynamicLabelOptimizer, paramsA2 as dynamic_params
from Global_spatiotemporal_joint_optimization import LabelOptimizer, paramsA1 as static_params, paramsA1
from load_json_data import load_trajectories_from_json, get_label_info_from_json
from initialize import initialize_features_from_data
from label import Label
from quality_evaluation import evaluate_label_layout_quality, calculate_paper_metrics

# 参数设置（合并静态和动态参数）
global_params = {
    **static_params,
    **dynamic_params,
    'max_x': 1000,  # 修改为坐标轴内最大x值
    'max_y': 1000,  # 修改为坐标轴内最大y值
    'min_x': 0,   # 添加最小x值
    'min_y': 0,   # 添加最小y值
}

def create_new_frame(labels, frame_size=(1000, 1000)):
    """创建新的空白帧并绘制标签"""
    # 创建新的空白图像
    frame = Image.new('RGB', frame_size, 'white')
    draw = ImageDraw.Draw(frame)
    
    # 绘制坐标轴
    # X轴
    draw.line([(50, 950), (950, 950)], fill='black', width=2)
    # Y轴
    draw.line([(50, 50), (50, 950)], fill='black', width=2)
    
    # 绘制刻度
    for i in range(0, 901, 100):
        # X轴刻度
        draw.line([(50 + i, 950), (50 + i, 940)], fill='black', width=1)
        draw.text((50 + i - 10, 955), str(i), fill='black')
        # Y轴刻度
        draw.line([(50, 950 - i), (60, 950 - i)], fill='black', width=1)
        draw.text((30, 950 - i - 5), str(i), fill='black')
    
    # 绘制标签
    for label in labels:
        # 确保标签位置在坐标轴内
        x = max(global_params['min_x'], min(global_params['max_x'], label.position[0]))
        y = max(global_params['min_y'], min(global_params['max_y'], label.position[1]))
        
        width = label.width
        height = label.length

        # 计算矩形坐标（左上角和右下角）
        left = x - height // 2
        top = y - width // 2
        right = x + height // 2
        bottom = y + width // 2

        # 确保矩形在坐标轴内
        left = max(global_params['min_x'], left)
        top = max(global_params['min_y'], top)
        right = min(global_params['max_x'], right)
        bottom = min(global_params['max_y'], bottom)

        # 绘制矩形
        draw.rectangle([left, top, right, bottom], outline='red', width=2)

        #  # 生成标签文本
        # label_text = f"Label{label.id}"
        
        # # 获取文本大小
        # text_bbox = draw.textbbox((0, 0), label_text)
        # text_width = text_bbox[2] - text_bbox[0]
        # text_height = text_bbox[3] - text_bbox[1]
        
        # # 绘制文本（居中）
        # draw.text((x - text_width//2, y - text_height//2), label_text, fill='red')


        # 绘制特征点
        feature_x = max(global_params['min_x'], min(global_params['max_x'], label.feature.position[0]))
        feature_y = max(global_params['min_y'], min(global_params['max_y'], label.feature.position[1]))
        draw.ellipse([feature_x-5, feature_y-5, feature_x+5, feature_y+5], 
                    fill=label.feature.color, outline='black')

        # 绘制连接线（从矩形中心到特征点中心）
        draw.line([(x, y), (feature_x, feature_y)], fill='blue', width=1)

    return frame

def main():
    # 指定JSON数据文件路径 - 可以选择不同的数据文件
    json_file_path = 'sample_generated.json' 
    
    # 从JSON文件加载轨迹数据
    positions_list = load_trajectories_from_json(json_file_path)

    # 从JSON文件获取标签信息
    label_info_list = get_label_info_from_json(json_file_path)

    # 使用轨迹数据初始化特征点
    features = initialize_features_from_data(
        positions_list,
        frame_interval=0.05
    )

    # 创建标签对象（使用JSON中的标签信息）
    labels = []
    for i, feature in enumerate(features):
        # 获取对应的标签信息
        
        label_info = label_info_list[i]
        label_length = label_info['length']
        label_width = label_info['width']
        label_text = label_info['text']
        
        # 确保初始位置在坐标轴内
        initial_x = min(global_params['max_x'], feature.position[0] + 50)
        initial_y = max(global_params['min_y'], min(global_params['max_y'], feature.position[1]))
        
        label = Label(
            id=feature.id,
            feature=feature,
            position=(initial_x, initial_y),
            length=label_length,
            width=label_width,
            velocity=(0, 0)
        )
        labels.append(label)
    

    # 全局静态优化
    static_optimizer = LabelOptimizer(labels, features, paramsA1, global_params['max_x'], global_params['max_y'])
    first_frame_position, all_joint_set_positions = static_optimizer.optimize()
    
    # 使用第一帧坐标更新标签位置
    for label in labels:
        if label.id in first_frame_position:
            # 确保位置在坐标轴内
            x = max(global_params['min_x'], min(global_params['max_x'], first_frame_position[label.id][0]))
            y = max(global_params['min_y'], min(global_params['max_y'], first_frame_position[label.id][1]))
            label.position = (x, y)
    
    print("全局静态优化完成")
    current_positions = first_frame_position
    velocities = {label.id: (0.0, 0.0) for label in labels}

    # 初始化动态优化器
    # 将静态优化器的约束转换为使用标签ID的格式
    initial_constraints = {}
    for feature_idx, constraint in static_optimizer.constraints.items():
        label_id = labels[feature_idx].id
        # 将极坐标约束转换为第一帧的位置约束
        if label_id in first_frame_position:
            initial_constraints[label_id] = first_frame_position[label_id]
    
    dynamic_optimizer = DynamicLabelOptimizer(
        labels=labels,
        features=features,
        params=dynamic_params,
        constraints=initial_constraints,  # 使用转换后的约束
        max_x=global_params['max_x'],
        max_y=global_params['max_y']
    )

    # 创建新的帧列表
    output_frames = []
    # 累加质量指标
    total_occ = 0.0
    total_int = 0.0
    total_dist = 0.0
    # 添加历史记录用于综合指标计算
    all_labels_history = []
    all_features_history = []
    num_frames = len(features[0].trajectory)  # 使用第一个特征的轨迹长度作为总帧数

    print(f"开始处理 {num_frames} 帧...")

    # 处理每一帧
    for frame_idx in range(num_frames):
        # 更新特征位置
        for feature in features:
            if frame_idx < len(feature.trajectory):
                # 确保特征位置在坐标轴内
                x = max(global_params['min_x'], min(global_params['max_x'], feature.trajectory[frame_idx][0]))
                y = max(global_params['min_y'], min(global_params['max_y'], feature.trajectory[frame_idx][1]))
                feature.position = (x, y)

        # 根据论文思想：更新当前帧的位置约束
        current_frame_constraints = {}
        for joint_set_info in all_joint_set_positions:
            if joint_set_info['frame'] == frame_idx:
                # 当前帧的 joint set 位置约束直接使用（已经是标签ID为键）
                current_frame_constraints.update(joint_set_info['positions'])
        
        # 更新动态优化器的约束
        if current_frame_constraints:
            dynamic_optimizer.constraints = current_frame_constraints
            print(f"Frame {frame_idx}: Updated constraints for labels {list(current_frame_constraints.keys())}")
        # else:
        #     # 如果当前帧没有新的约束，保持之前的约束不变
        #     print(f"Frame {frame_idx}: No new constraints, keeping previous constraints")

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
                time_delta=0.05,
                max_iter=1
            )

        current_positions = new_positions
        velocities = new_velocities

        # 更新标签位置
        for i, label_id in enumerate(current_positions):
            # 确保位置在坐标轴内
            x = max(global_params['min_x'], min(global_params['max_x'], current_positions[label_id][0]))
            y = max(global_params['min_y'], min(global_params['max_y'], current_positions[label_id][1]))
            labels[i].position = (x, y)
            labels[i].velocity = velocities[label_id]

        # 评估当前帧的质量并累加
        frame_metrics = evaluate_label_layout_quality(labels, features, frame_idx)
        total_occ += frame_metrics['occ']
        total_int += frame_metrics['int']
        total_dist += frame_metrics['dist']

        # 保存当前帧的标签和特征状态用于综合指标计算
        import copy
        labels_copy = copy.deepcopy(labels)
        features_copy = copy.deepcopy(features)
        all_labels_history.append(labels_copy)
        all_features_history.append(features_copy)

        # 创建新帧并添加到列表
        new_frame = create_new_frame(labels)
        output_frames.append(new_frame)

    # 计算平均质量指标
    final_avg_occ = total_occ / num_frames
    final_avg_int = total_int / num_frames
    final_avg_dist = total_dist / num_frames
    
    # 计算综合评价指标
    paper_metrics = calculate_paper_metrics(all_labels_history, all_features_history)
    
    # 构建质量指标字典
    quality_metrics = {
        'S_overlap': paper_metrics['s_overlap'],
        'S_position': paper_metrics['s_position'],
        'S_aesthetics': paper_metrics['s_aesthetics'],
        'S_angle_smoothness': paper_metrics['s_smoothness_angle'],
        'S_distance_smoothness': paper_metrics['s_smoothness_radius'],
        'total_frames': num_frames,
        'total_labels': len(labels)
    }
    
    # 输出所有评估指标
    print("\n=== 标签布局质量评估结果 ===")
    print(f"OCC: {final_avg_occ:.2f}")
    print(f"INT: {final_avg_int:.2f}") 
    print(f"DIST: {final_avg_dist:.1f}")
    print(f"S_overlap: {quality_metrics['S_overlap']:.2f}")
    print(f"S_position: {quality_metrics['S_position']:.2f}")
    print(f"S_aesthetics: {quality_metrics['S_aesthetics']:.2f}")
    print(f"S_angle_smoothness: {quality_metrics['S_angle_smoothness']:.2f}")
    print(f"S_distance_smoothness: {quality_metrics['S_distance_smoothness']:.2f}")
    print(f"总帧数: {quality_metrics['total_frames']}")
    print(f"总标签数: {quality_metrics['total_labels']}")
    
    

    # 保存GIF
    gif_output_path = 'output1.gif'
    output_frames[0].save(
        gif_output_path,
        save_all=True,
        append_images=output_frames[1:],
        duration=50,
        loop=0
    )
    print(f"已保存GIF到 {gif_output_path}")

if __name__ == '__main__':
    main()
