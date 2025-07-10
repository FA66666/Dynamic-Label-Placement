from PIL import Image, ImageSequence, ImageDraw
from Dynamic_label_movement_planning import DynamicLabelOptimizer, paramsA2 as dynamic_params
from Global_spatiotemporal_joint_optimization import LabelOptimizer, paramsA1 as static_params, paramsA1
from load_json_data import load_trajectories_from_json, get_label_info_from_json
from initialize import initialize_features_from_data
from label import Label
from quality_evaluation import evaluate_label_layout_quality, calculate_paper_metrics
import copy
import math

global_params = {
    **static_params,
    **dynamic_params,
    'max_x': 1000,  
    'max_y': 1000,  
    'min_x': 0,   
    'min_y': 0,   
}

def draw_force_arrow(draw, start_pos, force, scale=5.0, color='red', width=2):
    """绘制力的箭头"""
    fx, fy = force
    start_x, start_y = start_pos
    
    # 计算力的大小
    magnitude = math.hypot(fx, fy)
    if magnitude < 1e-6:
        return
    
    # 缩放力的显示长度
    end_x = start_x + fx * scale
    end_y = start_y + fy * scale
    
    # 绘制箭头主线
    draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=width)
    
    # 绘制箭头头部
    if magnitude > 1e-6:
        # 计算箭头方向
        angle = math.atan2(fy, fx)
        arrow_length = 8
        arrow_angle = math.pi / 6  # 30度
        
        # 箭头两边的点
        arrow1_x = end_x - arrow_length * math.cos(angle - arrow_angle)
        arrow1_y = end_y - arrow_length * math.sin(angle - arrow_angle)
        arrow2_x = end_x - arrow_length * math.cos(angle + arrow_angle)
        arrow2_y = end_y - arrow_length * math.sin(angle + arrow_angle)
        
        draw.line([(end_x, end_y), (arrow1_x, arrow1_y)], fill=color, width=width)
        draw.line([(end_x, end_y), (arrow2_x, arrow2_y)], fill=color, width=width)

def create_force_visualization_frame(labels, features, dynamic_optimizer, label_positions, velocities, force_type, frame_size=(1000, 1000)):
    """创建特定力类型的可视化帧"""
    frame = Image.new('RGB', frame_size, 'white')
    draw = ImageDraw.Draw(frame)
    
    # 绘制坐标轴
    draw.line([(50, 950), (950, 950)], fill='black', width=2)
    draw.line([(50, 50), (50, 950)], fill='black', width=2)
    
    for i in range(0, 901, 100): 
        draw.line([(50 + i, 950), (50 + i, 940)], fill='black', width=1)
        draw.text((50 + i - 10, 955), str(i), fill='black')
        draw.line([(50, 950 - i), (60, 950 - i)], fill='black', width=1)
        draw.text((30, 950 - i - 5), str(i), fill='black')
    
    # 绘制标签和特征点
    for i, label in enumerate(labels):
        x = max(global_params['min_x'], min(global_params['max_x'], label.position[0]))
        y = max(global_params['min_y'], min(global_params['max_y'], label.position[1])) 
        
        width = label.width
        height = label.length
        left = x - height // 2
        top = y - width // 2
        right = x + height // 2
        bottom = y + width // 2

        left = max(global_params['min_x'], left)
        top = max(global_params['min_y'], top)
        right = min(global_params['max_x'], right)
        bottom = min(global_params['max_y'], bottom)
        
        draw.rectangle([left, top, right, bottom], outline='gray', width=1)

        feature_x = max(global_params['min_x'], min(global_params['max_x'], label.feature.position[0]))
        feature_y = max(global_params['min_y'], min(global_params['max_y'], label.feature.position[1]))

        draw.ellipse([feature_x-3, feature_y-3, feature_x+3, feature_y+3], 
                    fill=label.feature.color, outline='black')
        draw.line([(x, y), (feature_x, feature_y)], fill='lightgray', width=1)
    
    # 根据力类型绘制相应的力
    force_colors = {
        'label_repulsion': 'red',
        'feature_repulsion': 'orange', 
        'pulling': 'green',
        'friction': 'blue',
        'time_constraint': 'purple',
        'space_constraint': 'brown',
        'resultant': 'black'
    }
    
    color = force_colors.get(force_type, 'black')
    
    for i, label in enumerate(labels):
        if i not in label_positions:
            continue
            
        x, y = label_positions[i]
        
        if force_type == 'label_repulsion':
            # 标签间排斥力
            for j in range(len(labels)):
                if i != j:
                    force = dynamic_optimizer.compute_label_label_repulsion(i, j, label_positions)
                    draw_force_arrow(draw, (x, y), force, scale=2.0, color=color)
                    
        elif force_type == 'feature_repulsion':
            # 标签-特征排斥力
            force = dynamic_optimizer.compute_label_feature_repulsion(i, label_positions)
            draw_force_arrow(draw, (x, y), force, scale=2.0, color=color)
            
        elif force_type == 'pulling':
            # 拉力
            force = dynamic_optimizer.compute_pulling_force(i, label_positions)
            draw_force_arrow(draw, (x, y), force, scale=2.0, color=color)
            
        elif force_type == 'friction':
            # 摩擦力
            force = dynamic_optimizer.compute_friction(i, velocities)
            draw_force_arrow(draw, (x, y), force, scale=10.0, color=color)
            
        elif force_type == 'time_constraint':
            # 时间约束力
            total_fx, total_fy = 0.0, 0.0
            for j in range(len(features)):
                if i != j:
                    fx, fy = dynamic_optimizer.compute_time_constraint(i, j, label_positions)
                    total_fx += fx
                    total_fy += fy
            draw_force_arrow(draw, (x, y), (total_fx, total_fy), scale=5.0, color=color)
            
        elif force_type == 'space_constraint':
            # 空间约束力
            force = dynamic_optimizer.compute_space_constraint(i, label_positions)
            draw_force_arrow(draw, (x, y), force, scale=2.0, color=color)
            
        elif force_type == 'resultant':
            # 合力
            force = dynamic_optimizer.compute_resultant_force(i, label_positions, velocities)
            draw_force_arrow(draw, (x, y), force, scale=1.0, color=color)
    
    # 添加标题
    draw.text((50, 20), f"Force Type: {force_type}", fill='black')
    
    return frame

def create_new_frame(labels, frame_size=(1000, 1000), dynamic_optimizer=None):
    frame = Image.new('RGB', frame_size, 'white')
    draw = ImageDraw.Draw(frame)
    draw.line([(50, 950), (950, 950)], fill='black', width=2)
    draw.line([(50, 50), (50, 950)], fill='black', width=2)
    
    for i in range(0, 901, 100): 
        draw.line([(50 + i, 950), (50 + i, 940)], fill='black', width=1)
        draw.text((50 + i - 10, 955), str(i), fill='black')
        draw.line([(50, 950 - i), (60, 950 - i)], fill='black', width=1)
        draw.text((30, 950 - i - 5), str(i), fill='black')
    
    
    for i, label in enumerate(labels):
        x = max(global_params['min_x'], min(global_params['max_x'], label.position[0]))
        y = max(global_params['min_y'], min(global_params['max_y'], label.position[1])) 
        
        width = label.width
        height = label.length
        left = x - height // 2
        top = y - width // 2
        right = x + height // 2
        bottom = y + width // 2

        left = max(global_params['min_x'], left)
        top = max(global_params['min_y'], top)
        right = min(global_params['max_x'], right)
        bottom = min(global_params['max_y'], bottom)
        
        draw.rectangle([left, top, right, bottom], outline='red', width=2)

        feature_x = max(global_params['min_x'], min(global_params['max_x'], label.feature.position[0]))
        feature_y = max(global_params['min_y'], min(global_params['max_y'], label.feature.position[1]))

        draw.ellipse([feature_x-5, feature_y-5, feature_x+5, feature_y+5], 
                    fill=label.feature.color, outline='black')
        draw.line([(x, y), (feature_x, feature_y)], fill='blue', width=1)
    return frame

def main():
    json_file_path = 'sample_generated.json' 
    positions_list = load_trajectories_from_json(json_file_path) 
    label_info_list = get_label_info_from_json(json_file_path)
    features = initialize_features_from_data(
        positions_list,
        frame_interval=0.05
    )

    labels = []
    for i, feature in enumerate(features):  
        label_info = label_info_list[i]
        label_length = label_info['length']
        label_width = label_info['width']
        
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
    
    static_optimizer = LabelOptimizer(labels, features, paramsA1, global_params['max_x'], global_params['max_y'])
    first_frame_position, all_joint_set_positions = static_optimizer.optimize()
    
    for label in labels:
        if label.id in first_frame_position:
            x = max(global_params['min_x'], min(global_params['max_x'], first_frame_position[label.id][0]))
            y = max(global_params['min_y'], min(global_params['max_y'], first_frame_position[label.id][1]))
            label.position = (x, y)
    
    print("全局静态优化完成")
    current_positions = first_frame_position
    velocities = {label.id: (0.0, 0.0) for label in labels}
 
    initial_constraints = {}
    for feature_idx, constraint in static_optimizer.constraints.items():
        label_id = labels[feature_idx].id 
        if label_id in first_frame_position:
            initial_constraints[label_id] = first_frame_position[label_id]
    
    dynamic_optimizer = DynamicLabelOptimizer(
        labels=labels,
        features=features,
        params=dynamic_params,
        constraints=initial_constraints,  
        max_x=global_params['max_x'],
        max_y=global_params['max_y']
    )

    output_frames = []
    
    # 为每种力类型创建帧存储
    force_types = ['label_repulsion', 'feature_repulsion', 'pulling', 'friction', 'time_constraint', 'space_constraint', 'resultant']
    force_frames = {force_type: [] for force_type in force_types}
    
    total_occ = 0.0
    total_int = 0.0
    
    all_labels_history = []
    all_features_history = []
    num_frames = len(features[0].trajectory)  
    
    for frame_idx in range(num_frames):
        for feature in features:
            if frame_idx < len(feature.trajectory):     
                x = max(global_params['min_x'], min(global_params['max_x'], feature.trajectory[frame_idx][0]))
                y = max(global_params['min_y'], min(global_params['max_y'], feature.trajectory[frame_idx][1]))
                feature.position = (x, y)
        current_frame_constraints = {}
        for joint_set_info in all_joint_set_positions:
            if joint_set_info['frame'] == frame_idx:
                
                current_frame_constraints.update(joint_set_info['positions'])
        
        # 实现约束持久化：如果没有新的交集，保持之前的约束
        if current_frame_constraints:
            # 有新的约束，更新现有约束
            dynamic_optimizer.constraints.update(current_frame_constraints)
            print(f"Frame {frame_idx}: Updated constraints for labels {list(current_frame_constraints.keys())}")
        # 如果没有新约束，保持现有约束不变（即论文要求的持久化）
    
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
       
        for i, label_id in enumerate(current_positions):
            x = max(global_params['min_x'], min(global_params['max_x'], current_positions[label_id][0]))
            y = max(global_params['min_y'], min(global_params['max_y'], current_positions[label_id][1]))
            labels[i].position = (x, y)
            labels[i].velocity = velocities[label_id]
   
        frame_metrics = evaluate_label_layout_quality(labels, features, frame_idx)
        # 逐帧累加OCC 与 INT
        total_occ += frame_metrics['occ']
        total_int += frame_metrics['int']

        labels_copy = copy.deepcopy(labels)
        features_copy = copy.deepcopy(features)
        all_labels_history.append(labels_copy)
        all_features_history.append(features_copy)

        # 创建普通帧
        new_frame = create_new_frame(labels, dynamic_optimizer=dynamic_optimizer)
        output_frames.append(new_frame)
        
        # 创建力可视化帧
        # 首先准备label_positions字典，将label索引映射到位置
        frame_label_positions = {}
        frame_velocities = {}
        for i, label in enumerate(labels):
            frame_label_positions[i] = label.position
            frame_velocities[i] = label.velocity
        
        # 为每种力类型创建可视化帧
        for force_type in force_types:
            force_frame = create_force_visualization_frame(
                labels, features, dynamic_optimizer, 
                frame_label_positions, frame_velocities, force_type
            )
            force_frames[force_type].append(force_frame)

    # 计算评估指标
    final_avg_occ = total_occ / num_frames
    final_avg_int = total_int / num_frames
    
    paper_metrics = calculate_paper_metrics(all_labels_history, all_features_history)
    quality_metrics = {
        'S_overlap': paper_metrics['s_overlap'],
        'S_position': paper_metrics['s_position'],
        'S_aesthetics': paper_metrics['s_aesthetics'],
        'S_angle_smoothness': paper_metrics['s_smoothness_angle'],
        'S_distance_smoothness': paper_metrics['s_smoothness_radius'],
        'total_frames': num_frames,
        'total_labels': len(labels)
    }
    
    
    print("\n=== 标签布局质量评估结果 ===")
    print(f"OCC: {final_avg_occ:.2f}")
    print(f"INT: {final_avg_int:.2f}") 
    print(f"S_overlap: {quality_metrics['S_overlap']:.2f}")
    print(f"S_position: {quality_metrics['S_position']:.2f}")
    print(f"S_aesthetics: {quality_metrics['S_aesthetics']:.2f}")
    print(f"S_angle_smoothness: {quality_metrics['S_angle_smoothness']:.2f}")
    print(f"S_distance_smoothness: {quality_metrics['S_distance_smoothness']:.2f}")
    print(f"总帧数: {quality_metrics['total_frames']}")
    print(f"总标签数: {quality_metrics['total_labels']}")
    
    # 保存普通动图
    gif_output_path = 'output1.gif'
    output_frames[0].save(
        gif_output_path,
        save_all=True,
        append_images=output_frames[1:],
        duration=50,
        loop=0
    )
    print(f"已保存普通动图到 {gif_output_path}")
    
    # 保存每种力的可视化动图
    for force_type in force_types:
        if force_frames[force_type]:
            force_gif_path = f'force_{force_type}.gif'
            force_frames[force_type][0].save(
                force_gif_path,
                save_all=True,
                append_images=force_frames[force_type][1:],
                duration=50,  # 稍慢一些以便观察
                loop=0
            )
            print(f"已保存{force_type}力可视化动图到 {force_gif_path}")

if __name__ == '__main__':
    main()
