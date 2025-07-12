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
        x =label.position[0]
        y = label.position[1]
        
        width = label.width
        height = label.length
        left = x - height // 2
        top = y - width // 2
        right = x + height // 2
        bottom = y + width // 2
        
        draw.rectangle([left, top, right, bottom], outline='red', width=2)

        feature_x = label.feature.position[0]
        feature_y = label.feature.position[1]

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
        
        initial_x =  feature.position[0] + 30
        initial_y = feature.position[1]
        
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
            x =  first_frame_position[label.id][0]
            y = first_frame_position[label.id][1]
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
    )

    output_frames = []
    
    total_occ = 0.0
    total_int = 0.0
    
    all_labels_history = []
    all_features_history = []
    num_frames = len(features[0].trajectory)  
    
    for frame_idx in range(num_frames):
        for feature in features:
            if frame_idx < len(feature.trajectory):     
                x = feature.trajectory[frame_idx][0]
                y =  feature.trajectory[frame_idx][1]
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
                time_delta=0.05
            )
        else:
            new_positions, new_velocities = dynamic_optimizer.optimize_labels(
                initial_positions=current_positions,
                initial_velocities=velocities,
                time_delta=0.05
            )
        current_positions = new_positions
        velocities = new_velocities
       
        for i, label_id in enumerate(current_positions):
            x = current_positions[label_id][0]
            y =  current_positions[label_id][1]
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

if __name__ == '__main__':
    main()
