"""
标签布局质量评估模块

该模块实现了标签布局质量评估的全部7个指标：
- OCC (Occlusion): 遮挡指标
- INT (Intersection): 交叉指标
- S_overlap: 重叠度指标
- S_position: 位置距离指标
- S_aesthetics: 美观度指标（引导线交叉次数）
- S_angle_smoothness: 角度平滑度指标
- S_distance_smoothness: 距离平滑度指标

主要函数：
- evaluate_comprehensive_quality: 计算全部7个指标的综合评估函数
- evaluate_single_frame_quality: 向后兼容函数，返回OCC和INT
- evaluate_layout_quality: 向后兼容函数，返回论文中的5个指标
"""

import math

def evaluate_single_frame_quality(sim_engine, frame_index=None):
    """
    评估单帧标签布局的两个核心质量指标（为保持向后兼容性保留）
    
    现在调用综合评估函数并返回OCC和INT指标
    """
    # 为了保持向后兼容，创建单帧数据并调用综合评估函数
    single_frame_data = [{
        'frame': frame_index or 0,
        'points': {}
    }]
    
    # 从当前仿真引擎状态构建帧数据
    for feature_id, feature in sim_engine.features.items():
        label = sim_engine.labels[feature_id]
        single_frame_data[0]['points'][str(feature_id)] = {
            'anchor': [feature.x, feature.y],
            'text': label.text,
            'size': [label.width, label.height]
        }
    
    # 调用综合评估函数
    all_metrics = evaluate_comprehensive_quality(sim_engine, single_frame_data)
    
    # 返回OCC和INT指标
    return {
        'occ': all_metrics['OCC'],
        'int': all_metrics['INT']
    }


def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    判断两条线段是否相交
    
    使用CCW（逆时针）算法检测线段相交
    
    参数：
        x1, y1, x2, y2: 第一条线段的两个端点坐标
        x3, y3, x4, y4: 第二条线段的两个端点坐标
        
    返回：
        bool: 如果两条线段相交返回True，否则返回False
    """
    def ccw(Ax, Ay, Bx, By, Cx, Cy):
        return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
    
    return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and \
           ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)


def evaluate_comprehensive_quality(simulation_engine, frames_data):
    """
    综合评估标签布局质量 - 计算全部7个指标
    
    计算七个主要指标：
    - OCC: 遮挡指标，测量每个标签平均遮挡的对象和标签数量，值越小越好
    - INT: 交叉指标，测量每条引导线平均被其他引导线交叉的次数，值越小越好
    - S_overlap: 重叠度指标 (pixel²)
    - S_position: 位置距离指标，计算所有帧中标签到特征点距离的平均值 (pixel)  
    - S_aesthetics: 美观度指标（引导线交叉次数）
    - S_angle_smoothness: 角度平滑度指标 (degree)
    - S_distance_smoothness: 距离平滑度指标 (pixel)
    
    参数：
        simulation_engine: 仿真引擎实例
        frames_data: 帧数据列表
        
    返回：
        dict: 包含全部7项质量指标的字典
    """
    from force_calculator import ForceCalculator
    from simulation import SimulationEngine
    
    M = len(frames_data)  # 帧数
    N = len(simulation_engine.labels)  # 标签数
    
    # 按论文公式初始化累加器
    total_label_overlap = 0  # ∑∑O_{i,j}
    total_feature_overlap = 0  # ∑∑P_{i,j}
    total_position_distance = 0  # 总的位置距离
    total_position_count = 0  # 有效的标签-特征对计数
    total_intersections = 0  # ∑∑I_i
    total_angle_changes = 0  # ∑∑|θ_{i,k} - θ_{i,k+1}|
    total_distance_changes = 0  # ∑∑|r_{i,k} - r_{i,k+1}|
    
    # OCC和INT累计值
    total_occ_sum = 0
    total_int_sum = 0
    
    prev_angles = {}
    prev_distances = {}
    
    force_calculator = ForceCalculator(simulation_engine.params)
    sim_engine = SimulationEngine(simulation_engine.params, force_calculator)
    sim_engine.initialize_from_data(frames_data[0])
    
    for k, frame in enumerate(frames_data):
        sim_engine.update_feature_positions(frame, simulation_engine.params['time_step'])
        sim_engine.step(simulation_engine.params['time_step'])
        labels = list(sim_engine.labels.values())
        features = list(sim_engine.features.values())
        
        # ===== 计算当前帧的OCC和INT指标 =====
        frame_occ, frame_int = _calculate_occ_int_for_frame(labels, features, sim_engine)
        total_occ_sum += frame_occ
        total_int_sum += frame_int
        
        # ===== 1. S_Overlap计算 =====
        frame_label_overlap = 0
        frame_feature_overlap = 0
        
        # 标签-标签重叠 O_{i,j}
        for i in range(N):
            for j in range(N):
                if i != j:
                    l1, l2 = labels[i], labels[j]
                    x_overlap = max(0, min(l1.x+l1.width, l2.x+l2.width) - max(l1.x, l2.x))
                    y_overlap = max(0, min(l1.y+l1.height, l2.y+l2.height) - max(l1.y, l2.y))
                    frame_label_overlap += x_overlap * y_overlap
        
        # 标签-特征重叠 P_{i,j}
        for i in range(N):
            for j in range(N):
                # 根据论文公式，P_{i,j}包括所有标签i与特征j的重叠（包括i=j的情况）
                label = labels[i]
                feature = features[j]
                
                lx_min, ly_min = label.x, label.y
                lx_max, ly_max = label.x + label.width, label.y + label.height
                feature_radius = getattr(feature, 'radius', 1)  # 默认半径1
                fx_min, fy_min = feature.x - feature_radius, feature.y - feature_radius
                fx_max, fy_max = feature.x + feature_radius, feature.y + feature_radius
                
                x_overlap = max(0, min(lx_max, fx_max) - max(lx_min, fx_min))
                y_overlap = max(0, min(ly_max, fy_max) - max(ly_min, fy_min))
                frame_feature_overlap += x_overlap * y_overlap
        
        total_label_overlap += frame_label_overlap
        total_feature_overlap += frame_feature_overlap
        
        # ===== 2. S_Position计算 =====
        frame_position_sum = 0
        for i in range(N):
            label = labels[i]
            if label.id in sim_engine.features:
                feature = sim_engine.features[label.id]
                distance = math.hypot(label.center_x - feature.x, label.center_y - feature.y)
                frame_position_sum += distance
                total_position_count += 1
        
        total_position_distance += frame_position_sum
        
        # ===== 3. S_Aesthetics计算 =====
        frame_intersections = 0
        for i in range(N):
            label_i = labels[i]
            feature_i = sim_engine.features[label_i.id]
            intersections_count = 0
            
            for j in range(N):
                if i != j:
                    label_j = labels[j]
                    feature_j = sim_engine.features[label_j.id]
                    
                    # 检查标签i的引导线与标签j的引导线是否相交
                    if lines_intersect(label_i.center_x, label_i.center_y, feature_i.x, feature_i.y,
                                     label_j.center_x, label_j.center_y, feature_j.x, feature_j.y):
                        intersections_count += 1
            
            frame_intersections += intersections_count
        
        total_intersections += frame_intersections
        
        # ===== 4. S_Smoothness计算 =====
        current_angles = {}
        current_distances = {}
        
        for i in range(N):
            label = labels[i]
            feature = sim_engine.features[label.id]
            
            # 计算角度 θ_i (特征点到标签的角度)
            angle = math.atan2(label.center_y - feature.y, label.center_x - feature.x)
            distance = math.hypot(label.center_x - feature.x, label.center_y - feature.y)
            
            current_angles[label.id] = angle
            current_distances[label.id] = distance
            
            # 只有k>0时才计算变化
            if k > 0 and label.id in prev_angles:
                # S^θ_Smoothness = (1/(M-1)) * ∑_{k=1}^{M-1} ∑_{i=1}^N |θ_{i,k} - θ_{i,k+1}|
                angle_diff = abs(angle - prev_angles[label.id])
                # 处理角度跳跃 (-π, π)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                total_angle_changes += angle_diff
                
                # S^r_Smoothness = (1/(M-1)) * ∑_{k=1}^{M-1} ∑_{i=1}^N |r_{i,k} - r_{i,k+1}|
                distance_diff = abs(distance - prev_distances[label.id])
                total_distance_changes += distance_diff
        
        prev_angles = current_angles.copy()
        prev_distances = current_distances.copy()
    
    # ===== 计算最终指标 =====
    # OCC和INT的平均值
    avg_occ = total_occ_sum / M if M > 0 else 0
    avg_int = total_int_sum / M if M > 0 else 0
    
    # 论文公式指标
    S_overlap = (total_label_overlap + total_feature_overlap) / M
    S_position = total_position_distance / M
    S_aesthetics = total_intersections / M / N
    S_angle_smoothness = math.degrees(total_angle_changes / (M - 1) / N) if M > 1 else 0
    S_distance_smoothness = total_distance_changes / (M - 1) / N if M > 1 else 0
    
    return {
        'OCC': avg_occ,
        'INT': avg_int,
        'S_overlap': S_overlap,
        'S_position': S_position, 
        'S_aesthetics': S_aesthetics,
        'S_angle_smoothness': S_angle_smoothness,
        'S_distance_smoothness': S_distance_smoothness,
        'total_frames': M,
        'total_labels': N,
        'total_position_count': total_position_count,
        'raw_label_overlap': total_label_overlap,
        'raw_feature_overlap': total_feature_overlap
    }


def _calculate_occ_int_for_frame(labels, features, sim_engine):
    """
    计算单帧的OCC和INT指标的辅助函数
    """
    N = len(labels)
    
    # 1. OCC (遮挡指标) - 计算每个标签遮挡其他对象的平均数量
    total_occlusions = 0
    
    for i, label_i in enumerate(labels):
        occlusions_by_label_i = 0
        
        # 计算label_i遮挡了多少其他标签（矩形与矩形的重叠检测）
        for j, label_j in enumerate(labels):
            if i != j:
                # 使用轴对齐边界框(AABB)检测两个标签是否重叠
                x_overlap = max(0, min(label_i.x + label_i.width, label_j.x + label_j.width) - 
                              max(label_i.x, label_j.x))
                y_overlap = max(0, min(label_i.y + label_i.height, label_j.y + label_j.height) - 
                              max(label_i.y, label_j.y))
                if x_overlap > 0 and y_overlap > 0:
                    occlusions_by_label_i += 1
        
        # 计算label_i遮挡了多少其他特征点（矩形与圆形的重叠检测）
        for feature in features:
            if feature.id != label_i.id:  # 排除标签自己对应的特征点
                # 获取特征点的可视半径（默认为1像素）
                feature_radius = getattr(feature, 'radius', 1)
                
                # 计算矩形到圆心的最短距离
                closest_x = max(label_i.x, min(feature.x, label_i.x + label_i.width))
                closest_y = max(label_i.y, min(feature.y, label_i.y + label_i.height))
                distance_to_rect = math.hypot(feature.x - closest_x, feature.y - closest_y)
                
                # 如果距离小于等于半径，则发生遮挡
                if distance_to_rect <= feature_radius:
                    occlusions_by_label_i += 1
        
        total_occlusions += occlusions_by_label_i
    
    # 计算平均遮挡数量
    avg_occ = total_occlusions / N if N > 0 else 0
    
    # 2. INT (交叉指标) - 计算引导线相互交叉的平均次数
    total_intersections = 0
    processed_pairs = set()  # 记录已处理的引导线对，避免重复计算
    
    for i, label_i in enumerate(labels):
        feature_i = sim_engine.features[label_i.id]
        intersections_for_label_i = 0
        
        for j, label_j in enumerate(labels):
            if i != j:
                # 生成有序的标签对键，确保每对引导线只被检测一次
                pair_key = tuple(sorted([i, j]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                feature_j = sim_engine.features[label_j.id]
                
                # 检测两条引导线是否相交
                if lines_intersect(label_i.center_x, label_i.center_y, feature_i.x, feature_i.y,
                                 label_j.center_x, label_j.center_y, feature_j.x, feature_j.y):
                    intersections_for_label_i += 1
        
        total_intersections += intersections_for_label_i
    
    # 计算平均交叉次数
    avg_int = (total_intersections * 2) / N if N > 0 else 0
    
    return avg_occ, avg_int


def evaluate_layout_quality(simulation_engine, frames_data):
    """
    使用论文中的精确公式评估标签布局质量（为保持向后兼容性保留）
    
    现在调用综合评估函数并返回论文中的5个指标
    """
    all_metrics = evaluate_comprehensive_quality(simulation_engine, frames_data)
    
    # 返回论文中的5个指标
    return {
        'S_overlap': all_metrics['S_overlap'],
        'S_position': all_metrics['S_position'], 
        'S_aesthetics': all_metrics['S_aesthetics'],
        'S_angle_smoothness': all_metrics['S_angle_smoothness'],
        'S_distance_smoothness': all_metrics['S_distance_smoothness'],
        'total_frames': all_metrics['total_frames'],
        'total_labels': all_metrics['total_labels'],
        'total_position_count': all_metrics['total_position_count'],
        'raw_label_overlap': all_metrics['raw_label_overlap'],
        'raw_feature_overlap': all_metrics['raw_feature_overlap']
    }
