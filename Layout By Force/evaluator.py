"""
标签布局质量评估模块

该模块实现了标签布局质量评估的核心指标：
- OCC (Occlusion): 遮挡指标
- INT (Intersection): 交叉指标  
- DIST (Distance): 距离指标

以及论文中的精确公式评估的标签布局质量指标：
- S_overlap: 重叠度指标
- S_position: 位置距离指标
- S_aesthetics: 美观度指标（引导线交叉次数）
- S_angle_smoothness: 角度平滑度指标
- S_distance_smoothness: 距离平滑度指标
"""

import math

def evaluate_single_frame_quality(sim_engine, frame_index=None):
    """
    评估单帧标签布局的三个核心质量指标
    
    指标说明：
    - OCC (Occlusion): 遮挡指标，测量每个标签平均遮挡的对象和标签数量，值越小越好
    - INT (Intersection): 交叉指标，测量每条引导线平均被其他引导线交叉的次数，值越小越好  
    - DIST (Distance): 距离指标，测量标签相对于理想位置的平均距离偏差，值越小越好
    
    参数：
        sim_engine: 仿真引擎实例，包含当前帧的标签和特征点状态
        frame_index: 帧索引（可选，用于调试）
        
    返回：
        dict: 包含 'occ', 'int', 'dist' 三个指标的字典
    """
    labels = list(sim_engine.labels.values())
    features = list(sim_engine.features.values())
    N = len(labels)
    
    # 1. OCC (遮挡指标) - 计算每个标签遮挡其他对象的平均数量
    # 包括：标签遮挡其他标签 + 标签遮挡其他特征点
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
    # 避免重复计算同一对引导线的交叉
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
                # 引导线：标签中心 → 对应特征点
                if lines_intersect(label_i.center_x, label_i.center_y, feature_i.x, feature_i.y,
                                 label_j.center_x, label_j.center_y, feature_j.x, feature_j.y):
                    intersections_for_label_i += 1
        
        total_intersections += intersections_for_label_i
    
    # 计算平均交叉次数
    # 由于每个交叉影响两个标签，需要乘以2来确保每个标签都计入相应的交叉数
    avg_int = (total_intersections * 2) / N if N > 0 else 0
    
    # 3. DIST - 标签位置质量评估
    # 论文定义：测量每个标签相对于其目标对象的平均额外移动距离，近似标签的抖动程度
    # 实现方式：由于缺少完整轨迹信息，这里计算标签与特征点的距离偏差作为近似
    total_distance_deviation = 0
    
    for label in labels:
        feature = sim_engine.features[label.id]
        
        # 计算标签中心到特征点的实际距离
        current_distance = math.hypot(label.center_x - feature.x, label.center_y - feature.y)
        
        # 理想距离：标签与特征点之间的最优距离
        # 使用标签对角线长度的一半，确保标签不与特征点重叠且距离适中
        ideal_distance = math.sqrt(label.width**2 + label.height**2) / 2
        
        # 距离偏差：实际距离与理想距离的绝对差值
        # 较大的偏差表示标签位置不理想，可能导致视觉混乱
        distance_deviation = abs(current_distance - ideal_distance)
        total_distance_deviation += distance_deviation
    
    # 计算平均距离偏差
    avg_dist = total_distance_deviation / N if N > 0 else 0
    
    return {
        'occ': avg_occ,      # 平均遮挡数量：每个标签遮挡的对象和标签的平均数量
        'int': avg_int,      # 平均交叉数量：每条引导线被其他引导线交叉的平均次数
        'dist': avg_dist     # 平均距离偏差：标签与理想位置的平均距离偏差（像素）
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


def evaluate_layout_quality(simulation_engine, frames_data):
    """
    使用论文中的精确公式评估标签布局质量
    
    计算五个主要指标：
    - S_overlap: 重叠度指标 (pixel²)
    - S_position: 位置距离指标 (pixel)  
    - S_aesthetics: 美观度指标（引导线交叉次数）
    - S_angle_smoothness: 角度平滑度指标 (degree)
    - S_distance_smoothness: 距离平滑度指标 (pixel)
    
    参数：
        simulation_engine: 仿真引擎实例
        frames_data: 帧数据列表
        
    返回：
        dict: 包含各项质量指标的字典
    """
    from force_calculator import ForceCalculator
    from simulation import SimulationEngine
    
    M = len(frames_data)  # 帧数
    N = len(simulation_engine.labels)  # 标签数
    
    # 按论文公式初始化累加器
    total_label_overlap = 0  # ∑∑O_{i,j}
    total_feature_overlap = 0  # ∑∑P_{i,j}
    total_position_distance = 0  # ∑∑r_i
    total_intersections = 0  # ∑∑I_i
    total_angle_changes = 0  # ∑∑|θ_{i,k} - θ_{i,k+1}|
    total_distance_changes = 0  # ∑∑|r_{i,k} - r_{i,k+1}|
    
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
        
        # 1. S_Overlap = (1/M) * ∑_{k=1}^M (∑_{i}∑_{j≠i} O_{i,j} + ∑_{i}∑_{j≠i} P_{i,j})
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
                if i != j:  # 标签i与特征j（j≠i）的重叠
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
        
        # 2. S_Position = (1/M) * ∑_{k=1}^M ∑_{i=1}^N r_i
        frame_position_distance = 0
        for i in range(N):
            label = labels[i]
            feature = sim_engine.features[label.id]
            distance = math.hypot(label.center_x - feature.x, label.center_y - feature.y)
            frame_position_distance += distance
        
        total_position_distance += frame_position_distance
        
        # 3. S_Aesthetics = (1/M) * ∑_{k=1}^M ∑_{i=1}^N I_i
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
        
        # 4. S_Smoothness: 计算相邻帧间的变化
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
    
    # 按论文公式计算最终指标
    S_overlap = (total_label_overlap + total_feature_overlap) / M
    S_position = total_position_distance / M / N
    S_aesthetics = total_intersections / M / N
    S_angle_smoothness = math.degrees(total_angle_changes / (M - 1) / N) if M > 1 else 0
    S_distance_smoothness = total_distance_changes / (M - 1) / N if M > 1 else 0
    
    return {
        'S_overlap': S_overlap,
        'S_position': S_position, 
        'S_aesthetics': S_aesthetics,
        'S_angle_smoothness': S_angle_smoothness,
        'S_distance_smoothness': S_distance_smoothness,
        'total_frames': M,
        'total_labels': N,
        'raw_label_overlap': total_label_overlap,
        'raw_feature_overlap': total_feature_overlap
    }
