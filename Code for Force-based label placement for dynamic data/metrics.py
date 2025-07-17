"""
标签布局质量评估模块

该模块实现了标签布局质量评估的核心指标：
- OCC (Occlusion): 遮挡指标
- INT (Intersection): 交叉指标

以及论文中的精确公式评估的标签布局质量指标：
- S_overlap: 重叠度指标
- S_position: 位置距离指标
- S_aesthetics: 美观度指标（引导线交叉次数）
- S_angle_smoothness: 角度平滑度指标
- S_distance_smoothness: 距离平滑度指标
"""
import math


def evaluate_metrics(history, labels, features):
    """
    使用给定的评价指标计算优化结果质量：
    - S_overlap: 标签之间重叠面积平均值
    - S_position: 标签与特征点距离平均值
    - S_aesthetic: 领导线交叉次数平均值
    - S_r_smooth / S_theta_smooth: 距离/角度平滑度

    Args:
        history (list): 优化历史记录
        labels (list): Label 对象列表
        features (list): Feature 对象列表

    Returns:
        dict: 各项评价指标
    """
    M = len(history)
    total_overlap = 0.0
    total_position = 0.0
    total_aesthetic = 0
    r_vals = {i: [] for i in range(len(labels))}
    theta_vals = {i: [] for i in range(len(labels))}

    def rect_overlap(a_center, a_w, a_h, b_center, b_w, b_h):
        dx = min(a_center[0] + a_w/2, b_center[0] + b_w/2) - max(a_center[0] - a_w/2, b_center[0] - b_w/2)
        dy = min(a_center[1] + a_h/2, b_center[1] + b_h/2) - max(a_center[1] - a_h/2, b_center[1] - b_h/2)
        return max(dx, 0) * max(dy, 0)

    def segments_intersect(p1, p2, p3, p4):
        def ori(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        o1, o2 = ori(p1, p2, p3), ori(p1, p2, p4)
        o3, o4 = ori(p3, p4, p1), ori(p3, p4, p2)
        return o1*o2 < 0 and o3*o4 < 0

    for frame in history:
        positions = frame['positions']
        features_pos = frame['features']
        overlap_sum = 0.0
        for i in positions:
            for j in positions:
                if j <= i:
                    continue
                overlap_sum += rect_overlap(positions[i], labels[i].length, labels[i].width,
                                            positions[j], labels[j].length, labels[j].width)
        total_overlap += overlap_sum

        pos_sum, cross_count = 0.0, 0
        for i in positions:
            xi, yi = positions[i]
            xf, yf = features_pos[i]
            dist = math.hypot(xi - xf, yi - yf)
            pos_sum += dist
            r_vals[i].append(dist)
            theta_vals[i].append(math.degrees(math.atan2(yi - yf, xi - xf)))
            for j in positions:
                if j <= i:
                    continue
                if segments_intersect(positions[i], features_pos[i], positions[j], features_pos[j]):
                    cross_count += 1
        total_position += pos_sum
        total_aesthetic += cross_count

    S_overlap = total_overlap / M
    S_position = total_position / M
    S_aesthetic = total_aesthetic / M

    M1 = M - 1
    sum_r = sum(abs(r_vals[i][k] - r_vals[i][k+1]) for i in r_vals for k in range(M1))
    sum_theta = sum(abs(theta_vals[i][k] - theta_vals[i][k+1]) for i in theta_vals for k in range(M1))
    S_r_smooth = sum_r / M1
    S_theta_smooth = sum_theta / M1

    return {
        'S_overlap': S_overlap,
        'S_position': S_position,
        'S_aesthetic': S_aesthetic,
        'S_r_smooth': S_r_smooth,
        'S_theta_smooth': S_theta_smooth
    }


def evaluate_single_frame_quality(labels, features, frame_positions, features_pos):
    """
    评估单帧标签布局的两个核心质量指标
    
    指标说明：
    - OCC (Occlusion): 遮挡指标，测量每个标签平均遮挡的对象和标签数量，值越小越好
    - INT (Intersection): 交叉指标，测量每条引导线平均被其他引导线交叉的次数，值越小越好
    
    参数：
        labels: 标签对象列表
        features: 特征对象列表
        frame_positions: 当前帧标签位置字典 {i: (x, y)}
        features_pos: 当前帧特征点位置列表 [(x, y), ...]
        
    返回：
        dict: 包含 'occ', 'int' 两个指标的字典
    """
    N = len(labels)
    
    # 1. OCC (遮挡指标) - 计算每个标签遮挡其他对象的平均数量
    total_occlusions = 0
    
    for i in range(N):
        if i not in frame_positions:
            continue
            
        label_i_pos = frame_positions[i]
        label_i = labels[i]
        occlusions_by_label_i = 0
        
        # 计算label_i遮挡了多少其他标签（矩形与矩形的重叠检测）
        for j in range(N):
            if i != j and j in frame_positions:
                label_j_pos = frame_positions[j]
                label_j = labels[j]
                
                # 使用轴对齐边界框(AABB)检测两个标签是否重叠
                x_overlap = max(0, min(label_i_pos[0] + label_i.length/2, label_j_pos[0] + label_j.length/2) - 
                              max(label_i_pos[0] - label_i.length/2, label_j_pos[0] - label_j.length/2))
                y_overlap = max(0, min(label_i_pos[1] + label_i.width/2, label_j_pos[1] + label_j.width/2) - 
                              max(label_i_pos[1] - label_i.width/2, label_j_pos[1] - label_j.width/2))
                if x_overlap > 0 and y_overlap > 0:
                    occlusions_by_label_i += 1
        
        # 计算label_i遮挡了多少其他特征点（矩形与圆形的重叠检测）
        for k, feature_pos in enumerate(features_pos):
            if k != i:  # 排除标签自己对应的特征点
                feature_radius = 1  # 默认半径
                
                # 计算矩形到圆心的最短距离
                closest_x = max(label_i_pos[0] - label_i.length/2, 
                               min(feature_pos[0], label_i_pos[0] + label_i.length/2))
                closest_y = max(label_i_pos[1] - label_i.width/2, 
                               min(feature_pos[1], label_i_pos[1] + label_i.width/2))
                distance_to_rect = math.hypot(feature_pos[0] - closest_x, feature_pos[1] - closest_y)
                
                # 如果距离小于等于半径，则发生遮挡
                if distance_to_rect <= feature_radius:
                    occlusions_by_label_i += 1
        
        total_occlusions += occlusions_by_label_i
    
    # 计算平均遮挡数量
    avg_occ = total_occlusions / N if N > 0 else 0
    
    # 2. INT (交叉指标) - 计算引导线相互交叉的平均次数
    total_intersections = 0
    processed_pairs = set()  # 记录已处理的引导线对，避免重复计算
    
    for i in range(N):
        if i not in frame_positions:
            continue
            
        label_i_pos = frame_positions[i]
        feature_i_pos = features_pos[i]
        intersections_for_label_i = 0
        
        for j in range(N):
            if i != j and j in frame_positions:
                # 生成有序的标签对键，确保每对引导线只被检测一次
                pair_key = tuple(sorted([i, j]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                label_j_pos = frame_positions[j]
                feature_j_pos = features_pos[j]
                
                # 检测两条引导线是否相交
                if lines_intersect(label_i_pos[0], label_i_pos[1], feature_i_pos[0], feature_i_pos[1],
                                 label_j_pos[0], label_j_pos[1], feature_j_pos[0], feature_j_pos[1]):
                    intersections_for_label_i += 1
        
        total_intersections += intersections_for_label_i
    
    # 计算平均交叉次数
    avg_int = (total_intersections * 2) / N if N > 0 else 0
    
    return {
        'occ': avg_occ,      # 平均遮挡数量
        'int': avg_int       # 平均交叉数量
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


def evaluate_enhanced_metrics(history, labels, features):
    """
    使用增强的评价指标体系评估优化结果质量
    
    结合原有指标和新增的OCC/INT指标，提供更全面的评估
    
    Args:
        history (list): 优化历史记录
        labels (list): Label 对象列表  
        features (list): Feature 对象列表
        
    Returns:
        dict: 包含所有评价指标的字典
    """
    # 获取原有的评价指标
    basic_metrics = evaluate_metrics(history, labels, features)
    
    # 计算每帧的OCC和INT指标
    M = len(history)
    total_occ = 0.0
    total_int = 0.0
    
    for frame in history:
        frame_positions = frame['positions']
        features_pos = frame['features']
        
        # 计算当前帧的OCC和INT指标
        frame_quality = evaluate_single_frame_quality(labels, features, frame_positions, features_pos)
        total_occ += frame_quality['occ']
        total_int += frame_quality['int']
    
    # 计算平均OCC和INT
    avg_occ = total_occ / M if M > 0 else 0
    avg_int = total_int / M if M > 0 else 0
    
    # 按指定顺序组织指标，优先显示OCC和INT
    enhanced_metrics = {
        'OCC': avg_occ,      # 平均遮挡指标
        'INT': avg_int,      # 平均交叉指标
    }
    # 添加原有指标
    enhanced_metrics.update(basic_metrics)
    
    return enhanced_metrics
