"""
标签布局质量评估模块

该模块实现了标签布局质量评估的核心指标：
- OCC (Occlusion): 遮挡指标
- INT (Intersection): 交叉指标  
- DIST (Distance): 距离指标
"""

import math

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

def evaluate_label_layout_quality(labels, features, frame_index=None):
    """
    评估标签布局的三个核心质量指标
    
    指标说明：
    - OCC (Occlusion): 遮挡指标，测量每个标签平均遮挡的对象和标签数量，值越小越好
    - INT (Intersection): 交叉指标，测量每条引导线平均被其他引导线交叉的次数，值越小越好  
    - DIST (Distance): 距离指标，测量标签相对于理想位置的平均距离偏差，值越小越好
    
    参数：
        labels: 标签列表，每个标签应该有position、width、length、id属性
        features: 特征点列表，每个特征点应该有position、id、radius属性
        frame_index: 帧索引（可选，用于调试）
        
    返回：
        dict: 包含 'occ', 'int', 'dist' 三个指标的字典
    """
    N = len(labels)
    if N == 0:
        return {'occ': 0, 'int': 0, 'dist': 0}
    
    # 创建特征点字典，方便快速查找
    feature_dict = {f.id: f for f in features}
    
    # 1. OCC (遮挡指标) - 计算每个标签遮挡其他对象的平均数量
    total_occlusions = 0
    
    for i, label_i in enumerate(labels):
        occlusions_by_label_i = 0
        
        # 获取标签i的边界框
        x_i, y_i = label_i.position
        width_i = label_i.width
        height_i = label_i.length
        
        # 标签边界框 (左上角坐标系)
        left_i = x_i - height_i // 2
        top_i = y_i - width_i // 2
        right_i = x_i + height_i // 2
        bottom_i = y_i + width_i // 2
        
        # 计算label_i遮挡了多少其他标签
        for j, label_j in enumerate(labels):
            if i != j:
                x_j, y_j = label_j.position
                width_j = label_j.width
                height_j = label_j.length
                
                # 标签j的边界框
                left_j = x_j - height_j // 2
                top_j = y_j - width_j // 2
                right_j = x_j + height_j // 2
                bottom_j = y_j + width_j // 2
                
                # 检测两个矩形是否重叠
                x_overlap = max(0, min(right_i, right_j) - max(left_i, left_j))
                y_overlap = max(0, min(bottom_i, bottom_j) - max(top_i, top_j))
                
                if x_overlap > 0 and y_overlap > 0:
                    occlusions_by_label_i += 1
        
        # 计算label_i遮挡了多少其他特征点
        for feature in features:
            if feature.id != label_i.id:  # 排除标签自己对应的特征点
                feature_x, feature_y = feature.position
                feature_radius = getattr(feature, 'radius', 1)
                
                # 计算矩形到圆心的最短距离
                closest_x = max(left_i, min(feature_x, right_i))
                closest_y = max(top_i, min(feature_y, bottom_i))
                distance_to_rect = math.hypot(feature_x - closest_x, feature_y - closest_y)
                
                # 如果距离小于等于半径，则发生遮挡
                if distance_to_rect <= feature_radius:
                    occlusions_by_label_i += 1
        
        total_occlusions += occlusions_by_label_i
    
    # 计算平均遮挡数量
    avg_occ = total_occlusions / N
    
    # 2. INT (交叉指标) - 计算引导线相互交叉的平均次数
    total_intersections = 0
    processed_pairs = set()  # 避免重复计算同一对引导线
    
    for i, label_i in enumerate(labels):
        if label_i.id not in feature_dict:
            continue
            
        feature_i = feature_dict[label_i.id]
        label_i_x, label_i_y = label_i.position
        feature_i_x, feature_i_y = feature_i.position
        
        for j, label_j in enumerate(labels):
            if i != j and label_j.id in feature_dict:
                # 确保每对引导线只被检测一次
                pair_key = tuple(sorted([i, j]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                
                feature_j = feature_dict[label_j.id]
                label_j_x, label_j_y = label_j.position
                feature_j_x, feature_j_y = feature_j.position
                
                # 检测两条引导线是否相交
                if lines_intersect(label_i_x, label_i_y, feature_i_x, feature_i_y,
                                 label_j_x, label_j_y, feature_j_x, feature_j_y):
                    total_intersections += 1
    
    # 计算平均交叉次数（每个交叉影响两个标签）
    avg_int = (total_intersections * 2) / N
    
    # 3. DIST (距离指标) - 标签位置质量评估
    total_distance_deviation = 0
    
    for label in labels:
        if label.id not in feature_dict:
            continue
            
        feature = feature_dict[label.id]
        label_x, label_y = label.position
        feature_x, feature_y = feature.position
        
        # 计算标签中心到特征点的实际距离
        current_distance = math.hypot(label_x - feature_x, label_y - feature_y)
        
        # 理想距离：使用标签对角线长度的一半
        ideal_distance = math.sqrt(label.width**2 + label.length**2) / 2
        
        # 距离偏差
        distance_deviation = abs(current_distance - ideal_distance)
        total_distance_deviation += distance_deviation
    
    # 计算平均距离偏差
    avg_dist = total_distance_deviation / N
    
    return {
        'occ': avg_occ,      # 平均遮挡数量
        'int': avg_int,      # 平均交叉数量
        'dist': avg_dist     # 平均距离偏差（像素）
    }

def print_quality_metrics(metrics, frame_index=None):
    """
    打印质量指标的格式化输出
    
    参数：
        metrics: evaluate_label_layout_quality返回的指标字典
        frame_index: 帧索引（可选）
    """
    if frame_index is not None:
        print(f"\n=== 帧 {frame_index} 的标签布局质量评估 ===")
    else:
        print("\n=== 标签布局质量评估 ===")
    
    print(f"OCC (遮挡指标): {metrics['occ']:.3f}")
    print(f"INT (交叉指标): {metrics['int']:.3f}")
    print(f"DIST (距离指标): {metrics['dist']:.3f}")
    
    # 提供质量评估的简单解释
    if metrics['occ'] < 0.5:
        print("✓ 遮挡情况良好")
    elif metrics['occ'] < 1.0:
        print("⚠ 遮挡情况一般")
    else:
        print("✗ 遮挡情况较差")
    
    if metrics['int'] < 0.5:
        print("✓ 引导线交叉情况良好")
    elif metrics['int'] < 1.0:
        print("⚠ 引导线交叉情况一般")
    else:
        print("✗ 引导线交叉情况较差")
    
    if metrics['dist'] < 50:
        print("✓ 标签位置质量良好")
    elif metrics['dist'] < 100:
        print("⚠ 标签位置质量一般")
    else:
        print("✗ 标签位置质量较差")

def evaluate_sequence_quality(all_frame_metrics):
    """
    评估整个序列的质量统计
    
    参数：
        all_frame_metrics: 所有帧的质量指标列表
        
    返回：
        dict: 包含平均值、最大值、最小值等统计信息
    """
    if not all_frame_metrics:
        return {}
    
    # 计算各指标的统计信息
    occ_values = [m['occ'] for m in all_frame_metrics]
    int_values = [m['int'] for m in all_frame_metrics]
    dist_values = [m['dist'] for m in all_frame_metrics]
    
    stats = {
        'occ': {
            'mean': sum(occ_values) / len(occ_values),
            'max': max(occ_values),
            'min': min(occ_values),
        },
        'int': {
            'mean': sum(int_values) / len(int_values),
            'max': max(int_values),
            'min': min(int_values),
        },
        'dist': {
            'mean': sum(dist_values) / len(dist_values),
            'max': max(dist_values),
            'min': min(dist_values),
        }
    }
    
    return stats

def print_sequence_quality_summary(stats):
    """
    打印序列质量统计摘要
    
    参数：
        stats: evaluate_sequence_quality返回的统计字典
    """
    print("\n=== 整个序列的质量统计摘要 ===")
    
    for metric_name, metric_stats in stats.items():
        metric_display = {
            'occ': 'OCC (遮挡指标)',
            'int': 'INT (交叉指标)',
            'dist': 'DIST (距离指标)'
        }[metric_name]
        
        print(f"\n{metric_display}:")
        print(f"  平均值: {metric_stats['mean']:.3f}")
        print(f"  最大值: {metric_stats['max']:.3f}")
        print(f"  最小值: {metric_stats['min']:.3f}")
