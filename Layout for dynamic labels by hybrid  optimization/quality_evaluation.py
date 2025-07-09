"""
动态标签布局质量评估模块

该模块实现了动态标签布局优化的质量评估体系，包含以下评价指标：

   - OCC (Occlusion): 遮挡指标 - 测量标签与其他对象的平均遮挡数量
   - INT (Intersection): 交叉指标 - 测量引导线相互交叉的平均次数
   - DIST (Distance): 距离指标 - 测量标签相对于理想位置的平均距离偏差
   - S_Overlap: 重叠指标 - 标签间和标签-特征间的重叠面积，单位：pixel²
   - S_Position: 位置指标 - 标签到对应特征点的距离，单位：pixel
   - S_Aesthetics: 美学指标 - 引导线交叉次数，单位：次
   - S_Smoothness: 平滑度指标 - 相邻帧间的位置变化平滑度
     * S_Smoothness_Angle: 角度平滑度，单位：degree
     * S_Smoothness_Radius: 径向平滑度，单位：pixel
"""

import math

def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    判断两条线段是否相交
    
    使用CCW（逆时针）算法检测两条线段是否相交。该算法基于向量叉积的几何性质，
    通过判断四个点的相对位置关系来确定两条线段是否相交。
    
    算法原理：
    - 对于两条线段AB和CD，如果它们相交，则点C和点D必须位于直线AB的两侧
    - 同时，点A和点B必须位于直线CD的两侧
    - 使用CCW函数判断三个点的转向（顺时针/逆时针）
    
    参数：
        x1, y1 (float): 第一条线段的起点坐标
        x2, y2 (float): 第一条线段的终点坐标
        x3, y3 (float): 第二条线段的起点坐标
        x4, y4 (float): 第二条线段的终点坐标
        
    返回：
        bool: 如果两条线段相交返回True，否则返回False
        
    示例：
        >>> lines_intersect(0, 0, 10, 10, 0, 10, 10, 0)
        True
        >>> lines_intersect(0, 0, 5, 5, 6, 6, 10, 10)
        False
    """
    def ccw(Ax, Ay, Bx, By, Cx, Cy):
        return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
    
    return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and \
           ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)

def evaluate_label_layout_quality(labels, features, frame_index=None):
    """
    评估单帧标签布局的质量指标
    
    该函数计算三个核心质量指标，用于实时优化过程中的质量评估。
    这些指标设计简单且计算高效，适合在优化迭代过程中频繁调用。
    
    指标详解：
    1. OCC (Occlusion): 遮挡指标
       - 测量每个标签平均遮挡的对象数量（包括其他标签和特征点）
       - 计算方式：统计每个标签遮挡的对象数量，然后求平均值
       - 取值范围：[0, +∞)，值越小表示遮挡越少
       
    2. INT (Intersection): 交叉指标
       - 测量引导线相互交叉的平均次数
       - 计算方式：统计所有引导线对的交叉次数，除以标签数量
       - 取值范围：[0, +∞)，值越小表示交叉越少
       
    3. DIST (Distance): 距离指标
       - 测量标签相对于理想位置的平均距离偏差
       - 理想距离定义为标签对角线长度的一半
       - 计算方式：|实际距离 - 理想距离|的平均值
       - 取值范围：[0, +∞)，值越小表示位置越理想
    
    参数：
        labels (list): 标签对象列表，每个标签需包含以下属性：
            - position (tuple): 标签中心坐标 (x, y)
            - width (int): 标签宽度（像素）
            - length (int): 标签长度（像素）
            - id: 标签唯一标识符
        features (list): 特征点对象列表，每个特征点需包含以下属性：
            - position (tuple): 特征点坐标 (x, y)
            - id: 特征点唯一标识符
            - radius (int, optional): 特征点半径，默认为1像素
        frame_index (int, optional): 帧索引，用于调试输出，默认为None
        
    返回：
        dict: 包含三个指标的字典
            - 'occ' (float): 平均遮挡数量
            - 'int' (float): 平均交叉数量
            - 'dist' (float): 平均距离偏差（像素）
            
    异常：
        如果标签列表为空，返回全零字典
        
    示例：
        >>> metrics = evaluate_label_layout_quality(labels, features, 0)
        >>> print(f"遮挡: {metrics['occ']:.3f}, 交叉: {metrics['int']:.3f}, 距离: {metrics['dist']:.3f}")
        遮挡: 0.250, 交叉: 0.167, 距离: 45.123
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


def calculate_paper_metrics(all_labels_history, all_features_history):
    """
    计算综合评价指标
    
    计算动态标签布局的四个核心质量指标，用于系统性能的最终评估。
    这些指标基于标准评估体系，提供全面的布局质量分析。
    
    指标详解：
    1. S_Overlap (重叠指标):
       - 公式: S_Overlap = (1/M) * Σ(ΣΣO_{i,j} + ΣΣP_{i,j})
       - 测量标签间重叠和标签-特征重叠的总面积
       - 单位: pixel² (像素平方)
       - 物理意义: 反映布局的空间冲突程度
       
    2. S_Position (位置指标):
       - 公式: S_Position = (1/M) * ΣΣr_i
       - 测量标签到对应特征点的距离
       - 单位: pixel (像素)
       - 物理意义: 反映标签与特征点的关联紧密程度
       
    3. S_Aesthetics (美学指标):
       - 公式: S_Aesthetics = (1/M) * ΣΣI_i
       - 测量引导线交叉的数量
       - 单位: 次数
       - 物理意义: 反映视觉美观程度
       
    4. S_Smoothness (平滑度指标):
       分为两个子指标：
       - S_Smoothness_Angle: 角度平滑度
         * 公式: (1/(M-1)) * ΣΣ|θ_{i,k} - θ_{i,k+1}|
         * 单位: degree (度)
         * 物理意义: 反映标签角度变化的连续性
       - S_Smoothness_Radius: 径向平滑度
         * 公式: (1/(M-1)) * ΣΣ|r_{i,k} - r_{i,k+1}|
         * 单位: pixel (像素)
         * 物理意义: 反映标签距离变化的连续性
    
    参数：
        all_labels_history (list): 所有帧的标签历史记录
            格式: [[labels_frame1], [labels_frame2], ...]
            每个labels_frame包含该帧的所有标签对象
        all_features_history (list): 所有帧的特征历史记录
            格式: [[features_frame1], [features_frame2], ...]
            每个features_frame包含该帧的所有特征点对象
    
    返回：
        dict: 包含五个指标的字典
            - 's_overlap' (float): 平均重叠面积 (pixel²)
            - 's_position' (float): 平均位置距离 (pixel)
            - 's_aesthetics' (float): 平均交叉次数
            - 's_smoothness_angle' (float): 平均角度变化 (degree)
            - 's_smoothness_radius' (float): 平均径向变化 (pixel)
    
    异常：
        如果历史记录为空，返回全零字典
        
    性能考虑：
        该函数计算复杂度为O(M * N²)，其中M为帧数，N为标签数
        建议在处理完整序列后调用，而非实时计算
        
    示例：
        >>> metrics = calculate_paper_metrics(all_labels_history, all_features_history)
        >>> print(f"重叠: {metrics['s_overlap']:.3f} pixel²")
        >>> print(f"位置: {metrics['s_position']:.3f} pixel")
        >>> print(f"美学: {metrics['s_aesthetics']:.3f}")
        >>> print(f"角度平滑: {metrics['s_smoothness_angle']:.3f}°")
        >>> print(f"径向平滑: {metrics['s_smoothness_radius']:.3f} pixel")
    """
    if not all_labels_history or not all_features_history:
        return {'s_overlap': 0, 's_position': 0, 's_aesthetics': 0, 's_smoothness_angle': 0, 's_smoothness_radius': 0}
    
    M = len(all_labels_history)  # 帧数
    
    # 1. S_Overlap - 重叠指标
    total_overlap = 0
    for frame_idx in range(M):
        labels = all_labels_history[frame_idx]
        features = all_features_history[frame_idx]
        
        # 标签-标签重叠
        for i, label_i in enumerate(labels):
            for j, label_j in enumerate(labels):
                if i != j:
                    overlap_area = calculate_rectangle_overlap(label_i, label_j)
                    total_overlap += overlap_area
        
        # 标签-特征重叠
        for label in labels:
            for feature in features:
                if label.id != feature.id:
                    overlap_area = calculate_label_feature_overlap(label, feature)
                    total_overlap += overlap_area
    
    s_overlap = total_overlap / M
    
    # 2. S_Position - 位置指标
    total_distance = 0
    for frame_idx in range(M):
        labels = all_labels_history[frame_idx]
        features = all_features_history[frame_idx]
        feature_dict = {f.id: f for f in features}
        
        for label in labels:
            if label.id in feature_dict:
                feature = feature_dict[label.id]
                distance = math.hypot(label.position[0] - feature.position[0], 
                                    label.position[1] - feature.position[1])
                total_distance += distance
    
    s_position = total_distance / M
    
    # 3. S_Aesthetics - 美学指标（引导线交叉）
    total_intersections = 0
    for frame_idx in range(M):
        labels = all_labels_history[frame_idx]
        features = all_features_history[frame_idx]
        feature_dict = {f.id: f for f in features}
        
        # 计算引导线交叉
        for i, label_i in enumerate(labels):
            if label_i.id in feature_dict:
                feature_i = feature_dict[label_i.id]
                for j, label_j in enumerate(labels):
                    if i != j and label_j.id in feature_dict:
                        feature_j = feature_dict[label_j.id]
                        if lines_intersect(label_i.position[0], label_i.position[1], 
                                         feature_i.position[0], feature_i.position[1],
                                         label_j.position[0], label_j.position[1], 
                                         feature_j.position[0], feature_j.position[1]):
                            total_intersections += 1
    
    s_aesthetics = total_intersections / M
    
    # 4. S_Smoothness - 平滑度指标
    total_angle_diff = 0
    total_radius_diff = 0
    
    if M > 1:
        for frame_idx in range(M - 1):
            labels_curr = all_labels_history[frame_idx]
            labels_next = all_labels_history[frame_idx + 1]
            features_curr = all_features_history[frame_idx]
            features_next = all_features_history[frame_idx + 1]
            
            feature_dict_curr = {f.id: f for f in features_curr}
            feature_dict_next = {f.id: f for f in features_next}
            
            for label_curr in labels_curr:
                # 找到下一帧对应的标签
                label_next = None
                for l in labels_next:
                    if l.id == label_curr.id:
                        label_next = l
                        break
                
                if (label_next and label_curr.id in feature_dict_curr and 
                    label_curr.id in feature_dict_next):
                    
                    feature_curr = feature_dict_curr[label_curr.id]
                    feature_next = feature_dict_next[label_curr.id]
                    
                    # 计算角度差异
                    angle_curr = math.atan2(label_curr.position[1] - feature_curr.position[1],
                                          label_curr.position[0] - feature_curr.position[0])
                    angle_next = math.atan2(label_next.position[1] - feature_next.position[1],
                                          label_next.position[0] - feature_next.position[0])
                    
                    angle_diff = abs(angle_curr - angle_next)
                    # 处理角度跳跃
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff
                    
                    total_angle_diff += math.degrees(angle_diff)
                    
                    # 计算径向距离差异
                    radius_curr = math.hypot(label_curr.position[0] - feature_curr.position[0],
                                           label_curr.position[1] - feature_curr.position[1])
                    radius_next = math.hypot(label_next.position[0] - feature_next.position[0],
                                           label_next.position[1] - feature_next.position[1])
                    
                    total_radius_diff += abs(radius_curr - radius_next)
    
    s_smoothness_angle = total_angle_diff / (M - 1) if M > 1 else 0
    s_smoothness_radius = total_radius_diff / (M - 1) if M > 1 else 0
    
    return {
        's_overlap': s_overlap,
        's_position': s_position,
        's_aesthetics': s_aesthetics,
        's_smoothness_angle': s_smoothness_angle,
        's_smoothness_radius': s_smoothness_radius
    }

def calculate_rectangle_overlap(label1, label2):
    """
    计算两个矩形标签的重叠面积
    
    使用矩形相交算法计算两个标签的重叠面积。该函数假设标签为轴对齐的矩形，
    以标签中心为参考点，根据width和length计算边界框。
    
    算法步骤：
    1. 根据标签中心和尺寸计算每个标签的边界框
    2. 计算两个边界框的交集区域
    3. 如果有交集，计算交集面积；否则返回0
    
    参数：
        label1 (Label): 第一个标签对象，需包含：
            - position (tuple): 标签中心坐标 (x, y)
            - width (int): 标签宽度
            - length (int): 标签长度
        label2 (Label): 第二个标签对象，格式同label1
        
    返回：
        float: 重叠面积（像素平方），如果无重叠返回0
        
    注意事项：
        - 标签坐标系以中心为原点
        - 边界框计算：left = x - length/2, right = x + length/2
        - 函数不检查输入参数的有效性
        
    示例：
        >>> label1 = Label(position=(100, 100), width=50, length=80)
        >>> label2 = Label(position=(120, 110), width=40, length=60)
        >>> overlap = calculate_rectangle_overlap(label1, label2)
        >>> print(f"重叠面积: {overlap} pixel²")
    """
    x1, y1 = label1.position
    w1, h1 = label1.width, label1.length
    left1, top1 = x1 - h1//2, y1 - w1//2
    right1, bottom1 = x1 + h1//2, y1 + w1//2
    
    x2, y2 = label2.position
    w2, h2 = label2.width, label2.length
    left2, top2 = x2 - h2//2, y2 - w2//2
    right2, bottom2 = x2 + h2//2, y2 + w2//2
    
    # 计算重叠区域
    overlap_left = max(left1, left2)
    overlap_top = max(top1, top2)
    overlap_right = min(right1, right2)
    overlap_bottom = min(bottom1, bottom2)
    
    if overlap_left < overlap_right and overlap_top < overlap_bottom:
        return (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
    return 0

def calculate_label_feature_overlap(label, feature):
    """
    计算标签与特征点的重叠面积
    
    计算矩形标签与圆形特征点的重叠面积。该函数使用简化的碰撞检测算法，
    先计算矩形到圆心的最短距离，然后判断是否发生重叠。
    
    算法步骤：
    1. 计算标签矩形的边界框
    2. 找到矩形边界上距离特征点最近的点
    3. 计算该点到特征点圆心的距离
    4. 如果距离小于等于特征点半径，则认为发生重叠
    5. 简化处理：重叠面积估算为整个特征点圆的面积
    
    参数：
        label (Label): 标签对象，需包含：
            - position (tuple): 标签中心坐标 (x, y)
            - width (int): 标签宽度
            - length (int): 标签长度
        feature (Feature): 特征点对象，需包含：
            - position (tuple): 特征点坐标 (x, y)
            - radius (int, optional): 特征点半径，默认为5像素
            
    返回：
        float: 重叠面积（像素平方），如果无重叠返回0
        
    注意事项：
        - 使用简化的重叠面积计算，可能存在一定误差
        - 实际应用中特征点半径通常很小，误差可忽略
        - 如果需要精确计算，可使用更复杂的几何算法
        
    示例：
        >>> label = Label(position=(100, 100), width=50, length=80)
        >>> feature = Feature(position=(90, 95), radius=8)
        >>> overlap = calculate_label_feature_overlap(label, feature)
        >>> print(f"标签-特征重叠面积: {overlap:.2f} pixel²")
    """
    label_x, label_y = label.position
    feature_x, feature_y = feature.position
    feature_radius = getattr(feature, 'radius', 5)  # 默认半径5像素
    
    # 标签边界
    left = label_x - label.length//2
    top = label_y - label.width//2
    right = label_x + label.length//2
    bottom = label_y + label.width//2
    
    # 计算矩形到圆心的最短距离
    closest_x = max(left, min(feature_x, right))
    closest_y = max(top, min(feature_y, bottom))
    distance = math.hypot(feature_x - closest_x, feature_y - closest_y)
    
    # 简化处理：如果距离小于半径，认为有重叠
    if distance <= feature_radius:
        # 估算重叠面积（简化为圆的面积）
        return math.pi * feature_radius * feature_radius
    return 0
