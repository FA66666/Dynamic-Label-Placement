

import math

def lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    def ccw(Ax, Ay, Bx, By, Cx, Cy):
        return (Cy - Ay) * (Bx - Ax) > (By - Ay) * (Cx - Ax)
    
    return ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and \
           ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4)

def evaluate_label_layout_quality(labels, features, frame_index=None):
    # 统一计算所有七个指标
    N = len(labels)
    if N == 0:
        return {
            'occ': 0, 'int': 0, 'dist': 0,
            's_overlap': 0, 's_position': 0, 's_aesthetics': 0,
            's_smoothness_angle': 0, 's_smoothness_radius': 0
        }

    feature_dict = {f.id: f for f in features}

    # OCC
    total_occlusions = 0
    for i, label_i in enumerate(labels):
        occlusions_by_label_i = 0
        x_i, y_i = label_i.position
        width_i = label_i.width
        height_i = label_i.length
        left_i = x_i - height_i // 2
        top_i = y_i - width_i // 2
        right_i = x_i + height_i // 2
        bottom_i = y_i + width_i // 2
        for j, label_j in enumerate(labels):
            if i != j:
                x_j, y_j = label_j.position
                width_j = label_j.width
                height_j = label_j.length
                left_j = x_j - height_j // 2
                top_j = y_j - width_j // 2
                right_j = x_j + height_j // 2
                bottom_j = y_j + width_j // 2
                x_overlap = max(0, min(right_i, right_j) - max(left_i, left_j))
                y_overlap = max(0, min(bottom_i, bottom_j) - max(top_i, top_j))
                if x_overlap > 0 and y_overlap > 0:
                    occlusions_by_label_i += 1
        for feature in features:
            if feature.id != label_i.id:
                feature_x, feature_y = feature.position
                feature_radius = getattr(feature, 'radius', 1)
                closest_x = max(left_i, min(feature_x, right_i))
                closest_y = max(top_i, min(feature_y, bottom_i))
                distance_to_rect = math.hypot(feature_x - closest_x, feature_y - closest_y)
                if distance_to_rect <= feature_radius:
                    occlusions_by_label_i += 1
        total_occlusions += occlusions_by_label_i
    avg_occ = total_occlusions / N

    # INT
    total_intersections = 0
    processed_pairs = set()
    for i, label_i in enumerate(labels):
        if label_i.id not in feature_dict:
            continue
        feature_i = feature_dict[label_i.id]
        label_i_x, label_i_y = label_i.position
        feature_i_x, feature_i_y = feature_i.position
        for j, label_j in enumerate(labels):
            if i != j and label_j.id in feature_dict:
                pair_key = tuple(sorted([i, j]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)
                feature_j = feature_dict[label_j.id]
                label_j_x, label_j_y = label_j.position
                feature_j_x, feature_j_y = feature_j.position
                if lines_intersect(label_i_x, label_i_y, feature_i_x, feature_i_y,
                                  label_j_x, label_j_y, feature_j_x, feature_j_y):
                    total_intersections += 1
    avg_int = (total_intersections * 2) / N

    # DIST
    total_distance = 0
    for label in labels:
        if label.id not in feature_dict:
            continue
        feature = feature_dict[label.id]
        label_x, label_y = label.position
        feature_x, feature_y = feature.position
        current_distance = math.hypot(label_x - feature_x, label_y - feature_y)
        total_distance += current_distance
    avg_dist = total_distance / N

    # S_Overlap, S_Position, S_Aesthetics, S_Smoothness
    # 单帧时直接用当前帧数据，多帧时传入历史
    all_labels_history = [labels]
    all_features_history = [features]
    s_metrics = calculate_paper_metrics(all_labels_history, all_features_history)

    return {
        'occ': avg_occ,
        'int': avg_int,
        'dist': avg_dist,
        's_overlap': s_metrics['s_overlap'],
        's_position': s_metrics['s_position'],
        's_aesthetics': s_metrics['s_aesthetics'],
        's_smoothness_angle': s_metrics['s_smoothness_angle'],
        's_smoothness_radius': s_metrics['s_smoothness_radius']
    }


def calculate_paper_metrics(all_labels_history, all_features_history):
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
                    overlap_area = calculate_label_feature_overlap(label, feature)
                    total_overlap += overlap_area
    
    s_overlap = total_overlap / M
    
    # 2. S_Position - 位置指标
    # 根据公式: S_Position = (1/M) * Σ(Σr_i)
    # 其中r_i是标签和特征点之间的距离，单位是像素
    total_distance_across_frames = 0
    for frame_idx in range(M):
        frame_distance_sum = 0
        labels = all_labels_history[frame_idx]
        features = all_features_history[frame_idx]
        feature_dict = {f.id: f for f in features}
        for label in labels:
            if label.id in feature_dict:
                feature = feature_dict[label.id]
                distance = math.hypot(label.position[0] - feature.position[0], 
                                    label.position[1] - feature.position[1])
                frame_distance_sum += distance
        total_distance_across_frames += frame_distance_sum
    s_position = total_distance_across_frames / M 
    
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
    label_x, label_y = label.position
    feature_x, feature_y = feature.position
    feature_radius = getattr(feature, 'radius', 1)  # 默认半径5像素
    
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
