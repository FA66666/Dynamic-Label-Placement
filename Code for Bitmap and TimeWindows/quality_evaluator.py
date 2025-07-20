import json
import math
import argparse

# ======================================================================
# 几何学辅助函数 (Geometric Helper Functions)
# ======================================================================

def lines_intersect(p1, p2, p3, p4):
    """
    判断两条线段 (p1, p2) 和 (p3, p4) 是否相交。
    这是一个标准的基于方向性的几何算法。

    参数:
        p1, p2: 第一条线段的端点, 格式为 (x, y)
        p3, p4: 第二条线段的端点, 格式为 (x, y)
        
    返回:
        bool: 如果相交则返回True, 否则返回False
    """
    def ccw(A, B, C):
        # 计算三点A, B, C的朝向
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    # 如果两条线段的端点在彼此的相对两侧，则它们相交
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)


# ======================================================================
# 核心评估函数 (Core Evaluation Function)
# ======================================================================

def evaluate_placement_quality(output_data):
    """
    根据标签放置算法的输出数据，综合评估其质量。
    此函数重新实现了7个核心指标。

    参数:
        output_data (dict): 从 output_positions.json 加载的数据。
    
    返回:
        dict: 包含所有7个评估指标结果的字典。
    """
    # --- 1. 初始化 ---
    # 按帧ID排序，确保处理顺序正确
    sorted_frames = sorted(output_data.items(), key=lambda item: int(item[0]))
    
    if not sorted_frames:
        return {metric: 0 for metric in ['OCC', 'INT', 'S_overlap', 'S_position', 'S_aesthetics', 'S_angle_smoothness', 'S_distance_smoothness']}

    M = len(sorted_frames)  # 总帧数
    N = len(sorted_frames[0][1]) # 标签数 (从第一帧获取)

    # 初始化所有指标的累加器
    total_occ_count = 0
    total_int_count = 0
    total_label_overlap_area = 0
    total_feature_overlap_area = 0
    total_position_distance = 0
    total_intersection_events = 0
    total_angle_change = 0
    total_distance_change = 0

    # 用于计算平滑度的前一帧数据
    prev_angles = {}
    prev_distances = {}

    # --- 2. 逐帧计算指标 ---
    for frame_idx, (frame_key, frame_data) in enumerate(sorted_frames):
        
        labels = list(frame_data.values())
        label_ids = list(frame_data.keys())

        # --- 当前帧的临时累加器 ---
        frame_intersections = 0
        
        # --- 计算 S_overlap (标签-标签重叠, 标签-特征点重叠) ---
        for i in range(N):
            label_i_data = labels[i]
            bbox_i = label_i_data['bbox']
            li_x, li_y, li_w, li_h = bbox_i
            
            # 标签-特征点重叠 (假设特征点为一个半径为1的区域)
            anchor_i = label_i_data['anchor']
            # 特征点的AABB (Axis-Aligned Bounding Box)
            fi_x, fi_y = anchor_i[0] - 1, anchor_i[1] - 1
            fi_w, fi_h = 2, 2
            x_overlap_feat = max(0, min(li_x + li_w, fi_x + fi_w) - max(li_x, fi_x))
            y_overlap_feat = max(0, min(li_y + li_h, fi_y + fi_h) - max(li_y, fi_y))
            total_feature_overlap_area += x_overlap_feat * y_overlap_feat

            for j in range(i + 1, N): # i+1 避免重复计算和自比较
                label_j_data = labels[j]
                bbox_j = label_j_data['bbox']
                lj_x, lj_y, lj_w, lj_h = bbox_j

                # 标签-标签重叠面积
                x_overlap_label = max(0, min(li_x + li_w, lj_x + lj_w) - max(li_x, lj_x))
                y_overlap_label = max(0, min(li_y + li_h, lj_y + lj_h) - max(li_y, lj_y))
                total_label_overlap_area += x_overlap_label * y_overlap_label
        
        
        # --- 计算 S_position, S_aesthetics, Smoothness, OCC, INT ---
        current_angles = {}
        current_distances = {}

        for i in range(N):
            label_i_id = label_ids[i]
            label_i_data = labels[i]
            bbox_i = label_i_data['bbox']
            anchor_i = tuple(label_i_data['anchor'])
            center_i = (bbox_i[0] + bbox_i[2] / 2, bbox_i[1] + bbox_i[3] / 2)

            # S_position: 累加标签中心到锚点的距离
            distance = math.hypot(center_i[0] - anchor_i[0], center_i[1] - anchor_i[1])
            total_position_distance += distance

            # 平滑度指标所需数据
            angle = math.atan2(center_i[1] - anchor_i[1], center_i[0] - anchor_i[0])
            current_angles[label_i_id] = angle
            current_distances[label_i_id] = distance

            # 计算角度和距离变化 (如果不是第一帧)
            if frame_idx > 0:
                if label_i_id in prev_angles:
                    angle_diff = abs(angle - prev_angles[label_i_id])
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff
                    total_angle_change += angle_diff
                
                if label_i_id in prev_distances:
                    dist_diff = abs(distance - prev_distances[label_i_id])
                    total_distance_change += dist_diff

            # OCC: 计算标签i遮挡了多少其他对象
            occlusions_by_i = 0
            for j in range(N):
                if i == j: continue
                # 遮挡其他标签
                bbox_j = labels[j]['bbox']
                if not (bbox_i[0] + bbox_i[2] < bbox_j[0] or bbox_i[0] > bbox_j[0] + bbox_j[2] or
                        bbox_i[1] + bbox_i[3] < bbox_j[1] or bbox_i[1] > bbox_j[1] + bbox_j[3]):
                    occlusions_by_i += 1
                
                # 遮挡其他特征点
                anchor_j = labels[j]['anchor']
                if (bbox_i[0] <= anchor_j[0] <= bbox_i[0] + bbox_i[2] and
                    bbox_i[1] <= anchor_j[1] <= bbox_i[1] + bbox_i[3]):
                     occlusions_by_i +=1
            total_occ_count += occlusions_by_i

            # S_aesthetics 和 INT (引导线交叉)
            for j in range(i + 1, N):
                label_j_data = labels[j]
                anchor_j = tuple(label_j_data['anchor'])
                center_j = (label_j_data['bbox'][0] + label_j_data['bbox'][2] / 2, 
                            label_j_data['bbox'][1] + label_j_data['bbox'][3] / 2)

                if lines_intersect(center_i, anchor_i, center_j, anchor_j):
                    frame_intersections += 1
        
        total_intersection_events += frame_intersections
        prev_angles = current_angles
        prev_distances = current_distances

    # --- 3. 计算最终平均值 (严格按照公式修正) ---
    # 避免除以零
    if N == 0 or M == 0: return {}

    # OCC: 平均每个标签在所有帧上遮挡的对象数量
    occ = (total_occ_count / M) / N if M > 0 else 0
    
    # INT: 平均每条引导线在所有帧上被交叉的次数
    # 每次交叉涉及2条线，所以乘以2
    int_metric = (total_intersection_events * 2 / M) / N if M > 0 else 0

    # 论文五指标 (严格按照公式修正)
    s_overlap = (total_label_overlap_area + total_feature_overlap_area) / M

    # 修正：分母从 M*N 改为 M，代表“平均每帧的总距离”
    s_position = total_position_distance / M

    # 修正：反映公式中的双重计数，代表“平均每帧引导线被交叉的总次数”
    s_aesthetics = (total_intersection_events * 2) / M

    if M > 1:
        # 修正：分母移除 N，代表“平均每次帧转换的总变化”
        s_angle_smoothness = math.degrees(total_angle_change / (M - 1))
        s_distance_smoothness = total_distance_change / (M - 1)
    else:
        s_angle_smoothness = 0
        s_distance_smoothness = 0
        
    return {
        'OCC ': occ,
        'INT ': int_metric,
        'S_overlap ': s_overlap,
        'S_position ': s_position,
        'S_aesthetics ': s_aesthetics,
        'S_angle_smoothness ': s_angle_smoothness,
        'S_distance_smoothness ': s_distance_smoothness
    }

# ======================================================================
# 主执行模块 (Main Execution Block)
# ======================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估标签布局算法的输出质量。")
    parser.add_argument("output_file", type=str, help="算法生成的 output_positions.json 文件路径。")
    
    args = parser.parse_args()
    
    try:
        with open(args.output_file, 'r') as f:
            placement_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{args.output_file}'")
        exit(1)
    except json.JSONDecodeError:
        print(f"错误: 文件 '{args.output_file}' 不是一个有效的JSON文件。")
        exit(1)
    # 执行评估
    quality_metrics = evaluate_placement_quality(placement_data)
    
    # 打印结果
    
    print("\n--- 标签布局质量评估结果 ---")
    for metric, value in quality_metrics.items():
        print(f"{metric:<35}: {value:.4f}")
    print("---------------------------------")
