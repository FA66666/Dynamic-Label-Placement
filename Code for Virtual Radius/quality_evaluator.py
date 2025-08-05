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
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def point_to_rect_distance(point, rect):
    """
    计算一个点到一个矩形的最短距离。
    
    参数:
        point (tuple): 点的坐标 (px, py)
        rect (tuple): 矩形的属性 (rx, ry, rw, rh)
    返回:
        float: 点到矩形的最短距离
    """
    px, py = point
    rx, ry, rw, rh = rect

    # 找到矩形上离点最近的点的x坐标 (通过“钳制”操作)
    closest_x = max(rx, min(px, rx + rw))

    # 找到矩形上离点最近的点的y坐标 (通过“钳制”操作)
    closest_y = max(ry, min(py, ry + rh))

    # 返回点与这个最近点之间的欧式距离
    return math.hypot(px - closest_x, py - closest_y)


# ======================================================================
# 核心评估函数 (Core Evaluation Function)
# ======================================================================

def evaluate_placement_quality(output_data):
    """
    根据标签放置算法的输出数据，综合评估其质量。
    此函数重新实现了7个核心指标。
    """
    # --- 1. 初始化 ---
    sorted_frames = sorted(output_data.items(), key=lambda item: int(item[0]))
    if not sorted_frames:
        return {metric: 0 for metric in ['OCC', 'INT', 'S_overlap', 'S_position', 'S_aesthetics', 'S_angle_smoothness', 'S_distance_smoothness']}

    M = len(sorted_frames)
    N = len(sorted_frames[0][1])

    total_occ_count = 0
    total_int_count = 0
    total_label_overlap_area = 0
    total_feature_overlap_area = 0
    total_position_distance = 0
    total_intersection_events = 0
    total_angle_change = 0
    total_distance_change = 0
    prev_angles, prev_distances = {}, {}

    # --- 2. 逐帧计算指标 ---
    for frame_idx, (frame_key, frame_data) in enumerate(sorted_frames):
        labels = list(frame_data.values())
        label_ids = list(frame_data.keys())
        frame_intersections = 0
        
        # S_overlap: 计算重叠面积
        for i in range(N):
            label_i_data = labels[i]; bbox_i = label_i_data['bbox']; li_x, li_y, li_w, li_h = bbox_i
            anchor_i = label_i_data['anchor']; fi_x, fi_y = anchor_i[0] - 1, anchor_i[1] - 1; fi_w, fi_h = 2, 2
            x_overlap_feat = max(0, min(li_x + li_w, fi_x + fi_w) - max(li_x, fi_x))
            y_overlap_feat = max(0, min(li_y + li_h, fi_y + fi_h) - max(li_y, fi_y))
            total_feature_overlap_area += x_overlap_feat * y_overlap_feat
            for j in range(i + 1, N):
                label_j_data = labels[j]; bbox_j = label_j_data['bbox']; lj_x, lj_y, lj_w, lj_h = bbox_j
                x_overlap_label = max(0, min(li_x + li_w, lj_x + lj_w) - max(li_x, lj_x))
                y_overlap_label = max(0, min(li_y + li_h, lj_y + lj_h) - max(li_y, lj_y))
                total_label_overlap_area += x_overlap_label * y_overlap_label
        
        current_angles, current_distances = {}, {}
        for i in range(N):
            label_i_id = label_ids[i]; label_i_data = labels[i]
            bbox_i = label_i_data['bbox']; anchor_i = tuple(label_i_data['anchor'])
            center_i = (bbox_i[0] + bbox_i[2] / 2, bbox_i[1] + bbox_i[3] / 2)

            # --- S_position 计算方式已修改 ---
            # 使用新的函数计算锚点到矩形的最短距离
            distance = point_to_rect_distance(anchor_i, bbox_i)
            total_position_distance += distance

            # 平滑度指标所需数据
            angle = math.atan2(center_i[1] - anchor_i[1], center_i[0] - anchor_i[0])
            current_angles[label_i_id] = angle
            current_distances[label_i_id] = distance

            if frame_idx > 0:
                if label_i_id in prev_angles:
                    angle_diff = abs(angle - prev_angles[label_i_id])
                    if angle_diff > math.pi: angle_diff = 2 * math.pi - angle_diff
                    total_angle_change += angle_diff
                if label_i_id in prev_distances:
                    total_distance_change += abs(distance - prev_distances[label_i_id])

            # OCC: 计算遮挡
            occlusions_by_i = 0
            for j in range(N):
                if i == j: continue
                bbox_j = labels[j]['bbox']
                if not (bbox_i[0] + bbox_i[2] < bbox_j[0] or bbox_i[0] > bbox_j[0] + bbox_j[2] or
                        bbox_i[1] + bbox_i[3] < bbox_j[1] or bbox_i[1] > bbox_j[1] + bbox_j[3]):
                    occlusions_by_i += 1
                anchor_j = labels[j]['anchor']
                if (bbox_i[0] <= anchor_j[0] <= bbox_i[0] + bbox_i[2] and
                    bbox_i[1] <= anchor_j[1] <= bbox_i[1] + bbox_i[3]):
                     occlusions_by_i +=1
            total_occ_count += occlusions_by_i

            # S_aesthetics 和 INT (引导线交叉)
            for j in range(i + 1, N):
                label_j_data = labels[j]; anchor_j = tuple(label_j_data['anchor'])
                center_j = (label_j_data['bbox'][0] + label_j_data['bbox'][2] / 2, 
                            label_j_data['bbox'][1] + label_j_data['bbox'][3] / 2)
                if lines_intersect(center_i, anchor_i, center_j, anchor_j):
                    frame_intersections += 1
        
        total_intersection_events += frame_intersections
        prev_angles, prev_distances = current_angles, current_distances

    # --- 3. 计算最终平均值 ---
    if N == 0 or M == 0: return {}
    occ = (total_occ_count / M) / N
    int_metric = (total_intersection_events * 2 / M) / N
    s_overlap = (total_label_overlap_area + total_feature_overlap_area) / M
    s_position = total_position_distance / M
    s_aesthetics = (total_intersection_events * 2) / M
    if M > 1:
        s_angle_smoothness = math.degrees(total_angle_change / (M - 1))
        s_distance_smoothness = total_distance_change / (M - 1)
    else:
        s_angle_smoothness, s_distance_smoothness = 0, 0
        
    return {
        'OCC': occ,
        'INT': int_metric,
        'S_overlap': s_overlap,
        'S_position': s_position,
        'S_aesthetics': s_aesthetics,
        'S_angle_smoothness': s_angle_smoothness,
        'S_distance_smoothness': s_distance_smoothness
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

    quality_metrics = evaluate_placement_quality(placement_data)
    
    print("\n--- 标签布局质量评估结果 ---")
    for metric, value in quality_metrics.items():
        print(f"{metric:<35}: {value:.4f}")
    print("---------------------------------")