"""
metrics.py

封装评价指标计算，独立于主流程。
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
