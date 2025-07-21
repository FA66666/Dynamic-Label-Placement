import json
import math
import os
import numpy as np
from scipy.integrate import quad # <--- 导入新的库
from bitmap_placer import BitmapPlacer
from core_logic import Feature, Label,Vec2
import config
from quality_evaluator import evaluate_placement_quality

def check_aabb_collision(r1, r2):
    """检查两个矩形 (x, y, w, h) 是否碰撞。"""
    if r1 is None or r2 is None: return False
    x1,y1,w1,h1=r1;x2,y2,w2,h2=r2;return not(x1+w1<x2 or x1>x2+w2 or y1+h1<y2 or y1>y2+h2)

def get_overlap_area(r1, r2):
    """计算两个矩形 r1 和 r2 的重叠面积。"""
    if r1 is None or r2 is None: return 0.0
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return x_overlap * y_overlap

def precise_bezier_arc_length(p0, p1, p2, p3):
    """
    使用数值积分 (高斯求积) 精确计算三阶贝塞尔曲线的弧长。
    L = ∫[0 to 1] sqrt( (dx/dt)^2 + (dy/dt)^2 ) dt
    """

    def integrand(t):
        # 首先计算贝塞尔曲线在t点的导数(dx/dt, dy/dt)，它代表了速度向量
        # B'(t) = 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2)
        inv_t = 1.0 - t
        
        # 速度向量的x和y分量
        dx_dt = 3 * inv_t**2 * (p1.x - p0.x) + 6 * inv_t * t * (p2.x - p1.x) + 3 * t**2 * (p3.x - p2.x)
        dy_dt = 3 * inv_t**2 * (p1.y - p0.y) + 6 * inv_t * t * (p2.y - p1.y) + 3 * t**2 * (p3.y - p2.y)
        
        # 速率是速度向量的模长
        speed = math.sqrt(dx_dt**2 + dy_dt**2)
        return speed

    # 调用scipy的quad函数进行数值积分
    length, _ = quad(integrand, 0, 1)
    return length

def find_best_avoidance_position(label_to_move, all_labels, collision_frame_idx, placer, current_frame_t):
    """
    为某个标签寻找一个最优的躲避位置。
    通过最小化一个包含重叠面积、移动距离、引导线长度和角度变化的成本函数来寻找最优解。
    """
    min_cost = float('inf')
    best_position = None
    w, h = label_to_move.feature.size

    other_bboxes = {}
    for other_id, other_label in all_labels.items():
        if other_id == label_to_move.id: continue
        predicted_anchor_pos = other_label.feature.predict_future(collision_frame_idx - current_frame_t)
        if other_label.current_offset:
            other_w, other_h = other_label.feature.size
            final_pos = predicted_anchor_pos + other_label.current_offset
            other_bboxes[other_id] = (final_pos.x, final_pos.y, other_w, other_h)
        else:
            if (current_frame_t - 1) in other_label.history:
                model = other_label.history[current_frame_t - 1].get('pos_model')
                if model is not None:
                    other_bboxes[other_id] = placer.get_position_box(other_id, model, predicted_anchor_pos.as_tuple)

    if (current_frame_t - 1) not in label_to_move.history: return None
    start_bbox = label_to_move.history[current_frame_t - 1]['bbox']
    start_anchor = label_to_move.feature.get_position_at(current_frame_t - 1)
    if start_anchor is None: return None
    anchor_to_move = label_to_move.feature.predict_future(collision_frame_idx - current_frame_t)
    prev_angle = label_to_move.previous_angle

    for radius in config.radii_to_search:
        for i in range(config.points_per_circle):
            angle = 2 * math.pi * i / config.points_per_circle
            
            cand_center_x = anchor_to_move.x + radius * math.cos(angle)
            cand_center_y = anchor_to_move.y + radius * math.sin(angle)
            cand_x = cand_center_x - w / 2
            cand_y = cand_center_y - h / 2
            candidate_bbox = (cand_x, cand_y, w, h)
            
            cost_overlap = 0.0
            for other_bbox in other_bboxes.values():
                cost_overlap += get_overlap_area(candidate_bbox, other_bbox)

            p0 = Vec2(start_bbox[0], start_bbox[1]) - start_anchor
            p3 = Vec2(candidate_bbox[0], candidate_bbox[1]) - anchor_to_move
            p1 = p0 + (p3 - p0) * 0.3 + Vec2(-(p3 - p0).y, (p3 - p0).x) * 0.3
            p2 = p0 + (p3 - p0) * 0.7 + Vec2(-(p3 - p0).y, (p3 - p0).x) * 0.3
            cost_move = precise_bezier_arc_length(p0, p1, p2, p3)

            cost_line = math.hypot(cand_center_x - anchor_to_move.x, cand_center_y - anchor_to_move.y)

            cand_angle = math.atan2(cand_center_y - anchor_to_move.y, cand_center_x - anchor_to_move.x)
            if prev_angle is not None:
                angle_diff = abs(cand_angle - prev_angle)
                cost_angle = min(angle_diff, 2 * math.pi - angle_diff)
            else:
                cost_angle = 0
                
            total_cost = (config.W_OVERLAP * cost_overlap + 
                          config.W_MOVE * cost_move + 
                          config.W_LINE * cost_line + 
                          config.W_ANGLE * cost_angle)

            if total_cost < min_cost:
                min_cost = total_cost
                best_position = candidate_bbox

    return best_position

def main():
    """主函数"""
    with open(config.INPUT_FILE, 'r') as f: all_frames_data = json.load(f)['frames']
    num_frames = len(all_frames_data)

    features = {}
    frame_0_points = all_frames_data[0]['points']
    for pid, data in frame_0_points.items():
        size = (math.ceil(data["size"][0]), math.ceil(data["size"][1] + data["size"][2]))
        features[pid] = Feature(pid, data['text'], size, data['anchor'], dt=config.DT)
    
    placer = BitmapPlacer(config.SCREEN_WIDTH, config.SCREEN_HEIGHT, config.LABEL_RADIUS)
    initial_solution = placer.run_initial_placement(frame_0_points)
    
    labels = {}
    for pid, feature in features.items():
        label = Label(feature)
        pos_model = initial_solution[pid]
        bbox = placer.get_position_box(pid, pos_model, feature.get_position_at(0).as_tuple)
        label.add_frame_data(0, bbox, pos_model)
        label._update_angle(feature.get_position_at(0), bbox)
        labels[pid] = label

    final_positions = {0: {pid: {'anchor': l.feature.get_position_at(0).as_tuple, 'bbox': l.history[0]['bbox']} for pid, l in labels.items()}}
    
    for t in range(1, num_frames):
        # A. 更新卡尔曼滤波器
        for pid, feature in features.items():
            if t < len(all_frames_data) and pid in all_frames_data[t]['points']:
                observed_pos = all_frames_data[t]['points'][pid]['anchor']
                feature.add_position(t, observed_pos)
                feature.update(np.array(observed_pos))
            feature.predict_step()

        # B. 碰撞预测
        collisions = {}
        label_ids = list(labels.keys())
        for i in range(len(label_ids)):
            for j in range(i + 1, len(label_ids)):
                id_A, id_B = label_ids[i], label_ids[j]
                label_A, label_B = labels[id_A], labels[id_B]

                for k in range(1, config.PREDICTION_WINDOW + 1):
                    future_t = t + k
                    if future_t >= num_frames: break
                    
                    predicted_anchor_A = features[id_A].predict_future(k)
                    predicted_anchor_B = features[id_B].predict_future(k)
                    w_A, h_A = features[id_A].size
                    w_B, h_B = features[id_B].size
                    bbox_A, bbox_B = None, None

                    if label_A.current_offset:
                        pos_A = predicted_anchor_A + label_A.current_offset
                        bbox_A = (pos_A.x, pos_A.y, w_A, h_A)
                    elif (t - 1) in label_A.history:
                        model_A = label_A.history[t-1]['pos_model']
                        if model_A is not None:
                            bbox_A = placer.get_position_box(id_A, model_A, predicted_anchor_A.as_tuple)
                    
                    if label_B.current_offset:
                        pos_B = predicted_anchor_B + label_B.current_offset
                        bbox_B = (pos_B.x, pos_B.y, w_B, h_B)
                    elif (t - 1) in label_B.history:
                        model_B = label_B.history[t-1]['pos_model']
                        if model_B is not None:
                            bbox_B = placer.get_position_box(id_B, model_B, predicted_anchor_B.as_tuple)

                    if check_aabb_collision(bbox_A, bbox_B):
                        label_to_move = label_B if int(id_B) > int(id_A) else label_A
                        if label_to_move.id in collisions: continue
                        
                        target_bbox = find_best_avoidance_position(label_to_move, labels, future_t, placer, t)
                        
                        if target_bbox:
                            collisions[label_to_move.id] = {"at": future_t, "bbox": target_bbox}
                        break

        # C. 设置或更新躲避计划
        for pid, info in collisions.items():
            label = labels[pid]
            if label.avoidance_plan and label.avoidance_plan['end_frame'] > t: continue
            
            if (t-1) not in label.history or label.feature.get_position_at(t-1) is None: continue
            
            start_offset = Vec2(*label.history[t-1]['bbox'][:2]) - label.feature.get_position_at(t-1)
            if label.current_offset: start_offset = label.current_offset
            
            predicted_target_anchor = features[pid].predict_future(info['at'] - t)
            target_offset = Vec2(*info['bbox'][:2]) - predicted_target_anchor
            
            label.set_avoidance_plan(t, info['at'], start_offset, target_offset)

        # D. 计算并记录最终位置
        frame_t_pos = {}
        for pid, label in labels.items():
            bbox, model = label.calculate_pos_at_frame(t, placer)
            anchor_pos = label.feature.get_position_at(t)
            if anchor_pos is None:
                anchor_pos = label.feature.predict_future(0)
            frame_t_pos[pid] = {'anchor': anchor_pos.as_tuple, 'bbox': bbox, 'pos_model': model}
        final_positions[t] = frame_t_pos
        
    # 5. 保存结果
    serializable_final_positions = {}
    for frame, data in final_positions.items():
        serializable_final_positions[str(frame)] = {}
        for pid, pos_data in data.items():
            serializable_final_positions[str(frame)][pid] = {'anchor': pos_data['anchor'], 'bbox': pos_data['bbox']}

    with open(config.OUTPUT_FILE, 'w') as f:
        json.dump(serializable_final_positions, f, indent=2)
    print(f"处理完成，结果已保存至 '{config.OUTPUT_FILE}'")

    # 6. 调用质量评估模块并打印结果
    quality_metrics = evaluate_placement_quality(serializable_final_positions)
    print("\n--- 标签布局质量评估结果 ---")
    for metric, value in quality_metrics.items():
        print(f"{metric:<35}: {value:.4f}")

if __name__ == '__main__':
    main()