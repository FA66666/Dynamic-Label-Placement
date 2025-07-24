import json
import math
import os
import numpy as np
from scipy.integrate import quad
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
    x1, y1, w1, h1 = r1; x2, y2, w2, h2 = r2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return x_overlap * y_overlap

def point_to_rect_shortest_distance(point, rect):
    """计算一个点到一个矩形的最短距离。"""
    px, py = point; rx, ry, rw, rh = rect
    closest_x = max(rx, min(px, rx + rw))
    closest_y = max(ry, min(py, ry + rh))
    return math.hypot(px - closest_x, py - closest_y)

def precise_bezier_arc_length(p0, p1, p2, p3):
    """使用数值积分精确计算三阶贝塞尔曲线的弧长。"""
    def integrand(t):
        inv_t = 1.0 - t
        dx_dt = 3*inv_t**2*(p1.x-p0.x) + 6*inv_t*t*(p2.x-p1.x) + 3*t**2*(p3.x-p2.x)
        dy_dt = 3*inv_t**2*(p1.y-p0.y) + 6*inv_t*t*(p2.y-p1.y) + 3*t**2*(p3.y-p2.y)
        return math.sqrt(dx_dt**2 + dy_dt**2)
    length, _ = quad(integrand, 0, 1)
    return length

def find_best_avoidance_position(label_to_move, all_labels, collision_frame_idx, placer, current_frame_t):
    """为某个标签寻找一个最优的躲避位置。"""
    min_cost = float('inf')
    best_position = None
    w, h = label_to_move.feature.size

    other_bboxes = {}
    for other_id, other_label in all_labels.items():
        if other_id == label_to_move.id: continue
        other_bboxes[other_id] = get_predicted_bbox(other_label, current_frame_t, collision_frame_idx - current_frame_t)

    if (current_frame_t - 1) not in label_to_move.history: return None
    start_anchor = label_to_move.feature.get_position_at(current_frame_t - 1)
    if start_anchor is None: return None
    anchor_to_move = label_to_move.feature.predict_future(collision_frame_idx - current_frame_t)
    prev_angle = label_to_move.previous_angle
    start_offset = label_to_move.current_offset # 起始偏移就是当前的偏移

    for radius in config.radii_to_search:
        for i in range(config.points_per_circle):
            angle = 2 * math.pi * i / config.points_per_circle
            cand_center_x = anchor_to_move.x + radius * math.cos(angle)
            cand_center_y = anchor_to_move.y + radius * math.sin(angle)
            cand_x = cand_center_x - w / 2; cand_y = cand_center_y - h / 2
            candidate_bbox = (cand_x, cand_y, w, h)
            
            cost_overlap = 0.0
            for other_bbox in other_bboxes.values():
                cost_overlap += get_overlap_area(candidate_bbox, other_bbox)
            
            p0 = start_offset
            p3 = Vec2(candidate_bbox[0], candidate_bbox[1]) - anchor_to_move
            p1 = p0 + (p3 - p0) * 0.3 + Vec2(-(p3 - p0).y, (p3 - p0).x) * 0.3
            p2 = p0 + (p3 - p0) * 0.7 + Vec2(-(p3 - p0).y, (p3 - p0).x) * 0.3
            cost_move = precise_bezier_arc_length(p0, p1, p2, p3)
            cost_line = point_to_rect_shortest_distance(anchor_to_move.as_tuple, candidate_bbox)
            
            cand_angle = math.degrees(math.atan2(cand_center_y - anchor_to_move.y, cand_center_x - anchor_to_move.x))
            if prev_angle is not None:
                angle_diff = abs(cand_angle - prev_angle)
                cost_angle = min(angle_diff, 360.0 - angle_diff)
            else:
                cost_angle = 0
            
            total_cost = (config.W_OVERLAP * cost_overlap + config.W_MOVE * cost_move + config.W_LINE * cost_line + config.W_ANGLE * cost_angle)

            if total_cost < min_cost:
                min_cost = total_cost
                best_position = candidate_bbox
    return best_position

# --- 预测逻辑修正：彻底移除对 pos_model 的依赖 ---
def get_predicted_bbox(label, t, k):
    """计算一个标签在未来k帧后的预测边界框。"""
    predicted_anchor = label.feature.predict_future(k)
    w, h = label.feature.size
    
    if label.current_offset:
        pos = predicted_anchor + label.current_offset
        return (pos.x, pos.y, w, h)
    
    # 如果没有current_offset（只可能在初始化阶段发生），则无法预测
    return None

def main():
    """主函数"""
    with open(config.INPUT_FILE, 'r') as f: all_frames_data = json.load(f)['frames']
    num_frames = len(all_frames_data)

    features = {}; frame_0_points = all_frames_data[0]['points']
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
        
        # --- 关键修正：在初始化时，根据第0帧布局计算并存储初始偏移量 ---
        initial_offset = Vec2(bbox[0], bbox[1]) - feature.get_position_at(0)
        label.current_offset = initial_offset
        
        labels[pid] = label

    final_positions = {0: {pid: {'anchor': l.feature.get_position_at(0).as_tuple, 'bbox': l.history[0]['bbox']} for pid, l in labels.items()}}
    
    for t in range(1, num_frames):
        for pid, feature in features.items():
            if t < len(all_frames_data) and pid in all_frames_data[t]['points']:
                observed_pos = all_frames_data[t]['points'][pid]['anchor']
                feature.add_position(t, observed_pos)
                feature.update(np.array(observed_pos))
            feature.predict_step()
            
        for pid, label in labels.items():
            if label.avoidance_plan and label.avoidance_plan['end_frame'] > t:
                is_plan_still_needed = False
                for k in range(1, config.PREDICTION_WINDOW + 1):
                    # --- 计划复核修正：使用正确的预测函数 ---
                    hypothetical_bbox = get_predicted_bbox(label, t, k)
                    
                    for other_pid, other_label in labels.items():
                        if pid == other_pid: continue
                        other_bbox = get_predicted_bbox(other_label, t, k)
                        if check_aabb_collision(hypothetical_bbox, other_bbox):
                            is_plan_still_needed = True
                            break
                    if is_plan_still_needed:
                        break
                
                if not is_plan_still_needed:
                    label.cancel_avoidance_plan()
        
        collisions = {}
        label_ids = list(labels.keys())
        for i in range(len(label_ids)):
            for j in range(i + 1, len(label_ids)):
                id_A, id_B = label_ids[i], label_ids[j]
                for k in range(1, config.PREDICTION_WINDOW + 1):
                    future_t = t + k
                    if future_t >= num_frames: break
                    
                    # --- 碰撞预测修正：使用正确的预测函数 ---
                    bbox_A = get_predicted_bbox(labels[id_A], t, k)
                    bbox_B = get_predicted_bbox(labels[id_B], t, k)

                    if check_aabb_collision(bbox_A, bbox_B):
                        label_to_move = labels[id_B] if int(id_B) > int(id_A) else labels[id_A]
                        if label_to_move.id in collisions: continue
                        target_bbox = find_best_avoidance_position(label_to_move, labels, future_t, placer, t)
                        if target_bbox:
                            collisions[label_to_move.id] = {"at": future_t, "bbox": target_bbox}
                        break

        for pid, info in collisions.items():
            label = labels[pid]
            if label.avoidance_plan and label.avoidance_plan['end_frame'] > t: continue
            
            # --- 设置计划修正：起始偏移永远是 current_offset ---
            start_offset = label.current_offset
            
            predicted_target_anchor = features[pid].predict_future(info['at'] - t)
            target_offset = Vec2(*info['bbox'][:2]) - predicted_target_anchor
            label.set_avoidance_plan(t, info['at'], start_offset, target_offset)

        frame_t_pos = {}
        for pid, label in labels.items():
            bbox, model = label.calculate_pos_at_frame(t, placer)
            anchor_pos = label.feature.get_position_at(t)
            if anchor_pos is None: anchor_pos = label.feature.predict_future(0)
            frame_t_pos[pid] = {'anchor': anchor_pos.as_tuple, 'bbox': bbox, 'pos_model': model}
        final_positions[t] = frame_t_pos
        
    serializable_final_positions = {}
    for frame, data in final_positions.items():
        serializable_final_positions[str(frame)] = {}
        for pid, pos_data in data.items():
            serializable_final_positions[str(frame)][pid] = {'anchor': pos_data['anchor'], 'bbox': pos_data['bbox']}
    with open(config.OUTPUT_FILE, 'w') as f:
        json.dump(serializable_final_positions, f, indent=2)
    print(f"处理完成，结果已保存至 '{config.OUTPUT_FILE}'")

    quality_metrics = evaluate_placement_quality(serializable_final_positions)
    print("\n--- 标签布局质量评估结果 ---")
    for metric, value in quality_metrics.items():
        print(f"{metric:<35}: {value:.4f}")

if __name__ == '__main__':
    main()