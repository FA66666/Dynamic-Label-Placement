import json
import math
import os
import numpy as np
from bitmap_placer import BitmapPlacer
from core_logic import Feature, Label,Vec2
import config # 导入配置文件
from quality_evaluator import evaluate_placement_quality # 新增：导入评估函数

def check_aabb_collision(r1, r2):
    """检查两个矩形 (x, y, w, h) 是否碰撞。"""
    x1,y1,w1,h1=r1;x2,y2,w2,h2=r2;return not(x1+w1<x2 or x1>x2+w2 or y1+h1<y2 or y1>y2+h2)

def find_sanctuary_on_circles(label_to_move, all_labels, collision_frame_idx, placer, current_frame_t):
    """
    为某个标签在多个同心圆上寻找一个无碰撞的“避难”位置。
    """
    # --- 1. 定义搜索参数 ---
    # 定义多个搜索半径
    config.radii_to_search = [25, 35, 45, 55, 65]
    # 定义每个圆上要测试的点的数量
    config.points_per_circle = 16

    # --- 2. 准备其他标签的预测位置 ---
    other_bboxes = {}
    for other_id, other_label in all_labels.items():
        if other_id == label_to_move.id:
            continue
        # 获取其他标签在碰撞时刻的预测位置
        predicted_anchor_pos = other_label.feature.predict_future(collision_frame_idx - current_frame_t)
        
        # 检查其他标签是否有自定义的偏移量，否则使用其模型
        if other_label.current_offset:
            other_w, other_h = other_label.feature.size
            final_pos = predicted_anchor_pos + other_label.current_offset
            other_bboxes[other_id] = (final_pos.x, final_pos.y, other_w, other_h)
        else:
            other_pos_model = other_label.history[current_frame_t - 1]['pos_model']
            # FIX: Handle cases where a label was animating in the previous frame
            if other_pos_model is not None:
                other_bboxes[other_id] = placer.get_position_box(other_id, other_pos_model, predicted_anchor_pos.as_tuple)
            elif (current_frame_t - 1) in other_label.history:
                # Fallback for animating labels: use last known offset
                last_bbox = other_label.history[current_frame_t - 1]['bbox']
                last_anchor = other_label.feature.get_position_at(current_frame_t - 1)
                offset = Vec2(last_bbox[0] - last_anchor.x, last_bbox[1] - last_anchor.y)
                new_pos = predicted_anchor_pos + offset
                w, h = other_label.feature.size
                other_bboxes[other_id] = (new_pos.x, new_pos.y, w, h)


    # --- 3. 在同心圆上进行迭代搜索 ---
    anchor_to_move = label_to_move.feature.predict_future(collision_frame_idx - current_frame_t)
    w, h = label_to_move.feature.size

    for radius in config.radii_to_search:
        for i in range(config.points_per_circle):
            angle = 2 * math.pi * i / config.points_per_circle
            
            # 以预测的锚点为中心，计算候选位置的中心点
            candidate_center_x = anchor_to_move.x + radius * math.cos(angle)
            candidate_center_y = anchor_to_move.y + radius * math.sin(angle)
            
            # 计算候选框的左上角坐标 (x, y)
            candidate_x = candidate_center_x - w / 2
            candidate_y = candidate_center_y - h / 2
            
            candidate_bbox = (candidate_x, candidate_y, w, h)
            
            # 检查这个候选框是否与任何其他标签的预测框发生碰撞
            if not any(check_aabb_collision(candidate_bbox, other_bbox) for other_bbox in other_bboxes.values() if other_bbox is not None):
                # 如果未发生碰撞，则找到了一个安全的“避难所”
                return candidate_bbox # 直接返回这个安全的位置

    # 如果遍历完所有同心圆上的所有点都找不到位置，则返回 None
    return None


def main():
    """主函数"""
    # 1. 加载原始数据 (从 config 加载文件名)
    with open(config.INPUT_FILE, 'r') as f: all_frames_data = json.load(f)['frames']
    num_frames = len(all_frames_data)

    # 2. 创建并初始化 Feature 对象 (从 config 加载 dt)
    features = {}
    frame_0_points = all_frames_data[0]['points']
    for pid, data in frame_0_points.items():
        size = (math.ceil(data["size"][0]), math.ceil(data["size"][1] + data["size"][2]))
        features[pid] = Feature(pid, data['text'], size, data['anchor'], dt=config.DT)
    
    # 3. 初始化 Placer 和 Label 对象 (从 config 加载屏幕尺寸和半径)
    placer = BitmapPlacer(config.SCREEN_WIDTH, config.SCREEN_HEIGHT, config.LABEL_RADIUS)
    initial_solution = placer.run_initial_placement(frame_0_points)
    
    labels = {}
    for pid, feature in features.items():
        label = Label(feature)
        pos_model = initial_solution[pid]
        bbox = placer.get_position_box(pid, pos_model, feature.get_position_at(0).as_tuple)
        label.add_frame_data(0, bbox, pos_model)
        labels[pid] = label

    # 4. 主循环
    final_positions = {0: {pid: {'anchor': l.feature.get_position_at(0).as_tuple, 'bbox': l.history[0]['bbox']} for pid, l in labels.items()}}
    
    for t in range(1, num_frames):
        # A. 更新卡尔曼滤波器
        for pid, feature in features.items():
            observed_pos = all_frames_data[t]['points'][pid]['anchor']
            feature.add_position(t, observed_pos)
            feature.update(np.array(observed_pos))
            feature.predict_step()

        # B. 碰撞预测 (从 config 加载预测窗口)
        collisions = {}
        label_ids = list(labels.keys())
        for i in range(len(label_ids)):
            for j in range(i + 1, len(label_ids)):
                id_A, id_B = label_ids[i], label_ids[j]
                label_A, label_B = labels[id_A], labels[id_B]

                for k in range(1, config.PREDICTION_WINDOW + 1):
                    future_t = t + k
                    if future_t >= num_frames: break
                    
                    # 预测两个标签未来的锚点和边界框
                    predicted_anchor_A = features[id_A].predict_future(k)
                    predicted_anchor_B = features[id_B].predict_future(k)
                    w_A, h_A = features[id_A].size
                    w_B, h_B = features[id_B].size

                    bbox_A, bbox_B = None, None

                    # --- FIX START: Robust BBox Prediction ---
                    # 获取A的位置
                    if label_A.current_offset:
                        pos_A = predicted_anchor_A + label_A.current_offset
                        bbox_A = (pos_A.x, pos_A.y, w_A, h_A)
                    else:
                        model_A = label_A.history[t-1]['pos_model']
                        if model_A is not None:
                            bbox_A = placer.get_position_box(id_A, model_A, predicted_anchor_A.as_tuple)
                        else: # Fallback for animating labels
                            last_bbox = label_A.history[t-1]['bbox']
                            last_anchor = label_A.feature.get_position_at(t-1)
                            offset = Vec2(last_bbox[0] - last_anchor.x, last_bbox[1] - last_anchor.y)
                            new_pos = predicted_anchor_A + offset
                            bbox_A = (new_pos.x, new_pos.y, w_A, h_A)

                    # 获取B的位置
                    if label_B.current_offset:
                        pos_B = predicted_anchor_B + label_B.current_offset
                        bbox_B = (pos_B.x, pos_B.y, w_B, h_B)
                    else:
                        model_B = label_B.history[t-1]['pos_model']
                        if model_B is not None:
                            bbox_B = placer.get_position_box(id_B, model_B, predicted_anchor_B.as_tuple)
                        else: # Fallback for animating labels
                            last_bbox = label_B.history[t-1]['bbox']
                            last_anchor = label_B.feature.get_position_at(t-1)
                            offset = Vec2(last_bbox[0] - last_anchor.x, last_bbox[1] - last_anchor.y)
                            new_pos = predicted_anchor_B + offset
                            bbox_B = (new_pos.x, new_pos.y, w_B, h_B)
                    # --- FIX END ---

                    # 检查碰撞
                    if bbox_A and bbox_B and check_aabb_collision(bbox_A, bbox_B):
                        # 优先移动ID较大的标签
                        label_to_move = label_B if int(id_B) > int(id_A) else label_A
                        
                        # 如果该标签本帧已有躲避计划，则跳过
                        if label_to_move.id in collisions: continue
                        
                        # 调用新的同心圆搜索算法寻找避难所
                        target_bbox = find_sanctuary_on_circles(label_to_move, labels, future_t, placer, t)
                        
                        if target_bbox:
                            # 将找到的安全位置存入碰撞字典，不再需要model
                            collisions[label_to_move.id] = {"at": future_t, "bbox": target_bbox}
                        break  # 找到碰撞后即处理，跳出k循环

        # C. 设置或更新躲避计划
        for pid, info in collisions.items():
            label = labels[pid]
            
            # 如果该标签已经有躲避计划，则跳过，防止重复设置
            if label.avoidance_plan and label.avoidance_plan['end_frame'] > t:
                continue
            
            # 起始偏移量
            if label.current_offset:
                start_offset = label.current_offset
            else:
                start_offset = Vec2(*label.history[t-1]['bbox'][:2]) - label.feature.get_position_at(t-1)

            # 目标偏移量
            predicted_target_anchor = features[pid].predict_future(info['at'] - t)
            target_offset = Vec2(*info['bbox'][:2]) - predicted_target_anchor
            
            # 设置新的躲避计划，注意这里不再需要传入 model
            label.set_avoidance_plan(t, info['at'], start_offset, target_offset)

        # D. 计算并记录最终位置
        frame_t_pos = {}
        for pid, label in labels.items():
            bbox, model = label.calculate_pos_at_frame(t, placer)
            frame_t_pos[pid] = {'anchor': label.feature.get_position_at(t).as_tuple, 'bbox': bbox, 'pos_model': model}
        final_positions[t] = frame_t_pos
        
    # 5. 保存结果 (从 config 加载文件名)
    # 创建一个可序列化的版本
    serializable_final_positions = {}
    for frame, data in final_positions.items():
        serializable_final_positions[frame] = {}
        for pid, pos_data in data.items():
            # pos_model 可能是 None，这没问题
            serializable_final_positions[frame][pid] = {
                'anchor': pos_data['anchor'],
                'bbox': pos_data['bbox'],
            }

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