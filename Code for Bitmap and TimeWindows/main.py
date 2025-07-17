import json
import math
import os
import numpy as np
from bitmap_placer import BitmapPlacer
from core_logic import Feature, Label,Vec2

def check_aabb_collision(r1, r2):
    """检查两个矩形 (x, y, w, h) 是否碰撞。"""
    x1,y1,w1,h1=r1;x2,y2,w2,h2=r2;return not(x1+w1<x2 or x1>x2+w2 or y1+h1<y2 or y1>y2+h2)

def find_sanctuary(label_to_move, all_labels, collision_frame_idx, placer, current_frame_t):
    """为某个标签寻找一个无碰撞的“避难”位置模型。"""
    other_bboxes = {}
    for other_id, other_label in all_labels.items():
        if other_id == label_to_move.id: continue
        predicted_anchor_pos = other_label.feature.predict_future(collision_frame_idx - current_frame_t)
        other_pos_model = other_label.history[current_frame_t - 1]['pos_model']
        other_bboxes[other_id] = placer.get_position_box(other_id, other_pos_model, predicted_anchor_pos.as_tuple)

    anchor_to_move = label_to_move.feature.predict_future(collision_frame_idx - current_frame_t)
    default_pos_model = label_to_move.history[current_frame_t - 1]['pos_model']

    for m in range(1, 9):
        if m == default_pos_model: continue
        candidate_bbox = placer.get_position_box(label_to_move.id, m, anchor_to_move.as_tuple)
        if not any(check_aabb_collision(candidate_bbox, other_bbox) for other_bbox in other_bboxes.values()):
            return m, candidate_bbox
    return None, None

def main():
    """主函数"""
    # 1. 配置
    CONFIG = {'width': 1000, 'height': 1000, 'radius': 2, 'window': 5, 'dt': 1.0}
    INPUT_FILE, OUTPUT_FILE = 'sample_input.json', 'output_positions.json'
    
    # 2. 加载原始数据
    with open(INPUT_FILE, 'r') as f: all_frames_data = json.load(f)['frames']
    num_frames = len(all_frames_data)

    # 3. 创建并初始化 Feature 对象
    features = {}
    frame_0_points = all_frames_data[0]['points']
    for pid, data in frame_0_points.items():
        size = (math.ceil(data["size"][0]), math.ceil(data["size"][1] + data["size"][2]))
        features[pid] = Feature(pid, data['text'], size, data['anchor'], dt=CONFIG['dt'])
    
    # 4. 初始化 Placer 和 Label 对象
    placer = BitmapPlacer(CONFIG['width'], CONFIG['height'], CONFIG['radius'])
    initial_solution = placer.run_initial_placement(frame_0_points)
    
    labels = {}
    for pid, feature in features.items():
        label = Label(feature)
        pos_model = initial_solution[pid]
        bbox = placer.get_position_box(pid, pos_model, feature.get_position_at(0).as_tuple)
        label.add_frame_data(0, bbox, pos_model)
        labels[pid] = label

    # 5. 主循环
    final_positions = {0: {pid: {'anchor': l.feature.get_position_at(0).as_tuple, 'bbox': l.history[0]['bbox']} for pid, l in labels.items()}}
    
    for t in range(1, num_frames):
        # A. 更新卡尔曼滤波器
        for pid, feature in features.items():
            observed_pos = all_frames_data[t]['points'][pid]['anchor']
            feature.add_position(t, observed_pos)
            feature.update(np.array(observed_pos))
            feature.predict_step()

        # B. 碰撞预测
        collisions = {}
        for id_A, label_A in labels.items():
            # 不再跳过已有计划的标签，让它们也参与新一轮预测
            for id_B, label_B in labels.items():
                if int(id_A) >= int(id_B): continue
                
                for k in range(1, CONFIG['window'] + 1):
                    future_t = t + k
                    if future_t >= num_frames: break
                    
                    # 确定每个标签在做预测时，应该基于哪个位置模型
                    # 如果有计划，就基于计划的目标模型；否则基于上一帧的模型
                    model_A = label_A.avoidance_plan['target_pos_model'] if label_A.avoidance_plan else label_A.history[t-1]['pos_model']
                    model_B = label_B.avoidance_plan['target_pos_model'] if label_B.avoidance_plan else label_B.history[t-1]['pos_model']

                    predicted_anchor_A = features[id_A].predict_future(k)
                    predicted_anchor_B = features[id_B].predict_future(k)
                    
                    bbox_A = placer.get_position_box(id_A, model_A, predicted_anchor_A.as_tuple)
                    bbox_B = placer.get_position_box(id_B, model_B, predicted_anchor_B.as_tuple)

                    if check_aabb_collision(bbox_A, bbox_B):
                        label_to_move = label_B if int(id_B) > int(id_A) else label_A
                        if label_to_move.id in collisions: continue
                        
                        model, bbox = find_sanctuary(label_to_move, labels, future_t, placer, t)
                        if model:
                            collisions[label_to_move.id] = {"at": future_t, "model": model, "bbox": bbox}
                        break 
                if id_B in collisions and id_B == (label_B if int(id_B) > int(id_A) else label_A).id : break
            if id_A in collisions : break

        # C. 设置或更新躲避计划
        for pid, info in collisions.items():
            label = labels[pid]


            start_offset = Vec2(*label.history[t-1]['bbox'][:2]) - label.feature.get_position_at(t-1)
            predicted_target_anchor = features[pid].predict_future(info['at'] - t)
            target_offset = Vec2(*info['bbox'][:2]) - predicted_target_anchor
            label.set_avoidance_plan(t, info['at'], start_offset, target_offset, info['model'])

        # D. 计算并记录最终位置
        frame_t_pos = {}
        for pid, label in labels.items():
            bbox, model = label.calculate_pos_at_frame(t, placer)
            frame_t_pos[pid] = {'anchor': label.feature.get_position_at(t).as_tuple, 'bbox': bbox, 'pos_model': model}
        final_positions[t] = frame_t_pos
        
    # 6. 保存结果
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_positions, f, indent=2)
    print(f"处理完成，结果已保存至 '{OUTPUT_FILE}'")

if __name__ == '__main__':
    main()