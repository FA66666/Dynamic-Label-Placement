import json
import math
import numpy as np
import os

# --- 辅助类：用于二维向量操作 ---
class Vec2:
    """一个简单的二维向量类，用于处理位置和速度。"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)

    def __repr__(self):
        return f"Vec2({self.x}, {self.y})"

    @property
    def as_tuple(self):
        return (self.x, self.y)

# --- 经过重构的 BitmapPlacer 类，用于初始布局 ---
class BitmapPlacer:
    """
    封装了基于位图的方法，用于初始的、静态的标签布局。
    从用户提供的 bitmap_method.py 重构而来，成为一个可复用的类。
    """
    def __init__(self, screen_width, screen_height, radius=1, n_val=32):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.radius = radius
        self.n = n_val
        self.label_data = {}
        self.anchor_x_arr = []
        self.anchor_y_arr = []
        self.label_size_arr = []
        self.min_height = 99999
        self.point_id_map = []

    def _get_array_index(self, minx, row, width):
        row_start_bit_x = row * self.screen_width + minx
        n_val = self.n
        array_index_start = row_start_bit_x // n_val
        bit_start_offset = row_start_bit_x % n_val
        array_index_end = (row_start_bit_x + width) // n_val
        bit_end_offset = (row_start_bit_x + width) % n_val
        return array_index_start, bit_start_offset, array_index_end, bit_end_offset

    def _parse_data(self, points_data):
        self.label_data = points_data
        self.anchor_x_arr, self.anchor_y_arr, self.label_size_arr = [], [], []
        # 创建一个从 point_id 到数组索引的一致性映射
        self.point_id_map = sorted(points_data.keys(), key=int)

        for point_id in self.point_id_map:
            p = points_data[point_id]
            # 这些数据仅用于初始布局
            self.anchor_x_arr.append(int(p["anchor"][0]))
            self.anchor_y_arr.append(int(p["anchor"][1]))
            self.label_size_arr.append((math.ceil(p["size"][0]), math.ceil(p["size"][1] + p["size"][2])))

    def _do_prep(self, points_data):
        self._parse_data(points_data)
        self.min_height = 99999
        for label_size in self.label_size_arr:
            if label_size[1] > 0:
                 self.min_height = min(label_size[1], self.min_height)

    def _position_model(self, point_index, m, anchor_x, anchor_y):
        width, height = self.label_size_arr[point_index]
        r = self.radius
        
        positions = {
            1: (anchor_x + r, anchor_y - height - r, width, height),
            2: (anchor_x - width - r, anchor_y - height - r, width, height),
            3: (anchor_x - width - r, anchor_y + r, width, height),
            4: (anchor_x + r, anchor_y + r, width, height),
            5: (anchor_x + r, int(anchor_y - height / 2), width, height),
            6: (int(anchor_x - width / 2), anchor_y - height - r, width, height),
            7: (anchor_x - width - r, int(anchor_y - height / 2), width, height),
            8: (int(anchor_x - width / 2), anchor_y + r, width, height)
        }
        return positions.get(m)

    def _init_s(self):
        size = (self.screen_height * self.screen_width + self.n - 1) // self.n
        return np.zeros(size, dtype=np.uint32)

    def _deal_with_node(self, rec_pos):
        x, y, w, h = rec_pos
        if x < 0:
            w += x
            x = 0
        if y < 0:
            h += y
            y = 0
        if x + w > self.screen_width:
            w = self.screen_width - x
        if y + h > self.screen_height:
            h = self.screen_height - y
        return [x, y, w, h]

    def _update_bitmap(self, bitmap, label_pos, remove=False):
        minx, miny, width, height = label_pos
        full_mask = (1 << self.n) - 1

        for row in range(miny, miny + height):
            if not (0 <= row < self.screen_height): continue
                
            array_index_start, bit_start_offset, array_index_end, bit_end_offset = self._get_array_index(minx, row, width)
            
            mask = (full_mask >> bit_start_offset) if bit_start_offset > 0 else full_mask
            if array_index_end > array_index_start:
                if not remove:
                    bitmap[array_index_start] |= mask
                else:
                    bitmap[array_index_start] &= ~mask

                for i in range(array_index_start + 1, array_index_end):
                    if not remove:
                        bitmap[i] = full_mask
                    else:
                        bitmap[i] = 0
                
                end_mask = full_mask ^ (full_mask >> bit_end_offset)
                if not remove:
                    bitmap[array_index_end] |= end_mask
                else:
                    bitmap[array_index_end] &= ~end_mask
            else:
                end_mask = full_mask ^ (full_mask >> bit_end_offset)
                final_mask = mask & end_mask
                if not remove:
                    bitmap[array_index_start] |= final_mask
                else:
                    bitmap[array_index_start] &= ~final_mask
        return bitmap

    def _lookup(self, bitmap, label_pos):
        minx, miny, width, height = label_pos
        full_mask = (1 << self.n) - 1

        for row in range(miny, miny + height):
            if not (0 <= row < self.screen_height): continue
            
            array_index_start, bit_start_offset, array_index_end, bit_end_offset = self._get_array_index(minx, row, width)
            
            if array_index_end > array_index_start:
                if (bitmap[array_index_start] & (full_mask >> bit_start_offset)) != 0: return False
                for i in range(array_index_start + 1, array_index_end):
                    if bitmap[i] != 0: return False
                if (bitmap[array_index_end] & (full_mask ^ (full_mask >> bit_end_offset))) != 0: return False
            else:
                mask = (full_mask >> bit_start_offset) & (full_mask ^ (full_mask >> bit_end_offset))
                if (bitmap[array_index_start] & mask) != 0: return False
        return True
    
    def _is_over_screen(self, obj):
        minx, miny, width, height = obj
        return not (minx >= 0 and miny >=0 and minx + width <= self.screen_width and miny + height <= self.screen_height)

    def _bitmaplabeling(self):
        solution = {}
        bitmap = self._init_s()
        
        # 使用第0帧的锚点进行初始化
        for i in range(len(self.anchor_x_arr)):
            r = self.radius
            rec_pos = self._deal_with_node([self.anchor_x_arr[i] - r, self.anchor_y_arr[i] - r, 2 * r, 2 * r])
            bitmap = self._update_bitmap(bitmap, [int(c) for c in rec_pos])
            
        # 基于第0帧的锚点放置标签
        for i in range(len(self.anchor_x_arr)):
            for m in range(1, 9):
                # 在这个一次性的布局中使用初始的锚点坐标
                label_pos = self._position_model(i, m, self.anchor_x_arr[i], self.anchor_y_arr[i])
                if self._is_over_screen(label_pos):
                    continue
                
                label_pos_int = tuple(map(int, label_pos))

                if self._lookup(bitmap, label_pos_int):
                    self._update_bitmap(bitmap, label_pos_int)
                    solution[self.point_id_map[i]] = m
                    break
        return solution

    def run_initial_placement(self, points_data):
        self._do_prep(points_data)
        solution = self._bitmaplabeling()
        
        # 确保所有点都有一个默认位置，以防未被放置
        for point_id in self.point_id_map:
             if point_id not in solution:
                 solution[point_id] = 4 # 默认为右上方
        return solution

    def get_position_box(self, point_id_str, pos_model, points_data):
        """公共方法，用于获取给定点和位置模型的边界框。"""
        if not self.label_data:
            self._parse_data(points_data)
            
        point_index = self.point_id_map.index(point_id_str)
        
        # 从 points_data 获取当前帧的锚点坐标
        current_anchor = points_data[point_id_str]['anchor']
        anchor_x, anchor_y = int(current_anchor[0]), int(current_anchor[1])

        return self._position_model(point_index, pos_model, anchor_x, anchor_y)

# --- 核心动态标签类 ---
class DynamicLabel:
    """管理单个标签在所有帧中的状态和移动。"""
    def __init__(self, point_id, frame_data, pos_model, initial_bbox):
        self.id = point_id
        self.history = {}
        self.avoidance_plan = None # 存储躲避计划 {start_frame, end_frame, start_offset, target_offset}
        
        anchor_pos = Vec2(frame_data['anchor'][0], frame_data['anchor'][1])
        
        self.add_frame_data(0, anchor_pos, initial_bbox, pos_model)

    def add_frame_data(self, frame_idx, anchor_pos, bbox, pos_model):
        self.history[frame_idx] = {
            'anchor': anchor_pos,
            'bbox': bbox,
            'pos_model': pos_model
        }
    
    def set_avoidance_plan(self, start_frame, end_frame, start_offset, target_offset, target_pos_model):
        self.avoidance_plan = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_offset": start_offset,
            "target_offset": target_offset,
            "target_pos_model": target_pos_model
        }

    def cancel_avoidance_plan(self):
        self.avoidance_plan = None

    def get_pos_at_frame(self, frame_idx, all_frames_data, placer):
        """计算给定帧的标签边界框，如果需要则进行插值移动。"""
        current_anchor_data = all_frames_data[frame_idx]['points'][self.id]['anchor']
        current_anchor = Vec2(current_anchor_data[0], current_anchor_data[1])
        
        last_pos_model = self.history[frame_idx-1]['pos_model']
        
        # 如果有正在执行的躲避计划
        if self.avoidance_plan:
            plan = self.avoidance_plan
            start_frame = plan['start_frame']
            end_frame = plan['end_frame']

            # 检查计划是否仍在有效期内
            if start_frame <= frame_idx < end_frame:
                duration = end_frame - start_frame
                time_step = (frame_idx - start_frame) / duration if duration > 0 else 1
                
                # 使用三次贝塞尔曲线进行平滑插值
                p0 = plan['start_offset']
                p3 = plan['target_offset']
                # 控制点，用于产生缓和的曲线效果
                p1 = p0 + (p3-p0)*0.3 + Vec2(-(p3-p0).y, (p3-p0).x) * 0.3
                p2 = p0 + (p3-p0)*0.7 + Vec2(-(p3-p0).y, (p3-p0).x) * 0.3

                t = time_step
                inv_t = 1 - t
                
                offset = (p0 * (inv_t**3)) + (p1 * 3 * (inv_t**2) * t) + (p2 * 3 * inv_t * (t**2)) + (p3 * (t**3))

                size_w, size_h = placer.label_size_arr[placer.point_id_map.index(self.id)]
                new_pos = current_anchor + offset
                final_bbox = (new_pos.x, new_pos.y, size_w, size_h)
                self.add_frame_data(frame_idx, current_anchor, final_bbox, last_pos_model)
                return final_bbox, last_pos_model

            # 如果计划在当前帧正好结束
            elif frame_idx == end_frame:
                new_pos_model = plan['target_pos_model']
                self.cancel_avoidance_plan()
                final_bbox = placer.get_position_box(self.id, new_pos_model, all_frames_data[frame_idx]['points'])
                self.add_frame_data(frame_idx, current_anchor, final_bbox, new_pos_model)
                return final_bbox, new_pos_model

        # 默认行为：没有计划或计划已结束
        default_bbox = placer.get_position_box(self.id, last_pos_model, all_frames_data[frame_idx]['points'])
        self.add_frame_data(frame_idx, current_anchor, default_bbox, last_pos_model)
        return default_bbox, last_pos_model


# --- 主要应用逻辑 ---
def check_aabb_collision(rect1, rect2):
    """检查两个矩形 (x, y, w, h) 是否碰撞。"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return not (x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + h2)

# ##################################################################
# # 关键修正：函数签名增加了 current_frame_t 参数。                 #
# ##################################################################
def find_sanctuary(label_to_move, other_labels, collision_frame_idx, all_frames_data, placer, current_frame_t):
    """为标签寻找一个最佳的无碰撞位置模型。"""
    # 预测所有其他标签在碰撞帧的位置
    other_bboxes_at_collision = {}
    for other_id, other_label in other_labels.items():
        if other_id == label_to_move.id: continue
        
        # ##################################################################
        # # 关键修正：基于最近的已知状态 (t-1) 来获取位置模型。         #
        # ##################################################################
        other_pos_model = other_label.history[current_frame_t - 1]['pos_model']
        other_bboxes_at_collision[other_id] = placer.get_position_box(
            other_id, other_pos_model, all_frames_data[collision_frame_idx]['points']
        )
    
    # ##################################################################
    # # 关键修正：同样，基于 t-1 来获取要移动的标签的默认模型。         #
    # ##################################################################
    default_pos_model = label_to_move.history[current_frame_t - 1]['pos_model']
        
    # 搜索一个新的位置模型
    for m in range(1, 9):
        if m == default_pos_model: continue
        
        candidate_bbox = placer.get_position_box(label_to_move.id, m, all_frames_data[collision_frame_idx]['points'])
        
        is_clear = True
        for other_bbox in other_bboxes_at_collision.values():
            if check_aabb_collision(candidate_bbox, other_bbox):
                is_clear = False
                break
        
        if is_clear:
            return m, candidate_bbox
    
    return None, None # 未找到可用位置

def main():
    """主执行函数。"""
    # 1. 配置
    SCREEN_WIDTH, SCREEN_HEIGHT = 1000, 1000
    RADIUS = 2
    PREDICTION_WINDOW = 5
    INPUT_FILE = 'sample_generated.json'
    OUTPUT_FILE = 'output_positions.json'
    
    # 2. 加载数据
    with open(INPUT_FILE, 'r') as f:
        data = json.load(f)
    all_frames_data = data['frames']
    num_frames = len(all_frames_data)

    # 3. 初始布局 (第0帧)
    print("正在为第0帧执行初始布局...")
    placer = BitmapPlacer(SCREEN_WIDTH, SCREEN_HEIGHT, RADIUS)
    frame_0_points = all_frames_data[0]['points']
    initial_solution = placer.run_initial_placement(frame_0_points)
    
    # 4. 初始化动态标签对象
    labels = {}
    for point_id, point_data in frame_0_points.items():
        pos_model = initial_solution[point_id]
        initial_bbox = placer.get_position_box(point_id, pos_model, frame_0_points)
        labels[point_id] = DynamicLabel(point_id, point_data, pos_model, initial_bbox)

    final_positions = {0: {lid: l.history[0] for lid, l in labels.items()}}
    print("初始布局完成。")

    # 5. 主循环 (逐帧模拟)
    for t in range(1, num_frames):
        print(f"正在处理第 {t}/{num_frames-1} 帧...", end='\r')
        
        # --- A. 碰撞预测阶段 ---
        collisions_found = {} # {label_id_to_move: {at_frame, target_model, target_bbox}}

        # 预测未来路径并检查重叠
        for label_id_A, label_A in labels.items():
            if label_A.avoidance_plan and label_A.avoidance_plan['start_frame'] == t: continue 

            for label_id_B, label_B in labels.items():
                if int(label_id_A) >= int(label_id_B): continue
                if label_B.avoidance_plan and label_B.avoidance_plan['start_frame'] == t: continue
                
                for k in range(1, PREDICTION_WINDOW + 1):
                    future_frame_idx = t + k
                    if future_frame_idx >= num_frames: break

                    # 预测A和B在未来帧的边界框
                    pos_model_A = labels[label_id_A].history[t-1]['pos_model']
                    pos_model_B = labels[label_id_B].history[t-1]['pos_model']
                    
                    future_points_data = all_frames_data[future_frame_idx]['points']
                    bbox_A = placer.get_position_box(label_id_A, pos_model_A, future_points_data)
                    bbox_B = placer.get_position_box(label_id_B, pos_model_B, future_points_data)

                    if check_aabb_collision(bbox_A, bbox_B):
                        label_to_move = label_B if int(label_id_B) > int(label_id_A) else label_A
                        if label_to_move.id in collisions_found: continue
                        
                        # ##################################################################
                        # # 关键修正：在调用时传入当前的帧数 t。                          #
                        # ##################################################################
                        new_model, new_bbox = find_sanctuary(label_to_move, labels, future_frame_idx, all_frames_data, placer, t)
                        
                        if new_model:
                            collisions_found[label_to_move.id] = {
                                "at_frame": future_frame_idx,
                                "target_model": new_model,
                                "target_bbox": new_bbox,
                            }
                        break 
                if label_id_B in collisions_found: break
            if label_id_B in collisions_found: break
        
        # --- B. 更新状态和计划 ---
        for label_id, collision_info in collisions_found.items():
            label = labels[label_id]
            if label.avoidance_plan: continue # 已经有计划了

            start_bbox = label.history[t-1]['bbox']
            start_anchor = label.history[t-1]['anchor']
            start_offset = Vec2(start_bbox[0], start_bbox[1]) - start_anchor
            
            target_bbox = collision_info['target_bbox']
            target_anchor_data = all_frames_data[collision_info['at_frame']]['points'][label_id]['anchor']
            target_anchor = Vec2(target_anchor_data[0], target_anchor_data[1])
            target_offset = Vec2(target_bbox[0], target_bbox[1]) - target_anchor

            label.set_avoidance_plan(t, collision_info['at_frame'], start_offset, target_offset, collision_info['target_model'])

        # --- C. 计算第 t 帧的最终位置 ---
        frame_t_positions = {}
        for label_id, label in labels.items():
            bbox, pos_model = label.get_pos_at_frame(t, all_frames_data, placer)
            frame_t_positions[label_id] = label.history[t]
        
        final_positions[t] = frame_t_positions
    
    print(f"\n已完成全部 {num_frames} 帧的处理。")

    # 6. 保存结果
    # 将 Vec2 对象转换为元组以便进行JSON序列化
    for frame_idx, frame_data in final_positions.items():
        for label_id, label_data in frame_data.items():
            label_data['anchor'] = label_data['anchor'].as_tuple
            
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_positions, f, indent=2)
        
    print(f"输出结果已保存至 '{OUTPUT_FILE}'")

if __name__ == '__main__':
    main()