# 文件: bitmap_method.py

import json
import math
import numpy as np

# --- 全局变量定义 ---
n = 32
label_data = {}
anchor_x_arr, anchor_y_arr, label_size_arr = [], [], []
screen_width, screen_height = 1000, 1000
radius = 10
min_height = 10
file_path = ""
HEURISTIC_DISTANCE_THRESHOLD = 100 # 启发式规则的激活距离阈值

# --- 核心及辅助函数 ---

def get_angle_position(point_index, angle, r):
    """计算给定角度的标签位置（返回整数坐标）"""
    anchor_x, anchor_y = anchor_x_arr[point_index], anchor_y_arr[point_index]
    width, height = label_size_arr[point_index]
    distance = r + max(width, height) / 2
    center_x = anchor_x + distance * math.cos(angle)
    center_y = anchor_y + distance * math.sin(angle)
    x = int(round(center_x - width / 2))
    y = int(round(center_y - height / 2))
    return (x, y, width, height)

def get_array_index(minx, row, width):
    """计算矩形行在位图一维数组中的起始和结束索引及偏移量"""
    row_start_bit_x = row * screen_width + minx
    array_index_start = int(row_start_bit_x // n)
    bit_start_offset = int(row_start_bit_x % n)
    array_index_end = int((row_start_bit_x + width) // n)
    bit_end_offset = int((row_start_bit_x + width) % n)
    return array_index_start, bit_start_offset, array_index_end, bit_end_offset

def deal_with_node(rec_pos):
    """处理矩形边界，确保其在画布内"""
    rec_pos = [int(round(x)) for x in rec_pos]
    if rec_pos[0] < 0: rec_pos[0] = 0
    if rec_pos[1] < 0: rec_pos[1] = 0
    if rec_pos[0] + rec_pos[2] > screen_width: rec_pos[2] = screen_width - rec_pos[0]
    if rec_pos[1] + rec_pos[3] > screen_height: rec_pos[3] = screen_height - rec_pos[1]
    return rec_pos

def is_over_screen(obj):
    """检查标签是否完全在屏幕内"""
    minx, miny, width, height = obj
    maxx, maxy = minx + width, miny + height
    return not (minx >= 0 and maxx <= screen_width and miny >= 0 and maxy <= screen_height)

def lookup(bitmap, label_pos):
    """在位图中检查一个区域是否被占用"""
    minx, miny, width, height = label_pos
    full_mask = (1 << n) - 1
    for row in range(miny, miny + height):
        array_index_start, bit_start_offset, array_index_end, bit_end_offset = get_array_index(minx, row, width)
        if array_index_end > array_index_start:
            if bitmap[array_index_start] & (full_mask >> bit_start_offset): return False
            for arr_index in range(array_index_start + 1, array_index_end):
                if bitmap[arr_index]: return False
            if bitmap[array_index_end] & (full_mask ^ (full_mask >> bit_end_offset)): return False
        else:
            if bitmap[array_index_start] & ((full_mask >> bit_start_offset) & (full_mask ^ (full_mask >> bit_end_offset))): return False
    return True

def update(bitmap, label_pos):
    """在位图中标记一个区域为已占用"""
    minx, miny, width, height = label_pos
    full_mask = (1 << n) - 1
    for row in range(miny, miny + height):
        array_index_start, bit_start_offset, array_index_end, bit_end_offset = get_array_index(minx, row, width)
        if array_index_end > array_index_start:
            bitmap[array_index_start] |= (full_mask >> bit_start_offset)
            for arr_index in range(array_index_start + 1, array_index_end):
                bitmap[arr_index] = full_mask
            bitmap[array_index_end] |= (full_mask ^ (full_mask >> bit_end_offset))
        else:
            bitmap[array_index_start] |= ((full_mask >> bit_start_offset) & (full_mask ^ (full_mask >> bit_end_offset)))
    return bitmap

def init_point_bitmap(bitmap, r):
    """在位图中初始化所有锚点区域为占用"""
    for i in range(len(anchor_x_arr)):
        rec_pos = [anchor_x_arr[i] - r, anchor_y_arr[i] - r, 2 * r, 2 * r]
        rec_pos = deal_with_node(rec_pos)
        update(bitmap, rec_pos)
    return bitmap

def get_quadrant_center(point_index, quadrant, r):
    """获取标签在指定象限的预估中心点"""
    angle_map = {1: math.pi/4, 2: 3*math.pi/4, 3: 5*math.pi/4, 4: 7*math.pi/4}
    angle = angle_map.get(quadrant, 0)
    pos = get_angle_position(point_index, angle, r)
    return (pos[0] + pos[2]/2, pos[1] + pos[3]/2)

def quadrant_to_angle_range(quadrant):
    """将象限转换为角度范围 (弧度)"""
    range_map = {
        1: (0, math.pi/2), 2: (math.pi/2, math.pi),
        3: (math.pi, 3*math.pi/2), 4: (3*math.pi/2, 2*math.pi)
    }
    return range_map.get(quadrant)

def distance_sq(p1, p2):
    """计算两点距离的平方"""
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def calculate_shortest_angle_diff(angle1, angle2):
    """计算两个角度之间的最短弧度差 (0 to pi)"""
    diff = abs(angle1 - angle2) % (2 * math.pi)
    return min(diff, 2 * math.pi - diff)


def bitmaplabeling(prev_solution={}, prev_anchors={}):
    """核心标签放置算法，引入成本函数以保证运动连续性"""
    solution = {}
    bitmap = np.zeros((screen_height * screen_width + n - 1) // n, dtype=np.uint32)
    bitmap = init_point_bitmap(bitmap, radius)
    
    angle_step = math.pi / 12
    point_angle_ranges = {}

    # 1. 速度计算与启发式约束生成
    velocities = {}
    for i in range(len(anchor_x_arr)):
        if i in prev_anchors:
            velocities[i] = (anchor_x_arr[i] - prev_anchors[i][0], anchor_y_arr[i] - prev_anchors[i][1])
        else:
            velocities[i] = (0, 0)

    for i in range(len(anchor_x_arr)):
        for j in range(i + 1, len(anchor_x_arr)):
            static_dist_sq = (anchor_x_arr[j] - anchor_x_arr[i])**2 + (anchor_y_arr[j] - anchor_y_arr[i])**2
            if static_dist_sq > HEURISTIC_DISTANCE_THRESHOLD**2:
                continue
            
            rel_vx, rel_vy = velocities[j][0] - velocities[i][0], velocities[j][1] - velocities[i][1]
            if abs(rel_vx) < 1 and abs(rel_vy) < 1: continue

            q_config1, q_config2 = None, None
            if rel_vx >= 0 and rel_vy < 0: q_config1, q_config2 = ({'i': 1, 'j': 3}, {'i': 3, 'j': 1})
            elif rel_vx < 0 and rel_vy < 0: q_config1, q_config2 = ({'i': 2, 'j': 4}, {'i': 4, 'j': 2})
            elif rel_vx < 0 and rel_vy >= 0: q_config1, q_config2 = ({'i': 1, 'j': 3}, {'i': 3, 'j': 1})
            elif rel_vx >= 0 and rel_vy >= 0: q_config1, q_config2 = ({'i': 2, 'j': 4}, {'i': 4, 'j': 2})
            
            if not q_config1: continue

            pos_i_c1, pos_j_c1 = get_quadrant_center(i, q_config1['i'], radius), get_quadrant_center(j, q_config1['j'], radius)
            dist_sq1 = distance_sq(pos_i_c1, pos_j_c1)
            pos_i_c2, pos_j_c2 = get_quadrant_center(i, q_config2['i'], radius), get_quadrant_center(j, q_config2['j'], radius)
            dist_sq2 = distance_sq(pos_i_c2, pos_j_c2)
            best_config = q_config1 if dist_sq1 > dist_sq2 else q_config2
            
            if i not in point_angle_ranges: point_angle_ranges[i] = []
            if j not in point_angle_ranges: point_angle_ranges[j] = []
            point_angle_ranges[i].append(quadrant_to_angle_range(best_config['i']))
            point_angle_ranges[j].append(quadrant_to_angle_range(best_config['j']))

    # 2. 放置阶段：应用成本函数
    for i in range(len(anchor_x_arr)):
        prev_angle = prev_solution.get(i)
        if prev_angle is not None:
            label_pos = get_angle_position(i, prev_angle, radius)
            if not is_over_screen(label_pos) and lookup(bitmap, label_pos):
                update(bitmap, label_pos)
                solution[i] = prev_angle
                continue

        # 阶段一：完整搜索，找到所有可用的位置
        valid_positions = []
        angle_ranges = point_angle_ranges.get(i, [(0, 2 * math.pi)])
        # 修复后的代码
        for start_angle, end_angle in angle_ranges:
            # 确定搜索的总角度范围
            total_angle_range = (end_angle - start_angle + 2 * math.pi) % (2 * math.pi)
            if total_angle_range == 0 and start_angle != end_angle: # 处理完整的360度搜索
                total_angle_range = 2 * math.pi

            # 计算需要搜索的总步数
            # 使用 1e-9 是为了处理浮点数精度问题，确保能覆盖到端点
            num_steps = int(total_angle_range / angle_step + 1e-9) + 1 

            for step in range(num_steps):
                current_angle = (start_angle + step * angle_step) % (2 * math.pi)
                
                label_pos = get_angle_position(i, current_angle, radius)
                if not is_over_screen(label_pos) and lookup(bitmap, label_pos):
                    valid_positions.append((label_pos, current_angle))
        
        # 阶段二：成本评估，选择成本最低的位置
        if valid_positions:
            min_cost = float('inf')
            best_choice = None
            reference_angle = prev_solution.get(i, valid_positions[0][1])

            for pos, angle in valid_positions:
                cost = calculate_shortest_angle_diff(angle, reference_angle)
                
                if cost < min_cost:
                    min_cost = cost
                    best_choice = (pos, angle)
            
            if best_choice:
                best_position, best_angle = best_choice
                update(bitmap, best_position)
                solution[i] = best_angle
            
    return solution, bitmap

# --- 公开的接口函数 ---

def define_path(pf_count, name, size=16):
    global file_path
    file_path = "sample_generated.json"

def read_dataset():
    with open(file_path, 'r') as file:
        return json.load(file)

def set_label_data(frame_points):
    global label_data
    label_data = frame_points

def parse_data():
    global anchor_x_arr, anchor_y_arr, label_size_arr, min_height
    anchor_x_arr, anchor_y_arr, label_size_arr = [], [], []
    min_height = float('inf')
    if not label_data: return
    point_indices = sorted([int(k) for k in label_data.keys()])
    for j in point_indices:
        p = label_data[str(j)]
        anchor_x_arr.append(int(round(p["anchor"][0])))
        anchor_y_arr.append(int(round(p["anchor"][1])))
        height = int(round(p["size"][1] + p["size"][2]))
        label_size_arr.append((int(round(p["size"][0])), height))
        min_height = min(height, min_height)

def do_prep():
    global r
    r = radius
    parse_data()

def do_alg(prev_solution={}, prev_anchors={}):
    """主算法入口，现在接收上一帧的锚点位置"""
    solution, _ = bitmaplabeling(prev_solution, prev_anchors)
    return solution

def solution_to_position(point_index, angle):
    """辅助函数，将解（角度）转换回矩形坐标"""
    return get_angle_position(point_index, angle, radius)