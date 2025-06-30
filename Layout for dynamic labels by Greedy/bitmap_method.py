import json
import math
import numpy as np

# --- 全局变量定义 ---
n = 32  # 位图压缩的位数，通常为32或64
label_data = {}
anchor_x_arr, anchor_y_arr, label_size_arr = [], [], []
screen_width, screen_height = 1000, 1000
radius = 10
min_height = 10
file_path = ""

# --- 核心功能函数 ---

def get_angle_position(point_index, angle, r):
    """计算给定角度的标签位置（返回整数坐标）"""
    anchor_x, anchor_y = anchor_x_arr[point_index], anchor_y_arr[point_index]
    width, height = label_size_arr[point_index]
    
    # 标签中心到锚点的距离
    distance = r + max(width, height) / 2
    
    # 标签中心点的位置
    center_x = anchor_x + distance * math.cos(angle)
    center_y = anchor_y + distance * math.sin(angle)
    
    # 标签左上角的位置并转换为整数
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
    if rec_pos[0] < 0:
        rec_pos[0] = 0
    if rec_pos[1] < 0:
        rec_pos[1] = 0
    if rec_pos[0] + rec_pos[2] > screen_width:
        rec_pos[2] = screen_width - rec_pos[0]
    if rec_pos[1] + rec_pos[3] > screen_height:
        rec_pos[3] = screen_height - rec_pos[1]
    return rec_pos

def is_over_screen(obj):
    """检查标签是否完全或部分在屏幕外"""
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
            start_mask = full_mask >> bit_start_offset
            if bitmap[array_index_start] & start_mask:
                return False
            for arr_index in range(array_index_start + 1, array_index_end):
                if bitmap[arr_index] & full_mask:
                    return False
            end_mask = full_mask ^ (full_mask >> bit_end_offset)
            if bitmap[array_index_end] & end_mask:
                return False
        else:
            start_mask = full_mask >> bit_start_offset
            end_mask = full_mask ^ (full_mask >> bit_end_offset)
            if bitmap[array_index_start] & (start_mask & end_mask):
                return False
    return True

def update(bitmap, label_pos):
    """在位图中标记一个区域为已占用"""
    minx, miny, width, height = label_pos
    full_mask = (1 << n) - 1
    
    for row in range(miny, miny + height):
        array_index_start, bit_start_offset, array_index_end, bit_end_offset = get_array_index(minx, row, width)
        
        if array_index_end > array_index_start:
            start_mask = full_mask >> bit_start_offset
            bitmap[array_index_start] |= start_mask
            for arr_index in range(array_index_start + 1, array_index_end):
                bitmap[arr_index] = full_mask
            end_mask = full_mask ^ (full_mask >> bit_end_offset)
            bitmap[array_index_end] |= end_mask
        else:
            start_mask = full_mask >> bit_start_offset
            end_mask = full_mask ^ (full_mask >> bit_end_offset)
            bitmap[array_index_start] |= (start_mask & end_mask)
    return bitmap

def init_point_bitmap(bitmap, r):
    """在位图中初始化所有锚点区域为占用"""
    for i in range(len(anchor_x_arr)):
        rec_pos = [anchor_x_arr[i] - r, anchor_y_arr[i] - r, 2 * r, 2 * r]
        rec_pos = deal_with_node(rec_pos)
        update(bitmap, rec_pos)
    return bitmap

# 在 bitmap_method.py 文件中找到并替换这个函数

def bitmaplabeling(prev_solution={}):
    """核心标签放置算法，已加入时间一致性处理"""
    solution = {}
    bitmap = np.zeros((screen_height * screen_width + n - 1) // n, dtype=np.uint32)
    bitmap = init_point_bitmap(bitmap, radius)
    
    angle_step = math.pi / 12  # 角度搜索精度 (15度)
    point_angle_ranges = {}
    
    # 预处理：计算点对之间的相对位置，以启发式地限制搜索角度
    for i in range(len(anchor_x_arr)):
        for j in range(i + 1, len(anchor_x_arr)):
            dx = anchor_x_arr[j] - anchor_x_arr[i]
            dy = anchor_y_arr[j] - anchor_y_arr[i]
            if dx == 0 and dy == 0: continue
            
            angle = math.atan2(dy, dx)
            if i not in point_angle_ranges: point_angle_ranges[i] = []
            if j not in point_angle_ranges: point_angle_ranges[j] = []
            point_angle_ranges[i].append(((angle + math.pi/2) % (2*math.pi), (angle + 3*math.pi/2) % (2*math.pi)))
            point_angle_ranges[j].append(((angle - math.pi/2) % (2*math.pi), (angle + math.pi/2) % (2*math.pi)))

    # 按顺序处理所有点
    for i in range(len(anchor_x_arr)):
        best_position = None
        best_angle = None
        
        # 1. 优先检查上一帧的位置以保持稳定
        prev_angle = prev_solution.get(i)
        if prev_angle is not None:
            # *** 修正点 1 ***
            label_pos = get_angle_position(i, prev_angle, radius)
            if not is_over_screen(label_pos) and lookup(bitmap, label_pos):
                best_position = label_pos
                best_angle = prev_angle
        
        # 2. 如果上一帧位置不可用，再进行完整搜索
        if best_position is None:
            angle_ranges = point_angle_ranges.get(i, [(0, 2 * math.pi)])
            for start_angle, end_angle in angle_ranges:
                if best_position: break
                
                current_angle = start_angle
                # 循环处理角度，包括跨越0度的情况
                while True:
                    if (start_angle < end_angle and current_angle >= end_angle) or \
                       (start_angle > end_angle and (current_angle >= end_angle and current_angle < start_angle)):
                        break
                    
                    # *** 修正点 2 ***
                    label_pos = get_angle_position(i, current_angle, radius)
                    if not is_over_screen(label_pos) and lookup(bitmap, label_pos):
                        best_position = label_pos
                        best_angle = current_angle
                        break
                    current_angle = (current_angle + angle_step) % (2 * math.pi)

        # 如果找到可行位置，更新解和位图
        if best_position is not None:
            update(bitmap, best_position)
            solution[i] = best_angle
            
    return solution, bitmap

# --- 公开的接口函数 ---

def define_path(pf_count, name, size=16):
    global file_path
    # 根据你的文件结构修改此路径
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

def do_alg(prev_solution={}):
    """主算法入口，返回最终解"""
    # 直接返回解，位图是中间过程，外部不需要
    solution, _ = bitmaplabeling(prev_solution)
    return solution

def solution_to_position(point_index, angle):
    """辅助函数，将解（角度）转换回矩形坐标"""
    return get_angle_position(point_index, angle, radius)