import math
import numpy as np
import config # 导入配置文件

# ----------------------------------------------------------------------
# BitmapPlacer 类
# 职责：封装了所有与位图法相关的操作。主要负责在动画的第0帧，
#      为所有标签计算出一个没有重叠的初始布局方案。
# ----------------------------------------------------------------------
class BitmapPlacer:
    """封装了基于位图的方法，用于初始的、静态的标签布局。"""
    def __init__(self, screen_width, screen_height, radius, n_val=config.N_VAL):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.radius = radius
        self.n = n_val # 从 config 加载默认值
        self.point_id_map = []
        self.anchor_x_arr = []
        self.anchor_y_arr = []
        self.label_size_arr = []

    def _get_array_index(self, minx, row, width):
        row_start_bit_x = row * self.screen_width + minx
        n_val = self.n
        array_index_start = row_start_bit_x // n_val
        bit_start_offset = row_start_bit_x % n_val
        array_index_end = (row_start_bit_x + width) // n_val
        bit_end_offset = (row_start_bit_x + width) % n_val
        return array_index_start, bit_start_offset, array_index_end, bit_end_offset

    def _parse_data(self, points_data):
        self.point_id_map = sorted(points_data.keys(), key=int)
        self.anchor_x_arr = [int(points_data[pid]["anchor"][0]) for pid in self.point_id_map]
        self.anchor_y_arr = [int(points_data[pid]["anchor"][1]) for pid in self.point_id_map]
        self.label_size_arr = [(math.ceil(points_data[pid]["size"][0]), math.ceil(points_data[pid]["size"][1] + points_data[pid]["size"][2])) for pid in self.point_id_map]

    def _do_prep(self, points_data):
        self._parse_data(points_data)

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
        if x < 0: w, x = w + x, 0
        if y < 0: h, y = h + y, 0
        if x + w > self.screen_width: w = self.screen_width - x
        if y + h > self.screen_height: h = self.screen_height - y
        return [x, y, w, h]

    def _update_bitmap(self, bitmap, label_pos):
        minx, miny, width, height = [int(v) for v in label_pos]
        full_mask = (1 << self.n) - 1
        for row in range(miny, miny + height):
            if not (0 <= row < self.screen_height): continue
            start_idx, start_off, end_idx, end_off = self._get_array_index(minx, row, width)
            if end_idx > start_idx:
                bitmap[start_idx] |= (full_mask >> start_off)
                for i in range(start_idx + 1, end_idx): bitmap[i] = full_mask
                if end_off > 0: bitmap[end_idx] |= (full_mask ^ (full_mask >> end_off))
            else:
                if start_off < end_off:
                    mask = (full_mask >> start_off) ^ (full_mask >> end_off)
                    bitmap[start_idx] |= mask
        return bitmap

    def _lookup(self, bitmap, label_pos):
        minx, miny, width, height = [int(v) for v in label_pos]
        full_mask = (1 << self.n) - 1
        for row in range(miny, miny + height):
            if not (0 <= row < self.screen_height): continue
            start_idx, start_off, end_idx, end_off = self._get_array_index(minx, row, width)
            if end_idx > start_idx:
                if (bitmap[start_idx] & (full_mask >> start_off)) != 0: return False
                for i in range(start_idx + 1, end_idx):
                    if bitmap[i] != 0: return False
                if end_off > 0 and (bitmap[end_idx] & (full_mask ^ (full_mask >> end_off))) != 0: return False
            else:
                if start_off < end_off:
                    mask = (full_mask >> start_off) ^ (full_mask >> end_off)
                    if (bitmap[start_idx] & mask) != 0: return False
        return True
    
    def _is_over_screen(self, obj):
        minx, miny, width, height = obj
        return not (minx >= 0 and miny >=0 and minx + width <= self.screen_width and miny + height <= self.screen_height)

    def _bitmaplabeling(self):
        solution = {}
        bitmap = self._init_s()
        for i in range(len(self.anchor_x_arr)):
            r = self.radius
            rec_pos = self._deal_with_node([self.anchor_x_arr[i] - r, self.anchor_y_arr[i] - r, 2 * r, 2 * r])
            bitmap = self._update_bitmap(bitmap, rec_pos)
        
        for i in range(len(self.anchor_x_arr)):
            for m in range(1, 9):
                label_pos = self._position_model(i, m, self.anchor_x_arr[i], self.anchor_y_arr[i])
                if self._is_over_screen(label_pos): continue
                if self._lookup(bitmap, label_pos):
                    self._update_bitmap(bitmap, label_pos)
                    solution[self.point_id_map[i]] = m
                    break
        return solution

    def run_initial_placement(self, points_data):
        self._do_prep(points_data)
        solution = self._bitmaplabeling()
        for point_id in self.point_id_map:
             if point_id not in solution:
                 solution[point_id] = 1
        return solution
        
    def get_position_box(self, point_id_str, pos_model, anchor_tuple):
        if not self.point_id_map: raise Exception("Placer not initialized with data.")
        point_index = self.point_id_map.index(point_id_str)
        return self._position_model(point_index, pos_model, anchor_tuple[0], anchor_tuple[1])