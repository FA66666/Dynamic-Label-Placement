#!/usr/bin/env python3
"""
将100帧动画数据扩展为2000帧，保持运动方向不变，速度减小为原来的1/20
"""

import json
import copy
from typing import Dict, List, Tuple, Any

def load_animation_data(filename: str) -> Dict[str, Any]:
    """加载动画数据"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_velocity(start_pos: List[float], end_pos: List[float], frames: int) -> List[float]:
    """计算速度向量（每帧的位移）"""
    if frames == 0:
        return [0.0, 0.0]
    
    dx = (end_pos[0] - start_pos[0]) / frames
    dy = (end_pos[1] - start_pos[1]) / frames
    return [dx, dy]

def extend_animation(data: Dict[str, Any], speed_factor: float = 1/20, target_frames: int = 2000) -> Dict[str, Any]:
    """
    扩展动画数据
    
    Args:
        data: 原始动画数据
        speed_factor: 速度因子，1/20表示速度减小为原来的1/20
        target_frames: 目标帧数
    
    Returns:
        扩展后的动画数据
    """
    frames = data["frames"]
    
    # 找到所有的点ID
    point_ids = list(frames[0]["points"].keys())
    
    # 计算每个点的初始位置和最终位置
    initial_positions = {}
    final_positions = {}
    
    for point_id in point_ids:
        initial_pos = frames[0]["points"][point_id]["anchor"]
        final_pos = frames[-1]["points"][point_id]["anchor"]
        initial_positions[point_id] = initial_pos[:]
        final_positions[point_id] = final_pos[:]
    
    # 计算原始动画的总帧数
    original_frames = len(frames)
    
    # 为每个点计算速度向量
    velocities = {}
    for point_id in point_ids:
        velocity = calculate_velocity(
            initial_positions[point_id],
            final_positions[point_id], 
            original_frames - 1  # 实际的间隔帧数
        )
        # 应用速度因子
        velocities[point_id] = [velocity[0] * speed_factor, velocity[1] * speed_factor]
    
    # 创建新的动画数据
    new_data = {
        "frames": []
    }
    
    # 生成新的帧
    for frame_num in range(target_frames + 1):  # 包含第0帧到第2000帧
        new_frame = {
            "frame": frame_num,
            "points": {}
        }
        
        for point_id in point_ids:
            # 计算新位置
            new_x = initial_positions[point_id][0] + velocities[point_id][0] * frame_num
            new_y = initial_positions[point_id][1] + velocities[point_id][1] * frame_num
            
            # 复制点的其他属性（从第一帧）
            original_point = frames[0]["points"][point_id]
            new_point = copy.deepcopy(original_point)
            new_point["anchor"] = [new_x, new_y]
            
            new_frame["points"][point_id] = new_point
        
        new_data["frames"].append(new_frame)
    
    return new_data

def save_animation_data(data: Dict[str, Any], filename: str):
    """保存动画数据"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def main():
    """主函数"""
    # 输入和输出文件名
    input_file = "sample_generated.json"
    output_file = "sample_extended_2000_frames.json"
    
    print("正在读取原始动画数据...")
    original_data = load_animation_data(input_file)
    
    print(f"原始数据包含 {len(original_data['frames'])} 帧")
    
    # 分析原始数据中的运动
    frames = original_data["frames"]
    point_ids = list(frames[0]["points"].keys())
    
    print("分析点的运动：")
    for point_id in point_ids:
        start_pos = frames[0]["points"][point_id]["anchor"]
        end_pos = frames[-1]["points"][point_id]["anchor"]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = (dx**2 + dy**2)**0.5
        
        print(f"  {point_id}: 从 ({start_pos[0]}, {start_pos[1]}) 到 ({end_pos[0]}, {end_pos[1]}), 总距离: {distance:.2f}")
    
    print("\n正在扩展动画数据...")
    # 扩展动画（速度减小为原来的1/20，扩展为2000帧）
    extended_data = extend_animation(original_data, speed_factor=1/20, target_frames=2000)
    
    print(f"扩展后数据包含 {len(extended_data['frames'])} 帧")
    
    # 验证扩展后的运动
    new_frames = extended_data["frames"]
    print("\n验证扩展后的运动：")
    for point_id in point_ids:
        start_pos = new_frames[0]["points"][point_id]["anchor"]
        end_pos = new_frames[-1]["points"][point_id]["anchor"]
        
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        distance = (dx**2 + dy**2)**0.5
        
        print(f"  {point_id}: 从 ({start_pos[0]:.2f}, {start_pos[1]:.2f}) 到 ({end_pos[0]:.2f}, {end_pos[1]:.2f}), 总距离: {distance:.2f}")
    
    print(f"\n正在保存扩展后的数据到 {output_file}...")
    save_animation_data(extended_data, output_file)
    
    print("完成！")
    print(f"原始动画: {len(original_data['frames'])} 帧")
    print(f"扩展动画: {len(extended_data['frames'])} 帧")
    print(f"速度比例: 1/20")

if __name__ == "__main__":
    main()
