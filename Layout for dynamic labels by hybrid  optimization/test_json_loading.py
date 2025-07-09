"""
测试修改后的main.py是否能正确使用JSON数据
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_json_data import load_trajectories_from_json, get_label_info_from_json
from initialize import initialize_features_from_data
from label import Label

def test_json_loading():
    """测试JSON数据加载功能"""
    print("=== 测试JSON数据加载功能 ===")
    
    # 测试sample_generated.json
    json_file_path = 'sample_generated.json'
    print(f"正在测试 {json_file_path}...")
    
    # 加载轨迹数据
    positions_list = load_trajectories_from_json(json_file_path)
    if not positions_list:
        print("错误：无法从JSON文件加载轨迹数据")
        return False
    
    # 获取标签信息
    label_info_list = get_label_info_from_json(json_file_path)
    
    print(f"✓ 成功加载 {len(positions_list)} 个特征点的轨迹数据")
    print(f"✓ 每个轨迹包含 {len(positions_list[0])} 个时间点")
    print(f"✓ 获取了 {len(label_info_list)} 个标签信息")
    
    # 测试特征点初始化
    try:
        features = initialize_features_from_data(
            positions_list,
            frame_interval=0.05
        )
        print(f"✓ 成功初始化 {len(features)} 个特征点")
    except Exception as e:
        print(f"✗ 特征点初始化失败: {e}")
        return False
    
    # 测试标签创建
    try:
        labels = []
        for i, feature in enumerate(features):
            if i < len(label_info_list):
                label_info = label_info_list[i]
                label_length = label_info['length']
                label_width = label_info['width']
                label_text = label_info['text']
            else:
                label_length = 40
                label_width = 16
                label_text = f'point_{i}'
            
            initial_x = min(1000, feature.position[0] + 50)
            initial_y = max(0, min(1000, feature.position[1]))
            
            label = Label(
                id=feature.id,
                feature=feature,
                position=(initial_x, initial_y),
                length=label_length,
                width=label_width,
                velocity=(0, 0)
            )
            labels.append(label)
        
        print(f"✓ 成功创建 {len(labels)} 个标签对象")
        
        # 显示前几个标签的信息
        for i, label in enumerate(labels[:3]):
            print(f"  标签 {i}: ID={label.id}, 位置={label.position}, 尺寸=({label.length}, {label.width})")
        
    except Exception as e:
        print(f"✗ 标签创建失败: {e}")
        return False
    
    return True

