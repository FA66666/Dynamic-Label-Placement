import json
import numpy as np

def load_trajectories_from_json(json_file_path):
    """
    从JSON文件加载轨迹数据
    
    参数:
        json_file_path: JSON文件路径
        
    返回:
        包含所有特征点轨迹的列表，格式与generate_trajectories()一致
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 获取所有帧
        frames = data.get('frames', [])
        if not frames:
            raise ValueError("JSON文件中没有找到frames数据")
        
        # 获取点的ID列表（从第一帧获取）
        first_frame = frames[0]
        point_ids = list(first_frame.get('points', {}).keys())
        
        if not point_ids:
            raise ValueError("JSON文件中没有找到点数据")
        
        # 初始化轨迹列表
        trajectories = [[] for _ in range(len(point_ids))]
        
        # 处理每一帧
        for frame_data in frames:
            frame_points = frame_data.get('points', {})
            
            # 为每个点添加当前帧的位置
            for i, point_id in enumerate(point_ids):
                if point_id in frame_points:
                    anchor = frame_points[point_id].get('anchor', [0, 0])
                    # 确保坐标在有效范围内
                    x = max(50, min(950, float(anchor[0])))
                    y = max(50, min(950, float(anchor[1])))
                    trajectories[i].append((x, y))
        
        print(f"成功从 {json_file_path} 加载轨迹数据")
        print(f"点的数量: {len(point_ids)}")
        print(f"帧数: {len(frames)}")
        
        return trajectories
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
        return []
    except json.JSONDecodeError:
        print(f"错误：JSON文件格式不正确 {json_file_path}")
        return []
    except Exception as e:
        print(f"错误：加载JSON数据时发生错误: {e}")
        return []

def get_label_info_from_json(json_file_path):
    """
    从JSON文件获取标签信息（尺寸、文本等）
    
    参数:
        json_file_path: JSON文件路径
        
    返回:
        包含标签信息的字典列表
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        if not frames:
            return []
        
        # 从第一帧获取标签信息
        first_frame = frames[0]
        points = first_frame.get('points', {})
        
        label_info = []
        for point_id, point_data in points.items():
            size = point_data.get('size', [40.0, 15.0, 0.0])
            text = point_data.get('text', f'point_{point_id}')
            
            label_info.append({
                'id': int(point_id),
                'text': text,
                'length': float(size[0]),  # 长度
                'width': float(size[1]),   # 宽度
                'height': float(size[2])   # 高度（通常为0）
            })
        
        return label_info
        
    except Exception as e:
        print(f"错误：获取标签信息时发生错误: {e}")
        return []

