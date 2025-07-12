import json
import numpy as np

def load_trajectories(file_path):
    """
    从JSON文件加载轨迹数据
    
    参数:
        json_file_path: JSON文件路径
        
    返回:
        包含所有特征点轨迹的列表，格式与generate_trajectories()一致
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 获取所有帧和点的 ID 列表
        frames = data.get('frames',[])
        first_frame = frames[0]        
        point_ids = list(first_frame.get('points',{}).keys())

        # 初始化轨迹列表
        trajectories = [[] for _ in range(len(point_ids))]

        for frame_data in frames:
            frame_points = frame_data.get('points', {})

            # 为每个点添加当前帧的位置
            for i, point_id in enumerate(point_ids):
                    anchor = frame_points[point_id].get('anchor',[0,0])
                    x,y = anchor[0],anchor[1]
                    trajectories[i].append((x, y))


        print(f"成功从 {file_path} 加载轨迹数据")
        print(f"点的数量: {len(point_ids)}")
        print(f"帧数: {len(frames)}")
        
        return trajectories
    
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return []
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return []      
    
def get_label_info(file_path):
    """
    从JSON文件获取标签信息（尺寸、文本等）
    
    参数:
        json_file_path: JSON文件路径
        
    返回:
        包含标签信息的字典列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
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


