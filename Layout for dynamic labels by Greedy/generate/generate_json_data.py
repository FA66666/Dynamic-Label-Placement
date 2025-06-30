import numpy as np
import json

def generate_json_data():
    """
    生成类似sample10_1.json格式的数据
    """
    # 定义7个点的初始位置和结束位置
    points = [
        # 移动的点
        {'start': (100, 700), 'end': (900, 700)},  # 点0：从左到右
        {'start': (500, 100), 'end': (500, 900)},  # 点1：从下到上
        {'start': (100, 100), 'end': (900, 900)},  # 点2：对角线移动
        
        # 静止的点
        {'start': (200, 800), 'end': (200, 800)},  # 点3：左上角
        {'start': (800, 800), 'end': (800, 800)},  # 点4：右上角
        {'start': (200, 200), 'end': (200, 200)},  # 点5：左下角
        {'start': (800, 200), 'end': (800, 200)},  # 点6：右下角
    ]

    frames = []
    num_frames = 100  # 总帧数

    for frame in range(num_frames + 1):
        progress = frame / num_frames
        frame_data = {
            "frame": frame,
            "points": {}
        }

        for i, point in enumerate(points):
            # 计算当前点的位置
            x = point['start'][0] + (point['end'][0] - point['start'][0]) * progress
            y = point['start'][1] + (point['end'][1] - point['start'][1]) * progress
            
            # 确保坐标在有效范围内
            x = max(50, min(950, x))
            y = max(50, min(950, y))

            # 添加点的数据
            frame_data["points"][str(i)] = {
                "anchor": [int(x), int(y)],
                "text": f"point_{i}",
                "size": [40.0, 15.0, 0.0]
            }

        frames.append(frame_data)

    # 创建最终的JSON数据结构
    json_data = {
        "frames": frames
    }

    # 将数据保存到文件
    with open('sample_generated.json', 'w') as f:
        json.dump(json_data, f, indent=2)

if __name__ == '__main__':
    generate_json_data()
    print("数据生成完成，已保存到 sample_generated.json") 