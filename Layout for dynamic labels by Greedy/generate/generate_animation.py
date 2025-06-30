import json
import math
import random
import numpy as np

def sine_motion(t, start, amplitude=50, frequency=2):
    x = start[0] + amplitude * math.sin(frequency * t * math.pi)
    y = start[1] + amplitude * math.cos(frequency * t * math.pi)
    return [x, y]

def circle_motion(t, center, radius=50):
    x = center[0] + radius * math.cos(2 * math.pi * t)
    y = center[1] + radius * math.sin(2 * math.pi * t)
    return [x, y]

def parabolic_motion(t, start, height=100):
    x = start[0] + 200 * t
    y = start[1] + height * (4 * t * (1 - t))
    return [x, y]

def exponential_motion(t, start):
    x = start[0] + 100 * (math.exp(t) - 1) / (math.e - 1)
    y = start[1] + 100 * (math.exp(1-t) - 1) / (math.e - 1)
    return [x, y]

def spiral_motion(t, center, radius=50):
    r = radius * t
    x = center[0] + r * math.cos(6 * math.pi * t)
    y = center[1] + r * math.sin(6 * math.pi * t)
    return [x, y]

def lemniscate_motion(t, center, size=50):
    theta = 2 * math.pi * t
    x = center[0] + size * math.cos(theta) / (1 + math.sin(theta)**2)
    y = center[1] + size * math.sin(theta) * math.cos(theta) / (1 + math.sin(theta)**2)
    return [x, y]

def hyperbolic_motion(t, center, a=50):
    x = center[0] + a * (2*t - 1)
    y = center[1] + a / (2*t - 1) if t != 0.5 else center[1] + a
    return [x, y]

def main():
    # 读取原始文件获取模板
    with open('sample10_1.json', 'r') as f:
        data = json.load(f)

    # 使用更小的size值，并将透明度设置为0（完全透明）
    size_template = [30.0, 8.0, 0.0]

    # 设置20个初始点的位置
    num_points = 20
    screen_width = 600
    screen_height = 400
    margin = 50

    initial_points = {}
    for i in range(num_points):
        x = random.randint(margin, screen_width - margin)
        y = random.randint(margin, screen_height - margin)
        initial_points[str(i)] = [x, y]

    # 为每个点随机分配运动类型
    motion_types = []
    for i in range(num_points):
        motion_type = random.choice([
            ('sine', sine_motion, {'amplitude': random.randint(30, 70), 'frequency': random.uniform(1, 3)}),
            ('circle', circle_motion, {'radius': random.randint(30, 70)}),
            ('parabolic', parabolic_motion, {'height': random.randint(50, 150)}),
            ('exponential', exponential_motion, {}),
            ('spiral', spiral_motion, {'radius': random.randint(30, 70)}),
            ('lemniscate', lemniscate_motion, {'size': random.randint(30, 70)}),
            ('hyperbolic', hyperbolic_motion, {'a': random.randint(30, 70)})
        ])
        motion_types.append(motion_type)

    # 生成帧数据
    num_frames = 100
    frames = []

    # 创建第一帧
    first_frame = {
        "frame": 0,
        "points": {
            str(i): {
                "anchor": initial_points[str(i)],
                "text": f"point_{i}",
                "size": size_template
            }
            for i in range(num_points)
        }
    }
    frames.append(first_frame)

    # 生成后续帧
    for frame in range(1, num_frames):
        t = frame / (num_frames - 1)
        new_points = {}
        
        for i in range(num_points):
            start_point = initial_points[str(i)]
            motion_name, motion_func, motion_params = motion_types[i]
            
            if motion_name == 'circle':
                new_pos = motion_func(t, start_point, **motion_params)
            elif motion_name == 'sine':
                new_pos = motion_func(t, start_point, **motion_params)
            elif motion_name == 'parabolic':
                new_pos = motion_func(t, start_point, **motion_params)
            elif motion_name == 'exponential':
                new_pos = motion_func(t, start_point)
            elif motion_name == 'spiral':
                new_pos = motion_func(t, start_point, **motion_params)
            elif motion_name == 'lemniscate':
                new_pos = motion_func(t, start_point, **motion_params)
            else:  # hyperbolic
                new_pos = motion_func(t, start_point, **motion_params)
            
            # 确保点不会超出屏幕边界
            new_pos[0] = max(margin, min(screen_width - margin, new_pos[0]))
            new_pos[1] = max(margin, min(screen_height - margin, new_pos[1]))
            
            new_points[str(i)] = {
                "anchor": [round(new_pos[0], 2), round(new_pos[1], 2)],
                "text": f"point_{i}",
                "size": size_template
            }
        
        frames.append({
            "frame": frame,
            "points": new_points
        })

    # 保存修改后的文件
    data["frames"] = frames
    with open('sample10_1.json', 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()