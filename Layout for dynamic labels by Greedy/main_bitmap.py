# 文件: main.py

import json
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import math

import bitmap_method

def plot_frame(frame_data, solution, screen_width, screen_height, filename):
    """可视化单帧的标签放置结果（无引导线）"""
    fig, ax = plt.subplots(figsize=(10, 10 * screen_height / screen_width))
    
    # 绘制锚点
    if 'points' in frame_data:
        for idx_str, point_info in frame_data['points'].items():
            x, y = point_info['anchor']
            ax.scatter(x, y, c='red', s=25, zorder=3)
    
    # 绘制成功放置的标签
    for idx, angle in solution.items():
        idx_str = str(idx)
        if idx_str not in frame_data['points']:
            continue

        x, y, width, height = bitmap_method.solution_to_position(idx, angle)
        text = frame_data['points'][idx_str]['text']
        
        ax.text(x + width / 2, y + height / 2, text,
                ha='center', va='center',
                fontsize=8, color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'),
                zorder=2)
    
    frame_num = frame_data.get("frame", "N/A")
    total_points = len(frame_data.get('points', {}))
    ax.set_title(f'Frame {frame_num} - Placed: {len(solution)}/{total_points}')
    ax.set_xlim(0, screen_width)
    ax.set_ylim(0, screen_height)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    fig.savefig(filename, dpi=120, bbox_inches='tight')
    plt.close(fig)

def main(screen_width=1000, screen_height=1000):
    """主执行函数，处理多帧数据并生成动画"""
    
    # 1. 初始化
    bitmap_method.define_path(0, "ignored") 
    bitmap_method.screen_width = screen_width
    bitmap_method.screen_height = screen_height
    
    try:
        all_frames_data = bitmap_method.read_dataset()
    except FileNotFoundError:
        print(f"错误: 未在 '{bitmap_method.file_path}' 找到数据文件，请检查路径。")
        return
        
    all_frames = all_frames_data.get('frames', [all_frames_data])

    # 2. 循环处理每一帧
    frame_files = []
    prev_solution = {}
    prev_anchors = {}

    for idx, frame in enumerate(all_frames):
        print(f"正在处理第 {idx} 帧...")
        
        current_points = frame.get('points', {})
        bitmap_method.set_label_data(current_points)
        bitmap_method.do_prep()
        
        current_solution = bitmap_method.do_alg(prev_solution, prev_anchors)
        
        frame_file = f"frame_{idx:03d}.png"
        plot_frame(frame, current_solution, screen_width, screen_height, frame_file)
        frame_files.append(frame_file)
        
        prev_solution = current_solution
        prev_anchors = {int(k): v['anchor'] for k, v in current_points.items()}
        
        print(f"第 {idx} 帧: {len(current_solution)} 个标签被放置。")

    # 3. 生成GIF动画
    output_gif = 'animation_final.gif'
    print(f"\n正在生成GIF动画: {output_gif}")
    if frame_files:
        with imageio.get_writer(output_gif, mode='I', duration=0.1, loop=0) as writer:
            for filename in frame_files:
                image = imageio.imread(filename)
                writer.append_data(image)
                os.remove(filename)
        print("动画生成成功！")
    else:
        print("没有帧被处理，无法生成GIF。")

if __name__ == "__main__":
    main(screen_width=1000, screen_height=1000)