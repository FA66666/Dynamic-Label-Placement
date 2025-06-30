import json
import matplotlib.pyplot as plt
import imageio.v2 as imageio # 使用 v2 版本的 imageio
import os
import math

# 导入我们已经修改过的 bitmap_method 模块
import bitmap_method

# 在 main.py 文件中找到并替换这个函数

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
        # 确保索引是字符串，以匹配 frame_data['points'] 的键
        idx_str = str(idx)
        if idx_str not in frame_data['points']:
            continue

        x, y, width, height = bitmap_method.solution_to_position(idx, angle)
        text = frame_data['points'][idx_str]['text']
        
        # 只绘制标签文本框
        ax.text(x + width / 2, y + height / 2, text,
                ha='center', va='center',
                fontsize=8, color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.3'),
                zorder=2)
        
        # --- 以下绘制引导线的代码已被移除 ---
        # anchor_x, anchor_y = frame_data['points'][idx_str]['anchor']
        # label_center_x = x + width / 2
        # label_center_y = y + height / 2
        # ax.plot([anchor_x, label_center_x], [anchor_y, label_center_y], 
        #         'k-', alpha=0.3, linewidth=0.7, zorder=1)
    
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
    
    # --- 1. 初始化 (只执行一次) ---
    bitmap_method.define_path(0, "ignored")
    bitmap_method.screen_width = screen_width
    bitmap_method.screen_height = screen_height
    
    try:
        all_frames_data = bitmap_method.read_dataset()
    except FileNotFoundError:
        print(f"Error: Data file not found at '{bitmap_method.file_path}'. Please check the path.")
        return
        
    all_frames = all_frames_data.get('frames', [all_frames_data])

    # --- 2. 循环处理每一帧 ---
    frame_files = []
    prev_solution = {}

    for idx, frame in enumerate(all_frames):
        print(f"Processing frame {idx}...")
        
        bitmap_method.set_label_data(frame.get('points', {}))
        bitmap_method.do_prep()
        
        # 执行算法，并传入上一帧的解以实现时间一致性
        current_solution = bitmap_method.do_alg(prev_solution)
        
        # 生成并保存当前帧的可视化图片
        frame_file = f"frame_{idx:03d}.png"
        plot_frame(frame, current_solution, screen_width, screen_height, frame_file)
        frame_files.append(frame_file)
        
        prev_solution = current_solution
        print(f"Frame {idx}: {len(current_solution)} labels placed.")

    # --- 3. 生成GIF动画 ---
    output_gif = 'animation_final.gif'
    print(f"\nCreating GIF animation: {output_gif}")
    if frame_files:
        with imageio.get_writer(output_gif, mode='I', duration=0.1, loop=0) as writer:
            for filename in frame_files:
                image = imageio.imread(filename)
                writer.append_data(image)
                os.remove(filename)
        print("Animation created successfully!")
    else:
        print("No frames were processed to create a GIF.")

if __name__ == "__main__":
    main(screen_width=1000, screen_height=1000)