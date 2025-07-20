import json
from PIL import Image, ImageDraw, ImageFont
import os
import config # 导入配置文件

def generate_visualization_gif():
    """
    根据算法输出的 JSON 文件，生成一个 GIF 动画来可视化标签的动态放置过程。
    """
    # --- 1. 从 config 文件加载文件路径 ---
    output_json_path = config.OUTPUT_FILE
    sample_json_path = config.INPUT_FILE

    # --- 2. 加载数据 ---
    print("正在加载位置数据...")
    with open(output_json_path, 'r') as f:
        output_data = json.load(f)

    with open(sample_json_path, 'r') as f:
        sample_data = json.load(f)

    static_info = {}
    initial_points = sample_data['frames'][0]['points']
    for point_id, data in initial_points.items():
        static_info[point_id] = {'text': data['text']}


    # --- 3. 逐帧绘制图像 ---
    image_frames = []
    sorted_frame_keys = sorted(output_data.keys(), key=int)

    print(f"开始绘制 {len(sorted_frame_keys)} 帧图像...")
    for i, frame_idx_str in enumerate(sorted_frame_keys):
        print(f"  - 正在处理第 {i+1}/{len(sorted_frame_keys)} 帧...", end='\r')
        
        # 创建新图像 (从 config 加载尺寸和背景色)
        img = Image.new('RGB', (config.SCREEN_WIDTH, config.SCREEN_HEIGHT), config.BG_COLOR)
        draw = ImageDraw.Draw(img)

        frame_positions = output_data[frame_idx_str]

        for label_id, pos_info in frame_positions.items():
            anchor = pos_info['anchor']
            bbox = pos_info['bbox']
            text = static_info.get(label_id, {}).get('text', '')
            
            box_rect = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            anchor_pos = (anchor[0], anchor[1])
            
            # 绘制连接线 (从 config 加载颜色)
            draw.line(
                (anchor_pos[0], anchor_pos[1], bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2),
                fill=config.LINE_COLOR,
                width=1
            )
            
            # 绘制标签矩形框 (从 config 加载颜色)
            draw.rectangle(
                box_rect,
                outline=config.BOX_OUTLINE_COLOR,
                fill=config.BOX_FILL_COLOR
            )
            
            # 绘制标签文本 (从 config 加载颜色)
            draw.text(
                (bbox[0] + 4, bbox[1] + 2),
                text,
                fill=config.TEXT_COLOR
            )
            
            # 绘制锚点圆圈 (从 config 加载半径和颜色)
            r = config.ANCHOR_RADIUS
            draw.ellipse(
                (anchor_pos[0] - r, anchor_pos[1] - r, anchor_pos[0] + r, anchor_pos[1] + r),
                fill=config.ANCHOR_COLOR
            )
        
        image_frames.append(img) 
    print("\n所有帧绘制完成。")

    # --- 4. 保存为 GIF 文件 (从 config 加载文件名和时长) ---
    print(f"正在生成 GIF 文件: {config.GIF_OUTPUT_FILENAME}...")
    image_frames[0].save(
        config.GIF_OUTPUT_FILENAME,
        save_all=True,
        append_images=image_frames[1:],
        duration=config.GIF_DURATION_MS,
        loop=0
    )
    print("🎉 GIF 生成成功！")


if __name__ == "__main__":
    generate_visualization_gif()