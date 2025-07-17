import json
from PIL import Image, ImageDraw, ImageFont
import os

def generate_visualization_gif():
    """
    根据算法输出的 JSON 文件，生成一个 GIF 动画来可视化标签的动态放置过程。
    """
    # --- 1. 配置参数 ---
    config = {
        'width': 1000,
        'height': 1000,
        'bg_color': 'white',
        'box_outline_color': 'black',
        'box_fill_color': '#f0f0f0',      # 浅灰色填充
        'text_color': 'black',
        'anchor_color': 'red',
        'line_color': 'grey',
        'output_filename': 'output.gif',
        'duration_ms': 50,               # 每帧的持续时间（毫秒），50ms = 20 FPS
        'anchor_radius': 4               # [新增] 锚点圆圈的半径，可在此调节
    }

    output_json_path = 'output_positions.json'
    sample_json_path = 'sample_input.json'

    # --- 2. 检查文件是否存在 ---
    if not os.path.exists(output_json_path):
        print(f"错误: 找不到输入文件 '{output_json_path}'。请先运行上一个脚本生成该文件。")
        return
    if not os.path.exists(sample_json_path):
        print(f"错误: 找不到输入文件 '{sample_json_path}'。")
        return

    # --- 3. 加载数据 ---
    print("正在加载位置数据...")
    with open(output_json_path, 'r') as f:
        output_data = json.load(f)

    with open(sample_json_path, 'r') as f:
        sample_data = json.load(f)

    # 从 sample_input.json 获取每个标签的静态信息（如文本）
    static_info = {}
    initial_points = sample_data['frames'][0]['points']
    for point_id, data in initial_points.items():
        static_info[point_id] = {'text': data['text']}

    # --- 4. 准备字体 ---
    try:
        font = ImageFont.truetype("Arial.ttf", size=12)
    except IOError:
        print("警告: 未找到 Arial 字体，将使用默认字体。")
        font = ImageFont.load_default()


    # --- 5. 逐帧绘制图像 ---
    image_frames = []
    # 确保帧是按数字顺序排序的
    sorted_frame_keys = sorted(output_data.keys(), key=int)

    print(f"开始绘制 {len(sorted_frame_keys)} 帧图像...")
    for i, frame_idx_str in enumerate(sorted_frame_keys):
        print(f"  - 正在处理第 {i+1}/{len(sorted_frame_keys)} 帧...", end='\r')
        
        # 创建一个新的白色背景图像
        img = Image.new('RGB', (config['width'], config['height']), config['bg_color'])
        draw = ImageDraw.Draw(img)

        frame_positions = output_data[frame_idx_str]

        # 在当前帧上绘制每一个标签
        for label_id, pos_info in frame_positions.items():
            anchor = pos_info['anchor']
            bbox = pos_info['bbox']  # 格式: [x, y, width, height]
            text = static_info.get(label_id, {}).get('text', '')

            # 定义矩形框的四个角 (x1, y1, x2, y2)
            box_rect = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            
            # 定义锚点的位置
            anchor_pos = (anchor[0], anchor[1])
            
            # 绘制连接线
            draw.line(
                (anchor_pos[0], anchor_pos[1], bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2),
                fill=config['line_color'],
                width=1
            )
            
            # 绘制标签矩形框
            draw.rectangle(
                box_rect,
                outline=config['box_outline_color'],
                fill=config['box_fill_color']
            )
            
            # 绘制标签文本
            draw.text(
                (bbox[0] + 4, bbox[1] + 2), # 文本位置稍微内缩
                text,
                font=font,
                fill=config['text_color']
            )
            
            # ###############################################################
            # # [修改] 将绘制锚点的方式从固定方块改为可调节半径的圆形
            # ###############################################################
            r = config['anchor_radius']
            draw.ellipse(
                (anchor_pos[0] - r, anchor_pos[1] - r, anchor_pos[0] + r, anchor_pos[1] + r),
                fill=config['anchor_color']
            )
        
        image_frames.append(img)
    
    print("\n所有帧绘制完成。")

    # --- 6. 保存为 GIF 文件 ---
    if image_frames:
        print(f"正在生成 GIF 文件: {config['output_filename']}...")
        image_frames[0].save(
            config['output_filename'],
            save_all=True,
            append_images=image_frames[1:],
            duration=config['duration_ms'],
            loop=0  # 0 表示无限循环
        )
        print("🎉 GIF 生成成功！")
    else:
        print("错误：没有可用于生成 GIF 的帧。")

if __name__ == "__main__":
    generate_visualization_gif()