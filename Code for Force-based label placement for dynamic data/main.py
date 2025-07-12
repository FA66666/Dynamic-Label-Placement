from initialize import initialize_features
from label import Label
from feature import Feature
import feature
from load_data import load_trajectories, get_label_info
from config import params_Adj, param_NoAdj, global_params
from Force_based_optimizer import ForceBasedOptimizer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 使用黑体或微软雅黑
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号


def main():
    file_path = 'sample_generated.json' 
    positions_list = load_trajectories(file_path)
    label_info_list = get_label_info(file_path)
    features = initialize_features(
        positions_list,
        frame_interval=0.05
    )

    labels = []
    initial_label_positions = {}
    initial_velocities = {}
    
    for i, feature in enumerate(features):
        label_info = label_info_list[i]
        label_length = label_info['length']
        label_width = label_info['width']

        initial_x = feature.position[0] + 30
        initial_y = feature.position[1]

        label = Label(
            id=feature.id,
            feature=feature,
            position=(initial_x, initial_y),
            length=label_length,
            width=label_width,
            velocity=(0, 0)
        )
        labels.append(label)
        
        # 准备优化所需的数据结构
        initial_label_positions[i] = (initial_x, initial_y)
        initial_velocities[i] = (0.0, 0.0)

    # 创建优化器
    force_based_optimizer = ForceBasedOptimizer(features=features, labels=labels)
    
    # 运行优化并记录每一步的结果
    num_steps = 100  # 优化步数
    dt = 0.1        # 时间步长
    
    # 记录优化过程
    optimization_history = []
    current_positions = initial_label_positions.copy()
    current_velocities = initial_velocities.copy()
    current_frame = 0  # 当前帧数
    
    # 记录初始状态
    optimization_history.append({
        'positions': current_positions.copy(),
        'velocities': current_velocities.copy(),
        'features': [(f.position[0], f.position[1]) for f in features],
        'frame': current_frame
    })
    
    for step in range(num_steps):
        # 更新特征点位置到下一帧
        if current_frame + 1 < len(positions_list[0]):  # 确保不超出轨迹范围
            current_frame += 1
            for i, feature in enumerate(features):
                if i < len(positions_list) and current_frame < len(positions_list[i]):
                    new_position = positions_list[i][current_frame]
                    feature.position = new_position
                    # 更新速度（如果需要用于预测力计算）
                    if current_frame > 0:
                        prev_position = positions_list[i][current_frame - 1]
                        feature.velocity = (
                            (new_position[0] - prev_position[0]) / dt,
                            (new_position[1] - prev_position[1]) / dt
                        )
        
        # 优化标签位置
        current_positions, current_velocities = force_based_optimizer.update_label_positions(
            current_positions, current_velocities, dt
        )
        
        # 记录每一步的状态
        optimization_history.append({
            'positions': current_positions.copy(),
            'velocities': current_velocities.copy(),
            'features': [(f.position[0], f.position[1]) for f in features],
            'frame': current_frame
        })
        
    
    # 创建动画
    create_optimization_animation(optimization_history, labels, save_path='label_optimization.gif')
    
    return current_positions, current_velocities

def create_optimization_animation(optimization_history, labels, save_path='label_optimization.gif'):
    """
    创建标签优化过程的动画
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 设置画布范围
    all_x = []
    all_y = []
    for frame in optimization_history:
        for pos in frame['positions'].values():
            all_x.append(pos[0])
            all_y.append(pos[1])
        for pos in frame['features']:
            all_x.append(pos[0])
            all_y.append(pos[1])
    
    margin = 100
    xlim = (min(all_x) - margin, max(all_x) + margin)
    ylim = (min(all_y) - margin, max(all_y) + margin)
    
    def animate(frame_idx):
        ax.clear()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        frame_data = optimization_history[frame_idx]
        ax.set_title(f'(帧 {frame_data["frame"]})')
        
        # 绘制特征点
        feature_positions = frame_data['features']
        feature_x = [pos[0] for pos in feature_positions]
        feature_y = [pos[1] for pos in feature_positions]
        ax.scatter(feature_x, feature_y, c='red', s=100, marker='o', label='特征点', zorder=3)
        
        # 绘制标签和连接线
        for i, label in enumerate(labels):
            if i in frame_data['positions']:
                label_pos = frame_data['positions'][i]
                feature_pos = feature_positions[i]
                
                # 绘制连接线
                ax.plot([label_pos[0], feature_pos[0]], 
                       [label_pos[1], feature_pos[1]], 
                       'k--', alpha=0.5, linewidth=1, zorder=1)
                
                # 绘制标签矩形
                label_x, label_y = label_pos
                width = label.length
                height = label.width
                
                # 创建矩形（以中心点为基准）
                rect_x = label_x - width/2
                rect_y = label_y - height/2
                
                rect = plt.Rectangle((rect_x, rect_y), width, height, 
                                   facecolor='lightblue', edgecolor='blue', 
                                   alpha=0.7, linewidth=1, zorder=2)
                ax.add_patch(rect)
                
                # 添加标签文本
                ax.text(label_x, label_y, f'L{i}', 
                       ha='center', va='center', fontsize=8, 
                       fontweight='bold', zorder=4)
                
                # 添加特征点编号
                ax.text(feature_pos[0], feature_pos[1], f'F{i}', 
                       ha='center', va='center', fontsize=8, 
                       fontweight='bold', color='white', zorder=4)
        
        ax.legend(loc='upper right')
        return []
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=len(optimization_history), 
                                 interval=100, blit=False, repeat=True)
    
    # 保存动画
    anim.save(save_path, writer='pillow', fps=10, dpi=100)
    print(f"动画已保存到 {save_path}")
    

if __name__ == "__main__":
    main()