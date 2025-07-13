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
import math
from metrics import evaluate_metrics

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
        frame_interval=0.1
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
    # 计算初始状态的特征预测力
    initial_prediction_forces = calculate_feature_prediction_forces(
        force_based_optimizer, current_positions, current_velocities)
    
    optimization_history.append({
        'positions': current_positions.copy(),
        'velocities': current_velocities.copy(),
        'features': [(f.position[0], f.position[1]) for f in features],
        'frame': current_frame,
        'prediction_forces': initial_prediction_forces.copy()
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
        
        # 计算当前步的特征预测力
        current_prediction_forces = calculate_feature_prediction_forces(
            force_based_optimizer, current_positions, current_velocities)
        
        # 记录每一步的状态
        optimization_history.append({
            'positions': current_positions.copy(),
            'velocities': current_velocities.copy(),
            'features': [(f.position[0], f.position[1]) for f in features],
            'frame': current_frame,
            'prediction_forces': current_prediction_forces.copy()
        })
        
    # 评价指标计算
    metrics = evaluate_metrics(optimization_history, labels, features)
    print("评价指标:", metrics)

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
        ax.scatter(feature_x, feature_y, c='red', s=31.4, marker='o', label='特征点', zorder=3)
        
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
    anim.save(save_path, writer='pillow', fps=20, dpi=100)
    print(f"动画已保存到 {save_path}")

def calculate_feature_prediction_forces(force_based_optimizer, label_positions, velocities):
    """
    计算每个标签的特征预测力大小
    
    Args:
        force_based_optimizer: 力优化器实例
        label_positions: 标签位置字典
        velocities: 标签速度字典
    
    Returns:
        dict: 每个标签的特征预测力大小字典
    """
    prediction_forces = {}
    
    for i in range(len(force_based_optimizer.labels)):
        if i not in label_positions:
            continue
            
        total_pred_fx = 0.0
        total_pred_fy = 0.0
        
        # 1. 计算特征点运动预测力
        feature_velocities = [feature.velocity for feature in force_based_optimizer.features]
        for j in range(len(force_based_optimizer.features)):
            pred_fx, pred_fy = force_based_optimizer.compute_point_movement_prediction_force(
                i, j, label_positions, feature_velocities)
            total_pred_fx += pred_fx * param_NoAdj['c_point_predict']
            total_pred_fy += pred_fy * param_NoAdj['c_point_predict']
        
        # 2. 计算其他标签运动预测力对当前标签的影响
        for j in range(len(force_based_optimizer.labels)):
            if i != j:
                pred_fx, pred_fy = force_based_optimizer.compute_movement_prediction_force(
                    i, j, label_positions, velocities)
                total_pred_fx += pred_fx * param_NoAdj['c_label_predict']
                total_pred_fy += pred_fy * param_NoAdj['c_label_predict']
        
        # 计算预测力的大小
        prediction_force_magnitude = math.hypot(total_pred_fx, total_pred_fy)
        prediction_forces[i] = prediction_force_magnitude
    
    return prediction_forces

# 新增评价指标计算函数


if __name__ == "__main__":
    main()