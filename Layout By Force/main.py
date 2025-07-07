# 文件名: main.py
import json
from point import Point
from label import Label
from force_calculator import ForceCalculator
from simulation import SimulationEngine
from visualizer import Visualizer


def evaluate_overlap(params, frames_data):
    """评估标签重叠程度的辅助函数"""
    force_calculator = ForceCalculator(params)
    simulation_engine = SimulationEngine(params, force_calculator)
    simulation_engine.initialize_from_data(frames_data[0])
    total_overlap = 0
    for frame in frames_data:
        simulation_engine.update_feature_positions(frame, params['time_step'])
        simulation_engine.step(params['time_step'])
        labels = list(simulation_engine.labels.values())
        for i in range(len(labels)):  # 计算所有标签对的重叠面积
            for j in range(i+1, len(labels)):
                l1, l2 = labels[i], labels[j]
                x_overlap = max(0, min(l1.x+l1.width, l2.x+l2.width) - max(l1.x, l2.x))
                y_overlap = max(0, min(l1.y+l1.height, l2.y+l2.height) - max(l1.y, l2.y))
                total_overlap += x_overlap * y_overlap
    return total_overlap

def main():
    for datafile in ['sample_generated.json', 'sample10_1.json']:
        try:
            with open(datafile, 'r') as f:
                full_data = json.load(f)
            frames_data = full_data['frames']
        except FileNotFoundError:
            print(f"错误：{datafile} 文件未找到。请确保该文件与脚本在同一目录下。")
            continue

        params = {  # 力学参数配置
            'wlabel-collision': 1200,    # 标签碰撞力权重
            'Dlabel-collision': 50,      # 标签碰撞距离阈值
            'wfeature-collision': 1000,  # 特征点碰撞力权重
            'Dfeature-collision': 50,    # 特征点碰撞距离阈值
            'wpull': 30,                 # 牵引力权重
            'Dpull': 40,                 # 牵引力距离阈值
            'c_friction': 0.7,           # 摩擦系数
            'Wtime': 40,
            'CellSize': 170,             # 空间网格大小
            'D_critical': 5,
            'R_adaptive': 10,
            'time_step': 0.05,           # 仿真时间步长
            'force_direction': 'xy'  # 力方向约束策略(project/xy)
        }
        force_calculator = ForceCalculator(params)
        simulation_engine = SimulationEngine(params, force_calculator)
        simulation_engine.initialize_from_data(frames_data[0])
        visualizer = Visualizer(simulation_engine, frames_data, params)
        outname = f"output_kalman_{datafile.replace('.json','')}.gif"
        visualizer.run_and_save(outname)

if __name__ == '__main__':
    main()