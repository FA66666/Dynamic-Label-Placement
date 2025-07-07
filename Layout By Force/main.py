# 文件名: main.py
import json
from point import Point
from label import Label
from force_calculator import ForceCalculator
from simulation import SimulationEngine
from visualizer import Visualizer


def evaluate_overlap(params, frames_data):
    force_calculator = ForceCalculator(params)
    simulation_engine = SimulationEngine(params, force_calculator)
    simulation_engine.initialize_from_data(frames_data[0])
    total_overlap = 0
    for frame in frames_data:
        simulation_engine.update_feature_positions(frame, params['time_step'])
        simulation_engine.step(params['time_step'])
        labels = list(simulation_engine.labels.values())
        for i in range(len(labels)):
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

        params = {
            'wlabel-collision': 1200,
            'Dlabel-collision': 50,
            'wfeature-collision': 1000,
            'Dfeature-collision': 50,
            'wpull': 30,
            'Dpull': 40,
            'c_friction': 0.7,
            'Wtime': 40,
            'CellSize': 170,
            'D_critical': 5,
            'R_adaptive': 10,
            'time_step': 0.05,
            'force_direction': 'x-y+'  # 'x+', 'x-', 'y+', 'y-', 'x+y', 'x-y', 'xy+', 'xy-', 'x+y+', 'x+y-', 'x-y+', 'x-y-', 'xy' - 控制碰撞力的方向约束
        }
        force_calculator = ForceCalculator(params)
        simulation_engine = SimulationEngine(params, force_calculator)
        simulation_engine.initialize_from_data(frames_data[0])
        visualizer = Visualizer(simulation_engine, frames_data, params)
        outname = f"output_kalman_{datafile.replace('.json','')}.gif"
        visualizer.run_and_save(outname)

if __name__ == '__main__':
    main()