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
    try:
        with open('sample_generated.json', 'r') as f:
            full_data = json.load(f)
        frames_data = full_data['frames']
    except FileNotFoundError:
        print("错误：sample_generated.json 文件未找到。请确保该文件与脚本在同一目录下。")
        return

    params = {
        'wlabel-collision': 1200,
        'Dlabel-collision': 50,
        'wfeature-collision': 1100,
        'Dfeature-collision': 50,
        'wpull': 30,
        'Dpull': 40,
        'c_friction': 0.7,
        'Wtime': 40,
        'CellSize': 100,
        'D_critical': 100,
        'R_adaptive': 100,
        'time_step': 0.05
    }
    force_calculator = ForceCalculator(params)
    simulation_engine = SimulationEngine(params, force_calculator)
    simulation_engine.initialize_from_data(frames_data[0])
    visualizer = Visualizer(simulation_engine, frames_data, params)
    visualizer.run_and_save("output_kalman_generic.gif")

if __name__ == '__main__':
    main()