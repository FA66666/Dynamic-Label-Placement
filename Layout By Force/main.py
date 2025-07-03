# 文件名: main.py
import json
from models import Point, Label
from force_calculator import ForceCalculator
from simulation import SimulationEngine
from visualizer import Visualizer

def main():
    try:
        with open('sample_generated.json', 'r') as f:
            full_data = json.load(f)
        frames_data = full_data['frames']
    except FileNotFoundError:
        print("错误：sample_generated.json 文件未找到。请确保该文件与脚本在同一目录下。")
        return

    params = {
        'wlabel-collision': 120, 'Dlabel-collision': 15,
        'wfeature-collision': 100, 'Dfeature-collision': 17,
        'wpull': 30, 'Dpull': 30,
        'c_friction': 0.7,
        'Wtime': 40,
        'CellSize': 100, 'D_critical': 40, 'R_adaptive': 80,
        'time_step': 0.05 
    }

    force_calculator = ForceCalculator(params)
    simulation_engine = SimulationEngine(params, force_calculator)
    simulation_engine.initialize_from_data(frames_data[0])
    
    visualizer = Visualizer(simulation_engine, frames_data, params)
    visualizer.run_and_save("output_kalman_generic.gif")

if __name__ == '__main__':
    main()