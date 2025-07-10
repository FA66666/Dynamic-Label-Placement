import json
import math
import numpy as np
from point import Point
from label import Label
from force_calculator import ForceCalculator
from simulation import SimulationEngine
from visualizer import Visualizer
from evaluator import evaluate_single_frame_quality, evaluate_layout_quality


def main():
    for datafile in ['sample_generated copy_slow_2x.json']:
    # for datafile in ['sample10_1.json']:
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
            'force_direction': 'project',     # 力方向约束策略(project/xy)
            'predict_frames': 20          # 预测帧数
        }
        
        print(f"\n处理文件: {datafile}")
        print(f"当前预测帧数：{params['predict_frames']}")
        
        # 创建仿真组件
        force_calculator = ForceCalculator(params)
        simulation_engine = SimulationEngine(params, force_calculator)
        simulation_engine.initialize_from_data(frames_data[0])
        
        # 运行仿真并评估OCC、INT指标
        step_evaluations = []
        total_occ_sum = 0
        total_int_sum = 0  
        frame_count = 0
        
        print("开始仿真和评估...")
        
        # 评估初始帧
        initial_quality = evaluate_single_frame_quality(simulation_engine)
        step_evaluations.append(initial_quality)
        total_occ_sum = initial_quality['occ']
        total_int_sum = initial_quality['int']
        frame_count = 1
        
        # 运行仿真，每一步都评估
        time_step = params['time_step']
        for frame_num in range(1, len(frames_data)):
            simulation_engine.update_feature_positions(frames_data[frame_num], time_step)
            
            # 每帧进行多个子步骤以提高稳定性
            sub_steps = 5
            for sub_step in range(sub_steps):
                simulation_engine.step(time_step / sub_steps)
                
                # 评估当前步骤
                quality = evaluate_single_frame_quality(simulation_engine)
                step_evaluations.append(quality)
                
                # 累积到总和中
                total_occ_sum += quality['occ']
                total_int_sum += quality['int'] 
                frame_count += 1
        
        print("仿真完成")
        
        # 计算最终平均值
        final_avg_occ = total_occ_sum / frame_count
        final_avg_int = total_int_sum / frame_count
        
        # 计算论文中的标准评价指标
        quality_metrics = evaluate_layout_quality(simulation_engine, frames_data)
        
        # 输出所有评估指标
        print("\n=== 标签布局质量评估结果 ===")
        print(f"OCC: {final_avg_occ:.2f}")
        print(f"INT: {final_avg_int:.2f}") 
        print(f"S_overlap: {quality_metrics['S_overlap']:.2f}")
        print(f"S_position: {quality_metrics['S_position']:.2f}")
        print(f"S_aesthetics: {quality_metrics['S_aesthetics']:.2f}")
        print(f"S_angle_smoothness: {quality_metrics['S_angle_smoothness']:.2f}")
        print(f"S_distance_smoothness: {quality_metrics['S_distance_smoothness']:.2f}")
        print(f"总帧数: {quality_metrics['total_frames']}")
        print(f"总标签数: {quality_metrics['total_labels']}")
        
        # 重新创建仿真引擎用于可视化
        force_calculator = ForceCalculator(params)
        simulation_engine = SimulationEngine(params, force_calculator)
        simulation_engine.initialize_from_data(frames_data[0])
        
        visualizer = Visualizer(simulation_engine, frames_data, params)
        outname = f"output_kalman_{datafile.replace('.json','')}.gif"
        visualizer.run_and_save(outname)
            
        print("="*50)

if __name__ == '__main__':
    main()