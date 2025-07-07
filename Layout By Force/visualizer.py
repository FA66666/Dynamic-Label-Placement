import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from point import Point
from label import Label
import numpy as np

class Visualizer:
    force_arrows = []

    def __init__(self, simulation_engine, frames_data, params):
        self.engine = simulation_engine
        self.frames_data = frames_data
        self.params = params
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self._setup_plot()
        self._create_artists()

    def _setup_plot(self):
        self.ax.set_xlim(0, 1000)
        self.ax.set_ylim(0, 1000)
        self.ax.set_aspect('equal')
        self.ax.grid(True)

    def _create_artists(self):
        """创建可视化对象（特征点、标签矩形、文本、引导线）"""
        self.feature_plots = {pid: self.ax.plot([], [], 'ko', markersize=5)[0] for pid in self.engine.features.keys()}
        self.label_rects = {pid: patches.Rectangle((0, 0), 1, 1, fc='white', ec='blue', alpha=0.8) for pid in self.engine.labels.keys()}
        self.label_texts = {pid: self.ax.text(0, 0, '', fontsize=8, ha='center', va='center') for pid in self.engine.labels.keys()}
        self.leader_lines = {pid: self.ax.plot([], [], color='gray', linestyle='--', linewidth=1)[0] for pid in self.engine.labels.keys()}
        for rect in self.label_rects.values():
            self.ax.add_patch(rect)
    
    def _update_frame(self, frame_num):
        """更新单帧动画内容"""
        time_step = self.params['time_step']
        if frame_num > 0:
            self.engine.update_feature_positions(self.frames_data[frame_num], time_step)
        sub_steps = 5  # 每帧分多个子步骤提高稳定性
        for _ in range(sub_steps):
            self.engine.step(time_step / sub_steps)
        self.ax.set_title(f"Frame: {frame_num}")
        for pid, feature in self.engine.features.items():
            self.feature_plots[pid].set_data([feature.x], [feature.y])

        if hasattr(self, 'force_arrows'):  # 清除上一帧的力箭头
            for arrow in self.force_arrows:
                arrow.remove()
        self.force_arrows = []

        for pid, label in self.engine.labels.items():
            self.label_rects[pid].set_xy((label.x, label.y))  # 更新标签位置和尺寸
            self.label_rects[pid].set_width(label.width)
            self.label_rects[pid].set_height(label.height)
            self.label_texts[pid].set_position((label.center_x, label.center_y))
            self.label_texts[pid].set_text(label.text)
            
            feature = self.engine.features[pid]
            self.leader_lines[pid].set_data([label.center_x, feature.x], [label.center_y, feature.y])  # 引导线连接

        return list(self.feature_plots.values()) + list(self.label_rects.values()) + list(self.label_texts.values()) + list(self.leader_lines.values())

    def run_and_save(self, output_filename="label_animation.gif", interval=50):
        """运行仿真并保存为GIF动画"""
        ani = FuncAnimation(
            self.fig, self._update_frame, frames=len(self.frames_data), 
            blit=True, interval=interval, repeat=False
        )
        print(f"开始渲染并保存动画到 {output_filename} ...")
        fps = 1000 / interval
        ani.save(output_filename, writer=PillowWriter(fps=fps))
        print(f"动画已成功保存到 {output_filename}")
        plt.close(self.fig)