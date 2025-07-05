# 文件名: visualizer.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from point import Point
from label import Label
import numpy as np

class Visualizer:
    # 为合力箭头类型提示
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
        self.feature_plots = {pid: self.ax.plot([], [], 'ko', markersize=5)[0] for pid in self.engine.features.keys()}
        self.label_rects = {pid: patches.Rectangle((0, 0), 1, 1, fc='white', ec='blue', alpha=0.8) for pid in self.engine.labels.keys()}
        self.label_texts = {pid: self.ax.text(0, 0, '', fontsize=8, ha='center', va='center') for pid in self.engine.labels.keys()}
        # 新增：创建引导线对象
        self.leader_lines = {pid: self.ax.plot([], [], color='gray', linestyle='--', linewidth=1)[0] for pid in self.engine.labels.keys()}
        for rect in self.label_rects.values():
            self.ax.add_patch(rect)
    
    def _update_frame(self, frame_num):
        time_step = self.params['time_step']
        if frame_num > 0:
            self.engine.update_feature_positions(self.frames_data[frame_num], time_step)
        sub_steps = 5
        for _ in range(sub_steps):
            self.engine.step(time_step / sub_steps)
        self.ax.set_title(f"Frame: {frame_num}")
        for pid, feature in self.engine.features.items():
            self.feature_plots[pid].set_data([feature.x], [feature.y])

        # 清除上一帧的箭头
        if hasattr(self, 'force_arrows'):
            for arrow in self.force_arrows:
                arrow.remove()
        self.force_arrows = []

        for pid, label in self.engine.labels.items():
            self.label_rects[pid].set_xy((label.x, label.y))
            self.label_rects[pid].set_width(label.width)
            self.label_rects[pid].set_height(label.height)
            self.label_texts[pid].set_position((label.center_x, label.center_y))
            self.label_texts[pid].set_text(label.text)
            
            # 新增：更新引导线位置
            feature = self.engine.features[pid]
            self.leader_lines[pid].set_data([label.center_x, feature.x], [label.center_y, feature.y])

            # 注释掉合力箭头绘制代码
            # # 计算合力并画箭头
            # ... (这部分保持不变)

        # 检查所有标签对是否重叠，如有重叠则保存图片
        overlapped = False
        overlapped_pairs = []
        label_list = list(self.engine.labels.values())
        for i in range(len(label_list)):
            l1 = label_list[i]
            for j in range(i+1, len(label_list)):
                l2 = label_list[j]
                # 计算两个标签的包围盒
                l1_xmin, l1_ymin = l1.x, l1.y
                l1_xmax, l1_ymax = l1.x + l1.width, l1.y + l1.height
                l2_xmin, l2_ymin = l2.x, l2.y
                l2_xmax, l2_ymax = l2.x + l2.width, l2.y + l2.height
                # 判断是否重叠
                if not (l1_xmax < l2_xmin or l1_xmin > l2_xmax or l1_ymax < l2_ymin or l1_ymin > l2_ymax):
                    overlapped = True
                    overlapped_pairs.append((l1.id, l2.id))
        if overlapped:
            # 文件名包含帧号和所有重叠对
            pair_str = '_'.join([f"{id1}_{id2}" for id1, id2 in overlapped_pairs])
            fname = f"overlap_compare_frame_{frame_num}_{pair_str}.png"

            # 创建左右对比图
            fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 9))
            for ax in [ax_left, ax_right]:
                ax.set_xlim(0, 1000)
                ax.set_ylim(0, 1000)
                ax.set_aspect('equal')
                ax.grid(True)

            # 左侧：实际位置（含合力箭头）
            for pid, feature in self.engine.features.items():
                ax_left.plot([feature.x], [feature.y], 'ko', markersize=5)
            for pid, label in self.engine.labels.items():
                rect = patches.Rectangle((label.x, label.y), label.width, label.height, fc='white', ec='blue', alpha=0.8)
                ax_left.add_patch(rect)
                ax_left.text(label.center_x, label.center_y, label.text, fontsize=8, ha='center', va='center')
                # 新增：为左侧对比图添加引导线
                feature = self.engine.features[pid]
                ax_left.plot([label.center_x, feature.x], [label.center_y, feature.y], color='gray', linestyle='--', linewidth=1)
                # 合力箭头
                lx, ly = int(label.center_x // self.params['CellSize']), int(label.center_y // self.params['CellSize'])
                candidate_neighbors = []
                grid = self.engine._build_grid()
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        cell_key = (lx + dx, ly + dy)
                        if cell_key in grid:
                            candidate_neighbors.extend(grid[cell_key])
                neighbor_labels = [p for p in candidate_neighbors if isinstance(p, Label) and p.id != label.id]
                neighbor_features = [p for p in candidate_neighbors if isinstance(p, Point)]
                fx, fy = self.engine.force_calculator.compute_total_force_for_label(label, neighbor_labels, neighbor_features)
                arrow_scale = 0.2
                ax_left.arrow(label.center_x, label.center_y, fx * arrow_scale, fy * arrow_scale, head_width=8, head_length=12, fc='red', ec='red', alpha=0.7)
            ax_left.set_title(f"Actual Frame: {frame_num}")

            # 右侧：预测位置
            for pid, feature in self.engine.features.items():
                ax_right.plot([feature.x], [feature.y], 'ko', markersize=5)
            for pid, label in self.engine.labels.items():
                pred_state = label.kf.predict(u=np.array([[label.ax],[label.ay]]))
                pred_x = pred_state[0,0] - label.width/2
                pred_y = pred_state[2,0] - label.height/2
                rect = patches.Rectangle((pred_x, pred_y), label.width, label.height, fc='none', ec='orange', lw=2, linestyle='--', alpha=0.8)
                ax_right.add_patch(rect)
                ax_right.text(pred_x + label.width/2, pred_y + label.height/2, label.text, fontsize=8, ha='center', va='center', color='orange')
                # 新增：为右侧对比图添加引导线
                feature = self.engine.features[pid]
                ax_right.plot([pred_x + label.width/2, feature.x], [pred_y + label.height/2, feature.y], color='gray', linestyle='--', linewidth=1)
            ax_right.set_title(f"Predicted Frame: {frame_num}")

            fig.suptitle(f"Frame: {frame_num}  Overlap: {pair_str}")
            fig.savefig(fname)
            plt.close(fig)
        # 新增：将引导线添加到返回的艺术家列表中以支持blit
        return list(self.feature_plots.values()) + list(self.label_rects.values()) + list(self.label_texts.values()) + list(self.leader_lines.values())

    def run_and_save(self, output_filename="label_animation.gif", interval=50):
        ani = FuncAnimation(
            self.fig, self._update_frame, frames=len(self.frames_data), 
            blit=True, interval=interval, repeat=False
        )
        print(f"开始渲染并保存动画到 {output_filename} ...")
        fps = 1000 / interval
        ani.save(output_filename, writer=PillowWriter(fps=fps))
        print(f"动画已成功保存到 {output_filename}")
        plt.close(self.fig)