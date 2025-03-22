import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from adjustTextDiy import adjust_text  # 直接导入本地目录的adjustText
import time

# 确保当前目录在Python路径中（如果需要）
sys.path.append(os.path.dirname(__file__))


def main():
    # 生成测试数据
    np.random.seed(42)
    x = np.random.rand(100) * 100
    y = np.random.rand(100) * 100
    labels = [f"Label {i}" for i in range(100)]

    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=100, c='blue', alpha=0.6, edgecolors='w')

    # 添加文本标签
    texts = []
    for xi, yi, label in zip(x, y, labels):
        text = ax.text(xi, yi, label, fontsize=8, ha='center', va='center')
        texts.append(text)
    start = time.perf_counter()
    # 调用优化后的adjust_text，使用新参数
    adjust_text(
        texts,
        x=x,
        y=y,
        ax=ax,
        force_text=(0.3, 0.3),  # 增强文本排斥力
        force_static=(0.1, 0.1),
        move_threshold=2.0,  # 移动超过2像素才更新
        stable_iterations=3,  # 连续3轮稳定后标记为稳定
        arrowprops=dict(arrowstyle="-|>", color='gray', lw=0.5),
        expand=(1.2, 1.2),  # 扩大文本区域
        only_move={'text': 'xy', 'static': 'xy'},  # 允许全方向移动
    )
    # 记录结束时间
    end = time.perf_counter()
    # 计算运行时间
    run_time = end - start
    print(f"运行时间：{run_time}秒")

    # 设置坐标轴范围
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    ax.set_title("Text Adjustment with Stability Threshold")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()