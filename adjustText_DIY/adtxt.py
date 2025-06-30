import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text  # 官方adjust_text库
from adjustTextDiy import adjust_text as adjust_text_diy  # 用户自定义的adjustTextDiy模块
import time


def generate_data():
    """生成测试数据（公共函数）"""
    np.random.seed(200)
    x = np.random.rand(100) * 100
    y = np.random.rand(100) * 100
    labels = [f"Label {i}" for i in range(100)]
    return x, y, labels


def plot_and_adjust(adjust_func, title, save_path, **kwargs):
    """通用绘图函数，接受调整函数和参数"""
    x, y, labels = generate_data()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, s=100, c='blue', alpha=0.6, edgecolors='w')

    texts = []
    for xi, yi, label in zip(x, y, labels):
        text = ax.text(xi, yi, label, fontsize=8, ha='center', va='center')
        texts.append(text)

    start = time.perf_counter()
    adjust_func(texts, x=x, y=y, ax=ax, **kwargs)
    end = time.perf_counter()
    run_time = end - start
    print(f"{title} 运行时间：{run_time:.4f}秒")

    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    # 调用自定义adjustTextDiy
    plot_and_adjust(
        adjust_func=adjust_text_diy,
        title="My Adjust Text",
        save_path="My_adjust.png",
        force_text=(0.3, 0.3),
        force_static=(0.1, 0.1),
        move_threshold=0.1,
        stable_iterations=10,
        arrowprops=dict(arrowstyle="-|>", color='gray', lw=0.5),
        expand=(1.2, 1.2),
        only_move={'text': 'xy', 'static': 'xy'},
    )

    # 调用官方adjust_text库
    plot_and_adjust(
        adjust_func=adjust_text,
        title="Official Adjust Text",
        save_path="official_adjust.png",
        force_text=0.3,  # 官方库参数可能不同，需调整
        force_points=0.1,
        arrowprops=dict(arrowstyle="-|>", color='gray', lw=0.5),
        expand_text=(1.2, 1.2),
        expand_points=(1.2, 1.2),
        only_move={'text': 'xy', 'points': 'xy'},
    )


if __name__ == "__main__":
    main()