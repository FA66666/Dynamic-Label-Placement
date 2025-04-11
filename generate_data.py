import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_trajectories():
    """
    生成7个点的运动轨迹数据，其中4个点静止，3个点移动
    返回: 包含7个特征点轨迹的列表
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    # 修改坐标范围，确保在坐标轴内
    ax.set_xlim(50, 950)
    ax.set_ylim(50, 950)

    # 定义7个点，其中4个静止，3个移动
    points = [
        # 移动的点
        # 点1：从左到右
        {'start': (100, 700), 'end': (900, 700), 'color': 'red'},
        # 点2：从下到上
        {'start': (500, 100), 'end': (500, 900), 'color': 'green'},
        # 点3：对角线移动
        {'start': (100, 100), 'end': (900, 900), 'color': 'blue'},
        
        # 静止的点
        # 点4：左上角
        {'start': (200, 800), 'end': (200, 800), 'color': 'purple'},
        # 点5：右上角
        {'start': (800, 800), 'end': (800, 800), 'color': 'orange'},
        # 点6：左下角
        {'start': (200, 200), 'end': (200, 200), 'color': 'cyan'},
        # 点7：右下角
        {'start': (800, 200), 'end': (800, 200), 'color': 'magenta'}
    ]

    # 创建动画对象
    dots = []
    for point in points:
        dot, = ax.plot([], [], 'o', color=point['color'], markersize=10)
        dots.append(dot)

    # 存储所有点的轨迹
    positions = [[] for _ in range(len(points))]

    def init():
        for dot in dots:
            dot.set_data([], [])
        return dots

    def animate(frame):
        # 计算当前进度（0到1之间）
        progress = frame / 100
        
        for i, point in enumerate(points):
            # 计算当前点的位置
            x = point['start'][0] + (point['end'][0] - point['start'][0]) * progress
            y = point['start'][1] + (point['end'][1] - point['start'][1]) * progress
            
            # 确保坐标在有效范围内
            x = max(50, min(950, x))
            y = max(50, min(950, y))
            
            # 存储坐标
            positions[i].append((float(x), float(y)))
            
            # 更新点的位置
            dots[i].set_data([x], [y])

        return dots

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=100,
        interval=50,
        blit=True
    )

    anim.save('input.gif', 
              writer='pillow', 
              dpi=100,
              savefig_kwargs={'facecolor': 'white'})

    plt.close()
    return positions

if __name__ == '__main__':
    positions = generate_trajectories()
    print(f"轨迹数据生成完成，每个轨迹包含 {len(positions[0])} 个点")
