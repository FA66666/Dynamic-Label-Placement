import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def generate_trajectories():
    """
    生成三个点的直线运动轨迹数据
    返回: (red_positions, green_positions, blue_positions)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axis_off()
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)

    # 定义起点和终点
    # 红点：从左下到右上
    red_start = (100, 100)
    red_end = (900, 900)
    
    # 绿点：从下到上
    green_start = (500, 100)
    green_end = (500, 900)
    
    # 蓝点：从右下到左上
    blue_start = (900, 100)
    blue_end = (100, 900)

    red_dot, = ax.plot([], [], 'ro', markersize=10)
    blue_dot, = ax.plot([], [], 'bo', markersize=10)
    green_dot, = ax.plot([], [], 'go', markersize=10)

    red_positions = []
    blue_positions = []
    green_positions = []

    def init():
        red_dot.set_data([], [])
        blue_dot.set_data([], [])
        green_dot.set_data([], [])
        return red_dot, blue_dot, green_dot

    def animate(frame):
        # 计算当前进度（0到1之间）
        progress = frame / 100
        
        # 红点：从左下到右上
        x_red = red_start[0] + (red_end[0] - red_start[0]) * progress
        y_red = red_start[1] + (red_end[1] - red_start[1]) * progress
        
        # 绿点：从下到上
        x_green = green_start[0]  # x坐标保持不变
        y_green = green_start[1] + (green_end[1] - green_start[1]) * progress
        
        # 蓝点：从右下到左上
        x_blue = blue_start[0] + (blue_end[0] - blue_start[0]) * progress
        y_blue = blue_start[1] + (blue_end[1] - blue_start[1]) * progress

        # 存储坐标
        red_positions.append((float(x_red), float(y_red)))
        blue_positions.append((float(x_blue), float(y_blue)))
        green_positions.append((float(x_green), float(y_green)))

        # 更新点的位置
        red_dot.set_data([x_red], [y_red])
        blue_dot.set_data([x_blue], [y_blue])
        green_dot.set_data([x_green], [y_green])

        return red_dot, blue_dot, green_dot

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
    return red_positions, green_positions, blue_positions

if __name__ == '__main__':
    red_positions, green_positions, blue_positions = generate_trajectories()
    print(f"轨迹数据生成完成，每个轨迹包含 {len(red_positions)} 个点")
